//! Lightweight hardware monitor.
//!
//! A background thread samples, once a second, both the whole system (CPU, RAM,
//! and the NVIDIA GPU if present) and *this process* — which is what the model
//! actually costs, since inference runs in-process. The latest snapshot is
//! published behind a mutex the UI reads each frame.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Default)]
pub struct HwStats {
    // whole system
    pub cpu_pct: f32,
    pub mem_used_gb: f32,
    pub mem_total_gb: f32,
    pub gpu_pct: Option<f32>,
    pub vram_used_mb: Option<f32>,
    pub vram_total_mb: Option<f32>,
    // this process = what the model costs
    pub proc_cpu_pct: f32, // share of the whole machine (all cores = 100%)
    pub proc_rss_gb: f32,  // resident memory
}

/// Clock ticks per second on Linux (sysconf(_SC_CLK_TCK) is 100 on x86).
const CLK_TCK: f32 = 100.0;
/// Page size in bytes (4 KiB on x86).
const PAGE: f32 = 4096.0;

/// Start the monitor thread; returns the shared snapshot.
pub fn spawn() -> Arc<Mutex<HwStats>> {
    let stats = Arc::new(Mutex::new(HwStats::default()));
    let out = Arc::clone(&stats);
    let ncpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) as f32;

    std::thread::spawn(move || {
        let mut prev_sys = read_cpu_times();
        let mut prev_proc = read_proc_ticks();
        let mut last = Instant::now();
        loop {
            std::thread::sleep(Duration::from_millis(1000));
            let elapsed = last.elapsed().as_secs_f32().max(0.001);
            last = Instant::now();

            // system CPU
            let now_sys = read_cpu_times();
            let cpu_pct = match (prev_sys, now_sys) {
                (Some((pb, pt)), Some((nb, nt))) if nt > pt => {
                    100.0 * (nb.saturating_sub(pb)) as f32 / (nt - pt) as f32
                }
                _ => 0.0,
            };
            prev_sys = now_sys;

            // process CPU (share of whole machine)
            let now_proc = read_proc_ticks();
            let proc_cpu_pct = match (prev_proc, now_proc) {
                (Some(p), Some(n)) => {
                    let cpu_secs = n.saturating_sub(p) as f32 / CLK_TCK;
                    (100.0 * cpu_secs / (elapsed * ncpu)).min(100.0)
                }
                _ => 0.0,
            };
            prev_proc = now_proc;

            let (mem_used_gb, mem_total_gb) = read_mem().unwrap_or((0.0, 0.0));
            let (gpu_pct, vram_used_mb, vram_total_mb) = read_gpu();
            let proc_rss_gb = read_proc_rss_gb().unwrap_or(0.0);

            if let Ok(mut s) = out.lock() {
                *s = HwStats {
                    cpu_pct,
                    mem_used_gb,
                    mem_total_gb,
                    gpu_pct,
                    vram_used_mb,
                    vram_total_mb,
                    proc_cpu_pct,
                    proc_rss_gb,
                };
            }
        }
    });
    stats
}

/// (busy, total) jiffies from the aggregate line of /proc/stat.
fn read_cpu_times() -> Option<(u64, u64)> {
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    let line = stat.lines().next()?;
    let nums: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|t| t.parse().ok())
        .collect();
    if nums.len() < 4 {
        return None;
    }
    let idle = nums[3] + nums.get(4).copied().unwrap_or(0);
    let total: u64 = nums.iter().sum();
    Some((total.saturating_sub(idle), total))
}

/// utime + stime for this process (clock ticks) from /proc/self/stat.
fn read_proc_ticks() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // comm (field 2) may contain spaces/parens; fields resume after the last ')'.
    let after = &s[s.rfind(')')? + 1..];
    let f: Vec<&str> = after.split_whitespace().collect();
    // field 3 = f[0] (state); utime = field 14 = f[11]; stime = field 15 = f[12].
    let utime: u64 = f.get(11)?.parse().ok()?;
    let stime: u64 = f.get(12)?.parse().ok()?;
    Some(utime + stime)
}

/// Resident set size of this process, in GB, from /proc/self/statm.
fn read_proc_rss_gb() -> Option<f32> {
    let s = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident: u64 = s.split_whitespace().nth(1)?.parse().ok()?;
    Some(resident as f32 * PAGE / (1024.0 * 1024.0 * 1024.0))
}

/// (used_gb, total_gb) from /proc/meminfo.
fn read_mem() -> Option<(f32, f32)> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb = 0f32;
    let mut avail_kb = 0f32;
    for line in info.lines() {
        if let Some(v) = line.strip_prefix("MemTotal:") {
            total_kb = v.split_whitespace().next()?.parse().ok()?;
        } else if let Some(v) = line.strip_prefix("MemAvailable:") {
            avail_kb = v.split_whitespace().next()?.parse().ok()?;
        }
    }
    let gb = 1024.0 * 1024.0;
    Some(((total_kb - avail_kb) / gb, total_kb / gb))
}

/// The GPU model name via nvidia-smi (e.g. "NVIDIA GeForce RTX 3060"), or None
/// if there is no NVIDIA GPU / nvidia-smi. Queried once at startup — it does not
/// change while the app runs.
pub fn gpu_name() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let name = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().to_string();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// (util%, vram_used_mb, vram_total_mb) via nvidia-smi, or Nones if unavailable.
fn read_gpu() -> (Option<f32>, Option<f32>, Option<f32>) {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(out) = out else {
        return (None, None, None);
    };
    if !out.status.success() {
        return (None, None, None);
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let first = text.lines().next().unwrap_or("");
    let parts: Vec<f32> = first.split(',').filter_map(|p| p.trim().parse().ok()).collect();
    if parts.len() == 3 {
        (Some(parts[0]), Some(parts[1]), Some(parts[2]))
    } else {
        (None, None, None)
    }
}
