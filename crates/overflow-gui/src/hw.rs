//! Lightweight hardware monitor.
//!
//! A background thread samples CPU, RAM and (if available) the NVIDIA GPU once a
//! second and publishes the latest snapshot behind a mutex the UI reads each
//! frame. CPU and RAM come straight from /proc; GPU from `nvidia-smi`.

use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Copy, Default)]
pub struct HwStats {
    pub cpu_pct: f32,
    pub mem_used_gb: f32,
    pub mem_total_gb: f32,
    pub gpu_pct: Option<f32>,
    pub vram_used_mb: Option<f32>,
    pub vram_total_mb: Option<f32>,
}

/// Start the monitor thread; returns the shared snapshot.
pub fn spawn() -> Arc<Mutex<HwStats>> {
    let stats = Arc::new(Mutex::new(HwStats::default()));
    let out = Arc::clone(&stats);
    std::thread::spawn(move || {
        let mut prev = read_cpu_times();
        loop {
            std::thread::sleep(Duration::from_millis(1000));
            let now = read_cpu_times();
            let cpu_pct = match (prev, now) {
                (Some((pb, pt)), Some((nb, nt))) if nt > pt => {
                    100.0 * (nb.saturating_sub(pb)) as f32 / (nt - pt) as f32
                }
                _ => 0.0,
            };
            prev = now;

            let (mem_used_gb, mem_total_gb) = read_mem().unwrap_or((0.0, 0.0));
            let (gpu_pct, vram_used_mb, vram_total_mb) = read_gpu();

            if let Ok(mut s) = out.lock() {
                *s = HwStats {
                    cpu_pct,
                    mem_used_gb,
                    mem_total_gb,
                    gpu_pct,
                    vram_used_mb,
                    vram_total_mb,
                };
            }
        }
    });
    stats
}

/// (busy, total) jiffies from the aggregate line of /proc/stat.
fn read_cpu_times() -> Option<(u64, u64)> {
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    let line = stat.lines().next()?; // "cpu  u n s idle iowait irq softirq steal ..."
    let nums: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|t| t.parse().ok())
        .collect();
    if nums.len() < 4 {
        return None;
    }
    let idle = nums[3] + nums.get(4).copied().unwrap_or(0); // idle + iowait
    let total: u64 = nums.iter().sum();
    Some((total.saturating_sub(idle), total))
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
    let parts: Vec<f32> = first
        .split(',')
        .filter_map(|p| p.trim().parse().ok())
        .collect();
    if parts.len() == 3 {
        (Some(parts[0]), Some(parts[1]), Some(parts[2]))
    } else {
        (None, None, None)
    }
}
