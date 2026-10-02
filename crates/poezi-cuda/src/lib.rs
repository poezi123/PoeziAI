//! GPU backend (A5): run the whole Qwen2 forward pass on the RTX 3060.
//!
//! Uses vendored CUDA (scripts/fetch-cuda.sh): only libnvrtc is shipped; the
//! driver's libcuda is already present. Kernels are compiled at runtime with
//! NVRTC via cudarc (dynamic-loading) — no system CUDA install. fp16 is decoded
//! in-kernel with bit ops so NVRTC needs no CUDA headers.
//!
//! A5.6 makes the forward pass *resident*: all activations stay in VRAM and the
//! per-op CPU<->GPU round-trips are gone. `GpuWeights` holds the immutable model
//! (weights, norms, biases) and `GpuSession` the per-generation state (scratch
//! buffers + KV cache). Only the final logits are copied back to the host.
//! Kernels are ported from the CPU paths in poezi-tensor/poezi-model and
//! validated against them (and end-to-end against llama.cpp).

use std::error::Error;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use poezi_gguf::GgmlType;

type R<T> = Result<T, Box<dyn Error>>;

const KERNELS: &str = r#"
__device__ __forceinline__ float h2f(unsigned short h) {
    unsigned int sign = (h >> 15) & 1u;
    unsigned int exp  = (h >> 10) & 0x1fu;
    unsigned int mant = h & 0x3ffu;
    float v;
    if (exp == 0u)          v = (float)mant * 5.9604645e-8f;
    else if (exp == 0x1fu)  v = mant ? (0.0f/0.0f) : (1.0f/0.0f);
    else                    v = (1.0f + (float)mant / 1024.0f) * exp2f((float)((int)exp - 15));
    return sign ? -v : v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
// A5.7: warp per output row, lanes cooperate *within* a block so adjacent lanes
// read adjacent weight bytes (coalesced). Each lane accumulates its share, then
// one warp reduction. Index maps match the CPU dequant exactly.

// ---- Q8_0: 32-value blocks; lane handles element `lane` of each block ----
extern "C" __global__ void matmul_q8_0(float* y, const unsigned char* w, const float* x, int in_dim, int out_dim){
    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x; int row=gid>>5, lane=gid&31;
    if (row>=out_dim) return;
    int nb=in_dim/32; long rowbase=(long)row*nb*34; float acc=0.0f;
    for (int b=0;b<nb;b++){ long base=rowbase+(long)b*34;
        float d=h2f((unsigned short)w[base]|((unsigned short)w[base+1]<<8));
        signed char q=(signed char)w[base+2+lane];            // coalesced across the warp
        acc += d*(float)q*x[b*32+lane];
    }
    acc=warp_sum(acc); if(lane==0) y[row]=acc;
}
__device__ __forceinline__ void get_sm(int j,const unsigned char* q,unsigned char* d,unsigned char* m){
    if(j<4){*d=q[j]&63;*m=q[j+4]&63;}
    else{*d=(q[j+4]&0xF)|((q[j-4]>>6)<<4);*m=(q[j+4]>>4)|((q[j-0]>>6)<<4);}
}
// ---- Q4_K: 256-value super-blocks; lane handles element `lane` of each 32 group ----
extern "C" __global__ void matmul_q4_k(float* y,const unsigned char* w,const float* x,int in_dim,int out_dim){
    unsigned int gid=blockIdx.x*blockDim.x+threadIdx.x; int row=gid>>5,lane=gid&31;
    if(row>=out_dim) return; int nb=in_dim/256; long rowbase=(long)row*nb*144; float acc=0.0f;
    for(int sb=0;sb<nb;sb++){ long base=rowbase+(long)sb*144;
        float d=h2f((unsigned short)w[base]|((unsigned short)w[base+1]<<8));
        float dmin=h2f((unsigned short)w[base+2]|((unsigned short)w[base+3]<<8));
        const unsigned char* scales=w+base+4; const unsigned char* qs=w+base+16;
        int vb=sb*256;
        for(int grp=0;grp<4;grp++){ unsigned char sc,m;
            get_sm(grp*2,scales,&sc,&m);   float d1=d*sc, mm1=dmin*m;
            get_sm(grp*2+1,scales,&sc,&m); float d2=d*sc, mm2=dmin*m;
            unsigned char byte=qs[grp*32+lane];               // coalesced
            acc += (d1*(float)(byte&0xF)-mm1) * x[vb+grp*64+lane];
            acc += (d2*(float)(byte>>4) -mm2) * x[vb+grp*64+32+lane];
        }
    }
    acc=warp_sum(acc); if(lane==0) y[row]=acc;
}
// ---- Q6_K: 256-value super-blocks; lane handles element `lane` of each 32 group ----
extern "C" __global__ void matmul_q6_k(float* y,const unsigned char* w,const float* x,int in_dim,int out_dim){
    unsigned int gid=blockIdx.x*blockDim.x+threadIdx.x; int row=gid>>5,lane=gid&31;
    if(row>=out_dim) return; int nb=in_dim/256; long rowbase=(long)row*nb*210; float acc=0.0f;
    int is=lane/16;
    for(int sb=0;sb<nb;sb++){ long base=rowbase+(long)sb*210;
        const unsigned char* ql=w+base; const unsigned char* qh=w+base+128;
        const signed char* sc=(const signed char*)(w+base+192);
        float d=h2f((unsigned short)w[base+208]|((unsigned short)w[base+209]<<8));
        for(int n=0;n<2;n++){ int qlb=n*64,qhb=n*32,scb=n*8,vb=sb*256+n*128;
            int ql0=ql[qlb+lane], ql1=ql[qlb+lane+32], qhb_=qh[qhb+lane];  // coalesced
            int q1=((ql0&0xF)|(((qhb_>>0)&3)<<4))-32;
            int q2=((ql1&0xF)|(((qhb_>>2)&3)<<4))-32;
            int q3=((ql0>>4) |(((qhb_>>4)&3)<<4))-32;
            int q4=((ql1>>4) |(((qhb_>>6)&3)<<4))-32;
            acc += d*(float)sc[scb+is+0]*(float)q1*x[vb+lane];
            acc += d*(float)sc[scb+is+2]*(float)q2*x[vb+lane+32];
            acc += d*(float)sc[scb+is+4]*(float)q3*x[vb+lane+64];
            acc += d*(float)sc[scb+is+6]*(float)q4*x[vb+lane+96];
        }
    }
    acc=warp_sum(acc); if(lane==0) y[row]=acc;
}
// ---- attention: one block per query head, softmax in shared memory ----
extern "C" __global__ void attention(float* out,const float* q,const float* kc,const float* vc,
                                     int n_pos,int head_dim,int n_head,int n_head_kv,int kv_dim){
    int h=blockIdx.x; if(h>=n_head) return;
    int group=n_head/n_head_kv; int kvh=h/group; int t=threadIdx.x,nt=blockDim.x;
    extern __shared__ float s[]; __shared__ float red[128];
    const float* qh=q+h*head_dim; float scale=rsqrtf((float)head_dim);
    for(int p=t;p<n_pos;p+=nt){ const float* kt=kc+(size_t)p*kv_dim+kvh*head_dim;
        float dp=0.0f; for(int i=0;i<head_dim;i++) dp+=qh[i]*kt[i]; s[p]=dp*scale; }
    __syncthreads();
    float m=-3.4e38f; for(int p=t;p<n_pos;p+=nt) m=fmaxf(m,s[p]);
    red[t]=m; __syncthreads();
    for(int o=nt>>1;o>0;o>>=1){ if(t<o) red[t]=fmaxf(red[t],red[t+o]); __syncthreads(); }
    float gmax=red[0]; __syncthreads();
    float sm=0.0f; for(int p=t;p<n_pos;p+=nt){ float e=__expf(s[p]-gmax); s[p]=e; sm+=e; }
    red[t]=sm; __syncthreads();
    for(int o=nt>>1;o>0;o>>=1){ if(t<o) red[t]+=red[t+o]; __syncthreads(); }
    float gsum=red[0]; __syncthreads();
    if(t<head_dim){ float acc=0.0f;
        for(int p=0;p<n_pos;p++) acc+=(s[p]/gsum)*vc[(size_t)p*kv_dim+kvh*head_dim+t];
        out[h*head_dim+t]=acc; }
}
// ---- elementwise / norm helpers (device-resident forward) ----
extern "C" __global__ void rmsnorm(float* out,const float* x,const float* w,int n,float eps){
    int t=threadIdx.x,nt=blockDim.x; __shared__ float red[256];
    float ss=0.0f; for(int i=t;i<n;i+=nt){ float v=x[i]; ss+=v*v; }
    red[t]=ss; __syncthreads();
    for(int o=nt>>1;o>0;o>>=1){ if(t<o) red[t]+=red[t+o]; __syncthreads(); }
    float scale=rsqrtf(red[0]/(float)n+eps);
    for(int i=t;i<n;i+=nt) out[i]=x[i]*scale*w[i];
}
extern "C" __global__ void add_inplace(float* a,const float* b,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]+=b[i];
}
extern "C" __global__ void swiglu(float* g,const float* u,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){ float x=g[i]; g[i]=(x/(1.0f+__expf(-x)))*u[i]; }
}
extern "C" __global__ void rope(float* x,int n_heads,int head_dim,int pos,float theta){
    int h=blockIdx.x; if(h>=n_heads) return; int half=head_dim/2; float* xh=x+h*head_dim;
    for(int k=threadIdx.x;k<half;k+=blockDim.x){
        float inv=1.0f/powf(theta,(2.0f*(float)k)/(float)head_dim);
        float ang=(float)pos*inv; float c=cosf(ang),s=sinf(ang);
        float x0=xh[k],x1=xh[k+half]; xh[k]=x0*c-x1*s; xh[k+half]=x0*s+x1*c; }
}
"#;

/// A quantized weight resident in VRAM, with its ggml type.
pub struct GpuWeight {
    buf: CudaSlice<u8>,
    ty: GgmlType,
}

/// Shared compiled kernels + device/stream. One per process.
pub struct GpuBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    f_q8: CudaFunction,
    f_q4k: CudaFunction,
    f_q6k: CudaFunction,
    f_attn: CudaFunction,
    f_rmsnorm: CudaFunction,
    f_add: CudaFunction,
    f_swiglu: CudaFunction,
    f_rope: CudaFunction,
}

impl GpuBackend {
    pub fn new() -> R<Self> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let m = ctx.load_module(compile_ptx(KERNELS)?)?;
        Ok(GpuBackend {
            f_q8: m.load_function("matmul_q8_0")?,
            f_q4k: m.load_function("matmul_q4_k")?,
            f_q6k: m.load_function("matmul_q6_k")?,
            f_attn: m.load_function("attention")?,
            f_rmsnorm: m.load_function("rmsnorm")?,
            f_add: m.load_function("add_inplace")?,
            f_swiglu: m.load_function("swiglu")?,
            f_rope: m.load_function("rope")?,
            ctx,
            stream,
        })
    }

    pub fn device_name(&self) -> String {
        self.ctx.name().unwrap_or_else(|_| "unknown".into())
    }

    pub fn upload_weight(&self, bytes: &[u8], ty: GgmlType) -> R<GpuWeight> {
        Ok(GpuWeight { buf: self.stream.clone_htod(bytes)?, ty })
    }
    fn upload_f32(&self, data: &[f32]) -> R<CudaSlice<f32>> {
        Ok(self.stream.clone_htod(data)?)
    }

    fn mm_fn(&self, ty: GgmlType) -> R<&CudaFunction> {
        Ok(match ty {
            GgmlType::Q8_0 => &self.f_q8,
            GgmlType::Q4K => &self.f_q4k,
            GgmlType::Q6K => &self.f_q6k,
            o => return Err(format!("gpu matmul: unsupported type {o:?}").into()),
        })
    }

    /// Device-to-device matmul: x_dev, y_dev already in VRAM (no host transfer).
    fn mm(&self, w: &GpuWeight, in_dim: usize, out_dim: usize, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>) -> R<()> {
        let func = self.mm_fn(w.ty)?.clone();
        let in_i = in_dim as i32;
        let out_i = out_dim as i32;
        const T: u32 = 256;
        let blocks = (out_dim as u32).div_ceil(T / 32);
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (T, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.stream.launch_builder(&func);
        b.arg(y);
        b.arg(&w.buf);
        b.arg(x);
        b.arg(&in_i);
        b.arg(&out_i);
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    fn rmsnorm(&self, out: &mut CudaSlice<f32>, x: &CudaSlice<f32>, w: &CudaSlice<f32>, n: usize, eps: f32) -> R<()> {
        let n_i = n as i32;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let f = self.f_rmsnorm.clone();
        let mut b = self.stream.launch_builder(&f);
        b.arg(out);
        b.arg(x);
        b.arg(w);
        b.arg(&n_i);
        b.arg(&eps);
        unsafe { b.launch(cfg) }?;
        Ok(())
    }
    fn add(&self, a: &mut CudaSlice<f32>, bvec: &CudaSlice<f32>, n: usize) -> R<()> {
        let n_i = n as i32;
        let f = self.f_add.clone();
        let mut b = self.stream.launch_builder(&f);
        b.arg(a);
        b.arg(bvec);
        b.arg(&n_i);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(())
    }
    fn swiglu(&self, g: &mut CudaSlice<f32>, u: &CudaSlice<f32>, n: usize) -> R<()> {
        let n_i = n as i32;
        let f = self.f_swiglu.clone();
        let mut b = self.stream.launch_builder(&f);
        b.arg(g);
        b.arg(u);
        b.arg(&n_i);
        unsafe { b.launch(LaunchConfig::for_num_elems(n as u32)) }?;
        Ok(())
    }
    fn rope(&self, x: &mut CudaSlice<f32>, n_heads: usize, head_dim: usize, pos: usize, theta: f32) -> R<()> {
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let p = pos as i32;
        let cfg = LaunchConfig { grid_dim: (n_heads as u32, 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
        let f = self.f_rope.clone();
        let mut b = self.stream.launch_builder(&f);
        b.arg(x);
        b.arg(&nh);
        b.arg(&hd);
        b.arg(&p);
        b.arg(&theta);
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    // --- test helpers (host in/out) ---
    pub fn matmul_bytes(&self, ty: GgmlType, w: &[u8], in_dim: usize, out_dim: usize, x: &[f32]) -> R<Vec<f32>> {
        let gw = self.upload_weight(w, ty)?;
        let x_dev = self.stream.clone_htod(x)?;
        let mut y_dev = self.stream.alloc_zeros::<f32>(out_dim)?;
        self.mm(&gw, in_dim, out_dim, &x_dev, &mut y_dev)?;
        Ok(self.stream.clone_dtoh(&y_dev)?)
    }
}

/// Hyperparameters the GPU forward needs.
#[derive(Clone, Copy)]
pub struct GpuConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub n_ff: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub kv_cap: usize,
}

struct GpuLayer {
    attn_norm: CudaSlice<f32>,
    wq: GpuWeight,
    bq: CudaSlice<f32>,
    wk: GpuWeight,
    bk: CudaSlice<f32>,
    wv: GpuWeight,
    bv: CudaSlice<f32>,
    wo: GpuWeight,
    ffn_norm: CudaSlice<f32>,
    w_gate: GpuWeight,
    w_up: GpuWeight,
    w_down: GpuWeight,
}

/// Immutable model resident in VRAM. Build once, share across sessions.
pub struct GpuWeights {
    be: Arc<GpuBackend>,
    cfg: GpuConfig,
    layers: Vec<GpuLayer>,
    out_norm: CudaSlice<f32>,
    output: GpuWeight,
}

/// Per-generation mutable state: scratch buffers + KV cache in VRAM.
pub struct GpuSession {
    x: CudaSlice<f32>,
    xb: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    att: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    kcache: Vec<CudaSlice<f32>>,
    vcache: Vec<CudaSlice<f32>>,
}

/// Builder: feed weights in, get a GpuWeights out.
pub struct GpuWeightsBuilder {
    be: Arc<GpuBackend>,
    cfg: GpuConfig,
    layers: Vec<GpuLayer>,
    out_norm: Option<CudaSlice<f32>>,
    output: Option<GpuWeight>,
}

impl GpuWeightsBuilder {
    pub fn out_norm(&mut self, w: &[f32]) -> R<()> {
        self.out_norm = Some(self.be.upload_f32(w)?);
        Ok(())
    }
    pub fn output(&mut self, bytes: &[u8], ty: GgmlType) -> R<()> {
        self.output = Some(self.be.upload_weight(bytes, ty)?);
        Ok(())
    }
    /// Push one layer. Weight byte slices + their types, norm/bias f32 slices.
    #[allow(clippy::too_many_arguments)]
    pub fn layer(
        &mut self,
        attn_norm: &[f32],
        wq: (&[u8], GgmlType), bq: &[f32],
        wk: (&[u8], GgmlType), bk: &[f32],
        wv: (&[u8], GgmlType), bv: &[f32],
        wo: (&[u8], GgmlType),
        ffn_norm: &[f32],
        w_gate: (&[u8], GgmlType),
        w_up: (&[u8], GgmlType),
        w_down: (&[u8], GgmlType),
    ) -> R<()> {
        let be = &self.be;
        self.layers.push(GpuLayer {
            attn_norm: be.upload_f32(attn_norm)?,
            wq: be.upload_weight(wq.0, wq.1)?, bq: be.upload_f32(bq)?,
            wk: be.upload_weight(wk.0, wk.1)?, bk: be.upload_f32(bk)?,
            wv: be.upload_weight(wv.0, wv.1)?, bv: be.upload_f32(bv)?,
            wo: be.upload_weight(wo.0, wo.1)?,
            ffn_norm: be.upload_f32(ffn_norm)?,
            w_gate: be.upload_weight(w_gate.0, w_gate.1)?,
            w_up: be.upload_weight(w_up.0, w_up.1)?,
            w_down: be.upload_weight(w_down.0, w_down.1)?,
        });
        Ok(())
    }
    pub fn build(self) -> R<GpuWeights> {
        Ok(GpuWeights {
            be: self.be,
            cfg: self.cfg,
            layers: self.layers,
            out_norm: self.out_norm.ok_or("out_norm not set")?,
            output: self.output.ok_or("output not set")?,
        })
    }
}

impl GpuBackend {
    pub fn weights_builder(self: &Arc<Self>, cfg: GpuConfig) -> GpuWeightsBuilder {
        GpuWeightsBuilder {
            be: Arc::clone(self),
            cfg,
            layers: Vec::with_capacity(cfg.n_layer),
            out_norm: None,
            output: None,
        }
    }
}

impl GpuWeights {
    pub fn device_name(&self) -> String {
        self.be.device_name()
    }

    /// Allocate a fresh generation session (scratch + KV cache).
    pub fn new_session(&self) -> R<GpuSession> {
        let c = &self.cfg;
        let z = |n: usize| self.be.stream.alloc_zeros::<f32>(n);
        let mut kcache = Vec::with_capacity(c.n_layer);
        let mut vcache = Vec::with_capacity(c.n_layer);
        for _ in 0..c.n_layer {
            kcache.push(z(c.kv_cap * c.kv_dim)?);
            vcache.push(z(c.kv_cap * c.kv_dim)?);
        }
        Ok(GpuSession {
            x: z(c.n_embd)?,
            xb: z(c.n_embd)?,
            q: z(c.n_embd)?,
            k: z(c.kv_dim)?,
            v: z(c.kv_dim)?,
            att: z(c.n_embd)?,
            gate: z(c.n_ff)?,
            up: z(c.n_ff)?,
            logits: z(c.vocab)?,
            kcache,
            vcache,
        })
    }

    /// One token. `x0` is the token embedding (host, n_embd). Returns logits.
    pub fn forward(&self, s: &mut GpuSession, x0: &[f32], pos: usize) -> R<Vec<f32>> {
        let c = &self.cfg;
        let be = &self.be;
        if pos >= c.kv_cap {
            return Err(format!("kv cache full (cap {})", c.kv_cap).into());
        }
        be.stream.memcpy_htod(x0, &mut s.x)?;

        for (li, layer) in self.layers.iter().enumerate() {
            // attention
            be.rmsnorm(&mut s.xb, &s.x, &layer.attn_norm, c.n_embd, c.rms_eps)?;
            be.mm(&layer.wq, c.n_embd, c.n_embd, &s.xb, &mut s.q)?;
            be.add(&mut s.q, &layer.bq, c.n_embd)?;
            be.mm(&layer.wk, c.n_embd, c.kv_dim, &s.xb, &mut s.k)?;
            be.add(&mut s.k, &layer.bk, c.kv_dim)?;
            be.mm(&layer.wv, c.n_embd, c.kv_dim, &s.xb, &mut s.v)?;
            be.add(&mut s.v, &layer.bv, c.kv_dim)?;
            be.rope(&mut s.q, c.n_head, c.head_dim, pos, c.rope_theta)?;
            be.rope(&mut s.k, c.n_head_kv, c.head_dim, pos, c.rope_theta)?;

            // store k,v into cache at `pos`, then attend
            let off = pos * c.kv_dim;
            {
                let mut kdst = s.kcache[li].slice_mut(off..off + c.kv_dim);
                be.stream.memcpy_dtod(&s.k, &mut kdst)?;
            }
            {
                let mut vdst = s.vcache[li].slice_mut(off..off + c.kv_dim);
                be.stream.memcpy_dtod(&s.v, &mut vdst)?;
            }
            self.attend(s, li, pos)?;

            be.mm(&layer.wo, c.n_embd, c.n_embd, &s.att, &mut s.xb)?;
            be.add(&mut s.x, &s.xb, c.n_embd)?;

            // feed-forward
            be.rmsnorm(&mut s.xb, &s.x, &layer.ffn_norm, c.n_embd, c.rms_eps)?;
            be.mm(&layer.w_gate, c.n_embd, c.n_ff, &s.xb, &mut s.gate)?;
            be.mm(&layer.w_up, c.n_embd, c.n_ff, &s.xb, &mut s.up)?;
            be.swiglu(&mut s.gate, &s.up, c.n_ff)?;
            be.mm(&layer.w_down, c.n_ff, c.n_embd, &s.gate, &mut s.xb)?;
            be.add(&mut s.x, &s.xb, c.n_embd)?;
        }

        be.rmsnorm(&mut s.xb, &s.x, &self.out_norm, c.n_embd, c.rms_eps)?;
        be.mm(&self.output, c.n_embd, c.vocab, &s.xb, &mut s.logits)?;
        Ok(be.stream.clone_dtoh(&s.logits)?)
    }

    fn attend(&self, s: &mut GpuSession, layer: usize, pos: usize) -> R<()> {
        let c = &self.cfg;
        let n_pos = (pos + 1) as i32;
        let hd = c.head_dim as i32;
        let nh = c.n_head as i32;
        let nhkv = c.n_head_kv as i32;
        let kvd = c.kv_dim as i32;
        let cfg = LaunchConfig {
            grid_dim: (c.n_head as u32, 1, 1),
            block_dim: (c.head_dim as u32, 1, 1),
            shared_mem_bytes: ((pos + 1) * 4) as u32,
        };
        let f = self.be.f_attn.clone();
        let mut b = self.be.stream.launch_builder(&f);
        b.arg(&mut s.att);
        b.arg(&s.q);
        b.arg(&s.kcache[layer]);
        b.arg(&s.vcache[layer]);
        b.arg(&n_pos);
        b.arg(&hd);
        b.arg(&nh);
        b.arg(&nhkv);
        b.arg(&kvd);
        unsafe { b.launch(cfg) }?;
        Ok(())
    }
}

pub fn gpu_available() -> bool {
    CudaContext::new(0).is_ok()
}
pub fn device_name() -> R<String> {
    Ok(CudaContext::new(0)?.name()?)
}

/// Q8_0 GPU self-test (fixed fp16 scale) vs a CPU reference. Returns max abs error.
pub fn self_test_q8(in_dim: usize, out_dim: usize) -> R<f32> {
    let nb = in_dim / 32;
    let rb = nb * 34;
    let d = 0.03125f32;
    let mut w = vec![0u8; out_dim * rb];
    for j in 0..out_dim {
        for b in 0..nb {
            let base = j * rb + b * 34;
            w[base] = 0x00;
            w[base + 1] = 0x28;
            for i in 0..32 {
                w[base + 2 + i] = (((j + b + i) % 251) as i32 - 125) as i8 as u8;
            }
        }
    }
    let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let be = GpuBackend::new()?;
    let gpu = be.matmul_bytes(GgmlType::Q8_0, &w, in_dim, out_dim, &x)?;
    let mut max_err = 0f32;
    for j in 0..out_dim {
        let mut acc = 0f32;
        for b in 0..nb {
            let base = j * rb + b * 34;
            let mut blk = 0f32;
            for i in 0..32 {
                blk += (w[base + 2 + i] as i8 as f32) * x[b * 32 + i];
            }
            acc += d * blk;
        }
        max_err = max_err.max((acc - gpu[j]).abs());
    }
    Ok(max_err)
}

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> impl FnMut() -> u32 {
        let mut s: u32 = 0x1234_5678;
        move || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s }
    }
    fn fp16_bytes(r: &mut impl FnMut() -> u32) -> [u8; 2] {
        let exp = 8 + (r() % 11);
        let mant = r() & 0x3ff;
        let h = ((exp << 10) | mant) as u16;
        [(h & 0xff) as u8, (h >> 8) as u8]
    }
    fn check(ty: GgmlType, block_bytes: usize, block_elems: usize) {
        if !gpu_available() { return; }
        let (in_dim, out_dim) = (block_elems * 3, 200);
        let nb = in_dim / block_elems;
        let row = nb * block_bytes;
        let mut r = rng();
        let mut w = vec![0u8; out_dim * row];
        for byte in w.iter_mut() { *byte = (r() & 0xff) as u8; }
        for j in 0..out_dim {
            for b in 0..nb {
                let base = j * row + b * block_bytes;
                match ty {
                    GgmlType::Q4K => {
                        let d = fp16_bytes(&mut r); let dm = fp16_bytes(&mut r);
                        w[base..base + 2].copy_from_slice(&d);
                        w[base + 2..base + 4].copy_from_slice(&dm);
                    }
                    GgmlType::Q6K => {
                        let d = fp16_bytes(&mut r);
                        w[base + 208..base + 210].copy_from_slice(&d);
                    }
                    _ => {}
                }
            }
        }
        let x: Vec<f32> = (0..in_dim).map(|_| (r() % 1000) as f32 / 1000.0 - 0.5).collect();
        let be = GpuBackend::new().expect("gpu");
        let gpu = be.matmul_bytes(ty, &w, in_dim, out_dim, &x).expect("gpu matmul");
        let mut cpu = vec![0f32; out_dim];
        poezi_tensor::matmul(ty, &w, in_dim, out_dim, &x, &mut cpu);
        let mut mr = 0f32;
        for j in 0..out_dim {
            let d = cpu[j].abs().max(1.0);
            mr = mr.max((cpu[j] - gpu[j]).abs() / d);
        }
        assert!(mr < 1e-3, "{ty:?}: max rel err {mr}");
    }
    #[test] fn gpu_q8_matches_cpu() { check(GgmlType::Q8_0, 34, 32); }
    #[test] fn gpu_q4k_matches_cpu() { check(GgmlType::Q4K, 144, 256); }
    #[test] fn gpu_q6k_matches_cpu() { check(GgmlType::Q6K, 210, 256); }
}
