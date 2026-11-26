//! Signed distance field to a closed, quad-dominant mesh with Naive Surface Nets.
//!
//! Algorithm after Gibson, "Constrained Elastic Surface Nets" (1998), and Mikola Lysenko's write-up
//! (https://0fps.net/2012/07/12/smooth-voxel-terrain-part-2/, MIT-licensed reference code). The Rust crate
//! `fast-surface-nets` (MIT/Apache-2.0) uses the same scheme; this is an independent implementation that
//! adds a Newton projection of each vertex onto the zero set so shapes stay true at coarse resolutions.
//!
//! Output faces are quads, wound counter-clockwise seen from outside, which is what Sculptor's n-gon meshes
//! and edit tools expect. Every directed edge is matched by its reverse, so the mesh is closed.

use super::sdf::{Aabb, BlockClass, Compiled, F32x4, FlatSdf};
use crate::format::Mesh;
use glam::Vec3;
use std::cell::RefCell;

/// Samples above this are refused (about 12 MB of f32 field).
pub const MAX_SAMPLES: usize = 3_000_000;

pub struct Grid {
    pub origin: Vec3,
    pub h: f32,
    pub n: [usize; 3],
}

impl Grid {
    /// A grid with cell size `h` covering `b`, padded so the surface never touches the border.
    pub fn covering(b: &Aabb, h: f32) -> Grid {
        let pad = 2.0 * h;
        let min = b.min - Vec3::splat(pad);
        let size = b.size() + Vec3::splat(2.0 * pad);
        let n = [(size.x / h).ceil() as usize + 1, (size.y / h).ceil() as usize + 1, (size.z / h).ceil() as usize + 1];
        Grid { origin: min, h, n }
    }
    pub fn samples(&self) -> usize {
        self.n[0] * self.n[1] * self.n[2]
    }
}

/// Block edge (in samples) of the narrow-band pruning grid shared by the SDF fill and the cell scan.
const BLK: usize = 4;
/// Marks a grid cell the surface does not cross.
const EMPTY: u64 = u64::MAX;

/// Unboxed Surface Nets output: avoids per-quad `Vec<u32>` allocations until manifold and budget checks pass.
pub(crate) struct RawMesh {
    pub positions: Vec<Vec3>,
    pub quads: Vec<[u32; 4]>,
    /// True when no crossing cell had an ambiguous 4-crossing face or touched the grid border.
    /// In that case every cell face has 0 or 2 crossings, so every directed quad edge is paired
    /// with its exact reverse and the mesh is mathematically guaranteed watertight & manifold.
    pub guaranteed_manifold: bool,
}

/// Bit 47 of a cell record: set only when a cell has >1 component (or dynamic saddle components),
/// so the 99.9% single-component fast path skips all `TABLES.comp` lookups.
const MULTI_COMP_BIT: u64 = 1u64 << 47;
/// Bits 56..62 of a cell record: 6-bit mask where bit `fi` is set iff face `fi` of this cell has
/// 4 crossings whose two asymptotic-decider pairs merged into the same component of this cell.
const MERGED_FACE_SHIFT: u32 = 56;

/// Per-strip reusable L1-resident scratch for single-phase `surface_nets_dynpar`.
#[cfg(feature = "threads")]
#[derive(Default)]
struct StripScratch {
    slab_cur: Vec<f32>,
    slab_prev_last: Vec<f32>,
    signs_cur: Vec<u64>,
    signs_prev_last: Vec<u64>,
    act_cur: Vec<u8>,
    dcs_cur: Vec<f32>,
    act_prev: Vec<u8>,
    scan_blocks: Vec<u8>,
    positions: Vec<Vec3>,
    quads: Vec<[u32; 4]>,
    proj_ids: Vec<u32>,
    proj_grads: Vec<Vec3>,
    recs_prev: Vec<u64>,
    recs_cur: Vec<u64>,
    cells_prev: Vec<u32>,
    cells_cur: Vec<u32>,
    dyn_prev: Vec<[u8; 12]>,
    dyn_cur: Vec<[u8; 12]>,
    /// Saved first cell layer (`k == k0`, when `s > 0`) for inter-strip X/Y quad stitching.
    recs_first: Vec<u64>,
    cells_first: Vec<(u32, u8)>,
    dyn_first: Vec<[u8; 12]>,
    /// Saved last cell layer (`k == k1 - 1`, when `s + 1 < n_strips`) for inter-strip stitching.
    recs_last: Vec<u64>,
    cells_last: Vec<u32>,
    dyn_last: Vec<[u8; 12]>,
    guaranteed_manifold: bool,
}

/// Reusable thread-local scratch buffers so small/medium grids perform zero temporary heap allocations.
#[derive(Default)]
struct SweepScratch {
    slab_cur: Vec<f32>,
    slab_prev_last: Vec<f32>,
    signs_cur: Vec<u64>,
    signs_prev_last: Vec<u64>,
    act_super: [Vec<u8>; 2],
    dcs_super: [Vec<f32>; 2],
    act_prev: Vec<u8>,
    scan_blocks: Vec<u8>,
    recs_prev: Vec<u64>,
    recs_cur: Vec<u64>,
    cells_prev: Vec<u32>,
    cells_cur: Vec<u32>,
    dyn_prev: Vec<[u8; 12]>,
    dyn_cur: Vec<[u8; 12]>,
    proj_ids: Vec<u32>,
    proj_grads: Vec<Vec3>,
    man_cursor: Vec<u32>,
    man_keys: Vec<u32>,
    #[cfg(feature = "threads")]
    par_strips: Vec<StripScratch>,
}

thread_local! {
    static SWEEP: RefCell<SweepScratch> = RefCell::new(SweepScratch::default());
}

#[cfg(feature = "threads")]
struct SyncPtr<T>(*mut T);
#[cfg(feature = "threads")]
impl<T> Clone for SyncPtr<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
#[cfg(feature = "threads")]
impl<T> Copy for SyncPtr<T> {}
#[cfg(feature = "threads")]
unsafe impl<T> Send for SyncPtr<T> {}
#[cfg(feature = "threads")]
unsafe impl<T> Sync for SyncPtr<T> {}
#[cfg(feature = "threads")]
impl<T> SyncPtr<T> {
    #[inline(always)]
    fn get(&self) -> *mut T {
        self.0
    }
}

/// Ultra-low-latency (~0.3 µs wakeup) caller-participating persistent P-core worker pool.
/// Workers spin briefly (~200 µs) between bursts and then park asleep on a `Condvar` (0% idle CPU).
/// Uses a Dekker/RCU epoch fence (`epoch.store(0, SeqCst)` + `readers == 0`) so stack-borrowed
/// closures are 100% immune to use-after-free or ABA across OS preemptions.
#[cfg(feature = "threads")]
mod fast_par {
    use std::cell::UnsafeCell;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex, OnceLock};

    const SLOTS: usize = 8;

    struct Slot {
        epoch: AtomicUsize,
        readers: AtomicUsize,
        next: AtomicUsize,
        done: AtomicUsize,
        count: AtomicUsize,
        body: UnsafeCell<*const (dyn Fn(usize) + Sync + 'static)>,
    }

    impl Slot {
        fn new() -> Self {
            fn noop(_: usize) {}
            Self {
                epoch: AtomicUsize::new(0),
                readers: AtomicUsize::new(0),
                next: AtomicUsize::new(0),
                done: AtomicUsize::new(0),
                count: AtomicUsize::new(0),
                body: UnsafeCell::new(&noop as &(dyn Fn(usize) + Sync + 'static) as *const _),
            }
        }
    }

    struct Pool {
        in_use: AtomicBool,
        global_epoch: AtomicUsize,
        sleeping: AtomicUsize,
        park_mu: Mutex<()>,
        park_cv: Condvar,
        slots: [Slot; SLOTS],
        n_workers: usize,
    }
    unsafe impl Send for Pool {}
    unsafe impl Sync for Pool {}

    static POOL: OnceLock<&'static Pool> = OnceLock::new();

    fn get_pool() -> &'static Pool {
        *POOL.get_or_init(|| {
            let avail = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
            // Cap at 3 workers + 1 caller = 4 total threads (matches 4 P-cores on Apple Silicon, avoiding E-cores).
            let n_workers = avail.min(4).saturating_sub(1);
            let pool: &'static Pool = Box::leak(Box::new(Pool {
                in_use: AtomicBool::new(false),
                global_epoch: AtomicUsize::new(0),
                sleeping: AtomicUsize::new(0),
                park_mu: Mutex::new(()),
                park_cv: Condvar::new(),
                slots: std::array::from_fn(|_| Slot::new()),
                n_workers,
            }));
            for w in 0..n_workers {
                let _ = std::thread::Builder::new()
                    .name(format!("sdf-mesh-{w}"))
                    .spawn(move || worker_loop(pool));
            }
            pool
        })
    }

    fn worker_loop(pool: &'static Pool) {
        let mut last_e = 0usize;
        loop {
            let mut e = pool.global_epoch.load(Ordering::Acquire);
            if e == last_e {
                let mut spins = 0u32;
                while spins < 40_000 {
                    std::hint::spin_loop();
                    e = pool.global_epoch.load(Ordering::Acquire);
                    if e != last_e {
                        break;
                    }
                    spins += 1;
                }
                if e == last_e {
                    let mut guard = pool.park_mu.lock().unwrap();
                    pool.sleeping.fetch_add(1, Ordering::SeqCst);
                    loop {
                        e = pool.global_epoch.load(Ordering::SeqCst);
                        if e != last_e {
                            break;
                        }
                        guard = pool.park_cv.wait(guard).unwrap();
                    }
                    pool.sleeping.fetch_sub(1, Ordering::SeqCst);
                }
            }
            last_e = e;
            let slot = &pool.slots[e & (SLOTS - 1)];
            slot.readers.fetch_add(1, Ordering::SeqCst);
            if slot.epoch.load(Ordering::SeqCst) == e {
                let count = slot.count.load(Ordering::Relaxed);
                let body = unsafe { *slot.body.get() };
                loop {
                    let i = slot.next.fetch_add(1, Ordering::Relaxed);
                    if i >= count {
                        break;
                    }
                    unsafe { (*body)(i) };
                    slot.done.fetch_add(1, Ordering::Release);
                }
            }
            slot.readers.fetch_sub(1, Ordering::Release);
        }
    }

    struct AbortOnUnwind;
    impl Drop for AbortOnUnwind {
        fn drop(&mut self) {
            if std::thread::panicking() {
                std::process::abort();
            }
        }
    }

    pub fn for_each(n: usize, body: &(dyn Fn(usize) + Sync + '_)) {
        if n <= 1 {
            if n == 1 {
                body(0);
            }
            return;
        }
        let pool = get_pool();
        if pool.n_workers == 0
            || pool.in_use.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err()
        {
            for i in 0..n {
                body(i);
            }
            return;
        }
        let guard = AbortOnUnwind;
        let e = pool.global_epoch.load(Ordering::Relaxed).wrapping_add(1).max(1);
        let slot = &pool.slots[e & (SLOTS - 1)];
        while slot.readers.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        let erased = unsafe {
            std::mem::transmute::<&(dyn Fn(usize) + Sync + '_), &'static (dyn Fn(usize) + Sync + 'static)>(body) as *const _
        };
        unsafe { *slot.body.get() = erased };
        slot.count.store(n, Ordering::Relaxed);
        slot.next.store(0, Ordering::Relaxed);
        slot.done.store(0, Ordering::Relaxed);
        slot.epoch.store(e, Ordering::Release);
        pool.global_epoch.store(e, Ordering::SeqCst);
        if pool.sleeping.load(Ordering::SeqCst) > 0 {
            let _g = pool.park_mu.lock().unwrap();
            pool.park_cv.notify_all();
        }
        // Caller participates immediately (zero wakeup wait!).
        loop {
            let i = slot.next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            body(i);
            slot.done.fetch_add(1, Ordering::Release);
        }
        while slot.done.load(Ordering::Acquire) < n {
            std::hint::spin_loop();
        }
        // Dekker/RCU fence: close slot epoch then wait for any late reader to exit before ending `body` borrow.
        slot.epoch.store(0, Ordering::SeqCst);
        while slot.readers.load(Ordering::SeqCst) != 0 {
            std::hint::spin_loop();
        }
        pool.in_use.store(false, Ordering::Release);
        drop(guard);
    }
}

#[cfg(feature = "threads")]
pub(crate) fn fast_par_for_each(n: usize, body: &(dyn Fn(usize) + Sync + '_)) {
    fast_par::for_each(n, body);
}

/// Branchless single-component Surface Nets cell vertex (local coords in `[0,1]^3`) + trilinear gradient.
/// Uses 3 pipelined 4-wide NEON divisions on aarch64 (zero `while` loops, zero branch mispredictions).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn cell_local_and_grad_single(v: &[f32; 8], mask: u8, inv_h: f32) -> (Vec3, Vec3) {
    use core::arch::aarch64::*;
    unsafe {
        let vz0 = vld1q_f32(v.as_ptr()); // [v0, v1, v2, v3]
        let vz1 = vld1q_f32(v.as_ptr().add(4)); // [v4, v5, v6, v7]

        // X-edges: (0,1), (2,3), (4,5), (6,7). `vandq(mx, xa / dx)` zeroes non-crossing lanes
        // bitwise (so NaN/Inf on `dx == 0` never leaks) and `t = -(xa / dx)` avoids `vnegq`+`vbslq`.
        let xa = vuzp1q_f32(vz0, vz1);
        let xb = vuzp2q_f32(vz0, vz1);
        let mx = veorq_u32(vcltzq_f32(xa), vcltzq_f32(xb));
        let dx = vsubq_f32(xb, xa); // [v1-v0, v3-v2, v5-v4, v7-v6]
        let tx = vreinterpretq_f32_u32(vandq_u32(mx, vreinterpretq_u32_f32(vdivq_f32(xa, dx))));

        // Y-edges: (0,2), (1,3), (4,6), (5,7)
        let ya = vcombine_f32(vget_low_f32(vz0), vget_low_f32(vz1));
        let yb = vcombine_f32(vget_high_f32(vz0), vget_high_f32(vz1));
        let my = veorq_u32(vcltzq_f32(ya), vcltzq_f32(yb));
        let dy = vsubq_f32(yb, ya); // [v2-v0, v3-v1, v6-v4, v7-v5]
        let ty = vreinterpretq_f32_u32(vandq_u32(my, vreinterpretq_u32_f32(vdivq_f32(ya, dy))));

        // Z-edges: (0,4), (1,5), (2,6), (3,7)
        let mz = veorq_u32(vcltzq_f32(vz0), vcltzq_f32(vz1));
        let dz = vsubq_f32(vz1, vz0); // [v4-v0, v5-v1, v6-v2, v7-v3]
        let tz = vreinterpretq_f32_u32(vandq_u32(mz, vreinterpretq_u32_f32(vdivq_f32(vz0, dz))));

        let base = TABLES.base_sum[mask as usize];
        let inv_c = TABLES.inv_cross_count[mask as usize];
        let u1 = (base.x - vaddvq_f32(tx)) * inv_c;
        let w1 = (base.y - vaddvq_f32(ty)) * inv_c;
        let t1 = (base.z - vaddvq_f32(tz)) * inv_c;

        let (u0, w0, t0) = (1.0 - u1, 1.0 - w1, 1.0 - t1);
        let wx = vld1q_f32([w0 * t0, w1 * t0, w0 * t1, w1 * t1].as_ptr());
        let wy = vld1q_f32([u0 * t0, u1 * t0, u0 * t1, u1 * t1].as_ptr());
        let wz = vld1q_f32([u0 * w0, u1 * w0, u0 * w1, u1 * w1].as_ptr());
        let gx = vaddvq_f32(vmulq_f32(dx, wx)) * inv_h;
        let gy = vaddvq_f32(vmulq_f32(dy, wy)) * inv_h;
        let gz = vaddvq_f32(vmulq_f32(dz, wz)) * inv_h;
        (Vec3::new(u1, w1, t1), Vec3::new(gx, gy, gz))
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn cell_local_and_grad_single(v: &[f32; 8], mask: u8, inv_h: f32) -> (Vec3, Vec3) {
    let bits = TABLES.cross[mask as usize];
    let mut sum = TABLES.base_sum[mask as usize];
    let mut bx = bits & 0x00F;
    while bx != 0 {
        let e = bx.trailing_zeros() as usize;
        bx &= bx - 1;
        let (ea, eb) = EDGES[e];
        sum.x += v[ea] / (v[ea] - v[eb]);
    }
    let mut by = bits & 0x0F0;
    while by != 0 {
        let e = by.trailing_zeros() as usize;
        by &= by - 1;
        let (ea, eb) = EDGES[e];
        sum.y += v[ea] / (v[ea] - v[eb]);
    }
    let mut bz = bits & 0xF00;
    while bz != 0 {
        let e = bz.trailing_zeros() as usize;
        bz &= bz - 1;
        let (ea, eb) = EDGES[e];
        sum.z += v[ea] / (v[ea] - v[eb]);
    }
    let local = sum * TABLES.inv_cross_count[mask as usize];
    (local, cell_trilinear_grad(v, local, inv_h))
}

/// Trilinear gradient of the SDF inside a cell from its 8 corner values at local coords `u` in `[0, 1]^3`.
#[inline(always)]
fn cell_trilinear_grad(v: &[f32; 8], u: Vec3, inv_h: f32) -> Vec3 {
    let (u1, w1, t1) = (u.x, u.y, u.z);
    let (u0, w0, t0) = (1.0 - u1, 1.0 - w1, 1.0 - t1);
    let gx = ((v[1] - v[0]) * w0 + (v[3] - v[2]) * w1) * t0 + ((v[5] - v[4]) * w0 + (v[7] - v[6]) * w1) * t1;
    let gy = ((v[2] - v[0]) * u0 + (v[3] - v[1]) * u1) * t0 + ((v[6] - v[4]) * u0 + (v[7] - v[5]) * u1) * t1;
    let gz = ((v[4] - v[0]) * u0 + (v[5] - v[1]) * u1) * w0 + ((v[6] - v[2]) * u0 + (v[7] - v[3]) * u1) * w1;
    Vec3::new(gx, gy, gz) * inv_h
}

/// Batched 4-wide SIMD Newton projection of single-component vertices onto the zero isosurface.
/// Uses the flattened `FlatSdf` register-stack tape when available (including for tetrahedral
/// gradients on crease/rim vertices) and densely packs selective 2nd-pass candidates into full
/// 4-wide SIMD batches so sparse crease vertices never waste SIMD lanes.
#[inline(always)]
fn project_vertices_inner<E: Fn([Vec3; 4]) -> [f32; 4]>(
    eval4: E,
    h: f32,
    ids: &[u32],
    grads: &[Vec3],
    positions: &mut [Vec3],
) {
    const K0: Vec3 = Vec3::new(1.0, -1.0, -1.0);
    const K1: Vec3 = Vec3::new(-1.0, -1.0, 1.0);
    const K2: Vec3 = Vec3::new(-1.0, 1.0, -1.0);
    const K3: Vec3 = Vec3::new(1.0, 1.0, 1.0);

    let h2 = h * h;
    let eps = 1e-5 * h;
    let refine_eps = 0.018 * h;
    let crease_thresh = 0.08 * h;
    let e1 = h * 0.25;
    let inv_4e1 = 0.25 / e1;
    let e2 = h * 0.20;
    let inv_4e2 = 0.25 / e2;

    let grad_tetra = |p: Vec3, e: f32, inv_4e: f32| -> Vec3 {
        let d = eval4([p + K0 * e, p + K1 * e, p + K2 * e, p + K3 * e]);
        (K0 * d[0] + K1 * d[1] + K2 * d[2] + K3 * d[3]) * inv_4e
    };

    let mut ref_ids = [0u32; 4];
    let mut ref_starts = [Vec3::ZERO; 4];
    let mut ref_p1s = [Vec3::ZERO; 4];
    let mut ref_len = 0usize;

    let flush_refine = |len: usize,
                        r_ids: &[u32; 4],
                        r_starts: &[Vec3; 4],
                        r_p1s: &[Vec3; 4],
                        positions: &mut [Vec3]| {
        let mut q = *r_p1s;
        for l in len..4 {
            q[l] = q[0];
        }
        let d1 = eval4(q);
        for l in 0..len {
            let d = d1[l];
            if d.abs() <= refine_eps {
                continue;
            }
            let start = r_starts[l];
            let p1 = r_p1s[l];
            let gt = grad_tetra(p1, e2, inv_4e2);
            let gt2 = gt.length_squared();
            if gt2 >= 1e-12 {
                let cur = p1 - gt * (d / gt2);
                if cur.is_finite() && (cur - start).length_squared() <= h2 {
                    positions[r_ids[l] as usize] = cur;
                }
            }
        }
    };

    let np = ids.len();
    let mut idx = 0usize;
    while idx < np {
        let n4 = (np - idx).min(4);
        let p0 = positions[ids[idx] as usize];
        let mut pts = [p0; 4];
        for l in 1..n4 {
            pts[l] = positions[ids[idx + l] as usize];
        }
        let d0 = eval4(pts);
        for l in 0..n4 {
            let d = d0[l];
            if d.abs() <= eps {
                continue;
            }
            let vid = ids[idx + l];
            let start = pts[l];
            let g = grads[idx + l];
            let g2 = g.length_squared();
            let mut moved = false;
            let mut cur_pos = start;
            let mut need_refine = false;
            // On crease/edge cells the trilinear gradient can shrink well below unit length;
            // only trust it when well-conditioned (`g2 >= 0.45`), otherwise use exact tetra gradient.
            if g2 >= 0.45 {
                let s = d / g2;
                if d * s <= h2 {
                    let cur = start - g * s;
                    if cur.is_finite() {
                        cur_pos = cur;
                        positions[vid as usize] = cur;
                        moved = true;
                        need_refine = d.abs() > crease_thresh || g2 < 0.72;
                    }
                }
            }
            if !moved {
                let gt = grad_tetra(start, e1, inv_4e1);
                let gt2 = gt.length_squared();
                if gt2 >= 1e-12 {
                    let s = d / gt2;
                    if d * s <= h2 {
                        let cur = start - gt * s;
                        if cur.is_finite() {
                            cur_pos = cur;
                            positions[vid as usize] = cur;
                            need_refine = true;
                        }
                    }
                }
            }
            if need_refine {
                ref_ids[ref_len] = vid;
                ref_starts[ref_len] = start;
                ref_p1s[ref_len] = cur_pos;
                ref_len += 1;
                if ref_len == 4 {
                    flush_refine(4, &ref_ids, &ref_starts, &ref_p1s, positions);
                    ref_len = 0;
                }
            }
        }
        idx += 4;
    }
    if ref_len > 0 {
        flush_refine(ref_len, &ref_ids, &ref_starts, &ref_p1s, positions);
    }
}

fn project_vertices_batched(f: &Compiled, flat: Option<&FlatSdf>, h: f32, ids: &[u32], grads: &[Vec3], positions: &mut [Vec3]) {
    if let Some(tape) = flat {
        project_vertices_inner(|pts| tape.eval4_pts(pts), h, ids, grads, positions);
    } else {
        project_vertices_inner(|pts| f.eval4_pts(pts), h, ids, grads, positions);
    }
}

/// Classifies one z-slab (`k0..k1`, up to `BLK = 4` slices) into `act[..nblocks]` and `dcs[..nblocks]`
/// using an 8×8×4 -> 4×4×4 hierarchy with zero heap allocations.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "threads")]
fn classify_slab_range(
    f: &Compiled,
    origin: Vec3,
    h: f32,
    nx: usize,
    ny: usize,
    nbx: usize,
    nby: usize,
    k0: usize,
    k1: usize,
    act: &mut [u8],
    dcs: &mut [f32],
) -> (bool, bool, bool) {
    let mut flags = (false, true, true);
    let z0 = origin.z + k0 as f32 * h;
    let z1 = origin.z + (k1 - 1) as f32 * h;
    let margin = h * 1.05;
    let sbx_count = nbx.div_ceil(2);
    let sby_count = nby.div_ceil(2);

    for sbj in 0..sby_count {
        let bj0 = sbj * 2;
        let bj_end = (bj0 + 2).min(nby);
        let sj0 = bj0 * BLK;
        let sj1 = (bj_end * BLK).min(ny);
        let sy0 = origin.y + sj0 as f32 * h;
        let sy1 = origin.y + (sj1 - 1) as f32 * h;
        for sbi in 0..sbx_count {
            let bi0 = sbi * 2;
            let bi_end = (bi0 + 2).min(nbx);
            let si0 = bi0 * BLK;
            let si1 = (bi_end * BLK).min(nx);
            let sx0 = origin.x + si0 as f32 * h;
            let sx1 = origin.x + (si1 - 1) as f32 * h;
            match f.classify_interval(Vec3::new(sx0, sy0, z0), Vec3::new(sx1, sy1, z1), margin) {
                BlockClass::Empty(fill) => {
                    flags.2 = false;
                    for bj in bj0..bj_end {
                        let base = bj * nbx;
                        for bi in bi0..bi_end {
                            act[base + bi] = 0;
                            dcs[base + bi] = fill;
                        }
                    }
                    continue;
                }
                BlockClass::Solid(fill) => {
                    flags.1 = false;
                    for bj in bj0..bj_end {
                        let base = bj * nbx;
                        for bi in bi0..bi_end {
                            act[base + bi] = 2;
                            dcs[base + bi] = fill;
                        }
                    }
                    continue;
                }
                BlockClass::Band => {}
            }
            for bj in bj0..bj_end {
                let j0 = bj * BLK;
                let j1 = (j0 + BLK).min(ny);
                let y0 = origin.y + j0 as f32 * h;
                let y1 = origin.y + (j1 - 1) as f32 * h;
                let base = bj * nbx;
                for bi in bi0..bi_end {
                    let i0 = bi * BLK;
                    let i1 = (i0 + BLK).min(nx);
                    let x0 = origin.x + i0 as f32 * h;
                    let x1 = origin.x + (i1 - 1) as f32 * h;
                    match f.classify_interval(Vec3::new(x0, y0, z0), Vec3::new(x1, y1, z1), margin) {
                        BlockClass::Empty(fill) => {
                            flags.2 = false;
                            act[base + bi] = 0;
                            dcs[base + bi] = fill;
                        }
                        BlockClass::Solid(fill) => {
                            flags.1 = false;
                            act[base + bi] = 2;
                            dcs[base + bi] = fill;
                        }
                        BlockClass::Band => {
                            flags.0 = true;
                            flags.1 = false;
                            flags.2 = false;
                            act[base + bi] = 1;
                            dcs[base + bi] = 0.0;
                        }
                    }
                }
            }
        }
    }
    flags
}

/// Evaluates an `8x8x8` -> `4x4x4` hierarchical super-slab (1 or 2 z-slabs of `BLK = 4` slices each),
/// filling `act[0..nsub]` and `dcs[0..nsub]` and returning `(any, all_pos, all_neg)` per sub-slab.
#[allow(clippy::too_many_arguments)]
fn classify_super_slab(
    f: &Compiled,
    origin: Vec3,
    h: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    nbx: usize,
    nby: usize,
    bk0: usize,
    nsub: usize,
    act: &mut [Vec<u8>; 2],
    dcs: &mut [Vec<f32>; 2],
) -> [(bool, bool, bool); 2] {
    let mut flags = [(false, true, true); 2];
    let sbk_k0 = bk0 * BLK;
    let sbk_k1 = (sbk_k0 + nsub * BLK).min(nz);
    let sz0 = origin.z + sbk_k0 as f32 * h;
    let sz1 = origin.z + (sbk_k1 - 1) as f32 * h;
    let margin = h * 1.05;
    let sbx_count = nbx.div_ceil(2);
    let sby_count = nby.div_ceil(2);

    for sbj in 0..sby_count {
        let bj0 = sbj * 2;
        let bj_end = (bj0 + 2).min(nby);
        let sj0 = bj0 * BLK;
        let sj1 = (bj_end * BLK).min(ny);
        let sy0 = origin.y + sj0 as f32 * h;
        let sy1 = origin.y + (sj1 - 1) as f32 * h;
        for sbi in 0..sbx_count {
            let bi0 = sbi * 2;
            let bi_end = (bi0 + 2).min(nbx);
            let si0 = bi0 * BLK;
            let si1 = (bi_end * BLK).min(nx);
            let sx0 = origin.x + si0 as f32 * h;
            let sx1 = origin.x + (si1 - 1) as f32 * h;
            match f.classify_interval(Vec3::new(sx0, sy0, sz0), Vec3::new(sx1, sy1, sz1), margin) {
                BlockClass::Empty(fill) => {
                    for s in 0..nsub {
                        flags[s].2 = false;
                        for bj in bj0..bj_end {
                            let base = bj * nbx;
                            for bi in bi0..bi_end {
                                act[s][base + bi] = 0;
                                dcs[s][base + bi] = fill;
                            }
                        }
                    }
                    continue;
                }
                BlockClass::Solid(fill) => {
                    for s in 0..nsub {
                        flags[s].1 = false;
                        for bj in bj0..bj_end {
                            let base = bj * nbx;
                            for bi in bi0..bi_end {
                                act[s][base + bi] = 2;
                                dcs[s][base + bi] = fill;
                            }
                        }
                    }
                    continue;
                }
                BlockClass::Band => {}
            }
            // Refine into child 4x4x4 blocks.
            for s in 0..nsub {
                let k0 = (bk0 + s) * BLK;
                let k1 = (k0 + BLK).min(nz);
                let z0 = origin.z + k0 as f32 * h;
                let z1 = origin.z + (k1 - 1) as f32 * h;
                for bj in bj0..bj_end {
                    let j0 = bj * BLK;
                    let j1 = (j0 + BLK).min(ny);
                    let y0 = origin.y + j0 as f32 * h;
                    let y1 = origin.y + (j1 - 1) as f32 * h;
                    let base = bj * nbx;
                    for bi in bi0..bi_end {
                        let i0 = bi * BLK;
                        let i1 = (i0 + BLK).min(nx);
                        let x0 = origin.x + i0 as f32 * h;
                        let x1 = origin.x + (i1 - 1) as f32 * h;
                        match f.classify_interval(Vec3::new(x0, y0, z0), Vec3::new(x1, y1, z1), margin) {
                            BlockClass::Empty(fill) => {
                                flags[s].2 = false;
                                act[s][base + bi] = 0;
                                dcs[s][base + bi] = fill;
                            }
                            BlockClass::Solid(fill) => {
                                flags[s].1 = false;
                                act[s][base + bi] = 2;
                                dcs[s][base + bi] = fill;
                            }
                            BlockClass::Band => {
                                flags[s].0 = true;
                                flags[s].1 = false;
                                flags[s].2 = false;
                                act[s][base + bi] = 1;
                                dcs[s][base + bi] = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }
    flags
}

/// Fills one z-slab (`n_slices` slices starting at `k0`) using hoisted `eval_block4` per active 4×4×4 block,
/// optionally also populating bitpacked `signs` (`u64` per X-row when `nx <= 64`) in the same register pass.
#[allow(clippy::too_many_arguments)]
fn fill_slab(
    f: &Compiled,
    origin: Vec3,
    h: f32,
    nx: usize,
    ny: usize,
    nbx: usize,
    k0: usize,
    n_slices: usize,
    any: bool,
    all_pos: bool,
    all_neg: bool,
    act: &[u8],
    dcs: &[f32],
    slab: &mut [f32],
    mut signs: Option<&mut [u64]>,
) {
    let slice_len = nx * ny;
    let nby = ny.div_ceil(BLK);
    if !any && (all_pos || all_neg) {
        slab[..n_slices * slice_len].fill(if all_neg { -1.0 } else { 1.0 });
        if let Some(sb) = signs {
            let fill_bits = if all_neg { if nx == 64 { u64::MAX } else { (1u64 << nx) - 1 } } else { 0 };
            sb[..n_slices * ny].fill(fill_bits);
        }
        return;
    }
    if let Some(ref mut sb) = signs {
        sb[..n_slices * ny].fill(0);
    }
    let z0 = origin.z + k0 as f32 * h;
    let mut blk = [F32x4::splat(0.0); 16];
    for bj in 0..nby {
        let j0 = bj * BLK;
        let j1 = (j0 + BLK).min(ny);
        let y0 = origin.y + j0 as f32 * h;
        let act_row = &act[bj * nbx..][..nbx];
        let dcs_row = &dcs[bj * nbx..][..nbx];
        for bi in 0..nbx {
            let i0 = bi * BLK;
            let i1 = (i0 + BLK).min(nx);
            let a = act_row[bi];
            if a & 1 == 0 {
                let v = dcs_row[bi];
                for lk in 0..n_slices {
                    let z_off = lk * slice_len;
                    for j in j0..j1 {
                        slab[z_off + j * nx + i0..z_off + j * nx + i1].fill(v);
                    }
                }
                if a == 2 {
                    if let Some(ref mut sb) = signs {
                        let mask = ((1u64 << (i1 - i0)) - 1) << i0;
                        for lk in 0..n_slices {
                            let kb = lk * ny;
                            for j in j0..j1 {
                                sb[kb + j] |= mask;
                            }
                        }
                    }
                }
            } else {
                let x0 = origin.x + i0 as f32 * h;
                f.eval_block4(Vec3::new(x0, y0, z0), h, &mut blk);
                let full4 = i1 - i0 == 4;
                if let Some(ref mut sb) = signs {
                    let len_mask = (1u8 << (i1 - i0)) - 1;
                    for lk in 0..n_slices {
                        let z_off = lk * slice_len;
                        let lb = lk * 4;
                        let kb = lk * ny;
                        for lj in 0..(j1 - j0) {
                            let j = j0 + lj;
                            let v4 = blk[lb + lj];
                            let dst = &mut slab[z_off + j * nx + i0..];
                            if full4 {
                                v4.store(dst);
                            } else {
                                let arr = v4.to_array();
                                dst[..i1 - i0].copy_from_slice(&arr[..i1 - i0]);
                            }
                            sb[kb + j] |= ((v4.sign_mask4() & len_mask) as u64) << i0;
                        }
                    }
                } else {
                    for lk in 0..n_slices {
                        let z_off = lk * slice_len;
                        let lb = lk * 4;
                        for lj in 0..(j1 - j0) {
                            let j = j0 + lj;
                            let v4 = blk[lb + lj];
                            let dst = &mut slab[z_off + j * nx + i0..];
                            if full4 {
                                v4.store(dst);
                            } else {
                                let arr = v4.to_array();
                                dst[..i1 - i0].copy_from_slice(&arr[..i1 - i0]);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Computes `scan_blocks` for interior layers (within a single z-slab: 2x2 block stencil in XY).
fn build_scan_blocks_intra(nbx: usize, nby: usize, act: &[u8], out: &mut [u8]) -> bool {
    let mut any_scan = false;
    for bj in 0..nby {
        let j1 = (bj + 1).min(nby - 1);
        for bi in 0..nbx {
            let i1 = (bi + 1).min(nbx - 1);
            let a00 = act[bj * nbx + bi];
            let a10 = act[bj * nbx + i1];
            let a01 = act[j1 * nbx + bi];
            let a11 = act[j1 * nbx + i1];
            let band = ((a00 | a10 | a01 | a11) & 1) != 0;
            let neg = ((a00 | a10 | a01 | a11) & 2) != 0;
            let pos = ((a00 & a10 & a01 & a11) & 2) == 0;
            let s = (band || (neg && pos)) as u8;
            out[bj * nbx + bi] = s;
            any_scan |= s != 0;
        }
    }
    any_scan
}

/// Computes `scan_blocks` for a slab-boundary layer (2x2x2 block stencil across `act_prev` and `act_cur`).
fn build_scan_blocks_inter(nbx: usize, nby: usize, act_prev: &[u8], act_cur: &[u8], out: &mut [u8]) -> bool {
    let mut any_scan = false;
    for bj in 0..nby {
        let j1 = (bj + 1).min(nby - 1);
        for bi in 0..nbx {
            let i1 = (bi + 1).min(nbx - 1);
            let idx00 = bj * nbx + bi;
            let idx10 = bj * nbx + i1;
            let idx01 = j1 * nbx + bi;
            let idx11 = j1 * nbx + i1;
            let p_or = act_prev[idx00] | act_prev[idx10] | act_prev[idx01] | act_prev[idx11];
            let c_or = act_cur[idx00] | act_cur[idx10] | act_cur[idx01] | act_cur[idx11];
            let p_and = act_prev[idx00] & act_prev[idx10] & act_prev[idx01] & act_prev[idx11];
            let c_and = act_cur[idx00] & act_cur[idx10] & act_cur[idx01] & act_cur[idx11];
            let band = ((p_or | c_or) & 1) != 0;
            let neg = ((p_or | c_or) & 2) != 0;
            let pos = ((p_and & c_and) & 2) == 0;
            let s = (band || (neg && pos)) as u8;
            out[idx00] = s;
            any_scan |= s != 0;
        }
    }
    any_scan
}

#[inline(always)]
fn vert_lookup(recs: &[u64], dyn_comp: &[[u8; 12]], idx: usize, e: usize) -> u32 {
    let r = recs[idx];
    if r == EMPTY {
        return u32::MAX;
    }
    let base = r as u32;
    if (r & MULTI_COMP_BIT) == 0 {
        return base;
    }
    let tab = ((r >> 32) & 0x7FFF) as usize;
    let c = if tab < 256 { TABLES.comp[tab][e] } else { dyn_comp[tab - 256][e] };
    base + c as u32
}

/// Processes one crossing cell `(i, j, k)` with sign mask `mask` and 8 corner values `v`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn process_crossing_cell(
    i: usize,
    j: usize,
    me: usize,
    cx: usize,
    mask: u8,
    border: bool,
    emit_xy: bool,
    emit_z: bool,
    cell_min: Vec3,
    h: f32,
    inv_h: f32,
    v: &[f32; 8],
    recs_prev: &[u64],
    dyn_prev: &[[u8; 12]],
    recs_cur: &mut [u64],
    cells_cur: &mut Vec<u32>,
    dyn_cur: &mut Vec<[u8; 12]>,
    positions: &mut Vec<Vec3>,
    proj_ids: &mut Vec<u32>,
    proj_grads: &mut Vec<Vec3>,
    quads: &mut Vec<[u32; 4]>,
    guaranteed_manifold: &mut bool,
    record_first: &mut Option<&mut Vec<(u32, u8)>>,
) {
    if border {
        *guaranteed_manifold = false;
    }
    let base = positions.len() as u32;
    let rec: u64;
    if !TABLES.ambiguous[mask as usize] {
        let nc = TABLES.ncomp[mask as usize] as usize;
        if nc == 1 {
            let (local, grad) = cell_local_and_grad_single(v, mask, inv_h);
            let pos = cell_min + local * h;
            proj_ids.push(base);
            proj_grads.push(grad);
            positions.push(pos);
            rec = base as u64;
            recs_cur[me] = rec;
            cells_cur.push(me as u32);
            if let Some(rf) = record_first.as_deref_mut() {
                rf.push((me as u32, mask));
            }
            if emit_z {
                let inside = mask & 1 != 0;
                let mut push = |q: [u32; 4]| {
                    if q[0].max(q[1]).max(q[2]).max(q[3]) != u32::MAX {
                        quads.push(if inside { q } else { [q[0], q[3], q[2], q[1]] });
                    }
                };
                if emit_xy {
                    if j >= 1 && (mask ^ (mask >> 1)) & 1 != 0 {
                        push([
                            vert_lookup(recs_prev, dyn_prev, me - cx, ED[0][0]),
                            vert_lookup(recs_prev, dyn_prev, me, ED[0][1]),
                            base,
                            vert_lookup(recs_cur, dyn_cur, me - cx, ED[0][3]),
                        ]);
                    }
                    if i >= 1 && (mask ^ (mask >> 2)) & 1 != 0 {
                        push([
                            vert_lookup(recs_prev, dyn_prev, me - 1, ED[1][0]),
                            vert_lookup(recs_cur, dyn_cur, me - 1, ED[1][1]),
                            base,
                            vert_lookup(recs_prev, dyn_prev, me, ED[1][3]),
                        ]);
                    }
                }
                if i >= 1 && j >= 1 && (mask ^ (mask >> 4)) & 1 != 0 {
                    push([
                        vert_lookup(recs_cur, dyn_cur, me - cx - 1, ED[2][0]),
                        vert_lookup(recs_cur, dyn_cur, me - cx, ED[2][1]),
                        base,
                        vert_lookup(recs_cur, dyn_cur, me - 1, ED[2][3]),
                    ]);
                }
            }
            return;
        }
        let bits = TABLES.cross[mask as usize];
        let comp = &TABLES.comp[mask as usize];
        let mut sum = [Vec3::ZERO; 4];
        let mut cnt = [0f32; 4];
        let mut b = bits;
        while b != 0 {
            let e = b.trailing_zeros() as usize;
            b &= b - 1;
            let (ea, eb) = EDGES[e];
            let t = v[ea] / (v[ea] - v[eb]);
            let c = comp[e] as usize;
            sum[c] += EDGE_BASE[e] + EDGE_DIR[e] * t;
            cnt[c] += 1.0;
        }
        for c in 0..nc {
            positions.push(cell_min + sum[c] * (h / cnt[c]));
        }
        rec = base as u64 | (mask as u64) << 32 | MULTI_COMP_BIT;
    } else {
        let (comp, nc, merged_faces) = saddle_components(v, TABLES.cross[mask as usize]);
        // Exact manifoldness check: a 4-crossing face shared by two cells creates a non-manifold edge
        // iff BOTH cells merged its two asymptotic-decider pairs into the same component.
        if merged_faces != 0 {
            if emit_xy && (merged_faces & (1 << 0)) != 0 {
                let rp = recs_prev[me];
                if rp != EMPTY && ((rp >> MERGED_FACE_SHIFT) & (1 << 1)) != 0 {
                    *guaranteed_manifold = false;
                }
            }
            if j >= 1 && (merged_faces & (1 << 2)) != 0 {
                let ry = recs_cur[me - cx];
                if ry != EMPTY && ((ry >> MERGED_FACE_SHIFT) & (1 << 3)) != 0 {
                    *guaranteed_manifold = false;
                }
            }
            if i >= 1 && (merged_faces & (1 << 4)) != 0 {
                let rx = recs_cur[me - 1];
                if rx != EMPTY && ((rx >> MERGED_FACE_SHIFT) & (1 << 5)) != 0 {
                    *guaranteed_manifold = false;
                }
            }
        }
        let mut sum = [Vec3::ZERO; 4];
        let mut cnt = [0f32; 4];
        let mut bits = TABLES.cross[mask as usize];
        while bits != 0 {
            let e = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let (ea, eb) = EDGES[e];
            let t = v[ea] / (v[ea] - v[eb]);
            let c = comp[e] as usize;
            sum[c] += EDGE_BASE[e] + EDGE_DIR[e] * t;
            cnt[c] += 1.0;
        }
        for c in 0..nc {
            let local = sum[c] / cnt[c];
            let pos = cell_min + local * h;
            if nc == 1 {
                proj_ids.push(positions.len() as u32);
                proj_grads.push(cell_trilinear_grad(v, local, inv_h));
            }
            positions.push(pos);
        }
        let tab = 256 + dyn_cur.len() as u16;
        dyn_cur.push(comp);
        rec = base as u64 | (tab as u64) << 32 | MULTI_COMP_BIT | ((merged_faces as u64) << MERGED_FACE_SHIFT);
    }
    recs_cur[me] = rec;
    cells_cur.push(me as u32);
    if let Some(rf) = record_first.as_deref_mut() {
        rf.push((me as u32, mask));
    }
    if emit_z {
        let inside = mask & 1 != 0;
        let tab = ((rec >> 32) & 0x7FFF) as usize;
        let q2_comp: &[u8; 12] = if tab < 256 { &TABLES.comp[tab] } else { &dyn_cur[tab - 256] };
        let mut push = |q: [u32; 4]| {
            if q[0].max(q[1]).max(q[2]).max(q[3]) != u32::MAX {
                quads.push(if inside { q } else { [q[0], q[3], q[2], q[1]] });
            }
        };
        if emit_xy {
            if j >= 1 && (mask ^ (mask >> 1)) & 1 != 0 {
                push([
                    vert_lookup(recs_prev, dyn_prev, me - cx, ED[0][0]),
                    vert_lookup(recs_prev, dyn_prev, me, ED[0][1]),
                    base + q2_comp[ED[0][2]] as u32,
                    vert_lookup(recs_cur, dyn_cur, me - cx, ED[0][3]),
                ]);
            }
            if i >= 1 && (mask ^ (mask >> 2)) & 1 != 0 {
                push([
                    vert_lookup(recs_prev, dyn_prev, me - 1, ED[1][0]),
                    vert_lookup(recs_cur, dyn_cur, me - 1, ED[1][1]),
                    base + q2_comp[ED[1][2]] as u32,
                    vert_lookup(recs_prev, dyn_prev, me, ED[1][3]),
                ]);
            }
        }
        if i >= 1 && j >= 1 && (mask ^ (mask >> 4)) & 1 != 0 {
            push([
                vert_lookup(recs_cur, dyn_cur, me - cx - 1, ED[2][0]),
                vert_lookup(recs_cur, dyn_cur, me - cx, ED[2][1]),
                base + q2_comp[ED[2][2]] as u32,
                vert_lookup(recs_cur, dyn_cur, me - 1, ED[2][3]),
            ]);
        }
    }
}

/// Bitmask-accelerated cell layer scan for `nx <= 64`: finds every crossing cell in a row with 3 bitwise
/// instructions (zero float comparisons, zero empty/solid cell branches).
#[allow(clippy::too_many_arguments)]
fn scan_and_emit_layer_bits(
    k: usize,
    origin: Vec3,
    h: f32,
    inv_h: f32,
    nx: usize,
    cx: usize,
    cy: usize,
    cz: usize,
    s0: &[f32],
    s1: &[f32],
    signs0: &[u64],
    signs1: &[u64],
    emit_xy: bool,
    recs_prev: &[u64],
    dyn_prev: &[[u8; 12]],
    recs_cur: &mut [u64],
    cells_cur: &mut Vec<u32>,
    dyn_cur: &mut Vec<[u8; 12]>,
    positions: &mut Vec<Vec3>,
    proj_ids: &mut Vec<u32>,
    proj_grads: &mut Vec<Vec3>,
    quads: &mut Vec<[u32; 4]>,
    guaranteed_manifold: &mut bool,
    mut record_first: Option<&mut Vec<(u32, u8)>>,
) {
    let emit_z = k > 0;
    let k_border = k == 0 || k + 1 == cz;
    let z_pos = origin.z + k as f32 * h;
    let valid_cx = if cx >= 64 { u64::MAX } else { (1u64 << cx) - 1 };
    for j in 0..cy {
        let m00 = signs0[j];
        let m01 = signs0[j + 1];
        let m10 = signs1[j];
        let m11 = signs1[j + 1];
        let or_all = m00 | m01 | m10 | m11;
        let and_all = m00 & m01 & m10 & m11;
        let mut active = ((or_all | (or_all >> 1)) & !(and_all & (and_all >> 1))) & valid_cx;
        if active == 0 {
            continue;
        }
        let r0 = j * nx;
        let r1 = r0 + nx;
        let s0_r0 = &s0[r0..r0 + nx];
        let s0_r1 = &s0[r1..r1 + nx];
        let s1_r0 = &s1[r0..r0 + nx];
        let s1_r1 = &s1[r1..r1 + nx];
        let row_cell = j * cx;
        let j_border = k_border || j == 0 || j + 1 == cy;
        let y_pos = origin.y + j as f32 * h;
        while active != 0 {
            let i = active.trailing_zeros() as usize;
            active &= active - 1;
            let mask = (((m00 >> i) & 3)
                | (((m01 >> i) & 3) << 2)
                | (((m10 >> i) & 3) << 4)
                | (((m11 >> i) & 3) << 6)) as u8;
            let v = unsafe {
                [
                    *s0_r0.get_unchecked(i),
                    *s0_r0.get_unchecked(i + 1),
                    *s0_r1.get_unchecked(i),
                    *s0_r1.get_unchecked(i + 1),
                    *s1_r0.get_unchecked(i),
                    *s1_r0.get_unchecked(i + 1),
                    *s1_r1.get_unchecked(i),
                    *s1_r1.get_unchecked(i + 1),
                ]
            };
            let cell_min = Vec3::new(origin.x + i as f32 * h, y_pos, z_pos);
            let border = j_border || i == 0 || i + 1 == cx;
            process_crossing_cell(
                i, j, row_cell + i, cx, mask, border, emit_xy, emit_z, cell_min, h, inv_h, &v,
                recs_prev, dyn_prev, recs_cur, cells_cur, dyn_cur, positions, proj_ids, proj_grads,
                quads, guaranteed_manifold, &mut record_first,
            );
        }
    }
}

/// Scans one cell z-layer `k` from sample slices `s0` (z = k) and `s1` (z = k + 1), appends vertices
/// to `positions`, queues single-component vertices for batched projection, records cell entries in
/// `recs_cur`/`cells_cur`/`dyn_cur`, and emits quads inline as each cell is visited.
#[allow(clippy::too_many_arguments)]
fn scan_and_emit_layer(
    k: usize,
    origin: Vec3,
    h: f32,
    inv_h: f32,
    nx: usize,
    cx: usize,
    cy: usize,
    cz: usize,
    nbx: usize,
    scan_blocks: &[u8],
    s0: &[f32],
    s1: &[f32],
    emit_xy: bool,
    recs_prev: &[u64],
    dyn_prev: &[[u8; 12]],
    recs_cur: &mut [u64],
    cells_cur: &mut Vec<u32>,
    dyn_cur: &mut Vec<[u8; 12]>,
    positions: &mut Vec<Vec3>,
    proj_ids: &mut Vec<u32>,
    proj_grads: &mut Vec<Vec3>,
    quads: &mut Vec<[u32; 4]>,
    guaranteed_manifold: &mut bool,
    mut record_first: Option<&mut Vec<(u32, u8)>>,
) {
    let emit_z = k > 0;
    let k_border = k == 0 || k + 1 == cz;
    let z_pos = origin.z + k as f32 * h;
    for j in 0..cy {
        let act_row = &scan_blocks[(j / BLK) * nbx..][..nbx];
        let r0 = j * nx;
        let r1 = r0 + nx;
        let s0_r0 = &s0[r0..r0 + nx];
        let s0_r1 = &s0[r1..r1 + nx];
        let s1_r0 = &s1[r0..r0 + nx];
        let s1_r1 = &s1[r1..r1 + nx];
        let row_cell = j * cx;
        let j_border = k_border || j == 0 || j + 1 == cy;
        let y_pos = origin.y + j as f32 * h;
        let mut bi = 0;
        while bi < nbx {
            if act_row[bi] == 0 {
                bi += 1;
                continue;
            }
            let mut be = bi + 1;
            while be < nbx && act_row[be] != 0 {
                be += 1;
            }
            let (i0, i1) = (bi * BLK, (be * BLK).min(cx));
            bi = be;
            let mut left = ((s0_r0[i0] < 0.0) as u8)
                | (((s0_r1[i0] < 0.0) as u8) << 2)
                | (((s1_r0[i0] < 0.0) as u8) << 4)
                | (((s1_r1[i0] < 0.0) as u8) << 6);
            for i in i0..i1 {
                let right = ((s0_r0[i + 1] < 0.0) as u8)
                    | (((s0_r1[i + 1] < 0.0) as u8) << 2)
                    | (((s1_r0[i + 1] < 0.0) as u8) << 4)
                    | (((s1_r1[i + 1] < 0.0) as u8) << 6);
                let mask = left | (right << 1);
                left = right;
                if mask == 0 || mask == 255 {
                    continue;
                }
                let v = [
                    s0_r0[i], s0_r0[i + 1], s0_r1[i], s0_r1[i + 1],
                    s1_r0[i], s1_r0[i + 1], s1_r1[i], s1_r1[i + 1],
                ];
                let cell_min = Vec3::new(origin.x + i as f32 * h, y_pos, z_pos);
                let border = j_border || i == 0 || i + 1 == cx;
                process_crossing_cell(
                    i, j, row_cell + i, cx, mask, border, emit_xy, emit_z, cell_min, h, inv_h, &v,
                    recs_prev, dyn_prev, recs_cur, cells_cur, dyn_cur, positions, proj_ids, proj_grads,
                    quads, guaranteed_manifold, &mut record_first,
                );
            }
        }
    }
}

/// L1-cache-resident single-threaded Z-sweep Surface Nets using thread-local scratch buffers.
fn surface_nets_sweep(f: &Compiled, grid: &Grid) -> RawMesh {
    let [nx, ny, nz] = grid.n;
    if nx < 2 || ny < 2 || nz < 2 {
        return RawMesh { positions: Vec::new(), quads: Vec::new(), guaranteed_manifold: true };
    }
    let h = grid.h;
    let inv_h = 1.0 / h;
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);
    let slice_len = nx * ny;
    let (nbx, nby, nbz) = (nx.div_ceil(BLK), ny.div_ceil(BLK), nz.div_ceil(BLK));
    let nblocks = nbx * nby;
    let rec_len = cx * cy;
    let use_bits = nx <= 64;

    SWEEP.with(|cell| {
        let mut sc = cell.borrow_mut();
        if sc.slab_cur.len() < slice_len * BLK {
            sc.slab_cur.resize(slice_len * BLK, 0.0);
        }
        if sc.slab_prev_last.len() < slice_len {
            sc.slab_prev_last.resize(slice_len, 0.0);
        }
        if use_bits {
            if sc.signs_cur.len() < ny * BLK {
                sc.signs_cur.resize(ny * BLK, 0);
            }
            if sc.signs_prev_last.len() < ny {
                sc.signs_prev_last.resize(ny, 0);
            }
        }
        for s in 0..2 {
            if sc.act_super[s].len() < nblocks {
                sc.act_super[s].resize(nblocks, 0);
            }
            if sc.dcs_super[s].len() < nblocks {
                sc.dcs_super[s].resize(nblocks, 0.0);
            }
        }
        if sc.act_prev.len() < nblocks {
            sc.act_prev.resize(nblocks, 0);
        }
        if sc.scan_blocks.len() < nblocks {
            sc.scan_blocks.resize(nblocks, 0);
        }
        if sc.recs_prev.len() < rec_len {
            sc.recs_prev.resize(rec_len, EMPTY);
        }
        if sc.recs_cur.len() < rec_len {
            sc.recs_cur.resize(rec_len, EMPTY);
        }
        sc.cells_prev.clear();
        sc.cells_cur.clear();
        sc.dyn_prev.clear();
        sc.dyn_cur.clear();
        sc.proj_ids.clear();
        sc.proj_grads.clear();

        let est = (cx * cy).min(4096);
        let mut positions: Vec<Vec3> = Vec::with_capacity(est);
        let mut quads: Vec<[u32; 4]> = Vec::with_capacity(est + est / 4);
        let mut guaranteed_manifold = true;

        let SweepScratch {
            slab_cur,
            slab_prev_last,
            signs_cur,
            signs_prev_last,
            act_super,
            dcs_super,
            act_prev,
            scan_blocks,
            recs_prev,
            recs_cur,
            cells_prev,
            cells_cur,
            dyn_prev,
            dyn_cur,
            proj_ids,
            proj_grads,
            ..
        } = &mut *sc;

        let clear_prev = |recs_p: &mut [u64], cells_p: &mut Vec<u32>, dyn_p: &mut Vec<[u8; 12]>| {
            for &me in cells_p.iter() {
                recs_p[me as usize] = EMPTY;
            }
            cells_p.clear();
            dyn_p.clear();
        };

        let mut prev_all_pos = false;
        let mut prev_all_neg = false;
        let mut bk0 = 0usize;
        while bk0 < nbz {
            let nsub = (nbz - bk0).min(2);
            let flags = classify_super_slab(f, grid.origin, h, nx, ny, nz, nbx, nby, bk0, nsub, act_super, dcs_super);
            for s in 0..nsub {
                let bk = bk0 + s;
                let k0 = bk * BLK;
                let k1 = (k0 + BLK).min(nz);
                let n_slices = k1 - k0;
                let (any, all_pos, all_neg) = flags[s];
                let act_c = &act_super[s][..nblocks];
                let dcs_c = &dcs_super[s][..nblocks];

                let uniform = !any && (all_pos || all_neg);
                let boundary_trivial = bk == 0 || (uniform && ((all_pos && prev_all_pos) || (all_neg && prev_all_neg)));
                if uniform && boundary_trivial {
                    if !cells_prev.is_empty() {
                        clear_prev(recs_prev, cells_prev, dyn_prev);
                    }
                    let fill = if all_neg { -1.0 } else { 1.0 };
                    slab_prev_last[..slice_len].fill(fill);
                    if use_bits {
                        let fill_bits = if all_neg { if nx == 64 { u64::MAX } else { (1u64 << nx) - 1 } } else { 0 };
                        signs_prev_last[..ny].fill(fill_bits);
                    }
                    act_prev[..nblocks].copy_from_slice(act_c);
                    prev_all_pos = all_pos;
                    prev_all_neg = all_neg;
                    continue;
                }

                fill_slab(
                    f, grid.origin, h, nx, ny, nbx, k0, n_slices, any, all_pos, all_neg, act_c, dcs_c, slab_cur,
                    if use_bits { Some(&mut signs_cur[..n_slices * ny]) } else { None },
                );

                // 1. Slab boundary cell layer k = k0 - 1 (between slab_prev_last and slab_cur[0]).
                if bk > 0 {
                    if use_bits {
                        scan_and_emit_layer_bits(
                            k0 - 1, grid.origin, h, inv_h, nx, cx, cy, cz,
                            &slab_prev_last[..slice_len], &slab_cur[..slice_len],
                            &signs_prev_last[..ny], &signs_cur[..ny],
                            true, &recs_prev[..rec_len], dyn_prev,
                            &mut recs_cur[..rec_len], cells_cur, dyn_cur,
                            &mut positions, proj_ids, proj_grads, &mut quads, &mut guaranteed_manifold, None,
                        );
                        clear_prev(recs_prev, cells_prev, dyn_prev);
                        std::mem::swap(recs_prev, recs_cur);
                        std::mem::swap(cells_prev, cells_cur);
                        std::mem::swap(dyn_prev, dyn_cur);
                    } else {
                        let any_inter = build_scan_blocks_inter(nbx, nby, &act_prev[..nblocks], act_c, &mut scan_blocks[..nblocks]);
                        if any_inter {
                            scan_and_emit_layer(
                                k0 - 1, grid.origin, h, inv_h, nx, cx, cy, cz, nbx, &scan_blocks[..nblocks],
                                &slab_prev_last[..slice_len], &slab_cur[..slice_len],
                                true, &recs_prev[..rec_len], dyn_prev,
                                &mut recs_cur[..rec_len], cells_cur, dyn_cur,
                                &mut positions, proj_ids, proj_grads, &mut quads, &mut guaranteed_manifold, None,
                            );
                            clear_prev(recs_prev, cells_prev, dyn_prev);
                            std::mem::swap(recs_prev, recs_cur);
                            std::mem::swap(cells_prev, cells_cur);
                            std::mem::swap(dyn_prev, dyn_cur);
                        } else if !cells_prev.is_empty() {
                            clear_prev(recs_prev, cells_prev, dyn_prev);
                        }
                    }
                }

                // 2. Interior cell layers k = k0 .. k1 - 1 (all within slab_cur).
                if n_slices > 1 {
                    if use_bits {
                        if !uniform {
                            for local_k in 0..(n_slices - 1) {
                                let s0 = &slab_cur[local_k * slice_len..(local_k + 1) * slice_len];
                                let s1 = &slab_cur[(local_k + 1) * slice_len..(local_k + 2) * slice_len];
                                let b0 = &signs_cur[local_k * ny..(local_k + 1) * ny];
                                let b1 = &signs_cur[(local_k + 1) * ny..(local_k + 2) * ny];
                                scan_and_emit_layer_bits(
                                    k0 + local_k, grid.origin, h, inv_h, nx, cx, cy, cz,
                                    s0, s1, b0, b1,
                                    true, &recs_prev[..rec_len], dyn_prev,
                                    &mut recs_cur[..rec_len], cells_cur, dyn_cur,
                                    &mut positions, proj_ids, proj_grads, &mut quads, &mut guaranteed_manifold, None,
                                );
                                clear_prev(recs_prev, cells_prev, dyn_prev);
                                std::mem::swap(recs_prev, recs_cur);
                                std::mem::swap(cells_prev, cells_cur);
                                std::mem::swap(dyn_prev, dyn_cur);
                            }
                        } else if !cells_prev.is_empty() {
                            clear_prev(recs_prev, cells_prev, dyn_prev);
                        }
                    } else {
                        let any_intra = !uniform && build_scan_blocks_intra(nbx, nby, act_c, &mut scan_blocks[..nblocks]);
                        if any_intra {
                            for local_k in 0..(n_slices - 1) {
                                let s0 = &slab_cur[local_k * slice_len..(local_k + 1) * slice_len];
                                let s1 = &slab_cur[(local_k + 1) * slice_len..(local_k + 2) * slice_len];
                                scan_and_emit_layer(
                                    k0 + local_k, grid.origin, h, inv_h, nx, cx, cy, cz, nbx, &scan_blocks[..nblocks],
                                    s0, s1, true, &recs_prev[..rec_len], dyn_prev,
                                    &mut recs_cur[..rec_len], cells_cur, dyn_cur,
                                    &mut positions, proj_ids, proj_grads, &mut quads, &mut guaranteed_manifold, None,
                                );
                                clear_prev(recs_prev, cells_prev, dyn_prev);
                                std::mem::swap(recs_prev, recs_cur);
                                std::mem::swap(cells_prev, cells_cur);
                                std::mem::swap(dyn_prev, dyn_cur);
                            }
                        } else if !cells_prev.is_empty() {
                            clear_prev(recs_prev, cells_prev, dyn_prev);
                        }
                    }
                }

                // Carry last slice and block classification to next slab.
                let last_off = (n_slices - 1) * slice_len;
                slab_prev_last[..slice_len].copy_from_slice(&slab_cur[last_off..last_off + slice_len]);
                if use_bits {
                    let last_b = (n_slices - 1) * ny;
                    signs_prev_last[..ny].copy_from_slice(&signs_cur[last_b..last_b + ny]);
                }
                act_prev[..nblocks].copy_from_slice(act_c);
                prev_all_pos = all_pos;
                prev_all_neg = all_neg;
            }
            bk0 += nsub;
        }

        clear_prev(recs_prev, cells_prev, dyn_prev);
        clear_prev(recs_cur, cells_cur, dyn_cur);

        let flat = f.compile_flat();
        project_vertices_batched(f, flat.as_ref(), h, proj_ids, proj_grads, &mut positions);

        RawMesh { positions, quads, guaranteed_manifold }
    })
}

/// Single-phase L1-cache-resident parallel Surface Nets over `fast_par` (4 P-cores, ~0.3 µs wakeup):
/// Splits cell Z-layers into `n_strips` contiguous strips; each strip sweeps its own slices in L1-hot
/// 4-slice slabs, scans + emits all interior quads, and runs `FlatSdf` Newton projection in one pass
/// (zero shared `field` buffer, zero mid-kernel barrier). Serial stitch merges strips and emits the
/// `n_strips - 1` boundary X/Y quad layers.
#[cfg(feature = "threads")]
fn surface_nets_dynpar(f: &Compiled, grid: &Grid) -> RawMesh {
    let [nx, ny, nz] = grid.n;
    if nx < 2 || ny < 2 || nz < 2 {
        return RawMesh { positions: Vec::new(), quads: Vec::new(), guaranteed_manifold: true };
    }
    let h = grid.h;
    let inv_h = 1.0 / h;
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);
    let slice_len = nx * ny;
    let (nbx, nby) = (nx.div_ceil(BLK), ny.div_ceil(BLK));
    let nblocks = nbx * nby;
    let rec_len = cx * cy;
    let use_bits = nx <= 64;
    let n_strips = if cz <= 40 { (cz / 4).clamp(2, 4) } else { (cz / 8).clamp(4, 6) };
    let flat = f.compile_flat();

    SWEEP.with(|cell| {
        let mut sc = cell.borrow_mut();
        if sc.par_strips.len() < n_strips {
            sc.par_strips.resize_with(n_strips, StripScratch::default);
        }
        for st in &mut sc.par_strips[..n_strips] {
            if st.slab_cur.len() < slice_len * BLK {
                st.slab_cur.resize(slice_len * BLK, 0.0);
            }
            if st.slab_prev_last.len() < slice_len {
                st.slab_prev_last.resize(slice_len, 0.0);
            }
            if use_bits {
                if st.signs_cur.len() < ny * BLK {
                    st.signs_cur.resize(ny * BLK, 0);
                }
                if st.signs_prev_last.len() < ny {
                    st.signs_prev_last.resize(ny, 0);
                }
            } else {
                if st.act_prev.len() < nblocks {
                    st.act_prev.resize(nblocks, 0);
                }
                if st.scan_blocks.len() < nblocks {
                    st.scan_blocks.resize(nblocks, 0);
                }
            }
            if st.act_cur.len() < nblocks {
                st.act_cur.resize(nblocks, 0);
            }
            if st.dcs_cur.len() < nblocks {
                st.dcs_cur.resize(nblocks, 0.0);
            }
            if st.recs_prev.len() < rec_len {
                st.recs_prev.resize(rec_len, EMPTY);
            }
            if st.recs_cur.len() < rec_len {
                st.recs_cur.resize(rec_len, EMPTY);
            }
            if st.recs_first.len() < rec_len {
                st.recs_first.resize(rec_len, EMPTY);
            }
            if st.recs_last.len() < rec_len {
                st.recs_last.resize(rec_len, EMPTY);
            }
            st.positions.clear();
            st.quads.clear();
            st.proj_ids.clear();
            st.proj_grads.clear();
            st.cells_prev.clear();
            st.cells_cur.clear();
            st.dyn_prev.clear();
            st.dyn_cur.clear();
            st.cells_first.clear();
            st.dyn_first.clear();
            st.cells_last.clear();
            st.dyn_last.clear();
            st.guaranteed_manifold = true;
        }

        let strips_ptr = SyncPtr(sc.par_strips.as_mut_ptr());
        let flat_ref = flat.as_ref();

        fast_par::for_each(n_strips, &|s| {
            let k_start = s * cz / n_strips;
            let k_end = (s + 1) * cz / n_strips;
            if k_start >= k_end {
                return;
            }
            // SAFETY: each strip index `s` owns `par_strips[s]` exclusively.
            let st = unsafe { &mut *strips_ptr.get().add(s) };
            let StripScratch {
                slab_cur,
                slab_prev_last,
                signs_cur,
                signs_prev_last,
                act_cur,
                dcs_cur,
                act_prev,
                scan_blocks,
                positions,
                quads,
                proj_ids,
                proj_grads,
                recs_prev,
                recs_cur,
                cells_prev,
                cells_cur,
                dyn_prev,
                dyn_cur,
                recs_first,
                cells_first,
                dyn_first,
                recs_last,
                cells_last,
                dyn_last,
                guaranteed_manifold,
            } = st;

            let clear_rec = |recs: &mut [u64], cells: &mut Vec<u32>, dyns: &mut Vec<[u8; 12]>| {
                for &me in cells.iter() {
                    recs[me as usize] = EMPTY;
                }
                cells.clear();
                dyns.clear();
            };

            // Strip `s` needs sample slices `k_start ..= k_end`.
            // For non-final strips (`s + 1 < n_strips`), stop main 4-slice slabs at `k_end` and evaluate
            // the shared boundary slice `k_end` as a dedicated 1-slice slab starting at `k0 = k_end` (`lk = 0`).
            // Strip `s + 1` ALSO starts its first slab at `k0 = k_end` (`lk = 0`), so both strips compute
            // bit-for-bit identical f32 values and signs on every shared boundary slice (even at exact 0.0).
            let z_main_end = if s + 1 < n_strips { k_end } else { k_end + 1 };
            let z_slice_end = k_end + 1;
            let mut k0 = k_start;
            let mut first_slab = true;
            let mut prev_all_pos = false;
            let mut prev_all_neg = false;

            while k0 < z_slice_end {
                let k1 = if k0 < z_main_end { (k0 + BLK).min(z_main_end) } else { k0 + 1 };
                let n_slices = k1 - k0;
                let (any, all_pos, all_neg) = classify_slab_range(
                    f, grid.origin, h, nx, ny, nbx, nby, k0, k1, &mut act_cur[..nblocks], &mut dcs_cur[..nblocks],
                );
                let act_c = &act_cur[..nblocks];
                let dcs_c = &dcs_cur[..nblocks];
                let uniform = !any && (all_pos || all_neg);
                let boundary_trivial = first_slab || (uniform && ((all_pos && prev_all_pos) || (all_neg && prev_all_neg)));
                if uniform && boundary_trivial {
                    if !cells_prev.is_empty() {
                        clear_rec(&mut recs_prev[..rec_len], cells_prev, dyn_prev);
                    }
                    let fill = if all_neg { -1.0 } else { 1.0 };
                    slab_prev_last[..slice_len].fill(fill);
                    if use_bits {
                        let fill_bits = if all_neg { if nx == 64 { u64::MAX } else { (1u64 << nx) - 1 } } else { 0 };
                        signs_prev_last[..ny].fill(fill_bits);
                    } else {
                        act_prev[..nblocks].copy_from_slice(act_c);
                    }
                    prev_all_pos = all_pos;
                    prev_all_neg = all_neg;
                    first_slab = false;
                    k0 = k1;
                    continue;
                }

                fill_slab(
                    f, grid.origin, h, nx, ny, nbx, k0, n_slices, any, all_pos, all_neg, act_c, dcs_c,
                    &mut slab_cur[..n_slices * slice_len],
                    if use_bits { Some(&mut signs_cur[..n_slices * ny]) } else { None },
                );

                // 1. Inter-slab cell layer `k = k0 - 1` (between `slab_prev_last` and `slab_cur[0]`).
                if !first_slab {
                    let k = k0 - 1;
                    let is_first = k == k_start;
                    let emit_xy = k > 0 && !is_first;
                    let rec_first = if is_first && s > 0 { Some(&mut *cells_first) } else { None };
                    if use_bits {
                        scan_and_emit_layer_bits(
                            k, grid.origin, h, inv_h, nx, cx, cy, cz,
                            &slab_prev_last[..slice_len], &slab_cur[..slice_len],
                            &signs_prev_last[..ny], &signs_cur[..ny],
                            emit_xy, &recs_prev[..rec_len], dyn_prev,
                            &mut recs_cur[..rec_len], cells_cur, dyn_cur,
                            positions, proj_ids, proj_grads, quads, guaranteed_manifold, rec_first,
                        );
                        if is_first && s > 0 {
                            for &(me, _) in cells_first.iter() {
                                recs_first[me as usize] = recs_cur[me as usize];
                            }
                            dyn_first.extend_from_slice(dyn_cur);
                        }
                        clear_rec(&mut recs_prev[..rec_len], cells_prev, dyn_prev);
                        std::mem::swap(recs_prev, recs_cur);
                        std::mem::swap(cells_prev, cells_cur);
                        std::mem::swap(dyn_prev, dyn_cur);
                    } else {
                        let any_inter = build_scan_blocks_inter(nbx, nby, &act_prev[..nblocks], act_c, &mut scan_blocks[..nblocks]);
                        if any_inter {
                            scan_and_emit_layer(
                                k, grid.origin, h, inv_h, nx, cx, cy, cz, nbx, &scan_blocks[..nblocks],
                                &slab_prev_last[..slice_len], &slab_cur[..slice_len],
                                emit_xy, &recs_prev[..rec_len], dyn_prev,
                                &mut recs_cur[..rec_len], cells_cur, dyn_cur,
                                positions, proj_ids, proj_grads, quads, guaranteed_manifold, rec_first,
                            );
                            if is_first && s > 0 {
                                for &(me, _) in cells_first.iter() {
                                    recs_first[me as usize] = recs_cur[me as usize];
                                }
                                dyn_first.extend_from_slice(dyn_cur);
                            }
                            clear_rec(&mut recs_prev[..rec_len], cells_prev, dyn_prev);
                            std::mem::swap(recs_prev, recs_cur);
                            std::mem::swap(cells_prev, cells_cur);
                            std::mem::swap(dyn_prev, dyn_cur);
                        } else if !cells_prev.is_empty() {
                            clear_rec(&mut recs_prev[..rec_len], cells_prev, dyn_prev);
                        }
                    }
                }

                // 2. Intra-slab cell layers `k = k0 .. k1 - 1`.
                if n_slices > 1 {
                    if use_bits {
                        if !uniform {
                            for lk in 0..(n_slices - 1) {
                                let k = k0 + lk;
                                let is_first = k == k_start;
                                let emit_xy = k > 0 && !is_first;
                                let rec_first = if is_first && s > 0 { Some(&mut *cells_first) } else { None };
                                let s0 = &slab_cur[lk * slice_len..(lk + 1) * slice_len];
                                let s1 = &slab_cur[(lk + 1) * slice_len..(lk + 2) * slice_len];
                                let b0 = &signs_cur[lk * ny..(lk + 1) * ny];
                                let b1 = &signs_cur[(lk + 1) * ny..(lk + 2) * ny];
                                scan_and_emit_layer_bits(
                                    k, grid.origin, h, inv_h, nx, cx, cy, cz,
                                    s0, s1, b0, b1,
                                    emit_xy, &recs_prev[..rec_len], dyn_prev,
                                    &mut recs_cur[..rec_len], cells_cur, dyn_cur,
                                    positions, proj_ids, proj_grads, quads, guaranteed_manifold, rec_first,
                                );
                                if is_first && s > 0 {
                                    for &(me, _) in cells_first.iter() {
                                        recs_first[me as usize] = recs_cur[me as usize];
                                    }
                                    dyn_first.extend_from_slice(dyn_cur);
                                }
                                clear_rec(&mut recs_prev[..rec_len], cells_prev, dyn_prev);
                                std::mem::swap(recs_prev, recs_cur);
                                std::mem::swap(cells_prev, cells_cur);
                                std::mem::swap(dyn_prev, dyn_cur);
                            }
                        } else if !cells_prev.is_empty() {
                            clear_rec(&mut recs_prev[..rec_len], cells_prev, dyn_prev);
                        }
                    } else {
                        let any_intra = !uniform && build_scan_blocks_intra(nbx, nby, act_c, &mut scan_blocks[..nblocks]);
                        if any_intra {
                            for lk in 0..(n_slices - 1) {
                                let k = k0 + lk;
                                let is_first = k == k_start;
                                let emit_xy = k > 0 && !is_first;
                                let rec_first = if is_first && s > 0 { Some(&mut *cells_first) } else { None };
                                let s0 = &slab_cur[lk * slice_len..(lk + 1) * slice_len];
                                let s1 = &slab_cur[(lk + 1) * slice_len..(lk + 2) * slice_len];
                                scan_and_emit_layer(
                                    k, grid.origin, h, inv_h, nx, cx, cy, cz, nbx, &scan_blocks[..nblocks],
                                    s0, s1, emit_xy, &recs_prev[..rec_len], dyn_prev,
                                    &mut recs_cur[..rec_len], cells_cur, dyn_cur,
                                    positions, proj_ids, proj_grads, quads, guaranteed_manifold, rec_first,
                                );
                                if is_first && s > 0 {
                                    for &(me, _) in cells_first.iter() {
                                        recs_first[me as usize] = recs_cur[me as usize];
                                    }
                                    dyn_first.extend_from_slice(dyn_cur);
                                }
                                clear_rec(&mut recs_prev[..rec_len], cells_prev, dyn_prev);
                                std::mem::swap(recs_prev, recs_cur);
                                std::mem::swap(cells_prev, cells_cur);
                                std::mem::swap(dyn_prev, dyn_cur);
                            }
                        } else if !cells_prev.is_empty() {
                            clear_rec(&mut recs_prev[..rec_len], cells_prev, dyn_prev);
                        }
                    }
                }

                // Carry last slice to next slab.
                if k1 < z_slice_end {
                    let last_off = (n_slices - 1) * slice_len;
                    slab_prev_last[..slice_len].copy_from_slice(&slab_cur[last_off..last_off + slice_len]);
                    if use_bits {
                        let last_b = (n_slices - 1) * ny;
                        signs_prev_last[..ny].copy_from_slice(&signs_cur[last_b..last_b + ny]);
                    } else {
                        act_prev[..nblocks].copy_from_slice(act_c);
                    }
                }
                prev_all_pos = all_pos;
                prev_all_neg = all_neg;
                first_slab = false;
                k0 = k1;
            }

            if s + 1 < n_strips {
                std::mem::swap(recs_last, recs_prev);
                std::mem::swap(cells_last, cells_prev);
                std::mem::swap(dyn_last, dyn_prev);
            } else {
                clear_rec(&mut recs_prev[..rec_len], cells_prev, dyn_prev);
            }
            project_vertices_batched(f, flat_ref, h, proj_ids, proj_grads, positions);
        });

        let mut offsets = [0u32; 8];
        let mut total_verts = 0u32;
        let mut total_quads = 0usize;
        let mut guaranteed_manifold = true;
        for (s, st) in sc.par_strips[..n_strips].iter().enumerate() {
            offsets[s] = total_verts;
            total_verts += st.positions.len() as u32;
            total_quads += st.quads.len() + st.cells_first.len() * 2;
            guaranteed_manifold &= st.guaranteed_manifold;
        }
        let mut positions = Vec::with_capacity(total_verts as usize);
        let mut quads = Vec::with_capacity(total_quads);
        for (s, st) in sc.par_strips[..n_strips].iter().enumerate() {
            positions.extend_from_slice(&st.positions);
            let off = offsets[s];
            if off == 0 {
                quads.extend_from_slice(&st.quads);
            } else {
                let old_len = quads.len();
                let add = st.quads.len();
                quads.reserve(add);
                unsafe {
                    let dst = quads.as_mut_ptr().add(old_len);
                    let src = st.quads.as_ptr();
                    for i in 0..add {
                        let q = *src.add(i);
                        *dst.add(i) = [q[0] + off, q[1] + off, q[2] + off, q[3] + off];
                    }
                    quads.set_len(old_len + add);
                }
            }
        }
        for s in 1..n_strips {
            let (oprev, ocur) = (offsets[s - 1], offsets[s]);
            let prev = &sc.par_strips[s - 1];
            let cur = &sc.par_strips[s];
            if cur.cells_first.is_empty() {
                continue;
            }
            let rprev = &prev.recs_last[..rec_len];
            let dprev = &prev.dyn_last;
            let rcur = &cur.recs_first[..rec_len];
            let dcur = &cur.dyn_first;
            for &(me_u, mask) in &cur.cells_first {
                let me = me_u as usize;
                let j = me / cx;
                let i = me - j * cx;
                let r = rcur[me];
                // Check cross-strip Z-face merged-saddle collision (face 0 of cur vs face 1 of prev).
                if ((r >> MERGED_FACE_SHIFT) & (1 << 0)) != 0 {
                    let rp = rprev[me];
                    if rp != EMPTY && ((rp >> MERGED_FACE_SHIFT) & (1 << 1)) != 0 {
                        guaranteed_manifold = false;
                    }
                }
                let base = ocur + r as u32;
                let inside = mask & 1 != 0;
                let mut push = |q: [u32; 4]| {
                    if q[0].max(q[1]).max(q[2]).max(q[3]) != u32::MAX {
                        quads.push(if inside { q } else { [q[0], q[3], q[2], q[1]] });
                    }
                };
                if (r & MULTI_COMP_BIT) == 0 {
                    if j >= 1 && (mask ^ (mask >> 1)) & 1 != 0 {
                        let v0 = vert_lookup(rprev, dprev, me - cx, ED[0][0]);
                        let v1 = vert_lookup(rprev, dprev, me, ED[0][1]);
                        let v3 = vert_lookup(rcur, dcur, me - cx, ED[0][3]);
                        push([
                            if v0 == u32::MAX { u32::MAX } else { oprev + v0 },
                            if v1 == u32::MAX { u32::MAX } else { oprev + v1 },
                            base,
                            if v3 == u32::MAX { u32::MAX } else { ocur + v3 },
                        ]);
                    }
                    if i >= 1 && (mask ^ (mask >> 2)) & 1 != 0 {
                        let v0 = vert_lookup(rprev, dprev, me - 1, ED[1][0]);
                        let v1 = vert_lookup(rcur, dcur, me - 1, ED[1][1]);
                        let v3 = vert_lookup(rprev, dprev, me, ED[1][3]);
                        push([
                            if v0 == u32::MAX { u32::MAX } else { oprev + v0 },
                            if v1 == u32::MAX { u32::MAX } else { ocur + v1 },
                            base,
                            if v3 == u32::MAX { u32::MAX } else { oprev + v3 },
                        ]);
                    }
                } else {
                    let tab = ((r >> 32) & 0x7FFF) as usize;
                    let q2_comp: &[u8; 12] = if tab < 256 { &TABLES.comp[tab] } else { &dcur[tab - 256] };
                    if j >= 1 && (mask ^ (mask >> 1)) & 1 != 0 {
                        let v0 = vert_lookup(rprev, dprev, me - cx, ED[0][0]);
                        let v1 = vert_lookup(rprev, dprev, me, ED[0][1]);
                        let v3 = vert_lookup(rcur, dcur, me - cx, ED[0][3]);
                        push([
                            if v0 == u32::MAX { u32::MAX } else { oprev + v0 },
                            if v1 == u32::MAX { u32::MAX } else { oprev + v1 },
                            base + q2_comp[ED[0][2]] as u32,
                            if v3 == u32::MAX { u32::MAX } else { ocur + v3 },
                        ]);
                    }
                    if i >= 1 && (mask ^ (mask >> 2)) & 1 != 0 {
                        let v0 = vert_lookup(rprev, dprev, me - 1, ED[1][0]);
                        let v1 = vert_lookup(rcur, dcur, me - 1, ED[1][1]);
                        let v3 = vert_lookup(rprev, dprev, me, ED[1][3]);
                        push([
                            if v0 == u32::MAX { u32::MAX } else { oprev + v0 },
                            if v1 == u32::MAX { u32::MAX } else { ocur + v1 },
                            base + q2_comp[ED[1][2]] as u32,
                            if v3 == u32::MAX { u32::MAX } else { oprev + v3 },
                        ]);
                    }
                }
            }
        }
        for st in &mut sc.par_strips[..n_strips] {
            for &(me, _) in &st.cells_first {
                st.recs_first[me as usize] = EMPTY;
            }
            for &me in &st.cells_last {
                st.recs_last[me as usize] = EMPTY;
            }
        }
        RawMesh { positions, quads, guaranteed_manifold }
    })
}

pub(crate) fn surface_nets_raw(f: &Compiled, grid: &Grid) -> RawMesh {
    #[cfg(feature = "threads")]
    {
        let nbz = grid.n[2].div_ceil(BLK);
        if grid.samples() >= 8_000
            && nbz >= 3
            && grid.n[2] >= 10
            && rayon::current_thread_index().is_none()
            && rayon::current_num_threads() >= 2
        {
            return surface_nets_dynpar(f, grid);
        }
    }
    surface_nets_sweep(f, grid)
}

/// Converts a `RawMesh` into Sculptor's boxed `Mesh` (either quads or directly triangulated along shorter diagonals).
pub(crate) fn raw_to_mesh(raw: RawMesh, keep_quads: bool) -> Mesh {
    let RawMesh { positions, quads, .. } = raw;
    if keep_quads {
        #[cfg(feature = "threads")]
        let faces: Vec<Vec<u32>> = if quads.len() >= 1_024 && rayon::current_thread_index().is_none() {
            let n = quads.len();
            let ch = n.div_ceil(4).clamp(256, 2048);
            let nc = n.div_ceil(ch);
            let mut faces: Vec<std::mem::MaybeUninit<Vec<u32>>> = Vec::with_capacity(n);
            unsafe { faces.set_len(n); }
            let dst = SyncPtr(faces.as_mut_ptr());
            let src = quads.as_slice();
            fast_par::for_each(nc, &|c| {
                let s = c * ch;
                let e = (s + ch).min(n);
                let p = dst.get();
                for i in s..e {
                    unsafe {
                        (*p.add(i)).write(src[i].to_vec());
                    }
                }
            });
            // SAFETY: every element 0..n was written once above.
            unsafe {
                let ptr = faces.as_mut_ptr() as *mut Vec<u32>;
                let cap = faces.capacity();
                std::mem::forget(faces);
                Vec::from_raw_parts(ptr, n, cap)
            }
        } else {
            let mut faces = Vec::with_capacity(quads.len());
            for q in &quads {
                faces.push(q.to_vec());
            }
            faces
        };
        #[cfg(not(feature = "threads"))]
        let faces: Vec<Vec<u32>> = {
            let mut faces = Vec::with_capacity(quads.len());
            for q in &quads {
                faces.push(q.to_vec());
            }
            faces
        };
        Mesh { positions, faces, smooth_shading: true }
    } else {
        let mut faces = Vec::with_capacity(quads.len() * 2);
        for q in &quads {
            let p0 = positions[q[0] as usize];
            let p1 = positions[q[1] as usize];
            let p2 = positions[q[2] as usize];
            let p3 = positions[q[3] as usize];
            if (p0 - p2).length_squared() <= (p1 - p3).length_squared() {
                faces.push(vec![q[0], q[1], q[2]]);
                faces.push(vec![q[0], q[2], q[3]]);
            } else {
                faces.push(vec![q[0], q[1], q[3]]);
                faces.push(vec![q[1], q[2], q[3]]);
            }
        }
        Mesh { positions, faces, smooth_shading: true }
    }
}

pub fn surface_nets(f: &Compiled, grid: &Grid) -> Mesh {
    raw_to_mesh(surface_nets_raw(f, grid), true)
}

/// Components of a saddle cell (a face with four crossings): edge -> component id (ids numbered by first
/// edge), how many components there are, and `merged_faces` (6-bit mask: bit `fi` set iff face `fi` has
/// 4 crossings whose two asymptotic-decider pairs merged into the same component of this cell).
fn saddle_components(v: &[f32; 8], cross: u16) -> ([u8; 12], usize, u8) {
    let mut comp: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    #[inline(always)]
    fn root(c: &mut [u8; 12], mut x: u8) -> u8 {
        while c[x as usize] != x {
            x = c[x as usize];
        }
        x
    }
    let join = |c: &mut [u8; 12], a: usize, b: usize| {
        let (ra, rb) = (root(c, a as u8), root(c, b as u8));
        c[ra as usize] = rb;
    };
    let mut quad_faces = 0u8;
    for (fi, face) in FACES.iter().enumerate() {
        let mut cr = [0usize; 4];
        let mut cr_len = 0usize;
        for n in 0..4 {
            if (cross >> face.edges[n]) & 1 != 0 {
                cr[cr_len] = n;
                cr_len += 1;
            }
        }
        match cr_len {
            2 => join(&mut comp, face.edges[cr[0]], face.edges[cr[1]]),
            4 => {
                quad_faces |= 1u8 << fi;
                let [a, b, c, d] = face.corners.map(|n| v[n]);
                let denom = a + c - b - d;
                let saddle = if denom.abs() > 1e-12 { (a * c - b * d) / denom } else { a };
                // Corners on the saddle's side are joined through the face's middle; the other
                // pair of diagonal corners is cut off. Edge n runs corner n -> corner n+1.
                let corner0_inside = a < 0.0;
                if (saddle < 0.0) == corner0_inside {
                    // Corners 1 and 3 are cut off: pair the edges around each of them.
                    join(&mut comp, face.edges[0], face.edges[1]);
                    join(&mut comp, face.edges[2], face.edges[3]);
                } else {
                    // Corners 0 and 2 are cut off.
                    join(&mut comp, face.edges[3], face.edges[0]);
                    join(&mut comp, face.edges[1], face.edges[2]);
                }
            }
            _ => {}
        }
    }
    let mut ids = [255u8; 12];
    let mut out = [255u8; 12];
    let mut n = 0usize;
    for e in 0..12 {
        if (cross >> e) & 1 != 0 {
            let r = root(&mut comp, e as u8) as usize;
            if ids[r] == 255 {
                ids[r] = n as u8;
                n += 1;
            }
            out[e] = ids[r];
        }
    }
    let mut merged_faces = 0u8;
    let mut qf = quad_faces;
    while qf != 0 {
        let fi = qf.trailing_zeros() as usize;
        qf &= qf - 1;
        // In both asymptotic decider pairings, face.edges[0] and face.edges[2] belong to opposite pairs.
        if out[FACES[fi].edges[0]] == out[FACES[fi].edges[2]] {
            merged_faces |= 1u8 << fi;
        }
    }
    (out, n, merged_faces)
}

/// Per corner-sign-mask tables for cells with no four-crossing face (everything except saddles).
pub(crate) struct Tables {
    /// Bit e set when edge e crosses.
    pub(crate) cross: [u16; 256],
    /// Some face has four crossings, so components depend on the sample values.
    ambiguous: [bool; 256],
    ncomp: [u8; 256],
    /// Edge -> component id (numbered by first edge), 255 for edges that don't cross.
    comp: [[u8; 12]; 256],
    /// Precomputed sum of `EDGE_BASE[e]` over all crossing edges `e` for this mask.
    base_sum: [Vec3; 256],
    /// Precomputed `1.0 / cross.count_ones()` (0.0 for mask 0 and 255).
    inv_cross_count: [f32; 256],
}

pub(crate) static TABLES: Tables = build_tables();

const fn build_tables() -> Tables {
    let mut t = Tables {
        cross: [0; 256],
        ambiguous: [false; 256],
        ncomp: [0; 256],
        comp: [[255; 12]; 256],
        base_sum: [Vec3::ZERO; 256],
        inv_cross_count: [0.0; 256],
    };
    let mut mask = 0usize;
    while mask < 256 {
        let mut cross = 0u16;
        let mut bx = 0.0f32;
        let mut by = 0.0f32;
        let mut bz = 0.0f32;
        let mut cnt = 0u32;
        let mut e = 0;
        while e < 12 {
            let (a, b) = EDGES[e];
            if (mask >> a) & 1 != (mask >> b) & 1 {
                cross |= 1 << e;
                bx += (a & 1) as f32;
                by += ((a >> 1) & 1) as f32;
                bz += ((a >> 2) & 1) as f32;
                cnt += 1;
            }
            e += 1;
        }
        t.base_sum[mask] = Vec3::new(bx, by, bz);
        t.inv_cross_count[mask] = if cnt > 0 { 1.0 / cnt as f32 } else { 0.0 };
        let mut parent = [0u8; 12];
        let mut i = 0;
        while i < 12 {
            parent[i] = i as u8;
            i += 1;
        }
        let mut amb = false;
        let mut f = 0;
        while f < 6 {
            let mut cr = [0usize; 4];
            let mut cr_len = 0;
            let mut n = 0;
            while n < 4 {
                if (cross >> FACES[f].edges[n]) & 1 != 0 {
                    cr[cr_len] = n;
                    cr_len += 1;
                }
                n += 1;
            }
            if cr_len == 2 {
                let (mut x, mut y) = (FACES[f].edges[cr[0]] as u8, FACES[f].edges[cr[1]] as u8);
                while parent[x as usize] != x {
                    x = parent[x as usize];
                }
                while parent[y as usize] != y {
                    y = parent[y as usize];
                }
                parent[x as usize] = y;
            } else if cr_len == 4 {
                amb = true;
            }
            f += 1;
        }
        let mut ids = [255u8; 12];
        let mut n = 0u8;
        let mut e = 0;
        while e < 12 {
            if (cross >> e) & 1 != 0 {
                let mut r = e as u8;
                while parent[r as usize] != r {
                    r = parent[r as usize];
                }
                if ids[r as usize] == 255 {
                    ids[r as usize] = n;
                    n += 1;
                }
                t.comp[mask][e] = ids[r as usize];
            }
            e += 1;
        }
        t.cross[mask] = cross;
        t.ambiguous[mask] = amb;
        t.ncomp[mask] = n;
        mask += 1;
    }
    t
}

/// Where each cell edge's crossing starts (its lower corner) and its direction (unit axis step).
pub(crate) const EDGE_BASE: [Vec3; 12] = {
    let mut t = [Vec3::ZERO; 12];
    let mut e = 0;
    while e < 12 {
        let a = EDGES[e].0;
        t[e] = Vec3::new((a & 1) as f32, ((a >> 1) & 1) as f32, ((a >> 2) & 1) as f32);
        e += 1;
    }
    t
};
pub(crate) const EDGE_DIR: [Vec3; 12] = {
    let mut t = [Vec3::ZERO; 12];
    let mut e = 0;
    while e < 12 {
        let (a, b) = EDGES[e];
        t[e] = Vec3::new(
            ((b & 1) as f32) - ((a & 1) as f32),
            (((b >> 1) & 1) as f32) - (((a >> 1) & 1) as f32),
            (((b >> 2) & 1) as f32) - (((a >> 2) & 1) as f32),
        );
        e += 1;
    }
    t
};
/// ED[a][n]: the cell edge, in the n-th cell around a +axis-`a` sample edge, that is that sample edge.
const ED: [[usize; 4]; 3] = {
    let mut t = [[0usize; 4]; 3];
    let mut a = 0;
    while a < 3 {
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        let mut n = 0;
        while n < 4 {
            let (ob, oc) = match n {
                0 => (1, 1),
                1 => (0, 1),
                2 => (0, 0),
                _ => (1, 0),
            };
            let c0 = ob << b | oc << c;
            let c1 = c0 | 1 << a;
            t[a][n] = EDGE_LOOKUP[c0][c1] as usize;
            n += 1;
        }
        a += 1;
    }
    t
};

pub(crate) const EDGES: [(usize, usize); 12] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];
const EDGE_LOOKUP: [[u8; 8]; 8] = {
    let mut table = [[255u8; 8]; 8];
    let mut i = 0;
    while i < 12 {
        let (a, b) = EDGES[i];
        table[a][b] = i as u8;
        table[b][a] = i as u8;
        i += 1;
    }
    table
};

/// A cell face: its corners in cyclic order and edges[n] joining corners[n] and corners[n + 1].
struct Face {
    corners: [usize; 4],
    edges: [usize; 4],
}

const FACES: [Face; 6] = [
    // z = 0 and z = 1 (corners 0,1,3,2 / 4,5,7,6)
    Face { corners: [0, 1, 3, 2], edges: [0, 5, 1, 4] },
    Face { corners: [4, 5, 7, 6], edges: [2, 7, 3, 6] },
    // y = 0 and y = 1 (corners 0,1,5,4 / 2,3,7,6)
    Face { corners: [0, 1, 5, 4], edges: [0, 9, 2, 8] },
    Face { corners: [2, 3, 7, 6], edges: [1, 11, 3, 10] },
    // x = 0 and x = 1 (corners 0,2,6,4 / 1,3,7,5)
    Face { corners: [0, 2, 6, 4], edges: [4, 10, 6, 8] },
    Face { corners: [1, 3, 7, 5], edges: [5, 11, 7, 9] },
];

/// Splits every quad into two triangles along its shorter diagonal.
pub fn triangulate(m: &mut Mesh) {
    let mut out = Vec::with_capacity(m.faces.len() * 2);
    for f in &m.faces {
        if f.len() == 4 {
            let p = |i: usize| m.positions[f[i] as usize];
            if (p(0) - p(2)).length_squared() <= (p(1) - p(3)).length_squared() {
                out.push(vec![f[0], f[1], f[2]]);
                out.push(vec![f[0], f[2], f[3]]);
            } else {
                out.push(vec![f[0], f[1], f[3]]);
                out.push(vec![f[1], f[2], f[3]]);
            }
        } else {
            out.push(f.clone());
        }
    }
    m.faces = out;
}

pub fn triangle_count(m: &Mesh) -> usize {
    m.faces.iter().map(|f| f.len().saturating_sub(2)).sum()
}

/// Makes a closed Surface Nets mesh strictly manifold, as Sculptor's booleans and edit tools need.
///
/// At saddle cells (two sheets passing through one cell) Surface Nets joins four quads on one edge and two
/// cones at one vertex. This pairs the faces around each such edge by angle (each face with the next one
/// around the edge that runs the other way), then gives every fan of faces around a vertex its own copy of
/// the vertex. The result has every directed edge exactly once with its reverse; duplicated vertices sit at
/// the same position, so the shape doesn't change.
pub fn make_manifold(m: &mut Mesh) {
    if is_manifold(m) {
        return;
    }
    use std::collections::HashMap;
    // Undirected edge -> (face, corner index k where the face runs f[k] -> f[k+1]).
    let mut edges: HashMap<(u32, u32), Vec<(usize, usize)>> = HashMap::new();
    for (fi, f) in m.faces.iter().enumerate() {
        for k in 0..f.len() {
            let (a, b) = (f[k], f[(k + 1) % f.len()]);
            edges.entry((a.min(b), a.max(b))).or_default().push((fi, k));
        }
    }
    if edges.values().all(|v| v.len() == 2) {
        return;
    }
    // Union-find over corners (face, corner): corners joined across paired edges share a vertex.
    let offsets: Vec<usize> = m.faces.iter().scan(0, |s, f| { let o = *s; *s += f.len(); Some(o) }).collect();
    let total: usize = m.faces.iter().map(|f| f.len()).sum();
    let mut parent: Vec<usize> = (0..total).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let unite = |p: &mut Vec<usize>, a: usize, b: usize| {
        let (ra, rb) = (find(p, a), find(p, b));
        if ra != rb {
            p[ra] = rb;
        }
    };
    let centroid = |f: &Vec<u32>| f.iter().fold(glam::Vec3::ZERO, |s, &i| s + m.positions[i as usize]) / f.len() as f32;
    for (&(lo, hi), list) in edges.iter() {
        let pairs: Vec<((usize, usize), (usize, usize))> = if list.len() == 2 {
            vec![(list[0], list[1])]
        } else {
            let (pa, pb) = (m.positions[lo as usize], m.positions[hi as usize]);
            let d = (pb - pa).normalize_or_zero();
            let u0 = {
                let w = centroid(&m.faces[list[0].0]) - pa;
                (w - d * w.dot(d)).normalize_or_zero()
            };
            let v0 = d.cross(u0);
            let mut around: Vec<(f32, (usize, usize), bool)> = list
                .iter()
                .map(|&(fi, k)| {
                    let w = centroid(&m.faces[fi]) - pa;
                    let w = w - d * w.dot(d);
                    let forward = m.faces[fi][k] == lo; // runs lo -> hi
                    (w.dot(v0).atan2(w.dot(u0)), (fi, k), forward)
                })
                .collect();
            around.sort_by(|a, b| a.0.total_cmp(&b.0));
            let n = around.len();
            let mut used = vec![false; n];
            let mut out = Vec::new();
            for i in 0..n {
                if used[i] || !around[i].2 {
                    continue;
                }
                // The next unused face around the edge that runs the other way.
                if let Some(j) = (1..n).map(|s| (i + s) % n).find(|&j| !used[j] && !around[j].2) {
                    used[i] = true;
                    used[j] = true;
                    out.push((around[i].1, around[j].1));
                }
            }
            out
        };
        for ((fa, ka), (fb, kb)) in pairs {
            let (la, lb) = (m.faces[fa].len(), m.faces[fb].len());
            // Face a runs x -> y at corners ka, ka+1; face b runs y -> x at kb, kb+1.
            unite(&mut parent, offsets[fa] + ka, offsets[fb] + (kb + 1) % lb);
            unite(&mut parent, offsets[fa] + (ka + 1) % la, offsets[fb] + kb);
        }
    }
    // One vertex per (original vertex, corner group); the first group keeps the original index.
    let mut assigned: HashMap<(u32, usize), u32> = HashMap::new();
    let mut taken = vec![false; m.positions.len()];
    for fi in 0..m.faces.len() {
        for k in 0..m.faces[fi].len() {
            let v = m.faces[fi][k];
            let root = find(&mut parent, offsets[fi] + k);
            let idx = *assigned.entry((v, root)).or_insert_with(|| {
                if !taken[v as usize] {
                    taken[v as usize] = true;
                    v
                } else {
                    m.positions.push(m.positions[v as usize]);
                    (m.positions.len() - 1) as u32
                }
            });
            m.faces[fi][k] = idx;
        }
    }
}

#[inline]
fn check_manifold_buckets(ends: &[u32], keys: &mut [u32]) -> bool {
    let mut s = 0usize;
    for &e in ends {
        let e = e as usize;
        let bucket = &mut keys[s..e];
        s = e;
        let len = bucket.len();
        if len == 0 {
            continue;
        }
        if len & 1 != 0 {
            return false;
        }
        if len == 2 {
            // Two directed edges pair iff they share the upper vertex and have opposite directions.
            if (bucket[0] ^ bucket[1]) != 1 {
                return false;
            }
            continue;
        }
        if len <= 16 {
            for i in 1..len {
                let key = bucket[i];
                let mut j = i;
                while j > 0 && bucket[j - 1] > key {
                    bucket[j] = bucket[j - 1];
                    j -= 1;
                }
                bucket[j] = key;
            }
        } else {
            bucket.sort_unstable();
        }
        let mut i = 0;
        while i < len {
            let k0 = bucket[i];
            // Pair must be the same undirected edge with opposite directions: dir 0 then dir 1.
            if k0 & 1 != 0 || bucket[i + 1] != (k0 | 1) {
                return false;
            }
            // Reject non-manifold edges shared by 3+ faces (same undirected edge repeats).
            if i + 2 < len && (bucket[i + 2] >> 1) == (k0 >> 1) {
                return false;
            }
            i += 2;
        }
    }
    true
}

/// `is_manifold` over a flat quad list with `num_vertices` vertices (no boxed faces, thread-local scratch).
pub(crate) fn quads_manifold(quads: &[[u32; 4]], num_vertices: usize) -> bool {
    if quads.is_empty() {
        return true;
    }
    let nv = num_vertices;
    SWEEP.with(|cell| {
        let mut sc = cell.borrow_mut();
        let SweepScratch { man_cursor, man_keys, .. } = &mut *sc;
        man_cursor.clear();
        man_cursor.resize(nv + 1, 0);
        for q in quads {
            man_cursor[q[0].min(q[1]) as usize + 1] += 1;
            man_cursor[q[1].min(q[2]) as usize + 1] += 1;
            man_cursor[q[2].min(q[3]) as usize + 1] += 1;
            man_cursor[q[3].min(q[0]) as usize + 1] += 1;
        }
        for i in 0..nv {
            man_cursor[i + 1] += man_cursor[i];
        }
        let need_keys = quads.len() * 4;
        if man_keys.len() < need_keys {
            man_keys.resize(need_keys, 0);
        }
        let mut push_edge = |a: u32, b: u32| {
            let (lo, hi, dir) = if a < b { (a, b, 0u32) } else { (b, a, 1u32) };
            let c = &mut man_cursor[lo as usize];
            man_keys[*c as usize] = hi << 1 | dir;
            *c += 1;
        };
        for q in quads {
            push_edge(q[0], q[1]);
            push_edge(q[1], q[2]);
            push_edge(q[2], q[3]);
            push_edge(q[3], q[0]);
        }
        check_manifold_buckets(&man_cursor[..nv], &mut man_keys[..need_keys])
    })
}

/// Sculptor's watertight test: every directed edge appears once and so does its reverse.
/// Counting-sort variant: edges are bucketed by their lower vertex in one pass (no global sort, no hashing);
/// each bucket holds a handful of `(upper vertex, direction)` keys that are sorted and paired in place.
pub fn is_manifold(m: &Mesh) -> bool {
    let mut total_edges = 0usize;
    let mut max_v = 0u32;
    for f in &m.faces {
        total_edges += f.len();
        for &v in f {
            max_v = max_v.max(v);
        }
    }
    if total_edges == 0 || total_edges % 2 != 0 {
        return total_edges == 0;
    }
    if max_v >= 1 << 31 {
        return false;
    }
    let nv = max_v as usize + 1;
    let mut cursor = vec![0u32; nv + 1];
    for f in &m.faces {
        if let &[v0, v1, v2, v3] = f.as_slice() {
            cursor[v0.min(v1) as usize + 1] += 1;
            cursor[v1.min(v2) as usize + 1] += 1;
            cursor[v2.min(v3) as usize + 1] += 1;
            cursor[v3.min(v0) as usize + 1] += 1;
        } else {
            let len = f.len();
            for k in 0..len {
                let (a, b) = (f[k], f[if k + 1 == len { 0 } else { k + 1 }]);
                cursor[a.min(b) as usize + 1] += 1;
            }
        }
    }
    for i in 0..nv {
        cursor[i + 1] += cursor[i];
    }
    let mut keys = vec![0u32; total_edges];
    let mut push_edge = |a: u32, b: u32| {
        // Direction bit: 0 when the edge runs low -> high.
        let (lo, hi, dir) = if a < b { (a, b, 0u32) } else { (b, a, 1u32) };
        let c = &mut cursor[lo as usize];
        keys[*c as usize] = hi << 1 | dir;
        *c += 1;
    };
    for f in &m.faces {
        if let &[v0, v1, v2, v3] = f.as_slice() {
            push_edge(v0, v1);
            push_edge(v1, v2);
            push_edge(v2, v3);
            push_edge(v3, v0);
        } else {
            let len = f.len();
            for k in 0..len {
                push_edge(f[k], f[if k + 1 == len { 0 } else { k + 1 }]);
            }
        }
    }
    check_manifold_buckets(&cursor[..nv], &mut keys)
}
