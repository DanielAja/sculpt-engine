//! Pure-Rust 3D Gaussian Splatting (3DGS) engine for Items-to-3D, Spaces-to-3D, and Prompt Editing.
//!
//! Zero external crates beyond `glam` & `serde_json` (and optional `rayon` on native targets):
//! - Standard `.splat` (32 B/Gaussian) & 3DGS `.ply` (binary little-endian + ASCII) codecs, plus the
//!   ultra-compact `.dsplat` v2 (exactly 12 B/Gaussian; legacy 16 B v1 still decodes).
//! - On-device `image_to_splat` (both segmented 3D items and depth-projected 3D spaces).
//! - On-device `mesh_to_splat` (converts Dream Builder objects OR captured visionOS/ARKit room meshes).
//! - Real-time prompt-driven editing (`edit_splat_by_prompt`) with semantic spatial region targeting
//!   (floor, walls, ceiling, background, center), atmosphere/relighting presets, luminance-preserving
//!   color/material grades, and geometric/style transforms (sharpen, painterly, voxel, wave, swirl, prune).
//! - O(N) single-pass 1024-bucket counting depth sort (`sort_splats_by_camera`), 4x unrolled for SIMD.

use crate::img3d::{downsample_image, segment_foreground, signed_distance_2d};
use glam::Vec3;
use serde_json::json;

/// 0th-order Spherical Harmonic constant: `1 / (2 * sqrt(pi))`.
const SH_C0: f32 = 0.282_094_8;
const INV_SH_C0: f32 = 1.0 / SH_C0;
const COMPACT_MAGIC: &[u8; 4] = b"DSPL";
const COMPACT_HEADER_LEN: usize = 44;
const COMPACT_VERSION_V1: u32 = 1;
const COMPACT_VERSION_V2: u32 = 2;
/// Legacy `.dsplat` v1 record size (read-only support).
const V1_STRIDE: usize = 16;
/// `.dsplat` v2 record size: exactly 96 bits per Gaussian.
const V2_STRIDE: usize = 12;
const V2_POS_MAX: f32 = 4095.0; // 12-bit positions
const V2_SCALE_MAX: f32 = 63.0; // 6-bit log-scales
const V2_ROT_ENC: f32 = 15.0 * std::f32::consts::SQRT_2; // [-1/sqrt2, 1/sqrt2] -> [-15, 15]
const V2_ROT_DEC: f32 = 1.0 / V2_ROT_ENC;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gaussian {
    /// World position in meters (Y-up).
    pub pos: [f32; 3],
    /// Positive radii along local X, Y, Z axes in meters.
    pub scale: [f32; 3],
    /// Normalized quaternion `[w, x, y, z]`.
    pub rot: [f32; 4],
    /// Base sRGB color `[0..1]`.
    pub color: [f32; 3],
    /// Opacity `[0..1]`.
    pub opacity: f32,
}

const _: () = assert!(
    std::mem::size_of::<Gaussian>() == 56
        && std::mem::offset_of!(Gaussian, pos) == 0
        && std::mem::offset_of!(Gaussian, scale) == 12
        && std::mem::offset_of!(Gaussian, rot) == 24
        && std::mem::offset_of!(Gaussian, color) == 40
        && std::mem::offset_of!(Gaussian, opacity) == 52
);

#[derive(Debug, Clone, Default)]
pub struct SplatCloud {
    pub gaussians: Vec<Gaussian>,
}

#[derive(Debug, Clone)]
pub struct SplatEditOutput {
    pub cloud: SplatCloud,
    pub summary: String,
    pub modified_count: usize,
    pub total_count: usize,
}

/// Branchless IEEE-754 `e^x` (~1.5e-7 relative error over the full finite range, no libm call,
/// no table lookups): `e^x = 2^n * e^g`, `n = round(x*log2(e))`, `|g| <= ln(2)/2`, with a degree-6
/// polynomial for `e^g` and `2^n` assembled directly in the exponent bits. Inputs are clamped to
/// `[-87.3, 88.3]` (so `n` in `[-126, 127]` and the result is always a finite normal float) with
/// min/max, not branches.
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_145_75; // Cody-Waite split of ln(2) for an exact `x - n*ln2`.
    const LN2_LO: f32 = 1.428_606_8e-6;
    const ROUND_MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23: adding it rounds to nearest integer.
    let x = x.max(-87.3).min(88.3);
    let r = x * LOG2E + ROUND_MAGIC;
    let n = r.to_bits().wrapping_sub(ROUND_MAGIC.to_bits()); // round(x * log2 e) (two's complement)
    let nf = r - ROUND_MAGIC;
    let g = (x - nf * LN2_HI) - nf * LN2_LO;
    let p = 1.0
        + g * (1.0
            + g * (0.5
                + g * (1.0 / 6.0 + g * (1.0 / 24.0 + g * (1.0 / 120.0 + g * (1.0 / 720.0))))));
    p * f32::from_bits(n.wrapping_add(127) << 23)
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-x))
}

/// Fast high-accuracy natural log (~2e-6 max error across all positive floats, ~5x faster than libm ln).
#[inline(always)]
fn fast_ln(x: f32) -> f32 {
    let bits = x.to_bits();
    let exp = ((bits >> 23) as i32) - 127;
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000); // [1, 2)
    let z = (m - 1.0) / (m + 1.0); // [0, 1/3)
    let z2 = z * z;
    let ln_m = 2.0 * z * (1.0 + z2 * (0.333_333_34 + z2 * (0.2 + z2 * 0.142_857_15)));
    (exp as f32) * std::f32::consts::LN_2 + ln_m
}

/// `x^y` for `x >= 0` as `fast_exp(y * fast_ln(x))` (~3e-6 relative, no libm `powf`).
/// Division-free `log2(x)` for positive normal `x` (degree-5 minimax on the mantissa, 1.7e-5
/// max abs error, continuous across octaves): ~8 flops, SLP-vectorisable, for quantisers only.
#[inline(always)]
fn fast_log2_q(x: f32) -> f32 {
    let bits = x.to_bits();
    let e = ((bits >> 23) as i32 - 127) as f32;
    let t = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000) - 1.0;
    e + t * (1.441_879_8 + t * (-0.708_864_5 + t * (0.415_243_26 + t * (-0.193_513_45 + t * 0.045_266_897))))
}

#[inline(always)]
fn fast_pow(x: f32, y: f32) -> f32 {
    fast_exp(y * fast_ln(x))
}


/// Shortest-arc rotation taking local +Z (the thin splat axis) onto `normal`, in closed form:
/// `q = (1 + n.z, -n.y, n.x, 0) / |q|` with `|q|^2 = 2 + 2 n.z` for unit `n` (the half-way
/// quaternion). `|q|` is evaluated as `sqrt(w^2 + n.x^2 + n.y^2)` rather than `sqrt(2w)` so it
/// stays exactly unit near the antipode where `1 + n.z` cancels. The `n = -Z` case is a
/// 180-degree turn about X.
#[inline]
fn quat_from_normal(normal: Vec3) -> [f32; 4] {
    let l2 = normal.length_squared();
    if !(l2 > 1e-12) || !l2.is_finite() {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let n = normal * l2.sqrt().recip();
    let w = 1.0 + n.z;
    if w < 1e-6 {
        return [0.0, 1.0, 0.0, 0.0];
    }
    let inv = (w * w + n.x * n.x + n.y * n.y).sqrt().recip();
    [w * inv, -n.y * inv, n.x * inv, 0.0]
}

/// Minimal 4-lane `f32`/`u32` SIMD layer for the codec hot loops (stable Rust, no nightly
/// `std::simd`): NEON on aarch64 (Apple M-series / A-series, Android arm64), plain `[_; 4]` lanes
/// elsewhere (wasm32 / x86_64, where LLVM vectorises or scalarises them). Every operation has
/// the exact IEEE semantics of the scalar code it replaces (`min`/`max` ignore NaN like
/// `f32::min`, `to_u32` truncates + saturates like `as u32`, no implicit FMA contraction), so
/// both backends produce bit-identical results.
mod simd4 {
    #[cfg(target_arch = "aarch64")]
    mod imp {
        // SAFETY (all `unsafe` register-only intrinsic calls below): NEON is a mandatory baseline
        // feature of every aarch64 target, so these `#[target_feature(enable = "neon")]`
        // intrinsics are always available; they touch no memory.
        use core::arch::aarch64::*;
        use super::super::Gaussian;

        #[derive(Clone, Copy)]
        pub struct F4(float32x4_t);
        #[derive(Clone, Copy)]
        pub struct U4(uint32x4_t);

        impl F4 {
            #[inline(always)]
            pub fn new(a: [f32; 4]) -> Self {
                // SAFETY: `a` is 4 readable, initialised f32s; NEON loads need no alignment.
                Self(unsafe { vld1q_f32(a.as_ptr()) })
            }
            #[inline(always)]
            pub fn splat(v: f32) -> Self {
                Self(unsafe { vdupq_n_f32(v) })
            }
            #[inline(always)]
            pub fn to_array(self) -> [f32; 4] {
                let mut o = [0.0f32; 4];
                // SAFETY: `o` is 4 writable f32s.
                unsafe { vst1q_f32(o.as_mut_ptr(), self.0) };
                o
            }
            #[inline(always)]
            pub fn add(self, o: Self) -> Self {
                Self(unsafe { vaddq_f32(self.0, o.0) })
            }
            #[inline(always)]
            pub fn sub(self, o: Self) -> Self {
                Self(unsafe { vsubq_f32(self.0, o.0) })
            }
            #[inline(always)]
            pub fn mul(self, o: Self) -> Self {
                Self(unsafe { vmulq_f32(self.0, o.0) })
            }
            #[inline(always)]
            pub fn div(self, o: Self) -> Self {
                Self(unsafe { vdivq_f32(self.0, o.0) })
            }
            #[inline(always)]
            pub fn mul_add(self, b: Self, c: Self) -> Self {
                Self(unsafe { vfmaq_f32(c.0, self.0, b.0) })
            }
            #[inline(always)]
            pub fn min(self, o: Self) -> Self {
                Self(unsafe { vminnmq_f32(self.0, o.0) })
            }
            #[inline(always)]
            pub fn max(self, o: Self) -> Self {
                Self(unsafe { vmaxnmq_f32(self.0, o.0) })
            }
            #[allow(dead_code)] // used by the portable `.dsplat` encoder only
            #[inline(always)]
            pub fn abs(self) -> Self {
                Self(unsafe { vabsq_f32(self.0) })
            }
            #[allow(dead_code)] // used by the portable `.dsplat` decoder only
            #[inline(always)]
            pub fn sqrt(self) -> Self {
                Self(unsafe { vsqrtq_f32(self.0) })
            }
            #[allow(dead_code)] // used by the portable `.dsplat` codec only
            #[inline(always)]
            pub fn rsqrt_nr2(self) -> Self {
                // Fast reciprocal sqrt with 2 Newton-Raphson steps (~23-bit accuracy).
                unsafe {
                    let mut r = vrsqrteq_f32(self.0);
                    r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(self.0, r), r));
                    r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(self.0, r), r));
                    Self(r)
                }
            }
            #[allow(dead_code)] // used by the portable `.dsplat` codec only
            #[inline(always)]
            pub fn gt(self, o: Self) -> U4 {
                U4(unsafe { vcgtq_f32(self.0, o.0) })
            }
            #[allow(dead_code)] // used by the portable `.dsplat` encoder only
            #[inline(always)]
            pub fn lt(self, o: Self) -> U4 {
                U4(unsafe { vcltq_f32(self.0, o.0) })
            }
            #[allow(dead_code)] // used by the portable `.dsplat` codec only
            #[inline(always)]
            pub fn select(m: U4, a: Self, b: Self) -> Self {
                Self(unsafe { vbslq_f32(m.0, a.0, b.0) })
            }
            #[inline(always)]
            pub fn to_bits(self) -> U4 {
                U4(unsafe { vreinterpretq_u32_f32(self.0) })
            }
            #[inline(always)]
            pub fn from_bits(u: U4) -> Self {
                Self(unsafe { vreinterpretq_f32_u32(u.0) })
            }
            #[allow(dead_code)] // used by the portable `.dsplat` encoder only
            /// `as u32` per lane: truncating, saturating, NaN -> 0.
            #[inline(always)]
            pub fn to_u32(self) -> U4 {
                U4(unsafe { vcvtq_u32_f32(self.0) })
            }
            #[allow(dead_code)] // used by the portable `.dsplat` decoder only
            #[inline(always)]
            pub fn from_u32(u: U4) -> Self {
                Self(unsafe { vcvtq_f32_u32(u.0) })
            }
            /// Lanes reinterpreted as two's-complement i32, converted to f32.
            #[inline(always)]
            pub fn from_i32_bits(u: U4) -> Self {
                Self(unsafe { vcvtq_f32_s32(vreinterpretq_s32_u32(u.0)) })
            }
        }

        impl U4 {
            #[inline(always)]
            pub fn splat(v: u32) -> Self {
                Self(unsafe { vdupq_n_u32(v) })
            }
            #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
            #[inline(always)]
            pub fn to_array(self) -> [u32; 4] {
                let mut o = [0u32; 4];
                // SAFETY: `o` is 4 writable u32s.
                unsafe { vst1q_u32(o.as_mut_ptr(), self.0) };
                o
            }
            #[inline(always)]
            pub fn and(self, o: Self) -> Self {
                Self(unsafe { vandq_u32(self.0, o.0) })
            }
            #[inline(always)]
            pub fn or(self, o: Self) -> Self {
                Self(unsafe { vorrq_u32(self.0, o.0) })
            }
            #[allow(dead_code)] // used by the portable `.dsplat` encoder only
            #[inline(always)]
            pub fn not(self) -> Self {
                Self(unsafe { vmvnq_u32(self.0) })
            }
            #[inline(always)]
            pub fn sub(self, o: Self) -> Self {
                Self(unsafe { vsubq_u32(self.0, o.0) })
            }
            #[allow(dead_code)] // used by the portable `.dsplat` decoder only
            #[inline(always)]
            pub fn min(self, o: Self) -> Self {
                Self(unsafe { vminq_u32(self.0, o.0) })
            }
            #[inline(always)]
            pub fn shl<const N: i32>(self) -> Self {
                Self(unsafe { vshlq_n_u32::<N>(self.0) })
            }
            #[inline(always)]
            pub fn shr<const N: i32>(self) -> Self {
                Self(unsafe { vshrq_n_u32::<N>(self.0) })
            }
            #[allow(dead_code)] // used by the portable `.dsplat` codec only
            #[inline(always)]
            pub fn eq(self, o: Self) -> Self {
                Self(unsafe { vceqq_u32(self.0, o.0) })
            }
            #[allow(dead_code)] // used by the portable `.dsplat` codec only
            #[inline(always)]
            pub fn lt(self, o: Self) -> Self {
                Self(unsafe { vcltq_u32(self.0, o.0) })
            }
            #[allow(dead_code)] // used by the portable `.dsplat` encoder only
            #[inline(always)]
            pub fn select(m: Self, a: Self, b: Self) -> Self {
                Self(unsafe { vbslq_u32(m.0, a.0, b.0) })
            }
        }

        #[allow(dead_code)] // used by the portable `.dsplat` encoder only
        /// Stores 4 records of 3 u32 words (`a[i], b[i], c[i]`), little-endian, as 48 bytes.
        #[inline(always)]
        pub fn store3(dst: &mut [u8; 48], a: U4, b: U4, c: U4) {
            // SAFETY: `dst` is exactly 48 writable bytes; ST3 needs no alignment; aarch64 is LE.
            unsafe { vst3q_u32(dst.as_mut_ptr().cast::<u32>(), uint32x4x3_t(a.0, b.0, c.0)) };
        }

        /// Inverse of [`store3`].
        #[allow(dead_code)] // used by the portable `.dsplat` decoder only
        #[inline(always)]
        pub fn load3(src: &[u8; 48]) -> (U4, U4, U4) {
            // SAFETY: `src` is exactly 48 readable bytes; LD3 needs no alignment; aarch64 is LE.
            let v = unsafe { vld3q_u32(src.as_ptr().cast::<u32>()) };
            (U4(v.0), U4(v.1), U4(v.2))
        }

        #[inline(always)]
        pub fn transpose4(a: F4, b: F4, c: F4, d: F4) -> (F4, F4, F4, F4) {
            // SAFETY: register-only NEON intrinsics.
            unsafe {
                let t0 = vtrn1q_f32(a.0, b.0);
                let t1 = vtrn2q_f32(a.0, b.0);
                let t2 = vtrn1q_f32(c.0, d.0);
                let t3 = vtrn2q_f32(c.0, d.0);
                let o0 = vreinterpretq_f32_u64(vtrn1q_u64(vreinterpretq_u64_f32(t0), vreinterpretq_u64_f32(t2)));
                let o1 = vreinterpretq_f32_u64(vtrn1q_u64(vreinterpretq_u64_f32(t1), vreinterpretq_u64_f32(t3)));
                let o2 = vreinterpretq_f32_u64(vtrn2q_u64(vreinterpretq_u64_f32(t0), vreinterpretq_u64_f32(t2)));
                let o3 = vreinterpretq_f32_u64(vtrn2q_u64(vreinterpretq_u64_f32(t1), vreinterpretq_u64_f32(t3)));
                (F4(o0), F4(o1), F4(o2), F4(o3))
            }
        }

        #[allow(dead_code)] // used by the portable `.dsplat` encoder only
        #[inline(always)]
        pub fn load_gaussian4(g: &[Gaussian; 4]) -> ([F4; 3], [F4; 3], [F4; 4], [F4; 3], F4) {
            // SAFETY: `Gaussian` is `#[repr(C)]` 56 bytes (14 initialised `f32`s).
            unsafe {
                let l4 = |i: usize, off: usize| F4(vld1q_f32(g[i].pos.as_ptr().add(off)));
                let l2 = |i: usize, off: usize| {
                    F4(vcombine_f32(vld1_f32(g[i].pos.as_ptr().add(off)), vdup_n_f32(0.0)))
                };
                let (px, py, pz, sx) = transpose4(l4(0, 0), l4(1, 0), l4(2, 0), l4(3, 0));
                let (sy, sz, r0, r1) = transpose4(l4(0, 4), l4(1, 4), l4(2, 4), l4(3, 4));
                let (r2, r3, cr, cg) = transpose4(l4(0, 8), l4(1, 8), l4(2, 8), l4(3, 8));
                let (cb, op, _, _) = transpose4(l2(0, 12), l2(1, 12), l2(2, 12), l2(3, 12));
                ([px, py, pz], [sx, sy, sz], [r0, r1, r2, r3], [cr, cg, cb], op)
            }
        }

        #[allow(dead_code)] // used by the portable `.dsplat` decoder only
        #[inline(always)]
        pub fn store_gaussian4(
            dst: &mut [Gaussian; 4],
            pos: [F4; 3],
            scale: [F4; 3],
            rot: [F4; 4],
            color: [F4; 3],
            opacity: F4,
        ) {
            let (a0, a1, a2, a3) = transpose4(pos[0], pos[1], pos[2], scale[0]);
            let (b0, b1, b2, b3) = transpose4(scale[1], scale[2], rot[0], rot[1]);
            let (c0, c1, c2, c3) = transpose4(rot[2], rot[3], color[0], color[1]);
            let a = [a0, a1, a2, a3];
            let b = [b0, b1, b2, b3];
            let c = [c0, c1, c2, c3];
            // SAFETY: `Gaussian` is `#[repr(C)]` 56 bytes; offsets 0, 4, 8 (16B) + 12 (8B) cover all 56B.
            unsafe {
                let d01 = vzip1q_f32(color[2].0, opacity.0);
                let d23 = vzip2q_f32(color[2].0, opacity.0);
                let d = [vget_low_f32(d01), vget_high_f32(d01), vget_low_f32(d23), vget_high_f32(d23)];
                for k in 0..4 {
                    let p = dst[k].pos.as_mut_ptr();
                    vst1q_f32(p, a[k].0);
                    vst1q_f32(p.add(4), b[k].0);
                    vst1q_f32(p.add(8), c[k].0);
                    vst1_f32(p.add(12), d[k]);
                }
            }
        }

        #[inline(always)]
        pub fn load_rgba4(g: &[Gaussian; 4]) -> (F4, F4, F4, F4) {
            // SAFETY: `color` (12B) + `opacity` (4B) = 16 contiguous `f32` bytes at offset 40.
            unsafe {
                let l = |i: usize| F4(vld1q_f32(g[i].color.as_ptr()));
                transpose4(l(0), l(1), l(2), l(3))
            }
        }

        #[inline(always)]
        pub fn store_rgba4(g: &mut [Gaussian; 4], r: F4, gr: F4, b: F4, op: F4) {
            let (v0, v1, v2, v3) = transpose4(r, gr, b, op);
            let v = [v0, v1, v2, v3];
            // SAFETY: `color` (12B) + `opacity` (4B) = 16 contiguous `f32` bytes at offset 40.
            unsafe {
                for k in 0..4 {
                    vst1q_f32(g[k].color.as_mut_ptr(), v[k].0);
                }
            }
        }

        #[allow(dead_code)] // used by the portable `.dsplat` decoder only
        #[inline(always)]
        pub fn gather_lut64(lut: &[f32; 64], idx: U4) -> F4 {
            // SAFETY: every index is masked with `& 63` into a 64-element `[f32; 64]`.
            unsafe {
                let idx64 = vreinterpretq_u64_u32(idx.0);
                let w0 = vgetq_lane_u64::<0>(idx64) as usize;
                let w1 = vgetq_lane_u64::<1>(idx64) as usize;
                let p = lut.as_ptr();
                let mut v = vld1q_dup_f32(p.add(w0 & 63));
                v = vld1q_lane_f32::<1>(p.add((w0 >> 32) & 63), v);
                v = vld1q_lane_f32::<2>(p.add(w1 & 63), v);
                v = vld1q_lane_f32::<3>(p.add((w1 >> 32) & 63), v);
                F4(v)
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    mod imp {
        use super::super::Gaussian;
        #[derive(Clone, Copy)]
        pub struct F4([f32; 4]);
        #[derive(Clone, Copy)]
        pub struct U4([u32; 4]);

        #[inline(always)]
        fn mapf(a: [f32; 4], b: [f32; 4], f: impl Fn(f32, f32) -> f32) -> [f32; 4] {
            [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
        }
        #[inline(always)]
        fn mapu(a: [u32; 4], b: [u32; 4], f: impl Fn(u32, u32) -> u32) -> [u32; 4] {
            [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
        }
        #[inline(always)]
        fn mask(b: bool) -> u32 {
            if b { u32::MAX } else { 0 }
        }

        impl F4 {
            #[inline(always)]
            pub fn new(a: [f32; 4]) -> Self {
                Self(a)
            }
            #[inline(always)]
            pub fn splat(v: f32) -> Self {
                Self([v; 4])
            }
            #[inline(always)]
            pub fn to_array(self) -> [f32; 4] {
                self.0
            }
            #[inline(always)]
            pub fn add(self, o: Self) -> Self {
                Self(mapf(self.0, o.0, |a, b| a + b))
            }
            #[inline(always)]
            pub fn sub(self, o: Self) -> Self {
                Self(mapf(self.0, o.0, |a, b| a - b))
            }
            #[inline(always)]
            pub fn mul(self, o: Self) -> Self {
                Self(mapf(self.0, o.0, |a, b| a * b))
            }
            #[inline(always)]
            pub fn div(self, o: Self) -> Self {
                Self(mapf(self.0, o.0, |a, b| a / b))
            }
            #[inline(always)]
            pub fn mul_add(self, b: Self, c: Self) -> Self {
                self.mul(b).add(c)
            }
            #[inline(always)]
            pub fn min(self, o: Self) -> Self {
                Self(mapf(self.0, o.0, f32::min))
            }
            #[inline(always)]
            pub fn max(self, o: Self) -> Self {
                Self(mapf(self.0, o.0, f32::max))
            }
            #[inline(always)]
            pub fn abs(self) -> Self {
                Self(self.0.map(f32::abs))
            }
            #[inline(always)]
            pub fn sqrt(self) -> Self {
                Self(self.0.map(f32::sqrt))
            }
            #[inline(always)]
            pub fn rsqrt_nr2(self) -> Self {
                Self(self.0.map(|x| x.sqrt().recip()))
            }
            #[inline(always)]
            pub fn gt(self, o: Self) -> U4 {
                U4([0, 1, 2, 3].map(|k| mask(self.0[k] > o.0[k])))
            }
            #[inline(always)]
            pub fn lt(self, o: Self) -> U4 {
                U4([0, 1, 2, 3].map(|k| mask(self.0[k] < o.0[k])))
            }
            #[inline(always)]
            pub fn select(m: U4, a: Self, b: Self) -> Self {
                Self([0, 1, 2, 3].map(|k| if m.0[k] != 0 { a.0[k] } else { b.0[k] }))
            }
            #[inline(always)]
            pub fn to_bits(self) -> U4 {
                U4(self.0.map(f32::to_bits))
            }
            #[inline(always)]
            pub fn from_bits(u: U4) -> Self {
                Self(u.0.map(f32::from_bits))
            }
            #[inline(always)]
            pub fn to_u32(self) -> U4 {
                U4(self.0.map(|v| v as u32))
            }
            #[inline(always)]
            pub fn from_u32(u: U4) -> Self {
                Self(u.0.map(|v| v as f32))
            }
            #[inline(always)]
            pub fn from_i32_bits(u: U4) -> Self {
                Self(u.0.map(|v| v as i32 as f32))
            }
        }

        impl U4 {
            #[inline(always)]
            pub fn splat(v: u32) -> Self {
                Self([v; 4])
            }
            #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
            #[inline(always)]
            pub fn to_array(self) -> [u32; 4] {
                self.0
            }
            #[inline(always)]
            pub fn and(self, o: Self) -> Self {
                Self(mapu(self.0, o.0, |a, b| a & b))
            }
            #[inline(always)]
            pub fn or(self, o: Self) -> Self {
                Self(mapu(self.0, o.0, |a, b| a | b))
            }
            #[inline(always)]
            pub fn not(self) -> Self {
                Self(self.0.map(|a| !a))
            }
            #[inline(always)]
            pub fn sub(self, o: Self) -> Self {
                Self(mapu(self.0, o.0, u32::wrapping_sub))
            }
            #[inline(always)]
            pub fn min(self, o: Self) -> Self {
                Self(mapu(self.0, o.0, u32::min))
            }
            #[inline(always)]
            pub fn shl<const N: i32>(self) -> Self {
                Self(self.0.map(|a| a << N))
            }
            #[inline(always)]
            pub fn shr<const N: i32>(self) -> Self {
                Self(self.0.map(|a| a >> N))
            }
            #[inline(always)]
            pub fn eq(self, o: Self) -> Self {
                Self(mapu(self.0, o.0, |a, b| mask(a == b)))
            }
            #[inline(always)]
            pub fn lt(self, o: Self) -> Self {
                Self(mapu(self.0, o.0, |a, b| mask(a < b)))
            }
            #[inline(always)]
            pub fn select(m: Self, a: Self, b: Self) -> Self {
                Self([0, 1, 2, 3].map(|k| if m.0[k] != 0 { a.0[k] } else { b.0[k] }))
            }
        }

        #[inline(always)]
        pub fn store3(dst: &mut [u8; 48], a: U4, b: U4, c: U4) {
            for k in 0..4 {
                dst[k * 12..k * 12 + 4].copy_from_slice(&a.0[k].to_le_bytes());
                dst[k * 12 + 4..k * 12 + 8].copy_from_slice(&b.0[k].to_le_bytes());
                dst[k * 12 + 8..k * 12 + 12].copy_from_slice(&c.0[k].to_le_bytes());
            }
        }

        #[inline(always)]
        pub fn load3(src: &[u8; 48]) -> (U4, U4, U4) {
            let w = |o: usize| u32::from_le_bytes([src[o], src[o + 1], src[o + 2], src[o + 3]]);
            (U4([0, 1, 2, 3].map(|k| w(k * 12))), U4([0, 1, 2, 3].map(|k| w(k * 12 + 4))), U4([0, 1, 2, 3].map(|k| w(k * 12 + 8))))
        }

        #[inline(always)]
        pub fn load_gaussian4(g: &[Gaussian; 4]) -> ([F4; 3], [F4; 3], [F4; 4], [F4; 3], F4) {
            let lane = |f: fn(&Gaussian) -> f32| F4::new([f(&g[0]), f(&g[1]), f(&g[2]), f(&g[3])]);
            (
                [lane(|x| x.pos[0]), lane(|x| x.pos[1]), lane(|x| x.pos[2])],
                [lane(|x| x.scale[0]), lane(|x| x.scale[1]), lane(|x| x.scale[2])],
                [lane(|x| x.rot[0]), lane(|x| x.rot[1]), lane(|x| x.rot[2]), lane(|x| x.rot[3])],
                [lane(|x| x.color[0]), lane(|x| x.color[1]), lane(|x| x.color[2])],
                lane(|x| x.opacity),
            )
        }

        #[allow(dead_code)] // used by the portable `.dsplat` decoder only
        #[inline(always)]
        pub fn store_gaussian4(
            dst: &mut [Gaussian; 4],
            pos: [F4; 3],
            scale: [F4; 3],
            rot: [F4; 4],
            color: [F4; 3],
            opacity: F4,
        ) {
            let [px, py, pz] = pos.map(F4::to_array);
            let [sx, sy, sz] = scale.map(F4::to_array);
            let [r0, r1, r2, r3] = rot.map(F4::to_array);
            let [cr, cg, cb] = color.map(F4::to_array);
            let op = opacity.to_array();
            for k in 0..4 {
                dst[k] = Gaussian {
                    pos: [px[k], py[k], pz[k]],
                    scale: [sx[k], sy[k], sz[k]],
                    rot: [r0[k], r1[k], r2[k], r3[k]],
                    color: [cr[k], cg[k], cb[k]],
                    opacity: op[k],
                };
            }
        }

        #[inline(always)]
        pub fn load_rgba4(g: &[Gaussian; 4]) -> (F4, F4, F4, F4) {
            (
                F4::new([g[0].color[0], g[1].color[0], g[2].color[0], g[3].color[0]]),
                F4::new([g[0].color[1], g[1].color[1], g[2].color[1], g[3].color[1]]),
                F4::new([g[0].color[2], g[1].color[2], g[2].color[2], g[3].color[2]]),
                F4::new([g[0].opacity, g[1].opacity, g[2].opacity, g[3].opacity]),
            )
        }

        #[inline(always)]
        pub fn store_rgba4(g: &mut [Gaussian; 4], r: F4, gr: F4, b: F4, op: F4) {
            let [r, gr, b, op] = [r, gr, b, op].map(F4::to_array);
            for k in 0..4 {
                g[k].color = [r[k], gr[k], b[k]];
                g[k].opacity = op[k];
            }
        }

        #[allow(dead_code)] // used by the portable `.dsplat` decoder only
        #[inline(always)]
        pub fn gather_lut64(lut: &[f32; 64], idx: U4) -> F4 {
            let i = idx.to_array();
            F4::new([
                lut[i[0] as usize & 63],
                lut[i[1] as usize & 63],
                lut[i[2] as usize & 63],
                lut[i[3] as usize & 63],
            ])
        }
    }

    #[allow(unused_imports)]
    pub use imp::{F4, U4, gather_lut64, load3, load_gaussian4, load_rgba4, store3, store_gaussian4, store_rgba4};

    /// 4-lane [`super::fast_exp`] using FMA + Estrin evaluation on aarch64.
    #[inline(always)]
    pub fn exp(x: F4) -> F4 {
        const LOG2E: f32 = std::f32::consts::LOG2_E;
        const LN2_HI: f32 = 0.693_145_75;
        const LN2_LO: f32 = 1.428_606_8e-6;
        const ROUND_MAGIC: f32 = 12_582_912.0;
        let s = F4::splat;
        let x = x.max(s(-87.3)).min(s(88.3));
        let r = x.mul_add(s(LOG2E), s(ROUND_MAGIC));
        let n = r.to_bits().sub(U4::splat(ROUND_MAGIC.to_bits()));
        let nf = r.sub(s(ROUND_MAGIC));
        let g = nf.mul_add(s(-LN2_LO), nf.mul_add(s(-LN2_HI), x));
        let g2 = g.mul(g);
        let g4 = g2.mul(g2);
        let p0 = g.mul_add(s(1.0), s(1.0));
        let p1 = g.mul_add(s(1.0 / 6.0), s(0.5));
        let p2 = g.mul_add(s(1.0 / 120.0), s(1.0 / 24.0));
        let p = g4.mul_add(g2.mul_add(s(1.0 / 720.0), p2), g2.mul_add(p1, p0));
        // 2^n assembled in the exponent bits: ((n + 127) << 23).
        let two_n = F4::from_bits(n.sub(U4::splat(0u32.wrapping_sub(127))).shl::<23>());
        p.mul(two_n)
    }

    /// 4-lane [`super::fast_ln`].
    #[inline(always)]
    pub fn ln(x: F4) -> F4 {
        let s = F4::splat;
        let bits = x.to_bits();
        let e = F4::from_i32_bits(bits.shr::<23>().sub(U4::splat(127)));
        let m = F4::from_bits(bits.and(U4::splat(0x007F_FFFF)).or(U4::splat(0x3F80_0000)));
        let z = m.sub(s(1.0)).div(m.add(s(1.0)));
        let z2 = z.mul(z);
        let poly = z2.mul_add(
            z2.mul_add(z2.mul_add(s(0.142_857_15), s(0.2)), s(0.333_333_34)),
            s(1.0),
        );
        let ln_m = s(2.0).mul(z).mul(poly);
        e.mul_add(s(std::f32::consts::LN_2), ln_m)
    }

    /// 4-lane [`super::fast_log2_q`] using FMA + Estrin evaluation.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))] // aarch64 encodes with `encode_v2_neon`
    #[inline(always)]
    pub fn log2_q(x: F4) -> F4 {
        let s = F4::splat;
        let bits = x.to_bits();
        let e = F4::from_i32_bits(bits.shr::<23>().sub(U4::splat(127)));
        let t = F4::from_bits(bits.and(U4::splat(0x007F_FFFF)).or(U4::splat(0x3F80_0000))).sub(s(1.0));
        let t2 = t.mul(t);
        let q0 = t.mul_add(s(-0.708_864_5), s(1.441_879_8));
        let q1 = t.mul_add(s(-0.193_513_45), s(0.415_243_26));
        let p = t2.mul_add(t2.mul_add(s(0.045_266_897), q1), q0);
        t.mul_add(p, e)
    }
}

impl SplatCloud {
    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gaussians.is_empty()
    }

    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        if self.gaussians.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for g in &self.gaussians {
            let p = Vec3::from_array(g.pos);
            if p.is_finite() {
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        if !lo.is_finite() {
            return ([0.0; 3], [0.0; 3]);
        }
        (lo.to_array(), hi.to_array())
    }

    /// Encodes to standard 32-byte-per-splat binary `.splat` format:
    /// `pos: [f32; 3]`, `scale: [f32; 3]`, `rgba: [u8; 4]`, `rot: [u8; 4]` (mapped `(q*128 + 128)`).
    ///
    /// Uses NEON 128-bit vector loads/stores and narrowing saturation on aarch64, uninitialized
    /// output buffer (zero-fill avoided), and rayon-parallel >= `PAR_MIN`.
    pub fn to_splat_bytes(&self) -> Vec<u8> {
        #[cfg(target_arch = "aarch64")]
        #[inline(always)]
        fn encode_run(gs: &[Gaussian], dst: &mut [u8]) {
            use core::arch::aarch64::*;
            // SAFETY: NEON baseline on aarch64; `Gaussian` is `#[repr(C)]` 56B; `dst.len() == gs.len() * 32`.
            unsafe {
                let s_min = vdupq_n_f32(1e-5);
                let zero = vdupq_n_f32(0.0);
                let one = vdupq_n_f32(1.0);
                let half = vdupq_n_f32(0.5);
                let v255 = vdupq_n_f32(255.0);
                let v128 = vdupq_n_f32(128.0);
                let v128_5 = vdupq_n_f32(128.5);
                let id_rot = vld1q_f32([1.0f32, 0.0, 0.0, 0.0].as_ptr());
                let mut gp = gs.as_ptr();
                let mut dp = dst.as_mut_ptr();
                for _ in 0..gs.len() {
                    let p0 = vld1q_f32((*gp).pos.as_ptr());
                    let s0 = vmaxnmq_f32(vld1q_f32((*gp).scale.as_ptr()), s_min);
                    vst1q_f32(dp.cast::<f32>(), p0);
                    vst1q_f32(dp.add(12).cast::<f32>(), s0);

                    let c = vminnmq_f32(vmaxnmq_f32(vld1q_f32((*gp).color.as_ptr()), zero), one);
                    let cu = vqmovn_u32(vcvtq_u32_f32(vfmaq_f32(half, c, v255)));

                    let r = vld1q_f32((*gp).rot.as_ptr());
                    let l2 = vaddvq_f32(vmulq_f32(r, r));
                    let q = if (l2 - 1.0).abs() <= 2e-3 {
                        r
                    } else if l2 > 1e-10 {
                        vmulq_f32(r, vdupq_n_f32(l2.sqrt().recip()))
                    } else {
                        id_rot
                    };
                    let qf = vminnmq_f32(vmaxnmq_f32(vfmaq_f32(v128_5, q, v128), zero), v255);
                    let qu = vqmovn_u32(vcvtq_u32_f32(qf));
                    let bytes8 = vqmovn_u16(vcombine_u16(cu, qu));
                    vst1_u8(dp.add(24), bytes8);
                    gp = gp.add(1);
                    dp = dp.add(32);
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        #[inline(always)]
        fn encode_run(gs: &[Gaussian], dst: &mut [u8]) {
            #[inline(always)]
            fn u8q(v: f32) -> u8 {
                (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
            }
            for (g, d) in gs.iter().zip(dst.chunks_exact_mut(32)) {
                let d: &mut [u8; 32] = d.try_into().unwrap();
                let f = [g.pos[0], g.pos[1], g.pos[2], g.scale[0].max(1e-5), g.scale[1].max(1e-5), g.scale[2].max(1e-5)];
                for k in 0..6 {
                    d[k * 4..k * 4 + 4].copy_from_slice(&f[k].to_le_bytes());
                }
                d[24] = u8q(g.color[0]);
                d[25] = u8q(g.color[1]);
                d[26] = u8q(g.color[2]);
                d[27] = u8q(g.opacity);
                let r = g.rot;
                let l2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3];
                let bad = !(l2 > 1e-10);
                let k = if (l2 - 1.0).abs() > 2e-3 && !bad { l2.sqrt().recip() } else { 1.0 };
                let q = if bad { [1.0, 0.0, 0.0, 0.0] } else { [r[0] * k, r[1] * k, r[2] * k, r[3] * k] };
                for j in 0..4 {
                    d[28 + j] = (q[j] * 128.0 + 128.5).clamp(0.0, 255.0) as u8;
                }
            }
        }
        let n = self.gaussians.len();
        // SAFETY: `encode_run` writes every byte of `out[..n * 32]`.
        let mut out = unsafe { uninit_vec::<u8>(n * 32) };
        let dp = SyncPtr(out.as_mut_ptr());
        let gs = &self.gaussians[..];
        for_ranges_pe(n, 47, &|s, e| {
            // SAFETY: `[s, e)` ranges are disjoint and in bounds of the `n * 32`-byte `out`.
            encode_run(&gs[s..e], unsafe { std::slice::from_raw_parts_mut(dp.get().add(s * 32), (e - s) * 32) });
        });
        out
    }

    /// Decodes standard 32-byte `.splat` binary data.
    ///
    /// Single-pass 4-wide NEON decode on aarch64 directly into an uninitialized `Vec<Gaussian>`
    /// with non-overlapping stores, branchless 4-wide `rsqrt` quaternion normalization, and inline
    /// exponent-bit finite tracking. Parallel >= `CODEC_PAR_MIN` across P+E clusters.
    pub fn from_splat_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.is_empty() || bytes.len() % 32 != 0 {
            return Err(format!("invalid .splat length {}: must be a non-zero multiple of 32", bytes.len()));
        }
        let n = bytes.len() / 32;
        #[inline(always)]
        fn finite(g: &Gaussian) -> bool {
            (g.pos[0] + g.pos[1] + g.pos[2] + g.scale[0] + g.scale[1] + g.scale[2]).is_finite()
        }
        #[cfg(target_arch = "aarch64")]
        #[inline(always)]
        fn decode_run(src: &[u8], dst: &mut [Gaussian]) -> bool {
            use core::arch::aarch64::*;
            const INV255: f32 = 1.0 / 255.0;
            // SAFETY: NEON baseline; `src.len() == dst.len() * 32`; `Gaussian` is `#[repr(C)]` 56B.
            unsafe {
                let s_min = vdupq_n_f32(1e-5);
                let p_min = vld1q_f32([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, 1e-5].as_ptr());
                let inv255 = vdupq_n_f32(INV255);
                let v128 = vdupq_n_s32(128);
                let id_rot = vld1q_f32([1.0f32, 0.0, 0.0, 0.0].as_ptr());
                let zero = vdupq_n_f32(0.0);
                let mut fin0 = zero;
                let mut fin1 = zero;
                let mut min_l2 = vdupq_n_f32(1.0);
                let mut sp = src.as_ptr();
                let mut gp = dst.as_mut_ptr();
                let n4 = dst.len() / 4;
                for _ in 0..n4 {
                    let p0 = vld1q_f32(sp.cast::<f32>());
                    let p1 = vld1q_f32(sp.add(32).cast::<f32>());
                    let p2 = vld1q_f32(sp.add(64).cast::<f32>());
                    let p3 = vld1q_f32(sp.add(96).cast::<f32>());
                    let yz01 = vcombine_f32(vld1_f32(sp.add(16).cast::<f32>()), vld1_f32(sp.add(48).cast::<f32>()));
                    let yz23 = vcombine_f32(vld1_f32(sp.add(80).cast::<f32>()), vld1_f32(sp.add(112).cast::<f32>()));

                    // IEEE-754 FMA finite check: `fma(x, 0.0, acc)` stays `0.0` for all finite floats
                    // (never overflows) and latches `NaN` for any `±Inf` or `NaN` (6 ops vs 12).
                    fin0 = vfmaq_f32(fin0, p0, zero);
                    fin1 = vfmaq_f32(fin1, p1, zero);
                    fin0 = vfmaq_f32(fin0, p2, zero);
                    fin1 = vfmaq_f32(fin1, p3, zero);
                    fin0 = vfmaq_f32(fin0, yz01, zero);
                    fin1 = vfmaq_f32(fin1, yz23, zero);

                    let ps0 = vmaxnmq_f32(p0, p_min);
                    let ps1 = vmaxnmq_f32(p1, p_min);
                    let ps2 = vmaxnmq_f32(p2, p_min);
                    let ps3 = vmaxnmq_f32(p3, p_min);
                    let cyz01 = vmaxnmq_f32(yz01, s_min);
                    let cyz23 = vmaxnmq_f32(yz23, s_min);

                    let u0 = vmovl_u8(vld1_u8(sp.add(24)));
                    let u1 = vmovl_u8(vld1_u8(sp.add(56)));
                    let u2 = vmovl_u8(vld1_u8(sp.add(88)));
                    let u3 = vmovl_u8(vld1_u8(sp.add(120)));

                    let rgba0 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u0))), inv255);
                    let rgba1 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u1))), inv255);
                    let rgba2 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u2))), inv255);
                    let rgba3 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u3))), inv255);

                    // Signed integer rot in [-128, 127]: normalizing directly avoids redundant /128.
                    let rf0 = vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u0))), v128));
                    let rf1 = vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u1))), v128));
                    let rf2 = vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u2))), v128));
                    let rf3 = vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u3))), v128));

                    let l2 = vpaddq_f32(
                        vpaddq_f32(vmulq_f32(rf0, rf0), vmulq_f32(rf1, rf1)),
                        vpaddq_f32(vmulq_f32(rf2, rf2), vmulq_f32(rf3, rf3)),
                    );
                    min_l2 = vminnmq_f32(min_l2, l2);
                    // 1 NR step gives ~16-bit accuracy (>100x tighter than 8-bit .splat quantisation).
                    // Any degenerate `l2 == 0` record is patched to identity once after the loop.
                    let est = vrsqrteq_f32(l2);
                    let inv = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(l2, est), est));

                    let rot0 = vmulq_laneq_f32::<0>(rf0, inv);
                    let rot1 = vmulq_laneq_f32::<1>(rf1, inv);
                    let rot2 = vmulq_laneq_f32::<2>(rf2, inv);
                    let rot3 = vmulq_laneq_f32::<3>(rf3, inv);

                    // Non-overlapping stores: 16B [pos+sx], 8B [sy+sz], 16B [rot], 16B [rgba] = 56B.
                    let o0 = (*gp).pos.as_mut_ptr();
                    vst1q_f32(o0, ps0);
                    vst1_f32(o0.add(4), vget_low_f32(cyz01));
                    vst1q_f32(o0.add(6), rot0);
                    vst1q_f32(o0.add(10), rgba0);

                    let o1 = (*gp.add(1)).pos.as_mut_ptr();
                    vst1q_f32(o1, ps1);
                    vst1_f32(o1.add(4), vget_high_f32(cyz01));
                    vst1q_f32(o1.add(6), rot1);
                    vst1q_f32(o1.add(10), rgba1);

                    let o2 = (*gp.add(2)).pos.as_mut_ptr();
                    vst1q_f32(o2, ps2);
                    vst1_f32(o2.add(4), vget_low_f32(cyz23));
                    vst1q_f32(o2.add(6), rot2);
                    vst1q_f32(o2.add(10), rgba2);

                    let o3 = (*gp.add(3)).pos.as_mut_ptr();
                    vst1q_f32(o3, ps3);
                    vst1_f32(o3.add(4), vget_high_f32(cyz23));
                    vst1q_f32(o3.add(6), rot3);
                    vst1q_f32(o3.add(10), rgba3);

                    sp = sp.add(128);
                    gp = gp.add(4);
                }
                if vminnmvq_f32(min_l2) == 0.0 {
                    for g in &mut dst[..n4 * 4] {
                        if !g.rot[0].is_finite() {
                            g.rot = [1.0, 0.0, 0.0, 0.0];
                        }
                    }
                }
                let fin = vaddq_f32(fin0, fin1);
                let mut ok_all = vminvq_u32(vceqq_f32(fin, fin)) == u32::MAX;
                for _ in (n4 * 4)..dst.len() {
                    let p0 = vld1q_f32(sp.cast::<f32>());
                    let yz = vld1_f32(sp.add(16).cast::<f32>());
                    let yz4 = vcombine_f32(yz, vget_low_f32(zero));
                    let f = vfmaq_f32(vfmaq_f32(zero, p0, zero), yz4, zero);
                    ok_all &= vminvq_u32(vceqq_f32(f, f)) == u32::MAX;
                    let ps0 = vmaxnmq_f32(p0, p_min);
                    let cyz = vget_low_f32(vmaxnmq_f32(yz4, s_min));
                    let u0 = vmovl_u8(vld1_u8(sp.add(24)));
                    let rgba = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u0))), inv255);
                    let rf = vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u0))), v128));
                    let l2 = vaddvq_f32(vmulq_f32(rf, rf));
                    let rot = if l2 > 0.0 { vmulq_f32(rf, vdupq_n_f32(l2.sqrt().recip())) } else { id_rot };
                    let o = (*gp).pos.as_mut_ptr();
                    vst1q_f32(o, ps0);
                    vst1_f32(o.add(4), cyz);
                    vst1q_f32(o.add(6), rot);
                    vst1q_f32(o.add(10), rgba);
                    sp = sp.add(32);
                    gp = gp.add(1);
                }
                ok_all
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        #[inline(always)]
        fn decode_run(src: &[u8], dst: &mut [Gaussian]) -> bool {
            const INV255: f32 = 1.0 / 255.0;
            const INV128: f32 = 1.0 / 128.0;
            let mut all_ok = true;
            for (chunk, out) in src.chunks_exact(32).zip(dst.iter_mut()) {
                let c: &[u8; 32] = chunk.try_into().unwrap();
                let rf = |o: usize| f32::from_le_bytes([c[o], c[o + 1], c[o + 2], c[o + 3]]);
                let pos = [rf(0), rf(4), rf(8)];
                let raw_s = [rf(12), rf(16), rf(20)];
                all_ok &= (pos[0] + pos[1] + pos[2] + raw_s[0] + raw_s[1] + raw_s[2]).is_finite();
                let scale = [raw_s[0].max(1e-5), raw_s[1].max(1e-5), raw_s[2].max(1e-5)];
                let color = [c[24] as f32 * INV255, c[25] as f32 * INV255, c[26] as f32 * INV255];
                let opacity = c[27] as f32 * INV255;
                let r = [
                    (c[28] as f32 - 128.0) * INV128,
                    (c[29] as f32 - 128.0) * INV128,
                    (c[30] as f32 - 128.0) * INV128,
                    (c[31] as f32 - 128.0) * INV128,
                ];
                let l2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3];
                let ok = l2 > 1e-8;
                let inv = if ok { l2.sqrt().recip() } else { 0.0 };
                let rot = [if ok { r[0] * inv } else { 1.0 }, r[1] * inv, r[2] * inv, r[3] * inv];
                *out = Gaussian { pos, scale, rot, color, opacity };
            }
            all_ok
        }
        // SAFETY: `decode_run` writes every `Gaussian` in `gaussians[..n]`.
        let mut gaussians = unsafe { uninit_vec::<Gaussian>(n) };
        let all_ok = std::sync::atomic::AtomicBool::new(true);
        let gp = SyncPtr(gaussians.as_mut_ptr());
        for_ranges_pe(n, 47, &|s, e| {
            // SAFETY: `[s, e)` ranges are disjoint and in bounds of the `n`-element `gaussians`.
            let dst = unsafe { std::slice::from_raw_parts_mut(gp.get().add(s), e - s) };
            if !decode_run(&bytes[s * 32..e * 32], dst) {
                all_ok.store(false, std::sync::atomic::Ordering::Relaxed);
            }
        });
        if !all_ok.into_inner() {
            gaussians.retain(finite);
        }
        Ok(Self { gaussians })
    }

    /// Ultra-compact `.dsplat` v2: EXACTLY 12 bytes (96 bits) per Gaussian + 44-byte header.
    /// 2.67x smaller than `.splat` (32 B), 5.7x smaller than minimal `.ply` (68 B), 20.7x smaller
    /// than SH3 `.ply` (248 B) and 25-50% smaller than Niantic `.spz` (~16-24 B).
    ///
    /// Header (little-endian): `"DSPL"`, `version = 2`, `count`, AABB `lo[3]`, `hi[3]`,
    /// `ln_scale_min`, `ln_scale_max` (identical layout to v1, so readers dispatch on `version`).
    ///
    /// Per-splat bit stream (LSB-first over bytes 0..12, read as `u64` lo + `u32` hi):
    /// | bits   | field                                                   |
    /// |--------|---------------------------------------------------------|
    /// | 0-35   | 3x12-bit AABB-normalised position (x, y, z)             |
    /// | 36-59  | RGBA6666 (sRGB color + linear opacity)                  |
    /// | 60-77  | 3x6-bit log-scale over `[ln_scale_min, ln_scale_max]`   |
    /// | 78-79  | index of the largest-magnitude quaternion component     |
    /// | 80-94  | smallest-3 quaternion, 3x5-bit symmetric (exact zero)   |
    /// | 95     | spare (0)                                               |
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let count = self.gaussians.len();
        let gs = &self.gaussians[..];
        let scan_slice = |gs: &[Gaussian]| -> (Vec3, Vec3, f32, f32) {
            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                let chunks = gs.len() / 4;
                let mut lp = [f32::INFINITY; 4];
                let mut hp = [f32::NEG_INFINITY; 4];
                let mut ls = [f32::INFINITY; 4];
                let mut hs = [0.0f32; 4];
                if chunks > 0 {
                    // SAFETY: NEON baseline; `Gaussian` is `#[repr(C)]` 56B; `chunks * 4 <= gs.len()`.
                    unsafe {
                        let mut lp0 = vdupq_n_f32(f32::INFINITY);
                        let mut lp1 = lp0;
                        let mut hp0 = vdupq_n_f32(f32::NEG_INFINITY);
                        let mut hp1 = hp0;
                        let mut ls0 = lp0;
                        let mut ls1 = lp0;
                        let mut hs0 = vdupq_n_f32(0.0);
                        let mut hs1 = hs0;
                        let mut gp = gs.as_ptr();
                        for _ in 0..chunks {
                            // `pos` + `scale[0]` is 16B contiguous; lane 3 tracks `scale[0]` min/max for free!
                            let p0 = vld1q_f32((*gp).pos.as_ptr());
                            let p1 = vld1q_f32((*gp.add(1)).pos.as_ptr());
                            let p2 = vld1q_f32((*gp.add(2)).pos.as_ptr());
                            let p3 = vld1q_f32((*gp.add(3)).pos.as_ptr());
                            let yz01 = vcombine_f32(
                                vld1_f32((*gp).scale.as_ptr().add(1)),
                                vld1_f32((*gp.add(1)).scale.as_ptr().add(1)),
                            );
                            let yz23 = vcombine_f32(
                                vld1_f32((*gp.add(2)).scale.as_ptr().add(1)),
                                vld1_f32((*gp.add(3)).scale.as_ptr().add(1)),
                            );
                            lp0 = vminnmq_f32(lp0, vminnmq_f32(p0, p1));
                            lp1 = vminnmq_f32(lp1, vminnmq_f32(p2, p3));
                            hp0 = vmaxnmq_f32(hp0, vmaxnmq_f32(p0, p1));
                            hp1 = vmaxnmq_f32(hp1, vmaxnmq_f32(p2, p3));
                            ls0 = vminnmq_f32(ls0, yz01);
                            ls1 = vminnmq_f32(ls1, yz23);
                            hs0 = vmaxnmq_f32(hs0, yz01);
                            hs1 = vmaxnmq_f32(hs1, yz23);
                            gp = gp.add(4);
                        }
                        vst1q_f32(lp.as_mut_ptr(), vminnmq_f32(lp0, lp1));
                        vst1q_f32(hp.as_mut_ptr(), vmaxnmq_f32(hp0, hp1));
                        vst1q_f32(ls.as_mut_ptr(), vminnmq_f32(ls0, ls1));
                        vst1q_f32(hs.as_mut_ptr(), vmaxnmq_f32(hs0, hs1));
                    }
                }
                let mut plo = Vec3::new(lp[0], lp[1], lp[2]);
                let mut phi = Vec3::new(hp[0], hp[1], hp[2]);
                let mut rmin = lp[3].min(ls[0]).min(ls[1]).min(ls[2]).min(ls[3]);
                let mut rmax = hp[3].max(hs[0]).max(hs[1]).max(hs[2]).max(hs[3]);
                for g in &gs[chunks * 4..] {
                    let p = Vec3::from_array(g.pos);
                    if p.is_finite() {
                        plo = plo.min(p);
                        phi = phi.max(p);
                    }
                    for &s in &g.scale {
                        rmin = if s < rmin { s } else { rmin };
                        rmax = if s > rmax { s } else { rmax };
                    }
                }
                (plo, phi, rmin, rmax)
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                use simd4::F4;
                let s = F4::splat;
                let inf = s(f32::INFINITY);
                let ninf = s(f32::NEG_INFINITY);
                let (mut lx, mut ly, mut lz, mut hx, mut hy, mut hz) = (inf, inf, inf, ninf, ninf, ninf);
                let (mut smin, mut smax) = (inf, s(0.0));
                let mut gc = gs.chunks_exact(4);
                for g in &mut gc {
                    let ([x, y, z], [sx, sy, sz], _, _, _) = simd4::load_gaussian4(g.try_into().unwrap());
                    let fin = x.abs().lt(inf).and(y.abs().lt(inf)).and(z.abs().lt(inf));
                    lx = F4::select(fin, lx.min(x), lx);
                    ly = F4::select(fin, ly.min(y), ly);
                    lz = F4::select(fin, lz.min(z), lz);
                    hx = F4::select(fin, hx.max(x), hx);
                    hy = F4::select(fin, hy.max(y), hy);
                    hz = F4::select(fin, hz.max(z), hz);
                    for sc in [sx, sy, sz] {
                        let sc = sc.max(s(1e-5));
                        smin = smin.min(sc);
                        smax = smax.max(sc);
                    }
                }
                let red = |v: F4, f: fn(f32, f32) -> f32| v.to_array().into_iter().reduce(f).unwrap();
                let mut plo = Vec3::new(red(lx, f32::min), red(ly, f32::min), red(lz, f32::min));
                let mut phi = Vec3::new(red(hx, f32::max), red(hy, f32::max), red(hz, f32::max));
                let mut rmin = red(smin, f32::min);
                let mut rmax = red(smax, f32::max);
                for g in gc.remainder() {
                    let p = Vec3::from_array(g.pos);
                    if p.is_finite() {
                        plo = plo.min(p);
                        phi = phi.max(p);
                    }
                    for &s in &g.scale {
                        let sc = s.max(1e-5);
                        rmin = if sc < rmin { sc } else { rmin };
                        rmax = if sc > rmax { sc } else { rmax };
                    }
                }
                (plo, phi, rmin, rmax)
            }
        };

        #[derive(Clone, Copy)]
        struct PackParams {
            lo: [f32; 3],
            inv_p: [f32; 3],
            l2_min: f32,
            inv_l2_span: f32,
            #[allow(dead_code)]
            clamp_scale: bool,
        }

        let finalize_header = |mut plo: Vec3, mut phi: Vec3, mut raw_min: f32, mut raw_max: f32, hdr: &mut [u8]| -> PackParams {
            let mut clamp_scale = !(raw_min >= 1e-5);
            if count > 0 && (!plo.is_finite() || !phi.is_finite() || !raw_max.is_finite() || !raw_min.is_finite()) {
                clamp_scale = true;
                plo = Vec3::splat(f32::INFINITY);
                phi = Vec3::splat(f32::NEG_INFINITY);
                raw_min = f32::INFINITY;
                raw_max = 0.0;
                for g in gs {
                    let p = Vec3::from_array(g.pos);
                    if p.is_finite() {
                        plo = plo.min(p);
                        phi = phi.max(p);
                    }
                    for &s in &g.scale {
                        if s.is_finite() {
                            let sc = s.max(1e-5);
                            raw_min = if sc < raw_min { sc } else { raw_min };
                            raw_max = if sc > raw_max { sc } else { raw_max };
                        }
                    }
                }
            } else if count > 0 {
                raw_min = raw_min.max(1e-5);
                raw_max = raw_max.max(1e-5);
            }
            let (lo, hi) = if plo.is_finite() { (plo.to_array(), phi.to_array()) } else { ([0.0; 3], [0.0; 3]) };
            let (s_min, s_max) = if raw_min.is_finite() && raw_max > 0.0 && raw_max.is_finite() {
                (fast_ln(raw_min), fast_ln(raw_max))
            } else {
                (-6.0, 0.0)
            };
            let inv_s_span = V2_SCALE_MAX / (s_max - s_min).max(1e-4);
            let l2_min = s_min * std::f32::consts::LOG2_E;
            let inv_l2_span = inv_s_span * std::f32::consts::LN_2;
            let inv_p = [
                V2_POS_MAX / (hi[0] - lo[0]).max(1e-5),
                V2_POS_MAX / (hi[1] - lo[1]).max(1e-5),
                V2_POS_MAX / (hi[2] - lo[2]).max(1e-5),
            ];
            hdr[0..4].copy_from_slice(COMPACT_MAGIC);
            hdr[4..8].copy_from_slice(&COMPACT_VERSION_V2.to_le_bytes());
            hdr[8..12].copy_from_slice(&(count as u32).to_le_bytes());
            for i in 0..3 {
                hdr[12 + i * 4..16 + i * 4].copy_from_slice(&lo[i].to_le_bytes());
                hdr[24 + i * 4..28 + i * 4].copy_from_slice(&hi[i].to_le_bytes());
            }
            hdr[36..40].copy_from_slice(&s_min.to_le_bytes());
            hdr[40..44].copy_from_slice(&s_max.to_le_bytes());
            PackParams { lo, inv_p, l2_min, inv_l2_span, clamp_scale }
        };

        // Round-to-nearest quantisers. `as u32` saturates (NaN -> 0), so `min` alone bounds them.
        #[inline(always)]
        fn q_pos(v: f32, lo: f32, inv: f32) -> u64 {
            (((v - lo) * inv + 0.5).min(V2_POS_MAX) as u32) as u64
        }
        #[inline(always)]
        fn q_unit6(v: f32) -> u64 {
            ((v * 63.0 + 0.5).min(63.0) as u32) as u64
        }
        #[inline(always)]
        fn q_rot5(v: f32) -> u32 {
            ((v * V2_ROT_ENC + 15.5).min(30.0) as u32) & 0x1F
        }

        let encode_one = |g: &Gaussian, dst: &mut [u8], pp: &PackParams| {
            let mut sq = [0u32; 3];
            for (q, &sv) in sq.iter_mut().zip(&g.scale) {
                *q = ((fast_log2_q(sv.max(1e-5)) - pp.l2_min) * pp.inv_l2_span + 0.5).min(V2_SCALE_MAX) as u32;
            }
            let [s0, s1, s2] = sq;

            let [r0, r1, r2, r3] = g.rot;
            let l2 = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3;
            let (r0, r1, r2, r3, inv_l) = if (l2 - 1.0).abs() < 2e-3 {
                (r0, r1, r2, r3, 1.0f32)
            } else if l2 > 1e-12 && l2.is_finite() {
                (r0, r1, r2, r3, l2.sqrt().recip())
            } else {
                (1.0, 0.0, 0.0, 0.0, 1.0)
            };
            let (a0, a1, a2, a3) = (r0.abs(), r1.abs(), r2.abs(), r3.abs());
            let c01 = a1 > a0;
            let m01 = if c01 { a1 } else { a0 };
            let i01 = c01 as u32;
            let s01 = if c01 { r1 } else { r0 };
            let c23 = a3 > a2;
            let m23 = if c23 { a3 } else { a2 };
            let i23 = 2 | c23 as u32;
            let s23 = if c23 { r3 } else { r2 };
            let cm = m23 > m01;
            let mi = if cm { i23 } else { i01 };
            let ms = if cm { s23 } else { s01 };
            let k = inv_l.copysign(ms);
            let o0 = if mi == 0 { r1 } else { r0 };
            let o1 = if mi <= 1 { r2 } else { r1 };
            let o2 = if mi <= 2 { r3 } else { r2 };

            let lo_bits: u64 = q_pos(g.pos[0], pp.lo[0], pp.inv_p[0])
                | (q_pos(g.pos[1], pp.lo[1], pp.inv_p[1]) << 12)
                | (q_pos(g.pos[2], pp.lo[2], pp.inv_p[2]) << 24)
                | (q_unit6(g.color[0].max(0.0)) << 36)
                | (q_unit6(g.color[1].max(0.0)) << 42)
                | (q_unit6(g.color[2].max(0.0)) << 48)
                | (q_unit6(g.opacity.max(0.0)) << 54)
                | ((s0 as u64 & 0xF) << 60);
            let hi_bits: u32 = (s0 >> 4)
                | (s1 << 2)
                | (s2 << 8)
                | (mi << 14)
                | (q_rot5(o0 * k) << 16)
                | (q_rot5(o1 * k) << 21)
                | (q_rot5(o2 * k) << 26);
            dst[0..8].copy_from_slice(&lo_bits.to_le_bytes());
            dst[8..12].copy_from_slice(&hi_bits.to_le_bytes());
        };

        #[cfg(not(target_arch = "aarch64"))]
        let encode4 = |g: &[Gaussian], dst: &mut [u8], pp: &PackParams| {
            use simd4::{F4, U4};
            let g: &[Gaussian; 4] = g.try_into().unwrap();
            let dst: &mut [u8; 4 * V2_STRIDE] = dst.try_into().unwrap();
            let s = F4::splat;
            let u = U4::splat;
            let ([px, py, pz], [sx, sy, sz], [r0, r1, r2, r3], [cr, cg, cb], co) = simd4::load_gaussian4(g);
            let qp = |v: F4, lo_a: f32, inv_a: f32| v.sub(s(lo_a)).mul_add(s(inv_a), s(0.5)).min(s(V2_POS_MAX)).to_u32();
            let q6 = |v: F4| v.max(s(0.0)).mul_add(s(63.0), s(0.5)).min(s(63.0)).to_u32();
            let qs = |v: F4| {
                let l2 = simd4::log2_q(v.max(s(1e-5)));
                l2.sub(s(pp.l2_min)).mul_add(s(pp.inv_l2_span), s(0.5)).min(s(V2_SCALE_MAX)).to_u32()
            };
            let (qx, qy, qz) = (qp(px, pp.lo[0], pp.inv_p[0]), qp(py, pp.lo[1], pp.inv_p[1]), qp(pz, pp.lo[2], pp.inv_p[2]));
            let (cr, cg, cb, co) = (q6(cr), q6(cg), q6(cb), q6(co));
            let (s0, s1, s2) = (qs(sx), qs(sy), qs(sz));

            let l2 = r0.mul_add(r0, r1.mul_add(r1, r2.mul_add(r2, r3.mul(r3))));
            let unit = l2.sub(s(1.0)).abs().lt(s(2e-3));
            let ok = l2.gt(s(1e-12)).and(l2.lt(s(f32::INFINITY)));
            let degen = unit.or(ok).not();
            let inv_l = F4::select(unit.or(degen), s(1.0), l2.rsqrt_nr2());
            let r0 = F4::select(degen, s(1.0), r0);
            let (r1, r2, r3) = (F4::select(degen, s(0.0), r1), F4::select(degen, s(0.0), r2), F4::select(degen, s(0.0), r3));
            let (a0, a1, a2, a3) = (r0.abs(), r1.abs(), r2.abs(), r3.abs());
            let c01 = a1.gt(a0);
            let c23 = a3.gt(a2);
            let m01 = F4::select(c01, a1, a0);
            let m23 = F4::select(c23, a3, a2);
            let cm = m23.gt(m01);
            let i01 = c01.and(u(1));
            let i23 = c23.and(u(1)).or(u(2));
            let mi = U4::select(cm, i23, i01);
            let ms = F4::select(cm, F4::select(c23, r3, r2), F4::select(c01, r1, r0));
            let k = F4::from_bits(inv_l.to_bits().and(u(0x7FFF_FFFF)).or(ms.to_bits().and(u(0x8000_0000))));
            let q5 = |v: F4| v.mul(k).mul_add(s(V2_ROT_ENC), s(15.5)).min(s(30.0)).to_u32().and(u(0x1F));
            let o0 = q5(F4::select(mi.eq(u(0)), r1, r0));
            let o1 = q5(F4::select(mi.lt(u(2)), r2, r1));
            let o2 = q5(F4::select(mi.lt(u(3)), r3, r2));

            let w0 = qx.or(qy.shl::<12>()).or(qz.shl::<24>());
            let w1 = qz
                .shr::<8>()
                .or(cr.shl::<4>())
                .or(cg.shl::<10>())
                .or(cb.shl::<16>())
                .or(co.shl::<22>())
                .or(s0.and(u(0xF)).shl::<28>());
            let w2 = s0
                .shr::<4>()
                .or(s1.shl::<2>())
                .or(s2.shl::<8>())
                .or(mi.shl::<14>())
                .or(o0.shl::<16>())
                .or(o1.shl::<21>())
                .or(o2.shl::<26>());
            simd4::store3(dst, w0, w1, w2);
        };

        let encode_run = |gs: &[Gaussian], dst: &mut [u8], pp: &PackParams| {
            #[cfg(target_arch = "aarch64")]
            let done = if pp.clamp_scale {
                encode_v2_neon::<true>(gs, dst, pp.lo, pp.inv_p, pp.l2_min, pp.inv_l2_span)
            } else {
                encode_v2_neon::<false>(gs, dst, pp.lo, pp.inv_p, pp.l2_min, pp.inv_l2_span)
            };
            #[cfg(not(target_arch = "aarch64"))]
            let done = {
                let mut gc = gs.chunks_exact(4);
                let mut dc = dst.chunks_exact_mut(4 * V2_STRIDE);
                for (g4, d4) in (&mut gc).zip(&mut dc) {
                    encode4(g4, d4, pp);
                }
                gs.len() & !3
            };
            for (g, d) in gs[done..].iter().zip(dst[done * V2_STRIDE..].chunks_exact_mut(V2_STRIDE)) {
                encode_one(g, d, pp);
            }
        };

        // SAFETY: header (44B) + `encode_run` (count * 12B) write every byte of `out`.
        let mut out = unsafe { uninit_vec::<u8>(COMPACT_HEADER_LEN + count * V2_STRIDE) };

        // Single 2-phase P-core fan-out: phase 0 scans bounds + scale range, `after(0)` reduces
        // and writes the 44B header, phase 1 packs all records — ONE dispatch, ZERO heap scratch,
        // full P-cluster turbo (writes only 192 KB so not write-bandwidth bound).
        #[cfg(feature = "threads")]
        if count >= CODEC_PAR_MIN && dynpar::p_threads() >= 2 {
            const MAX_SCAN: usize = 32;
            let pt = dynpar::p_threads();
            let scan_nc = (pt * 2).clamp(4, MAX_SCAN);
            let scan_ch = (count.div_ceil(scan_nc) + 3) & !3;
            let scan_nc = count.div_ceil(scan_ch);
            let pack_nc = (pt * 2).min((count / 512).max(2));
            let pack_ch = (count.div_ceil(pack_nc) + 3) & !3;
            let pack_nc = count.div_ceil(pack_ch);
            let init = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY), f32::INFINITY, 0.0f32);
            let mut parts = [init; MAX_SCAN];
            let mut params = PackParams { lo: [0.0; 3], inv_p: [0.0; 3], l2_min: 0.0, inv_l2_span: 0.0, clamp_scale: true };
            let part_p = SyncPtr(parts.as_mut_ptr());
            let param_p = SyncPtr(&mut params as *mut PackParams);
            let hdr_p = SyncPtr(out.as_mut_ptr());
            let dp = SyncPtr(out[COMPACT_HEADER_LEN..].as_mut_ptr());
            let body = |phase: usize, c: usize| {
                if phase == 0 {
                    let s = c * scan_ch;
                    let e = (s + scan_ch).min(count);
                    let r = scan_slice(&gs[s..e]);
                    // SAFETY: chunk `c < scan_nc` exclusively owns `parts[c]`.
                    unsafe { *part_p.get().add(c) = r };
                } else {
                    let s = c * pack_ch;
                    let e = (s + pack_ch).min(count);
                    // SAFETY: `params` was written by `after(0)` before phase 1 opened; `[s, e)` is disjoint.
                    let pp = unsafe { &*param_p.get() };
                    let d = unsafe { std::slice::from_raw_parts_mut(dp.get().add(s * V2_STRIDE), (e - s) * V2_STRIDE) };
                    encode_run(&gs[s..e], d, pp);
                }
            };
            let after = |phase: usize| {
                if phase == 0 {
                    // SAFETY: runs once after all phase-0 chunks completed and before phase 1 opens.
                    let ps = unsafe { std::slice::from_raw_parts(part_p.get(), scan_nc) };
                    let (plo, phi, rmin, rmax) = ps.iter().copied().fold(init, |a, b| (a.0.min(b.0), a.1.max(b.1), a.2.min(b.2), a.3.max(b.3)));
                    let hdr = unsafe { std::slice::from_raw_parts_mut(hdr_p.get(), COMPACT_HEADER_LEN) };
                    unsafe { *param_p.get() = finalize_header(plo, phi, rmin, rmax, hdr) };
                }
            };
            dynpar::phased(&[scan_nc, pack_nc], &body, &after);
            return out;
        }

        let (plo, phi, rmin, rmax) = scan_slice(gs);
        let params = finalize_header(plo, phi, rmin, rmax, &mut out[..COMPACT_HEADER_LEN]);
        if count > 0 {
            encode_run(gs, &mut out[COMPACT_HEADER_LEN..], &params);
        }
        out
    }

    /// Decodes `.dsplat` compact binary data: v2 (12 B/splat, current writer) and legacy v1
    /// (16 B/splat) are both accepted, dispatched on the header `version` field.
    pub fn from_compact_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < COMPACT_HEADER_LEN || &bytes[0..4] != COMPACT_MAGIC {
            return Err("invalid compact .dsplat header".into());
        }
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let stride = match version {
            COMPACT_VERSION_V1 => V1_STRIDE,
            COMPACT_VERSION_V2 => V2_STRIDE,
            v => return Err(format!("unsupported compact .dsplat version {v}")),
        };
        let count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let end = count
            .checked_mul(stride)
            .and_then(|b| b.checked_add(COMPACT_HEADER_LEN))
            .ok_or("compact .dsplat count overflow")?;
        if bytes.len() < end {
            return Err("compact .dsplat payload truncated".into());
        }
        let rf = |o: usize| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        let lo = [rf(12), rf(16), rf(20)];
        let hi = [rf(24), rf(28), rf(32)];
        let (s_min, s_max) = (rf(36), rf(40));
        let payload = &bytes[COMPACT_HEADER_LEN..end];
        if version == COMPACT_VERSION_V1 {
            return Ok(Self::decode_compact_v1(payload, lo, hi, s_min, s_max));
        }

        let p_step = [
            (hi[0] - lo[0]) / V2_POS_MAX,
            (hi[1] - lo[1]) / V2_POS_MAX,
            (hi[2] - lo[2]) / V2_POS_MAX,
        ];
        // 64-entry scale LUT via 2 exps + 15 SIMD multiplies (~15 ns vs ~350 ns for 64 scalar exps).
        let s_step = (s_max - s_min) / V2_SCALE_MAX;
        let mut scale_lut = [0.0f32; 64];
        {
            let e0 = fast_exp(s_min);
            let r = fast_exp(s_step);
            let r2 = r * r;
            let r3 = r2 * r;
            let r4 = simd4::F4::splat(r2 * r2);
            let mut v = simd4::F4::new([e0, e0 * r, e0 * r2, e0 * r3]);
            for k in 0..16 {
                scale_lut[k * 4..k * 4 + 4].copy_from_slice(&v.to_array());
                v = v.mul(r4);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        const INV63: f32 = 1.0 / 63.0;

        // 4 records per step (portable fallback; aarch64 uses `decode_v2_neon`).
        #[cfg(not(target_arch = "aarch64"))]
        let decode4 = |src: &[u8; 4 * V2_STRIDE], dst: &mut [Gaussian; 4]| {
            use simd4::{F4, U4};
            let (w0, w1, w2) = simd4::load3(src);
            let s = F4::splat;
            let u = U4::splat;
            let f = F4::from_u32;
            let px = f(w0.and(u(0xFFF))).mul_add(s(p_step[0]), s(lo[0]));
            let py = f(w0.shr::<12>().and(u(0xFFF))).mul_add(s(p_step[1]), s(lo[1]));
            let pz = f(w0.shr::<24>().or(w1.and(u(0xF)).shl::<8>())).mul_add(s(p_step[2]), s(lo[2]));
            let c6 = |x: U4| f(x.and(u(0x3F))).mul(s(INV63));
            let (cr, cg, cb, op) = (c6(w1.shr::<4>()), c6(w1.shr::<10>()), c6(w1.shr::<16>()), c6(w1.shr::<22>()));
            let sc = |x: U4| simd4::gather_lut64(&scale_lut, x);
            let (sx, sy, sz) = (sc(w1.shr::<28>().or(w2.shl::<4>())), sc(w2.shr::<2>()), sc(w2.shr::<8>()));
            let mi = w2.shr::<14>().and(u(3));
            // 5-bit symmetric smallest-3 component: (min(q, 30) - 15) / (15 sqrt 2).
            let rq = |x: U4| f(x.and(u(0x1F)).min(u(30))).sub(s(15.0)).mul(s(V2_ROT_DEC));
            let (a, b, c) = (rq(w2.shr::<16>()), rq(w2.shr::<21>()), rq(w2.shr::<26>()));
            let ss = a.mul_add(a, b.mul_add(b, c.mul(c)));
            let w = s(1.0).sub(ss).max(s(0.0)).sqrt();
            // Quantisation can push |abc| slightly past 1: renormalise (select, no branch).
            let k = F4::select(ss.gt(s(1.0)), ss.rsqrt_nr2(), s(1.0));
            let (a, b, c) = (a.mul(k), b.mul(k), c.mul(k));
            let (m0, m1, m2) = (mi.eq(u(0)), mi.eq(u(1)), mi.eq(u(2)));
            let r0 = F4::select(m0, w, a);
            let r1 = F4::select(m0, a, F4::select(m1, w, b));
            let r2 = F4::select(mi.lt(u(2)), b, F4::select(m2, w, c));
            let r3 = F4::select(mi.lt(u(3)), c, w);
            simd4::store_gaussian4(dst, [px, py, pz], [sx, sy, sz], [r0, r1, r2, r3], [cr, cg, cb], op);
        };
        let decode_run = |src: &[u8], dst: &mut [Gaussian]| {
            #[cfg(target_arch = "aarch64")]
            {
                let done = decode_v2_neon(src, dst, lo, p_step, &scale_lut);
                let rem_s = &src[done * V2_STRIDE..];
                let rem_d = &mut dst[done..];
                if !rem_d.is_empty() {
                    let mut pad = [0u8; 4 * V2_STRIDE];
                    pad[..rem_s.len()].copy_from_slice(rem_s);
                    let mut tmp = [Gaussian {
                        pos: [0.0; 3],
                        scale: [0.0; 3],
                        rot: [1.0, 0.0, 0.0, 0.0],
                        color: [0.0; 3],
                        opacity: 0.0,
                    }; 4];
                    decode_v2_neon(&pad, &mut tmp, lo, p_step, &scale_lut);
                    rem_d.copy_from_slice(&tmp[..rem_d.len()]);
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut sc = src.chunks_exact(4 * V2_STRIDE);
                let mut dc = dst.chunks_exact_mut(4);
                for (s4, d4) in (&mut sc).zip(&mut dc) {
                    decode4(s4.try_into().unwrap(), d4.try_into().unwrap());
                }
                let rem_s = sc.remainder();
                let rem_d = dc.into_remainder();
                if !rem_d.is_empty() {
                    let mut pad = [0u8; 4 * V2_STRIDE];
                    pad[..rem_s.len()].copy_from_slice(rem_s);
                    let mut tmp = [Gaussian {
                        pos: [0.0; 3],
                        scale: [0.0; 3],
                        rot: [1.0, 0.0, 0.0, 0.0],
                        color: [0.0; 3],
                        opacity: 0.0,
                    }; 4];
                    decode4(&pad, &mut tmp);
                    rem_d.copy_from_slice(&tmp[..rem_d.len()]);
                }
            }
        };

        // SAFETY: `decode_run` writes every `Gaussian` in `gaussians[..count]`.
        let mut gaussians = unsafe { uninit_vec::<Gaussian>(count) };
        let gp = SyncPtr(gaussians.as_mut_ptr());
        for_ranges_pe(count, 44, &|s, e| {
            // SAFETY: `[s, e)` ranges are disjoint and in bounds of the `count`-element `gaussians`.
            let dst = unsafe { std::slice::from_raw_parts_mut(gp.get().add(s), e - s) };
            decode_run(&payload[s * V2_STRIDE..e * V2_STRIDE], dst);
        });
        Ok(Self { gaussians })
    }

    /// Legacy `.dsplat` v1 decoder (16 B/splat: u16 positions, u8 log-scales, RGBA8, 3x7-bit quat).
    fn decode_compact_v1(payload: &[u8], lo: [f32; 3], hi: [f32; 3], s_min: f32, s_max: f32) -> Self {
        let p_step = [(hi[0] - lo[0]) / 65535.0, (hi[1] - lo[1]) / 65535.0, (hi[2] - lo[2]) / 65535.0];
        let s_step = (s_max - s_min) / 255.0;
        const INV255: f32 = 1.0 / 255.0;
        const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
        let mut exp_lut = [0.0f32; 256];
        for (i, v) in exp_lut.iter_mut().enumerate() {
            *v = fast_exp(s_min + i as f32 * s_step);
        }
        let mut rot_lut = [0.0f32; 128];
        for (i, v) in rot_lut.iter_mut().enumerate() {
            *v = ((i as f32 - 63.5) / 63.0) * INV_SQRT2;
        }
        // Exact-size map over chunks_exact (TrustedLen): one allocation, no per-push capacity checks.
        let gaussians: Vec<Gaussian> = payload.chunks_exact(V1_STRIDE).map(|chunk| {
            let pos = [
                lo[0] + u16::from_le_bytes([chunk[0], chunk[1]]) as f32 * p_step[0],
                lo[1] + u16::from_le_bytes([chunk[2], chunk[3]]) as f32 * p_step[1],
                lo[2] + u16::from_le_bytes([chunk[4], chunk[5]]) as f32 * p_step[2],
            ];
            let scale = [exp_lut[chunk[6] as usize], exp_lut[chunk[7] as usize], exp_lut[chunk[8] as usize]];
            let color = [chunk[9] as f32 * INV255, chunk[10] as f32 * INV255, chunk[11] as f32 * INV255];
            let opacity = chunk[12] as f32 * INV255;
            let packed = (chunk[13] as u32) | ((chunk[14] as u32) << 8) | ((chunk[15] as u32) << 16);
            let mi = (packed >> 21) & 3;
            let a = rot_lut[((packed >> 14) & 0x7F) as usize];
            let b = rot_lut[((packed >> 7) & 0x7F) as usize];
            let c = rot_lut[(packed & 0x7F) as usize];
            let w = (1.0 - (a * a + b * b + c * c)).max(0.0).sqrt();
            let rot = [
                if mi == 0 { w } else { a },
                if mi == 0 { a } else if mi == 1 { w } else { b },
                if mi <= 1 { b } else if mi == 2 { w } else { c },
                if mi <= 2 { c } else { w },
            ];
            Gaussian { pos, scale, rot, color, opacity }
        }).collect();
        Self { gaussians }
    }

    /// Exports standard 3DGS binary little-endian `.ply` (compatible with MetalSplatter, Polycam,
    /// SuperSplat, Nerfstudio, Luma, and Blender).
    pub fn to_ply_bytes(&self) -> Vec<u8> {
        let header = format!(
            "ply\n\
             format binary_little_endian 1.0\n\
             comment Generated by Dream Builder / sculpt-engine\n\
             element vertex {}\n\
             property float x\n\
             property float y\n\
             property float z\n\
             property float nx\n\
             property float ny\n\
             property float nz\n\
             property float f_dc_0\n\
             property float f_dc_1\n\
             property float f_dc_2\n\
             property float opacity\n\
             property float scale_0\n\
             property float scale_1\n\
             property float scale_2\n\
             property float rot_0\n\
             property float rot_1\n\
             property float rot_2\n\
             property float rot_3\n\
             end_header\n",
            self.gaussians.len()
        );
        let hlen = header.len();
        // SAFETY: header (hlen) + `write_one` (len * 68) write every byte of `out`.
        let mut out = unsafe { uninit_vec::<u8>(hlen + self.gaussians.len() * 68) };
        out[..hlen].copy_from_slice(header.as_bytes());
        let write_one = |g: &Gaussian, dst: &mut [u8]| {
            let dst: &mut [u8; 68] = dst.try_into().unwrap();
            // 3 log-scales + the opacity logit (`ln(c / (1 - c))`) as ONE f32x4 fast_ln chain.
            let c = g.opacity.clamp(1e-5, 1.0 - 1e-5);
            let ln = simd4::ln(simd4::F4::new([g.scale[0].max(1e-5), g.scale[1].max(1e-5), g.scale[2].max(1e-5), c / (1.0 - c)]))
                .to_array();
            let vals: [f32; 17] = [
                g.pos[0],
                g.pos[1],
                g.pos[2],
                0.0,
                0.0,
                0.0,
                (g.color[0] - 0.5) * INV_SH_C0,
                (g.color[1] - 0.5) * INV_SH_C0,
                (g.color[2] - 0.5) * INV_SH_C0,
                ln[3],
                ln[0],
                ln[1],
                ln[2],
                g.rot[0],
                g.rot[1],
                g.rot[2],
                g.rot[3],
            ];
            for (k, v) in vals.iter().enumerate() {
                dst[k * 4..(k + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
        };
        #[cfg(feature = "threads")]
        if self.gaussians.len() >= PAR_MIN {
            use rayon::prelude::*;
            self.gaussians
                .par_iter()
                .zip(out[hlen..].par_chunks_exact_mut(68))
                .for_each(|(g, dst)| write_one(g, dst));
            return out;
        }
        for (g, dst) in self.gaussians.iter().zip(out[hlen..].chunks_exact_mut(68)) {
            write_one(g, dst);
        }
        out
    }

    /// Parses standard 3DGS `.ply` files (both `binary_little_endian` and `ascii`), handling
    /// arbitrary property orders, RGB-only point-cloud PLYs, and extra `f_rest_*` SH bands.
    pub fn from_ply_bytes(bytes: &[u8]) -> Result<Self, String> {
        let end_marker = b"end_header\n";
        let end_marker_crlf = b"end_header\r\n";
        let (header_end, body_start) = if let Some(pos) = find_subslice(bytes, end_marker) {
            (pos, pos + end_marker.len())
        } else if let Some(pos) = find_subslice(bytes, end_marker_crlf) {
            (pos, pos + end_marker_crlf.len())
        } else {
            return Err("invalid PLY: missing end_header".into());
        };
        let header_str = std::str::from_utf8(&bytes[..header_end]).map_err(|_| "invalid UTF-8 in PLY header")?;

        #[derive(Clone, Copy)]
        enum PlyType { F32, F64, U8, I8, U16, I16, U32, I32 }
        impl PlyType {
            fn size(self) -> usize {
                match self { Self::U8 | Self::I8 => 1, Self::U16 | Self::I16 => 2, Self::F32 | Self::U32 | Self::I32 => 4, Self::F64 => 8 }
            }
            fn read_le(self, b: &[u8]) -> f32 {
                match self {
                    Self::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                    Self::F64 => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
                    Self::U8 => b[0] as f32,
                    Self::I8 => (b[0] as i8) as f32,
                    Self::U16 => u16::from_le_bytes([b[0], b[1]]) as f32,
                    Self::I16 => i16::from_le_bytes([b[0], b[1]]) as f32,
                    Self::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
                    Self::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
                }
            }
        }

        let mut is_ascii = false;
        let mut is_binary_le = false;
        let mut vertex_count = 0usize;
        let mut in_vertex = false;
        let mut props: Vec<(String, PlyType, usize)> = Vec::new();
        let mut stride = 0usize;

        for line in header_str.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            match parts[0] {
                "format" if parts.len() >= 2 => {
                    if parts[1] == "binary_little_endian" {
                        is_binary_le = true;
                    } else if parts[1] == "ascii" {
                        is_ascii = true;
                    } else {
                        return Err(format!("unsupported PLY format: {}", parts[1]));
                    }
                }
                "element" if parts.len() >= 3 => {
                    in_vertex = parts[1] == "vertex";
                    if in_vertex {
                        vertex_count = parts[2].parse::<usize>().map_err(|_| "bad PLY vertex count")?;
                    }
                }
                "property" if in_vertex && parts.len() >= 3 && parts[1] != "list" => {
                    let ty = match parts[1] {
                        "float" | "float32" => PlyType::F32,
                        "double" | "float64" => PlyType::F64,
                        "uchar" | "uint8" => PlyType::U8,
                        "char" | "int8" => PlyType::I8,
                        "ushort" | "uint16" => PlyType::U16,
                        "short" | "int16" => PlyType::I16,
                        "uint" | "uint32" => PlyType::U32,
                        "int" | "int32" => PlyType::I32,
                        other => return Err(format!("unsupported PLY property type: {other}")),
                    };
                    props.push((parts[2].to_string(), ty, stride));
                    stride += ty.size();
                }
                _ => {}
            }
        }
        if !is_binary_le && !is_ascii {
            return Err("PLY header missing format line".into());
        }
        if vertex_count == 0 {
            return Ok(Self::default());
        }
        let max_cap = vertex_count.min(2_000_000);
        let prop_idx = |name: &str| props.iter().position(|(n, _, _)| n == name);
        let ix = prop_idx("x").ok_or("PLY missing x property")?;
        let iy = prop_idx("y").ok_or("PLY missing y property")?;
        let iz = prop_idx("z").ok_or("PLY missing z property")?;
        let idc = [prop_idx("f_dc_0"), prop_idx("f_dc_1"), prop_idx("f_dc_2")];
        let irgb = [prop_idx("red"), prop_idx("green"), prop_idx("blue")];
        let iop = prop_idx("opacity");
        let isc = [prop_idx("scale_0"), prop_idx("scale_1"), prop_idx("scale_2")];
        let irot = [prop_idx("rot_0"), prop_idx("rot_1"), prop_idx("rot_2"), prop_idx("rot_3")];

        let mut gaussians: Vec<Gaussian>;
        let body = &bytes[body_start..];

        if is_binary_le {
            if body.len() < vertex_count.saturating_mul(stride) {
                return Err("PLY binary body truncated".into());
            }
            // Fast path: standard all-f32 3DGS PLY (avoids per-property enum dispatch inside inner loop).
            let fast_f32 = matches!(props[ix].1, PlyType::F32)
                && matches!(props[iy].1, PlyType::F32)
                && matches!(props[iz].1, PlyType::F32)
                && idc.iter().all(|o| o.is_some_and(|i| matches!(props[i].1, PlyType::F32)))
                && iop.is_some_and(|i| matches!(props[i].1, PlyType::F32))
                && isc.iter().all(|o| o.is_some_and(|i| matches!(props[i].1, PlyType::F32)))
                && irot.iter().all(|o| o.is_some_and(|i| matches!(props[i].1, PlyType::F32)));

            if fast_f32 {
                let ox = props[ix].2;
                let oy = props[iy].2;
                let oz = props[iz].2;
                let odc = [props[idc[0].unwrap()].2, props[idc[1].unwrap()].2, props[idc[2].unwrap()].2];
                let oop = props[iop.unwrap()].2;
                let osc = [props[isc[0].unwrap()].2, props[isc[1].unwrap()].2, props[isc[2].unwrap()].2];
                let orot = [
                    props[irot[0].unwrap()].2,
                    props[irot[1].unwrap()].2,
                    props[irot[2].unwrap()].2,
                    props[irot[3].unwrap()].2,
                ];
                // One bounds check + one unaligned 32-bit load per property (indexing 4 bytes
                // separately costs 4 checks and 4 byte loads that LLVM cannot merge).
                #[inline(always)]
                fn rf(row: &[u8], off: usize) -> f32 {
                    f32::from_le_bytes(row[off..off + 4].try_into().unwrap())
                }
                // N contiguous f32s behind a single bounds check (inner indexing is proven in-range).
                #[inline(always)]
                fn rfn<const N: usize>(row: &[u8], off: usize) -> [f32; N] {
                    let b = &row[off..off + 4 * N];
                    let mut o = [0.0f32; N];
                    for (k, v) in o.iter_mut().enumerate() {
                        *v = f32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap());
                    }
                    o
                }
                struct Offs {
                    pos: [usize; 3],
                    dc: [usize; 3],
                    op: usize,
                    sc: [usize; 3],
                    rot: [usize; 4],
                }
                /// `CONTIG`: every group (xyz, f_dc_*, scale_*, rot_*) is laid out consecutively,
                /// as in every standard 3DGS exporter -> 5 bounds checks per row instead of 14.
                #[inline(always)]
                fn decode_row<const CONTIG: bool>(row: &[u8], o: &Offs) -> Gaussian {
                    let p = if CONTIG { rfn::<3>(row, o.pos[0]) } else { [rf(row, o.pos[0]), rf(row, o.pos[1]), rf(row, o.pos[2])] };
                    let dc = if CONTIG { rfn::<3>(row, o.dc[0]) } else { [rf(row, o.dc[0]), rf(row, o.dc[1]), rf(row, o.dc[2])] };
                    let sc = if CONTIG { rfn::<3>(row, o.sc[0]) } else { [rf(row, o.sc[0]), rf(row, o.sc[1]), rf(row, o.sc[2])] };
                    let r = if CONTIG {
                        rfn::<4>(row, o.rot[0])
                    } else {
                        [rf(row, o.rot[0]), rf(row, o.rot[1]), rf(row, o.rot[2]), rf(row, o.rot[3])]
                    };
                    let color = [
                        (0.5 + SH_C0 * dc[0]).clamp(0.0, 1.0),
                        (0.5 + SH_C0 * dc[1]).clamp(0.0, 1.0),
                        (0.5 + SH_C0 * dc[2]).clamp(0.0, 1.0),
                    ];
                    // 3 log-scales + negated opacity logit as ONE explicit f32x4 fast_exp chain
                    // (NEON on aarch64; LLVM's SLP vectoriser left the array form scalar).
                    let e = simd4::exp(simd4::F4::new([sc[0], sc[1], sc[2], -rf(row, o.op)])).to_array();
                    let opacity = 1.0 / (1.0 + e[3]);
                    let scale = [e[0].clamp(1e-4, 2.0), e[1].clamp(1e-4, 2.0), e[2].clamp(1e-4, 2.0)];
                    let l2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3];
                    let rot = if l2 > 1e-8 {
                        let inv = l2.sqrt().recip();
                        [r[0] * inv, r[1] * inv, r[2] * inv, r[3] * inv]
                    } else {
                        [1.0, 0.0, 0.0, 0.0]
                    };
                    Gaussian { pos: p, scale, rot, color, opacity }
                }
                let o = Offs { pos: [ox, oy, oz], dc: odc, op: oop, sc: osc, rot: orot };
                let contiguous = oy == ox + 4
                    && oz == ox + 8
                    && odc[1] == odc[0] + 4
                    && odc[2] == odc[0] + 8
                    && osc[1] == osc[0] + 4
                    && osc[2] == osc[0] + 8
                    && orot[1] == orot[0] + 4
                    && orot[2] == orot[0] + 8
                    && orot[3] == orot[0] + 12;
                // Exact-size chunked decode directly into uninitialized buffer, tracking finiteness
                // inline so we skip a second full-cloud scan when all positions are finite.
                let body = &body[..max_cap * stride];
                let decode_slice = |src: &[u8], dst: &mut [Gaussian]| -> bool {
                    let mut all_ok = true;
                    if contiguous {
                        for (row, slot) in src.chunks_exact(stride).zip(dst.iter_mut()) {
                            let g = decode_row::<true>(row, &o);
                            all_ok &= (g.pos[0] + g.pos[1] + g.pos[2]).is_finite();
                            *slot = g;
                        }
                    } else {
                        for (row, slot) in src.chunks_exact(stride).zip(dst.iter_mut()) {
                            let g = decode_row::<false>(row, &o);
                            all_ok &= (g.pos[0] + g.pos[1] + g.pos[2]).is_finite();
                            *slot = g;
                        }
                    }
                    all_ok
                };
                // SAFETY: `decode_slice` writes every `Gaussian` in `gaussians[..max_cap]`.
                gaussians = unsafe { uninit_vec::<Gaussian>(max_cap) };
                let all_ok;
                #[cfg(feature = "threads")]
                if max_cap >= PAR_MIN {
                    use rayon::prelude::*;
                    const CH: usize = 8192;
                    all_ok = body
                        .par_chunks(CH * stride)
                        .zip(gaussians.par_chunks_mut(CH))
                        .map(|(s, d)| decode_slice(s, d))
                        .reduce(|| true, |a, b| a & b);
                } else {
                    all_ok = decode_slice(body, &mut gaussians);
                }
                #[cfg(not(feature = "threads"))]
                {
                    all_ok = decode_slice(body, &mut gaussians);
                }
                if !all_ok {
                    gaussians.retain(|g| (g.pos[0] + g.pos[1] + g.pos[2]).is_finite());
                }
            } else {
                gaussians = Vec::with_capacity(max_cap);
                for v in 0..max_cap {
                    let row = &body[v * stride..(v + 1) * stride];
                    let get = |idx: usize| {
                        let (_, ty, off) = props[idx];
                        ty.read_le(&row[off..off + ty.size()])
                    };
                    let pos = [get(ix), get(iy), get(iz)];
                    let color = if let (Some(a), Some(b), Some(c)) = (idc[0], idc[1], idc[2]) {
                        [
                            (0.5 + SH_C0 * get(a)).clamp(0.0, 1.0),
                            (0.5 + SH_C0 * get(b)).clamp(0.0, 1.0),
                            (0.5 + SH_C0 * get(c)).clamp(0.0, 1.0),
                        ]
                    } else if let (Some(a), Some(b), Some(c)) = (irgb[0], irgb[1], irgb[2]) {
                        let s = if matches!(props[a].1, PlyType::U8) { 1.0 / 255.0 } else { 1.0 };
                        [(get(a) * s).clamp(0.0, 1.0), (get(b) * s).clamp(0.0, 1.0), (get(c) * s).clamp(0.0, 1.0)]
                    } else {
                        [0.75, 0.75, 0.78]
                    };
                    let opacity = iop.map(|i| sigmoid(get(i))).unwrap_or(0.92);
                    let scale = if let (Some(a), Some(b), Some(c)) = (isc[0], isc[1], isc[2]) {
                        [fast_exp(get(a)).clamp(1e-4, 2.0), fast_exp(get(b)).clamp(1e-4, 2.0), fast_exp(get(c)).clamp(1e-4, 2.0)]
                    } else {
                        [0.012, 0.012, 0.012]
                    };
                    let rot = if let (Some(a), Some(b), Some(c), Some(d)) = (irot[0], irot[1], irot[2], irot[3]) {
                        let r = [get(a), get(b), get(c), get(d)];
                        let l = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt();
                        if l > 1e-5 { [r[0] / l, r[1] / l, r[2] / l, r[3] / l] } else { [1.0, 0.0, 0.0, 0.0] }
                    } else {
                        [1.0, 0.0, 0.0, 0.0]
                    };
                    if pos.iter().all(|v| v.is_finite()) {
                        gaussians.push(Gaussian { pos, scale, rot, color, opacity });
                    }
                }
            }
        } else {
            gaussians = Vec::with_capacity(max_cap);
            let text = std::str::from_utf8(body).map_err(|_| "invalid UTF-8 in ASCII PLY body")?;
            let mut vals = vec![0.0f32; props.len()];
            for line in text.lines().take(max_cap) {
                let mut count = 0usize;
                for tok in line.split_whitespace() {
                    if count < vals.len() {
                        vals[count] = tok.parse::<f32>().unwrap_or(0.0);
                        count += 1;
                    }
                }
                if count < props.len() {
                    continue;
                }
                let pos = [vals[ix], vals[iy], vals[iz]];
                let color = if let (Some(a), Some(b), Some(c)) = (idc[0], idc[1], idc[2]) {
                    [
                        (0.5 + SH_C0 * vals[a]).clamp(0.0, 1.0),
                        (0.5 + SH_C0 * vals[b]).clamp(0.0, 1.0),
                        (0.5 + SH_C0 * vals[c]).clamp(0.0, 1.0),
                    ]
                } else if let (Some(a), Some(b), Some(c)) = (irgb[0], irgb[1], irgb[2]) {
                    let s = if matches!(props[a].1, PlyType::U8) { 1.0 / 255.0 } else { 1.0 };
                    [(vals[a] * s).clamp(0.0, 1.0), (vals[b] * s).clamp(0.0, 1.0), (vals[c] * s).clamp(0.0, 1.0)]
                } else {
                    [0.75, 0.75, 0.78]
                };
                let opacity = iop.map(|i| sigmoid(vals[i])).unwrap_or(0.92);
                let scale = if let (Some(a), Some(b), Some(c)) = (isc[0], isc[1], isc[2]) {
                    [fast_exp(vals[a]).clamp(1e-4, 2.0), fast_exp(vals[b]).clamp(1e-4, 2.0), fast_exp(vals[c]).clamp(1e-4, 2.0)]
                } else {
                    [0.012, 0.012, 0.012]
                };
                let rot = if let (Some(a), Some(b), Some(c), Some(d)) = (irot[0], irot[1], irot[2], irot[3]) {
                    let r = [vals[a], vals[b], vals[c], vals[d]];
                    let l = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt();
                    if l > 1e-5 { [r[0] / l, r[1] / l, r[2] / l, r[3] / l] } else { [1.0, 0.0, 0.0, 0.0] }
                } else {
                    [1.0, 0.0, 0.0, 0.0]
                };
                if pos.iter().all(|v| v.is_finite()) {
                    gaussians.push(Gaussian { pos, scale, rot, color, opacity });
                }
            }
        }

        Ok(Self { gaussians })
    }

    pub fn stats_json(&self) -> String {
        let (lo, hi) = self.bounds();
        let center = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
        let size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let avg_op = if self.gaussians.is_empty() {
            0.0
        } else {
            self.gaussians.iter().map(|g| g.opacity).sum::<f32>() / self.gaussians.len() as f32
        };
        json!({
            "count": self.gaussians.len(),
            "boundsMin": lo,
            "boundsMax": hi,
            "center": center,
            "size": size,
            "avgOpacity": avg_op,
            "splatBytes": self.gaussians.len() * 32,
            "dsplatBytes": COMPACT_HEADER_LEN + self.gaussians.len() * V2_STRIDE,
        })
        .to_string()
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Generates a 3D Gaussian Splat cloud from a single RGBA8 photo (+ optional depth map in meters or `[0..1]`).
/// - `is_space == false`: segments the foreground item and inflates oriented front/back/rim 3D Gaussians.
/// - `is_space == true`: projects the full photo onto a 110° walkable spatial room frustum using depth.
pub fn image_to_splat(
    width: u32,
    height: u32,
    rgba: &[u8],
    depth: Option<&[f32]>,
    is_space: bool,
    max_splats: usize,
) -> Result<SplatCloud, String> {
    let budget = max_splats.clamp(256, 120_000);
    let max_dim = if is_space {
        ((budget as f32).sqrt().ceil() as usize).clamp(48, 180)
    } else {
        (((budget as f32 * 0.6).sqrt().ceil()) as usize).clamp(48, 160)
    };
    let work = downsample_image(width, height, rgba, depth, max_dim)?;
    let (w, h) = (work.w, work.h);

    if is_space {
        // Space-to-3D Gaussian Splat: true camera-frustum pinhole projection (64° horizontal FOV,
        // matching wide-angle AR cameras) centered at [0,0,0] looking down -Z so mounting at
        // `ARFrame.camera.transform` overlays 1:1 seamlessly on top of the live environment.
        let total = w * h;
        let stride_f = ((total as f32 / budget as f32).sqrt()).max(1.0);
        let step = stride_f.ceil() as usize;
        let fov_x = 64.0_f32.to_radians();
        let aspect = (w as f32 / h.max(1) as f32).clamp(0.45, 2.4);
        let half_tan_x = (fov_x * 0.5).tan();
        let half_tan_y = half_tan_x / aspect;
        let mut gaussians = Vec::with_capacity(budget.min(total));

        // Detect whether supplied depth is already metric (in meters, e.g. ARKit LiDAR) or normalized [0..1].
        let depth_stats = work.depth.as_ref().map(|d| {
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for &v in d {
                if v.is_finite() && v > 0.0 {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            let is_metric = hi > 1.05 && hi < 25.0;
            (lo, (hi - lo).max(1e-4), is_metric)
        });

        let x_cols: Vec<(usize, f32, f32)> = (0..w)
            .step_by(step)
            .map(|x| {
                let nx = (x as f32 + 0.5) / w as f32 * 2.0 - 1.0;
                let side_pull = nx.abs().powf(1.6) * 0.34;
                (x, nx * half_tan_x, side_pull)
            })
            .collect();
        // Generous 1.18x overlap factor so soft radial Gaussian discs blend continuously without holes.
        let pixel_span_k = (2.0 * half_tan_x / (w as f32 / step as f32)) * 1.18;
        const INV255: f32 = 1.0 / 255.0;

        for y in (0..h).step_by(step) {
            let ny = 1.0 - (y as f32 + 0.5) / h as f32 * 2.0; // +1 top, -1 bottom
            let vy = ny * half_tan_y;
            let floor_pull = if ny < -0.12 { (-ny - 0.12) * 0.92 } else { 0.0 };
            let ceil_pull = if ny > 0.35 { (ny - 0.35) * 0.58 } else { 0.0 };
            let row_base = y * w;
            for &(x, vx, side_pull) in &x_cols {
                let p = work.rgba[row_base + x];
                if p[3] < 24 {
                    continue;
                }
                let r = p[0] as f32 * INV255;
                let g = p[1] as f32 * INV255;
                let b = p[2] as f32 * INV255;
                let lum = 0.299 * r + 0.587 * g + 0.114 * b;

                let z_depth = if let (Some(d), Some((lo, span, is_metric))) = (&work.depth, depth_stats) {
                    let raw = d[row_base + x];
                    if is_metric && raw.is_finite() && raw > 0.05 {
                        raw.clamp(0.45, 5.5)
                    } else {
                        let t = ((raw - lo) / span).clamp(0.0, 1.0);
                        0.95 + t * 2.85
                    }
                } else {
                    let lum_relief = (0.5 - lum) * 0.32;
                    (2.35 - floor_pull - ceil_pull - side_pull + lum_relief).clamp(0.95, 3.6)
                };

                // Exact pinhole unprojection along camera ray (-Z forward).
                let pos = [vx * z_depth, vy * z_depth, -z_depth];
                let s_t = (z_depth * pixel_span_k).clamp(0.012, 0.14);
                let s_n = (s_t * 0.25).max(0.004);
                gaussians.push(Gaussian {
                    pos,
                    scale: [s_t, s_t, s_n],
                    // quat_from_normal normalises internally: pass the raw (reversed) ray.
                    rot: quat_from_normal(Vec3::new(-vx, -vy, 1.0)),
                    color: [r, g, b],
                    opacity: (p[3] as f32 * INV255 * 0.92).clamp(0.12, 0.96),
                });
                if gaussians.len() >= budget {
                    break;
                }
            }
            if gaussians.len() >= budget {
                break;
            }
        }
        return Ok(SplatCloud { gaussians });
    }

    // Item-to-3D Gaussian Splat: segmented foreground, dual-sided smooth dome inflation + true RGB colors.
    let (mask, _dom) = segment_foreground(&work);
    let sd2d = signed_distance_2d(&mask, w, h);
    let max_in = sd2d.iter().fold(0.0f32, |m, &d| m.max(-d)).max(1.0);

    let (mut min_x, mut min_y) = (w, h);
    let (mut max_x, mut max_y) = (0usize, 0usize);
    let mut fg_pixels = 0usize;
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                fg_pixels += 1;
            }
        }
    }
    if fg_pixels == 0 {
        return Err("no foreground pixels detected for item splat".into());
    }

    let crop_w = (max_x - min_x + 1) as f32;
    let crop_h = (max_y - min_y + 1) as f32;
    let longest = crop_w.max(crop_h).max(1.0);
    let target_m = 0.28f32; // ~28 cm tabletop hero item in RealityKit/XR meters
    let m_per_px = target_m / longest;
    let cx = (min_x + max_x) as f32 * 0.5;
    let cy = (min_y + max_y) as f32 * 0.5;
    let max_half_z = (max_in * m_per_px * 1.1).clamp(0.018, target_m * 0.34);

    // Front + back surfaces -> ~2 * fg_pixels candidates.
    let step = (((fg_pixels * 2) as f32 / budget as f32).sqrt().ceil() as usize).max(1);
    let s_t = (m_per_px * step as f32 * 0.82).clamp(0.0025, 0.025);
    let s_n = (s_t * 0.42).max(0.0012);

    let mut gaussians = Vec::with_capacity(budget.min(fg_pixels * 2));
    for y in (min_y..=max_y).step_by(step) {
        for x in (min_x..=max_x).step_by(step) {
            let idx = y * w + x;
            if !mask[idx] {
                continue;
            }
            let p = work.rgba[idx];
            let r = p[0] as f32 / 255.0;
            let g = p[1] as f32 / 255.0;
            let b = p[2] as f32 / 255.0;
            let lum = 0.299 * r + 0.587 * g + 0.114 * b;

            let in_d = (-sd2d[idx]).max(0.0) / max_in;
            let relief = if let Some(ref d) = work.depth {
                d[idx].clamp(0.0, 1.0)
            } else {
                0.78 + 0.24 * lum
            };
            let hz = max_half_z * in_d.sqrt() * relief;
            let px = (x as f32 - cx) * m_per_px;
            let py = (cy - y as f32) * m_per_px;

            // Approximate surface normal from 2D distance field gradient.
            let gx = if x + 1 < w && x > 0 { sd2d[y * w + x + 1] - sd2d[y * w + x - 1] } else { 0.0 };
            let gy = if y + 1 < h && y > 0 { sd2d[(y + 1) * w + x] - sd2d[(y - 1) * w + x] } else { 0.0 };
            // quat_from_normal normalises internally (no separate normalize() per face).
            let n_front = Vec3::new(gx * 0.35, -gy * 0.35, 1.0);
            let n_back = Vec3::new(gx * 0.35, -gy * 0.35, -1.0);

            gaussians.push(Gaussian {
                pos: [px, py, hz],
                scale: [s_t, s_t, s_n],
                rot: quat_from_normal(n_front),
                color: [r, g, b],
                opacity: 0.94,
            });
            if hz > s_n * 0.8 && gaussians.len() < budget {
                // Slightly softer ambient back face so orbiting the item feels solid from 360 degrees.
                gaussians.push(Gaussian {
                    pos: [px, py, -hz],
                    scale: [s_t, s_t, s_n],
                    rot: quat_from_normal(n_back),
                    color: [r * 0.88, g * 0.88, b * 0.88],
                    opacity: 0.90,
                });
            }
            if gaussians.len() >= budget {
                break;
            }
        }
        if gaussians.len() >= budget {
            break;
        }
    }

    Ok(SplatCloud { gaussians })
}

/// Converts any triangle mesh (a Dream Builder object OR a captured visionOS/ARKit room mesh)
/// into an oriented 3D Gaussian Splat cloud via area-weighted stratified triangle sampling.
///
/// Every per-triangle invariant (orientation quaternion, AO/semantic shading, floor/ceiling
/// height thresholds) is hoisted out of the per-splat loop, and because the stratified area
/// targets `(i + u) / N` are strictly increasing the triangle lookup is a forward walk over the
/// CDF (amortised O(1)) instead of a binary search per splat. The SplitMix64 stream is
/// counter-based (call `k` hashes `seed + k * golden`), so with `threads` and >= `PAR_MIN`
/// splats chunks are sampled in parallel with bit-identical output.
pub fn mesh_to_splat(
    positions: &[f32],
    indices: &[u32],
    vertex_colors: Option<&[f32]>,
    fallback_rgba: [f32; 4],
    is_space: bool,
    max_splats: usize,
) -> Result<SplatCloud, String> {
    if positions.len() < 9 || positions.len() % 3 != 0 {
        return Err("positions must be non-empty xyz triples".into());
    }
    if indices.len() < 3 || indices.len() % 3 != 0 {
        return Err("indices must be non-empty triangle triples".into());
    }
    let vcount = positions.len() / 3;
    let vc = vertex_colors.filter(|vc| vc.len() >= vcount * 3);
    let has_vc = vc.is_some();
    let budget = max_splats.clamp(64, 150_000);

    let pos_at = |i: u32| -> Vec3 {
        let base = (i as usize) * 3;
        Vec3::new(positions[base], positions[base + 1], positions[base + 2])
    };
    let col_at = |vc: &[f32], i: u32| -> [f32; 3] {
        let base = (i as usize) * 3;
        [vc[base].clamp(0.0, 1.0), vc[base + 1].clamp(0.0, 1.0), vc[base + 2].clamp(0.0, 1.0)]
    };

    let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
    for c in positions.chunks_exact(3) {
        if c[1].is_finite() {
            min_y = min_y.min(c[1]);
            max_y = max_y.max(c[1]);
        }
    }
    let y_span = (max_y - min_y).max(0.05);

    /// Per-triangle sampling record. Colour: with vertex colours `c[0..3]` are barycentrically
    /// blended; otherwise the splat takes `c[0]` when `(p.y - thr) * sgn < 0`, else `c[1]`.
    struct TriInfo {
        a: Vec3,
        b: Vec3,
        c: Vec3,
        col: [[f32; 3]; 3],
        thr: f32,
        sgn: f32,
        rot: [f32; 4],
        cdf: f32,
    }

    let light = Vec3::new(0.3, 0.85, 0.4).normalize();
    let shade_rgb = |base: [f32; 3], k: f32| [(base[0] * k).clamp(0.0, 1.0), (base[1] * k).clamp(0.0, 1.0), (base[2] * k).clamp(0.0, 1.0)];
    let mut tris = Vec::with_capacity(indices.len() / 3);
    let mut total_area = 0.0f32;
    for t in indices.chunks_exact(3) {
        if t[0] as usize >= vcount || t[1] as usize >= vcount || t[2] as usize >= vcount {
            continue;
        }
        let (a, b, c) = (pos_at(t[0]), pos_at(t[1]), pos_at(t[2]));
        let cross = (b - a).cross(c - a);
        let len = cross.length();
        let area = 0.5 * len;
        if !(area > 1e-9) || !area.is_finite() {
            continue;
        }
        total_area += area;
        let normal = cross / len;
        let (col, thr, sgn) = if let Some(vc) = vc {
            ([col_at(vc, t[0]), col_at(vc, t[1]), col_at(vc, t[2])], 0.0, 1.0)
        } else if is_space {
            // Semantic architectural shading for raw ARKit/visionOS room scans.
            let ny = normal.y;
            let ao = 0.84 + 0.16 * (0.5 + 0.5 * normal.dot(light));
            const FLOOR: [f32; 3] = [0.54, 0.45, 0.36]; // Warm oak floor
            const CEIL: [f32; 3] = [0.88, 0.89, 0.91]; // Soft plaster ceiling
            const TABLE: [f32; 3] = [0.64, 0.58, 0.50]; // Table / counter / horizontal furniture top
            const WALL: [f32; 3] = [0.80, 0.79, 0.76]; // Warm studio walls
            if ny > 0.72 {
                // floor below 14% of the room height, else furniture top
                ([shade_rgb(FLOOR, ao), shade_rgb(TABLE, ao), [0.0; 3]], min_y + 0.14 * y_span, 1.0)
            } else if ny < -0.65 {
                // ceiling above 78% of the room height, else wall
                ([shade_rgb(CEIL, ao), shade_rgb(WALL, ao), [0.0; 3]], min_y + 0.78 * y_span, -1.0)
            } else {
                let w = shade_rgb(WALL, ao);
                ([w, w, [0.0; 3]], f32::INFINITY, 1.0)
            }
        } else {
            let shade = 0.86 + 0.14 * (0.5 + 0.5 * normal.y);
            let s = shade_rgb([fallback_rgba[0], fallback_rgba[1], fallback_rgba[2]], shade);
            ([s, s, [0.0; 3]], f32::INFINITY, 1.0)
        };
        tris.push(TriInfo { a, b, c, col, thr, sgn, rot: quat_from_normal(normal), cdf: total_area });
    }
    if tris.is_empty() || !(total_area > 0.0) {
        return Err("mesh has zero surface area".into());
    }

    let target_count = budget;
    let spacing = (total_area / target_count as f32).sqrt();
    let s_t = (spacing * 0.88).clamp(0.003, 0.16);
    let s_n = (s_t * 0.28).max(0.001);
    let opacity = fallback_rgba[3].clamp(0.25, 0.98);
    let inv_count = 1.0 / target_count as f32;
    let last = tris.len() - 1;

    /// SplitMix64 call `k` (1-based) of the stream seeded with `golden`, as a [0,1) f32.
    #[inline(always)]
    fn rng(k: u64) -> f32 {
        let mut z = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(k + 1);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32) * (1.0 / (1u32 << 24) as f32)
    }
    let u_area = |i: usize| ((i as f32 + rng(3 * i as u64 + 1)) * inv_count) * total_area;
    // Sample `i`, advancing the CDF cursor `tri_idx` (targets are strictly increasing).
    let sample = |i: usize, tri_idx: &mut usize| -> Gaussian {
        let u = u_area(i);
        let mut ti = *tri_idx;
        while ti < last && tris[ti].cdf < u {
            ti += 1;
        }
        *tri_idx = ti;
        let tri = &tris[ti];
        let k = 3 * i as u64;
        let r1 = rng(k + 2).sqrt();
        let r2 = rng(k + 3);
        let w0 = 1.0 - r1;
        let w1 = r1 * (1.0 - r2);
        let w2 = r1 * r2;
        let p = tri.a * w0 + tri.b * w1 + tri.c * w2;
        let color = if has_vc {
            let [ca, cb, cc] = tri.col;
            [
                ca[0] * w0 + cb[0] * w1 + cc[0] * w2,
                ca[1] * w0 + cb[1] * w1 + cc[1] * w2,
                ca[2] * w0 + cb[2] * w1 + cc[2] * w2,
            ]
        } else if (p.y - tri.thr) * tri.sgn < 0.0 {
            tri.col[0]
        } else {
            tri.col[1]
        };
        Gaussian { pos: p.to_array(), scale: [s_t, s_t, s_n], rot: tri.rot, color, opacity }
    };

    #[cfg(feature = "threads")]
    if target_count >= PAR_MIN {
        use rayon::prelude::*;
        const CH: usize = 8192;
        let starts: Vec<usize> = (0..target_count).step_by(CH).collect();
        let parts: Vec<Vec<Gaussian>> = starts
            .par_iter()
            .map(|&s| {
                let e = (s + CH).min(target_count);
                let u = u_area(s);
                let mut ti = tris.partition_point(|t| t.cdf < u).min(last);
                (s..e).map(|i| sample(i, &mut ti)).collect()
            })
            .collect();
        let mut gaussians = Vec::with_capacity(target_count);
        for p in parts {
            gaussians.extend_from_slice(&p);
        }
        return Ok(SplatCloud { gaussians });
    }
    let mut ti = 0usize;
    let gaussians: Vec<Gaussian> = (0..target_count).map(|i| sample(i, &mut ti)).collect();
    Ok(SplatCloud { gaussians })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpatialRegion {
    All,
    Floor,
    Ceiling,
    Walls,
    Center,
    Background,
    Left,
    Right,
    Front,
    Back,
}

/// A prompt clause tokenised ONCE (split on non-alphanumerics except `-`), so each keyword list
/// is a scan over ~5 short tokens instead of re-splitting the clause for every candidate word
/// (the old per-word re-split ran ~170x per clause).
struct Clause<'a> {
    text: &'a str,
    toks: Vec<&'a str>,
}

impl<'a> Clause<'a> {
    fn new(text: &'a str) -> Self {
        let toks = text.split(|c: char| !c.is_alphanumeric() && c != '-').filter(|t| !t.is_empty()).collect();
        Self { text, toks }
    }

    /// Whole-token match for single words, substring match for multi-word phrases.
    #[inline]
    fn has(&self, words: &[&str]) -> bool {
        words.iter().any(|w| if w.contains(' ') { self.text.contains(*w) } else { self.toks.iter().any(|t| t == w) })
    }
}

fn parse_region(clause: &Clause) -> SpatialRegion {
    if clause.has(&["floor", "ground", "carpet", "bottom", "rug"]) {
        SpatialRegion::Floor
    } else if clause.has(&["ceiling", "roof", "sky", "top"]) {
        SpatialRegion::Ceiling
    } else if clause.has(&["wall", "walls", "sides"]) {
        SpatialRegion::Walls
    } else if clause.has(&["background", "outside", "border", "edges", "surroundings"]) {
        SpatialRegion::Background
    } else if clause.has(&["center", "middle", "subject", "foreground", "object", "item"]) {
        SpatialRegion::Center
    } else if clause.has(&["left"]) {
        SpatialRegion::Left
    } else if clause.has(&["right"]) {
        SpatialRegion::Right
    } else if clause.has(&["front"]) {
        SpatialRegion::Front
    } else if clause.has(&["back", "rear"]) {
        SpatialRegion::Back
    } else {
        SpatialRegion::All
    }
}

/// Cloud-bounds-derived constants shared by every region test and effect of a clause group
/// (computed once per group instead of re-deriving span/center/divisors per Gaussian).
#[derive(Clone, Copy)]
struct EditBounds {
    lo: Vec3,
    center: Vec3,
    span: Vec3,
    inv_span: Vec3,
    span_max: f32,
    /// `1 / (max(span.x, span.z) / 2)` for the horizontal radial (walls) test.
    inv_radial: f32,
    /// `1 / max(|span| / 2, 1e-3)^2` for the squared centre-distance tests (no sqrt).
    inv_half_diag2: f32,
}

impl EditBounds {
    fn new(lo: [f32; 3], hi: [f32; 3]) -> Self {
        let lo = Vec3::from_array(lo);
        let hi = Vec3::from_array(hi);
        let span = (hi - lo).max(Vec3::splat(1e-3));
        let half_diag = (span.length() * 0.5).max(1e-3);
        Self {
            lo,
            center: (lo + hi) * 0.5,
            span,
            inv_span: span.recip(),
            span_max: span.max_element(),
            inv_radial: 1.0 / (span.x.max(span.z) * 0.5),
            inv_half_diag2: 1.0 / (half_diag * half_diag),
        }
    }

    /// Placeholder for groups whose regions/effects never read the bounds (skips a full pass).
    fn unused() -> Self {
        Self::new([0.0; 3], [1.0; 3])
    }
}

/// Soft membership weight of position `p` in `region`. Only the coordinates a region needs are
/// computed; the sqrt is limited to the walls band and centre tests compare squared distances.
#[inline(always)]
fn region_weight(region: SpatialRegion, p: Vec3, b: &EditBounds) -> f32 {
    match region {
        SpatialRegion::All => 1.0,
        SpatialRegion::Floor => {
            let ry = (p.y - b.lo.y) * b.inv_span.y;
            if ry < 0.22 { ((0.22 - ry) * (1.0 / 0.14)).clamp(0.0, 1.0) } else { 0.0 }
        }
        SpatialRegion::Ceiling => {
            let ry = (p.y - b.lo.y) * b.inv_span.y;
            if ry > 0.74 { ((ry - 0.74) * (1.0 / 0.14)).clamp(0.0, 1.0) } else { 0.0 }
        }
        SpatialRegion::Walls => {
            let ry = (p.y - b.lo.y) * b.inv_span.y;
            if !(0.14..=0.86).contains(&ry) {
                return 0.0;
            }
            let (dx, dz) = (p.x - b.center.x, p.z - b.center.z);
            let radial = (dx * dx + dz * dz).sqrt() * b.inv_radial;
            if radial > 0.45 { ((radial - 0.45) * 4.0).clamp(0.0, 1.0) } else { 0.0 }
        }
        SpatialRegion::Background => {
            let d2 = (p - b.center).length_squared() * b.inv_half_diag2;
            let rz = (p.z - b.lo.z) * b.inv_span.z;
            if d2 > 0.58 * 0.58 || rz < 0.22 { 1.0 } else { 0.0 }
        }
        SpatialRegion::Center => {
            let d2 = (p - b.center).length_squared() * b.inv_half_diag2;
            if d2 < 0.62 * 0.62 { 1.0 } else { 0.0 }
        }
        SpatialRegion::Left => if (p.x - b.lo.x) * b.inv_span.x < 0.46 { 1.0 } else { 0.0 },
        SpatialRegion::Right => if (p.x - b.lo.x) * b.inv_span.x > 0.54 { 1.0 } else { 0.0 },
        SpatialRegion::Front => if (p.z - b.lo.z) * b.inv_span.z > 0.52 { 1.0 } else { 0.0 },
        SpatialRegion::Back => if (p.z - b.lo.z) * b.inv_span.z < 0.48 { 1.0 } else { 0.0 },
    }
}

fn parse_named_color(clause: &Clause) -> Option<(&'static str, [f32; 3])> {
    const COLORS: &[(&str, [f32; 3])] = &[
        ("crimson", [0.80, 0.12, 0.18]),
        ("red", [0.88, 0.20, 0.18]),
        ("orange", [0.95, 0.50, 0.12]),
        ("amber", [0.94, 0.68, 0.16]),
        ("yellow", [0.96, 0.85, 0.22]),
        ("gold", [0.88, 0.72, 0.24]),
        ("golden", [0.90, 0.74, 0.28]),
        ("bronze", [0.72, 0.46, 0.24]),
        ("copper", [0.78, 0.44, 0.26]),
        ("emerald", [0.16, 0.72, 0.42]),
        ("green", [0.24, 0.68, 0.30]),
        ("mint", [0.46, 0.86, 0.68]),
        ("teal", [0.14, 0.62, 0.62]),
        ("cyan", [0.16, 0.84, 0.92]),
        ("blue", [0.18, 0.48, 0.90]),
        ("navy", [0.12, 0.22, 0.48]),
        ("indigo", [0.26, 0.20, 0.72]),
        ("purple", [0.58, 0.24, 0.82]),
        ("violet", [0.54, 0.28, 0.86]),
        ("magenta", [0.88, 0.18, 0.68]),
        ("pink", [0.94, 0.48, 0.68]),
        ("white", [0.95, 0.96, 0.97]),
        ("cream", [0.94, 0.90, 0.80]),
        ("silver", [0.80, 0.83, 0.88]),
        ("chrome", [0.84, 0.87, 0.92]),
        ("gray", [0.52, 0.54, 0.56]),
        ("grey", [0.52, 0.54, 0.56]),
        ("black", [0.12, 0.13, 0.15]),
        ("wood", [0.54, 0.36, 0.22]),
        ("oak", [0.62, 0.44, 0.26]),
        ("walnut", [0.36, 0.23, 0.15]),
        ("brown", [0.46, 0.30, 0.18]),
        ("marble", [0.88, 0.87, 0.85]),
        ("clay", [0.76, 0.48, 0.34]),
        ("brick", [0.68, 0.26, 0.20]),
        ("terracotta", [0.78, 0.38, 0.24]),
    ];
    for &(name, rgb) in COLORS {
        if clause.has(&[name]) {
            return Some((name, rgb));
        }
    }
    None
}

#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

// Active-effect bits, in application order (colour grades first, then geometry/style).
const FX_SUNSET: u32 = 1 << 0;
const FX_CYBERPUNK: u32 = 1 << 1;
const FX_NIGHT: u32 = 1 << 2;
const FX_SNOW: u32 = 1 << 3;
const FX_AUTUMN: u32 = 1 << 4;
const FX_UNDERWATER: u32 = 1 << 5;
const FX_PASTEL: u32 = 1 << 6;
const FX_NOIR: u32 = 1 << 7;
const FX_HOLOGRAM: u32 = 1 << 8;
const FX_LAVA: u32 = 1 << 9;
const FX_TINT: u32 = 1 << 10;
const FX_WARMER: u32 = 1 << 11;
const FX_COOLER: u32 = 1 << 12;
const FX_BRIGHTER: u32 = 1 << 13;
const FX_DARKER: u32 = 1 << 14;
const FX_VIBRANT: u32 = 1 << 15;
const FX_MUTED: u32 = 1 << 16;
const FX_INVERT: u32 = 1 << 17;
const FX_SHARPEN: u32 = 1 << 18;
const FX_SOFTER: u32 = 1 << 19;
const FX_VOXEL: u32 = 1 << 20;
const FX_WAVE: u32 = 1 << 21;
const FX_SWIRL: u32 = 1 << 22;
const FX_FLOAT: u32 = 1 << 23;
const FX_EXPLODE: u32 = 1 << 24;
const FX_BIGGER: u32 = 1 << 25;
const FX_SMALLER: u32 = 1 << 26;
const FX_TALLER: u32 = 1 << 27;
const FX_WIDER: u32 = 1 << 28;
const FX_COLOR_MASK: u32 = (FX_INVERT << 1) - 1;
const FX_GEO_MASK: u32 = ((FX_WIDER << 1) - 1) & !FX_COLOR_MASK;

/// Keyword table for every effect except `FX_TINT` (named colours are parsed separately).
const FX_WORDS: &[(u32, &[&str])] = &[
    (FX_SUNSET, &["sunset", "golden hour", "dusk", "sunrise"]),
    (FX_CYBERPUNK, &["cyberpunk", "neon", "synthwave", "vaporwave"]),
    (FX_NIGHT, &["night", "midnight", "moonlight", "nocturnal", "evening"]),
    (FX_SNOW, &["snow", "snowy", "winter", "frost", "icy", "frozen"]),
    (FX_AUTUMN, &["autumn", "fall", "harvest"]),
    (FX_UNDERWATER, &["underwater", "ocean", "aquatic", "deep sea"]),
    (FX_PASTEL, &["pastel", "dreamy", "cotton candy", "soft"]),
    (FX_NOIR, &["noir", "monochrome", "grayscale", "black and white"]),
    (FX_HOLOGRAM, &["hologram", "matrix", "sci-fi", "scifi", "holographic"]),
    (FX_LAVA, &["lava", "fire", "volcanic", "magma", "inferno"]),
    (FX_WARMER, &["warm", "warmer", "cozy"]),
    (FX_COOLER, &["cool", "cooler", "cold", "chilly"]),
    (FX_BRIGHTER, &["bright", "brighter", "lighter", "lit", "glow", "glowing"]),
    (FX_DARKER, &["dark", "darker", "dim", "dimmer", "moody"]),
    (FX_VIBRANT, &["vibrant", "vivid", "saturated", "pop"]),
    (FX_MUTED, &["muted", "desaturate", "faded", "vintage"]),
    (FX_INVERT, &["invert", "negative"]),
    (FX_SHARPEN, &["sharp", "sharpen", "crisp", "crisper", "finer", "detailed"]),
    (FX_SOFTER, &["softer", "fluffy", "painterly", "blur", "blurry", "cloud", "impressionist"]),
    (FX_VOXEL, &["voxel", "pixel", "pixelate", "blocky", "minecraft", "retro"]),
    (FX_WAVE, &["wave", "wavy", "ripple", "ripples"]),
    (FX_SWIRL, &["swirl", "twist", "spiral", "vortex"]),
    (FX_FLOAT, &["float", "levitate", "rise", "hover"]),
    (FX_EXPLODE, &["explode", "scatter", "disperse", "shatter"]),
    (FX_BIGGER, &["bigger", "larger", "expand", "grow", "giant"]),
    (FX_SMALLER, &["smaller", "shrink", "tiny", "mini", "compact"]),
    (FX_TALLER, &["taller", "stretch", "elongate"]),
    (FX_WIDER, &["wider", "broaden"]),
];

/// Clause-invariant state for the edit kernels (computed once per clause).
struct EditCtx {
    region: SpatialRegion,
    b: EditBounds,
    tint: [f32; 3],
}

/// Gaussians per edit block: the per-block weight/luma scratch stays in L1 and the block's
/// Gaussians stay in L1 across all of a clause group's effect loops.
const EDIT_BLOCK: usize = 256;

/// Runs a colour effect over the block's in-region Gaussians: `f(g, w, lum0, rel)`.
/// When `ALL` is true, `w` is constant `1.0` so LLVM folds `w` out of every expression and skips
/// the weight check/load completely.
#[inline(always)]
fn each_color<const ALL: bool, F: Fn(&mut Gaussian, f32, f32, Vec3)>(
    gs: &mut [Gaussian],
    w: &[f32],
    lum: &[f32],
    b: &EditBounds,
    f: F,
) {
    if ALL {
        for (g, &li) in gs.iter_mut().zip(lum) {
            let rel = (Vec3::from_array(g.pos) - b.lo) * b.inv_span;
            f(g, 1.0, li, rel);
        }
    } else {
        for ((g, &wi), &li) in gs.iter_mut().zip(w).zip(lum) {
            if wi > 1e-4 {
                let rel = (Vec3::from_array(g.pos) - b.lo) * b.inv_span;
                f(g, wi, li, rel);
            }
        }
    }
}

/// Runs a geometry/style effect over the block's in-region Gaussians: `f(g, w)`.
#[inline(always)]
fn each_geo<const ALL: bool, F: Fn(&mut Gaussian, f32)>(gs: &mut [Gaussian], w: &[f32], f: F) {
    if ALL {
        for g in gs.iter_mut() {
            f(g, 1.0);
        }
    } else {
        for (g, &wi) in gs.iter_mut().zip(w) {
            if wi > 1e-4 {
                f(g, wi);
            }
        }
    }
}

/// 4-wide NEON/SIMD fast path for global luminance-preserving tint (`FX_TINT` with `SpatialRegion::All`,
/// e.g. `"make it red"`): single pass, zero scratch buffers, zero branches.
#[inline(always)]
fn apply_tint_all_simd(gs: &mut [Gaussian], rgb: [f32; 3]) -> usize {
    use simd4::F4;
    let s = F4::splat;
    let zero = s(0.0);
    let one = s(1.0);
    let mut chunks = gs.chunks_exact_mut(4);
    for g4 in &mut chunks {
        let g4: &mut [Gaussian; 4] = g4.try_into().unwrap();
        let (cr, cg, cb, op) = simd4::load_rgba4(g4);
        let lum = cr.mul_add(s(0.299), cg.mul_add(s(0.587), cb.mul(s(0.114))));
        let shade = lum.mul_add(s(0.85), s(0.38));
        let tr = shade.mul(s(rgb[0])).max(zero).min(one);
        let tg = shade.mul(s(rgb[1])).max(zero).min(one);
        let tb = shade.mul(s(rgb[2])).max(zero).min(one);
        let nr = tr.mul_add(s(0.82), cr.mul(s(0.18)));
        let ng = tg.mul_add(s(0.82), cg.mul(s(0.18)));
        let nb = tb.mul_add(s(0.82), cb.mul(s(0.18)));
        simd4::store_rgba4(g4, nr, ng, nb, op);
    }
    for g in chunks.into_remainder() {
        let c = g.color;
        let lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
        let shade = 0.38 + 0.85 * lum;
        let tinted = [
            (rgb[0] * shade).clamp(0.0, 1.0),
            (rgb[1] * shade).clamp(0.0, 1.0),
            (rgb[2] * shade).clamp(0.0, 1.0),
        ];
        g.color = lerp3(c, tinted, 0.82);
    }
    gs.len()
}

impl EditCtx {
    /// Applies every effect whose bit is set in `fx` to a block of <= `EDIT_BLOCK` Gaussians and
    /// returns how many were in-region (touched).
    fn apply_block(&self, fx: u32, gs: &mut [Gaussian]) -> usize {
        if self.region == SpatialRegion::All {
            self.apply_block_inner::<true>(fx, gs)
        } else {
            self.apply_block_inner::<false>(fx, gs)
        }
    }

    fn apply_block_inner<const ALL: bool>(&self, fx: u32, gs: &mut [Gaussian]) -> usize {
        debug_assert!(gs.len() <= EDIT_BLOCK);
        if fx == 0 {
            return 0;
        }
        let n = gs.len();
        let mut w_buf = [0.0f32; EDIT_BLOCK];
        let hits = if ALL {
            n
        } else {
            let mut h = 0usize;
            for (wi, g) in w_buf.iter_mut().zip(gs.iter()) {
                *wi = region_weight(self.region, Vec3::from_array(g.pos), &self.b);
                h += (*wi > 1e-4) as usize;
            }
            if h == 0 {
                return 0;
            }
            h
        };
        let w = if ALL { &[][..] } else { &w_buf[..n] };

        let mut cm = fx & FX_COLOR_MASK;
        if cm != 0 {
            let mut lum = [0.0f32; EDIT_BLOCK];
            for (l, g) in lum.iter_mut().zip(gs.iter()) {
                let c = g.color;
                *l = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
            }
            let lum = &lum[..n];
            let b = &self.b;
            while cm != 0 {
                let bit = cm & cm.wrapping_neg();
                cm ^= bit;
                match bit {
                    FX_SUNSET => each_color::<ALL, _>(gs, w, lum, b, |g, w, lum, rel| {
                        let c = g.color;
                        let sky = lerp3([0.98, 0.42, 0.16], [0.96, 0.24, 0.46], rel.y.clamp(0.0, 1.0));
                        let target = [
                            (c[0] * 0.55 + sky[0] * (0.35 + 0.45 * lum)).clamp(0.0, 1.0),
                            (c[1] * 0.45 + sky[1] * (0.30 + 0.35 * lum)).clamp(0.0, 1.0),
                            (c[2] * 0.35 + sky[2] * (0.25 + 0.25 * lum)).clamp(0.0, 1.0),
                        ];
                        g.color = lerp3(c, target, 0.85 * w);
                    }),
                    FX_CYBERPUNK => each_color::<ALL, _>(gs, w, lum, b, |g, w, lum, rel| {
                        let c = g.color;
                        let side = rel.x.clamp(0.0, 1.0);
                        let neon = lerp3([0.08, 0.92, 0.98], [0.96, 0.14, 0.78], side);
                        let dark_base = [c[0] * 0.18 + 0.04, c[1] * 0.14 + 0.04, c[2] * 0.28 + 0.12];
                        let rim = fast_pow((lum * 1.45).clamp(0.0, 1.0), 1.3);
                        let target = lerp3(dark_base, neon, (0.25 + 0.75 * rim).clamp(0.0, 1.0));
                        g.color = lerp3(c, target, 0.90 * w);
                    }),
                    FX_NIGHT => each_color::<ALL, _>(gs, w, lum, b, |g, w, _, rel| {
                        let c = g.color;
                        let target = [
                            (c[0] * 0.28 + 0.02).clamp(0.0, 1.0),
                            (c[1] * 0.36 + 0.04).clamp(0.0, 1.0),
                            (c[2] * 0.58 + 0.10 + rel.y * 0.08).clamp(0.0, 1.0),
                        ];
                        g.color = lerp3(c, target, 0.85 * w);
                    }),
                    FX_SNOW => each_color::<ALL, _>(gs, w, lum, b, |g, w, _, rel| {
                        let c = g.color;
                        // y of the rotated +Z axis (rotation-matrix element m21) in closed form.
                        let [qw, qx, qy, qz] = g.rot;
                        let up = (2.0 * (qy * qz - qw * qx)).max(0.0);
                        let snow_amt = (up * 0.75 + rel.y * 0.30).clamp(0.0, 0.92);
                        let cool = [c[0] * 0.72 + 0.10, c[1] * 0.82 + 0.12, (c[2] * 0.92 + 0.18).min(1.0)];
                        let target = lerp3(cool, [0.94, 0.97, 1.0], snow_amt);
                        g.color = lerp3(c, target, 0.88 * w);
                    }),
                    FX_AUTUMN => each_color::<ALL, _>(gs, w, lum, b, |g, w, lum, rel| {
                        let target = [
                            (0.35 + lum * 0.65).clamp(0.0, 1.0),
                            (0.16 + lum * 0.44 * (1.0 - rel.x * 0.3)).clamp(0.0, 1.0),
                            (0.06 + lum * 0.18).clamp(0.0, 1.0),
                        ];
                        g.color = lerp3(g.color, target, 0.78 * w);
                    }),
                    FX_UNDERWATER => each_color::<ALL, _>(gs, w, lum, b, |g, w, _, rel| {
                        let c = g.color;
                        let depth_grad = lerp3([0.04, 0.24, 0.42], [0.16, 0.78, 0.82], rel.y.clamp(0.0, 1.0));
                        let target = lerp3([c[0] * 0.3, c[1] * 0.65, c[2] * 0.85], depth_grad, 0.62);
                        g.color = lerp3(c, target, 0.85 * w);
                    }),
                    FX_PASTEL => each_geo::<ALL, _>(gs, w, |g, w| {
                        let c = g.color;
                        let target = [
                            (c[0] * 0.55 + 0.42).clamp(0.0, 1.0),
                            (c[1] * 0.55 + 0.40).clamp(0.0, 1.0),
                            (c[2] * 0.55 + 0.44).clamp(0.0, 1.0),
                        ];
                        g.color = lerp3(c, target, 0.80 * w);
                    }),
                    FX_NOIR => each_color::<ALL, _>(gs, w, lum, b, |g, w, lum, _| {
                        let contrast = ((lum - 0.5) * 1.35 + 0.5).clamp(0.0, 1.0);
                        g.color = lerp3(g.color, [contrast, contrast, contrast], 0.95 * w);
                    }),
                    FX_HOLOGRAM => each_color::<ALL, _>(gs, w, lum, b, |g, w, lum, _| {
                        let scan = 0.65 + 0.35 * ((g.pos[1] * 45.0).sin() * 0.5 + 0.5);
                        let holo = [0.06 * scan, (0.78 + 0.22 * lum) * scan, (0.85 + 0.15 * lum) * scan];
                        g.color = lerp3(g.color, holo, 0.90 * w);
                        g.opacity = (g.opacity * (1.0 - 0.25 * w)).clamp(0.18, 0.95);
                    }),
                    FX_LAVA => each_color::<ALL, _>(gs, w, lum, b, |g, w, lum, _| {
                        let hot = lerp3([0.14, 0.04, 0.03], [1.0, 0.52, 0.08], (lum * 1.35).clamp(0.0, 1.0));
                        g.color = lerp3(g.color, hot, 0.88 * w);
                    }),
                    FX_TINT => {
                        let rgb = self.tint;
                        each_color::<ALL, _>(gs, w, lum, b, |g, w, lum, _| {
                            // Luminance-preserving tint so shading and texture remain crisp.
                            let shade = 0.38 + 0.85 * lum;
                            let tinted = [(rgb[0] * shade).clamp(0.0, 1.0), (rgb[1] * shade).clamp(0.0, 1.0), (rgb[2] * shade).clamp(0.0, 1.0)];
                            g.color = lerp3(g.color, tinted, 0.82 * w);
                        })
                    }
                    FX_WARMER => each_geo::<ALL, _>(gs, w, |g, w| {
                        let c = g.color;
                        g.color = lerp3(c, [(c[0] * 1.16 + 0.04).min(1.0), (c[1] * 1.04 + 0.01).min(1.0), (c[2] * 0.86).min(1.0)], w);
                    }),
                    FX_COOLER => each_geo::<ALL, _>(gs, w, |g, w| {
                        let c = g.color;
                        g.color = lerp3(c, [(c[0] * 0.86).min(1.0), (c[1] * 1.02).min(1.0), (c[2] * 1.18 + 0.05).min(1.0)], w);
                    }),
                    FX_BRIGHTER => each_geo::<ALL, _>(gs, w, |g, w| {
                        let c = g.color;
                        g.color = lerp3(c, [(c[0] * 1.25 + 0.06).min(1.0), (c[1] * 1.25 + 0.06).min(1.0), (c[2] * 1.25 + 0.06).min(1.0)], w);
                    }),
                    FX_DARKER => each_geo::<ALL, _>(gs, w, |g, w| {
                        let c = g.color;
                        g.color = lerp3(c, [c[0] * 0.68, c[1] * 0.68, c[2] * 0.68], w);
                    }),
                    FX_VIBRANT => each_geo::<ALL, _>(gs, w, |g, w| {
                        let c = g.color;
                        let l = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
                        g.color = lerp3(c, [(l + (c[0] - l) * 1.45).clamp(0.0, 1.0), (l + (c[1] - l) * 1.45).clamp(0.0, 1.0), (l + (c[2] - l) * 1.45).clamp(0.0, 1.0)], w);
                    }),
                    FX_MUTED => each_geo::<ALL, _>(gs, w, |g, w| {
                        let c = g.color;
                        let l = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
                        g.color = lerp3(c, [l + (c[0] - l) * 0.45, l + (c[1] - l) * 0.45, l + (c[2] - l) * 0.45], w);
                    }),
                    FX_INVERT => each_geo::<ALL, _>(gs, w, |g, w| {
                        let c = g.color;
                        g.color = lerp3(c, [1.0 - c[0], 1.0 - c[1], 1.0 - c[2]], w);
                    }),
                    _ => {}
                }
            }
        }

        let mut gm = fx & FX_GEO_MASK;
        let center = self.b.center;
        let span = self.b.span;
        let span_max = self.b.span_max;
        while gm != 0 {
            let bit = gm & gm.wrapping_neg();
            gm ^= bit;
            match bit {
                FX_SHARPEN => each_geo::<ALL, _>(gs, w, |g, w| {
                    let k = 1.0 - 0.28 * w;
                    g.scale = [(g.scale[0] * k).max(1e-4), (g.scale[1] * k).max(1e-4), (g.scale[2] * k).max(1e-4)];
                    g.opacity = (g.opacity + 0.08 * w).min(1.0);
                }),
                FX_SOFTER => each_geo::<ALL, _>(gs, w, |g, w| {
                    let k = 1.0 + 0.42 * w;
                    g.scale = [g.scale[0] * k, g.scale[1] * k, g.scale[2] * k];
                    g.opacity = (g.opacity * (1.0 - 0.15 * w)).max(0.12);
                }),
                FX_VOXEL => {
                    let step = (span_max / 28.0).clamp(0.008, 0.12);
                    let s = step * 0.58;
                    each_geo::<ALL, _>(gs, w, |g, w| {
                        let qx = (g.pos[0] / step).round() * step;
                        let qy = (g.pos[1] / step).round() * step;
                        let qz = (g.pos[2] / step).round() * step;
                        g.pos = [g.pos[0] + (qx - g.pos[0]) * w, g.pos[1] + (qy - g.pos[1]) * w, g.pos[2] + (qz - g.pos[2]) * w];
                        g.scale = [s, s, s];
                        g.rot = [1.0, 0.0, 0.0, 0.0];
                    })
                }
                FX_WAVE => each_geo::<ALL, _>(gs, w, |g, w| {
                    let amp = span_max * 0.045 * w;
                    g.pos[1] += (g.pos[0] * 14.0 + g.pos[2] * 10.0).sin() * amp;
                }),
                FX_SWIRL => each_geo::<ALL, _>(gs, w, |g, w| {
                    let rel_p = Vec3::from_array(g.pos) - center;
                    let angle = (rel_p.y / span.y.max(0.05)) * 1.15 * w;
                    let (s, cs) = angle.sin_cos();
                    g.pos[0] = center.x + rel_p.x * cs - rel_p.z * s;
                    g.pos[2] = center.z + rel_p.x * s + rel_p.z * cs;
                }),
                FX_FLOAT => each_geo::<ALL, _>(gs, w, |g, w| {
                    g.pos[1] += span.y * 0.18 * w;
                }),
                FX_EXPLODE => each_geo::<ALL, _>(gs, w, |g, w| {
                    let dir = (Vec3::from_array(g.pos) - center).normalize_or_zero();
                    let push = span_max * 0.22 * w;
                    g.pos[0] += dir.x * push;
                    g.pos[1] += dir.y * push;
                    g.pos[2] += dir.z * push;
                }),
                FX_BIGGER => each_geo::<ALL, _>(gs, w, |g, w| {
                    let k = 1.0 + 0.25 * w;
                    g.pos = ((Vec3::from_array(g.pos) - center) * k + center).to_array();
                    g.scale = [g.scale[0] * k, g.scale[1] * k, g.scale[2] * k];
                }),
                FX_SMALLER => each_geo::<ALL, _>(gs, w, |g, w| {
                    let k = 1.0 - 0.22 * w;
                    g.pos = ((Vec3::from_array(g.pos) - center) * k + center).to_array();
                    g.scale = [(g.scale[0] * k).max(1e-4), (g.scale[1] * k).max(1e-4), (g.scale[2] * k).max(1e-4)];
                }),
                FX_TALLER => each_geo::<ALL, _>(gs, w, |g, w| {
                    g.pos[1] = center.y + (g.pos[1] - center.y) * (1.0 + 0.25 * w);
                    g.scale[1] *= 1.0 + 0.18 * w;
                }),
                FX_WIDER => each_geo::<ALL, _>(gs, w, |g, w| {
                    g.pos[0] = center.x + (g.pos[0] - center.x) * (1.0 + 0.25 * w);
                    g.scale[0] *= 1.0 + 0.18 * w;
                }),
                _ => {}
            }
        }
        hits
    }
}

/// Effects whose kernels read the cloud bounds (`rel`, `center`, `span`): only groups containing
/// one of these (or a spatial region) pay for a `bounds()` pass.
const FX_NEEDS_BOUNDS: u32 = FX_SUNSET | FX_CYBERPUNK | FX_NIGHT | FX_SNOW | FX_AUTUMN | FX_UNDERWATER | FX_MOVES;
/// Effects that move Gaussians (invalidate cached bounds and end a fused clause group).
const FX_MOVES: u32 = FX_GEO_MASK & !(FX_SHARPEN | FX_SOFTER);
/// Max clauses fused into one pass over the cloud.
const MAX_GROUP: usize = 8;

/// Applies a group of clauses (all sharing the same bounds) in ONE pass over the cloud: each
/// L1-sized block runs every clause's effects in prompt order, which is exactly equivalent to one
/// pass per clause because Gaussians are independent and no clause but the last moves splats.
/// Rayon-parallel >= `PAR_MIN`. Returns per-clause hit counts.
fn run_edit_group(gs: &mut [Gaussian], group: &[(EditCtx, u32)]) -> [usize; MAX_GROUP] {
    if group.len() == 1 && group[0].0.region == SpatialRegion::All && group[0].1 == FX_TINT {
        let rgb = group[0].0.tint;
        let mut out = [0usize; MAX_GROUP];
        #[cfg(feature = "threads")]
        if gs.len() >= PAR_MIN {
            use rayon::prelude::*;
            out[0] = gs.par_chunks_mut(EDIT_BLOCK * 16).map(|c| apply_tint_all_simd(c, rgb)).sum();
            return out;
        }
        out[0] = apply_tint_all_simd(gs, rgb);
        return out;
    }
    let run = |chunk: &mut [Gaussian]| {
        let mut h = [0usize; MAX_GROUP];
        for blk in chunk.chunks_mut(EDIT_BLOCK) {
            for (hk, (ctx, fx)) in h.iter_mut().zip(group) {
                *hk += ctx.apply_block(*fx, blk);
            }
        }
        h
    };
    #[cfg(feature = "threads")]
    if gs.len() >= PAR_MIN {
        use rayon::prelude::*;
        return gs.par_chunks_mut(EDIT_BLOCK * 16).map(run).reduce(
            || [0usize; MAX_GROUP],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );
    }
    run(gs)
}

/// Applies natural-language prompt edits directly to a 3D Gaussian Splat cloud (item or full space).
pub fn edit_splat_by_prompt(cloud: &SplatCloud, prompt: &str) -> SplatEditOutput {
    let mut out = cloud.clone();
    let (summary, modified_count) = edit_splat_in_place(&mut out, prompt);
    let total_count = out.gaussians.len();
    SplatEditOutput { cloud: out, summary, modified_count, total_count }
}

/// A parsed colour/geometry clause waiting to be fused into the next group pass.
struct PendingFx<'a> {
    region: SpatialRegion,
    fx: u32,
    tint: [f32; 3],
    text: &'a str,
}

/// Bounds of the cloud, reused across clauses until a clause moves or removes splats.
fn cached_bounds(gs: &SplatCloud, cache: &mut Option<EditBounds>) -> EditBounds {
    *cache.get_or_insert_with(|| {
        let (lo, hi) = gs.bounds();
        EditBounds::new(lo, hi)
    })
}

/// Executes the pending clause group in one fused pass and records its ops in prompt order.
fn flush_edit_group(
    out: &mut SplatCloud,
    pending: &mut Vec<PendingFx<'_>>,
    cache: &mut Option<EditBounds>,
    modified_total: &mut usize,
    applied_ops: &mut Vec<String>,
) {
    if pending.is_empty() {
        return;
    }
    let needs_bounds = pending.iter().any(|p| p.region != SpatialRegion::All || p.fx & FX_NEEDS_BOUNDS != 0);
    let b = if needs_bounds { cached_bounds(out, cache) } else { EditBounds::unused() };
    let group: Vec<(EditCtx, u32)> = pending.iter().map(|p| (EditCtx { region: p.region, b, tint: p.tint }, p.fx)).collect();
    let hits = run_edit_group(&mut out.gaussians, &group);
    for (p, &h) in pending.iter().zip(&hits) {
        if h > 0 {
            *modified_total = (*modified_total).max(h);
            applied_ops.push(p.text.to_string());
        }
    }
    if pending.iter().any(|p| p.fx & FX_MOVES != 0) {
        *cache = None;
    }
    pending.clear();
}

/// Zero-clone in-place prompt editor for FFI/WASM hosts.
///
/// Each clause is tokenised once; consecutive colour/style clauses are fused into a single pass
/// over the cloud (a clause that moves splats closes its group), and the cloud bounds are computed
/// at most once per group, only when a region or bounds-dependent effect needs them, and reused
/// until splats move or are removed.
pub fn edit_splat_in_place(out: &mut SplatCloud, prompt: &str) -> (String, usize) {
    let trimmed = prompt.trim();
    if trimmed.is_empty() || out.gaussians.is_empty() {
        return ("No changes applied.".into(), 0);
    }
    let lower_owned;
    let lower: &str = if trimmed.bytes().any(|b| b.is_ascii_uppercase()) {
        lower_owned = trimmed.to_ascii_lowercase();
        &lower_owned
    } else {
        trimmed
    };

    // Split compound prompts ("make the floor wood and the walls cyberpunk").
    let mut clauses: Vec<&str> = Vec::new();
    for part in lower.split(" and ") {
        for sub in part.split(',') {
            let t = sub.trim();
            if !t.is_empty() {
                clauses.push(t);
            }
        }
    }
    if clauses.is_empty() {
        clauses.push(&lower);
    }

    let mut applied_ops: Vec<String> = Vec::new();
    let mut modified_total = 0usize;
    let mut cache: Option<EditBounds> = None;
    let mut pending: Vec<PendingFx> = Vec::with_capacity(MAX_GROUP);

    for &text in &clauses {
        let clause = Clause::new(text);
        let region = parse_region(&clause);

        // 1. Pruning / removal operations
        if clause.has(&["remove", "delete", "clear", "erase", "hide", "crop", "isolate"]) {
            let target_region = if region == SpatialRegion::All && clause.has(&["isolate", "crop"]) {
                SpatialRegion::Background
            } else {
                region
            };
            if target_region != SpatialRegion::All {
                flush_edit_group(out, &mut pending, &mut cache, &mut modified_total, &mut applied_ops);
                let b = cached_bounds(out, &mut cache);
                let before = out.gaussians.len();
                out.gaussians.retain(|g| region_weight(target_region, Vec3::from_array(g.pos), &b) < 0.5);
                let removed = before.saturating_sub(out.gaussians.len());
                if removed > 0 {
                    cache = None;
                    modified_total += removed;
                    applied_ops.push(format!("removed {removed} splats from {target_region:?}"));
                }
                continue;
            }
        }

        if clause.has(&["denoise", "clean", "prune", "tidy"]) {
            flush_edit_group(out, &mut pending, &mut cache, &mut modified_total, &mut applied_ops);
            let before = out.gaussians.len();
            out.gaussians.retain(|g| g.opacity >= 0.15);
            let removed = before.saturating_sub(out.gaussians.len());
            if removed > 0 {
                cache = None;
            }
            modified_total += removed;
            applied_ops.push(format!("denoised ({removed} faint splats pruned)"));
        }

        // 2. Hoist every keyword test into ONE active-effects bitmask per clause. The per-Gaussian
        //    kernel then visits only set bits (lowest first == original application order) instead of
        //    branching on all 29 flags, and does no colour/geometry work when no such bit is set.
        let mut fx = 0u32;
        for &(bit, words) in FX_WORDS {
            if clause.has(words) {
                fx |= bit;
            }
        }
        // Only apply a named color if it isn't purely consumed by "golden hour" or "black and white".
        let named_color = if fx & FX_NOIR != 0 || text.contains("golden hour") { None } else { parse_named_color(&clause) };
        let mut tint = [0.0f32; 3];
        if let Some((_, rgb)) = named_color {
            fx |= FX_TINT;
            tint = rgb;
        }
        if fx != 0 {
            pending.push(PendingFx { region, fx, tint, text });
            if fx & FX_MOVES != 0 || pending.len() == MAX_GROUP {
                flush_edit_group(out, &mut pending, &mut cache, &mut modified_total, &mut applied_ops);
            }
        }
    }
    flush_edit_group(out, &mut pending, &mut cache, &mut modified_total, &mut applied_ops);

    let summary = if applied_ops.is_empty() {
        format!("Tried \"{prompt}\" — try a style like cyberpunk, sunset, snowy, gold, sharpen, voxel, or \"make the floor wood\".")
    } else {
        format!("Applied {} across {} splats.", applied_ops.join(" + "), out.gaussians.len())
    };
    (summary, modified_total)
}

const SORT_BUCKETS: usize = 1024;
const SORT_MASK: usize = SORT_BUCKETS - 1;
const SORT_MAX_B: f32 = (SORT_BUCKETS - 1) as f32;
/// Bucket-major 4-lane histogram: `hist[bucket][lane]` (lane = element index & 3).
type Hist4 = [[u32; 4]; SORT_BUCKETS];

/// Pass 1: `depth = dot(pos, fwd) - base` stored as raw f32 bits (the buffer is later recycled
/// as the u32 index output), with 8 independent min/max lanes (no loop-carried dependency, so
/// LLVM keeps two NEON/WASM-SIMD f32x4 accumulators). `if d < lo` (not `f32::min`) maps 1:1 onto
/// `fmin`-free compare+select and silently skips NaN depths.
/// `dot(p, f)`: fused multiply-adds on aarch64 (single-rounding `fmadd`, 2 ops instead of 5);
/// plain mul/add elsewhere (wasm32 has no scalar FMA, `mul_add` would be a libm call).
#[inline(always)]
fn dot3(p: &[f32; 3], f: [f32; 3]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        p[2].mul_add(f[2], p[1].mul_add(f[1], p[0] * f[0]))
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        p[0] * f[0] + p[1] * f[1] + p[2] * f[2]
    }
}

/// Allocates a `Vec<T>` of length `len` without zero-filling; caller must initialise every
/// element before reading.
#[inline(always)]
unsafe fn uninit_vec<T: Copy>(len: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(len);
    // SAFETY: `T: Copy` has no drop glue; caller overwrites all `len` elements before reading.
    unsafe { v.set_len(len) };
    v
}

/// 4x4 `f32` register transpose: rows `a..d` -> columns (8 permutes).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn tr4(
    a: core::arch::aarch64::float32x4_t,
    b: core::arch::aarch64::float32x4_t,
    c: core::arch::aarch64::float32x4_t,
    d: core::arch::aarch64::float32x4_t,
) -> [core::arch::aarch64::float32x4_t; 4] {
    use core::arch::aarch64::*;
    // SAFETY: register-only NEON permutes (NEON is baseline on aarch64).
    unsafe {
        let t0 = vreinterpretq_f64_f32(vtrn1q_f32(a, b));
        let t1 = vreinterpretq_f64_f32(vtrn2q_f32(a, b));
        let t2 = vreinterpretq_f64_f32(vtrn1q_f32(c, d));
        let t3 = vreinterpretq_f64_f32(vtrn2q_f32(c, d));
        [
            vreinterpretq_f32_f64(vzip1q_f64(t0, t2)),
            vreinterpretq_f32_f64(vzip1q_f64(t1, t3)),
            vreinterpretq_f32_f64(vzip2q_f64(t0, t2)),
            vreinterpretq_f32_f64(vzip2q_f64(t1, t3)),
        ]
    }
}

/// `.dsplat` v2 NEON encoder for `gs.len() & !3` splats (returns that count; the caller encodes
/// the tail). ~135 NEON ops per 4 splats (the generic `simd4` kernel needs ~240, and this loop
/// is SIMD-pipe bound, not memory bound):
/// - round-to-nearest saturating `fcvtnu` replaces `+0.5 / max(0) / trunc` everywhere;
/// - fields are packed with shift-left-insert (`sli`), which also discards any out-of-range
///   high bits, so positions / quaternion components need no clamp (`+inf` saturates to all-ones
///   whose low 12 bits are exactly the `4095` clamp);
/// - log2(scale) is `(bits - bias) * 2^-23` (one integer sub + fixed-point convert) plus a
///   cubic mantissa correction `t(1-t)(A + t(B + Ct))` (|err| < 1.2e-4 log2 units, < 0.3% of
///   the finest quantisation step), with the quantiser gain folded into the coefficients;
/// - smallest-3 selection uses absolute compares (`facgt`) on signed values (no `abs`), and
///   degenerate quaternions are patched to identity with one bit-select on the packed word.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn encode_v2_neon<const CLAMP_S: bool>(
    gs: &[Gaussian],
    dst: &mut [u8],
    lo: [f32; 3],
    inv_p: [f32; 3],
    l2_min: f32,
    k: f32,
) -> usize {
    use core::arch::aarch64::*;
    let n4 = (gs.len() / 4).min(dst.len() / (4 * V2_STRIDE));
    // Bias-centered IEEE-754 log2: `(bits(s) - bias) * 2^-23 == e + t + 0.0430 - l2_min`,
    // max error +-0.043 octaves (+-3.0% relative scale, < 1/2 step of 6-bit log-scale),
    // cutting each scale quantisation to 5 NEON ops (down from 12).
    let bias = ((127.0 + l2_min as f64 - 0.043_035_666) * 8_388_608.0).round() as i32;
    // SAFETY: NEON baseline; `Gaussian` is `#[repr(C)]` 56 B = 14 initialised f32s; `n4 * 4 <= gs.len()`
    // and `n4 * 48 <= dst.len()`.
    unsafe {
        // Pack constants into lane vectors to keep live NEON register count <= 11 (zero stack spills).
        let in_v = vld1q_f32([inv_p[0], inv_p[1], inv_p[2], 63.0].as_ptr());
        let off_x = vdupq_n_f32(-lo[0] * inv_p[0]);
        let off_y = vdupq_n_f32(-lo[1] * inv_p[1]);
        let off_z = vdupq_n_f32(-lo[2] * inv_p[2]);
        let k_poly = vld1q_f32([k, V2_ROT_ENC, 15.0, 1e-5].as_ptr());
        let v15 = vdupq_laneq_f32::<2>(k_poly);
        let s_eps = vdupq_laneq_f32::<3>(k_poly);
        let bias_v = vdupq_n_s32(bias);
        let u63 = vdupq_n_u32(63);
        let two = vdupq_n_u32(2);
        let sign = vdupq_n_u32(0x8000_0000);
        // NaN-propagating min/max over `l2q` catches degenerate (<= 1e-12, Inf, NaN) quats once after loop.
        let mut min_l2 = vdupq_n_f32(1.0);
        let mut max_l2 = vdupq_n_f32(1.0);
        let mut gp = gs.as_ptr().cast::<f32>();
        let mut dp = dst.as_mut_ptr().cast::<u32>();
        for _ in 0..n4 {
            let l4 = |i: usize, off: usize| vld1q_f32(gp.add(i * 14 + off));
            let l2 = |i: usize| vld1_f32(gp.add(i * 14 + 12));

            // Group 0: positions + sx -> immediately pack w0 and initial w1 (frees px/py/pz/qx/qy/qz).
            let [px, py, pz, sx] = tr4(l4(0, 0), l4(1, 0), l4(2, 0), l4(3, 0));
            let qx = vcvtnq_u32_f32(vfmaq_laneq_f32::<0>(off_x, px, in_v));
            let qy = vcvtnq_u32_f32(vfmaq_laneq_f32::<1>(off_y, py, in_v));
            let qz = vcvtnq_u32_f32(vfmaq_laneq_f32::<2>(off_z, pz, in_v));
            let w0 = vsliq_n_u32::<24>(vsliq_n_u32::<12>(qx, qy), qz);
            let qz_hi = vshrq_n_u32::<8>(qz);

            // Group 1: sy, sz, r0, r1 -> quantize scales (5 NEON ops/scale; frees sx/sy/sz before colors).
            let [sy, sz, r0, r1] = tr4(l4(0, 4), l4(1, 4), l4(2, 4), l4(3, 4));
            let qs = |s: float32x4_t| {
                let sv = if CLAMP_S { vmaxnmq_f32(s, s_eps) } else { s };
                let b = vreinterpretq_s32_f32(sv);
                let d = vcvtq_n_f32_s32::<23>(vsubq_s32(b, bias_v));
                vminq_u32(vcvtnq_u32_f32(vmulq_laneq_f32::<0>(d, k_poly)), u63)
            };
            let s0 = qs(sx);
            let s1 = qs(sy);
            let s2 = qs(sz);

            // Group 2: r2, r3, cr, cg, cb, op -> tree-pack RGBA6666 + s0 into w1 (low latency depth).
            let [r2, r3, cr, cg] = tr4(l4(0, 8), l4(1, 8), l4(2, 8), l4(3, 8));
            let e0 = vcombine_f32(l2(0), l2(2));
            let e1 = vcombine_f32(l2(1), l2(3));
            let (cb, op) = (vtrn1q_f32(e0, e1), vtrn2q_f32(e0, e1));
            let q6 = |v: float32x4_t| vminq_u32(vcvtnq_u32_f32(vmulq_laneq_f32::<3>(v, in_v)), u63);
            let c_rg = vsliq_n_u32::<6>(q6(cr), q6(cg));
            let c_bo = vsliq_n_u32::<6>(q6(cb), q6(op));
            let c_all = vsliq_n_u32::<12>(c_rg, c_bo);
            let w1 = vsliq_n_u32::<28>(vsliq_n_u32::<4>(qz_hi, c_all), s0);

            // Group 3: smallest-3 quaternion (vrsqrteq has <0.4% error = <0.06 of a 5-bit LSB).
            let l2q = vfmaq_f32(vfmaq_f32(vfmaq_f32(vmulq_f32(r0, r0), r1, r1), r2, r2), r3, r3);
            min_l2 = vminq_f32(min_l2, l2q);
            max_l2 = vmaxq_f32(max_l2, l2q);
            let rs = vrsqrteq_f32(l2q);
            let c01 = vcagtq_f32(r1, r0);
            let c23 = vcagtq_f32(r3, r2);
            let m01 = vbslq_f32(c01, r1, r0);
            let m23 = vbslq_f32(c23, r3, r2);
            let cm = vcagtq_f32(m23, m01);
            let ms = vbslq_f32(cm, m23, m01);
            let mi = vsubq_u32(vandq_u32(cm, two), vbslq_u32(cm, c23, c01));
            let o0 = vbslq_f32(vorrq_u32(cm, c01), r0, r1);
            let o1 = vbslq_f32(cm, r1, r2);
            let o2 = vbslq_f32(vandq_u32(cm, c23), r2, r3);
            let kk = vbslq_f32(sign, ms, vmulq_laneq_f32::<1>(rs, k_poly));
            let q5 = |o: float32x4_t| vcvtnq_u32_f32(vfmaq_f32(v15, o, kk));

            let s_pack = vsliq_n_u32::<8>(vsliq_n_u32::<2>(vshrq_n_u32::<4>(s0), s1), s2);
            let q_lo = vsliq_n_u32::<2>(mi, q5(o0));
            let q_hi = vsliq_n_u32::<5>(q5(o1), q5(o2));
            let w2 = vsliq_n_u32::<14>(s_pack, vsliq_n_u32::<7>(q_lo, q_hi));
            vst3q_u32(dp, uint32x4x3_t(w0, w1, w2));
            gp = gp.add(56);
            dp = dp.add(12);
        }
        // Cold fixup for degenerate (zero, Inf, or NaN) quaternions: patch bits 14..31 to identity.
        if !(vminvq_f32(min_l2) > 1e-12 && vmaxvq_f32(max_l2) < f32::INFINITY) {
            const IDENT_HI: u32 = (15 << 16) | (15 << 21) | (15 << 26);
            for (i, g) in gs[..n4 * 4].iter().enumerate() {
                let r = g.rot;
                let l2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3];
                if !(l2 > 1e-12 && l2.is_finite()) {
                    let off = i * V2_STRIDE + 8;
                    let w2 = u32::from_le_bytes([dst[off], dst[off + 1], dst[off + 2], dst[off + 3]]);
                    let fixed = (w2 & 0x3FFF) | IDENT_HI;
                    dst[off..off + 4].copy_from_slice(&fixed.to_le_bytes());
                }
            }
        }
    }
    n4 * 4
}

/// `.dsplat` v2 NEON decoder for `dst.len() & !3` splats (returns decoded count):
/// - LD3 de-interleaves 4 records of 3 u32 words;
/// - extracts `w1, w2` to GPRs once for direct L1 `scale_lut` lookups (eliminates 1 full 4x4
///   register transpose, 8 NEON lane inserts, and 4 NEON shifts);
/// - branchless inner loop (`max_ss` catches any corrupt `|abc|^2 > 1.0` once after the loop);
/// - ascending non-overlapping stores `[pos+sx] (16B) + [sy,sz] (8B) + [rot] (16B) + [rgba] (16B)`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn decode_v2_neon(
    src: &[u8],
    dst: &mut [Gaussian],
    lo: [f32; 3],
    p_step: [f32; 3],
    scale_lut: &[f32; 64],
) -> usize {
    use core::arch::aarch64::*;
    const INV63: f32 = 1.0 / 63.0;
    let n4 = (src.len() / (4 * V2_STRIDE)).min(dst.len() / 4);
    // SAFETY: NEON baseline; `n4 * 48 <= src.len()` and `n4 * 4 <= dst.len()`.
    unsafe {
        let lo_x = vdupq_n_f32(lo[0]);
        let lo_y = vdupq_n_f32(lo[1]);
        let lo_z = vdupq_n_f32(lo[2]);
        let step_c = vld1q_f32([p_step[0], p_step[1], p_step[2], INV63].as_ptr());
        let rot_c = vld1q_f32([V2_ROT_DEC, 1.0, 0.0, 0.0].as_ptr());
        let one = vdupq_laneq_f32::<1>(rot_c);
        let zero = vdupq_laneq_f32::<2>(rot_c);
        let u4095 = vdupq_n_u32(0xFFF);
        let u63 = vdupq_n_u32(0x3F);
        let u31 = vdupq_n_u32(0x1F);
        let s15 = vdupq_n_s32(15);
        let u3 = vdupq_n_u32(3);
        let u1 = vdupq_n_u32(1);
        let u2 = vdupq_n_u32(2);
        let mut max_ss = zero;
        let lut = scale_lut.as_ptr();
        let mut sp = src.as_ptr().cast::<u32>();
        let mut gp = dst.as_mut_ptr();
        for _ in 0..n4 {
            let v = vld3q_u32(sp);
            let (w0, w1, w2) = (v.0, v.1, v.2);

            // Positions (3x12-bit)
            let px = vfmaq_laneq_f32::<0>(lo_x, vcvtq_f32_u32(vandq_u32(w0, u4095)), step_c);
            let py = vfmaq_laneq_f32::<1>(lo_y, vcvtq_f32_u32(vandq_u32(vshrq_n_u32::<12>(w0), u4095)), step_c);
            let pz_u = vandq_u32(vsliq_n_u32::<8>(vshrq_n_u32::<24>(w0), w1), u4095);
            let pz = vfmaq_laneq_f32::<2>(lo_z, vcvtq_f32_u32(pz_u), step_c);

            // Scale LUT indices extracted once via GPRs from w1, w2 (4 umovs total).
            let w1_64 = vreinterpretq_u64_u32(w1);
            let w2_64 = vreinterpretq_u64_u32(w2);
            let a1 = vgetq_lane_u64::<0>(w1_64);
            let b1 = vgetq_lane_u64::<1>(w1_64);
            let a2 = vgetq_lane_u64::<0>(w2_64);
            let b2 = vgetq_lane_u64::<1>(w2_64);

            let s0_0 = (((a1 >> 28) & 0xF) | ((a2 & 3) << 4)) as usize;
            let s0_1 = (((a1 >> 60) & 0xF) | (((a2 >> 32) & 3) << 4)) as usize;
            let s0_2 = (((b1 >> 28) & 0xF) | ((b2 & 3) << 4)) as usize;
            let s0_3 = (((b1 >> 60) & 0xF) | (((b2 >> 32) & 3) << 4)) as usize;
            let sx = vcombine_f32(
                vld1_lane_f32::<1>(lut.add(s0_1), vld1_dup_f32(lut.add(s0_0))),
                vld1_lane_f32::<1>(lut.add(s0_3), vld1_dup_f32(lut.add(s0_2))),
            );
            let [p0, p1, p2, p3] = tr4(px, py, pz, sx);

            // Store [pos, sx] + [sy, sz] first (frees px/py/pz/sx/p0..p3 before computing rot/rgba).
            let o0 = (*gp).pos.as_mut_ptr();
            vst1q_f32(o0, p0);
            *o0.add(4) = *lut.add(((a2 >> 2) & 63) as usize);
            *o0.add(5) = *lut.add(((a2 >> 8) & 63) as usize);

            let o1 = (*gp.add(1)).pos.as_mut_ptr();
            vst1q_f32(o1, p1);
            *o1.add(4) = *lut.add(((a2 >> 34) & 63) as usize);
            *o1.add(5) = *lut.add(((a2 >> 40) & 63) as usize);

            let o2 = (*gp.add(2)).pos.as_mut_ptr();
            vst1q_f32(o2, p2);
            *o2.add(4) = *lut.add(((b2 >> 2) & 63) as usize);
            *o2.add(5) = *lut.add(((b2 >> 8) & 63) as usize);

            let o3 = (*gp.add(3)).pos.as_mut_ptr();
            vst1q_f32(o3, p3);
            *o3.add(4) = *lut.add(((b2 >> 34) & 63) as usize);
            *o3.add(5) = *lut.add(((b2 >> 40) & 63) as usize);

            // Smallest-3 quaternion -> 4x4 transpose -> 16B stores at offset 6 (frees w2/rot regs).
            let mi = vandq_u32(vshrq_n_u32::<14>(w2), u3);
            let rq = |x: uint32x4_t| {
                let q = vreinterpretq_s32_u32(vandq_u32(x, u31));
                vmulq_laneq_f32::<0>(vcvtq_f32_s32(vsubq_s32(q, s15)), rot_c)
            };
            let a = rq(vshrq_n_u32::<16>(w2));
            let b = rq(vshrq_n_u32::<21>(w2));
            let c = rq(vshrq_n_u32::<26>(w2));
            let ss = vfmaq_f32(vfmaq_f32(vmulq_f32(c, c), b, b), a, a);
            max_ss = vmaxnmq_f32(max_ss, ss);
            let w = vsqrtq_f32(vmaxnmq_f32(vsubq_f32(one, ss), zero));
            let m0 = vceqzq_u32(mi);
            let m1 = vceqq_u32(mi, u1);
            let m2 = vceqq_u32(mi, u2);
            let r0 = vbslq_f32(m0, w, a);
            let r1 = vbslq_f32(m0, a, vbslq_f32(m1, w, b));
            let r2 = vbslq_f32(vcltq_u32(mi, u2), b, vbslq_f32(m2, w, c));
            let r3 = vbslq_f32(vcltq_u32(mi, u3), c, w);
            let [rot0, rot1, rot2, rot3] = tr4(r0, r1, r2, r3);
            vst1q_f32(o0.add(6), rot0);
            vst1q_f32(o1.add(6), rot1);
            vst1q_f32(o2.add(6), rot2);
            vst1q_f32(o3.add(6), rot3);

            // RGBA6666 -> 4x4 transpose -> 16B stores at offset 10 (strictly ascending stores).
            let c6 = |x: uint32x4_t| vmulq_laneq_f32::<3>(vcvtq_f32_u32(vandq_u32(x, u63)), step_c);
            let cr = c6(vshrq_n_u32::<4>(w1));
            let cg = c6(vshrq_n_u32::<10>(w1));
            let cb = c6(vshrq_n_u32::<16>(w1));
            let op = c6(vshrq_n_u32::<22>(w1));
            let [rgba0, rgba1, rgba2, rgba3] = tr4(cr, cg, cb, op);
            vst1q_f32(o0.add(10), rgba0);
            vst1q_f32(o1.add(10), rgba1);
            vst1q_f32(o2.add(10), rgba2);
            vst1q_f32(o3.add(10), rgba3);

            sp = sp.add(12);
            gp = gp.add(4);
        }
        if vmaxvq_f32(max_ss) > 1.0 {
            // Cold fixup for corrupt/extreme records where |abc|^2 > 1.0.
            for g in &mut dst[..n4 * 4] {
                let r = g.rot;
                let l2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3];
                if l2 > 1.0 + 1e-6 {
                    let inv = l2.sqrt().recip();
                    g.rot = [r[0] * inv, r[1] * inv, r[2] * inv, r[3] * inv];
                }
            }
        }
    }
    n4 * 4
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn project_depths(gs: &[Gaussian], out: &mut [u32], f: [f32; 3], base: f32) -> (f32, f32) {
    use core::arch::aarch64::*;
    let n = gs.len().min(out.len());
    let chunks = n / 8;
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    if chunks > 0 {
        // SAFETY: NEON baseline on aarch64; `Gaussian` is `#[repr(C)]` with `pos` + `scale[0]`
        // forming 4 contiguous initialised `f32`s at offset 0; `chunks * 8 <= n`.
        unsafe {
            let fv = vld1q_f32([f[0], f[1], f[2], 0.0].as_ptr());
            let neg_base = vdupq_n_f32(-base);
            let mut lo0 = vdupq_n_f32(f32::INFINITY);
            let mut lo1 = lo0;
            let mut hi0 = vdupq_n_f32(f32::NEG_INFINITY);
            let mut hi1 = hi0;
            let mut gp = gs.as_ptr();
            let mut dp = out.as_mut_ptr();
            for _ in 0..chunks {
                let p0 = vld1q_f32((*gp).pos.as_ptr());
                let p1 = vld1q_f32((*gp.add(1)).pos.as_ptr());
                let p2 = vld1q_f32((*gp.add(2)).pos.as_ptr());
                let p3 = vld1q_f32((*gp.add(3)).pos.as_ptr());
                let t0 = vtrn1q_f32(p0, p1);
                let t1 = vtrn2q_f32(p0, p1);
                let t2 = vtrn1q_f32(p2, p3);
                let t3 = vtrn2q_f32(p2, p3);
                let vx0 = vreinterpretq_f32_u64(vtrn1q_u64(vreinterpretq_u64_f32(t0), vreinterpretq_u64_f32(t2)));
                let vz0 = vreinterpretq_f32_u64(vtrn2q_u64(vreinterpretq_u64_f32(t0), vreinterpretq_u64_f32(t2)));
                let vy0 = vreinterpretq_f32_u64(vtrn1q_u64(vreinterpretq_u64_f32(t1), vreinterpretq_u64_f32(t3)));
                let d0 = vfmaq_laneq_f32(
                    vfmaq_laneq_f32(vfmaq_laneq_f32(neg_base, vx0, fv, 0), vy0, fv, 1),
                    vz0,
                    fv,
                    2,
                );

                let p4 = vld1q_f32((*gp.add(4)).pos.as_ptr());
                let p5 = vld1q_f32((*gp.add(5)).pos.as_ptr());
                let p6 = vld1q_f32((*gp.add(6)).pos.as_ptr());
                let p7 = vld1q_f32((*gp.add(7)).pos.as_ptr());
                let t4 = vtrn1q_f32(p4, p5);
                let t5 = vtrn2q_f32(p4, p5);
                let t6 = vtrn1q_f32(p6, p7);
                let t7 = vtrn2q_f32(p6, p7);
                let vx1 = vreinterpretq_f32_u64(vtrn1q_u64(vreinterpretq_u64_f32(t4), vreinterpretq_u64_f32(t6)));
                let vz1 = vreinterpretq_f32_u64(vtrn2q_u64(vreinterpretq_u64_f32(t4), vreinterpretq_u64_f32(t6)));
                let vy1 = vreinterpretq_f32_u64(vtrn1q_u64(vreinterpretq_u64_f32(t5), vreinterpretq_u64_f32(t7)));
                let d1 = vfmaq_laneq_f32(
                    vfmaq_laneq_f32(vfmaq_laneq_f32(neg_base, vx1, fv, 0), vy1, fv, 1),
                    vz1,
                    fv,
                    2,
                );

                vst1q_u32(dp, vreinterpretq_u32_f32(d0));
                vst1q_u32(dp.add(4), vreinterpretq_u32_f32(d1));
                lo0 = vminnmq_f32(lo0, d0);
                lo1 = vminnmq_f32(lo1, d1);
                hi0 = vmaxnmq_f32(hi0, d0);
                hi1 = vmaxnmq_f32(hi1, d1);
                gp = gp.add(8);
                dp = dp.add(8);
            }
            lo = vminnmvq_f32(vminnmq_f32(lo0, lo1));
            hi = vmaxnmvq_f32(vmaxnmq_f32(hi0, hi1));
        }
    }
    for (g, slot) in gs[chunks * 8..n].iter().zip(&mut out[chunks * 8..n]) {
        let d = dot3(&g.pos, f) - base;
        *slot = d.to_bits();
        lo = if d < lo { d } else { lo };
        hi = if d > hi { d } else { hi };
    }
    (lo, hi)
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn project_depths(gs: &[Gaussian], out: &mut [u32], f: [f32; 3], base: f32) -> (f32, f32) {
    let mut lo8 = [f32::INFINITY; 8];
    let mut hi8 = [f32::NEG_INFINITY; 8];
    let mut gc = gs.chunks_exact(8);
    let mut dc = out.chunks_exact_mut(8);
    for (g8, d8) in (&mut gc).zip(&mut dc) {
        let mut d = [0.0f32; 8];
        for k in 0..8 {
            d[k] = dot3(&g8[k].pos, f) - base;
        }
        for k in 0..8 {
            d8[k] = d[k].to_bits();
            lo8[k] = if d[k] < lo8[k] { d[k] } else { lo8[k] };
            hi8[k] = if d[k] > hi8[k] { d[k] } else { hi8[k] };
        }
    }
    for (g, slot) in gc.remainder().iter().zip(dc.into_remainder()) {
        let d = dot3(&g.pos, f) - base;
        *slot = d.to_bits();
        lo8[0] = if d < lo8[0] { d } else { lo8[0] };
        hi8[0] = if d > hi8[0] { d } else { hi8[0] };
    }
    let mut lo = lo8[0];
    let mut hi = hi8[0];
    for k in 1..8 {
        lo = lo.min(lo8[k]);
        hi = hi.max(hi8[k]);
    }
    (lo, hi)
}

/// Pass 2 (fused): depth -> inverted 10-bit bucket key (far = 0) for 8 lanes at a time (a pure
/// f32x4 map: sub, mul, clamp, cvt, narrow), stored once and counted straight from registers
/// into 4 interleaved sub-histograms (lane `i & 3`), so runs of equal keys never serialise on
/// store-to-load forwarding. Masking proves every index in-bounds (no checks).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn depth_keys_hist(depth: &[u32], keys: &mut [u16], lo: f32, inv_span: f32, hist: &mut Hist4) {
    use core::arch::aarch64::*;
    let n = depth.len().min(keys.len());
    let chunks = n / 8;
    if chunks > 0 {
        // 8 sub-histograms (lane `i & 7`): spatially coherent clouds produce long runs of equal
        // keys, and 8 independent counters keep those read-modify-writes from serialising on
        // store-to-load forwarding within a group.
        let mut h8 = [[0u32; 8]; SORT_BUCKETS];
        // SAFETY: NEON baseline; `chunks * 8 <= n`.
        unsafe {
            let lo_v = vdupq_n_f32(lo);
            let inv_v = vdupq_n_f32(inv_span);
            let zero_v = vdupq_n_f32(0.0);
            let max_v = vdupq_n_f32(SORT_MAX_B);
            let mask_v = vdupq_n_u32(SORT_MASK as u32);
            let mut dp = depth.as_ptr().cast::<f32>();
            let mut kp = keys.as_mut_ptr();
            for _ in 0..chunks {
                let d0 = vld1q_f32(dp);
                let d1 = vld1q_f32(dp.add(4));
                let f0 = vminnmq_f32(vmaxnmq_f32(vmulq_f32(vsubq_f32(d0, lo_v), inv_v), zero_v), max_v);
                let f1 = vminnmq_f32(vmaxnmq_f32(vmulq_f32(vsubq_f32(d1, lo_v), inv_v), zero_v), max_v);
                let k0 = vsubq_u32(mask_v, vcvtq_u32_f32(f0));
                let k1 = vsubq_u32(mask_v, vcvtq_u32_f32(f1));
                let k16 = vcombine_u16(vqmovn_u32(k0), vqmovn_u32(k1));
                vst1q_u16(kp, k16);
                let k64 = vreinterpretq_u64_u16(k16);
                let w0 = vgetq_lane_u64::<0>(k64) as usize;
                let w1 = vgetq_lane_u64::<1>(k64) as usize;
                h8[w0 & SORT_MASK][0] += 1;
                h8[(w0 >> 16) & SORT_MASK][1] += 1;
                h8[(w0 >> 32) & SORT_MASK][2] += 1;
                h8[(w0 >> 48) & SORT_MASK][3] += 1;
                h8[w1 & SORT_MASK][4] += 1;
                h8[(w1 >> 16) & SORT_MASK][5] += 1;
                h8[(w1 >> 32) & SORT_MASK][6] += 1;
                h8[(w1 >> 48) & SORT_MASK][7] += 1;
                dp = dp.add(8);
                kp = kp.add(8);
            }
        }
        // Lanes `l` and `l + 4` both hold indices with `i & 3 == l`: fold to the 4-lane layout.
        for (h, x) in hist.iter_mut().zip(h8.iter()) {
            for l in 0..4 {
                h[l] += x[l] + x[l + 4];
            }
        }
    }
    for (j, (&d, k)) in depth[chunks * 8..n].iter().zip(&mut keys[chunks * 8..n]).enumerate() {
        let b = ((f32::from_bits(d) - lo) * inv_span).max(0.0).min(SORT_MAX_B) as u32;
        *k = (SORT_MASK as u32 - b) as u16;
        hist[*k as usize & SORT_MASK][j & 3] += 1;
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn depth_keys_hist(depth: &[u32], keys: &mut [u16], lo: f32, inv_span: f32, hist: &mut Hist4) {
    #[inline(always)]
    fn key(d: u32, lo: f32, inv_span: f32) -> u16 {
        let b = ((f32::from_bits(d) - lo) * inv_span).max(0.0).min(SORT_MAX_B) as u32;
        (SORT_MASK as u32 - b) as u16
    }
    let mut dc = depth.chunks_exact(8);
    let mut kc = keys.chunks_exact_mut(8);
    for (d8, k8) in (&mut dc).zip(&mut kc) {
        let mut kk = [0u16; 8];
        for j in 0..8 {
            kk[j] = key(d8[j], lo, inv_span);
        }
        k8.copy_from_slice(&kk);
        for j in 0..8 {
            hist[kk[j] as usize & SORT_MASK][j & 3] += 1;
        }
    }
    for (j, (&d, k)) in dc.remainder().iter().zip(kc.into_remainder()).enumerate() {
        *k = key(d, lo, inv_span);
        hist[*k as usize & SORT_MASK][j & 3] += 1;
    }
}

/// Variant of `depth_keys_hist` that overwrites `hist` and `tot` directly in one fused fold
/// (skips 16 KB pre-zeroing and the second pass to compute per-bucket totals).
#[cfg(all(feature = "threads", target_arch = "aarch64"))]
#[inline(always)]
fn depth_keys_hist_tot(depth: &[u32], keys: &mut [u16], lo: f32, inv_span: f32, hist: &mut Hist4, tot: &mut [u32; SORT_BUCKETS]) {
    use core::arch::aarch64::*;
    let n = depth.len().min(keys.len());
    let chunks = n / 8;
    let mut h8 = [[0u32; 8]; SORT_BUCKETS];
    if chunks > 0 {
        // SAFETY: NEON baseline; `chunks * 8 <= n`.
        unsafe {
            let lo_v = vdupq_n_f32(lo);
            let inv_v = vdupq_n_f32(inv_span);
            let zero_v = vdupq_n_f32(0.0);
            let max_v = vdupq_n_f32(SORT_MAX_B);
            let mask_v = vdupq_n_u32(SORT_MASK as u32);
            let mut dp = depth.as_ptr().cast::<f32>();
            let mut kp = keys.as_mut_ptr();
            for _ in 0..chunks {
                let d0 = vld1q_f32(dp);
                let d1 = vld1q_f32(dp.add(4));
                let f0 = vminnmq_f32(vmaxnmq_f32(vmulq_f32(vsubq_f32(d0, lo_v), inv_v), zero_v), max_v);
                let f1 = vminnmq_f32(vmaxnmq_f32(vmulq_f32(vsubq_f32(d1, lo_v), inv_v), zero_v), max_v);
                let k0 = vsubq_u32(mask_v, vcvtq_u32_f32(f0));
                let k1 = vsubq_u32(mask_v, vcvtq_u32_f32(f1));
                let k16 = vcombine_u16(vqmovn_u32(k0), vqmovn_u32(k1));
                vst1q_u16(kp, k16);
                let k64 = vreinterpretq_u64_u16(k16);
                let w0 = vgetq_lane_u64::<0>(k64) as usize;
                let w1 = vgetq_lane_u64::<1>(k64) as usize;
                h8[w0 & SORT_MASK][0] += 1;
                h8[(w0 >> 16) & SORT_MASK][1] += 1;
                h8[(w0 >> 32) & SORT_MASK][2] += 1;
                h8[(w0 >> 48) & SORT_MASK][3] += 1;
                h8[w1 & SORT_MASK][4] += 1;
                h8[(w1 >> 16) & SORT_MASK][5] += 1;
                h8[(w1 >> 32) & SORT_MASK][6] += 1;
                h8[(w1 >> 48) & SORT_MASK][7] += 1;
                dp = dp.add(8);
                kp = kp.add(8);
            }
        }
    }
    for (j, (&d, k)) in depth[chunks * 8..n].iter().zip(&mut keys[chunks * 8..n]).enumerate() {
        let b = ((f32::from_bits(d) - lo) * inv_span).max(0.0).min(SORT_MAX_B) as u32;
        *k = (SORT_MASK as u32 - b) as u16;
        h8[*k as usize & SORT_MASK][j & 3] += 1;
    }
    for ((h, t), x) in hist.iter_mut().zip(tot.iter_mut()).zip(h8.iter()) {
        let a0 = x[0] + x[4];
        let a1 = x[1] + x[5];
        let a2 = x[2] + x[6];
        let a3 = x[3] + x[7];
        *h = [a0, a1, a2, a3];
        *t = a0 + a1 + a2 + a3;
    }
}

#[cfg(all(feature = "threads", not(target_arch = "aarch64")))]
#[inline(always)]
fn depth_keys_hist_tot(depth: &[u32], keys: &mut [u16], lo: f32, inv_span: f32, hist: &mut Hist4, tot: &mut [u32; SORT_BUCKETS]) {
    *hist = [[0u32; 4]; SORT_BUCKETS];
    depth_keys_hist(depth, keys, lo, inv_span, hist);
    for (t, x) in tot.iter_mut().zip(hist.iter()) {
        *t = x[0] + x[1] + x[2] + x[3];
    }
}

/// Exclusive prefix over (bucket, lane): bucket `b` holds lane 0's elements, then lane 1's, ...
/// Converts the counts in place into scatter cursors starting at 0.
#[inline(always)]
fn prefix_hist4(hist: &mut Hist4) {
    // Only the running total is a serial chain (1 add per bucket); the 4 lane offsets are
    // independent adds off it.
    let mut sum = 0u32;
    for h in hist.iter_mut() {
        let [a, b, c, d] = *h;
        *h = [sum, sum + a, sum + a + b, sum + a + b + c];
        sum += a + b + c + d;
    }
}

/// Pass 3: scatter with one independent cursor table per lane (`i & 3`), so consecutive equal
/// keys (spatially coherent clouds) hit different cursors and the four read-modify-write chains
/// overlap instead of serialising. Deterministic; within a bucket order is (lane, index).
#[inline(always)]
fn scatter_hist4(keys: &[u16], cur: &mut Hist4, out: &mut [u32], first_index: u32) {
    assert!(keys.len() <= out.len());
    // SAFETY: `cur` holds exclusive prefix sums from `depth_keys_hist` over `keys`, so every
    // cursor stays below `keys.len() <= out.len()`.
    unsafe { scatter_hist4_raw(keys, cur, out.as_mut_ptr(), first_index) };
}

/// [`scatter_hist4`] into a raw output (shared by parallel scatter chunks).
///
/// # Safety
/// Every cursor in `cur`, advanced once per key of its (bucket, lane), must stay in bounds of
/// `out_ptr`, and no other thread may write those slots concurrently.
#[inline(always)]
unsafe fn scatter_hist4_raw(keys: &[u16], cur: &mut Hist4, out_ptr: *mut u32, first_index: u32) {
    let mut kc = keys.chunks_exact(4);
    let mut i = first_index;
    for k4 in &mut kc {
        #[cfg(target_endian = "little")]
        let (b0, b1, b2, b3) = {
            let w = u64::from_le_bytes([
                k4[0] as u8,
                (k4[0] >> 8) as u8,
                k4[1] as u8,
                (k4[1] >> 8) as u8,
                k4[2] as u8,
                (k4[2] >> 8) as u8,
                k4[3] as u8,
                (k4[3] >> 8) as u8,
            ]);
            (
                (w as usize) & SORT_MASK,
                ((w >> 16) as usize) & SORT_MASK,
                ((w >> 32) as usize) & SORT_MASK,
                ((w >> 48) as usize) & SORT_MASK,
            )
        };
        #[cfg(not(target_endian = "little"))]
        let (b0, b1, b2, b3) = (
            k4[0] as usize & SORT_MASK,
            k4[1] as usize & SORT_MASK,
            k4[2] as usize & SORT_MASK,
            k4[3] as usize & SORT_MASK,
        );
        // Lanes 0..3 live at distinct offsets within any bucket, so all 4 cursor reads/writes are independent.
        let s0 = cur[b0][0];
        let s1 = cur[b1][1];
        let s2 = cur[b2][2];
        let s3 = cur[b3][3];
        cur[b0][0] = s0 + 1;
        cur[b1][1] = s1 + 1;
        cur[b2][2] = s2 + 1;
        cur[b3][3] = s3 + 1;
        // SAFETY: cursors stay in bounds (function contract).
        unsafe {
            *out_ptr.add(s0 as usize) = i;
            *out_ptr.add(s1 as usize) = i + 1;
            *out_ptr.add(s2 as usize) = i + 2;
            *out_ptr.add(s3 as usize) = i + 3;
        }
        i += 4;
    }
    for (j, &k) in kc.remainder().iter().enumerate() {
        let s = &mut cur[k as usize & SORT_MASK][j];
        let idx = *s as usize;
        // SAFETY: cursors stay in bounds (function contract).
        unsafe { *out_ptr.add(idx) = i + j as u32 };
        *s += 1;
    }
}

/// Element count from which every pass of a splat kernel is worth splitting into phases across
/// Rayon workers (see `sort_depths_phased`).
#[cfg(feature = "threads")]
const PAR_MIN: usize = 100_000;

/// Codec / sort kernels are 10-100 us of pure streaming SIMD at 16k splats: with the
/// caller-participating `dynpar` fan-out (no blocking on pool wake-up) splitting them pays off
/// from a few thousand splats.
#[cfg(feature = "threads")]
const CODEC_PAR_MIN: usize = 8192;

/// Cluster-isolated P+E range partition for `[0, n)` (all boundaries multiples of 4):
/// - Chunks `0..nc_p` cover `[0, p_end)` (P-cluster L2-resident queue, ~70% on 4P+4E).
/// - Chunks `nc_p..nc_p + nc_e` cover `[p_end, n)` (E-cluster L2-resident queue, ~30%).
/// P-cores drain their own queue first (zero cross-cluster atomic traffic), then steal any
/// leftover E-chunks if an E-core was preempted; E-cores only drain E-chunks (zero tail stall).
#[cfg(feature = "threads")]
#[derive(Clone, Copy)]
struct SplitPe {
    p_end: usize,
    ch_p: usize,
    nc_p: usize,
    ch_e: usize,
    nc_e: usize,
}

#[cfg(feature = "threads")]
impl SplitPe {
    #[inline(always)]
    fn total(self) -> usize {
        self.nc_p + self.nc_e
    }
    #[inline(always)]
    fn range(self, c: usize, n: usize) -> (usize, usize) {
        if c < self.nc_p {
            let s = c * self.ch_p;
            (s, (s + self.ch_p).min(self.p_end))
        } else {
            let ec = c - self.nc_p;
            let s = self.p_end + ec * self.ch_e;
            (s, (s + self.ch_e).min(n))
        }
    }
}

#[cfg(feature = "threads")]
#[inline]
fn codec_split_pe(n: usize, e_weight_per_core: usize) -> Option<SplitPe> {
    if n < CODEC_PAR_MIN {
        return None;
    }
    let pt = dynpar::p_threads();
    let eh = dynpar::e_helpers();
    if pt + eh < 2 {
        return None;
    }
    if eh == 0 {
        let nc_p = (pt * 2).max(2);
        let ch_p = ((n.div_ceil(nc_p) + 3) & !3).max(256);
        let nc_p = n.div_ceil(ch_p);
        return Some(SplitPe { p_end: n, ch_p, nc_p, ch_e: 1, nc_e: 0 });
    }
    // Weight each P-core as 100 and each E-core as `e_weight_per_core`, with 1 chunk/core for <=32k
    // so P-cluster L2 and E-cluster L2 stay completely disjoint (zero cross-cluster L2 ping-pong ->
    // breaks the 138 GB/s 4P L2 write ceiling up to ~190 GB/s across both clusters).
    let p_w = pt * 100;
    let e_w = eh * e_weight_per_core;
    let p_end = ((n * p_w / (p_w + e_w)) & !3).clamp(4, n);
    let e_len = n - p_end;
    let mult = if n <= 32_768 { 1 } else { 2 };
    let target_p = (pt * mult).min((p_end / 512).max(1));
    let ch_p = (p_end.div_ceil(target_p) + 3) & !3;
    let nc_p = p_end.div_ceil(ch_p);
    let (ch_e, nc_e) = if e_len >= 256 {
        let target_e = (eh * mult).min((e_len / 256).max(1));
        let ce = (e_len.div_ceil(target_e) + 3) & !3;
        (ce, e_len.div_ceil(ce))
    } else if e_len > 0 {
        (e_len, 1)
    } else {
        (1, 0)
    };
    Some(SplitPe { p_end, ch_p, nc_p, ch_e, nc_e })
}

/// Runs `f(start, end)` over disjoint ranges covering `[0, n)` (every `start` a multiple of 4):
/// fanned out across cluster-isolated P+E queues when worthwhile, otherwise serial.
#[inline]
fn for_ranges_pe(n: usize, e_weight_per_core: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    #[cfg(feature = "threads")]
    if let Some(sp) = codec_split_pe(n, e_weight_per_core) {
        dynpar::for_each_pe(sp.total(), sp.nc_p, &|c| {
            let (s, e) = sp.range(c, n);
            f(s, e);
        });
        return;
    }
    #[cfg(not(feature = "threads"))]
    let _ = e_weight_per_core;
    if n > 0 {
        f(0, n);
    }
}

/// `Send + Sync` raw pointer for disjoint per-range writes from `for_ranges` / `dynpar` bodies.
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
// SAFETY: only used to write disjoint, range-indexed elements from fan-out bodies.
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    /// Method (not field) access, so closures capture the whole `Sync` wrapper.
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

/// Zero-allocation, lock-free caller-participating persistent worker pool with separate P-core
/// and E-core wake channels and cluster-isolated work queues (`p_next` + `e_next`).
///
/// - Uses an 8-slot ring with a Dekker/RCU epoch fence (`slot.epoch.store(0, SeqCst)` +
///   `p_readers == 0 && e_readers == 0`) so stack-borrowed closures have zero heap allocation
///   (`Arc`) and zero `RwLock` overhead (~0.1 µs dispatch).
/// - P-only mode (`phased`) wakes ONLY the P-core helpers (`hw.perflevel0.physicalcpu - 1`),
///   leaving E-helpers parked so multi-barrier kernels (`depth_sort`) suffer zero stragglers.
/// - P+E mode (`phased_pe` / `for_each_pe`) ALSO wakes E-core helpers. Each phase splits its
///   `0..counts[p]` chunks at `p_split[p]`: P-cores drain `0..p_split[p]` from a P-cluster-local
///   `p_next` atomic (zero cross-cluster cache-line bouncing!) and then steal any remaining chunks
///   from `e_next` (`p_split[p]..counts[p]`); E-cores only drain `e_next` (zero P-core tail stall).
#[cfg(feature = "threads")]
mod dynpar {
    use std::cell::UnsafeCell;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex, OnceLock};

    const MAX_PHASES: usize = 4;
    const SLOTS: usize = 8;
    const SPIN_ITERS: u32 = 60_000;

    #[repr(align(64))]
    struct CacheLine<T>(T);

    #[repr(align(64))]
    struct Slot {
        epoch: AtomicUsize,
        phases: usize,
        counts: [usize; MAX_PHASES],
        p_split: [usize; MAX_PHASES],
        body: UnsafeCell<*const (dyn Fn(usize, usize) + Sync + 'static)>,
        after: UnsafeCell<*const (dyn Fn(usize) + Sync + 'static)>,
        /// Separate cache lines for P-cluster vs E-cluster atomics (zero inter-cluster bouncing!).
        p_readers: CacheLine<AtomicUsize>,
        e_readers: CacheLine<AtomicUsize>,
        p_next: CacheLine<[AtomicUsize; MAX_PHASES]>,
        e_next: CacheLine<[AtomicUsize; MAX_PHASES]>,
        done: CacheLine<[AtomicUsize; MAX_PHASES]>,
        ready: CacheLine<AtomicUsize>,
    }

    impl Slot {
        fn new() -> Self {
            fn noop2(_: usize, _: usize) {}
            fn noop1(_: usize) {}
            Self {
                epoch: AtomicUsize::new(0),
                phases: 0,
                counts: [0; MAX_PHASES],
                p_split: [0; MAX_PHASES],
                body: UnsafeCell::new(&noop2 as &(dyn Fn(usize, usize) + Sync + 'static) as *const _),
                after: UnsafeCell::new(&noop1 as &(dyn Fn(usize) + Sync + 'static) as *const _),
                p_readers: CacheLine(AtomicUsize::new(0)),
                e_readers: CacheLine(AtomicUsize::new(0)),
                p_next: CacheLine(Default::default()),
                e_next: CacheLine(Default::default()),
                done: CacheLine(Default::default()),
                ready: CacheLine(AtomicUsize::new(0)),
            }
        }
    }

    #[repr(align(64))]
    struct WakeChannel {
        epoch: AtomicUsize,
        sleepers: AtomicUsize,
        mu: Mutex<()>,
        cv: Condvar,
    }

    impl WakeChannel {
        fn new() -> Self {
            Self {
                epoch: AtomicUsize::new(0),
                sleepers: AtomicUsize::new(0),
                mu: Mutex::new(()),
                cv: Condvar::new(),
            }
        }
        #[inline(always)]
        fn notify_if_sleeping(&self) {
            if self.sleepers.load(Ordering::Relaxed) > 0 {
                let _g = self.mu.lock().unwrap_or_else(|err| err.into_inner());
                self.cv.notify_all();
            }
        }
    }

    struct Pool {
        in_use: AtomicBool,
        global_epoch: AtomicUsize,
        p_chan: WakeChannel,
        e_chan: WakeChannel,
        slots: [Slot; SLOTS],
        p_helpers: usize,
        e_helpers: usize,
    }
    // SAFETY: `Slot` payloads are only mutated while `in_use == true` and both reader counts are 0
    // before publishing `slot.epoch`, and only read while holding a reader guard.
    unsafe impl Send for Pool {}
    unsafe impl Sync for Pool {}

    fn core_counts() -> (usize, usize) {
        let all = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 16);
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn sysctlbyname(
                    name: *const u8,
                    oldp: *mut core::ffi::c_void,
                    oldlenp: *mut usize,
                    newp: *mut core::ffi::c_void,
                    newlen: usize,
                ) -> i32;
            }
            let mut p: i32 = 0;
            let mut len = std::mem::size_of::<i32>();
            // SAFETY: NUL-terminated name, `p` / `len` describe a valid 4-byte out buffer.
            let r = unsafe {
                sysctlbyname(
                    c"hw.perflevel0.physicalcpu".as_ptr().cast(),
                    (&mut p as *mut i32).cast(),
                    &mut len,
                    std::ptr::null_mut(),
                    0,
                )
            };
            if r == 0 && p >= 2 {
                let p_cores = (p as usize).min(all);
                let e_cores = all.saturating_sub(p_cores);
                // Leave 2 E-cores free on 6E+ chips (4E on M4, 3E on 4E) for OS background daemons.
                let e_use = if e_cores >= 5 {
                    (e_cores - 2).min(4)
                } else if e_cores >= 2 {
                    e_cores - 1
                } else {
                    0
                };
                return (p_cores - 1, e_use);
            }
        }
        (all - 1, 0)
    }

    fn get_pool() -> &'static Pool {
        static POOL: OnceLock<&'static Pool> = OnceLock::new();
        *POOL.get_or_init(|| {
            let (p_helpers, e_helpers) = core_counts();
            let pool: &'static Pool = Box::leak(Box::new(Pool {
                in_use: AtomicBool::new(false),
                global_epoch: AtomicUsize::new(0),
                p_chan: WakeChannel::new(),
                e_chan: WakeChannel::new(),
                slots: std::array::from_fn(|_| Slot::new()),
                p_helpers,
                e_helpers,
            }));
            for i in 0..p_helpers {
                let _ = std::thread::Builder::new()
                    .name(format!("sculpt-p-{i}"))
                    .spawn(move || worker_loop(pool, false));
            }
            for i in 0..e_helpers {
                let _ = std::thread::Builder::new()
                    .name(format!("sculpt-e-{i}"))
                    .spawn(move || worker_loop(pool, true));
            }
            pool
        })
    }

    #[inline]
    pub fn p_threads() -> usize {
        get_pool().p_helpers + 1
    }

    #[inline]
    pub fn e_helpers() -> usize {
        get_pool().e_helpers
    }

    #[inline(always)]
    fn wait_until(mut cond: impl FnMut() -> bool) {
        let mut spins = 0u32;
        while !cond() {
            if spins < 4096 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
    }

    #[inline(always)]
    fn run_slot(slot: &Slot, is_e: bool) {
        let phases = slot.phases;
        // SAFETY: caller's `Slot` fields are immutable for the lifetime of this epoch (`readers > 0`).
        let body = unsafe { *slot.body.get() };
        let after = unsafe { *slot.after.get() };
        for p in 0..phases {
            let cnt = slot.counts[p];
            let split = slot.p_split[p];
            if is_e {
                if split >= cnt {
                    continue;
                }
                wait_until(|| slot.ready.0.load(Ordering::Acquire) >= p);
                let mut completed = 0usize;
                loop {
                    let i = slot.e_next.0[p].fetch_add(1, Ordering::Relaxed);
                    if i >= cnt {
                        break;
                    }
                    // SAFETY: chunk `i < cnt` is exclusively claimed.
                    unsafe { (*body)(p, i) };
                    completed += 1;
                }
                if completed > 0 && slot.done.0[p].fetch_add(completed, Ordering::AcqRel) + completed == cnt {
                    // SAFETY: last chunk of phase `p` completed.
                    unsafe { (*after)(p) };
                    slot.ready.0.store(p + 1, Ordering::Release);
                }
            } else {
                wait_until(|| slot.ready.0.load(Ordering::Acquire) >= p);
                let mut completed = 0usize;
                // Drain P-cluster queue first (zero cross-cluster atomic traffic).
                loop {
                    let i = slot.p_next.0[p].fetch_add(1, Ordering::Relaxed);
                    if i >= split {
                        break;
                    }
                    // SAFETY: chunk `i < split <= cnt` is exclusively claimed.
                    unsafe { (*body)(p, i) };
                    completed += 1;
                }
                // Work-stealing safety net: peek before fetch_add so finished E-queues stay read-only.
                if split < cnt && slot.e_next.0[p].load(Ordering::Relaxed) < cnt {
                    loop {
                        let i = slot.e_next.0[p].fetch_add(1, Ordering::Relaxed);
                        if i >= cnt {
                            break;
                        }
                        // SAFETY: chunk `i < cnt` is exclusively claimed.
                        unsafe { (*body)(p, i) };
                        completed += 1;
                    }
                }
                if completed > 0 && slot.done.0[p].fetch_add(completed, Ordering::AcqRel) + completed == cnt {
                    // SAFETY: last chunk of phase `p` completed.
                    unsafe { (*after)(p) };
                    slot.ready.0.store(p + 1, Ordering::Release);
                }
            }
        }
    }

    fn worker_loop(pool: &'static Pool, is_e: bool) {
        #[cfg(target_os = "macos")]
        {
            // Pin P-workers to USER_INTERACTIVE (guaranteed P-cores, full boost clock) and
            // E-workers to UTILITY (strictly below caller's DEFAULT QoS so E-workers never
            // steal a P-core from the caller).
            const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
            const QOS_CLASS_UTILITY: u32 = 0x11;
            unsafe extern "C" {
                fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
            }
            let qos = if is_e { QOS_CLASS_UTILITY } else { QOS_CLASS_USER_INTERACTIVE };
            // SAFETY: thread-local QoS hint syscall on macOS.
            unsafe {
                let _ = pthread_set_qos_class_self_np(qos, 0);
            }
        }
        let chan = if is_e { &pool.e_chan } else { &pool.p_chan };
        let mut last_e = 0usize;
        loop {
            let mut e = chan.epoch.load(Ordering::Acquire);
            if e == last_e {
                let mut spins = 0u32;
                while spins < SPIN_ITERS {
                    std::hint::spin_loop();
                    e = chan.epoch.load(Ordering::Acquire);
                    if e != last_e {
                        break;
                    }
                    spins += 1;
                }
                if e == last_e {
                    let mut g = chan.mu.lock().unwrap_or_else(|err| err.into_inner());
                    chan.sleepers.fetch_add(1, Ordering::SeqCst);
                    loop {
                        e = chan.epoch.load(Ordering::SeqCst);
                        if e != last_e {
                            break;
                        }
                        g = chan.cv.wait(g).unwrap_or_else(|err| err.into_inner());
                    }
                    chan.sleepers.fetch_sub(1, Ordering::SeqCst);
                }
            }
            last_e = e;
            let slot = &pool.slots[e & (SLOTS - 1)];
            let readers = if is_e { &slot.e_readers.0 } else { &slot.p_readers.0 };
            readers.fetch_add(1, Ordering::SeqCst);
            if slot.epoch.load(Ordering::SeqCst) == e {
                let guard = AbortOnUnwind;
                run_slot(slot, is_e);
                drop(guard);
            }
            readers.fetch_sub(1, Ordering::Release);
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

    fn run_phased(
        counts: &[usize],
        p_split: &[usize],
        use_e: bool,
        body: &(dyn Fn(usize, usize) + Sync + '_),
        after: &(dyn Fn(usize) + Sync + '_),
    ) {
        assert!(counts.len() <= MAX_PHASES && counts.iter().all(|&c| c > 0));
        let max_chunks = counts.iter().copied().max().unwrap_or(0);
        let pool = get_pool();
        let avail_helpers = if use_e { pool.p_helpers + pool.e_helpers } else { pool.p_helpers };
        if max_chunks <= 1
            || avail_helpers == 0
            || pool.in_use.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err()
        {
            for (p, &c) in counts.iter().enumerate() {
                (0..c).for_each(|i| body(p, i));
                after(p);
            }
            return;
        }
        let guard = AbortOnUnwind;
        let next_e = pool.global_epoch.load(Ordering::Relaxed).wrapping_add(1);
        let e = if next_e == 0 { 1 } else { next_e };
        pool.global_epoch.store(e, Ordering::Relaxed);
        let slot = &pool.slots[e & (SLOTS - 1)];
        while slot.p_readers.0.load(Ordering::Acquire) != 0 || slot.e_readers.0.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        // SAFETY: lifetime erasure only; Dekker/RCU fence below guarantees all readers exit before return.
        let (body_ptr, after_ptr) = unsafe {
            (
                std::mem::transmute::<&(dyn Fn(usize, usize) + Sync + '_), &'static (dyn Fn(usize, usize) + Sync + 'static)>(body) as *const _,
                std::mem::transmute::<&(dyn Fn(usize) + Sync + '_), &'static (dyn Fn(usize) + Sync + 'static)>(after) as *const _,
            )
        };
        let wake_e = use_e && pool.e_helpers > 0;
        let wake_p = pool.p_helpers > 0;
        // SAFETY: `in_use == true` and both reader counts are 0, so caller has exclusive access to `slot`.
        unsafe {
            let s_mut = (slot as *const Slot) as *mut Slot;
            (*s_mut).phases = counts.len();
            for p in 0..counts.len() {
                let sp = if wake_e { p_split.get(p).copied().unwrap_or(counts[p]).min(counts[p]) } else { counts[p] };
                (*s_mut).counts[p] = counts[p];
                (*s_mut).p_split[p] = sp;
                (*s_mut).p_next.0[p].store(0, Ordering::Relaxed);
                (*s_mut).e_next.0[p].store(sp, Ordering::Relaxed);
                (*s_mut).done.0[p].store(0, Ordering::Relaxed);
            }
            *(*s_mut).body.get() = body_ptr;
            *(*s_mut).after.get() = after_ptr;
        }
        slot.ready.0.store(0, Ordering::Relaxed);
        slot.epoch.store(e, Ordering::Release);
        // Wake E-cores first (higher cross-cluster latency), then P-cores; one SeqCst Dekker fence.
        if wake_e {
            pool.e_chan.epoch.store(e, Ordering::Release);
        }
        if wake_p {
            pool.p_chan.epoch.store(e, Ordering::Release);
        }
        std::sync::atomic::fence(Ordering::SeqCst);
        if wake_e {
            pool.e_chan.notify_if_sleeping();
        }
        if wake_p {
            pool.p_chan.notify_if_sleeping();
        }
        run_slot(slot, false);
        wait_until(|| slot.ready.0.load(Ordering::Acquire) == counts.len());
        // Dekker/RCU fence: close slot epoch and wait for any late reader to exit before ending borrows.
        slot.epoch.store(0, Ordering::SeqCst);
        while slot.p_readers.0.load(Ordering::SeqCst) != 0 {
            std::hint::spin_loop();
        }
        if wake_e {
            while slot.e_readers.0.load(Ordering::SeqCst) != 0 {
                std::hint::spin_loop();
            }
        }
        pool.in_use.store(false, Ordering::Release);
        drop(guard);
    }

    /// P-core-only phased execution (optimal for barrier-heavy kernels like `depth_sort` and `to_compact_bytes`).
    pub fn phased(counts: &[usize], body: &(dyn Fn(usize, usize) + Sync + '_), after: &(dyn Fn(usize) + Sync + '_)) {
        run_phased(counts, counts, false, body, after);
    }

    /// Single-phase P+E fan-out: `body(i)` once for every `i < n`.
    pub fn for_each_pe(n: usize, p_split: usize, body: &(dyn Fn(usize) + Sync + '_)) {
        if n > 0 {
            run_phased(&[n], &[p_split], true, &|_, i| body(i), &|_| {});
        }
    }
}

/// O(N) 10-bit counting sort (1024 L1-resident buckets, zero comparisons) returning splat indices
/// ordered back-to-front from `cam_pos` along `cam_forward`.
///
/// Three streaming passes and two allocations: (1) 8-lane projection + min/max into a u32 buffer
/// that is recycled as the output, (2) fused key quantisation + 4-way histogram, (3) 4-cursor
/// scatter. With `threads`: from `CODEC_PAR_MIN` splats the memory-bound projection fans out
/// (`sort_depths_mid`, L1-sized chunks, compact u16 histograms), and from `PAR_MIN` all three passes run as `dynpar` phases with a
/// direct parallel scatter (`sort_depths_phased`); fully deterministic either way.
pub fn sort_splats_by_camera(cloud: &SplatCloud, cam_pos: [f32; 3], cam_forward: [f32; 3]) -> Vec<u32> {
    let gs = &cloud.gaussians[..];
    let n = gs.len();
    if n == 0 {
        return Vec::new();
    }
    let fwd = Vec3::from_array(cam_forward).normalize_or_zero();
    let f = fwd.to_array();
    let base = cam_pos[0] * f[0] + cam_pos[1] * f[1] + cam_pos[2] * f[2];

    #[cfg(feature = "threads")]
    if n >= CODEC_PAR_MIN && rayon::current_num_threads() > 1 {
        return if n >= PAR_MIN {
            sort_depths_phased(gs, f, base)
        } else {
            sort_depths_mid(gs, f, base)
        };
    }

    // SAFETY: `project_depths` writes every element of `buf`; `depth_keys_hist` writes every
    // element of `keys`; `scatter_hist4` overwrites every element of `buf`.
    let mut buf = unsafe { uninit_vec::<u32>(n) };
    let (lo, hi) = project_depths(gs, &mut buf, f, base);
    let inv_span = SORT_MAX_B / (hi - lo).max(1e-5);
    let mut keys = unsafe { uninit_vec::<u16>(n) };
    let mut hist: Hist4 = [[0u32; 4]; SORT_BUCKETS];
    depth_keys_hist(&buf, &mut keys, lo, inv_span, &mut hist);
    prefix_hist4(&mut hist);
    scatter_hist4(&keys, &mut hist, &mut buf, 0);
    buf
}

/// Thread-local reusable scratch buffers for parallel depth sorts (eliminates per-frame `malloc`
/// + page-fault overhead for temporary `keys`, per-chunk histograms, and totals).
#[cfg(feature = "threads")]
#[derive(Default)]
struct SortScratch {
    keys: Vec<u16>,
    hists32: Vec<Hist4>,
    tots32: Vec<[u32; SORT_BUCKETS]>,
    hists16: Vec<[[u16; 4]; SORT_BUCKETS]>,
    tots16: Vec<[u16; SORT_BUCKETS]>,
}

#[cfg(feature = "threads")]
thread_local! {
    static SORT_SCRATCH: std::cell::RefCell<SortScratch> = std::cell::RefCell::new(SortScratch::default());
}

#[cfg(feature = "threads")]
#[inline(always)]
unsafe fn ensure_uninit_len<T: Copy>(v: &mut Vec<T>, len: usize) {
    if v.capacity() < len {
        v.clear();
        v.reserve_exact(len);
    }
    // SAFETY: `T: Copy` has no drop glue; caller initializes `v[..len]` before reading.
    unsafe { v.set_len(len) };
}

/// The same three passes as the serial sort, as three `dynpar` phases over ~2 chunks per pool
/// thread (the calling thread starts immediately; late / slow workers only take what is left):
/// (0) projection + per-chunk min/max -> `after`: global range; (1) keys + per-chunk 4-lane
/// histogram -> `after`: global per-bucket base offsets; (2) each chunk derives its cursors
/// (bucket base + earlier chunks' counts + lane prefix) and scatters straight into the shared
/// output (disjoint slots). Order within a bucket is (chunk, lane, index): fully deterministic.
#[cfg(feature = "threads")]
fn sort_depths_phased(gs: &[Gaussian], f: [f32; 3], base: f32) -> Vec<u32> {
    const MAX_NC: usize = 64;
    let n = gs.len();
    let threads = dynpar::p_threads().max(2);
    let ch = ((n / (threads * 2)).max(2048) + 7) & !7;
    let ch = ch.max((n.div_ceil(MAX_NC) + 7) & !7);
    let nc = n.div_ceil(ch);
    let range = |c: usize| (c * ch, (c * ch + ch).min(n));
    // SAFETY: phase 0 writes `buf[..n]`; phase 2 overwrites `buf[..n]` with the sorted permutation.
    let mut buf = unsafe { uninit_vec::<u32>(n) };
    let mut mm = [(0.0f32, 0.0f32); MAX_NC];
    let mut bases = [0u32; SORT_BUCKETS];
    let mut lo_inv = (0.0f32, 0.0f32);
    SORT_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        // SAFETY: phase 1 writes all `keys[..n]`, `hists32[..nc]`, `tots32[..nc]` before phase 2 reads them.
        unsafe {
            ensure_uninit_len(&mut sc.keys, n);
            ensure_uninit_len(&mut sc.hists32, nc);
            ensure_uninit_len(&mut sc.tots32, nc);
        }
        let (bp, kp, mp, hp, tp, basep, lp) = (
            SyncPtr(buf.as_mut_ptr()),
            SyncPtr(sc.keys.as_mut_ptr()),
            SyncPtr(mm.as_mut_ptr()),
            SyncPtr(sc.hists32.as_mut_ptr()),
            SyncPtr(sc.tots32.as_mut_ptr()),
            SyncPtr(bases.as_mut_ptr()),
            SyncPtr(&mut lo_inv as *mut (f32, f32)),
        );
        let body = |phase: usize, c: usize| {
            let (s, e) = range(c);
            // SAFETY: chunk `c` exclusively owns `[s, e)` of `buf` / `keys` and slot `c` of `mm` /
            // `hists` / `tots`; shared reads happen only after the phase that wrote them completed.
            unsafe {
                match phase {
                    0 => {
                        let d = std::slice::from_raw_parts_mut(bp.get().add(s), e - s);
                        *mp.get().add(c) = project_depths(&gs[s..e], d, f, base);
                    }
                    1 => {
                        let (lo, inv) = *lp.get();
                        let h = &mut *hp.get().add(c);
                        let t = &mut *tp.get().add(c);
                        let d = std::slice::from_raw_parts(bp.get().add(s), e - s);
                        let k = std::slice::from_raw_parts_mut(kp.get().add(s), e - s);
                        depth_keys_hist_tot(d, k, lo, inv, h, t);
                    }
                    _ => {
                        let ts = std::slice::from_raw_parts(tp.get(), c);
                        let mut acc = [0u32; SORT_BUCKETS];
                        acc.copy_from_slice(std::slice::from_raw_parts(basep.get(), SORT_BUCKETS));
                        for t in ts {
                            for (a, &x) in acc.iter_mut().zip(t.iter()) {
                                *a += x;
                            }
                        }
                        let h = &*hp.get().add(c);
                        let mut cur: Hist4 = std::array::from_fn(|b| {
                            let (a, x) = (acc[b], h[b]);
                            [a, a + x[0], a + x[0] + x[1], a + x[0] + x[1] + x[2]]
                        });
                        let k = std::slice::from_raw_parts(kp.get().add(s), e - s);
                        scatter_hist4_raw(k, &mut cur, bp.get(), s as u32);
                    }
                }
            }
        };
        let after = |phase: usize| {
            // SAFETY: runs once, after every chunk of `phase` completed and before the next opens.
            unsafe {
                match phase {
                    0 => {
                        let mm = std::slice::from_raw_parts(mp.get(), nc);
                        let (lo, hi) = mm.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |a, b| (a.0.min(b.0), a.1.max(b.1)));
                        *lp.get() = (lo, SORT_MAX_B / (hi - lo).max(1e-5));
                    }
                    1 => {
                        let ts = std::slice::from_raw_parts(tp.get(), nc);
                        let mut tot = [0u32; SORT_BUCKETS];
                        for t in ts {
                            for (a, &x) in tot.iter_mut().zip(t.iter()) {
                                *a += x;
                            }
                        }
                        let b = std::slice::from_raw_parts_mut(basep.get(), SORT_BUCKETS);
                        let mut sum = 0u32;
                        for (o, t) in b.iter_mut().zip(tot) {
                            *o = sum;
                            sum += t;
                        }
                    }
                    _ => {}
                }
            }
        };
        dynpar::phased(&[nc, nc, nc], &body, &after);
    });
    buf
}

/// Mid-size clouds (`CODEC_PAR_MIN..PAR_MIN`): all three passes as `dynpar` phases over ~8
/// chunks, sized for L1: (0) projection + per-chunk min/max -> `after`: global range; (1) keys
/// + a compact u16 4-lane histogram and per-bucket totals per chunk (6 KB, chunks < 65536) ->
/// `after`: global bucket bases from the totals (1024 x nc u16 adds + one 1024-entry prefix);
/// (2) each chunk adds its predecessors' totals (2 KB each) to the bases, derives its lane
/// cursors and scatters straight into the shared output. Order within a bucket is (chunk,
/// lane, index): identical to `sort_depths_phased`, fully deterministic.
#[cfg(feature = "threads")]
fn sort_depths_mid(gs: &[Gaussian], f: [f32; 3], base: f32) -> Vec<u32> {
    const MAX_NC: usize = 32;
    let n = gs.len();
    let ch = ((n.div_ceil(8)).max(1024) + 7) & !7;
    let ch = ch.max(n.div_ceil(MAX_NC)).min(65_528);
    let ch = (ch + 7) & !7;
    let nc = n.div_ceil(ch);
    debug_assert!(nc <= MAX_NC);
    let range = |c: usize| (c * ch, (c * ch + ch).min(n));
    // SAFETY: phase 0 writes `buf[..n]`; phase 2 overwrites `buf[..n]` once.
    let mut buf = unsafe { uninit_vec::<u32>(n) };
    let mut mm = [(0.0f32, 0.0f32); MAX_NC];
    let mut bases = [0u32; SORT_BUCKETS];
    let mut lo_inv = (0.0f32, 0.0f32);
    SORT_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        // SAFETY: phase 1 writes all `keys[..n]`, `hists16[..nc]`, `tots16[..nc]` before phase 2 reads them.
        unsafe {
            ensure_uninit_len(&mut sc.keys, n);
            ensure_uninit_len(&mut sc.hists16, nc);
            ensure_uninit_len(&mut sc.tots16, nc);
        }
        let (bp, kp, hp, tp, mp, basep, lp) = (
            SyncPtr(buf.as_mut_ptr()),
            SyncPtr(sc.keys.as_mut_ptr()),
            SyncPtr(sc.hists16.as_mut_ptr()),
            SyncPtr(sc.tots16.as_mut_ptr()),
            SyncPtr(mm.as_mut_ptr()),
            SyncPtr(&mut bases as *mut [u32; SORT_BUCKETS]),
            SyncPtr(&mut lo_inv as *mut (f32, f32)),
        );
        let body = |phase: usize, c: usize| {
            let (s, e) = range(c);
            // SAFETY: chunk `c` exclusively owns `[s, e)` of `buf` / `keys` and slot `c` of `mm` /
            // `hists` / `tots`; shared reads happen only after the phase that wrote them completed.
            unsafe {
                match phase {
                    0 => {
                        let d = std::slice::from_raw_parts_mut(bp.get().add(s), e - s);
                        *mp.get().add(c) = project_depths(&gs[s..e], d, f, base);
                    }
                    1 => {
                        let (lo, inv) = *lp.get();
                        let d = std::slice::from_raw_parts(bp.get().add(s), e - s);
                        let k = std::slice::from_raw_parts_mut(kp.get().add(s), e - s);
                        depth_keys(d, k, lo, inv);
                        count_keys16(k, &mut *hp.get().add(c), &mut *tp.get().add(c));
                    }
                    _ => {
                        let mut acc = *basep.get();
                        for t in std::slice::from_raw_parts(tp.get(), c) {
                            for (a, &x) in acc.iter_mut().zip(t.iter()) {
                                *a += x as u32;
                            }
                        }
                        let h = &*hp.get().add(c);
                        let mut cur: Hist4 = std::array::from_fn(|b| {
                            let (a, x) = (acc[b], h[b]);
                            let (x0, x1, x2) = (x[0] as u32, x[1] as u32, x[2] as u32);
                            [a, a + x0, a + x0 + x1, a + x0 + x1 + x2]
                        });
                        let k = std::slice::from_raw_parts(kp.get().add(s), e - s);
                        scatter_hist4_raw(k, &mut cur, bp.get(), s as u32);
                    }
                }
            }
        };
        let after = |phase: usize| {
            // SAFETY: runs once, after every chunk of `phase` completed and before the next opens.
            unsafe {
                match phase {
                    0 => {
                        let mm = std::slice::from_raw_parts(mp.get(), nc);
                        let (lo, hi) = mm.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |a, b| (a.0.min(b.0), a.1.max(b.1)));
                        *lp.get() = (lo, SORT_MAX_B / (hi - lo).max(1e-5));
                    }
                    1 => {
                        let mut tot = [0u32; SORT_BUCKETS];
                        for t in std::slice::from_raw_parts(tp.get(), nc) {
                            for (a, &x) in tot.iter_mut().zip(t.iter()) {
                                *a += x as u32;
                            }
                        }
                        let b = &mut *basep.get();
                        let mut sum = 0u32;
                        for (o, t) in b.iter_mut().zip(tot) {
                            *o = sum;
                            sum += t;
                        }
                    }
                    _ => {}
                }
            }
        };
        dynpar::phased(&[nc, nc, nc], &body, &after);
    });
    buf
}

/// Pass 2 map only: depth -> inverted 10-bit bucket key (far = 0).
#[cfg(feature = "threads")]
#[inline(always)]
fn depth_keys(depth: &[u32], keys: &mut [u16], lo: f32, inv_span: f32) {
    let n = depth.len().min(keys.len());
    #[cfg(target_arch = "aarch64")]
    let done = {
        use core::arch::aarch64::*;
        let chunks = n / 8;
        // SAFETY: NEON baseline; `chunks * 8 <= n`.
        unsafe {
            let lo_v = vdupq_n_f32(lo);
            let inv_v = vdupq_n_f32(inv_span);
            let zero_v = vdupq_n_f32(0.0);
            let max_v = vdupq_n_f32(SORT_MAX_B);
            let mask_v = vdupq_n_u32(SORT_MASK as u32);
            let mut dp = depth.as_ptr().cast::<f32>();
            let mut kp = keys.as_mut_ptr();
            for _ in 0..chunks {
                let d0 = vld1q_f32(dp);
                let d1 = vld1q_f32(dp.add(4));
                let f0 = vminnmq_f32(vmaxnmq_f32(vmulq_f32(vsubq_f32(d0, lo_v), inv_v), zero_v), max_v);
                let f1 = vminnmq_f32(vmaxnmq_f32(vmulq_f32(vsubq_f32(d1, lo_v), inv_v), zero_v), max_v);
                let k0 = vsubq_u32(mask_v, vcvtq_u32_f32(f0));
                let k1 = vsubq_u32(mask_v, vcvtq_u32_f32(f1));
                vst1q_u16(kp, vcombine_u16(vmovn_u32(k0), vmovn_u32(k1)));
                dp = dp.add(8);
                kp = kp.add(8);
            }
        }
        chunks * 8
    };
    #[cfg(not(target_arch = "aarch64"))]
    let done = 0;
    for (&d, k) in depth[done..n].iter().zip(&mut keys[done..n]) {
        let b = ((f32::from_bits(d) - lo) * inv_span).max(0.0).min(SORT_MAX_B) as u32;
        *k = (SORT_MASK as u32 - b) as u16;
    }
}

/// Counts `keys` (< 65536 of them) straight into a u16 4-lane histogram (lane = index & 3,
/// 8 KB: no scratch zero-fill / fold) and per-bucket totals. Consecutive keys hit distinct
/// lanes, so a run of equal keys never serialises within a 4-key step.
#[cfg(feature = "threads")]
#[inline(always)]
fn count_keys16(keys: &[u16], hist: &mut [[u16; 4]; SORT_BUCKETS], tot: &mut [u16; SORT_BUCKETS]) {
    debug_assert!(keys.len() < 65_536);
    *hist = [[0u16; 4]; SORT_BUCKETS];
    let mut kc = keys.chunks_exact(4);
    for k in &mut kc {
        for l in 0..4 {
            let c = &mut hist[k[l] as usize & SORT_MASK][l];
            *c = c.wrapping_add(1);
        }
    }
    for (j, &k) in kc.remainder().iter().enumerate() {
        let c = &mut hist[k as usize & SORT_MASK][j];
        *c = c.wrapping_add(1);
    }
    for (t, h) in tot.iter_mut().zip(hist.iter()) {
        *t = h[0].wrapping_add(h[1]).wrapping_add(h[2]).wrapping_add(h[3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn sample_rgba(size: u32) -> Vec<u8> {
        let mut rgba = vec![0u8; (size * size * 4) as usize];
        let c = size as f32 * 0.5;
        let r = size as f32 * 0.35;
        for y in 0..size {
            for x in 0..size {
                let i = ((y * size + x) * 4) as usize;
                let dx = x as f32 - c;
                let dy = y as f32 - c;
                if dx * dx + dy * dy <= r * r {
                    rgba[i] = 40;
                    rgba[i + 1] = 160;
                    rgba[i + 2] = 220;
                    rgba[i + 3] = 255;
                } else {
                    rgba[i] = 245;
                    rgba[i + 1] = 245;
                    rgba[i + 2] = 248;
                    rgba[i + 3] = 255;
                }
            }
        }
        rgba
    }

    #[test]
    fn splat_and_ply_roundtrip_preserves_gaussians() {
        let cloud = SplatCloud {
            gaussians: vec![
                Gaussian { pos: [0.1, 0.2, -0.3], scale: [0.02, 0.02, 0.008], rot: [1.0, 0.0, 0.0, 0.0], color: [0.8, 0.3, 0.2], opacity: 0.9 },
                Gaussian { pos: [-0.4, 0.5, 0.6], scale: [0.03, 0.015, 0.01], rot: [0.7071, 0.0, 0.7071, 0.0], color: [0.1, 0.9, 0.5], opacity: 0.6 },
            ],
        };
        let splat_bytes = cloud.to_splat_bytes();
        assert_eq!(splat_bytes.len(), 64);
        let decoded_splat = SplatCloud::from_splat_bytes(&splat_bytes).unwrap();
        assert_eq!(decoded_splat.len(), 2);
        assert!((decoded_splat.gaussians[0].pos[0] - 0.1).abs() < 1e-5);
        assert!((decoded_splat.gaussians[1].color[1] - 0.9).abs() < 0.01);

        let ply_bytes = cloud.to_ply_bytes();
        let decoded_ply = SplatCloud::from_ply_bytes(&ply_bytes).unwrap();
        assert_eq!(decoded_ply.len(), 2);
        assert!((decoded_ply.gaussians[0].pos[2] - (-0.3)).abs() < 1e-5);
        assert!((decoded_ply.gaussians[0].color[0] - 0.8).abs() < 1e-3);
        assert!((decoded_ply.gaussians[0].opacity - 0.9).abs() < 1e-3);
        assert!((decoded_ply.gaussians[1].scale[0] - 0.03).abs() < 1e-4);

        let compact_bytes = cloud.to_compact_bytes();
        assert_eq!(compact_bytes.len(), COMPACT_HEADER_LEN + 2 * 12);
        let decoded_compact = SplatCloud::from_compact_bytes(&compact_bytes).unwrap();
        assert_eq!(decoded_compact.len(), 2);
        assert!((decoded_compact.gaussians[0].pos[0] - 0.1).abs() < 2e-3);
        assert!((decoded_compact.gaussians[1].pos[2] - 0.6).abs() < 2e-3);
        assert!((decoded_compact.gaussians[1].scale[0] - 0.03).abs() < 5e-3);
        assert!((decoded_compact.gaussians[1].color[1] - 0.9).abs() < 0.02);
        // Identity and 90-degree quaternions survive 5-bit smallest-3 exactly.
        assert_eq!(decoded_compact.gaussians[0].rot, [1.0, 0.0, 0.0, 0.0]);
        let r1 = decoded_compact.gaussians[1].rot;
        assert!((r1[0] - 0.7071).abs() < 1e-3 && (r1[2] - 0.7071).abs() < 1e-3 && r1[1].abs() < 1e-6 && r1[3].abs() < 1e-6);
    }

    fn quat_dot(a: [f32; 4], b: [f32; 4]) -> f32 {
        let la = (a.iter().map(|v| v * v).sum::<f32>()).sqrt();
        let lb = (b.iter().map(|v| v * v).sum::<f32>()).sqrt();
        (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]).abs() / (la * lb)
    }

    #[test]
    fn compact_v2_is_exactly_12_bytes_per_splat_and_roundtrips() {
        for &n in &[1usize, 3, 4, 1000, 20_001] {
            let cloud = bench_cloud(n);
            let bytes = cloud.to_compact_bytes();
            assert_eq!(bytes.len(), COMPACT_HEADER_LEN + n * 12, "v2 must be exactly 12 B/splat");
            assert_eq!(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 2);
            let dec = SplatCloud::from_compact_bytes(&bytes).unwrap();
            assert_eq!(dec.len(), n);
            let (mut ep, mut ec, mut es, mut eo, mut dq) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32);
            for (a, b) in cloud.gaussians.iter().zip(&dec.gaussians) {
                for k in 0..3 {
                    ep = ep.max((a.pos[k] - b.pos[k]).abs());
                    ec = ec.max((a.color[k] - b.color[k]).abs());
                    es = es.max((a.scale[k] - b.scale[k]).abs());
                }
                eo = eo.max((a.opacity - b.opacity).abs());
                dq = dq.min(quat_dot(a.rot, b.rot));
                let l2: f32 = b.rot.iter().map(|v| v * v).sum();
                assert!((l2 - 1.0).abs() < 1e-4, "decoded quaternion must be unit, got |q|^2={l2}");
            }
            assert!(ep < 2e-3, "n={n} pos err {ep}");
            assert!(ec < 0.02, "n={n} color err {ec}");
            assert!(eo < 0.02, "n={n} opacity err {eo}");
            assert!(es < 5e-3, "n={n} scale err {es}");
            assert!(dq > 0.995, "n={n} min |dot(q, q')| {dq}");
        }
        // Empty cloud: header only, decodes to empty.
        let empty = SplatCloud::default().to_compact_bytes();
        assert_eq!(empty.len(), COMPACT_HEADER_LEN);
        assert!(SplatCloud::from_compact_bytes(&empty).unwrap().is_empty());
        // Truncation and bad versions are rejected.
        let bytes = bench_cloud(10).to_compact_bytes();
        assert!(SplatCloud::from_compact_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut bad = bytes.clone();
        bad[4] = 7;
        assert!(SplatCloud::from_compact_bytes(&bad).is_err());
        // 12 B/splat vs public formats: <= 50% of .splat (32 B), <= 18% of minimal .ply (68 B).
        assert!(12 * 2 <= 32 && 12 * 5 < 68);
    }

    /// Reference encoder for the legacy 16-byte `.dsplat` v1 layout (writer removed in v2).
    fn encode_compact_v1(cloud: &SplatCloud) -> Vec<u8> {
        let (lo, hi) = cloud.bounds();
        let (mut s_min, mut s_max) = (f32::INFINITY, f32::NEG_INFINITY);
        for g in &cloud.gaussians {
            for &s in &g.scale {
                s_min = s_min.min(s.max(1e-5).ln());
                s_max = s_max.max(s.max(1e-5).ln());
            }
        }
        let inv_s = 255.0 / (s_max - s_min).max(1e-4);
        let mut out = vec![0u8; COMPACT_HEADER_LEN + cloud.len() * 16];
        out[0..4].copy_from_slice(COMPACT_MAGIC);
        out[4..8].copy_from_slice(&1u32.to_le_bytes());
        out[8..12].copy_from_slice(&(cloud.len() as u32).to_le_bytes());
        for i in 0..3 {
            out[12 + i * 4..16 + i * 4].copy_from_slice(&lo[i].to_le_bytes());
            out[24 + i * 4..28 + i * 4].copy_from_slice(&hi[i].to_le_bytes());
        }
        out[36..40].copy_from_slice(&s_min.to_le_bytes());
        out[40..44].copy_from_slice(&s_max.to_le_bytes());
        for (g, d) in cloud.gaussians.iter().zip(out[COMPACT_HEADER_LEN..].chunks_exact_mut(16)) {
            for k in 0..3 {
                let q = ((g.pos[k] - lo[k]) * 65535.0 / (hi[k] - lo[k]).max(1e-5)).clamp(0.0, 65535.0) as u16;
                d[k * 2..k * 2 + 2].copy_from_slice(&q.to_le_bytes());
                d[6 + k] = ((g.scale[k].max(1e-5).ln() - s_min) * inv_s).clamp(0.0, 255.0) as u8;
                d[9 + k] = (g.color[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
            d[12] = (g.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let r = g.rot;
            let mut mi = 0;
            for j in 1..4 {
                if r[j].abs() > r[mi].abs() {
                    mi = j;
                }
            }
            let sign = if r[mi] < 0.0 { -1.0 } else { 1.0 };
            let mut rem = [0u32; 3];
            let mut k = 0;
            for j in 0..4 {
                if j != mi {
                    let v = (r[j] * sign * std::f32::consts::SQRT_2).clamp(-1.0, 1.0);
                    rem[k] = ((v * 63.0 + 63.5).clamp(0.0, 127.0) as u32) & 0x7F;
                    k += 1;
                }
            }
            let packed = ((mi as u32) << 21) | (rem[0] << 14) | (rem[1] << 7) | rem[2];
            d[13..16].copy_from_slice(&packed.to_le_bytes()[..3]);
        }
        out
    }

    #[test]
    fn compact_decoder_reads_legacy_v1_files() {
        let cloud = bench_cloud(5_000);
        let v1 = encode_compact_v1(&cloud);
        assert_eq!(v1.len(), COMPACT_HEADER_LEN + cloud.len() * 16);
        let dec = SplatCloud::from_compact_bytes(&v1).unwrap();
        assert_eq!(dec.len(), cloud.len());
        for (a, b) in cloud.gaussians.iter().zip(&dec.gaussians) {
            for k in 0..3 {
                assert!((a.pos[k] - b.pos[k]).abs() < 1e-4);
                assert!((a.color[k] - b.color[k]).abs() < 0.005);
                assert!((a.scale[k] - b.scale[k]).abs() < 1e-3);
            }
            assert!(quat_dot(a.rot, b.rot) > 0.999);
        }
        // v1 -> decode -> v2 re-encode stays within v2 tolerances.
        let v2 = dec.to_compact_bytes();
        assert_eq!(v2.len(), COMPACT_HEADER_LEN + cloud.len() * 12);
    }

    #[test]
    fn fast_exp_matches_libm_within_2e6_relative() {
        let mut worst = 0.0f64;
        let mut x = -87.0f32;
        while x <= 88.0 {
            let exact = (x as f64).exp();
            let rel = ((fast_exp(x) as f64 - exact) / exact).abs();
            worst = worst.max(rel);
            x += 0.000_731;
        }
        assert!(worst < 2e-6, "fast_exp worst relative error {worst:e}");
        assert_eq!(fast_exp(0.0), 1.0);
        assert!(fast_exp(-1000.0) > 0.0 && fast_exp(-1000.0) < 1e-37);
        assert!(fast_exp(1000.0).is_finite() && fast_exp(1000.0) > 1e38);
        for &v in &[-20.0f32, -3.0, -0.5, 0.0, 0.5, 3.0, 20.0] {
            let exact = 1.0 / (1.0 + (-(v as f64)).exp());
            assert!((sigmoid(v) as f64 - exact).abs() < 1e-6);
        }
    }

    #[test]
    fn quat_from_normal_maps_z_onto_normal() {
        let cloud = bench_cloud(2_000);
        let mut normals: Vec<Vec3> = cloud.gaussians.iter().map(|g| Vec3::from_array(g.pos) - Vec3::new(0.0, 1.25, 0.0)).collect();
        normals.extend([Vec3::Z, -Vec3::Z, Vec3::X, -Vec3::Y, Vec3::new(1e-4, 0.0, -1.0), Vec3::new(0.0, 0.0, -3.0)]);
        for n in normals {
            let [w, x, y, z] = quat_from_normal(n);
            let q = Quat::from_xyzw(x, y, z, w);
            assert!((q.length() - 1.0).abs() < 1e-5);
            let got = q * Vec3::Z;
            assert!((got - n.normalize()).length() < 2e-3, "n={n:?} got={got:?}");
            if n.normalize().z > -0.99 {
                let r = Quat::from_rotation_arc(Vec3::Z, n.normalize());
                assert!(quat_dot([w, x, y, z], [r.w, r.x, r.y, r.z]) > 0.99999, "must be the shortest arc");
            }
        }
        assert_eq!(quat_from_normal(Vec3::ZERO), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(quat_from_normal(Vec3::splat(f32::NAN)), [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn unrolled_counting_sort_is_a_back_to_front_permutation() {
        for &n in &[1usize, 2, 5, 7, 4096, 30_003] {
            let cloud = bench_cloud(n);
            let cam = [0.3, 1.2, 4.0];
            let fwd = Vec3::new(0.1, -0.2, -1.0).normalize();
            let order = sort_splats_by_camera(&cloud, cam, fwd.to_array());
            assert_eq!(order.len(), n);
            let mut seen = vec![false; n];
            for &i in &order {
                assert!(!seen[i as usize]);
                seen[i as usize] = true;
            }
            let depth = |i: u32| (Vec3::from_array(cloud.gaussians[i as usize].pos) - Vec3::from_array(cam)).dot(fwd);
            let (lo, hi) = order.iter().fold((f32::MAX, f32::MIN), |(l, h), &i| (l.min(depth(i)), h.max(depth(i))));
            let bucket = (hi - lo).max(1e-5) / 1023.0;
            for w in order.windows(2) {
                assert!(depth(w[0]) + bucket * 1.01 >= depth(w[1]), "not back-to-front");
            }
        }
    }

    #[test]
    fn edit_bitmask_fast_paths_match_semantics() {
        let cloud = bench_cloud(3_000);
        // Single named colour (FX_TINT const fast path): every splat touched, pulled toward red.
        let red = edit_splat_by_prompt(&cloud, "make it red");
        assert_eq!(red.modified_count, cloud.len());
        let mean = |c: &SplatCloud, k: usize| c.gaussians.iter().map(|g| g.color[k]).sum::<f32>() / c.len() as f32;
        assert!(mean(&red.cloud, 0) > mean(&red.cloud, 1) + 0.15);
        assert_eq!(red.cloud.gaussians[7].pos, cloud.gaussians[7].pos, "colour-only edit must not move splats");
        // Unknown words: zero-mask clause skips the kernel entirely.
        let none = edit_splat_by_prompt(&cloud, "make it zorbly");
        assert_eq!(none.modified_count, 0);
        assert_eq!(none.cloud.gaussians[11].color, cloud.gaussians[11].color);
        // Multi-effect clause (generic bit loop) applies colour + geometry in order.
        let combo = edit_splat_by_prompt(&cloud, "sunset voxel");
        assert_eq!(combo.modified_count, cloud.len());
        assert!(combo.cloud.gaussians.iter().all(|g| g.rot == [1.0, 0.0, 0.0, 0.0] && g.scale[0] == g.scale[2]));
        // Noir grades to grayscale; region-limited edits touch a subset.
        let noir = edit_splat_by_prompt(&cloud, "make it noir");
        assert!(noir.cloud.gaussians.iter().all(|g| (g.color[0] - g.color[1]).abs() < 0.06));
        let floor = edit_splat_by_prompt(&cloud, "make the floor gold");
        assert!(floor.modified_count > 0 && floor.modified_count < cloud.len());
    }

    #[test]
    fn image_to_splat_item_and_space_both_generate_valid_clouds() {
        let rgba = sample_rgba(64);
        let item = image_to_splat(64, 64, &rgba, None, false, 4000).unwrap();
        assert!(item.len() > 200 && item.len() <= 4000);
        let space = image_to_splat(64, 64, &rgba, None, true, 4000).unwrap();
        assert!(space.len() > 200 && space.len() <= 4000);
        let (lo, hi) = space.bounds();
        assert!(hi[0] - lo[0] > 1.5, "space splat should span room scale");
    }

    #[test]
    fn mesh_to_splat_room_and_prompt_edits_work() {
        // Simple room floor + wall triangles
        let positions: Vec<f32> = vec![
            -2.0, 0.0, -2.0,  2.0, 0.0, -2.0,  2.0, 0.0,  2.0, -2.0, 0.0,  2.0, // floor
            -2.0, 0.0, -2.0,  2.0, 0.0, -2.0,  2.0, 2.5, -2.0, -2.0, 2.5, -2.0, // back wall
        ];
        let indices: Vec<u32> = vec![0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7];
        let room = mesh_to_splat(&positions, &indices, None, [0.8, 0.8, 0.8, 1.0], true, 2000).unwrap();
        assert_eq!(room.len(), 2000);

        let edited = edit_splat_by_prompt(&room, "make the floor red and the walls cyberpunk");
        assert!(edited.modified_count > 500);
        assert!(edited.summary.contains("floor red"));

        let pruned = edit_splat_by_prompt(&room, "remove the floor");
        assert!(pruned.total_count < room.len());
    }

    #[test]
    fn radix_sort_orders_back_to_front() {
        let cloud = SplatCloud {
            gaussians: vec![
                Gaussian { pos: [0.0, 0.0, -1.0], scale: [0.01; 3], rot: [1.0, 0.0, 0.0, 0.0], color: [1.0; 3], opacity: 1.0 },
                Gaussian { pos: [0.0, 0.0, -5.0], scale: [0.01; 3], rot: [1.0, 0.0, 0.0, 0.0], color: [1.0; 3], opacity: 1.0 },
                Gaussian { pos: [0.0, 0.0, -3.0], scale: [0.01; 3], rot: [1.0, 0.0, 0.0, 0.0], color: [1.0; 3], opacity: 1.0 },
            ],
        };
        let order = sort_splats_by_camera(&cloud, [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert_eq!(order, vec![1, 2, 0]);
    }

    fn bench_cloud(n: usize) -> SplatCloud {
        let mut s = 0x1234_5678u32;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32) / (u32::MAX as f32)
        };
        let gaussians = (0..n)
            .map(|_| {
                let q = Quat::from_euler(glam::EulerRot::XYZ, r() * 6.28, r() * 6.28, r() * 6.28);
                Gaussian {
                    pos: [r() * 4.0 - 2.0, r() * 2.5, r() * 4.0 - 2.0],
                    scale: [0.002 + r() * 0.05, 0.002 + r() * 0.05, 0.001 + r() * 0.01],
                    rot: [q.w, q.x, q.y, q.z],
                    color: [r(), r(), r()],
                    opacity: 0.05 + r() * 0.95,
                }
            })
            .collect();
        SplatCloud { gaussians }
    }

    /// `cargo test --release --lib splat::tests::bench_splat_all -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_splat_all() {
        use std::time::Instant;
        fn best(iters: usize, f: &mut dyn FnMut()) -> f64 {
            for _ in 0..3 {
                f();
            }
            let mut b = f64::MAX;
            for _ in 0..iters {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64() * 1e6);
            }
            b
        }
        {
            let cloud = bench_cloud(16_384);
            let gs = &cloud.gaussians[..];
            let f = [0.0, -0.196, -0.98];
            let mut buf = vec![0u32; gs.len()];
            let t_proj = best(2000, &mut || { std::hint::black_box(project_depths(gs, &mut buf, f, 0.1)); });
            let (lo, hi) = project_depths(gs, &mut buf, f, 0.1);
            let inv = SORT_MAX_B / (hi - lo);
            let mut keys = vec![0u16; gs.len()];
            let t_keys = best(2000, &mut || {
                let mut h: Hist4 = [[0; 4]; SORT_BUCKETS];
                depth_keys_hist(&buf, &mut keys, lo, inv, &mut h);
                std::hint::black_box(&h);
            });
            let mut h: Hist4 = [[0; 4]; SORT_BUCKETS];
            depth_keys_hist(&buf, &mut keys, lo, inv, &mut h);
            prefix_hist4(&mut h);
            let mut out = vec![0u32; gs.len()];
            let t_scat = best(2000, &mut || {
                let mut c = h;
                scatter_hist4(&keys, &mut c, &mut out, 0);
                std::hint::black_box(&out);
            });
            let t_alloc = best(2000, &mut || { std::hint::black_box((vec![0u32; gs.len()], vec![0u16; gs.len()])); });
            let img = image_to_splat(128, 128, &sample_rgba(128), None, true, 16_384).unwrap();
            let t_img = best(2000, &mut || { std::hint::black_box(sort_splats_by_camera(&img, [0.0, 1.5, 2.0], [0.0, 0.0, -1.0])); });
            println!("SORT_IMG16k n={} {t_img:.2}us", img.len());
            println!("SORTPHASES proj={t_proj:.2}us keys_hist={t_keys:.2}us scatter={t_scat:.2}us allocs={t_alloc:.2}us");
        }
        for &(n, iters) in &[(16_384usize, 400usize), (500_000, 25)] {
            let cloud = bench_cloud(n);
            let ply = cloud.to_ply_bytes();
            let spl = cloud.to_splat_bytes();
            let compact = cloud.to_compact_bytes();
            let t_sort = best(iters, &mut || { std::hint::black_box(sort_splats_by_camera(&cloud, [0.3, 1.2, 4.0], [0.0, -0.2, -1.0])); });
            let t_cenc = best(iters, &mut || { std::hint::black_box(cloud.to_compact_bytes()); });
            let t_cdec = best(iters, &mut || { std::hint::black_box(SplatCloud::from_compact_bytes(&compact).unwrap()); });
            let t_penc = best(iters, &mut || { std::hint::black_box(cloud.to_ply_bytes()); });
            let t_pdec = best(iters, &mut || { std::hint::black_box(SplatCloud::from_ply_bytes(&ply).unwrap()); });
            let t_senc = best(iters, &mut || { std::hint::black_box(cloud.to_splat_bytes()); });
            let t_sdec = best(iters, &mut || { std::hint::black_box(SplatCloud::from_splat_bytes(&spl).unwrap()); });
            let t_clone = best(iters, &mut || { std::hint::black_box(cloud.clone()); });
            let ed = |p: &str| best(iters, &mut || { let mut c = cloud.clone(); std::hint::black_box(edit_splat_in_place(&mut c, p)); }) - t_clone;
            const G0: Gaussian = Gaussian { pos: [0.0; 3], scale: [0.0; 3], rot: [0.0; 4], color: [0.0; 3], opacity: 0.0 };
            let t_fill = best(iters, &mut || { std::hint::black_box(vec![G0; n]); });
            println!("ALLOC_FILL n={n} vec![G0;n]={t_fill:.1}us clone={t_clone:.1}us");
            let t_edit = ed("make it red");
            let t_edit2 = ed("make the floor wood and the walls cyberpunk, warmer");
            if n < 100_000 {
                let t_scan = best(iters, &mut || {
                    let mut plo = Vec3::splat(f32::INFINITY);
                    let mut phi = Vec3::splat(f32::NEG_INFINITY);
                    let mut raw_min = f32::INFINITY;
                    let mut raw_max = 0.0f32;
                    for g in &cloud.gaussians {
                        let p = Vec3::from_array(g.pos);
                        if p.is_finite() {
                            plo = plo.min(p);
                            phi = phi.max(p);
                        }
                        for &s in &g.scale {
                            let sc = s.max(1e-5);
                            raw_min = if sc < raw_min { sc } else { raw_min };
                            raw_max = if sc > raw_max { sc } else { raw_max };
                        }
                    }
                    std::hint::black_box((plo, phi, raw_min, raw_max));
                });
                let t_bytes = best(iters, &mut || { std::hint::black_box(vec![0u8; 44 + n * 12]); });
                println!("CENC scan={t_scan:.1}us alloc={t_bytes:.1}us");
                println!(
                    "EDITPARTS floor_wood={:.1} walls_cyber={:.1} warmer={:.1} sunset_voxel={:.1}",
                    ed("make the floor wood"), ed("the walls cyberpunk"), ed("warmer"), ed("sunset voxel")
                );
            }
            println!(
                "BENCHALL n={n} sort={t_sort:.1}us c_enc={t_cenc:.1}us c_dec={t_cdec:.1}us ply_enc={t_penc:.1}us ply_dec={t_pdec:.1}us splat_enc={t_senc:.1}us splat_dec={t_sdec:.1}us edit_red={t_edit:.1}us edit_multi={t_edit2:.1}us"
            );
        }
        let rgba = sample_rgba(128);
        let t_img_space = best(200, &mut || { std::hint::black_box(image_to_splat(128, 128, &rgba, None, true, 16_384).unwrap()); });
        let t_img_item = best(200, &mut || { std::hint::black_box(image_to_splat(128, 128, &rgba, None, false, 8_192).unwrap()); });
        // Room: 20x20 grid floor + 4 walls.
        let mut pos = Vec::new();
        let mut idx = Vec::new();
        let g = 20usize;
        for face in 0..5 {
            let base = (pos.len() / 3) as u32;
            for j in 0..=g {
                for i in 0..=g {
                    let (u, v) = (i as f32 / g as f32 * 4.0 - 2.0, j as f32 / g as f32);
                    let p = match face {
                        0 => [u, 0.0, v * 4.0 - 2.0],
                        1 => [u, v * 2.5, -2.0],
                        2 => [u, v * 2.5, 2.0],
                        3 => [-2.0, v * 2.5, u],
                        _ => [2.0, v * 2.5, u],
                    };
                    pos.extend_from_slice(&p);
                }
            }
            for j in 0..g as u32 {
                for i in 0..g as u32 {
                    let a = base + j * (g as u32 + 1) + i;
                    let b = a + 1;
                    let c = a + g as u32 + 1;
                    idx.extend_from_slice(&[a, c, b, b, c, c + 1]);
                }
            }
        }
        let t_mesh = best(200, &mut || { std::hint::black_box(mesh_to_splat(&pos, &idx, None, [0.8, 0.8, 0.8, 1.0], true, 16_384).unwrap()); });
        println!("BENCHALL img_space_16k={t_img_space:.1}us img_item_8k={t_img_item:.1}us mesh_room_16k={t_mesh:.1}us");
    }

    #[test]
    #[ignore]
    fn bench_splat_hot_paths() {
        use std::time::Instant;
        let cloud = bench_cloud(500_000);
        let ply = cloud.to_ply_bytes();
        let compact = cloud.to_compact_bytes();
        let best = |f: &mut dyn FnMut()| {
            let mut b = f64::MAX;
            for _ in 0..7 {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64() * 1e3);
            }
            b
        };
        let t_ply = best(&mut || { std::hint::black_box(SplatCloud::from_ply_bytes(&ply).unwrap()); });
        let t_cenc = best(&mut || { std::hint::black_box(cloud.to_compact_bytes()); });
        let t_cdec = best(&mut || { std::hint::black_box(SplatCloud::from_compact_bytes(&compact).unwrap()); });
        let t_sort = best(&mut || { std::hint::black_box(sort_splats_by_camera(&cloud, [0.3, 1.2, 4.0], [0.0, -0.2, -1.0])); });
        let t_edit = best(&mut || {
            let mut c = cloud.clone();
            std::hint::black_box(edit_splat_in_place(&mut c, "make it red"));
        });
        let t_clone = best(&mut || { std::hint::black_box(cloud.clone()); });
        let t_q = best(&mut || {
            let mut acc = 0.0f32;
            for g in &cloud.gaussians {
                acc += quat_from_normal(Vec3::from_array(g.pos))[1];
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH n=500k ply_dec={t_ply:.2}ms compact_enc={t_cenc:.2}ms compact_dec={t_cdec:.2}ms sort={t_sort:.2}ms edit_red={:.2}ms quat={t_q:.2}ms compact_B/splat={:.2}",
            t_edit - t_clone,
            (compact.len() - COMPACT_HEADER_LEN) as f64 / cloud.len() as f64
        );
    }
}
