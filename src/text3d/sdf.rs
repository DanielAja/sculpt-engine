//! Signed distance evaluation of shape programs. Negative inside, positive outside, millimetres.
//!
//! Primitive and operator formulas follow Inigo Quilez's distance-function articles
//! (https://iquilezles.org/articles/distfunctions/, code under the MIT licence, (c) Inigo Quilez) and his
//! polynomial smooth minimum (https://iquilezles.org/articles/smin/). Written fresh in Rust here.

use super::program::{Node, Shape};
use glam::{EulerRot, Mat3, Quat, Vec2, Vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb { min: Vec3::splat(f32::INFINITY), max: Vec3::splat(f32::NEG_INFINITY) };
    pub fn new(min: Vec3, max: Vec3) -> Aabb {
        Aabb { min, max }
    }
    pub fn half(h: Vec3) -> Aabb {
        Aabb { min: -h, max: h }
    }
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }
    pub fn union(&self, o: &Aabb) -> Aabb {
        Aabb { min: self.min.min(o.min), max: self.max.max(o.max) }
    }
    pub fn intersect(&self, o: &Aabb) -> Aabb {
        Aabb { min: self.min.max(o.min), max: self.max.min(o.max) }
    }
    pub fn grow(&self, d: f32) -> Aabb {
        Aabb { min: self.min - Vec3::splat(d), max: self.max + Vec3::splat(d) }
    }
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    /// Distance from `p` to the box (0 inside).
    #[inline(always)]
    pub fn distance(&self, p: Vec3) -> f32 {
        self.distance_squared(p).sqrt()
    }
    /// Squared distance from `p` to the box (0 inside).
    #[inline(always)]
    pub fn distance_squared(&self, p: Vec3) -> f32 {
        (self.min - p).max(p - self.max).max(Vec3::ZERO).length_squared()
    }
    /// Squared distance between this box and the box `lo..hi` (0 when they overlap).
    #[inline(always)]
    fn distance_squared_box(&self, lo: Vec3, hi: Vec3) -> f32 {
        (self.min - hi).max(lo - self.max).max(Vec3::ZERO).length_squared()
    }
    fn corners(&self) -> [Vec3; 8] {
        let (a, b) = (self.min, self.max);
        [
            Vec3::new(a.x, a.y, a.z),
            Vec3::new(b.x, a.y, a.z),
            Vec3::new(a.x, b.y, a.z),
            Vec3::new(b.x, b.y, a.z),
            Vec3::new(a.x, a.y, b.z),
            Vec3::new(b.x, a.y, b.z),
            Vec3::new(a.x, b.y, b.z),
            Vec3::new(b.x, b.y, b.z),
        ]
    }
}

// ---------------------------------------------------------------------------
// 4-wide SIMD float vector: native AArch64 NEON, portable fallback elsewhere.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod simd4 {
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub(crate) struct F32x4(pub(crate) float32x4_t);

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub(crate) struct M32x4(pub(crate) uint32x4_t);

    impl M32x4 {
        #[inline(always)]
        pub(crate) fn and(self, o: Self) -> Self {
            unsafe { Self(vandq_u32(self.0, o.0)) }
        }
        #[inline(always)]
        pub(crate) fn or(self, o: Self) -> Self {
            unsafe { Self(vorrq_u32(self.0, o.0)) }
        }
        #[inline(always)]
        pub(crate) fn not(self) -> Self {
            unsafe { Self(vmvnq_u32(self.0)) }
        }
    }

    impl F32x4 {
        #[inline(always)]
        pub(crate) fn splat(v: f32) -> Self {
            unsafe { Self(vdupq_n_f32(v)) }
        }
        #[inline(always)]
        pub(crate) fn from_array(a: [f32; 4]) -> Self {
            unsafe { Self(vld1q_f32(a.as_ptr())) }
        }
        #[inline(always)]
        pub(crate) fn to_array(self) -> [f32; 4] {
            let mut a = [0.0f32; 4];
            unsafe { vst1q_f32(a.as_mut_ptr(), self.0) };
            a
        }
        #[inline(always)]
        pub(crate) fn load(slice: &[f32]) -> Self {
            debug_assert!(slice.len() >= 4);
            unsafe { Self(vld1q_f32(slice.as_ptr())) }
        }
        #[inline(always)]
        pub(crate) fn store(self, slice: &mut [f32]) {
            debug_assert!(slice.len() >= 4);
            unsafe { vst1q_f32(slice.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        pub(crate) fn add(self, o: Self) -> Self {
            unsafe { Self(vaddq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub(crate) fn sub(self, o: Self) -> Self {
            unsafe { Self(vsubq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub(crate) fn mul(self, o: Self) -> Self {
            unsafe { Self(vmulq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub(crate) fn mul_add(self, b: Self, c: Self) -> Self {
            unsafe { Self(vfmaq_f32(c.0, self.0, b.0)) }
        }
        #[inline(always)]
        pub(crate) fn sign_mask4(self) -> u8 {
            const BITS: [u32; 4] = [1, 2, 4, 8];
            unsafe {
                let m = vcltzq_f32(self.0);
                let b = vandq_u32(m, vld1q_u32(BITS.as_ptr()));
                vaddvq_u32(b) as u8
            }
        }
        #[inline(always)]
        pub(crate) fn div(self, o: Self) -> Self {
            unsafe { Self(vdivq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub(crate) fn neg(self) -> Self {
            unsafe { Self(vnegq_f32(self.0)) }
        }
        #[inline(always)]
        pub(crate) fn abs(self) -> Self {
            unsafe { Self(vabsq_f32(self.0)) }
        }
        #[inline(always)]
        pub(crate) fn sqrt(self) -> Self {
            unsafe { Self(vsqrtq_f32(self.0)) }
        }
        #[inline(always)]
        pub(crate) fn max(self, o: Self) -> Self {
            unsafe { Self(vmaxnmq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub(crate) fn min(self, o: Self) -> Self {
            unsafe { Self(vminnmq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub(crate) fn clamp01(self) -> Self {
            self.max(Self::splat(0.0)).min(Self::splat(1.0))
        }
        #[inline(always)]
        pub(crate) fn lt(self, o: Self) -> M32x4 {
            unsafe { M32x4(vcltq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub(crate) fn le(self, o: Self) -> M32x4 {
            unsafe { M32x4(vcleq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub(crate) fn gt(self, o: Self) -> M32x4 {
            unsafe { M32x4(vcgtq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub(crate) fn ge(self, o: Self) -> M32x4 {
            unsafe { M32x4(vcgeq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub(crate) fn select(mask: M32x4, if_true: Self, if_false: Self) -> Self {
            unsafe { Self(vbslq_f32(mask.0, if_true.0, if_false.0)) }
        }
        #[inline(always)]
        pub(crate) fn max_elem(self) -> f32 {
            unsafe { vmaxnmvq_f32(self.0) }
        }
        #[inline(always)]
        pub(crate) fn min_elem(self) -> f32 {
            unsafe { vminnmvq_f32(self.0) }
        }
        #[inline(always)]
        pub(crate) fn rev4(self) -> Self {
            unsafe {
                let r = vrev64q_f32(self.0);
                Self(vextq_f32(r, r, 2))
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod simd4 {
    #[derive(Clone, Copy)]
    pub(crate) struct F32x4(pub(crate) [f32; 4]);

    #[derive(Clone, Copy)]
    pub(crate) struct M32x4(pub(crate) [bool; 4]);

    impl M32x4 {
        #[inline(always)]
        pub(crate) fn and(self, o: Self) -> Self {
            Self([self.0[0] && o.0[0], self.0[1] && o.0[1], self.0[2] && o.0[2], self.0[3] && o.0[3]])
        }
        #[inline(always)]
        pub(crate) fn or(self, o: Self) -> Self {
            Self([self.0[0] || o.0[0], self.0[1] || o.0[1], self.0[2] || o.0[2], self.0[3] || o.0[3]])
        }
        #[inline(always)]
        pub(crate) fn not(self) -> Self {
            Self([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
        }
    }

    impl F32x4 {
        #[inline(always)]
        pub(crate) fn splat(v: f32) -> Self {
            Self([v, v, v, v])
        }
        #[inline(always)]
        pub(crate) fn from_array(a: [f32; 4]) -> Self {
            Self(a)
        }
        #[inline(always)]
        pub(crate) fn to_array(self) -> [f32; 4] {
            self.0
        }
        #[inline(always)]
        pub(crate) fn load(s: &[f32]) -> Self {
            Self([s[0], s[1], s[2], s[3]])
        }
        #[inline(always)]
        pub(crate) fn store(self, s: &mut [f32]) {
            s[0] = self.0[0];
            s[1] = self.0[1];
            s[2] = self.0[2];
            s[3] = self.0[3];
        }
        #[inline(always)]
        pub(crate) fn add(self, o: Self) -> Self {
            Self([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2], self.0[3] + o.0[3]])
        }
        #[inline(always)]
        pub(crate) fn sub(self, o: Self) -> Self {
            Self([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2], self.0[3] - o.0[3]])
        }
        #[inline(always)]
        pub(crate) fn mul(self, o: Self) -> Self {
            Self([self.0[0] * o.0[0], self.0[1] * o.0[1], self.0[2] * o.0[2], self.0[3] * o.0[3]])
        }
        #[inline(always)]
        pub(crate) fn mul_add(self, b: Self, c: Self) -> Self {
            Self([
                self.0[0].mul_add(b.0[0], c.0[0]),
                self.0[1].mul_add(b.0[1], c.0[1]),
                self.0[2].mul_add(b.0[2], c.0[2]),
                self.0[3].mul_add(b.0[3], c.0[3]),
            ])
        }
        #[inline(always)]
        pub(crate) fn sign_mask4(self) -> u8 {
            ((self.0[0] < 0.0) as u8)
                | (((self.0[1] < 0.0) as u8) << 1)
                | (((self.0[2] < 0.0) as u8) << 2)
                | (((self.0[3] < 0.0) as u8) << 3)
        }
        #[inline(always)]
        pub(crate) fn div(self, o: Self) -> Self {
            Self([self.0[0] / o.0[0], self.0[1] / o.0[1], self.0[2] / o.0[2], self.0[3] / o.0[3]])
        }
        #[inline(always)]
        pub(crate) fn neg(self) -> Self {
            Self([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
        }
        #[inline(always)]
        pub(crate) fn abs(self) -> Self {
            Self([self.0[0].abs(), self.0[1].abs(), self.0[2].abs(), self.0[3].abs()])
        }
        #[inline(always)]
        pub(crate) fn sqrt(self) -> Self {
            Self([self.0[0].sqrt(), self.0[1].sqrt(), self.0[2].sqrt(), self.0[3].sqrt()])
        }
        #[inline(always)]
        pub(crate) fn max(self, o: Self) -> Self {
            Self([self.0[0].max(o.0[0]), self.0[1].max(o.0[1]), self.0[2].max(o.0[2]), self.0[3].max(o.0[3])])
        }
        #[inline(always)]
        pub(crate) fn min(self, o: Self) -> Self {
            Self([self.0[0].min(o.0[0]), self.0[1].min(o.0[1]), self.0[2].min(o.0[2]), self.0[3].min(o.0[3])])
        }
        #[inline(always)]
        pub(crate) fn clamp01(self) -> Self {
            self.max(Self::splat(0.0)).min(Self::splat(1.0))
        }
        #[inline(always)]
        pub(crate) fn lt(self, o: Self) -> M32x4 {
            M32x4([self.0[0] < o.0[0], self.0[1] < o.0[1], self.0[2] < o.0[2], self.0[3] < o.0[3]])
        }
        #[inline(always)]
        pub(crate) fn le(self, o: Self) -> M32x4 {
            M32x4([self.0[0] <= o.0[0], self.0[1] <= o.0[1], self.0[2] <= o.0[2], self.0[3] <= o.0[3]])
        }
        #[inline(always)]
        pub(crate) fn gt(self, o: Self) -> M32x4 {
            M32x4([self.0[0] > o.0[0], self.0[1] > o.0[1], self.0[2] > o.0[2], self.0[3] > o.0[3]])
        }
        #[inline(always)]
        pub(crate) fn ge(self, o: Self) -> M32x4 {
            M32x4([self.0[0] >= o.0[0], self.0[1] >= o.0[1], self.0[2] >= o.0[2], self.0[3] >= o.0[3]])
        }
        #[inline(always)]
        pub(crate) fn select(mask: M32x4, if_true: Self, if_false: Self) -> Self {
            Self([
                if mask.0[0] { if_true.0[0] } else { if_false.0[0] },
                if mask.0[1] { if_true.0[1] } else { if_false.0[1] },
                if mask.0[2] { if_true.0[2] } else { if_false.0[2] },
                if mask.0[3] { if_true.0[3] } else { if_false.0[3] },
            ])
        }
        #[inline(always)]
        pub(crate) fn max_elem(self) -> f32 {
            self.0[0].max(self.0[1]).max(self.0[2].max(self.0[3]))
        }
        #[inline(always)]
        pub(crate) fn min_elem(self) -> f32 {
            self.0[0].min(self.0[1]).min(self.0[2].min(self.0[3]))
        }
        #[inline(always)]
        pub(crate) fn rev4(self) -> Self {
            Self([self.0[3], self.0[2], self.0[1], self.0[0]])
        }
    }
}

pub(crate) use simd4::F32x4;

/// Axis-aligned signed permutation (multiples of 90 degrees): `out[r] = (if neg[r] { -1 } else { 1 }) * in[axis[r]]`.
#[derive(Debug, Clone, Copy)]
struct AxisPerm {
    axis: [u8; 3],
    neg: [bool; 3],
}

impl AxisPerm {
    fn from_mat(m: &Mat3) -> Option<Self> {
        let mut axis = [0u8; 3];
        let mut neg = [false; 3];
        let rows = [
            [m.x_axis.x, m.y_axis.x, m.z_axis.x],
            [m.x_axis.y, m.y_axis.y, m.z_axis.y],
            [m.x_axis.z, m.y_axis.z, m.z_axis.z],
        ];
        for r in 0..3 {
            let mut found = None;
            let mut zeros = 0u8;
            for c in 0..3 {
                let v = rows[r][c];
                if v.abs() > 0.99999 {
                    found = Some((c as u8, v < 0.0));
                } else if v.abs() < 1e-5 {
                    zeros += 1;
                }
            }
            if let Some((c, n)) = found {
                if zeros == 2 {
                    axis[r] = c;
                    neg[r] = n;
                    continue;
                }
            }
            return None;
        }
        Some(AxisPerm { axis, neg })
    }

    #[inline(always)]
    fn apply_vec3(self, p: Vec3) -> Vec3 {
        let c = |r: usize| {
            let v = p[self.axis[r] as usize];
            if self.neg[r] { -v } else { v }
        };
        Vec3::new(c(0), c(1), c(2))
    }

    #[inline(always)]
    fn apply_aabb(self, min: Vec3, max: Vec3) -> (Vec3, Vec3) {
        let ax = |r: usize| -> (f32, f32) {
            let c = self.axis[r] as usize;
            if self.neg[r] { (-max[c], -min[c]) } else { (min[c], max[c]) }
        };
        let (x0, x1) = ax(0);
        let (y0, y1) = ax(1);
        let (z0, z1) = ax(2);
        (Vec3::new(x0, y0, z0), Vec3::new(x1, y1, z1))
    }

    #[inline(always)]
    fn apply_f32x4(self, x: F32x4, y: F32x4, z: F32x4) -> (F32x4, F32x4, F32x4) {
        let pick = |r: usize| {
            let v = match self.axis[r] {
                0 => x,
                1 => y,
                _ => z,
            };
            if self.neg[r] { v.neg() } else { v }
        };
        (pick(0), pick(1), pick(2))
    }

    /// Returns true if this is a pure X-axis 90° rotation (`axis == [0, 2, 1]`, like cup/mug/teapot handles).
    #[inline(always)]
    fn is_x_rot90(self) -> bool {
        self.axis == [0, 2, 1]
    }
}

/// Classification of a 3D box against the zero level-set (with narrow-band margin).
#[derive(Debug, Clone, Copy)]
pub(crate) enum BlockClass {
    /// Strictly outside (`d > margin`): value is lower bound `lo`.
    Empty(f32),
    /// Strictly inside (`d < -margin`): value is upper bound `hi`.
    Solid(f32),
    /// May cross the zero level-set (or within `margin` of it).
    Band,
}

/// Flattened non-recursive register-stack SDF tape for blazing-fast 4-wide NEON evaluation
/// (used in `project_vertices_batched` to eliminate recursive dispatch and transform flag checks).
pub(crate) struct FlatSdf {
    ops: Vec<FlatOp>,
}

#[derive(Clone, Copy)]
enum FlatOp {
    Sphere { at: Vec3, r: f32, mir_mask: u8 },
    Cuboid { at: Vec3, inner: Vec3, round: f32, mir_mask: u8 },
    Cylinder { at: Vec3, r_in: f32, h_in: f32, round: f32, mir_mask: u8 },
    CoaxialCylDiff {
        cx: f32,
        cy: f32,
        z0: f32,
        r0: f32,
        h0: f32,
        rd0: f32,
        z1: f32,
        r1: f32,
        h1: f32,
        rd1: f32,
        mir_mask: u8,
    },
    Torus { at: Vec3, major: f32, minor: f32, mir_mask: u8 },
    TorusXRot90 { at: Vec3, major: f32, minor: f32, mir_mask: u8 },
    Capsule { a: Vec3, ba: Vec3, inv_ba_dot: f32, r: f32, mir_mask: u8 },
    Ellipsoid { at: Vec3, inv_r: Vec3, inv_r2: Vec3, neg_min_r: f32, mir_mask: u8 },
    Min,
    SMin { k: f32, inv_k: f32 },
    MaxNeg,
    SMaxNeg { k: f32, inv_k: f32 },
    Max,
    SMax { k: f32, inv_k: f32 },
    Hollow { hh: f32 },
}

impl FlatSdf {
    #[inline(always)]
    fn apply_mir(mut x: F32x4, mut y: F32x4, mut z: F32x4, mir_mask: u8) -> (F32x4, F32x4, F32x4) {
        if mir_mask & 1 != 0 {
            x = x.abs();
        }
        if mir_mask & 2 != 0 {
            y = y.abs();
        }
        if mir_mask & 4 != 0 {
            z = z.abs();
        }
        (x, y, z)
    }

    #[inline(always)]
    pub(crate) fn eval4_pts(&self, pts: [Vec3; 4]) -> [f32; 4] {
        let x0 = F32x4::from_array([pts[0].x, pts[1].x, pts[2].x, pts[3].x]);
        let y0 = F32x4::from_array([pts[0].y, pts[1].y, pts[2].y, pts[3].y]);
        let z0 = F32x4::from_array([pts[0].z, pts[1].z, pts[2].z, pts[3].z]);
        let zero = F32x4::splat(0.0);
        let mut stack = [zero; 16];
        let mut sp = 0usize;
        for &op in &self.ops {
            match op {
                FlatOp::Sphere { at, r, mir_mask } => {
                    let (mx, my, mz) = Self::apply_mir(x0, y0, z0, mir_mask);
                    let x = mx.sub(F32x4::splat(at.x));
                    let y = my.sub(F32x4::splat(at.y));
                    let z = mz.sub(F32x4::splat(at.z));
                    stack[sp] = x.mul_add(x, y.mul_add(y, z.mul(z))).sqrt().sub(F32x4::splat(r));
                    sp += 1;
                }
                FlatOp::Cuboid { at, inner, round, mir_mask } => {
                    let (mx, my, mz) = Self::apply_mir(x0, y0, z0, mir_mask);
                    let qx = mx.sub(F32x4::splat(at.x)).abs().sub(F32x4::splat(inner.x));
                    let qy = my.sub(F32x4::splat(at.y)).abs().sub(F32x4::splat(inner.y));
                    let qz = mz.sub(F32x4::splat(at.z)).abs().sub(F32x4::splat(inner.z));
                    let ox = qx.max(zero);
                    let oy = qy.max(zero);
                    let oz = qz.max(zero);
                    stack[sp] = ox
                        .mul_add(ox, oy.mul_add(oy, oz.mul(oz)))
                        .sqrt()
                        .add(qx.max(qy).max(qz).min(zero))
                        .sub(F32x4::splat(round));
                    sp += 1;
                }
                FlatOp::Cylinder { at, r_in, h_in, round, mir_mask } => {
                    let (mx, my, mz) = Self::apply_mir(x0, y0, z0, mir_mask);
                    let x = mx.sub(F32x4::splat(at.x));
                    let y = my.sub(F32x4::splat(at.y));
                    let z = mz.sub(F32x4::splat(at.z));
                    let dr = x.mul_add(x, y.mul(y)).sqrt().sub(F32x4::splat(r_in));
                    let dz = z.abs().sub(F32x4::splat(h_in));
                    let or = dr.max(zero);
                    let oz = dz.max(zero);
                    stack[sp] = dr.max(dz).min(zero).add(or.mul_add(or, oz.mul(oz)).sqrt()).sub(F32x4::splat(round));
                    sp += 1;
                }
                FlatOp::CoaxialCylDiff { cx, cy, z0: cz0, r0, h0, rd0, z1: cz1, r1, h1, rd1, mir_mask } => {
                    let (mx, my, mz) = Self::apply_mir(x0, y0, z0, mir_mask);
                    let x = mx.sub(F32x4::splat(cx));
                    let y = my.sub(F32x4::splat(cy));
                    let rxy = x.mul_add(x, y.mul(y)).sqrt();
                    let dr0 = rxy.sub(F32x4::splat(r0));
                    let dz0 = mz.sub(F32x4::splat(cz0)).abs().sub(F32x4::splat(h0));
                    let or0 = dr0.max(zero);
                    let oz0 = dz0.max(zero);
                    let d0 = dr0.max(dz0).min(zero).add(or0.mul_add(or0, oz0.mul(oz0)).sqrt()).sub(F32x4::splat(rd0));
                    let dr1 = rxy.sub(F32x4::splat(r1));
                    let dz1 = mz.sub(F32x4::splat(cz1)).abs().sub(F32x4::splat(h1));
                    let or1 = dr1.max(zero);
                    let oz1 = dz1.max(zero);
                    let d1 = dr1.max(dz1).min(zero).add(or1.mul_add(or1, oz1.mul(oz1)).sqrt()).sub(F32x4::splat(rd1));
                    stack[sp] = d0.max(d1.neg());
                    sp += 1;
                }
                FlatOp::Torus { at, major, minor, mir_mask } => {
                    let (mx, my, mz) = Self::apply_mir(x0, y0, z0, mir_mask);
                    let x = mx.sub(F32x4::splat(at.x));
                    let y = my.sub(F32x4::splat(at.y));
                    let z = mz.sub(F32x4::splat(at.z));
                    let q = x.mul_add(x, y.mul(y)).sqrt().sub(F32x4::splat(major));
                    stack[sp] = q.mul_add(q, z.mul(z)).sqrt().sub(F32x4::splat(minor));
                    sp += 1;
                }
                FlatOp::TorusXRot90 { at, major, minor, mir_mask } => {
                    let (mx, my, mz) = Self::apply_mir(x0, y0, z0, mir_mask);
                    let x = mx.sub(F32x4::splat(at.x));
                    let y = my.sub(F32x4::splat(at.y));
                    let z = mz.sub(F32x4::splat(at.z));
                    let q = x.mul_add(x, z.mul(z)).sqrt().sub(F32x4::splat(major));
                    stack[sp] = q.mul_add(q, y.mul(y)).sqrt().sub(F32x4::splat(minor));
                    sp += 1;
                }
                FlatOp::Capsule { a, ba, inv_ba_dot, r, mir_mask } => {
                    let (mx, my, mz) = Self::apply_mir(x0, y0, z0, mir_mask);
                    let px = mx.sub(F32x4::splat(a.x));
                    let py = my.sub(F32x4::splat(a.y));
                    let pz = mz.sub(F32x4::splat(a.z));
                    let bx = F32x4::splat(ba.x);
                    let by = F32x4::splat(ba.y);
                    let bz = F32x4::splat(ba.z);
                    let t = px.mul_add(bx, py.mul_add(by, pz.mul(bz))).mul(F32x4::splat(inv_ba_dot)).clamp01();
                    let qx = px.sub(bx.mul(t));
                    let qy = py.sub(by.mul(t));
                    let qz = pz.sub(bz.mul(t));
                    stack[sp] = qx.mul_add(qx, qy.mul_add(qy, qz.mul(qz))).sqrt().sub(F32x4::splat(r));
                    sp += 1;
                }
                FlatOp::Ellipsoid { at, inv_r, inv_r2, neg_min_r, mir_mask } => {
                    let (mx, my, mz) = Self::apply_mir(x0, y0, z0, mir_mask);
                    let x = mx.sub(F32x4::splat(at.x));
                    let y = my.sub(F32x4::splat(at.y));
                    let z = mz.sub(F32x4::splat(at.z));
                    let rx = x.mul(F32x4::splat(inv_r.x));
                    let ry = y.mul(F32x4::splat(inv_r.y));
                    let rz = z.mul(F32x4::splat(inv_r.z));
                    let k0 = rx.mul_add(rx, ry.mul_add(ry, rz.mul(rz))).sqrt();
                    let r2x = x.mul(F32x4::splat(inv_r2.x));
                    let r2y = y.mul(F32x4::splat(inv_r2.y));
                    let r2z = z.mul(F32x4::splat(inv_r2.z));
                    let k1 = r2x.mul_add(r2x, r2y.mul_add(r2y, r2z.mul(r2z))).sqrt();
                    let eps = F32x4::splat(1e-9);
                    let val = k0.mul(k0.sub(F32x4::splat(1.0))).div(k1.max(eps));
                    stack[sp] = F32x4::select(k1.gt(eps), val, F32x4::splat(neg_min_r));
                    sp += 1;
                }
                FlatOp::Min => {
                    sp -= 1;
                    stack[sp - 1] = stack[sp - 1].min(stack[sp]);
                }
                FlatOp::SMin { k, inv_k } => {
                    sp -= 1;
                    stack[sp - 1] = smin4(stack[sp - 1], stack[sp], F32x4::splat(k), F32x4::splat(inv_k));
                }
                FlatOp::MaxNeg => {
                    sp -= 1;
                    stack[sp - 1] = stack[sp - 1].max(stack[sp].neg());
                }
                FlatOp::SMaxNeg { k, inv_k } => {
                    sp -= 1;
                    stack[sp - 1] = smax4(stack[sp - 1], stack[sp].neg(), F32x4::splat(k), F32x4::splat(inv_k));
                }
                FlatOp::Max => {
                    sp -= 1;
                    stack[sp - 1] = stack[sp - 1].max(stack[sp]);
                }
                FlatOp::SMax { k, inv_k } => {
                    sp -= 1;
                    stack[sp - 1] = smax4(stack[sp - 1], stack[sp], F32x4::splat(k), F32x4::splat(inv_k));
                }
                FlatOp::Hollow { hh } => {
                    let hhv = F32x4::splat(hh);
                    stack[sp - 1] = stack[sp - 1].add(hhv).abs().sub(hhv);
                }
            }
        }
        stack[0].to_array()
    }
}

/// Samples per batched row evaluation chunk (stack scratch size, multiple of 4).
const ROW: usize = 32;

/// Bounding box of the `n` samples `p0 + dir * i`.
#[inline(always)]
fn row_box(p0: Vec3, dir: Vec3, n: usize) -> (Vec3, Vec3) {
    let e = p0 + dir * (n.max(1) - 1) as f32;
    (p0.min(e), p0.max(e))
}

/// Precomputed polygon edge for fast scalar & 4-wide SIMD distance evaluation.
#[derive(Debug, Clone, Copy)]
struct PolyEdge {
    vi: Vec2,
    vj_y: f32,
    e: Vec2,
    inv_e2: f32,
}

fn build_poly_edges(pts: &[Vec2]) -> Vec<PolyEdge> {
    let n = pts.len();
    let mut out = Vec::with_capacity(n);
    let mut j = n - 1;
    for i in 0..n {
        let vi = pts[i];
        let vj = pts[j];
        let e = vj - vi;
        let inv_e2 = 1.0 / dot2(e).max(1e-12);
        out.push(PolyEdge { vi, vj_y: vj.y, e, inv_e2 });
        j = i;
    }
    out
}

#[derive(Debug, Clone)]
enum Kind {
    Sphere(f32),
    /// Precomputed (inner = half - round, round)
    Cuboid { inner: Vec3, round: f32 },
    /// Precomputed (r_in = r - round, h_in = hh - round, round)
    Cylinder { r_in: f32, h_in: f32, round: f32 },
    /// Precomputed capped cone constants
    Cone { r1: f32, r2: f32, hh: f32, k1: Vec2, k2: Vec2, inv_k2_dot: f32 },
    Torus(f32, f32),
    /// Precomputed (a, ba = b - a, inv_ba_dot, r)
    Capsule { a: Vec3, ba: Vec3, inv_ba_dot: f32, r: f32 },
    /// Precomputed (half, inv_r, inv_r2, neg_min_r)
    Ellipsoid { half: Vec3, inv_r: Vec3, inv_r2: Vec3, neg_min_r: f32 },
    /// Precomputed pyramid constants
    Pyramid { base: f32, h: f32, inv_s: f32, h_over_s: f32, m2: f32, inv_m2_p: f32, inv_m2: f32 },
    Lathe(Vec<PolyEdge>),
    Extrude(Vec<PolyEdge>, f32),
    Union(Vec<Compiled>, f32),
    Difference(Vec<Compiled>, f32),
    Intersection(Vec<Compiled>, f32),
    Array(Box<Compiled>, u32, Vec3),
}

/// A node ready for evaluation: inverse placement precomputed and bounds in the parent's frame.
#[derive(Debug, Clone)]
pub struct Compiled {
    inv_mat: Mat3,
    abs_inv_mat: Mat3,
    axis_perm: Option<AxisPerm>,
    at: Vec3,
    inv_scale_stretch: Vec3,
    /// Distance correction for scale and stretch (keeps the field a bound: |grad| <= 1).
    lip: f32,
    /// Lower bound on `eval(p) / aabb.distance(p)` outside `self.aabb` (<= 1.0).
    aabb_scale: f32,
    has_at: bool,
    has_rot: bool,
    has_scale: bool,
    /// True when `self.aabb` is already exact (every face touches the solid shape).
    exact_aabb: bool,
    mirror: Option<usize>,
    hollow: f32,
    kind: Kind,
    local_aabb: Aabb,
    /// Bounds in the parent's frame.
    pub aabb: Aabb,
}

fn v3(a: [f32; 3]) -> Vec3 {
    Vec3::from_array(a)
}

pub fn rotation(deg: [f32; 3]) -> Quat {
    let r = v3(deg) * (std::f32::consts::PI / 180.0);
    Quat::from_euler(EulerRot::XYZ, r.x, r.y, r.z)
}

impl Compiled {
    pub fn new(n: &Node) -> Compiled {
        let (kind, local_aabb, kind_exact) = match &n.shape {
            Shape::Sphere { r } => (Kind::Sphere(*r), Aabb::half(Vec3::splat(*r)), true),
            Shape::Cuboid { size, round } => {
                let half = v3(*size) * 0.5;
                let r = round.min(half.min_element());
                (Kind::Cuboid { inner: half - Vec3::splat(r), round: r }, Aabb::half(half), true)
            }
            Shape::Cylinder { r, h, round } => {
                let hh = h * 0.5;
                let rd = round.min(*r).min(hh);
                (
                    Kind::Cylinder { r_in: *r - rd, h_in: hh - rd, round: rd },
                    Aabb::half(Vec3::new(*r, *r, hh)),
                    true,
                )
            }
            Shape::Cone { r1, r2, h } => {
                let hh = h * 0.5;
                let k1 = Vec2::new(*r2, hh);
                let k2 = Vec2::new(*r2 - *r1, 2.0 * hh);
                let d2 = dot2(k2);
                let r = r1.max(*r2);
                (
                    Kind::Cone { r1: *r1, r2: *r2, hh, k1, k2, inv_k2_dot: if d2 > 0.0 { 1.0 / d2 } else { 0.0 } },
                    Aabb::half(Vec3::new(r, r, hh)),
                    true,
                )
            }
            Shape::Torus { major, minor } => {
                (Kind::Torus(*major, *minor), Aabb::half(Vec3::new(major + minor, major + minor, *minor)), true)
            }
            Shape::Capsule { from, to, r } => {
                let a = v3(*from);
                let b = v3(*to);
                let ba = b - a;
                let d2 = ba.length_squared();
                let lb = Aabb::new(a.min(b) - Vec3::splat(*r), a.max(b) + Vec3::splat(*r));
                (Kind::Capsule { a, ba, inv_ba_dot: if d2 > 0.0 { 1.0 / d2 } else { 0.0 }, r: *r }, lb, true)
            }
            Shape::Ellipsoid { size } => {
                let half = v3(*size) * 0.5;
                let inv_r = Vec3::ONE / half;
                (
                    Kind::Ellipsoid { half, inv_r, inv_r2: inv_r * inv_r, neg_min_r: -half.min_element() },
                    Aabb::half(half),
                    true,
                )
            }
            Shape::Pyramid { base, h } => {
                let inv_s = 1.0 / *base;
                let h_over_s = *h * inv_s;
                let m2 = h_over_s * h_over_s + 0.25;
                let lb = Aabb::new(Vec3::new(-base / 2.0, -base / 2.0, -h / 2.0), Vec3::new(base / 2.0, base / 2.0, h / 2.0));
                (
                    Kind::Pyramid { base: *base, h: *h, inv_s, h_over_s, m2, inv_m2_p: 1.0 / (m2 + 0.25), inv_m2: 1.0 / m2 },
                    lb,
                    true,
                )
            }
            Shape::Lathe { profile } => {
                let mut pts: Vec<Vec2> = profile.iter().map(|p| Vec2::new(p[0].max(0.0), p[1])).collect();
                // Close the profile along the axis so it is a polygon in (radius, z).
                let (first, last) = (pts[0], pts[pts.len() - 1]);
                if last.x > 0.0 {
                    pts.push(Vec2::new(0.0, last.y));
                }
                if first.x > 0.0 {
                    pts.insert(0, Vec2::new(0.0, first.y));
                }
                let r = pts.iter().fold(0.0f32, |m, p| m.max(p.x));
                let (lo, hi) = pts.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| (lo.min(p.y), hi.max(p.y)));
                let lb = Aabb::new(Vec3::new(-r, -r, lo), Vec3::new(r, r, hi));
                (Kind::Lathe(build_poly_edges(&pts)), lb, true)
            }
            Shape::Extrude { outline, h } => {
                let pts: Vec<Vec2> = outline.iter().map(|p| Vec2::new(p[0], p[1])).collect();
                let hh = h * 0.5;
                let lo = pts.iter().fold(Vec2::splat(f32::INFINITY), |m, p| m.min(*p));
                let hi = pts.iter().fold(Vec2::splat(f32::NEG_INFINITY), |m, p| m.max(*p));
                let lb = Aabb::new(lo.extend(-hh), hi.extend(hh));
                (Kind::Extrude(build_poly_edges(&pts), hh), lb, true)
            }
            Shape::Union { children, smooth } => {
                let ch: Vec<Compiled> = children.iter().map(Compiled::new).collect();
                let lb = ch.iter().fold(Aabb::EMPTY, |b, c| b.union(&c.aabb)).grow(*smooth * 0.25);
                let ex = *smooth <= 0.0 && !ch.is_empty() && ch.iter().all(|c| c.exact_aabb);
                (Kind::Union(ch, *smooth), lb, ex)
            }
            Shape::Difference { children, smooth } => {
                let ch: Vec<Compiled> = children.iter().map(Compiled::new).collect();
                let lb = ch[0].aabb;
                (Kind::Difference(ch, *smooth), lb, false)
            }
            Shape::Intersection { children, smooth } => {
                let ch: Vec<Compiled> = children.iter().map(Compiled::new).collect();
                let lb = ch.iter().skip(1).fold(ch[0].aabb, |b, c| b.intersect(&c.aabb));
                (Kind::Intersection(ch, *smooth), lb, false)
            }
            Shape::Array { child, count, offset } => {
                let c = Compiled::new(child);
                let off = v3(*offset);
                let last = Aabb::new(c.aabb.min + off * (*count as f32 - 1.0), c.aabb.max + off * (*count as f32 - 1.0));
                let lb = c.aabb.union(&last);
                let ex = *count > 0 && c.exact_aabb;
                (Kind::Array(Box::new(c), *count, off), lb, ex)
            }
        };
        let stretch = v3(n.stretch);
        let at = v3(n.at);
        let has_at = at != Vec3::ZERO;
        let has_rot = n.rot != [0.0, 0.0, 0.0];
        let rot = if has_rot { rotation(n.rot) } else { Quat::IDENTITY };
        let raw_inv_mat = if has_rot { Mat3::from_quat(rot.inverse()) } else { Mat3::IDENTITY };
        let axis_perm = if has_rot { AxisPerm::from_mat(&raw_inv_mat) } else { None };
        let inv_mat = if let Some(perm) = axis_perm {
            Mat3::from_cols(
                perm.apply_vec3(Vec3::X),
                perm.apply_vec3(Vec3::Y),
                perm.apply_vec3(Vec3::Z),
            )
        } else {
            raw_inv_mat
        };
        let abs_inv_mat = Mat3::from_cols(inv_mat.x_axis.abs(), inv_mat.y_axis.abs(), inv_mat.z_axis.abs());
        let scale_stretch = stretch * n.scale;
        let has_scale = scale_stretch != Vec3::ONE;
        let kind_aabb_scale = match &kind {
            Kind::Ellipsoid { half, .. } => (half.min_element() / half.max_element().max(1e-6)).max(1e-4),
            Kind::Union(children, _) | Kind::Difference(children, _) | Kind::Intersection(children, _) => {
                children.iter().map(|c| c.aabb_scale).fold(1.0f32, f32::min)
            }
            Kind::Array(c, _, _) => c.aabb_scale,
            _ => 1.0,
        };
        let stretch_ratio = if has_scale {
            (scale_stretch.min_element() / scale_stretch.max_element().max(1e-6)).max(1e-4)
        } else {
            1.0
        };
        let mut b = Aabb::EMPTY;
        if !local_aabb.is_empty() {
            for k in local_aabb.corners() {
                let p = rot * (k * stretch * n.scale) + at;
                b = b.union(&Aabb::new(p, p));
            }
        }
        let mirror = n.mirror.map(|a| a.index());
        if let Some(a) = mirror {
            let mut m = b;
            m.min[a] = -b.max[a];
            m.max[a] = -b.min[a];
            b = b.union(&m);
        }
        let rot_preserves_exact = !has_rot || axis_perm.is_some();
        // Mirror around axis `a` preserves exactness when the unmirrored box lies on one side of 0
        // (or is symmetric), which is always true when `local` touches its faces since both the + and -
        // copies preserve the maximum extent in every direction.
        let exact_aabb = kind_exact && rot_preserves_exact && !b.is_empty();
        Compiled {
            inv_mat,
            abs_inv_mat,
            axis_perm,
            at,
            inv_scale_stretch: Vec3::ONE / scale_stretch,
            lip: n.scale * stretch.min_element(),
            aabb_scale: kind_aabb_scale * stretch_ratio,
            has_at,
            has_rot,
            has_scale,
            exact_aabb,
            mirror,
            hollow: n.hollow,
            kind,
            local_aabb,
            aabb: b,
        }
    }

    /// Signed distance at `p`, given in the parent's frame.
    #[inline(always)]
    pub fn eval(&self, mut p: Vec3) -> f32 {
        if let Some(a) = self.mirror {
            p[a] = p[a].abs();
        }
        if self.has_at {
            p -= self.at;
        }
        if self.has_rot {
            if let Some(perm) = self.axis_perm {
                p = perm.apply_vec3(p);
            } else {
                p = self.inv_mat * p;
            }
        }
        if self.has_scale {
            p *= self.inv_scale_stretch;
        }
        let d = self.kind_eval(p) * self.lip;
        if self.hollow > 0.0 {
            (d + self.hollow * 0.5).abs() - self.hollow * 0.5
        } else {
            d
        }
    }

    /// Evaluates 4 arbitrary points simultaneously using 4-wide SIMD (NEON on aarch64).
    #[inline]
    pub fn eval4_pts(&self, pts: [Vec3; 4]) -> [f32; 4] {
        let x = F32x4::from_array([pts[0].x, pts[1].x, pts[2].x, pts[3].x]);
        let y = F32x4::from_array([pts[0].y, pts[1].y, pts[2].y, pts[3].y]);
        let z = F32x4::from_array([pts[0].z, pts[1].z, pts[2].z, pts[3].z]);
        self.eval4(x, y, z).to_array()
    }

    /// Compiles a small SDF tree (`<= 16` ops, axis-aligned/unscaled) into a flat register-stack tape
    /// with zero recursive dispatch or per-node transform branches.
    pub(crate) fn compile_flat(&self) -> Option<FlatSdf> {
        let mut ops = Vec::with_capacity(8);
        if self.emit_flat(Vec3::ZERO, 0, &mut ops, 0) && !ops.is_empty() && ops.len() <= 24 {
            Some(FlatSdf { ops })
        } else {
            None
        }
    }

    fn emit_flat(&self, parent_at: Vec3, parent_mir: u8, ops: &mut Vec<FlatOp>, depth: usize) -> bool {
        if depth >= 14 || self.has_scale || self.lip != 1.0 {
            return false;
        }
        // Mirror only commutes with parent_at if parent_at along the mirrored axis is 0.
        let mut mir_mask = parent_mir;
        if let Some(a) = self.mirror {
            if parent_at[a] != 0.0 {
                return false;
            }
            mir_mask |= 1u8 << a;
        }
        let at = parent_at + if self.has_at { self.at } else { Vec3::ZERO };
        if self.has_rot {
            if let (Some(perm), Kind::Torus(major, minor)) = (self.axis_perm, &self.kind) {
                if perm.is_x_rot90() {
                    ops.push(FlatOp::TorusXRot90 { at, major: *major, minor: *minor, mir_mask });
                    if self.hollow > 0.0 {
                        ops.push(FlatOp::Hollow { hh: self.hollow * 0.5 });
                    }
                    return true;
                }
            }
            return false;
        }
        match &self.kind {
            Kind::Sphere(r) => ops.push(FlatOp::Sphere { at, r: *r, mir_mask }),
            Kind::Cuboid { inner, round } => ops.push(FlatOp::Cuboid { at, inner: *inner, round: *round, mir_mask }),
            Kind::Cylinder { r_in, h_in, round } => {
                ops.push(FlatOp::Cylinder { at, r_in: *r_in, h_in: *h_in, round: *round, mir_mask })
            }
            Kind::Torus(major, minor) => ops.push(FlatOp::Torus { at, major: *major, minor: *minor, mir_mask }),
            Kind::Capsule { a, ba, inv_ba_dot, r } => {
                ops.push(FlatOp::Capsule { a: *a + at, ba: *ba, inv_ba_dot: *inv_ba_dot, r: *r, mir_mask })
            }
            Kind::Ellipsoid { inv_r, inv_r2, neg_min_r, .. } => {
                ops.push(FlatOp::Ellipsoid { at, inv_r: *inv_r, inv_r2: *inv_r2, neg_min_r: *neg_min_r, mir_mask })
            }
            Kind::Union(children, k) => {
                if children.is_empty() || children.len() > 8 {
                    return false;
                }
                if !children[0].emit_flat(at, mir_mask, ops, depth + 1) {
                    return false;
                }
                for c in &children[1..] {
                    if !c.emit_flat(at, mir_mask, ops, depth + 1) {
                        return false;
                    }
                    ops.push(if *k <= 0.0 { FlatOp::Min } else { FlatOp::SMin { k: *k, inv_k: 1.0 / *k } });
                }
            }
            Kind::Difference(children, k) => {
                if children.is_empty() || children.len() > 6 {
                    return false;
                }
                if children.len() == 2
                    && *k <= 0.0
                    && !children[0].has_rot && !children[0].has_scale && children[0].mirror.is_none() && children[0].hollow <= 0.0 && children[0].lip == 1.0
                    && !children[1].has_rot && !children[1].has_scale && children[1].mirror.is_none() && children[1].hollow <= 0.0 && children[1].lip == 1.0
                    && children[0].at.x == children[1].at.x
                    && children[0].at.y == children[1].at.y
                {
                    if let (
                        Kind::Cylinder { r_in: r0, h_in: h0, round: rd0 },
                        Kind::Cylinder { r_in: r1, h_in: h1, round: rd1 },
                    ) = (&children[0].kind, &children[1].kind)
                    {
                        ops.push(FlatOp::CoaxialCylDiff {
                            cx: at.x + children[0].at.x,
                            cy: at.y + children[0].at.y,
                            z0: at.z + children[0].at.z,
                            r0: *r0,
                            h0: *h0,
                            rd0: *rd0,
                            z1: at.z + children[1].at.z,
                            r1: *r1,
                            h1: *h1,
                            rd1: *rd1,
                            mir_mask,
                        });
                        if self.hollow > 0.0 {
                            ops.push(FlatOp::Hollow { hh: self.hollow * 0.5 });
                        }
                        return true;
                    }
                }
                if !children[0].emit_flat(at, mir_mask, ops, depth + 1) {
                    return false;
                }
                for c in &children[1..] {
                    if !c.emit_flat(at, mir_mask, ops, depth + 1) {
                        return false;
                    }
                    ops.push(if *k <= 0.0 { FlatOp::MaxNeg } else { FlatOp::SMaxNeg { k: *k, inv_k: 1.0 / *k } });
                }
            }
            Kind::Intersection(children, k) => {
                if children.is_empty() || children.len() > 6 {
                    return false;
                }
                if !children[0].emit_flat(at, mir_mask, ops, depth + 1) {
                    return false;
                }
                for c in &children[1..] {
                    if !c.emit_flat(at, mir_mask, ops, depth + 1) {
                        return false;
                    }
                    ops.push(if *k <= 0.0 { FlatOp::Max } else { FlatOp::SMax { k: *k, inv_k: 1.0 / *k } });
                }
            }
            _ => return false,
        }
        if self.hollow > 0.0 {
            ops.push(FlatOp::Hollow { hh: self.hollow * 0.5 });
        }
        true
    }

    #[inline(always)]
    fn eval4(&self, mut x: F32x4, mut y: F32x4, mut z: F32x4) -> F32x4 {
        if let Some(a) = self.mirror {
            match a {
                0 => x = x.abs(),
                1 => y = y.abs(),
                _ => z = z.abs(),
            }
        }
        if self.has_at {
            x = x.sub(F32x4::splat(self.at.x));
            y = y.sub(F32x4::splat(self.at.y));
            z = z.sub(F32x4::splat(self.at.z));
        }
        if self.has_rot {
            if let Some(perm) = self.axis_perm {
                (x, y, z) = perm.apply_f32x4(x, y, z);
            } else {
                let m = &self.inv_mat;
                let nx = x.mul(F32x4::splat(m.x_axis.x))
                    .add(y.mul(F32x4::splat(m.y_axis.x)))
                    .add(z.mul(F32x4::splat(m.z_axis.x)));
                let ny = x.mul(F32x4::splat(m.x_axis.y))
                    .add(y.mul(F32x4::splat(m.y_axis.y)))
                    .add(z.mul(F32x4::splat(m.z_axis.y)));
                let nz = x.mul(F32x4::splat(m.x_axis.z))
                    .add(y.mul(F32x4::splat(m.y_axis.z)))
                    .add(z.mul(F32x4::splat(m.z_axis.z)));
                x = nx;
                y = ny;
                z = nz;
            }
        }
        if self.has_scale {
            x = x.mul(F32x4::splat(self.inv_scale_stretch.x));
            y = y.mul(F32x4::splat(self.inv_scale_stretch.y));
            z = z.mul(F32x4::splat(self.inv_scale_stretch.z));
        }
        let d = if self.lip != 1.0 {
            self.kind_eval4(x, y, z).mul(F32x4::splat(self.lip))
        } else {
            self.kind_eval4(x, y, z)
        };
        if self.hollow > 0.0 {
            let hh = F32x4::splat(self.hollow * 0.5);
            d.add(hh).abs().sub(hh)
        } else {
            d
        }
    }

    #[inline(always)]
    fn is_cheap_leaf(&self) -> bool {
        !matches!(
            self.kind,
            Kind::Union(_, _) | Kind::Difference(_, _) | Kind::Intersection(_, _) | Kind::Array(_, _, _) | Kind::Lathe(_) | Kind::Extrude(_, _)
        )
    }

    #[inline(always)]
    fn kind_eval4(&self, x: F32x4, y: F32x4, z: F32x4) -> F32x4 {
        let zero = F32x4::splat(0.0);
        match &self.kind {
            Kind::Sphere(r) => x.mul_add(x, y.mul_add(y, z.mul(z))).sqrt().sub(F32x4::splat(*r)),
            Kind::Cuboid { inner, round, .. } => {
                let qx = x.abs().sub(F32x4::splat(inner.x));
                let qy = y.abs().sub(F32x4::splat(inner.y));
                let qz = z.abs().sub(F32x4::splat(inner.z));
                let ox = qx.max(zero);
                let oy = qy.max(zero);
                let oz = qz.max(zero);
                ox.mul_add(ox, oy.mul_add(oy, oz.mul(oz)))
                    .sqrt()
                    .add(qx.max(qy).max(qz).min(zero))
                    .sub(F32x4::splat(*round))
            }
            Kind::Cylinder { r_in, h_in, round, .. } => {
                let dr = x.mul_add(x, y.mul(y)).sqrt().sub(F32x4::splat(*r_in));
                let dz = z.abs().sub(F32x4::splat(*h_in));
                let or = dr.max(zero);
                let oz = dz.max(zero);
                dr.max(dz).min(zero).add(or.mul_add(or, oz.mul(oz)).sqrt()).sub(F32x4::splat(*round))
            }
            Kind::Cone { r1, r2, hh, k1, k2, inv_k2_dot } => {
                let qx = x.mul_add(x, y.mul(y)).sqrt();
                let qy = z;
                let r_sel = F32x4::select(qy.lt(zero), F32x4::splat(*r1), F32x4::splat(*r2));
                let cax = qx.sub(qx.min(r_sel));
                let cay = qy.abs().sub(F32x4::splat(*hh));
                let k1x = F32x4::splat(k1.x);
                let k1y = F32x4::splat(k1.y);
                let k2x = F32x4::splat(k2.x);
                let k2y = F32x4::splat(k2.y);
                let t = k1x
                    .sub(qx)
                    .mul_add(k2x, k1y.sub(qy).mul(k2y))
                    .mul(F32x4::splat(*inv_k2_dot))
                    .clamp01();
                let cbx = k2x.mul_add(t, qx.sub(k1x));
                let cby = k2y.mul_add(t, qy.sub(k1y));
                let d = cax.mul_add(cax, cay.mul(cay)).min(cbx.mul_add(cbx, cby.mul(cby))).sqrt();
                F32x4::select(cbx.lt(zero).and(cay.lt(zero)), d.neg(), d)
            }
            Kind::Torus(a, b) => {
                let q = x.mul_add(x, y.mul(y)).sqrt().sub(F32x4::splat(*a));
                q.mul_add(q, z.mul(z)).sqrt().sub(F32x4::splat(*b))
            }
            Kind::Capsule { a, ba, inv_ba_dot, r, .. } => {
                let px = x.sub(F32x4::splat(a.x));
                let py = y.sub(F32x4::splat(a.y));
                let pz = z.sub(F32x4::splat(a.z));
                let bx = F32x4::splat(ba.x);
                let by = F32x4::splat(ba.y);
                let bz = F32x4::splat(ba.z);
                let t = px.mul_add(bx, py.mul_add(by, pz.mul(bz))).mul(F32x4::splat(*inv_ba_dot)).clamp01();
                let qx = px.sub(bx.mul(t));
                let qy = py.sub(by.mul(t));
                let qz = pz.sub(bz.mul(t));
                qx.mul_add(qx, qy.mul_add(qy, qz.mul(qz))).sqrt().sub(F32x4::splat(*r))
            }
            Kind::Ellipsoid { inv_r, inv_r2, neg_min_r, .. } => {
                let rx = x.mul(F32x4::splat(inv_r.x));
                let ry = y.mul(F32x4::splat(inv_r.y));
                let rz = z.mul(F32x4::splat(inv_r.z));
                let k0 = rx.mul_add(rx, ry.mul_add(ry, rz.mul(rz))).sqrt();
                let r2x = x.mul(F32x4::splat(inv_r2.x));
                let r2y = y.mul(F32x4::splat(inv_r2.y));
                let r2z = z.mul(F32x4::splat(inv_r2.z));
                let k1 = r2x.mul_add(r2x, r2y.mul_add(r2y, r2z.mul(r2z))).sqrt();
                let eps = F32x4::splat(1e-9);
                let val = k0.mul(k0.sub(F32x4::splat(1.0))).div(k1.max(eps));
                F32x4::select(k1.gt(eps), val, F32x4::splat(*neg_min_r))
            }
            Kind::Pyramid { base, h, inv_s, h_over_s, m2, inv_m2_p, inv_m2 } => {
                let is = F32x4::splat(*inv_s);
                let px = x.mul(is);
                let py = z.add(F32x4::splat(*h * 0.5)).mul(is);
                let pz = y.mul(is);
                let ax = px.abs();
                let az = pz.abs();
                let half = F32x4::splat(0.5);
                let xx = ax.max(az).sub(half);
                let zz = ax.min(az).sub(half);
                let hs = F32x4::splat(*h_over_s);
                let qx = zz;
                let qy = hs.mul(py).sub(half.mul(xx));
                let qz = hs.mul(xx).add(half.mul(py));
                let s = qx.neg().max(zero);
                let t = qy.sub(half.mul(zz)).mul(F32x4::splat(*inv_m2_p)).clamp01();
                let m2v = F32x4::splat(*m2);
                let qs = qx.add(s);
                let a = m2v.mul(qs).mul(qs).add(qy.mul(qy));
                let qt = qx.add(half.mul(t));
                let qym = qy.sub(m2v.mul(t));
                let b = m2v.mul(qt).mul(qt).add(qym.mul(qym));
                let cond = qy.min(qx.neg().mul(m2v).sub(qy.mul(half)));
                let d2 = F32x4::select(cond.gt(zero), zero, a.min(b));
                let dist = d2.add(qz.mul(qz)).mul(F32x4::splat(*inv_m2)).sqrt();
                let sgn = F32x4::select(qz.max(py.neg()).le(zero), F32x4::splat(-1.0), F32x4::splat(1.0));
                dist.mul(sgn).mul(F32x4::splat(*base))
            }
            Kind::Lathe(edges) => polygon4(x.mul_add(x, y.mul(y)).sqrt(), z, edges),
            Kind::Extrude(edges, hh) => {
                let wx = polygon4(x, y, edges);
                let wy = z.abs().sub(F32x4::splat(*hh));
                let ox = wx.max(zero);
                let oy = wy.max(zero);
                wx.max(wy).min(zero).add(ox.mul_add(ox, oy.mul(oy)).sqrt())
            }
            Kind::Union(children, k) => {
                if children.is_empty() {
                    return F32x4::splat(f32::INFINITY);
                }
                let mut d = children[0].eval4(x, y, z);
                if children.len() == 1 {
                    return d;
                }
                let k = *k;
                let inv_k = if k > 0.0 { 1.0 / k } else { 0.0 };
                if children.len() == 2 && children[1].is_cheap_leaf() {
                    let e = children[1].eval4(x, y, z);
                    return if k <= 0.0 { d.min(e) } else { smin4(d, e, F32x4::splat(k), F32x4::splat(inv_k)) };
                }
                let bmin = Vec3::new(x.min_elem(), y.min_elem(), z.min_elem());
                let bmax = Vec3::new(x.max_elem(), y.max_elem(), z.max_elem());
                for c in &children[1..] {
                    let d_max = d.max_elem();
                    let reach = (d_max + k) / c.aabb_scale;
                    let bd2 = c.aabb.distance_squared_box(bmin, bmax);
                    if reach <= 0.0 {
                        if bd2 > 0.0 {
                            continue;
                        }
                    } else if bd2 > reach * reach {
                        continue;
                    }
                    let e = c.eval4(x, y, z);
                    d = if k <= 0.0 {
                        d.min(e)
                    } else {
                        smin4(d, e, F32x4::splat(k), F32x4::splat(inv_k))
                    };
                }
                d
            }
            Kind::Difference(children, k) => {
                let mut d = children[0].eval4(x, y, z);
                if children.len() <= 1 {
                    return d;
                }
                let k = *k;
                let inv_k = if k > 0.0 { 1.0 / k } else { 0.0 };
                if children.len() == 2 && children[1].is_cheap_leaf() {
                    let ne = children[1].eval4(x, y, z).neg();
                    return if k <= 0.0 { d.max(ne) } else { smax4(d, ne, F32x4::splat(k), F32x4::splat(inv_k)) };
                }
                let bmin = Vec3::new(x.min_elem(), y.min_elem(), z.min_elem());
                let bmax = Vec3::new(x.max_elem(), y.max_elem(), z.max_elem());
                for c in &children[1..] {
                    let reach = (k + d.abs().max_elem()) / c.aabb_scale;
                    if c.aabb.distance_squared_box(bmin, bmax) > reach * reach {
                        continue;
                    }
                    let ne = c.eval4(x, y, z).neg();
                    d = if k <= 0.0 {
                        d.max(ne)
                    } else {
                        smax4(d, ne, F32x4::splat(k), F32x4::splat(inv_k))
                    };
                }
                d
            }
            Kind::Intersection(children, k) => {
                let mut d = children[0].eval4(x, y, z);
                let k = *k;
                let inv_k = if k > 0.0 { 1.0 / k } else { 0.0 };
                for c in &children[1..] {
                    let e = c.eval4(x, y, z);
                    d = if k <= 0.0 {
                        d.max(e)
                    } else {
                        smax4(d, e, F32x4::splat(k), F32x4::splat(inv_k))
                    };
                }
                d
            }
            Kind::Array(c, n, off) => {
                let mut d = F32x4::splat(f32::INFINITY);
                for i in 0..*n {
                    let fi = i as f32;
                    let e = c.eval4(
                        x.sub(F32x4::splat(off.x * fi)),
                        y.sub(F32x4::splat(off.y * fi)),
                        z.sub(F32x4::splat(off.z * fi)),
                    );
                    d = d.min(e);
                }
                d
            }
        }
    }

    /// Conservative interval `[lo, hi]` bounding `self.eval(p)` for all `p` in the axis-aligned box `[min, max]`.
    #[inline]
    pub fn eval_interval(&self, mut min: Vec3, mut max: Vec3) -> (f32, f32) {
        if let Some(a) = self.mirror {
            let (l, h) = iv_abs(min[a], max[a]);
            min[a] = l;
            max[a] = h;
        }
        if self.has_at {
            min -= self.at;
            max -= self.at;
        }
        if self.has_rot {
            if let Some(perm) = self.axis_perm {
                (min, max) = perm.apply_aabb(min, max);
            } else {
                let c = (min + max) * 0.5;
                let e = (max - min) * 0.5;
                let rc = self.inv_mat * c;
                let re = self.abs_inv_mat * e;
                let eps = rc.abs().max_element() * 1e-5 + 1e-5;
                min = rc - re - Vec3::splat(eps);
                max = rc + re + Vec3::splat(eps);
            }
        }
        if self.has_scale {
            let a = min * self.inv_scale_stretch;
            let b = max * self.inv_scale_stretch;
            min = a.min(b);
            max = a.max(b);
        }
        let (d0, d1) = self.kind_interval(min, max);
        let (d0, d1) = (d0 * self.lip, d1 * self.lip);
        if self.hollow > 0.0 {
            let hh = self.hollow * 0.5;
            let (a0, a1) = iv_abs(d0 + hh, d1 + hh);
            (a0 - hh, a1 - hh)
        } else {
            (d0, d1)
        }
    }

    fn kind_interval(&self, min: Vec3, max: Vec3) -> (f32, f32) {
        match &self.kind {
            Kind::Sphere(r) => {
                let abs_lo = min.max(-max).max(Vec3::ZERO);
                let abs_hi = min.abs().max(max.abs());
                (abs_lo.length() - *r, abs_hi.length() - *r)
            }
            Kind::Cuboid { inner, round, .. } => {
                let abs_lo = min.max(-max).max(Vec3::ZERO);
                let abs_hi = min.abs().max(max.abs());
                let eval_abs = |a: Vec3| -> f32 {
                    let q = a - *inner;
                    q.max(Vec3::ZERO).length() + q.max_element().min(0.0) - *round
                };
                (eval_abs(abs_lo), eval_abs(abs_hi))
            }
            Kind::Cylinder { r_in, h_in, round, .. } => {
                let abs_lo = min.max(-max).max(Vec3::ZERO);
                let abs_hi = min.abs().max(max.abs());
                let eval_cyl = |rxy: f32, az: f32| -> f32 {
                    let dr = rxy - *r_in;
                    let dz = az - *h_in;
                    if dr <= 0.0 && dz <= 0.0 {
                        dr.max(dz) - *round
                    } else if dr <= 0.0 {
                        dz - *round
                    } else if dz <= 0.0 {
                        dr - *round
                    } else {
                        (dr * dr + dz * dz).sqrt() - *round
                    }
                };
                (
                    eval_cyl((abs_lo.x * abs_lo.x + abs_lo.y * abs_lo.y).sqrt(), abs_lo.z),
                    eval_cyl((abs_hi.x * abs_hi.x + abs_hi.y * abs_hi.y).sqrt(), abs_hi.z),
                )
            }
            Kind::Torus(a, b) => {
                let abs_lo = min.max(-max).max(Vec3::ZERO);
                let abs_hi = min.abs().max(max.abs());
                let r_lo = (abs_lo.x * abs_lo.x + abs_lo.y * abs_lo.y).sqrt();
                let r_hi = (abs_hi.x * abs_hi.x + abs_hi.y * abs_hi.y).sqrt();
                let (q_lo, q_hi) = iv_abs(r_lo - *a, r_hi - *a);
                (
                    (q_lo * q_lo + abs_lo.z * abs_lo.z).sqrt() - *b,
                    (q_hi * q_hi + abs_hi.z * abs_hi.z).sqrt() - *b,
                )
            }
            Kind::Ellipsoid { half, inv_r, inv_r2, neg_min_r } => {
                let abs_lo = min.max(-max).max(Vec3::ZERO);
                let abs_hi = min.abs().max(max.abs());
                let k0_lo = (abs_lo * *inv_r).length();
                let k0_hi = (abs_hi * *inv_r).length();
                let min_r = -*neg_min_r;
                let max_r = half.max_element();
                if k0_lo >= 1.0 {
                    let k1_lo = (abs_lo * *inv_r2).length();
                    let k1_hi = (abs_hi * *inv_r2).length();
                    let d_lo = (k0_lo * (k0_lo - 1.0) / k1_hi).max(min_r * (k0_lo - 1.0));
                    let d_hi = (k0_hi * (k0_hi - 1.0) / k1_lo).min(max_r * (k0_hi - 1.0));
                    (d_lo, d_hi.max(d_lo))
                } else if k0_hi <= 1.0 {
                    let k1_lo = (abs_lo * *inv_r2).length();
                    let mut d_lo = max_r * (k0_lo - 1.0);
                    if k1_lo > 1e-9 {
                        let num_lo = if k0_lo <= 0.5 && k0_hi >= 0.5 {
                            -0.25
                        } else {
                            (k0_lo * (k0_lo - 1.0)).min(k0_hi * (k0_hi - 1.0))
                        };
                        d_lo = d_lo.max(num_lo / k1_lo);
                    }
                    let d_hi = min_r * (k0_hi - 1.0);
                    (d_lo, d_hi.max(d_lo))
                } else {
                    (max_r * (k0_lo - 1.0), max_r * (k0_hi - 1.0))
                }
            }
            Kind::Pyramid { base, h, .. } => {
                // Quilez's sdPyramid has a sign discontinuity across the base plane z = -h/2, so
                // 1-Lipschitz center-minus-half-diag is only valid when the box stays strictly on one side.
                let base_z = -*h * 0.5;
                let bdist2 = self.local_aabb.distance_squared_box(min, max);
                if bdist2 > 0.0 && max.z < base_z {
                    let bdist = bdist2.sqrt();
                    let c = (min + max) * 0.5;
                    let hd = ((max - min) * 0.5).length();
                    let dc = self.kind_eval(c);
                    (bdist, (dc + hd).max(bdist))
                } else if min.z <= base_z + 1e-4 && max.z >= base_z - 1e-4 {
                    if bdist2 > 0.0 {
                        let bdist = bdist2.sqrt();
                        (bdist, bdist + (max - min).length())
                    } else {
                        let bound = base.max(*h) + (max - min).length();
                        (-bound, bound)
                    }
                } else {
                    let c = (min + max) * 0.5;
                    let hd = ((max - min) * 0.5).length();
                    let dc = self.kind_eval(c);
                    let lo = if bdist2 > 0.0 { (dc - hd).max(bdist2.sqrt()) } else { dc - hd };
                    let hi = (dc + hd).max(lo);
                    (lo, hi)
                }
            }
            Kind::Capsule { .. } | Kind::Cone { .. } | Kind::Lathe(_) | Kind::Extrude(_, _) => {
                let c = (min + max) * 0.5;
                let hd = ((max - min) * 0.5).length();
                let dc = self.kind_eval(c);
                let bdist2 = self.local_aabb.distance_squared_box(min, max);
                let lo = if bdist2 > 0.0 { (dc - hd).max(bdist2.sqrt()) } else { dc - hd };
                let hi = (dc + hd).max(lo);
                (lo, hi)
            }
            Kind::Union(children, k) => {
                let (mut lo, mut hi) = (f32::INFINITY, f32::INFINITY);
                for c in children {
                    let (clo, chi) = c.eval_interval(min, max);
                    if lo.is_finite() {
                        lo = smin(lo, clo, *k);
                        hi = smin(hi, chi, *k);
                    } else {
                        lo = clo;
                        hi = chi;
                    }
                }
                (lo, hi)
            }
            Kind::Difference(children, k) => {
                let (mut lo, mut hi) = children[0].eval_interval(min, max);
                for c in &children[1..] {
                    let (clo, chi) = c.eval_interval(min, max);
                    lo = smax(lo, -chi, *k);
                    hi = smax(hi, -clo, *k);
                }
                (lo, hi)
            }
            Kind::Intersection(children, k) => {
                let (mut lo, mut hi) = children[0].eval_interval(min, max);
                for c in &children[1..] {
                    let (clo, chi) = c.eval_interval(min, max);
                    lo = smax(lo, clo, *k);
                    hi = smax(hi, chi, *k);
                }
                (lo, hi)
            }
            Kind::Array(c, count, off) => {
                let (mut lo, mut hi) = (f32::INFINITY, f32::INFINITY);
                for i in 0..*count {
                    let shift = *off * i as f32;
                    let (clo, chi) = c.eval_interval(min - shift, max - shift);
                    lo = lo.min(clo);
                    hi = hi.min(chi);
                }
                (lo, hi)
            }
        }
    }

    /// Fast narrow-band classification of axis-aligned box `[min, max]` against `±margin`, with
    /// early-outs for hard unions (`Solid`) and hard differences (`Empty`).
    #[inline]
    pub(crate) fn classify_interval(&self, mut min: Vec3, mut max: Vec3, margin: f32) -> BlockClass {
        if self.mirror.is_none() && !self.has_rot && !self.has_scale && self.hollow <= 0.0 && self.lip == 1.0 {
            if self.has_at {
                min -= self.at;
                max -= self.at;
            }
            match &self.kind {
                Kind::Union(children, k) if *k <= 0.0 && !children.is_empty() => {
                    let mut lo = f32::INFINITY;
                    let mut hi = f32::INFINITY;
                    for c in children {
                        let (clo, chi) = c.eval_interval(min, max);
                        if chi < -margin {
                            return BlockClass::Solid(chi.max(-1e6));
                        }
                        lo = lo.min(clo);
                        hi = hi.min(chi);
                    }
                    return if lo > margin {
                        BlockClass::Empty(lo.min(1e6))
                    } else if hi < -margin {
                        BlockClass::Solid(hi.max(-1e6))
                    } else {
                        BlockClass::Band
                    };
                }
                Kind::Difference(children, k) if *k <= 0.0 && children.len() >= 2 => {
                    let (mut lo, mut hi) = children[0].eval_interval(min, max);
                    if lo > margin {
                        return BlockClass::Empty(lo.min(1e6));
                    }
                    for c in &children[1..] {
                        let (clo, chi) = c.eval_interval(min, max);
                        lo = lo.max(-chi);
                        if lo > margin {
                            return BlockClass::Empty(lo.min(1e6));
                        }
                        hi = hi.max(-clo);
                    }
                    return if hi < -margin {
                        BlockClass::Solid(hi.max(-1e6))
                    } else {
                        BlockClass::Band
                    };
                }
                _ => {}
            }
            let (lo, hi) = self.kind_interval(min, max);
            return if lo > margin {
                BlockClass::Empty(lo.min(1e6))
            } else if hi < -margin {
                BlockClass::Solid(hi.max(-1e6))
            } else {
                BlockClass::Band
            };
        }
        let (lo, hi) = self.eval_interval(min, max);
        if lo > margin {
            BlockClass::Empty(lo.min(1e6))
        } else if hi < -margin {
            BlockClass::Solid(hi.max(-1e6))
        } else {
            BlockClass::Band
        }
    }

    /// Evaluates a full 4×4×4 lattice block starting at `p0` with uniform step `h` into `out`,
    /// where `out[k * 4 + j]` holds the 4 X-samples at `(p0.x + [0,1,2,3]*h, p0.y + j*h, p0.z + k*h)`.
    /// Amortizes CSG interval pruning, coordinate transforms, and radial `sqrt`s across all 64 samples.
    pub(crate) fn eval_block4(&self, p0: Vec3, h: f32, out: &mut [F32x4; 16]) {
        if let Some(a) = self.mirror {
            // Fast path when the entire 4×4×4 block lies on one side of the mirror plane (no fold inside block):
            // mirror is a pure sign flip on axis `a`, so we unwrap cleanly into `eval_block4`!
            let lo_a = p0[a];
            let hi_a = p0[a] + 3.0 * h;
            if lo_a >= 0.0 {
                self.eval_block4_unmirrored(p0, h, out);
                return;
            }
            if hi_a <= 0.0 {
                // Mirror along X, Y, or Z when whole block is <= 0: flip coordinate order along that axis!
                let mut q0 = p0;
                q0[a] = -hi_a;
                let mut tmp = [F32x4::splat(0.0); 16];
                self.eval_block4_unmirrored(q0, h, &mut tmp);
                if a == 0 {
                    for m in 0..16 {
                        out[m] = tmp[m].rev4();
                    }
                } else if a == 1 {
                    for k in 0..4 {
                        let kb = k * 4;
                        out[kb] = tmp[kb + 3];
                        out[kb + 1] = tmp[kb + 2];
                        out[kb + 2] = tmp[kb + 1];
                        out[kb + 3] = tmp[kb];
                    }
                } else {
                    for k in 0..4 {
                        let src_k = (3 - k) * 4;
                        let dst_k = k * 4;
                        for j in 0..4 {
                            out[dst_k + j] = tmp[src_k + j];
                        }
                    }
                }
                return;
            }
            self.eval_block4_via_eval4(p0, h, out);
            return;
        }
        self.eval_block4_unmirrored(p0, h, out);
    }

    fn eval_block4_unmirrored(&self, p0: Vec3, h: f32, out: &mut [F32x4; 16]) {
        let zero = F32x4::splat(0.0);
        let ramp = F32x4::from_array([0.0, 1.0, 2.0, 3.0]);

        if !self.has_rot && !self.has_scale {
            let q0 = if self.has_at { p0 - self.at } else { p0 };
            let handled = match &self.kind {
                Kind::Sphere(r) => {
                    let x = F32x4::splat(q0.x).add(F32x4::splat(h).mul(ramp));
                    let x2 = x.mul(x);
                    let rv = F32x4::splat(*r);
                    let y2 = [0, 1, 2, 3].map(|j| {
                        let y = q0.y + j as f32 * h;
                        y * y
                    });
                    for k in 0..4 {
                        let z = q0.z + k as f32 * h;
                        let z2 = z * z;
                        let kb = k * 4;
                        for j in 0..4 {
                            out[kb + j] = x2.add(F32x4::splat(y2[j] + z2)).sqrt().sub(rv);
                        }
                    }
                    true
                }
                Kind::Cuboid { inner, round } => {
                    let x = F32x4::splat(q0.x).add(F32x4::splat(h).mul(ramp));
                    let qx = x.abs().sub(F32x4::splat(inner.x));
                    let ox = qx.max(zero);
                    let ox2 = ox.mul(ox);
                    let rv = F32x4::splat(*round);
                    let qys = [0, 1, 2, 3].map(|j| (q0.y + j as f32 * h).abs() - inner.y);
                    for k in 0..4 {
                        let qz = (q0.z + k as f32 * h).abs() - inner.z;
                        let oz = qz.max(0.0);
                        let oz2 = oz * oz;
                        let kb = k * 4;
                        for j in 0..4 {
                            let qy = qys[j];
                            let oy = qy.max(0.0);
                            let oyz2 = oy * oy + oz2;
                            let qyz = F32x4::splat(qy.max(qz));
                            out[kb + j] = if oyz2 == 0.0 {
                                qx.max(qyz).sub(rv)
                            } else {
                                ox2.add(F32x4::splat(oyz2)).sqrt().add(qx.max(qyz).min(zero)).sub(rv)
                            };
                        }
                    }
                    true
                }
                Kind::Cylinder { r_in, h_in, round } => {
                    let x = F32x4::splat(q0.x).add(F32x4::splat(h).mul(ramp));
                    let x2 = x.mul(x);
                    let ri = F32x4::splat(*r_in);
                    let rv = F32x4::splat(*round);
                    let mut drs = [zero; 4];
                    for j in 0..4 {
                        let y = q0.y + j as f32 * h;
                        drs[j] = x2.add(F32x4::splat(y * y)).sqrt().sub(ri);
                    }
                    for k in 0..4 {
                        let dz = (q0.z + k as f32 * h).abs() - *h_in;
                        let kb = k * 4;
                        if dz <= 0.0 {
                            let dzv = F32x4::splat(dz);
                            for j in 0..4 {
                                out[kb + j] = drs[j].max(dzv).sub(rv);
                            }
                        } else {
                            let oz2 = F32x4::splat(dz * dz);
                            for j in 0..4 {
                                let or = drs[j].max(zero);
                                out[kb + j] = or.mul_add(or, oz2).sqrt().sub(rv);
                            }
                        }
                    }
                    true
                }
                Kind::Torus(a, b) => {
                    let x = F32x4::splat(q0.x).add(F32x4::splat(h).mul(ramp));
                    let x2 = x.mul(x);
                    let av = F32x4::splat(*a);
                    let bv = F32x4::splat(*b);
                    let mut q2s = [zero; 4];
                    for j in 0..4 {
                        let y = q0.y + j as f32 * h;
                        let q = x2.add(F32x4::splat(y * y)).sqrt().sub(av);
                        q2s[j] = q.mul(q);
                    }
                    for k in 0..4 {
                        let z = q0.z + k as f32 * h;
                        let z2 = F32x4::splat(z * z);
                        let kb = k * 4;
                        for j in 0..4 {
                            out[kb + j] = q2s[j].add(z2).sqrt().sub(bv);
                        }
                    }
                    true
                }
                Kind::Cone { r1, r2, hh, k1, k2, inv_k2_dot } => {
                    let x = F32x4::splat(q0.x).add(F32x4::splat(h).mul(ramp));
                    let x2 = x.mul(x);
                    let mut qxs = [zero; 4];
                    for j in 0..4 {
                        let y = q0.y + j as f32 * h;
                        qxs[j] = x2.add(F32x4::splat(y * y)).sqrt();
                    }
                    let (k1x, k1y, k2x, k2y, ik2) = (
                        F32x4::splat(k1.x),
                        F32x4::splat(k1.y),
                        F32x4::splat(k2.x),
                        F32x4::splat(k2.y),
                        F32x4::splat(*inv_k2_dot),
                    );
                    for k in 0..4 {
                        let qy_s = q0.z + k as f32 * h;
                        let qy = F32x4::splat(qy_s);
                        let r_sel = F32x4::splat(if qy_s < 0.0 { *r1 } else { *r2 });
                        let cay_s = qy_s.abs() - *hh;
                        let cay2 = F32x4::splat(cay_s * cay_s);
                        let cay_neg = cay_s < 0.0;
                        let k1y_qy_k2y = F32x4::splat((k1.y - qy_s) * k2.y);
                        let kb = k * 4;
                        for j in 0..4 {
                            let qx = qxs[j];
                            let cax = qx.sub(qx.min(r_sel));
                            let t = k1x.sub(qx).mul_add(k2x, k1y_qy_k2y).mul(ik2).clamp01();
                            let cbx = k2x.mul_add(t, qx.sub(k1x));
                            let cby = k2y.mul_add(t, qy.sub(k1y));
                            let d = cax.mul_add(cax, cay2).min(cbx.mul_add(cbx, cby.mul(cby))).sqrt();
                            out[kb + j] = if cay_neg {
                                F32x4::select(cbx.lt(zero), d.neg(), d)
                            } else {
                                d
                            };
                        }
                    }
                    true
                }
                Kind::Ellipsoid { inv_r, inv_r2, neg_min_r, .. } => {
                    let x = F32x4::splat(q0.x).add(F32x4::splat(h).mul(ramp));
                    let ax = x.mul(F32x4::splat(inv_r.x));
                    let ax2 = ax.mul(ax);
                    let bx = x.mul(F32x4::splat(inv_r2.x));
                    let bx2 = bx.mul(bx);
                    let (eps, one, nm) = (F32x4::splat(1e-9), F32x4::splat(1.0), F32x4::splat(*neg_min_r));
                    let ay2 = [0, 1, 2, 3].map(|j| {
                        let v = (q0.y + j as f32 * h) * inv_r.y;
                        v * v
                    });
                    let by2 = [0, 1, 2, 3].map(|j| {
                        let v = (q0.y + j as f32 * h) * inv_r2.y;
                        v * v
                    });
                    for k in 0..4 {
                        let az = (q0.z + k as f32 * h) * inv_r.z;
                        let bz = (q0.z + k as f32 * h) * inv_r2.z;
                        let (az2, bz2) = (az * az, bz * bz);
                        let kb = k * 4;
                        for j in 0..4 {
                            let k0 = ax2.add(F32x4::splat(ay2[j] + az2)).sqrt();
                            let k1 = bx2.add(F32x4::splat(by2[j] + bz2)).sqrt();
                            let val = k0.mul(k0.sub(one)).div(k1.max(eps));
                            out[kb + j] = F32x4::select(k1.gt(eps), val, nm);
                        }
                    }
                    true
                }
                Kind::Lathe(edges) => {
                    let x = F32x4::splat(q0.x).add(F32x4::splat(h).mul(ramp));
                    let x2 = x.mul(x);
                    let mut rxs = [zero; 4];
                    for j in 0..4 {
                        let y = q0.y + j as f32 * h;
                        rxs[j] = x2.add(F32x4::splat(y * y)).sqrt();
                    }
                    for k in 0..4 {
                        let zv = F32x4::splat(q0.z + k as f32 * h);
                        let kb = k * 4;
                        for j in 0..4 {
                            out[kb + j] = polygon4(rxs[j], zv, edges);
                        }
                    }
                    true
                }
                Kind::Extrude(edges, hh) => {
                    let x = F32x4::splat(q0.x).add(F32x4::splat(h).mul(ramp));
                    let mut wxs = [zero; 4];
                    for j in 0..4 {
                        wxs[j] = polygon4(x, F32x4::splat(q0.y + j as f32 * h), edges);
                    }
                    for k in 0..4 {
                        let wy = (q0.z + k as f32 * h).abs() - *hh;
                        let kb = k * 4;
                        if wy <= 0.0 {
                            let wyv = F32x4::splat(wy);
                            for j in 0..4 {
                                out[kb + j] = wxs[j].max(wyv);
                            }
                        } else {
                            let oy2 = F32x4::splat(wy * wy);
                            for j in 0..4 {
                                let ox = wxs[j].max(zero);
                                out[kb + j] = ox.mul_add(ox, oy2).sqrt();
                            }
                        }
                    }
                    true
                }
                Kind::Union(children, k) => {
                    let m = children.len();
                    if m == 0 {
                        out.fill(F32x4::splat(f32::INFINITY));
                    } else if m == 1 {
                        children[0].eval_block4(q0, h, out);
                    } else {
                        let bmin = q0;
                        let bmax = q0 + Vec3::splat(3.0 * h);
                        let k = *k;
                        let mut stack_iv = [(0.0f32, 0.0f32); 32];
                        let mut heap_iv = Vec::new();
                        let ivs: &mut [(f32, f32)] = if m <= 32 {
                            &mut stack_iv[..m]
                        } else {
                            heap_iv.resize(m, (0.0, 0.0));
                            &mut heap_iv
                        };
                        let mut best_max = f32::INFINITY;
                        let mut best_idx = 0usize;
                        for (i, c) in children.iter().enumerate() {
                            let iv = c.eval_interval(bmin, bmax);
                            ivs[i] = iv;
                            if iv.1 < best_max {
                                best_max = iv.1;
                                best_idx = i;
                            }
                        }
                        if k <= 0.0 {
                            children[best_idx].eval_block4(q0, h, out);
                            let mut any_other = false;
                            for (i, &(c_min, _)) in ivs.iter().enumerate() {
                                if i != best_idx && c_min < best_max {
                                    any_other = true;
                                    break;
                                }
                            }
                            if any_other {
                                let mut out_max = block_max(out);
                                let mut tmp = [zero; 16];
                                for (i, c) in children.iter().enumerate() {
                                    if i == best_idx || ivs[i].0 >= out_max {
                                        continue;
                                    }
                                    c.eval_block4(q0, h, &mut tmp);
                                    out_max = block_combine_min(out, &tmp);
                                }
                            }
                        } else {
                            let cutoff = best_max + 2.0 * k;
                            let mut active_count = 0usize;
                            let mut last_active = 0usize;
                            for (i, &(c_min, _)) in ivs.iter().enumerate() {
                                if i == best_idx || c_min < cutoff {
                                    active_count += 1;
                                    last_active = i;
                                }
                            }
                            if active_count <= 1 {
                                children[last_active].eval_block4(q0, h, out);
                            } else {
                                let mut tmp = [zero; 16];
                                let mut first = true;
                                let mut out_max = f32::INFINITY;
                                let k4 = F32x4::splat(k);
                                let ik4 = F32x4::splat(1.0 / k);
                                for (i, c) in children.iter().enumerate() {
                                    let c_min = ivs[i].0;
                                    if i != best_idx && (c_min >= cutoff || (!first && c_min >= out_max + k)) {
                                        continue;
                                    }
                                    if first {
                                        c.eval_block4(q0, h, out);
                                        out_max = block_max(out);
                                        first = false;
                                    } else {
                                        c.eval_block4(q0, h, &mut tmp);
                                        out_max = block_combine_smin(out, &tmp, k4, ik4);
                                    }
                                }
                            }
                        }
                    }
                    true
                }
                Kind::Difference(children, k) => {
                    let m = children.len();
                    if m == 0 {
                        out.fill(F32x4::splat(f32::INFINITY));
                    } else if m == 1 {
                        children[0].eval_block4(q0, h, out);
                    } else {
                        let k = *k;
                        let bmin = q0;
                        let bmax = q0 + Vec3::splat(3.0 * h);
                        // Fast fused path for coaxial hollow cylinders (e.g. cup/mug/pipe): shares radial sqrt!
                        let coaxial_cyls = m == 2
                            && k <= 0.0
                            && !children[0].has_rot && !children[0].has_scale && children[0].mirror.is_none() && children[0].hollow <= 0.0 && children[0].lip == 1.0
                            && !children[1].has_rot && !children[1].has_scale && children[1].mirror.is_none() && children[1].hollow <= 0.0 && children[1].lip == 1.0
                            && children[0].at.x == children[1].at.x
                            && children[0].at.y == children[1].at.y;
                        if coaxial_cyls {
                            if let (
                                Kind::Cylinder { r_in: r0, h_in: h0, round: rd0 },
                                Kind::Cylinder { r_in: r1, h_in: h1, round: rd1 },
                            ) = (&children[0].kind, &children[1].kind)
                            {
                                let c0 = q0 - children[0].at;
                                let c1_z = q0.z - children[1].at.z;
                                let x = F32x4::splat(c0.x).add(F32x4::splat(h).mul(ramp));
                                let x2 = x.mul(x);
                                let mut rxy = [zero; 4];
                                for j in 0..4 {
                                    let y = c0.y + j as f32 * h;
                                    rxy[j] = x2.add(F32x4::splat(y * y)).sqrt();
                                }
                                let (r0v, rd0v) = (F32x4::splat(*r0), F32x4::splat(*rd0));
                                let (r1v, rd1v) = (F32x4::splat(*r1), F32x4::splat(*rd1));
                                let mut dr0 = [zero; 4];
                                let mut dr1 = [zero; 4];
                                for j in 0..4 {
                                    dr0[j] = rxy[j].sub(r0v);
                                    dr1[j] = rxy[j].sub(r1v);
                                }
                                for k_i in 0..4 {
                                    let dz0 = (c0.z + k_i as f32 * h).abs() - *h0;
                                    let dz1 = (c1_z + k_i as f32 * h).abs() - *h1;
                                    let kb = k_i * 4;
                                    for j in 0..4 {
                                        let d0 = if dz0 <= 0.0 {
                                            dr0[j].max(F32x4::splat(dz0)).sub(rd0v)
                                        } else {
                                            let or = dr0[j].max(zero);
                                            or.mul_add(or, F32x4::splat(dz0 * dz0)).sqrt().sub(rd0v)
                                        };
                                        let d1 = if dz1 <= 0.0 {
                                            dr1[j].max(F32x4::splat(dz1)).sub(rd1v)
                                        } else {
                                            let or = dr1[j].max(zero);
                                            or.mul_add(or, F32x4::splat(dz1 * dz1)).sqrt().sub(rd1v)
                                        };
                                        out[kb + j] = d0.max(d1.neg());
                                    }
                                }
                                self.apply_lip_hollow_block4(out);
                                return;
                            }
                        }
                        children[0].eval_block4(q0, h, out);
                        let mut out_min = block_min(out);
                        let mut tmp = [zero; 16];
                        let k4 = F32x4::splat(k);
                        let ik4 = F32x4::splat(if k > 0.0 { 1.0 / k } else { 0.0 });
                        for c in &children[1..] {
                            let thr = k - out_min;
                            let d2 = c.aabb.distance_squared_box(bmin, bmax);
                            if thr <= 0.0 {
                                if d2 > 0.0 {
                                    continue;
                                }
                            } else {
                                let reach = thr / c.aabb_scale;
                                if d2 >= reach * reach {
                                    continue;
                                }
                            }
                            if c.eval_interval(bmin, bmax).0 >= thr {
                                continue;
                            }
                            c.eval_block4(q0, h, &mut tmp);
                            out_min = if k <= 0.0 {
                                block_combine_max_neg(out, &tmp)
                            } else {
                                block_combine_smax_neg(out, &tmp, k4, ik4)
                            };
                        }
                    }
                    true
                }
                Kind::Intersection(children, k) => {
                    let m = children.len();
                    if m == 0 {
                        out.fill(F32x4::splat(f32::INFINITY));
                    } else if m == 1 {
                        children[0].eval_block4(q0, h, out);
                    } else {
                        let k = *k;
                        let bmin = q0;
                        let bmax = q0 + Vec3::splat(3.0 * h);
                        let mut stack_iv = [(0.0f32, 0.0f32); 32];
                        let mut heap_iv = Vec::new();
                        let ivs: &mut [(f32, f32)] = if m <= 32 {
                            &mut stack_iv[..m]
                        } else {
                            heap_iv.resize(m, (0.0, 0.0));
                            &mut heap_iv
                        };
                        let mut best_min = f32::NEG_INFINITY;
                        let mut best_idx = 0usize;
                        for (i, c) in children.iter().enumerate() {
                            let iv = c.eval_interval(bmin, bmax);
                            ivs[i] = iv;
                            if iv.0 > best_min {
                                best_min = iv.0;
                                best_idx = i;
                            }
                        }
                        if k <= 0.0 {
                            children[best_idx].eval_block4(q0, h, out);
                            let mut out_min = block_min(out);
                            let mut tmp = [zero; 16];
                            for (i, c) in children.iter().enumerate() {
                                if i == best_idx || ivs[i].1 <= out_min {
                                    continue;
                                }
                                c.eval_block4(q0, h, &mut tmp);
                                out_min = block_combine_max(out, &tmp);
                            }
                        } else {
                            let cutoff = best_min - 2.0 * k;
                            let mut active_count = 0usize;
                            let mut last_active = 0usize;
                            for (i, &(_, c_max)) in ivs.iter().enumerate() {
                                if i == best_idx || c_max > cutoff {
                                    active_count += 1;
                                    last_active = i;
                                }
                            }
                            if active_count <= 1 {
                                children[last_active].eval_block4(q0, h, out);
                            } else {
                                let mut tmp = [zero; 16];
                                let mut first = true;
                                let mut out_min = f32::NEG_INFINITY;
                                let k4 = F32x4::splat(k);
                                let ik4 = F32x4::splat(1.0 / k);
                                for (i, c) in children.iter().enumerate() {
                                    let c_max = ivs[i].1;
                                    if i != best_idx && (c_max <= cutoff || (!first && c_max <= out_min - k)) {
                                        continue;
                                    }
                                    if first {
                                        c.eval_block4(q0, h, out);
                                        out_min = block_min(out);
                                        first = false;
                                    } else {
                                        c.eval_block4(q0, h, &mut tmp);
                                        out_min = block_combine_smax(out, &tmp, k4, ik4);
                                    }
                                }
                            }
                        }
                    }
                    true
                }
                Kind::Array(c, count, off) => {
                    let count = *count as usize;
                    if count == 0 {
                        out.fill(F32x4::splat(f32::INFINITY));
                    } else if count == 1 {
                        c.eval_block4(q0, h, out);
                    } else {
                        let bmin = q0;
                        let bmax = q0 + Vec3::splat(3.0 * h);
                        let mut stack_iv = [(0.0f32, 0.0f32); 64];
                        let mut heap_iv = Vec::new();
                        let ivs: &mut [(f32, f32)] = if count <= 64 {
                            &mut stack_iv[..count]
                        } else {
                            heap_iv.resize(count, (0.0, 0.0));
                            &mut heap_iv
                        };
                        let mut best_max = f32::INFINITY;
                        let mut best_idx = 0usize;
                        for (i, slot) in ivs.iter_mut().enumerate() {
                            let shift = *off * i as f32;
                            let iv = c.eval_interval(bmin - shift, bmax - shift);
                            *slot = iv;
                            if iv.1 < best_max {
                                best_max = iv.1;
                                best_idx = i;
                            }
                        }
                        c.eval_block4(q0 - *off * best_idx as f32, h, out);
                        let mut out_max = block_max(out);
                        let mut tmp = [zero; 16];
                        for (i, &(c_min, _)) in ivs.iter().enumerate() {
                            if i == best_idx || c_min >= out_max {
                                continue;
                            }
                            c.eval_block4(q0 - *off * i as f32, h, &mut tmp);
                            out_max = block_combine_min(out, &tmp);
                        }
                    }
                    true
                }
                _ => false,
            };
            if handled {
                self.apply_lip_hollow_block4(out);
                return;
            }
        } else if !self.has_scale {
            // Fast path for 90°-X-rotated Torus (cup/mug/teapot/bucket handles): hoists radial sqrt across Y!
            if let (Some(perm), Kind::Torus(a, b)) = (self.axis_perm, &self.kind) {
                if perm.is_x_rot90() {
                    let q0 = if self.has_at { p0 - self.at } else { p0 };
                    let lx = if perm.neg[0] {
                        F32x4::splat(-q0.x).sub(F32x4::splat(h).mul(ramp))
                    } else {
                        F32x4::splat(q0.x).add(F32x4::splat(h).mul(ramp))
                    };
                    let lx2 = lx.mul(lx);
                    let av = F32x4::splat(*a);
                    let bv = F32x4::splat(*b);
                    let mut lz2 = [0.0f32; 4];
                    for j in 0..4 {
                        let v = q0.y + j as f32 * h;
                        lz2[j] = v * v;
                    }
                    for k in 0..4 {
                        let ly = q0.z + k as f32 * h;
                        let q = lx2.add(F32x4::splat(ly * ly)).sqrt().sub(av);
                        let q2 = q.mul(q);
                        let kb = k * 4;
                        for j in 0..4 {
                            out[kb + j] = q2.add(F32x4::splat(lz2[j])).sqrt().sub(bv);
                        }
                    }
                    self.apply_lip_hollow_block4(out);
                    return;
                }
            }
        }

        // General linear transform (hoist affine matrix/scale once for all 64 samples).
        let mut q0 = if self.has_at { p0 - self.at } else { p0 };
        let (mut dx, mut dy, mut dz) = (Vec3::new(h, 0.0, 0.0), Vec3::new(0.0, h, 0.0), Vec3::new(0.0, 0.0, h));
        if self.has_rot {
            if let Some(perm) = self.axis_perm {
                q0 = perm.apply_vec3(q0);
                dx = perm.apply_vec3(dx);
                dy = perm.apply_vec3(dy);
                dz = perm.apply_vec3(dz);
            } else {
                q0 = self.inv_mat * q0;
                dx = self.inv_mat * dx;
                dy = self.inv_mat * dy;
                dz = self.inv_mat * dz;
            }
        }
        if self.has_scale {
            q0 *= self.inv_scale_stretch;
            dx *= self.inv_scale_stretch;
            dy *= self.inv_scale_stretch;
            dz *= self.inv_scale_stretch;
        }
        let rx0 = F32x4::splat(q0.x).add(F32x4::splat(dx.x).mul(ramp));
        let ry0 = F32x4::splat(q0.y).add(F32x4::splat(dx.y).mul(ramp));
        let rz0 = F32x4::splat(q0.z).add(F32x4::splat(dx.z).mul(ramp));
        let (dyx, dyy, dyz) = (F32x4::splat(dy.x), F32x4::splat(dy.y), F32x4::splat(dy.z));
        let (dzx, dzy, dzz) = (F32x4::splat(dz.x), F32x4::splat(dz.y), F32x4::splat(dz.z));
        let (mut kx, mut ky, mut kz) = (rx0, ry0, rz0);
        for k in 0..4 {
            let (mut x, mut y, mut z) = (kx, ky, kz);
            let kb = k * 4;
            for j in 0..4 {
                out[kb + j] = self.kind_eval4(x, y, z);
                x = x.add(dyx);
                y = y.add(dyy);
                z = z.add(dyz);
            }
            kx = kx.add(dzx);
            ky = ky.add(dzy);
            kz = kz.add(dzz);
        }
        self.apply_lip_hollow_block4(out);
    }

    #[inline(always)]
    fn apply_lip_hollow_block4(&self, out: &mut [F32x4; 16]) {
        if self.hollow > 0.0 {
            let lip4 = F32x4::splat(self.lip);
            let hh4 = F32x4::splat(self.hollow * 0.5);
            for v in out.iter_mut() {
                *v = v.mul(lip4).add(hh4).abs().sub(hh4);
            }
        } else if self.lip != 1.0 {
            let lip4 = F32x4::splat(self.lip);
            for v in out.iter_mut() {
                *v = v.mul(lip4);
            }
        }
    }

    fn eval_block4_via_eval4(&self, p0: Vec3, h: f32, out: &mut [F32x4; 16]) {
        let ramp = F32x4::from_array([0.0, 1.0, 2.0, 3.0]);
        let x = F32x4::splat(p0.x).add(F32x4::splat(h).mul(ramp));
        for k in 0..4 {
            let zv = F32x4::splat(p0.z + k as f32 * h);
            let kb = k * 4;
            for j in 0..4 {
                let yv = F32x4::splat(p0.y + j as f32 * h);
                out[kb + j] = self.eval4(x, yv, zv);
            }
        }
    }

    /// Signed distances at `out.len()` samples `p0 + (i * dx, 0, 0)` (parent's frame), the batched
    /// counterpart of [`Compiled::eval`]. Transforms are applied once per row and leaf primitives run
    /// 4-wide in NEON SIMD; combinators prune strictly dominated children per row via interval arithmetic.
    pub fn eval_row(&self, p0: Vec3, dx: f32, out: &mut [f32]) {
        let mut start = 0usize;
        for chunk in out.chunks_mut(ROW) {
            self.row(p0 + Vec3::new(dx * start as f32, 0.0, 0.0), Vec3::new(dx, 0.0, 0.0), chunk);
            start += ROW;
        }
    }

    /// Row evaluation, `out.len() <= ROW`, samples at `p0 + dir * i`.
    fn row(&self, mut p0: Vec3, dir: Vec3, out: &mut [f32]) {
        if out.is_empty() {
            return;
        }
        let n = out.len();
        if let Some(a) = self.mirror {
            if dir[a] != 0.0 {
                // |p[a]| is piecewise linear along the row: split it where the coordinate changes sign.
                let (pa, da) = (p0[a], dir[a]);
                let first_nonneg = pa >= 0.0;
                let m = (1..n).find(|&i| ((pa + da * i as f32) >= 0.0) != first_nonneg).unwrap_or(n);
                let flip = |p: &mut Vec3, d: &mut Vec3, neg: bool| {
                    if neg {
                        p[a] = -p[a];
                        d[a] = -d[a];
                    }
                };
                let (mut q0, mut qd) = (p0, dir);
                flip(&mut q0, &mut qd, !first_nonneg);
                self.row_lin(q0, qd, &mut out[..m]);
                if m < n {
                    let (mut q1, mut qd1) = (p0 + dir * m as f32, dir);
                    flip(&mut q1, &mut qd1, first_nonneg);
                    self.row_lin(q1, qd1, &mut out[m..]);
                }
                return;
            }
            p0[a] = p0[a].abs();
        }
        self.row_lin(p0, dir, out);
    }

    /// Row evaluation once the mirror fold is linear along the row.
    fn row_lin(&self, mut p0: Vec3, mut dir: Vec3, out: &mut [f32]) {
        if self.has_at {
            p0 -= self.at;
        }
        if self.has_rot {
            p0 = self.inv_mat * p0;
            dir = self.inv_mat * dir;
        }
        if self.has_scale {
            p0 *= self.inv_scale_stretch;
            dir *= self.inv_scale_stretch;
        }
        self.kind_row(p0, dir, out);
        let lip = self.lip;
        if self.hollow > 0.0 {
            let hh = self.hollow * 0.5;
            let lip4 = F32x4::splat(lip);
            let hh4 = F32x4::splat(hh);
            let mut i = 0;
            while i + 4 <= out.len() {
                let v = F32x4::load(&out[i..]);
                v.mul(lip4).add(hh4).abs().sub(hh4).store(&mut out[i..]);
                i += 4;
            }
            for o in &mut out[i..] {
                *o = (*o * lip + hh).abs() - hh;
            }
        } else if lip != 1.0 {
            let lip4 = F32x4::splat(lip);
            let mut i = 0;
            while i + 4 <= out.len() {
                F32x4::load(&out[i..]).mul(lip4).store(&mut out[i..]);
                i += 4;
            }
            for o in &mut out[i..] {
                *o *= lip;
            }
        }
    }

    /// Runs `f(x4, y4, z4)` over `out` along `p0 + dir * i`, 4 lanes at a time.
    #[inline(always)]
    fn eval_lanes4(p0: Vec3, dir: Vec3, out: &mut [f32], mut f: impl FnMut(F32x4, F32x4, F32x4) -> F32x4) {
        let ramp = F32x4::from_array([0.0, 1.0, 2.0, 3.0]);
        let mut x = F32x4::splat(p0.x).add(F32x4::splat(dir.x).mul(ramp));
        let mut y = F32x4::splat(p0.y).add(F32x4::splat(dir.y).mul(ramp));
        let mut z = F32x4::splat(p0.z).add(F32x4::splat(dir.z).mul(ramp));
        let sx = F32x4::splat(dir.x * 4.0);
        let sy = F32x4::splat(dir.y * 4.0);
        let sz = F32x4::splat(dir.z * 4.0);
        let mut i = 0usize;
        let n = out.len();
        while i + 4 <= n {
            f(x, y, z).store(&mut out[i..]);
            x = x.add(sx);
            y = y.add(sy);
            z = z.add(sz);
            i += 4;
        }
        if i < n {
            let tail = f(x, y, z).to_array();
            let rem = n - i;
            out[i..n].copy_from_slice(&tail[..rem]);
        }
    }

    /// Runs `f(x4)` over `out` along `x = x0 + dx * i` (when `dir.y == 0 && dir.z == 0`), 4 lanes at a time.
    #[inline(always)]
    fn eval_lanes4_x(x0: f32, dx: f32, out: &mut [f32], mut f: impl FnMut(F32x4) -> F32x4) {
        let ramp = F32x4::from_array([0.0, 1.0, 2.0, 3.0]);
        let mut x = F32x4::splat(x0).add(F32x4::splat(dx).mul(ramp));
        let sx = F32x4::splat(dx * 4.0);
        let mut i = 0usize;
        let n = out.len();
        while i + 4 <= n {
            f(x).store(&mut out[i..]);
            x = x.add(sx);
            i += 4;
        }
        if i < n {
            let tail = f(x).to_array();
            out[i..n].copy_from_slice(&tail[..n - i]);
        }
    }

    fn kind_row(&self, p0: Vec3, dir: Vec3, out: &mut [f32]) {
        let n = out.len();
        let zero = F32x4::splat(0.0);
        let x_only = dir.y == 0.0 && dir.z == 0.0;
        match &self.kind {
            Kind::Sphere(r) => {
                let rv = F32x4::splat(*r);
                if x_only {
                    let yz2 = F32x4::splat(p0.y * p0.y + p0.z * p0.z);
                    Self::eval_lanes4_x(p0.x, dir.x, out, |x| x.mul(x).add(yz2).sqrt().sub(rv));
                } else {
                    Self::eval_lanes4(p0, dir, out, |x, y, z| x.mul(x).add(y.mul(y)).add(z.mul(z)).sqrt().sub(rv));
                }
            }
            Kind::Cuboid { inner, round, .. } => {
                let (ix, iy, iz, rv) = (F32x4::splat(inner.x), F32x4::splat(inner.y), F32x4::splat(inner.z), F32x4::splat(*round));
                if x_only {
                    let qy = p0.y.abs() - inner.y;
                    let qz = p0.z.abs() - inner.z;
                    let qyz = F32x4::splat(qy.max(qz));
                    let oy = qy.max(0.0);
                    let oz = qz.max(0.0);
                    let oyz2 = oy * oy + oz * oz;
                    if oyz2 == 0.0 {
                        // Inside Y and Z bounds: no sqrt needed!
                        Self::eval_lanes4_x(p0.x, dir.x, out, |x| x.abs().sub(ix).max(qyz).sub(rv));
                    } else {
                        let oyz2v = F32x4::splat(oyz2);
                        Self::eval_lanes4_x(p0.x, dir.x, out, |x| {
                            let qx = x.abs().sub(ix);
                            let ox = qx.max(zero);
                            ox.mul(ox).add(oyz2v).sqrt().add(qx.max(qyz).min(zero)).sub(rv)
                        });
                    }
                } else {
                    Self::eval_lanes4(p0, dir, out, |x, y, z| {
                        let qx = x.abs().sub(ix);
                        let qy = y.abs().sub(iy);
                        let qz = z.abs().sub(iz);
                        let ox = qx.max(zero);
                        let oy = qy.max(zero);
                        let oz = qz.max(zero);
                        ox.mul(ox).add(oy.mul(oy)).add(oz.mul(oz)).sqrt().add(qx.max(qy).max(qz).min(zero)).sub(rv)
                    });
                }
            }
            Kind::Cylinder { r_in, h_in, round, .. } => {
                let (ri, hi, rv) = (F32x4::splat(*r_in), F32x4::splat(*h_in), F32x4::splat(*round));
                if x_only {
                    let y2 = F32x4::splat(p0.y * p0.y);
                    let dz = p0.z.abs() - *h_in;
                    let dzv = F32x4::splat(dz);
                    if dz <= 0.0 {
                        // Within cylinder height: second sqrt vanishes identically!
                        Self::eval_lanes4_x(p0.x, dir.x, out, |x| {
                            x.mul(x).add(y2).sqrt().sub(ri).max(dzv).sub(rv)
                        });
                    } else {
                        let oz2 = F32x4::splat(dz * dz);
                        Self::eval_lanes4_x(p0.x, dir.x, out, |x| {
                            let dr = x.mul(x).add(y2).sqrt().sub(ri);
                            let or = dr.max(zero);
                            or.mul(or).add(oz2).sqrt().sub(rv)
                        });
                    }
                } else {
                    Self::eval_lanes4(p0, dir, out, |x, y, z| {
                        let dr = x.mul(x).add(y.mul(y)).sqrt().sub(ri);
                        let dz = z.abs().sub(hi);
                        let or = dr.max(zero);
                        let oz = dz.max(zero);
                        dr.max(dz).min(zero).add(or.mul(or).add(oz.mul(oz)).sqrt()).sub(rv)
                    });
                }
            }
            Kind::Cone { r1, r2, hh, k1, k2, inv_k2_dot } => {
                let (r1v, r2v, hhv) = (F32x4::splat(*r1), F32x4::splat(*r2), F32x4::splat(*hh));
                let (k1x, k1y, k2x, k2y, ik2) = (
                    F32x4::splat(k1.x),
                    F32x4::splat(k1.y),
                    F32x4::splat(k2.x),
                    F32x4::splat(k2.y),
                    F32x4::splat(*inv_k2_dot),
                );
                Self::eval_lanes4(p0, dir, out, |x, y, z| {
                    let qx = x.mul(x).add(y.mul(y)).sqrt();
                    let qy = z;
                    let r_sel = F32x4::select(qy.lt(zero), r1v, r2v);
                    let cax = qx.sub(qx.min(r_sel));
                    let cay = qy.abs().sub(hhv);
                    let t = k1x.sub(qx).mul(k2x).add(k1y.sub(qy).mul(k2y)).mul(ik2).clamp01();
                    let cbx = qx.sub(k1x).add(k2x.mul(t));
                    let cby = qy.sub(k1y).add(k2y.mul(t));
                    let d = cax.mul(cax).add(cay.mul(cay)).min(cbx.mul(cbx).add(cby.mul(cby))).sqrt();
                    F32x4::select(cbx.lt(zero).and(cay.lt(zero)), d.neg(), d)
                });
            }
            Kind::Torus(a, b) => {
                let (av, bv) = (F32x4::splat(*a), F32x4::splat(*b));
                if x_only {
                    let y2 = F32x4::splat(p0.y * p0.y);
                    let z2 = F32x4::splat(p0.z * p0.z);
                    Self::eval_lanes4_x(p0.x, dir.x, out, |x| {
                        let q = x.mul(x).add(y2).sqrt().sub(av);
                        q.mul(q).add(z2).sqrt().sub(bv)
                    });
                } else {
                    Self::eval_lanes4(p0, dir, out, |x, y, z| {
                        let q = x.mul(x).add(y.mul(y)).sqrt().sub(av);
                        q.mul(q).add(z.mul(z)).sqrt().sub(bv)
                    });
                }
            }
            Kind::Capsule { a, ba, inv_ba_dot, r, .. } => {
                let (ax, ay, az) = (F32x4::splat(a.x), F32x4::splat(a.y), F32x4::splat(a.z));
                let (bx, by, bz) = (F32x4::splat(ba.x), F32x4::splat(ba.y), F32x4::splat(ba.z));
                let (inv, rv) = (F32x4::splat(*inv_ba_dot), F32x4::splat(*r));
                Self::eval_lanes4(p0, dir, out, |x, y, z| {
                    let px = x.sub(ax);
                    let py = y.sub(ay);
                    let pz = z.sub(az);
                    let t = px.mul(bx).add(py.mul(by)).add(pz.mul(bz)).mul(inv).clamp01();
                    let qx = px.sub(bx.mul(t));
                    let qy = py.sub(by.mul(t));
                    let qz = pz.sub(bz.mul(t));
                    qx.mul(qx).add(qy.mul(qy)).add(qz.mul(qz)).sqrt().sub(rv)
                });
            }
            Kind::Ellipsoid { inv_r, inv_r2, neg_min_r, .. } => {
                let (rxv, ryv, rzv) = (F32x4::splat(inv_r.x), F32x4::splat(inv_r.y), F32x4::splat(inv_r.z));
                let (r2x, r2y, r2z) = (F32x4::splat(inv_r2.x), F32x4::splat(inv_r2.y), F32x4::splat(inv_r2.z));
                let (eps, one, nm) = (F32x4::splat(1e-9), F32x4::splat(1.0), F32x4::splat(*neg_min_r));
                if x_only {
                    let yz0 = F32x4::splat((p0.y * inv_r.y).powi(2) + (p0.z * inv_r.z).powi(2));
                    let yz1 = F32x4::splat((p0.y * inv_r2.y).powi(2) + (p0.z * inv_r2.z).powi(2));
                    Self::eval_lanes4_x(p0.x, dir.x, out, |x| {
                        let ax = x.mul(rxv);
                        let k0 = ax.mul(ax).add(yz0).sqrt();
                        let bx = x.mul(r2x);
                        let k1 = bx.mul(bx).add(yz1).sqrt();
                        let val = k0.mul(k0.sub(one)).div(k1.max(eps));
                        F32x4::select(k1.gt(eps), val, nm)
                    });
                } else {
                    Self::eval_lanes4(p0, dir, out, |x, y, z| {
                        let ax = x.mul(rxv);
                        let ay = y.mul(ryv);
                        let az = z.mul(rzv);
                        let k0 = ax.mul(ax).add(ay.mul(ay)).add(az.mul(az)).sqrt();
                        let bx = x.mul(r2x);
                        let by = y.mul(r2y);
                        let bz = z.mul(r2z);
                        let k1 = bx.mul(bx).add(by.mul(by)).add(bz.mul(bz)).sqrt();
                        let val = k0.mul(k0.sub(one)).div(k1.max(eps));
                        F32x4::select(k1.gt(eps), val, nm)
                    });
                }
            }
            Kind::Pyramid { base, h, inv_s, h_over_s, m2, inv_m2_p, inv_m2 } => {
                let (is, hh, hs, m2v, imp, im2, bv) = (
                    F32x4::splat(*inv_s),
                    F32x4::splat(*h * 0.5),
                    F32x4::splat(*h_over_s),
                    F32x4::splat(*m2),
                    F32x4::splat(*inv_m2_p),
                    F32x4::splat(*inv_m2),
                    F32x4::splat(*base),
                );
                let (half, neg1, pos1) = (F32x4::splat(0.5), F32x4::splat(-1.0), F32x4::splat(1.0));
                Self::eval_lanes4(p0, dir, out, |x, y, z| {
                    let px = x.mul(is);
                    let py = z.add(hh).mul(is);
                    let pz = y.mul(is);
                    let ax = px.abs();
                    let az = pz.abs();
                    let xx = ax.max(az).sub(half);
                    let zz = ax.min(az).sub(half);
                    let qx = zz;
                    let qy = hs.mul(py).sub(half.mul(xx));
                    let qz = hs.mul(xx).add(half.mul(py));
                    let s = qx.neg().max(zero);
                    let t = qy.sub(half.mul(zz)).mul(imp).clamp01();
                    let qs = qx.add(s);
                    let a = m2v.mul(qs).mul(qs).add(qy.mul(qy));
                    let qt = qx.add(half.mul(t));
                    let qym = qy.sub(m2v.mul(t));
                    let b = m2v.mul(qt).mul(qt).add(qym.mul(qym));
                    let cond = qy.min(qx.neg().mul(m2v).sub(qy.mul(half)));
                    let d2 = F32x4::select(cond.gt(zero), zero, a.min(b));
                    let dist = d2.add(qz.mul(qz)).mul(im2).sqrt();
                    let sgn = F32x4::select(qz.max(py.neg()).le(zero), neg1, pos1);
                    dist.mul(sgn).mul(bv)
                });
            }
            Kind::Lathe(edges) => {
                Self::eval_lanes4(p0, dir, out, |x, y, z| polygon4(x.mul(x).add(y.mul(y)).sqrt(), z, edges));
            }
            Kind::Extrude(edges, hh) => {
                let hhv = F32x4::splat(*hh);
                Self::eval_lanes4(p0, dir, out, |x, y, z| {
                    let wx = polygon4(x, y, edges);
                    let wy = z.abs().sub(hhv);
                    let ox = wx.max(zero);
                    let oy = wy.max(zero);
                    wx.max(wy).min(zero).add(ox.mul(ox).add(oy.mul(oy)).sqrt())
                });
            }
            Kind::Union(children, k) => {
                let m = children.len();
                if m == 0 {
                    out.fill(f32::INFINITY);
                    return;
                }
                if m == 1 {
                    children[0].row(p0, dir, out);
                    return;
                }
                let k = *k;
                let (lo, hi) = row_box(p0, dir, n);
                if k <= 0.0 {
                    // Hard union: fast AABB-distance ordering avoids recursive eval_interval on every row.
                    let mut stack_d2 = [0.0f32; 32];
                    let mut heap_d2 = Vec::new();
                    let d2s: &mut [f32] = if m <= 32 {
                        &mut stack_d2[..m]
                    } else {
                        heap_d2.resize(m, 0.0);
                        &mut heap_d2
                    };
                    let mut best_d2 = f32::INFINITY;
                    let mut best_idx = 0usize;
                    for (i, c) in children.iter().enumerate() {
                        let d2 = c.aabb.distance_squared_box(lo, hi);
                        d2s[i] = d2;
                        if d2 < best_d2 {
                            best_d2 = d2;
                            best_idx = i;
                        }
                    }
                    children[best_idx].row(p0, dir, out);
                    let mut out_max = slice_max(out);
                    let mut tmp = [0f32; ROW];
                    for (i, c) in children.iter().enumerate() {
                        if i == best_idx {
                            continue;
                        }
                        let d2 = d2s[i];
                        if out_max <= 0.0 {
                            if d2 > 0.0 {
                                continue;
                            }
                        } else {
                            let reach = out_max / c.aabb_scale;
                            if d2 >= reach * reach {
                                continue;
                            }
                        }
                        if matches!(c.kind, Kind::Union(_, _) | Kind::Difference(_, _) | Kind::Intersection(_, _) | Kind::Array(_, _, _))
                            && c.eval_interval(lo, hi).0 >= out_max
                        {
                            continue;
                        }
                        c.row(p0, dir, &mut tmp[..n]);
                        out_max = combine_min(out, &tmp[..n]);
                    }
                } else {
                    // Smooth union: preserve child order for overlapping blends, prune strictly dominated children.
                    let mut stack_iv = [(0.0f32, 0.0f32); 32];
                    let mut heap_iv = Vec::new();
                    let ivs: &mut [(f32, f32)] = if m <= 32 {
                        &mut stack_iv[..m]
                    } else {
                        heap_iv.resize(m, (0.0, 0.0));
                        &mut heap_iv
                    };
                    let mut best_max = f32::INFINITY;
                    let mut best_idx = 0usize;
                    for (i, c) in children.iter().enumerate() {
                        let iv = c.eval_interval(lo, hi);
                        ivs[i] = iv;
                        if iv.1 < best_max {
                            best_max = iv.1;
                            best_idx = i;
                        }
                    }
                    let cutoff = best_max + 2.0 * k;
                    let mut active_count = 0usize;
                    let mut last_active = 0usize;
                    for (i, &(c_min, _)) in ivs.iter().enumerate() {
                        if i == best_idx || c_min < cutoff {
                            active_count += 1;
                            last_active = i;
                        }
                    }
                    if active_count <= 1 {
                        children[last_active].row(p0, dir, out);
                        return;
                    }
                    let mut tmp = [0f32; ROW];
                    let mut first = true;
                    let mut out_max = f32::INFINITY;
                    let inv_k = 1.0 / k;
                    for (i, c) in children.iter().enumerate() {
                        let c_min = ivs[i].0;
                        if i != best_idx && (c_min >= cutoff || (!first && c_min >= out_max + k)) {
                            continue;
                        }
                        if first {
                            c.row(p0, dir, out);
                            out_max = slice_max(out);
                            first = false;
                        } else {
                            c.row(p0, dir, &mut tmp[..n]);
                            out_max = combine_smin(out, &tmp[..n], k, inv_k);
                        }
                    }
                }
            }
            Kind::Difference(children, k) => {
                let m = children.len();
                if m == 0 {
                    out.fill(f32::INFINITY);
                    return;
                }
                if m == 1 {
                    children[0].row(p0, dir, out);
                    return;
                }
                let k = *k;
                let (lo, hi) = row_box(p0, dir, n);
                children[0].row(p0, dir, out);
                let mut out_min = slice_min(out);
                let mut tmp = [0f32; ROW];
                let inv_k = if k > 0.0 { 1.0 / k } else { 0.0 };
                for c in &children[1..] {
                    let d2 = c.aabb.distance_squared_box(lo, hi);
                    let thr = k - out_min;
                    if thr <= 0.0 {
                        if d2 > 0.0 {
                            continue;
                        }
                    } else {
                        let reach = thr / c.aabb_scale;
                        if d2 >= reach * reach {
                            continue;
                        }
                    }
                    c.row(p0, dir, &mut tmp[..n]);
                    out_min = combine_smax_neg_min(out, &tmp[..n], k, inv_k);
                }
            }
            Kind::Intersection(children, k) => {
                let m = children.len();
                if m == 0 {
                    out.fill(f32::INFINITY);
                    return;
                }
                if m == 1 {
                    children[0].row(p0, dir, out);
                    return;
                }
                let k = *k;
                let (lo, hi) = row_box(p0, dir, n);
                let mut stack_iv = [(0.0f32, 0.0f32); 32];
                let mut heap_iv = Vec::new();
                let ivs: &mut [(f32, f32)] = if m <= 32 {
                    &mut stack_iv[..m]
                } else {
                    heap_iv.resize(m, (0.0, 0.0));
                    &mut heap_iv
                };
                let mut best_min = f32::NEG_INFINITY;
                let mut best_idx = 0usize;
                for (i, c) in children.iter().enumerate() {
                    let iv = c.eval_interval(lo, hi);
                    ivs[i] = iv;
                    if iv.0 > best_min {
                        best_min = iv.0;
                        best_idx = i;
                    }
                }
                if k <= 0.0 {
                    children[best_idx].row(p0, dir, out);
                    let mut out_min = slice_min(out);
                    let mut tmp = [0f32; ROW];
                    for (i, c) in children.iter().enumerate() {
                        if i == best_idx || ivs[i].1 <= out_min {
                            continue;
                        }
                        c.row(p0, dir, &mut tmp[..n]);
                        out_min = combine_max(out, &tmp[..n]);
                    }
                } else {
                    let cutoff = best_min - 2.0 * k;
                    let mut active_count = 0usize;
                    let mut last_active = 0usize;
                    for (i, &(_, c_max)) in ivs.iter().enumerate() {
                        if i == best_idx || c_max > cutoff {
                            active_count += 1;
                            last_active = i;
                        }
                    }
                    if active_count <= 1 {
                        children[last_active].row(p0, dir, out);
                        return;
                    }
                    let mut tmp = [0f32; ROW];
                    let mut first = true;
                    let mut out_min = f32::NEG_INFINITY;
                    let inv_k = 1.0 / k;
                    for (i, c) in children.iter().enumerate() {
                        let c_max = ivs[i].1;
                        if i != best_idx && (c_max <= cutoff || (!first && c_max <= out_min - k)) {
                            continue;
                        }
                        if first {
                            c.row(p0, dir, out);
                            out_min = slice_min(out);
                            first = false;
                        } else {
                            c.row(p0, dir, &mut tmp[..n]);
                            out_min = combine_smax(out, &tmp[..n], k, inv_k);
                        }
                    }
                }
            }
            Kind::Array(c, count, off) => {
                let count = *count as usize;
                if count == 0 {
                    out.fill(f32::INFINITY);
                    return;
                }
                if count == 1 {
                    c.row(p0, dir, out);
                    return;
                }
                let (lo, hi) = row_box(p0, dir, n);
                let mut stack_iv = [(0.0f32, 0.0f32); 64];
                let mut heap_iv = Vec::new();
                let ivs: &mut [(f32, f32)] = if count <= 64 {
                    &mut stack_iv[..count]
                } else {
                    heap_iv.resize(count, (0.0, 0.0));
                    &mut heap_iv
                };
                let mut best_max = f32::INFINITY;
                let mut best_idx = 0usize;
                for (i, slot) in ivs.iter_mut().enumerate() {
                    let shift = *off * i as f32;
                    let iv = c.eval_interval(lo - shift, hi - shift);
                    *slot = iv;
                    if iv.1 < best_max {
                        best_max = iv.1;
                        best_idx = i;
                    }
                }
                c.row(p0 - *off * best_idx as f32, dir, out);
                let mut out_max = slice_max(out);
                let mut tmp = [0f32; ROW];
                for (i, &(c_min, _)) in ivs.iter().enumerate() {
                    if i == best_idx || c_min >= out_max {
                        continue;
                    }
                    c.row(p0 - *off * i as f32, dir, &mut tmp[..n]);
                    out_max = combine_min(out, &tmp[..n]);
                }
            }
        }
    }

    #[inline(always)]
    fn kind_eval(&self, p: Vec3) -> f32 {
        match &self.kind {
            Kind::Sphere(r) => p.length() - r,
            Kind::Cuboid { inner, round, .. } => {
                let q = p.abs() - *inner;
                q.max(Vec3::ZERO).length() + q.max_element().min(0.0) - round
            }
            Kind::Cylinder { r_in, h_in, round, .. } => {
                let d = Vec2::new((p.x * p.x + p.y * p.y).sqrt() - *r_in, p.z.abs() - *h_in);
                d.x.max(d.y).min(0.0) + d.max(Vec2::ZERO).length() - round
            }
            Kind::Cone { r1, r2, hh, k1, k2, inv_k2_dot } => {
                let q = Vec2::new((p.x * p.x + p.y * p.y).sqrt(), p.z);
                let ca = Vec2::new(q.x - q.x.min(if q.y < 0.0 { *r1 } else { *r2 }), q.y.abs() - *hh);
                let cb = q - *k1 + *k2 * ((*k1 - q).dot(*k2) * *inv_k2_dot).clamp(0.0, 1.0);
                let s = if cb.x < 0.0 && ca.y < 0.0 { -1.0 } else { 1.0 };
                s * dot2(ca).min(dot2(cb)).sqrt()
            }
            Kind::Torus(a, b) => Vec2::new((p.x * p.x + p.y * p.y).sqrt() - a, p.z).length() - b,
            Kind::Capsule { a, ba, inv_ba_dot, r, .. } => {
                let pa = p - *a;
                let t = (pa.dot(*ba) * *inv_ba_dot).clamp(0.0, 1.0);
                (pa - *ba * t).length() - r
            }
            Kind::Ellipsoid { inv_r, inv_r2, neg_min_r, .. } => {
                let k0 = (p * *inv_r).length();
                let k1 = (p * *inv_r2).length();
                if k1 > 1e-9 { k0 * (k0 - 1.0) / k1 } else { *neg_min_r }
            }
            Kind::Pyramid { base, h, inv_s, h_over_s, m2, inv_m2_p, inv_m2 } => {
                let qx = p.x * *inv_s;
                let qy = (p.z + *h * 0.5) * *inv_s;
                let qz = p.y * *inv_s;
                pyramid_fast(Vec3::new(qx, qy, qz), *h_over_s, *m2, *inv_m2_p, *inv_m2) * *base
            }
            Kind::Lathe(edges) => polygon(Vec2::new((p.x * p.x + p.y * p.y).sqrt(), p.z), edges),
            Kind::Extrude(edges, hh) => {
                let w = Vec2::new(polygon(p.truncate(), edges), p.z.abs() - hh);
                w.x.max(w.y).min(0.0) + w.max(Vec2::ZERO).length()
            }
            Kind::Union(children, k) => {
                let mut d = f32::INFINITY;
                for c in children {
                    // A child whose bounds are farther than the blend reach can't change the result's sign.
                    if d.is_finite() {
                        let reach = (d + k) / c.aabb_scale;
                        if reach <= 0.0 {
                            if c.aabb.distance_squared(p) > 0.0 {
                                continue;
                            }
                        } else if c.aabb.distance_squared(p) > reach * reach {
                            continue;
                        }
                    }
                    let e = c.eval(p);
                    d = if d.is_finite() { smin(d, e, *k) } else { e };
                }
                d
            }
            Kind::Difference(children, k) => {
                let mut d = children[0].eval(p);
                for c in &children[1..] {
                    let reach = (k + d.abs()) / c.aabb_scale;
                    if c.aabb.distance_squared(p) > reach * reach {
                        continue;
                    }
                    d = smax(d, -c.eval(p), *k);
                }
                d
            }
            Kind::Intersection(children, k) => {
                let mut d = children[0].eval(p);
                for c in &children[1..] {
                    d = smax(d, c.eval(p), *k);
                }
                d
            }
            Kind::Array(c, n, off) => {
                let mut d = f32::INFINITY;
                for i in 0..*n {
                    d = d.min(c.eval(p - *off * i as f32));
                }
                d
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4×4×4 block combinators (16 F32x4 lanes, all in NEON registers)
// ---------------------------------------------------------------------------

#[inline(always)]
fn block_max(b: &[F32x4; 16]) -> f32 {
    let mut m0 = b[0].max(b[1]).max(b[2].max(b[3]));
    let m1 = b[4].max(b[5]).max(b[6].max(b[7]));
    let m2 = b[8].max(b[9]).max(b[10].max(b[11]));
    let m3 = b[12].max(b[13]).max(b[14].max(b[15]));
    m0 = m0.max(m1).max(m2.max(m3));
    m0.max_elem()
}

#[inline(always)]
fn block_min(b: &[F32x4; 16]) -> f32 {
    let mut m0 = b[0].min(b[1]).min(b[2].min(b[3]));
    let m1 = b[4].min(b[5]).min(b[6].min(b[7]));
    let m2 = b[8].min(b[9]).min(b[10].min(b[11]));
    let m3 = b[12].min(b[13]).min(b[14].min(b[15]));
    m0 = m0.min(m1).min(m2.min(m3));
    m0.min_elem()
}

#[inline(always)]
fn block_combine_min(out: &mut [F32x4; 16], src: &[F32x4; 16]) -> f32 {
    let mut mx = F32x4::splat(f32::NEG_INFINITY);
    for i in 0..16 {
        let v = out[i].min(src[i]);
        out[i] = v;
        mx = mx.max(v);
    }
    mx.max_elem()
}

#[inline(always)]
fn block_combine_smin(out: &mut [F32x4; 16], src: &[F32x4; 16], k4: F32x4, ik4: F32x4) -> f32 {
    let mut mx = F32x4::splat(f32::NEG_INFINITY);
    for i in 0..16 {
        let v = smin4(out[i], src[i], k4, ik4);
        out[i] = v;
        mx = mx.max(v);
    }
    mx.max_elem()
}

#[inline(always)]
fn block_combine_max(out: &mut [F32x4; 16], src: &[F32x4; 16]) -> f32 {
    let mut mn = F32x4::splat(f32::INFINITY);
    for i in 0..16 {
        let v = out[i].max(src[i]);
        out[i] = v;
        mn = mn.min(v);
    }
    mn.min_elem()
}

#[inline(always)]
fn block_combine_smax(out: &mut [F32x4; 16], src: &[F32x4; 16], k4: F32x4, ik4: F32x4) -> f32 {
    let mut mn = F32x4::splat(f32::INFINITY);
    for i in 0..16 {
        let v = smax4(out[i], src[i], k4, ik4);
        out[i] = v;
        mn = mn.min(v);
    }
    mn.min_elem()
}

#[inline(always)]
fn block_combine_max_neg(out: &mut [F32x4; 16], src: &[F32x4; 16]) -> f32 {
    let mut mn = F32x4::splat(f32::INFINITY);
    for i in 0..16 {
        let v = out[i].max(src[i].neg());
        out[i] = v;
        mn = mn.min(v);
    }
    mn.min_elem()
}

#[inline(always)]
fn block_combine_smax_neg(out: &mut [F32x4; 16], src: &[F32x4; 16], k4: F32x4, ik4: F32x4) -> f32 {
    let mut mn = F32x4::splat(f32::INFINITY);
    for i in 0..16 {
        let v = smax4(out[i], src[i].neg(), k4, ik4);
        out[i] = v;
        mn = mn.min(v);
    }
    mn.min_elem()
}

// ---------------------------------------------------------------------------
// 4-wide SIMD row combinators
// ---------------------------------------------------------------------------

#[inline(always)]
fn smin4(a: F32x4, b: F32x4, k: F32x4, inv_k: F32x4) -> F32x4 {
    let h = k.sub(a.sub(b).abs()).max(F32x4::splat(0.0)).mul(inv_k);
    a.min(b).sub(h.mul(h).mul(k).mul(F32x4::splat(0.25)))
}

#[inline(always)]
fn smax4(a: F32x4, b: F32x4, k: F32x4, inv_k: F32x4) -> F32x4 {
    smin4(a.neg(), b.neg(), k, inv_k).neg()
}

#[inline(always)]
fn slice_max(s: &[f32]) -> f32 {
    let mut acc = F32x4::splat(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 4 <= s.len() {
        acc = acc.max(F32x4::load(&s[i..]));
        i += 4;
    }
    let mut m = acc.max_elem();
    for &v in &s[i..] {
        m = m.max(v);
    }
    m
}

#[inline(always)]
fn slice_min(s: &[f32]) -> f32 {
    let mut acc = F32x4::splat(f32::INFINITY);
    let mut i = 0;
    while i + 4 <= s.len() {
        acc = acc.min(F32x4::load(&s[i..]));
        i += 4;
    }
    let mut m = acc.min_elem();
    for &v in &s[i..] {
        m = m.min(v);
    }
    m
}

#[inline(always)]
fn combine_min(out: &mut [f32], src: &[f32]) -> f32 {
    let mut mx4 = F32x4::splat(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 4 <= out.len() {
        let v = F32x4::load(&out[i..]).min(F32x4::load(&src[i..]));
        v.store(&mut out[i..]);
        mx4 = mx4.max(v);
        i += 4;
    }
    let mut mx = mx4.max_elem();
    for (o, &e) in out[i..].iter_mut().zip(&src[i..]) {
        let v = o.min(e);
        *o = v;
        mx = mx.max(v);
    }
    mx
}

#[inline(always)]
fn combine_max(out: &mut [f32], src: &[f32]) -> f32 {
    let mut mn4 = F32x4::splat(f32::INFINITY);
    let mut i = 0;
    while i + 4 <= out.len() {
        let v = F32x4::load(&out[i..]).max(F32x4::load(&src[i..]));
        v.store(&mut out[i..]);
        mn4 = mn4.min(v);
        i += 4;
    }
    let mut mn = mn4.min_elem();
    for (o, &e) in out[i..].iter_mut().zip(&src[i..]) {
        let v = o.max(e);
        *o = v;
        mn = mn.min(v);
    }
    mn
}

#[inline(always)]
fn combine_smin(out: &mut [f32], src: &[f32], k: f32, inv_k: f32) -> f32 {
    let k4 = F32x4::splat(k);
    let ik4 = F32x4::splat(inv_k);
    let mut mx4 = F32x4::splat(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 4 <= out.len() {
        let v = smin4(F32x4::load(&out[i..]), F32x4::load(&src[i..]), k4, ik4);
        v.store(&mut out[i..]);
        mx4 = mx4.max(v);
        i += 4;
    }
    let mut mx = mx4.max_elem();
    for (o, &e) in out[i..].iter_mut().zip(&src[i..]) {
        let v = smin(*o, e, k);
        *o = v;
        mx = mx.max(v);
    }
    mx
}

#[inline(always)]
fn combine_smax(out: &mut [f32], src: &[f32], k: f32, inv_k: f32) -> f32 {
    let k4 = F32x4::splat(k);
    let ik4 = F32x4::splat(inv_k);
    let mut mn4 = F32x4::splat(f32::INFINITY);
    let mut i = 0;
    while i + 4 <= out.len() {
        let v = smax4(F32x4::load(&out[i..]), F32x4::load(&src[i..]), k4, ik4);
        v.store(&mut out[i..]);
        mn4 = mn4.min(v);
        i += 4;
    }
    let mut mn = mn4.min_elem();
    for (o, &e) in out[i..].iter_mut().zip(&src[i..]) {
        let v = smax(*o, e, k);
        *o = v;
        mn = mn.min(v);
    }
    mn
}

#[inline(always)]
fn combine_smax_neg_min(out: &mut [f32], src: &[f32], k: f32, inv_k: f32) -> f32 {
    let mut mn4 = F32x4::splat(f32::INFINITY);
    let mut i = 0;
    if k <= 0.0 {
        while i + 4 <= out.len() {
            let v = F32x4::load(&out[i..]).max(F32x4::load(&src[i..]).neg());
            v.store(&mut out[i..]);
            mn4 = mn4.min(v);
            i += 4;
        }
        let mut mn = mn4.min_elem();
        for (o, &e) in out[i..].iter_mut().zip(&src[i..]) {
            let v = o.max(-e);
            *o = v;
            mn = mn.min(v);
        }
        mn
    } else {
        let k4 = F32x4::splat(k);
        let ik4 = F32x4::splat(inv_k);
        while i + 4 <= out.len() {
            let v = smax4(F32x4::load(&out[i..]), F32x4::load(&src[i..]).neg(), k4, ik4);
            v.store(&mut out[i..]);
            mn4 = mn4.min(v);
            i += 4;
        }
        let mut mn = mn4.min_elem();
        for (o, &e) in out[i..].iter_mut().zip(&src[i..]) {
            let v = smax(*o, -e, k);
            *o = v;
            mn = mn.min(v);
        }
        mn
    }
}

/// Polynomial smooth minimum with blend radius `k` (Quilez).
#[inline(always)]
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

#[inline(always)]
pub fn smax(a: f32, b: f32, k: f32) -> f32 {
    -smin(-a, -b, k)
}

#[inline(always)]
fn dot2(v: Vec2) -> f32 {
    v.dot(v)
}

#[inline(always)]
fn iv_abs(lo: f32, hi: f32) -> (f32, f32) {
    if lo >= 0.0 {
        (lo, hi)
    } else if hi <= 0.0 {
        (-hi, -lo)
    } else {
        (0.0, (-lo).max(hi))
    }
}

/// Square pyramid with unit base on y = 0 and apex at height `h` (Quilez, sdPyramid; Y up in this frame).
#[inline(always)]
fn pyramid_fast(p: Vec3, h: f32, m2: f32, inv_m2_p: f32, inv_m2: f32) -> f32 {
    let (mut x, mut z) = (p.x.abs(), p.z.abs());
    if z > x {
        std::mem::swap(&mut x, &mut z);
    }
    x -= 0.5;
    z -= 0.5;
    let y = p.y;
    let q = Vec3::new(z, h * y - 0.5 * x, h * x + 0.5 * y);
    let s = (-q.x).max(0.0);
    let t = ((q.y - 0.5 * z) * inv_m2_p).clamp(0.0, 1.0);
    let a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    let b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
    let d2 = if q.y.min(-q.x * m2 - q.y * 0.5) > 0.0 { 0.0 } else { a.min(b) };
    ((d2 + q.z * q.z) * inv_m2).sqrt() * q.z.max(-y).signum()
}

/// Signed distance to a closed polygon (Quilez, sdPolygon), negative inside, either winding.
#[inline]
fn polygon(p: Vec2, edges: &[PolyEdge]) -> f32 {
    let mut d = dot2(p - edges[0].vi);
    let mut s = 1.0;
    for edge in edges {
        let w = p - edge.vi;
        let b = w - edge.e * (w.dot(edge.e) * edge.inv_e2).clamp(0.0, 1.0);
        d = d.min(dot2(b));
        let c0 = p.y >= edge.vi.y;
        let c1 = p.y < edge.vj_y;
        let c2 = edge.e.x * w.y > edge.e.y * w.x;
        if (c0 && c1 && c2) || (!c0 && !c1 && !c2) {
            s = -s;
        }
    }
    s * d.sqrt()
}

/// 4-wide SIMD signed distance to a closed polygon.
#[inline]
fn polygon4(px: F32x4, py: F32x4, edges: &[PolyEdge]) -> F32x4 {
    let d0x = px.sub(F32x4::splat(edges[0].vi.x));
    let d0y = py.sub(F32x4::splat(edges[0].vi.y));
    let mut d = d0x.mul(d0x).add(d0y.mul(d0y));
    let mut s = F32x4::splat(1.0);
    for edge in edges {
        let vix = F32x4::splat(edge.vi.x);
        let viy = F32x4::splat(edge.vi.y);
        let ex = F32x4::splat(edge.e.x);
        let ey = F32x4::splat(edge.e.y);
        let wx = px.sub(vix);
        let wy = py.sub(viy);
        let t = wx.mul(ex).add(wy.mul(ey)).mul(F32x4::splat(edge.inv_e2)).clamp01();
        let bx = wx.sub(ex.mul(t));
        let by = wy.sub(ey.mul(t));
        d = d.min(bx.mul(bx).add(by.mul(by)));
        let c0 = py.ge(viy);
        let c1 = py.lt(F32x4::splat(edge.vj_y));
        let c2 = ex.mul(wy).gt(ey.mul(wx));
        let flip = c0.and(c1).and(c2).or(c0.not().and(c1.not()).and(c2.not()));
        s = F32x4::select(flip, s.neg(), s);
    }
    s.mul(d.sqrt())
}

/// Central-difference gradient of `f` at `p` with step `e`.
pub fn gradient(f: &Compiled, p: Vec3, e: f32) -> Vec3 {
    let dx = Vec3::new(e, 0.0, 0.0);
    let dy = Vec3::new(0.0, e, 0.0);
    let dz = Vec3::new(0.0, 0.0, e);
    Vec3::new(f.eval(p + dx) - f.eval(p - dx), f.eval(p + dy) - f.eval(p - dy), f.eval(p + dz) - f.eval(p - dz)) / (2.0 * e)
}

/// Visits the lattice `b.min + step * (i, j, k)`, `0 <= i, j, k < N`, in k-major then j then i order and calls
/// `visit(i, j, k, p, d)` with `d = dist(p)` for every sample that is not provably outside.
///
/// Every 4x4x4 block is tested first with `block_active(b_min, b_max)` via interval arithmetic; blocks
/// strictly outside (`d_min > 0`) are skipped cleanly without assuming 1-Lipschitz scaling.
fn scan_lattice<const N: usize>(
    b: &Aabb,
    step: Vec3,
    block_active: &mut dyn FnMut(Vec3, Vec3) -> bool,
    dist: &mut dyn FnMut(Vec3, f32, &mut [f32]),
    visit: &mut dyn FnMut(usize, usize, usize, Vec3, f32),
) {
    const BLK: usize = 4;
    let nb = N.div_ceil(BLK);
    let mut active = vec![true; nb * nb * nb];
    for bk in 0..nb {
        let k0 = bk * BLK;
        let k1 = (k0 + BLK - 1).min(N - 1);
        for bj in 0..nb {
            let j0 = bj * BLK;
            let j1 = (j0 + BLK - 1).min(N - 1);
            for bi in 0..nb {
                let i0 = bi * BLK;
                let i1 = (i0 + BLK - 1).min(N - 1);
                let lo = b.min + step * Vec3::new(i0 as f32, j0 as f32, k0 as f32);
                let hi = b.min + step * Vec3::new(i1 as f32, j1 as f32, k1 as f32);
                active[(bk * nb + bj) * nb + bi] = block_active(lo, hi);
            }
        }
    }
    let mut row = [0f32; N];
    for k in 0..N {
        for j in 0..N {
            let arow = &active[((k / BLK) * nb + j / BLK) * nb..][..nb];
            let mut bi = 0;
            while bi < nb {
                if !arow[bi] {
                    bi += 1;
                    continue;
                }
                let mut be = bi + 1;
                while be < nb && arow[be] {
                    be += 1;
                }
                let (i0, i1) = (bi * BLK, (be * BLK).min(N));
                let p0 = b.min + step * Vec3::new(i0 as f32, j as f32, k as f32);
                dist(p0, step.x, &mut row[..i1 - i0]);
                for i in i0..i1 {
                    visit(i, j, k, b.min + step * Vec3::new(i as f32, j as f32, k as f32), row[i - i0]);
                }
                bi = be;
            }
        }
    }
}

/// Gradient of `f` at `p` from four tetrahedral probes evaluated in a single 4-wide SIMD pass.
#[inline]
pub fn gradient_tetra(f: &Compiled, p: Vec3, e: f32) -> Vec3 {
    const K0: Vec3 = Vec3::new(1.0, -1.0, -1.0);
    const K1: Vec3 = Vec3::new(-1.0, -1.0, 1.0);
    const K2: Vec3 = Vec3::new(-1.0, 1.0, -1.0);
    const K3: Vec3 = Vec3::new(1.0, 1.0, 1.0);
    let d = f.eval4_pts([p + K0 * e, p + K1 * e, p + K2 * e, p + K3 * e]);
    (K0 * d[0] + K1 * d[1] + K2 * d[2] + K3 * d[3]) / (4.0 * e)
}

/// Bounds of what is actually solid, from a coarse sampling of the field (within ~1/28 of the size).
///
/// Fast paths:
/// 1. `f.exact_aabb` (all unrotated / 90°-rotated primitives and hard unions of them) returns in O(1).
/// 2. Pure translations (`has_at` only) unwrap cleanly into the local shape's `tight_aabb` + `at`.
/// 3. Hard unions decompose into the union of children's `tight_aabb`.
/// 4. Smooth unions (`smin`) can only bulge outside `union(child.aabb)` where two children's blend
///    regions overlap (`dist_i <= k/4`, `dist_j <= k`). When all pairwise blend overlaps lie inside
///    `child_union` (true for nearly every creature/character/object), `child_union` is exact in O(1)!
/// 5. Differences (`base - cuts`) seed every untouched face of `base` and only probe blocks that can
///    expand an uncertain face.
/// 6. General CSG scans a 28×28×28 outside-in 4×4×4 block grid, skipping any block already contained
///    in `out` without even calling `eval_interval`, and unioning solid (`d_max < 0`) blocks in O(1).
pub fn tight_aabb(f: &Compiled) -> Aabb {
    let b = f.aabb;
    if b.is_empty() || f.exact_aabb {
        return b;
    }
    let shift = if f.has_at { f.at } else { Vec3::ZERO };
    let no_linear_xform = !f.has_rot && !f.has_scale && f.mirror.is_none();

    if no_linear_xform {
        match &f.kind {
            Kind::Union(children, k) => {
                if *k <= 0.0 {
                    let mut u = Aabb::EMPTY;
                    for c in children {
                        u = u.union(&tight_aabb(c));
                    }
                    if !u.is_empty() {
                        u = Aabb::new(u.min + shift, u.max + shift);
                        return u.intersect(&b);
                    }
                } else if !children.is_empty() {
                    let mut child_boxes = Vec::with_capacity(children.len());
                    let mut u = Aabb::EMPTY;
                    for c in children {
                        let cb = tight_aabb(c);
                        child_boxes.push(cb);
                        u = u.union(&cb);
                    }
                    // Check if any pairwise smooth-blend overlap box actually extends outside `u` AND
                    // could contain solid points (verified with eval_interval on each protruding slab).
                    let k = *k;
                    let u_world = Aabb::new(u.min + shift, u.max + shift).intersect(&b);
                    let mut bulge_box = Aabb::EMPTY;
                    for i in 0..children.len() {
                        let g1_i = child_boxes[i].grow(k * 0.25 / children[i].aabb_scale);
                        let g2_i = child_boxes[i].grow(k / children[i].aabb_scale);
                        for j in (i + 1)..children.len() {
                            let g1_j = child_boxes[j].grow(k * 0.25 / children[j].aabb_scale);
                            let g2_j = child_boxes[j].grow(k / children[j].aabb_scale);
                            for ov in [g1_i.intersect(&g2_j), g1_j.intersect(&g2_i)] {
                                if ov.is_empty() {
                                    continue;
                                }
                                let ov_w = Aabb::new(ov.min + shift, ov.max + shift).intersect(&b);
                                if ov_w.is_empty() {
                                    continue;
                                }
                                // Test each of the up to 6 protruding strips of `ov_w \ u_world`.
                                for axis in 0..3usize {
                                    if ov_w.min[axis] < u_world.min[axis] - 1e-4 {
                                        let mut strip = ov_w;
                                        strip.max[axis] = u_world.min[axis];
                                        if f.eval_interval(strip.min, strip.max).0 <= 0.0 {
                                            bulge_box = bulge_box.union(&strip);
                                        }
                                    }
                                    if ov_w.max[axis] > u_world.max[axis] + 1e-4 {
                                        let mut strip = ov_w;
                                        strip.min[axis] = u_world.max[axis];
                                        if f.eval_interval(strip.min, strip.max).0 <= 0.0 {
                                            bulge_box = bulge_box.union(&strip);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if bulge_box.is_empty() && !u_world.is_empty() {
                        return u_world;
                    }
                    if !u_world.is_empty() {
                        // Only scan the small protruding bulge box (seeded with `u_world`), not the whole shape!
                        return tight_aabb_scan::<8>(f, bulge_box, u_world).intersect(&b);
                    }
                }
            }
            _ => {}
        }
    }

    // Seed `out` from known-solid sub-boxes to prune interior & untouched faces immediately.
    let mut out = Aabb::EMPTY;
    if no_linear_xform {
        match &f.kind {
            Kind::Difference(children, k) if children.len() >= 2 && children[0].exact_aabb => {
                // Any face of `base` that no cut's grown AABB reaches is untouched (guaranteed exact!),
                // AND any face touched by a cut that still has a solid witness point on `base`'s rim/corners
                // (e.g. the open rim of a hollow cup/mug/pot/glass) is also exact!
                let base = Aabb::new(children[0].aabb.min + shift, children[0].aabb.max + shift);
                let mut cut_u = Aabb::EMPTY;
                for c in &children[1..] {
                    cut_u = cut_u.union(&Aabb::new(c.aabb.min + shift, c.aabb.max + shift));
                }
                let cut_g = cut_u.grow(*k + 1e-3);
                let mut seed = base;
                let mut exact_faces = 0u8;
                let check_face = |axis: usize, is_max: bool, untouched: bool| -> bool {
                    if untouched {
                        return true;
                    }
                    has_solid_face_witness(f, &children[0], shift, axis, is_max)
                };
                if check_face(0, false, cut_g.min.x > base.min.x) { exact_faces += 1; } else { seed.min.x = base.center().x; }
                if check_face(0, true, cut_g.max.x < base.max.x) { exact_faces += 1; } else { seed.max.x = base.center().x; }
                if check_face(1, false, cut_g.min.y > base.min.y) { exact_faces += 1; } else { seed.min.y = base.center().y; }
                if check_face(1, true, cut_g.max.y < base.max.y) { exact_faces += 1; } else { seed.max.y = base.center().y; }
                if check_face(2, false, cut_g.min.z > base.min.z) { exact_faces += 1; } else { seed.min.z = base.center().z; }
                if check_face(2, true, cut_g.max.z < base.max.z) { exact_faces += 1; } else { seed.max.z = base.center().z; }
                if exact_faces == 6 {
                    return base.intersect(&b);
                }
                if exact_faces > 0 && !seed.is_empty() {
                    out = seed;
                }
            }
            _ => {}
        }
    }
    if b.size().max_element() <= 48.0 {
        tight_aabb_scan::<16>(f, b, out).intersect(&b)
    } else {
        tight_aabb_scan::<28>(f, b, out).intersect(&b)
    }
}

/// Checks whether `base` (an exact-AABB child of `f`, shifted by `shift`) still has at least one solid
/// witness point of `f` on its `(axis, is_max)` face.
fn has_solid_face_witness(f: &Compiled, base: &Compiled, shift: Vec3, axis: usize, is_max: bool) -> bool {
    if base.has_rot || base.has_scale || base.mirror.is_some() || base.hollow > 0.0 {
        return false;
    }
    let c = base.at + shift;
    let sgn = if is_max { 1.0f32 } else { -1.0f32 };
    const EPS: f32 = 1e-3;
    match &base.kind {
        Kind::Cylinder { r_in, h_in, round } => {
            let r = r_in + round;
            let hh = h_in + round;
            if axis == 2 {
                let z = c.z + sgn * (hh - EPS);
                let rr = (*r_in - EPS).max(0.0);
                for (dx, dy) in [(rr, 0.0), (-rr, 0.0), (0.0, rr), (0.0, -rr)] {
                    if f.eval(Vec3::new(c.x + dx, c.y + dy, z)) <= EPS {
                        return true;
                    }
                }
            } else {
                let off = sgn * (r - EPS);
                let hz = (*h_in - EPS).max(0.0);
                for dz in [-hz, 0.0, hz] {
                    let p = if axis == 0 {
                        Vec3::new(c.x + off, c.y, c.z + dz)
                    } else {
                        Vec3::new(c.x, c.y + off, c.z + dz)
                    };
                    if f.eval(p) <= EPS {
                        return true;
                    }
                }
            }
            false
        }
        Kind::Cuboid { inner, round } => {
            let half = *inner + Vec3::splat(*round);
            let mut p = c;
            p[axis] = c[axis] + sgn * (half[axis] - EPS);
            let a1 = (axis + 1) % 3;
            let a2 = (axis + 2) % 3;
            let e1 = (inner[a1] - EPS).max(0.0);
            let e2 = (inner[a2] - EPS).max(0.0);
            for (s1, s2) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (0.0, 0.0)] {
                p[a1] = c[a1] + s1 * e1;
                p[a2] = c[a2] + s2 * e2;
                if f.eval(p) <= EPS {
                    return true;
                }
            }
            false
        }
        Kind::Cone { r1, r2, hh, .. } => {
            if axis == 2 {
                let z = c.z + sgn * (*hh - EPS);
                let rr = ((if is_max { *r2 } else { *r1 }) - EPS).max(0.0);
                for (dx, dy) in [(rr, 0.0), (-rr, 0.0), (0.0, rr), (0.0, -rr)] {
                    if f.eval(Vec3::new(c.x + dx, c.y + dy, z)) <= EPS {
                        return true;
                    }
                }
            }
            false
        }
        _ => false,
    }
}

fn tight_aabb_scan<const N: usize>(f: &Compiled, b: Aabb, mut out: Aabb) -> Aabb {
    const BLK: usize = 4;
    let nb = N / BLK;
    let step = b.size() / (N as f32 - 1.0);
    // Outside-in block order: faces first so `out` expands immediately and prunes interior blocks.
    let mut order = [0usize; 8];
    let mut len = 0usize;
    let (mut lo_i, mut hi_i) = (0usize, nb);
    while lo_i < hi_i {
        order[len] = lo_i;
        len += 1;
        lo_i += 1;
        if lo_i < hi_i {
            hi_i -= 1;
            order[len] = hi_i;
            len += 1;
        }
    }
    let ord = &order[..len];
    let mut row = [0f32; BLK];

    for &bk in ord {
        let k0 = bk * BLK;
        let k1 = k0 + BLK - 1;
        let zlo = b.min.z + step.z * k0 as f32;
        let zhi = b.min.z + step.z * k1 as f32;
        let z_inside = !out.is_empty() && zlo >= out.min.z && zhi <= out.max.z;
        for &bj in ord {
            let j0 = bj * BLK;
            let j1 = j0 + BLK - 1;
            let ylo = b.min.y + step.y * j0 as f32;
            let yhi = b.min.y + step.y * j1 as f32;
            let yz_inside = z_inside && ylo >= out.min.y && yhi <= out.max.y;
            for &bi in ord {
                let i0 = bi * BLK;
                let i1 = i0 + BLK - 1;
                let xlo = b.min.x + step.x * i0 as f32;
                let xhi = b.min.x + step.x * i1 as f32;
                if yz_inside && xlo >= out.min.x && xhi <= out.max.x {
                    continue;
                }
                let lo = Vec3::new(xlo, ylo, zlo);
                let hi = Vec3::new(xhi, yhi, zhi);
                let (d_min, d_max) = f.eval_interval(lo, hi);
                if d_min > 0.0 {
                    continue;
                }
                if d_max < 0.0 {
                    out = out.union(&Aabb::new(lo, hi));
                    continue;
                }
                for k in k0..=k1 {
                    let z = b.min.z + step.z * k as f32;
                    for j in j0..=j1 {
                        let y = b.min.y + step.y * j as f32;
                        let p0 = Vec3::new(xlo, y, z);
                        f.eval_row(p0, step.x, &mut row);
                        for ii in 0..BLK {
                            let d = row[ii];
                            if d < 0.0 {
                                let p = Vec3::new(xlo + step.x * ii as f32, y, z);
                                out = out.union(&Aabb::new(p + d, p - d));
                            }
                        }
                    }
                }
            }
        }
    }
    if out.is_empty() { b } else { out }
}

/// Where an accessory goes: the top of the highest feature (a head, not the gap between a head and a tail).
/// Returns the point on top and the width of that feature.
pub fn summit(fs: &[Compiled]) -> Option<(Vec3, f32)> {
    let b = fs.iter().fold(Aabb::EMPTY, |b, f| b.union(&f.aabb));
    if b.is_empty() {
        return None;
    }
    const N: usize = 40;
    let step = b.size() / (N as f32 - 1.0);
    let mut inside = Vec::new();
    // Distance to the union: any part whose interval lower bound <= 0 may hold interior samples.
    let mut tmp = [0f32; N];
    scan_lattice::<N>(
        &b,
        step,
        &mut |lo, hi| fs.iter().any(|f| f.eval_interval(lo, hi).0 <= 0.0),
        &mut |p, dx, o| {
            let n = o.len();
            let (lo, hi) = row_box(p, Vec3::new(dx, 0.0, 0.0), n);
            let mut out_max = f32::INFINITY;
            let mut first = true;
            for f in fs {
                let (c_min, c_max) = f.eval_interval(lo, hi);
                if c_min > 0.0 || (!first && c_min >= out_max) {
                    continue;
                }
                if first {
                    f.eval_row(p, dx, o);
                    out_max = slice_max(o);
                    first = false;
                } else {
                    f.eval_row(p, dx, &mut tmp[..n]);
                    out_max = combine_min(o, &tmp[..n]);
                }
                let _ = c_max;
            }
            if first {
                o.fill(f32::INFINITY);
            }
        },
        &mut |_, _, _, p, d| {
            if d < 0.0 {
                inside.push(p);
            }
        },
    );
    let top = *inside.iter().max_by(|a, c| a.z.total_cmp(&c.z))?;
    let r = b.size().max_element() * 0.15;
    let near: Vec<Vec3> = inside.iter().copied().filter(|p| (p.truncate() - top.truncate()).length() < r && p.z > top.z - r).collect();
    let (lo, hi) = near.iter().fold((Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)), |(l, h), p| (l.min(*p), h.max(*p)));
    let c = (lo + hi) * 0.5;
    Some((Vec3::new(c.x, c.y, top.z + step.z * 0.5), (hi - lo).truncate().max_element() + step.x))
}

#[cfg(test)]
mod tests {
    use super::super::program::*;
    use super::*;

    fn assert_interval_contains_samples(node: &Node, seed: u32) {
        let c = Compiled::new(node);
        let mut s = 0x9E37_79B9u32 ^ seed.wrapping_mul(0x85EB_CA6B);
        let mut rnd = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32
        };
        let b = c.aabb.grow(20.0);
        let sz = b.size();
        for _ in 0..40 {
            let p0 = b.min + Vec3::new(rnd() * sz.x, rnd() * sz.y, rnd() * sz.z);
            let p1 = b.min + Vec3::new(rnd() * sz.x, rnd() * sz.y, rnd() * sz.z);
            let lo = p0.min(p1);
            let hi = p0.max(p1);
            let (iv_lo, iv_hi) = c.eval_interval(lo, hi);
            assert!(iv_lo <= iv_hi + 1e-4, "invalid interval [{iv_lo}, {iv_hi}]");
            for _ in 0..25 {
                let p = lo + (hi - lo) * Vec3::new(rnd(), rnd(), rnd());
                let v = c.eval(p);
                assert!(
                    v >= iv_lo - 1e-3 && v <= iv_hi + 1e-3,
                    "eval({p}) = {v} outside interval [{iv_lo}, {iv_hi}] on box [{lo}, {hi}]"
                );
            }
            // Check eval4_pts matches eval on 4 random points.
            let pts = [
                lo + (hi - lo) * Vec3::new(rnd(), rnd(), rnd()),
                lo + (hi - lo) * Vec3::new(rnd(), rnd(), rnd()),
                lo + (hi - lo) * Vec3::new(rnd(), rnd(), rnd()),
                lo + (hi - lo) * Vec3::new(rnd(), rnd(), rnd()),
            ];
            let v4 = c.eval4_pts(pts);
            for l in 0..4 {
                let pv = c.eval(pts[l]);
                assert!((v4[l] - pv).abs() < 1e-3, "case {seed}: eval4[{l}] = {} != eval = {pv}", v4[l]);
            }
            // Also check eval_row matches eval point-by-point along X.
            let n = 32usize;
            let dx = (hi.x - lo.x) / (n as f32 - 1.0);
            let mut row = [0.0f32; 32];
            let p_row = Vec3::new(lo.x, (lo.y + hi.y) * 0.5, (lo.z + hi.z) * 0.5);
            c.eval_row(p_row, dx, &mut row);
            for (i, &rv) in row.iter().enumerate() {
                let pt = p_row + Vec3::new(dx * i as f32, 0.0, 0.0);
                let pv = c.eval(pt);
                assert!((rv - pv).abs() < 1e-3, "case {seed}: eval_row[{i}] = {rv} != eval = {pv} at {pt} (iv=[{iv_lo}, {iv_hi}])");
                assert!(rv >= iv_lo - 1e-3 && rv <= iv_hi + 1e-3);
            }
            // Check eval_block4 matches eval at all 64 samples.
            let bh = ((hi - lo).max_element() / 4.0).max(0.25);
            let mut blk = [F32x4::splat(0.0); 16];
            c.eval_block4(lo, bh, &mut blk);
            for k in 0..4 {
                for j in 0..4 {
                    let row4 = blk[k * 4 + j].to_array();
                    for i in 0..4 {
                        let pt = lo + Vec3::new(i as f32 * bh, j as f32 * bh, k as f32 * bh);
                        let pv = c.eval(pt);
                        assert!(
                            (row4[i] - pv).abs() < 1e-3,
                            "case {seed}: eval_block4[{i},{j},{k}] = {} != eval = {pv} at {pt}",
                            row4[i]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn interval_bounds_and_row_pruning_soundness() {
        let cases = [
            sphere(20.0).at(5.0, -3.0, 10.0),
            rbox(30.0, 20.0, 14.0, 2.0).rot(25.0, -15.0, 40.0).mirror(Axis::X),
            rcylinder(12.0, 40.0, 2.0).stretched(1.0, 0.4, 1.5).hollow(2.0),
            cone(18.0, 6.0, 32.0).rot(10.0, 35.0, -20.0).at(0.0, 5.0, 0.0),
            torus(18.0, 5.0).rot(90.0, 15.0, 0.0).at(20.0, 0.0, 15.0),
            capsule([-10.0, -5.0, 0.0], [15.0, 12.0, 20.0], 5.0).mirror(Axis::Y),
            ellipsoid(40.0, 14.0, 10.0).rot(15.0, 25.0, -10.0),
            pyramid(28.0, 24.0).at(-5.0, 5.0, 12.0),
            lathe(&[(0.0, 0.0), (16.0, 0.0), (22.0, 20.0), (10.0, 40.0), (0.0, 40.0)]),
            extrude(&[(0.0, 0.0), (24.0, 0.0), (12.0, 20.0)], 8.0).rot(0.0, 30.0, 15.0),
            union(
                4.0,
                vec![
                    sphere(15.0).at(-20.0, 0.0, 15.0),
                    cuboid(20.0, 20.0, 20.0).at(15.0, 5.0, 10.0),
                    ellipsoid(30.0, 12.0, 10.0).at(0.0, -15.0, 10.0),
                ],
            ),
            difference(
                2.0,
                vec![
                    rcylinder(25.0, 50.0, 2.0).at(0.0, 0.0, 25.0),
                    cylinder(20.0, 50.0).at(0.0, 0.0, 30.0),
                    sphere(12.0).at(20.0, 0.0, 25.0),
                ],
            ),
            intersection(
                1.5,
                vec![
                    sphere(24.0).at(0.0, 0.0, 20.0),
                    cuboid(32.0, 32.0, 32.0).rot(20.0, 20.0, 0.0).at(0.0, 0.0, 20.0),
                ],
            ),
            array(rbox(8.0, 14.0, 10.0, 1.0).at(-20.0, 0.0, 5.0), 5, [10.0, 0.0, 2.0]),
        ];
        for (idx, node) in cases.iter().enumerate() {
            assert_interval_contains_samples(node, idx as u32 + 1);
        }
    }
}
