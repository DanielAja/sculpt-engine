//! Signed distance evaluation of shape programs. Negative inside, positive outside, millimetres.
//!
//! Primitive and operator formulas follow Inigo Quilez's distance-function articles
//! (https://iquilezles.org/articles/distfunctions/, code under the MIT licence, (c) Inigo Quilez) and his
//! polynomial smooth minimum (https://iquilezles.org/articles/smin/). Written fresh in Rust here.

use super::program::{Node, Shape};
use glam::{EulerRot, Quat, Vec2, Vec3};

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

#[derive(Debug, Clone)]
enum Kind {
    Sphere(f32),
    /// Precomputed (inner = half - round, half, round)
    Cuboid { inner: Vec3, half: Vec3, round: f32 },
    /// Precomputed (r, hh, r_in = r - round, h_in = hh - round, round)
    Cylinder { r: f32, hh: f32, r_in: f32, h_in: f32, round: f32 },
    /// Precomputed capped cone constants
    Cone { r1: f32, r2: f32, hh: f32, k1: Vec2, k2: Vec2, inv_k2_dot: f32 },
    Torus(f32, f32),
    /// Precomputed (a, b, ba = b - a, inv_ba_dot, r)
    Capsule { a: Vec3, b: Vec3, ba: Vec3, inv_ba_dot: f32, r: f32 },
    /// Precomputed (half, inv_r, inv_r2, neg_min_r)
    Ellipsoid { half: Vec3, inv_r: Vec3, inv_r2: Vec3, neg_min_r: f32 },
    /// Precomputed pyramid constants
    Pyramid { base: f32, h: f32, inv_s: f32, h_over_s: f32, m2: f32, inv_m2_p: f32, inv_m2: f32 },
    Lathe(Vec<Vec2>),
    Extrude(Vec<Vec2>, f32),
    Union(Vec<Compiled>, f32),
    Difference(Vec<Compiled>, f32),
    Intersection(Vec<Compiled>, f32),
    Array(Box<Compiled>, u32, Vec3),
}

/// A node ready for evaluation: inverse placement precomputed and bounds in the parent's frame.
#[derive(Debug, Clone)]
pub struct Compiled {
    inv_rot: Quat,
    at: Vec3,
    inv_scale_stretch: Vec3,
    /// Distance correction for scale and stretch (keeps the field a bound: |grad| <= 1).
    lip: f32,
    has_at: bool,
    has_rot: bool,
    has_scale: bool,
    mirror: Option<usize>,
    hollow: f32,
    kind: Kind,
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
        let kind = match &n.shape {
            Shape::Sphere { r } => Kind::Sphere(*r),
            Shape::Cuboid { size, round } => {
                let half = v3(*size) * 0.5;
                let r = round.min(half.min_element());
                Kind::Cuboid { inner: half - Vec3::splat(r), half, round: r }
            }
            Shape::Cylinder { r, h, round } => {
                let hh = h * 0.5;
                let rd = round.min(*r).min(hh);
                Kind::Cylinder { r: *r, hh, r_in: *r - rd, h_in: hh - rd, round: rd }
            }
            Shape::Cone { r1, r2, h } => {
                let hh = h * 0.5;
                let k1 = Vec2::new(*r2, hh);
                let k2 = Vec2::new(*r2 - *r1, 2.0 * hh);
                let d2 = dot2(k2);
                Kind::Cone { r1: *r1, r2: *r2, hh, k1, k2, inv_k2_dot: if d2 > 0.0 { 1.0 / d2 } else { 0.0 } }
            }
            Shape::Torus { major, minor } => Kind::Torus(*major, *minor),
            Shape::Capsule { from, to, r } => {
                let a = v3(*from);
                let b = v3(*to);
                let ba = b - a;
                let d2 = ba.length_squared();
                Kind::Capsule { a, b, ba, inv_ba_dot: if d2 > 0.0 { 1.0 / d2 } else { 0.0 }, r: *r }
            }
            Shape::Ellipsoid { size } => {
                let half = v3(*size) * 0.5;
                let inv_r = Vec3::ONE / half;
                Kind::Ellipsoid { half, inv_r, inv_r2: inv_r * inv_r, neg_min_r: -half.min_element() }
            }
            Shape::Pyramid { base, h } => {
                let inv_s = 1.0 / *base;
                let h_over_s = *h * inv_s;
                let m2 = h_over_s * h_over_s + 0.25;
                Kind::Pyramid { base: *base, h: *h, inv_s, h_over_s, m2, inv_m2_p: 1.0 / (m2 + 0.25), inv_m2: 1.0 / m2 }
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
                Kind::Lathe(pts)
            }
            Shape::Extrude { outline, h } => Kind::Extrude(outline.iter().map(|p| Vec2::new(p[0], p[1])).collect(), h * 0.5),
            Shape::Union { children, smooth } => Kind::Union(children.iter().map(Compiled::new).collect(), *smooth),
            Shape::Difference { children, smooth } => Kind::Difference(children.iter().map(Compiled::new).collect(), *smooth),
            Shape::Intersection { children, smooth } => Kind::Intersection(children.iter().map(Compiled::new).collect(), *smooth),
            Shape::Array { child, count, offset } => Kind::Array(Box::new(Compiled::new(child)), *count, v3(*offset)),
        };
        let stretch = v3(n.stretch);
        let at = v3(n.at);
        let has_at = at != Vec3::ZERO;
        let has_rot = n.rot != [0.0, 0.0, 0.0];
        let rot = if has_rot { rotation(n.rot) } else { Quat::IDENTITY };
        let scale_stretch = stretch * n.scale;
        let has_scale = scale_stretch != Vec3::ONE;
        let mut c = Compiled {
            inv_rot: rot.inverse(),
            at,
            inv_scale_stretch: Vec3::ONE / scale_stretch,
            lip: n.scale * stretch.min_element(),
            has_at,
            has_rot,
            has_scale,
            mirror: n.mirror.map(|a| a.index()),
            hollow: n.hollow,
            kind,
            aabb: Aabb::EMPTY,
        };
        let local = c.local_aabb();
        let mut b = Aabb::EMPTY;
        if !local.is_empty() {
            for k in local.corners() {
                let p = rot * (k * stretch * n.scale) + c.at;
                b = b.union(&Aabb::new(p, p));
            }
        }
        if let Some(a) = c.mirror {
            let mut m = b;
            m.min[a] = -b.max[a];
            m.max[a] = -b.min[a];
            b = b.union(&m);
        }
        c.aabb = b;
        c
    }

    fn local_aabb(&self) -> Aabb {
        match &self.kind {
            Kind::Sphere(r) => Aabb::half(Vec3::splat(*r)),
            Kind::Cuboid { half, .. } => Aabb::half(*half),
            Kind::Cylinder { r, hh, .. } => Aabb::half(Vec3::new(*r, *r, *hh)),
            Kind::Cone { r1, r2, hh, .. } => {
                let r = r1.max(*r2);
                Aabb::half(Vec3::new(r, r, *hh))
            }
            Kind::Torus(a, b) => Aabb::half(Vec3::new(a + b, a + b, *b)),
            Kind::Capsule { a, b, r, .. } => Aabb::new(a.min(*b) - Vec3::splat(*r), a.max(*b) + Vec3::splat(*r)),
            Kind::Ellipsoid { half, .. } => Aabb::half(*half),
            Kind::Pyramid { base: s, h, .. } => Aabb::new(Vec3::new(-s / 2.0, -s / 2.0, -h / 2.0), Vec3::new(s / 2.0, s / 2.0, h / 2.0)),
            Kind::Lathe(pts) => {
                let r = pts.iter().fold(0.0f32, |m, p| m.max(p.x));
                let (lo, hi) = pts.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| (lo.min(p.y), hi.max(p.y)));
                Aabb::new(Vec3::new(-r, -r, lo), Vec3::new(r, r, hi))
            }
            Kind::Extrude(pts, hh) => {
                let lo = pts.iter().fold(Vec2::splat(f32::INFINITY), |m, p| m.min(*p));
                let hi = pts.iter().fold(Vec2::splat(f32::NEG_INFINITY), |m, p| m.max(*p));
                Aabb::new(lo.extend(-hh), hi.extend(*hh))
            }
            Kind::Union(c, k) => c.iter().fold(Aabb::EMPTY, |b, c| b.union(&c.aabb)).grow(k * 0.25),
            Kind::Difference(c, _) => c[0].aabb,
            Kind::Intersection(c, _) => c.iter().skip(1).fold(c[0].aabb, |b, c| b.intersect(&c.aabb)),
            Kind::Array(c, n, off) => {
                let last = Aabb::new(c.aabb.min + *off * (*n as f32 - 1.0), c.aabb.max + *off * (*n as f32 - 1.0));
                c.aabb.union(&last)
            }
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
            p = self.inv_rot * p;
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

    #[inline(always)]
    fn kind_eval(&self, p: Vec3) -> f32 {
        match &self.kind {
            Kind::Sphere(r) => p.length() - r,
            Kind::Cuboid { inner, round, .. } => {
                let q = p.abs() - *inner;
                q.max(Vec3::ZERO).length() + q.max_element().min(0.0) - round
            }
            Kind::Cylinder { r_in, h_in, round, .. } => {
                let d = Vec2::new(p.x.hypot(p.y) - *r_in, p.z.abs() - *h_in);
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
            Kind::Lathe(pts) => polygon(Vec2::new((p.x * p.x + p.y * p.y).sqrt(), p.z), pts),
            Kind::Extrude(pts, hh) => {
                let w = Vec2::new(polygon(p.truncate(), pts), p.z.abs() - hh);
                w.x.max(w.y).min(0.0) + w.max(Vec2::ZERO).length()
            }
            Kind::Union(children, k) => {
                let mut d = f32::INFINITY;
                for c in children {
                    // A child whose bounds are farther than the blend reach can't change the result's sign.
                    if d.is_finite() {
                        let reach = d + k;
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
                    let reach = k + d.abs();
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
fn polygon(p: Vec2, v: &[Vec2]) -> f32 {
    let n = v.len();
    let mut d = dot2(p - v[0]);
    let mut s = 1.0;
    let mut j = n - 1;
    for i in 0..n {
        let e = v[j] - v[i];
        let w = p - v[i];
        let b = w - e * (w.dot(e) / dot2(e).max(1e-12)).clamp(0.0, 1.0);
        d = d.min(dot2(b));
        let c = (p.y >= v[i].y, p.y < v[j].y, e.x * w.y > e.y * w.x);
        if (c.0 && c.1 && c.2) || (!c.0 && !c.1 && !c.2) {
            s = -s;
        }
        j = i;
    }
    s * d.sqrt()
}

/// Central-difference gradient of `f` at `p` with step `e`.
pub fn gradient(f: &Compiled, p: Vec3, e: f32) -> Vec3 {
    let dx = Vec3::new(e, 0.0, 0.0);
    let dy = Vec3::new(0.0, e, 0.0);
    let dz = Vec3::new(0.0, 0.0, e);
    Vec3::new(f.eval(p + dx) - f.eval(p - dx), f.eval(p + dy) - f.eval(p - dy), f.eval(p + dz) - f.eval(p - dz)) / (2.0 * e)
}

/// Bounds of what is actually solid, from a coarse sampling of the field (within about 1/40 of the size).
/// The compiled bounds are conservative (a difference keeps its base's box), which would make layout float
/// things like an igloo, whose lower half is cut away.
pub fn tight_aabb(f: &Compiled) -> Aabb {
    let b = f.aabb;
    if b.is_empty() {
        return b;
    }
    const N: usize = 40;
    let step = b.size() / (N as f32 - 1.0);
    let mut out = Aabb::EMPTY;
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                let p = b.min + step * glam::Vec3::new(i as f32, j as f32, k as f32);
                let d = f.eval(p);
                if d < 0.0 {
                    // The ball of radius |d| around p is solid, so the shape reaches at least that far.
                    out = out.union(&Aabb::new(p + d, p - d));
                }
            }
        }
    }
    if out.is_empty() { b } else { out.intersect(&b) }
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
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                let p = b.min + step * Vec3::new(i as f32, j as f32, k as f32);
                if fs.iter().any(|f| f.eval(p) < 0.0) {
                    inside.push(p);
                }
            }
        }
    }
    let top = *inside.iter().max_by(|a, c| a.z.total_cmp(&c.z))?;
    let r = b.size().max_element() * 0.15;
    let near: Vec<Vec3> = inside.iter().copied().filter(|p| (p.truncate() - top.truncate()).length() < r && p.z > top.z - r).collect();
    let (lo, hi) = near.iter().fold((Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)), |(l, h), p| (l.min(*p), h.max(*p)));
    let c = (lo + hi) * 0.5;
    Some((Vec3::new(c.x, c.y, top.z + step.z * 0.5), (hi - lo).truncate().max_element() + step.x))
}
