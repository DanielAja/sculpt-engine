//! Triangle BVH: binned SAH build, refit, closest ray hit and sphere (brush footprint) queries.
//!
//! Between refits, moved vertices may leave their node boxes. Instead of refitting after every dab, the
//! tree tracks `slack`: the largest distance any vertex has moved since the last refit. Queries grow
//! every box by `slack`, so they stay exact; `refit` runs once the slack is large compared to the brush.

use glam::{Vec3, Vec3A};

/// 32 bytes, laid out for upload to the GPU path tracer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Node {
    pub min: [f32; 3],
    /// Interior: index of the left child (the right one follows). Leaf: first entry in `tri_ids`.
    pub left_first: u32,
    pub max: [f32; 3],
    /// Triangle count; 0 for interior nodes.
    pub count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub t: f32,
    pub tri: u32,
    pub point: Vec3,
}

#[derive(Debug, Clone, Default)]
pub struct Bvh {
    pub nodes: Vec<Node>,
    pub tri_ids: Vec<u32>,
    pub slack: f32,
}

const LEAF: usize = 4;
const BINS: usize = 12;

#[derive(Clone, Copy)]
struct Aabb {
    min: Vec3A,
    max: Vec3A,
}
impl Aabb {
    const EMPTY: Aabb = Aabb { min: Vec3A::splat(f32::INFINITY), max: Vec3A::splat(f32::NEG_INFINITY) };
    #[inline(always)]
    fn grow(&mut self, p: Vec3A) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
    #[inline(always)]
    fn union(&mut self, b: &Aabb) {
        self.min = self.min.min(b.min);
        self.max = self.max.max(b.max);
    }
    #[inline(always)]
    fn area(&self) -> f32 {
        let d = (self.max - self.min).max(Vec3A::ZERO);
        d.x * d.y + d.y * d.z + d.z * d.x
    }
}

#[derive(Clone, Copy)]
struct TriInfo {
    bb: Aabb,
    cent: Vec3A,
}

#[inline(always)]
fn tri_box(p: &[Vec3], t: &[u32; 3]) -> Aabb {
    let a = Vec3A::from(p[t[0] as usize]);
    let b = Vec3A::from(p[t[1] as usize]);
    let c = Vec3A::from(p[t[2] as usize]);
    Aabb { min: a.min(b).min(c), max: a.max(b).max(c) }
}

#[cfg(feature = "threads")]
const PARALLEL_BUILD_MIN: usize = 4096;

#[inline]
fn compute_bounds_and_split(ids: &mut [u32], info: &[TriInfo]) -> (Aabb, Option<usize>) {
    let count = ids.len();
    let mut bb = Aabb::EMPTY;
    let mut cb = Aabb::EMPTY;
    for &t in ids.iter() {
        let ti = &info[t as usize];
        bb.union(&ti.bb);
        cb.grow(ti.cent);
    }
    if count <= LEAF {
        return (bb, None);
    }
    let ext = cb.max - cb.min;
    let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
    if ext[axis] <= 0.0 {
        return (bb, None); // all centroids coincide: keep a (rare) fat leaf
    }
    let min_c = cb.min[axis];
    let scale = BINS as f32 / ext[axis];
    let bin_of = |t: u32| (((info[t as usize].cent[axis] - min_c) * scale) as usize).min(BINS - 1);
    let mut bins = [(Aabb::EMPTY, 0u32); BINS];
    for &t in ids.iter() {
        let b = &mut bins[bin_of(t)];
        b.0.union(&info[t as usize].bb);
        b.1 += 1;
    }
    let mut right_area = [0f32; BINS];
    let mut acc = (Aabb::EMPTY, 0u32);
    for i in (1..BINS).rev() {
        acc.0.union(&bins[i].0);
        acc.1 += bins[i].1;
        right_area[i] = acc.0.area() * acc.1 as f32;
    }
    let mut best = (f32::INFINITY, 0);
    let mut left = (Aabb::EMPTY, 0u32);
    for i in 0..BINS - 1 {
        left.0.union(&bins[i].0);
        left.1 += bins[i].1;
        let cost = left.0.area() * left.1 as f32 + right_area[i + 1];
        if left.1 > 0 && (left.1 as usize) < count && cost < best.0 {
            best = (cost, i);
        }
    }
    if best.0 >= bb.area() * count as f32 && count <= 16 {
        return (bb, None);
    }
    let (mut i, mut j) = (0usize, count);
    while i < j {
        if bin_of(ids[i]) <= best.1 {
            i += 1;
        } else {
            j -= 1;
            ids.swap(i, j);
        }
    }
    if i == 0 || i == count {
        return (bb, None);
    }
    (bb, Some(i))
}

fn build_seq(ids: &mut [u32], first_base: u32, info: &[TriInfo]) -> Vec<Node> {
    let n = ids.len();
    let mut nodes = Vec::with_capacity(2 * n / LEAF + 1);
    nodes.push(Node { left_first: 0, count: n as u32, ..Default::default() });
    let mut stack = Vec::with_capacity(64);
    stack.push((0usize, 0usize, n));
    while let Some((ni, rel_first, count)) = stack.pop() {
        let slice = &mut ids[rel_first..rel_first + count];
        let (bb, split) = compute_bounds_and_split(slice, info);
        nodes[ni].min = bb.min.to_array();
        nodes[ni].max = bb.max.to_array();
        let Some(i) = split else {
            nodes[ni].left_first = first_base + rel_first as u32;
            nodes[ni].count = count as u32;
            continue;
        };
        let l = nodes.len();
        nodes.push(Node::default());
        nodes.push(Node::default());
        nodes[ni].left_first = l as u32;
        nodes[ni].count = 0;
        stack.push((l + 1, rel_first + i, count - i));
        stack.push((l, rel_first, i));
    }
    nodes
}

#[cfg(feature = "threads")]
fn build_par(ids: &mut [u32], first_base: u32, info: &[TriInfo]) -> Vec<Node> {
    if ids.len() < PARALLEL_BUILD_MIN {
        return build_seq(ids, first_base, info);
    }
    let (bb, split) = compute_bounds_and_split(ids, info);
    let Some(split_idx) = split else {
        return vec![Node { min: bb.min.to_array(), left_first: first_base, max: bb.max.to_array(), count: ids.len() as u32 }];
    };
    let (left_ids, right_ids) = ids.split_at_mut(split_idx);
    let (mut lnodes, mut rnodes) = rayon::join(
        || build_par(left_ids, first_base, info),
        || build_par(right_ids, first_base + split_idx as u32, info),
    );
    let l_len = lnodes.len();
    let r_start = 2 + l_len;
    let mut out = Vec::with_capacity(1 + l_len + rnodes.len());
    out.push(Node { min: bb.min.to_array(), left_first: 1, max: bb.max.to_array(), count: 0 });
    for n in &mut lnodes {
        if n.count == 0 {
            n.left_first += 2;
        }
    }
    let r_shift = (r_start - 1) as u32;
    for n in &mut rnodes {
        if n.count == 0 {
            n.left_first += r_shift;
        }
    }
    out.push(lnodes[0]);
    out.push(rnodes[0]);
    out.extend_from_slice(&lnodes[1..]);
    out.extend_from_slice(&rnodes[1..]);
    out
}

/// Fixed-capacity stack with heap overflow for panic-free, zero-alloc BVH traversal.
struct LocalStack {
    buf: [usize; 64],
    len: usize,
    overflow: Vec<usize>,
}
impl LocalStack {
    #[inline(always)]
    fn new() -> Self {
        LocalStack { buf: [0; 64], len: 0, overflow: Vec::new() }
    }
    #[inline(always)]
    fn push(&mut self, v: usize) {
        if self.len < 64 {
            self.buf[self.len] = v;
            self.len += 1;
        } else {
            self.overflow.push(v);
        }
    }
    #[inline(always)]
    fn pop(&mut self) -> Option<usize> {
        if !self.overflow.is_empty() {
            return self.overflow.pop();
        }
        if self.len > 0 {
            self.len -= 1;
            Some(self.buf[self.len])
        } else {
            None
        }
    }
}

impl Bvh {
    pub fn build(positions: &[Vec3], tris: &[[u32; 3]]) -> Bvh {
        let n = tris.len();
        if n == 0 {
            return Bvh::default();
        }
        let mut tri_ids: Vec<u32> = (0..n as u32).collect();
        #[cfg(feature = "threads")]
        let info: Vec<TriInfo> = if n >= 4096 {
            use rayon::prelude::*;
            tris.par_iter()
                .map(|t| {
                    let bb = tri_box(positions, t);
                    TriInfo { bb, cent: (bb.min + bb.max) * 0.5 }
                })
                .collect()
        } else {
            tris.iter()
                .map(|t| {
                    let bb = tri_box(positions, t);
                    TriInfo { bb, cent: (bb.min + bb.max) * 0.5 }
                })
                .collect()
        };
        #[cfg(not(feature = "threads"))]
        let info: Vec<TriInfo> = tris
            .iter()
            .map(|t| {
                let bb = tri_box(positions, t);
                TriInfo { bb, cent: (bb.min + bb.max) * 0.5 }
            })
            .collect();

        #[cfg(feature = "threads")]
        let nodes = build_par(&mut tri_ids, 0, &info);
        #[cfg(not(feature = "threads"))]
        let nodes = build_seq(&mut tri_ids, 0, &info);

        Bvh { nodes, tri_ids, slack: 0.0 }
    }

    /// Recomputes every box bottom up (children always come after their parent) and clears the slack.
    pub fn refit(&mut self, positions: &[Vec3], tris: &[[u32; 3]]) {
        for ni in (0..self.nodes.len()).rev() {
            let nd = self.nodes[ni];
            let (min, max) = if nd.count > 0 {
                let mut lo = Vec3A::splat(f32::INFINITY);
                let mut hi = Vec3A::splat(f32::NEG_INFINITY);
                let start = nd.left_first as usize;
                let end = start + nd.count as usize;
                for &t in &self.tri_ids[start..end] {
                    let tri = tris[t as usize];
                    let a = Vec3A::from(positions[tri[0] as usize]);
                    let b = Vec3A::from(positions[tri[1] as usize]);
                    let c = Vec3A::from(positions[tri[2] as usize]);
                    lo = lo.min(a).min(b).min(c);
                    hi = hi.max(a).max(b).max(c);
                }
                (lo, hi)
            } else {
                let l = nd.left_first as usize;
                let r = l + 1;
                let lo = Vec3A::from_array(self.nodes[l].min).min(Vec3A::from_array(self.nodes[r].min));
                let hi = Vec3A::from_array(self.nodes[l].max).max(Vec3A::from_array(self.nodes[r].max));
                (lo, hi)
            };
            self.nodes[ni].min = min.to_array();
            self.nodes[ni].max = max.to_array();
        }
        self.slack = 0.0;
    }

    #[inline(always)]
    fn slab(n: &Node, inv: Vec3, neg_o_inv: Vec3, s_inv: Vec3, tmax: f32) -> Option<f32> {
        let t0 = Vec3::from_array(n.min).mul_add(inv, neg_o_inv);
        let t1 = Vec3::from_array(n.max).mul_add(inv, neg_o_inv);
        let lo = (t0.min(t1) - s_inv).max_element().max(0.0);
        let hi = (t0.max(t1) + s_inv).min_element().min(tmax);
        (lo <= hi).then_some(lo)
    }

    /// Closest hit along `dir` (need not be unit length; `t` is in units of `dir`). Back faces count.
    pub fn ray(&self, positions: &[Vec3], tris: &[[u32; 3]], o: Vec3, dir: Vec3) -> Option<Hit> {
        if self.nodes.is_empty() || tris.is_empty() {
            return None;
        }
        let inv = dir.recip();
        let neg_o_inv = -o * inv;
        let s_inv = Vec3::splat(self.slack) * inv.abs();
        let mut best: Option<Hit> = None;
        let mut tmax = f32::INFINITY;
        let mut stack = LocalStack::new();
        if Self::slab(&self.nodes[0], inv, neg_o_inv, s_inv, tmax).is_some() {
            stack.push(0usize);
        }
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni];
            if n.count > 0 {
                for &t in &self.tri_ids[n.left_first as usize..(n.left_first + n.count) as usize] {
                    let tri = tris[t as usize];
                    let (a, b, c) = (positions[tri[0] as usize], positions[tri[1] as usize], positions[tri[2] as usize]);
                    if let Some(tt) = ray_tri(o, dir, a, b, c) {
                        if tt < tmax {
                            tmax = tt;
                            best = Some(Hit { t: tt, tri: t, point: o + dir * tt });
                        }
                    }
                }
                continue;
            }
            let (l, r) = (n.left_first as usize, n.left_first as usize + 1);
            let dl = Self::slab(&self.nodes[l], inv, neg_o_inv, s_inv, tmax);
            let dr = Self::slab(&self.nodes[r], inv, neg_o_inv, s_inv, tmax);
            // Push the farther child first so the nearer one is visited first.
            match (dl, dr) {
                (Some(a), Some(b)) if a <= b => {
                    stack.push(r);
                    stack.push(l);
                }
                (Some(_), Some(_)) => {
                    stack.push(l);
                    stack.push(r);
                }
                (Some(_), None) => stack.push(l),
                (None, Some(_)) => stack.push(r),
                (None, None) => {}
            }
        }
        best
    }

    /// Appends every vertex within `r` of `c` (strictly closer) to `out`, sorted by index, no duplicates.
    /// `stamp` must have one entry per vertex; it is reused across calls to avoid clearing.
    pub fn sphere_vertices(&self, positions: &[Vec3], tris: &[[u32; 3]], c: Vec3, r: f32, stamp: &mut Stamp, out: &mut Vec<u32>) {
        out.clear();
        if self.nodes.is_empty() || !(r > 0.0) {
            return;
        }
        let tag = stamp.next();
        let reach = r + self.slack;
        let reach2 = reach * reach;
        let r2 = r * r;
        let mut stack = LocalStack::new();
        stack.push(0usize);
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni];
            let q = c.clamp(Vec3::from(n.min), Vec3::from(n.max));
            if (q - c).length_squared() > reach2 {
                continue;
            }
            if n.count == 0 {
                stack.push(n.left_first as usize);
                stack.push(n.left_first as usize + 1);
                continue;
            }
            for &t in &self.tri_ids[n.left_first as usize..(n.left_first + n.count) as usize] {
                for &v in &tris[t as usize] {
                    let s = &mut stamp.marks[v as usize];
                    if *s != tag {
                        *s = tag;
                        if (positions[v as usize] - c).length_squared() < r2 {
                            out.push(v);
                        }
                    }
                }
            }
        }
        out.sort_unstable();
    }
}

/// Per vertex "seen" marks that never need clearing (bump the tag instead).
#[derive(Debug, Clone, Default)]
pub struct Stamp {
    pub marks: Vec<u32>,
    tag: u32,
}
impl Stamp {
    pub fn new(n: usize) -> Stamp {
        Stamp { marks: vec![0; n], tag: 0 }
    }
    pub fn resize(&mut self, n: usize) {
        self.marks.resize(n, 0);
    }
    pub fn next(&mut self) -> u32 {
        self.tag = self.tag.wrapping_add(1);
        if self.tag == 0 {
            self.marks.fill(0);
            self.tag = 1;
        }
        self.tag
    }
}

/// Möller-Trumbore, both sides.
pub fn ray_tri(o: Vec3, d: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let e1 = b - a;
    let e2 = c - a;
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 0.0).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{uv_sphere, SculptMesh};

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
        fn v(&mut self) -> Vec3 {
            Vec3::new(self.f(), self.f(), self.f()) * 2.0 - 1.0
        }
    }

    fn brute_ray(m: &SculptMesh, o: Vec3, d: Vec3) -> Option<f32> {
        m.tris.iter().filter_map(|t| ray_tri(o, d, m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize])).reduce(f32::min)
    }

    #[test]
    fn every_triangle_once_and_boxes_contain() {
        for (seg, rings) in [(64, 32), (128, 64)] {
            let m = SculptMesh::from_mesh(&uv_sphere(10.0, seg, rings));
            let b = Bvh::build(&m.positions, &m.tris);
            let mut ids = b.tri_ids.clone();
            ids.sort_unstable();
            assert_eq!(ids, (0..m.tris.len() as u32).collect::<Vec<_>>());
            let leaves: u32 = b.nodes.iter().map(|n| n.count).sum();
            assert_eq!(leaves as usize, m.tris.len());
            for (ni, n) in b.nodes.iter().enumerate() {
                if n.count == 0 {
                    assert!(n.left_first as usize > ni, "parent {ni} must precede children");
                } else {
                    for &t in &b.tri_ids[n.left_first as usize..(n.left_first + n.count) as usize] {
                        for &v in &m.tris[t as usize] {
                            let p = m.positions[v as usize];
                            assert!(p.cmpge(Vec3::from(n.min)).all() && p.cmple(Vec3::from(n.max)).all());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rays_agree_with_brute_force_before_and_after_moves() {
        let mut m = SculptMesh::from_mesh(&uv_sphere(10.0, 48, 24));
        let mut b = Bvh::build(&m.positions, &m.tris);
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let check = |m: &SculptMesh, b: &Bvh, rng: &mut Rng| {
            for _ in 0..500 {
                let o = rng.v() * 30.0;
                let d = (rng.v() * 5.0 - o).normalize();
                let got = b.ray(&m.positions, &m.tris, o, d).map(|h| h.t);
                let want = brute_ray(m, o, d);
                match (got, want) {
                    (Some(g), Some(w)) => assert!((g - w).abs() < 1e-4, "{g} vs {w}"),
                    (g, w) => assert_eq!(g.is_some(), w.is_some()),
                }
            }
        };
        check(&m, &b, &mut rng);
        // Move some vertices without refitting: slack keeps queries exact.
        let mut moved: f32 = 0.0;
        for i in (0..m.positions.len()).step_by(7) {
            let d = rng.v() * 1.5;
            moved = moved.max(d.length());
            m.positions[i] += d;
        }
        b.slack = moved;
        check(&m, &b, &mut rng);
        b.refit(&m.positions, &m.tris);
        check(&m, &b, &mut rng);
    }

    #[test]
    fn sphere_query_matches_brute_force() {
        let m = SculptMesh::from_mesh(&uv_sphere(10.0, 48, 24));
        let b = Bvh::build(&m.positions, &m.tris);
        let mut stamp = Stamp::new(m.positions.len());
        let mut out = Vec::new();
        let mut rng = Rng(42);
        for _ in 0..50 {
            let c = rng.v() * 11.0;
            let r = 1.0 + rng.f() * 6.0;
            b.sphere_vertices(&m.positions, &m.tris, c, r, &mut stamp, &mut out);
            let want: Vec<u32> = (0..m.positions.len() as u32).filter(|&v| (m.positions[v as usize] - c).length() < r).collect();
            assert_eq!(out, want);
        }
    }
}
