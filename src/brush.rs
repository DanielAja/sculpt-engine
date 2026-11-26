//! Brush dabs. Includes Nomad Sculpt-grade brushes (`Draw`, `Clay`, `Inflate`, `Crease`, `Pinch`,
//! `Grab`, `Smooth`, `Flatten`, `Mask`), configurable falloff curves, per-vertex masking, 3-axis
//! mirror symmetry, and radial symmetry around Z.
//!
//! Each centre reads the positions from before it and writes new ones afterwards (Jacobi style), so the
//! result does not depend on how many threads ran it.

use crate::bvh::{Bvh, Stamp};
use crate::mesh::SculptMesh;
use glam::{Mat3, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrushKind {
    Draw,
    Clay,
    Inflate,
    Crease,
    Pinch,
    Grab,
    Smooth,
    Flatten,
    Mask,
}

impl BrushKind {
    /// Parses brush names across all platforms; anything unknown defaults to `Draw`.
    pub fn parse(s: &str) -> BrushKind {
        match s {
            "clay" => BrushKind::Clay,
            "inflate" => BrushKind::Inflate,
            "crease" => BrushKind::Crease,
            "pinch" => BrushKind::Pinch,
            "grab" => BrushKind::Grab,
            "smooth" => BrushKind::Smooth,
            "flatten" => BrushKind::Flatten,
            "mask" => BrushKind::Mask,
            _ => BrushKind::Draw,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Falloff {
    /// `(1 - (d/r)^2)^2` — organic smooth bell (default, matches Sculptor's original curve).
    #[default]
    Smooth,
    /// `1 - d/r` — crisp cone falloff.
    Linear,
    /// `(1 - d/r)^3` — sharp tip for creases and fine detail.
    Sharp,
    /// `1.0` inside radius — uniform stamp/clay buildup.
    Constant,
}

impl Falloff {
    pub fn parse(s: &str) -> Falloff {
        match s {
            "linear" => Falloff::Linear,
            "sharp" => Falloff::Sharp,
            "constant" => Falloff::Constant,
            _ => Falloff::Smooth,
        }
    }

    #[inline(always)]
    pub fn eval(self, dist_sq: f32, inv_r2: f32) -> f32 {
        match self {
            Falloff::Smooth => {
                let t = (1.0 - dist_sq * inv_r2).max(0.0);
                t * t
            }
            Falloff::Linear => (1.0 - (dist_sq * inv_r2).sqrt()).max(0.0),
            Falloff::Sharp => {
                let t = (1.0 - (dist_sq * inv_r2).sqrt()).max(0.0);
                t * t * t
            }
            Falloff::Constant => {
                if dist_sq * inv_r2 < 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrushParams {
    pub kind: BrushKind,
    /// Object local units (the stroke converts from world mm).
    pub radius: f32,
    /// -1 to 1; negative digs in for draw, clay, inflate, unmasks for mask, raises for crease.
    pub strength: f32,
    /// Mirror across the object's local X, Y, Z planes.
    pub symmetry: [bool; 3],
    /// Brush footprint falloff curve.
    pub falloff: Falloff,
    /// Radial symmetry copies around the object's local Z axis (1 = off, 2..=16).
    pub radial_symmetry: u32,
}

impl Default for BrushParams {
    fn default() -> Self {
        BrushParams {
            kind: BrushKind::Draw,
            radius: 4.0,
            strength: 0.6,
            symmetry: [false; 3],
            falloff: Falloff::Smooth,
            radial_symmetry: 1,
        }
    }
}

/// Computes all unique symmetry `(center, linear_transform)` pairs for a dab/grab at `center`
/// with 3-axis mirror symmetry and optional radial symmetry around Z (`radial_symmetry` in 1..=16).
pub fn symmetry_transforms(center: Vec3, radius: f32, symmetry: [bool; 3], radial_symmetry: u32) -> Vec<(Vec3, Mat3)> {
    let min_dist2 = (radius * 0.01) * (radius * 0.01);
    let mut out: Vec<(Vec3, Mat3)> = Vec::with_capacity(16);
    out.push((center, Mat3::IDENTITY));
    for axis in 0..3 {
        if symmetry[axis] {
            let n = out.len();
            let mut mirror_diag = Vec3::ONE;
            mirror_diag[axis] = -1.0;
            let mirror_mat = Mat3::from_diagonal(mirror_diag);
            for i in 0..n {
                let (c, m) = out[i];
                let mut mirrored = c;
                mirrored[axis] = -mirrored[axis];
                if (mirrored - c).length_squared() > min_dist2 {
                    out.push((mirrored, mirror_mat * m));
                }
            }
        }
    }
    let radial = radial_symmetry.clamp(1, 16);
    if radial > 1 {
        let base_len = out.len();
        for k in 1..radial {
            let angle = std::f32::consts::TAU * (k as f32) / (radial as f32);
            let rot = Mat3::from_rotation_z(angle);
            for i in 0..base_len {
                let (c, m) = out[i];
                let rc = rot * c;
                if out.iter().all(|(existing, _)| (*existing - rc).length_squared() > min_dist2) {
                    out.push((rc, rot * m));
                }
            }
        }
    }
    out
}

/// Reusable buffers for dabs on one mesh.
#[derive(Debug, Default)]
pub struct Scratch {
    stamp: Stamp,
    pub poly_stamp: Stamp,
    pub vert_stamp: Stamp,
    hits: Vec<u32>,
    weights: Vec<f32>,
    eff_weights: Vec<f32>,
    new_pos: Vec<Vec3>,
    moved_verts: Vec<u32>,
}

impl Scratch {
    /// Pre-sized for a mesh of `n` vertices, so the first dab doesn't allocate.
    pub fn new(n: usize) -> Scratch {
        Scratch { stamp: Stamp::new(n), vert_stamp: Stamp::new(n), ..Default::default() }
    }
}

/// Applies one dab and refreshes normals around moved vertices using reusable `Stamp`s (zero heap
/// allocations and zero sorting per dab). Appends refreshed vertices to `dirty`.
pub fn dab_and_update_normals(
    m: &mut SculptMesh,
    bvh: &Bvh,
    s: &mut Scratch,
    center: Vec3,
    p: &BrushParams,
    moved: &mut Vec<(u32, Vec3)>,
    dirty: &mut Vec<u32>,
) -> f32 {
    if p.kind == BrushKind::Mask {
        dab_mask(m, bvh, s, center, p, dirty);
        return 0.0;
    }
    let before_len = moved.len();
    let max_move = dab(m, bvh, s, center, p, moved);
    if moved.len() == before_len {
        return max_move;
    }
    s.moved_verts.clear();
    s.moved_verts.extend(moved[before_len..].iter().map(|&(v, _)| v));
    m.update_normals_stamped(&s.moved_verts, &mut s.poly_stamp, &mut s.vert_stamp, dirty);
    max_move
}

/// Paints per-vertex mask weights `[0.0, 1.0]` (negative strength unmasks). Appends touched vertices to `dirty`.
pub fn dab_mask(
    m: &mut SculptMesh,
    bvh: &Bvh,
    s: &mut Scratch,
    center: Vec3,
    p: &BrushParams,
    dirty: &mut Vec<u32>,
) {
    let r = p.radius;
    if !(r > 0.0) || m.positions.is_empty() {
        return;
    }
    if m.masks.len() != m.positions.len() {
        m.masks.resize(m.positions.len(), 0.0);
    }
    let syms = symmetry_transforms(center, r, p.symmetry, p.radial_symmetry);
    s.stamp.resize(m.positions.len());
    let st = p.strength.clamp(-1.0, 1.0);
    let inv_r2 = 1.0 / (r * r);
    for (c, _) in syms {
        bvh.sphere_vertices(&m.positions, &m.tris, c, r, &mut s.stamp, &mut s.hits);
        for &v in &s.hits {
            let d2 = (m.positions[v as usize] - c).length_squared();
            let f = p.falloff.eval(d2, inv_r2);
            if f > 0.0 {
                let prev = m.masks[v as usize];
                let next = (prev + st * f * 0.65).clamp(0.0, 1.0);
                if (next - prev).abs() > 1e-6 {
                    m.masks[v as usize] = next;
                    dirty.push(v);
                }
            }
        }
    }
}

/// Footprints at or above this size are split across threads.
const PARALLEL_MIN: usize = 4096;

/// Applies one dab at `center` (object space). Appends (vertex, position before) for every moved vertex to
/// `moved` (a vertex may repeat when mirrored footprints overlap) and returns
/// the largest distance any vertex moved. Normals are refreshed by the caller (`SculptMesh::update_normals`).
pub fn dab(m: &mut SculptMesh, bvh: &Bvh, s: &mut Scratch, center: Vec3, p: &BrushParams, moved: &mut Vec<(u32, Vec3)>) -> f32 {
    let r = p.radius;
    if !(r > 0.0) || m.positions.is_empty() {
        return 0.0;
    }
    if p.kind == BrushKind::Mask {
        let mut ignored_dirty = Vec::new();
        dab_mask(m, bvh, s, center, p, &mut ignored_dirty);
        return 0.0;
    }
    if p.kind == BrushKind::Grab {
        // Grab is driven interactively by SculptTarget::begin_grab / move_grab.
        return 0.0;
    }
    let syms = symmetry_transforms(center, r, p.symmetry, p.radial_symmetry);
    s.stamp.resize(m.positions.len());
    if m.masks.len() != m.positions.len() {
        m.masks.resize(m.positions.len(), 0.0);
    }
    let st = p.strength.clamp(-1.0, 1.0);
    let inv_r2 = 1.0 / (r * r);
    let mut max_move_sq: f32 = 0.0;
    for (c, _) in syms {
        // shortcut: vertices used by no face are never brushed (the web port moves them too); harmless for real meshes.
        bvh.sphere_vertices(&m.positions, &m.tris, c, r, &mut s.stamp, &mut s.hits);
        if s.hits.is_empty() {
            continue;
        }
        s.weights.clear();
        s.weights.reserve(s.hits.len());
        s.eff_weights.clear();
        s.eff_weights.reserve(s.hits.len());
        let (mut avg_n, mut avg_p, mut wsum) = (Vec3::ZERO, Vec3::ZERO, 0f32);
        for &v in &s.hits {
            let q = m.positions[v as usize];
            let f = p.falloff.eval((q - c).length_squared(), inv_r2);
            let mask_unprotected = (1.0 - m.masks[v as usize]).clamp(0.0, 1.0);
            s.weights.push(f);
            s.eff_weights.push(f * mask_unprotected);
            avg_n += m.normals[v as usize] * f;
            avg_p += q * f;
            wsum += f;
        }
        if !(wsum > 0.0) {
            continue;
        }
        let avg_n_len2 = avg_n.length_squared();
        let n = if avg_n_len2 > 1e-12 { avg_n / avg_n_len2.sqrt() } else { Vec3::Z };
        let plane = avg_p / wsum;
        let (pos, nrm, nb, hits, weights) = (&m.positions, &m.normals, &m.neighbours, &s.hits, &s.eff_weights);
        let st_r_scale = st * r * 0.1;
        let abs_st = st.abs();
        s.new_pos.clear();
        match p.kind {
            BrushKind::Draw => {
                let disp = n * st_r_scale;
                par_map(hits.len(), |i| pos[hits[i] as usize] + disp * weights[i], &mut s.new_pos);
            }
            BrushKind::Clay => {
                // Flat-topped organic clay strip buildup along plane offset by strength:
                // fills crevices smoothly toward the elevated plane without ballooning existing peaks.
                let target_plane = plane + n * (st * r * 0.12);
                let target_dot = target_plane.dot(n);
                let blend = (0.45 + 0.55 * abs_st).min(1.0);
                par_map(
                    hits.len(),
                    |i| {
                        let v = hits[i] as usize;
                        let src = pos[v];
                        let dist = target_dot - src.dot(n);
                        let delta = if st >= 0.0 { dist.max(0.0) } else { dist.min(0.0) };
                        src + n * (delta * blend * weights[i])
                    },
                    &mut s.new_pos,
                );
            }
            BrushKind::Inflate => {
                par_map(
                    hits.len(),
                    |i| {
                        let v = hits[i] as usize;
                        pos[v] + nrm[v] * (st_r_scale * weights[i])
                    },
                    &mut s.new_pos,
                );
            }
            BrushKind::Crease => {
                // Pinches vertices inward toward brush center along tangent plane while carving
                // a sharp groove along -n (positive strength) or raising a pinched ridge (+n, negative strength).
                let pinch_k = abs_st * 0.35;
                let groove_k = -st_r_scale * 1.15;
                par_map(
                    hits.len(),
                    |i| {
                        let v = hits[i] as usize;
                        let src = pos[v];
                        let w = weights[i];
                        let to_c = c - src;
                        let tang = to_c - n * to_c.dot(n);
                        src + tang * (pinch_k * w) + n * (groove_k * w * w.sqrt())
                    },
                    &mut s.new_pos,
                );
            }
            BrushKind::Pinch => {
                // Pulls vertices toward brush center along tangent plane to sharpen ridges/edges.
                let pinch_k = st * 0.5;
                par_map(
                    hits.len(),
                    |i| {
                        let v = hits[i] as usize;
                        let src = pos[v];
                        let to_c = c - src;
                        let tang = to_c - n * to_c.dot(n);
                        src + tang * (pinch_k * weights[i])
                    },
                    &mut s.new_pos,
                );
            }
            BrushKind::Smooth => {
                par_map(
                    hits.len(),
                    |i| {
                        let v = hits[i] as usize;
                        let src = pos[v];
                        let row = nb.row(v);
                        if row.is_empty() {
                            return src;
                        }
                        let mut avg = Vec3::ZERO;
                        for &j in row {
                            avg += pos[j as usize];
                        }
                        avg /= row.len() as f32;
                        src + (avg - src) * (abs_st * weights[i])
                    },
                    &mut s.new_pos,
                );
            }
            BrushKind::Flatten => {
                let plane_dot = plane.dot(n);
                par_map(
                    hits.len(),
                    |i| {
                        let v = hits[i] as usize;
                        let src = pos[v];
                        src - n * ((src.dot(n) - plane_dot) * abs_st * weights[i])
                    },
                    &mut s.new_pos,
                );
            }
            BrushKind::Grab | BrushKind::Mask => unreachable!(),
        }
        moved.reserve(s.hits.len());
        for (i, &v) in s.hits.iter().enumerate() {
            let old = m.positions[v as usize];
            let np = s.new_pos[i];
            let d2 = (np - old).length_squared();
            if d2 > 0.0 {
                max_move_sq = max_move_sq.max(d2);
                m.positions[v as usize] = np;
                moved.push((v, old));
            }
        }
    }
    max_move_sq.sqrt()
}

#[cfg(feature = "threads")]
fn par_map<F: Fn(usize) -> Vec3 + Sync + Send>(n: usize, f: F, out: &mut Vec<Vec3>) {
    use rayon::prelude::*;
    if n >= PARALLEL_MIN {
        (0..n).into_par_iter().with_min_len(1024).map(f).collect_into_vec(out);
    } else {
        out.extend((0..n).map(f));
    }
}

#[cfg(not(feature = "threads"))]
fn par_map<F: Fn(usize) -> Vec3 + Sync + Send>(n: usize, f: F, out: &mut Vec<Vec3>) {
    let _ = PARALLEL_MIN;
    out.extend((0..n).map(f));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::Mesh;
    use crate::mesh::uv_sphere;

    fn grid(n: u32) -> Mesh {
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                positions.push(Vec3::new(x as f32 - n as f32 / 2.0, y as f32 - n as f32 / 2.0, 0.0));
            }
        }
        let id = |x: u32, y: u32| y * (n + 1) + x;
        let mut faces = Vec::new();
        for y in 0..n {
            for x in 0..n {
                faces.push(vec![id(x, y), id(x + 1, y), id(x + 1, y + 1), id(x, y + 1)]);
            }
        }
        Mesh { positions, faces, smooth_shading: true }
    }

    fn run(m: &mut SculptMesh, c: Vec3, p: BrushParams) -> Vec<u32> {
        let bvh = Bvh::build(&m.positions, &m.tris);
        let mut s = Scratch::default();
        let mut moved = Vec::new();
        dab(m, &bvh, &mut s, c, &p, &mut moved);
        let vs: Vec<u32> = moved.iter().map(|x| x.0).collect();
        m.update_normals(&vs);
        vs
    }

    fn params(kind: BrushKind, radius: f32, strength: f32) -> BrushParams {
        BrushParams { kind, radius, strength, ..Default::default() }
    }

    #[test]
    fn draw_on_a_flat_grid_matches_hand_maths() {
        let mut m = SculptMesh::from_mesh(&grid(10));
        run(&mut m, Vec3::ZERO, params(BrushKind::Draw, 2.0, 0.5));
        // Centre vertex: f = 1, moves 0.5 * 1 * 2 * 0.1 = 0.1 along +Z.
        let centre = m.positions.iter().find(|p| p.x == 0.0 && p.y == 0.0).unwrap();
        assert!((centre.z - 0.1).abs() < 1e-6);
        // d = 1: t = 0.75, f = 0.5625, dz = 0.05625.
        let side = m.positions.iter().find(|p| p.x == 1.0 && p.y == 0.0).unwrap();
        assert!((side.z - 0.05625).abs() < 1e-6, "{}", side.z);
        // Outside the radius nothing moves.
        assert!(m.positions.iter().filter(|p| p.x.abs() + p.y.abs() >= 3.0).all(|p| p.z == 0.0));
    }

    #[test]
    fn negative_strength_digs_and_inflate_follows_normals() {
        let mut a = SculptMesh::from_mesh(&grid(10));
        run(&mut a, Vec3::ZERO, params(BrushKind::Draw, 2.0, -0.5));
        assert!(a.positions.iter().all(|p| p.z <= 0.0));
        let mut s = SculptMesh::from_mesh(&uv_sphere(10.0, 32, 16));
        let before = s.positions.clone();
        let moved = run(&mut s, Vec3::new(10.0, 0.0, 0.0), params(BrushKind::Inflate, 4.0, 1.0));
        for v in moved {
            assert!(s.positions[v as usize].length() > before[v as usize].length());
        }
    }

    #[test]
    fn smooth_and_flatten_reduce_bumps() {
        let mut m = SculptMesh::from_mesh(&grid(10));
        for _ in 0..3 {
            run(&mut m, Vec3::ZERO, params(BrushKind::Draw, 3.0, 1.0));
        }
        let peak = |m: &SculptMesh| m.positions.iter().map(|p| p.z).fold(0.0, f32::max);
        let p0 = peak(&m);
        let mut sm = m.clone();
        run(&mut sm, Vec3::ZERO, params(BrushKind::Smooth, 3.0, 1.0));
        assert!(peak(&sm) < p0);
        let mut fl = m.clone();
        run(&mut fl, Vec3::new(0.0, 0.0, p0 * 0.5), params(BrushKind::Flatten, 2.5, 1.0));
        assert!(peak(&fl) < p0);
    }

    #[test]
    fn symmetry_mirrors_the_dab() {
        let mut m = SculptMesh::from_mesh(&uv_sphere(10.0, 32, 16));
        let mut p = params(BrushKind::Draw, 3.0, 1.0);
        p.symmetry = [true, false, false];
        run(&mut m, Vec3::new(10.0, 0.0, 0.0), p);
        // The sphere's vertices are symmetric in X, so the result must be too.
        for (i, q) in m.positions.iter().enumerate() {
            let twin = m.positions.iter().map(|o| (Vec3::new(-o.x, o.y, o.z) - *q).length()).fold(f32::MAX, f32::min);
            assert!(twin < 1e-4, "vertex {i} has no mirror twin ({twin})");
        }
    }

    #[cfg(feature = "threads")]
    #[test]
    fn threads_do_not_change_the_result() {
        let base = SculptMesh::from_mesh(&uv_sphere(10.0, 256, 128));
        let p = params(BrushKind::Smooth, 14.0, 0.8);
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let four = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let mut a = base.clone();
        let mut b = base.clone();
        let ma = one.install(|| run(&mut a, Vec3::new(10.0, 0.0, 0.0), p));
        let mb = four.install(|| run(&mut b, Vec3::new(10.0, 0.0, 0.0), p));
        assert!(ma.len() >= PARALLEL_MIN, "footprint too small to exercise threads: {}", ma.len());
        assert_eq!(ma, mb);
        assert_eq!(a.positions, b.positions);
    }

    #[test]
    fn clay_builds_flat_topped_without_ballooning_peaks() {
        let mut m = SculptMesh::from_mesh(&grid(10));
        // Create a crevice at origin and raise surrounding slightly.
        let ci = m.positions.iter().position(|p| p.x == 0.0 && p.y == 0.0).unwrap();
        m.positions[ci].z = -0.5;
        m.update_normals(&[ci as u32]);
        run(&mut m, Vec3::ZERO, params(BrushKind::Clay, 2.5, 0.8));
        // Crevice at origin should fill upward toward the flat clay plane.
        assert!(m.positions[ci].z > -0.5, "clay should fill crevice: {}", m.positions[ci].z);
        // Repeat clay many times: peak remains flat-topped relative to neighbors (does not spike wildly).
        for _ in 0..5 {
            run(&mut m, Vec3::ZERO, params(BrushKind::Clay, 2.5, 0.8));
        }
        assert!(m.positions[ci].z > 0.1);
    }

    #[test]
    fn crease_digs_and_pinches_inward() {
        let mut m = SculptMesh::from_mesh(&grid(10));
        run(&mut m, Vec3::ZERO, params(BrushKind::Crease, 3.0, 0.8));
        let centre = m.positions.iter().find(|p| p.x.abs() < 1e-4 && p.y.abs() < 1e-4).unwrap();
        assert!(centre.z < -0.05, "crease should dig inward: {}", centre.z);
        // Side vertex at (1, 0, 0) should move closer to origin in X (pinched inward).
        let side_idx = grid(10).positions.iter().position(|p| p.x == 1.0 && p.y == 0.0).unwrap();
        assert!(m.positions[side_idx].x < 1.0, "crease should pinch inward: {}", m.positions[side_idx].x);
    }

    #[test]
    fn pinch_pulls_vertices_toward_center_along_tangent() {
        let mut m = SculptMesh::from_mesh(&grid(10));
        let side_idx = grid(10).positions.iter().position(|p| p.x == 1.0 && p.y == 0.0).unwrap();
        run(&mut m, Vec3::ZERO, params(BrushKind::Pinch, 3.0, 0.8));
        assert!(m.positions[side_idx].x < 0.9, "pinch should pull toward center: {}", m.positions[side_idx].x);
        assert!(m.positions[side_idx].z.abs() < 1e-5, "pinch on flat grid stays on tangent plane");
    }

    #[test]
    fn mask_protects_vertices_and_ops_work() {
        let mut m = SculptMesh::from_mesh(&grid(10));
        let ci = m.positions.iter().position(|p| p.x == 0.0 && p.y == 0.0).unwrap();
        // Paint full mask around origin.
        for _ in 0..3 {
            run(&mut m, Vec3::ZERO, params(BrushKind::Mask, 2.5, 1.0));
        }
        assert!((m.masks[ci] - 1.0).abs() < 1e-5, "centre should be fully masked: {}", m.masks[ci]);
        assert!(m.has_mask());
        // Draw at origin: fully masked centre vertex must not move!
        run(&mut m, Vec3::ZERO, params(BrushKind::Draw, 2.5, 1.0));
        assert!(m.positions[ci].z.abs() < 1e-6, "masked vertex should not move: {}", m.positions[ci].z);
        // Invert mask: centre becomes 0, far vertices become 1.
        m.mask_invert();
        assert!(m.masks[ci] < 1e-5);
        // Blur mask: centre gets some bleed from masked neighbors.
        let far_idx = m.positions.iter().position(|p| p.x == 5.0 && p.y == 5.0).unwrap();
        assert!((m.masks[far_idx] - 1.0).abs() < 1e-5);
        m.mask_blur();
        assert!(m.masks[ci] >= 0.0);
        // Now centre is mostly unmasked, so Draw moves it.
        run(&mut m, Vec3::ZERO, params(BrushKind::Draw, 2.5, 1.0));
        assert!(m.positions[ci].z > 0.05);
        // Clear mask removes all weights.
        m.mask_clear();
        assert!(!m.has_mask());
    }

    #[test]
    fn falloff_curves_behave_distinctly() {
        let mut smooth = SculptMesh::from_mesh(&grid(10));
        let mut sharp = SculptMesh::from_mesh(&grid(10));
        let mut constant = SculptMesh::from_mesh(&grid(10));
        let side_idx = grid(10).positions.iter().position(|p| p.x == 1.0 && p.y == 0.0).unwrap();
        let mut p = params(BrushKind::Draw, 2.0, 0.5);
        p.falloff = Falloff::Smooth;
        run(&mut smooth, Vec3::ZERO, p);
        p.falloff = Falloff::Sharp;
        run(&mut sharp, Vec3::ZERO, p);
        p.falloff = Falloff::Constant;
        run(&mut constant, Vec3::ZERO, p);
        // At d = r/2: Sharp (0.5^3 = 0.125) < Smooth (0.75^2 = 0.5625) < Constant (1.0)
        assert!(sharp.positions[side_idx].z < smooth.positions[side_idx].z);
        assert!(smooth.positions[side_idx].z < constant.positions[side_idx].z);
    }

    #[test]
    fn three_axis_and_radial_symmetry() {
        let mut m = SculptMesh::from_mesh(&uv_sphere(10.0, 32, 16));
        let mut p = params(BrushKind::Draw, 2.5, 1.0);
        p.radial_symmetry = 4;
        run(&mut m, Vec3::new(10.0, 0.0, 0.0), p);
        // All 4 cardinal points (+X, -X, +Y, -Y) should be displaced equally outward!
        let max_at = |dir: Vec3| {
            m.positions
                .iter()
                .filter(|q| q.normalize().dot(dir) > 0.98)
                .map(|q| q.length())
                .fold(0.0f32, f32::max)
        };
        let r_px = max_at(Vec3::X);
        let r_nx = max_at(Vec3::NEG_X);
        let r_py = max_at(Vec3::Y);
        let r_ny = max_at(Vec3::NEG_Y);
        assert!(r_px > 10.1, "radial +X should displace: {r_px}");
        assert!((r_px - r_nx).abs() < 1e-3, "{r_px} vs {r_nx}");
        assert!((r_px - r_py).abs() < 1e-3, "{r_px} vs {r_py}");
        assert!((r_px - r_ny).abs() < 1e-3, "{r_px} vs {r_ny}");
    }

    #[test]
    fn zero_radius_and_degenerate_params_are_safe() {
        let mut m = SculptMesh::from_mesh(&grid(8));
        let before = m.positions.clone();
        let moved = run(&mut m, Vec3::ZERO, params(BrushKind::Draw, 0.0, 1.0));
        assert!(moved.is_empty());
        assert_eq!(m.positions, before);
        let moved_neg = run(&mut m, Vec3::ZERO, params(BrushKind::Clay, -2.0, 1.0));
        assert!(moved_neg.is_empty());
        assert_eq!(m.positions, before);
    }
}
