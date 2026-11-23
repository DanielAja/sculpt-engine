//! Brush dabs. The maths is the web port's Sculpt-lite (`sculptor.web/swift/Sources/SculptorWeb/Sculpt.swift`)
//! so a stroke gives the same mesh on every platform:
//! falloff `f = (1 - (d/r)^2)^2`; draw moves along the footprint's average normal by `s*f*r*0.1`,
//! inflate along each vertex normal by the same amount, smooth toward the edge neighbour average by
//! `|s|*f`, flatten toward the footprint's average plane by `|s|*f`.
//!
//! Each centre reads the positions from before it and writes new ones afterwards (Jacobi style), so the
//! result does not depend on how many threads ran it.

use crate::bvh::{Bvh, Stamp};
use crate::mesh::SculptMesh;
use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrushKind {
    Draw,
    Inflate,
    Smooth,
    Flatten,
}

impl BrushKind {
    /// The web port's names; anything unknown is `draw`, as there.
    pub fn parse(s: &str) -> BrushKind {
        match s {
            "inflate" => BrushKind::Inflate,
            "smooth" => BrushKind::Smooth,
            "flatten" => BrushKind::Flatten,
            _ => BrushKind::Draw,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrushParams {
    pub kind: BrushKind,
    /// Object local units (the stroke converts from world mm).
    pub radius: f32,
    /// -1 to 1; negative digs in for draw and inflate.
    pub strength: f32,
    /// Mirror across the object's local X, Y, Z planes.
    pub symmetry: [bool; 3],
}

/// Reusable buffers for dabs on one mesh.
#[derive(Debug, Default)]
pub struct Scratch {
    stamp: Stamp,
    hits: Vec<u32>,
    weights: Vec<f32>,
    new_pos: Vec<Vec3>,
}

impl Scratch {
    /// Pre-sized for a mesh of `n` vertices, so the first dab doesn't allocate.
    pub fn new(n: usize) -> Scratch {
        Scratch { stamp: Stamp::new(n), ..Default::default() }
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
    let mut centers = [Vec3::ZERO; 8];
    centers[0] = center;
    let mut n_centers = 1usize;
    let min_dist2 = (r * 0.01) * (r * 0.01);
    for axis in 0..3 {
        if p.symmetry[axis] {
            for i in 0..n_centers {
                let c = centers[i];
                let mut mirrored = c;
                mirrored[axis] = -mirrored[axis];
                if (mirrored - c).length_squared() > min_dist2 {
                    centers[n_centers] = mirrored;
                    n_centers += 1;
                }
            }
        }
    }
    s.stamp.resize(m.positions.len());
    let st = p.strength.clamp(-1.0, 1.0);
    let inv_r2 = 1.0 / (r * r);
    let mut max_move_sq: f32 = 0.0;
    for &c in &centers[..n_centers] {
        // shortcut: vertices used by no face are never brushed (the web port moves them too); harmless for real meshes.
        bvh.sphere_vertices(&m.positions, &m.tris, c, r, &mut s.stamp, &mut s.hits);
        if s.hits.is_empty() {
            continue;
        }
        s.weights.clear();
        s.weights.reserve(s.hits.len());
        let (mut avg_n, mut avg_p, mut wsum) = (Vec3::ZERO, Vec3::ZERO, 0f32);
        for &v in &s.hits {
            let q = m.positions[v as usize];
            let t = 1.0 - (q - c).length_squared() * inv_r2;
            let f = t * t;
            s.weights.push(f);
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
        let (pos, nrm, nb, hits, weights) = (&m.positions, &m.normals, &m.neighbours, &s.hits, &s.weights);
        let kind = p.kind;
        s.new_pos.clear();
        par_map(
            hits.len(),
            |i| {
                let v = hits[i] as usize;
                let f = weights[i];
                let src = pos[v];
                match kind {
                    BrushKind::Inflate => src + nrm[v] * (st * f * r * 0.1),
                    BrushKind::Smooth => {
                        let row = nb.row(v);
                        if row.is_empty() {
                            return src;
                        }
                        let mut avg = Vec3::ZERO;
                        for &j in row {
                            avg += pos[j as usize];
                        }
                        avg /= row.len() as f32;
                        src + (avg - src) * (st.abs() * f)
                    }
                    BrushKind::Flatten => src - n * ((src - plane).dot(n) * st.abs() * f),
                    BrushKind::Draw => src + n * (st * f * r * 0.1),
                }
            },
            &mut s.new_pos,
        );
        moved.reserve(s.hits.len());
        for (i, &v) in s.hits.iter().enumerate() {
            let old = m.positions[v as usize];
            let np = s.new_pos[i];
            max_move_sq = max_move_sq.max((np - old).length_squared());
            m.positions[v as usize] = np;
            moved.push((v, old));
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
        BrushParams { kind, radius, strength, symmetry: [false; 3] }
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
}
