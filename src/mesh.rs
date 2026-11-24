//! `SculptMesh`: the working mesh for sculpting and rendering.
//!
//! Keeps Sculptor's n-gon faces as "polys" (so displacement-only edits write back the exact same faces),
//! their fan triangles for the GPU and the BVH, and adjacency in compressed sparse row form.

use crate::bvh::Stamp;
use crate::format::Mesh;
use glam::Vec3;

/// Compressed sparse rows: `items[start[i]..start[i + 1]]` belongs to row `i`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Csr {
    pub start: Vec<u32>,
    pub items: Vec<u32>,
}

impl Csr {
    pub fn row(&self, i: usize) -> &[u32] {
        &self.items[self.start[i] as usize..self.start[i + 1] as usize]
    }
    fn from_rows(n: usize, mut each: impl FnMut(&mut dyn FnMut(usize, u32))) -> Csr {
        let mut count = vec![0u32; n + 1];
        each(&mut |r, _| count[r + 1] += 1);
        for i in 0..n {
            count[i + 1] += count[i];
        }
        let mut fill = count.clone();
        let mut items = vec![0u32; count[n] as usize];
        each(&mut |r, v| {
            items[fill[r] as usize] = v;
            fill[r] += 1;
        });
        Csr { start: count, items }
    }
}

#[derive(Debug, Clone)]
pub struct SculptMesh {
    pub positions: Vec<Vec3>,
    /// Area weighted, normalised (Sculptor's `Mesh.vertexNormals()`).
    pub normals: Vec<Vec3>,
    /// Per-vertex sculpt mask weights in `[0.0, 1.0]` (0 = unmasked, 1 = fully protected).
    pub masks: Vec<f32>,
    /// Faces (n-gons) as rows of vertex indices.
    pub polys: Csr,
    /// Fan triangles of every poly, in poly order.
    pub tris: Vec<[u32; 3]>,
    pub smooth_shading: bool,
    /// Unnormalised Newell vector per poly (direction = normal, length = 2 x area).
    poly_vec: Vec<Vec3>,
    vertex_polys: Csr,
    /// Edge neighbours through poly edges (not fan diagonals), sorted.
    pub neighbours: Csr,
}

impl SculptMesh {
    pub fn from_mesh(m: &Mesh) -> SculptMesh {
        let n = m.positions.len();
        let fc = m.faces.iter().filter(|f| f.len() >= 3).count();
        let polys = Csr::from_rows(fc, |push| {
            for (fi, f) in m.faces.iter().filter(|f| f.len() >= 3).enumerate() {
                for &v in f.iter() {
                    push(fi, v);
                }
            }
        });
        let mut out = SculptMesh {
            positions: m.positions.clone(),
            normals: vec![Vec3::ZERO; n],
            masks: vec![0.0; n],
            polys,
            tris: Vec::new(),
            smooth_shading: m.smooth_shading,
            poly_vec: Vec::new(),
            vertex_polys: Csr::default(),
            neighbours: Csr::default(),
        };
        out.rebuild_topology();
        out
    }

    pub fn mask_clear(&mut self) {
        self.masks.fill(0.0);
    }

    pub fn mask_invert(&mut self) {
        for m in &mut self.masks {
            *m = (1.0 - *m).clamp(0.0, 1.0);
        }
    }

    pub fn mask_blur(&mut self) {
        let n = self.masks.len();
        if n == 0 || self.neighbours.start.len() <= n {
            return;
        }
        let prev = self.masks.clone();
        for v in 0..n {
            let row = self.neighbours.row(v);
            if row.is_empty() {
                continue;
            }
            let mut sum = 0.0f32;
            for &j in row {
                sum += prev[j as usize];
            }
            let avg = sum / row.len() as f32;
            self.masks[v] = (prev[v] * 0.5 + avg * 0.5).clamp(0.0, 1.0);
        }
    }

    pub fn has_mask(&self) -> bool {
        self.masks.iter().any(|&m| m > 1e-4)
    }

    pub fn poly_count(&self) -> usize {
        self.polys.start.len() - 1
    }

    /// Recomputes triangles, adjacency and normals from `positions` and `polys`.
    pub fn rebuild_topology(&mut self) {
        let n = self.positions.len();
        let pc = self.poly_count();
        self.tris.clear();
        self.tris.reserve(self.polys.items.len().saturating_sub(2 * pc));
        for p in 0..pc {
            let r = self.polys.row(p);
            for k in 1..r.len() - 1 {
                self.tris.push([r[0], r[k], r[k + 1]]);
            }
        }
        let polys = &self.polys;
        self.vertex_polys = Csr::from_rows(n, |push| {
            for p in 0..pc {
                for &v in polys.row(p) {
                    push(v as usize, p as u32);
                }
            }
        });
        let mut nb = Csr::from_rows(n, |push| {
            for p in 0..pc {
                let r = polys.row(p);
                let mut prev = r[r.len() - 1];
                for &cur in r {
                    push(prev as usize, cur);
                    push(cur as usize, prev);
                    prev = cur;
                }
            }
        });
        // Sort and dedupe each row in place (shared edges appear twice).
        let mut start = Vec::with_capacity(n + 1);
        start.push(0u32);
        let mut write = 0usize;
        for v in 0..n {
            let s = nb.start[v] as usize;
            let e = nb.start[v + 1] as usize;
            let len = e - s;
            if len > 1 {
                let row = &mut nb.items[s..e];
                if len <= 16 {
                    // Insertion sort avoids pdqsort overhead for typical degree 4..8 vertices.
                    for i in 1..len {
                        let key = row[i];
                        let mut j = i;
                        while j > 0 && row[j - 1] > key {
                            row[j] = row[j - 1];
                            j -= 1;
                        }
                        row[j] = key;
                    }
                } else {
                    row.sort_unstable();
                }
            }
            let mut last = u32::MAX;
            for k in s..e {
                let x = nb.items[k];
                if x != last {
                    nb.items[write] = x;
                    write += 1;
                    last = x;
                }
            }
            start.push(write as u32);
        }
        nb.items.truncate(write);
        self.neighbours = Csr { start, items: nb.items };
        #[cfg(feature = "threads")]
        {
            use rayon::prelude::*;
            if pc >= 4096 {
                let positions = &self.positions;
                let polys = &self.polys;
                self.poly_vec = (0..pc)
                    .into_par_iter()
                    .with_min_len(1024)
                    .map(|p| newell_poly(positions, polys.row(p)))
                    .collect();
            } else {
                self.poly_vec = (0..pc).map(|p| self.newell(p)).collect();
            }
            if n >= 4096 {
                let vertex_polys = &self.vertex_polys;
                let poly_vec = &self.poly_vec;
                self.normals.resize(n, Vec3::ZERO);
                self.normals.par_iter_mut().enumerate().with_min_len(1024).for_each(|(v, dst)| {
                    *dst = vertex_normal_raw(vertex_polys.row(v), poly_vec);
                });
            } else {
                for v in 0..n {
                    self.normals[v] = self.vertex_normal(v);
                }
            }
        }
        #[cfg(not(feature = "threads"))]
        {
            self.poly_vec = (0..pc).map(|p| self.newell(p)).collect();
            for v in 0..n {
                self.normals[v] = self.vertex_normal(v);
            }
        }
    }

    /// Newell's method, written exactly as Sculptor's `Mesh.newellVector`.
    #[inline]
    fn newell(&self, p: usize) -> Vec3 {
        newell_poly(&self.positions, self.polys.row(p))
    }

    #[inline]
    fn vertex_normal(&self, v: usize) -> Vec3 {
        vertex_normal_raw(self.vertex_polys.row(v), &self.poly_vec)
    }

    /// Zero-allocation, zero-sort normal refresh using reusable `Stamp`s.
    /// Appends every vertex whose normal was recomputed to `out_verts` (unsorted, unique per call).
    pub fn update_normals_stamped(
        &mut self,
        moved: &[u32],
        poly_stamp: &mut Stamp,
        vert_stamp: &mut Stamp,
        out_verts: &mut Vec<u32>,
    ) {
        let pc = self.poly_count();
        let n = self.positions.len();
        if poly_stamp.marks.len() < pc {
            poly_stamp.resize(pc);
        }
        if vert_stamp.marks.len() < n {
            vert_stamp.resize(n);
        }
        let ptag = poly_stamp.next();
        let vtag = vert_stamp.next();
        let start_len = out_verts.len();
        for &v in moved {
            for &p in self.vertex_polys.row(v as usize) {
                let pm = &mut poly_stamp.marks[p as usize];
                if *pm != ptag {
                    *pm = ptag;
                    self.poly_vec[p as usize] = newell_poly(&self.positions, self.polys.row(p as usize));
                    for &u in self.polys.row(p as usize) {
                        let vm = &mut vert_stamp.marks[u as usize];
                        if *vm != vtag {
                            *vm = vtag;
                            out_verts.push(u);
                        }
                    }
                }
            }
        }
        let touched = &out_verts[start_len..];
        for &v in touched {
            self.normals[v as usize] = vertex_normal_raw(self.vertex_polys.row(v as usize), &self.poly_vec);
        }
    }

    /// Refreshes normals around moved vertices (their polys, then every vertex of those polys).
    /// Returns the vertices whose normals were recomputed (sorted, a superset of `moved`).
    pub fn update_normals(&mut self, moved: &[u32]) -> Vec<u32> {
        let mut polys = Vec::with_capacity(moved.len() * 4);
        for &v in moved {
            polys.extend_from_slice(self.vertex_polys.row(v as usize));
        }
        polys.sort_unstable();
        polys.dedup();
        let mut verts = Vec::with_capacity(polys.len() * 4);
        for &p in &polys {
            self.poly_vec[p as usize] = self.newell(p as usize);
            verts.extend_from_slice(self.polys.row(p as usize));
        }
        verts.sort_unstable();
        verts.dedup();
        #[cfg(feature = "threads")]
        if verts.len() >= 4096 {
            use rayon::prelude::*;
            let vertex_polys = &self.vertex_polys;
            let poly_vec = &self.poly_vec;
            let new_nrms: Vec<Vec3> = verts
                .par_iter()
                .with_min_len(1024)
                .map(|&v| vertex_normal_raw(vertex_polys.row(v as usize), poly_vec))
                .collect();
            for (&v, n) in verts.iter().zip(new_nrms) {
                self.normals[v as usize] = n;
            }
            return verts;
        }
        for &v in &verts {
            self.normals[v as usize] = self.vertex_normal(v as usize);
        }
        verts
    }

    pub fn to_mesh(&self) -> Mesh {
        Mesh {
            positions: self.positions.clone(),
            faces: (0..self.poly_count()).map(|p| self.polys.row(p).to_vec()).collect(),
            smooth_shading: self.smooth_shading,
        }
    }

    /// Flat shaded copy: every triangle gets its own three vertices and the face normal.
    /// Returns (positions, normals); used by the renderer when `smooth_shading` is false.
    pub fn flat_buffers(&self) -> (Vec<Vec3>, Vec<Vec3>) {
        let (p, n, _) = self.flat_buffers_masked();
        (p, n)
    }

    /// Flat shaded copy with per-corner mask weights.
    pub fn flat_buffers_masked(&self) -> (Vec<Vec3>, Vec<Vec3>, Vec<f32>) {
        let mut pos = Vec::with_capacity(self.tris.len() * 3);
        let mut nrm = Vec::with_capacity(self.tris.len() * 3);
        let mut msk = Vec::with_capacity(self.tris.len() * 3);
        let mut t = 0;
        for p in 0..self.poly_count() {
            let n = self.poly_vec[p].normalize_or_zero();
            for _ in 0..self.polys.row(p).len() - 2 {
                for &v in &self.tris[t] {
                    pos.push(self.positions[v as usize]);
                    nrm.push(n);
                    msk.push(self.masks.get(v as usize).copied().unwrap_or(0.0));
                }
                t += 1;
            }
        }
        (pos, nrm, msk)
    }
}

#[inline]
fn newell_poly(positions: &[Vec3], r: &[u32]) -> Vec3 {
    let mut n = Vec3::ZERO;
    let mut a = positions[r[r.len() - 1] as usize];
    for &idx in r {
        let b = positions[idx as usize];
        n.x += (a.y - b.y) * (a.z + b.z);
        n.y += (a.z - b.z) * (a.x + b.x);
        n.z += (a.x - b.x) * (a.y + b.y);
        a = b;
    }
    n
}

#[inline]
fn vertex_normal_raw(polys: &[u32], poly_vec: &[Vec3]) -> Vec3 {
    let mut s = Vec3::ZERO;
    for &p in polys {
        s += poly_vec[p as usize];
    }
    s.normalize_or_zero()
}

/// UV sphere of `segments` x `rings`, radius `r`, quads with triangle caps (Sculptor's layout: poles first and last).
pub fn uv_sphere(r: f32, segments: u32, rings: u32) -> Mesh {
    let mut positions = vec![Vec3::new(0.0, 0.0, r)];
    for i in 1..rings {
        let phi = std::f32::consts::PI * i as f32 / rings as f32;
        for j in 0..segments {
            let th = std::f32::consts::TAU * j as f32 / segments as f32;
            positions.push(Vec3::new(r * phi.sin() * th.cos(), r * phi.sin() * th.sin(), r * phi.cos()));
        }
    }
    positions.push(Vec3::new(0.0, 0.0, -r));
    let bottom = positions.len() as u32 - 1;
    let ring = |i: u32, j: u32| 1 + (i - 1) * segments + j % segments;
    let mut faces = Vec::new();
    for j in 0..segments {
        faces.push(vec![0, ring(1, j), ring(1, j + 1)]);
    }
    for i in 1..rings - 1 {
        for j in 0..segments {
            faces.push(vec![ring(i, j), ring(i + 1, j), ring(i + 1, j + 1), ring(i, j + 1)]);
        }
    }
    for j in 0..segments {
        faces.push(vec![ring(rings - 1, j + 1), ring(rings - 1, j), bottom]);
    }
    Mesh { positions, faces, smooth_shading: true }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube() -> Mesh {
        let p = |x: f32, y: f32, z: f32| Vec3::new(x, y, z) * 10.0;
        Mesh {
            positions: vec![p(-1., -1., -1.), p(1., -1., -1.), p(1., 1., -1.), p(-1., 1., -1.), p(-1., -1., 1.), p(1., -1., 1.), p(1., 1., 1.), p(-1., 1., 1.)],
            faces: vec![vec![0, 3, 2, 1], vec![4, 5, 6, 7], vec![0, 1, 5, 4], vec![1, 2, 6, 5], vec![2, 3, 7, 6], vec![3, 0, 4, 7]],
            smooth_shading: false,
        }
    }

    #[test]
    fn cube_adjacency_and_normals() {
        let m = SculptMesh::from_mesh(&cube());
        assert_eq!(m.tris.len(), 12);
        assert_eq!(m.neighbours.row(0), &[1, 3, 4]);
        let n = m.normals[6];
        assert!((n - Vec3::ONE.normalize()).length() < 1e-6, "{n}");
        assert_eq!(m.to_mesh(), cube());
    }

    #[test]
    fn local_normal_update_matches_full_rebuild() {
        let mut m = SculptMesh::from_mesh(&uv_sphere(10.0, 16, 8));
        for v in [3u32, 4, 20] {
            m.positions[v as usize] *= 1.3;
        }
        m.update_normals(&[3, 4, 20]);
        let mut full = m.clone();
        full.rebuild_topology();
        assert_eq!(m.normals, full.normals);
    }

    #[test]
    fn sphere_is_closed() {
        let m = SculptMesh::from_mesh(&uv_sphere(10.0, 32, 16));
        // Every vertex of a closed manifold has at least 3 neighbours; Euler characteristic 2.
        let edges = m.neighbours.items.len() / 2;
        assert_eq!(m.positions.len() as i64 - edges as i64 + m.poly_count() as i64, 2);
        assert!(m.normals.iter().zip(&m.positions).all(|(n, p)| n.dot(p.normalize()) > 0.95));
    }
}
