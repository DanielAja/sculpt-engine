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
    /// Rows of the faces with at least three corners, in order.
    fn from_faces(faces: &[Vec<u32>]) -> Csr {
        const CH: usize = 32768;
        let counts = collect_chunks(faces.len(), CH, |_, r| {
            let (mut polys, mut items) = (0usize, 0usize);
            for f in &faces[r] {
                if f.len() >= 3 {
                    polys += 1;
                    items += f.len();
                }
            }
            (polys, items)
        });
        let mut offsets = Vec::with_capacity(counts.len());
        let (mut p, mut it) = (0usize, 0usize);
        for &(np, ni) in &counts {
            offsets.push((p, it));
            p += np;
            it += ni;
        }
        let mut start = Vec::with_capacity(p + 1);
        let mut items = Vec::with_capacity(it);
        // SAFETY: `u32` is Copy; `start[0..=p]` and `items[0..it]` are completely written below before any read.
        unsafe {
            start.set_len(p + 1);
            items.set_len(it);
        }
        start[p] = it as u32;
        let (sp, ip) = (SyncPtr(start.as_mut_ptr()), SyncPtr(items.as_mut_ptr()));
        run_chunks(faces.len(), CH, |c, r| {
            let (mut p, mut it) = offsets[c];
            for f in &faces[r] {
                let len = f.len();
                if len >= 3 {
                    // SAFETY: chunk `c` owns poly slots `offsets[c].0..` and item slots `offsets[c].1..`, which the
                    // counting pass sized exactly; no other chunk writes there.
                    unsafe {
                        sp.add(p).write(it as u32);
                        match f.as_slice() {
                            [a, b, c, d] => (ip.add(it) as *mut [u32; 4]).write([*a, *b, *c, *d]),
                            [a, b, c] => (ip.add(it) as *mut [u32; 3]).write([*a, *b, *c]),
                            s => std::ptr::copy_nonoverlapping(s.as_ptr(), ip.add(it), len),
                        }
                    }
                    p += 1;
                    it += len;
                }
            }
        });
        Csr { start, items }
    }
}

/// Raw pointer that may cross threads; every use writes disjoint elements.
struct SyncPtr<T>(*mut T);
impl<T> Clone for SyncPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    #[inline(always)]
    fn add(self, i: usize) -> *mut T {
        // SAFETY: callers stay inside the allocation the pointer came from.
        unsafe { self.0.add(i) }
    }
}
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

/// Runs `f(chunk_index, item_range)` over `0..n` in `chunk` sized pieces (in parallel for big inputs).
fn run_chunks(n: usize, chunk: usize, f: impl Fn(usize, std::ops::Range<usize>) + Sync + Send) {
    let nc = n.div_ceil(chunk);
    #[cfg(feature = "threads")]
    if n >= PAR_MIN {
        use rayon::prelude::*;
        (0..nc).into_par_iter().for_each(|c| f(c, c * chunk..((c + 1) * chunk).min(n)));
        return;
    }
    for c in 0..nc {
        f(c, c * chunk..((c + 1) * chunk).min(n));
    }
}

/// Like [`run_chunks`], returning one result per chunk in order.
fn collect_chunks<T: Send>(n: usize, chunk: usize, f: impl Fn(usize, std::ops::Range<usize>) -> T + Sync + Send) -> Vec<T> {
    let nc = n.div_ceil(chunk);
    #[cfg(feature = "threads")]
    if n >= PAR_MIN {
        use rayon::prelude::*;
        return (0..nc).into_par_iter().map(|c| f(c, c * chunk..((c + 1) * chunk).min(n))).collect();
    }
    (0..nc).map(|c| f(c, c * chunk..((c + 1) * chunk).min(n))).collect()
}

/// Items from which work is spread over the thread pool.
#[cfg(feature = "threads")]
const PAR_MIN: usize = 65_536;

#[inline]
fn join2<A: Send, B: Send>(a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B + Send) -> (A, B) {
    #[cfg(feature = "threads")]
    {
        rayon::join(a, b)
    }
    #[cfg(not(feature = "threads"))]
    {
        (a(), b())
    }
}

/// Polys touching each vertex, ascending poly order within a row.
fn build_vertex_polys(n: usize, polys: &Csr) -> Csr {
    let pc = polys.start.len() - 1;
    let total = polys.items.len();
    #[cfg(feature = "threads")]
    if pc >= PAR_MIN && n > 0 && n <= 4_000_000 {
        use rayon::prelude::*;
        // Partition polys across chunks: each chunk builds its own vertex histogram and scatters
        // its poly IDs in ascending order, avoiding atomic contention and per-vertex row sorting.
        let t = rayon::current_num_threads().clamp(2, 4);
        let chunk = pc.div_ceil(t);
        let chunks: Vec<std::ops::Range<usize>> = (0..t)
            .map(|c| c * chunk..((c + 1) * chunk).min(pc))
            .filter(|r| !r.is_empty())
            .collect();
        let mut counts: Vec<Vec<u32>> = chunks
            .par_iter()
            .map(|r| {
                let mut cnt = vec![0u32; n];
                let s0 = polys.start[r.start] as usize;
                let s1 = polys.start[r.end] as usize;
                for &v in &polys.items[s0..s1] {
                    cnt[v as usize] += 1;
                }
                cnt
            })
            .collect();
        let mut start = Vec::with_capacity(n + 1);
        // SAFETY: `u32` is Copy; every element `0..=n` is written in the prefix-sum loop below.
        unsafe { start.set_len(n + 1); }
        let mut acc = 0u32;
        for v in 0..n {
            start[v] = acc;
            for c in &mut counts {
                let d = c[v];
                c[v] = acc;
                acc += d;
            }
        }
        start[n] = acc;
        let mut items: Vec<u32> = Vec::with_capacity(total);
        // SAFETY: `u32` is Copy; every slot in `0..total` belongs to one `(chunk, poly, vertex)`
        // and is written exactly once below.
        unsafe { items.set_len(total); }
        let dst = SyncPtr(items.as_mut_ptr());
        chunks.into_par_iter().zip(counts.into_par_iter()).for_each(|(r, mut off)| {
            for p in r {
                let pu = p as u32;
                for &v in polys.row(p) {
                    let slot = &mut off[v as usize];
                    // SAFETY: `off[v]` ranges over chunk's disjoint slice within vertex `v`'s row.
                    unsafe { dst.add(*slot as usize).write(pu) };
                    *slot += 1;
                }
            }
        });
        return Csr { start, items };
    }
    let mut start = vec![0u32; n + 1];
    for &v in &polys.items {
        start[v as usize + 1] += 1;
    }
    for i in 0..n {
        start[i + 1] += start[i];
    }
    let mut items = Vec::with_capacity(total);
    // SAFETY: `u32` is Copy; every slot in `0..total` is written in the scatter loop below.
    unsafe { items.set_len(total); }
    for p in 0..pc {
        let pu = p as u32;
        for &v in polys.row(p) {
            let slot = &mut start[v as usize];
            items[*slot as usize] = pu;
            *slot += 1;
        }
    }
    if n > 0 {
        start.copy_within(0..n, 1);
        start[0] = 0;
    }
    Csr { start, items }
}

/// Edge neighbours through poly edges (not fan diagonals), sorted and unique per vertex.
fn build_neighbours(n: usize, polys: &Csr, vertex_polys: &Csr) -> Csr {
    let parts = collect_chunks(n, 32768, |_, r| {
        let mut lens = Vec::with_capacity(r.len());
        let mut items: Vec<u32> = Vec::with_capacity(r.len() * 8 + 64);
        for v in r {
            let vu = v as u32;
            let s = items.len();
            for &p in vertex_polys.row(v) {
                let row = polys.row(p as usize);
                match row {
                    &[a, b, c, d] => {
                        if a == vu { items.push(d); items.push(b); }
                        if b == vu { items.push(a); items.push(c); }
                        if c == vu { items.push(b); items.push(d); }
                        if d == vu { items.push(c); items.push(a); }
                    }
                    &[a, b, c] => {
                        if a == vu { items.push(c); items.push(b); }
                        if b == vu { items.push(a); items.push(c); }
                        if c == vu { items.push(b); items.push(a); }
                    }
                    _ => {
                        let m = row.len();
                        for i in 0..m {
                            if row[i] == vu {
                                items.push(row[if i == 0 { m - 1 } else { i - 1 }]);
                                items.push(row[if i + 1 == m { 0 } else { i + 1 }]);
                            }
                        }
                    }
                }
            }
            let len = items.len() - s;
            if len > 1 {
                let row = &mut items[s..];
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
                let mut w = s + 1;
                for k in s + 1..items.len() {
                    if items[k] != items[w - 1] {
                        items[w] = items[k];
                        w += 1;
                    }
                }
                items.truncate(w);
            }
            lens.push((items.len() - s) as u32);
        }
        (lens, items)
    });
    let mut start = Vec::with_capacity(n + 1);
    start.push(0u32);
    let mut offsets = Vec::with_capacity(parts.len());
    let mut total = 0usize;
    let mut acc = 0u32;
    for (lens, items) in &parts {
        offsets.push(total);
        for &l in lens {
            acc += l;
            start.push(acc);
        }
        total += items.len();
    }
    let mut items = Vec::with_capacity(total);
    // SAFETY: `u32` is Copy; every chunk fills its disjoint slice of `items[0..total]` below.
    unsafe { items.set_len(total); }
    let dst = SyncPtr(items.as_mut_ptr());
    let copy = |c: usize| {
        let src = &parts[c].1;
        // SAFETY: chunk `c` fills `offsets[c]..offsets[c] + src.len()` only.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.add(offsets[c]), src.len()) };
    };
    #[cfg(feature = "threads")]
    if n >= PAR_MIN {
        use rayon::prelude::*;
        (0..parts.len()).into_par_iter().for_each(copy);
        return Csr { start, items };
    }
    (0..parts.len()).for_each(copy);
    Csr { start, items }
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
        let polys = Csr::from_faces(&m.faces);
        let mut normals = Vec::with_capacity(n);
        // SAFETY: `Vec3` is Copy; `rebuild_topology` writes all `n` elements of `normals` before reading.
        unsafe { normals.set_len(n); }
        let mut out = SculptMesh {
            positions: m.positions.clone(),
            normals,
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
        let polys = &self.polys;
        let positions = &self.positions;
        // Fan triangles + Newell vectors only depend on `polys` and `positions`; overlap them with
        // `vertex_polys` + `neighbours` on the pool before computing per-vertex normals.
        let ((tris, poly_vec), (vertex_polys, neighbours)) = join2(
            || {
                join2(
                    || {
                        let counts = collect_chunks(pc, 16384, |_, r| {
                            r.map(|p| polys.row(p).len().saturating_sub(2)).sum::<usize>()
                        });
                        let mut offsets = Vec::with_capacity(counts.len());
                        let mut total = 0usize;
                        for &c in &counts {
                            offsets.push(total);
                            total += c;
                        }
                        let mut tris: Vec<[u32; 3]> = Vec::with_capacity(total);
                        // SAFETY: `[u32; 3]` is Copy; every chunk fills its disjoint `counts[c]` slots.
                        unsafe { tris.set_len(total); }
                        let dst = SyncPtr(tris.as_mut_ptr());
                        run_chunks(pc, 16384, |c, r| {
                            let mut at = offsets[c];
                            for p in r {
                                let row = polys.row(p);
                                match row {
                                    &[a, b, c, d] => {
                                        // SAFETY: chunk `c` owns `offsets[c]..offsets[c] + counts[c]`.
                                        unsafe {
                                            dst.add(at).write([a, b, c]);
                                            dst.add(at + 1).write([a, c, d]);
                                        }
                                        at += 2;
                                    }
                                    &[a, b, c] => {
                                        unsafe { dst.add(at).write([a, b, c]) };
                                        at += 1;
                                    }
                                    _ => {
                                        for k in 1..row.len().saturating_sub(1) {
                                            unsafe { dst.add(at).write([row[0], row[k], row[k + 1]]) };
                                            at += 1;
                                        }
                                    }
                                }
                            }
                        });
                        tris
                    },
                    || {
                        #[cfg(feature = "threads")]
                        if pc >= PAR_MIN {
                            use rayon::prelude::*;
                            return (0..pc)
                                .into_par_iter()
                                .with_min_len(4096)
                                .map(|p| newell_poly(positions, polys.row(p)))
                                .collect();
                        }
                        (0..pc).map(|p| newell_poly(positions, polys.row(p))).collect()
                    },
                )
            },
            || {
                let vp = build_vertex_polys(n, polys);
                let nb = build_neighbours(n, polys, &vp);
                (vp, nb)
            },
        );
        self.tris = tris;
        self.poly_vec = poly_vec;
        self.vertex_polys = vertex_polys;
        self.neighbours = neighbours;
        if self.normals.len() != n {
            self.normals.resize(n, Vec3::ZERO);
        }
        #[cfg(feature = "threads")]
        if n >= PAR_MIN {
            use rayon::prelude::*;
            let vertex_polys = &self.vertex_polys;
            let poly_vec = &self.poly_vec;
            self.normals.par_iter_mut().enumerate().with_min_len(4096).for_each(|(v, dst)| {
                *dst = vertex_normal_raw(vertex_polys.row(v), poly_vec);
            });
            return;
        }
        for v in 0..n {
            self.normals[v] = self.vertex_normal(v);
        }
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
                    let row = self.polys.row(p as usize);
                    self.poly_vec[p as usize] = newell_poly(&self.positions, row);
                    for &u in row {
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
        #[cfg(feature = "threads")]
        if touched.len() >= PAR_MIN {
            use rayon::prelude::*;
            let vertex_polys = &self.vertex_polys;
            let poly_vec = &self.poly_vec;
            let nrms = SyncPtr(self.normals.as_mut_ptr());
            touched.par_chunks(4096).for_each(|chunk| {
                for &v in chunk {
                    // SAFETY: `vert_stamp` guarantees every vertex in `touched` is unique.
                    unsafe {
                        nrms.add(v as usize).write(vertex_normal_raw(vertex_polys.row(v as usize), poly_vec));
                    }
                }
            });
            return;
        }
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
            let row = self.polys.row(p as usize);
            self.poly_vec[p as usize] = newell_poly(&self.positions, row);
            verts.extend_from_slice(row);
        }
        verts.sort_unstable();
        verts.dedup();
        #[cfg(feature = "threads")]
        if verts.len() >= PAR_MIN {
            use rayon::prelude::*;
            let vertex_polys = &self.vertex_polys;
            let poly_vec = &self.poly_vec;
            let nrms = SyncPtr(self.normals.as_mut_ptr());
            verts.par_chunks(4096).for_each(|chunk| {
                for &v in chunk {
                    // SAFETY: `verts` was deduped, so every chunk writes disjoint elements of `normals`.
                    unsafe {
                        nrms.add(v as usize).write(vertex_normal_raw(vertex_polys.row(v as usize), poly_vec));
                    }
                }
            });
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
    match r {
        &[i0, i1, i2] => {
            let p0 = positions[i0 as usize];
            (positions[i1 as usize] - p0).cross(positions[i2 as usize] - p0)
        }
        &[i0, i1, i2, i3] => {
            let d02 = positions[i2 as usize] - positions[i0 as usize];
            let d13 = positions[i3 as usize] - positions[i1 as usize];
            d02.cross(d13)
        }
        _ => {
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
    }
}

#[inline]
fn vertex_normal_raw(polys: &[u32], poly_vec: &[Vec3]) -> Vec3 {
    let s = match polys {
        &[p0, p1, p2, p3] => {
            (poly_vec[p0 as usize] + poly_vec[p1 as usize])
                + (poly_vec[p2 as usize] + poly_vec[p3 as usize])
        }
        &[p0, p1, p2] => poly_vec[p0 as usize] + poly_vec[p1 as usize] + poly_vec[p2 as usize],
        _ => {
            let mut acc = Vec3::ZERO;
            for &p in polys {
                acc += poly_vec[p as usize];
            }
            acc
        }
    };
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

    /// Chunked / parallel topology builders against a plain sets-and-loops reference.
    #[test]
    fn topology_matches_naive_reference() {
        for (seg, rings) in [(5, 3), (24, 12), (120, 90)] {
            let mut m = uv_sphere(5.0, seg, rings);
            m.faces.insert(3, vec![0, 1]); // too short: dropped
            let sm = SculptMesh::from_mesh(&m);
            let faces: Vec<&Vec<u32>> = m.faces.iter().filter(|f| f.len() >= 3).collect();
            assert_eq!(sm.poly_count(), faces.len());
            let mut tris = Vec::new();
            let mut vp = vec![Vec::new(); m.positions.len()];
            let mut nb = vec![std::collections::BTreeSet::new(); m.positions.len()];
            for (p, f) in faces.iter().enumerate() {
                assert_eq!(sm.polys.row(p), &f[..]);
                for k in 1..f.len() - 1 {
                    tris.push([f[0], f[k], f[k + 1]]);
                }
                for (i, &v) in f.iter().enumerate() {
                    vp[v as usize].push(p as u32);
                    nb[v as usize].insert(f[(i + 1) % f.len()]);
                    nb[f[(i + 1) % f.len()] as usize].insert(v);
                }
            }
            assert_eq!(sm.tris, tris);
            for v in 0..m.positions.len() {
                assert_eq!(sm.vertex_polys.row(v), &vp[v][..], "vertex_polys {v}");
                assert_eq!(sm.neighbours.row(v), &nb[v].iter().copied().collect::<Vec<_>>()[..], "neighbours {v}");
            }
        }
    }

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
