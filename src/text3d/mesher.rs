//! Signed distance field to a closed, quad-dominant mesh with Naive Surface Nets.
//!
//! Algorithm after Gibson, "Constrained Elastic Surface Nets" (1998), and Mikola Lysenko's write-up
//! (https://0fps.net/2012/07/12/smooth-voxel-terrain-part-2/, MIT-licensed reference code). The Rust crate
//! `fast-surface-nets` (MIT/Apache-2.0) uses the same scheme; this is an independent implementation that
//! adds a Newton projection of each vertex onto the zero set so shapes stay true at coarse resolutions.
//!
//! Output faces are quads, wound counter-clockwise seen from outside, which is what Sculptor's n-gon meshes
//! and edit tools expect. Every directed edge is matched by its reverse, so the mesh is closed.

use super::sdf::{gradient, Aabb, Compiled};
use crate::format::Mesh;
use glam::Vec3;

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

pub fn surface_nets(f: &Compiled, grid: &Grid) -> Mesh {
    let [nx, ny, nz] = grid.n;
    let h = grid.h;
    let idx = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    let mut field = vec![0f32; nx * ny * nz];
    let slice_len = nx * ny;
    #[cfg(feature = "threads")]
    if field.len() >= 4096 {
        use rayon::prelude::*;
        field.par_chunks_mut(slice_len).enumerate().for_each(|(k, slice)| {
            let z = grid.origin.z + k as f32 * h;
            for j in 0..ny {
                let y = grid.origin.y + j as f32 * h;
                let row = &mut slice[j * nx..(j + 1) * nx];
                for (i, cell) in row.iter_mut().enumerate() {
                    *cell = f.eval(Vec3::new(grid.origin.x + i as f32 * h, y, z));
                }
            }
        });
    } else {
        for k in 0..nz {
            let z = grid.origin.z + k as f32 * h;
            for j in 0..ny {
                let y = grid.origin.y + j as f32 * h;
                let base = (k * ny + j) * nx;
                for i in 0..nx {
                    field[base + i] = f.eval(Vec3::new(grid.origin.x + i as f32 * h, y, z));
                }
            }
        }
    }
    #[cfg(not(feature = "threads"))]
    {
        let _ = slice_len;
        for k in 0..nz {
            let z = grid.origin.z + k as f32 * h;
            for j in 0..ny {
                let y = grid.origin.y + j as f32 * h;
                let base = (k * ny + j) * nx;
                for i in 0..nx {
                    field[base + i] = f.eval(Vec3::new(grid.origin.x + i as f32 * h, y, z));
                }
            }
        }
    }

    // One vertex per surface component in each cell the surface crosses, at the mean of that component's
    // edge crossings. Most cells have one component; saddle cells (two sheets, as in a thin wall) get two,
    // which keeps the mesh manifold. Components follow Dual Marching Cubes: on each cell face, crossing edges
    // are joined in pairs, and a face with four crossings is resolved by the asymptotic decider (Nielson and
    // Hamann 1991), which depends only on that face's values, so both cells sharing the face agree.
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);
    let cidx = |i: usize, j: usize, k: usize| (k * cy + j) * cx + i;
    let mut cell_slot = vec![u32::MAX; cx * cy * cz];
    let mut slots: Vec<[u32; 12]> = Vec::new();
    let mut positions: Vec<Vec3> = Vec::new();
    let h2 = h * h;
    for k in 0..cz {
        for j in 0..cy {
            for i in 0..cx {
                let mut v = [0f32; 8];
                let mut mask = 0u8;
                for (c, o) in CORNERS.iter().enumerate() {
                    v[c] = field[idx(i + o[0], j + o[1], k + o[2])];
                    if v[c] < 0.0 {
                        mask |= 1 << c;
                    }
                }
                if mask == 0 || mask == 255 {
                    continue;
                }
                let mut edge_crosses = [false; 12];
                let mut total_crosses = 0usize;
                for e in 0..12 {
                    let cr = (v[EDGES[e].0] < 0.0) != (v[EDGES[e].1] < 0.0);
                    edge_crosses[e] = cr;
                    total_crosses += cr as usize;
                }
                let mut comp: [usize; 12] = std::array::from_fn(|e| e);
                fn root(c: &mut [usize; 12], mut x: usize) -> usize {
                    while c[x] != x {
                        x = c[x];
                    }
                    x
                }
                let join = |c: &mut [usize; 12], a: usize, b: usize| {
                    let (ra, rb) = (root(c, a), root(c, b));
                    c[ra] = rb;
                };
                for face in FACES.iter() {
                    let mut cr = [0usize; 4];
                    let mut cr_len = 0usize;
                    for n in 0..4 {
                        if edge_crosses[face.edges[n]] {
                            cr[cr_len] = n;
                            cr_len += 1;
                        }
                    }
                    match cr_len {
                        2 => join(&mut comp, face.edges[cr[0]], face.edges[cr[1]]),
                        4 => {
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
                let cell_min = grid.origin + Vec3::new(i as f32, j as f32, k as f32) * h;
                let mut slot = [u32::MAX; 12];
                for e in 0..12 {
                    if !edge_crosses[e] || slot[e] != u32::MAX {
                        continue;
                    }
                    let r = root(&mut comp, e);
                    let mut sum = Vec3::ZERO;
                    let mut count = 0.0;
                    let mut members = [0usize; 12];
                    let mut m_len = 0usize;
                    for e2 in 0..12 {
                        if edge_crosses[e2] && root(&mut comp, e2) == r {
                            let (a, b) = EDGES[e2];
                            let t = v[a] / (v[a] - v[b]);
                            let pa = Vec3::new(CORNERS[a][0] as f32, CORNERS[a][1] as f32, CORNERS[a][2] as f32);
                            let pb = Vec3::new(CORNERS[b][0] as f32, CORNERS[b][1] as f32, CORNERS[b][2] as f32);
                            sum += pa + (pb - pa) * t;
                            count += 1.0;
                            members[m_len] = e2;
                            m_len += 1;
                        }
                    }
                    let start = cell_min + sum / count * h;
                    // Newton steps toward the zero set (only for single-component cells: in a saddle cell the
                    // nearest surface may belong to the other sheet).
                    let mut p = start;
                    if m_len == total_crosses {
                        for _ in 0..2 {
                            let d = f.eval(p);
                            let g = gradient(f, p, h * 0.25);
                            let g2 = g.length_squared();
                            if g2 < 1e-12 {
                                break;
                            }
                            p -= g * (d / g2);
                        }
                        if !p.is_finite() || (p - start).length_squared() > h2 {
                            p = start;
                        }
                    }
                    let id = positions.len() as u32;
                    positions.push(p);
                    for &e2 in &members[..m_len] {
                        slot[e2] = id;
                    }
                }
                cell_slot[cidx(i, j, k)] = slots.len() as u32;
                slots.push(slot);
            }
        }
    }

    // One quad per sample edge with a sign change, joining the four cells around it (each cell contributes
    // the vertex of the component that this edge belongs to).
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(slots.len() * 3);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let d0 = field[idx(i, j, k)];
                let p = [i, j, k];
                for a in 0..3 {
                    let (b, c) = ((a + 1) % 3, (a + 2) % 3);
                    if p[a] + 1 >= grid.n[a] || p[b] < 1 || p[c] < 1 || p[b] + 1 >= grid.n[b] || p[c] + 1 >= grid.n[c] {
                        continue;
                    }
                    let mut q = p;
                    q[a] += 1;
                    let d1 = field[idx(q[0], q[1], q[2])];
                    if (d0 < 0.0) == (d1 < 0.0) {
                        continue;
                    }
                    let cell = |db: usize, dc: usize| {
                        let mut s = p;
                        s[b] = s[b] - 1 + db;
                        s[c] = s[c] - 1 + dc;
                        let sl = cell_slot[cidx(s[0], s[1], s[2])];
                        if sl == u32::MAX {
                            return u32::MAX;
                        }
                        // The grid edge inside this cell: from corner (p - s) along axis a.
                        let mut off = [0usize; 3];
                        for t in 0..3 {
                            off[t] = p[t] - s[t];
                        }
                        let c0 = off[0] | off[1] << 1 | off[2] << 2;
                        let c1 = c0 | 1 << a;
                        let e = EDGE_LOOKUP[c0][c1] as usize;
                        slots[sl as usize][e]
                    };
                    let quad = [cell(0, 0), cell(1, 0), cell(1, 1), cell(0, 1)];
                    if quad.contains(&u32::MAX) {
                        continue;
                    }
                    if d0 < 0.0 {
                        faces.push(quad.to_vec());
                    } else {
                        faces.push(vec![quad[0], quad[3], quad[2], quad[1]]);
                    }
                }
            }
        }
    }
    Mesh { positions, faces, smooth_shading: true }
}

const CORNERS: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
const EDGES: [(usize, usize); 12] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];
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

/// Sculptor's watertight test: every directed edge appears once and so does its reverse.
pub fn is_manifold(m: &Mesh) -> bool {
    let mut e: std::collections::HashMap<(u32, u32), u8> = std::collections::HashMap::with_capacity(m.faces.len() * 4);
    for f in &m.faces {
        for k in 0..f.len() {
            let n = e.entry((f[k], f[(k + 1) % f.len()])).or_default();
            *n += 1;
            if *n > 1 {
                return false;
            }
        }
    }
    e.keys().all(|(a, b)| e.contains_key(&(*b, *a)))
}
