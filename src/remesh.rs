//! Free Voxel / SDF Remesh: BVH-accelerated mesh-to-SDF + manifold Surface Nets.
//!
//! Converts any closed (or mostly-closed) `SculptMesh` into a uniform-density, watertight,
//! outward-facing quad manifold `Mesh` at a requested voxel resolution (`24..=128` along the
//! longest bounding-box axis) — a 100% free, zero-server alternative to paid remeshers.
//!
//! Speed strategy:
//! 1. Inside/outside sign is computed across the full grid via 3-axis scanline raycasts
//!    (`+X`, `+Y`, `+Z`) with 2-of-3 majority parity voting — only `O(N^2)` rays instead of `O(N^3)`.
//! 2. Exact BVH closest-triangle distance is evaluated ONLY at the thin surface band (corners of
//!    cells adjacent to a sign change) — `O(N^2)` distance queries instead of `O(N^3)`.
//! 3. Dual-component Surface Nets with the Nielson-Hamann asymptotic decider + `make_manifold`
//!    extracts a strictly watertight quad mesh, and single-component vertices are gently projected
//!    toward their exact closest surface point so fine sculpted silhouettes stay crisp.

use crate::bvh::{ray_tri, Bvh};
use crate::format::Mesh;
use crate::mesh::SculptMesh;
use crate::text3d::mesher::{make_manifold, MAX_SAMPLES};
use glam::{Vec3, Vec3A};

pub const MIN_RESOLUTION: u32 = 24;
pub const MAX_RESOLUTION: u32 = 128;

/// Remeshes `mesh` into a fresh uniform-density watertight quad manifold mesh.
/// `resolution` is clamped to `24..=128` voxels along the longest axis.
pub fn voxel_remesh(mesh: &SculptMesh, bvh: &Bvh, resolution: u32) -> Result<Mesh, String> {
    if mesh.positions.is_empty() || mesh.tris.is_empty() || bvh.nodes.is_empty() {
        return Err("cannot remesh an empty mesh".into());
    }
    let res = resolution.clamp(MIN_RESOLUTION, MAX_RESOLUTION);

    let (mut bmin, mut bmax) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for &p in &mesh.positions {
        if p.is_finite() {
            bmin = bmin.min(p);
            bmax = bmax.max(p);
        }
    }
    let size = bmax - bmin;
    let longest = size.max_element();
    if !(longest > 1e-4) || !longest.is_finite() {
        return Err("mesh bounding box is degenerate".into());
    }

    let mut h = longest / (res as f32);
    let make_grid = |cell: f32| -> (Vec3, [usize; 3]) {
        let pad = 2.0 * cell;
        let origin = bmin - Vec3::splat(pad);
        let span = size + Vec3::splat(2.0 * pad);
        let n = [
            ((span.x / cell).ceil() as usize + 1).max(5),
            ((span.y / cell).ceil() as usize + 1).max(5),
            ((span.z / cell).ceil() as usize + 1).max(5),
        ];
        (origin, n)
    };
    let (mut origin, mut n) = make_grid(h);
    while n[0].saturating_mul(n[1]).saturating_mul(n[2]) > MAX_SAMPLES {
        h *= 1.1;
        let g = make_grid(h);
        origin = g.0;
        n = g.1;
    }

    let [nx, ny, nz] = n;
    let total = nx * ny * nz;
    let idx = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;

    // 1. 3-axis scanline parity vote (O(N^2) rays total, immune to single-axis edge grazing).
    let mut votes = vec![0u8; total];

    // +X scanlines (ny * nz rays)
    {
        let mut hits = Vec::with_capacity(16);
        let ox = origin.x - h * 2.0;
        let jy = h * 0.00013;
        let jz = h * 0.00019;
        for k in 0..nz {
            let z = origin.z + k as f32 * h + jz;
            for j in 0..ny {
                let y = origin.y + j as f32 * h + jy;
                hits.clear();
                collect_axis_ray_hits(bvh, &mesh.positions, &mesh.tris, Vec3::new(ox, y, z), 0, &mut hits);
                if hits.is_empty() {
                    continue;
                }
                hits.sort_unstable_by(|a, b| a.total_cmp(b));
                let mut h_idx = 0usize;
                let mut inside = false;
                let base = (k * ny + j) * nx;
                for i in 0..nx {
                    let t_grid = (origin.x + i as f32 * h) - ox;
                    while h_idx < hits.len() && hits[h_idx] < t_grid {
                        inside = !inside;
                        h_idx += 1;
                    }
                    if inside {
                        votes[base + i] += 1;
                    }
                }
            }
        }
    }

    // +Y scanlines (nx * nz rays)
    {
        let mut hits = Vec::with_capacity(16);
        let oy = origin.y - h * 2.0;
        let jx = h * 0.00017;
        let jz = h * 0.00023;
        for k in 0..nz {
            let z = origin.z + k as f32 * h + jz;
            for i in 0..nx {
                let x = origin.x + i as f32 * h + jx;
                hits.clear();
                collect_axis_ray_hits(bvh, &mesh.positions, &mesh.tris, Vec3::new(x, oy, z), 1, &mut hits);
                if hits.is_empty() {
                    continue;
                }
                hits.sort_unstable_by(|a, b| a.total_cmp(b));
                let mut h_idx = 0usize;
                let mut inside = false;
                let kz = k * ny * nx;
                for j in 0..ny {
                    let t_grid = (origin.y + j as f32 * h) - oy;
                    while h_idx < hits.len() && hits[h_idx] < t_grid {
                        inside = !inside;
                        h_idx += 1;
                    }
                    if inside {
                        votes[kz + j * nx + i] += 1;
                    }
                }
            }
        }
    }

    // +Z scanlines (nx * ny rays)
    {
        let mut hits = Vec::with_capacity(16);
        let oz = origin.z - h * 2.0;
        let jx = h * 0.00029;
        let jy = h * 0.00011;
        for j in 0..ny {
            let y = origin.y + j as f32 * h + jy;
            for i in 0..nx {
                let x = origin.x + i as f32 * h + jx;
                hits.clear();
                collect_axis_ray_hits(bvh, &mesh.positions, &mesh.tris, Vec3::new(x, y, oz), 2, &mut hits);
                if hits.is_empty() {
                    continue;
                }
                hits.sort_unstable_by(|a, b| a.total_cmp(b));
                let mut h_idx = 0usize;
                let mut inside = false;
                let xy = j * nx + i;
                for k in 0..nz {
                    let t_grid = (origin.z + k as f32 * h) - oz;
                    while h_idx < hits.len() && hits[h_idx] < t_grid {
                        inside = !inside;
                        h_idx += 1;
                    }
                    if inside {
                        votes[k * ny * nx + xy] += 1;
                    }
                }
            }
        }
    }

    // 2. Mark active surface-band grid points (corners of cells with a sign change).
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);
    let sy = nx;
    let sz = nx * ny;
    let corner_off: [usize; 8] = [0, 1, sy, sy + 1, sz, sz + 1, sz + sy, sz + sy + 1];

    let mut is_band = vec![false; total];
    let mut band_points: Vec<usize> = Vec::with_capacity(total / 8);
    for k in 0..cz {
        let kz = k * sz;
        for j in 0..cy {
            let row = kz + j * sy;
            for i in 0..cx {
                let b = row + i;
                let mut mask = 0u8;
                for c in 0..8 {
                    if votes[b + corner_off[c]] >= 2 {
                        mask |= 1 << c;
                    }
                }
                if mask != 0 && mask != 255 {
                    for c in 0..8 {
                        let gi = b + corner_off[c];
                        if !is_band[gi] {
                            is_band[gi] = true;
                            band_points.push(gi);
                        }
                    }
                }
            }
        }
    }

    if band_points.is_empty() {
        return Err("mesh encloses zero volume at this remesh resolution".into());
    }

    // 3. Populate signed distance field: exact BVH closest-triangle distance on the surface band,
    //    fast +/- 2h outside the band (strictly preserving sign for every grid point).
    let mut field: Vec<f32> = votes.iter().map(|&v| if v >= 2 { -2.0 * h } else { 2.0 * h }).collect();
    let eps = (1e-5 * h).max(1e-7);

    let eval_point = |gi: usize| -> f32 {
        let k = gi / sz;
        let rem = gi - k * sz;
        let j = rem / sy;
        let i = rem - j * sy;
        let p = origin + Vec3::new(i as f32, j as f32, k as f32) * h;
        let (d2, _) = closest_point_bvh(bvh, &mesh.positions, &mesh.tris, p);
        let d = d2.sqrt().max(eps);
        if votes[gi] >= 2 { -d } else { d }
    };

    #[cfg(feature = "threads")]
    let band_vals: Vec<f32> = if band_points.len() >= 2048 {
        use rayon::prelude::*;
        band_points.par_iter().map(|&gi| eval_point(gi)).collect()
    } else {
        band_points.iter().map(|&gi| eval_point(gi)).collect()
    };
    #[cfg(not(feature = "threads"))]
    let band_vals: Vec<f32> = band_points.iter().map(|&gi| eval_point(gi)).collect();

    for (&gi, val) in band_points.iter().zip(band_vals) {
        field[gi] = val;
    }

    // 4. Dual-component Surface Nets with asymptotic decider + gentle closest-surface projection.
    let cidx = |i: usize, j: usize, k: usize| (k * cy + j) * cx + i;
    let mut cell_slot = vec![u32::MAX; cx * cy * cz];
    let mut slots: Vec<[u32; 12]> = Vec::new();
    let mut positions: Vec<Vec3> = Vec::new();
    let max_snap2 = (h * 0.65) * (h * 0.65);

    for k in 0..cz {
        let kz = k * sz;
        for j in 0..cy {
            let base_row = kz + j * sy;
            for i in 0..cx {
                let b = base_row + i;
                let mut v = [0f32; 8];
                let mut mask = 0u8;
                for c in 0..8 {
                    let val = field[b + corner_off[c]];
                    v[c] = val;
                    if val < 0.0 {
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
                    for n_i in 0..4 {
                        if edge_crosses[face.edges[n_i]] {
                            cr[cr_len] = n_i;
                            cr_len += 1;
                        }
                    }
                    match cr_len {
                        2 => join(&mut comp, face.edges[cr[0]], face.edges[cr[1]]),
                        4 => {
                            let [a, b, c, d] = face.corners.map(|n_i| v[n_i]);
                            let denom = a + c - b - d;
                            let saddle = if denom.abs() > 1e-12 { (a * c - b * d) / denom } else { a };
                            let corner0_inside = a < 0.0;
                            if (saddle < 0.0) == corner0_inside {
                                join(&mut comp, face.edges[0], face.edges[1]);
                                join(&mut comp, face.edges[2], face.edges[3]);
                            } else {
                                join(&mut comp, face.edges[3], face.edges[0]);
                                join(&mut comp, face.edges[1], face.edges[2]);
                            }
                        }
                        _ => {}
                    }
                }
                let cell_min = origin + Vec3::new(i as f32, j as f32, k as f32) * h;
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
                            let denom = v[a] - v[b];
                            let t = if denom.abs() > 1e-12 { (v[a] / denom).clamp(0.0, 1.0) } else { 0.5 };
                            let pa = CORNERS_F32[a];
                            let pb = CORNERS_F32[b];
                            sum += pa + (pb - pa) * t;
                            count += 1.0;
                            members[m_len] = e2;
                            m_len += 1;
                        }
                    }
                    let start = cell_min + (sum / count) * h;
                    let mut p = start;
                    // Project single-component vertices gently onto the original surface for crisp fidelity.
                    if m_len == total_crosses {
                        let (d2, cp) = closest_point_bvh(bvh, &mesh.positions, &mesh.tris, start);
                        if d2 <= max_snap2 && cp.is_finite() {
                            p = start.lerp(cp, 0.7);
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

    // 5. Emit quads across every grid edge with a sign change.
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(slots.len() * 3);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let d0 = field[idx(i, j, k)];
                let p = [i, j, k];
                for a in 0..3 {
                    let (b, c) = ((a + 1) % 3, (a + 2) % 3);
                    if p[a] + 1 >= n[a] || p[b] < 1 || p[c] < 1 || p[b] + 1 >= n[b] || p[c] + 1 >= n[c] {
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
                        let mut off = [0usize; 3];
                        for t in 0..3 {
                            off[t] = p[t] - s[t];
                        }
                        let c0 = off[0] | (off[1] << 1) | (off[2] << 2);
                        let c1 = c0 | (1 << a);
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

    if faces.is_empty() {
        return Err("voxel remesh produced no surface faces".into());
    }

    let mut out = Mesh { positions, faces, smooth_shading: true };
    make_manifold(&mut out);
    Ok(out)
}

/// Collects all positive ray intersection distances `t > 0` along positive axis `axis` (0=+X, 1=+Y, 2=+Z).
/// Uses exact NaN-free AABB interval tests (no division by zero on perpendicular axes).
fn collect_axis_ray_hits(bvh: &Bvh, positions: &[Vec3], tris: &[[u32; 3]], o: Vec3, axis: usize, out: &mut Vec<f32>) {
    if bvh.nodes.is_empty() || tris.is_empty() {
        return;
    }
    let dir = match axis {
        0 => Vec3::X,
        1 => Vec3::Y,
        _ => Vec3::Z,
    };
    let b = (axis + 1) % 3;
    let c = (axis + 2) % 3;
    let oa = o[axis];
    let ob = o[b];
    let oc = o[c];
    let slack = bvh.slack;
    let mut stack = [0usize; 64];
    let mut sp = 0usize;
    let mut overflow = Vec::new();

    let hits_box = |min: [f32; 3], max: [f32; 3]| -> bool {
        ob >= min[b] - slack
            && ob <= max[b] + slack
            && oc >= min[c] - slack
            && oc <= max[c] + slack
            && max[axis] + slack >= oa
    };

    if hits_box(bvh.nodes[0].min, bvh.nodes[0].max) {
        stack[sp] = 0;
        sp += 1;
    }
    while sp > 0 || !overflow.is_empty() {
        let ni = if !overflow.is_empty() {
            overflow.pop().unwrap()
        } else {
            sp -= 1;
            stack[sp]
        };
        let node = &bvh.nodes[ni];
        if node.count > 0 {
            for &t in &bvh.tri_ids[node.left_first as usize..(node.left_first + node.count) as usize] {
                let tri = tris[t as usize];
                let pa = positions[tri[0] as usize];
                let pb = positions[tri[1] as usize];
                let pc = positions[tri[2] as usize];
                if let Some(hit_t) = ray_tri(o, dir, pa, pb, pc) {
                    out.push(hit_t);
                }
            }
            continue;
        }
        let l = node.left_first as usize;
        let r = l + 1;
        if hits_box(bvh.nodes[l].min, bvh.nodes[l].max) {
            if sp < 64 {
                stack[sp] = l;
                sp += 1;
            } else {
                overflow.push(l);
            }
        }
        if hits_box(bvh.nodes[r].min, bvh.nodes[r].max) {
            if sp < 64 {
                stack[sp] = r;
                sp += 1;
            } else {
                overflow.push(r);
            }
        }
    }
}

#[inline(always)]
fn box_dist_sq(p: Vec3A, min: [f32; 3], max: [f32; 3], slack: f32) -> f32 {
    let s = Vec3A::splat(slack);
    let lo = Vec3A::from_array(min) - s;
    let hi = Vec3A::from_array(max) + s;
    let clamped = p.clamp(lo, hi);
    (p - clamped).length_squared()
}

/// Exact squared distance and closest point on the mesh via nearest-first BVH branch-and-bound.
pub fn closest_point_bvh(bvh: &Bvh, positions: &[Vec3], tris: &[[u32; 3]], p: Vec3) -> (f32, Vec3) {
    if bvh.nodes.is_empty() || tris.is_empty() {
        return (f32::INFINITY, p);
    }
    let pa = Vec3A::from(p);
    let slack = bvh.slack;
    let mut best_d2 = f32::INFINITY;
    let mut best_pt = p;
    let mut stack = [(0usize, 0.0f32); 64];
    let mut sp = 0usize;
    let mut overflow: Vec<(usize, f32)> = Vec::new();
    let root_d2 = box_dist_sq(pa, bvh.nodes[0].min, bvh.nodes[0].max, slack);
    stack[sp] = (0, root_d2);
    sp += 1;

    while sp > 0 || !overflow.is_empty() {
        let (ni, nd2) = if !overflow.is_empty() {
            overflow.pop().unwrap()
        } else {
            sp -= 1;
            stack[sp]
        };
        if nd2 >= best_d2 {
            continue;
        }
        let node = &bvh.nodes[ni];
        if node.count > 0 {
            for &t in &bvh.tri_ids[node.left_first as usize..(node.left_first + node.count) as usize] {
                let tri = tris[t as usize];
                let cp = closest_point_on_triangle(
                    p,
                    positions[tri[0] as usize],
                    positions[tri[1] as usize],
                    positions[tri[2] as usize],
                );
                let d2 = (cp - p).length_squared();
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_pt = cp;
                }
            }
            continue;
        }
        let l = node.left_first as usize;
        let r = l + 1;
        let dl = box_dist_sq(pa, bvh.nodes[l].min, bvh.nodes[l].max, slack);
        let dr = box_dist_sq(pa, bvh.nodes[r].min, bvh.nodes[r].max, slack);
        let push = |stack: &mut [(usize, f32); 64], sp: &mut usize, overflow: &mut Vec<(usize, f32)>, item: (usize, f32)| {
            if *sp < 64 {
                stack[*sp] = item;
                *sp += 1;
            } else {
                overflow.push(item);
            }
        };
        // Visit nearer child first so `best_d2` shrinks rapidly and prunes the far subtree.
        if dl <= dr {
            if dr < best_d2 {
                push(&mut stack, &mut sp, &mut overflow, (r, dr));
            }
            if dl < best_d2 {
                push(&mut stack, &mut sp, &mut overflow, (l, dl));
            }
        } else {
            if dl < best_d2 {
                push(&mut stack, &mut sp, &mut overflow, (l, dl));
            }
            if dr < best_d2 {
                push(&mut stack, &mut sp, &mut overflow, (r, dr));
            }
        }
    }
    (best_d2, best_pt)
}

/// Closest point on triangle `(a, b, c)` to `p` (Christer Ericson, Real-Time Collision Detection).
#[inline]
fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

const CORNERS_F32: [Vec3; 8] = [
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(0.0, 1.0, 1.0),
    Vec3::new(1.0, 1.0, 1.0),
];
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

struct Face {
    corners: [usize; 4],
    edges: [usize; 4],
}

const FACES: [Face; 6] = [
    Face { corners: [0, 1, 3, 2], edges: [0, 5, 1, 4] },
    Face { corners: [4, 5, 7, 6], edges: [2, 7, 3, 6] },
    Face { corners: [0, 1, 5, 4], edges: [0, 9, 2, 8] },
    Face { corners: [2, 3, 7, 6], edges: [1, 11, 3, 10] },
    Face { corners: [0, 2, 6, 4], edges: [4, 10, 6, 8] },
    Face { corners: [1, 3, 7, 5], edges: [5, 11, 7, 9] },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::uv_sphere;
    use crate::text3d::mesher::is_manifold;

    fn signed_volume(m: &Mesh) -> f32 {
        let mut vol6 = 0.0f32;
        for f in &m.faces {
            let a = m.positions[f[0] as usize];
            for k in 1..f.len() - 1 {
                let b = m.positions[f[k] as usize];
                let c = m.positions[f[k + 1] as usize];
                vol6 += a.dot(b.cross(c));
            }
        }
        vol6 / 6.0
    }

    #[test]
    fn remesh_sphere_is_watertight_and_preserves_radius_and_volume() {
        let src = SculptMesh::from_mesh(&uv_sphere(10.0, 32, 16));
        let bvh = Bvh::build(&src.positions, &src.tris);
        let remeshed = voxel_remesh(&src, &bvh, 32).unwrap();
        assert!(remeshed.positions.len() > 200);
        assert!(remeshed.faces.len() > 200);
        assert!(is_manifold(&remeshed), "remeshed sphere must be strictly watertight manifold");
        let avg_r = remeshed.positions.iter().map(|p| p.length()).sum::<f32>() / remeshed.positions.len() as f32;
        assert!((avg_r - 10.0).abs() < 0.35, "average radius should match 10.0: {avg_r}");
        let vol = signed_volume(&remeshed);
        let expected_vol = (4.0 / 3.0) * std::f32::consts::PI * 1000.0;
        assert!(vol > expected_vol * 0.88 && vol < expected_vol * 1.12, "volume {vol} vs expected {expected_vol}");
    }

    #[test]
    fn remesh_cube_is_watertight_outward_facing() {
        let p = |x: f32, y: f32, z: f32| Vec3::new(x, y, z) * 10.0;
        let cube = Mesh {
            positions: vec![
                p(-1., -1., -1.), p(1., -1., -1.), p(1., 1., -1.), p(-1., 1., -1.),
                p(-1., -1., 1.), p(1., -1., 1.), p(1., 1., 1.), p(-1., 1., 1.),
            ],
            faces: vec![
                vec![0, 3, 2, 1], vec![4, 5, 6, 7], vec![0, 1, 5, 4],
                vec![1, 2, 6, 5], vec![2, 3, 7, 6], vec![3, 0, 4, 7],
            ],
            smooth_shading: false,
        };
        let src = SculptMesh::from_mesh(&cube);
        let bvh = Bvh::build(&src.positions, &src.tris);
        let remeshed = voxel_remesh(&src, &bvh, 24).unwrap();
        assert!(is_manifold(&remeshed), "remeshed cube must be watertight manifold");
        let vol = signed_volume(&remeshed);
        assert!(vol > 6500.0 && vol < 9000.0, "remeshed cube volume {vol} (expected ~8000)");
    }
}
