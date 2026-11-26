//! Draw in 3D: freehand 3D stroke-to-watertight-mesh (Tube / 3D Ribbon).
//!
//! Turns a sequence of 3D sample points `(pos, radius)` — captured from pointer/pencil drags
//! either on a camera-facing depth plane or surface-snapped to an existing mesh's BVH — into a
//! strictly watertight, outward-facing manifold `Mesh`:
//! - Catmull-Rom spline smoothing & uniform arc-length resampling
//! - Twist-free rotation-minimizing frames (parallel transport / Bishop frames)
//! - Configurable cross-section profiles (`Round`, `Square`, `Ribbon`)
//! - Smooth start & end tapering
//! - Optional closed loop (with twist-compensation closure) or Hemispherical / Flat end caps
//! - `[bool; 3]` symmetry mirroring (with automatic winding flip on odd reflections)

use crate::format::{add_mesh_command, Material, Mesh};
use glam::{Quat, Vec3};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Draw3DPoint {
    pub pos: Vec3,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TubeProfile {
    #[default]
    Round,
    Square,
    Ribbon,
}

impl TubeProfile {
    pub fn parse(s: &str) -> TubeProfile {
        match s.to_ascii_lowercase().as_str() {
            "square" => TubeProfile::Square,
            "ribbon" => TubeProfile::Ribbon,
            _ => TubeProfile::Round,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CapStyle {
    #[default]
    Hemisphere,
    Flat,
}

impl CapStyle {
    pub fn parse(s: &str) -> CapStyle {
        match s.to_ascii_lowercase().as_str() {
            "flat" => CapStyle::Flat,
            _ => CapStyle::Hemisphere,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Draw3DOptions {
    pub profile: TubeProfile,
    /// Cross-section ring subdivisions (e.g. 16 for Round, 4 for Square, 12 for Ribbon).
    pub radial_segments: u32,
    /// Start taper in `0.0..=1.0` (0 = full radius, 1 = sharp tip).
    pub taper_start: f32,
    /// End taper in `0.0..=1.0`.
    pub taper_end: f32,
    /// Connect last sample seamlessly back to the first (no end caps).
    pub closed: bool,
    pub cap_style: CapStyle,
    /// Mirror across X, Y, Z planes.
    pub symmetry: [bool; 3],
    /// Thickness ratio (minor/major axis) for `TubeProfile::Ribbon`.
    pub ribbon_aspect: f32,
    /// Catmull-Rom samples per input span (1 = polyline, 4 = smooth).
    pub smooth_subdivisions: u32,
}

impl Default for Draw3DOptions {
    fn default() -> Self {
        Draw3DOptions {
            profile: TubeProfile::Round,
            radial_segments: 16,
            taper_start: 0.0,
            taper_end: 0.0,
            closed: false,
            cap_style: CapStyle::Hemisphere,
            symmetry: [false; 3],
            ribbon_aspect: 0.28,
            smooth_subdivisions: 4,
        }
    }
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct Draw3DOptionsIn {
    profile: Option<String>,
    radial_segments: Option<u32>,
    taper_start: Option<f32>,
    taper_end: Option<f32>,
    closed: Option<bool>,
    cap_style: Option<String>,
    symmetry: Option<[bool; 3]>,
    ribbon_aspect: Option<f32>,
    smooth_subdivisions: Option<u32>,
    name: Option<String>,
}

impl Draw3DOptions {
    pub fn from_json(json: &str) -> (Draw3DOptions, String) {
        let parsed: Draw3DOptionsIn = serde_json::from_str(json).unwrap_or_default();
        let profile = parsed.profile.as_deref().map(TubeProfile::parse).unwrap_or_default();
        let default_segs = match profile {
            TubeProfile::Square => 4,
            TubeProfile::Ribbon => 12,
            TubeProfile::Round => 16,
        };
        let opts = Draw3DOptions {
            profile,
            radial_segments: parsed.radial_segments.unwrap_or(default_segs).clamp(3, 48),
            taper_start: parsed.taper_start.unwrap_or(0.0).clamp(0.0, 1.0),
            taper_end: parsed.taper_end.unwrap_or(0.0).clamp(0.0, 1.0),
            closed: parsed.closed.unwrap_or(false),
            cap_style: parsed.cap_style.as_deref().map(CapStyle::parse).unwrap_or_default(),
            symmetry: parsed.symmetry.unwrap_or([false; 3]),
            ribbon_aspect: parsed.ribbon_aspect.unwrap_or(0.28).clamp(0.08, 1.0),
            smooth_subdivisions: parsed.smooth_subdivisions.unwrap_or(4).clamp(1, 12),
        };
        let name = parsed.name.unwrap_or_else(|| match profile {
            TubeProfile::Ribbon => "3D Ribbon".into(),
            TubeProfile::Square => "3D Beam".into(),
            TubeProfile::Round => "3D Tube".into(),
        });
        (opts, name)
    }
}

/// Builds a watertight, outward-facing manifold `Mesh` (centered around a clean `pivot` point)
/// from 3D stroke points `(pos, radius)` and `Draw3DOptions`.
/// Returns `(local_mesh, world_pivot)` so the mesh can be placed at `world_pivot` as a scene object.
pub fn build_tube_mesh_with_pivot(raw_points: &[Draw3DPoint], opts: &Draw3DOptions) -> Result<(Mesh, Vec3), String> {
    let opts = Draw3DOptions {
        radial_segments: opts.radial_segments.clamp(3, 48),
        taper_start: if opts.taper_start.is_finite() { opts.taper_start.clamp(0.0, 1.0) } else { 0.0 },
        taper_end: if opts.taper_end.is_finite() { opts.taper_end.clamp(0.0, 1.0) } else { 0.0 },
        ribbon_aspect: if opts.ribbon_aspect.is_finite() { opts.ribbon_aspect.clamp(0.08, 1.0) } else { 0.28 },
        smooth_subdivisions: opts.smooth_subdivisions.clamp(1, 12),
        ..*opts
    };

    // 1. Clean & deduplicate consecutive coincident samples and sanitize radii.
    let mut clean: Vec<Draw3DPoint> = Vec::with_capacity(raw_points.len());
    for &pt in raw_points {
        if !pt.pos.is_finite() {
            continue;
        }
        let radius = if pt.radius.is_finite() { pt.radius.clamp(0.15, 500.0) } else { 1.0 };
        if let Some(last) = clean.last() {
            if (pt.pos - last.pos).length_squared() < 1e-4 {
                continue;
            }
        }
        clean.push(Draw3DPoint { pos: pt.pos, radius });
    }
    if opts.closed && clean.len() >= 2 && (clean[0].pos - clean.last().unwrap().pos).length_squared() < 1e-4 {
        clean.pop();
    }
    if clean.len() < 2 {
        return Err("Draw in 3D needs at least 2 distinct points".into());
    }

    // 2. Smooth with Catmull-Rom spline & resample uniformly by arc length.
    let mut spine = smooth_and_resample(&clean, opts.closed, opts.smooth_subdivisions);
    // Deduplicate any coincident resampled points (e.g. from tight loops or backtracking).
    let mut dedup: Vec<Draw3DPoint> = Vec::with_capacity(spine.len());
    for pt in spine.drain(..) {
        if !pt.pos.is_finite() {
            continue;
        }
        if let Some(last) = dedup.last() {
            if (pt.pos - last.pos).length_squared() < 1e-6 {
                continue;
            }
        }
        dedup.push(Draw3DPoint {
            pos: pt.pos,
            radius: if pt.radius.is_finite() { pt.radius.clamp(0.15, 500.0) } else { 1.0 },
        });
    }
    if opts.closed && dedup.len() >= 2 && (dedup[0].pos - dedup.last().unwrap().pos).length_squared() < 1e-6 {
        dedup.pop();
    }
    if dedup.len() < 2 {
        return Err("Draw in 3D curve collapsed".into());
    }

    // 3. Build single tube mesh in world space.
    let base = sweep_single_tube(&dedup, &opts);
    if base.positions.is_empty() || base.faces.is_empty() || !base.positions.iter().all(|p| p.is_finite()) {
        return Err("Draw in 3D produced invalid geometry".into());
    }

    // 4. Mirror according to symmetry flags `[bool; 3]`.
    let mut combined = combine_with_symmetry(&base, opts.symmetry);

    // 5. Choose clean object pivot: preserve 0 on mirrored axes so future symmetric sculpting aligns.
    let (mut bmin, mut bmax) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for &p in &combined.positions {
        bmin = bmin.min(p);
        bmax = bmax.max(p);
    }
    let mut pivot = (bmin + bmax) * 0.5;
    for axis in 0..3 {
        if opts.symmetry[axis] {
            pivot[axis] = 0.0;
        }
    }
    for p in &mut combined.positions {
        *p -= pivot;
    }

    Ok((combined, pivot))
}

/// Convenience wrapper returning the mesh in world coordinates (no recentering).
pub fn build_tube_mesh(raw_points: &[Draw3DPoint], opts: &Draw3DOptions) -> Result<Mesh, String> {
    let (mut mesh, pivot) = build_tube_mesh_with_pivot(raw_points, opts)?;
    for p in &mut mesh.positions {
        *p += pivot;
    }
    Ok(mesh)
}

/// Builds a `SceneCommand.addMesh` JSON string ready for Sculptor's Swift/Web/Android document core.
pub fn build_tube_add_mesh_command(
    raw_points: &[Draw3DPoint],
    opts: &Draw3DOptions,
    name: &str,
    material: Option<&Material>,
) -> Result<(Mesh, Vec3, String), String> {
    let (mesh, pivot) = build_tube_mesh_with_pivot(raw_points, opts)?;
    let cmd = add_mesh_command(&mesh, name, pivot, material);
    Ok((mesh, pivot, cmd))
}

fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

fn catmull_rom_f32(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

fn smooth_and_resample(pts: &[Draw3DPoint], closed: bool, subdiv: u32) -> Vec<Draw3DPoint> {
    let n = pts.len();
    let sub = subdiv.max(1) as usize;
    let mut dense: Vec<Draw3DPoint> = Vec::with_capacity(n * sub + 1);
    let segs = if closed { n } else { n - 1 };

    let get = |idx: isize| -> Draw3DPoint {
        if closed {
            pts[idx.rem_euclid(n as isize) as usize]
        } else {
            pts[idx.clamp(0, (n - 1) as isize) as usize]
        }
    };

    for i in 0..segs {
        let p0 = get(i as isize - 1);
        let p1 = get(i as isize);
        let p2 = get(i as isize + 1);
        let p3 = get(i as isize + 2);
        for s in 0..sub {
            let t = s as f32 / sub as f32;
            let pos = catmull_rom(p0.pos, p1.pos, p2.pos, p3.pos, t);
            let r = catmull_rom_f32(p0.radius, p1.radius, p2.radius, p3.radius, t).max(0.15);
            dense.push(Draw3DPoint { pos, radius: r });
        }
    }
    if !closed {
        dense.push(*pts.last().unwrap());
    }

    // Arc-length table for clean uniform spacing.
    let m = dense.len();
    let mut arc = Vec::with_capacity(m + 1);
    arc.push(0.0f32);
    for i in 1..m {
        let prev = *arc.last().unwrap();
        arc.push(prev + (dense[i].pos - dense[i - 1].pos).length());
    }
    if closed {
        let prev = *arc.last().unwrap();
        arc.push(prev + (dense[0].pos - dense[m - 1].pos).length());
    }
    let total_len = *arc.last().unwrap();
    if !(total_len > 1e-4) {
        return pts.to_vec();
    }

    let avg_r = dense.iter().map(|p| p.radius).sum::<f32>() / dense.len() as f32;
    let target_step = (avg_r * 0.55).clamp(0.4, 8.0);
    let max_samples = 240usize;
    let min_samples = if closed { 8 } else { 4 };
    let count = ((total_len / target_step).ceil() as usize).clamp(min_samples, max_samples);

    let mut out = Vec::with_capacity(count + 1);
    let mut seg = 0usize;
    let sample_at = |target_s: f32, seg: &mut usize| -> Draw3DPoint {
        while *seg + 1 < arc.len() - 1 && arc[*seg + 1] < target_s {
            *seg += 1;
        }
        let s0 = arc[*seg];
        let s1 = arc[*seg + 1];
        let t = if s1 - s0 > 1e-6 { ((target_s - s0) / (s1 - s0)).clamp(0.0, 1.0) } else { 0.0 };
        let a = dense[*seg];
        let b = dense[(*seg + 1) % m];
        Draw3DPoint {
            pos: a.pos.lerp(b.pos, t),
            radius: a.radius + (b.radius - a.radius) * t,
        }
    };

    if closed {
        for i in 0..count {
            let s = total_len * (i as f32 / count as f32);
            out.push(sample_at(s, &mut seg));
        }
    } else {
        for i in 0..=count {
            let s = total_len * (i as f32 / count as f32);
            out.push(sample_at(s, &mut seg));
        }
    }
    out
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn taper_factor(u: f32, taper_start: f32, taper_end: f32, closed: bool) -> f32 {
    if closed {
        return 1.0;
    }
    let mut scale = 1.0f32;
    if taper_start > 1e-4 && u < 0.28 {
        let s = smoothstep(0.0, 0.28, u);
        scale *= (1.0 - taper_start * 0.90) + (taper_start * 0.90) * s;
    }
    if taper_end > 1e-4 && u > 0.72 {
        let s = smoothstep(1.0, 0.72, u);
        scale *= (1.0 - taper_end * 0.90) + (taper_end * 0.90) * s;
    }
    scale.max(0.08)
}

fn profile_coords(profile: TubeProfile, segs: usize, ribbon_aspect: f32) -> Vec<(f32, f32)> {
    match profile {
        TubeProfile::Square => {
            // 4 clean corners rotated 45 deg so flat faces align with N/B.
            let s = std::f32::consts::FRAC_1_SQRT_2;
            vec![(s, s), (-s, s), (-s, -s), (s, -s)]
        }
        TubeProfile::Ribbon => (0..segs)
            .map(|j| {
                let th = std::f32::consts::TAU * (j as f32) / (segs as f32);
                (th.cos(), th.sin() * ribbon_aspect)
            })
            .collect(),
        TubeProfile::Round => (0..segs)
            .map(|j| {
                let th = std::f32::consts::TAU * (j as f32) / (segs as f32);
                (th.cos(), th.sin())
            })
            .collect(),
    }
}

/// NaN-free shortest-arc rotation between two unit vectors, safely handling exact 180° reversals.
fn safe_rotation_arc(from: Vec3, to: Vec3, fallback_axis: Vec3) -> Quat {
    let dot = from.dot(to).clamp(-1.0, 1.0);
    if dot > 0.99999 {
        return Quat::IDENTITY;
    }
    if dot < -0.99999 {
        let mut axis = fallback_axis - from * fallback_axis.dot(from);
        if axis.length_squared() < 1e-6 {
            let alt = if from.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
            axis = alt - from * alt.dot(from);
        }
        return Quat::from_axis_angle(axis.normalize(), std::f32::consts::PI);
    }
    Quat::from_rotation_arc(from, to)
}

fn sweep_single_tube(spine: &[Draw3DPoint], opts: &Draw3DOptions) -> Mesh {
    let m = spine.len();
    let closed = opts.closed && m >= 3;

    // 1. Unit tangents (robust against 180° hairpins where central difference cancels to zero).
    let mut tangents = Vec::with_capacity(m);
    for i in 0..m {
        let mut t = if closed {
            let prev = spine[(i + m - 1) % m].pos;
            let next = spine[(i + 1) % m].pos;
            next - prev
        } else if i == 0 {
            spine[1].pos - spine[0].pos
        } else if i == m - 1 {
            spine[m - 1].pos - spine[m - 2].pos
        } else {
            spine[i + 1].pos - spine[i - 1].pos
        };
        if t.length_squared() < 1e-8 && i + 1 < m {
            t = spine[i + 1].pos - spine[i].pos;
        }
        if t.length_squared() < 1e-8 && i > 0 {
            t = spine[i].pos - spine[i - 1].pos;
        }
        let unit = t.normalize_or_zero();
        let fallback = if i > 0 { tangents[i - 1] } else { Vec3::Z };
        tangents.push(if unit.length_squared() > 0.5 { unit } else { fallback });
    }

    // 2. Rotation-minimizing frames (parallel transport / Bishop frames).
    let mut normals = Vec::with_capacity(m);
    let mut binormals = Vec::with_capacity(m);
    let ref_up = if tangents[0].z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
    let n0 = (ref_up - tangents[0] * ref_up.dot(tangents[0])).normalize();
    normals.push(n0);
    binormals.push(tangents[0].cross(n0).normalize());

    for i in 1..m {
        let q = safe_rotation_arc(tangents[i - 1], tangents[i], normals[i - 1]);
        let mut ni = q * normals[i - 1];
        // Re-orthonormalize to eliminate floating-point drift.
        ni = (ni - tangents[i] * ni.dot(tangents[i])).normalize_or_zero();
        if ni.length_squared() < 0.5 || !ni.is_finite() {
            let alt = if tangents[i].z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
            ni = (alt - tangents[i] * alt.dot(tangents[i])).normalize();
        }
        normals.push(ni);
        binormals.push(tangents[i].cross(ni).normalize());
    }

    // Close twist seamlessly for closed loops.
    if closed {
        let q_close = safe_rotation_arc(tangents[m - 1], tangents[0], normals[m - 1]);
        let mut n_end = q_close * normals[m - 1];
        n_end = (n_end - tangents[0] * n_end.dot(tangents[0])).normalize_or_zero();
        if n_end.length_squared() > 0.5 && n_end.is_finite() {
            let cos_a = n_end.dot(normals[0]).clamp(-1.0, 1.0);
            let sin_a = n_end.dot(binormals[0]);
            let twist = sin_a.atan2(cos_a);
            if twist.is_finite() {
                for i in 1..m {
                    let ang = -twist * (i as f32 / m as f32);
                    let (s, c) = ang.sin_cos();
                    let n = normals[i];
                    let b = binormals[i];
                    normals[i] = n * c + b * s;
                    binormals[i] = -n * s + b * c;
                }
            }
        }
    }

    // 3. Cross-section profile.
    let segs = match opts.profile {
        TubeProfile::Square => 4,
        _ => opts.radial_segments.clamp(3, 48) as usize,
    };
    let profile = profile_coords(opts.profile, segs, opts.ribbon_aspect);
    let k = profile.len();

    let mut positions: Vec<Vec3> = Vec::with_capacity((m + 8) * k + 2);
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity((m + 8) * k);

    let push_ring = |positions: &mut Vec<Vec3>, center: Vec3, n: Vec3, b: Vec3, r: f32| -> u32 {
        let base = positions.len() as u32;
        for &(px, py) in &profile {
            positions.push(center + n * (px * r) + b * (py * r));
        }
        base
    };

    // Main spine rings.
    let mut ring_starts: Vec<u32> = Vec::with_capacity(m);
    let mut radii: Vec<f32> = Vec::with_capacity(m);
    for i in 0..m {
        let u = if m > 1 { i as f32 / (m - 1) as f32 } else { 0.0 };
        let r = spine[i].radius * taper_factor(u, opts.taper_start, opts.taper_end, closed);
        radii.push(r);
        ring_starts.push(push_ring(&mut positions, spine[i].pos, normals[i], binormals[i], r));
    }

    // Connect consecutive rings with outward-facing CCW quads.
    let connect_rings = |faces: &mut Vec<Vec<u32>>, r0: u32, r1: u32| {
        for j in 0..k {
            let jn = (j + 1) % k;
            let a = r0 + j as u32;
            let b = r0 + jn as u32;
            let c = r1 + jn as u32;
            let d = r1 + j as u32;
            faces.push(vec![a, b, c, d]);
        }
    };

    for i in 0..m - 1 {
        connect_rings(&mut faces, ring_starts[i], ring_starts[i + 1]);
    }
    if closed {
        connect_rings(&mut faces, ring_starts[m - 1], ring_starts[0]);
    } else {
        // 4. Watertight end caps.
        let cap_depth_scale = match opts.profile {
            TubeProfile::Ribbon => opts.ribbon_aspect.max(0.25),
            _ => 1.0,
        };
        match opts.cap_style {
            CapStyle::Flat => {
                // Start cap facing -T[0].
                let start_center = positions.len() as u32;
                positions.push(spine[0].pos);
                let r0 = ring_starts[0];
                for j in 0..k {
                    let jn = (j + 1) % k;
                    faces.push(vec![start_center, r0 + jn as u32, r0 + j as u32]);
                }
                // End cap facing +T[m-1].
                let end_center = positions.len() as u32;
                positions.push(spine[m - 1].pos);
                let rn = ring_starts[m - 1];
                for j in 0..k {
                    let jn = (j + 1) % k;
                    faces.push(vec![end_center, rn + j as u32, rn + jn as u32]);
                }
            }
            CapStyle::Hemisphere => {
                let cap_rings = 3usize;
                // Start hemisphere stepping along -T[0].
                let r_start = radii[0];
                let mut prev_ring = ring_starts[0];
                for step in 1..cap_rings {
                    let phi = std::f32::consts::FRAC_PI_2 * (step as f32 / cap_rings as f32);
                    let c_pos = spine[0].pos - tangents[0] * (r_start * cap_depth_scale * phi.sin());
                    let r_ring = r_start * phi.cos();
                    let cur = push_ring(&mut positions, c_pos, normals[0], binormals[0], r_ring);
                    // Moving toward -T, so cur precedes prev_ring along +T: connect(cur, prev_ring).
                    connect_rings(&mut faces, cur, prev_ring);
                    prev_ring = cur;
                }
                let start_pole = positions.len() as u32;
                positions.push(spine[0].pos - tangents[0] * (r_start * cap_depth_scale));
                for j in 0..k {
                    let jn = (j + 1) % k;
                    faces.push(vec![start_pole, prev_ring + jn as u32, prev_ring + j as u32]);
                }

                // End hemisphere stepping along +T[m-1].
                let r_end = radii[m - 1];
                let mut prev_ring = ring_starts[m - 1];
                for step in 1..cap_rings {
                    let phi = std::f32::consts::FRAC_PI_2 * (step as f32 / cap_rings as f32);
                    let c_pos = spine[m - 1].pos + tangents[m - 1] * (r_end * cap_depth_scale * phi.sin());
                    let r_ring = r_end * phi.cos();
                    let cur = push_ring(&mut positions, c_pos, normals[m - 1], binormals[m - 1], r_ring);
                    connect_rings(&mut faces, prev_ring, cur);
                    prev_ring = cur;
                }
                let end_pole = positions.len() as u32;
                positions.push(spine[m - 1].pos + tangents[m - 1] * (r_end * cap_depth_scale));
                for j in 0..k {
                    let jn = (j + 1) % k;
                    faces.push(vec![end_pole, prev_ring + j as u32, prev_ring + jn as u32]);
                }
            }
        }
    }

    Mesh {
        positions,
        faces,
        smooth_shading: opts.profile != TubeProfile::Square,
    }
}

fn combine_with_symmetry(base: &Mesh, symmetry: [bool; 3]) -> Mesh {
    let mut signs = vec![Vec3::ONE];
    for axis in 0..3 {
        if symmetry[axis] {
            let n = signs.len();
            for i in 0..n {
                let mut s = signs[i];
                s[axis] = -s[axis];
                signs.push(s);
            }
        }
    }
    if signs.len() == 1 {
        return base.clone();
    }
    let mut positions = Vec::with_capacity(base.positions.len() * signs.len());
    let mut faces = Vec::with_capacity(base.faces.len() * signs.len());
    for s in signs {
        let base_idx = positions.len() as u32;
        let flip = (s.x * s.y * s.z) < 0.0;
        for &p in &base.positions {
            positions.push(p * s);
        }
        for f in &base.faces {
            if flip {
                faces.push(f.iter().rev().map(|&idx| idx + base_idx).collect());
            } else {
                faces.push(f.iter().map(|&idx| idx + base_idx).collect());
            }
        }
    }
    Mesh {
        positions,
        faces,
        smooth_shading: base.smooth_shading,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn helix_points() -> Vec<Draw3DPoint> {
        (0..12)
            .map(|i| {
                let t = i as f32 * 0.45;
                Draw3DPoint {
                    pos: Vec3::new(15.0 + t.cos() * 6.0, t.sin() * 6.0, i as f32 * 2.5),
                    radius: 2.0,
                }
            })
            .collect()
    }

    #[test]
    fn round_square_ribbon_tubes_are_watertight_and_outward_facing() {
        let pts = helix_points();
        for (profile, cap) in [
            (TubeProfile::Round, CapStyle::Hemisphere),
            (TubeProfile::Round, CapStyle::Flat),
            (TubeProfile::Square, CapStyle::Flat),
            (TubeProfile::Ribbon, CapStyle::Hemisphere),
        ] {
            let opts = Draw3DOptions {
                profile,
                cap_style: cap,
                taper_start: 0.4,
                taper_end: 0.7,
                ..Default::default()
            };
            let mesh = build_tube_mesh(&pts, &opts).unwrap();
            assert!(is_manifold(&mesh), "profile {profile:?} cap {cap:?} must be watertight manifold");
            let vol = signed_volume(&mesh);
            assert!(vol > 10.0, "profile {profile:?} cap {cap:?} must have positive outward volume: {vol}");
        }
    }

    #[test]
    fn closed_loop_and_symmetry_are_watertight_and_outward() {
        let loop_pts: Vec<Draw3DPoint> = (0..16)
            .map(|i| {
                let a = std::f32::consts::TAU * i as f32 / 16.0;
                Draw3DPoint { pos: Vec3::new(18.0 + a.cos() * 8.0, a.sin() * 8.0, (a * 2.0).sin() * 2.0), radius: 2.0 }
            })
            .collect();
        let opts = Draw3DOptions {
            closed: true,
            symmetry: [true, false, false],
            ..Default::default()
        };
        let (mesh, pivot) = build_tube_mesh_with_pivot(&loop_pts, &opts).unwrap();
        assert_eq!(pivot.x, 0.0, "X symmetry keeps pivot.x at 0");
        assert!(is_manifold(&mesh), "closed mirrored tube must be watertight manifold");
        assert!(signed_volume(&mesh) > 100.0, "both original and mirrored tubes must have outward normals");
    }

    #[test]
    fn add_mesh_command_json_is_valid() {
        let pts = helix_points();
        let (_, _, cmd) = build_tube_add_mesh_command(&pts, &Draw3DOptions::default(), "3D Tube", None).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&cmd).unwrap();
        assert!(parsed.get("addMesh").is_some());
        assert_eq!(parsed["addMesh"]["name"], "3D Tube");
        assert!(parsed["addMesh"]["_0"]["positions"].is_array());
        assert!(parsed["addMesh"]["_0"]["faces"].is_array());
    }

    #[test]
    fn single_point_and_coincident_points_return_clean_error() {
        let single = vec![Draw3DPoint { pos: Vec3::new(1.0, 2.0, 3.0), radius: 2.0 }];
        assert!(build_tube_mesh(&single, &Draw3DOptions::default()).is_err());

        let dupes = vec![
            Draw3DPoint { pos: Vec3::new(5.0, 5.0, 5.0), radius: 2.0 },
            Draw3DPoint { pos: Vec3::new(5.0, 5.0, 5.0), radius: 2.0 },
            Draw3DPoint { pos: Vec3::new(5.0001, 5.0, 5.0), radius: 2.0 },
        ];
        assert!(build_tube_mesh(&dupes, &Draw3DOptions::default()).is_err());
    }

    #[test]
    fn sharp_180_degree_reversal_has_no_nan_and_is_watertight() {
        // Explicit 180-degree hairpin backtrack along +X then -X!
        let hairpin = vec![
            Draw3DPoint { pos: Vec3::new(0.0, 0.0, 0.0), radius: 1.5 },
            Draw3DPoint { pos: Vec3::new(10.0, 0.0, 0.0), radius: 1.5 },
            Draw3DPoint { pos: Vec3::new(20.0, 0.0, 0.0), radius: 1.5 },
            Draw3DPoint { pos: Vec3::new(10.0, 0.0, 0.0), radius: 1.5 },
            Draw3DPoint { pos: Vec3::new(0.0, 0.0, 0.0), radius: 1.5 },
        ];
        let opts = Draw3DOptions { smooth_subdivisions: 1, ..Default::default() };
        let mesh = build_tube_mesh(&hairpin, &opts).unwrap();
        assert!(mesh.positions.iter().all(|p| p.is_finite()), "no NaN/Inf positions on 180-degree reversal");
        assert!(is_manifold(&mesh), "180-degree hairpin tube must remain strictly manifold");
    }

    #[test]
    fn extreme_radii_and_tapers_stay_finite_and_watertight() {
        let pts = vec![
            Draw3DPoint { pos: Vec3::new(0.0, 0.0, 0.0), radius: -10.0 },
            Draw3DPoint { pos: Vec3::new(5.0, 0.0, 5.0), radius: f32::NAN },
            Draw3DPoint { pos: Vec3::new(10.0, 2.0, 10.0), radius: 1e9 },
            Draw3DPoint { pos: Vec3::new(15.0, 0.0, 15.0), radius: 0.0 },
        ];
        let opts = Draw3DOptions {
            taper_start: 5.0,
            taper_end: f32::NAN,
            ribbon_aspect: -2.0,
            radial_segments: 0,
            ..Default::default()
        };
        let mesh = build_tube_mesh(&pts, &opts).unwrap();
        assert!(mesh.positions.iter().all(|p| p.is_finite()));
        assert!(is_manifold(&mesh));
    }
}
