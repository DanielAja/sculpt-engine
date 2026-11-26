//! Built-in 38-shape parametric catalog + 2D Sketch / Point-Placement → Live Extrude engine.
//!
//! Coordinate convention matches Sculptor across Web, Apple, Android, and Desktop:
//! - Right-handed **Z-up**, millimetres (`X` right, `Y` forward, `Z` up).
//! - Every generated mesh is strictly **watertight (closed 2-manifold)** with outward CCW
//!   normals, positive signed volume, centered in X/Y around `0`, and resting cleanly on `Z = 0`.
//! - Default characteristic scale is `20.0 mm` (matching Sculptor's existing `20×20×20 mm` primitives).

use crate::format::{add_mesh_command, Mesh};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::f32::consts::{PI, TAU};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShapeCatalogEntry {
    pub id: &'static str,
    pub name: &'static str,
    pub category: &'static str,
    pub category_label: &'static str,
    pub default_size: [f32; 3],
    pub tags: &'static [&'static str],
}

pub const SHAPE_CATALOG: &[ShapeCatalogEntry] = &[
    // Basics (8)
    ShapeCatalogEntry { id: "box", name: "Box", category: "basics", category_label: "Basics", default_size: [20.0, 20.0, 20.0], tags: &["cube", "block", "square"] },
    ShapeCatalogEntry { id: "sphere", name: "Sphere", category: "basics", category_label: "Basics", default_size: [20.0, 20.0, 20.0], tags: &["ball", "orb", "round"] },
    ShapeCatalogEntry { id: "cylinder", name: "Cylinder", category: "basics", category_label: "Basics", default_size: [20.0, 20.0, 20.0], tags: &["drum", "rod", "post"] },
    ShapeCatalogEntry { id: "cone", name: "Cone", category: "basics", category_label: "Basics", default_size: [20.0, 20.0, 22.0], tags: &["spike", "funnel", "tip"] },
    ShapeCatalogEntry { id: "torus", name: "Torus", category: "basics", category_label: "Basics", default_size: [26.0, 26.0, 9.0], tags: &["donut", "ring", "loop"] },
    ShapeCatalogEntry { id: "capsule", name: "Capsule", category: "basics", category_label: "Basics", default_size: [14.0, 14.0, 26.0], tags: &["pill", "stadium"] },
    ShapeCatalogEntry { id: "pyramid", name: "Pyramid", category: "basics", category_label: "Basics", default_size: [22.0, 22.0, 20.0], tags: &["egypt", "point"] },
    ShapeCatalogEntry { id: "prism", name: "Prism", category: "basics", category_label: "Basics", default_size: [20.0, 18.0, 20.0], tags: &["triangle", "wedge"] },
    // Rounded & Organic (7)
    ShapeCatalogEntry { id: "rounded_box", name: "Rounded Box", category: "organic", category_label: "Rounded & Organic", default_size: [20.0, 20.0, 20.0], tags: &["bevel", "cube", "soft"] },
    ShapeCatalogEntry { id: "egg", name: "Egg", category: "organic", category_label: "Rounded & Organic", default_size: [16.0, 16.0, 22.0], tags: &["ovoid", "oval"] },
    ShapeCatalogEntry { id: "teardrop", name: "Teardrop", category: "organic", category_label: "Rounded & Organic", default_size: [16.0, 16.0, 24.0], tags: &["drop", "rain", "flame"] },
    ShapeCatalogEntry { id: "heart", name: "Heart", category: "organic", category_label: "Rounded & Organic", default_size: [22.0, 20.0, 9.0], tags: &["love", "symbol"] },
    ShapeCatalogEntry { id: "star_prism", name: "Star", category: "organic", category_label: "Rounded & Organic", default_size: [24.0, 23.0, 8.0], tags: &["star", "badge", "five"] },
    ShapeCatalogEntry { id: "pebble", name: "Pebble", category: "organic", category_label: "Rounded & Organic", default_size: [24.0, 18.0, 13.0], tags: &["stone", "rock", "smooth"] },
    ShapeCatalogEntry { id: "dome", name: "Dome", category: "organic", category_label: "Rounded & Organic", default_size: [22.0, 22.0, 11.0], tags: &["hemisphere", "cap", "igloo"] },
    // Polyhedra & Gems (6)
    ShapeCatalogEntry { id: "tetrahedron", name: "Tetrahedron", category: "polyhedra", category_label: "Polyhedra & Gems", default_size: [22.0, 20.0, 18.0], tags: &["d4", "platonic", "crystal"] },
    ShapeCatalogEntry { id: "octahedron", name: "Octahedron", category: "polyhedra", category_label: "Polyhedra & Gems", default_size: [20.0, 20.0, 24.0], tags: &["d8", "diamond", "platonic"] },
    ShapeCatalogEntry { id: "dodecahedron", name: "Dodecahedron", category: "polyhedra", category_label: "Polyhedra & Gems", default_size: [22.0, 22.0, 22.0], tags: &["d12", "pentagon", "platonic"] },
    ShapeCatalogEntry { id: "icosahedron", name: "Icosahedron", category: "polyhedra", category_label: "Polyhedra & Gems", default_size: [22.0, 22.0, 22.0], tags: &["d20", "geodesic", "platonic"] },
    ShapeCatalogEntry { id: "gem", name: "Gem", category: "polyhedra", category_label: "Polyhedra & Gems", default_size: [22.0, 22.0, 16.0], tags: &["diamond", "jewel", "facet"] },
    ShapeCatalogEntry { id: "hex_prism", name: "Hex Prism", category: "polyhedra", category_label: "Polyhedra & Gems", default_size: [22.0, 19.0, 20.0], tags: &["hexagon", "tile", "bolt"] },
    // Architecture (6)
    ShapeCatalogEntry { id: "arch", name: "Arch", category: "architecture", category_label: "Architecture", default_size: [24.0, 10.0, 22.0], tags: &["bridge", "doorway", "gate"] },
    ShapeCatalogEntry { id: "wedge", name: "Wedge", category: "architecture", category_label: "Architecture", default_size: [22.0, 22.0, 16.0], tags: &["ramp", "slope", "incline"] },
    ShapeCatalogEntry { id: "stairs", name: "Stairs", category: "architecture", category_label: "Architecture", default_size: [22.0, 26.0, 20.0], tags: &["steps", "flight", "terrace"] },
    ShapeCatalogEntry { id: "column", name: "Column", category: "architecture", category_label: "Architecture", default_size: [14.0, 14.0, 30.0], tags: &["pillar", "fluted", "greek"] },
    ShapeCatalogEntry { id: "roof_gable", name: "Gable Roof", category: "architecture", category_label: "Architecture", default_size: [24.0, 22.0, 15.0], tags: &["house", "pitched", "attic"] },
    ShapeCatalogEntry { id: "wall_window", name: "Window Frame", category: "architecture", category_label: "Architecture", default_size: [26.0, 6.0, 22.0], tags: &["portal", "wall", "frame"] },
    // Mechanical & Hardware (6)
    ShapeCatalogEntry { id: "tube", name: "Tube", category: "mechanical", category_label: "Mechanical & Hardware", default_size: [20.0, 20.0, 24.0], tags: &["pipe", "hollow", "sleeve"] },
    ShapeCatalogEntry { id: "ring", name: "Washer Ring", category: "mechanical", category_label: "Mechanical & Hardware", default_size: [24.0, 24.0, 4.0], tags: &["washer", "spacer", "collar"] },
    ShapeCatalogEntry { id: "gear", name: "Spur Gear", category: "mechanical", category_label: "Mechanical & Hardware", default_size: [26.0, 26.0, 6.0], tags: &["cog", "wheel", "teeth"] },
    ShapeCatalogEntry { id: "hex_nut", name: "Hex Nut", category: "mechanical", category_label: "Mechanical & Hardware", default_size: [22.0, 19.0, 9.0], tags: &["fastener", "hardware", "bolt"] },
    ShapeCatalogEntry { id: "spring", name: "Helix Spring", category: "mechanical", category_label: "Mechanical & Hardware", default_size: [18.0, 18.0, 26.0], tags: &["coil", "spiral", "helix"] },
    ShapeCatalogEntry { id: "l_bracket", name: "L-Bracket", category: "mechanical", category_label: "Mechanical & Hardware", default_size: [22.0, 16.0, 22.0], tags: &["angle", "mount", "corner"] },
    // Containers & Everyday (5)
    ShapeCatalogEntry { id: "bowl", name: "Bowl", category: "containers", category_label: "Containers & Everyday", default_size: [24.0, 24.0, 12.0], tags: &["dish", "vessel", "hollow"] },
    ShapeCatalogEntry { id: "cup", name: "Cup", category: "containers", category_label: "Containers & Everyday", default_size: [18.0, 18.0, 22.0], tags: &["mug", "tumbler", "beaker"] },
    ShapeCatalogEntry { id: "funnel", name: "Funnel", category: "containers", category_label: "Containers & Everyday", default_size: [24.0, 24.0, 24.0], tags: &["pour", "cone", "hopper"] },
    ShapeCatalogEntry { id: "tray", name: "Tray", category: "containers", category_label: "Containers & Everyday", default_size: [28.0, 20.0, 7.0], tags: &["box", "organizer", "dish"] },
    ShapeCatalogEntry { id: "vase", name: "Vase", category: "containers", category_label: "Containers & Everyday", default_size: [18.0, 18.0, 28.0], tags: &["pottery", "urn", "lathe"] },
];

pub fn shape_catalog_json() -> String {
    serde_json::to_string(SHAPE_CATALOG).unwrap_or_else(|_| "[]".to_string())
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ShapeParams {
    /// Characteristic scale in mm (default 20.0; scales relative to 20.0 mm baseline).
    pub size: Option<f32>,
    /// Optional non-uniform scale multipliers `[sx, sy, sz]`.
    pub scale: Option<[f32; 3]>,
    /// World placement `[x, y, z]` in mm (default `[0, 0, 0]`).
    pub position: Option<[f32; 3]>,
    /// Optional object name override.
    pub name: Option<String>,
    /// Optional radial segment count for curved shapes.
    pub segments: Option<usize>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ExtrudeProfileOptions {
    /// `"freehand"` or `"points"` (controls default simplification & corner smoothing).
    pub mode: Option<String>,
    /// Extrusion height in mm (positive = +Z, negative = -Z; default `16.0`).
    pub height: Option<f32>,
    /// Optional chamfered top/bottom bevel in mm (`0.0..=height*0.35`, default `0.0`).
    pub bevel: Option<f32>,
    /// Optional top cap scale relative to centroid (`0.2..=1.5`, default `1.0`).
    pub taper: Option<f32>,
    /// Douglas-Peucker simplification tolerance in mm (default `0.45` in freehand mode, `0.0` in points mode).
    #[serde(alias = "simplify_tolerance")]
    pub simplify_tolerance: Option<f32>,
    /// Chaikin corner-smoothing passes (`0..=3`; default `1` in freehand mode, `0` in points mode).
    #[serde(alias = "smooth_passes")]
    pub smooth_passes: Option<u32>,
    /// Optional custom name (default `"Extruded Shape"`).
    pub name: Option<String>,
    /// Optional extra world translation `[x, y, z]` added on top of the drawn workplane centroid.
    pub position: Option<[f32; 3]>,
    /// Whether to mark the mesh with smooth shading (default `false` for crisp extrusions).
    #[serde(alias = "smooth_shading")]
    pub smooth_shading: Option<bool>,
}

/// Generates a watertight mesh for any catalog shape ID, resting on `Z = 0`.
pub fn generate_shape_mesh(id: &str, params_json: &str) -> Result<(Mesh, String, Vec3), String> {
    let params: ShapeParams = if params_json.trim().is_empty() {
        ShapeParams::default()
    } else {
        serde_json::from_str(params_json).map_err(|e| format!("invalid params JSON: {e}"))?
    };

    let key = id.trim().to_ascii_lowercase();
    let entry = SHAPE_CATALOG
        .iter()
        .find(|e| e.id == key)
        .ok_or_else(|| format!("unknown shape id '{id}'"))?;

    let segs = params.segments.unwrap_or(32).clamp(12, 96);
    let mut mesh = match key.as_str() {
        // Basics (8)
        "box" => build_box(20.0, 20.0, 20.0),
        "sphere" => build_lathe_sphere(10.0, segs, (segs / 2).max(10)),
        "cylinder" => build_lathe(&[(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)], segs, true),
        "cone" => build_lathe(&[(0.0, 0.0), (10.0, 0.0), (0.0, 22.0)], segs, true),
        "torus" => build_torus(9.0, 4.0, segs, (segs / 2).max(12)),
        "capsule" => build_capsule(7.0, 12.0, segs, (segs / 4).max(6)),
        "pyramid" => build_pyramid(22.0, 22.0, 20.0),
        "prism" => build_regular_prism(3, 11.5, 20.0, false),

        // Rounded & Organic (7)
        "rounded_box" => build_rounded_box(20.0, 20.0, 20.0, 3.6, 5),
        "egg" => build_egg(8.2, 22.0, segs, (segs / 2).max(14)),
        "teardrop" => build_teardrop(8.0, 24.0, segs, (segs / 2).max(16)),
        "heart" => build_heart(22.0, 9.0),
        "star_prism" => build_star_prism(5, 12.0, 5.2, 8.0),
        "pebble" => build_pebble(12.0, 9.2, 6.5),
        "dome" => build_dome(11.0, segs, (segs / 4).max(8)),

        // Polyhedra & Gems (6)
        "tetrahedron" => build_tetrahedron(13.0),
        "octahedron" => build_octahedron(10.0, 12.0),
        "dodecahedron" => build_dodecahedron(11.0),
        "icosahedron" => build_icosahedron(11.0),
        "gem" => build_gem(11.0, 16.0, 10),
        "hex_prism" => build_regular_prism(6, 11.0, 20.0, false),

        // Architecture (6)
        "arch" => build_arch(24.0, 10.0, 22.0, 4.5, 14),
        "wedge" => build_wedge(22.0, 22.0, 16.0),
        "stairs" => build_stairs(22.0, 26.0, 20.0, 5),
        "column" => build_column(6.5, 30.0, 12),
        "roof_gable" => build_roof_gable(24.0, 22.0, 15.0),
        "wall_window" => build_wall_window(26.0, 6.0, 22.0, 4.0),

        // Mechanical & Hardware (6)
        "tube" => build_annular_circle(10.0, 7.0, 24.0, segs, true),
        "ring" => build_annular_circle(12.0, 6.5, 4.0, segs, true),
        "gear" => build_gear(12, 13.0, 9.6, 3.8, 6.0),
        "hex_nut" => build_hex_nut(11.0, 5.2, 9.0),
        "spring" => build_spring(7.2, 1.8, 4.0, 24.0, 24, 10),
        "l_bracket" => build_l_bracket(22.0, 16.0, 22.0, 4.5),

        // Containers & Everyday (5)
        "bowl" => build_bowl(12.0, 12.0, 2.2, segs),
        "cup" => build_cup(9.0, 7.4, 22.0, 1.8, segs),
        "funnel" => build_funnel(12.0, 2.6, 24.0, 1.4, segs),
        "tray" => build_tray(28.0, 20.0, 7.0, 2.2, 2.0),
        "vase" => build_vase(9.0, 28.0, 1.8, segs),

        _ => return Err(format!("unknown shape id '{id}'")),
    };

    // Apply uniform size multiplier (relative to 20.0 mm baseline) + optional XYZ scale.
    let size_mul = params.size.unwrap_or(20.0).clamp(0.2, 5000.0) / 20.0;
    let s = params.scale.unwrap_or([1.0, 1.0, 1.0]);
    let sx = (s[0].abs().max(0.05)) * size_mul;
    let sy = (s[1].abs().max(0.05)) * size_mul;
    let sz = (s[2].abs().max(0.05)) * size_mul;
    if (sx - 1.0).abs() > 1e-5 || (sy - 1.0).abs() > 1e-5 || (sz - 1.0).abs() > 1e-5 {
        for p in &mut mesh.positions {
            p.x *= sx;
            p.y *= sy;
            p.z *= sz;
        }
    }

    ground_and_center(&mut mesh);
    ensure_outward_winding(&mut mesh);

    let name = params
        .name
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| entry.name.to_string());
    let pos = params.position.map(Vec3::from_array).unwrap_or(Vec3::ZERO);
    Ok((mesh, name, pos))
}

/// Generates a `SceneCommand.addMesh` JSON string for any built-in catalog shape.
pub fn generate_shape_command_json(id: &str, params_json: &str) -> Result<String, String> {
    let (mesh, name, pos) = generate_shape_mesh(id, params_json)?;
    Ok(add_mesh_command(&mesh, &name, pos, None))
}

/// Extrudes a 2D closed outline (`[x0, y0, x1, y1, ...]` on the Z=0 workplane) into a watertight
/// 3D `Mesh` with optional freehand simplification, Chaikin smoothing, bevel, and taper.
pub fn extrude_profile_2d_mesh(
    points_xy: &[f32],
    options_json: &str,
) -> Result<(Mesh, String, Vec3), String> {
    if points_xy.len() < 6 || points_xy.len() % 2 != 0 {
        return Err("points_xy must contain at least 3 (x, y) coordinate pairs".to_string());
    }
    let opts: ExtrudeProfileOptions = if options_json.trim().is_empty() {
        ExtrudeProfileOptions::default()
    } else {
        serde_json::from_str(options_json).map_err(|e| format!("invalid options JSON: {e}"))?
    };

    let freehand = opts
        .mode
        .as_deref()
        .map(|m| m.eq_ignore_ascii_case("freehand"))
        .unwrap_or(false);
    let raw_height = opts.height.unwrap_or(16.0);
    let height = if raw_height.is_finite() && raw_height.abs() >= 0.2 {
        raw_height.clamp(-2000.0, 2000.0)
    } else if raw_height < 0.0 {
        -0.2
    } else {
        16.0
    };
    let abs_h = height.abs();
    let taper = opts.taper.unwrap_or(1.0).clamp(0.2, 1.5);
    let requested_bevel = opts.bevel.unwrap_or(0.0).max(0.0);
    let simplify_tol = opts
        .simplify_tolerance
        .unwrap_or(if freehand { 0.45 } else { 0.0 })
        .max(0.0);
    let smooth_passes = opts
        .smooth_passes
        .unwrap_or(if freehand { 1 } else { 0 })
        .min(3);

    // 1. Collect finite points and remove consecutive duplicates.
    let mut pts: Vec<Vec2> = points_xy
        .chunks_exact(2)
        .filter(|c| c[0].is_finite() && c[1].is_finite())
        .map(|c| Vec2::new(c[0], c[1]))
        .collect();
    dedup_closed_ring(&mut pts, 1e-3);
    if pts.len() < 3 {
        return Err("outline needs at least 3 distinct points".to_string());
    }

    // 2. Simplify freehand strokes with closed-loop Douglas-Peucker.
    if simplify_tol > 1e-4 && pts.len() > 4 {
        pts = simplify_closed_polygon(&pts, simplify_tol);
        dedup_closed_ring(&mut pts, 1e-3);
    }

    // 3. Optional Chaikin corner rounding for organic freehand shapes.
    if smooth_passes > 0 && pts.len() >= 3 {
        for _ in 0..smooth_passes {
            if pts.len() >= 256 {
                break;
            }
            pts = chaikin_closed(&pts);
        }
        dedup_closed_ring(&mut pts, 1e-3);
    }

    // 4. Clean self-intersections & collinear spikes, enforce CCW winding.
    pts = sanitize_simple_polygon(pts)?;

    // Center around 2D bounding-box center so the object's transform pivot sits cleanly inside it.
    let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
    for &p in &pts {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    let center = Vec2::new((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
    let local_pts: Vec<Vec2> = pts.iter().map(|&p| p - center).collect();

    let smooth_shading = opts.smooth_shading.unwrap_or(false);
    let mut mesh = build_extruded_polygon_2d(&local_pts, abs_h, requested_bevel, taper, smooth_shading)?;
    ensure_outward_winding(&mut mesh);

    let extra_pos = opts.position.map(Vec3::from_array).unwrap_or(Vec3::ZERO);
    let z_base = if height < 0.0 { height } else { 0.0 };
    let world_pos = Vec3::new(center.x + extra_pos.x, center.y + extra_pos.y, z_base + extra_pos.z);
    let name = opts
        .name
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| if freehand { "Drawn Shape".to_string() } else { "Extruded Shape".to_string() });

    Ok((mesh, name, world_pos))
}

/// Extrudes a 2D closed outline and returns a `SceneCommand.addMesh` JSON string.
pub fn extrude_profile_2d_command_json(points_xy: &[f32], options_json: &str) -> Result<String, String> {
    let (mesh, name, pos) = extrude_profile_2d_mesh(points_xy, options_json)?;
    Ok(add_mesh_command(&mesh, &name, pos, None))
}

// ---------------------------------------------------------------------------
// Geometry Helpers & Watertight Primitive Builders
// ---------------------------------------------------------------------------

fn ground_and_center(mesh: &mut Mesh) {
    if mesh.positions.is_empty() {
        return;
    }
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for &p in &mesh.positions {
        min = min.min(p);
        max = max.max(p);
    }
    let shift = Vec3::new((min.x + max.x) * 0.5, (min.y + max.y) * 0.5, min.z);
    for p in &mut mesh.positions {
        *p -= shift;
        if p.z.abs() < 1e-6 {
            p.z = 0.0;
        }
    }
}

pub(crate) fn signed_volume(mesh: &Mesh) -> f32 {
    let mut vol6 = 0.0f64;
    for f in &mesh.faces {
        if f.len() < 3 {
            continue;
        }
        let v0 = mesh.positions[f[0] as usize];
        for i in 1..f.len() - 1 {
            let v1 = mesh.positions[f[i] as usize];
            let v2 = mesh.positions[f[i + 1] as usize];
            let c = v1.cross(v2);
            vol6 += (v0.x as f64) * (c.x as f64)
                + (v0.y as f64) * (c.y as f64)
                + (v0.z as f64) * (c.z as f64);
        }
    }
    (vol6 / 6.0) as f32
}

fn ensure_outward_winding(mesh: &mut Mesh) {
    if signed_volume(mesh) < 0.0 {
        for f in &mut mesh.faces {
            f.reverse();
        }
    }
}

fn build_box(sx: f32, sy: f32, sz: f32) -> Mesh {
    let (hx, hy) = (sx * 0.5, sy * 0.5);
    let positions = vec![
        Vec3::new(-hx, -hy, 0.0),
        Vec3::new(hx, -hy, 0.0),
        Vec3::new(hx, hy, 0.0),
        Vec3::new(-hx, hy, 0.0),
        Vec3::new(-hx, -hy, sz),
        Vec3::new(hx, -hy, sz),
        Vec3::new(hx, hy, sz),
        Vec3::new(-hx, hy, sz),
    ];
    let faces = vec![
        vec![0, 3, 2, 1], // bottom (-Z)
        vec![4, 5, 6, 7], // top (+Z)
        vec![0, 1, 5, 4], // front (-Y)
        vec![1, 2, 6, 5], // right (+X)
        vec![2, 3, 7, 6], // back (+Y)
        vec![3, 0, 4, 7], // left (-X)
    ];
    Mesh { positions, faces, smooth_shading: false }
}

/// Revolves a 2D `(radius, z)` profile around the Z axis.
/// Points with `r <= 1e-5` collapse into single pole vertices; the first and last points are
/// automatically closed to poles if they have `r > 0`, ensuring a strictly watertight manifold.
fn build_lathe(profile: &[(f32, f32)], segs: usize, smooth_shading: bool) -> Mesh {
    let mut pts: Vec<(f32, f32)> = Vec::with_capacity(profile.len() + 2);
    if let Some(&(r0, z0)) = profile.first() {
        if r0 > 1e-4 {
            pts.push((0.0, z0));
        }
    }
    for &(r, z) in profile {
        let r = r.max(0.0);
        if let Some(&(pr, pz)) = pts.last() {
            if (pr - r).abs() < 1e-5 && (pz - z).abs() < 1e-5 {
                continue;
            }
        }
        pts.push((r, z));
    }
    if let Some((rl, zl)) = pts.last().copied() {
        if rl > 1e-4 {
            pts.push((0.0, zl));
        }
    }

    let mut positions: Vec<Vec3> = Vec::new();
    enum Ring {
        Pole(u32),
        Circle(u32),
    }
    let mut rings: Vec<Ring> = Vec::with_capacity(pts.len());
    for &(r, z) in &pts {
        if r <= 1e-4 {
            let idx = positions.len() as u32;
            positions.push(Vec3::new(0.0, 0.0, z));
            rings.push(Ring::Pole(idx));
        } else {
            let start = positions.len() as u32;
            for i in 0..segs {
                let a = (i as f32 / segs as f32) * TAU;
                positions.push(Vec3::new(r * a.cos(), r * a.sin(), z));
            }
            rings.push(Ring::Circle(start));
        }
    }

    let mut faces: Vec<Vec<u32>> = Vec::new();
    for w in rings.windows(2) {
        match (&w[0], &w[1]) {
            (Ring::Pole(p), Ring::Circle(c)) => {
                for i in 0..segs {
                    let j = (i + 1) % segs;
                    faces.push(vec![*p, c + j as u32, c + i as u32]);
                }
            }
            (Ring::Circle(c), Ring::Pole(p)) => {
                for i in 0..segs {
                    let j = (i + 1) % segs;
                    faces.push(vec![c + i as u32, c + j as u32, *p]);
                }
            }
            (Ring::Circle(a), Ring::Circle(b)) => {
                for i in 0..segs {
                    let j = (i + 1) % segs;
                    faces.push(vec![a + i as u32, a + j as u32, b + j as u32, b + i as u32]);
                }
            }
            (Ring::Pole(_), Ring::Pole(_)) => {}
        }
    }

    let mut m = Mesh { positions, faces, smooth_shading };
    ensure_outward_winding(&mut m);
    m
}

fn build_lathe_sphere(radius: f32, segs: usize, rings: usize) -> Mesh {
    let mut prof = Vec::with_capacity(rings + 1);
    for i in 0..=rings {
        let t = i as f32 / rings as f32;
        let phi = -PI * 0.5 + t * PI;
        let r = if i == 0 || i == rings { 0.0 } else { radius * phi.cos() };
        let z = radius + radius * phi.sin();
        prof.push((r, z));
    }
    build_lathe(&prof, segs, true)
}

fn build_torus(major_r: f32, minor_r: f32, major_segs: usize, minor_segs: usize) -> Mesh {
    let mut positions = Vec::with_capacity(major_segs * minor_segs);
    for i in 0..major_segs {
        let u = (i as f32 / major_segs as f32) * TAU;
        let (cu, su) = (u.cos(), u.sin());
        for j in 0..minor_segs {
            let v = (j as f32 / minor_segs as f32) * TAU;
            let r = major_r + minor_r * v.cos();
            positions.push(Vec3::new(r * cu, r * su, minor_r + minor_r * v.sin()));
        }
    }
    let mut faces = Vec::with_capacity(major_segs * minor_segs);
    for i in 0..major_segs {
        let i1 = (i + 1) % major_segs;
        for j in 0..minor_segs {
            let j1 = (j + 1) % minor_segs;
            let a = (i * minor_segs + j) as u32;
            let b = (i1 * minor_segs + j) as u32;
            let c = (i1 * minor_segs + j1) as u32;
            let d = (i * minor_segs + j1) as u32;
            faces.push(vec![a, b, c, d]);
        }
    }
    let mut m = Mesh { positions, faces, smooth_shading: true };
    ensure_outward_winding(&mut m);
    m
}

fn build_capsule(radius: f32, cyl_h: f32, segs: usize, hemi_rings: usize) -> Mesh {
    let mut prof = Vec::with_capacity(hemi_rings * 2 + 2);
    for i in 0..=hemi_rings {
        let phi = -PI * 0.5 + (i as f32 / hemi_rings as f32) * (PI * 0.5);
        let r = if i == 0 { 0.0 } else { radius * phi.cos() };
        let z = radius + radius * phi.sin();
        prof.push((r, z));
    }
    for i in 0..=hemi_rings {
        let phi = (i as f32 / hemi_rings as f32) * (PI * 0.5);
        let r = if i == hemi_rings { 0.0 } else { radius * phi.cos() };
        let z = radius + cyl_h + radius * phi.sin();
        prof.push((r, z));
    }
    build_lathe(&prof, segs, true)
}

fn build_pyramid(sx: f32, sy: f32, sz: f32) -> Mesh {
    let (hx, hy) = (sx * 0.5, sy * 0.5);
    let positions = vec![
        Vec3::new(-hx, -hy, 0.0),
        Vec3::new(hx, -hy, 0.0),
        Vec3::new(hx, hy, 0.0),
        Vec3::new(-hx, hy, 0.0),
        Vec3::new(0.0, 0.0, sz),
    ];
    let faces = vec![
        vec![0, 3, 2, 1],
        vec![0, 1, 4],
        vec![1, 2, 4],
        vec![2, 3, 4],
        vec![3, 0, 4],
    ];
    Mesh { positions, faces, smooth_shading: false }
}

fn build_regular_prism(sides: usize, radius: f32, height: f32, smooth: bool) -> Mesh {
    let mut pts = Vec::with_capacity(sides);
    for i in 0..sides {
        let a = (i as f32 / sides as f32) * TAU - PI * 0.5;
        pts.push(Vec2::new(radius * a.cos(), radius * a.sin()));
    }
    build_extruded_polygon_2d(&pts, height, 0.0, 1.0, smooth).unwrap()
}

/// Strictly watertight subdivided cube pushed onto a rounded-box SDF surface (pure quads, no poles).
fn build_rounded_box(sx: f32, sy: f32, sz: f32, radius: f32, sub: usize) -> Mesh {
    let n = sub.max(2);
    let half = Vec3::new(sx * 0.5, sy * 0.5, sz * 0.5);
    let r = radius.min(half.min_element() * 0.48);
    let inner = half - Vec3::splat(r);

    // Map quantized grid integer coordinates `(ix, iy, iz)` in `0..=n` on the cube boundary to unique vertex index.
    let mut index_map = std::collections::HashMap::<(usize, usize, usize), u32>::new();
    let mut positions = Vec::new();
    let mut get_vert = |ix: usize, iy: usize, iz: usize| -> u32 {
        if let Some(&idx) = index_map.get(&(ix, iy, iz)) {
            return idx;
        }
        let u = (ix as f32 / n as f32) * 2.0 - 1.0;
        let v = (iy as f32 / n as f32) * 2.0 - 1.0;
        let w = (iz as f32 / n as f32) * 2.0 - 1.0;
        let p = Vec3::new(u * half.x, v * half.y, w * half.z);
        let q = p.clamp(-inner, inner);
        let d = p - q;
        let dl = d.length();
        let pos = if dl > 1e-6 { q + (d / dl) * r } else { q };
        let idx = positions.len() as u32;
        positions.push(pos + Vec3::new(0.0, 0.0, half.z));
        index_map.insert((ix, iy, iz), idx);
        idx
    };

    let mut faces = Vec::with_capacity(6 * n * n);
    // +Z (iz = n)
    for iy in 0..n {
        for ix in 0..n {
            faces.push(vec![
                get_vert(ix, iy, n),
                get_vert(ix + 1, iy, n),
                get_vert(ix + 1, iy + 1, n),
                get_vert(ix, iy + 1, n),
            ]);
        }
    }
    // -Z (iz = 0)
    for iy in 0..n {
        for ix in 0..n {
            faces.push(vec![
                get_vert(ix, iy, 0),
                get_vert(ix, iy + 1, 0),
                get_vert(ix + 1, iy + 1, 0),
                get_vert(ix + 1, iy, 0),
            ]);
        }
    }
    // +Y (iy = n)
    for iz in 0..n {
        for ix in 0..n {
            faces.push(vec![
                get_vert(ix, n, iz),
                get_vert(ix, n, iz + 1),
                get_vert(ix + 1, n, iz + 1),
                get_vert(ix + 1, n, iz),
            ]);
        }
    }
    // -Y (iy = 0)
    for iz in 0..n {
        for ix in 0..n {
            faces.push(vec![
                get_vert(ix, 0, iz),
                get_vert(ix + 1, 0, iz),
                get_vert(ix + 1, 0, iz + 1),
                get_vert(ix, 0, iz + 1),
            ]);
        }
    }
    // +X (ix = n)
    for iz in 0..n {
        for iy in 0..n {
            faces.push(vec![
                get_vert(n, iy, iz),
                get_vert(n, iy + 1, iz),
                get_vert(n, iy + 1, iz + 1),
                get_vert(n, iy, iz + 1),
            ]);
        }
    }
    // -X (ix = 0)
    for iz in 0..n {
        for iy in 0..n {
            faces.push(vec![
                get_vert(0, iy, iz),
                get_vert(0, iy, iz + 1),
                get_vert(0, iy + 1, iz + 1),
                get_vert(0, iy + 1, iz),
            ]);
        }
    }

    let mut m = Mesh { positions, faces, smooth_shading: true };
    ensure_outward_winding(&mut m);
    m
}

fn build_egg(radius: f32, height: f32, segs: usize, rings: usize) -> Mesh {
    let mut prof = Vec::with_capacity(rings + 1);
    for i in 0..=rings {
        let t = i as f32 / rings as f32;
        let phi = t * PI;
        let z = height * 0.5 * (1.0 - phi.cos());
        // Classic ovoid taper: fuller near bottom, narrower at crown.
        let taper = 1.0 - 0.24 * (t - 0.35);
        let r = if i == 0 || i == rings { 0.0 } else { (radius * phi.sin() * taper).max(0.0) };
        prof.push((r, z));
    }
    build_lathe(&prof, segs, true)
}

fn build_teardrop(radius: f32, height: f32, segs: usize, rings: usize) -> Mesh {
    let mut prof = Vec::with_capacity(rings + 1);
    for i in 0..=rings {
        let t = i as f32 / rings as f32;
        let phi = t * PI;
        let z = height * 0.5 * (1.0 - phi.cos());
        let r = if i == 0 || i == rings {
            0.0
        } else {
            radius * phi.sin() * (0.5 * (1.0 + phi.cos())).powf(0.45) * 1.18
        };
        prof.push((r, z));
    }
    build_lathe(&prof, segs, true)
}

fn build_heart(size: f32, height: f32) -> Mesh {
    let n = 40usize;
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f32 / n as f32) * TAU;
        let x = 16.0 * t.sin().powi(3);
        let y = 13.0 * t.cos() - 5.0 * (2.0 * t).cos() - 2.0 * (3.0 * t).cos() - (4.0 * t).cos();
        pts.push(Vec2::new(x, y) * (size / 32.0));
    }
    // Enforce CCW & build nicely beveled heart.
    let pts = sanitize_simple_polygon(pts).unwrap();
    build_extruded_polygon_2d(&pts, height, 1.4, 1.0, true).unwrap()
}

fn build_star_prism(points: usize, outer_r: f32, inner_r: f32, height: f32) -> Mesh {
    let n = points * 2;
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i as f32 / n as f32) * TAU + PI * 0.5;
        let r = if i % 2 == 0 { outer_r } else { inner_r };
        pts.push(Vec2::new(r * a.cos(), r * a.sin()));
    }
    build_extruded_polygon_2d(&pts, height, 0.9, 1.0, false).unwrap()
}

fn build_pebble(rx: f32, ry: f32, rz: f32) -> Mesh {
    // Subdivided cube sphere (smooth organic river stone, zero poles, 100% manifold).
    let mut m = build_rounded_box(2.0, 2.0, 2.0, 0.99, 7);
    for p in &mut m.positions {
        let d = (*p - Vec3::new(0.0, 0.0, 1.0)).normalize_or_zero();
        let bump = 1.0
            + 0.08 * (d.x * 2.3 + d.y * 1.7).sin()
            + 0.06 * (d.y * 3.1 - d.z * 2.1).cos()
            + 0.04 * (d.z * 2.7 + d.x * 2.9).sin();
        p.x = d.x * rx * bump;
        p.y = d.y * ry * bump;
        p.z = d.z * rz * bump;
    }
    ground_and_center(&mut m);
    ensure_outward_winding(&mut m);
    m
}

fn build_dome(radius: f32, segs: usize, rings: usize) -> Mesh {
    let mut prof = Vec::with_capacity(rings + 2);
    prof.push((0.0, 0.0));
    for i in 0..=rings {
        let phi = (i as f32 / rings as f32) * (PI * 0.5);
        let r = if i == rings { 0.0 } else { radius * phi.cos() };
        let z = radius * phi.sin();
        prof.push((r, z));
    }
    build_lathe(&prof, segs, true)
}

fn build_tetrahedron(radius: f32) -> Mesh {
    let h = radius * 0.45;
    let r_base = radius * 0.94;
    let mut positions = Vec::with_capacity(4);
    for i in 0..3 {
        let a = (i as f32 / 3.0) * TAU;
        positions.push(Vec3::new(r_base * a.cos(), r_base * a.sin(), 0.0));
    }
    positions.push(Vec3::new(0.0, 0.0, radius + h));
    let faces = vec![
        vec![0, 2, 1],
        vec![0, 1, 3],
        vec![1, 2, 3],
        vec![2, 0, 3],
    ];
    let mut m = Mesh { positions, faces, smooth_shading: false };
    ensure_outward_winding(&mut m);
    m
}

fn build_octahedron(radius: f32, half_h: f32) -> Mesh {
    let positions = vec![
        Vec3::new(radius, 0.0, half_h),
        Vec3::new(0.0, radius, half_h),
        Vec3::new(-radius, 0.0, half_h),
        Vec3::new(0.0, -radius, half_h),
        Vec3::new(0.0, 0.0, half_h * 2.0),
        Vec3::new(0.0, 0.0, 0.0),
    ];
    let faces = vec![
        vec![0, 1, 4],
        vec![1, 2, 4],
        vec![2, 3, 4],
        vec![3, 0, 4],
        vec![1, 0, 5],
        vec![2, 1, 5],
        vec![3, 2, 5],
        vec![0, 3, 5],
    ];
    Mesh { positions, faces, smooth_shading: false }
}

fn build_icosahedron(radius: f32) -> Mesh {
    let phi = (1.0 + 5.0f32.sqrt()) * 0.5;
    let raw = [
        Vec3::new(-1.0, phi, 0.0), Vec3::new(1.0, phi, 0.0), Vec3::new(-1.0, -phi, 0.0), Vec3::new(1.0, -phi, 0.0),
        Vec3::new(0.0, -1.0, phi), Vec3::new(0.0, 1.0, phi), Vec3::new(0.0, -1.0, -phi), Vec3::new(0.0, 1.0, -phi),
        Vec3::new(phi, 0.0, -1.0), Vec3::new(phi, 0.0, 1.0), Vec3::new(-phi, 0.0, -1.0), Vec3::new(-phi, 0.0, 1.0),
    ];
    let positions: Vec<Vec3> = raw.iter().map(|v| v.normalize() * radius).collect();
    let faces = vec![
        vec![0, 11, 5], vec![0, 5, 1], vec![0, 1, 7], vec![0, 7, 10], vec![0, 10, 11],
        vec![1, 5, 9], vec![5, 11, 4], vec![11, 10, 2], vec![10, 7, 6], vec![7, 1, 8],
        vec![3, 9, 4], vec![3, 4, 2], vec![3, 2, 6], vec![3, 6, 8], vec![3, 8, 9],
        vec![4, 9, 5], vec![2, 4, 11], vec![6, 2, 10], vec![8, 6, 7], vec![9, 8, 1],
    ];
    let mut m = Mesh { positions, faces, smooth_shading: false };
    ground_and_center(&mut m);
    ensure_outward_winding(&mut m);
    m
}

/// Exact dual of the icosahedron: 20 vertices, 12 regular pentagonal faces (strictly watertight).
fn build_dodecahedron(radius: f32) -> Mesh {
    let ico = build_icosahedron(1.0);
    // Re-center ico at origin for exact dual.
    let mut ico_pos = ico.positions.clone();
    let c = ico_pos.iter().copied().sum::<Vec3>() / ico_pos.len() as f32;
    for p in &mut ico_pos {
        *p -= c;
    }
    // Dual vertices = normalized centroids of each of the 20 icosahedron faces.
    let positions: Vec<Vec3> = ico
        .faces
        .iter()
        .map(|f| {
            let m = (ico_pos[f[0] as usize] + ico_pos[f[1] as usize] + ico_pos[f[2] as usize]) / 3.0;
            m.normalize() * radius
        })
        .collect();

    // Dual faces = for each of the 12 icosahedron vertices, collect the 5 incident face indices sorted CCW around the vertex normal.
    let mut faces = Vec::with_capacity(12);
    for (vi, &vn) in ico_pos.iter().enumerate() {
        let normal = vn.normalize();
        let ref_axis = if normal.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
        let u = ref_axis.cross(normal).normalize();
        let v = normal.cross(u);
        let mut incident: Vec<(f32, u32)> = ico
            .faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f.contains(&(vi as u32)))
            .map(|(fi, _)| {
                let d = positions[fi];
                let ang = d.dot(v).atan2(d.dot(u));
                (ang, fi as u32)
            })
            .collect();
        incident.sort_by(|a, b| a.0.total_cmp(&b.0));
        faces.push(incident.into_iter().map(|(_, fi)| fi).collect());
    }

    let mut m = Mesh { positions, faces, smooth_shading: false };
    ground_and_center(&mut m);
    ensure_outward_winding(&mut m);
    m
}

fn build_gem(radius: f32, height: f32, facets: usize) -> Mesh {
    let prof = [
        (0.0, 0.0),
        (radius * 0.98, height * 0.62),
        (radius, height * 0.68),
        (radius * 0.66, height),
        (0.0, height),
    ];
    build_lathe(&prof, facets.max(6), false)
}

/// Extrudes a 2D polygon in the `(X, Z)` plane along `Y` (used for arch, roof_gable, l_bracket).
fn extrude_xz_along_y(pts_xz: &[Vec2], depth_y: f32) -> Mesh {
    let mut m = build_extruded_polygon_2d(pts_xz, depth_y, 0.0, 1.0, false).unwrap();
    // Map (x, y_2d, z_ext) -> (x, z_ext, y_2d) so original 2D Y becomes vertical Z!
    for p in &mut m.positions {
        let (x, z_vert, y_depth) = (p.x, p.y, p.z);
        *p = Vec3::new(x, y_depth, z_vert);
    }
    ground_and_center(&mut m);
    ensure_outward_winding(&mut m);
    m
}

/// Extrudes a 2D polygon in the `(Y, Z)` plane along `X` (used for wedge, stairs).
fn extrude_yz_along_x(pts_yz: &[Vec2], width_x: f32) -> Mesh {
    let mut m = build_extruded_polygon_2d(pts_yz, width_x, 0.0, 1.0, false).unwrap();
    // Map (y_2d, z_vert, x_ext) -> (x_ext, y_2d, z_vert)
    for p in &mut m.positions {
        let (y_horiz, z_vert, x_width) = (p.x, p.y, p.z);
        *p = Vec3::new(x_width, y_horiz, z_vert);
    }
    ground_and_center(&mut m);
    ensure_outward_winding(&mut m);
    m
}

fn build_arch(width: f32, depth: f32, height: f32, thickness: f32, arc_segs: usize) -> Mesh {
    let outer_r = width * 0.5;
    let inner_r = (outer_r - thickness).max(2.0);
    let spring_z = (height - outer_r).max(4.0);
    let mut pts = Vec::with_capacity(arc_segs * 2 + 6);
    // Outer contour (right base -> up -> semicircular arch -> left base)
    pts.push(Vec2::new(outer_r, 0.0));
    for i in 0..=arc_segs {
        let a = (i as f32 / arc_segs as f32) * PI;
        pts.push(Vec2::new(outer_r * a.cos(), spring_z + outer_r * a.sin()));
    }
    pts.push(Vec2::new(-outer_r, 0.0));
    // Inner contour (left base -> up -> inner arch reversed -> right base)
    pts.push(Vec2::new(-inner_r, 0.0));
    for i in (0..=arc_segs).rev() {
        let a = (i as f32 / arc_segs as f32) * PI;
        pts.push(Vec2::new(inner_r * a.cos(), spring_z + inner_r * a.sin()));
    }
    pts.push(Vec2::new(inner_r, 0.0));
    extrude_xz_along_y(&pts, depth)
}

fn build_wedge(width: f32, depth: f32, height: f32) -> Mesh {
    let pts = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(depth, 0.0),
        Vec2::new(depth, height),
    ];
    extrude_yz_along_x(&pts, width)
}

fn build_stairs(width: f32, depth: f32, height: f32, steps: usize) -> Mesh {
    let s = steps.max(2);
    let mut pts = Vec::with_capacity(s * 2 + 2);
    pts.push(Vec2::new(0.0, 0.0));
    pts.push(Vec2::new(depth, 0.0));
    for i in (0..s).rev() {
        let y_back = depth * ((i + 1) as f32 / s as f32);
        let y_front = depth * (i as f32 / s as f32);
        let z_top = height * ((i + 1) as f32 / s as f32);
        pts.push(Vec2::new(y_back, z_top));
        pts.push(Vec2::new(y_front, z_top));
    }
    extrude_yz_along_x(&pts, width)
}

fn build_column(radius: f32, height: f32, flutes: usize) -> Mesh {
    let segs = (flutes * 4).max(36);
    // Rings: base plinth -> torus molding -> fluted shaft -> neck -> capital -> abacus.
    let rings_spec: [(f32, f32, f32); 11] = [
        (0.0, 0.0, 0.0),
        (radius * 1.25, 0.0, 0.0),
        (radius * 1.25, height * 0.07, 0.0),
        (radius * 1.05, height * 0.11, 0.0),
        (radius * 0.96, height * 0.14, 0.08),
        (radius * 0.92, height * 0.50, 0.09),
        (radius * 0.86, height * 0.84, 0.08),
        (radius * 0.92, height * 0.88, 0.0),
        (radius * 1.18, height * 0.93, 0.0),
        (radius * 1.25, height, 0.0),
        (0.0, height, 0.0),
    ];

    let mut positions = Vec::new();
    enum Ring {
        Pole(u32),
        Circle(u32),
    }
    let mut rings = Vec::with_capacity(rings_spec.len());
    for &(r_base, z, flute_depth) in &rings_spec {
        if r_base <= 1e-4 {
            let idx = positions.len() as u32;
            positions.push(Vec3::new(0.0, 0.0, z));
            rings.push(Ring::Pole(idx));
        } else {
            let start = positions.len() as u32;
            for i in 0..segs {
                let a = (i as f32 / segs as f32) * TAU;
                let flute = if flute_depth > 0.0 {
                    1.0 - flute_depth * (0.5 - 0.5 * (a * flutes as f32).cos())
                } else {
                    1.0
                };
                let r = r_base * flute;
                positions.push(Vec3::new(r * a.cos(), r * a.sin(), z));
            }
            rings.push(Ring::Circle(start));
        }
    }
    let mut faces = Vec::new();
    for w in rings.windows(2) {
        match (&w[0], &w[1]) {
            (Ring::Pole(p), Ring::Circle(c)) => {
                for i in 0..segs {
                    let j = (i + 1) % segs;
                    faces.push(vec![*p, c + j as u32, c + i as u32]);
                }
            }
            (Ring::Circle(c), Ring::Pole(p)) => {
                for i in 0..segs {
                    let j = (i + 1) % segs;
                    faces.push(vec![c + i as u32, c + j as u32, *p]);
                }
            }
            (Ring::Circle(a), Ring::Circle(b)) => {
                for i in 0..segs {
                    let j = (i + 1) % segs;
                    faces.push(vec![a + i as u32, a + j as u32, b + j as u32, b + i as u32]);
                }
            }
            _ => {}
        }
    }
    let mut m = Mesh { positions, faces, smooth_shading: false };
    ensure_outward_winding(&mut m);
    m
}

fn build_roof_gable(width: f32, depth: f32, height: f32) -> Mesh {
    let hw = width * 0.5;
    let eave = height * 0.16;
    let pts = vec![
        Vec2::new(-hw, 0.0),
        Vec2::new(hw, 0.0),
        Vec2::new(hw, eave),
        Vec2::new(0.0, height),
        Vec2::new(-hw, eave),
    ];
    extrude_xz_along_y(&pts, depth)
}

/// Strictly watertight annular extrusion between equal-length CCW outer & inner 2D loops.
fn build_annular_extrusion_2d(
    outer: &[Vec2],
    inner: &[Vec2],
    height: f32,
    smooth_shading: bool,
) -> Mesh {
    let n = outer.len();
    assert_eq!(n, inner.len());
    let mut positions = Vec::with_capacity(n * 4);
    for &p in outer {
        positions.push(Vec3::new(p.x, p.y, 0.0));
    }
    for &p in outer {
        positions.push(Vec3::new(p.x, p.y, height));
    }
    for &p in inner {
        positions.push(Vec3::new(p.x, p.y, 0.0));
    }
    for &p in inner {
        positions.push(Vec3::new(p.x, p.y, height));
    }

    let o0 = 0u32;
    let o1 = n as u32;
    let i0 = (2 * n) as u32;
    let i1 = (3 * n) as u32;
    let mut faces = Vec::with_capacity(n * 4);
    for k in 0..n {
        let kn = (k + 1) % n;
        let (k, kn) = (k as u32, kn as u32);
        // Outer wall
        faces.push(vec![o0 + k, o0 + kn, o1 + kn, o1 + k]);
        // Inner wall (faces inward toward hole)
        faces.push(vec![i0 + kn, i0 + k, i1 + k, i1 + kn]);
        // Top cap (+Z)
        faces.push(vec![o1 + k, o1 + kn, i1 + kn, i1 + k]);
        // Bottom cap (-Z)
        faces.push(vec![i0 + k, i0 + kn, o0 + kn, o0 + k]);
    }
    let mut m = Mesh { positions, faces, smooth_shading };
    ensure_outward_winding(&mut m);
    m
}

fn build_wall_window(width: f32, depth: f32, height: f32, frame: f32) -> Mesh {
    let hw = width * 0.5;
    let outer = [
        Vec2::new(-hw, 0.0),
        Vec2::new(hw, 0.0),
        Vec2::new(hw, height),
        Vec2::new(-hw, height),
    ];
    let inner = [
        Vec2::new(-hw + frame, frame),
        Vec2::new(hw - frame, frame),
        Vec2::new(hw - frame, height - frame),
        Vec2::new(-hw + frame, height - frame),
    ];
    let mut m = build_annular_extrusion_2d(&outer, &inner, depth, false);
    for p in &mut m.positions {
        let (x, z_vert, y_depth) = (p.x, p.y, p.z);
        *p = Vec3::new(x, y_depth, z_vert);
    }
    ground_and_center(&mut m);
    ensure_outward_winding(&mut m);
    m
}

fn build_annular_circle(outer_r: f32, inner_r: f32, height: f32, segs: usize, smooth: bool) -> Mesh {
    let mut outer = Vec::with_capacity(segs);
    let mut inner = Vec::with_capacity(segs);
    for i in 0..segs {
        let a = (i as f32 / segs as f32) * TAU;
        let (ca, sa) = (a.cos(), a.sin());
        outer.push(Vec2::new(outer_r * ca, outer_r * sa));
        inner.push(Vec2::new(inner_r * ca, inner_r * sa));
    }
    build_annular_extrusion_2d(&outer, &inner, height, smooth)
}

fn build_gear(teeth: usize, outer_r: f32, root_r: f32, bore_r: f32, height: f32) -> Mesh {
    let n = teeth * 4;
    let mut outer = Vec::with_capacity(n);
    let mut inner = Vec::with_capacity(n);
    let step = TAU / teeth as f32;
    for t in 0..teeth {
        let base = t as f32 * step;
        let spec = [
            (base + step * 0.00, root_r),
            (base + step * 0.16, outer_r),
            (base + step * 0.44, outer_r),
            (base + step * 0.60, root_r),
        ];
        for (a, r) in spec {
            let (ca, sa) = (a.cos(), a.sin());
            outer.push(Vec2::new(r * ca, r * sa));
            inner.push(Vec2::new(bore_r * ca, bore_r * sa));
        }
    }
    build_annular_extrusion_2d(&outer, &inner, height, false)
}

fn build_hex_nut(outer_r: f32, bore_r: f32, height: f32) -> Mesh {
    // 24 outer vertices (4 subdivisions per hex edge) paired 1:1 with 24 circular bore vertices!
    let sub = 4usize;
    let n = 6 * sub;
    let mut outer = Vec::with_capacity(n);
    let mut inner = Vec::with_capacity(n);
    let corners: Vec<Vec2> = (0..6)
        .map(|i| {
            let a = (i as f32 / 6.0) * TAU;
            Vec2::new(outer_r * a.cos(), outer_r * a.sin())
        })
        .collect();
    for i in 0..6 {
        let a = corners[i];
        let b = corners[(i + 1) % 6];
        for s in 0..sub {
            let t = s as f32 / sub as f32;
            let p = a.lerp(b, t);
            outer.push(p);
            let ang = p.y.atan2(p.x);
            inner.push(Vec2::new(bore_r * ang.cos(), bore_r * ang.sin()));
        }
    }
    build_annular_extrusion_2d(&outer, &inner, height, false)
}

fn build_spring(
    coil_r: f32,
    wire_r: f32,
    turns: f32,
    height: f32,
    segs_per_turn: usize,
    tube_segs: usize,
) -> Mesh {
    let steps = ((turns * segs_per_turn as f32).round() as usize).max(24);
    let mut spine = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let a = t * turns * TAU;
        spine.push(Vec3::new(coil_r * a.cos(), coil_r * a.sin(), wire_r + t * height));
    }

    // Parallel-transport (rotation-minimizing) frames along helix + flat end caps.
    let mut tangents = Vec::with_capacity(spine.len());
    for i in 0..spine.len() {
        let d = if i == 0 {
            spine[1] - spine[0]
        } else if i + 1 == spine.len() {
            spine[i] - spine[i - 1]
        } else {
            spine[i + 1] - spine[i - 1]
        };
        tangents.push(d.normalize_or_zero());
    }

    let mut positions = Vec::with_capacity((spine.len()) * tube_segs + 2);
    let mut normal = Vec3::new(1.0, 0.0, 0.0);
    // Project initial normal orthogonal to tangents[0]
    normal = (normal - tangents[0] * normal.dot(tangents[0])).normalize_or_zero();
    if normal.length_squared() < 0.5 {
        normal = Vec3::Z.cross(tangents[0]).normalize();
    }

    for i in 0..spine.len() {
        let t = tangents[i];
        if i > 0 {
            let prev_t = tangents[i - 1];
            let axis = prev_t.cross(t);
            let sin_a = axis.length();
            if sin_a > 1e-6 {
                let cos_a = prev_t.dot(t).clamp(-1.0, 1.0);
                let q = glam::Quat::from_axis_angle(axis / sin_a, sin_a.atan2(cos_a));
                normal = (q * normal).normalize();
            }
            normal = (normal - t * normal.dot(t)).normalize();
        }
        let binormal = t.cross(normal).normalize();
        for j in 0..tube_segs {
            let a = (j as f32 / tube_segs as f32) * TAU;
            let offset = (normal * a.cos() + binormal * a.sin()) * wire_r;
            positions.push(spine[i] + offset);
        }
    }

    let start_cap = positions.len() as u32;
    positions.push(spine[0]);
    let end_cap = positions.len() as u32;
    positions.push(*spine.last().unwrap());

    let mut faces = Vec::with_capacity((spine.len() - 1) * tube_segs + tube_segs * 2);
    for i in 0..spine.len() - 1 {
        let r0 = (i * tube_segs) as u32;
        let r1 = ((i + 1) * tube_segs) as u32;
        for j in 0..tube_segs {
            let j1 = ((j + 1) % tube_segs) as u32;
            let j = j as u32;
            faces.push(vec![r0 + j, r0 + j1, r1 + j1, r1 + j]);
        }
    }
    // Start cap
    for j in 0..tube_segs {
        let j1 = ((j + 1) % tube_segs) as u32;
        faces.push(vec![start_cap, j1, j as u32]);
    }
    // End cap
    let last_ring = ((spine.len() - 1) * tube_segs) as u32;
    for j in 0..tube_segs {
        let j1 = ((j + 1) % tube_segs) as u32;
        faces.push(vec![end_cap, last_ring + j as u32, last_ring + j1]);
    }

    let mut m = Mesh { positions, faces, smooth_shading: true };
    ground_and_center(&mut m);
    ensure_outward_winding(&mut m);
    m
}

fn build_l_bracket(width: f32, depth: f32, height: f32, thickness: f32) -> Mesh {
    let pts = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(width, 0.0),
        Vec2::new(width, thickness),
        Vec2::new(thickness, thickness),
        Vec2::new(thickness, height),
        Vec2::new(0.0, height),
    ];
    extrude_xz_along_y(&pts, depth)
}

fn build_bowl(outer_r: f32, height: f32, wall: f32, segs: usize) -> Mesh {
    let steps = 10usize;
    let inner_r = (outer_r - wall).max(2.0);
    let mut prof = Vec::with_capacity(steps * 2 + 4);
    prof.push((0.0, 0.0));
    prof.push((outer_r * 0.45, 0.0)); // flat foot so it stands cleanly
    for i in 1..=steps {
        let t = i as f32 / steps as f32;
        let a = t * PI * 0.5;
        let r = outer_r * (0.45 + 0.55 * a.sin());
        let z = height * (1.0 - a.cos());
        prof.push((r, z));
    }
    // Rounded rim into interior
    prof.push(((outer_r + inner_r) * 0.5, height + wall * 0.2));
    for i in (1..=steps).rev() {
        let t = i as f32 / steps as f32;
        let a = t * PI * 0.5;
        let r = inner_r * (0.35 + 0.65 * a.sin());
        let z = wall + (height - wall) * (1.0 - a.cos());
        prof.push((r, z));
    }
    prof.push((0.0, wall));
    build_lathe(&prof, segs, true)
}

fn build_cup(top_r: f32, bot_r: f32, height: f32, wall: f32, segs: usize) -> Mesh {
    let prof = [
        (0.0, 0.0),
        (bot_r, 0.0),
        (top_r, height),
        (top_r - wall * 0.5, height + wall * 0.15),
        (top_r - wall, height),
        ((bot_r - wall).max(1.0), wall),
        (0.0, wall),
    ];
    build_lathe(&prof, segs, true)
}

fn build_funnel(top_r: f32, stem_r: f32, height: f32, wall: f32, segs: usize) -> Mesh {
    // True through-hole thickened funnel: annular closed cross-section revolved 360°!
    let neck_z = height * 0.42;
    let inner_stem = (stem_r - wall * 0.7).max(0.8);
    let ring_pts = [
        (stem_r, 0.0),
        (stem_r, neck_z),
        (top_r, height),
        (top_r - wall, height),
        (inner_stem, neck_z + wall * 0.35),
        (inner_stem, 0.0),
    ];
    let k = ring_pts.len();
    let mut positions = Vec::with_capacity(segs * k);
    for i in 0..segs {
        let a = (i as f32 / segs as f32) * TAU;
        let (ca, sa) = (a.cos(), a.sin());
        for &(r, z) in &ring_pts {
            positions.push(Vec3::new(r * ca, r * sa, z));
        }
    }
    let mut faces = Vec::with_capacity(segs * k);
    for i in 0..segs {
        let i1 = (i + 1) % segs;
        for j in 0..k {
            let j1 = (j + 1) % k;
            let a = (i * k + j) as u32;
            let b = (i1 * k + j) as u32;
            let c = (i1 * k + j1) as u32;
            let d = (i * k + j1) as u32;
            faces.push(vec![a, b, c, d]);
        }
    }
    let mut m = Mesh { positions, faces, smooth_shading: true };
    ensure_outward_winding(&mut m);
    m
}

fn build_tray(sx: f32, sy: f32, sz: f32, wall: f32, floor_t: f32) -> Mesh {
    // 8-point chamfered rectangle loop replicated across 4 concentric cross-section rings + center poles.
    let chamfer_rect = |hx: f32, hy: f32, ch: f32| -> [Vec2; 8] {
        let c = ch.min(hx * 0.4).min(hy * 0.4);
        [
            Vec2::new(hx - c, -hy),
            Vec2::new(hx, -hy + c),
            Vec2::new(hx, hy - c),
            Vec2::new(hx - c, hy),
            Vec2::new(-hx + c, hy),
            Vec2::new(-hx, hy - c),
            Vec2::new(-hx, -hy + c),
            Vec2::new(-hx + c, -hy),
        ]
    };
    let (hx, hy) = (sx * 0.5, sy * 0.5);
    let r_bot_out = chamfer_rect(hx - 0.8, hy - 0.8, 2.2);
    let r_top_out = chamfer_rect(hx, hy, 2.6);
    let r_top_in = chamfer_rect(hx - wall, hy - wall, 1.8);
    let r_bot_in = chamfer_rect(hx - wall - 0.6, hy - wall - 0.6, 1.4);

    let mut positions = Vec::with_capacity(2 + 4 * 8);
    positions.push(Vec3::new(0.0, 0.0, 0.0)); // 0: bottom center
    for p in r_bot_out { positions.push(Vec3::new(p.x, p.y, 0.0)); }
    for p in r_top_out { positions.push(Vec3::new(p.x, p.y, sz)); }
    for p in r_top_in { positions.push(Vec3::new(p.x, p.y, sz)); }
    for p in r_bot_in { positions.push(Vec3::new(p.x, p.y, floor_t)); }
    let inner_center = positions.len() as u32;
    positions.push(Vec3::new(0.0, 0.0, floor_t));

    let ring = |idx: u32, k: usize| 1 + idx * 8 + k as u32;
    let mut faces = Vec::with_capacity(8 * 5);
    for k in 0..8 {
        let kn = (k + 1) % 8;
        // Bottom fan
        faces.push(vec![0, ring(0, kn), ring(0, k)]);
        // 3 quad strips: outer wall, rim top, inner wall
        for r in 0..3u32 {
            faces.push(vec![ring(r, k), ring(r, kn), ring(r + 1, kn), ring(r + 1, k)]);
        }
        // Inner floor fan
        faces.push(vec![ring(3, k), ring(3, kn), inner_center]);
    }
    let mut m = Mesh { positions, faces, smooth_shading: false };
    ensure_outward_winding(&mut m);
    m
}

fn build_vase(max_r: f32, height: f32, wall: f32, segs: usize) -> Mesh {
    // Classic S-curve ceramic vase: solid foot, bulging belly, graceful neck, flared lip, hollow inside.
    let outer_pts: [(f32, f32); 11] = [
        (0.0, 0.0),
        (max_r * 0.56, 0.0),
        (max_r * 0.68, height * 0.14),
        (max_r * 0.96, height * 0.34),
        (max_r, height * 0.50),
        (max_r * 0.80, height * 0.66),
        (max_r * 0.54, height * 0.80),
        (max_r * 0.56, height * 0.90),
        (max_r * 0.72, height * 0.97),
        (max_r * 0.70, height),
        (max_r * 0.70 - wall * 0.5, height + wall * 0.12),
    ];
    let mut prof: Vec<(f32, f32)> = outer_pts.to_vec();
    for &(r, z) in outer_pts[2..10].iter().rev() {
        prof.push(((r - wall).max(1.5), z.max(wall + 0.5)));
    }
    prof.push((0.0, wall + 0.5));
    build_lathe(&prof, segs, true)
}

// ---------------------------------------------------------------------------
// 2D Polygon Cleanup, Ear-Clipping Triangulation & Beveled Extrusion
// ---------------------------------------------------------------------------

fn signed_area_2d(pts: &[Vec2]) -> f32 {
    let mut a = 0.0f32;
    for i in 0..pts.len() {
        let p = pts[i];
        let q = pts[(i + 1) % pts.len()];
        a += p.x * q.y - q.x * p.y;
    }
    a * 0.5
}

fn dedup_closed_ring(pts: &mut Vec<Vec2>, eps: f32) {
    let eps2 = eps * eps;
    let mut out: Vec<Vec2> = Vec::with_capacity(pts.len());
    for &p in pts.iter() {
        if let Some(&last) = out.last() {
            if last.distance_squared(p) < eps2 {
                continue;
            }
        }
        out.push(p);
    }
    while out.len() > 1 && out.first().unwrap().distance_squared(*out.last().unwrap()) < eps2 {
        out.pop();
    }
    *pts = out;
}

fn point_line_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 < 1e-12 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

fn dp_open(pts: &[Vec2], tol: f32, out: &mut Vec<Vec2>) {
    if pts.len() <= 2 {
        out.push(pts[0]);
        if pts.len() == 2 {
            out.push(pts[1]);
        }
        return;
    }
    let a = pts[0];
    let b = *pts.last().unwrap();
    let mut max_d = -1.0f32;
    let mut max_i = 0usize;
    for (i, &p) in pts.iter().enumerate().skip(1).take(pts.len() - 2) {
        let d = point_line_distance(p, a, b);
        if d > max_d {
            max_d = d;
            max_i = i;
        }
    }
    if max_d > tol {
        let mut left = Vec::new();
        dp_open(&pts[..=max_i], tol, &mut left);
        let mut right = Vec::new();
        dp_open(&pts[max_i..], tol, &mut right);
        out.extend_from_slice(&left[..left.len().saturating_sub(1)]);
        out.extend_from_slice(&right);
    } else {
        out.push(a);
        out.push(b);
    }
}

fn simplify_closed_polygon(pts: &[Vec2], tol: f32) -> Vec<Vec2> {
    if pts.len() <= 4 {
        return pts.to_vec();
    }
    // Split at the two farthest vertices so closed-loop simplification stays symmetric.
    let mut best_i = 0usize;
    let mut best_d = -1.0f32;
    for (i, &p) in pts.iter().enumerate().skip(1) {
        let d = p.distance_squared(pts[0]);
        if d > best_d {
            best_d = d;
            best_i = i;
        }
    }
    let mut half1 = Vec::new();
    dp_open(&pts[..=best_i], tol, &mut half1);
    let mut second = pts[best_i..].to_vec();
    second.push(pts[0]);
    let mut half2 = Vec::new();
    dp_open(&second, tol, &mut half2);

    let mut out = half1;
    if half2.len() > 2 {
        out.extend_from_slice(&half2[1..half2.len() - 1]);
    }
    if out.len() < 3 {
        pts.to_vec()
    } else {
        out
    }
}

fn chaikin_closed(pts: &[Vec2]) -> Vec<Vec2> {
    let n = pts.len();
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        out.push(a * 0.75 + b * 0.25);
        out.push(a * 0.25 + b * 0.75);
    }
    out
}

fn cross2(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn segments_strictly_intersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool {
    let d1 = cross2(a, b, c);
    let d2 = cross2(a, b, d);
    let d3 = cross2(c, d, a);
    let d4 = cross2(c, d, b);
    ((d1 > 1e-6 && d2 < -1e-6) || (d1 < -1e-6 && d2 > 1e-6))
        && ((d3 > 1e-6 && d4 < -1e-6) || (d3 < -1e-6 && d4 > 1e-6))
}

fn sanitize_simple_polygon(mut pts: Vec<Vec2>) -> Result<Vec<Vec2>, String> {
    dedup_closed_ring(&mut pts, 1e-3);
    // Untangle simple self-crossing segments via 2-opt loop reversal (keeps messy freehand sketches valid!).
    for _pass in 0..8 {
        let n = pts.len();
        if n < 4 {
            break;
        }
        let mut swapped = false;
        'outer: for i in 0..n {
            let i1 = (i + 1) % n;
            for j in (i + 2)..n {
                let j1 = (j + 1) % n;
                if j1 == i {
                    continue;
                }
                if segments_strictly_intersect(pts[i], pts[i1], pts[j], pts[j1]) {
                    pts[i1..=j].reverse();
                    swapped = true;
                    break 'outer;
                }
            }
        }
        if !swapped {
            break;
        }
    }

    // Remove nearly-collinear zero-area spikes.
    let mut changed = true;
    while changed && pts.len() > 3 {
        changed = false;
        let n = pts.len();
        for i in 0..n {
            let prev = pts[(i + n - 1) % n];
            let curr = pts[i];
            let next = pts[(i + 1) % n];
            let area2 = cross2(prev, curr, next).abs();
            let base = prev.distance(next);
            if area2 < 1e-4 * base.max(1e-3) {
                pts.remove(i);
                changed = true;
                break;
            }
        }
    }

    if pts.len() < 3 {
        return Err("outline collapsed to fewer than 3 non-collinear vertices".to_string());
    }
    let area = signed_area_2d(&pts);
    if area.abs() < 1e-3 {
        return Err("outline has near-zero area".to_string());
    }
    if area < 0.0 {
        pts.reverse();
    }
    Ok(pts)
}

fn point_in_tri_2d(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let c1 = cross2(a, b, p);
    let c2 = cross2(b, c, p);
    let c3 = cross2(c, a, p);
    c1 >= -1e-6 && c2 >= -1e-6 && c3 >= -1e-6
}

/// Ear-clipping triangulation of a simple CCW 2D polygon. Returns triangle index triples into `pts`.
fn triangulate_polygon_2d(pts: &[Vec2]) -> Result<Vec<[u32; 3]>, String> {
    let n = pts.len();
    if n < 3 {
        return Err("polygon needs at least 3 vertices".to_string());
    }
    if n == 3 {
        return Ok(vec![[0, 1, 2]]);
    }

    let mut rem: Vec<usize> = (0..n).collect();
    let mut tris = Vec::with_capacity(n - 2);
    let mut guard = n * n + 16;

    while rem.len() > 3 && guard > 0 {
        guard -= 1;
        let m = rem.len();
        let mut clipped = false;
        for i in 0..m {
            let a = rem[(i + m - 1) % m];
            let b = rem[i];
            let c = rem[(i + 1) % m];
            let (pa, pb, pc) = (pts[a], pts[b], pts[c]);
            if cross2(pa, pb, pc) <= 1e-7 {
                continue;
            }
            let mut contains_other = false;
            for &other in &rem {
                if other == a || other == b || other == c {
                    continue;
                }
                if point_in_tri_2d(pts[other], pa, pb, pc) {
                    contains_other = true;
                    break;
                }
            }
            if !contains_other {
                tris.push([a as u32, b as u32, c as u32]);
                rem.remove(i);
                clipped = true;
                break;
            }
        }
        if !clipped {
            // Fallback for near-degenerate ear: pick highest positive cross-product vertex.
            let mut best = 0usize;
            let mut best_c = f32::NEG_INFINITY;
            for i in 0..m {
                let a = rem[(i + m - 1) % m];
                let b = rem[i];
                let c = rem[(i + 1) % m];
                let cr = cross2(pts[a], pts[b], pts[c]);
                if cr > best_c {
                    best_c = cr;
                    best = i;
                }
            }
            let a = rem[(best + m - 1) % m];
            let b = rem[best];
            let c = rem[(best + 1) % m];
            tris.push([a as u32, b as u32, c as u32]);
            rem.remove(best);
        }
    }
    if rem.len() == 3 {
        tris.push([rem[0] as u32, rem[1] as u32, rem[2] as u32]);
    }
    Ok(tris)
}

fn inset_polygon_safe(pts: &[Vec2], amount: f32) -> Option<Vec<Vec2>> {
    if amount <= 1e-4 {
        return None;
    }
    let n = pts.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let prev = pts[(i + n - 1) % n];
        let curr = pts[i];
        let next = pts[(i + 1) % n];
        let e0 = (curr - prev).normalize_or_zero();
        let e1 = (next - curr).normalize_or_zero();
        // Inward normals for CCW polygon: (-dy, dx)
        let n0 = Vec2::new(-e0.y, e0.x);
        let n1 = Vec2::new(-e1.y, e1.x);
        let bis = n0 + n1;
        let bl = bis.length();
        let dir = if bl > 1e-4 {
            let unit = bis / bl;
            let denom = unit.dot(n0).max(0.45);
            unit * (amount / denom).min(amount * 2.0)
        } else {
            n0 * amount
        };
        out.push(curr + dir);
    }
    // Validate inset polygon preserved CCW positive area and has no self-intersections.
    if signed_area_2d(&out) <= signed_area_2d(pts) * 0.08 {
        return None;
    }
    for i in 0..n {
        let i1 = (i + 1) % n;
        for j in (i + 2)..n {
            let j1 = (j + 1) % n;
            if j1 == i {
                continue;
            }
            if segments_strictly_intersect(out[i], out[i1], out[j], out[j1]) {
                return None;
            }
        }
    }
    Some(out)
}

fn build_extruded_polygon_2d(
    pts_ccw: &[Vec2],
    height: f32,
    bevel: f32,
    taper: f32,
    smooth_shading: bool,
) -> Result<Mesh, String> {
    let n = pts_ccw.len();
    let tris_bot = triangulate_polygon_2d(pts_ccw)?;
    let centroid = pts_ccw.iter().copied().sum::<Vec2>() / n as f32;
    let top_pts: Vec<Vec2> = if (taper - 1.0).abs() > 1e-4 {
        pts_ccw.iter().map(|&p| centroid + (p - centroid) * taper).collect()
    } else {
        pts_ccw.to_vec()
    };

    // Safely clamp bevel to keep both caps simple & manifold.
    let min_edge = (0..n)
        .map(|i| pts_ccw[i].distance(pts_ccw[(i + 1) % n]))
        .fold(f32::INFINITY, f32::min);
    let max_bevel = (height * 0.32).min(min_edge * 0.45);
    let b_amt = bevel.min(max_bevel);
    let bot_inset = if b_amt > 1e-3 { inset_polygon_safe(pts_ccw, b_amt) } else { None };
    let top_inset = if b_amt > 1e-3 { inset_polygon_safe(&top_pts, b_amt * taper.min(1.0)) } else { None };
    let use_bevel = bot_inset.is_some() && top_inset.is_some();

    let mut rings: Vec<(Vec<Vec2>, f32)> = Vec::with_capacity(4);
    if use_bevel {
        let bi = bot_inset.unwrap();
        let ti = top_inset.unwrap();
        rings.push((bi, 0.0));
        rings.push((pts_ccw.to_vec(), b_amt));
        rings.push((top_pts, height - b_amt));
        rings.push((ti, height));
    } else {
        rings.push((pts_ccw.to_vec(), 0.0));
        rings.push((top_pts, height));
    }

    let mut positions = Vec::with_capacity(rings.len() * n);
    for (ring, z) in &rings {
        for &p in ring {
            positions.push(Vec3::new(p.x, p.y, *z));
        }
    }

    let mut faces = Vec::with_capacity((rings.len() - 1) * n + tris_bot.len() * 2);
    // Side walls between consecutive rings
    for r in 0..rings.len() - 1 {
        let a = (r * n) as u32;
        let b = ((r + 1) * n) as u32;
        for i in 0..n {
            let j = ((i + 1) % n) as u32;
            let i = i as u32;
            faces.push(vec![a + i, a + j, b + j, b + i]);
        }
    }

    // Bottom cap (CW for -Z normal)
    let bot_tris = if use_bevel {
        triangulate_polygon_2d(&rings[0].0).unwrap_or_else(|_| tris_bot.clone())
    } else {
        tris_bot.clone()
    };
    for [a, b, c] in bot_tris {
        faces.push(vec![a, c, b]);
    }

    // Top cap (CCW for +Z normal)
    let top_off = ((rings.len() - 1) * n) as u32;
    let top_tris = if use_bevel {
        triangulate_polygon_2d(&rings.last().unwrap().0).unwrap_or(tris_bot)
    } else {
        tris_bot
    };
    for [a, b, c] in top_tris {
        faces.push(vec![top_off + a, top_off + b, top_off + c]);
    }

    let mut m = Mesh { positions, faces, smooth_shading };
    ensure_outward_winding(&mut m);
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Verifies that a mesh is strictly a closed, oriented 2-manifold:
    /// every directed half-edge `(a, b)` appears once and its reverse `(b, a)` appears once.
    fn assert_watertight_manifold(label: &str, m: &Mesh) {
        assert!(!m.positions.is_empty(), "{label}: empty positions");
        assert!(!m.faces.is_empty(), "{label}: empty faces");
        for (i, p) in m.positions.iter().enumerate() {
            assert!(p.is_finite(), "{label}: non-finite position at {i}");
        }
        let mut edges: HashMap<(u32, u32), usize> = HashMap::new();
        for f in &m.faces {
            assert!(f.len() >= 3, "{label}: degenerate face");
            for i in 0..f.len() {
                let a = f[i];
                let b = f[(i + 1) % f.len()];
                assert_ne!(a, b, "{label}: zero-length edge");
                assert!((a as usize) < m.positions.len(), "{label}: index out of bounds");
                *edges.entry((a, b)).or_insert(0) += 1;
            }
        }
        for (&(a, b), &count) in &edges {
            assert_eq!(count, 1, "{label}: non-manifold half-edge ({a}->{b}) count={count}");
            assert_eq!(
                edges.get(&(b, a)).copied().unwrap_or(0),
                1,
                "{label}: open boundary or flipped twin for edge ({a}->{b})"
            );
        }
        let vol = signed_volume(m);
        assert!(vol > 1.0, "{label}: expected positive volume > 1 mm^3, got {vol}");

        let min_z = m.positions.iter().map(|p| p.z).fold(f32::INFINITY, f32::min);
        let max_z = m.positions.iter().map(|p| p.z).fold(f32::NEG_INFINITY, f32::max);
        assert!(min_z.abs() < 1e-3, "{label}: expected min.z == 0, got {min_z}");
        assert!(max_z > 1.0, "{label}: expected positive height, got {max_z}");
    }

    #[test]
    fn all_38_catalog_shapes_are_watertight_and_rest_on_ground() {
        assert_eq!(SHAPE_CATALOG.len(), 38);
        let cat: serde_json::Value = serde_json::from_str(&shape_catalog_json()).unwrap();
        assert_eq!(cat.as_array().unwrap().len(), 38);

        for entry in SHAPE_CATALOG {
            let (mesh, name, pos) = generate_shape_mesh(entry.id, "{}").unwrap_or_else(|e| {
                panic!("failed to generate '{}': {e}", entry.id);
            });
            assert_eq!(name, entry.name);
            assert_eq!(pos, Vec3::ZERO);
            assert_watertight_manifold(entry.id, &mesh);

            let cmd_json = generate_shape_command_json(entry.id, "{}").unwrap();
            let cmd: serde_json::Value = serde_json::from_str(&cmd_json).unwrap();
            assert!(cmd["addMesh"]["_0"]["positions"].is_array(), "{}: bad command JSON", entry.id);
        }
    }

    #[test]
    fn extrude_profile_concave_l_shape_and_bevel_taper_are_watertight() {
        // Concave L-shape placed via points
        let pts = [
            0.0, 0.0,
            24.0, 0.0,
            24.0, 8.0,
            8.0, 8.0,
            8.0, 22.0,
            0.0, 22.0,
        ];
        let (mesh, name, pos) = extrude_profile_2d_mesh(
            &pts,
            r#"{"mode":"points","height":14,"bevel":1.2,"taper":0.9,"name":"L-Plate"}"#,
        )
        .unwrap();
        assert_eq!(name, "L-Plate");
        assert!((pos.x - 12.0).abs() < 1e-3);
        assert!((pos.y - 11.0).abs() < 1e-3);
        assert_watertight_manifold("concave_l_bevel_taper", &mesh);
    }

    #[test]
    fn extrude_profile_freehand_noisy_loop_simplifies_smooths_and_is_watertight() {
        // Simulate a dense noisy freehand drawn star-like loop with 48 points
        let mut pts = Vec::with_capacity(96);
        for i in 0..48 {
            let a = (i as f32 / 48.0) * TAU;
            let r = 14.0 + 4.5 * (a * 3.0).sin() + 0.15 * (i as f32 * 7.0).cos();
            pts.push(10.0 + r * a.cos());
            pts.push(-5.0 + r * a.sin());
        }
        let (mesh, name, _) = extrude_profile_2d_mesh(
            &pts,
            r#"{"mode":"freehand","height":18,"bevel":1.0,"simplifyTolerance":0.5,"smoothPasses":1}"#,
        )
        .unwrap();
        assert_eq!(name, "Drawn Shape");
        assert_watertight_manifold("freehand_organic", &mesh);

        let cmd_json = extrude_profile_2d_command_json(&pts, r#"{"mode":"freehand","height":12}"#).unwrap();
        let cmd: serde_json::Value = serde_json::from_str(&cmd_json).unwrap();
        assert!(cmd["addMesh"].is_object());
    }

    #[test]
    fn extrude_profile_handles_negative_height_and_cw_input() {
        // Clockwise square + negative height
        let cw = [0.0, 0.0, 0.0, 15.0, 15.0, 15.0, 15.0, 0.0];
        let (mesh, _, pos) = extrude_profile_2d_mesh(&cw, r#"{"height":-10}"#).unwrap();
        assert!((pos.z + 10.0).abs() < 1e-3);
        assert_watertight_manifold("negative_height_cw", &mesh);
    }
}
