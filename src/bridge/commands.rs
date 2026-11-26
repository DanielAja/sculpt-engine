//! SceneCommand JSON executor, primitive generators, modifier stack, and SDF booleans.

use crate::bridge::{
    edit, evaluate_object_mesh, mesh_triangles, new_uuid, ok_envelope, scene_world_matrix,
    AppliedCommand, BoundingBox, BridgeError, CommandControl, EngineDoc, SceneSnapshot,
};
use crate::bvh::Bvh;
use crate::draw3d::{build_tube_mesh_with_pivot, Draw3DOptions, Draw3DPoint};
use crate::format::{Color, Material, Mesh, MeshSelection, Modifier, ModifierEntry, Scene, SceneObject, Transform};
use crate::mesh::SculptMesh;
use glam::{Mat4, Quat, Vec3};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::f32::consts::{PI, TAU};

pub struct CommandOutput {
    pub created: Vec<String>,
    pub new_elements: Option<MeshSelection>,
    pub label: String,
    pub affected: Vec<String>,
}

pub fn make_starter_cube() -> SceneObject {
    SceneObject {
        id: new_uuid(),
        name: "Cube".to_string(),
        mesh: make_primitive_mesh("cube", Vec3::splat(20.0), 1, 1),
        transform: Transform { position: Vec3::new(0.0, 0.0, 10.0), rotation: Quat::IDENTITY, scale: Vec3::ONE },
        material: Material::default(),
        modifiers: Vec::new(),
        parent: None,
        is_visible: true,
        is_locked: false,
        primitive_kind: Some("cube".to_string()),
    }
}

/// Builds a closed primitive centered at origin with extents `size` (mm, Z up).
pub fn make_primitive_mesh(kind: &str, size: Vec3, segments: usize, rings: usize) -> Mesh {
    let sx = size.x.max(0.01);
    let sy = size.y.max(0.01);
    let sz = size.z.max(0.0);
    let segs = segments.clamp(3, 128);
    let rngs = rings.clamp(2, 64);

    match kind {
        "cube" => {
            let (hx, hy, hz) = (sx * 0.5, sy * 0.5, sz.max(0.01) * 0.5);
            let positions = vec![
                Vec3::new(-hx, -hy, -hz),
                Vec3::new(hx, -hy, -hz),
                Vec3::new(hx, hy, -hz),
                Vec3::new(-hx, hy, -hz),
                Vec3::new(-hx, -hy, hz),
                Vec3::new(hx, -hy, hz),
                Vec3::new(hx, hy, hz),
                Vec3::new(-hx, hy, hz),
            ];
            let faces = vec![
                vec![0, 3, 2, 1], // -Z bottom
                vec![4, 5, 6, 7], // +Z top
                vec![0, 1, 5, 4], // -Y front
                vec![1, 2, 6, 5], // +X right
                vec![2, 3, 7, 6], // +Y back
                vec![3, 0, 4, 7], // -X left
            ];
            Mesh { positions, faces, smooth_shading: false }
        }
        "sphere" | "blob" => {
            let (rx, ry, rz) = (sx * 0.5, sy * 0.5, sz.max(0.01) * 0.5);
            let s = if kind == "blob" { 24 } else { segs };
            let r = if kind == "blob" { 16 } else { rngs };
            let mut positions = Vec::with_capacity(2 + (r - 1) * s);
            positions.push(Vec3::new(0.0, 0.0, rz));
            for j in 1..r {
                let phi = PI * (j as f32) / (r as f32);
                let sp = phi.sin();
                let cp = phi.cos();
                for i in 0..s {
                    let th = TAU * (i as f32) / (s as f32);
                    let bump = if kind == "blob" {
                        1.0 + 0.14 * (3.0 * th).sin() * sp + 0.1 * (2.0 * phi).cos()
                    } else {
                        1.0
                    };
                    positions.push(Vec3::new(rx * sp * th.cos() * bump, ry * sp * th.sin() * bump, rz * cp * bump));
                }
            }
            let south = positions.len() as u32;
            positions.push(Vec3::new(0.0, 0.0, -rz));
            let mut faces = Vec::new();
            // Top cap
            for i in 0..s {
                let next = (i + 1) % s;
                faces.push(vec![0, (1 + i) as u32, (1 + next) as u32]);
            }
            // Middle rings
            for j in 0..(r - 2) {
                let r0 = 1 + j * s;
                let r1 = 1 + (j + 1) * s;
                for i in 0..s {
                    let next = (i + 1) % s;
                    faces.push(vec![(r0 + i) as u32, (r1 + i) as u32, (r1 + next) as u32, (r0 + next) as u32]);
                }
            }
            // Bottom cap
            let last_ring = 1 + (r - 2) * s;
            for i in 0..s {
                let next = (i + 1) % s;
                faces.push(vec![south, (last_ring + next) as u32, (last_ring + i) as u32]);
            }
            Mesh { positions, faces, smooth_shading: true }
        }
        "cylinder" => {
            let (rx, ry, hz) = (sx * 0.5, sy * 0.5, sz.max(0.01) * 0.5);
            let mut positions = Vec::with_capacity(2 * segs + 2);
            for i in 0..segs {
                let th = TAU * (i as f32) / (segs as f32);
                positions.push(Vec3::new(rx * th.cos(), ry * th.sin(), -hz));
            }
            for i in 0..segs {
                let th = TAU * (i as f32) / (segs as f32);
                positions.push(Vec3::new(rx * th.cos(), ry * th.sin(), hz));
            }
            let bot_center = positions.len() as u32;
            positions.push(Vec3::new(0.0, 0.0, -hz));
            let top_center = positions.len() as u32;
            positions.push(Vec3::new(0.0, 0.0, hz));
            let mut faces = Vec::with_capacity(segs * 3);
            for i in 0..segs {
                let next = (i + 1) % segs;
                faces.push(vec![i as u32, next as u32, (segs + next) as u32, (segs + i) as u32]);
                faces.push(vec![bot_center, next as u32, i as u32]);
                faces.push(vec![top_center, (segs + i) as u32, (segs + next) as u32]);
            }
            Mesh { positions, faces, smooth_shading: false }
        }
        "cone" => {
            let (rx, ry, hz) = (sx * 0.5, sy * 0.5, sz.max(0.01) * 0.5);
            let mut positions = Vec::with_capacity(segs + 2);
            for i in 0..segs {
                let th = TAU * (i as f32) / (segs as f32);
                positions.push(Vec3::new(rx * th.cos(), ry * th.sin(), -hz));
            }
            let apex = positions.len() as u32;
            positions.push(Vec3::new(0.0, 0.0, hz));
            let bot_center = positions.len() as u32;
            positions.push(Vec3::new(0.0, 0.0, -hz));
            let mut faces = Vec::with_capacity(segs * 2);
            for i in 0..segs {
                let next = (i + 1) % segs;
                faces.push(vec![i as u32, next as u32, apex]);
                faces.push(vec![bot_center, next as u32, i as u32]);
            }
            Mesh { positions, faces, smooth_shading: false }
        }
        "torus" => {
            let minor = (sz.max(0.5) * 0.5).min(sx * 0.49);
            let major = (sx * 0.5 - minor).max(minor * 1.05);
            let scale_y = sy / sx.max(0.01);
            let mut positions = Vec::with_capacity(segs * rngs);
            for i in 0..segs {
                let u = TAU * (i as f32) / (segs as f32);
                let (cu, su) = (u.cos(), u.sin());
                for j in 0..rngs {
                    let v = TAU * (j as f32) / (rngs as f32);
                    let r = major + minor * v.cos();
                    positions.push(Vec3::new(r * cu, r * su * scale_y, minor * v.sin()));
                }
            }
            let mut faces = Vec::with_capacity(segs * rngs);
            for i in 0..segs {
                let ni = (i + 1) % segs;
                for j in 0..rngs {
                    let nj = (j + 1) % rngs;
                    let a = (i * rngs + j) as u32;
                    let b = (ni * rngs + j) as u32;
                    let c = (ni * rngs + nj) as u32;
                    let d = (i * rngs + nj) as u32;
                    faces.push(vec![a, b, c, d]);
                }
            }
            Mesh { positions, faces, smooth_shading: true }
        }
        "plane" => {
            let (hx, hy) = (sx * 0.5, sy * 0.5);
            Mesh {
                positions: vec![
                    Vec3::new(-hx, -hy, 0.0),
                    Vec3::new(hx, -hy, 0.0),
                    Vec3::new(hx, hy, 0.0),
                    Vec3::new(-hx, hy, 0.0),
                ],
                faces: vec![vec![0, 1, 2, 3]],
                smooth_shading: false,
            }
        }
        _ => make_primitive_mesh("cube", size, 1, 1),
    }
}

fn display_name_for_kind(kind: &str) -> &'static str {
    match kind {
        "cube" => "Cube",
        "sphere" => "Sphere",
        "cylinder" => "Cylinder",
        "cone" => "Cone",
        "torus" => "Torus",
        "plane" => "Plane",
        "blob" => "Blob",
        _ => "Shape",
    }
}

// ── JSON parsing helpers ──

fn parse_uuid(v: &Value) -> Option<String> {
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    v.get("rawValue").and_then(|x| x.as_str()).map(|s| s.to_string())
}

fn parse_uuid_list(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(parse_uuid).collect()).unwrap_or_default()
}

fn parse_vec3(v: &Value) -> Option<Vec3> {
    let a = v.as_array()?;
    if a.len() < 3 {
        return None;
    }
    let x = a[0].as_f64()? as f32;
    let y = a[1].as_f64()? as f32;
    let z = a[2].as_f64()? as f32;
    if x.is_finite() && y.is_finite() && z.is_finite() { Some(Vec3::new(x, y, z)) } else { None }
}

fn parse_quat(v: &Value) -> Option<Quat> {
    let a = v.as_array()?;
    if a.len() < 4 {
        return None;
    }
    let q = Quat::from_xyzw(
        a[0].as_f64()? as f32,
        a[1].as_f64()? as f32,
        a[2].as_f64()? as f32,
        a[3].as_f64()? as f32,
    );
    if q.is_finite() && q.length_squared() > 1e-12 { Some(q.normalize()) } else { Some(Quat::IDENTITY) }
}

fn parse_mat4(v: &Value) -> Option<Mat4> {
    let a = v.as_array()?;
    if a.len() == 16 {
        let mut arr = [0.0f32; 16];
        for (i, x) in a.iter().enumerate() {
            arr[i] = x.as_f64()? as f32;
        }
        return Some(Mat4::from_cols_array(&arr));
    }
    // Swift simd_float4x4 sometimes encodes as 4 column arrays [[x,y,z,w], ...]
    if a.len() == 4 {
        let mut arr = [0.0f32; 16];
        for (c, col) in a.iter().enumerate() {
            let ca = col.as_array()?;
            if ca.len() < 4 { return None; }
            for r in 0..4 {
                arr[c * 4 + r] = ca[r].as_f64()? as f32;
            }
        }
        return Some(Mat4::from_cols_array(&arr));
    }
    None
}

fn parse_transform(v: &Value) -> Option<Transform> {
    let pos = parse_vec3(v.get("position")?)?;
    let rot = parse_quat(v.get("rotation")?)?;
    let scale = parse_vec3(v.get("scale")?)?;
    Some(Transform { position: pos, rotation: rot, scale })
}

fn parse_material(v: &Value) -> Material {
    let name = v.get("name").and_then(|x| x.as_str()).unwrap_or("Clay").to_string();
    let bc = v.get("baseColor");
    let color = Color {
        r: bc.and_then(|c| c.get("r")).and_then(|x| x.as_f64()).unwrap_or(233.0 / 255.0) as f32,
        g: bc.and_then(|c| c.get("g")).and_then(|x| x.as_f64()).unwrap_or(201.0 / 255.0) as f32,
        b: bc.and_then(|c| c.get("b")).and_then(|x| x.as_f64()).unwrap_or(168.0 / 255.0) as f32,
        a: bc.and_then(|c| c.get("a")).and_then(|x| x.as_f64()).unwrap_or(1.0) as f32,
    };
    let metallic = v.get("metallic").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
    let roughness = v.get("roughness").and_then(|x| x.as_f64()).unwrap_or(0.6) as f32;
    Material {
        name,
        base_color: Color {
            r: color.r.clamp(0.0, 1.0),
            g: color.g.clamp(0.0, 1.0),
            b: color.b.clamp(0.0, 1.0),
            a: color.a.clamp(0.0, 1.0),
        },
        metallic: metallic.clamp(0.0, 1.0),
        roughness: roughness.clamp(0.0, 1.0),
    }
}

fn parse_mesh_value(v: &Value) -> Result<Mesh, BridgeError> {
    let smooth = v.get("smoothShading").and_then(|x| x.as_bool()).unwrap_or(false);
    let mut positions = Vec::new();
    if let Some(arr) = v.get("positions").and_then(|x| x.as_array()) {
        if arr.first().is_some_and(|x| x.is_array()) {
            for pt in arr {
                if let Some(p) = parse_vec3(pt) {
                    positions.push(p);
                }
            }
        } else {
            for chunk in arr.chunks_exact(3) {
                let x = chunk[0].as_f64().unwrap_or(0.0) as f32;
                let y = chunk[1].as_f64().unwrap_or(0.0) as f32;
                let z = chunk[2].as_f64().unwrap_or(0.0) as f32;
                positions.push(Vec3::new(x, y, z));
            }
        }
    }
    let n = positions.len() as u32;
    let mut faces = Vec::new();
    if let Some(farr) = v.get("faces").and_then(|x| x.as_array()) {
        for fv in farr {
            let idx_arr = fv.as_array().or_else(|| fv.get("indices").and_then(|x| x.as_array()));
            if let Some(ia) = idx_arr {
                let idx: Vec<u32> = ia.iter().filter_map(|x| x.as_u64().map(|u| u as u32)).collect();
                if idx.len() >= 3 && idx.iter().all(|&i| i < n) {
                    faces.push(idx);
                }
            }
        }
    }
    Ok(Mesh { positions, faces, smooth_shading: smooth })
}

fn parse_modifier(v: &Value) -> Result<Modifier, BridgeError> {
    let obj = v.as_object().ok_or_else(|| BridgeError::new("engine.badJSON"))?;
    let (case, body) = obj.iter().next().ok_or_else(|| BridgeError::new("engine.badJSON"))?;
    let s = body.get("_0").unwrap_or(body);
    match case.as_str() {
        "mirror" => Ok(Modifier::Mirror {
            axis: s.get("axis").and_then(|x| x.as_i64()).unwrap_or(0).clamp(0, 2),
            merge: s.get("merge").and_then(|x| x.as_bool()).unwrap_or(true),
            merge_distance: (s.get("mergeDistance").and_then(|x| x.as_f64()).unwrap_or(0.01) as f32).clamp(0.0, 10.0),
        }),
        "subdivision" => Ok(Modifier::Subdivision {
            levels: s.get("levels").and_then(|x| x.as_i64()).unwrap_or(1).clamp(1, 3),
        }),
        "array" => Ok(Modifier::Array {
            count: s.get("count").and_then(|x| x.as_i64()).unwrap_or(2).clamp(1, 250),
            relative_offset: s.get("relativeOffset").and_then(parse_vec3).unwrap_or(Vec3::new(1.1, 0.0, 0.0)),
            constant_offset: s.get("constantOffset").and_then(parse_vec3).unwrap_or(Vec3::ZERO),
            use_relative: s.get("useRelative").and_then(|x| x.as_bool()).unwrap_or(true),
        }),
        "solidify" => Ok(Modifier::Solidify {
            thickness: (s.get("thickness").and_then(|x| x.as_f64()).unwrap_or(2.0) as f32).clamp(-1000.0, 1000.0),
            offset: (s.get("offset").and_then(|x| x.as_f64()).unwrap_or(-1.0) as f32).clamp(-1.0, 1.0),
        }),
        _ => Err(BridgeError::with_args("engine.badJSON", vec![case.clone()])),
    }
}

fn parse_mesh_selection(v: &Value) -> MeshSelection {
    let vertices = v
        .get("vertices")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
        .unwrap_or_default();
    let faces = v
        .get("faces")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
        .unwrap_or_default();
    let edges = v
        .get("edges")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|e| {
                    let a = e.get("a").and_then(|x| x.as_i64())?;
                    let b = e.get("b").and_then(|x| x.as_i64())?;
                    Some(if a < b { (a, b) } else { (b, a) })
                })
                .collect()
        })
        .unwrap_or_default();
    MeshSelection { vertices, edges, faces }
}

// ── Command execution on Scene ──

pub fn apply_command_json(
    scene: &mut Scene,
    command_json: &str,
    control: Option<&CommandControl>,
) -> Result<CommandOutput, BridgeError> {
    let val: Value = serde_json::from_str(command_json)
        .map_err(|e| BridgeError::with_args("engine.badJSON", vec![e.to_string()]))?;
    apply_command_value(scene, &val, control)
}

pub fn apply_command_value(
    scene: &mut Scene,
    val: &Value,
    control: Option<&CommandControl>,
) -> Result<CommandOutput, BridgeError> {
    let obj = val.as_object().ok_or_else(|| BridgeError::new("engine.badJSON"))?;
    let (case, body) = obj.iter().next().ok_or_else(|| BridgeError::new("engine.badJSON"))?;

    match case.as_str() {
        "addPrimitive" => {
            let spec = body.get("_0").ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let kind = spec.get("kind").and_then(|x| x.as_str()).unwrap_or("cube");
            let size = spec.get("size").and_then(parse_vec3).unwrap_or(Vec3::splat(20.0));
            let segments = spec.get("segments").and_then(|x| x.as_u64()).unwrap_or(32) as usize;
            let rings = spec.get("rings").and_then(|x| x.as_u64()).unwrap_or(16) as usize;
            let mesh = make_primitive_mesh(kind, size, segments, rings);
            let pos = body
                .get("position")
                .and_then(parse_vec3)
                .unwrap_or_else(|| Vec3::new(0.0, 0.0, size.z * 0.5));
            let base_name = body
                .get("name")
                .and_then(|x| x.as_str())
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| display_name_for_kind(kind));
            let name = unique_name(scene, base_name);
            let material = body.get("material").map(parse_material).unwrap_or_default();
            let id = new_uuid();
            scene.objects.push(SceneObject {
                id: id.clone(),
                name,
                mesh,
                transform: Transform { position: pos, rotation: Quat::IDENTITY, scale: Vec3::ONE },
                material,
                modifiers: Vec::new(),
                parent: None,
                is_visible: true,
                is_locked: false,
                primitive_kind: Some(kind.to_string()),
            });
            Ok(CommandOutput {
                created: vec![id.clone()],
                new_elements: None,
                label: format!("Add {}", display_name_for_kind(kind)),
                affected: vec![id],
            })
        }

        "addMesh" => {
            let mesh_val = body.get("_0").ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let mesh = parse_mesh_value(mesh_val)?;
            let pos = body.get("position").and_then(parse_vec3).unwrap_or(Vec3::ZERO);
            let base_name = body
                .get("name")
                .and_then(|x| x.as_str())
                .filter(|s| !s.trim().is_empty())
                .unwrap_or("Mesh");
            let name = unique_name(scene, base_name);
            let material = body.get("material").map(parse_material).unwrap_or_default();
            let id = new_uuid();
            scene.objects.push(SceneObject {
                id: id.clone(),
                name: name.clone(),
                mesh,
                transform: Transform { position: pos, rotation: Quat::IDENTITY, scale: Vec3::ONE },
                material,
                modifiers: Vec::new(),
                parent: None,
                is_visible: true,
                is_locked: false,
                primitive_kind: None,
            });
            Ok(CommandOutput {
                created: vec![id.clone()],
                new_elements: None,
                label: format!("Add {name}"),
                affected: vec![id],
            })
        }

        "setMesh" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let mesh_val = body.get("_1").ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let mesh = parse_mesh_value(mesh_val)?;
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            o.mesh = mesh;
            o.primitive_kind = None;
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Edit Mesh".into(), affected: vec![id] })
        }

        "deleteObjects" => {
            let ids: HashSet<String> = body.get("_0").map(parse_uuid_list).unwrap_or_default().into_iter().collect();
            if ids.is_empty() {
                return Err(BridgeError::new("command.invalidSelection"));
            }
            for o in &mut scene.objects {
                if o.parent.as_ref().is_some_and(|p| ids.contains(p)) {
                    o.parent = None;
                }
            }
            let before = scene.objects.len();
            scene.objects.retain(|o| !ids.contains(&o.id));
            if scene.objects.len() == before {
                return Err(BridgeError::new("command.objectNotFound"));
            }
            Ok(CommandOutput {
                created: Vec::new(),
                new_elements: None,
                label: "Delete".into(),
                affected: ids.into_iter().collect(),
            })
        }

        "duplicateObjects" => {
            let ids = body.get("_0").map(parse_uuid_list).unwrap_or_default();
            let offset = body.get("offset").and_then(parse_vec3).unwrap_or(Vec3::new(20.0, 0.0, 0.0));
            let mut created = Vec::new();
            let mut clones = Vec::new();
            for id in &ids {
                if let Some(src) = scene.objects.iter().find(|o| &o.id == id) {
                    let mut copy = src.clone();
                    copy.id = new_uuid();
                    copy.name = unique_name(scene, &format!("{} Copy", src.name));
                    copy.transform.position += offset;
                    copy.parent = None;
                    created.push(copy.id.clone());
                    clones.push(copy);
                }
            }
            if clones.is_empty() {
                return Err(BridgeError::new("command.objectNotFound"));
            }
            scene.objects.extend(clones);
            Ok(CommandOutput { created: created.clone(), new_elements: None, label: "Duplicate".into(), affected: created })
        }

        "translate" => {
            let ids = body.get("_0").map(parse_uuid_list).unwrap_or_default();
            let delta = body.get("by").and_then(parse_vec3).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            for id in &ids {
                let parent_mat = scene
                    .objects
                    .iter()
                    .find(|o| &o.id == id)
                    .and_then(|o| o.parent.as_deref())
                    .map(|pid| scene_world_matrix(scene, pid))
                    .unwrap_or(Mat4::IDENTITY);
                let local_delta = parent_mat.inverse().transform_vector3(delta);
                if let Some(o) = scene.objects.iter_mut().find(|o| &o.id == id) {
                    o.transform.position += local_delta;
                }
            }
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Move".into(), affected: ids })
        }

        "rotate" => {
            let ids = body.get("_0").map(parse_uuid_list).unwrap_or_default();
            let q = body.get("by").and_then(parse_quat).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let pivot = body.get("pivot").and_then(parse_vec3).unwrap_or(Vec3::ZERO);
            let rot_mat = Mat4::from_translation(pivot) * Mat4::from_quat(q) * Mat4::from_translation(-pivot);
            for id in &ids {
                apply_world_matrix_to_object(scene, id, rot_mat);
            }
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Rotate".into(), affected: ids })
        }

        "scale" => {
            let ids = body.get("_0").map(parse_uuid_list).unwrap_or_default();
            let factor = body.get("by").and_then(parse_vec3).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let pivot = body.get("pivot").and_then(parse_vec3).unwrap_or(Vec3::ZERO);
            for id in &ids {
                let world = scene_world_matrix(scene, id);
                let world_pos = world.w_axis.truncate();
                let new_world_pos = pivot + (world_pos - pivot) * factor;
                let parent_mat = scene
                    .objects
                    .iter()
                    .find(|o| &o.id == id)
                    .and_then(|o| o.parent.as_deref())
                    .map(|pid| scene_world_matrix(scene, pid))
                    .unwrap_or(Mat4::IDENTITY);
                let new_local_pos = parent_mat.inverse().transform_point3(new_world_pos);
                if let Some(o) = scene.objects.iter_mut().find(|o| &o.id == id) {
                    o.transform.position = new_local_pos;
                    o.transform.scale *= factor;
                }
            }
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Scale".into(), affected: ids })
        }

        "setTransform" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let tr = body.get("_1").and_then(parse_transform).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            o.transform = tr;
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Transform".into(), affected: vec![id] })
        }

        "setDimensions" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let target = body.get("_1").and_then(parse_vec3).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let world = scene_world_matrix(scene, &id);
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            let eval = evaluate_object_mesh(o);
            let cur = BoundingBox::from_mesh(&eval, world).size();
            let mut ns = o.transform.scale;
            if cur.x > 1e-5 && target.x > 0.0 { ns.x *= target.x / cur.x; }
            if cur.y > 1e-5 && target.y > 0.0 { ns.y *= target.y / cur.y; }
            if cur.z > 1e-5 && target.z > 0.0 { ns.z *= target.z / cur.z; }
            o.transform.scale = ns;
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Resize".into(), affected: vec![id] })
        }

        "applyTransform" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            let m = Mat4::from_scale_rotation_translation(o.transform.scale, o.transform.rotation, Vec3::ZERO);
            let flip = m.determinant() < 0.0;
            for p in &mut o.mesh.positions {
                *p = m.transform_point3(*p);
            }
            if flip {
                for f in &mut o.mesh.faces {
                    f.reverse();
                }
            }
            o.transform.rotation = Quat::IDENTITY;
            o.transform.scale = Vec3::ONE;
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Apply Transform".into(), affected: vec![id] })
        }

        "renameObject" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let name = body.get("_1").and_then(|x| x.as_str()).unwrap_or("").trim().to_string();
            if name.is_empty() {
                return Err(BridgeError::with_args("command.invalidParameter", vec!["Name cannot be empty".into()]));
            }
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            o.name = name;
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Rename".into(), affected: vec![id] })
        }

        "setVisibility" => {
            let ids = body.get("_0").map(parse_uuid_list).unwrap_or_default();
            let vis = body.get("_1").and_then(|x| x.as_bool()).unwrap_or(true);
            for id in &ids {
                if let Some(o) = scene.objects.iter_mut().find(|o| &o.id == id) {
                    o.is_visible = vis;
                }
            }
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: if vis { "Show" } else { "Hide" }.into(), affected: ids })
        }

        "setSmoothShading" => {
            let ids = body.get("_0").map(parse_uuid_list).unwrap_or_default();
            let smooth = body.get("_1").and_then(|x| x.as_bool()).unwrap_or(true);
            for id in &ids {
                if let Some(o) = scene.objects.iter_mut().find(|o| &o.id == id) {
                    o.mesh.smooth_shading = smooth;
                }
            }
            Ok(CommandOutput {
                created: Vec::new(),
                new_elements: None,
                label: if smooth { "Shade Smooth" } else { "Shade Flat" }.into(),
                affected: ids,
            })
        }

        "setLocked" => {
            let ids = body.get("_0").map(parse_uuid_list).unwrap_or_default();
            let locked = body.get("_1").and_then(|x| x.as_bool()).unwrap_or(true);
            for id in &ids {
                if let Some(o) = scene.objects.iter_mut().find(|o| &o.id == id) {
                    o.is_locked = locked;
                }
            }
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: if locked { "Lock" } else { "Unlock" }.into(), affected: ids })
        }

        "setParent" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let parent = body.get("parent").and_then(parse_uuid);
            if let Some(ref pid) = parent {
                // Cycle detection
                let mut cur = Some(pid.as_str());
                let mut steps = 0usize;
                while let Some(c) = cur {
                    if c == id || steps > scene.objects.len() {
                        return Err(BridgeError::new("command.cycle"));
                    }
                    cur = scene.objects.iter().find(|o| o.id == c).and_then(|o| o.parent.as_deref());
                    steps += 1;
                }
            }
            let world = scene_world_matrix(scene, &id);
            let parent_world = parent.as_deref().map(|p| scene_world_matrix(scene, p)).unwrap_or(Mat4::IDENTITY);
            let local = parent_world.inverse() * world;
            let (scale, rotation, position) = local.to_scale_rotation_translation();
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            o.parent = parent;
            o.transform = Transform { position, rotation, scale };
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Reparent".into(), affected: vec![id] })
        }

        "reorder" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let to = body.get("toIndex").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let from = scene.objects.iter().position(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            let item = scene.objects.remove(from);
            let idx = to.min(scene.objects.len());
            scene.objects.insert(idx, item);
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Reorder".into(), affected: vec![id] })
        }

        "joinObjects" => {
            let ids = body.get("_0").map(parse_uuid_list).unwrap_or_default();
            if ids.len() < 2 {
                return Err(BridgeError::new("command.invalidSelection"));
            }
            let target_id = ids[0].clone();
            let target_world = scene_world_matrix(scene, &target_id);
            let inv_target = target_world.inverse();
            let mut merged_pos = Vec::new();
            let mut merged_faces = Vec::new();
            let mut smooth = false;
            for (k, id) in ids.iter().enumerate() {
                let o = scene.objects.iter().find(|o| &o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
                if k == 0 { smooth = o.mesh.smooth_shading; }
                let eval = evaluate_object_mesh(o);
                let rel = inv_target * scene_world_matrix(scene, id);
                let base = merged_pos.len() as u32;
                let flip = rel.determinant() < 0.0;
                for &p in &eval.positions {
                    merged_pos.push(rel.transform_point3(p));
                }
                for f in &eval.faces {
                    let mut nf: Vec<u32> = f.iter().map(|&i| i + base).collect();
                    if flip { nf.reverse(); }
                    merged_faces.push(nf);
                }
            }
            let remove_set: HashSet<&str> = ids[1..].iter().map(|s| s.as_str()).collect();
            if let Some(t) = scene.objects.iter_mut().find(|o| o.id == target_id) {
                t.mesh = Mesh { positions: merged_pos, faces: merged_faces, smooth_shading: smooth };
                t.modifiers.clear();
                t.primitive_kind = None;
            }
            scene.objects.retain(|o| !remove_set.contains(o.id.as_str()));
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Join".into(), affected: vec![target_id] })
        }

        "setMaterial" => {
            let ids = body.get("_0").map(parse_uuid_list).unwrap_or_default();
            let mat = body.get("_1").map(parse_material).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            for id in &ids {
                if let Some(o) = scene.objects.iter_mut().find(|o| &o.id == id) {
                    o.material = mat.clone();
                }
            }
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Set Material".into(), affected: ids })
        }

        "addModifier" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let m = body.get("_1").map(parse_modifier).transpose()?.ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            o.modifiers.push(ModifierEntry { id: new_uuid(), modifier: m, is_enabled: true });
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Add Modifier".into(), affected: vec![id] })
        }

        "updateModifier" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let eid = body.get("entryID").and_then(|x| x.as_str()).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let m = body.get("_2").map(parse_modifier).transpose()?.ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            if let Some(e) = o.modifiers.iter_mut().find(|e| e.id == eid) {
                e.modifier = m;
            }
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Update Modifier".into(), affected: vec![id] })
        }

        "setModifierEnabled" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let eid = body.get("entryID").and_then(|x| x.as_str()).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let enabled = body.get("_2").and_then(|x| x.as_bool()).unwrap_or(true);
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            if let Some(e) = o.modifiers.iter_mut().find(|e| e.id == eid) {
                e.is_enabled = enabled;
            }
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Toggle Modifier".into(), affected: vec![id] })
        }

        "removeModifier" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let eid = body.get("entryID").and_then(|x| x.as_str()).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            o.modifiers.retain(|e| e.id != eid);
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Remove Modifier".into(), affected: vec![id] })
        }

        "moveModifier" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let eid = body.get("entryID").and_then(|x| x.as_str()).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let to = body.get("toIndex").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            if let Some(from) = o.modifiers.iter().position(|e| e.id == eid) {
                let item = o.modifiers.remove(from);
                let idx = to.min(o.modifiers.len());
                o.modifiers.insert(idx, item);
            }
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Reorder Modifier".into(), affected: vec![id] })
        }

        "applyModifiers" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            o.mesh = evaluate_object_mesh(o);
            o.modifiers.clear();
            o.primitive_kind = None;
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Apply Modifiers".into(), affected: vec![id] })
        }

        "boolean" => {
            let op = body.get("_0").and_then(|x| x.as_str()).unwrap_or("union");
            let target_id = body.get("target").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let tool_id = body.get("tool").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let delete_tool = body.get("deleteTool").and_then(|x| x.as_bool()).unwrap_or(false);
            if target_id == tool_id {
                return Err(BridgeError::new("command.invalidSelection"));
            }
            let target_obj = scene.objects.iter().find(|o| o.id == target_id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?.clone();
            let tool_obj = scene.objects.iter().find(|o| o.id == tool_id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?.clone();
            let target_w = scene_world_matrix(scene, &target_id);
            let tool_w = scene_world_matrix(scene, &tool_id);
            let mesh_a = transform_mesh(&evaluate_object_mesh(&target_obj), target_w);
            let mesh_b = transform_mesh(&evaluate_object_mesh(&tool_obj), tool_w);

            let res_world = sdf_boolean(&mesh_a, &mesh_b, op, control)?;
            if res_world.faces.is_empty() {
                return Err(BridgeError::new("command.boolean.emptyResult"));
            }
            let res_local = transform_mesh(&res_world, target_w.inverse());
            if let Some(t) = scene.objects.iter_mut().find(|o| o.id == target_id) {
                t.mesh = res_local;
                t.modifiers.clear();
                t.primitive_kind = None;
            }
            if delete_tool {
                scene.objects.retain(|o| o.id != tool_id);
            } else if let Some(tool) = scene.objects.iter_mut().find(|o| o.id == tool_id) {
                tool.is_visible = false;
            }
            let label = match op {
                "difference" => "Subtract",
                "intersect" => "Intersect",
                _ => "Union",
            };
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: label.into(), affected: vec![target_id] })
        }

        "addTube3D" => {
            let spec = body.get("_0").ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let pts_val = spec.get("points").and_then(|x| x.as_array()).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let pts: Vec<Draw3DPoint> = pts_val
                .iter()
                .filter_map(|p| {
                    let pos = p.get("position").and_then(parse_vec3)?;
                    let radius = p.get("radius").and_then(|r| r.as_f64()).unwrap_or(2.5) as f32;
                    Some(Draw3DPoint { pos, radius })
                })
                .collect();
            let (opts, default_name) = Draw3DOptions::from_json(&spec.to_string());
            let (mesh, origin) = build_tube_mesh_with_pivot(&pts, &opts)
                .map_err(|e| BridgeError::with_args("command.invalidParameter", vec![e]))?;
            let base_name = body.get("name").and_then(|x| x.as_str()).unwrap_or(&default_name);
            let name = unique_name(scene, base_name);
            let id = new_uuid();
            scene.objects.push(SceneObject {
                id: id.clone(),
                name,
                mesh,
                transform: Transform { position: origin, rotation: Quat::IDENTITY, scale: Vec3::ONE },
                material: Material::default(),
                modifiers: Vec::new(),
                parent: None,
                is_visible: true,
                is_locked: false,
                primitive_kind: None,
            });
            Ok(CommandOutput { created: vec![id.clone()], new_elements: None, label: "Draw 3D".into(), affected: vec![id] })
        }

        "voxelRemesh" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let res = body.get("resolution").and_then(|x| x.as_u64()).unwrap_or(48) as u32;
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            let sm = SculptMesh::from_mesh(&o.mesh);
            let bvh = Bvh::build(&sm.positions, &sm.tris);
            let remeshed = crate::remesh::voxel_remesh(&sm, &bvh, res)
                .map_err(|e| BridgeError::with_args("command.invalidParameter", vec![e]))?;
            o.mesh = remeshed;
            o.primitive_kind = None;
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Voxel Remesh".into(), affected: vec![id] })
        }

        "sculptStroke" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let pts: Vec<Vec3> = body
                .get("points")
                .and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(parse_vec3).collect())
                .unwrap_or_default();
            let params = body.get("params");
            let kind_str = params.and_then(|p| p.get("kind")).and_then(|x| x.as_str()).unwrap_or("draw");
            let radius = params.and_then(|p| p.get("radiusMm")).and_then(|x| x.as_f64()).unwrap_or(6.0) as f32;
            let strength = params.and_then(|p| p.get("strength")).and_then(|x| x.as_f64()).unwrap_or(0.5) as f32;
            let sym_x = params.and_then(|p| p.get("symmetryX")).and_then(|x| x.as_bool()).unwrap_or(false);
            let falloff_str = params.and_then(|p| p.get("falloff")).and_then(|x| x.as_str()).unwrap_or("smooth");
            let sym_y = params.and_then(|p| p.get("symmetryY")).and_then(|x| x.as_bool()).unwrap_or(false);
            let sym_z = params.and_then(|p| p.get("symmetryZ")).and_then(|x| x.as_bool()).unwrap_or(false);
            let radial = params.and_then(|p| p.get("radialSymmetry")).and_then(|x| x.as_u64()).unwrap_or(1) as u32;
            let world = scene_world_matrix(scene, &id);
            let inv = world.inverse();
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            let mut sm = SculptMesh::from_mesh(&o.mesh);
            let mut bvh = Bvh::build(&sm.positions, &sm.tris);
            let mut scratch = crate::brush::Scratch::new(sm.positions.len());
            let mut moved = Vec::new();
            let mut dirty = Vec::new();
            let brush = crate::brush::BrushParams {
                kind: crate::brush::BrushKind::parse(kind_str),
                radius: radius.max(0.5),
                strength,
                symmetry: [sym_x, sym_y, sym_z],
                falloff: crate::brush::Falloff::parse(falloff_str),
                radial_symmetry: radial.clamp(1, 16),
            };
            for p in pts {
                let lp = inv.transform_point3(p);
                moved.clear();
                dirty.clear();
                let _ = crate::brush::dab_and_update_normals(&mut sm, &bvh, &mut scratch, lp, &brush, &mut moved, &mut dirty);
                if !moved.is_empty() {
                    bvh.refit(&sm.positions, &sm.tris);
                }
            }
            o.mesh = sm.to_mesh();
            o.primitive_kind = None;
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Sculpt".into(), affected: vec![id] })
        }

        // ── Edit-mode commands ──
        "moveElements" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let sel = body.get("_1").map(parse_mesh_selection).unwrap_or_default();
            let delta = body.get("by").and_then(parse_vec3).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            for v in edit::expanded_vertices(&sel, &o.mesh) {
                if let Some(p) = o.mesh.positions.get_mut(v) {
                    *p += delta;
                }
            }
            o.primitive_kind = None;
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Move Elements".into(), affected: vec![id] })
        }

        "transformElements" => {
            let id = body.get("_0").and_then(parse_uuid).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let sel = body.get("_1").map(parse_mesh_selection).unwrap_or_default();
            let mat = body.get("by").and_then(parse_mat4).ok_or_else(|| BridgeError::new("engine.badJSON"))?;
            let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
            for v in edit::expanded_vertices(&sel, &o.mesh) {
                if let Some(p) = o.mesh.positions.get_mut(v) {
                    *p = mat.transform_point3(*p);
                }
            }
            o.primitive_kind = None;
            Ok(CommandOutput { created: Vec::new(), new_elements: None, label: "Transform Elements".into(), affected: vec![id] })
        }

        "extrude" | "inset" | "bevel" | "loopCut" | "subdivide" | "mergeVertices" | "deleteElements"
        | "flipNormals" | "recalculateNormals" => edit::apply_edit_command(scene, case, body),

        _ => Err(BridgeError::with_args("engine.badJSON", vec![case.clone()])),
    }
}

fn apply_world_matrix_to_object(scene: &mut Scene, id: &str, world_delta: Mat4) {
    let world = scene_world_matrix(scene, id);
    let parent_world = scene
        .objects
        .iter()
        .find(|o| o.id == id)
        .and_then(|o| o.parent.as_deref())
        .map(|pid| scene_world_matrix(scene, pid))
        .unwrap_or(Mat4::IDENTITY);
    let new_local = parent_world.inverse() * world_delta * world;
    let (scale, rotation, position) = new_local.to_scale_rotation_translation();
    if let Some(o) = scene.objects.iter_mut().find(|o| o.id == id) {
        if position.is_finite() && rotation.is_finite() && scale.is_finite() {
            o.transform = Transform { position, rotation: rotation.normalize(), scale };
        }
    }
}

fn unique_name(scene: &Scene, base: &str) -> String {
    let existing: HashSet<&str> = scene.objects.iter().map(|o| o.name.as_str()).collect();
    if !existing.contains(base) {
        return base.to_string();
    }
    for n in 2..1000 {
        let cand = format!("{base} {n}");
        if !existing.contains(cand.as_str()) {
            return cand;
        }
    }
    base.to_string()
}

pub fn transform_mesh(mesh: &Mesh, mat: Mat4) -> Mesh {
    let flip = mat.determinant() < 0.0;
    let positions = mesh.positions.iter().map(|&p| mat.transform_point3(p)).collect();
    let faces = if flip {
        mesh.faces.iter().map(|f| f.iter().rev().copied().collect()).collect()
    } else {
        mesh.faces.clone()
    };
    Mesh { positions, faces, smooth_shading: mesh.smooth_shading }
}

// ── Document-level apply / applyGroup / preview / off-main ──

pub fn doc_apply(doc: &mut EngineDoc, command_json: &str) -> String {
    let mut next = doc.scene.clone();
    match apply_command_json(&mut next, command_json, None) {
        Ok(out) => {
            doc.record_scene_change(out.label, next, &out.created, out.new_elements);
            let mut m = serde_json::Map::new();
            m.insert("created".into(), json!(out.created));
            ok_envelope(m)
        }
        Err(e) => e.to_envelope(),
    }
}

pub fn doc_apply_group(doc: &mut EngineDoc, json_str: &str) -> String {
    let val: Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(e) => return BridgeError::with_args("engine.badJSON", vec![e.to_string()]).to_envelope(),
    };
    if let Some(r) = val.get("ifRevision").and_then(|x| x.as_i64()) {
        if r as i32 != doc.revision {
            return BridgeError::new("engine.changedMeanwhile").to_envelope();
        }
    }
    let label = val.get("label").and_then(|x| x.as_str()).unwrap_or("Edit").to_string();
    let cmds = val.get("commands").and_then(|x| x.as_array()).cloned().unwrap_or_default();
    if cmds.is_empty() {
        let mut m = serde_json::Map::new();
        m.insert("created".into(), json!([]));
        return ok_envelope(m);
    }
    doc.begin_undo_group(label);
    let mut created = Vec::new();
    for c in &cmds {
        let mut next = doc.scene.clone();
        match apply_command_value(&mut next, c, None) {
            Ok(out) => {
                created.extend(out.created.clone());
                doc.record_scene_change(out.label, next, &out.created, out.new_elements);
            }
            Err(e) => {
                doc.end_undo_group();
                return e.to_envelope();
            }
        }
    }
    doc.end_undo_group();
    let mut m = serde_json::Map::new();
    m.insert("created".into(), json!(created));
    ok_envelope(m)
}

pub fn doc_preview(doc: &mut EngineDoc, command_json: &str) -> String {
    let mut scratch = doc.scene.clone();
    match apply_command_json(&mut scratch, command_json, None) {
        Ok(out) => {
            doc.preview_transforms.clear();
            doc.preview_meshes.clear();
            for id in out.affected {
                let before = doc.scene.objects.iter().find(|o| o.id == id);
                let after = scratch.objects.iter().find(|o| o.id == id);
                if let (Some(b), Some(a)) = (before, after) {
                    if a.transform != b.transform {
                        doc.preview_transforms.insert(id.clone(), a.transform);
                    }
                    if a.mesh != b.mesh {
                        doc.preview_meshes.insert(id, a.mesh.clone());
                    }
                }
            }
            crate::bridge::ok_empty()
        }
        Err(e) => e.to_envelope(),
    }
}

pub fn apply_off_main(
    snapshot: &SceneSnapshot,
    command_json: &str,
    control: Option<&CommandControl>,
) -> Result<AppliedCommand, BridgeError> {
    let mut scene = snapshot.scene.clone();
    let out = apply_command_json(&mut scene, command_json, control)?;
    Ok(AppliedCommand {
        scene,
        created: out.created,
        new_elements: out.new_elements,
        label: out.label,
        start_revision: snapshot.revision,
    })
}

// ── Modifiers: Mirror, Subdivision, Array, Solidify ──

pub fn apply_modifier(mesh: &Mesh, modifier: &Modifier) -> Mesh {
    match modifier {
        Modifier::Mirror { axis, .. } => {
            let ax = (*axis).clamp(0, 2) as usize;
            let mut positions = mesh.positions.clone();
            let base = positions.len() as u32;
            for &p in &mesh.positions {
                let mut m = p;
                match ax {
                    0 => m.x = -m.x,
                    1 => m.y = -m.y,
                    _ => m.z = -m.z,
                }
                positions.push(m);
            }
            let mut faces = mesh.faces.clone();
            for f in &mesh.faces {
                faces.push(f.iter().rev().map(|&i| i + base).collect());
            }
            Mesh { positions, faces, smooth_shading: mesh.smooth_shading }
        }
        Modifier::Subdivision { levels } => {
            let mut cur = mesh.clone();
            for _ in 0..(*levels).clamp(1, 3) {
                cur = catmull_clark_step(&cur);
            }
            cur
        }
        Modifier::Array { count, relative_offset, constant_offset, use_relative } => {
            let c = (*count).clamp(1, 250) as usize;
            if c <= 1 {
                return mesh.clone();
            }
            let b = BoundingBox::from_mesh(mesh, Mat4::IDENTITY);
            let sz = b.size();
            let step = if *use_relative { sz * *relative_offset + *constant_offset } else { *constant_offset };
            let mut positions = Vec::with_capacity(mesh.positions.len() * c);
            let mut faces = Vec::with_capacity(mesh.faces.len() * c);
            for k in 0..c {
                let off = step * (k as f32);
                let base = positions.len() as u32;
                for &p in &mesh.positions {
                    positions.push(p + off);
                }
                for f in &mesh.faces {
                    faces.push(f.iter().map(|&i| i + base).collect());
                }
            }
            Mesh { positions, faces, smooth_shading: mesh.smooth_shading }
        }
        Modifier::Solidify { thickness, offset } => {
            if thickness.abs() < 1e-5 {
                return mesh.clone();
            }
            let normals = crate::bridge::mesh_vertex_normals(mesh);
            let n = mesh.positions.len();
            let d_outer = thickness * (1.0 + *offset) * 0.5;
            let d_inner = thickness * (*offset - 1.0) * 0.5;
            let mut positions = Vec::with_capacity(n * 2);
            for i in 0..n {
                positions.push(mesh.positions[i] + normals[i] * d_outer);
            }
            for i in 0..n {
                positions.push(mesh.positions[i] + normals[i] * d_inner);
            }
            let base = n as u32;
            let mut faces = Vec::with_capacity(mesh.faces.len() * 2);
            let mut edge_count: HashMap<(u32, u32), usize> = HashMap::new();
            let mut directed: Vec<(u32, u32)> = Vec::new();
            for f in &mesh.faces {
                faces.push(f.clone());
                faces.push(f.iter().rev().map(|&i| i + base).collect());
                for i in 0..f.len() {
                    let a = f[i];
                    let b = f[(i + 1) % f.len()];
                    let key = if a < b { (a, b) } else { (b, a) };
                    *edge_count.entry(key).or_insert(0) += 1;
                    directed.push((a, b));
                }
            }
            // Bridge boundary edges
            for (a, b) in directed {
                let key = if a < b { (a, b) } else { (b, a) };
                if edge_count.get(&key).copied().unwrap_or(0) == 1 {
                    faces.push(vec![a, a + base, b + base, b]);
                }
            }
            Mesh { positions, faces, smooth_shading: mesh.smooth_shading }
        }
    }
}

pub fn catmull_clark_step(mesh: &Mesh) -> Mesh {
    let nv = mesh.positions.len();
    let nf = mesh.faces.len();
    if nv == 0 || nf == 0 {
        return mesh.clone();
    }
    let mut face_pts = Vec::with_capacity(nf);
    let mut vert_faces: Vec<Vec<usize>> = vec![Vec::new(); nv];
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (fi, f) in mesh.faces.iter().enumerate() {
        let mut c = Vec3::ZERO;
        for &vi in f {
            c += mesh.positions[vi as usize];
            vert_faces[vi as usize].push(fi);
        }
        face_pts.push(c / (f.len().max(1) as f32));
        for i in 0..f.len() {
            let a = f[i];
            let b = f[(i + 1) % f.len()];
            let key = if a < b { (a, b) } else { (b, a) };
            edge_faces.entry(key).or_default().push(fi);
        }
    }

    let mut positions = Vec::with_capacity(nv + nf + edge_faces.len());
    // 0..nv: updated original vertices
    let mut vert_edges: Vec<Vec<(u32, u32)>> = vec![Vec::new(); nv];
    for &key in edge_faces.keys() {
        vert_edges[key.0 as usize].push(key);
        vert_edges[key.1 as usize].push(key);
    }
    for v in 0..nv {
        let p = mesh.positions[v];
        let boundary: Vec<(u32, u32)> = vert_edges[v]
            .iter()
            .copied()
            .filter(|k| edge_faces[k].len() == 1)
            .collect();
        if !boundary.is_empty() {
            let mut sum = p * 6.0;
            for &(a, b) in &boundary {
                let other = if a as usize == v { b } else { a };
                sum += mesh.positions[other as usize];
            }
            positions.push(sum / (6.0 + boundary.len() as f32));
        } else {
            let n = vert_faces[v].len() as f32;
            if n < 3.0 {
                positions.push(p);
            } else {
                let f_avg = vert_faces[v].iter().map(|&fi| face_pts[fi]).sum::<Vec3>() / n;
                let r_avg = vert_edges[v]
                    .iter()
                    .map(|&(a, b)| (mesh.positions[a as usize] + mesh.positions[b as usize]) * 0.5)
                    .sum::<Vec3>()
                    / (vert_edges[v].len().max(1) as f32);
                positions.push((f_avg + 2.0 * r_avg + (n - 3.0) * p) / n);
            }
        }
    }
    // nv..nv+nf: face points
    let face_base = positions.len() as u32;
    positions.extend_from_slice(&face_pts);
    // edge points
    let mut edge_idx: HashMap<(u32, u32), u32> = HashMap::with_capacity(edge_faces.len());
    for (&(a, b), fs) in &edge_faces {
        let idx = positions.len() as u32;
        let mid = (mesh.positions[a as usize] + mesh.positions[b as usize]) * 0.5;
        let ep = if fs.len() == 2 {
            (mesh.positions[a as usize] + mesh.positions[b as usize] + face_pts[fs[0]] + face_pts[fs[1]]) * 0.25
        } else {
            mid
        };
        positions.push(ep);
        edge_idx.insert((a, b), idx);
    }

    let mut faces = Vec::with_capacity(nf * 4);
    for (fi, f) in mesh.faces.iter().enumerate() {
        let m = f.len();
        let fp = face_base + fi as u32;
        for i in 0..m {
            let v = f[i];
            let prev = f[(i + m - 1) % m];
            let next = f[(i + 1) % m];
            let e_prev = edge_idx[&if prev < v { (prev, v) } else { (v, prev) }];
            let e_next = edge_idx[&if v < next { (v, next) } else { (next, v) }];
            faces.push(vec![v, e_next, fp, e_prev]);
        }
    }
    Mesh { positions, faces, smooth_shading: mesh.smooth_shading }
}

// ── Watertight SDF Boolean (Union / Difference / Intersect) ──

/// Fast, strictly watertight SDF boolean using BVH inside/outside parity + Surface Nets.
pub fn sdf_boolean(
    mesh_a: &Mesh,
    mesh_b: &Mesh,
    op: &str,
    control: Option<&CommandControl>,
) -> Result<Mesh, BridgeError> {
    if control.is_some_and(|c| c.is_cancelled()) {
        return Err(BridgeError::new("engine.cancelled"));
    }
    let sm_a = SculptMesh::from_mesh(mesh_a);
    let sm_b = SculptMesh::from_mesh(mesh_b);
    if sm_a.tris.is_empty() || sm_b.tris.is_empty() {
        return Err(BridgeError::new("command.boolean.emptyResult"));
    }
    let bvh_a = Bvh::build(&sm_a.positions, &sm_a.tris);
    let bvh_b = Bvh::build(&sm_b.positions, &sm_b.tris);

    let ba = BoundingBox::from_mesh(mesh_a, Mat4::IDENTITY);
    let bb = BoundingBox::from_mesh(mesh_b, Mat4::IDENTITY);
    let bounds = match op {
        "difference" => ba,
        "intersect" => BoundingBox { min: ba.min.max(bb.min), max: ba.max.min(bb.max) },
        _ => ba.union(bb),
    };
    if bounds.is_empty() || bounds.max_dimension() < 1e-4 {
        return Err(BridgeError::new("command.boolean.emptyResult"));
    }

    let size = bounds.size();
    let longest = size.max_element();
    let res = 64usize;
    let mut h = longest / (res as f32);
    let make_grid = |cell: f32| {
        let pad = 2.0 * cell;
        let origin = bounds.min - Vec3::splat(pad);
        let span = size + Vec3::splat(2.0 * pad);
        let n = [
            ((span.x / cell).ceil() as usize + 1).clamp(5, 96),
            ((span.y / cell).ceil() as usize + 1).clamp(5, 96),
            ((span.z / cell).ceil() as usize + 1).clamp(5, 96),
        ];
        (origin, n)
    };
    let (mut origin, mut n) = make_grid(h);
    while n[0] * n[1] * n[2] > 320_000 {
        h *= 1.1;
        let g = make_grid(h);
        origin = g.0;
        n = g.1;
    }
    let [nx, ny, nz] = n;
    let total = nx * ny * nz;
    if let Some(c) = control { c.set_progress(0.15); }

    let inside_a = scanline_inside(&sm_a, &bvh_a, origin, n, h);
    if control.is_some_and(|c| c.is_cancelled()) {
        return Err(BridgeError::new("engine.cancelled"));
    }
    if let Some(c) = control { c.set_progress(0.45); }
    let inside_b = scanline_inside(&sm_b, &bvh_b, origin, n, h);
    if control.is_some_and(|c| c.is_cancelled()) {
        return Err(BridgeError::new("engine.cancelled"));
    }
    if let Some(c) = control { c.set_progress(0.65); }

    let mut inside = vec![false; total];
    let mut any_in = false;
    for i in 0..total {
        let v = match op {
            "difference" => inside_a[i] && !inside_b[i],
            "intersect" => inside_a[i] && inside_b[i],
            _ => inside_a[i] || inside_b[i],
        };
        inside[i] = v;
        any_in |= v;
    }
    if !any_in {
        return Err(BridgeError::new("command.boolean.emptyResult"));
    }

    // Surface band exact SDF for crisp edges via Surface Nets
    let idx = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    let mut active = vec![false; total];
    for k in 0..(nz - 1) {
        for j in 0..(ny - 1) {
            for i in 0..(nx - 1) {
                let c0 = inside[idx(i, j, k)];
                let mut diff = false;
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            if inside[idx(i + dx, j + dy, k + dz)] != c0 {
                                diff = true;
                            }
                        }
                    }
                }
                if diff {
                    for dz in 0..2 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                active[idx(i + dx, j + dy, k + dz)] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut phi = vec![0.0f32; total];
    for k in 0..nz {
        let z = origin.z + k as f32 * h;
        for j in 0..ny {
            let y = origin.y + j as f32 * h;
            for i in 0..nx {
                let id = idx(i, j, k);
                if !active[id] {
                    phi[id] = if inside[id] { -h * 2.0 } else { h * 2.0 };
                    continue;
                }
                let p = Vec3::new(origin.x + i as f32 * h, y, z);
                let da = bvh_closest_dist(&bvh_a, &sm_a.positions, &sm_a.tris, p, h * 4.0);
                let db = bvh_closest_dist(&bvh_b, &sm_b.positions, &sm_b.tris, p, h * 4.0);
                let sa = if inside_a[id] { -da } else { da };
                let sb = if inside_b[id] { -db } else { db };
                phi[id] = match op {
                    "difference" => sa.max(-sb),
                    "intersect" => sa.max(sb),
                    _ => sa.min(sb),
                };
            }
        }
    }
    if let Some(c) = control { c.set_progress(0.85); }

    let mesh = surface_nets_extract(&phi, origin, n, h);
    if let Some(c) = control { c.set_progress(1.0); }
    Ok(mesh)
}

fn scanline_inside(sm: &SculptMesh, bvh: &Bvh, origin: Vec3, n: [usize; 3], h: f32) -> Vec<bool> {
    let [nx, ny, nz] = n;
    let total = nx * ny * nz;
    let mut votes = vec![0u8; total];
    // Cast rays along +X, +Y, +Z using BVH raycast stepping for parity voting
    let cast = |o: Vec3, d: Vec3, hits: &mut Vec<f32>| {
        hits.clear();
        let mut t_cur = 0.0f32;
        for _ in 0..64 {
            let ro = o + d * t_cur;
            let Some(hit) = bvh.ray(&sm.positions, &sm.tris, ro, d) else { break };
            let t_hit = t_cur + hit.t;
            hits.push(t_hit);
            t_cur = t_hit + 1e-4;
        }
    };
    let mut hits = Vec::with_capacity(16);
    // +X
    let ox = origin.x - h * 2.0;
    for k in 0..nz {
        let z = origin.z + k as f32 * h + h * 0.00019;
        for j in 0..ny {
            let y = origin.y + j as f32 * h + h * 0.00013;
            cast(Vec3::new(ox, y, z), Vec3::X, &mut hits);
            if hits.is_empty() { continue; }
            let mut hi = 0;
            let mut ins = false;
            let base = (k * ny + j) * nx;
            for i in 0..nx {
                let tg = (origin.x + i as f32 * h) - ox;
                while hi < hits.len() && hits[hi] < tg { ins = !ins; hi += 1; }
                if ins { votes[base + i] += 1; }
            }
        }
    }
    // +Y
    let oy = origin.y - h * 2.0;
    for k in 0..nz {
        let z = origin.z + k as f32 * h + h * 0.00023;
        for i in 0..nx {
            let x = origin.x + i as f32 * h + h * 0.00017;
            cast(Vec3::new(x, oy, z), Vec3::Y, &mut hits);
            if hits.is_empty() { continue; }
            let mut hi = 0;
            let mut ins = false;
            for j in 0..ny {
                let tg = (origin.y + j as f32 * h) - oy;
                while hi < hits.len() && hits[hi] < tg { ins = !ins; hi += 1; }
                if ins { votes[(k * ny + j) * nx + i] += 1; }
            }
        }
    }
    // +Z
    let oz = origin.z - h * 2.0;
    for j in 0..ny {
        let y = origin.y + j as f32 * h + h * 0.00029;
        for i in 0..nx {
            let x = origin.x + i as f32 * h + h * 0.00011;
            cast(Vec3::new(x, y, oz), Vec3::Z, &mut hits);
            if hits.is_empty() { continue; }
            let mut hi = 0;
            let mut ins = false;
            for k in 0..nz {
                let tg = (origin.z + k as f32 * h) - oz;
                while hi < hits.len() && hits[hi] < tg { ins = !ins; hi += 1; }
                if ins { votes[(k * ny + j) * nx + i] += 1; }
            }
        }
    }
    votes.into_iter().map(|v| v >= 2).collect()
}

fn surface_nets_extract(phi: &[f32], origin: Vec3, n: [usize; 3], h: f32) -> Mesh {
    let [nx, ny, nz] = n;
    let idx = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    let cell_idx = |i: usize, j: usize, k: usize| (k * (ny - 1) + j) * (nx - 1) + i;
    let mut cell_vert = vec![u32::MAX; (nx - 1) * (ny - 1) * (nz - 1)];
    let mut positions = Vec::new();
    const EDGES: [(usize, usize); 12] = [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    for k in 0..(nz - 1) {
        for j in 0..(ny - 1) {
            for i in 0..(nx - 1) {
                let mut vals = [0.0f32; 8];
                let mut pos = [Vec3::ZERO; 8];
                let mut mask = 0u8;
                for c in 0..8 {
                    let (dx, dy, dz) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
                    vals[c] = phi[idx(i + dx, j + dy, k + dz)];
                    pos[c] = origin + Vec3::new((i + dx) as f32 * h, (j + dy) as f32 * h, (k + dz) as f32 * h);
                    if vals[c] < 0.0 { mask |= 1 << c; }
                }
                if mask == 0 || mask == 255 { continue; }
                let mut sum = Vec3::ZERO;
                let mut cnt = 0.0f32;
                for (a, b) in EDGES {
                    if (vals[a] < 0.0) != (vals[b] < 0.0) {
                        let denom = vals[a] - vals[b];
                        let t = if denom.abs() > 1e-12 { (vals[a] / denom).clamp(0.0, 1.0) } else { 0.5 };
                        sum += pos[a].lerp(pos[b], t);
                        cnt += 1.0;
                    }
                }
                if cnt > 0.0 {
                    cell_vert[cell_idx(i, j, k)] = positions.len() as u32;
                    positions.push(sum / cnt);
                }
            }
        }
    }

    let mut faces = Vec::new();
    // X-axis edges: shared by 4 cells (i, j-1..=j, k-1..=k)
    for k in 1..(nz - 1) {
        for j in 1..(ny - 1) {
            for i in 0..(nx - 1) {
                let a = phi[idx(i, j, k)] < 0.0;
                let b = phi[idx(i + 1, j, k)] < 0.0;
                if a == b { continue; }
                let v0 = cell_vert[cell_idx(i, j - 1, k - 1)];
                let v1 = cell_vert[cell_idx(i, j, k - 1)];
                let v2 = cell_vert[cell_idx(i, j, k)];
                let v3 = cell_vert[cell_idx(i, j - 1, k)];
                if v0 != u32::MAX && v1 != u32::MAX && v2 != u32::MAX && v3 != u32::MAX {
                    faces.push(if a { vec![v0, v1, v2, v3] } else { vec![v0, v3, v2, v1] });
                }
            }
        }
    }
    // Y-axis edges
    for k in 1..(nz - 1) {
        for j in 0..(ny - 1) {
            for i in 1..(nx - 1) {
                let a = phi[idx(i, j, k)] < 0.0;
                let b = phi[idx(i, j + 1, k)] < 0.0;
                if a == b { continue; }
                let v0 = cell_vert[cell_idx(i - 1, j, k - 1)];
                let v1 = cell_vert[cell_idx(i - 1, j, k)];
                let v2 = cell_vert[cell_idx(i, j, k)];
                let v3 = cell_vert[cell_idx(i, j, k - 1)];
                if v0 != u32::MAX && v1 != u32::MAX && v2 != u32::MAX && v3 != u32::MAX {
                    faces.push(if a { vec![v0, v1, v2, v3] } else { vec![v0, v3, v2, v1] });
                }
            }
        }
    }
    // Z-axis edges
    for k in 0..(nz - 1) {
        for j in 1..(ny - 1) {
            for i in 1..(nx - 1) {
                let a = phi[idx(i, j, k)] < 0.0;
                let b = phi[idx(i, j, k + 1)] < 0.0;
                if a == b { continue; }
                let v0 = cell_vert[cell_idx(i - 1, j - 1, k)];
                let v1 = cell_vert[cell_idx(i, j - 1, k)];
                let v2 = cell_vert[cell_idx(i, j, k)];
                let v3 = cell_vert[cell_idx(i - 1, j, k)];
                if v0 != u32::MAX && v1 != u32::MAX && v2 != u32::MAX && v3 != u32::MAX {
                    faces.push(if a { vec![v0, v1, v2, v3] } else { vec![v0, v3, v2, v1] });
                }
            }
        }
    }
    Mesh { positions, faces, smooth_shading: false }
}

fn bvh_closest_dist(bvh: &Bvh, positions: &[Vec3], tris: &[[u32; 3]], p: Vec3, max_d: f32) -> f32 {
    if bvh.nodes.is_empty() { return max_d; }
    let mut best_d2 = max_d * max_d;
    let mut stack = [0usize; 64];
    let mut sp = 1usize;
    while sp > 0 {
        sp -= 1;
        let n = &bvh.nodes[stack[sp]];
        let clamped = p.clamp(Vec3::from(n.min), Vec3::from(n.max));
        if clamped.distance_squared(p) >= best_d2 { continue; }
        if n.count > 0 {
            let s = n.left_first as usize;
            let e = s + n.count as usize;
            for &ti in &bvh.tri_ids[s..e] {
                let t = tris[ti as usize];
                let d2 = point_tri_dist_sq(p, positions[t[0] as usize], positions[t[1] as usize], positions[t[2] as usize]);
                if d2 < best_d2 { best_d2 = d2; }
            }
        } else if sp + 2 <= stack.len() {
            stack[sp] = n.left_first as usize;
            stack[sp + 1] = n.left_first as usize + 1;
            sp += 2;
        }
    }
    best_d2.sqrt()
}

fn point_tri_dist_sq(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 { return ap.length_squared(); }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 { return bp.length_squared(); }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length_squared();
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 { return cp.length_squared(); }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length_squared();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length_squared();
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (p - (a + ab * v + ac * w)).length_squared()
}

// Keep unused warning quiet if needed
#[allow(dead_code)]
fn _tri_helper(m: &Mesh) -> usize {
    mesh_triangles(m).len()
}
