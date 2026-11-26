//! Edit mode: element selection, mesh edit tools, element picking, gizmo commands, and edit-cage overlay.

use crate::bridge::{
    commands::{catmull_clark_step, CommandOutput},
    mesh_face_normal, mesh_unique_edges, mesh_vertex_normals, ok_empty, ok_envelope,
    scene_world_matrix, BoundingBox, BridgeError, EditMode, ElementMode, EngineDoc,
};
use crate::bvh::Bvh;
use crate::format::{Mesh, MeshSelection, Scene};
use crate::mesh::SculptMesh;
use glam::{Mat4, Quat, Vec2, Vec3};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub struct OverlayCacheKey {
    pub object_id: String,
    pub mesh: Mesh,
    pub selection: MeshSelection,
    pub mode: ElementMode,
    pub dot_radius_bits: u32,
    pub scale_bits: [u32; 3],
    pub active_vertex: Option<usize>,
    pub dark: bool,
}

#[derive(Debug, Clone)]
pub struct EditOverlayData {
    pub revision: i32,
    pub matrix: Vec<f32>,
    pub line_positions: Vec<f32>,
    pub line_colors: Vec<f32>,
    pub line_indices: Vec<u32>,
    pub dot_positions: Vec<f32>,
    pub dot_colors: Vec<f32>,
    pub dot_indices: Vec<u32>,
    pub face_positions: Vec<f32>,
    pub face_colors: Vec<f32>,
    pub face_indices: Vec<u32>,
}

pub fn expanded_vertices(sel: &MeshSelection, mesh: &Mesh) -> HashSet<usize> {
    let n = mesh.positions.len();
    let mut out = HashSet::new();
    for &v in &sel.vertices {
        if v >= 0 && (v as usize) < n {
            out.insert(v as usize);
        }
    }
    for &(a, b) in &sel.edges {
        if a >= 0 && (a as usize) < n { out.insert(a as usize); }
        if b >= 0 && (b as usize) < n { out.insert(b as usize); }
    }
    for &fi in &sel.faces {
        if fi >= 0 {
            if let Some(f) = mesh.faces.get(fi as usize) {
                for &v in f {
                    if (v as usize) < n {
                        out.insert(v as usize);
                    }
                }
            }
        }
    }
    out
}

pub fn expanded_faces(sel: &MeshSelection, mesh: &Mesh) -> HashSet<usize> {
    let mut out: HashSet<usize> = sel
        .faces
        .iter()
        .filter_map(|&fi| (fi >= 0 && (fi as usize) < mesh.faces.len()).then_some(fi as usize))
        .collect();
    let verts = expanded_vertices(sel, mesh);
    if !verts.is_empty() {
        for (fi, f) in mesh.faces.iter().enumerate() {
            if !f.is_empty() && f.iter().all(|&v| verts.contains(&(v as usize))) {
                out.insert(fi);
            }
        }
    }
    out
}

pub fn selected_edges(sel: &MeshSelection, mesh: &Mesh) -> Vec<(i64, i64)> {
    let verts = expanded_vertices(sel, mesh);
    let mut set: HashSet<(i64, i64)> = sel.edges.iter().map(|&(a, b)| if a < b { (a, b) } else { (b, a) }).collect();
    for (a, b) in mesh_unique_edges(mesh) {
        if verts.contains(&(a as usize)) && verts.contains(&(b as usize)) {
            set.insert((a as i64, b as i64));
        }
    }
    let mut out: Vec<(i64, i64)> = set.into_iter().collect();
    out.sort_unstable();
    out
}

pub fn elements_from_vertices(verts: &HashSet<usize>, mode: ElementMode, mesh: &Mesh) -> MeshSelection {
    match mode {
        ElementMode::Vertex => {
            let mut v: Vec<i64> = verts.iter().map(|&i| i as i64).collect();
            v.sort_unstable();
            MeshSelection { vertices: v, edges: Vec::new(), faces: Vec::new() }
        }
        ElementMode::Edge => {
            let edges = mesh_unique_edges(mesh)
                .into_iter()
                .filter(|&(a, b)| verts.contains(&(a as usize)) && verts.contains(&(b as usize)))
                .map(|(a, b)| (a as i64, b as i64))
                .collect();
            MeshSelection { vertices: Vec::new(), edges, faces: Vec::new() }
        }
        ElementMode::Face => {
            let faces = mesh
                .faces
                .iter()
                .enumerate()
                .filter(|(_, f)| !f.is_empty() && f.iter().all(|&v| verts.contains(&(v as usize))))
                .map(|(i, _)| i as i64)
                .collect();
            MeshSelection { vertices: Vec::new(), edges: Vec::new(), faces }
        }
    }
}

pub fn element_stats(doc: &EngineDoc) -> Option<Value> {
    let id = doc.selection.active.as_deref()?;
    let obj = doc.scene.objects.iter().find(|o| o.id == id)?;
    let mesh = &obj.mesh;
    let edges = mesh_unique_edges(mesh);
    let verts = expanded_vertices(&doc.selection.elements, mesh);
    let sel_edges_set: HashSet<(i64, i64)> = doc.selection.elements.edges.iter().copied().collect();
    let edge_cnt = edges
        .iter()
        .filter(|&&(a, b)| (verts.contains(&(a as usize)) && verts.contains(&(b as usize))) || sel_edges_set.contains(&(a as i64, b as i64)))
        .count();
    let face_cnt = expanded_faces(&doc.selection.elements, mesh).len();
    Some(json!({
        "vertices": verts.len(),
        "edges": edge_cnt,
        "faces": face_cnt,
        "totalVertices": mesh.positions.len(),
        "totalEdges": edges.len(),
        "totalFaces": mesh.faces.len(),
    }))
}

pub fn set_edit_mode(doc: &mut EngineDoc, mode_name: &str) -> String {
    if mode_name == "object" {
        if matches!(doc.edit_mode, EditMode::Edit(_)) {
            doc.selection.elements = MeshSelection::default();
            doc.edit_mode = EditMode::Object;
        }
        doc.last_picked_vertex = None;
        return ok_empty();
    }
    let Some(new_mode) = ElementMode::parse(mode_name) else {
        return BridgeError::with_args("engine.badJSON", vec![mode_name.to_string()]).to_envelope();
    };
    let Some(active_id) = doc.selection.active.clone() else {
        return BridgeError::new("edit.selectFirst").to_envelope();
    };
    let Some(obj) = doc.scene.objects.iter().find(|o| o.id == active_id) else {
        return BridgeError::new("edit.selectFirst").to_envelope();
    };
    if obj.is_locked {
        return BridgeError::with_args("edit.locked", vec![obj.name.clone()]).to_envelope();
    }
    let mesh = obj.mesh.clone();
    match doc.edit_mode {
        EditMode::Edit(cur) => {
            if cur == new_mode {
                return ok_empty();
            }
            let verts = expanded_vertices(&doc.selection.elements, &mesh);
            doc.selection.elements = if verts.is_empty() {
                MeshSelection::default()
            } else {
                elements_from_vertices(&verts, new_mode, &mesh)
            };
        }
        EditMode::Object => {
            doc.selection.elements = MeshSelection::default();
        }
    }
    doc.last_picked_vertex = None;
    doc.edit_mode = EditMode::Edit(new_mode);
    ok_empty()
}

pub fn select_linked(doc: &mut EngineDoc) -> String {
    let EditMode::Edit(mode) = doc.edit_mode else { return ok_empty() };
    let Some(active_id) = doc.selection.active.clone() else { return ok_empty() };
    let Some(obj) = doc.scene.objects.iter().find(|o| o.id == active_id) else { return ok_empty() };
    let mesh = &obj.mesh;
    let seed = expanded_vertices(&doc.selection.elements, mesh);
    if seed.is_empty() {
        return BridgeError::new("edit.linkedNeedsSelection").to_envelope();
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); mesh.positions.len()];
    for f in &mesh.faces {
        for i in 0..f.len() {
            let a = f[i] as usize;
            let b = f[(i + 1) % f.len()] as usize;
            if a < adj.len() && b < adj.len() {
                adj[a].push(b);
                adj[b].push(a);
            }
        }
    }
    let mut seen = seed;
    let mut stack: Vec<usize> = seen.iter().copied().collect();
    while let Some(v) = stack.pop() {
        for &n in &adj[v] {
            if seen.insert(n) {
                stack.push(n);
            }
        }
    }
    doc.selection.elements = elements_from_vertices(&seen, mode, mesh);
    ok_empty()
}

// ── Edit tools ──

pub fn run_tool(doc: &mut EngineDoc, tool: &str, options_json: &str) -> String {
    let opts: Value = serde_json::from_str(options_json).unwrap_or(json!({}));
    match doc.edit_mode {
        EditMode::Object => {
            let ids = doc.ordered_selection();
            if ids.is_empty() {
                return BridgeError::new("tool.selectObject").to_envelope();
            }
            let mut next = doc.scene.clone();
            let label = match tool {
                "shadeSmooth" => {
                    for id in &ids {
                        if let Some(o) = next.objects.iter_mut().find(|o| &o.id == id) { o.mesh.smooth_shading = true; }
                    }
                    "Shade Smooth"
                }
                "shadeFlat" => {
                    for id in &ids {
                        if let Some(o) = next.objects.iter_mut().find(|o| &o.id == id) { o.mesh.smooth_shading = false; }
                    }
                    "Shade Flat"
                }
                "flipNormals" => {
                    for id in &ids {
                        if let Some(o) = next.objects.iter_mut().find(|o| &o.id == id) {
                            for f in &mut o.mesh.faces { f.reverse(); }
                        }
                    }
                    "Flip Normals"
                }
                "recalcNormals" => {
                    for id in &ids {
                        if let Some(o) = next.objects.iter_mut().find(|o| &o.id == id) {
                            recalculate_normals_outward(&mut o.mesh);
                        }
                    }
                    "Recalculate Normals"
                }
                _ => return BridgeError::with_args("tool.needsEdit", vec![tool.to_string()]).to_envelope(),
            };
            doc.record_scene_change(label.to_string(), next, &[], None);
            ok_empty()
        }
        EditMode::Edit(mode) => {
            let Some(active_id) = doc.selection.active.clone() else {
                return BridgeError::new("edit.selectFirst").to_envelope();
            };
            let mut next = doc.scene.clone();
            let Some(obj) = next.objects.iter_mut().find(|o| o.id == active_id) else {
                return BridgeError::new("edit.selectFirst").to_envelope();
            };
            let sel = doc.selection.elements.clone();
            let res = apply_tool_to_mesh(&mut obj.mesh, &sel, mode, tool, &opts);
            match res {
                Ok((label, new_sel)) => {
                    obj.primitive_kind = None;
                    doc.record_scene_change(label, next, &[], new_sel);
                    ok_empty()
                }
                Err(e) => e.to_envelope(),
            }
        }
    }
}

pub fn apply_edit_command(scene: &mut Scene, case: &str, body: &Value) -> Result<CommandOutput, BridgeError> {
    let id = body
        .get("_0")
        .and_then(|v| v.as_str().or_else(|| v.get("rawValue").and_then(|x| x.as_str())))
        .ok_or_else(|| BridgeError::new("engine.badJSON"))?
        .to_string();
    let o = scene.objects.iter_mut().find(|o| o.id == id).ok_or_else(|| BridgeError::new("command.objectNotFound"))?;
    let label = match case {
        "flipNormals" => {
            let faces: HashSet<usize> = body
                .get("faces")
                .and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_u64().map(|u| u as usize)).collect())
                .unwrap_or_default();
            for (fi, f) in o.mesh.faces.iter_mut().enumerate() {
                if faces.is_empty() || faces.contains(&fi) {
                    f.reverse();
                }
            }
            "Flip Normals"
        }
        "recalculateNormals" => {
            recalculate_normals_outward(&mut o.mesh);
            "Recalculate Normals"
        }
        "subdivide" => {
            o.mesh = catmull_clark_step(&o.mesh);
            o.primitive_kind = None;
            "Subdivide"
        }
        _ => {
            o.primitive_kind = None;
            case
        }
    };
    Ok(CommandOutput { created: Vec::new(), new_elements: None, label: label.to_string(), affected: vec![id] })
}

fn apply_tool_to_mesh(
    mesh: &mut Mesh,
    sel: &MeshSelection,
    mode: ElementMode,
    tool: &str,
    opts: &Value,
) -> Result<(String, Option<MeshSelection>), BridgeError> {
    let sel_empty = sel.vertices.is_empty() && sel.edges.is_empty() && sel.faces.is_empty();
    match tool {
        "extrude" => {
            if sel_empty {
                return Err(BridgeError::with_args("tool.extrude.empty", vec![mode.as_str().to_string()]));
            }
            let dist = opts.get("extrudeDistanceMm").and_then(|x| x.as_f64()).unwrap_or(10.0) as f32;
            let faces = expanded_faces(sel, mesh);
            if !faces.is_empty() {
                let mut new_faces_sel = Vec::new();
                let mut avg_n = Vec3::ZERO;
                for &fi in &faces { avg_n += mesh_face_normal(mesh, fi); }
                let dir = if avg_n.length_squared() > 1e-12 { avg_n.normalize() } else { Vec3::Z };
                let mut vmap: HashMap<u32, u32> = HashMap::new();
                let mut old_faces_list: Vec<usize> = faces.into_iter().collect();
                old_faces_list.sort_unstable();
                // Duplicate vertices of selected faces and bridge boundary edges
                let mut edge_uses: HashMap<(u32, u32), usize> = HashMap::new();
                let mut directed_edges: Vec<(u32, u32)> = Vec::new();
                for &fi in &old_faces_list {
                    let f = mesh.faces[fi].clone();
                    for i in 0..f.len() {
                        let a = f[i];
                        let b = f[(i + 1) % f.len()];
                        let key = if a < b { (a, b) } else { (b, a) };
                        *edge_uses.entry(key).or_insert(0) += 1;
                        directed_edges.push((a, b));
                    }
                    for &v in &f {
                        vmap.entry(v).or_insert_with(|| {
                            let idx = mesh.positions.len() as u32;
                            mesh.positions.push(mesh.positions[v as usize] + dir * dist);
                            idx
                        });
                    }
                }
                for &fi in &old_faces_list {
                    let nf: Vec<u32> = mesh.faces[fi].iter().map(|v| vmap[v]).collect();
                    mesh.faces[fi] = nf;
                    new_faces_sel.push(fi as i64);
                }
                for (a, b) in directed_edges {
                    let key = if a < b { (a, b) } else { (b, a) };
                    if edge_uses[&key] == 1 {
                        mesh.faces.push(vec![a, b, vmap[&b], vmap[&a]]);
                    }
                }
                return Ok(("Extrude".into(), Some(MeshSelection { vertices: Vec::new(), edges: Vec::new(), faces: new_faces_sel })));
            }
            // Vertex/edge extrude fallback: duplicate selected vertices along average normal
            let verts = expanded_vertices(sel, mesh);
            let normals = mesh_vertex_normals(mesh);
            let mut new_vs = Vec::new();
            for v in verts {
                let n = normals.get(v).copied().unwrap_or(Vec3::Z);
                let idx = mesh.positions.len() as i64;
                mesh.positions.push(mesh.positions[v] + n * dist);
                new_vs.push(idx);
            }
            Ok(("Extrude".into(), Some(MeshSelection { vertices: new_vs, edges: Vec::new(), faces: Vec::new() })))
        }
        "inset" => {
            let faces = expanded_faces(sel, mesh);
            if faces.is_empty() {
                return Err(BridgeError::new("tool.inset.noFaces"));
            }
            let thickness = (opts.get("insetThicknessMm").and_then(|x| x.as_f64()).unwrap_or(2.0) as f32).max(0.05);
            let mut flist: Vec<usize> = faces.into_iter().collect();
            flist.sort_unstable();
            let mut new_sel_faces = Vec::new();
            for fi in flist {
                let f = mesh.faces[fi].clone();
                if f.len() < 3 { continue; }
                let center: Vec3 = f.iter().map(|&v| mesh.positions[v as usize]).sum::<Vec3>() / (f.len() as f32);
                let mut inner = Vec::with_capacity(f.len());
                for &v in &f {
                    let p = mesh.positions[v as usize];
                    let to_c = center - p;
                    let len = to_c.length();
                    let np = if len > 1e-5 { p + to_c * (thickness / len).min(0.85) } else { p };
                    let idx = mesh.positions.len() as u32;
                    mesh.positions.push(np);
                    inner.push(idx);
                }
                mesh.faces[fi] = inner.clone();
                new_sel_faces.push(fi as i64);
                for i in 0..f.len() {
                    let ni = (i + 1) % f.len();
                    mesh.faces.push(vec![f[i], f[ni], inner[ni], inner[i]]);
                }
            }
            Ok(("Inset".into(), Some(MeshSelection { vertices: Vec::new(), edges: Vec::new(), faces: new_sel_faces })))
        }
        "bevel" => {
            if mode == ElementMode::Vertex {
                if sel.vertices.is_empty() {
                    return Err(BridgeError::new("tool.bevel.noVertices"));
                }
            } else if selected_edges(sel, mesh).is_empty() {
                return Err(BridgeError::new("tool.bevel.noEdges"));
            }
            let segs = (opts.get("bevelSegments").and_then(|x| x.as_i64()).unwrap_or(1)).clamp(1, 2);
            for _ in 0..segs {
                *mesh = catmull_clark_step(mesh);
            }
            Ok(("Bevel".into(), Some(MeshSelection::default())))
        }
        "loopCut" => {
            let edges = selected_edges(sel, mesh);
            if edges.is_empty() {
                return Err(BridgeError::new(if mode == ElementMode::Edge { "tool.loopCut.tapEdge" } else { "tool.loopCut.needsEdge" }));
            }
            *mesh = catmull_clark_step(mesh);
            Ok(("Loop Cut".into(), Some(MeshSelection::default())))
        }
        "subdivide" => {
            *mesh = catmull_clark_step(mesh);
            Ok(("Subdivide".into(), Some(MeshSelection::default())))
        }
        "merge" => {
            let merge_mode = opts.get("mergeMode").and_then(|x| x.as_str()).unwrap_or("atCenter");
            let mut verts: Vec<usize> = expanded_vertices(sel, mesh).into_iter().collect();
            verts.sort_unstable();
            if merge_mode == "byDistance" {
                if verts.is_empty() {
                    verts = (0..mesh.positions.len()).collect();
                }
                let eps = (opts.get("mergeDistanceMm").and_then(|x| x.as_f64()).unwrap_or(0.1) as f32).max(1e-5);
                weld_vertices(mesh, &verts, eps);
                return Ok(("Merge Vertices".into(), Some(MeshSelection::default())));
            }
            if verts.len() < 2 {
                return Err(BridgeError::new("tool.merge.needTwo"));
            }
            let target_pos = if merge_mode == "atFirst" {
                mesh.positions[verts[0]]
            } else {
                verts.iter().map(|&v| mesh.positions[v]).sum::<Vec3>() / (verts.len() as f32)
            };
            let keep = verts[0] as u32;
            mesh.positions[verts[0]] = target_pos;
            let merged: HashSet<u32> = verts[1..].iter().map(|&v| v as u32).collect();
            for f in &mut mesh.faces {
                for vi in f.iter_mut() {
                    if merged.contains(vi) { *vi = keep; }
                }
                f.dedup();
                if f.len() > 1 && f.first() == f.last() { f.pop(); }
            }
            mesh.faces.retain(|f| f.len() >= 3);
            Ok(("Merge Vertices".into(), Some(MeshSelection { vertices: vec![keep as i64], edges: Vec::new(), faces: Vec::new() })))
        }
        "delete" => {
            if sel_empty {
                return Err(BridgeError::with_args("tool.delete.empty", vec![mode.as_str().to_string()]));
            }
            match mode {
                ElementMode::Face => {
                    let drop = expanded_faces(sel, mesh);
                    let mut keep = Vec::new();
                    for (fi, f) in mesh.faces.iter().enumerate() {
                        if !drop.contains(&fi) { keep.push(f.clone()); }
                    }
                    mesh.faces = keep;
                }
                ElementMode::Edge => {
                    let drop: HashSet<(u32, u32)> = selected_edges(sel, mesh).into_iter().map(|(a, b)| (a as u32, b as u32)).collect();
                    mesh.faces.retain(|f| {
                        for i in 0..f.len() {
                            let a = f[i];
                            let b = f[(i + 1) % f.len()];
                            let key = if a < b { (a, b) } else { (b, a) };
                            if drop.contains(&key) { return false; }
                        }
                        true
                    });
                }
                ElementMode::Vertex => {
                    let drop = expanded_vertices(sel, mesh);
                    mesh.faces.retain(|f| !f.iter().any(|&v| drop.contains(&(v as usize))));
                }
            }
            Ok(("Delete".into(), Some(MeshSelection::default())))
        }
        "flipNormals" => {
            let faces = expanded_faces(sel, mesh);
            for (fi, f) in mesh.faces.iter_mut().enumerate() {
                if faces.is_empty() || faces.contains(&fi) { f.reverse(); }
            }
            Ok(("Flip Normals".into(), None))
        }
        "recalcNormals" => {
            recalculate_normals_outward(mesh);
            Ok(("Recalculate Normals".into(), None))
        }
        "shadeSmooth" => {
            mesh.smooth_shading = true;
            Ok(("Shade Smooth".into(), None))
        }
        "shadeFlat" => {
            mesh.smooth_shading = false;
            Ok(("Shade Flat".into(), None))
        }
        _ => Err(BridgeError::with_args("engine.badJSON", vec![tool.to_string()])),
    }
}

fn weld_vertices(mesh: &mut Mesh, verts: &[usize], eps: f32) {
    let eps2 = eps * eps;
    let mut remap: Vec<u32> = (0..mesh.positions.len() as u32).collect();
    for (ii, &a) in verts.iter().enumerate() {
        if remap[a] != a as u32 { continue; }
        let pa = mesh.positions[a];
        for &b in &verts[(ii + 1)..] {
            if remap[b] == b as u32 && pa.distance_squared(mesh.positions[b]) <= eps2 {
                remap[b] = a as u32;
            }
        }
    }
    for f in &mut mesh.faces {
        for v in f.iter_mut() { *v = remap[*v as usize]; }
        f.dedup();
        if f.len() > 1 && f.first() == f.last() { f.pop(); }
    }
    mesh.faces.retain(|f| f.len() >= 3);
}

pub fn recalculate_normals_outward(mesh: &mut Mesh) {
    let mut signed_vol = 0.0f64;
    for f in &mesh.faces {
        if f.len() < 3 { continue; }
        let a = mesh.positions[f[0] as usize].as_dvec3();
        for k in 1..(f.len() - 1) {
            let b = mesh.positions[f[k] as usize].as_dvec3();
            let c = mesh.positions[f[k + 1] as usize].as_dvec3();
            signed_vol += a.dot(b.cross(c));
        }
    }
    if signed_vol < 0.0 {
        for f in &mut mesh.faces { f.reverse(); }
    }
}

// ── Element picking, Box select, Gizmo command, Can drag ──

struct PickCamera {
    view_proj: [f32; 16],
    width: f32,
    height: f32,
    eye: Vec3,
    forward: Vec3,
    ortho: bool,
    ortho_back: f32,
    ray_origin: Vec3,
    ray_dir: Vec3,
}

impl PickCamera {
    fn parse(json_str: &str) -> Option<Self> {
        let v: Value = serde_json::from_str(json_str).ok()?;
        let vp_arr = v.get("viewProj")?.as_array()?;
        if vp_arr.len() != 16 { return None; }
        let mut view_proj = [0.0f32; 16];
        for (i, x) in vp_arr.iter().enumerate() { view_proj[i] = x.as_f64()? as f32; }
        let v3 = |key: &str| -> Vec3 {
            v.get(key)
                .and_then(|x| x.as_array())
                .filter(|a| a.len() >= 3)
                .map(|a| Vec3::new(
                    a[0].as_f64().unwrap_or(0.0) as f32,
                    a[1].as_f64().unwrap_or(0.0) as f32,
                    a[2].as_f64().unwrap_or(0.0) as f32,
                ))
                .unwrap_or(Vec3::ZERO)
        };
        Some(Self {
            view_proj,
            width: v.get("width")?.as_f64()? as f32,
            height: v.get("height")?.as_f64()? as f32,
            eye: v3("eye"),
            forward: v3("forward"),
            ortho: v.get("ortho").and_then(|x| x.as_bool()).unwrap_or(false),
            ortho_back: v.get("orthoBack").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
            ray_origin: v3("rayOrigin"),
            ray_dir: v3("rayDirection"),
        })
    }

    fn project(&self, p: Vec3) -> Option<Vec2> {
        let m = &self.view_proj;
        let x = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
        let y = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
        let w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
        if w <= 1e-6 { return None; }
        Some(Vec2::new((x / w + 1.0) * 0.5 * self.width, (1.0 - y / w) * 0.5 * self.height))
    }
}

fn dist_point_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let l2 = ab.length_squared();
    if l2 <= 1e-8 { return p.distance(a); }
    let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

fn raycast_mesh_face(origin: Vec3, dir: Vec3, mesh: &Mesh, world: Mat4) -> Option<(f32, usize)> {
    let det = world.determinant();
    if !det.is_finite() || det.abs() < 1e-12 { return None; }
    let inv = world.inverse();
    let lo = inv.transform_point3(origin);
    let ld = inv.transform_vector3(dir);
    let ld_len = ld.length();
    if ld_len <= 1e-12 { return None; }
    let ld_n = ld / ld_len;
    let sm = SculptMesh::from_mesh(mesh);
    if sm.tris.is_empty() { return None; }
    let bvh = Bvh::build(&sm.positions, &sm.tris);
    let hit = bvh.ray(&sm.positions, &sm.tris, lo, ld_n)?;
    let world_p = world.transform_point3(lo + ld_n * hit.t);
    let dist = world_p.distance(origin);
    // Map triangle index back to polygon face index
    let mut tri_acc = 0usize;
    for (fi, f) in mesh.faces.iter().enumerate() {
        if f.len() < 3 { continue; }
        let cnt = f.len() - 2;
        if (hit.tri as usize) < tri_acc + cnt {
            return Some((dist, fi));
        }
        tri_acc += cnt;
    }
    None
}

pub fn tap_element(doc: &mut EngineDoc, camera_json: &str, x: f32, y: f32, density: f32, extend: bool) -> String {
    let EditMode::Edit(mode) = doc.edit_mode else {
        return json!({"ok": true, "hit": null}).to_string();
    };
    let Some(active_id) = doc.selection.active.clone() else {
        return json!({"ok": true, "hit": null}).to_string();
    };
    let Some(cam) = PickCamera::parse(camera_json) else {
        return json!({"ok": true, "hit": null}).to_string();
    };
    let scene = doc.previewed_scene();
    let Some(obj) = scene.objects.iter().find(|o| o.id == active_id) else {
        return json!({"ok": true, "hit": null}).to_string();
    };
    let mesh = &obj.mesh;
    let world = scene_world_matrix(&scene, &active_id);
    let pt = Vec2::new(x, y);
    let dens = density.max(1.0);

    let is_visible = |world_p: Vec3| -> bool {
        let orig = if cam.ortho { world_p - cam.forward * cam.ortho_back } else { cam.eye };
        let to = world_p - orig;
        let dist = to.length();
        if dist <= 1e-4 { return true; }
        match raycast_mesh_face(orig, to / dist, mesh, world) {
            None => true,
            Some((hit_d, _)) => hit_d >= dist - (dist * 0.002).max(0.02),
        }
    };

    enum Hit { V(usize), E((u32, u32)), F(usize) }
    let hit = match mode {
        ElementMode::Face => raycast_mesh_face(cam.ray_origin, cam.ray_dir, mesh, world).map(|(_, fi)| Hit::F(fi)),
        ElementMode::Vertex => {
            let radius = 14.0 * dens;
            let mut cands: Vec<(usize, f32)> = mesh
                .positions
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| {
                    let sp = cam.project(world.transform_point3(p))?;
                    let d = sp.distance(pt);
                    (d <= radius).then_some((i, d))
                })
                .collect();
            cands.sort_by(|a, b| a.1.total_cmp(&b.1));
            cands
                .into_iter()
                .take(8)
                .find(|&(i, _)| is_visible(world.transform_point3(mesh.positions[i])))
                .map(|(i, _)| Hit::V(i))
        }
        ElementMode::Edge => {
            let radius = 10.0 * dens;
            let proj: Vec<Option<Vec2>> = mesh.positions.iter().map(|&p| cam.project(world.transform_point3(p))).collect();
            let mut cands: Vec<((u32, u32), f32)> = Vec::new();
            for (a, b) in mesh_unique_edges(mesh) {
                if let (Some(pa), Some(pb)) = (proj[a as usize], proj[b as usize]) {
                    let d = dist_point_segment(pt, pa, pb);
                    if d <= radius { cands.push(((a, b), d)); }
                }
            }
            cands.sort_by(|a, b| a.1.total_cmp(&b.1));
            cands
                .into_iter()
                .take(8)
                .find(|&((a, b), _)| {
                    let mid = world.transform_point3((mesh.positions[a as usize] + mesh.positions[b as usize]) * 0.5);
                    is_visible(mid)
                })
                .map(|(e, _)| Hit::E(e))
        }
    };

    let Some(hit) = hit else {
        let was_non_empty = !doc.selection.elements.vertices.is_empty()
            || !doc.selection.elements.edges.is_empty()
            || !doc.selection.elements.faces.is_empty();
        if was_non_empty && !extend {
            doc.selection.elements = MeshSelection::default();
            doc.last_picked_vertex = None;
            return json!({"ok": true, "hit": null, "cleared": true}).to_string();
        }
        return json!({"ok": true, "hit": null}).to_string();
    };

    let mut elems = if extend { doc.selection.elements.clone() } else { MeshSelection::default() };
    let (kind, count) = match hit {
        Hit::V(v) => {
            let vi = v as i64;
            if extend && elems.vertices.contains(&vi) {
                elems.vertices.retain(|&x| x != vi);
                doc.last_picked_vertex = None;
            } else {
                if !elems.vertices.contains(&vi) { elems.vertices.push(vi); }
                doc.last_picked_vertex = Some(v);
            }
            ("vertex", elems.vertices.len())
        }
        Hit::E((a, b)) => {
            let ek = (a as i64, b as i64);
            if extend && elems.edges.contains(&ek) {
                elems.edges.retain(|&x| x != ek);
            } else if !elems.edges.contains(&ek) {
                elems.edges.push(ek);
            }
            ("edge", elems.edges.len())
        }
        Hit::F(fi) => {
            let f = fi as i64;
            if extend && elems.faces.contains(&f) {
                elems.faces.retain(|&x| x != f);
            } else if !elems.faces.contains(&f) {
                elems.faces.push(f);
            }
            ("face", elems.faces.len())
        }
    };
    doc.selection.elements = elems;
    json!({"ok": true, "hit": kind, "count": count}).to_string()
}

pub fn box_select(doc: &mut EngineDoc, camera_json: &str, x0: f32, y0: f32, x1: f32, y1: f32, extend: bool) -> String {
    let Some(cam) = PickCamera::parse(camera_json) else {
        return json!({"ok": true, "count": 0}).to_string();
    };
    let lo = Vec2::new(x0.min(x1), y0.min(y1));
    let hi = Vec2::new(x0.max(x1), y0.max(y1));
    doc.refresh_cache();
    let scene = doc.previewed_scene();
    match doc.edit_mode {
        EditMode::Object => {
            let mut ids = Vec::new();
            for o in scene.objects.iter().filter(|o| o.is_visible) {
                let mesh = doc.evaluated_mesh(&o.id).unwrap_or(&o.mesh);
                let b = BoundingBox::from_mesh(mesh, scene_world_matrix(&scene, &o.id));
                if b.is_empty() { continue; }
                let mut smin = Vec2::splat(f32::INFINITY);
                let mut smax = Vec2::splat(f32::NEG_INFINITY);
                let mut any = false;
                for px in [b.min.x, b.max.x] {
                    for py in [b.min.y, b.max.y] {
                        for pz in [b.min.z, b.max.z] {
                            if let Some(sp) = cam.project(Vec3::new(px, py, pz)) {
                                smin = smin.min(sp);
                                smax = smax.max(sp);
                                any = true;
                            }
                        }
                    }
                }
                if any && smin.x <= hi.x && smax.x >= lo.x && smin.y <= hi.y && smax.y >= lo.y {
                    ids.push(o.id.clone());
                }
            }
            let found = ids.len();
            if extend {
                for id in &ids {
                    if !doc.selection.objects.contains(id) { doc.selection.objects.push(id.clone()); }
                }
                if doc.selection.active.is_none() { doc.selection.active = ids.last().cloned(); }
            } else {
                doc.selection.objects = ids.clone();
                doc.selection.active = ids.last().cloned();
            }
            json!({"ok": true, "count": doc.selection.objects.len(), "found": found}).to_string()
        }
        EditMode::Edit(mode) => {
            let Some(active_id) = doc.selection.active.clone() else {
                return json!({"ok": true, "count": 0}).to_string();
            };
            let Some(obj) = scene.objects.iter().find(|o| o.id == active_id) else {
                return json!({"ok": true, "count": 0}).to_string();
            };
            let w = scene_world_matrix(&scene, &active_id);
            let inside: HashSet<usize> = obj
                .mesh
                .positions
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| {
                    let sp = cam.project(w.transform_point3(p))?;
                    (sp.x >= lo.x && sp.x <= hi.x && sp.y >= lo.y && sp.y <= hi.y).then_some(i)
                })
                .collect();
            let mut elems = if extend { doc.selection.elements.clone() } else { MeshSelection::default() };
            match mode {
                ElementMode::Vertex => {
                    for v in inside {
                        let vi = v as i64;
                        if !elems.vertices.contains(&vi) { elems.vertices.push(vi); }
                    }
                    elems.vertices.sort_unstable();
                }
                ElementMode::Edge => {
                    for (a, b) in mesh_unique_edges(&obj.mesh) {
                        if inside.contains(&(a as usize)) && inside.contains(&(b as usize)) {
                            let e = (a as i64, b as i64);
                            if !elems.edges.contains(&e) { elems.edges.push(e); }
                        }
                    }
                }
                ElementMode::Face => {
                    for (fi, f) in obj.mesh.faces.iter().enumerate() {
                        if !f.is_empty() && f.iter().all(|&v| inside.contains(&(v as usize))) {
                            let idx = fi as i64;
                            if !elems.faces.contains(&idx) { elems.faces.push(idx); }
                        }
                    }
                }
            }
            let count = match mode {
                ElementMode::Vertex => elems.vertices.len(),
                ElementMode::Edge => elems.edges.len(),
                ElementMode::Face => elems.faces.len(),
            };
            doc.selection.elements = elems;
            json!({"ok": true, "count": count}).to_string()
        }
    }
}

fn locked_or_nothing(doc: &EngineDoc) -> BridgeError {
    let sel: HashSet<&str> = doc.selection.objects.iter().map(|s| s.as_str()).collect();
    let locked: Vec<&str> = doc
        .scene
        .objects
        .iter()
        .filter(|o| sel.contains(o.id.as_str()) && o.is_locked)
        .map(|o| o.name.as_str())
        .collect();
    if locked.len() == 1 {
        BridgeError::with_args("gizmo.locked", vec![locked[0].to_string()])
    } else if locked.len() > 1 {
        BridgeError::new("gizmo.lockedMany")
    } else {
        BridgeError::new("gizmo.nothing")
    }
}

pub fn can_drag(doc: &EngineDoc) -> String {
    match doc.edit_mode {
        EditMode::Object => {
            let sel: HashSet<&str> = doc.selection.objects.iter().map(|s| s.as_str()).collect();
            let ok = doc.scene.objects.iter().any(|o| sel.contains(o.id.as_str()) && !o.is_locked && o.is_visible);
            if ok { ok_empty() } else { locked_or_nothing(doc).to_envelope() }
        }
        EditMode::Edit(_) => {
            let Some(id) = doc.selection.active.as_deref() else {
                return BridgeError::new("gizmo.nothing").to_envelope();
            };
            let Some(obj) = doc.scene.objects.iter().find(|o| o.id == id) else {
                return BridgeError::new("gizmo.nothing").to_envelope();
            };
            if obj.is_locked {
                return BridgeError::with_args("gizmo.locked", vec![obj.name.clone()]).to_envelope();
            }
            if expanded_vertices(&doc.selection.elements, &obj.mesh).is_empty() {
                BridgeError::new("gizmo.nothing").to_envelope()
            } else {
                ok_empty()
            }
        }
    }
}

pub fn gizmo_command(doc: &EngineDoc, json_str: &str) -> String {
    let val: Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return BridgeError::new("engine.badJSON").to_envelope(),
    };
    let parse_v3 = |key: &str| -> Option<Vec3> {
        let a = val.get(key)?.as_array()?;
        if a.len() < 3 { return None; }
        Some(Vec3::new(a[0].as_f64()? as f32, a[1].as_f64()? as f32, a[2].as_f64()? as f32))
    };
    let Some(pivot) = parse_v3("pivot") else {
        return BridgeError::new("engine.badJSON").to_envelope();
    };
    let tr = parse_v3("translate");
    let rot_axis = parse_v3("rotateAxis");
    let radians = val.get("radians").and_then(|x| x.as_f64()).map(|x| x as f32);
    let scale = parse_v3("scale");

    let (cmd, label): (Option<Value>, &str) = match doc.edit_mode {
        EditMode::Object => {
            let sel: HashSet<&str> = doc.selection.objects.iter().map(|s| s.as_str()).collect();
            let roots: Vec<Value> = doc
                .scene
                .objects
                .iter()
                .filter(|o| {
                    if !sel.contains(o.id.as_str()) || o.is_locked || !o.is_visible { return false; }
                    let mut p = o.parent.as_deref();
                    let mut steps = 0;
                    while let Some(pid) = p {
                        if steps > doc.scene.objects.len() { break; }
                        if sel.contains(pid) { return false; }
                        p = doc.scene.objects.iter().find(|x| x.id == pid).and_then(|x| x.parent.as_deref());
                        steps += 1;
                    }
                    true
                })
                .map(|o| json!({"rawValue": o.id}))
                .collect();
            if roots.is_empty() {
                return locked_or_nothing(doc).to_envelope();
            }
            if let Some(t) = tr {
                if t.length() > 1e-6 {
                    (Some(json!({"translate": {"_0": roots, "by": [t.x, t.y, t.z]}})), "Move")
                } else { (None, "") }
            } else if let (Some(ax), Some(r)) = (rot_axis, radians) {
                if r.abs() > 1e-6 && ax.length() > 1e-6 {
                    let q = Quat::from_axis_angle(ax.normalize(), r);
                    (Some(json!({"rotate": {"_0": roots, "by": [q.x, q.y, q.z, q.w], "pivot": [pivot.x, pivot.y, pivot.z]}})), "Rotate")
                } else { (None, "") }
            } else if let Some(s) = scale {
                if (s - Vec3::ONE).length() > 1e-6 {
                    (Some(json!({"scale": {"_0": roots, "by": [s.x, s.y, s.z], "pivot": [pivot.x, pivot.y, pivot.z]}})), "Scale")
                } else { (None, "") }
            } else {
                (None, "")
            }
        }
        EditMode::Edit(_) => {
            let Some(id) = doc.selection.active.as_deref() else {
                return BridgeError::new("gizmo.nothing").to_envelope();
            };
            let Some(obj) = doc.scene.objects.iter().find(|o| o.id == id) else {
                return BridgeError::new("gizmo.nothing").to_envelope();
            };
            if obj.is_locked {
                return BridgeError::with_args("gizmo.locked", vec![obj.name.clone()]).to_envelope();
            }
            if expanded_vertices(&doc.selection.elements, &obj.mesh).is_empty() {
                return BridgeError::new("gizmo.nothing").to_envelope();
            }
            let world = scene_world_matrix(&doc.scene, id);
            let inv = world.inverse();
            let sel_json = json!({
                "vertices": doc.selection.elements.vertices,
                "edges": doc.selection.elements.edges.iter().map(|&(a, b)| json!({"a": a, "b": b})).collect::<Vec<_>>(),
                "faces": doc.selection.elements.faces,
            });
            if let Some(t) = tr {
                if t.length() > 1e-6 {
                    let lt = inv.transform_vector3(t);
                    (Some(json!({"moveElements": {"_0": {"rawValue": id}, "_1": sel_json, "by": [lt.x, lt.y, lt.z]}})), "Move Elements")
                } else { (None, "") }
            } else {
                let m = if let (Some(ax), Some(r)) = (rot_axis, radians) {
                    (r.abs() > 1e-6 && ax.length() > 1e-6)
                        .then(|| Mat4::from_translation(pivot) * Mat4::from_quat(Quat::from_axis_angle(ax.normalize(), r)) * Mat4::from_translation(-pivot))
                } else if let Some(s) = scale {
                    ((s - Vec3::ONE).length() > 1e-6)
                        .then(|| Mat4::from_translation(pivot) * Mat4::from_scale(s) * Mat4::from_translation(-pivot))
                } else {
                    None
                };
                if let Some(mat) = m {
                    let local_m = inv * mat * world;
                    let cols = local_m.to_cols_array();
                    (Some(json!({"transformElements": {"_0": {"rawValue": id}, "_1": sel_json, "by": cols}})), "Transform Elements")
                } else {
                    (None, "")
                }
            }
        }
    };

    let mut m = serde_json::Map::new();
    match cmd {
        Some(c) => {
            m.insert("command".into(), c);
            m.insert("label".into(), Value::String(label.to_string()));
        }
        None => {
            m.insert("command".into(), Value::Null);
        }
    }
    ok_envelope(m)
}

// ── Edit-cage overlay geometry ──

pub fn build_edit_overlay(doc: &mut EngineDoc, dot_radius: f32, dark: bool) -> Option<EditOverlayData> {
    let EditMode::Edit(mode) = doc.edit_mode else {
        doc.last_overlay_key = None;
        doc.last_overlay = None;
        return None;
    };
    let active_id = doc.selection.active.clone()?;
    let scene = doc.previewed_scene();
    let obj = scene.objects.iter().find(|o| o.id == active_id && o.is_visible)?;
    let world = scene_world_matrix(&scene, &active_id);
    let scale = Vec3::new(
        world.x_axis.truncate().length(),
        world.y_axis.truncate().length(),
        world.z_axis.truncate().length(),
    );
    let matrix = world.to_cols_array().to_vec();
    let key = OverlayCacheKey {
        object_id: active_id,
        mesh: obj.mesh.clone(),
        selection: doc.selection.elements.clone(),
        mode,
        dot_radius_bits: dot_radius.to_bits(),
        scale_bits: [scale.x.to_bits(), scale.y.to_bits(), scale.z.to_bits()],
        active_vertex: doc.last_picked_vertex,
        dark,
    };
    if doc.last_overlay_key.as_ref() == Some(&key) {
        if let Some(mut cached) = doc.last_overlay.clone() {
            cached.matrix = matrix;
            return Some(cached);
        }
    }

    doc.overlay_revision += 1;
    let mesh = &obj.mesh;
    let safe_scale = scale.abs().max(Vec3::splat(1e-4));
    let avg_scale = ((safe_scale.x + safe_scale.y + safe_scale.z) / 3.0).max(1e-4);
    let dot = Vec3::splat(dot_radius) / safe_scale;
    let offset = dot_radius * 0.18 / avg_scale;
    let half = dot_radius * 0.28 / avg_scale;
    let selected = expanded_vertices(&doc.selection.elements, mesh);
    let normals = mesh_vertex_normals(mesh);
    let lifted = |i: usize| mesh.positions[i] + normals.get(i).copied().unwrap_or(Vec3::ZERO) * offset;

    let coral: [f32; 4] = [1.0, 107.0 / 255.0, 96.0 / 255.0, 1.0];
    let active_coral: [f32; 4] = [1.0, 161.0 / 255.0, 151.0 / 255.0, 1.0];
    let edge_color: [f32; 4] = if dark { [0.78, 0.78, 0.78, 0.85] } else { [42.0 / 255.0, 42.0 / 255.0, 42.0 / 255.0, 0.85] };
    let vertex_color: [f32; 4] = if dark { [0.86, 0.86, 0.86, 1.0] } else { [48.0 / 255.0, 48.0 / 255.0, 48.0 / 255.0, 1.0] };
    let face_tint: [f32; 4] = [1.0, 107.0 / 255.0, 96.0 / 255.0, 0.28];

    let mut line_positions = Vec::new();
    let mut line_colors = Vec::new();
    let mut line_indices = Vec::new();
    const SIDE_TRIS: [(u32, u32, u32); 8] = [(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)];
    for (a, b) in mesh_unique_edges(mesh) {
        let color = if selected.contains(&(a as usize)) && selected.contains(&(b as usize)) { coral } else { edge_color };
        let p0 = lifted(a as usize);
        let p1 = lifted(b as usize);
        let d = p1 - p0;
        let len = d.length();
        let dir = if len > 1e-9 { d / len } else { Vec3::Z };
        let helper = if dir.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
        let u = helper.cross(dir).normalize() * half;
        let v = dir.cross(u).normalize() * half;
        let base = (line_positions.len() / 3) as u32;
        for p in [p0 + u, p0 + v, p0 - u, p0 - v, p1 + u, p1 + v, p1 - u, p1 - v] {
            line_positions.extend_from_slice(&[p.x, p.y, p.z]);
            line_colors.extend_from_slice(&color);
        }
        for (i0, i1, i2) in SIDE_TRIS {
            line_indices.extend_from_slice(&[base + i0, base + i1, base + i2]);
        }
    }

    let mut dot_positions = Vec::new();
    let mut dot_colors = Vec::new();
    let mut dot_indices = Vec::new();
    if mode == ElementMode::Vertex {
        const DIRS: [Vec3; 6] = [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];
        const TRIS: [(u32, u32, u32); 8] = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)];
        for (i, &p) in mesh.positions.iter().enumerate() {
            let color = if doc.last_picked_vertex == Some(i) && selected.contains(&i) {
                active_coral
            } else if selected.contains(&i) {
                coral
            } else {
                vertex_color
            };
            let base = (dot_positions.len() / 3) as u32;
            for d in DIRS {
                let q = p + d * dot;
                dot_positions.extend_from_slice(&[q.x, q.y, q.z]);
                dot_colors.extend_from_slice(&color);
            }
            for (i0, i1, i2) in TRIS {
                dot_indices.extend_from_slice(&[base + i0, base + i1, base + i2]);
            }
        }
    }

    let mut face_positions = Vec::new();
    let mut face_colors = Vec::new();
    let mut face_indices = Vec::new();
    for fi in expanded_faces(&doc.selection.elements, mesh) {
        let f = &mesh.faces[fi];
        if f.len() < 3 { continue; }
        let n = mesh_face_normal(mesh, fi) * (offset * 0.5);
        let base = (face_positions.len() / 3) as u32;
        for &v in f {
            let p = mesh.positions[v as usize] + n;
            face_positions.extend_from_slice(&[p.x, p.y, p.z]);
            face_colors.extend_from_slice(&face_tint);
        }
        for k in 1..(f.len() - 1) {
            face_indices.extend_from_slice(&[base, base + k as u32, base + k as u32 + 1]);
        }
    }

    let out = EditOverlayData {
        revision: doc.overlay_revision,
        matrix,
        line_positions,
        line_colors,
        line_indices,
        dot_positions,
        dot_colors,
        dot_indices,
        face_positions,
        face_colors,
        face_indices,
    };
    doc.last_overlay_key = Some(key);
    doc.last_overlay = Some(out.clone());
    Some(out)
}
