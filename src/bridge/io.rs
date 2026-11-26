//! Multi-format Export (STL, OBJ+MTL, 3MF, GLB, USDZ), Printability Check & Auto-Fix,
//! and STL/OBJ Import with memory-class limits, simplification, and placement.

use crate::bridge::{
    commands::transform_mesh, evaluate_object_mesh, mesh_triangles, new_uuid, ok_envelope,
    scene_world_matrix, BoundingBox, BridgeError, CommandControl, EngineDoc, SceneSnapshot,
};
use crate::format::{Material, Mesh, MeshSelection, Scene, SceneObject, Selection, Transform};
use glam::{Mat4, Quat, Vec3};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

pub struct ExportFileOut {
    pub file_name: String,
    pub data: Vec<u8>,
}

struct FlatObject {
    name: String,
    mesh: Mesh,
    material: Material,
}

fn sanitize_filename(name: &str, fallback: &str) -> String {
    let cleaned: String = name
        .trim()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' { c } else { '_' })
        .collect();
    let t = cleaned.trim();
    if t.is_empty() { fallback.to_string() } else { t.to_string() }
}

fn flatten_scene(scene: &Scene, ids: Option<&HashSet<String>>, apply_modifiers: bool) -> Vec<FlatObject> {
    let mut out = Vec::new();
    for o in &scene.objects {
        if let Some(wanted) = ids {
            if !wanted.contains(&o.id) { continue; }
        } else if !o.is_visible {
            continue;
        }
        let local = if apply_modifiers { evaluate_object_mesh(o) } else { o.mesh.clone() };
        let world = scene_world_matrix(scene, &o.id);
        let mesh = transform_mesh(&local, world);
        if !mesh.faces.is_empty() {
            out.push(FlatObject { name: o.name.clone(), mesh, material: o.material.clone() });
        }
    }
    out
}

fn merge_flats(flats: &[FlatObject]) -> Mesh {
    let mut positions = Vec::new();
    let mut faces = Vec::new();
    for f in flats {
        let base = positions.len() as u32;
        positions.extend_from_slice(&f.mesh.positions);
        for face in &f.mesh.faces {
            faces.push(face.iter().map(|&i| i + base).collect());
        }
    }
    Mesh { positions, faces, smooth_shading: false }
}

// ── Exports ──

pub fn export_stl(snapshot: &SceneSnapshot, objects_json: Option<&str>) -> Result<Vec<u8>, BridgeError> {
    let ids: Option<HashSet<String>> = objects_json
        .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
        .map(|v| v.into_iter().collect());
    let flats = flatten_scene(&snapshot.scene, ids.as_ref(), true);
    let merged = merge_flats(&flats);
    if merged.faces.is_empty() {
        return Err(BridgeError::new("export.empty"));
    }
    Ok(write_binary_stl(&merged))
}

pub fn export_files(snapshot: &SceneSnapshot, options_json: &str) -> Result<Vec<ExportFileOut>, BridgeError> {
    let v: Value = serde_json::from_str(options_json)
        .map_err(|e| BridgeError::with_args("engine.badJSON", vec![e.to_string()]))?;
    let format = v.get("format").and_then(|x| x.as_str()).unwrap_or("stl");
    let ids: Option<HashSet<String>> = v
        .get("objects")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect());
    let apply_modifiers = v.get("applyModifiers").and_then(|x| x.as_bool()).unwrap_or(true);
    let merge = v.get("merge").and_then(|x| x.as_bool()).unwrap_or(false);
    let fallback = v.get("fallbackFileName").and_then(|x| x.as_str()).unwrap_or("Model");
    let raw_name = v.get("fileName").and_then(|x| x.as_str()).unwrap_or(fallback);
    let base = sanitize_filename(raw_name, fallback);

    let flats = flatten_scene(&snapshot.scene, ids.as_ref(), apply_modifiers);
    if flats.is_empty() {
        return Err(BridgeError::new("export.empty"));
    }

    match format {
        "stl" => {
            let merged = merge_flats(&flats);
            Ok(vec![ExportFileOut { file_name: format!("{base}.stl"), data: write_binary_stl(&merged) }])
        }
        "obj" => {
            let (obj_bytes, mtl_bytes) = write_obj_and_mtl(&flats, &format!("{base}.mtl"), merge);
            Ok(vec![
                ExportFileOut { file_name: format!("{base}.obj"), data: obj_bytes },
                ExportFileOut { file_name: format!("{base}.mtl"), data: mtl_bytes },
            ])
        }
        "3mf" => Ok(vec![ExportFileOut {
            file_name: format!("{base}.3mf"),
            data: write_3mf(&flats, merge),
        }]),
        "glb" => Ok(vec![ExportFileOut {
            file_name: format!("{base}.glb"),
            data: write_glb(&flats, merge),
        }]),
        "usdz" => Ok(vec![ExportFileOut {
            file_name: format!("{base}.usdz"),
            data: write_usdz(&flats, &base, merge),
        }]),
        _ => Err(BridgeError::with_args("engine.badJSON", vec![format.to_string()])),
    }
}

fn write_binary_stl(mesh: &Mesh) -> Vec<u8> {
    let tris = mesh_triangles(mesh);
    let mut out = Vec::with_capacity(84 + tris.len() * 50);
    let mut header = [0u8; 80];
    let tag = b"Sculptor Binary STL (mm, Z-up)";
    header[..tag.len()].copy_from_slice(tag);
    out.extend_from_slice(&header);
    out.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    for t in tris {
        let a = mesh.positions[t[0] as usize];
        let b = mesh.positions[t[1] as usize];
        let c = mesh.positions[t[2] as usize];
        let cr = (b - a).cross(c - a);
        let len = cr.length();
        let n = if len > 1e-12 { cr / len } else { Vec3::Z };
        for v in [n, a, b, c] {
            out.extend_from_slice(&v.x.to_le_bytes());
            out.extend_from_slice(&v.y.to_le_bytes());
            out.extend_from_slice(&v.z.to_le_bytes());
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}

fn write_obj_and_mtl(flats: &[FlatObject], mtl_name: &str, merge: bool) -> (Vec<u8>, Vec<u8>) {
    let mut obj = String::new();
    let mut mtl = String::new();
    obj.push_str("# Exported by Sculptor (mm, Z-up)\n");
    obj.push_str(&format!("mtllib {mtl_name}\n"));
    let mut v_offset = 1usize;
    let items: Vec<FlatObject> = if merge {
        vec![FlatObject {
            name: "Model".into(),
            mesh: merge_flats(flats),
            material: flats[0].material.clone(),
        }]
    } else {
        flats
            .iter()
            .map(|f| FlatObject { name: f.name.clone(), mesh: f.mesh.clone(), material: f.material.clone() })
            .collect()
    };
    for (idx, item) in items.iter().enumerate() {
        let mat_id = format!("mat_{idx}");
        let c = item.material.base_color;
        mtl.push_str(&format!(
            "newmtl {mat_id}\nKd {:.4} {:.4} {:.4}\nd {:.4}\nPm {:.4}\nPr {:.4}\n\n",
            c.r, c.g, c.b, c.a, item.material.metallic, item.material.roughness
        ));
        obj.push_str(&format!("o {}\nusemtl {mat_id}\n", item.name.replace(' ', "_")));
        for &p in &item.mesh.positions {
            // Convert Z-up to standard OBJ Y-up (x, z, -y)
            obj.push_str(&format!("v {:.5} {:.5} {:.5}\n", p.x, p.z, -p.y));
        }
        for f in &item.mesh.faces {
            if f.len() < 3 { continue; }
            obj.push('f');
            for &vi in f {
                obj.push_str(&format!(" {}", vi as usize + v_offset));
            }
            obj.push('\n');
        }
        v_offset += item.mesh.positions.len();
    }
    (obj.into_bytes(), mtl.into_bytes())
}

fn write_3mf(flats: &[FlatObject], merge: bool) -> Vec<u8> {
    let merged = if merge { Some(merge_flats(flats)) } else { None };
    let mut model = String::from(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
<resources>
"#,
    );
    let meshes: Vec<(&str, &Mesh)> = if let Some(ref m) = merged {
        vec![("Model", m)]
    } else {
        flats.iter().map(|f| (f.name.as_str(), &f.mesh)).collect()
    };
    for (idx, (name, mesh)) in meshes.iter().enumerate() {
        let oid = idx + 1;
        let esc_name = name.replace('&', "&amp;").replace('<', "&lt;").replace('"', "&quot;");
        model.push_str(&format!("<object id=\"{oid}\" name=\"{esc_name}\" type=\"model\"><mesh><vertices>\n"));
        for &p in &mesh.positions {
            model.push_str(&format!("<vertex x=\"{:.5}\" y=\"{:.5}\" z=\"{:.5}\"/>\n", p.x, p.y, p.z));
        }
        model.push_str("</vertices><triangles>\n");
        for t in mesh_triangles(mesh) {
            model.push_str(&format!("<triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>\n", t[0], t[1], t[2]));
        }
        model.push_str("</triangles></mesh></object>\n");
    }
    model.push_str("</resources><build>\n");
    for idx in 0..meshes.len() {
        model.push_str(&format!("<item objectid=\"{}\"/>\n", idx + 1));
    }
    model.push_str("</build></model>\n");

    let content_types = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"#;
    let rels = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"#;

    build_store_zip(
        &[
            ("[Content_Types].xml", content_types.as_bytes()),
            ("_rels/.rels", rels.as_bytes()),
            ("3D/3dmodel.model", model.as_bytes()),
        ],
        false,
    )
}

fn write_glb(flats: &[FlatObject], merge: bool) -> Vec<u8> {
    let mut bin = Vec::<u8>::new();
    let mut buffer_views = Vec::<Value>::new();
    let mut accessors = Vec::<Value>::new();
    let mut materials = Vec::<Value>::new();
    let mut meshes = Vec::<Value>::new();
    let mut nodes = Vec::<Value>::new();

    let items: Vec<FlatObject> = if merge {
        vec![FlatObject { name: "Model".into(), mesh: merge_flats(flats), material: flats[0].material.clone() }]
    } else {
        flats.iter().map(|f| FlatObject { name: f.name.clone(), mesh: f.mesh.clone(), material: f.material.clone() }).collect()
    };

    for (idx, item) in items.iter().enumerate() {
        let tris = mesh_triangles(&item.mesh);
        if tris.is_empty() { continue; }
        let mut pos_bytes = Vec::with_capacity(item.mesh.positions.len() * 12);
        let mut pmin = Vec3::splat(f32::INFINITY);
        let mut pmax = Vec3::splat(f32::NEG_INFINITY);
        for &p in &item.mesh.positions {
            // glTF is metres, Y-up: (x, z, -y) * 0.001
            let g = Vec3::new(p.x * 0.001, p.z * 0.001, -p.y * 0.001);
            pmin = pmin.min(g);
            pmax = pmax.max(g);
            pos_bytes.extend_from_slice(&g.x.to_le_bytes());
            pos_bytes.extend_from_slice(&g.y.to_le_bytes());
            pos_bytes.extend_from_slice(&g.z.to_le_bytes());
        }
        let mut idx_bytes = Vec::with_capacity(tris.len() * 12);
        for t in &tris {
            for &vi in t { idx_bytes.extend_from_slice(&vi.to_le_bytes()); }
        }
        let pos_off = bin.len();
        bin.extend_from_slice(&pos_bytes);
        while bin.len() % 4 != 0 { bin.push(0); }
        let idx_off = bin.len();
        bin.extend_from_slice(&idx_bytes);
        while bin.len() % 4 != 0 { bin.push(0); }

        let pos_bv = buffer_views.len();
        buffer_views.push(json!({"buffer": 0, "byteOffset": pos_off, "byteLength": pos_bytes.len(), "target": 34962}));
        let idx_bv = buffer_views.len();
        buffer_views.push(json!({"buffer": 0, "byteOffset": idx_off, "byteLength": idx_bytes.len(), "target": 34963}));

        let pos_acc = accessors.len();
        accessors.push(json!({
            "bufferView": pos_bv, "componentType": 5126, "count": item.mesh.positions.len(), "type": "VEC3",
            "min": [pmin.x, pmin.y, pmin.z], "max": [pmax.x, pmax.y, pmax.z]
        }));
        let idx_acc = accessors.len();
        accessors.push(json!({
            "bufferView": idx_bv, "componentType": 5125, "count": tris.len() * 3, "type": "SCALAR"
        }));

        let c = item.material.base_color;
        let mat_idx = materials.len();
        materials.push(json!({
            "name": item.material.name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [c.r, c.g, c.b, c.a],
                "metallicFactor": item.material.metallic,
                "roughnessFactor": item.material.roughness
            }
        }));
        let mesh_idx = meshes.len();
        meshes.push(json!({
            "name": item.name,
            "primitives": [{"attributes": {"POSITION": pos_acc}, "indices": idx_acc, "material": mat_idx}]
        }));
        nodes.push(json!({"name": item.name, "mesh": mesh_idx}));
        let _ = idx;
    }

    let root_nodes: Vec<usize> = (0..nodes.len()).collect();
    let gltf = json!({
        "asset": {"version": "2.0", "generator": "Sculptor Rust Engine"},
        "scene": 0,
        "scenes": [{"nodes": root_nodes}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": bin.len()}]
    });
    let mut json_bytes = gltf.to_string().into_bytes();
    while json_bytes.len() % 4 != 0 { json_bytes.push(b' '); }
    while bin.len() % 4 != 0 { bin.push(0); }

    let total_len = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total_len as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // JSON
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004E4942u32.to_le_bytes()); // BIN
    out.extend_from_slice(&bin);
    out
}

fn write_usdz(flats: &[FlatObject], base: &str, merge: bool) -> Vec<u8> {
    let mut usda = String::from(
        "#usda 1.0\n(\n    defaultPrim = \"Scene\"\n    metersPerUnit = 0.001\n    upAxis = \"Y\"\n)\ndef Xform \"Scene\" {\n",
    );
    let items: Vec<FlatObject> = if merge {
        vec![FlatObject { name: "Model".into(), mesh: merge_flats(flats), material: flats[0].material.clone() }]
    } else {
        flats.iter().map(|f| FlatObject { name: f.name.clone(), mesh: f.mesh.clone(), material: f.material.clone() }).collect()
    };
    for (idx, item) in items.iter().enumerate() {
        let tris = mesh_triangles(&item.mesh);
        if tris.is_empty() { continue; }
        let pts: Vec<String> = item.mesh.positions.iter().map(|p| format!("({:.4}, {:.4}, {:.4})", p.x, p.z, -p.y)).collect();
        let counts: Vec<&str> = vec!["3"; tris.len()];
        let indices: Vec<String> = tris.iter().flat_map(|t| t.iter().map(|v| v.to_string())).collect();
        usda.push_str(&format!(
            "    def Mesh \"Mesh_{idx}\" {{\n        int[] faceVertexCounts = [{}]\n        int[] faceVertexIndices = [{}]\n        point3f[] points = [{}]\n    }}\n",
            counts.join(", "),
            indices.join(", "),
            pts.join(", ")
        ));
    }
    usda.push_str("}\n");
    build_store_zip(&[(&format!("{base}.usda"), usda.as_bytes())], true)
}

/// Minimal uncompressed ZIP archive (STORE method; optional 64-byte alignment for USDZ).
fn build_store_zip(entries: &[(&str, &[u8])], align_64: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for &(name, data) in entries {
        let name_bytes = name.as_bytes();
        let crc = crc32(data);
        let offset = out.len() as u32;
        let extra_len: u16 = if align_64 {
            let header_end = out.len() + 30 + name_bytes.len();
            let rem = header_end % 64;
            if rem == 0 { 0 } else { (64 - rem) as u16 }
        } else {
            0
        };
        // Local file header
        out.extend_from_slice(&0x04034B50u32.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // STORE
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(&extra_len.to_le_bytes());
        out.extend_from_slice(name_bytes);
        if extra_len > 0 {
            out.resize(out.len() + extra_len as usize, 0);
        }
        out.extend_from_slice(data);

        // Central directory header
        central.extend_from_slice(&0x02014B50u32.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u32.to_le_bytes());
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name_bytes);
    }
    let cd_offset = out.len() as u32;
    let cd_size = central.len() as u32;
    out.extend_from_slice(&central);
    // EOCD
    out.extend_from_slice(&0x06054B50u32.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

// ── Printability check & auto-fix ──

const PLATE_MM: [f32; 3] = [220.0, 220.0, 250.0];

pub fn print_check(snapshot: &SceneSnapshot, options_json: &str) -> String {
    let v: Value = serde_json::from_str(options_json).unwrap_or(json!({}));
    let ids: Option<HashSet<String>> = v
        .get("objects")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect());
    let apply_modifiers = v.get("applyModifiers").and_then(|x| x.as_bool()).unwrap_or(true);
    let flats = flatten_scene(&snapshot.scene, ids.as_ref(), apply_modifiers);
    let merged = merge_flats(&flats);
    let tris = mesh_triangles(&merged);
    let bounds = BoundingBox::from_mesh(&merged, Mat4::IDENTITY);

    let mut area = 0.0f64;
    let mut vol6 = 0.0f64;
    let mut edge_counts: HashMap<(u32, u32), usize> = HashMap::new();
    let mut degenerate = 0usize;
    for t in &tris {
        let a = merged.positions[t[0] as usize].as_dvec3();
        let b = merged.positions[t[1] as usize].as_dvec3();
        let c = merged.positions[t[2] as usize].as_dvec3();
        let cr = (b - a).cross(c - a);
        let a2 = cr.length();
        if a2 < 1e-8 { degenerate += 1; }
        area += 0.5 * a2;
        vol6 += a.dot(b.cross(c));
        for &(u, w) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let key = if u < w { (u, w) } else { (w, u) };
            *edge_counts.entry(key).or_insert(0) += 1;
        }
    }
    let open_edges = edge_counts.values().filter(|&&c| c == 1).count();
    let non_manifold = edge_counts.values().filter(|&&c| c > 2).count();
    let volume = (vol6 / 6.0).abs() as f32;
    let sz = bounds.size();

    let mut issues = Vec::<Value>::new();
    if open_edges > 0 {
        issues.push(json!({"kind": "notWatertight", "severity": "error", "canAutoFix": false, "key": "print.notWatertight", "args": [open_edges.to_string()]}));
    }
    if non_manifold > 0 {
        issues.push(json!({"kind": "nonManifoldEdges", "severity": "error", "canAutoFix": false, "key": "print.nonManifold", "args": [non_manifold.to_string()]}));
    }
    if vol6 < -1e-4 {
        issues.push(json!({"kind": "invertedNormals", "severity": "error", "canAutoFix": true, "key": "print.invertedNormals", "args": []}));
    }
    if degenerate > 0 {
        issues.push(json!({"kind": "degenerateFaces", "severity": "warning", "canAutoFix": false, "key": "print.degenerateFaces", "args": [degenerate.to_string()]}));
    }
    if !bounds.is_empty() && bounds.min.z < -0.05 {
        issues.push(json!({"kind": "belowGround", "severity": "warning", "canAutoFix": true, "key": "print.belowGround", "args": [format!("{:.1}", -bounds.min.z)]}));
    }
    if !bounds.is_empty() && (sz.x > PLATE_MM[0] || sz.y > PLATE_MM[1] || sz.z > PLATE_MM[2]) {
        issues.push(json!({"kind": "exceedsBuildPlate", "severity": "warning", "canAutoFix": false, "key": "print.exceedsPlate", "args": []}));
    }
    if !bounds.is_empty() && bounds.max_dimension() < 1.0 {
        issues.push(json!({"kind": "tiny", "severity": "warning", "canAutoFix": false, "key": "print.tiny", "args": []}));
    }

    let printable = !issues.iter().any(|i| i["severity"] == "error") && !tris.is_empty();
    let bjson = if bounds.is_empty() {
        Value::Null
    } else {
        json!({"min": [bounds.min.x, bounds.min.y, bounds.min.z], "max": [bounds.max.x, bounds.max.y, bounds.max.z]})
    };
    json!({
        "issues": issues,
        "bounds": bjson,
        "triangles": tris.len(),
        "volume": volume,
        "area": area as f32,
        "printable": printable,
        "plate": PLATE_MM,
        "revision": snapshot.revision,
    })
    .to_string()
}

pub fn fix_plan(snapshot: &SceneSnapshot, options_json: &str) -> String {
    let v: Value = serde_json::from_str(options_json).unwrap_or(json!({}));
    let ids: Option<HashSet<String>> = v
        .get("objects")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect());
    let apply_modifiers = v.get("applyModifiers").and_then(|x| x.as_bool()).unwrap_or(true);

    let mut commands = Vec::<Value>::new();
    let mut repaired = 0usize;
    let mut min_z = f32::INFINITY;
    let mut target_ids = Vec::<String>::new();

    for o in &snapshot.scene.objects {
        if let Some(ref wanted) = ids {
            if !wanted.contains(&o.id) { continue; }
        } else if !o.is_visible {
            continue;
        }
        target_ids.push(o.id.clone());
        let local = if apply_modifiers { evaluate_object_mesh(o) } else { o.mesh.clone() };
        let world = scene_world_matrix(&snapshot.scene, &o.id);
        let wm = transform_mesh(&local, world);
        let b = BoundingBox::from_mesh(&wm, Mat4::IDENTITY);
        if !b.is_empty() { min_z = min_z.min(b.min.z); }
        let mut vol6 = 0.0f64;
        for t in mesh_triangles(&wm) {
            let a = wm.positions[t[0] as usize].as_dvec3();
            let b = wm.positions[t[1] as usize].as_dvec3();
            let c = wm.positions[t[2] as usize].as_dvec3();
            vol6 += a.dot(b.cross(c));
        }
        if vol6 < -1e-4 {
            commands.push(json!({"recalculateNormals": {"_0": {"rawValue": o.id}}}));
            repaired += 1;
        }
    }
    let mut moved = false;
    if min_z.is_finite() && min_z < -0.05 && !target_ids.is_empty() {
        let refs: Vec<Value> = target_ids.iter().map(|id| json!({"rawValue": id})).collect();
        commands.push(json!({"translate": {"_0": refs, "by": [0.0, 0.0, -min_z]}}));
        moved = true;
    }
    json!({
        "commands": commands,
        "repaired": repaired,
        "moved": moved,
        "revision": snapshot.revision,
    })
    .to_string()
}

// ── Import (STL / OBJ) ──

#[derive(Debug, Clone)]
pub struct ImportLimits {
    pub memory_class: &'static str,
    pub undo_budget_bytes: i64,
    pub max_triangles: usize,
    pub simplify_ceiling: usize,
    pub max_file_bytes: i64,
}

impl ImportLimits {
    pub fn for_device(physical: i64, _available: i64) -> Self {
        let gb = physical / (1024 * 1024 * 1024);
        if gb >= 8 {
            Self { memory_class: "high", undo_budget_bytes: 384 * 1024 * 1024, max_triangles: 500_000, simplify_ceiling: 3_000_000, max_file_bytes: 256 * 1024 * 1024 }
        } else if gb >= 4 {
            Self { memory_class: "standard", undo_budget_bytes: 192 * 1024 * 1024, max_triangles: 250_000, simplify_ceiling: 1_500_000, max_file_bytes: 128 * 1024 * 1024 }
        } else {
            Self { memory_class: "constrained", undo_budget_bytes: 96 * 1024 * 1024, max_triangles: 100_000, simplify_ceiling: 600_000, max_file_bytes: 64 * 1024 * 1024 }
        }
    }
    pub fn to_json(&self) -> String {
        json!({
            "memoryClass": self.memory_class,
            "undoBudgetBytes": self.undo_budget_bytes,
            "maxTriangles": self.max_triangles,
            "simplifyCeiling": self.simplify_ceiling,
            "maxFileBytes": self.max_file_bytes,
        })
        .to_string()
    }
}

pub struct LoadedImport {
    pub file_name: String,
    pub parts: Vec<(String, Mesh)>,
    pub notes: Vec<String>,
}

pub struct PreparedImport {
    pub parts: Vec<(String, Mesh, Vec3)>,
    pub notes: Vec<Vec<String>>,
    pub triangles: usize,
}

pub fn import_load(
    job: Option<&CommandControl>,
    file_name: &str,
    bytes: &[u8],
    physical_memory: i64,
    available_memory: i64,
    default_name: &str,
) -> Result<(LoadedImport, Value), BridgeError> {
    let limits = ImportLimits::for_device(physical_memory, available_memory);
    if bytes.len() as i64 > limits.max_file_bytes {
        return Err(BridgeError::with_args(
            "import.fileTooLarge",
            vec![file_name.into(), bytes.len().to_string(), limits.max_file_bytes.to_string()],
        ));
    }
    let (base, ext) = match file_name.rfind('.') {
        Some(i) if i > 0 => (&file_name[..i], file_name[i + 1..].to_ascii_lowercase()),
        _ => (file_name, String::new()),
    };
    let display = if base.trim().is_empty() { default_name.to_string() } else { base.trim().to_string() };
    let mut notes = Vec::new();
    let parts = match ext.as_str() {
        "stl" => {
            let mesh = parse_stl(bytes, job).map_err(|_| BridgeError::with_args("import.unreadable", vec![file_name.into()]))?;
            vec![(display, mesh)]
        }
        "obj" => {
            let text = String::from_utf8_lossy(bytes);
            let mut raw = parse_obj(&text, &display);
            let rot = Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);
            for (_, m) in &mut raw { *m = transform_mesh(m, rot); }
            notes.push("import.note.yUp".to_string());
            raw
        }
        _ => return Err(BridgeError::with_args("import.unsupported", vec![file_name.into()])),
    };
    let parts: Vec<(String, Mesh)> = parts.into_iter().filter(|(_, m)| !m.faces.is_empty()).collect();
    if parts.is_empty() {
        return Err(BridgeError::with_args("import.empty", vec![file_name.into()]));
    }
    let tris: usize = parts.iter().map(|(_, m)| crate::bridge::mesh_triangle_count(m)).sum();
    if tris > limits.simplify_ceiling {
        return Err(BridgeError::with_args(
            "import.tooManyTriangles",
            vec![file_name.into(), tris.to_string(), limits.simplify_ceiling.to_string()],
        ));
    }
    let verdict = if tris <= limits.max_triangles / 2 {
        "importDirectly"
    } else if tris <= limits.max_triangles {
        "ask"
    } else {
        "simplifyOnly"
    };
    let simplify_target = (limits.max_triangles / 2).min(tris);
    if let Some(j) = job { j.set_progress(1.0); }
    let info = json!({
        "triangles": tris,
        "verdict": verdict,
        "simplifyTarget": simplify_target,
        "maxTriangles": limits.max_triangles,
        "simplifyCeiling": limits.simplify_ceiling,
        "fileName": file_name,
    });
    Ok((LoadedImport { file_name: file_name.to_string(), parts, notes }, info))
}

pub fn import_prepare(
    job: Option<&CommandControl>,
    loaded: &LoadedImport,
    target: i32,
    default_name: &str,
) -> Result<PreparedImport, BridgeError> {
    let total: usize = loaded.parts.iter().map(|(_, m)| crate::bridge::mesh_triangle_count(m)).sum();
    let mut notes: Vec<Vec<String>> = loaded.notes.iter().map(|n| vec![n.clone()]).collect();
    let mut raw = Vec::new();
    for (name, mesh) in &loaded.parts {
        if job.is_some_and(|j| j.is_cancelled()) {
            return Err(BridgeError::new("import.cancelled"));
        }
        let m = if target > 0 && total > target as usize {
            decimate_grid(mesh, (target as usize).max(4))
        } else {
            mesh.clone()
        };
        raw.push((name.clone(), m));
    }
    if target > 0 && total > target as usize {
        notes.push(vec!["import.note.simplified".into(), total.to_string()]);
    }
    let mut combined = BoundingBox::EMPTY;
    for (_, m) in &raw {
        combined = combined.union(BoundingBox::from_mesh(m, Mat4::IDENTITY));
    }
    let mut scale = 1.0f32;
    if combined.max_dimension() > 0.0 && combined.max_dimension() < 1.0 {
        scale = 1000.0;
        notes.push(vec!["import.note.metres".into()]);
    }
    let anchor = Vec3::new(combined.center().x, combined.center().y, combined.min.z) * scale;
    let mut parts = Vec::new();
    let mut tris = 0usize;
    for (name, mut mesh) in raw {
        let b = BoundingBox::from_mesh(&mesh, Mat4::IDENTITY);
        let pivot = Vec3::new(b.center().x, b.center().y, b.min.z) * scale;
        for p in &mut mesh.positions { *p = *p * scale - pivot; }
        tris += crate::bridge::mesh_triangle_count(&mesh);
        let clean = if name.trim().is_empty() { default_name.to_string() } else { name.trim().to_string() };
        parts.push((clean, mesh, pivot - anchor));
    }
    if let Some(j) = job { j.set_progress(1.0); }
    Ok(PreparedImport { parts, notes, triangles: tris })
}

pub fn commit_import(doc: &mut EngineDoc, prepared: &PreparedImport, as_new_document: bool, label: &str) -> String {
    let mut objs = Vec::new();
    let mut created = Vec::new();
    for (name, mesh, pos) in &prepared.parts {
        let id = new_uuid();
        created.push(id.clone());
        objs.push(SceneObject {
            id,
            name: name.clone(),
            mesh: mesh.clone(),
            transform: Transform { position: *pos, rotation: Quat::IDENTITY, scale: Vec3::ONE },
            material: Material::default(),
            modifiers: Vec::new(),
            parent: None,
            is_visible: true,
            is_locked: false,
            primitive_kind: None,
        });
    }
    if as_new_document {
        let sel = Selection { objects: created.clone(), active: created.last().cloned(), elements: MeshSelection::default() };
        doc.replace_scene_fresh(Scene { objects: objs }, sel, None);
    } else {
        doc.edit_mode = crate::bridge::EditMode::Object;
        let mut next = doc.scene.clone();
        next.objects.extend(objs);
        doc.record_scene_change(label.to_string(), next, &created, None);
    }
    let mut m = serde_json::Map::new();
    m.insert("created".into(), json!(created));
    m.insert("notes".into(), json!(prepared.notes));
    ok_envelope(m)
}

fn parse_stl(bytes: &[u8], job: Option<&CommandControl>) -> Result<Mesh, ()> {
    if bytes.len() >= 84 {
        let count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        if 84 + count.saturating_mul(50) == bytes.len() && count > 0 {
            let mut positions = Vec::with_capacity(count * 3);
            let mut faces = Vec::with_capacity(count);
            let mut vmap: HashMap<[u32; 3], u32> = HashMap::with_capacity(count);
            let mut off = 84;
            for i in 0..count {
                if i % 4096 == 0 && job.is_some_and(|j| j.is_cancelled()) { return Err(()); }
                off += 12; // skip normal
                let mut tri = [0u32; 3];
                for slot in &mut tri {
                    let x = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
                    let y = f32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
                    let z = f32::from_le_bytes(bytes[off + 8..off + 12].try_into().unwrap());
                    off += 12;
                    let key = [(x * 1e4).round() as i32 as u32, (y * 1e4).round() as i32 as u32, (z * 1e4).round() as i32 as u32];
                    *slot = *vmap.entry(key).or_insert_with(|| {
                        let id = positions.len() as u32;
                        positions.push(Vec3::new(x, y, z));
                        id
                    });
                }
                off += 2;
                if tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0] {
                    faces.push(tri.to_vec());
                }
            }
            return Ok(Mesh { positions, faces, smooth_shading: false });
        }
    }
    // ASCII STL
    let text = std::str::from_utf8(bytes).map_err(|_| ())?;
    let mut positions = Vec::new();
    let mut faces = Vec::new();
    let mut cur = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("vertex ") {
            let nums: Vec<f32> = rest.split_whitespace().filter_map(|w| w.parse().ok()).collect();
            if nums.len() >= 3 {
                let id = positions.len() as u32;
                positions.push(Vec3::new(nums[0], nums[1], nums[2]));
                cur.push(id);
                if cur.len() == 3 {
                    faces.push(std::mem::take(&mut cur));
                }
            }
        }
    }
    if faces.is_empty() { Err(()) } else { Ok(Mesh { positions, faces, smooth_shading: false }) }
}

fn parse_obj(text: &str, default_name: &str) -> Vec<(String, Mesh)> {
    let mut all_pos = Vec::new();
    let mut cur_name = default_name.to_string();
    let mut cur_faces: Vec<Vec<u32>> = Vec::new();
    let mut raw_parts: Vec<(String, Vec<Vec<u32>>)> = Vec::new();

    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') { continue; }
        let mut it = t.split_whitespace();
        match it.next() {
            Some("v") => {
                let nums: Vec<f32> = it.filter_map(|w| w.parse().ok()).collect();
                if nums.len() >= 3 { all_pos.push(Vec3::new(nums[0], nums[1], nums[2])); }
            }
            Some("o") | Some("g") => {
                if !cur_faces.is_empty() {
                    raw_parts.push((cur_name.clone(), std::mem::take(&mut cur_faces)));
                }
                let rest = it.collect::<Vec<_>>().join(" ");
                if !rest.is_empty() { cur_name = rest; }
            }
            Some("f") => {
                let mut f = Vec::new();
                for tok in it {
                    if let Some(first) = tok.split('/').next() {
                        if let Ok(idx) = first.parse::<i64>() {
                            let resolved = if idx > 0 { (idx - 1) as usize } else { all_pos.len().saturating_sub((-idx) as usize) };
                            if resolved < all_pos.len() { f.push(resolved as u32); }
                        }
                    }
                }
                if f.len() >= 3 { cur_faces.push(f); }
            }
            _ => {}
        }
    }
    if !cur_faces.is_empty() {
        raw_parts.push((cur_name, cur_faces));
    }
    raw_parts
        .into_iter()
        .map(|(name, faces)| {
            let mut remap: HashMap<u32, u32> = HashMap::new();
            let mut positions = Vec::new();
            let mut new_faces = Vec::with_capacity(faces.len());
            for f in faces {
                let nf: Vec<u32> = f
                    .into_iter()
                    .map(|vi| {
                        *remap.entry(vi).or_insert_with(|| {
                            let id = positions.len() as u32;
                            positions.push(all_pos[vi as usize]);
                            id
                        })
                    })
                    .collect();
                new_faces.push(nf);
            }
            (name, Mesh { positions, faces: new_faces, smooth_shading: false })
        })
        .collect()
}

/// Fast vertex-clustering decimation when an imported mesh exceeds target triangle budget.
fn decimate_grid(mesh: &Mesh, target_tris: usize) -> Mesh {
    let cur = crate::bridge::mesh_triangle_count(mesh);
    if cur <= target_tris || mesh.positions.is_empty() {
        return mesh.clone();
    }
    let b = BoundingBox::from_mesh(mesh, Mat4::IDENTITY);
    let ratio = (target_tris as f32 / cur as f32).sqrt().clamp(0.05, 0.95);
    let cells = ((mesh.positions.len() as f32).cbrt() * ratio * 1.5).max(8.0);
    let step = (b.max_dimension() / cells).max(1e-4);
    let mut cell_map: HashMap<(i32, i32, i32), (Vec3, u32, u32)> = HashMap::new();
    let mut v_to_cell = vec![0u32; mesh.positions.len()];
    let mut next_id = 0u32;
    for (i, &p) in mesh.positions.iter().enumerate() {
        let key = (
            ((p.x - b.min.x) / step).floor() as i32,
            ((p.y - b.min.y) / step).floor() as i32,
            ((p.z - b.min.z) / step).floor() as i32,
        );
        let entry = cell_map.entry(key).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            (Vec3::ZERO, 0, id)
        });
        entry.0 += p;
        entry.1 += 1;
        v_to_cell[i] = entry.2;
    }
    let mut positions = vec![Vec3::ZERO; next_id as usize];
    for (sum, cnt, id) in cell_map.values() {
        positions[*id as usize] = *sum / (*cnt as f32);
    }
    let mut faces = Vec::new();
    for f in &mesh.faces {
        let mut nf: Vec<u32> = f.iter().map(|&v| v_to_cell[v as usize]).collect();
        nf.dedup();
        if nf.len() > 1 && nf.first() == nf.last() { nf.pop(); }
        if nf.len() >= 3 { faces.push(nf); }
    }
    Mesh { positions, faces, smooth_shading: mesh.smooth_shading }
}
