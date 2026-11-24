//! The `.sculpt` project format (Sculptor `ProjectFile`, format version 1).
//!
//! Decoding accepts what Swift's `ProjectFile.decode` accepts and applies the same repairs;
//! encoding writes the same bytes as Swift's `ProjectFile.encode`.

pub mod json;
mod de;

use glam::{Quat, Vec3};
use json::J;

pub const CURRENT_FORMAT_VERSION: i64 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    /// "This file isn't a Sculptor project, or it is damaged."
    NotAProject,
    NewerVersion(i64),
    UnsupportedVersion(i64),
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatError::NotAProject => write!(f, "This file isn't a Sculptor project, or it is damaged."),
            FormatError::NewerVersion(v) => write!(f, "This project was made with a newer version of Sculptor (format {v}). Update the app to open it."),
            FormatError::UnsupportedVersion(v) => write!(f, "This project uses an old format ({v}) that can't be opened."),
        }
    }
}
impl std::error::Error for FormatError {}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectFile {
    pub format_version: i64,
    pub scene: Scene,
    pub selection: Selection,
    /// Base64 of the viewport's `OrbitCamera` JSON, kept as written.
    pub camera_state: Option<String>,
    /// ISO 8601, whole seconds, UTC (`2026-09-21T14:13:20Z`).
    pub created_at: String,
    pub modified_at: String,
    pub app_version: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Scene {
    pub objects: Vec<SceneObject>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SceneObject {
    /// Uppercase UUID string, as Swift's `UUID.uuidString`.
    pub id: String,
    pub name: String,
    pub mesh: Mesh,
    pub transform: Transform,
    pub material: Material,
    pub modifiers: Vec<ModifierEntry>,
    pub parent: Option<String>,
    pub is_visible: bool,
    pub is_locked: bool,
    pub primitive_kind: Option<String>,
}

/// Indexed n-gon mesh, faces counter-clockwise seen from outside. Indices are always valid.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub faces: Vec<Vec<u32>>,
    pub smooth_shading: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Transform { position: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE }
    }
}

impl Transform {
    /// T · R · S, as Sculptor's `Transform.matrix`.
    pub fn matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }
    fn is_finite(&self) -> bool {
        self.position.is_finite() && self.scale.is_finite() && self.rotation.is_finite()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    pub name: String,
    /// sRGB, 0 to 1.
    pub base_color: Color,
    pub metallic: f32,
    pub roughness: f32,
}

impl Default for Material {
    fn default() -> Self {
        Material { name: "Clay".into(), base_color: Color { r: 233.0 / 255.0, g: 201.0 / 255.0, b: 168.0 / 255.0, a: 1.0 }, metallic: 0.0, roughness: 0.6 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModifierEntry {
    pub id: String,
    pub modifier: Modifier,
    pub is_enabled: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Modifier {
    Mirror { axis: i64, merge: bool, merge_distance: f32 },
    Subdivision { levels: i64 },
    Array { count: i64, relative_offset: Vec3, constant_offset: Vec3, use_relative: bool },
    Solidify { thickness: f32, offset: f32 },
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Selection {
    /// Sorted, unique.
    pub objects: Vec<String>,
    pub active: Option<String>,
    pub elements: MeshSelection,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MeshSelection {
    pub vertices: Vec<i64>,
    pub edges: Vec<(i64, i64)>,
    pub faces: Vec<i64>,
}

impl ProjectFile {
    /// Decodes a project, rejecting newer and unknown formats and repairing damaged content.
    pub fn decode(bytes: &[u8]) -> Result<ProjectFile, FormatError> {
        de::decode(bytes)
    }

    /// Pretty, key sorted JSON, byte identical to Swift's `ProjectFile.encode()`.
    /// Streams mesh arrays directly to avoid building millions of temporary AST nodes.
    pub fn encode(&self) -> String {
        let est: usize = 2048
            + self
                .scene
                .objects
                .iter()
                .map(|o| o.mesh.positions.len() * 64 + o.mesh.faces.len() * 96 + 512)
                .sum::<usize>();
        let mut out = String::with_capacity(est);
        out.push_str("{\n  \"appVersion\" : ");
        json::string(&self.app_version, &mut out);
        if let Some(c) = &self.camera_state {
            out.push_str(",\n  \"cameraState\" : ");
            json::string(c, &mut out);
        }
        out.push_str(",\n  \"createdAt\" : ");
        json::string(&self.created_at, &mut out);
        out.push_str(",\n  \"formatVersion\" : ");
        json::write_i64(self.format_version, &mut out);
        out.push_str(",\n  \"modifiedAt\" : ");
        json::string(&self.modified_at, &mut out);
        out.push_str(",\n  \"scene\" : {\n    \"objects\" : ");
        json::container(&mut out, Some(2), '[', ']', self.scene.objects.len(), |out, i, l| {
            write_object_pretty(&self.scene.objects[i], out, l.unwrap());
        });
        out.push_str("\n  },\n  \"selection\" : ");
        json::write_value(&selection_json(&self.selection), &mut out, Some(1));
        out.push_str("\n}");
        out
    }
}

fn write_mesh_stream(m: &Mesh, out: &mut String, level: Option<usize>) {
    match level {
        None => {
            out.push_str("{\"faces\":[");
            for (fi, f) in m.faces.iter().enumerate() {
                if fi > 0 {
                    out.push(',');
                }
                out.push('[');
                for (ii, &idx) in f.iter().enumerate() {
                    if ii > 0 {
                        out.push(',');
                    }
                    json::write_u32(idx, out);
                }
                out.push(']');
            }
            out.push_str("],\"positions\":[");
            for (pi, p) in m.positions.iter().enumerate() {
                if pi > 0 {
                    out.push(',');
                }
                json::swift_float(p.x, out);
                out.push(',');
                json::swift_float(p.y, out);
                out.push(',');
                json::swift_float(p.z, out);
            }
            out.push_str("],\"smoothShading\":");
            out.push_str(if m.smooth_shading { "true}" } else { "false}" });
        }
        Some(l) => {
            out.push_str("{\n");
            json::indent(out, l + 1);
            out.push_str("\"faces\" : ");
            json::container(out, Some(l + 1), '[', ']', m.faces.len(), |out, fi, fl| {
                let f = &m.faces[fi];
                let fl = fl.unwrap();
                json::container(out, Some(fl), '[', ']', f.len(), |out, ii, _| {
                    json::write_u32(f[ii], out);
                });
            });
            out.push_str(",\n");
            json::indent(out, l + 1);
            out.push_str("\"positions\" : [");
            out.push('\n');
            let npos = m.positions.len();
            if npos == 0 {
                out.push('\n');
            } else {
                for (pi, p) in m.positions.iter().enumerate() {
                    let last_p = pi + 1 == npos;
                    json::indent(out, l + 2);
                    json::swift_float(p.x, out);
                    out.push_str(",\n");
                    json::indent(out, l + 2);
                    json::swift_float(p.y, out);
                    out.push_str(",\n");
                    json::indent(out, l + 2);
                    json::swift_float(p.z, out);
                    out.push_str(if last_p { "\n" } else { ",\n" });
                }
            }
            json::indent(out, l + 1);
            out.push_str("],\n");
            json::indent(out, l + 1);
            out.push_str("\"smoothShading\" : ");
            out.push_str(if m.smooth_shading { "true\n" } else { "false\n" });
            json::indent(out, l);
            out.push('}');
        }
    }
}

fn write_object_pretty(o: &SceneObject, out: &mut String, l: usize) {
    let t = &o.transform;
    out.push_str("{\n");
    // Sorted keys: id, isLocked, isVisible, material, mesh, modifiers, name, [parent], [primitiveKind], transform
    json::indent(out, l + 1);
    out.push_str("\"id\" : ");
    json::write_value(&uuid(&o.id), out, Some(l + 1));
    out.push_str(",\n");
    json::indent(out, l + 1);
    out.push_str("\"isLocked\" : ");
    out.push_str(if o.is_locked { "true,\n" } else { "false,\n" });
    json::indent(out, l + 1);
    out.push_str("\"isVisible\" : ");
    out.push_str(if o.is_visible { "true,\n" } else { "false,\n" });
    json::indent(out, l + 1);
    out.push_str("\"material\" : ");
    json::write_value(&material_json(&o.material), out, Some(l + 1));
    out.push_str(",\n");
    json::indent(out, l + 1);
    out.push_str("\"mesh\" : ");
    write_mesh_stream(&o.mesh, out, Some(l + 1));
    out.push_str(",\n");
    json::indent(out, l + 1);
    out.push_str("\"modifiers\" : ");
    json::write_value(&J::Arr(o.modifiers.iter().map(modifier_json).collect()), out, Some(l + 1));
    out.push_str(",\n");
    json::indent(out, l + 1);
    out.push_str("\"name\" : ");
    json::string(&o.name, out);
    if let Some(p) = &o.parent {
        out.push_str(",\n");
        json::indent(out, l + 1);
        out.push_str("\"parent\" : ");
        json::write_value(&uuid(p), out, Some(l + 1));
    }
    if let Some(k) = &o.primitive_kind {
        out.push_str(",\n");
        json::indent(out, l + 1);
        out.push_str("\"primitiveKind\" : ");
        json::string(k, out);
    }
    out.push_str(",\n");
    json::indent(out, l + 1);
    out.push_str("\"transform\" : ");
    let tr = J::Obj(vec![
        ("position", v3(t.position)),
        ("rotation", J::Arr(t.rotation.to_array().iter().map(|&x| J::F32(x)).collect())),
        ("scale", v3(t.scale)),
    ]);
    json::write_value(&tr, out, Some(l + 1));
    out.push('\n');
    json::indent(out, l);
    out.push('}');
}

fn v3(v: Vec3) -> J {
    J::Arr(vec![J::F32(v.x), J::F32(v.y), J::F32(v.z)])
}
fn uuid(s: &str) -> J {
    J::Obj(vec![("rawValue", J::Str(s.to_string()))])
}

/// The Mesh JSON shape (flat positions, faces as index arrays).
pub fn mesh_json(m: &Mesh) -> J {
    let mut flat = Vec::with_capacity(m.positions.len() * 3);
    for p in &m.positions {
        flat.extend([J::F32(p.x), J::F32(p.y), J::F32(p.z)]);
    }
    J::Obj(vec![
        ("faces", J::Arr(m.faces.iter().map(|f| J::Arr(f.iter().map(|&i| J::Int(i as i64)).collect())).collect())),
        ("positions", J::Arr(flat)),
        ("smoothShading", J::Bool(m.smooth_shading)),
    ])
}

/// `SceneCommand.setMesh(id, mesh)` in Swift's synthesized Codable shape, compact.
pub fn set_mesh_command(id: &str, mesh: &Mesh) -> String {
    let mut out = String::with_capacity(128 + mesh.positions.len() * 32 + mesh.faces.len() * 24);
    out.push_str("{\"setMesh\":{\"_0\":{\"rawValue\":");
    json::string(id, &mut out);
    out.push_str("},\"_1\":");
    write_mesh_stream(mesh, &mut out, None);
    out.push_str("}}");
    out
}

/// `SceneCommand.addMesh(mesh, name: name, position: position, material: material)` in Swift's Codable shape.
pub fn add_mesh_command(mesh: &Mesh, name: &str, position: Vec3, material: Option<&Material>) -> String {
    let mut out = String::with_capacity(256 + mesh.positions.len() * 32 + mesh.faces.len() * 24);
    out.push_str("{\"addMesh\":{\"_0\":");
    write_mesh_stream(mesh, &mut out, None);
    out.push_str(",\"name\":");
    json::string(name, &mut out);
    out.push_str(",\"position\":[");
    json::swift_float(position.x, &mut out);
    out.push(',');
    json::swift_float(position.y, &mut out);
    out.push(',');
    json::swift_float(position.z, &mut out);
    out.push(']');
    if let Some(m) = material {
        out.push_str(",\"material\":");
        json::write_value(&material_json(m), &mut out, None);
    }
    out.push_str("}}");
    out
}

fn material_json(m: &Material) -> J {
    let c = m.base_color;
    J::Obj(vec![
        ("baseColor", J::Obj(vec![("a", J::F32(c.a)), ("b", J::F32(c.b)), ("g", J::F32(c.g)), ("r", J::F32(c.r))])),
        ("metallic", J::F32(m.metallic)),
        ("name", J::Str(m.name.clone())),
        ("roughness", J::F32(m.roughness)),
    ])
}

fn modifier_json(e: &ModifierEntry) -> J {
    let (key, settings) = match &e.modifier {
        Modifier::Mirror { axis, merge, merge_distance } => {
            ("mirror", J::Obj(vec![("axis", J::Int(*axis)), ("merge", J::Bool(*merge)), ("mergeDistance", J::F32(*merge_distance))]))
        }
        Modifier::Subdivision { levels } => ("subdivision", J::Obj(vec![("levels", J::Int(*levels))])),
        Modifier::Array { count, relative_offset, constant_offset, use_relative } => (
            "array",
            J::Obj(vec![
                ("constantOffset", v3(*constant_offset)),
                ("count", J::Int(*count)),
                ("relativeOffset", v3(*relative_offset)),
                ("useRelative", J::Bool(*use_relative)),
            ]),
        ),
        Modifier::Solidify { thickness, offset } => ("solidify", J::Obj(vec![("offset", J::F32(*offset)), ("thickness", J::F32(*thickness))])),
    };
    J::Obj(vec![
        ("id", J::Str(e.id.clone())),
        ("isEnabled", J::Bool(e.is_enabled)),
        ("modifier", J::Obj(vec![(key, J::Obj(vec![("_0", settings)]))])),
    ])
}

fn selection_json(s: &Selection) -> J {
    let ints = |v: &[i64]| J::Arr(v.iter().map(|&i| J::Int(i)).collect());
    let mut f = vec![
        (
            "elements",
            J::Obj(vec![
                ("edges", J::Arr(s.elements.edges.iter().map(|&(a, b)| J::Obj(vec![("a", J::Int(a)), ("b", J::Int(b))])).collect())),
                ("faces", ints(&s.elements.faces)),
                ("vertices", ints(&s.elements.vertices)),
            ]),
        ),
        ("objects", J::Arr(s.objects.iter().map(|o| uuid(o)).collect())),
    ];
    if let Some(a) = &s.active {
        f.push(("active", uuid(a)));
    }
    J::Obj(f)
}

// MARK: repair (ProjectFile.repaired in Swift)

fn unit(f: f32, fallback: f32) -> f32 {
    if f.is_finite() { f.clamp(0.0, 1.0) } else { fallback }
}

pub(crate) fn repair_scene(scene: &mut Scene) {
    let ids: std::collections::HashSet<String> = scene.objects.iter().map(|o| o.id.clone()).collect();
    for o in &mut scene.objects {
        if !o.transform.is_finite() {
            o.transform = Transform::default();
        }
        for m in &mut o.modifiers {
            sanitize_modifier(&mut m.modifier);
        }
        let c = &mut o.material.base_color;
        for x in [&mut c.r, &mut c.g, &mut c.b, &mut c.a] {
            *x = if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 };
        }
        o.material.metallic = unit(o.material.metallic, 0.0);
        o.material.roughness = unit(o.material.roughness, 0.6);
        if o.parent.as_ref().is_some_and(|p| *p == o.id || !ids.contains(p)) {
            o.parent = None;
        }
    }
}

fn sanitize_modifier(m: &mut Modifier) {
    let fin = |v: &mut Vec3, d: Vec3| {
        if !v.is_finite() {
            *v = d
        }
    };
    match m {
        Modifier::Mirror { merge_distance, .. } => {
            *merge_distance = if merge_distance.is_finite() { merge_distance.clamp(0.0, 10.0) } else { 0.01 }
        }
        Modifier::Subdivision { levels } => *levels = (*levels).clamp(1, 3),
        Modifier::Array { count, relative_offset, constant_offset, .. } => {
            *count = (*count).clamp(1, 250);
            fin(relative_offset, Vec3::new(1.1, 0.0, 0.0));
            fin(constant_offset, Vec3::ZERO);
        }
        Modifier::Solidify { thickness, offset } => {
            *thickness = if thickness.is_finite() { thickness.clamp(-1000.0, 1000.0) } else { 2.0 };
            *offset = if offset.is_finite() { offset.clamp(-1.0, 1.0) } else { -1.0 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN: &str = include_str!("../../tests/fixtures/project.sculpt");

    #[test]
    fn golden_reencodes_byte_for_byte() {
        let p = ProjectFile::decode(GOLDEN.as_bytes()).unwrap();
        assert_eq!(p.scene.objects.len(), 3);
        assert_eq!(p.encode(), GOLDEN);
    }

    #[test]
    fn versions_and_garbage() {
        assert_eq!(ProjectFile::decode(b"nope"), Err(FormatError::NotAProject));
        let newer = GOLDEN.replacen("\"formatVersion\" : 1", "\"formatVersion\" : 2", 1);
        assert_eq!(ProjectFile::decode(newer.as_bytes()), Err(FormatError::NewerVersion(2)));
        let old = GOLDEN.replacen("\"formatVersion\" : 1", "\"formatVersion\" : 0", 1);
        assert_eq!(ProjectFile::decode(old.as_bytes()), Err(FormatError::UnsupportedVersion(0)));
        let broken = GOLDEN.replacen("\"isLocked\" : false", "\"isLocked\" : 3", 1);
        assert_eq!(ProjectFile::decode(broken.as_bytes()), Err(FormatError::NotAProject));
    }

    #[test]
    fn repairs_damaged_content() {
        let src = r#"{"formatVersion":1,"appVersion":"x","createdAt":0,"modifiedAt":"2026-01-02T03:04:05.250Z",
          "selection":{"objects":[{"rawValue":"b0000000-0000-4000-8000-000000000000"},{"rawValue":"A0000000-0000-4000-8000-000000000000"}],"elements":{"vertices":[3,1,3],"edges":[{"a":2,"b":1}],"faces":[]}},
          "scene":{"objects":[{"id":{"rawValue":"A0000000-0000-4000-8000-000000000000"},"name":"Bad","isLocked":false,"isVisible":true,
            "parent":{"rawValue":"A0000000-0000-4000-8000-000000000000"},
            "material":{"name":"M","baseColor":{"r":2,"g":"nan","b":0.5,"a":1},"metallic":"inf","roughness":-1},
            "transform":{"position":["nan",0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},
            "modifiers":[{"id":"C0000000-0000-4000-8000-000000000000","isEnabled":true,"modifier":{"subdivision":{"_0":{"levels":9}}}}],
            "mesh":{"positions":[[0,0,0],[1,0,0],["inf",1,0]],"faces":[[0,1,2],[0,1],[0,1,7],{"indices":[0,0,1]}]}}]}}"#;
        let p = ProjectFile::decode(src.as_bytes()).unwrap();
        let o = &p.scene.objects[0];
        assert_eq!(o.mesh.faces, vec![vec![0, 1, 2]]);
        assert_eq!(o.mesh.positions[2], Vec3::ZERO);
        assert!(!o.mesh.smooth_shading);
        assert_eq!(o.transform, Transform::default());
        assert_eq!(o.parent, None);
        assert_eq!(o.material.base_color, Color { r: 1.0, g: 0.0, b: 0.5, a: 1.0 });
        assert_eq!((o.material.metallic, o.material.roughness), (0.0, 0.0));
        assert_eq!(o.modifiers[0].modifier, Modifier::Subdivision { levels: 3 });
        assert_eq!(p.created_at, "2001-01-01T00:00:00Z");
        assert_eq!(p.modified_at, "2026-01-02T03:04:05Z");
        assert_eq!(p.selection.objects, vec!["A0000000-0000-4000-8000-000000000000", "B0000000-0000-4000-8000-000000000000"]);
        assert_eq!(p.selection.elements.vertices, vec![1, 3]);
        // Re-decoding our own output gives the same project.
        assert_eq!(ProjectFile::decode(p.encode().as_bytes()).unwrap(), p);
    }

    #[test]
    fn set_mesh_command_shape() {
        let m = Mesh { positions: vec![Vec3::ZERO, Vec3::X, Vec3::Y], faces: vec![vec![0, 1, 2]], smooth_shading: true };
        assert_eq!(
            set_mesh_command("ABC", &m),
            r#"{"setMesh":{"_0":{"rawValue":"ABC"},"_1":{"faces":[[0,1,2]],"positions":[0,0,0,1,0,0,0,1,0],"smoothShading":true}}}"#
        );
    }
}
