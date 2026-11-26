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

#[cfg(feature = "threads")]
impl Drop for ProjectFile {
    fn drop(&mut self) {
        use rayon::prelude::*;
        for obj in &mut self.scene.objects {
            if obj.mesh.faces.len() >= 16_384 {
                let faces = std::mem::take(&mut obj.mesh.faces);
                faces.into_par_iter().with_min_len(4096).for_each(drop);
            }
        }
    }
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
                .map(|o| o.mesh.positions.len() * 72 + o.mesh.faces.len() * 120 + 512)
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

/// Items per independently formatted chunk of a big array.
const CHUNK: usize = 16384;
/// Arrays shorter than this are formatted on the calling thread.
#[cfg(feature = "threads")]
const PAR_MIN: usize = 16 * 1024;

/// Formats `0..n` in `CHUNK` sized pieces (in parallel for big arrays) and appends them in order.
/// Uses accurate `bytes_per_item` estimates to avoid reallocations and drops temporary buffers in parallel.
fn write_chunked(out: &mut String, n: usize, bytes_per_item: usize, fmt: impl Fn(std::ops::Range<usize>, &mut Vec<u8>) + Sync) {
    let nchunks = n.div_ceil(CHUNK);
    let range = |c: usize| c * CHUNK..((c + 1) * CHUNK).min(n);
    // SAFETY: only ASCII is appended below.
    let dst = unsafe { out.as_mut_vec() };
    #[cfg(feature = "threads")]
    if n >= PAR_MIN {
        use rayon::prelude::*;
        let bufs: Vec<Vec<u8>> = (0..nchunks)
            .into_par_iter()
            .map(|c| {
                let r = range(c);
                let mut b = Vec::with_capacity(r.len() * bytes_per_item + 256);
                fmt(r, &mut b);
                b
            })
            .collect();
        let total: usize = bufs.iter().map(Vec::len).sum();
        let base = dst.len();
        dst.reserve(total);
        let mut offsets = Vec::with_capacity(nchunks);
        let mut acc = base;
        for b in &bufs {
            offsets.push(acc);
            acc += b.len();
        }
        let ptr = dst.as_mut_ptr() as usize;
        bufs.into_par_iter().zip(offsets.into_par_iter()).for_each(|(b, off)| {
            // SAFETY: the destination ranges are disjoint and inside the reserved capacity.
            unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), (ptr as *mut u8).add(off), b.len()) };
        });
        // SAFETY: bytes `base..base + total` were initialised above.
        unsafe { dst.set_len(base + total) };
        return;
    }
    dst.reserve(n * bytes_per_item + 256);
    for c in 0..nchunks {
        fmt(range(c), dst);
    }
}

#[inline(always)]
unsafe fn push_bytes_ptr(p: *mut u8, src: &[u8]) -> *mut u8 {
    unsafe {
        let len = src.len();
        if len == 12 {
            (p as *mut u64).write_unaligned((src.as_ptr() as *const u64).read_unaligned());
            (p.add(8) as *mut u32).write_unaligned((src.as_ptr().add(8) as *const u32).read_unaligned());
            return p.add(12);
        }
        if len == 20 {
            (p as *mut u128).write_unaligned((src.as_ptr() as *const u128).read_unaligned());
            (p.add(16) as *mut u32).write_unaligned((src.as_ptr().add(16) as *const u32).read_unaligned());
            return p.add(20);
        }
        std::ptr::copy_nonoverlapping(src.as_ptr(), p, len);
        p.add(len)
    }
}

fn write_mesh_stream(m: &Mesh, out: &mut String, level: Option<usize>) {
    let (nf, npos) = (m.faces.len(), m.positions.len());
    match level {
        None => {
            out.push_str("{\"faces\":[");
            write_chunked(out, nf, 28, |r, b| {
                b.reserve(r.len() * 28 + 64);
                let mut p = unsafe { b.as_mut_ptr().add(b.len()) };
                let mut end = unsafe { b.as_mut_ptr().add(b.capacity()) };
                for gi in r {
                    let f = &m.faces[gi];
                    let need = 4 + f.len() * 11;
                    if (end as usize).saturating_sub(p as usize) < need {
                        unsafe { b.set_len(p as usize - b.as_ptr() as usize) };
                        b.reserve(need.max(4096));
                        p = unsafe { b.as_mut_ptr().add(b.len()) };
                        end = unsafe { b.as_mut_ptr().add(b.capacity()) };
                    }
                    unsafe {
                        if gi > 0 {
                            *p = b',';
                            p = p.add(1);
                        }
                        *p = b'[';
                        p = p.add(1);
                        if f.len() == 3 {
                            p = json::write_u32_ptr(p, f[0]);
                            *p = b',';
                            p = json::write_u32_ptr(p.add(1), f[1]);
                            *p = b',';
                            p = json::write_u32_ptr(p.add(1), f[2]);
                        } else {
                            for (ii, &idx) in f.iter().enumerate() {
                                if ii > 0 {
                                    *p = b',';
                                    p = p.add(1);
                                }
                                p = json::write_u32_ptr(p, idx);
                            }
                        }
                        *p = b']';
                        p = p.add(1);
                    }
                }
                unsafe { b.set_len(p as usize - b.as_ptr() as usize) };
            });
            out.push_str("],\"positions\":[");
            write_chunked(out, npos, 36, |r, b| {
                b.reserve(r.len() * 36 + 128);
                let mut p = unsafe { b.as_mut_ptr().add(b.len()) };
                let mut end = unsafe { b.as_mut_ptr().add(b.capacity()) };
                for pi in r {
                    let v = m.positions[pi];
                    if (end as usize).saturating_sub(p as usize) < 96 {
                        unsafe { b.set_len(p as usize - b.as_ptr() as usize) };
                        b.reserve(4096);
                        p = unsafe { b.as_mut_ptr().add(b.len()) };
                        end = unsafe { b.as_mut_ptr().add(b.capacity()) };
                    }
                    unsafe {
                        if pi > 0 {
                            *p = b',';
                            p = p.add(1);
                        }
                        p = json::write_swift_f32_ptr(p, v.x);
                        *p = b',';
                        p = json::write_swift_f32_ptr(p.add(1), v.y);
                        *p = b',';
                        p = json::write_swift_f32_ptr(p.add(1), v.z);
                    }
                }
                unsafe { b.set_len(p as usize - b.as_ptr() as usize) };
            });
            out.push_str("],\"smoothShading\":");
            out.push_str(if m.smooth_shading { "true}" } else { "false}" });
        }
        Some(l) => {
            out.push_str("{\n");
            json::indent(out, l + 1);
            out.push_str("\"faces\" : [\n");
            if nf == 0 {
                out.push('\n'); // Swift writes empty containers as "[\n\n<indent>]".
            } else if l <= 12 {
                let ind2 = (l + 2) * 2;
                let ind3 = (l + 3) * 2;
                let mut open_buf = [b' '; 64];
                open_buf[ind2] = b'[';
                open_buf[ind2 + 1] = b'\n';
                let open_len = ind2 + 2 + ind3;
                let open = &open_buf[..open_len];

                let mut sep_buf = [b' '; 32];
                sep_buf[0] = b',';
                sep_buf[1] = b'\n';
                let sep_len = 2 + ind3;
                let sep = &sep_buf[..sep_len];

                let mut close_c_buf = [b' '; 32];
                close_c_buf[0] = b'\n';
                close_c_buf[1 + ind2] = b']';
                close_c_buf[2 + ind2] = b',';
                close_c_buf[3 + ind2] = b'\n';
                let close_c = &close_c_buf[..4 + ind2];
                let mut close_l_buf = [b' '; 32];
                close_l_buf[0] = b'\n';
                close_l_buf[1 + ind2] = b']';
                close_l_buf[2 + ind2] = b'\n';
                let close_l = &close_l_buf[..3 + ind2];

                let est_item = open_len + 4 * (8 + sep_len) + close_c.len();
                write_chunked(out, nf, est_item, |r, b| {
                    b.reserve(r.len() * est_item + 256);
                    let mut p = unsafe { b.as_mut_ptr().add(b.len()) };
                    let mut end = unsafe { b.as_mut_ptr().add(b.capacity()) };
                    for gi in r {
                        let f = &m.faces[gi];
                        let need = open_len + f.len() * (10 + sep_len) + close_c.len() + 16;
                        if !f.is_empty() && (end as usize).saturating_sub(p as usize) >= need {
                            unsafe {
                                p = push_bytes_ptr(p, open);
                                if f.len() == 3 {
                                    p = json::write_u32_ptr(p, f[0]);
                                    p = push_bytes_ptr(p, sep);
                                    p = json::write_u32_ptr(p, f[1]);
                                    p = push_bytes_ptr(p, sep);
                                    p = json::write_u32_ptr(p, f[2]);
                                } else if f.len() == 4 {
                                    p = json::write_u32_ptr(p, f[0]);
                                    p = push_bytes_ptr(p, sep);
                                    p = json::write_u32_ptr(p, f[1]);
                                    p = push_bytes_ptr(p, sep);
                                    p = json::write_u32_ptr(p, f[2]);
                                    p = push_bytes_ptr(p, sep);
                                    p = json::write_u32_ptr(p, f[3]);
                                } else {
                                    p = json::write_u32_ptr(p, f[0]);
                                    for &idx in &f[1..] {
                                        p = push_bytes_ptr(p, sep);
                                        p = json::write_u32_ptr(p, idx);
                                    }
                                }
                                p = push_bytes_ptr(p, if gi + 1 < nf { close_c } else { close_l });
                            }
                        } else {
                            unsafe { b.set_len(p as usize - b.as_ptr() as usize) };
                            if f.is_empty() {
                                json::push_indent(b, l + 2);
                                b.extend_from_slice(b"[\n\n");
                                json::push_indent(b, l + 2);
                                b.extend_from_slice(if gi + 1 < nf { b"],\n" } else { b"]\n" });
                            } else {
                                b.reserve(need.max(4096));
                                let mut p2 = unsafe { b.as_mut_ptr().add(b.len()) };
                                unsafe {
                                    p2 = push_bytes_ptr(p2, open);
                                    p2 = json::write_u32_ptr(p2, f[0]);
                                    for &idx in &f[1..] {
                                        p2 = push_bytes_ptr(p2, sep);
                                        p2 = json::write_u32_ptr(p2, idx);
                                    }
                                    p2 = push_bytes_ptr(p2, if gi + 1 < nf { close_c } else { close_l });
                                    b.set_len(p2 as usize - b.as_ptr() as usize);
                                }
                            }
                            p = unsafe { b.as_mut_ptr().add(b.len()) };
                            end = unsafe { b.as_mut_ptr().add(b.capacity()) };
                        }
                    }
                    unsafe { b.set_len(p as usize - b.as_ptr() as usize) };
                });
            } else {
                write_chunked(out, nf, 72 + l * 10, |r, b| {
                    for gi in r {
                        let f = &m.faces[gi];
                        json::push_indent(b, l + 2);
                        b.extend_from_slice(b"[\n");
                        if f.is_empty() {
                            b.push(b'\n');
                        }
                        for (ii, &idx) in f.iter().enumerate() {
                            json::push_indent(b, l + 3);
                            json::push_u32(b, idx);
                            b.extend_from_slice(if ii + 1 < f.len() { b",\n" } else { b"\n" });
                        }
                        json::push_indent(b, l + 2);
                        b.extend_from_slice(if gi + 1 < nf { b"],\n" } else { b"]\n" });
                    }
                });
            }
            json::indent(out, l + 1);
            out.push_str("],\n");
            json::indent(out, l + 1);
            out.push_str("\"positions\" : [\n");
            if npos == 0 {
                out.push('\n');
            } else if l <= 12 {
                let ind2_len = (l + 2) * 2;
                let ind2_buf = [b' '; 32];
                let ind2 = &ind2_buf[..ind2_len];
                let est_item = 3 * (ind2_len + 14);
                write_chunked(out, npos, est_item, |r, b| {
                    b.reserve(r.len() * est_item + 256);
                    let mut p = unsafe { b.as_mut_ptr().add(b.len()) };
                    let mut end = unsafe { b.as_mut_ptr().add(b.capacity()) };
                    for pi in r {
                        if (end as usize).saturating_sub(p as usize) < 256 {
                            unsafe { b.set_len(p as usize - b.as_ptr() as usize) };
                            b.reserve(4096);
                            p = unsafe { b.as_mut_ptr().add(b.len()) };
                            end = unsafe { b.as_mut_ptr().add(b.capacity()) };
                        }
                        let v = m.positions[pi];
                        unsafe {
                            if ind2_len == 8 {
                                (p as *mut u64).write_unaligned(0x2020_2020_2020_2020);
                                p = p.add(8);
                            } else {
                                p = push_bytes_ptr(p, ind2);
                            }
                            p = json::write_swift_f32_ptr(p, v.x);
                            *p = b',';
                            *p.add(1) = b'\n';
                            if ind2_len == 8 {
                                (p.add(2) as *mut u64).write_unaligned(0x2020_2020_2020_2020);
                                p = p.add(10);
                            } else {
                                p = push_bytes_ptr(p.add(2), ind2);
                            }
                            p = json::write_swift_f32_ptr(p, v.y);
                            *p = b',';
                            *p.add(1) = b'\n';
                            if ind2_len == 8 {
                                (p.add(2) as *mut u64).write_unaligned(0x2020_2020_2020_2020);
                                p = p.add(10);
                            } else {
                                p = push_bytes_ptr(p.add(2), ind2);
                            }
                            p = json::write_swift_f32_ptr(p, v.z);
                            if pi + 1 == npos {
                                *p = b'\n';
                                p = p.add(1);
                            } else {
                                *p = b',';
                                *p.add(1) = b'\n';
                                p = p.add(2);
                            }
                        }
                    }
                    unsafe { b.set_len(p as usize - b.as_ptr() as usize) };
                });
            } else {
                write_chunked(out, npos, 64 + l * 6, |r, b| {
                    for pi in r {
                        let p = m.positions[pi];
                        for (k, v) in [p.x, p.y, p.z].into_iter().enumerate() {
                            json::push_indent(b, l + 2);
                            json::swift_float_into(v, b);
                            b.extend_from_slice(if pi + 1 == npos && k == 2 { b"\n" } else { b",\n" });
                        }
                    }
                });
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

    /// The chunked (parallel above `PAR_MIN`) writers must match the generic tree writer byte for byte.
    #[test]
    fn streamed_mesh_matches_tree_writer() {
        for (seg, rings) in [(3, 2), (40, 30), (260, 130)] {
            let mut m = crate::mesh::uv_sphere(7.3, seg, rings);
            m.smooth_shading = seg % 2 == 0;
            for (i, p) in m.positions.iter_mut().enumerate() {
                *p *= 1.0 + (i % 7) as f32 * 1e-3; // varied digits
            }
            for level in [None, Some(0), Some(2)] {
                let mut got = String::new();
                write_mesh_stream(&m, &mut got, level);
                let want = match level {
                    None => mesh_json(&m).compact(),
                    Some(l) => {
                        let mut s = String::new();
                        json::write_value(&mesh_json(&m), &mut s, Some(l));
                        s
                    }
                };
                assert!(got == want, "seg {seg} level {level:?}: {} vs {} bytes", got.len(), want.len());
            }
        }
        let empty = Mesh::default();
        let mut got = String::new();
        write_mesh_stream(&empty, &mut got, Some(1));
        let mut want = String::new();
        json::write_value(&mesh_json(&empty), &mut want, Some(1));
        assert_eq!(got, want);
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
