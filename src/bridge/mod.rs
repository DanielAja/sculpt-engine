//! Pure-Rust document, command, edit-mode, import/export, printability, and voice-planning bridge.
//!
//! Replaces the legacy Swift `libSculptorAndroid.so` JNI bridge with a single zero-Swift-SDK
//! `libsculpt_engine.so` backend shared by Android runtime and JVM unit tests.

pub mod commands;
pub mod edit;
pub mod io;
pub mod voice;

use crate::bvh::Bvh;
use crate::format::{FormatError, Mesh, MeshSelection, ModifierEntry, ProjectFile, Scene, SceneObject, Selection, Transform};
use crate::mesh::SculptMesh;
use glam::{Mat4, Quat, Vec3};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

pub const BRIDGE_VERSION: i32 = 4;
pub const CORE_VERSION: &str = "1.1.0-rust";

// ── UUID v4 generator (no external crate needed) ──

static UUID_COUNTER: AtomicU64 = AtomicU64::new(1);

pub fn new_uuid() -> String {
    let seq = UUID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    // SplitMix64 mixing for unique RFC 4122 v4 uppercase UUID strings.
    let mut s1 = nanos ^ seq.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    s1 = (s1 ^ (s1 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    s1 = (s1 ^ (s1 >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let hi = s1 ^ (s1 >> 31);

    let mut s2 = (nanos.rotate_left(17)) ^ seq.wrapping_add(0xD1B5_4A32_D192_ED03);
    s2 = (s2 ^ (s2 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    s2 = (s2 ^ (s2 >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let lo = s2 ^ (s2 >> 31);

    let a = (hi >> 32) as u32;
    let b = ((hi >> 16) & 0xFFFF) as u16;
    let c = (((hi & 0xFFFF) as u16) & 0x0FFF) | 0x4000;
    let d = ((((lo >> 48) & 0xFFFF) as u16) & 0x3FFF) | 0x8000;
    let e = lo & 0x0000_FFFF_FFFF_FFFF;
    format!("{a:08X}-{b:04X}-{c:04X}-{d:04X}-{e:012X}")
}

pub fn iso8601_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    format_unix_iso8601(secs)
}

fn format_unix_iso8601(secs: i64) -> String {
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let hh = rem / 3600;
    let mm = (rem % 3600) / 60;
    let ss = rem % 60;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    format!("{year:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

// ── Errors & JSON envelopes ──

#[derive(Debug, Clone)]
pub struct BridgeError {
    pub key: String,
    pub args: Vec<String>,
}

impl BridgeError {
    pub fn new(key: impl Into<String>) -> Self {
        Self { key: key.into(), args: Vec::new() }
    }
    pub fn with_args(key: impl Into<String>, args: Vec<String>) -> Self {
        Self { key: key.into(), args }
    }
    pub fn to_error_json(&self) -> String {
        json!({ "key": self.key, "args": self.args }).to_string()
    }
    pub fn to_envelope(&self) -> String {
        json!({ "ok": false, "error": { "key": self.key, "args": self.args } }).to_string()
    }
}

impl From<FormatError> for BridgeError {
    fn from(e: FormatError) -> Self {
        match e {
            FormatError::NotAProject => Self::new("project.notAProject"),
            FormatError::NewerVersion(v) => Self::with_args("project.newerVersion", vec![v.to_string()]),
            FormatError::UnsupportedVersion(v) => Self::with_args("project.unsupportedVersion", vec![v.to_string()]),
        }
    }
}

pub fn ok_envelope(extra: serde_json::Map<String, Value>) -> String {
    let mut map = extra;
    map.insert("ok".into(), Value::Bool(true));
    Value::Object(map).to_string()
}

pub fn ok_empty() -> String {
    r#"{"ok":true}"#.to_string()
}

// ── Core geometry helpers ──

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl BoundingBox {
    pub const EMPTY: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }
    pub fn include(&mut self, p: Vec3) {
        if p.is_finite() {
            self.min = self.min.min(p);
            self.max = self.max.max(p);
        }
    }
    pub fn union(self, other: Self) -> Self {
        if other.is_empty() {
            return self;
        }
        if self.is_empty() {
            return other;
        }
        Self { min: self.min.min(other.min), max: self.max.max(other.max) }
    }
    pub fn center(&self) -> Vec3 {
        if self.is_empty() { Vec3::ZERO } else { (self.min + self.max) * 0.5 }
    }
    pub fn size(&self) -> Vec3 {
        if self.is_empty() { Vec3::ZERO } else { (self.max - self.min).max(Vec3::ZERO) }
    }
    pub fn max_dimension(&self) -> f32 {
        self.size().max_element()
    }
    pub fn from_mesh(mesh: &Mesh, world: Mat4) -> Self {
        let mut b = Self::EMPTY;
        for &p in &mesh.positions {
            b.include(world.transform_point3(p));
        }
        b
    }
}

/// XYZ intrinsic Euler angles in degrees matching Swift `Transform.eulerDegrees` (R = Rx * Ry * Rz).
pub fn quat_to_euler_degrees(q: Quat) -> Vec3 {
    let q = q.normalize();
    let m = Mat4::from_quat(q);
    // R = Rx(a) * Ry(b) * Rz(c):
    // m02 = sin(b), m01 = -cos(b)sin(c), m00 = cos(b)cos(c), m12 = -sin(a)cos(b), m22 = cos(a)cos(b)
    let r02 = m.z_axis.x.clamp(-1.0, 1.0);
    let y = r02.asin();
    let (x, z) = if r02.abs() < 0.99999 {
        ((-m.z_axis.y).atan2(m.z_axis.z), (-m.y_axis.x).atan2(m.x_axis.x))
    } else {
        (m.y_axis.z.atan2(m.y_axis.y), 0.0)
    };
    Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees())
}

pub fn euler_degrees_to_quat(e: Vec3) -> Quat {
    let rx = Quat::from_rotation_x(e.x.to_radians());
    let ry = Quat::from_rotation_y(e.y.to_radians());
    let rz = Quat::from_rotation_z(e.z.to_radians());
    (rx * ry * rz).normalize()
}

pub fn scene_world_matrix(scene: &Scene, id: &str) -> Mat4 {
    let mut chain = Vec::new();
    let mut cur = Some(id);
    let mut steps = 0usize;
    while let Some(cid) = cur {
        if steps > scene.objects.len() {
            break;
        }
        if let Some(obj) = scene.objects.iter().find(|o| o.id == cid) {
            chain.push(obj.transform.matrix());
            cur = obj.parent.as_deref();
        } else {
            break;
        }
        steps += 1;
    }
    let mut m = Mat4::IDENTITY;
    for mat in chain.into_iter().rev() {
        m *= mat;
    }
    m
}

pub fn mesh_triangles(mesh: &Mesh) -> Vec<[u32; 3]> {
    let n = mesh.positions.len() as u32;
    let mut tris = Vec::with_capacity(mesh.faces.len() * 2);
    for f in &mesh.faces {
        if f.len() < 3 || f.iter().any(|&i| i >= n) {
            continue;
        }
        for k in 1..(f.len() - 1) {
            tris.push([f[0], f[k], f[k + 1]]);
        }
    }
    tris
}

pub fn mesh_triangle_count(mesh: &Mesh) -> usize {
    let n = mesh.positions.len() as u32;
    mesh.faces
        .iter()
        .filter(|f| f.len() >= 3 && f.iter().all(|&i| i < n))
        .map(|f| f.len() - 2)
        .sum()
}

pub fn mesh_face_normal(mesh: &Mesh, face_idx: usize) -> Vec3 {
    let Some(f) = mesh.faces.get(face_idx) else { return Vec3::Z };
    if f.len() < 3 {
        return Vec3::Z;
    }
    let mut n = Vec3::ZERO;
    for i in 0..f.len() {
        let a = mesh.positions.get(f[i] as usize).copied().unwrap_or(Vec3::ZERO);
        let b = mesh.positions.get(f[(i + 1) % f.len()] as usize).copied().unwrap_or(Vec3::ZERO);
        n += Vec3::new(
            (a.y - b.y) * (a.z + b.z),
            (a.z - b.z) * (a.x + b.x),
            (a.x - b.x) * (a.y + b.y),
        );
    }
    let len = n.length();
    if len > 1e-12 { n / len } else { Vec3::Z }
}

pub fn mesh_vertex_normals(mesh: &Mesh) -> Vec<Vec3> {
    let mut acc = vec![Vec3::ZERO; mesh.positions.len()];
    let n = mesh.positions.len();
    for f in &mesh.faces {
        if f.len() < 3 || f.iter().any(|&i| i as usize >= n) {
            continue;
        }
        for k in 1..(f.len() - 1) {
            let (i0, i1, i2) = (f[0] as usize, f[k] as usize, f[k + 1] as usize);
            let cr = (mesh.positions[i1] - mesh.positions[i0]).cross(mesh.positions[i2] - mesh.positions[i0]);
            acc[i0] += cr;
            acc[i1] += cr;
            acc[i2] += cr;
        }
    }
    for v in &mut acc {
        let len = v.length();
        *v = if len > 1e-12 { *v / len } else { Vec3::Z };
    }
    acc
}

pub fn mesh_unique_edges(mesh: &Mesh) -> Vec<(u32, u32)> {
    let n = mesh.positions.len() as u32;
    let mut set = HashSet::new();
    let mut edges = Vec::new();
    for f in &mesh.faces {
        if f.len() < 2 {
            continue;
        }
        for i in 0..f.len() {
            let a = f[i];
            let b = f[(i + 1) % f.len()];
            if a < n && b < n && a != b {
                let key = if a < b { (a, b) } else { (b, a) };
                if set.insert(key) {
                    edges.push(key);
                }
            }
        }
    }
    edges.sort_unstable();
    edges
}

/// Evaluates an object's modifier stack (`Mirror`, `Subdivision`, `Array`, `Solidify`).
pub fn evaluate_object_mesh(obj: &SceneObject) -> Mesh {
    let mut mesh = obj.mesh.clone();
    for entry in &obj.modifiers {
        if !entry.is_enabled {
            continue;
        }
        mesh = commands::apply_modifier(&mesh, &entry.modifier);
    }
    mesh
}

/// Builds render buffers matching `RenderMesh` in `Engine.swift`.
#[derive(Debug, Clone)]
pub struct RenderMeshData {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub indices: Vec<u32>,
    pub hull_positions: Vec<f32>,
    pub hull_normals: Vec<f32>,
    pub hull_indices: Vec<u32>,
}

impl RenderMeshData {
    pub fn from_mesh(mesh: &Mesh) -> Self {
        let tris = mesh_triangles(mesh);
        if tris.is_empty() {
            return Self {
                positions: Vec::new(),
                normals: Vec::new(),
                indices: Vec::new(),
                hull_positions: Vec::new(),
                hull_normals: Vec::new(),
                hull_indices: Vec::new(),
            };
        }
        let smooth = mesh_vertex_normals(mesh);
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::with_capacity(tris.len() * 3);
        if mesh.smooth_shading {
            positions.reserve(mesh.positions.len() * 3);
            normals.reserve(mesh.positions.len() * 3);
            for (i, &p) in mesh.positions.iter().enumerate() {
                positions.extend_from_slice(&[p.x, p.y, p.z]);
                let n = smooth[i];
                normals.extend_from_slice(&[n.x, n.y, n.z]);
            }
            for t in &tris {
                indices.extend_from_slice(t);
            }
        } else {
            positions.reserve(tris.len() * 9);
            normals.reserve(tris.len() * 9);
            for t in &tris {
                let a = mesh.positions[t[0] as usize];
                let b = mesh.positions[t[1] as usize];
                let c = mesh.positions[t[2] as usize];
                let cr = (b - a).cross(c - a);
                let len = cr.length();
                let n = if len > 1e-12 { cr / len } else { Vec3::Z };
                let base = (positions.len() / 3) as u32;
                for p in [a, b, c] {
                    positions.extend_from_slice(&[p.x, p.y, p.z]);
                    normals.extend_from_slice(&[n.x, n.y, n.z]);
                }
                indices.extend_from_slice(&[base, base + 1, base + 2]);
            }
        }
        let mut hull_positions = Vec::with_capacity(mesh.positions.len() * 3);
        let mut hull_normals = Vec::with_capacity(mesh.positions.len() * 3);
        for (i, &p) in mesh.positions.iter().enumerate() {
            hull_positions.extend_from_slice(&[p.x, p.y, p.z]);
            let n = smooth[i];
            hull_normals.extend_from_slice(&[n.x, n.y, n.z]);
        }
        let mut hull_indices = Vec::with_capacity(tris.len() * 3);
        for t in &tris {
            hull_indices.extend_from_slice(t);
        }
        Self { positions, normals, indices, hull_positions, hull_normals, hull_indices }
    }
}

// ── Document, Undo Stack, Preview & Cache ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementMode {
    Vertex,
    Edge,
    Face,
}

impl ElementMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Edge => "edge",
            Self::Face => "face",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "vertex" => Some(Self::Vertex),
            "edge" => Some(Self::Edge),
            "face" => Some(Self::Face),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditMode {
    Object,
    Edit(ElementMode),
}

impl EditMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Object => "object",
            Self::Edit(m) => m.as_str(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UndoEntry {
    pub label: String,
    pub scene: Scene,
    pub selection: Selection,
    pub approx_bytes: usize,
}

#[derive(Debug, Clone)]
struct CachedEval {
    mesh: Mesh,
    modifiers: Vec<ModifierEntry>,
    evaluated: Mesh,
    geometry_revision: i32,
    render: Option<RenderMeshData>,
}

#[derive(Debug, Clone)]
pub struct SceneSnapshot {
    pub scene: Scene,
    pub selection: Selection,
    pub revision: i32,
}

#[derive(Debug, Clone)]
pub struct AppliedCommand {
    pub scene: Scene,
    pub created: Vec<String>,
    pub new_elements: Option<MeshSelection>,
    pub label: String,
    pub start_revision: i32,
}

pub struct CommandControl {
    pub progress_bits: AtomicU64,
    pub cancelled: AtomicBool,
}

impl CommandControl {
    pub fn new() -> Self {
        Self { progress_bits: AtomicU64::new(0.0f64.to_bits()), cancelled: AtomicBool::new(false) }
    }
    pub fn set_progress(&self, f: f64) {
        self.progress_bits.store(f.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }
    pub fn progress(&self) -> f64 {
        f64::from_bits(self.progress_bits.load(Ordering::Relaxed))
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

pub struct EngineDoc {
    pub scene: Scene,
    pub selection: Selection,
    pub edit_mode: EditMode,
    pub revision: i32,
    pub created_at: String,
    pub undo_stack: Vec<UndoEntry>,
    pub redo_stack: Vec<UndoEntry>,
    pub undo_memory_budget: usize,
    undo_group_depth: usize,
    undo_group_saved: Option<UndoEntry>,
    pub preview_transforms: HashMap<String, Transform>,
    pub preview_meshes: HashMap<String, Mesh>,
    cache: HashMap<String, CachedEval>,
    revision_counter: i32,
    pub last_picked_vertex: Option<usize>,
    pub overlay_revision: i32,
    pub last_overlay_key: Option<edit::OverlayCacheKey>,
    pub last_overlay: Option<edit::EditOverlayData>,
    pub plan_build: Option<voice::PlanBuildState>,
}

impl EngineDoc {
    pub fn starter() -> Self {
        let cube = commands::make_starter_cube();
        let id = cube.id.clone();
        Self {
            scene: Scene { objects: vec![cube] },
            selection: Selection { objects: vec![id.clone()], active: Some(id), elements: MeshSelection::default() },
            edit_mode: EditMode::Object,
            revision: 0,
            created_at: iso8601_now(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            undo_memory_budget: 192 * 1024 * 1024,
            undo_group_depth: 0,
            undo_group_saved: None,
            preview_transforms: HashMap::new(),
            preview_meshes: HashMap::new(),
            cache: HashMap::new(),
            revision_counter: 0,
            last_picked_vertex: None,
            overlay_revision: 0,
            last_overlay_key: None,
            last_overlay: None,
            plan_build: None,
        }
    }

    pub fn reset(&mut self) {
        let budget = self.undo_memory_budget;
        *self = Self::starter();
        self.undo_memory_budget = budget;
    }

    pub fn replace_scene_fresh(&mut self, scene: Scene, selection: Selection, created_at: Option<String>) {
        let budget = self.undo_memory_budget;
        self.scene = scene;
        self.selection = selection;
        self.edit_mode = EditMode::Object;
        self.revision = 0;
        self.created_at = created_at.unwrap_or_else(iso8601_now);
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.undo_memory_budget = budget;
        self.undo_group_depth = 0;
        self.undo_group_saved = None;
        self.preview_transforms.clear();
        self.preview_meshes.clear();
        self.cache.clear();
        self.last_picked_vertex = None;
        self.last_overlay_key = None;
        self.last_overlay = None;
        self.plan_build = None;
    }

    fn approx_scene_bytes(scene: &Scene) -> usize {
        scene
            .objects
            .iter()
            .map(|o| o.mesh.positions.len() * 12 + o.mesh.faces.len() * 32 + 256)
            .sum::<usize>()
            .max(512)
    }

    pub fn begin_undo_group(&mut self, label: impl Into<String>) {
        if self.undo_group_depth == 0 {
            self.undo_group_saved = Some(UndoEntry {
                label: label.into(),
                scene: self.scene.clone(),
                selection: self.selection.clone(),
                approx_bytes: Self::approx_scene_bytes(&self.scene),
            });
        }
        self.undo_group_depth += 1;
    }

    pub fn end_undo_group(&mut self) {
        if self.undo_group_depth == 0 {
            return;
        }
        self.undo_group_depth -= 1;
        if self.undo_group_depth == 0 {
            if let Some(saved) = self.undo_group_saved.take() {
                if saved.scene != self.scene {
                    self.push_undo_entry(saved);
                }
            }
        }
    }

    fn push_undo_entry(&mut self, entry: UndoEntry) {
        self.undo_stack.push(entry);
        self.redo_stack.clear();
        self.trim_undo_budget();
    }

    fn trim_undo_budget(&mut self) {
        let mut total: usize = self.undo_stack.iter().map(|e| e.approx_bytes).sum();
        while total > self.undo_memory_budget && self.undo_stack.len() > 1 {
            let removed = self.undo_stack.remove(0);
            total = total.saturating_sub(removed.approx_bytes);
        }
    }

    pub fn record_scene_change(&mut self, label: String, new_scene: Scene, created: &[String], new_elements: Option<MeshSelection>) {
        if self.undo_group_depth == 0 {
            self.push_undo_entry(UndoEntry {
                label,
                scene: self.scene.clone(),
                selection: self.selection.clone(),
                approx_bytes: Self::approx_scene_bytes(&self.scene),
            });
        }
        self.scene = new_scene;
        self.preview_transforms.clear();
        self.preview_meshes.clear();
        self.revision += 1;
        if !created.is_empty() {
            let mut objs = created.to_vec();
            objs.sort();
            objs.dedup();
            self.selection = Selection {
                objects: objs,
                active: created.last().cloned(),
                elements: MeshSelection::default(),
            };
        } else if let Some(elems) = new_elements {
            self.selection.elements = elems;
        }
        self.prune_selection();
    }

    pub fn prune_selection(&mut self) {
        let valid: HashSet<&str> = self.scene.objects.iter().map(|o| o.id.as_str()).collect();
        self.selection.objects.retain(|id| valid.contains(id.as_str()));
        if self.selection.active.as_deref().is_some_and(|a| !valid.contains(a)) {
            self.selection.active = self.selection.objects.last().cloned();
        }
        if matches!(self.edit_mode, EditMode::Edit(_)) && self.selection.active.is_none() {
            self.edit_mode = EditMode::Object;
            self.selection.elements = MeshSelection::default();
        }
    }

    pub fn undo(&mut self) -> bool {
        let Some(prev) = self.undo_stack.pop() else { return false };
        let cur = UndoEntry {
            label: prev.label.clone(),
            scene: self.scene.clone(),
            selection: self.selection.clone(),
            approx_bytes: Self::approx_scene_bytes(&self.scene),
        };
        self.redo_stack.push(cur);
        self.scene = prev.scene;
        self.selection = prev.selection;
        self.preview_transforms.clear();
        self.preview_meshes.clear();
        self.revision += 1;
        self.prune_selection();
        true
    }

    pub fn redo(&mut self) -> bool {
        let Some(next) = self.redo_stack.pop() else { return false };
        let cur = UndoEntry {
            label: next.label.clone(),
            scene: self.scene.clone(),
            selection: self.selection.clone(),
            approx_bytes: Self::approx_scene_bytes(&self.scene),
        };
        self.undo_stack.push(cur);
        self.scene = next.scene;
        self.selection = next.selection;
        self.preview_transforms.clear();
        self.preview_meshes.clear();
        self.revision += 1;
        self.prune_selection();
        true
    }

    pub fn previewed_scene(&self) -> Scene {
        if self.preview_transforms.is_empty() && self.preview_meshes.is_empty() {
            return self.scene.clone();
        }
        let mut s = self.scene.clone();
        for o in &mut s.objects {
            if let Some(&t) = self.preview_transforms.get(&o.id) {
                o.transform = t;
            }
            if let Some(m) = self.preview_meshes.get(&o.id) {
                o.mesh = m.clone();
            }
        }
        s
    }

    pub fn refresh_cache(&mut self) {
        let scene = self.previewed_scene();
        let mut seen = HashSet::with_capacity(scene.objects.len());
        for o in &scene.objects {
            seen.insert(o.id.clone());
            if let Some(c) = self.cache.get(&o.id) {
                if c.mesh == o.mesh && c.modifiers == o.modifiers {
                    continue;
                }
            }
            self.revision_counter += 1;
            let evaluated = evaluate_object_mesh(o);
            self.cache.insert(
                o.id.clone(),
                CachedEval {
                    mesh: o.mesh.clone(),
                    modifiers: o.modifiers.clone(),
                    evaluated,
                    geometry_revision: self.revision_counter,
                    render: None,
                },
            );
        }
        self.cache.retain(|k, _| seen.contains(k));
    }

    pub fn evaluated_mesh(&self, id: &str) -> Option<&Mesh> {
        self.cache.get(id).map(|c| &c.evaluated)
    }

    pub fn ordered_selection(&self) -> Vec<String> {
        let sel: HashSet<&str> = self.selection.objects.iter().map(|s| s.as_str()).collect();
        let mut ids: Vec<String> = self
            .scene
            .objects
            .iter()
            .filter(|o| sel.contains(o.id.as_str()))
            .map(|o| o.id.clone())
            .collect();
        if let Some(active) = &self.selection.active {
            if let Some(pos) = ids.iter().position(|id| id == active) {
                let a = ids.remove(pos);
                ids.insert(0, a);
            }
        }
        ids
    }

    pub fn selection_pivot(&self, scene: &Scene) -> Option<Vec3> {
        match self.edit_mode {
            EditMode::Object => {
                let mut b = BoundingBox::EMPTY;
                for id in &self.selection.objects {
                    let Some(o) = scene.objects.iter().find(|o| &o.id == id && o.is_visible) else { continue };
                    let mesh = self.cache.get(id).map(|c| &c.evaluated).unwrap_or(&o.mesh);
                    b = b.union(BoundingBox::from_mesh(mesh, scene_world_matrix(scene, id)));
                }
                if b.is_empty() { None } else { Some(b.center()) }
            }
            EditMode::Edit(_) => {
                let id = self.selection.active.as_deref()?;
                let obj = scene.objects.iter().find(|o| o.id == id && o.is_visible)?;
                let verts = edit::expanded_vertices(&self.selection.elements, &obj.mesh);
                if verts.is_empty() {
                    return None;
                }
                let mut b = BoundingBox::EMPTY;
                for v in verts {
                    if let Some(&p) = obj.mesh.positions.get(v) {
                        b.include(p);
                    }
                }
                if b.is_empty() {
                    None
                } else {
                    Some(scene_world_matrix(scene, id).transform_point3(b.center()))
                }
            }
        }
    }

    pub fn state_json(&mut self) -> String {
        self.refresh_cache();
        let previewed = self.previewed_scene();
        let sel_set: HashSet<&str> = self.selection.objects.iter().map(|s| s.as_str()).collect();

        let objects: Vec<Value> = self
            .scene
            .objects
            .iter()
            .map(|o| {
                let euler = quat_to_euler_degrees(o.transform.rotation);
                let c = o.material.base_color;
                let mods: Vec<Value> = o.modifiers.iter().map(modifier_entry_to_value).collect();
                json!({
                    "id": o.id,
                    "name": o.name,
                    "kind": o.primitive_kind,
                    "visible": o.is_visible,
                    "locked": o.is_locked,
                    "parent": o.parent,
                    "smooth": o.mesh.smooth_shading,
                    "geometryRevision": self.cache.get(&o.id).map(|c| c.geometry_revision).unwrap_or(0),
                    "position": [o.transform.position.x, o.transform.position.y, o.transform.position.z],
                    "rotation": [euler.x, euler.y, euler.z],
                    "scale": [o.transform.scale.x, o.transform.scale.y, o.transform.scale.z],
                    "material": {
                        "name": o.material.name,
                        "color": [c.r, c.g, c.b, c.a],
                        "metallic": o.material.metallic,
                        "roughness": o.material.roughness,
                    },
                    "modifiers": mods,
                })
            })
            .collect();

        let selection_ordered: Vec<String> = self
            .scene
            .objects
            .iter()
            .filter(|o| sel_set.contains(o.id.as_str()))
            .map(|o| o.id.clone())
            .collect();

        let mut root = serde_json::Map::new();
        root.insert("revision".into(), json!(self.revision));
        root.insert("canUndo".into(), json!(!self.undo_stack.is_empty()));
        root.insert("canRedo".into(), json!(!self.redo_stack.is_empty()));
        root.insert("undoLabel".into(), json!(self.undo_stack.last().map(|u| u.label.as_str())));
        root.insert("redoLabel".into(), json!(self.redo_stack.last().map(|u| u.label.as_str())));
        root.insert("selection".into(), json!(selection_ordered));
        root.insert("active".into(), json!(self.selection.active));
        root.insert("objects".into(), Value::Array(objects));
        root.insert("editMode".into(), json!(self.edit_mode.as_str()));
        root.insert(
            "pivot".into(),
            self.selection_pivot(&previewed).map(|p| json!([p.x, p.y, p.z])).unwrap_or(Value::Null),
        );

        if let Some(active) = &self.selection.active {
            if let Some(c) = self.cache.get(active) {
                let bounds = BoundingBox::from_mesh(&c.evaluated, scene_world_matrix(&self.scene, active));
                let sz = bounds.size();
                root.insert(
                    "activeInfo".into(),
                    json!({
                        "dimensions": [sz.x, sz.y, sz.z],
                        "triangles": mesh_triangle_count(&c.evaluated),
                    }),
                );
            }
        }
        if matches!(self.edit_mode, EditMode::Edit(_)) {
            if let Some(stats) = edit::element_stats(self) {
                root.insert("elements".into(), stats);
            }
        }
        Value::Object(root).to_string()
    }

    pub fn render_list_json(&mut self) -> String {
        self.refresh_cache();
        let scene = self.previewed_scene();
        let sel_set: HashSet<&str> = self.selection.objects.iter().map(|s| s.as_str()).collect();
        let editing = match self.edit_mode {
            EditMode::Edit(_) => self.selection.active.as_deref(),
            EditMode::Object => None,
        };
        let mut items = Vec::new();
        let mut bounds = BoundingBox::EMPTY;
        for o in scene.objects.iter().filter(|o| o.is_visible) {
            let Some(c) = self.cache.get(&o.id) else { continue };
            let m = scene_world_matrix(&scene, &o.id);
            let cols = m.to_cols_array();
            let col = o.material.base_color;
            let tri_count = mesh_triangle_count(&c.evaluated);
            items.push(json!({
                "id": o.id,
                "geometryRevision": c.geometry_revision,
                "empty": tri_count == 0,
                "matrix": cols,
                "color": [col.r, col.g, col.b, col.a],
                "metallic": o.material.metallic,
                "roughness": o.material.roughness,
                "selected": sel_set.contains(o.id.as_str()) && editing != Some(o.id.as_str()),
                "active": self.selection.active.as_deref() == Some(o.id.as_str()),
            }));
            bounds = bounds.union(BoundingBox::from_mesh(&c.evaluated, m));
        }
        let bjson = if bounds.is_empty() {
            Value::Null
        } else {
            json!({
                "min": [bounds.min.x, bounds.min.y, bounds.min.z],
                "max": [bounds.max.x, bounds.max.y, bounds.max.z],
            })
        };
        json!({ "objects": items, "bounds": bjson }).to_string()
    }

    pub fn bounds_json(&mut self, ids: &[String], elements: bool) -> String {
        self.refresh_cache();
        let scene = self.previewed_scene();
        let mut b = BoundingBox::EMPTY;
        if elements {
            if let (EditMode::Edit(_), Some(id)) = (self.edit_mode, self.selection.active.as_deref()) {
                if let Some(obj) = scene.objects.iter().find(|o| o.id == id) {
                    let verts = edit::expanded_vertices(&self.selection.elements, &obj.mesh);
                    if !verts.is_empty() {
                        let w = scene_world_matrix(&scene, id);
                        let mut eb = BoundingBox::EMPTY;
                        for v in verts {
                            if let Some(&p) = obj.mesh.positions.get(v) {
                                eb.include(w.transform_point3(p));
                            }
                        }
                        if !eb.is_empty() {
                            b = if eb.max_dimension() < 1.0 {
                                BoundingBox { min: eb.center() - Vec3::splat(5.0), max: eb.center() + Vec3::splat(5.0) }
                            } else {
                                eb
                            };
                        }
                    }
                }
            }
        }
        if b.is_empty() {
            let wanted: HashSet<&str> = ids.iter().map(|s| s.as_str()).collect();
            for o in &scene.objects {
                if wanted.is_empty() && !o.is_visible {
                    continue;
                }
                if !wanted.is_empty() && !wanted.contains(o.id.as_str()) {
                    continue;
                }
                if let Some(c) = self.cache.get(&o.id) {
                    b = b.union(BoundingBox::from_mesh(&c.evaluated, scene_world_matrix(&scene, &o.id)));
                }
            }
        }
        if b.is_empty() {
            "null".to_string()
        } else {
            json!({
                "min": [b.min.x, b.min.y, b.min.z],
                "max": [b.max.x, b.max.y, b.max.z],
            })
            .to_string()
        }
    }

    pub fn render_mesh(&mut self, id: &str) -> Option<RenderMeshData> {
        self.refresh_cache();
        let c = self.cache.get_mut(id)?;
        if c.render.is_none() {
            c.render = Some(RenderMeshData::from_mesh(&c.evaluated));
        }
        c.render.clone()
    }

    pub fn pick(&mut self, origin: Vec3, direction: Vec3) -> Option<String> {
        if !(direction.length_squared() > 0.0) || !origin.is_finite() || !direction.is_finite() {
            return None;
        }
        let dir = direction.normalize();
        self.refresh_cache();
        let scene = self.previewed_scene();
        let mut best: Option<(f32, String)> = None;
        for o in scene.objects.iter().filter(|o| o.is_visible) {
            let Some(c) = self.cache.get(&o.id) else { continue };
            let tris = mesh_triangles(&c.evaluated);
            if tris.is_empty() {
                continue;
            }
            let world = scene_world_matrix(&scene, &o.id);
            let det = world.determinant();
            if !det.is_finite() || det.abs() < 1e-12 {
                continue;
            }
            let inv = world.inverse();
            let lo = inv.transform_point3(origin);
            let ld = inv.transform_vector3(dir);
            let ld_len = ld.length();
            if ld_len <= 1e-12 {
                continue;
            }
            let ld_n = ld / ld_len;
            let sm = SculptMesh::from_mesh(&c.evaluated);
            let bvh = Bvh::build(&sm.positions, &sm.tris);
            if let Some(hit) = bvh.ray(&sm.positions, &sm.tris, lo, ld_n) {
                let world_p = world.transform_point3(lo + ld_n * hit.t);
                let dist = world_p.distance(origin);
                if best.as_ref().is_none_or(|(bd, _)| dist < *bd) {
                    best = Some((dist, o.id.clone()));
                }
            }
        }
        best.map(|(_, id)| id)
    }

    pub fn summary_text(&mut self) -> String {
        self.refresh_cache();
        if self.scene.objects.is_empty() {
            return "(empty scene)".to_string();
        }
        let mut lines = Vec::new();
        for o in &self.scene.objects {
            let mesh = self.cache.get(&o.id).map(|c| &c.evaluated).unwrap_or(&o.mesh);
            let b = BoundingBox::from_mesh(mesh, scene_world_matrix(&self.scene, &o.id));
            let s = b.size();
            let c = b.center();
            let kind = o.primitive_kind.as_deref().unwrap_or("mesh");
            lines.push(format!(
                "{} ({}, {:.0}x{:.0}x{:.0} mm at {:.0}, {:.0}, {:.0})",
                o.name, kind, s.x, s.y, s.z, c.x, c.y, c.z
            ));
        }
        lines.join("\n")
    }

    pub fn summary_json(&mut self) -> String {
        self.refresh_cache();
        let mut tris = 0usize;
        let mut bounds = BoundingBox::EMPTY;
        for o in self.scene.objects.iter().filter(|o| o.is_visible) {
            let mesh = self.cache.get(&o.id).map(|c| &c.evaluated).unwrap_or(&o.mesh);
            tris += mesh_triangle_count(mesh);
            bounds = bounds.union(BoundingBox::from_mesh(mesh, scene_world_matrix(&self.scene, &o.id)));
        }
        let selected = self.ordered_selection();
        let selected_name = if selected.len() == 1 {
            self.scene.objects.iter().find(|o| o.id == selected[0]).map(|o| o.name.clone())
        } else {
            None
        };
        let sz = if bounds.is_empty() {
            Value::Null
        } else {
            let s = bounds.size();
            json!([s.x, s.y, s.z])
        };
        let text = self.summary_text();
        json!({
            "objects": self.scene.objects.len(),
            "visibleObjects": self.scene.objects.iter().filter(|o| o.is_visible).count(),
            "selectedObjects": selected.len(),
            "selectedName": selected_name,
            "triangles": tris,
            "size": sz,
            "text": text,
        })
        .to_string()
    }

    pub fn to_project_file(&self, camera_json: Option<&str>) -> ProjectFile {
        let camera_state = camera_json.map(|s| base64_encode(s.as_bytes()));
        ProjectFile {
            format_version: crate::format::CURRENT_FORMAT_VERSION,
            scene: self.scene.clone(),
            selection: self.selection.clone(),
            camera_state,
            created_at: self.created_at.clone(),
            modified_at: iso8601_now(),
            app_version: CORE_VERSION.to_string(),
        }
    }

    pub fn open_project_bytes(&mut self, bytes: &[u8]) -> String {
        match ProjectFile::decode(bytes) {
            Ok(pf) => {
                let count = pf.scene.objects.len();
                let cam = pf
                    .camera_state
                    .as_deref()
                    .and_then(base64_decode)
                    .and_then(|b| String::from_utf8(b).ok());
                self.replace_scene_fresh(pf.scene.clone(), pf.selection.clone(), Some(pf.created_at.clone()));
                let mut m = serde_json::Map::new();
                m.insert("objects".into(), json!(count));
                if let Some(c) = cam {
                    m.insert("camera".into(), Value::String(c));
                }
                ok_envelope(m)
            }
            Err(e) => BridgeError::from(e).to_envelope(),
        }
    }
}

pub fn modifier_entry_to_value(e: &ModifierEntry) -> Value {
    let mod_val = match &e.modifier {
        crate::format::Modifier::Mirror { axis, merge, merge_distance } => json!({
            "mirror": { "_0": { "axis": axis, "merge": merge, "mergeDistance": merge_distance } }
        }),
        crate::format::Modifier::Subdivision { levels } => json!({
            "subdivision": { "_0": { "levels": levels } }
        }),
        crate::format::Modifier::Array { count, relative_offset, constant_offset, use_relative } => json!({
            "array": { "_0": {
                "count": count,
                "relativeOffset": [relative_offset.x, relative_offset.y, relative_offset.z],
                "constantOffset": [constant_offset.x, constant_offset.y, constant_offset.z],
                "useRelative": use_relative,
            } }
        }),
        crate::format::Modifier::Solidify { thickness, offset } => json!({
            "solidify": { "_0": { "thickness": thickness, "offset": offset } }
        }),
    };
    json!({
        "id": e.id,
        "isEnabled": e.is_enabled,
        "modifier": mod_val,
    })
}

// ── Simple Base64 (for ProjectFile.cameraState) ──

pub fn base64_encode(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 { TABLE[((n >> 6) & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[(n & 63) as usize] as char } else { '=' });
    }
    out
}

pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    fn val(b: u8) -> Option<u32> {
        match b {
            b'A'..=b'Z' => Some((b - b'A') as u32),
            b'a'..=b'z' => Some((b - b'a' + 26) as u32),
            b'0'..=b'9' => Some((b - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let clean: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if clean.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(clean.len() / 4 * 3);
    for chunk in clean.chunks_exact(4) {
        let a = val(chunk[0])?;
        let b = val(chunk[1])?;
        out.push(((a << 2) | (b >> 4)) as u8);
        if chunk[2] != b'=' {
            let c = val(chunk[2])?;
            out.push((((b & 0xF) << 4) | (c >> 2)) as u8);
            if chunk[3] != b'=' {
                let d = val(chunk[3])?;
                out.push((((c & 0x3) << 6) | d) as u8);
            }
        }
    }
    Some(out)
}

// ── Global Handle Registry ──

pub enum HeldObject {
    Applied(AppliedCommand),
    Control(Arc<CommandControl>),
    ImportJob(Arc<CommandControl>),
    ImportLoaded(io::LoadedImport),
    ImportPrepared(io::PreparedImport),
}

pub struct BridgeRegistry {
    next_id: AtomicI64,
    pub docs: Mutex<HashMap<i64, Arc<Mutex<EngineDoc>>>>,
    pub snapshots: Mutex<HashMap<i64, Arc<SceneSnapshot>>>,
    pub held: Mutex<HashMap<i64, HeldObject>>,
}

impl BridgeRegistry {
    pub fn global() -> &'static Self {
        static REG: OnceLock<BridgeRegistry> = OnceLock::new();
        REG.get_or_init(|| BridgeRegistry {
            next_id: AtomicI64::new(1),
            docs: Mutex::new(HashMap::new()),
            snapshots: Mutex::new(HashMap::new()),
            held: Mutex::new(HashMap::new()),
        })
    }

    pub fn alloc_id(&self) -> i64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn new_doc(&self) -> i64 {
        let id = self.alloc_id();
        self.docs.lock().unwrap().insert(id, Arc::new(Mutex::new(EngineDoc::starter())));
        id
    }

    pub fn get_doc(&self, id: i64) -> Option<Arc<Mutex<EngineDoc>>> {
        self.docs.lock().unwrap().get(&id).cloned()
    }

    pub fn free_doc(&self, id: i64) {
        self.docs.lock().unwrap().remove(&id);
    }

    pub fn put_snapshot(&self, snap: SceneSnapshot) -> i64 {
        let id = self.alloc_id();
        self.snapshots.lock().unwrap().insert(id, Arc::new(snap));
        id
    }

    pub fn get_snapshot(&self, id: i64) -> Option<Arc<SceneSnapshot>> {
        self.snapshots.lock().unwrap().get(&id).cloned()
    }

    pub fn free_snapshot(&self, id: i64) {
        self.snapshots.lock().unwrap().remove(&id);
    }

    pub fn put_held(&self, obj: HeldObject) -> i64 {
        let id = self.alloc_id();
        self.held.lock().unwrap().insert(id, obj);
        id
    }

    pub fn free_held(&self, id: i64) {
        self.held.lock().unwrap().remove(&id);
    }
}

pub fn with_doc<R>(handle: i64, default: R, f: impl FnOnce(&mut EngineDoc) -> R) -> R {
    let Some(arc) = BridgeRegistry::global().get_doc(handle) else { return default };
    let mut guard = arc.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}
