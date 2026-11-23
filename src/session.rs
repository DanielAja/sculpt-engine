//! `Session`: the engine as a host sees it, the same on every platform (DESIGN.md section 7).
//! It owns the scene's meshes, the per object sculpt targets, the undo history, the camera and the
//! brush. Hosts drive it with screen points (pixels, top left origin) and render it into their surface.
//! The native viewer, the wasm module and the uniffi bindings are thin wrappers around this type.

use crate::brush::{BrushKind, BrushParams};
use crate::camera::OrbitCamera;
use crate::format::{FormatError, Material, Mesh, ProjectFile};
use crate::mesh::SculptMesh;
use crate::render::{world_matrix, Renderer};
use crate::stroke::{History, SculptTarget};
use glam::{Mat4, Vec3};
use std::collections::HashMap;
use std::sync::Arc;

/// Builds an `Object` from the flat data hosts pass across the bindings (validated: this is a trust boundary).
/// `world` is a column major 4x4; `color` is sRGB r, g, b, a.
pub fn object_from_parts(id: &str, mesh_json: &str, world: &[f32], color: &[f32], metallic: f32, roughness: f32) -> Result<Object, String> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct MeshIn {
        positions: Vec<f32>,
        faces: Vec<Vec<u32>>,
        #[serde(default)]
        smooth_shading: bool,
    }
    let m: MeshIn = serde_json::from_str(mesh_json).map_err(|e| e.to_string())?;
    let n = (m.positions.len() / 3) as u32;
    if m.positions.len() % 3 != 0 || m.faces.iter().any(|f| f.len() < 3 || f.iter().any(|&i| i >= n)) {
        return Err("invalid mesh: positions must be x, y, z triples and faces valid indices".into());
    }
    if world.len() != 16 || !world.iter().all(|x| x.is_finite()) || color.len() != 4 {
        return Err("world must be 16 finite floats and color 4 floats".into());
    }
    let mesh = Mesh { positions: m.positions.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect(), faces: m.faces, smooth_shading: m.smooth_shading };
    let c = |i: usize| color[i].clamp(0.0, 1.0);
    Ok(Object {
        id: id.to_string(),
        mesh: Arc::new(mesh),
        world: Mat4::from_cols_slice(world),
        material: Material { name: String::new(), base_color: crate::format::Color { r: c(0), g: c(1), b: c(2), a: c(3) }, metallic: metallic.clamp(0.0, 1.0), roughness: roughness.clamp(0.0, 1.0) },
    })
}

pub struct Object {
    pub id: String,
    pub mesh: Arc<Mesh>,
    pub world: Mat4,
    pub material: Material,
}

/// A finished stroke: the `SceneCommand.setMesh` JSON for hosts with the Swift core, and the label.
#[derive(Debug, Clone, PartialEq)]
pub struct StrokeResult {
    pub object_id: String,
    pub command_json: String,
    pub label: String,
}

pub struct Session {
    pub renderer: Renderer,
    pub camera: OrbitCamera,
    pub brush: BrushParams,
    /// Dab spacing as a fraction of the radius.
    pub spacing: f32,
    pub size: (u32, u32),
    pub objects: Vec<Object>,
    pub history: History,
    /// The project the scene came from, kept so `project_json` writes everything else back unchanged.
    pub project: Option<ProjectFile>,
    targets: HashMap<String, SculptTarget>,
    sculpting: Option<String>,
}

impl Session {
    pub fn new(renderer: Renderer, size: (u32, u32)) -> Session {
        Session {
            renderer,
            camera: OrbitCamera::default(),
            brush: BrushParams { kind: BrushKind::Draw, radius: 4.0, strength: 0.6, symmetry: [true, false, false] },
            spacing: 0.25,
            size,
            objects: Vec::new(),
            history: History::default(),
            project: None,
            targets: HashMap::new(),
            sculpting: None,
        }
    }

    fn aspect(&self) -> f32 {
        self.size.0 as f32 / self.size.1.max(1) as f32
    }

    // ---- scene ----

    /// Opens a `.sculpt` project. Each mesh displays with its own `smoothShading` flag, as in Sculptor.
    pub fn open_project(&mut self, bytes: &[u8]) -> Result<(), FormatError> {
        let p = ProjectFile::decode(bytes)?;
        let objs = p.scene.objects.iter().filter(|o| o.is_visible).map(|o| Object { id: o.id.clone(), mesh: Arc::new(o.mesh.clone()), world: world_matrix(&p, &o.id), material: o.material.clone() }).collect();
        self.set_objects(objs);
        self.project = Some(p);
        self.frame_all();
        Ok(())
    }

    /// The project with sculpted meshes written back, byte compatible with Sculptor.
    pub fn project_json(&self) -> Option<String> {
        let mut p = self.project.clone()?;
        for o in &mut p.scene.objects {
            if let Some(s) = self.objects.iter().find(|x| x.id == o.id) {
                o.mesh = (*s.mesh).clone();
            }
        }
        Some(p.encode())
    }

    pub fn set_objects(&mut self, objects: Vec<Object>) {
        self.targets.clear();
        self.sculpting = None;
        self.history = History::default();
        self.renderer.clear();
        for o in &objects {
            self.upload(o);
        }
        self.objects = objects;
    }

    fn upload(&mut self, o: &Object) {
        self.renderer.upsert(&o.id, &SculptMesh::from_mesh(&o.mesh), o.world, &o.material);
    }

    /// Adds or replaces one object from the host's scene (hosts that own the scene through the Swift core).
    /// `mesh` is the object's base mesh (n-gon faces), so a stroke's `setMesh` keeps its faces.
    pub fn upsert_object(&mut self, o: Object) {
        if self.sculpting.as_deref() == Some(o.id.as_str()) {
            self.sculpting = None;
        }
        self.targets.remove(&o.id);
        self.upload(&o);
        match self.objects.iter().position(|x| x.id == o.id) {
            Some(i) => self.objects[i] = o,
            None => self.objects.push(o),
        }
    }

    pub fn remove_object(&mut self, id: &str) {
        self.objects.retain(|o| o.id != id);
        self.targets.remove(id);
        if self.sculpting.as_deref() == Some(id) {
            self.sculpting = None;
        }
        self.renderer.remove(id);
    }

    pub fn set_transform(&mut self, id: &str, world: Mat4) -> bool {
        let Some(o) = self.objects.iter_mut().find(|o| o.id == id) else { return false };
        o.world = world;
        if let Some(t) = self.targets.get_mut(id) {
            t.set_world(world);
        }
        self.renderer.set_transform(id, world)
    }

    pub fn set_material(&mut self, id: &str, material: Material) -> bool {
        let Some(o) = self.objects.iter_mut().find(|o| o.id == id) else { return false };
        o.material = material;
        self.renderer.set_material(id, &o.material)
    }

    pub fn set_visible(&mut self, id: &str, visible: bool) -> bool {
        self.renderer.set_visible(id, visible)
    }

    /// Screen position (pixels, top left origin) and clip depth of a world point; None behind the camera.
    pub fn project(&self, p: Vec3) -> Option<(f32, f32, f32)> {
        let c = self.camera.view_projection(self.aspect()) * p.extend(1.0);
        if c.w <= 0.0 {
            return None;
        }
        let n = c.truncate() / c.w;
        Some(((n.x + 1.0) * 0.5 * self.size.0 as f32, (1.0 - n.y) * 0.5 * self.size.1 as f32, n.z))
    }

    pub fn frame_all(&mut self) {
        let (a, b) = self.renderer.scene_bounds();
        self.camera = self.camera.fitted(a, b, self.aspect());
    }

    // ---- view ----

    pub fn resize(&mut self, width: u32, height: u32) {
        self.size = (width.max(1), height.max(1));
    }

    pub fn render(&mut self, target: &wgpu::TextureView) {
        self.renderer.render(target, self.size, &self.camera);
    }

    /// World ray through a screen point (pixels).
    pub fn ray(&self, x: f32, y: f32) -> (Vec3, Vec3) {
        self.camera.ray(x / self.size.0 as f32, y / self.size.1 as f32, self.aspect())
    }

    fn target(&mut self, id: &str) -> &mut SculptTarget {
        if !self.targets.contains_key(id) {
            let o = self.objects.iter().find(|o| o.id == id).expect("known object");
            let t = SculptTarget::new(id, &o.mesh, o.world);
            self.targets.insert(id.into(), t);
        }
        self.targets.get_mut(id).unwrap()
    }

    /// The closest object under a screen point and the world hit point.
    pub fn pick(&mut self, x: f32, y: f32) -> Option<(String, Vec3)> {
        let (o, d) = self.ray(x, y);
        let ids: Vec<String> = self.objects.iter().map(|o| o.id.clone()).collect();
        let mut best: Option<(f32, String, Vec3)> = None;
        for id in ids {
            if let Some(h) = self.target(&id).pick(o, d) {
                let dist = (h.point - o).length();
                if best.as_ref().is_none_or(|b| dist < b.0) {
                    best = Some((dist, id, h.point));
                }
            }
        }
        best.map(|b| (b.1, b.2))
    }

    // ---- sculpting ----

    /// Starts a stroke on the object under the point. False when the point misses everything.
    pub fn sculpt_begin(&mut self, x: f32, y: f32, pressure: f32) -> bool {
        let Some((id, _)) = self.pick(x, y) else { return false };
        self.target(&id).begin_stroke();
        self.sculpting = Some(id);
        self.sculpt_move(x, y, pressure);
        true
    }

    pub fn sculpt_move(&mut self, x: f32, y: f32, pressure: f32) {
        let Some(id) = self.sculpting.clone() else { return };
        let (o, d) = self.ray(x, y);
        let p = BrushParams { strength: self.brush.strength * pressure.clamp(0.0, 1.0), ..self.brush };
        let spacing = self.spacing;
        let t = self.target(&id);
        if let Some(h) = t.pick(o, d) {
            t.drag_to(h.point, &p, spacing);
        }
        let dirty = t.take_dirty();
        let mesh = &self.targets[&id].mesh;
        self.renderer.update_vertices(&id, mesh, &dirty);
    }

    /// Ends the stroke: one "Sculpt" history step, and the `setMesh` command for Swift hosts.
    pub fn sculpt_end(&mut self) -> Option<StrokeResult> {
        let id = self.sculpting.take()?;
        let (mesh, cmd) = self.target(&id).end_stroke()?;
        let o = self.objects.iter_mut().find(|o| o.id == id)?;
        let after = Arc::new(mesh);
        self.history.push("Sculpt", &id, o.mesh.clone(), after.clone());
        o.mesh = after;
        Some(StrokeResult { object_id: id, command_json: cmd, label: "Sculpt".into() })
    }

    pub fn sculpt_cancel(&mut self) {
        let Some(id) = self.sculpting.take() else { return };
        let t = self.target(&id);
        t.cancel_stroke();
        let dirty = t.take_dirty();
        let mesh = &self.targets[&id].mesh;
        self.renderer.update_vertices(&id, mesh, &dirty);
    }

    // ---- undo ----

    pub fn undo(&mut self) -> bool {
        let step = self.history.undo();
        self.restore(step)
    }

    pub fn redo(&mut self) -> bool {
        let step = self.history.redo();
        self.restore(step)
    }

    /// Replaces an object's mesh from the host (its own undo, or any Swift core edit).
    pub fn set_mesh(&mut self, id: &str, mesh: Mesh) {
        self.restore(Some((id.to_string(), Arc::new(mesh))));
    }

    fn restore(&mut self, step: Option<(String, Arc<Mesh>)>) -> bool {
        let Some((id, mesh)) = step else { return false };
        let Some(i) = self.objects.iter().position(|o| o.id == id) else { return false };
        self.objects[i].mesh = mesh.clone();
        self.target(&id).set_mesh(&mesh);
        let (w, m) = (self.objects[i].world, self.objects[i].material.clone());
        let sm = &self.targets[&id].mesh;
        self.renderer.upsert(&id, sm, w, &m);
        true
    }
}
