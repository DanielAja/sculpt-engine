//! `Session`: the engine as a host sees it, the same on every platform (DESIGN.md section 7).
//! It owns the scene's meshes, the per object sculpt targets, the undo history, the camera and the
//! brush. Hosts drive it with screen points (pixels, top left origin) and render it into their surface.
//! The native viewer, the wasm module and the uniffi bindings are thin wrappers around this type.

use crate::brush::{BrushKind, BrushParams, Falloff};
use crate::camera::OrbitCamera;
use crate::draw3d::{build_tube_add_mesh_command, Draw3DOptions, Draw3DPoint};
use crate::format::{set_mesh_command, FormatError, Material, Mesh, ProjectFile};
use crate::mesh::SculptMesh;
use crate::remesh::voxel_remesh;
use crate::render::{world_matrix, Renderer};
use crate::stroke::{History, SculptTarget};
use glam::{Mat4, Vec3};
use std::collections::HashMap;
use std::sync::Arc;

/// Builds a `Mesh` from flat binary buffers (`positions` x/y/z triples, `face_sizes` counts per n-gon,
/// `face_indices` flattened vertex indices). Validated at the host trust boundary (zero JSON overhead).
pub fn mesh_from_buffers(positions: &[f32], face_sizes: &[u32], face_indices: &[u32], smooth_shading: bool) -> Result<Mesh, String> {
    if positions.len() % 3 != 0 || !positions.iter().all(|x| x.is_finite()) {
        return Err("invalid positions: must be finite x, y, z triples".into());
    }
    let n = (positions.len() / 3) as u32;
    let total: usize = face_sizes.iter().map(|&s| s as usize).sum();
    if total != face_indices.len() {
        return Err("face_sizes sum does not match face_indices length".into());
    }
    let mut faces = Vec::with_capacity(face_sizes.len());
    let mut off = 0usize;
    for &sz in face_sizes {
        let len = sz as usize;
        if len < 3 {
            return Err("each face must have at least 3 vertices".into());
        }
        let slice = &face_indices[off..off + len];
        if slice.iter().any(|&i| i >= n) {
            return Err("face index out of bounds".into());
        }
        faces.push(slice.to_vec());
        off += len;
    }
    let pos = positions.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect();
    Ok(Mesh { positions: pos, faces, smooth_shading })
}

/// Zero-JSON binary variant of `object_from_parts` for high-throughput Sculptor host interop (E4.5).
pub fn object_from_buffers(
    id: &str,
    positions: &[f32],
    face_sizes: &[u32],
    face_indices: &[u32],
    smooth_shading: bool,
    world: &[f32],
    color: &[f32],
    metallic: f32,
    roughness: f32,
) -> Result<Object, String> {
    if world.len() != 16 || !world.iter().all(|x| x.is_finite()) || color.len() != 4 {
        return Err("world must be 16 finite floats and color 4 floats".into());
    }
    let mesh = mesh_from_buffers(positions, face_sizes, face_indices, smooth_shading)?;
    let c = |i: usize| color[i].clamp(0.0, 1.0);
    Ok(Object {
        id: id.to_string(),
        mesh: Arc::new(mesh),
        world: Mat4::from_cols_slice(world),
        material: Material {
            name: String::new(),
            base_color: crate::format::Color { r: c(0), g: c(1), b: c(2), a: c(3) },
            metallic: metallic.clamp(0.0, 1.0),
            roughness: roughness.clamp(0.0, 1.0),
        },
    })
}

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

struct Draw3DStroke {
    points: Vec<Draw3DPoint>,
    snap_surface: bool,
    surface_offset: f32,
    plane_depth: f32,
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
    drawing3d: Option<Draw3DStroke>,
}

impl Session {
    pub fn new(renderer: Renderer, size: (u32, u32)) -> Session {
        Session {
            renderer,
            camera: OrbitCamera::default(),
            brush: BrushParams { kind: BrushKind::Draw, radius: 4.0, strength: 0.6, symmetry: [true, false, false], ..BrushParams::default() },
            spacing: 0.25,
            size,
            objects: Vec::new(),
            history: History::default(),
            project: None,
            targets: HashMap::new(),
            sculpting: None,
            drawing3d: None,
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

    pub fn set_brush_ex(
        &mut self,
        kind: &str,
        radius: f32,
        strength: f32,
        falloff: &str,
        sym_x: bool,
        sym_y: bool,
        sym_z: bool,
        radial_count: u32,
    ) {
        self.brush = BrushParams {
            kind: BrushKind::parse(kind),
            radius: radius.max(0.1),
            strength: strength.clamp(-1.0, 1.0),
            symmetry: [sym_x, sym_y, sym_z],
            falloff: Falloff::parse(falloff),
            radial_symmetry: radial_count.clamp(1, 16),
        };
    }

    /// Starts a stroke on the object under the point. False when the point misses everything.
    pub fn sculpt_begin(&mut self, x: f32, y: f32, pressure: f32) -> bool {
        let Some((id, hit_pt)) = self.pick(x, y) else { return false };
        if self.brush.kind == BrushKind::Grab {
            let p = BrushParams { strength: self.brush.strength * pressure.clamp(0.0, 1.0), ..self.brush };
            let plane_n = self.camera.backward();
            self.target(&id).begin_grab(hit_pt, plane_n, &p);
            self.sculpting = Some(id);
            return true;
        }
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
        if t.is_grabbing() {
            t.drag_grab_ray(o, d);
        } else if let Some(h) = t.pick(o, d) {
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

    /// Applies `"clear"`, `"invert"`, or `"blur"` to an object's sculpt mask and updates its GPU shading.
    pub fn mask_op(&mut self, id: &str, op: &str) -> bool {
        if !self.objects.iter().any(|o| o.id == id) {
            return false;
        }
        let ok = self.target(id).mask_op(op);
        if ok {
            let dirty = self.target(id).take_dirty();
            let mesh = &self.targets[id].mesh;
            self.renderer.update_vertices(id, mesh, &dirty);
        }
        ok
    }

    // ---- free voxel / SDF remesh ----

    /// Remeshes object `id` into a uniform-density watertight manifold quad mesh at `resolution`
    /// (`24..=128` voxels along the longest axis). Pushes to history and returns `(Mesh, setMesh JSON)`.
    pub fn voxel_remesh(&mut self, id: &str, resolution: u32) -> Result<(Mesh, String), String> {
        let idx = self.objects.iter().position(|o| o.id == id).ok_or_else(|| format!("unknown object {id}"))?;
        let (remeshed, cmd) = {
            let t = self.target(id);
            let m = voxel_remesh(&t.mesh, &t.bvh, resolution)?;
            let cmd = set_mesh_command(id, &m);
            (m, cmd)
        };
        let after = Arc::new(remeshed.clone());
        let before = self.objects[idx].mesh.clone();
        self.history.push("Voxel Remesh", id, before, after.clone());
        self.objects[idx].mesh = after.clone();
        self.target(id).set_mesh(&after);
        let (w, mat) = (self.objects[idx].world, self.objects[idx].material.clone());
        let sm = &self.targets[id].mesh;
        self.renderer.upsert(id, sm, w, &mat);
        Ok((remeshed, cmd))
    }

    // ---- Draw in 3D ----

    /// Picks a 3D world point from screen pixel `(x, y)`:
    /// - If `snap_surface` is true and a scene object is under `(x, y)`, returns the surface hit
    ///   point offset toward the camera by `surface_offset`.
    /// - Otherwise intersects the ray with the camera-facing plane at `camera.target() + backward * plane_depth`.
    pub fn draw3d_pick_point(&mut self, x: f32, y: f32, snap_surface: bool, surface_offset: f32, plane_depth: f32) -> Option<Vec3> {
        let (o, d) = self.ray(x, y);
        if snap_surface {
            if let Some((_, hit_pt)) = self.pick(x, y) {
                return Some(hit_pt - d * surface_offset);
            }
        }
        let plane_n = self.camera.backward();
        let plane_p = self.camera.target() + plane_n * plane_depth;
        let denom = d.dot(plane_n);
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = (plane_p - o).dot(plane_n) / denom;
        if t <= 0.0 {
            return None;
        }
        Some(o + d * t)
    }

    /// Starts an interactive Draw in 3D stroke at screen pixel `(x, y)`. Returns the picked 3D world point.
    pub fn draw3d_begin(&mut self, x: f32, y: f32, radius: f32, snap_surface: bool, surface_offset: f32, plane_depth: f32) -> Option<Vec3> {
        let pt = self.draw3d_pick_point(x, y, snap_surface, surface_offset, plane_depth)?;
        self.drawing3d = Some(Draw3DStroke {
            points: vec![Draw3DPoint { pos: pt, radius: radius.max(0.2) }],
            snap_surface,
            surface_offset,
            plane_depth,
        });
        Some(pt)
    }

    /// Appends a sampled 3D point during an active Draw in 3D stroke. Returns the picked world point.
    pub fn draw3d_move(&mut self, x: f32, y: f32, radius: f32) -> Option<Vec3> {
        let (snap_surface, surface_offset, plane_depth) = {
            let st = self.drawing3d.as_ref()?;
            (st.snap_surface, st.surface_offset, st.plane_depth)
        };
        let pt = self.draw3d_pick_point(x, y, snap_surface, surface_offset, plane_depth)?;
        let st = self.drawing3d.as_mut()?;
        let r = radius.max(0.2);
        let min_step = (r * 0.25).clamp(0.3, 4.0);
        if let Some(last) = st.points.last() {
            if (pt - last.pos).length() < min_step {
                return None;
            }
        }
        st.points.push(Draw3DPoint { pos: pt, radius: r });
        Some(pt)
    }

    /// Ends the Draw in 3D stroke and returns the `SceneCommand.addMesh` JSON string.
    pub fn draw3d_end(&mut self, options_json: &str) -> Result<String, String> {
        let st = self.drawing3d.take().ok_or_else(|| "no active Draw in 3D stroke".to_string())?;
        let (opts, name) = Draw3DOptions::from_json(options_json);
        let (_, _, cmd) = build_tube_add_mesh_command(&st.points, &opts, &name, None)?;
        Ok(cmd)
    }

    pub fn draw3d_cancel(&mut self) {
        self.drawing3d = None;
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
