//! Strokes on one object, and an undo history for hosts that don't have Sculptor's Swift `Document`.
//!
//! A stroke previews on the working mesh and never enters history until `end_stroke`, which returns the
//! `SceneCommand.setMesh` JSON the Swift core accepts: one "Sculpt" undo step, as in the web port.

use crate::brush::{self, BrushParams, Scratch};
use crate::bvh::{Bvh, Hit};
use crate::format::{set_mesh_command, Mesh};
use crate::mesh::SculptMesh;
use glam::{Mat4, Vec3};
use std::sync::Arc;

/// The object being sculpted: its mesh, BVH and world placement. Lives across strokes so the BVH is
/// built once per object, not once per stroke.
pub struct SculptTarget {
    pub id: String,
    pub mesh: SculptMesh,
    pub bvh: Bvh,
    world: Mat4,
    world_inverse: Mat4,
    /// Average absolute scale: world radius / scale = local radius (as the web port does).
    scale: f32,
    scratch: Scratch,
    stroke: Option<Stroke>,
    /// Vertices changed since the host last asked (`take_dirty`), for partial GPU uploads.
    dirty: Vec<u32>,
}

struct Stroke {
    /// Sparse undo for `cancel`: (vertex, position before the stroke), each vertex once.
    before: Vec<(u32, Vec3)>,
    saved: Vec<bool>,
    dabs: usize,
    last: Option<Vec3>,
}

impl SculptTarget {
    pub fn new(id: impl Into<String>, mesh: &Mesh, world: Mat4) -> SculptTarget {
        let mesh = SculptMesh::from_mesh(mesh);
        let bvh = Bvh::build(&mesh.positions, &mesh.tris);
        let (s, _, _) = world.to_scale_rotation_translation();
        SculptTarget {
            id: id.into(),
            mesh,
            bvh,
            world,
            world_inverse: world.inverse(),
            scale: ((s.x.abs() + s.y.abs() + s.z.abs()) / 3.0).max(1e-6),
            scratch: Scratch::default(),
            stroke: None,
            dirty: Vec::new(),
        }
    }

    /// Closest hit of a world space ray, in world space.
    pub fn pick(&self, origin: Vec3, dir: Vec3) -> Option<Hit> {
        let o = self.world_inverse.transform_point3(origin);
        let d = self.world_inverse.transform_vector3(dir);
        self.bvh.ray(&self.mesh.positions, &self.mesh.tris, o, d).map(|h| Hit { point: self.world.transform_point3(h.point), ..h })
    }

    pub fn begin_stroke(&mut self) {
        self.stroke = Some(Stroke { before: Vec::new(), saved: vec![false; self.mesh.positions.len()], dabs: 0, last: None });
    }

    pub fn in_stroke(&self) -> bool {
        self.stroke.is_some()
    }

    /// Dabs at each world space point, exactly as given (the web port's `sculptMove`).
    pub fn dab_points(&mut self, points: &[Vec3], params: &BrushParams) {
        for &p in points {
            self.dab_world(p, params);
        }
    }

    /// Dabs along the segment from the previous dab to `p`, every `spacing x radius` (pointer input).
    pub fn drag_to(&mut self, p: Vec3, params: &BrushParams, spacing: f32) {
        let Some(st) = &self.stroke else { return };
        let Some(last) = st.last else { return self.dab_world(p, params) };
        let step = (params.radius * spacing.max(0.05)).max(1e-6);
        let len = (p - last).length();
        for i in 1..=(len / step) as usize {
            self.dab_world(last + (p - last) * (i as f32 * step / len), params);
        }
    }

    fn dab_world(&mut self, p: Vec3, params: &BrushParams) {
        let Some(st) = &mut self.stroke else { return };
        let local = self.world_inverse.transform_point3(p);
        let lp = BrushParams { radius: params.radius / self.scale, ..*params };
        let mut moved = Vec::new();
        let moved_by = brush::dab(&mut self.mesh, &self.bvh, &mut self.scratch, local, &lp, &mut moved);
        st.dabs += 1;
        st.last = Some(p);
        let mut vs = Vec::with_capacity(moved.len());
        for (v, old) in moved {
            if !st.saved[v as usize] {
                st.saved[v as usize] = true;
                st.before.push((v, old));
            }
            vs.push(v);
        }
        vs.sort_unstable();
        vs.dedup();
        let changed = self.mesh.update_normals(&vs);
        self.dirty.extend_from_slice(&changed);
        // Keep queries exact without a refit; refit once vertices have drifted a quarter radius.
        self.bvh.slack += moved_by;
        if self.bvh.slack > lp.radius * 0.25 {
            self.bvh.refit(&self.mesh.positions, &self.mesh.tris);
        }
    }

    /// Ends the stroke. Returns the mesh and the `setMesh` command, or `None` when nothing was dabbed.
    pub fn end_stroke(&mut self) -> Option<(Mesh, String)> {
        let st = self.stroke.take()?;
        if st.dabs == 0 || st.before.is_empty() {
            return None;
        }
        self.bvh.refit(&self.mesh.positions, &self.mesh.tris);
        let mesh = self.mesh.to_mesh();
        let cmd = set_mesh_command(&self.id, &mesh);
        Some((mesh, cmd))
    }

    /// Restores the mesh to how it was before the stroke.
    pub fn cancel_stroke(&mut self) {
        let Some(st) = self.stroke.take() else { return };
        let vs: Vec<u32> = st.before.iter().map(|b| b.0).collect();
        for (v, p) in st.before {
            self.mesh.positions[v as usize] = p;
        }
        let changed = self.mesh.update_normals(&vs);
        self.dirty.extend_from_slice(&changed);
        self.bvh.refit(&self.mesh.positions, &self.mesh.tris);
    }

    /// Replaces the mesh (undo, redo, or an edit from the host); rebuilds the BVH.
    pub fn set_mesh(&mut self, mesh: &Mesh) {
        self.stroke = None;
        self.mesh = SculptMesh::from_mesh(mesh);
        self.bvh = Bvh::build(&self.mesh.positions, &self.mesh.tris);
        self.dirty.clear();
    }

    /// Vertices whose position or normal changed since the last call, sorted and unique.
    pub fn take_dirty(&mut self) -> Vec<u32> {
        let mut d = std::mem::take(&mut self.dirty);
        d.sort_unstable();
        d.dedup();
        d
    }
}

/// Linear undo of whole meshes (the Swift `Document`'s model: label plus the state before).
/// Meshes are shared (`Arc`), so a history entry costs one mesh, not a scene.
pub struct History {
    entries: Vec<Entry>,
    /// Entries before this index are undoable; after it, redoable.
    cursor: usize,
    pub limit: usize,
    pub byte_budget: usize,
}

#[derive(Clone)]
pub struct Entry {
    pub label: String,
    pub object: String,
    pub before: Arc<Mesh>,
    pub after: Arc<Mesh>,
}

fn mesh_bytes(m: &Mesh) -> usize {
    m.positions.len() * 12 + m.faces.iter().map(|f| f.len() * 4 + 24).sum::<usize>()
}

impl Default for History {
    fn default() -> Self {
        History { entries: Vec::new(), cursor: 0, limit: 200, byte_budget: 512 << 20 }
    }
}

impl History {
    pub fn push(&mut self, label: &str, object: &str, before: Arc<Mesh>, after: Arc<Mesh>) {
        self.entries.truncate(self.cursor);
        self.entries.push(Entry { label: label.into(), object: object.into(), before, after });
        // Drop the oldest steps past the count limit or the memory budget (always keep the newest).
        while self.entries.len() > 1
            && (self.entries.len() > self.limit || self.entries.iter().map(|e| mesh_bytes(&e.before) + mesh_bytes(&e.after)).sum::<usize>() > self.byte_budget)
        {
            self.entries.remove(0);
        }
        self.cursor = self.entries.len();
    }
    pub fn undo_label(&self) -> Option<&str> {
        self.cursor.checked_sub(1).map(|i| self.entries[i].label.as_str())
    }
    pub fn redo_label(&self) -> Option<&str> {
        self.entries.get(self.cursor).map(|e| e.label.as_str())
    }
    /// The object and the mesh to restore.
    pub fn undo(&mut self) -> Option<(String, Arc<Mesh>)> {
        let i = self.cursor.checked_sub(1)?;
        self.cursor = i;
        Some((self.entries[i].object.clone(), self.entries[i].before.clone()))
    }
    pub fn redo(&mut self) -> Option<(String, Arc<Mesh>)> {
        let e = self.entries.get(self.cursor)?;
        self.cursor += 1;
        Some((e.object.clone(), e.after.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brush::BrushKind;
    use crate::mesh::uv_sphere;

    fn draw() -> BrushParams {
        BrushParams { kind: BrushKind::Draw, radius: 4.0, strength: 1.0, symmetry: [false; 3] }
    }

    #[test]
    fn stroke_end_cancel_and_world_transform() {
        let sphere = uv_sphere(10.0, 32, 16);
        // Object scaled 2x and moved: a world radius of 4 is a local radius of 2.
        let world = Mat4::from_scale_rotation_translation(Vec3::splat(2.0), glam::Quat::IDENTITY, Vec3::new(100.0, 0.0, 0.0));
        let mut t = SculptTarget::new("ID", &sphere, world);
        let hit = t.pick(Vec3::new(200.0, 0.0, 0.0), Vec3::NEG_X).unwrap();
        assert!((hit.point - Vec3::new(120.0, 0.0, 0.0)).length() < 0.5, "{}", hit.point);

        t.begin_stroke();
        t.dab_points(&[hit.point], &draw());
        assert!(t.mesh.positions != sphere.positions);
        t.cancel_stroke();
        assert_eq!(t.mesh.positions, sphere.positions);
        assert!(!t.take_dirty().is_empty());

        t.begin_stroke();
        assert!(t.end_stroke().is_none(), "an empty stroke is not an undo step");
        t.begin_stroke();
        t.drag_to(hit.point, &draw(), 0.25);
        t.drag_to(hit.point + Vec3::new(0.0, 0.0, 6.0), &draw(), 0.25);
        let (mesh, cmd) = t.end_stroke().unwrap();
        assert_eq!(mesh.faces, sphere.faces, "displacement keeps the n-gon faces");
        assert!(cmd.starts_with(r#"{"setMesh":{"_0":{"rawValue":"ID"},"_1":{"faces":[[0,1,2]"#), "{}", &cmd[..80]);
        // Only vertices near +X moved.
        for (a, b) in mesh.positions.iter().zip(&sphere.positions) {
            if a != b {
                assert!(b.x > 5.0, "{b}");
            }
        }
    }

    #[test]
    fn history_undo_redo_and_limits() {
        let a = Arc::new(uv_sphere(1.0, 8, 4));
        let b = Arc::new(uv_sphere(2.0, 8, 4));
        let mut h = History { limit: 2, ..Default::default() };
        h.push("Sculpt", "X", a.clone(), b.clone());
        h.push("Sculpt", "X", b.clone(), a.clone());
        h.push("Smooth", "X", a.clone(), b.clone());
        assert_eq!(h.undo_label(), Some("Smooth"));
        assert!(Arc::ptr_eq(&h.undo().unwrap().1, &a));
        assert!(Arc::ptr_eq(&h.undo().unwrap().1, &b));
        assert!(h.undo().is_none(), "limit 2 dropped the oldest step");
        assert!(Arc::ptr_eq(&h.redo().unwrap().1, &a));
        h.push("New", "X", a.clone(), a.clone());
        assert!(h.redo().is_none(), "a new step clears redo");
    }
}
