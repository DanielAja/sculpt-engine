//! Strokes on one object, and an undo history for hosts that don't have Sculptor's Swift `Document`.
//!
//! A stroke previews on the working mesh and never enters history until `end_stroke`, which returns the
//! `SceneCommand.setMesh` JSON the Swift core accepts: one "Sculpt" undo step, as in the web port.

use crate::brush::{self, BrushKind, BrushParams, Scratch};
use crate::bvh::{Bvh, Hit};
use crate::format::{set_mesh_command, Mesh};
use crate::mesh::SculptMesh;
use glam::{Mat3, Mat4, Vec3};
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
    moved_buf: Vec<(u32, Vec3)>,
    stroke: Option<Stroke>,
    /// Vertices changed since the host last asked (`take_dirty`), for partial GPU uploads.
    dirty: Vec<u32>,
}

struct GrabState {
    origin_world: Vec3,
    plane_normal: Vec3,
    /// Per grabbed vertex: `(vertex_index, initial_local_pos, accumulated_linear_delta_matrix)`.
    /// New local position = `initial_local_pos + delta_mat * delta_local`.
    entries: Vec<(u32, Vec3, Mat3)>,
    verts: Vec<u32>,
    last_delta_len: f32,
}

struct Stroke {
    /// Sparse undo for `cancel`: (vertex, position before the stroke), each vertex once.
    before: Vec<(u32, Vec3)>,
    saved: Vec<bool>,
    dabs: usize,
    last: Option<Vec3>,
    grab: Option<GrabState>,
}

impl SculptTarget {
    pub fn new(id: impl Into<String>, mesh: &Mesh, world: Mat4) -> SculptTarget {
        let mesh = SculptMesh::from_mesh(mesh);
        let bvh = Bvh::build(&mesh.positions, &mesh.tris);
        let vlen = mesh.positions.len();
        let (s, _, _) = world.to_scale_rotation_translation();
        SculptTarget {
            id: id.into(),
            mesh,
            bvh,
            world,
            world_inverse: world.inverse(),
            scale: ((s.x.abs() + s.y.abs() + s.z.abs()) / 3.0).max(1e-6),
            scratch: Scratch::new(vlen),
            moved_buf: Vec::new(),
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
        self.stroke = Some(Stroke {
            before: Vec::new(),
            saved: vec![false; self.mesh.positions.len()],
            dabs: 0,
            last: None,
            grab: None,
        });
    }

    /// Starts a Nomad-style Grab stroke at `hit_world` dragging on a plane with normal `plane_normal`.
    /// Captures all vertices inside the brush footprint (including 3-axis & radial symmetry) with
    /// their falloff and mask-protected weights.
    pub fn begin_grab(&mut self, hit_world: Vec3, plane_normal: Vec3, params: &BrushParams) {
        self.begin_stroke();
        let Some(st) = &mut self.stroke else { return };
        let local_center = self.world_inverse.transform_point3(hit_world);
        let r = (params.radius / self.scale).max(1e-5);
        let inv_r2 = 1.0 / (r * r);
        let abs_st = params.strength.abs().clamp(0.0, 1.0);
        let syms = brush::symmetry_transforms(local_center, r, params.symmetry, params.radial_symmetry);

        let mut stamp = crate::bvh::Stamp::new(self.mesh.positions.len());
        let mut hits = Vec::new();
        let mut accum: std::collections::HashMap<u32, (Vec3, Mat3, f32)> = std::collections::HashMap::new();
        for (sym_c, sym_mat) in syms {
            self.bvh.sphere_vertices(&self.mesh.positions, &self.mesh.tris, sym_c, r, &mut stamp, &mut hits);
            for &v in &hits {
                let pos = self.mesh.positions[v as usize];
                let d2 = (pos - sym_c).length_squared();
                let f = params.falloff.eval(d2, inv_r2);
                let mask_unprotected = (1.0 - self.mesh.masks.get(v as usize).copied().unwrap_or(0.0)).clamp(0.0, 1.0);
                let w = f * mask_unprotected * abs_st;
                if w > 1e-6 {
                    let entry = accum.entry(v).or_insert((pos, Mat3::ZERO, 0.0));
                    entry.1 += sym_mat * w;
                    entry.2 += w;
                }
            }
        }
        let mut entries: Vec<(u32, Vec3, Mat3)> = accum
            .into_iter()
            .map(|(v, (p, m, w_sum))| {
                // Clamp overlapping symmetry/radial weights near symmetry axes so vertices never overshoot 100% drag.
                let mat = if w_sum > abs_st && w_sum > 1e-6 { m * (abs_st / w_sum) } else { m };
                (v, p, mat)
            })
            .collect();
        entries.sort_unstable_by_key(|e| e.0);
        let mut verts = Vec::with_capacity(entries.len());
        for &(v, p, _) in &entries {
            st.saved[v as usize] = true;
            st.before.push((v, p));
            verts.push(v);
        }
        st.grab = Some(GrabState {
            origin_world: hit_world,
            plane_normal: plane_normal.normalize_or_zero(),
            entries,
            verts,
            last_delta_len: 0.0,
        });
    }

    pub fn is_grabbing(&self) -> bool {
        self.stroke.as_ref().is_some_and(|s| s.grab.is_some())
    }

    /// Moves a Grab stroke by intersecting the ray `(origin, dir)` with the grab drag plane.
    pub fn drag_grab_ray(&mut self, origin: Vec3, dir: Vec3) {
        let Some((p0, pn)) = self.stroke.as_ref().and_then(|s| s.grab.as_ref()).map(|g| (g.origin_world, g.plane_normal)) else {
            return;
        };
        let denom = dir.dot(pn);
        if denom.abs() < 1e-6 {
            return;
        }
        let t = (p0 - origin).dot(pn) / denom;
        if t <= 0.0 {
            return;
        }
        let cur_world = origin + dir * t;
        self.move_grab(cur_world - p0);
    }

    /// Translates the grabbed vertices by `delta_world` (in world mm).
    pub fn move_grab(&mut self, delta_world: Vec3) {
        let Some(st) = &mut self.stroke else { return };
        let Some(grab) = &mut st.grab else { return };
        if grab.entries.is_empty() {
            return;
        }
        let delta_local = self.world_inverse.transform_vector3(delta_world);
        let mut max_step = 0.0f32;
        for &(v, init_pos, mat) in &grab.entries {
            let np = init_pos + mat * delta_local;
            let prev = self.mesh.positions[v as usize];
            max_step = max_step.max((np - prev).length_squared());
            self.mesh.positions[v as usize] = np;
        }
        st.dabs += 1;
        let step_len = max_step.sqrt();
        grab.last_delta_len += step_len;
        self.mesh.update_normals_stamped(
            &grab.verts,
            &mut self.scratch.poly_stamp,
            &mut self.scratch.vert_stamp,
            &mut self.dirty,
        );
        self.bvh.slack += step_len;
        if self.bvh.slack > 2.0 {
            self.bvh.refit(&self.mesh.positions, &self.mesh.tris);
        }
    }

    /// Applies a mask operation (`"clear"`, `"invert"`, `"blur"`) and marks all vertices dirty so the
    /// renderer refreshes vertex mask shading.
    pub fn mask_op(&mut self, op: &str) -> bool {
        match op {
            "clear" => self.mesh.mask_clear(),
            "invert" => self.mesh.mask_invert(),
            "blur" => self.mesh.mask_blur(),
            _ => return false,
        }
        self.dirty.extend(0..self.mesh.positions.len() as u32);
        true
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
        self.moved_buf.clear();
        let moved_by = brush::dab_and_update_normals(
            &mut self.mesh,
            &self.bvh,
            &mut self.scratch,
            local,
            &lp,
            &mut self.moved_buf,
            &mut self.dirty,
        );
        if params.kind == BrushKind::Mask {
            st.dabs += 1;
            st.last = Some(p);
            return;
        }
        st.dabs += 1;
        st.last = Some(p);
        for &(v, old) in &self.moved_buf {
            if !st.saved[v as usize] {
                st.saved[v as usize] = true;
                st.before.push((v, old));
            }
        }
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

    /// Updates the object's world transform without rebuilding the mesh or BVH.
    pub fn set_world(&mut self, world: Mat4) {
        let (s, _, _) = world.to_scale_rotation_translation();
        self.world = world;
        self.world_inverse = world.inverse();
        self.scale = ((s.x.abs() + s.y.abs() + s.z.abs()) / 3.0).max(1e-6);
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
        BrushParams { kind: BrushKind::Draw, radius: 4.0, strength: 1.0, ..Default::default() }
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

    #[test]
    fn grab_translates_captured_footprint_with_symmetry_and_cancel() {
        let sphere = uv_sphere(10.0, 32, 16);
        let mut t = SculptTarget::new("ID", &sphere, Mat4::IDENTITY);
        let p = BrushParams {
            kind: BrushKind::Grab,
            radius: 4.0,
            strength: 1.0,
            symmetry: [true, false, false],
            ..Default::default()
        };
        t.begin_grab(Vec3::new(10.0, 0.0, 0.0), Vec3::NEG_Y, &p);
        assert!(t.is_grabbing());
        t.move_grab(Vec3::new(3.0, 0.0, 2.0));
        // Pole near +X should move outward along +X and +Z; mirror near -X should move outward along -X and +Z!
        let max_x = t.mesh.positions.iter().map(|q| q.x).fold(f32::NEG_INFINITY, f32::max);
        let min_x = t.mesh.positions.iter().map(|q| q.x).fold(f32::INFINITY, f32::min);
        assert!(max_x > 12.0, "grab +X should pull outward: {max_x}");
        assert!(min_x < -12.0, "mirrored grab -X should pull outward: {min_x}");
        t.cancel_stroke();
        assert_eq!(t.mesh.positions, sphere.positions);

        // Now test committing grab stroke.
        t.begin_grab(Vec3::new(10.0, 0.0, 0.0), Vec3::NEG_Y, &p);
        t.move_grab(Vec3::new(2.0, 0.0, 0.0));
        let (mesh, cmd) = t.end_stroke().unwrap();
        assert!(mesh.positions != sphere.positions);
        assert!(cmd.contains("\"setMesh\""));
    }

    #[test]
    fn grab_with_radial_symmetry_and_mask_keeps_masked_vertices_stationary() {
        let sphere = uv_sphere(10.0, 32, 16);
        let mut t = SculptTarget::new("ID", &sphere, Mat4::IDENTITY);
        // Mask the +Y hemisphere completely, leave +X/-X/-Y unmasked.
        for (i, pos) in t.mesh.positions.iter().enumerate() {
            if pos.y > 5.0 {
                t.mesh.masks[i] = 1.0;
            }
        }
        let p = BrushParams {
            kind: BrushKind::Grab,
            radius: 4.0,
            strength: 1.0,
            radial_symmetry: 4,
            ..Default::default()
        };
        t.begin_grab(Vec3::new(10.0, 0.0, 0.0), Vec3::NEG_Y, &p);
        t.move_grab(Vec3::new(3.0, 0.0, 0.0));
        // Unmasked +X, -X, -Y lobes should all displace outward > 12.0
        let max_x = t.mesh.positions.iter().map(|q| q.x).fold(f32::NEG_INFINITY, f32::max);
        let min_x = t.mesh.positions.iter().map(|q| q.x).fold(f32::INFINITY, f32::min);
        let min_y = t.mesh.positions.iter().map(|q| q.y).fold(f32::INFINITY, f32::min);
        assert!(max_x > 12.0, "+X unmasked lobe should move: {max_x}");
        assert!(min_x < -12.0, "-X radial lobe should move: {min_x}");
        assert!(min_y < -12.0, "-Y radial lobe should move: {min_y}");
        // Every masked vertex (y > 5.0) MUST stay strictly stationary!
        for (i, (orig, cur)) in sphere.positions.iter().zip(&t.mesh.positions).enumerate() {
            if orig.y > 5.0 {
                assert!(
                    (*cur - *orig).length() < 1e-6,
                    "masked vertex {i} at {orig} moved to {cur} during radial Grab!"
                );
            }
        }
    }
}
