//! uniffi bindings (Swift for Apple, Kotlin for Android): the same `Session` API as the web build.
//! The host owns the layer or window and hands the engine a raw pointer to it:
//! a `CAMetalLayer *` on Apple, an `ANativeWindow *` on Android (see `nativeWindowFromSurface`).

use crate::camera::OrbitCamera;
use crate::format::{Color, Material, Mesh};
use crate::mesh::SculptMesh;
use crate::render::{Gpu, Renderer, Tonemap};
use crate::session::Session;
use glam::{Mat4, Vec3};
use std::sync::{Arc, Mutex};

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum EngineError {
    #[error("{0}")]
    Gpu(String),
    #[error("{0}")]
    Format(String),
    #[error("{0}")]
    Invalid(String),
}

struct Inner {
    session: Session,
    surface: Option<(wgpu::Surface<'static>, wgpu::SurfaceConfiguration)>,
}

#[derive(uniffi::Object)]
pub struct EngineView {
    inner: Mutex<Inner>,
}

fn gpu_err(e: impl std::fmt::Display) -> EngineError {
    EngineError::Gpu(e.to_string())
}

fn open(target: Option<wgpu::SurfaceTargetUnsafe>, width: u32, height: u32) -> Result<Arc<EngineView>, EngineError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    // SAFETY: the host guarantees the layer or window outlives this view (documented in the guides).
    let surface = match target {
        Some(t) => Some(unsafe { instance.create_surface_unsafe(t) }.map_err(gpu_err)?),
        None => None,
    };
    let gpu = pollster::block_on(Gpu::new(instance, surface.as_ref())).map_err(EngineError::Gpu)?;
    let (format, surface) = match surface {
        Some(s) => {
            let c = s.get_default_config(&gpu.adapter, width.max(1), height.max(1)).ok_or_else(|| gpu_err("surface not supported by the adapter"))?;
            let c = wgpu::SurfaceConfiguration { format: c.format.add_srgb_suffix(), ..c };
            s.configure(&gpu.device, &c);
            (c.format, Some((s, c)))
        }
        None => (wgpu::TextureFormat::Rgba8UnormSrgb, None),
    };
    let session = Session::new(Renderer::new(gpu, format), (width.max(1), height.max(1)));
    Ok(Arc::new(EngineView { inner: Mutex::new(Inner { session, surface }) }))
}

#[uniffi::export]
impl EngineView {
    /// Apple: `layer` is a `CAMetalLayer *` (for example `Unmanaged.passUnretained(layer).toOpaque()`).
    #[uniffi::constructor]
    pub fn with_metal_layer(layer: u64, width: u32, height: u32) -> Result<Arc<EngineView>, EngineError> {
        if layer == 0 {
            return Err(EngineError::Invalid("null layer".into()));
        }
        #[cfg(any(target_vendor = "apple"))]
        {
            open(Some(wgpu::SurfaceTargetUnsafe::CoreAnimationLayer(layer as *mut std::ffi::c_void)), width, height)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (width, height);
            Err(EngineError::Invalid("CAMetalLayer is Apple only".into()))
        }
    }

    /// Android: `window` is an `ANativeWindow *` from `nativeWindowFromSurface(surface)`.
    #[uniffi::constructor]
    pub fn with_android_window(window: u64, width: u32, height: u32) -> Result<Arc<EngineView>, EngineError> {
        let ptr = std::ptr::NonNull::new(window as *mut std::ffi::c_void).ok_or_else(|| EngineError::Invalid("null window".into()))?;
        let target = wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(raw_window_handle::RawDisplayHandle::Android(raw_window_handle::AndroidDisplayHandle::new())),
            raw_window_handle: raw_window_handle::RawWindowHandle::AndroidNdk(raw_window_handle::AndroidNdkWindowHandle::new(ptr)),
        };
        open(Some(target), width, height)
    }

    /// No surface: for snapshots (`snapshot_png`) and tests.
    #[uniffi::constructor]
    pub fn offscreen(width: u32, height: u32) -> Result<Arc<EngineView>, EngineError> {
        open(None, width, height)
    }

    pub fn backend(&self) -> String {
        let i = self.inner.lock().unwrap();
        let info = i.session.renderer.gpu.adapter.get_info();
        format!("{:?} ({})", info.backend, info.name)
    }

    pub fn resize(&self, width: u32, height: u32) {
        let mut i = self.inner.lock().unwrap();
        let Inner { session, surface } = &mut *i;
        session.resize(width, height);
        if let Some((s, c)) = surface {
            c.width = width.max(1);
            c.height = height.max(1);
            s.configure(&session.renderer.gpu.device, c);
        }
    }

    /// Draws one frame into the layer or window. Call from the host's display link.
    pub fn render(&self) {
        let mut i = self.inner.lock().unwrap();
        let Inner { session, surface } = &mut *i;
        let Some((s, c)) = surface else { return };
        let frame = match s.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                s.configure(&session.renderer.gpu.device, c);
                return;
            }
            _ => return,
        };
        session.render(&frame.texture.create_view(&Default::default()));
        session.renderer.gpu.queue.present(frame);
    }

    pub fn open_project(&self, bytes: Vec<u8>) -> Result<(), EngineError> {
        self.inner.lock().unwrap().session.open_project(&bytes).map_err(|e| EngineError::Format(e.to_string()))
    }
    pub fn project_json(&self) -> Option<String> {
        self.inner.lock().unwrap().session.project_json()
    }
    pub fn set_camera_json(&self, json: String) -> Result<(), EngineError> {
        let c: OrbitCamera = serde_json::from_str(&json).map_err(|e| EngineError::Invalid(e.to_string()))?;
        self.inner.lock().unwrap().session.camera = c;
        Ok(())
    }
    pub fn camera_json(&self) -> String {
        serde_json::to_string(&self.inner.lock().unwrap().session.camera).unwrap()
    }
    pub fn orbit(&self, dx: f32, dy: f32) {
        let s = &mut self.inner.lock().unwrap().session;
        s.camera = s.camera.orbit(-dx * 0.4, dy * 0.4);
    }
    pub fn pan(&self, dx: f32, dy: f32) {
        let s = &mut self.inner.lock().unwrap().session;
        s.camera = s.camera.pan(dx, dy, s.size.1 as f32);
    }
    pub fn zoom(&self, factor: f32) {
        let s = &mut self.inner.lock().unwrap().session;
        s.camera = s.camera.zoom(factor);
    }
    /// kind: "draw", "clay", "inflate", "crease", "pinch", "grab", "smooth", "flatten", "mask";
    /// radius in world mm; strength -1 to 1.
    pub fn set_brush(&self, kind: String, radius: f32, strength: f32, symmetry_x: bool) {
        self.inner
            .lock()
            .unwrap()
            .session
            .set_brush_ex(&kind, radius, strength, "smooth", symmetry_x, false, false, 1);
    }

    /// Extended Nomad-grade brush settings with falloff, 3-axis symmetry, and radial symmetry (1..16).
    pub fn set_brush_ex(
        &self,
        kind: String,
        radius: f32,
        strength: f32,
        falloff: String,
        sym_x: bool,
        sym_y: bool,
        sym_z: bool,
        radial_count: u32,
    ) {
        self.inner
            .lock()
            .unwrap()
            .session
            .set_brush_ex(&kind, radius, strength, &falloff, sym_x, sym_y, sym_z, radial_count);
    }

    /// Mask operation on object `id`: "clear", "invert", or "blur".
    pub fn mask_op(&self, id: String, op: String) -> bool {
        self.inner.lock().unwrap().session.mask_op(&id, &op)
    }

    /// Free Voxel / SDF Remesh: returns `SceneCommand.setMesh` JSON for object `id` at `resolution` (24..128).
    pub fn voxel_remesh(&self, id: String, resolution: u32) -> Result<String, EngineError> {
        self.inner
            .lock()
            .unwrap()
            .session
            .voxel_remesh(&id, resolution)
            .map(|(_, cmd)| cmd)
            .map_err(EngineError::Invalid)
    }

    /// Interactive Draw in 3D: picks a world point from screen pixel `(x, y)` (empty if none).
    pub fn draw3d_pick_point(&self, x: f32, y: f32, snap_surface: bool, surface_offset: f32, plane_depth: f32) -> Vec<f32> {
        self.inner
            .lock()
            .unwrap()
            .session
            .draw3d_pick_point(x, y, snap_surface, surface_offset, plane_depth)
            .map(|p| vec![p.x, p.y, p.z])
            .unwrap_or_default()
    }

    pub fn draw3d_begin(&self, x: f32, y: f32, radius: f32, snap_surface: bool, surface_offset: f32, plane_depth: f32) -> Vec<f32> {
        self.inner
            .lock()
            .unwrap()
            .session
            .draw3d_begin(x, y, radius, snap_surface, surface_offset, plane_depth)
            .map(|p| vec![p.x, p.y, p.z])
            .unwrap_or_default()
    }

    pub fn draw3d_move(&self, x: f32, y: f32, radius: f32) -> Vec<f32> {
        self.inner
            .lock()
            .unwrap()
            .session
            .draw3d_move(x, y, radius)
            .map(|p| vec![p.x, p.y, p.z])
            .unwrap_or_default()
    }

    pub fn draw3d_end(&self, options_json: String) -> Result<String, EngineError> {
        self.inner
            .lock()
            .unwrap()
            .session
            .draw3d_end(&options_json)
            .map_err(EngineError::Invalid)
    }

    pub fn draw3d_cancel(&self) {
        self.inner.lock().unwrap().session.draw3d_cancel();
    }

    /// Sets viewport background sRGB `[0..1]`.
    pub fn set_background(&self, r: f32, g: f32, b: f32) {
        self.inner.lock().unwrap().session.renderer.settings.background = [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)];
    }
    /// Screen points are in the layer's pixels (points times scale), top left origin.
    pub fn pick(&self, x: f32, y: f32) -> Option<String> {
        self.inner.lock().unwrap().session.pick(x, y).map(|h| h.0)
    }
    pub fn sculpt_begin(&self, x: f32, y: f32, pressure: f32) -> bool {
        self.inner.lock().unwrap().session.sculpt_begin(x, y, pressure)
    }
    pub fn sculpt_move(&self, x: f32, y: f32, pressure: f32) {
        self.inner.lock().unwrap().session.sculpt_move(x, y, pressure)
    }
    /// The `SceneCommand.setMesh` JSON to pass to the Swift core's `perform` / `apply` ("Sculpt").
    pub fn sculpt_end(&self) -> Option<String> {
        self.inner.lock().unwrap().session.sculpt_end().map(|r| r.command_json)
    }
    pub fn sculpt_cancel(&self) {
        self.inner.lock().unwrap().session.sculpt_cancel()
    }
    pub fn undo(&self) -> bool {
        self.inner.lock().unwrap().session.undo()
    }
    pub fn redo(&self) -> bool {
        self.inner.lock().unwrap().session.redo()
    }
    /// "agx", "aces" or "linear".
    pub fn set_look(&self, tonemap: String, ssao: bool, shadows: bool) {
        let s = &mut self.inner.lock().unwrap().session.renderer.settings;
        s.tonemap = match tonemap.as_str() {
            "aces" => Tonemap::Aces,
            "linear" => Tonemap::Linear,
            _ => Tonemap::AgX,
        };
        s.ssao = ssao;
        s.shadows = shadows;
    }

    /// Adds or replaces an object from the host's scene: base mesh JSON (Sculptor's Mesh shape), column
    /// major world matrix (16 floats, millimetres, Z up), sRGB colour (4 floats), metallic, roughness.
    pub fn upsert_object(&self, id: String, mesh_json: String, world: Vec<f32>, color: Vec<f32>, metallic: f32, roughness: f32) -> Result<(), EngineError> {
        let o = crate::session::object_from_parts(&id, &mesh_json, &world, &color, metallic, roughness).map_err(EngineError::Invalid)?;
        self.inner.lock().unwrap().session.upsert_object(o);
        Ok(())
    }
    /// Zero-JSON binary mesh upload (E4.5): flat xyz `positions`, per-face vertex counts `face_sizes`, and `face_indices`.
    pub fn upsert_object_buffers(
        &self,
        id: String,
        positions: Vec<f32>,
        face_sizes: Vec<u32>,
        face_indices: Vec<u32>,
        smooth_shading: bool,
        world: Vec<f32>,
        color: Vec<f32>,
        metallic: f32,
        roughness: f32,
    ) -> Result<(), EngineError> {
        let o = crate::session::object_from_buffers(&id, &positions, &face_sizes, &face_indices, smooth_shading, &world, &color, metallic, roughness)
            .map_err(EngineError::Invalid)?;
        self.inner.lock().unwrap().session.upsert_object(o);
        Ok(())
    }
    /// Zero-JSON binary mesh replacement for an existing object (E4.5).
    pub fn set_mesh_buffers(&self, id: String, positions: Vec<f32>, face_sizes: Vec<u32>, face_indices: Vec<u32>, smooth_shading: bool) -> Result<(), EngineError> {
        let m = crate::session::mesh_from_buffers(&positions, &face_sizes, &face_indices, smooth_shading).map_err(EngineError::Invalid)?;
        self.inner.lock().unwrap().session.set_mesh(&id, m);
        Ok(())
    }
    pub fn remove_object(&self, id: String) {
        self.inner.lock().unwrap().session.remove_object(&id)
    }
    pub fn set_transform(&self, id: String, world: Vec<f32>) -> Result<(), EngineError> {
        if world.len() != 16 || !world.iter().all(|x| x.is_finite()) {
            return Err(EngineError::Invalid("world must be 16 finite floats".into()));
        }
        self.inner.lock().unwrap().session.set_transform(&id, Mat4::from_cols_slice(&world));
        Ok(())
    }
    pub fn set_material(&self, id: String, color: Vec<f32>, metallic: f32, roughness: f32) -> Result<(), EngineError> {
        if color.len() != 4 || !color.iter().all(|x| x.is_finite()) || !metallic.is_finite() || !roughness.is_finite() {
            return Err(EngineError::Invalid("color must be 4 finite floats".into()));
        }
        let c = |i: usize| color[i].clamp(0.0, 1.0);
        let m = Material {
            name: String::new(),
            base_color: Color { r: c(0), g: c(1), b: c(2), a: c(3) },
            metallic: metallic.clamp(0.0, 1.0),
            roughness: roughness.clamp(0.0, 1.0),
        };
        self.inner.lock().unwrap().session.set_material(&id, m);
        Ok(())
    }
    pub fn set_visible(&self, id: String, visible: bool) {
        self.inner.lock().unwrap().session.set_visible(&id, visible);
    }
    pub fn set_highlight(&self, id: String, on: bool) {
        self.inner.lock().unwrap().session.renderer.set_highlight(&id, on)
    }
    pub fn frame_all(&self) {
        self.inner.lock().unwrap().session.frame_all()
    }
    /// The object and world point under a screen point (pixels).
    pub fn hit(&self, x: f32, y: f32) -> Option<Hit> {
        self.inner.lock().unwrap().session.pick(x, y).map(|(id, p)| Hit { id, x: p.x, y: p.y, z: p.z })
    }
    /// Screen x, y (pixels) and clip depth of a world point; empty when behind the camera.
    pub fn project(&self, x: f32, y: f32, z: f32) -> Vec<f32> {
        self.inner.lock().unwrap().session.project(Vec3::new(x, y, z)).map(|(a, b, c)| vec![a, b, c]).unwrap_or_default()
    }

    /// Renders the current scene offscreen to PNG bytes (any view, surface or not).
    pub fn snapshot_png(&self, width: u32, height: u32) -> Vec<u8> {
        let mut i = self.inner.lock().unwrap();
        let s = &mut i.session;
        let cam = s.camera;
        snapshot(&mut s.renderer, width, height, &cam)
    }
}

fn snapshot(r: &mut Renderer, width: u32, height: u32, cam: &OrbitCamera) -> Vec<u8> {
    let size = (width.max(1), height.max(1));
    let px = r.render_rgba(size, cam);
    png(&px, size)
}

fn png(px: &[u8], size: (u32, u32)) -> Vec<u8> {
    let mut out = Vec::new();
    let mut e = ::png::Encoder::new(&mut out, size.0, size.1);
    e.set_color(::png::ColorType::Rgba);
    e.set_depth(::png::BitDepth::Eight);
    e.write_header().and_then(|mut w| w.write_image_data(px)).expect("png");
    out
}

#[derive(uniffi::Record)]
pub struct Hit {
    pub id: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// One object for Dream Builder's `SnapshotRenderer`: triangle buffers in engine space (mm, Z up),
/// a column major 4x4 world matrix, and a Sculptor material.
#[derive(uniffi::Record)]
pub struct SnapshotObject {
    pub positions: Vec<f32>,
    pub indices: Vec<u32>,
    pub world: Vec<f32>,
    /// sRGB r, g, b, a.
    pub base_color: Vec<f32>,
    pub metallic: f32,
    pub roughness: f32,
}

static SNAPSHOT_RENDERER: Mutex<Option<Renderer>> = Mutex::new(None);

/// A realistic still of loose triangle buffers, no project needed (Dream Builder, thumbnails).
#[uniffi::export]
pub fn render_snapshot(objects: Vec<SnapshotObject>, camera_json: String, width: u32, height: u32) -> Result<Vec<u8>, EngineError> {
    let cam: OrbitCamera = serde_json::from_str(&camera_json).map_err(|e| EngineError::Invalid(e.to_string()))?;
    let mut guard = SNAPSHOT_RENDERER.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let gpu = pollster::block_on(Gpu::headless()).map_err(EngineError::Gpu)?;
        let mut r = Renderer::new(gpu, wgpu::TextureFormat::Rgba8UnormSrgb);
        r.settings.grid = false;
        *guard = Some(r);
    }
    let r = guard.as_mut().unwrap();
    r.clear();
    for (k, o) in objects.iter().enumerate() {
        let n = (o.positions.len() / 3) as u32;
        if o.positions.len() % 3 != 0 || o.indices.len() % 3 != 0 || o.indices.iter().any(|&i| i >= n) || o.world.len() != 16 || o.base_color.len() != 4 {
            r.clear();
            return Err(EngineError::Invalid(format!("object {k}: bad buffers")));
        }
        let mesh = Mesh {
            positions: o.positions.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect(),
            faces: o.indices.chunks_exact(3).map(|t| t.to_vec()).collect(),
            smooth_shading: true,
        };
        let c = &o.base_color;
        let m = Material { name: String::new(), base_color: Color { r: c[0], g: c[1], b: c[2], a: c[3] }, metallic: o.metallic, roughness: o.roughness };
        r.upsert(&k.to_string(), &SculptMesh::from_mesh(&mesh), Mat4::from_cols_slice(&o.world), &m);
    }
    let bytes = png(&r.render_rgba((width.max(1), height.max(1)), &cam), (width.max(1), height.max(1)));
    r.clear();
    Ok(bytes)
}

/// Text to 3D (src/text3d): plans and meshes `prompt`. JSON in and out, see `text3d::host::generate_json`.
#[uniffi::export]
pub fn text_to_3d(prompt: String, options_json: String) -> String {
    crate::text3d::host::generate_json(&prompt, &options_json)
}

/// Plans `prompt` without meshing (`{ok, understood, notes, program}`).
#[uniffi::export]
pub fn plan_text(prompt: String, seed: u32) -> String {
    crate::text3d::host::plan_json(&prompt, seed)
}

/// Meshes a shape program, or an LLM reply containing one (Apple Foundation Models, a local model).
#[uniffi::export]
pub fn mesh_program(program_or_reply: String, options_json: String) -> String {
    crate::text3d::host::mesh_program_json(&program_or_reply, &options_json)
}

/// The system prompt and JSON Schema for asking an LLM for a shape program.
#[uniffi::export]
pub fn program_prompt() -> String {
    crate::text3d::host::llm_system_prompt()
}

#[uniffi::export]
pub fn program_schema() -> String {
    crate::text3d::host::program_schema()
}

/// On-device free Image-to-3D reconstruction (RGBA8 + optional depth map -> text3d-compatible JSON).
#[uniffi::export]
pub fn image_to_3d(width: u32, height: u32, rgba: Vec<u8>, depth: Vec<f32>, options_json: String) -> String {
    let depth_opt = if depth.is_empty() { None } else { Some(depth.as_slice()) };
    crate::img3d::image_to_3d_json(width, height, &rgba, depth_opt, &options_json)
}

/// Result returned by prompt-based Gaussian Splat editing across Swift/Kotlin FFI.
#[derive(uniffi::Record)]
pub struct SplatEditRecord {
    pub splat_bytes: Vec<u8>,
    pub summary: String,
    pub count: u32,
    pub modified: u32,
}

/// On-device Image-to-Gaussian-Splat (items when `is_space == false`, walkable spaces when `is_space == true`).
/// Returns standard 32-byte-per-Gaussian `.splat` bytes.
#[uniffi::export]
pub fn image_to_splat(
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    depth: Vec<f32>,
    is_space: bool,
    max_splats: u32,
) -> Result<Vec<u8>, EngineError> {
    let depth_opt = if depth.is_empty() { None } else { Some(depth.as_slice()) };
    let cloud = crate::splat::image_to_splat(width, height, &rgba, depth_opt, is_space, max_splats as usize)
        .map_err(EngineError::Invalid)?;
    Ok(cloud.to_splat_bytes())
}

/// Converts any triangle mesh (single object OR captured ARKit/visionOS room mesh) into 3D Gaussian Splats.
#[uniffi::export]
pub fn mesh_to_splat(
    positions: Vec<f32>,
    indices: Vec<u32>,
    vertex_colors: Vec<f32>,
    fallback_rgba: Vec<f32>,
    is_space: bool,
    max_splats: u32,
) -> Result<Vec<u8>, EngineError> {
    let vc = if vertex_colors.is_empty() { None } else { Some(vertex_colors.as_slice()) };
    let fb = if fallback_rgba.len() >= 4 {
        [fallback_rgba[0], fallback_rgba[1], fallback_rgba[2], fallback_rgba[3]]
    } else {
        [0.8, 0.8, 0.8, 1.0]
    };
    let cloud = crate::splat::mesh_to_splat(&positions, &indices, vc, fb, is_space, max_splats as usize)
        .map_err(EngineError::Invalid)?;
    Ok(cloud.to_splat_bytes())
}

/// Edits an item or captured space Gaussian Splat cloud using a natural-language prompt.
#[uniffi::export]
pub fn edit_splat_prompt(splat_bytes: Vec<u8>, prompt: String) -> Result<SplatEditRecord, EngineError> {
    let mut cloud = crate::splat::SplatCloud::from_splat_bytes(&splat_bytes).map_err(EngineError::Format)?;
    let (summary, modified) = crate::splat::edit_splat_in_place(&mut cloud, &prompt);
    let count = cloud.len() as u32;
    Ok(SplatEditRecord {
        splat_bytes: cloud.to_splat_bytes(),
        summary,
        count,
        modified: modified as u32,
    })
}

/// Converts compact `.splat` bytes to standard 3DGS binary `.ply` bytes.
#[uniffi::export]
pub fn splat_to_ply(splat_bytes: Vec<u8>) -> Result<Vec<u8>, EngineError> {
    let cloud = crate::splat::SplatCloud::from_splat_bytes(&splat_bytes).map_err(EngineError::Format)?;
    Ok(cloud.to_ply_bytes())
}

/// Imports standard 3DGS `.ply` (binary little-endian or ASCII) into compact `.splat` bytes.
#[uniffi::export]
pub fn ply_to_splat(ply_bytes: Vec<u8>) -> Result<Vec<u8>, EngineError> {
    let cloud = crate::splat::SplatCloud::from_ply_bytes(&ply_bytes).map_err(EngineError::Format)?;
    Ok(cloud.to_splat_bytes())
}

/// Packs `.splat` (32 B/splat) into ultra-compact `.dsplat` v2 (12 B/splat, 2.67x smaller than .splat).
#[uniffi::export]
pub fn splat_to_compact(splat_bytes: Vec<u8>) -> Result<Vec<u8>, EngineError> {
    let cloud = crate::splat::SplatCloud::from_splat_bytes(&splat_bytes).map_err(EngineError::Format)?;
    Ok(cloud.to_compact_bytes())
}

/// Unpacks `.dsplat` (v2 12 B/splat or v1 16 B/splat) back to standard `.splat` (32 B/splat).
#[uniffi::export]
pub fn compact_to_splat(compact_bytes: Vec<u8>) -> Result<Vec<u8>, EngineError> {
    let cloud = crate::splat::SplatCloud::from_compact_bytes(&compact_bytes).map_err(EngineError::Format)?;
    Ok(cloud.to_splat_bytes())
}

/// Returns JSON statistics (`count`, `boundsMin`, `boundsMax`, `center`, `size`, `avgOpacity`, `splatBytes`).
#[uniffi::export]
pub fn splat_stats(splat_bytes: Vec<u8>) -> Result<String, EngineError> {
    let cloud = crate::splat::SplatCloud::from_splat_bytes(&splat_bytes).map_err(EngineError::Format)?;
    Ok(cloud.stats_json())
}

/// Fast O(N) radix sort of splats back-to-front relative to a camera position and forward vector.
#[uniffi::export]
pub fn sort_splats(splat_bytes: Vec<u8>, cam_pos: Vec<f32>, cam_forward: Vec<f32>) -> Result<Vec<u32>, EngineError> {
    if cam_pos.len() != 3 || cam_forward.len() != 3 {
        return Err(EngineError::Invalid("cam_pos and cam_forward must have 3 floats".into()));
    }
    let cloud = crate::splat::SplatCloud::from_splat_bytes(&splat_bytes).map_err(EngineError::Format)?;
    Ok(crate::splat::sort_splats_by_camera(
        &cloud,
        [cam_pos[0], cam_pos[1], cam_pos[2]],
        [cam_forward[0], cam_forward[1], cam_forward[2]],
    ))
}

/// Draw in 3D free function: builds a watertight tube/ribbon `SceneCommand.addMesh` JSON from
/// flat `[x, y, z, radius, ...]` samples and options JSON.
#[uniffi::export]
pub fn draw3d_from_points(points_xyzr: Vec<f32>, options_json: String) -> Result<String, EngineError> {
    if points_xyzr.len() < 8 || points_xyzr.len() % 4 != 0 {
        return Err(EngineError::Invalid("points_xyzr must contain at least 2 (x, y, z, radius) quadruples".into()));
    }
    let pts: Vec<crate::draw3d::Draw3DPoint> = points_xyzr
        .chunks_exact(4)
        .map(|c| crate::draw3d::Draw3DPoint { pos: Vec3::new(c[0], c[1], c[2]), radius: c[3] })
        .collect();
    let (opts, name) = crate::draw3d::Draw3DOptions::from_json(&options_json);
    crate::draw3d::build_tube_add_mesh_command(&pts, &opts, &name, None)
        .map(|(_, _, cmd)| cmd)
        .map_err(EngineError::Invalid)
}

/// Built-in 38-shape catalog metadata as JSON array (`[{id, name, category, categoryLabel, defaultSize, tags}]`).
#[uniffi::export]
pub fn shape_catalog_json() -> String {
    crate::shapes::shape_catalog_json()
}

/// Generates a watertight built-in shape `SceneCommand.addMesh` JSON by catalog `id` and optional `params_json`.
#[uniffi::export]
pub fn generate_shape(id: String, params_json: String) -> Result<String, EngineError> {
    crate::shapes::generate_shape_command_json(&id, &params_json).map_err(EngineError::Invalid)
}

/// Extrudes a 2D closed outline (`[x0, y0, ...]` on the Z=0 workplane) into a watertight `SceneCommand.addMesh` JSON.
#[uniffi::export]
pub fn extrude_profile_2d(points_xy: Vec<f32>, options_json: String) -> Result<String, EngineError> {
    crate::shapes::extrude_profile_2d_command_json(&points_xy, &options_json)
        .map_err(EngineError::Invalid)
}

// ── Pure-Rust Document, Edit, IO & Voice Bridge (replaces legacy Swift libSculptorAndroid.so) ──

#[derive(uniffi::Record)]
pub struct BridgeRenderMesh {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub indices: Vec<u32>,
    pub hull_positions: Vec<f32>,
    pub hull_normals: Vec<f32>,
    pub hull_indices: Vec<u32>,
}

#[derive(uniffi::Record)]
pub struct BridgeOverlayData {
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

#[derive(uniffi::Record)]
pub struct BridgeExportFile {
    pub file_name: String,
    pub data: Vec<u8>,
}

#[uniffi::export]
pub fn doc_bridge_version() -> i32 {
    crate::bridge::BRIDGE_VERSION
}

#[uniffi::export]
pub fn doc_core_version() -> String {
    crate::bridge::CORE_VERSION.to_string()
}

#[uniffi::export]
pub fn doc_new() -> i64 {
    crate::bridge::BridgeRegistry::global().new_doc()
}

#[uniffi::export]
pub fn doc_free(handle: i64) {
    crate::bridge::BridgeRegistry::global().free_doc(handle);
}

#[uniffi::export]
pub fn doc_reset(handle: i64) {
    crate::bridge::with_doc(handle, (), |d| d.reset());
}

#[uniffi::export]
pub fn doc_apply(handle: i64, command_json: String) -> String {
    crate::bridge::with_doc(handle, crate::bridge::BridgeError::new("engine.badHandle").to_envelope(), |d| {
        crate::bridge::commands::doc_apply(d, &command_json)
    })
}

#[uniffi::export]
pub fn doc_apply_group(handle: i64, json: String) -> String {
    crate::bridge::with_doc(handle, crate::bridge::BridgeError::new("engine.badHandle").to_envelope(), |d| {
        crate::bridge::commands::doc_apply_group(d, &json)
    })
}

#[uniffi::export]
pub fn doc_undo(handle: i64) -> bool {
    crate::bridge::with_doc(handle, false, |d| d.undo())
}

#[uniffi::export]
pub fn doc_redo(handle: i64) -> bool {
    crate::bridge::with_doc(handle, false, |d| d.redo())
}

#[uniffi::export]
pub fn doc_preview(handle: i64, command_json: String) -> String {
    crate::bridge::with_doc(handle, crate::bridge::BridgeError::new("engine.badHandle").to_envelope(), |d| {
        crate::bridge::commands::doc_preview(d, &command_json)
    })
}

#[uniffi::export]
pub fn doc_clear_preview(handle: i64) {
    crate::bridge::with_doc(handle, (), |d| {
        d.preview_transforms.clear();
        d.preview_meshes.clear();
    });
}

#[uniffi::export]
pub fn doc_set_selection(handle: i64, ids_json: String, active: Option<String>) {
    let ids: Vec<String> = serde_json::from_str(&ids_json).unwrap_or_default();
    crate::bridge::with_doc(handle, (), |d| {
        let valid: std::collections::HashSet<&str> = d.scene.objects.iter().map(|o| o.id.as_str()).collect();
        let mut objs: Vec<String> = ids.into_iter().filter(|id| valid.contains(id.as_str())).collect();
        objs.sort();
        objs.dedup();
        let act = active.filter(|a| objs.contains(a)).or_else(|| objs.last().cloned());
        d.selection = crate::format::Selection { objects: objs, active: act, elements: Default::default() };
    });
}

#[uniffi::export]
pub fn doc_select_object(handle: i64, id: String, extend: bool) {
    crate::bridge::with_doc(handle, (), |d| {
        let Some(obj) = d.scene.objects.iter().find(|o| o.id == id) else { return };
        let locked = obj.is_locked;
        if matches!(d.edit_mode, crate::bridge::EditMode::Edit(_)) && d.selection.active.as_deref() != Some(id.as_str()) {
            d.selection.objects = vec![id.clone()];
            d.selection.active = Some(id);
            d.selection.elements = Default::default();
            if locked { d.edit_mode = crate::bridge::EditMode::Object; }
            d.last_picked_vertex = None;
            return;
        }
        if extend {
            if let Some(pos) = d.selection.objects.iter().position(|x| x == &id) {
                d.selection.objects.remove(pos);
                if d.selection.active.as_deref() == Some(id.as_str()) {
                    d.selection.active = d.selection.objects.last().cloned();
                }
            } else {
                d.selection.objects.push(id.clone());
                d.selection.active = Some(id);
            }
        } else {
            d.selection.objects = vec![id.clone()];
            d.selection.active = Some(id);
        }
    });
}

#[uniffi::export]
pub fn doc_select_all(handle: i64) {
    crate::bridge::with_doc(handle, (), |d| {
        if let (crate::bridge::EditMode::Edit(mode), Some(aid)) = (d.edit_mode, d.selection.active.clone()) {
            if let Some(obj) = d.scene.objects.iter().find(|o| o.id == aid) {
                let all_v: std::collections::HashSet<usize> = (0..obj.mesh.positions.len()).collect();
                d.selection.elements = crate::bridge::edit::elements_from_vertices(&all_v, mode, &obj.mesh);
                return;
            }
        }
        let ids: Vec<String> = d.scene.objects.iter().filter(|o| o.is_visible && !o.is_locked).map(|o| o.id.clone()).collect();
        d.selection.active = ids.last().cloned();
        d.selection.objects = ids;
    });
}

#[uniffi::export]
pub fn doc_deselect_all(handle: i64) {
    crate::bridge::with_doc(handle, (), |d| {
        d.selection.objects.clear();
        d.selection.active = None;
        d.selection.elements = Default::default();
        d.last_picked_vertex = None;
    });
}

#[uniffi::export]
pub fn doc_set_undo_budget(handle: i64, bytes: i64) {
    if bytes > 0 {
        crate::bridge::with_doc(handle, (), |d| d.undo_memory_budget = bytes as usize);
    }
}

#[uniffi::export]
pub fn doc_state_json(handle: i64) -> String {
    crate::bridge::with_doc(handle, "{}".to_string(), |d| d.state_json())
}

#[uniffi::export]
pub fn doc_render_list_json(handle: i64) -> String {
    crate::bridge::with_doc(handle, r#"{"objects":[],"bounds":null}"#.to_string(), |d| d.render_list_json())
}

#[uniffi::export]
pub fn doc_summary_json(handle: i64) -> String {
    crate::bridge::with_doc(handle, "{}".to_string(), |d| d.summary_json())
}

#[uniffi::export]
pub fn doc_bounds_json(handle: i64, ids_json: String, elements: bool) -> String {
    let ids: Vec<String> = serde_json::from_str(&ids_json).unwrap_or_default();
    crate::bridge::with_doc(handle, "null".to_string(), |d| d.bounds_json(&ids, elements))
}

#[uniffi::export]
pub fn doc_render_mesh(handle: i64, id: String) -> Option<BridgeRenderMesh> {
    crate::bridge::with_doc(handle, None, |d| d.render_mesh(&id)).map(|r| BridgeRenderMesh {
        positions: r.positions,
        normals: r.normals,
        indices: r.indices,
        hull_positions: r.hull_positions,
        hull_normals: r.hull_normals,
        hull_indices: r.hull_indices,
    })
}

#[uniffi::export]
pub fn doc_pick(handle: i64, ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) -> Option<String> {
    crate::bridge::with_doc(handle, None, |d| d.pick(Vec3::new(ox, oy, oz), Vec3::new(dx, dy, dz)))
}

#[uniffi::export]
pub fn doc_set_edit_mode(handle: i64, mode: String) -> String {
    crate::bridge::with_doc(handle, crate::bridge::BridgeError::new("engine.badHandle").to_envelope(), |d| {
        crate::bridge::edit::set_edit_mode(d, &mode)
    })
}

#[uniffi::export]
pub fn doc_run_tool(handle: i64, tool: String, options_json: String) -> String {
    crate::bridge::with_doc(handle, crate::bridge::BridgeError::new("engine.badHandle").to_envelope(), |d| {
        crate::bridge::edit::run_tool(d, &tool, &options_json)
    })
}

#[uniffi::export]
pub fn doc_select_no_elements(handle: i64) {
    crate::bridge::with_doc(handle, (), |d| {
        d.selection.elements = Default::default();
        d.last_picked_vertex = None;
    });
}

#[uniffi::export]
pub fn doc_select_linked(handle: i64) -> String {
    crate::bridge::with_doc(handle, crate::bridge::ok_empty(), |d| crate::bridge::edit::select_linked(d))
}

#[uniffi::export]
pub fn doc_tap_element(handle: i64, camera_json: String, x: f32, y: f32, density: f32, extend: bool) -> String {
    crate::bridge::with_doc(handle, r#"{"ok":true,"hit":null}"#.to_string(), |d| {
        crate::bridge::edit::tap_element(d, &camera_json, x, y, density, extend)
    })
}

#[uniffi::export]
pub fn doc_box_select(handle: i64, camera_json: String, x0: f32, y0: f32, x1: f32, y1: f32, extend: bool) -> String {
    crate::bridge::with_doc(handle, r#"{"ok":true,"count":0}"#.to_string(), |d| {
        crate::bridge::edit::box_select(d, &camera_json, x0, y0, x1, y1, extend)
    })
}

#[uniffi::export]
pub fn doc_gizmo_command(handle: i64, json: String) -> String {
    crate::bridge::with_doc(handle, crate::bridge::BridgeError::new("engine.badHandle").to_envelope(), |d| {
        crate::bridge::edit::gizmo_command(d, &json)
    })
}

#[uniffi::export]
pub fn doc_can_drag(handle: i64) -> String {
    crate::bridge::with_doc(handle, crate::bridge::BridgeError::new("engine.badHandle").to_envelope(), |d| {
        crate::bridge::edit::can_drag(d)
    })
}

#[uniffi::export]
pub fn doc_edit_overlay(handle: i64, dot_radius: f32, dark: bool) -> Option<BridgeOverlayData> {
    crate::bridge::with_doc(handle, None, |d| crate::bridge::edit::build_edit_overlay(d, dot_radius, dark)).map(|o| BridgeOverlayData {
        revision: o.revision,
        matrix: o.matrix,
        line_positions: o.line_positions,
        line_colors: o.line_colors,
        line_indices: o.line_indices,
        dot_positions: o.dot_positions,
        dot_colors: o.dot_colors,
        dot_indices: o.dot_indices,
        face_positions: o.face_positions,
        face_colors: o.face_colors,
        face_indices: o.face_indices,
    })
}

#[uniffi::export]
pub fn doc_project_json(handle: i64) -> String {
    crate::bridge::with_doc(handle, crate::bridge::BridgeError::new("engine.badHandle").to_envelope(), |d| {
        let mut m = serde_json::Map::new();
        m.insert("project".into(), serde_json::Value::String(d.to_project_file(None).encode()));
        crate::bridge::ok_envelope(m)
    })
}

#[uniffi::export]
pub fn doc_open_project(handle: i64, json: String) -> String {
    crate::bridge::with_doc(handle, crate::bridge::BridgeError::new("engine.badHandle").to_envelope(), |d| {
        d.open_project_bytes(json.as_bytes())
    })
}

#[uniffi::export]
pub fn doc_project_data(handle: i64, camera_json: Option<String>) -> Result<Vec<u8>, EngineError> {
    let reg = crate::bridge::BridgeRegistry::global();
    let arc = reg.get_doc(handle).ok_or_else(|| EngineError::Invalid(crate::bridge::BridgeError::new("engine.badHandle").to_error_json()))?;
    let guard = arc.lock().unwrap_or_else(|e| e.into_inner());
    Ok(guard.to_project_file(camera_json.as_deref()).encode().into_bytes())
}

#[uniffi::export]
pub fn doc_open_project_data(handle: i64, bytes: Vec<u8>) -> String {
    crate::bridge::with_doc(handle, crate::bridge::BridgeError::new("engine.badHandle").to_envelope(), |d| {
        d.open_project_bytes(&bytes)
    })
}

#[uniffi::export]
pub fn doc_reencode_project(bytes: Vec<u8>) -> Result<Vec<u8>, EngineError> {
    crate::format::ProjectFile::decode(&bytes)
        .map(|pf| pf.encode().into_bytes())
        .map_err(|e| EngineError::Format(crate::bridge::BridgeError::from(e).to_error_json()))
}

#[uniffi::export]
pub fn doc_snapshot(handle: i64) -> i64 {
    let reg = crate::bridge::BridgeRegistry::global();
    let Some(arc) = reg.get_doc(handle) else { return 0 };
    let snap = {
        let d = arc.lock().unwrap_or_else(|e| e.into_inner());
        crate::bridge::SceneSnapshot { scene: d.scene.clone(), selection: d.selection.clone(), revision: d.revision }
    };
    reg.put_snapshot(snap)
}

#[uniffi::export]
pub fn doc_free_snapshot(snapshot: i64) {
    crate::bridge::BridgeRegistry::global().free_snapshot(snapshot);
}

#[uniffi::export]
pub fn doc_snapshot_revision(snapshot: i64) -> i32 {
    crate::bridge::BridgeRegistry::global().get_snapshot(snapshot).map(|s| s.revision).unwrap_or(0)
}

#[uniffi::export]
pub fn doc_new_command_control() -> i64 {
    crate::bridge::BridgeRegistry::global().put_held(crate::bridge::HeldObject::Control(Arc::new(crate::bridge::CommandControl::new())))
}

#[uniffi::export]
pub fn doc_command_progress(control: i64) -> f64 {
    let guard = crate::bridge::BridgeRegistry::global().held.lock().unwrap();
    match guard.get(&control) {
        Some(crate::bridge::HeldObject::Control(c)) | Some(crate::bridge::HeldObject::ImportJob(c)) => c.progress(),
        _ => 0.0,
    }
}

#[uniffi::export]
pub fn doc_cancel_command(control: i64) {
    let guard = crate::bridge::BridgeRegistry::global().held.lock().unwrap();
    if let Some(crate::bridge::HeldObject::Control(c) | crate::bridge::HeldObject::ImportJob(c)) = guard.get(&control) {
        c.cancel();
    }
}

#[uniffi::export]
pub fn doc_apply_off_main(snapshot: i64, command_json: String, control: i64) -> Result<i64, EngineError> {
    let reg = crate::bridge::BridgeRegistry::global();
    let snap = reg.get_snapshot(snapshot).ok_or_else(|| EngineError::Invalid(crate::bridge::BridgeError::new("engine.badHandle").to_error_json()))?;
    let ctl = {
        let guard = reg.held.lock().unwrap();
        match guard.get(&control) {
            Some(crate::bridge::HeldObject::Control(c)) => Some(c.clone()),
            _ => None,
        }
    };
    let applied = crate::bridge::commands::apply_off_main(&snap, &command_json, ctl.as_deref())
        .map_err(|e| EngineError::Invalid(e.to_error_json()))?;
    Ok(reg.put_held(crate::bridge::HeldObject::Applied(applied)))
}

#[uniffi::export]
pub fn doc_commit_if_unchanged(handle: i64, result: i64, start_revision: i32) -> String {
    let reg = crate::bridge::BridgeRegistry::global();
    let applied = {
        let guard = reg.held.lock().unwrap();
        match guard.get(&result) {
            Some(crate::bridge::HeldObject::Applied(a)) => a.clone(),
            _ => return crate::bridge::BridgeError::new("engine.badHandle").to_envelope(),
        }
    };
    crate::bridge::with_doc(handle, crate::bridge::BridgeError::new("engine.badHandle").to_envelope(), |d| {
        if d.revision != start_revision || applied.start_revision != start_revision {
            let mut m = serde_json::Map::new();
            m.insert("committed".into(), serde_json::Value::Bool(false));
            return crate::bridge::ok_envelope(m);
        }
        d.record_scene_change(applied.label, applied.scene, &applied.created, applied.new_elements);
        let mut m = serde_json::Map::new();
        m.insert("committed".into(), serde_json::Value::Bool(true));
        m.insert("created".into(), serde_json::json!(applied.created));
        crate::bridge::ok_envelope(m)
    })
}

#[uniffi::export]
pub fn doc_free_held(id: i64) {
    crate::bridge::BridgeRegistry::global().free_held(id);
}

#[uniffi::export]
pub fn doc_export_stl(snapshot: i64, objects_json: Option<String>) -> Result<Vec<u8>, EngineError> {
    let snap = crate::bridge::BridgeRegistry::global()
        .get_snapshot(snapshot)
        .ok_or_else(|| EngineError::Invalid(crate::bridge::BridgeError::new("engine.badHandle").to_error_json()))?;
    crate::bridge::io::export_stl(&snap, objects_json.as_deref()).map_err(|e| EngineError::Invalid(e.to_error_json()))
}

#[uniffi::export]
pub fn doc_export_files(snapshot: i64, options_json: String) -> Result<Vec<BridgeExportFile>, EngineError> {
    let snap = crate::bridge::BridgeRegistry::global()
        .get_snapshot(snapshot)
        .ok_or_else(|| EngineError::Invalid(crate::bridge::BridgeError::new("engine.badHandle").to_error_json()))?;
    crate::bridge::io::export_files(&snap, &options_json)
        .map(|v| v.into_iter().map(|f| BridgeExportFile { file_name: f.file_name, data: f.data }).collect())
        .map_err(|e| EngineError::Invalid(e.to_error_json()))
}

#[uniffi::export]
pub fn doc_print_check(snapshot: i64, options_json: String) -> String {
    let Some(snap) = crate::bridge::BridgeRegistry::global().get_snapshot(snapshot) else { return "null".to_string() };
    crate::bridge::io::print_check(&snap, &options_json)
}

#[uniffi::export]
pub fn doc_fix_plan(snapshot: i64, options_json: String) -> String {
    let Some(snap) = crate::bridge::BridgeRegistry::global().get_snapshot(snapshot) else {
        return r#"{"commands":[],"repaired":0,"moved":false,"revision":0}"#.to_string();
    };
    crate::bridge::io::fix_plan(&snap, &options_json)
}

#[uniffi::export]
pub fn doc_new_import_job() -> i64 {
    crate::bridge::BridgeRegistry::global().put_held(crate::bridge::HeldObject::ImportJob(Arc::new(crate::bridge::CommandControl::new())))
}

#[uniffi::export]
pub fn doc_import_load(
    job: i64,
    file_name: String,
    bytes: Vec<u8>,
    physical_memory: i64,
    available_memory: i64,
    default_name: String,
) -> String {
    let reg = crate::bridge::BridgeRegistry::global();
    let ctl = {
        let guard = reg.held.lock().unwrap();
        match guard.get(&job) {
            Some(crate::bridge::HeldObject::ImportJob(c)) => Some(c.clone()),
            _ => None,
        }
    };
    match crate::bridge::io::import_load(ctl.as_deref(), &file_name, &bytes, physical_memory, available_memory, &default_name) {
        Ok((loaded, info)) => {
            let lid = reg.put_held(crate::bridge::HeldObject::ImportLoaded(loaded));
            let mut m = serde_json::Map::new();
            m.insert("loaded".into(), serde_json::json!(lid));
            m.insert("info".into(), info);
            crate::bridge::ok_envelope(m)
        }
        Err(e) => e.to_envelope(),
    }
}

#[uniffi::export]
pub fn doc_import_prepare(job: i64, loaded: i64, target: i32, default_name: String) -> String {
    let reg = crate::bridge::BridgeRegistry::global();
    let (ctl, res) = {
        let guard = reg.held.lock().unwrap();
        let ctl = match guard.get(&job) {
            Some(crate::bridge::HeldObject::ImportJob(c)) => Some(c.clone()),
            _ => None,
        };
        let res = match guard.get(&loaded) {
            Some(crate::bridge::HeldObject::ImportLoaded(l)) => {
                Some(crate::bridge::io::import_prepare(ctl.as_deref(), l, target, &default_name))
            }
            _ => None,
        };
        (ctl, res)
    };
    let _ = ctl;
    match res {
        Some(Ok(prep)) => {
            let tris = prep.triangles;
            let pid = reg.put_held(crate::bridge::HeldObject::ImportPrepared(prep));
            let mut m = serde_json::Map::new();
            m.insert("prepared".into(), serde_json::json!(pid));
            m.insert("triangles".into(), serde_json::json!(tris));
            crate::bridge::ok_envelope(m)
        }
        Some(Err(e)) => e.to_envelope(),
        None => crate::bridge::BridgeError::new("engine.badHandle").to_envelope(),
    }
}

#[uniffi::export]
pub fn doc_commit_import(handle: i64, prepared: i64, as_new_document: bool, label: String) -> String {
    let reg = crate::bridge::BridgeRegistry::global();
    let guard = reg.held.lock().unwrap();
    let Some(crate::bridge::HeldObject::ImportPrepared(prep)) = guard.get(&prepared) else {
        return crate::bridge::BridgeError::new("engine.badHandle").to_envelope();
    };
    crate::bridge::with_doc(handle, crate::bridge::BridgeError::new("engine.badHandle").to_envelope(), |d| {
        crate::bridge::io::commit_import(d, prep, as_new_document, &label)
    })
}

#[uniffi::export]
pub fn doc_import_limits(physical_memory: i64, available_memory: i64) -> String {
    crate::bridge::io::ImportLimits::for_device(physical_memory, available_memory).to_json()
}

#[uniffi::export]
pub fn doc_voice_call(handle: i64, name: String, args_json: String) -> String {
    crate::bridge::with_doc(handle, crate::bridge::BridgeError::new("engine.badHandle").to_envelope(), |d| {
        crate::bridge::voice::doc_voice_call(d, &name, &args_json)
    })
}

#[uniffi::export]
pub fn voice_call(name: String, args_json: String, payload: Option<String>) -> String {
    crate::bridge::voice::call(&name, &args_json, payload.as_deref())
}

/// JNI entry for the Android sample: `ANativeWindow_fromSurface` (the caller releases it with the view).
#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
pub extern "C" fn Java_com_aja_sculptengine_NativeWindow_fromSurface(env: *mut std::ffi::c_void, _class: *mut std::ffi::c_void, surface: *mut std::ffi::c_void) -> i64 {
    unsafe extern "C" {
        fn ANativeWindow_fromSurface(env: *mut std::ffi::c_void, surface: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    }
    unsafe { ANativeWindow_fromSurface(env, surface) as i64 }
}
