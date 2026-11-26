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

/// JNI entry for the Android sample: `ANativeWindow_fromSurface` (the caller releases it with the view).
#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
pub extern "C" fn Java_com_aja_sculptengine_NativeWindow_fromSurface(env: *mut std::ffi::c_void, _class: *mut std::ffi::c_void, surface: *mut std::ffi::c_void) -> i64 {
    unsafe extern "C" {
        fn ANativeWindow_fromSurface(env: *mut std::ffi::c_void, surface: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    }
    unsafe { ANativeWindow_fromSurface(env, surface) as i64 }
}
