//! uniffi bindings (Swift for Apple, Kotlin for Android): the same `Session` API as the web build.
//! The host owns the layer or window and hands the engine a raw pointer to it:
//! a `CAMetalLayer *` on Apple, an `ANativeWindow *` on Android (see `nativeWindowFromSurface`).

use crate::brush::BrushKind;
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
    /// kind: "draw", "inflate", "smooth", "flatten"; radius in world mm; strength -1 to 1.
    pub fn set_brush(&self, kind: String, radius: f32, strength: f32, symmetry_x: bool) {
        let b = &mut self.inner.lock().unwrap().session.brush;
        b.kind = BrushKind::parse(&kind);
        b.radius = radius.max(0.1);
        b.strength = strength.clamp(-1.0, 1.0);
        b.symmetry = [symmetry_x, false, false];
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

/// A realistic still of loose triangle buffers, no project needed (Dream Builder, thumbnails).
#[uniffi::export]
pub fn render_snapshot(objects: Vec<SnapshotObject>, camera_json: String, width: u32, height: u32) -> Result<Vec<u8>, EngineError> {
    let cam: OrbitCamera = serde_json::from_str(&camera_json).map_err(|e| EngineError::Invalid(e.to_string()))?;
    let gpu = pollster::block_on(Gpu::headless()).map_err(EngineError::Gpu)?;
    let mut r = Renderer::new(gpu, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.settings.grid = false;
    for (k, o) in objects.iter().enumerate() {
        let n = (o.positions.len() / 3) as u32;
        if o.positions.len() % 3 != 0 || o.indices.len() % 3 != 0 || o.indices.iter().any(|&i| i >= n) || o.world.len() != 16 || o.base_color.len() != 4 {
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
    Ok(png(&r.render_rgba((width.max(1), height.max(1)), &cam), (width.max(1), height.max(1))))
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
