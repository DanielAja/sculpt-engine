//! wasm-bindgen API: `Session` for a canvas. WebGPU when the browser has it, WebGL2 otherwise.
//! Coordinates are CSS pixels times `devicePixelRatio` (the canvas's drawing buffer pixels).

use crate::format::{Material, Mesh};
use crate::render::{Gpu, Renderer, Tonemap};
use crate::session::{Object, Session};
use glam::{Mat4, Vec3};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WebEngine {
    session: Session,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen]
impl WebEngine {
    /// Opens the GPU on `canvas`: WebGPU when available, else WebGL2 (always WebGL2 with `force_webgl`).
    /// Resolves to an engine showing a clay sphere until a project is opened.
    pub async fn create(canvas: web_sys::HtmlCanvasElement, force_webgl: bool) -> Result<WebEngine, JsValue> {
        console_error_panic_hook_set();
        let (w, h) = (canvas.width().max(1), canvas.height().max(1));
        let desc = wgpu::InstanceDescriptor::new_without_display_handle();
        let instance = if force_webgl {
            wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::GL, ..desc })
        } else {
            wgpu::util::new_instance_with_webgpu_detection(desc).await
        };
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas)).map_err(err)?;
        let gpu = Gpu::new(instance, Some(&surface)).await.map_err(err)?;
        let caps = surface.get_capabilities(&gpu.adapter);
        // Prefer an sRGB format; browsers usually offer only non-sRGB ones, and the tone map pass encodes then.
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: Default::default(),
        };
        surface.configure(&gpu.device, &config);
        let mut session = Session::new(Renderer::new(gpu, format), (w, h));
        session.set_objects(vec![Object {
            id: "SPHERE".into(),
            mesh: Arc::new(crate::mesh::uv_sphere(20.0, 256, 128)),
            world: Mat4::from_translation(Vec3::new(0.0, 0.0, 20.0)),
            material: Material::default(),
        }]);
        session.frame_all();
        Ok(WebEngine { session, surface, config })
    }

    /// "webgpu" or "webgl2" (and the adapter name).
    pub fn backend(&self) -> String {
        let info = self.session.renderer.gpu.adapter.get_info();
        format!("{:?} ({})", info.backend, info.name)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.session.renderer.gpu.device, &self.config);
        self.session.resize(width, height);
    }

    pub fn render(&mut self) -> Result<(), JsValue> {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.session.renderer.gpu.device, &self.config);
                return Ok(());
            }
            _ => return Ok(()),
        };
        self.session.render(&frame.texture.create_view(&Default::default()));
        self.session.renderer.gpu.queue.present(frame);
        Ok(())
    }

    #[wasm_bindgen(js_name = openProject)]
    pub fn open_project(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        self.session.open_project(bytes).map_err(err)
    }

    /// The `.sculpt` text with sculpted meshes, or undefined when no project was opened.
    #[wasm_bindgen(js_name = projectJson)]
    pub fn project_json(&self) -> Option<String> {
        self.session.project_json()
    }

    /// Sculptor's OrbitCamera JSON (the `cameraState` shape).
    #[wasm_bindgen(js_name = setCamera)]
    pub fn set_camera(&mut self, json: &str) -> Result<(), JsValue> {
        self.session.camera = serde_json::from_str(json).map_err(err)?;
        Ok(())
    }
    #[wasm_bindgen(js_name = cameraJson)]
    pub fn camera_json(&self) -> String {
        serde_json::to_string(&self.session.camera).unwrap()
    }
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.session.camera = self.session.camera.orbit(-dx * 0.4, dy * 0.4);
    }
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.session.camera = self.session.camera.pan(dx, dy, self.session.size.1 as f32);
    }
    pub fn zoom(&mut self, factor: f32) {
        self.session.camera = self.session.camera.zoom(factor);
    }

    /// Brush kind ("draw", "inflate", "smooth", "flatten"), world radius in mm, strength -1 to 1, X symmetry.
    #[wasm_bindgen(js_name = setBrush)]
    pub fn set_brush(&mut self, kind: &str, radius: f32, strength: f32, symmetry_x: bool) {
        self.session.brush.kind = crate::brush::BrushKind::parse(kind);
        self.session.brush.radius = radius.max(0.1);
        self.session.brush.strength = strength.clamp(-1.0, 1.0);
        self.session.brush.symmetry = [symmetry_x, false, false];
    }

    /// Picked object id under a point, if any.
    pub fn pick(&mut self, x: f32, y: f32) -> Option<String> {
        self.session.pick(x, y).map(|h| h.0)
    }
    #[wasm_bindgen(js_name = sculptBegin)]
    pub fn sculpt_begin(&mut self, x: f32, y: f32, pressure: f32) -> bool {
        self.session.sculpt_begin(x, y, pressure)
    }
    #[wasm_bindgen(js_name = sculptMove)]
    pub fn sculpt_move(&mut self, x: f32, y: f32, pressure: f32) {
        self.session.sculpt_move(x, y, pressure)
    }
    /// The `setMesh` SceneCommand JSON (for the Swift core's `apply`), or undefined for an empty stroke.
    #[wasm_bindgen(js_name = sculptEnd)]
    pub fn sculpt_end(&mut self) -> Option<String> {
        self.session.sculpt_end().map(|r| r.command_json)
    }
    #[wasm_bindgen(js_name = sculptCancel)]
    pub fn sculpt_cancel(&mut self) {
        self.session.sculpt_cancel()
    }
    pub fn undo(&mut self) -> bool {
        self.session.undo()
    }
    pub fn redo(&mut self) -> bool {
        self.session.redo()
    }

    /// Replaces an object's mesh with Mesh JSON (`{"positions":[...],"faces":[...],"smoothShading":b}`).
    #[wasm_bindgen(js_name = setMeshJson)]
    pub fn set_mesh_json(&mut self, id: &str, json: &str) -> Result<(), JsValue> {
        let m: MeshIn = serde_json::from_str(json).map_err(err)?;
        let n = (m.positions.len() / 3) as u32;
        if m.positions.len() % 3 != 0 || m.faces.iter().any(|f| f.len() < 3 || f.iter().any(|&i| i >= n)) {
            return Err(err("invalid mesh"));
        }
        let positions = m.positions.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect();
        self.session.set_mesh(id, Mesh { positions, faces: m.faces, smooth_shading: m.smooth_shading });
        Ok(())
    }

    /// "agx", "aces" or "linear"; SSAO and shadows on or off.
    #[wasm_bindgen(js_name = setLook)]
    pub fn set_look(&mut self, tonemap: &str, ssao: bool, shadows: bool) {
        let s = &mut self.session.renderer.settings;
        s.tonemap = match tonemap {
            "aces" => Tonemap::Aces,
            "linear" => Tonemap::Linear,
            _ => Tonemap::AgX,
        };
        s.ssao = ssao;
        s.shadows = shadows;
    }
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct MeshIn {
    positions: Vec<f32>,
    faces: Vec<Vec<u32>>,
    #[serde(default)]
    smooth_shading: bool,
}

/// Panics show up in the browser console instead of as "unreachable executed".
fn console_error_panic_hook_set() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            web_sys::console::error_1(&JsValue::from_str(&info.to_string()));
        }))
    });
}
