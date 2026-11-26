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

    /// Brush kind ("draw", "clay", "inflate", "crease", "pinch", "grab", "smooth", "flatten", "mask"),
    /// world radius in mm, strength -1 to 1, X symmetry (backwards compatible).
    #[wasm_bindgen(js_name = setBrush)]
    pub fn set_brush(&mut self, kind: &str, radius: f32, strength: f32, symmetry_x: bool) {
        self.session.set_brush_ex(kind, radius, strength, "smooth", symmetry_x, false, false, 1);
    }

    /// Extended Nomad-grade brush settings: kind, world radius in mm, strength -1..1, falloff
    /// ("smooth" | "linear" | "sharp" | "constant"), 3-axis symmetry, and radial symmetry (1..16).
    #[wasm_bindgen(js_name = setBrushEx)]
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
        self.session.set_brush_ex(kind, radius, strength, falloff, sym_x, sym_y, sym_z, radial_count);
    }

    /// Mask operation on object `id`: `"clear"`, `"invert"`, or `"blur"`.
    #[wasm_bindgen(js_name = maskOp)]
    pub fn mask_op(&mut self, id: &str, op: &str) -> bool {
        self.session.mask_op(id, op)
    }

    /// Free Voxel / SDF Remesh: returns `SceneCommand.setMesh` JSON for object `id` at `resolution` (24..128).
    #[wasm_bindgen(js_name = voxelRemesh)]
    pub fn voxel_remesh(&mut self, id: &str, resolution: u32) -> Result<String, JsValue> {
        self.session.voxel_remesh(id, resolution).map(|(_, cmd)| cmd).map_err(err)
    }

    /// Interactive Draw in 3D: picks a 3D world point from screen pixel `(x, y)` (empty if none).
    #[wasm_bindgen(js_name = draw3dPickPoint)]
    pub fn draw3d_pick_point(&mut self, x: f32, y: f32, snap_surface: bool, surface_offset: f32, plane_depth: f32) -> Vec<f32> {
        self.session
            .draw3d_pick_point(x, y, snap_surface, surface_offset, plane_depth)
            .map(|p| vec![p.x, p.y, p.z])
            .unwrap_or_default()
    }

    /// Interactive Draw in 3D: begins a stroke at `(x, y)` and returns `[x, y, z]` world point.
    #[wasm_bindgen(js_name = draw3dBegin)]
    pub fn draw3d_begin(&mut self, x: f32, y: f32, radius: f32, snap_surface: bool, surface_offset: f32, plane_depth: f32) -> Vec<f32> {
        self.session
            .draw3d_begin(x, y, radius, snap_surface, surface_offset, plane_depth)
            .map(|p| vec![p.x, p.y, p.z])
            .unwrap_or_default()
    }

    /// Interactive Draw in 3D: appends a sample at `(x, y)` and returns `[x, y, z]` if added.
    #[wasm_bindgen(js_name = draw3dMove)]
    pub fn draw3d_move(&mut self, x: f32, y: f32, radius: f32) -> Vec<f32> {
        self.session
            .draw3d_move(x, y, radius)
            .map(|p| vec![p.x, p.y, p.z])
            .unwrap_or_default()
    }

    /// Interactive Draw in 3D: finishes the stroke and returns `SceneCommand.addMesh` JSON.
    #[wasm_bindgen(js_name = draw3dEnd)]
    pub fn draw3d_end(&mut self, options_json: &str) -> Result<String, JsValue> {
        self.session.draw3d_end(options_json).map_err(err)
    }

    #[wasm_bindgen(js_name = draw3dCancel)]
    pub fn draw3d_cancel(&mut self) {
        self.session.draw3d_cancel();
    }

    /// Sets the viewport background color in sRGB `[0..1]`.
    #[wasm_bindgen(js_name = setBackground)]
    pub fn set_background(&mut self, r: f32, g: f32, b: f32) {
        self.session.renderer.settings.background = [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)];
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

    /// Adds or replaces an object from the host's scene: base mesh JSON (Sculptor's Mesh shape), column
    /// major world matrix (16 floats, mm, Z up), sRGB colour (4 floats), metallic, roughness.
    #[wasm_bindgen(js_name = upsertObject)]
    pub fn upsert_object(&mut self, id: &str, mesh_json: &str, world: &[f32], color: &[f32], metallic: f32, roughness: f32) -> Result<(), JsValue> {
        let o = crate::session::object_from_parts(id, mesh_json, world, color, metallic, roughness).map_err(err)?;
        self.session.upsert_object(o);
        Ok(())
    }
    /// Zero-JSON binary mesh upload (E4.5).
    #[wasm_bindgen(js_name = upsertObjectBuffers)]
    pub fn upsert_object_buffers(
        &mut self,
        id: &str,
        positions: &[f32],
        face_sizes: &[u32],
        face_indices: &[u32],
        smooth_shading: bool,
        world: &[f32],
        color: &[f32],
        metallic: f32,
        roughness: f32,
    ) -> Result<(), JsValue> {
        let o = crate::session::object_from_buffers(id, positions, face_sizes, face_indices, smooth_shading, world, color, metallic, roughness)
            .map_err(err)?;
        self.session.upsert_object(o);
        Ok(())
    }
    /// Zero-JSON binary mesh replacement (E4.5).
    #[wasm_bindgen(js_name = setMeshBuffers)]
    pub fn set_mesh_buffers(&mut self, id: &str, positions: &[f32], face_sizes: &[u32], face_indices: &[u32], smooth_shading: bool) -> Result<(), JsValue> {
        let m = crate::session::mesh_from_buffers(positions, face_sizes, face_indices, smooth_shading).map_err(err)?;
        self.session.set_mesh(id, m);
        Ok(())
    }
    #[wasm_bindgen(js_name = removeObject)]
    pub fn remove_object(&mut self, id: &str) {
        self.session.remove_object(id)
    }
    #[wasm_bindgen(js_name = setTransform)]
    pub fn set_transform(&mut self, id: &str, world: &[f32]) -> Result<(), JsValue> {
        if world.len() != 16 || !world.iter().all(|x| x.is_finite()) {
            return Err(err("world must be 16 finite floats"));
        }
        self.session.set_transform(id, Mat4::from_cols_slice(world));
        Ok(())
    }
    #[wasm_bindgen(js_name = setMaterial)]
    pub fn set_material(&mut self, id: &str, color: &[f32], metallic: f32, roughness: f32) -> Result<(), JsValue> {
        if color.len() != 4 || !color.iter().all(|x| x.is_finite()) || !metallic.is_finite() || !roughness.is_finite() {
            return Err(err("color must be 4 finite floats"));
        }
        let c = |i: usize| color[i].clamp(0.0, 1.0);
        let m = Material {
            name: String::new(),
            base_color: crate::format::Color { r: c(0), g: c(1), b: c(2), a: c(3) },
            metallic: metallic.clamp(0.0, 1.0),
            roughness: roughness.clamp(0.0, 1.0),
        };
        self.session.set_material(id, m);
        Ok(())
    }
    #[wasm_bindgen(js_name = setVisible)]
    pub fn set_visible(&mut self, id: &str, visible: bool) {
        self.session.set_visible(id, visible);
    }
    #[wasm_bindgen(js_name = setHighlight)]
    pub fn set_highlight(&mut self, id: &str, on: bool) {
        self.session.renderer.set_highlight(id, on)
    }
    #[wasm_bindgen(js_name = frameAll)]
    pub fn frame_all(&mut self) {
        self.session.frame_all()
    }
    /// `{"id": ..., "point": [x, y, z]}` for the surface under a point, or undefined.
    pub fn hit(&mut self, x: f32, y: f32) -> Option<String> {
        self.session.pick(x, y).map(|(id, p)| serde_json::json!({ "id": id, "point": [p.x, p.y, p.z] }).to_string())
    }
    /// Canvas x, y (pixels) and clip depth of a world point; empty when behind the camera.
    pub fn project(&self, x: f32, y: f32, z: f32) -> Vec<f32> {
        self.session.project(Vec3::new(x, y, z)).map(|(a, b, c)| vec![a, b, c]).unwrap_or_default()
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

// Text to 3D (src/text3d). Free functions: they need no GPU, so hosts can run them in a worker.

/// Plans and meshes `prompt`. `options` is JSON: `{maxTriangles, detail, seed, quads}` (all optional).
/// Returns `{ok, understood, notes, triangles, program, parts:[{name, pivot, material, triangles, mesh}]}`.
#[wasm_bindgen(js_name = textTo3d)]
pub fn text_to_3d(prompt: &str, options: &str) -> String {
    crate::text3d::host::generate_json(prompt, options)
}

/// Plans without meshing: `{ok, understood, notes, program}`.
#[wasm_bindgen(js_name = planText)]
pub fn plan_text(prompt: &str, seed: u32) -> String {
    crate::text3d::host::plan_json(prompt, seed)
}

/// Meshes a shape program, or an LLM reply that contains one.
#[wasm_bindgen(js_name = meshProgram)]
pub fn mesh_program(program_or_reply: &str, options: &str) -> String {
    crate::text3d::host::mesh_program_json(program_or_reply, options)
}

/// The system prompt that makes an LLM answer with a shape program.
#[wasm_bindgen(js_name = programPrompt)]
pub fn program_prompt() -> String {
    crate::text3d::host::llm_system_prompt()
}

/// JSON Schema of a shape program, for constrained decoding.
#[wasm_bindgen(js_name = programSchema)]
pub fn program_schema() -> String {
    crate::text3d::host::program_schema()
}

/// On-device free Image-to-3D reconstruction (RGBA8 + optional depth -> text3d-compatible JSON).
#[wasm_bindgen(js_name = imageTo3d)]
pub fn image_to_3d(width: u32, height: u32, rgba: &[u8], depth: &[f32], options_json: &str) -> String {
    let depth_opt = if depth.is_empty() { None } else { Some(depth) };
    crate::img3d::image_to_3d_json(width, height, rgba, depth_opt, options_json)
}

/// On-device Image-to-Gaussian-Splat (`isSpace=false` for items, `isSpace=true` for full spaces).
#[wasm_bindgen(js_name = imageToSplat)]
pub fn image_to_splat(
    width: u32,
    height: u32,
    rgba: &[u8],
    depth: &[f32],
    is_space: bool,
    max_splats: u32,
) -> Result<Vec<u8>, JsValue> {
    let depth_opt = if depth.is_empty() { None } else { Some(depth) };
    let cloud = crate::splat::image_to_splat(width, height, rgba, depth_opt, is_space, max_splats as usize).map_err(err)?;
    Ok(cloud.to_splat_bytes())
}

/// Converts a triangle mesh into a 3D Gaussian Splat cloud (`.splat` bytes).
#[wasm_bindgen(js_name = meshToSplat)]
pub fn mesh_to_splat(
    positions: &[f32],
    indices: &[u32],
    vertex_colors: &[f32],
    fallback_rgba: &[f32],
    is_space: bool,
    max_splats: u32,
) -> Result<Vec<u8>, JsValue> {
    let vc = if vertex_colors.is_empty() { None } else { Some(vertex_colors) };
    let fb = if fallback_rgba.len() >= 4 {
        [fallback_rgba[0], fallback_rgba[1], fallback_rgba[2], fallback_rgba[3]]
    } else {
        [0.8, 0.8, 0.8, 1.0]
    };
    let cloud = crate::splat::mesh_to_splat(positions, indices, vc, fb, is_space, max_splats as usize).map_err(err)?;
    Ok(cloud.to_splat_bytes())
}

/// Edits an item or captured space Gaussian Splat via prompt; returns edited `.splat` bytes.
#[wasm_bindgen(js_name = editSplatPrompt)]
pub fn edit_splat_prompt(splat_bytes: &[u8], prompt: &str) -> Result<Vec<u8>, JsValue> {
    let mut cloud = crate::splat::SplatCloud::from_splat_bytes(splat_bytes).map_err(err)?;
    crate::splat::edit_splat_in_place(&mut cloud, prompt);
    Ok(cloud.to_splat_bytes())
}

/// Edits a Gaussian Splat and returns the human-readable edit summary JSON `{summary, count, modified}`.
#[wasm_bindgen(js_name = editSplatSummary)]
pub fn edit_splat_summary(splat_bytes: &[u8], prompt: &str) -> Result<String, JsValue> {
    let mut cloud = crate::splat::SplatCloud::from_splat_bytes(splat_bytes).map_err(err)?;
    let (summary, modified) = crate::splat::edit_splat_in_place(&mut cloud, prompt);
    Ok(serde_json::json!({ "summary": summary, "count": cloud.len(), "modified": modified }).to_string())
}

#[wasm_bindgen(js_name = splatToPly)]
pub fn splat_to_ply(splat_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let cloud = crate::splat::SplatCloud::from_splat_bytes(splat_bytes).map_err(err)?;
    Ok(cloud.to_ply_bytes())
}

#[wasm_bindgen(js_name = plyToSplat)]
pub fn ply_to_splat(ply_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let cloud = crate::splat::SplatCloud::from_ply_bytes(ply_bytes).map_err(err)?;
    Ok(cloud.to_splat_bytes())
}

#[wasm_bindgen(js_name = splatToCompact)]
pub fn splat_to_compact(splat_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let cloud = crate::splat::SplatCloud::from_splat_bytes(splat_bytes).map_err(err)?;
    Ok(cloud.to_compact_bytes())
}

#[wasm_bindgen(js_name = compactToSplat)]
pub fn compact_to_splat(compact_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let cloud = crate::splat::SplatCloud::from_compact_bytes(compact_bytes).map_err(err)?;
    Ok(cloud.to_splat_bytes())
}

#[wasm_bindgen(js_name = splatStats)]
pub fn splat_stats(splat_bytes: &[u8]) -> Result<String, JsValue> {
    let cloud = crate::splat::SplatCloud::from_splat_bytes(splat_bytes).map_err(err)?;
    Ok(cloud.stats_json())
}

#[wasm_bindgen(js_name = sortSplats)]
pub fn sort_splats(splat_bytes: &[u8], cam_pos: &[f32], cam_forward: &[f32]) -> Result<Vec<u32>, JsValue> {
    if cam_pos.len() != 3 || cam_forward.len() != 3 {
        return Err(err("cam_pos and cam_forward must have 3 floats"));
    }
    let cloud = crate::splat::SplatCloud::from_splat_bytes(splat_bytes).map_err(err)?;
    Ok(crate::splat::sort_splats_by_camera(
        &cloud,
        [cam_pos[0], cam_pos[1], cam_pos[2]],
        [cam_forward[0], cam_forward[1], cam_forward[2]],
    ))
}

/// Draw in 3D free function (no GPU needed): builds a watertight tube/ribbon `SceneCommand.addMesh`
/// JSON from flat `[x, y, z, radius, ...]` samples and options JSON.
#[wasm_bindgen(js_name = draw3dFromPoints)]
pub fn draw3d_from_points(points_xyzr: &[f32], options_json: &str) -> Result<String, JsValue> {
    if points_xyzr.len() < 8 || points_xyzr.len() % 4 != 0 {
        return Err(err("points_xyzr must contain at least 2 (x, y, z, radius) quadruples"));
    }
    let pts: Vec<crate::draw3d::Draw3DPoint> = points_xyzr
        .chunks_exact(4)
        .map(|c| crate::draw3d::Draw3DPoint { pos: Vec3::new(c[0], c[1], c[2]), radius: c[3] })
        .collect();
    let (opts, name) = crate::draw3d::Draw3DOptions::from_json(options_json);
    crate::draw3d::build_tube_add_mesh_command(&pts, &opts, &name, None)
        .map(|(_, _, cmd)| cmd)
        .map_err(err)
}

/// GPU-free Voxel / SDF Remesh helper: takes Sculptor Mesh JSON and returns a `setMesh` command JSON.
#[wasm_bindgen(js_name = voxelRemeshMeshJson)]
pub fn voxel_remesh_mesh_json(id: &str, mesh_json: &str, resolution: u32) -> Result<String, JsValue> {
    let m: MeshIn = serde_json::from_str(mesh_json).map_err(err)?;
    let n = (m.positions.len() / 3) as u32;
    if m.positions.len() % 3 != 0 || m.faces.iter().any(|f| f.len() < 3 || f.iter().any(|&i| i >= n)) {
        return Err(err("invalid mesh"));
    }
    let mesh = Mesh {
        positions: m.positions.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect(),
        faces: m.faces,
        smooth_shading: m.smooth_shading,
    };
    let sm = crate::mesh::SculptMesh::from_mesh(&mesh);
    let bvh = crate::bvh::Bvh::build(&sm.positions, &sm.tris);
    let remeshed = crate::remesh::voxel_remesh(&sm, &bvh, resolution).map_err(err)?;
    Ok(crate::format::set_mesh_command(id, &remeshed))
}

/// GPU-free Nomad Sculpt dab helper (for workers & headless tests): applies one or more dabs on Mesh JSON
/// with optional initial per-vertex masks, returning `{ command, masks }`.
#[wasm_bindgen(js_name = sculptDabMeshJson)]
pub fn sculpt_dab_mesh_json(
    id: &str,
    mesh_json: &str,
    kind: &str,
    radius: f32,
    strength: f32,
    falloff: &str,
    sym_x: bool,
    sym_y: bool,
    sym_z: bool,
    radial_count: u32,
    centers_xyz: &[f32],
    initial_masks: &[f32],
) -> Result<String, JsValue> {
    let m: MeshIn = serde_json::from_str(mesh_json).map_err(err)?;
    let n = (m.positions.len() / 3) as u32;
    if m.positions.len() % 3 != 0 || m.faces.iter().any(|f| f.len() < 3 || f.iter().any(|&i| i >= n)) {
        return Err(err("invalid mesh"));
    }
    let mesh = Mesh {
        positions: m.positions.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect(),
        faces: m.faces,
        smooth_shading: m.smooth_shading,
    };
    let mut sm = crate::mesh::SculptMesh::from_mesh(&mesh);
    if initial_masks.len() == sm.positions.len() {
        sm.masks.copy_from_slice(initial_masks);
    }
    let bvh = crate::bvh::Bvh::build(&sm.positions, &sm.tris);
    let mut scratch = crate::brush::Scratch::new(sm.positions.len());
    let mut moved = Vec::new();
    let mut dirty = Vec::new();
    let p = crate::brush::BrushParams {
        kind: crate::brush::BrushKind::parse(kind),
        radius: radius.max(0.1),
        strength: strength.clamp(-1.0, 1.0),
        symmetry: [sym_x, sym_y, sym_z],
        falloff: crate::brush::Falloff::parse(falloff),
        radial_symmetry: radial_count.clamp(1, 16),
    };
    for c in centers_xyz.chunks_exact(3) {
        crate::brush::dab_and_update_normals(&mut sm, &bvh, &mut scratch, Vec3::new(c[0], c[1], c[2]), &p, &mut moved, &mut dirty);
    }
    let out_mesh = sm.to_mesh();
    let cmd = crate::format::set_mesh_command(id, &out_mesh);
    let cmd_val: serde_json::Value = serde_json::from_str(&cmd).map_err(err)?;
    Ok(serde_json::json!({ "command": cmd_val, "masks": sm.masks }).to_string())
}
