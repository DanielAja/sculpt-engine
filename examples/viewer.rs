//! Native demo: opens a `.sculpt` (or a clay sphere) in a window and sculpts it with the mouse.
//!
//! cargo run --release --example viewer [project.sculpt]
//!   left drag on a mesh: sculpt (Option/Alt inverts) | left drag elsewhere or right drag: orbit
//!   middle or shift drag: pan | wheel: zoom | 1 draw, 2 inflate, 3 smooth, 4 flatten | X symmetry
//!   [ and ]: radius | Cmd/Ctrl Z, Shift Z: undo, redo | T tone map | O SSAO | S shadows | F frame
//!
//! Headless modes (no window):
//!   --shot out.png [--stroke]   render the scene (after a scripted stroke) to a PNG
//!   --bench                     average frame time at 1920x1080 over 120 frames

use sculpt_engine::brush::{BrushKind, BrushParams};
use sculpt_engine::camera::OrbitCamera;
use sculpt_engine::format::{Material, Mesh, ProjectFile};
use sculpt_engine::glam::{Mat4, Vec3};
use sculpt_engine::mesh::{uv_sphere, SculptMesh};
use sculpt_engine::render::{world_matrix, Gpu, Renderer, Tonemap};
use sculpt_engine::session::{Object, Session};
use sculpt_engine::stroke::SculptTarget;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

/// The scene as the demo keeps it: objects with their base meshes and world matrices.
struct Obj {
    id: String,
    mesh: Arc<Mesh>,
    world: Mat4,
    material: Material,
}

fn load(args: &[String]) -> Vec<Obj> {
    if let Some(path) = args.iter().find(|a| a.ends_with(".sculpt")) {
        let bytes = std::fs::read(path).expect("read project");
        let p = ProjectFile::decode(&bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
        return p
            .scene
            .objects
            .iter()
            .filter(|o| o.is_visible)
            .map(|o| {
                Obj { id: o.id.clone(), mesh: Arc::new(o.mesh.clone()), world: world_matrix(&p, &o.id), material: o.material.clone() }
            })
            .collect();
    }
    // About 260k triangles of clay (`--big`: about 1M).
    let (seg, rings) = if args.iter().any(|a| a == "--big") { (1024, 512) } else { (512, 256) };
    vec![Obj { id: "SPHERE".into(), mesh: Arc::new(uv_sphere(20.0, seg, rings)), world: Mat4::from_translation(Vec3::new(0.0, 0.0, 20.0)), material: Material::default() }]
}

fn upload(r: &mut Renderer, objs: &[Obj]) {
    r.clear();
    for o in objs {
        r.upsert(&o.id, &SculptMesh::from_mesh(&o.mesh), o.world, &o.material);
    }
}

fn framed(r: &Renderer, aspect: f32) -> OrbitCamera {
    let (a, b) = r.scene_bounds();
    OrbitCamera::default().fitted(a, b, aspect)
}

fn brush(kind: BrushKind, radius: f32, invert: bool, symmetry: bool) -> BrushParams {
    BrushParams { kind, radius, strength: if invert { -0.6 } else { 0.6 }, symmetry: [symmetry, false, false], ..BrushParams::default() }
}

fn headless(args: &[String]) {
    let gpu = pollster::block_on(Gpu::headless()).expect("gpu");
    let mut r = Renderer::new(gpu.clone(), wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut objs = load(args);
    if args.iter().any(|a| a == "--stroke") {
        // A curved draw stroke across the front of the first object, then an inflate dab, with X symmetry.
        let o = &mut objs[0];
        let mut t = SculptTarget::new(&o.id, &o.mesh, o.world);
        t.begin_stroke();
        let p = brush(BrushKind::Draw, 6.0, false, true);
        for i in 0..=40 {
            let a = -1.2 + 2.4 * i as f32 / 40.0;
            let (origin, dir) = (Vec3::new(8.0 + 6.0 * a.sin(), -100.0, 20.0 + 10.0 * a), Vec3::Y);
            if let Some(h) = t.pick(origin, dir) {
                t.drag_to(h.point, &p, 0.25);
            }
        }
        t.end_stroke();
        t.begin_stroke();
        if let Some(h) = t.pick(Vec3::new(0.0, -100.0, 32.0), Vec3::Y) {
            t.dab_points(&[h.point; 6], &brush(BrushKind::Inflate, 5.0, false, false));
        }
        let (mesh, _) = t.end_stroke().unwrap();
        o.mesh = Arc::new(mesh);
    }
    upload(&mut r, &objs);
    r.settings.ssao = !args.iter().any(|a| a == "--no-ssao");
    r.settings.shadows = !args.iter().any(|a| a == "--no-shadows");
    if args.iter().any(|a| a == "--debug-ao") {
        r.settings.debug_ao = true;
        r.settings.tonemap = Tonemap::Linear;
    }
    if args.iter().any(|a| a == "--bench") {
        let size = (1920, 1080);
        let cam = framed(&r, 16.0 / 9.0);
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let tris: usize = objs.iter().map(|o| SculptMesh::from_mesh(&o.mesh).tris.len()).sum();
        for (label, ssao, shadows) in [("full (SSAO, shadows)", true, true), ("no SSAO", false, true), ("studio (no SSAO, no shadows)", false, false)] {
            r.settings.ssao = ssao;
            r.settings.shadows = shadows;
            for _ in 0..10 {
                r.render(&view, size, &cam);
            }
            gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            let n = 120;
            let t0 = Instant::now();
            for i in 0..n {
                r.render(&view, size, &cam.orbit(i as f32 * 0.5, 0.0));
                gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            }
            println!("frame 1920x1080, {tris} tris, {label}: {:.2} ms", t0.elapsed().as_secs_f64() * 1000.0 / n as f64);
        }
        return;
    }
    let out = args.iter().position(|a| a == "--shot").and_then(|i| args.get(i + 1)).expect("--shot <out.png>");
    let size = (1280, 800);
    let mut cam = framed(&r, 1.6);
    cam.yaw_degrees = 20.0;
    let px = r.render_rgba(size, &cam);
    image::save_buffer(out, &px, size.0, size.1, image::ColorType::Rgba8).unwrap();
    println!("wrote {out}");
}

struct App {
    args: Vec<String>,
    state: Option<State>,
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    s: Session,
    drag: Option<MouseButton>,
    sculpting: bool,
    cursor: (f64, f64),
    mods: ModifiersState,
    frames: (u32, Instant),
    total_frames: u64,
}

impl State {
    fn title(&self, frame_ms: f64) {
        let (b, r) = (&self.s.brush, &self.s.renderer.settings);
        self.window.set_title(&format!(
            "sculpt-engine | {:?} r={:.1} sym={} | tone={:?} ssao={} shadows={} | {frame_ms:.2} ms/frame | undo: {}",
            b.kind, b.radius, b.symmetry[0], r.tonemap, r.ssao, r.shadows, self.s.history.undo_label().unwrap_or("-")
        ));
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(el.create_window(Window::default_attributes().with_title("sculpt-engine").with_inner_size(winit::dpi::LogicalSize::new(1280, 800))).unwrap());
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).unwrap();
        let gpu = pollster::block_on(Gpu::new(instance, Some(&surface))).unwrap();
        let size = window.inner_size();
        let config = surface.get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1)).unwrap();
        let config = wgpu::SurfaceConfiguration { format: config.format.add_srgb_suffix(), present_mode: wgpu::PresentMode::AutoVsync, ..config };
        surface.configure(&gpu.device, &config);
        println!("adapter: {:?}, tier {:?}, surface {:?}", gpu.adapter.get_info().name, gpu.tier, config.format);
        let mut s = Session::new(Renderer::new(gpu, config.format), (config.width, config.height));
        match self.args.iter().find(|a| a.ends_with(".sculpt")) {
            Some(path) => s.open_project(&std::fs::read(path).expect("read project")).unwrap_or_else(|e| panic!("{path}: {e}")),
            None => {
                s.set_objects(load(&self.args).into_iter().map(|o| Object { id: o.id, mesh: o.mesh, world: o.world, material: o.material }).collect());
                s.frame_all();
            }
        }
        self.state = Some(State { window, surface, config, s, drag: None, sculpting: false, cursor: (0.0, 0.0), mods: ModifiersState::empty(), frames: (0, Instant::now()), total_frames: 0 });
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(st) = self.state.as_mut() else { return };
        let s = &mut st.s;
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                st.config.width = size.width.max(1);
                st.config.height = size.height.max(1);
                st.surface.configure(&s.renderer.gpu.device, &st.config);
                s.resize(st.config.width, st.config.height);
            }
            WindowEvent::ModifiersChanged(m) => st.mods = m.state(),
            WindowEvent::CursorMoved { position, .. } => {
                let (dx, dy) = ((position.x - st.cursor.0) as f32, (position.y - st.cursor.1) as f32);
                st.cursor = (position.x, position.y);
                let h = st.config.height as f32;
                match st.drag {
                    Some(_) if st.sculpting => s.sculpt_move(position.x as f32, position.y as f32, 1.0),
                    Some(MouseButton::Middle) => s.camera = s.camera.pan(dx, dy, h),
                    Some(MouseButton::Left) if st.mods.shift_key() => s.camera = s.camera.pan(dx, dy, h),
                    Some(_) => s.camera = s.camera.orbit(-dx * 0.4, dy * 0.4),
                    None => {}
                }
            }
            WindowEvent::MouseInput { state, button, .. } => match state {
                ElementState::Pressed => {
                    st.drag = Some(button);
                    if button == MouseButton::Left && !st.mods.shift_key() {
                        let invert = st.mods.alt_key();
                        s.brush.strength = if invert { -s.brush.strength.abs() } else { s.brush.strength.abs() };
                        st.sculpting = s.sculpt_begin(st.cursor.0 as f32, st.cursor.1 as f32, 1.0);
                    }
                }
                ElementState::Released => {
                    st.drag = None;
                    if std::mem::take(&mut st.sculpting) {
                        s.sculpt_end();
                    }
                }
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 20.0,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32,
                };
                s.camera = s.camera.zoom((y * 0.01).exp());
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                let cmd = st.mods.super_key() || st.mods.control_key();
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Escape) => el.exit(),
                    Key::Character("1") => s.brush.kind = BrushKind::Draw,
                    Key::Character("2") => s.brush.kind = BrushKind::Inflate,
                    Key::Character("3") => s.brush.kind = BrushKind::Smooth,
                    Key::Character("4") => s.brush.kind = BrushKind::Flatten,
                    Key::Character("x") => s.brush.symmetry[0] = !s.brush.symmetry[0],
                    Key::Character("[") => s.brush.radius = (s.brush.radius / 1.2).max(0.5),
                    Key::Character("]") => s.brush.radius = (s.brush.radius * 1.2).min(100.0),
                    Key::Character("z" | "Z") if cmd && st.mods.shift_key() => {
                        s.redo();
                    }
                    Key::Character("z") if cmd => {
                        s.undo();
                    }
                    Key::Character("t") => {
                        s.renderer.settings.tonemap = match s.renderer.settings.tonemap {
                            Tonemap::AgX => Tonemap::Aces,
                            Tonemap::Aces => Tonemap::Linear,
                            Tonemap::Linear => Tonemap::AgX,
                        }
                    }
                    Key::Character("o") => s.renderer.settings.ssao = !s.renderer.settings.ssao,
                    Key::Character("s") => s.renderer.settings.shadows = !s.renderer.settings.shadows,
                    Key::Character("f") => s.frame_all(),
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                let frame = match st.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                        st.surface.configure(&s.renderer.gpu.device, &st.config);
                        return;
                    }
                    _ => return,
                };
                s.render(&frame.texture.create_view(&Default::default()));
                st.window.pre_present_notify();
                s.renderer.gpu.queue.present(frame);
                st.frames.0 += 1;
                st.total_frames += 1;
                // --capture out.png: screenshot this window's region once it has settled, then quit.
                if st.total_frames == 90 {
                    if let Some(out) = self.args.iter().position(|a| a == "--capture").and_then(|i| self.args.get(i + 1)) {
                        let (pos, size, scale) = (st.window.inner_position().unwrap(), st.window.inner_size(), st.window.scale_factor());
                        let r = format!("{},{},{},{}", (pos.x as f64 / scale) as i32, (pos.y as f64 / scale) as i32, (size.width as f64 / scale) as i32, (size.height as f64 / scale) as i32);
                        let ok = std::process::Command::new("screencapture").args(["-x", "-R", &r, out]).status().map(|x| x.success()).unwrap_or(false);
                        println!("capture {r} -> {out}: {ok}");
                        el.exit();
                    }
                }
                if st.frames.1.elapsed().as_secs_f64() > 0.5 {
                    let ms = st.frames.1.elapsed().as_secs_f64() * 1000.0 / st.frames.0 as f64;
                    st.frames = (0, Instant::now());
                    st.title(ms);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--shot" || a == "--bench") {
        return headless(&args);
    }
    let el = EventLoop::new().unwrap();
    el.run_app(&mut App { args, state: None }).unwrap();
}
