//! Golden image tests: headless renders compared with stored PNGs (DESIGN.md section 10).
//! Goldens are per backend; set UPDATE_GOLDENS=1 to rewrite them after an intended change.

use sculpt_engine::camera::OrbitCamera;
use sculpt_engine::format::{Color, Material, ProjectFile};
use sculpt_engine::glam::{Mat4, Vec3};
use sculpt_engine::mesh::{uv_sphere, SculptMesh};
use sculpt_engine::render::{Gpu, Renderer, Tonemap};
use std::path::PathBuf;

const SIZE: (u32, u32) = (320, 240);

fn renderer() -> Option<Renderer> {
    match pollster::block_on(Gpu::headless()) {
        Ok(gpu) => Some(Renderer::new(gpu, wgpu::TextureFormat::Rgba8UnormSrgb)),
        Err(e) => {
            eprintln!("skipping render goldens: {e}");
            None
        }
    }
}

fn check(name: &str, pixels: &[u8]) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let path = dir.join(format!("{name}.png"));
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/render-out");
    std::fs::create_dir_all(&out).unwrap();
    image::save_buffer(out.join(format!("{name}.png")), pixels, SIZE.0, SIZE.1, image::ColorType::Rgba8).unwrap();
    if std::env::var("UPDATE_GOLDENS").is_ok() || !path.exists() {
        image::save_buffer(&path, pixels, SIZE.0, SIZE.1, image::ColorType::Rgba8).unwrap();
        eprintln!("wrote golden {}", path.display());
        return;
    }
    let want = image::open(&path).unwrap().to_rgba8();
    assert_eq!(want.dimensions(), SIZE);
    let (mut sum, mut bad) = (0u64, 0usize);
    for (a, b) in pixels.iter().zip(want.as_raw()) {
        let d = (*a as i32 - *b as i32).unsigned_abs();
        sum += d as u64;
        if d > 8 {
            bad += 1;
        }
    }
    let mean = sum as f64 / pixels.len() as f64;
    let frac = bad as f64 / pixels.len() as f64;
    assert!(mean < 1.5 && frac < 0.005, "{name}: mean abs error {mean:.3}, {:.3}% channels off by more than 8 (see target/render-out)", frac * 100.0);
}

fn mat(name: &str, hex: u32, metallic: f32, roughness: f32) -> Material {
    let c = |s: u32| ((hex >> s) & 0xff) as f32 / 255.0;
    Material { name: name.into(), base_color: Color { r: c(16), g: c(8), b: c(0), a: 1.0 }, metallic, roughness }
}

#[test]
fn golden_images() {
    let Some(mut r) = renderer() else { return };
    let project = ProjectFile::decode(include_bytes!("fixtures/project.sculpt")).unwrap();
    r.set_project(&project);
    let cam = OrbitCamera { distance: 140.0, ..OrbitCamera::default() };
    let px = r.render_rgba(SIZE, &cam);
    check("project", &px);

    // Material row: clay, glossy plastic, rough steel, polished gold.
    r.clear();
    let sphere = SculptMesh::from_mesh(&uv_sphere(9.0, 64, 32));
    let mats = [mat("Clay", 0xE9C9A8, 0.0, 0.6), mat("Sky", 0x4D96FF, 0.0, 0.15), mat("Steel", 0xA8ADB4, 1.0, 0.45), mat("Gold", 0xE8B64C, 1.0, 0.15)];
    for (i, m) in mats.iter().enumerate() {
        r.upsert(&format!("s{i}"), &sphere, Mat4::from_translation(Vec3::new(-30.0 + 20.0 * i as f32, 0.0, 9.0)), m);
    }
    let cam = OrbitCamera { target: [0.0, 0.0, 9.0], distance: 110.0, yaw_degrees: 0.0, pitch_degrees: 15.0, ..OrbitCamera::default() };
    check("materials", &r.render_rgba(SIZE, &cam));

    // The "Studio" preset: linear tone map, no SSAO (today's Android look).
    r.settings.tonemap = Tonemap::Linear;
    r.settings.ssao = false;
    check("materials-studio", &r.render_rgba(SIZE, &cam));
}

/// The host API end to end: open the golden project, sculpt with screen points, get the Swift command,
/// undo, redo, and write the project back.
#[test]
fn session_sculpt_undo_and_save() {
    use sculpt_engine::session::Session;
    let Some(r) = renderer() else { return };
    let mut s = Session::new(r, (800, 600));
    s.open_project(include_bytes!("fixtures/project.sculpt")).unwrap();
    s.camera = OrbitCamera { target: [0.0, 0.0, 10.0], distance: 80.0, yaw_degrees: 0.0, pitch_degrees: 0.0, ..OrbitCamera::default() };
    // The cube is centred at the screen centre in this front view.
    assert_eq!(s.pick(400.0, 300.0).map(|h| h.0), Some("0A1B2C3D-0000-4000-8000-000000000001".to_string()));
    assert!(!s.sculpt_begin(5.0, 5.0, 1.0), "a miss starts no stroke");
    // The cube has only its 8 corners: a small brush on a face moves nothing (and is no undo step).
    assert!(s.sculpt_begin(400.0, 300.0, 1.0));
    assert!(s.sculpt_end().is_none());
    s.brush.radius = 20.0;
    assert!(s.sculpt_begin(400.0, 300.0, 1.0));
    s.sculpt_move(420.0, 300.0, 1.0);
    let r = s.sculpt_end().unwrap();
    assert!(r.command_json.starts_with(r#"{"setMesh":{"_0":{"rawValue":"0A1B2C3D-0000-4000-8000-000000000001"}"#));
    let before = ProjectFile::decode(include_bytes!("fixtures/project.sculpt")).unwrap();
    let moved = ProjectFile::decode(s.project_json().unwrap().as_bytes()).unwrap();
    assert_ne!(moved.scene.objects[0].mesh.positions, before.scene.objects[0].mesh.positions);
    assert_eq!(moved.scene.objects[0].mesh.faces, before.scene.objects[0].mesh.faces);
    assert_eq!(moved.scene.objects[0].mesh.smooth_shading, false, "the file's shading flag is kept");
    assert!(s.undo());
    assert_eq!(s.project_json().unwrap(), include_str!("fixtures/project.sculpt"), "undo restores the exact bytes");
    assert!(s.redo());
    assert_eq!(s.project_json().unwrap(), moved.encode());
}

/// Hosts that own the scene push objects in, pick them and project points; bad input is refused.
#[test]
fn session_host_objects() {
    use sculpt_engine::session::{object_from_parts, Session};
    let Some(r) = renderer() else { return };
    let mut s = Session::new(r, (800, 600));
    let world: Vec<f32> = Mat4::from_translation(Vec3::new(0.0, 0.0, 10.0)).to_cols_array().to_vec();
    let cube = r#"{"positions":[-10,-10,-10,10,-10,-10,10,10,-10,-10,10,-10,-10,-10,10,10,-10,10,10,10,10,-10,10,10],
      "faces":[[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]],"smoothShading":false}"#;
    s.upsert_object(object_from_parts("A", cube, &world, &[0.2, 0.6, 0.6, 1.0], 0.0, 0.5).unwrap());
    s.camera = OrbitCamera { target: [0.0, 0.0, 10.0], distance: 80.0, yaw_degrees: 0.0, pitch_degrees: 0.0, ..OrbitCamera::default() };
    let (id, p) = s.pick(400.0, 300.0).unwrap();
    assert_eq!(id, "A");
    assert!((p - Vec3::new(0.0, -10.0, 10.0)).length() < 1e-3, "{p}");
    let (x, y, _) = s.project(Vec3::new(0.0, 0.0, 10.0)).unwrap();
    assert!((x - 400.0).abs() < 0.01 && (y - 300.0).abs() < 0.01);
    s.remove_object("A");
    assert!(s.pick(400.0, 300.0).is_none());
    assert!(object_from_parts("B", r#"{"positions":[0,0,0],"faces":[[0,1,2]]}"#, &world, &[1.0; 4], 0.0, 0.5).is_err());
    assert!(object_from_parts("B", cube, &[0.0; 15], &[1.0; 4], 0.0, 0.5).is_err());
}
