//! Contact sheet of text-to-3D results, rendered by the engine itself (headless wgpu):
//! `cargo run --release --example text3d_shot -- out.png "a red mug" "a castle" ...`
use sculpt_engine::camera::OrbitCamera;
use sculpt_engine::glam::{Mat4, Vec3};
use sculpt_engine::mesh::SculptMesh;
use sculpt_engine::render::{Gpu, Renderer};
use sculpt_engine::text3d::{generate, GenOptions};

const CELL: (u32, u32) = (360, 300);

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let out = if args.is_empty() { "text3d.png".to_string() } else { args.remove(0) };
    if args.is_empty() {
        args = ["a red mug", "a snowman", "a cat with a hat on a table", "a castle", "a dragon", "a pine tree next to a house"]
            .iter().map(|s| s.to_string()).collect();
    }
    let gpu = pollster::block_on(Gpu::headless()).expect("gpu");
    let cols = 3.min(args.len() as u32);
    let rows = (args.len() as u32).div_ceil(cols);
    let mut sheet = image::RgbaImage::new(CELL.0 * cols, CELL.1 * rows);
    for (n, prompt) in args.iter().enumerate() {
        let g = generate(prompt, &GenOptions::default()).expect("generate");
        let mut r = Renderer::new(gpu.clone(), wgpu::TextureFormat::Rgba8UnormSrgb);
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for (i, p) in g.parts.iter().enumerate() {
            for v in &p.mesh.positions {
                lo = lo.min(*v + p.pivot);
                hi = hi.max(*v + p.pivot);
            }
            r.upsert(&i.to_string(), &SculptMesh::from_mesh(&p.mesh), Mat4::from_translation(p.pivot), &p.material);
        }
        let c = (lo + hi) * 0.5;
        let cam = OrbitCamera { target: c.to_array(), distance: (hi - lo).length() * 1.45, yaw_degrees: 30.0, pitch_degrees: 22.0, ..OrbitCamera::default() };
        let px = r.render_rgba(CELL, &cam);
        let img = image::RgbaImage::from_raw(CELL.0, CELL.1, px).expect("pixels");
        image::imageops::overlay(&mut sheet, &img, ((n as u32 % cols) * CELL.0) as i64, ((n as u32 / cols) * CELL.1) as i64);
        println!("{prompt:?}: {} triangles", g.triangles());
    }
    sheet.save(&out).expect("save");
    println!("wrote {out}");
}
