//! Text to 3D from the command line: `cargo run --release --no-default-features --example text3d -- "a red mug" [out.obj]`.
//! Prints the plan, timing and triangle count, and writes an OBJ (with an MTL for the part colours), or with a
//! `.json` path the exact host JSON (options from `TEXT3D_OPTIONS`, e.g. '{"detail":16}').
use sculpt_engine::text3d::{generate, GenOptions};
use std::fmt::Write;

fn main() {
    let mut args = std::env::args().skip(1);
    let prompt = args.next().unwrap_or_else(|| "a snowman next to a pine tree".into());
    let out = args.next();
    let t = std::time::Instant::now();
    let g = generate(&prompt, &GenOptions::default()).expect("generate");
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("{prompt:?}: {} parts, {} triangles, {ms:.1} ms, understood: {}", g.parts.len(), g.triangles(), g.understood);
    for n in &g.notes {
        println!("  note: {n}");
    }
    let Some(path) = out else {
        println!("{}", g.program.to_json());
        return;
    };
    if path.ends_with(".json") {
        // The exact JSON hosts receive (fixtures for the Swift decoders): small options keep it compact.
        let opts = std::env::var("TEXT3D_OPTIONS").unwrap_or_else(|_| "{}".into());
        std::fs::write(&path, sculpt_engine::text3d::host::generate_json(&prompt, &opts)).unwrap();
        println!("wrote {path}");
        return;
    }
    let (mut obj, mut mtl) = (String::new(), String::new());
    let stem = std::path::Path::new(&path).with_extension("mtl");
    writeln!(obj, "mtllib {}", stem.file_name().unwrap().to_string_lossy()).unwrap();
    let mut base = 1;
    for (i, p) in g.parts.iter().enumerate() {
        let c = p.material.base_color;
        writeln!(mtl, "newmtl m{i}\nKd {} {} {}", c.r, c.g, c.b).unwrap();
        writeln!(obj, "o {}\nusemtl m{i}", p.name.replace(' ', "_")).unwrap();
        for v in &p.mesh.positions {
            let w = *v + p.pivot;
            writeln!(obj, "v {} {} {}", w.x, w.z, -w.y).unwrap(); // Z up to Y up for OBJ viewers
        }
        for f in &p.mesh.faces {
            let idx: Vec<String> = f.iter().map(|i| (i + base).to_string()).collect();
            writeln!(obj, "f {}", idx.join(" ")).unwrap();
        }
        base += p.mesh.positions.len() as u32;
    }
    std::fs::write(&path, obj).unwrap();
    std::fs::write(stem, mtl).unwrap();
    println!("wrote {path}");
}
