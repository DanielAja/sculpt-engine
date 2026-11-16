use super::program::*;
use super::sdf::{Aabb, Compiled};
use super::*;
use std::collections::HashMap;

fn opts() -> GenOptions {
    GenOptions { detail: 40, ..GenOptions::default() }
}

/// Sculptor's definition (SculptorCore `isWatertightAndConsistent`): every directed edge appears exactly once
/// and its reverse appears exactly once, so the mesh is closed, manifold and consistently wound.
fn closed(m: &crate::format::Mesh) -> bool {
    let mut e: HashMap<(u32, u32), i32> = HashMap::new();
    for f in &m.faces {
        for k in 0..f.len() {
            *e.entry((f[k], f[(k + 1) % f.len()])).or_default() += 1;
        }
    }
    e.iter().all(|(&(a, b), &n)| n == 1 && e.get(&(b, a)) == Some(&1))
}

fn bounds(parts: &[GenPart]) -> Aabb {
    let mut b = Aabb::EMPTY;
    for p in parts {
        for v in &p.mesh.positions {
            let w = *v + p.pivot;
            b = b.union(&Aabb::new(w, w));
        }
    }
    b
}

fn part_bounds(g: &Generated, prefix: &str) -> Aabb {
    let ps: Vec<GenPart> = g.parts.iter().filter(|p| p.name.starts_with(prefix)).cloned().collect();
    assert!(!ps.is_empty(), "no part starting with {prefix}: {:?}", g.parts.iter().map(|p| &p.name).collect::<Vec<_>>());
    bounds(&ps)
}

#[test]
fn every_template_meshes_closed_within_budget() {
    let words = [
        "mug", "glass", "bowl", "plate", "vase", "bottle", "jar", "pot", "bucket", "teapot", "table", "chair", "stool", "bench", "bed",
        "shelf", "sofa", "lamp", "candle", "book", "phone", "tree", "pine tree", "mushroom", "flower", "cactus", "rock", "cloud",
        "mountain", "snowman", "house", "tower", "castle", "tent", "igloo", "car", "truck", "bus", "boat", "rocket", "plane", "ufo",
        "robot", "person", "teddy bear", "duck", "fish", "snake", "turtle", "frog", "apple", "pear", "banana", "carrot", "pumpkin",
        "egg", "donut", "cake", "cupcake", "ice cream", "burger", "lollipop", "heart", "star", "crown", "ring", "gem", "trophy",
        "sword", "hammer", "key", "pawn", "ghost", "planet", "hat", "dice", "coin", "barrel", "chest", "umbrella", "balloon", "sphere",
        "cube", "cylinder", "tube", "cone", "torus", "capsule", "pyramid", "disc", "ellipsoid", "dragon", "cat", "dog", "horse",
        "unicorn", "cow", "pig", "sheep", "lion", "bear", "elephant", "giraffe", "rabbit", "mouse", "deer", "dinosaur", "crocodile",
        "bird", "eagle", "owl", "penguin", "swan", "flamingo", "wombat", "crow", "taxi", "cottage", "rose", "kettle", "crate", "lemon",
        "axe", "statue", "canoe", "helicopter", "cookie", "lighthouse",
    ];
    let dream_builder = GenOptions { max_triangles: 8_000, quads: false, ..GenOptions::default() };
    let mut strict_at_8k = 0;
    for w in words {
        // Dream Builder's budget: always within it, and manifold for nearly every template.
        let d = generate(&format!("a {w}"), &dream_builder).unwrap_or_else(|e| panic!("{w} at 8k: {e}"));
        assert!(d.triangles() <= 8_000, "{w}: {} triangles at 8k", d.triangles());
        strict_at_8k += d.parts.iter().all(|p| closed(&p.mesh)) as usize;
        // Sculptor's defaults.
        let g = generate(&format!("a {w}"), &GenOptions::default()).unwrap_or_else(|e| panic!("{w}: {e}"));
        assert!(g.parts.iter().all(|p| closed(&p.mesh)), "{w}: not manifold at the default detail");
        let g = generate(&format!("a {w}"), &opts()).unwrap_or_else(|e| panic!("{w}: {e}"));
        assert!(g.understood, "{w} not understood");
        assert!(g.triangles() > 50 && g.triangles() <= opts().max_triangles, "{w}: {} triangles", g.triangles());
        for p in &g.parts {
            assert!(closed(&p.mesh), "{w}: part {} is not closed", p.name);
            assert!(p.mesh.positions.iter().all(|v| v.is_finite()), "{w}: non-finite vertex");
        }
        let b = bounds(&g.parts);
        assert!(b.min.z > -3.0 && b.min.z < 8.0, "{w}: should stand on the ground, min z {}", b.min.z);
        assert!(b.size().max_element() < 600.0, "{w}: {:?}", b.size());
    }
    println!("manifold at Dream Builder's 8k budget: {strict_at_8k} of {}", words.len());
    assert!(strict_at_8k * 10 >= words.len() * 9, "at least 90 % manifold at 8k ({strict_at_8k} of {})", words.len());
}

#[test]
fn deterministic_and_seeded() {
    let a = generate("a tree next to a red car", &opts()).unwrap();
    let b = generate("a tree next to a red car", &opts()).unwrap();
    assert_eq!(a, b);
    let c = generate("a tree next to a red car", &GenOptions { seed: 7, ..opts() }).unwrap();
    assert_ne!(a.program, c.program, "another seed gives another take");
}

#[test]
fn colours_materials_and_shades() {
    let g = text::plan("a blue mug", 0);
    assert_eq!(g.program.parts[0].color.as_deref(), Some("#2f6fd6"));
    let g = text::plan("a gold robot", 0);
    assert_eq!(g.program.parts[0].metallic, 1.0);
    assert_eq!(g.program.parts[0].color.as_deref(), Some("#e6b84d"));
    let g = text::plan("a light blue vase", 0);
    assert_ne!(g.program.parts[0].color.as_deref(), Some("#2f6fd6"));
    let g = text::plan("a wooden green chair", 0);
    assert_eq!(g.program.parts[0].color.as_deref(), Some("#4caf50"), "an explicit colour wins over the material's");
    assert_eq!(g.program.parts[0].roughness, 0.75);
}

#[test]
fn counts_rows_and_stacks() {
    let g = generate("three cubes", &opts()).unwrap();
    assert_eq!(g.parts.len(), 3);
    let xs: Vec<f32> = g.parts.iter().map(|p| p.pivot.x).collect();
    assert!(xs[0] < xs[1] && xs[1] < xs[2]);
    let g = generate("a stack of 4 cubes", &opts()).unwrap();
    let b = bounds(&g.parts);
    assert!((b.size().z - 160.0).abs() < 4.0, "stacked height {}", b.size().z);
    let g = generate("flowers", &opts()).unwrap();
    assert_eq!(g.parts.iter().filter(|p| p.name.ends_with("Petals")).count(), 3, "a bare plural makes three");
}

#[test]
fn relations() {
    let g = generate("a cat on top of a table", &opts()).unwrap();
    let table = part_bounds(&g, "Table");
    let cat = part_bounds(&g, "Cat");
    assert!(cat.min.z >= table.max.z - 3.0, "cat {:?} should sit on table {:?}", cat, table);
    assert!((cat.center().x - table.center().x).abs() < 40.0);

    let g = generate("a table under a lamp", &opts()).unwrap();
    assert!(part_bounds(&g, "Lamp").min.z >= part_bounds(&g, "Table").max.z - 3.0);

    let g = generate("a mug and a teapot", &opts()).unwrap();
    let (m, t) = (part_bounds(&g, "Mug"), part_bounds(&g, "Teapot"));
    assert!(m.max.x < t.min.x, "side by side without overlap");

    let g = generate("a dog with a red hat", &opts()).unwrap();
    let hat = part_bounds(&g, "Hat");
    let dog = part_bounds(&g, "Body");
    assert!(hat.min.z > dog.center().z, "the hat sits on top");
    assert!(g.parts.iter().any(|p| p.name == "Hat" && (p.material.base_color.r - 0.84).abs() < 0.02), "the accessory takes its colour");
}

#[test]
fn dimensions_and_sizes() {
    let g = generate("a cube 100 mm", &opts()).unwrap();
    let s = bounds(&g.parts).size();
    assert!((s.x - 100.0).abs() < 3.0 && (s.z - 100.0).abs() < 3.0, "{s:?}");
    let g = generate("a vase 5 cm tall", &opts()).unwrap();
    assert!((bounds(&g.parts).size().z - 50.0).abs() < 2.0);
    let small = bounds(&generate("a small sphere", &opts()).unwrap().parts).size().x;
    let big = bounds(&generate("a huge sphere", &opts()).unwrap().parts).size().x;
    assert!(big > small * 3.0);
    let tall = bounds(&generate("a tall cylinder", &opts()).unwrap().parts).size();
    assert!(tall.z > tall.x * 1.5);
}

#[test]
fn unknown_words_fall_back() {
    let g = generate("a wombat", &opts()).unwrap();
    assert!(g.understood && g.notes[0].contains("wombat") && g.notes[0].contains("like an animal"), "{:?}", g.notes);
    let g = generate("a zorblax", &opts()).unwrap();
    assert!(!g.understood && g.notes[0].contains("zorblax"));
    assert_eq!(g.parts.len(), 1);
    assert_eq!(g.parts[0].name, "Zorblax", "a single part is named after the object");
    let g = generate("", &opts()).unwrap();
    assert!(!g.understood);
}

#[test]
fn sphere_is_accurate() {
    let p = Program { version: 1, name: "s".into(), real_scale: None, parts: vec![Part::new("S", "#ffffff", sphere(25.0))] };
    let parts = mesh_program(&p, &GenOptions { detail: 24, ..opts() }).unwrap();
    let m = &parts[0];
    let worst = m.mesh.positions.iter().map(|v| ((*v + m.pivot).length() - 25.0).abs()).fold(0.0f32, f32::max);
    assert!(worst < 0.25, "vertices within 1 % of the surface, worst {worst}");
}

#[test]
fn budget_and_triangles() {
    let g = generate("a castle", &GenOptions { max_triangles: 2_000, ..GenOptions::default() }).unwrap();
    assert!(g.triangles() <= 2_000);
    let g = generate("a duck", &GenOptions { quads: false, ..opts() }).unwrap();
    assert!(g.parts.iter().all(|p| p.mesh.faces.iter().all(|f| f.len() == 3) && closed(&p.mesh)));
}

#[test]
fn shapes_that_cancel_are_an_error() {
    let p = Program { version: 1, name: "x".into(), real_scale: None, parts: vec![Part::new("X", "#fff", difference(0.0, vec![sphere(10.0), sphere(20.0)]))] };
    assert!(mesh_program(&p, &opts()).is_err());
}

#[test]
fn program_json_round_trip() {
    let plan = text::plan("a snowman next to a pine tree", 0);
    let back = Program::from_json(&plan.program.to_json()).unwrap();
    assert_eq!(plan.program, back);
}

#[test]
fn llm_replies() {
    // The examples in the system prompt are valid programs.
    for chunk in llm::SYSTEM_PROMPT.split("Example,").skip(1) {
        let p = llm::extract_program(chunk).expect("prompt example parses");
        assert!(mesh_program(&p, &opts()).is_ok());
    }
    let fenced = "Sure! Here it is:\n```json\n{\"version\":1,\"name\":\"ball\",\"parts\":[{\"name\":\"Ball\",\"color\":\"#ff0000\",\"shape\":{\"type\":\"sphere\",\"r\":20,\"at\":[0,0,20]}}]}\n```\nEnjoy.";
    assert_eq!(llm::extract_program(fenced).unwrap().parts[0].name, "Ball");
    let bare = r#"{"type":"box","size":[10,10,10]}"#;
    assert_eq!(llm::extract_program(bare).unwrap().parts.len(), 1);
    let part = r##"{"name":"Cup","color":"#00ff00","shape":{"type":"cylinder","r":10,"h":20}}"##;
    assert_eq!(llm::extract_program(part).unwrap().parts[0].name, "Cup");
    for bad in [
        "no json here",
        r#"{"version":1,"parts":[]}"#,
        r#"{"version":1,"parts":[{"shape":{"type":"sphere","r":-4}}]}"#,
        r#"{"version":1,"parts":[{"shape":{"type":"sphere","r":"big"}}]}"#,
        r#"{"version":1,"parts":[{"shape":{"type":"array","count":5000,"offset":[1,0,0],"child":{"type":"sphere","r":1}}}]}"#,
        r#"{"version":1,"parts":[{"shape":{"type":"teapot"}}]}"#,
        r#"{"version":9,"parts":[{"shape":{"type":"sphere","r":4}}]}"#,
        r#"{"version":1,"parts":[{"shape":{"type":"union","children":[]}}]}"#,
    ] {
        assert!(llm::extract_program(bad).is_err(), "accepted {bad}");
    }
    let deep = (0..40).fold(r#"{"type":"sphere","r":1}"#.to_string(), |acc, _| format!(r#"{{"type":"union","children":[{acc}]}}"#));
    assert!(llm::extract_program(&deep).is_err(), "nesting is limited");
    let schema = llm::schema();
    assert!(schema["$defs"]["node"]["properties"]["type"]["enum"].as_array().unwrap().len() == 14);
}

#[test]
fn host_json() {
    let v: serde_json::Value = serde_json::from_str(&host::generate_json("a red apple", r#"{"detail":30}"#)).unwrap();
    assert_eq!(v["ok"], true);
    let part = &v["parts"][0];
    assert!(part["mesh"]["positions"].as_array().unwrap().len() % 3 == 0);
    assert_eq!(part["pivot"].as_array().unwrap().len(), 3);
    let v: serde_json::Value = serde_json::from_str(&host::mesh_program_json("garbage", "{}")).unwrap();
    assert_eq!(v["ok"], false);
    let v: serde_json::Value = serde_json::from_str(&host::plan_json("a boat", 0)).unwrap();
    assert_eq!(v["program"]["realScale"], 50.0);
}

#[test]
fn primitives_bounds_contain_surface() {
    // The compiled bounds must contain the mesh, or the mesher would clip shapes.
    let shapes = vec![
        cone(20.0, 5.0, 30.0).rot(30.0, 10.0, 0.0),
        torus(20.0, 5.0).rot(90.0, 0.0, 0.0).at(10.0, 0.0, 30.0),
        pyramid(40.0, 30.0),
        lathe(&[(0.0, 0.0), (20.0, 0.0), (10.0, 40.0), (0.0, 40.0)]),
        extrude(&[(0.0, 0.0), (30.0, 0.0), (0.0, 20.0)], 10.0).rot(0.0, 45.0, 0.0),
        capsule([0.0, 0.0, 0.0], [30.0, 10.0, 20.0], 6.0).mirror(Axis::X),
        array(rbox(10.0, 10.0, 10.0, 2.0), 4, [12.0, 0.0, 3.0]).stretched(1.0, 2.0, 1.0),
        ellipsoid(30.0, 20.0, 10.0).hollow(2.0),
    ];
    for s in shapes {
        let c = Compiled::new(&s);
        let p = Program { version: 1, name: "t".into(), real_scale: None, parts: vec![Part::new("T", "#fff", s)] };
        let parts = mesh_program(&p, &opts()).unwrap();
        let b = bounds(&parts);
        assert!(closed(&parts[0].mesh));
        assert!(b.min.cmpge(c.aabb.min - 0.5).all() && b.max.cmple(c.aabb.max + 0.5).all(), "{b:?} outside {:?}", c.aabb);
    }
}

#[test]
#[ignore]
fn diag_manifold() {
    for w in ["a lamp", "a crown", "a trophy", "a bucket", "a vase"] {
        let plan = text::plan(w, 0);
        for part in &plan.program.parts {
            let c = Compiled::new(&part.shape);
            for h in (3..40).map(|i| i as f32 * 0.1) {
                let grid = mesher::Grid::covering(&c.aabb, h);
                let raw = mesher::surface_nets(&c, &grid);
                let mut fixed = raw.clone();
                mesher::make_manifold(&mut fixed);
                let stats = |m: &crate::format::Mesh| {
                    let mut e: HashMap<(u32, u32), i32> = HashMap::new();
                    for f in &m.faces { for k in 0..f.len() { *e.entry((f[k], f[(k + 1) % f.len()])).or_default() += 1; } }
                    let dup = e.values().filter(|&&n| n > 1).count();
                    let lonely = e.keys().filter(|(a, b)| !e.contains_key(&(*b, *a))).count();
                    (dup, lonely)
                };
                println!("{w} {} h={h}: raw {:?} fixed {:?}", part.name, stats(&raw), stats(&fixed));
            }
        }
    }
}
