use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use sculpt_engine::brush::{dab_and_update_normals, BrushKind, BrushParams, Scratch};
use sculpt_engine::bvh::Bvh;
use sculpt_engine::format::{Mesh, ProjectFile, Scene, SceneObject, Selection, Transform, Material};
use sculpt_engine::glam::Vec3;
use sculpt_engine::mesh::{uv_sphere, SculptMesh};

/// ~1M triangles: 708 x 708 quads split in two by the fan.
fn big() -> Mesh {
    uv_sphere(50.0, 708, 708)
}

fn benches(c: &mut Criterion) {
    let m100k = uv_sphere(50.0, 224, 224);
    let m1m = big();
    let s1m = SculptMesh::from_mesh(&m1m);
    println!("1m mesh: {} tris, {} verts", s1m.tris.len(), s1m.positions.len());

    c.bench_function("sculpt_mesh_from_1m", |b| b.iter(|| SculptMesh::from_mesh(&m1m)));
    let s100k = SculptMesh::from_mesh(&m100k);
    c.bench_function("bvh_build/100k", |b| b.iter(|| Bvh::build(&s100k.positions, &s100k.tris)));
    c.bench_function("bvh_build/1m", |b| b.iter(|| Bvh::build(&s1m.positions, &s1m.tris)));
    let mut bvh = Bvh::build(&s1m.positions, &s1m.tris);
    c.bench_function("bvh_refit/1m", |b| b.iter(|| bvh.refit(&s1m.positions, &s1m.tris)));
    c.bench_function("ray_pick/1m", |b| {
        let mut i = 0u32;
        b.iter(|| {
            i = i.wrapping_add(1);
            let a = i as f32 * 0.37;
            let o = Vec3::new(a.cos(), a.sin(), (a * 0.1).sin()) * 200.0;
            bvh.ray(&s1m.positions, &s1m.tris, o, -o)
        })
    });

    // Footprints: radius 5 on a radius 50 sphere of 1M tris covers ~2.5k verts; radius 20 ~ 40k.
    for (name, radius) in [("small", 5.0f32), ("large", 20.0)] {
        for kind in [BrushKind::Draw, BrushKind::Inflate, BrushKind::Smooth, BrushKind::Flatten] {
            let p = BrushParams { kind, radius, strength: 0.5, symmetry: [false; 3], ..Default::default() };
            c.bench_function(&format!("dab/{kind:?}/{name}").to_lowercase(), |b| {
                b.iter_batched(
                    || {
                        let mut s = Scratch::new(s1m.positions.len());
                        s.poly_stamp.resize(s1m.poly_count());
                        (s1m.clone(), s, Vec::with_capacity(65536), Vec::with_capacity(65536))
                    },
                    |(mut m, mut s, mut moved, mut dirty)| {
                        dab_and_update_normals(&mut m, &bvh, &mut s, Vec3::new(50.0, 0.0, 0.0), &p, &mut moved, &mut dirty);
                        (m, s, moved, dirty) // dropped outside the timed part
                    },
                    BatchSize::LargeInput,
                )
            });
        }
    }

    let project = ProjectFile {
        format_version: 1,
        scene: Scene { objects: vec![SceneObject { id: "0A1B2C3D-0000-4000-8000-000000000001".into(), name: "Big".into(), mesh: m1m.clone(), transform: Transform::default(), material: Material::default(), modifiers: vec![], parent: None, is_visible: true, is_locked: false, primitive_kind: None }] },
        selection: Selection::default(),
        camera_state: None,
        created_at: "2026-11-01T00:00:00Z".into(),
        modified_at: "2026-11-01T00:00:00Z".into(),
        app_version: "1.0".into(),
    };
    let text = project.encode();
    println!("1m project: {} MB", text.len() >> 20);
    let mut g = c.benchmark_group("sculpt_format");
    g.sample_size(10);
    g.bench_function("encode_1m", |b| b.iter(|| project.encode().len()));
    g.bench_function("decode_1m", |b| b.iter(|| ProjectFile::decode(text.as_bytes()).unwrap()));
    g.finish();

    let mut t3d = c.benchmark_group("text3d");
    t3d.sample_size(10);
    let opts = sculpt_engine::text3d::GenOptions { max_triangles: 8_000, detail: 64, seed: 0, quads: true };
    t3d.bench_function("cup_8k", |b| b.iter(|| sculpt_engine::text3d::generate("a ceramic coffee cup", &opts).unwrap().triangles()));
    t3d.bench_function("chair_8k", |b| b.iter(|| sculpt_engine::text3d::generate("a wooden chair", &opts).unwrap().triangles()));
    t3d.bench_function("scene_cat_table", |b| b.iter(|| sculpt_engine::text3d::generate("a cat on a table", &opts).unwrap().triangles()));
    t3d.finish();

    #[cfg(feature = "render")]
    {
        let mut ibl = c.benchmark_group("ibl");
        ibl.sample_size(10);
        ibl.bench_function("prefilter_and_lut", |b| {
            b.iter(|| {
                let p = sculpt_engine::render::ibl::prefilter(&sculpt_engine::render::ibl::studio);
                let l = sculpt_engine::render::ibl::brdf_lut();
                (p.len(), l.len())
            })
        });
        ibl.finish();
    }
}

criterion_group!(core, benches);
criterion_main!(core);
