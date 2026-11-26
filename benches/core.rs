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

    let mut img_splat = c.benchmark_group("img3d_and_splat");
    img_splat.sample_size(10);
    let mut sample_rgba = vec![240u8; 128 * 128 * 4];
    for y in 0..128usize {
        for x in 0..128usize {
            let dx = x as f32 - 64.0;
            let dy = y as f32 - 64.0;
            if dx * dx + dy * dy < 44.0 * 44.0 {
                let i = (y * 128 + x) * 4;
                sample_rgba[i] = 210;
                sample_rgba[i + 1] = 65;
                sample_rgba[i + 2] = 48;
                sample_rgba[i + 3] = 255;
            }
        }
    }
    let img_opts = sculpt_engine::img3d::Image3DOptions::default();
    img_splat.bench_function("image_to_3d_128", |b| {
        b.iter(|| sculpt_engine::img3d::build_image_part(128, 128, &sample_rgba, None, &img_opts).unwrap().mesh.faces.len())
    });
    img_splat.bench_function("image_to_splat_space_12k", |b| {
        b.iter(|| sculpt_engine::splat::image_to_splat(128, 128, &sample_rgba, None, true, 12_000).unwrap().len())
    });
    // 128x128 with max_splats=16384 -> step=1 -> exactly 16,384 Gaussians (16,000 would pick step=2 -> 4,096).
    let cloud = sculpt_engine::splat::image_to_splat(128, 128, &sample_rgba, None, true, 16_384).unwrap();
    assert!(cloud.len() >= 16_000, "16k splat bench must really test ~16k Gaussians, got {}", cloud.len());
    println!("16k space cloud: {} Gaussians", cloud.len());
    let item8k = sculpt_engine::splat::image_to_splat(128, 128, &sample_rgba, None, false, 8_192).unwrap();
    println!("8k item cloud: {} Gaussians", item8k.len());
    img_splat.bench_function("image_to_splat_item_8k", |b| {
        b.iter(|| sculpt_engine::splat::image_to_splat(128, 128, &sample_rgba, None, false, 8_192).unwrap().len())
    });
    // Room mesh: floor + ceiling + 4 walls (12 triangles), 4 x 2.5 x 4 m.
    let (room_pos, room_idx) = {
        let (x, y, z) = (2.0f32, 2.5f32, 2.0f32);
        let pos: Vec<f32> = vec![
            -x, 0.0, -z, x, 0.0, -z, x, 0.0, z, -x, 0.0, z, // floor 0..3
            -x, y, -z, x, y, -z, x, y, z, -x, y, z, // ceiling 4..7
        ];
        let idx: Vec<u32> = vec![
            0, 2, 1, 0, 3, 2, // floor
            4, 5, 6, 4, 6, 7, // ceiling
            0, 1, 5, 0, 5, 4, // back
            3, 6, 2, 3, 7, 6, // front
            0, 4, 7, 0, 7, 3, // left
            1, 2, 6, 1, 6, 5, // right
        ];
        (pos, idx)
    };
    let room16k = sculpt_engine::splat::mesh_to_splat(&room_pos, &room_idx, None, [0.8, 0.8, 0.8, 1.0], true, 16_384).unwrap();
    println!("16k room cloud: {} Gaussians", room16k.len());
    img_splat.bench_function("mesh_to_splat_room_16k", |b| {
        b.iter(|| sculpt_engine::splat::mesh_to_splat(&room_pos, &room_idx, None, [0.8, 0.8, 0.8, 1.0], true, 16_384).unwrap().len())
    });
    let ply = cloud.to_ply_bytes();
    let splat_raw = cloud.to_splat_bytes();
    img_splat.bench_function("splat_prompt_edit_16k", |b| {
        b.iter(|| sculpt_engine::splat::edit_splat_by_prompt(&cloud, "make the floor wood and the walls cyberpunk").total_count)
    });
    img_splat.bench_function("splat_radix_sort_16k", |b| {
        b.iter(|| sculpt_engine::splat::sort_splats_by_camera(&cloud, [0.0, 1.5, 2.0], [0.0, 0.0, -1.0]).len())
    });
    img_splat.bench_function("splat_ply_roundtrip_16k", |b| {
        b.iter(|| sculpt_engine::splat::SplatCloud::from_ply_bytes(&ply).unwrap().to_ply_bytes().len())
    });
    let compact = cloud.to_compact_bytes();
    println!("16k splat sizes: PLY={} KB, SPLAT={} KB, DSPLAT={} KB", ply.len() / 1024, splat_raw.len() / 1024, compact.len() / 1024);
    img_splat.bench_function("splat_decode_16k", |b| {
        b.iter(|| sculpt_engine::splat::SplatCloud::from_splat_bytes(&splat_raw).unwrap().len())
    });
    img_splat.bench_function("splat_compact_roundtrip_16k", |b| {
        b.iter(|| sculpt_engine::splat::SplatCloud::from_compact_bytes(&compact).unwrap().to_compact_bytes().len())
    });
    img_splat.finish();

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


// ---------------------------------------------------------------------------------------------
// vs_industry_standard: sculpt-engine head to head with clean, idiomatic reference implementations
// written inline here (no extra dependencies), on identical input, on this machine. Every pair
// asserts that both sides produce a valid result of equivalent content before it is timed.
// Bench ids: `<name>/sculpt` vs `<name>/ref_*` (several references per name where useful).
// ---------------------------------------------------------------------------------------------
mod industry {
    use sculpt_engine::glam::{Vec2, Vec3};
    use sculpt_engine::splat::{Gaussian, SplatCloud};

    pub const EYE: [f32; 3] = [0.0, 1.5, 2.0];
    pub const DIR: [f32; 3] = [0.0, 0.0, -1.0];

    #[inline]
    pub fn depth_of(g: &Gaussian) -> f32 {
        (g.pos[0] - EYE[0]) * DIR[0] + (g.pos[1] - EYE[1]) * DIR[1] + (g.pos[2] - EYE[2]) * DIR[2]
    }

    // ---- 1. depth sort ----------------------------------------------------------------------

    /// REFERENCE (a): what most renderers/engines do. Pair `(depth, index)`, `sort_unstable_by` with
    /// `partial_cmp`, far (largest depth) first. Exact comparison sort, O(n log n).
    pub fn sort_pairs(c: &SplatCloud) -> Vec<u32> {
        let mut v: Vec<(f32, u32)> = c.gaussians.iter().enumerate().map(|(i, g)| (depth_of(g), i as u32)).collect();
        v.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        v.into_iter().map(|(_, i)| i).collect()
    }

    /// REFERENCE (b): antimatter15/splat style 16-bit counting sort: quantise depth between min and max
    /// into 65536 buckets, histogram, prefix sum, single scatter. Output far (bucket 0) first.
    pub fn sort_counting16(c: &SplatCloud) -> Vec<u32> {
        let n = c.gaussians.len();
        let depths: Vec<f32> = c.gaussians.iter().map(depth_of).collect();
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &d in &depths {
            lo = lo.min(d);
            hi = hi.max(d);
        }
        let inv = 65535.0 / (hi - lo).max(1e-5);
        let keys: Vec<u16> = depths.iter().map(|&d| 65535 - (((d - lo) * inv) as u32).min(65535) as u16).collect();
        let mut starts = vec![0u32; 65536];
        for &k in &keys {
            starts[k as usize] += 1;
        }
        let mut acc = 0u32;
        for s in starts.iter_mut() {
            let c = *s;
            *s = acc;
            acc += c;
        }
        let mut out = vec![0u32; n];
        for (i, &k) in keys.iter().enumerate() {
            let slot = &mut starts[k as usize];
            out[*slot as usize] = i as u32;
            *slot += 1;
        }
        out
    }

    /// Valid permutation, ordered far to near up to `tol` (the quantisation step of the sorter).
    pub fn check_order(c: &SplatCloud, order: &[u32], tol: f32, what: &str) {
        let n = c.gaussians.len();
        assert_eq!(order.len(), n, "{what}: length");
        let mut seen = vec![false; n];
        for &i in order {
            assert!(!std::mem::replace(&mut seen[i as usize], true), "{what}: duplicate index");
        }
        let d: Vec<f32> = c.gaussians.iter().map(depth_of).collect();
        for w in order.windows(2) {
            assert!(d[w[0] as usize] >= d[w[1] as usize] - tol, "{what}: not back to front");
        }
    }

    pub fn depth_span(c: &SplatCloud) -> f32 {
        let d: Vec<f32> = c.gaussians.iter().map(depth_of).collect();
        d.iter().cloned().fold(f32::MIN, f32::max) - d.iter().cloned().fold(f32::MAX, f32::min)
    }

    /// Synthetic cloud of `n` Gaussians: the real 16k cloud's attributes, positions scattered in a 4 m box.
    pub fn synthetic(base: &SplatCloud, n: usize) -> SplatCloud {
        let mut s = 0x9E37_79B9u32;
        let mut rnd = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0
        };
        let gaussians = (0..n)
            .map(|i| {
                let mut g = base.gaussians[i % base.gaussians.len()];
                g.pos = [rnd(), rnd(), rnd()];
                g
            })
            .collect();
        SplatCloud { gaussians }
    }

    // ---- 2. serialization -------------------------------------------------------------------

    /// REFERENCE: the standard antimatter15 32-byte `.splat` writer (pos f32x3, scale f32x3, rgba u8x4,
    /// rot u8x4 = q/|q|*128+128), naive `Vec` + `extend_from_slice`.
    pub fn write_splat(c: &SplatCloud) -> Vec<u8> {
        let mut out = Vec::with_capacity(c.gaussians.len() * 32);
        for g in &c.gaussians {
            for v in g.pos {
                out.extend_from_slice(&v.to_le_bytes());
            }
            for v in g.scale {
                out.extend_from_slice(&v.max(1e-5).to_le_bytes());
            }
            for v in g.color {
                out.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
            out.push((g.opacity.clamp(0.0, 1.0) * 255.0).round() as u8);
            let n = g.rot.iter().map(|q| q * q).sum::<f32>().sqrt().max(1e-9);
            for q in g.rot {
                out.push((q / n * 128.0 + 128.0).round().clamp(0.0, 255.0) as u8);
            }
        }
        out
    }

    /// REFERENCE: naive `.splat` decoder (per-field `from_le_bytes`, collect into `Vec<Gaussian>`).
    pub fn read_splat(b: &[u8]) -> Vec<Gaussian> {
        b.chunks_exact(32)
            .map(|r| {
                let f = |o: usize| f32::from_le_bytes(r[o..o + 4].try_into().unwrap());
                let rot = [0, 1, 2, 3].map(|k| (r[28 + k] as f32 - 128.0) / 128.0);
                let n = rot.iter().map(|q| q * q).sum::<f32>().sqrt().max(1e-9);
                Gaussian {
                    pos: [f(0), f(4), f(8)],
                    scale: [f(12), f(16), f(20)],
                    rot: rot.map(|q| q / n),
                    color: [r[24] as f32 / 255.0, r[25] as f32 / 255.0, r[26] as f32 / 255.0],
                    opacity: r[27] as f32 / 255.0,
                }
            })
            .collect()
    }

    /// REFERENCE: standard 3DGS binary PLY, 17 f32 per vertex = 68 B (xyz, nxyz, f_dc*3, opacity logit,
    /// log scale*3, rot wxyz), libm `ln` for the transforms, naive per-float `extend_from_slice`.
    pub fn write_ply(c: &SplatCloud) -> Vec<u8> {
        const SH_C0: f32 = 0.282_094_8;
        let mut out = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\n\
             property float nx\nproperty float ny\nproperty float nz\nproperty float f_dc_0\nproperty float f_dc_1\n\
             property float f_dc_2\nproperty float opacity\nproperty float scale_0\nproperty float scale_1\n\
             property float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n\
             end_header\n",
            c.gaussians.len()
        )
        .into_bytes();
        out.reserve(c.gaussians.len() * 68);
        for g in &c.gaussians {
            let o = g.opacity.clamp(1e-5, 1.0 - 1e-5);
            let vals: [f32; 17] = [
                g.pos[0], g.pos[1], g.pos[2], 0.0, 0.0, 0.0,
                (g.color[0] - 0.5) / SH_C0, (g.color[1] - 0.5) / SH_C0, (g.color[2] - 0.5) / SH_C0,
                (o / (1.0 - o)).ln(),
                g.scale[0].max(1e-5).ln(), g.scale[1].max(1e-5).ln(), g.scale[2].max(1e-5).ln(),
                g.rot[0], g.rot[1], g.rot[2], g.rot[3],
            ];
            for v in vals {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    // ---- 3. meshing -------------------------------------------------------------------------

    fn sd_cyl(p: Vec3, r: f32, h: f32) -> f32 {
        let d = Vec2::new(Vec2::new(p.x, p.z).length() - r, p.y.abs() - h);
        d.x.max(d.y).min(0.0) + d.max(Vec2::ZERO).length()
    }

    /// Closed-form coffee cup: hollow cylinder (outer minus inner capped cylinder) union a torus handle.
    pub fn sd_cup(p: Vec3) -> f32 {
        let body = sd_cyl(p, 0.45, 0.4).max(-sd_cyl(p - Vec3::new(0.0, 0.1, 0.0), 0.38, 0.4));
        let q = Vec2::new(Vec2::new(p.x - 0.55, p.y).length() - 0.22, p.z);
        body.min(q.length() - 0.05)
    }

    /// REFERENCE: textbook naive Surface Nets (Lysenko): evaluate the SDF at EVERY sample of an n^3 grid
    /// over [-1,1]^3, one vertex per sign-crossing cell (mean of edge crossings), one quad per
    /// sign-crossing grid edge. Returns (verts, quads, samples evaluated, active cells).
    pub fn naive_surface_nets(n: usize, sdf: impl Fn(Vec3) -> f32) -> (Vec<[f32; 3]>, Vec<[u32; 4]>, usize, usize) {
        let h = 2.0 / (n - 1) as f32;
        let pos = |x: usize, y: usize, z: usize| Vec3::new(x as f32, y as f32, z as f32) * h - Vec3::ONE;
        let at = |x: usize, y: usize, z: usize| x + n * (y + n * z);
        let mut field = vec![0.0f32; n * n * n];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    field[at(x, y, z)] = sdf(pos(x, y, z));
                }
            }
        }
        let m = n - 1;
        let cat = |x: usize, y: usize, z: usize| x + m * (y + m * z);
        let mut cell_vert = vec![u32::MAX; m * m * m];
        let mut verts: Vec<[f32; 3]> = Vec::new();
        for z in 0..m {
            for y in 0..m {
                for x in 0..m {
                    let c: [f32; 8] = std::array::from_fn(|i| field[at(x + (i & 1), y + ((i >> 1) & 1), z + (i >> 2))]);
                    let neg = c.iter().filter(|&&v| v < 0.0).count();
                    if neg == 0 || neg == 8 {
                        continue;
                    }
                    let (mut sum, mut cnt) = (Vec3::ZERO, 0.0);
                    for i in 0..8usize {
                        for bit in 0..3 {
                            let j = i | (1 << bit);
                            if j == i || (c[i] < 0.0) == (c[j] < 0.0) {
                                continue;
                            }
                            let t = c[i] / (c[i] - c[j]);
                            let a = Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, (i >> 2) as f32);
                            let b = Vec3::new((j & 1) as f32, ((j >> 1) & 1) as f32, (j >> 2) as f32);
                            sum += a + (b - a) * t;
                            cnt += 1.0;
                        }
                    }
                    cell_vert[cat(x, y, z)] = verts.len() as u32;
                    verts.push((pos(x, y, z) + sum / cnt * h).to_array());
                }
            }
        }
        let active = verts.len();
        let mut quads = Vec::new();
        for z in 1..n - 1 {
            for y in 1..n - 1 {
                for x in 1..n - 1 {
                    let here = field[at(x, y, z)] < 0.0;
                    // Edges leaving (x,y,z) along +x, +y, +z; each is shared by the four cells around it.
                    let edges = [
                        (field[at(x + 1, y, z)] < 0.0, [(x, y - 1, z - 1), (x, y, z - 1), (x, y, z), (x, y - 1, z)]),
                        (field[at(x, y + 1, z)] < 0.0, [(x - 1, y, z - 1), (x - 1, y, z), (x, y, z), (x, y, z - 1)]),
                        (field[at(x, y, z + 1)] < 0.0, [(x - 1, y - 1, z), (x, y - 1, z), (x, y, z), (x - 1, y, z)]),
                    ];
                    for (other, cells) in edges {
                        if other == here {
                            continue;
                        }
                        let mut q = cells.map(|(cx, cy, cz)| cell_vert[cat(cx, cy, cz)]);
                        if q.contains(&u32::MAX) {
                            continue;
                        }
                        if here {
                            q.reverse();
                        }
                        quads.push(q);
                    }
                }
            }
        }
        (verts, quads, n * n * n, active)
    }

    // ---- 4. BVH -----------------------------------------------------------------------------

    pub struct RefNode {
        pub min: Vec3,
        pub max: Vec3,
        /// Interior: child node indices (count 0). Leaf: tri count, children both u32::MAX.
        pub left: u32,
        pub right: u32,
        pub count: u32,
    }

    /// REFERENCE: textbook top-down median-split BVH, recursive, leaf size 4 (same as sculpt), split on
    /// the longest centroid axis at the median. `full_sort` = sort the whole range by centroid at every
    /// level (`sort_unstable_by`, O(n log^2 n)); otherwise `select_nth_unstable_by` (the better O(n log n)
    /// classic nth_element variant).
    pub fn median_bvh(positions: &[Vec3], tris: &[[u32; 3]], full_sort: bool) -> (Vec<RefNode>, Vec<u32>) {
        let lo: Vec<Vec3> = tris.iter().map(|t| t.iter().map(|&i| positions[i as usize]).fold(Vec3::splat(f32::MAX), Vec3::min)).collect();
        let hi: Vec<Vec3> = tris.iter().map(|t| t.iter().map(|&i| positions[i as usize]).fold(Vec3::splat(f32::MIN), Vec3::max)).collect();
        let cen: Vec<Vec3> = lo.iter().zip(&hi).map(|(a, b)| (*a + *b) * 0.5).collect();
        struct Ctx<'a> {
            lo: &'a [Vec3],
            hi: &'a [Vec3],
            cen: &'a [Vec3],
            full_sort: bool,
            nodes: Vec<RefNode>,
        }
        fn rec(cx: &mut Ctx, ids: &mut [u32]) -> u32 {
            let (mut bmin, mut bmax) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            let (mut cmin, mut cmax) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for &i in ids.iter() {
                bmin = bmin.min(cx.lo[i as usize]);
                bmax = bmax.max(cx.hi[i as usize]);
                cmin = cmin.min(cx.cen[i as usize]);
                cmax = cmax.max(cx.cen[i as usize]);
            }
            let me = cx.nodes.len() as u32;
            cx.nodes.push(RefNode { min: bmin, max: bmax, left: u32::MAX, right: u32::MAX, count: ids.len() as u32 });
            if ids.len() <= 4 {
                return me;
            }
            let e = cmax - cmin;
            let axis = if e.x >= e.y && e.x >= e.z { 0 } else if e.y >= e.z { 1 } else { 2 };
            let cen = cx.cen;
            let cmp = |a: &u32, b: &u32| cen[*a as usize][axis].partial_cmp(&cen[*b as usize][axis]).unwrap();
            let mid = ids.len() / 2;
            if cx.full_sort {
                ids.sort_unstable_by(cmp);
            } else {
                ids.select_nth_unstable_by(mid, cmp);
            }
            let (l, r) = ids.split_at_mut(mid);
            let li = rec(cx, l);
            let ri = rec(cx, r);
            let node = &mut cx.nodes[me as usize];
            node.left = li;
            node.right = ri;
            node.count = 0;
            me
        }
        let mut ids: Vec<u32> = (0..tris.len() as u32).collect();
        let mut cx = Ctx { lo: &lo, hi: &hi, cen: &cen, full_sort, nodes: Vec::with_capacity(tris.len() / 2) };
        rec(&mut cx, &mut ids);
        (cx.nodes, ids)
    }

    // ---- 4b. project JSON -------------------------------------------------------------------

    #[derive(serde::Serialize)]
    pub struct RefMesh<'a> {
        pub faces: &'a [Vec<u32>],
        pub positions: &'a [f32],
        #[serde(rename = "smoothShading")]
        pub smooth_shading: bool,
    }
    #[derive(serde::Serialize)]
    pub struct RefObject<'a> {
        pub id: &'a str,
        pub name: &'a str,
        pub mesh: RefMesh<'a>,
    }
    #[derive(serde::Serialize)]
    pub struct RefScene<'a> {
        pub objects: Vec<RefObject<'a>>,
    }
    #[derive(serde::Serialize)]
    pub struct RefProject<'a> {
        #[serde(rename = "formatVersion")]
        pub format_version: u32,
        pub scene: RefScene<'a>,
    }

    #[derive(serde::Deserialize)]
    pub struct DMesh {
        pub faces: Vec<Vec<u32>>,
        pub positions: Vec<f32>,
    }
    #[derive(serde::Deserialize)]
    pub struct DObject {
        pub mesh: DMesh,
    }
    #[derive(serde::Deserialize)]
    pub struct DScene {
        pub objects: Vec<DObject>,
    }
    #[derive(serde::Deserialize)]
    pub struct DProject {
        pub scene: DScene,
    }
}

fn vs_industry_standard(c: &mut Criterion) {
    use industry::*;
    use sculpt_engine::splat::SplatCloud;
    use std::hint::black_box;
    use std::time::Duration;

    let mut g = c.benchmark_group("vs_industry_standard");
    g.measurement_time(Duration::from_secs(3));
    g.warm_up_time(Duration::from_secs(1));
    g.sample_size(20);

    let mut rgba = vec![240u8; 128 * 128 * 4];
    for y in 0..128usize {
        for x in 0..128usize {
            let (dx, dy) = (x as f32 - 64.0, y as f32 - 64.0);
            if dx * dx + dy * dy < 44.0 * 44.0 {
                let i = (y * 128 + x) * 4;
                rgba[i..i + 4].copy_from_slice(&[210, 65, 48, 255]);
            }
        }
    }
    let cloud = sculpt_engine::splat::image_to_splat(128, 128, &rgba, None, true, 16_384).unwrap();
    assert_eq!(cloud.len(), 16_384);

    // ---- 1. depth sort (back to front from eye [0,1.5,2], dir [0,0,-1]) ----
    let big_cloud = synthetic(&cloud, 500_000);
    for (label, cl) in [("16k", &cloud), ("500k", &big_cloud)] {
        let span = depth_span(cl);
        // sculpt: 1024 buckets; counting16: 65536 buckets; pair sort: exact.
        check_order(cl, &sculpt_engine::splat::sort_splats_by_camera(cl, EYE, DIR), span / 1023.0 * 1.01 + 1e-6, "sculpt");
        check_order(cl, &sort_pairs(cl), 0.0, "pairs");
        check_order(cl, &sort_counting16(cl), span / 65535.0 * 1.01 + 1e-6, "counting16");
        if label == "500k" {
            g.sample_size(10);
        }
        let name = format!("depth_sort_{label}");
        g.bench_function(format!("{name}/sculpt"), |b| b.iter(|| black_box(sculpt_engine::splat::sort_splats_by_camera(black_box(cl), EYE, DIR))));
        g.bench_function(format!("{name}/ref_pairs_sort_unstable_by"), |b| b.iter(|| black_box(sort_pairs(black_box(cl)))));
        g.bench_function(format!("{name}/ref_counting16"), |b| b.iter(|| black_box(sort_counting16(black_box(cl)))));
    }
    g.sample_size(20);

    // ---- 2. serialization of the 16k cloud ----
    let n = cloud.len();
    let compact = cloud.to_compact_bytes();
    let splat_ref = write_splat(&cloud);
    let splat_crate = cloud.to_splat_bytes();
    let ply_ref = write_ply(&cloud);
    let ply_crate = cloud.to_ply_bytes();
    // Equivalence: reference .splat matches the crate's .splat (floats exact, 8-bit fields within 1 LSB);
    // reference PLY body matches the crate's PLY body (fast_ln/logit vs libm within 1e-4); compact decodes
    // to the same cloud within its 12-bit position quantisation.
    assert_eq!(splat_ref.len(), splat_crate.len());
    for (a, b) in splat_ref.chunks_exact(32).zip(splat_crate.chunks_exact(32)) {
        assert_eq!(a[..24], b[..24]);
        assert!(a[24..].iter().zip(&b[24..]).all(|(x, y)| x.abs_diff(*y) <= 1), "splat 8-bit fields differ by >1");
    }
    assert_eq!(ply_ref.len() - n * 68, ply_ref.windows(11).position(|w| w == b"end_header\n").unwrap() + 11);
    let (pb_ref, pb_crate) = (&ply_ref[ply_ref.len() - n * 68..], &ply_crate[ply_crate.len() - n * 68..]);
    for (a, b) in pb_ref.chunks_exact(4).zip(pb_crate.chunks_exact(4)) {
        let (a, b) = (f32::from_le_bytes(a.try_into().unwrap()), f32::from_le_bytes(b.try_into().unwrap()));
        assert!((a - b).abs() <= 1e-4 * (1.0 + a.abs()), "ply field {a} vs {b}");
    }
    let (lo, hi) = cloud.bounds();
    let pos_tol = (0..3).map(|k| (hi[k] - lo[k]) / 4095.0).fold(0.0f32, f32::max);
    let from_compact = SplatCloud::from_compact_bytes(&compact).unwrap();
    let from_ref_splat = read_splat(&splat_ref);
    let from_ref_ply = SplatCloud::from_ply_bytes(&ply_ref).unwrap();
    assert!(from_compact.len() == n && from_ref_splat.len() == n && from_ref_ply.len() == n);
    for i in 0..n {
        let o = &cloud.gaussians[i];
        for k in 0..3 {
            assert!((from_compact.gaussians[i].pos[k] - o.pos[k]).abs() <= pos_tol, "compact pos");
            assert_eq!(from_ref_splat[i].pos[k], o.pos[k]);
            assert_eq!(from_ref_ply.gaussians[i].pos[k], o.pos[k]);
        }
        assert!((from_ref_splat[i].opacity - o.opacity).abs() <= 1.0 / 255.0);
        assert!((from_ref_ply.gaussians[i].opacity - o.opacity).abs() <= 1e-3);
    }
    let bps = |len: usize| len as f64 / n as f64;
    println!(
        "serialized size, 16k cloud (B/splat): dsplat-v2 {:.2} | .splat {:.2} | PLY {:.2}  => dsplat is {:.2}x smaller than .splat, {:.2}x smaller than PLY",
        bps(compact.len()), bps(splat_ref.len()), bps(ply_ref.len()),
        splat_ref.len() as f64 / compact.len() as f64, ply_ref.len() as f64 / compact.len() as f64
    );
    g.bench_function("encode_16k/sculpt_to_compact_bytes_12B", |b| b.iter(|| black_box(cloud.to_compact_bytes()).len()));
    g.bench_function("encode_16k/ref_splat_32B_inline", |b| b.iter(|| black_box(write_splat(black_box(&cloud))).len()));
    g.bench_function("encode_16k/ref_splat_32B_crate_to_splat_bytes", |b| b.iter(|| black_box(cloud.to_splat_bytes()).len()));
    g.bench_function("encode_16k/ref_ply_68B_inline", |b| b.iter(|| black_box(write_ply(black_box(&cloud))).len()));
    g.bench_function("decode_16k/sculpt_from_compact_bytes", |b| b.iter(|| black_box(SplatCloud::from_compact_bytes(black_box(&compact)).unwrap()).len()));
    g.bench_function("decode_16k/ref_splat_naive_inline", |b| b.iter(|| black_box(read_splat(black_box(&splat_ref))).len()));
    g.bench_function("decode_16k/ref_splat_crate_from_splat_bytes", |b| b.iter(|| black_box(SplatCloud::from_splat_bytes(black_box(&splat_ref)).unwrap()).len()));
    g.bench_function("decode_16k/ref_ply_crate_from_ply_bytes", |b| b.iter(|| black_box(SplatCloud::from_ply_bytes(black_box(&ply_ref)).unwrap()).len()));

    // ---- 3. meshing ----
    // CONTENT DIFFERS: sculpt runs a whole text -> multi-part shape program (planner, templates, narrow-band
    // Surface Nets, quad conversion, 8k triangle budget); the reference only evaluates one closed-form cup SDF
    // on the full n^3 grid and runs naive Surface Nets. sculpt's own cell counts are not exposed by the API,
    // so only the reference's grid samples / active cells are reported.
    g.sample_size(10);
    for detail in [64u32, 128] {
        let opts = sculpt_engine::text3d::GenOptions { max_triangles: 8_000, detail, seed: 0, quads: true };
        let gen_ = sculpt_engine::text3d::generate("a ceramic coffee cup", &opts).unwrap();
        let (verts, quads, evaluated, active) = naive_surface_nets(detail as usize, sd_cup);
        assert!(gen_.triangles() > 0 && !gen_.parts.is_empty(), "sculpt produced no mesh");
        assert!(!quads.is_empty() && verts.iter().flatten().all(|v| v.abs() <= 1.0 + 1e-4), "reference mesh invalid");
        assert!(quads.iter().flatten().all(|&i| (i as usize) < verts.len()));
        println!(
            "meshing detail {detail}: sculpt {} tris ({} parts) | reference {} samples evaluated, {} active cells, {} verts, {} quads",
            gen_.triangles(), gen_.parts.len(), evaluated, active, verts.len(), quads.len()
        );
        g.bench_function(format!("meshing_d{detail}/sculpt_text3d_cup"), |b| {
            b.iter(|| black_box(sculpt_engine::text3d::generate("a ceramic coffee cup", black_box(&opts)).unwrap().triangles()))
        });
        g.bench_function(format!("meshing_d{detail}/ref_naive_full_grid_surface_nets"), |b| {
            b.iter(|| black_box(naive_surface_nets(detail as usize, sd_cup)).1.len())
        });
    }

    // ---- 4. BVH + project format ----
    for (label, rings) in [("100k", 224u32), ("1m", 708)] {
        let m = sculpt_engine::mesh::uv_sphere(50.0, rings, rings);
        let s = SculptMesh::from_mesh(&m);
        let bvh = Bvh::build(&s.positions, &s.tris);
        let (ref_nodes, ref_ids) = median_bvh(&s.positions, &s.tris, false);
        let leaf_tris: u32 = ref_nodes.iter().map(|n| n.count).sum();
        assert_eq!(bvh.tri_ids.len(), s.tris.len());
        assert_eq!(ref_ids.len(), s.tris.len());
        assert_eq!(leaf_tris as usize, s.tris.len(), "reference leaves must cover every triangle once");
        let (r0, s0) = (&ref_nodes[0], &bvh.nodes[0]);
        assert!((r0.min - Vec3::from_array(s0.min)).abs().max_element() < 1e-2 && (r0.max - Vec3::from_array(s0.max)).abs().max_element() < 1e-2, "root boxes differ");
        println!("bvh {label}: {} tris, sculpt {} nodes | reference {} nodes", s.tris.len(), bvh.nodes.len(), ref_nodes.len());
        g.sample_size(10);
        g.bench_function(format!("bvh_build_{label}/sculpt_binned_sah"), |b| b.iter(|| black_box(Bvh::build(black_box(&s.positions), black_box(&s.tris))).nodes.len()));
        g.bench_function(format!("bvh_build_{label}/ref_median_split_nth_element"), |b| b.iter(|| black_box(median_bvh(&s.positions, &s.tris, false)).0.len()));
        g.bench_function(format!("bvh_build_{label}/ref_median_split_full_sort"), |b| b.iter(|| black_box(median_bvh(&s.positions, &s.tris, true)).0.len()));
    }

    // Project JSON on the 1M-triangle sphere. Reference = serde derive `Serialize` over the same mesh data
    // (flat positions built once outside the timed loop) through `serde_json::to_string_pretty`; the sculpt
    // encoder additionally writes id/name/transform/material/selection (negligible next to the mesh).
    let m1m = big();
    let project = ProjectFile {
        format_version: 1,
        scene: Scene { objects: vec![SceneObject { id: "0A1B2C3D-0000-4000-8000-000000000001".into(), name: "Big".into(), mesh: m1m.clone(), transform: Transform::default(), material: Material::default(), modifiers: vec![], parent: None, is_visible: true, is_locked: false, primitive_kind: None }] },
        selection: Selection::default(),
        camera_state: None,
        created_at: "2026-11-01T00:00:00Z".into(),
        modified_at: "2026-11-01T00:00:00Z".into(),
        app_version: "1.0".into(),
    };
    let flat: Vec<f32> = m1m.positions.iter().flat_map(|p| p.to_array()).collect();
    let ref_project = RefProject {
        format_version: 1,
        scene: RefScene { objects: vec![RefObject { id: "0A1B2C3D-0000-4000-8000-000000000001", name: "Big", mesh: RefMesh { faces: &m1m.faces, positions: &flat, smooth_shading: m1m.smooth_shading } }] },
    };
    let text = project.encode();
    let ref_text = serde_json::to_string_pretty(&ref_project).unwrap();
    let decoded = ProjectFile::decode(text.as_bytes()).unwrap();
    let typed: DProject = serde_json::from_str(&ref_text).unwrap();
    let ref_mesh = &typed.scene.objects[0].mesh;
    assert_eq!(decoded.scene.objects[0].mesh.positions.len() * 3, ref_mesh.positions.len());
    assert_eq!(decoded.scene.objects[0].mesh.faces, ref_mesh.faces);
    assert_eq!(ref_mesh.positions, flat);
    let value: serde_json::Value = serde_json::from_slice(text.as_bytes()).unwrap();
    assert_eq!(value["scene"]["objects"][0]["mesh"]["faces"].as_array().unwrap().len(), m1m.faces.len());
    println!("project json 1m: sculpt {} MB | serde_json pretty {} MB", text.len() >> 20, ref_text.len() >> 20);
    g.sample_size(10);
    g.bench_function("project_encode_1m/sculpt", |b| b.iter(|| black_box(project.encode()).len()));
    g.bench_function("project_encode_1m/ref_serde_json_to_string_pretty", |b| b.iter(|| black_box(serde_json::to_string_pretty(black_box(&ref_project)).unwrap()).len()));
    g.bench_function("project_decode_1m/sculpt", |b| b.iter(|| black_box(ProjectFile::decode(black_box(text.as_bytes())).unwrap())));
    g.bench_function("project_decode_1m/ref_serde_json_Value", |b| b.iter(|| black_box(serde_json::from_slice::<serde_json::Value>(black_box(text.as_bytes())).unwrap())));
    g.bench_function("project_decode_1m/ref_serde_json_typed_derive", |b| b.iter(|| black_box(serde_json::from_slice::<DProject>(black_box(text.as_bytes())).unwrap()).scene.objects.len()));
    g.finish();
}

criterion_group!(core, benches, vs_industry_standard);
criterion_main!(core);
