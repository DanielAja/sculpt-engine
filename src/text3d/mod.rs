//! Text to 3D: the basis of Sculptor's "Make it" and Dream Builder's summoning.
//!
//! ```text
//! words ──▶ planner (text.rs, offline, deterministic) ─┐
//! words ──▶ any LLM with llm::SYSTEM_PROMPT ───────────┼─▶ Program (JSON SDF/CSG) ─▶ surface nets ─▶ meshes
//! a person or a saved program ─────────────────────────┘
//! ```
//!
//! Everything here is pure Rust with no GPU, so it runs in the browser (wasm), on iOS and visionOS
//! (xcframework), on Android and in Tauri, and costs nothing to keep running. See docs/text-to-3d-design.md.

pub mod compose;
pub mod lexicon;
pub mod llm;
pub mod mesher;
pub mod program;
pub mod sdf;
mod templates;
pub mod text;

use crate::format::{Color, Material, Mesh};
use glam::Vec3;
use program::Program;
use sdf::{Aabb, Compiled};
use serde_json::{json, Value};
use std::cell::RefCell;

#[derive(Debug, Clone, PartialEq)]
pub struct GenOptions {
    /// Upper bound for the whole program's triangle count (Dream Builder uses 8,000 per object).
    pub max_triangles: usize,
    /// Cells along the program's longest side; more is finer.
    pub detail: u32,
    /// Variation: the same words with another seed give a different take ("Try another").
    pub seed: u32,
    /// Quads (Sculptor's n-gon faces, nicer to edit) or triangles (RealityKit, game engines).
    pub quads: bool,
}

impl Default for GenOptions {
    fn default() -> Self {
        GenOptions { max_triangles: 40_000, detail: 72, seed: 0, quads: true }
    }
}

impl GenOptions {
    pub fn from_json(text: &str) -> GenOptions {
        let mut o = GenOptions::default();
        if let Ok(v) = serde_json::from_str::<Value>(text) {
            if let Some(n) = v.get("maxTriangles").and_then(Value::as_u64) {
                o.max_triangles = (n as usize).clamp(200, 500_000);
            }
            if let Some(n) = v.get("detail").and_then(Value::as_u64) {
                o.detail = (n as u32).clamp(8, 256);
            }
            if let Some(n) = v.get("seed").and_then(Value::as_u64) {
                o.seed = n as u32;
            }
            if let Some(b) = v.get("quads").and_then(Value::as_bool) {
                o.quads = b;
            }
        }
        o
    }
}

/// One meshed part. `mesh` positions are relative to `pivot` (the part's bottom centre, in program space).
#[derive(Debug, Clone, PartialEq)]
pub struct GenPart {
    pub name: String,
    pub material: Material,
    pub pivot: Vec3,
    pub mesh: Mesh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Generated {
    pub program: Program,
    pub parts: Vec<GenPart>,
    /// False when the planner didn't recognise the words and fell back to a rough shape.
    pub understood: bool,
    pub notes: Vec<String>,
}

impl Generated {
    pub fn triangles(&self) -> usize {
        self.parts.iter().map(|p| mesher::triangle_count(&p.mesh)).sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenError(pub String);
impl std::fmt::Display for GenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for GenError {}

/// Plans `prompt` with the built-in planner and meshes the result.
pub fn generate(prompt: &str, opts: &GenOptions) -> Result<Generated, GenError> {
    let plan = text::plan(prompt, opts.seed);
    let parts = mesh_program(&plan.program, opts)?;
    Ok(Generated { program: plan.program, parts, understood: plan.understood, notes: plan.notes })
}

/// "#rrggbb" (or "#rgb") to Sculptor's sRGB colour.
pub fn parse_color(hex: &str) -> Option<Color> {
    let h = hex.trim().trim_start_matches('#');
    let h: String = if h.len() == 3 { h.chars().flat_map(|c| [c, c]).collect() } else { h.to_string() };
    if h.len() != 6 || !h.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let c = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).ok().map(|v| v as f32 / 255.0);
    Some(Color { r: c(0)?, g: c(2)?, b: c(4)?, a: 1.0 })
}

#[derive(Default)]
struct PredictScratch {
    bits: Vec<u64>,
    signs: Vec<u8>,
    active: Vec<bool>,
}

thread_local! {
    static PREDICT: RefCell<PredictScratch> = RefCell::new(PredictScratch::default());
}

/// Predicts the Surface Nets triangle count of `c` at cell size `h` using a coarse grid probe
/// accelerated by 8x8x8 -> 4x4x4 interval pruning, hoisted `eval_block4`, and `u64` popcnt edge counting.
fn predict_part_triangles(c: &Compiled, h: f32) -> f32 {
    use sdf::{BlockClass, F32x4};
    let full = mesher::Grid::covering(&c.aabb, h);
    let ratio = (full.samples() as f32 / 10_500.0).cbrt().clamp(1.6, 5.5);
    let h_p = h * ratio;
    let pg = mesher::Grid::covering(&c.aabb, h_p);
    let [nx, ny, nz] = pg.n;
    if nx < 2 || ny < 2 || nz < 2 {
        return 0.0;
    }
    const BLK: usize = 4;
    let (nbx, nby, nbz) = (nx.div_ceil(BLK), ny.div_ceil(BLK), nz.div_ceil(BLK));

    PREDICT.with(|cell| {
        let mut sc = cell.borrow_mut();
        if sc.active.len() < nbx {
            sc.active.resize(nbx, false);
        }
        let PredictScratch { bits, signs, active } = &mut *sc;
        let mut blk = [F32x4::splat(0.0); 16];

        if nx <= 64 {
            // Fast path: 1 u64 per X-row (typically < 3 KB total, L1-resident) + 8x8x4 -> 4x4x4 pruning + block eval + popcnt.
            let n_rows = ny * nz;
            bits.clear();
            bits.resize(n_rows, 0);
            let fill_slab_bits = |bk: usize, bits_slab: &mut [u64]| {
                let k0 = bk * BLK;
                let k1 = (k0 + BLK).min(nz);
                let zlo = pg.origin.z + k0 as f32 * h_p;
                let zhi = pg.origin.z + (k1 - 1) as f32 * h_p;
                let nsbx = nbx.div_ceil(2);
                let mut blk = [F32x4::splat(0.0); 16];
                let mut bj0 = 0usize;
                while bj0 < nby {
                    let bj_end = (bj0 + 2).min(nby);
                    let sj0 = bj0 * BLK;
                    let sj1 = (bj_end * BLK).min(ny);
                    let sylo = pg.origin.y + sj0 as f32 * h_p;
                    let syhi = pg.origin.y + (sj1 - 1) as f32 * h_p;
                    let mut sbi_active = [0u8; 16];
                    let mut any_super_band = false;
                    for sbi in 0..nsbx {
                        let bi0 = sbi * 2;
                        let bi_end = (bi0 + 2).min(nbx);
                        let si0 = bi0 * BLK;
                        let si1 = (bi_end * BLK).min(nx);
                        match c.classify_interval(
                            Vec3::new(pg.origin.x + si0 as f32 * h_p, sylo, zlo),
                            Vec3::new(pg.origin.x + (si1 - 1) as f32 * h_p, syhi, zhi),
                            0.0,
                        ) {
                            BlockClass::Solid(_) => {
                                let mask = ((1u64 << (si1 - si0)) - 1) << si0;
                                for lk in 0..(k1 - k0) {
                                    let kb = lk * ny;
                                    for j in sj0..sj1 {
                                        bits_slab[kb + j] |= mask;
                                    }
                                }
                                sbi_active[sbi] = 2;
                            }
                            BlockClass::Band => {
                                sbi_active[sbi] = 1;
                                any_super_band = true;
                            }
                            BlockClass::Empty(_) => {}
                        }
                    }
                    if !any_super_band {
                        bj0 = bj_end;
                        continue;
                    }
                    for bj in bj0..bj_end {
                        let j0 = bj * BLK;
                        let j1 = (j0 + BLK).min(ny);
                        let ylo = pg.origin.y + j0 as f32 * h_p;
                        let yhi = pg.origin.y + (j1 - 1) as f32 * h_p;
                        for sbi in 0..nsbx {
                            if sbi_active[sbi] != 1 {
                                continue;
                            }
                            let bi0 = sbi * 2;
                            let bi_end = (bi0 + 2).min(nbx);
                            for bi in bi0..bi_end {
                                let i0 = bi * BLK;
                                let i1 = (i0 + BLK).min(nx);
                                let xlo = pg.origin.x + i0 as f32 * h_p;
                                match c.classify_interval(
                                    Vec3::new(xlo, ylo, zlo),
                                    Vec3::new(pg.origin.x + (i1 - 1) as f32 * h_p, yhi, zhi),
                                    0.0,
                                ) {
                                    BlockClass::Solid(_) => {
                                        let mask = ((1u64 << (i1 - i0)) - 1) << i0;
                                        for lk in 0..(k1 - k0) {
                                            let kb = lk * ny;
                                            for j in j0..j1 {
                                                bits_slab[kb + j] |= mask;
                                            }
                                        }
                                    }
                                    BlockClass::Band => {
                                        c.eval_block4(Vec3::new(xlo, ylo, zlo), h_p, &mut blk);
                                        let len_mask = (1u8 << (i1 - i0)) - 1;
                                        for lk in 0..(k1 - k0) {
                                            let kb = lk * ny;
                                            let lb = lk * 4;
                                            for lj in 0..(j1 - j0) {
                                                let sm = (blk[lb + lj].sign_mask4() & len_mask) as u64;
                                                bits_slab[kb + j0 + lj] |= sm << i0;
                                            }
                                        }
                                    }
                                    BlockClass::Empty(_) => {}
                                }
                            }
                        }
                    }
                    bj0 = bj_end;
                }
            };
            #[cfg(feature = "threads")]
            if nbz >= 3 && rayon::current_thread_index().is_none() {
                let bits_ptr = bits.as_mut_ptr() as usize;
                mesher::fast_par_for_each(nbz, &|bk| {
                    let k0 = bk * BLK;
                    let k1 = (k0 + BLK).min(nz);
                    // SAFETY: each `bk` writes exclusively to `bits[k0 * ny .. k1 * ny]`.
                    let slab = unsafe { std::slice::from_raw_parts_mut((bits_ptr as *mut u64).add(k0 * ny), (k1 - k0) * ny) };
                    fill_slab_bits(bk, slab);
                });
            } else {
                for bk in 0..nbz {
                    let k0 = bk * BLK;
                    let k1 = (k0 + BLK).min(nz);
                    fill_slab_bits(bk, &mut bits[k0 * ny..k1 * ny]);
                }
            }
            #[cfg(not(feature = "threads"))]
            for bk in 0..nbz {
                let k0 = bk * BLK;
                let k1 = (k0 + BLK).min(nz);
                fill_slab_bits(bk, &mut bits[k0 * ny..k1 * ny]);
            }

            let valid = if nx >= 3 { ((1u64 << (nx - 2)) - 1) << 1 } else { 0 };
            let mut edges = 0usize;
            for k in 1..nz - 1 {
                let kb = k * ny;
                for j in 1..ny - 1 {
                    let s = bits[kb + j];
                    let sy = bits[kb + j + 1];
                    let sz = bits[kb + ny + j];
                    if (s | sy | sz) == 0 {
                        continue;
                    }
                    edges += ((s ^ (s >> 1)) & valid).count_ones() as usize
                        + ((s ^ sy) & valid).count_ones() as usize
                        + ((s ^ sz) & valid).count_ones() as usize;
                }
            }
            edges as f32 * 2.0 * ratio * ratio
        } else {
            // Wide fallback when nx > 64.
            let sy = nx;
            let sz = nx * ny;
            signs.clear();
            signs.resize(nx * ny * nz, 0);
            for bk in 0..nbz {
                let k0 = bk * BLK;
                let k1 = (k0 + BLK).min(nz);
                let zlo = pg.origin.z + k0 as f32 * h_p;
                let zhi = pg.origin.z + (k1 - 1) as f32 * h_p;
                for bj in 0..nby {
                    let j0 = bj * BLK;
                    let j1 = (j0 + BLK).min(ny);
                    let ylo = pg.origin.y + j0 as f32 * h_p;
                    let yhi = pg.origin.y + (j1 - 1) as f32 * h_p;
                    for bi in 0..nbx {
                        let i0 = bi * BLK;
                        let i1 = (i0 + BLK).min(nx);
                        let xlo = pg.origin.x + i0 as f32 * h_p;
                        let lo = Vec3::new(xlo, ylo, zlo);
                        let hi = Vec3::new(pg.origin.x + (i1 - 1) as f32 * h_p, yhi, zhi);
                        match c.classify_interval(lo, hi, 0.0) {
                            BlockClass::Solid(_) => {
                                for k in k0..k1 {
                                    for j in j0..j1 {
                                        signs[k * sz + j * sy + i0..k * sz + j * sy + i1].fill(1);
                                    }
                                }
                            }
                            BlockClass::Band => {
                                c.eval_block4(lo, h_p, &mut blk);
                                for lk in 0..(k1 - k0) {
                                    let kb = (k0 + lk) * sz;
                                    let lb = lk * 4;
                                    for lj in 0..(j1 - j0) {
                                        let sm = blk[lb + lj].sign_mask4();
                                        let row = &mut signs[kb + (j0 + lj) * sy + i0..];
                                        for li in 0..(i1 - i0) {
                                            row[li] = (sm >> li) & 1;
                                        }
                                    }
                                }
                            }
                            BlockClass::Empty(_) => {}
                        }
                    }
                }
            }
            let _ = active;
            let mut edges = 0usize;
            for k in 1..nz - 1 {
                for j in 1..ny - 1 {
                    let base = k * sz + j * sy;
                    let row = &signs[base..base + nx];
                    let row_y = &signs[base + sy..base + sy + nx];
                    let row_z = &signs[base + sz..base + sz + nx];
                    for i in 1..nx - 1 {
                        let s = row[i];
                        edges += (s != row[i + 1]) as usize + (s != row_y[i]) as usize + (s != row_z[i]) as usize;
                    }
                }
            }
            edges as f32 * 2.0 * ratio * ratio
        }
    })
}

/// Meshes every part of a program within the triangle budget. Validates first.
pub fn mesh_program(program: &Program, opts: &GenOptions) -> Result<Vec<GenPart>, GenError> {
    mesh_program_with_attempts(program, opts).map(|(parts, _)| parts)
}

fn mesh_program_with_attempts(program: &Program, opts: &GenOptions) -> Result<(Vec<GenPart>, usize), GenError> {
    program.validate().map_err(|e| GenError(e.0))?;
    let compiled: Vec<Compiled> = program.parts.iter().map(|p| Compiled::new(&p.shape)).collect();
    let bounds = compiled.iter().fold(Aabb::EMPTY, |b, c| b.union(&c.aabb));
    if bounds.is_empty() {
        return Err(GenError("the program has no shapes".into()));
    }
    let longest = bounds.size().max_element().max(1e-3);
    let base_h = longest / opts.detail as f32;
    // Small parts get finer cells so an eye or a handle isn't a lump, within limits.
    let mut hs: Vec<f32> = compiled
        .iter()
        .map(|c| {
            // Thin parts (a balloon string, a sword blade) need cells finer than their thickness.
            let l = c.aabb.size().max_element().max(1e-3);
            let thin = c.aabb.size().min_element().max(1e-3);
            base_h.min(l / 14.0).min(thin / 3.0).max(longest / 600.0)
        })
        .collect();
    for (c, h) in compiled.iter().zip(hs.iter_mut()) {
        let samples = mesher::Grid::covering(&c.aabb, *h).samples();
        if samples > mesher::MAX_SAMPLES {
            *h *= (samples as f32 / mesher::MAX_SAMPLES as f32).cbrt() * 1.02;
        }
    }
    let total_samples: usize = compiled.iter().zip(hs.iter()).map(|(c, h)| mesher::Grid::covering(&c.aabb, *h).samples()).sum();
    if total_samples > opts.max_triangles {
        let preds: Vec<f32> = compiled
            .iter()
            .zip(hs.iter())
            .map(|(c, h)| predict_part_triangles(c, *h))
            .collect();
        let predicted: f32 = preds.iter().sum();
        if predicted > opts.max_triangles as f32 * 0.90 {
            let target = opts.max_triangles as f32 * 0.88;
            let f_all = (predicted / target).sqrt();
            let mut small_tris = 0.0f32;
            let mut heavy_tris = 0.0f32;
            let mut is_small = vec![false; compiled.len()];
            for (i, (c, &p)) in compiled.iter().zip(preds.iter()).enumerate() {
                let thin = c.aabb.size().min_element().max(1e-3);
                if p <= target * 0.10 && hs[i] * f_all > thin * 0.45 {
                    is_small[i] = true;
                    let new_h = (hs[i] * f_all).min(thin * 0.45).max(hs[i]);
                    let ratio = hs[i] / new_h;
                    small_tris += p * ratio * ratio;
                    hs[i] = new_h;
                } else {
                    heavy_tris += p;
                }
            }
            if heavy_tris > 0.0 {
                let rem_target = (target - small_tris).max(target * 0.5);
                let f = (heavy_tris / rem_target).sqrt();
                for (i, h) in hs.iter_mut().enumerate() {
                    if !is_small[i] {
                        *h *= f;
                    }
                }
            }
        }
    }
    let mut raw_meshes: Vec<(mesher::RawMesh, bool)> = Vec::with_capacity(compiled.len());
    const ATTEMPTS: usize = 6;
    let mut attempts_used = 0usize;
    let mesh_one = |c: &Compiled, h: &mut f32, attempt: usize| -> (mesher::RawMesh, bool) {
        let mut grid = mesher::Grid::covering(&c.aabb, *h);
        if grid.samples() > mesher::MAX_SAMPLES {
            *h *= (grid.samples() as f32 / mesher::MAX_SAMPLES as f32).cbrt() * 1.02;
            grid = mesher::Grid::covering(&c.aabb, *h);
        }
        let mut raw = mesher::surface_nets_raw(c, &grid);
        let mut manifold = raw.guaranteed_manifold || mesher::quads_manifold(&raw.quads, raw.positions.len());
        // A wall thinner than a cell puts two sheets in one cell and the mesh stops being manifold
        // (Sculptor's booleans and edit tools need it to be). Finer cells fix that; on the last budget
        // attempt the repair pass is used instead so the budget always holds.
        let mut fine = *h;
        for _ in 0..3 {
            if attempt + 1 == ATTEMPTS || manifold {
                break;
            }
            fine *= 0.7;
            let g = mesher::Grid::covering(&c.aabb, fine);
            if g.samples() > mesher::MAX_SAMPLES {
                break;
            }
            raw = mesher::surface_nets_raw(c, &g);
            manifold = raw.guaranteed_manifold || mesher::quads_manifold(&raw.quads, raw.positions.len());
        }
        (raw, manifold)
    };
    for attempt in 0..ATTEMPTS {
        attempts_used = attempt + 1;
        #[cfg(feature = "threads")]
        let par_parts = compiled.len() >= 2
            && compiled.iter().zip(hs.iter()).map(|(c, h)| mesher::Grid::covering(&c.aabb, *h).samples()).sum::<usize>() >= 24_000;
        #[cfg(feature = "threads")]
        if par_parts {
            use rayon::prelude::*;
            raw_meshes = compiled
                .par_iter()
                .zip(hs.par_iter_mut())
                .map(|(c, h)| mesh_one(c, h, attempt))
                .collect();
        } else {
            raw_meshes.clear();
            for (c, h) in compiled.iter().zip(hs.iter_mut()) {
                raw_meshes.push(mesh_one(c, h, attempt));
            }
        }
        #[cfg(not(feature = "threads"))]
        {
            raw_meshes.clear();
            for (c, h) in compiled.iter().zip(hs.iter_mut()) {
                raw_meshes.push(mesh_one(c, h, attempt));
            }
        }
        let tris: usize = raw_meshes.iter().map(|(r, _)| r.quads.len() * 2).sum();
        if tris <= opts.max_triangles {
            break;
        }
        let f = (tris as f32 / opts.max_triangles as f32).sqrt() * 1.08;
        for h in hs.iter_mut() {
            *h *= f;
        }
    }
    let tris: usize = raw_meshes.iter().map(|(r, _)| r.quads.len() * 2).sum();
    if tris > opts.max_triangles {
        return Err(GenError(format!("couldn't fit the model in {} triangles", opts.max_triangles)));
    }
    let mut out = Vec::with_capacity(program.parts.len());
    for (part, (mut raw, manifold)) in program.parts.iter().zip(raw_meshes) {
        if raw.positions.is_empty() {
            continue;
        }
        let b = raw.positions.iter().fold(Aabb::EMPTY, |b, p| b.union(&Aabb::new(*p, *p)));
        let pivot = Vec3::new(b.center().x, b.center().y, b.min.z);
        for p in raw.positions.iter_mut() {
            *p -= pivot;
        }
        let mut mesh = if !manifold {
            let mut m = mesher::raw_to_mesh(raw, true);
            mesher::make_manifold(&mut m);
            if !opts.quads {
                mesher::triangulate(&mut m);
            }
            m
        } else {
            mesher::raw_to_mesh(raw, opts.quads)
        };
        let _ = &mut mesh;
        let base_color = part.color.as_deref().and_then(parse_color).unwrap_or(Material::default().base_color);
        let name = if part.name.trim().is_empty() { "Part".to_string() } else { part.name.trim().chars().take(60).collect() };
        out.push(GenPart {
            material: Material { name: name.clone(), base_color, metallic: part.metallic, roughness: part.roughness },
            name,
            pivot,
            mesh,
        });
    }
    if out.is_empty() {
        return Err(GenError("the shapes cancel out: nothing is left to mesh".into()));
    }
    Ok((out, attempts_used))
}

fn mesh_value(m: &Mesh) -> Value {
    let mut flat = Vec::with_capacity(m.positions.len() * 3);
    for p in &m.positions {
        flat.extend([p.x, p.y, p.z]);
    }
    json!({ "positions": flat, "faces": m.faces, "smoothShading": m.smooth_shading })
}

/// The JSON hosts receive: the program (to save, show or hand to an LLM to edit) and one mesh per part in
/// Sculptor's Mesh shape (flat positions, faces as index arrays) with its pivot and material.
pub fn parts_json(parts: &[GenPart]) -> Value {
    Value::Array(
        parts
            .iter()
            .map(|p| {
                let c = p.material.base_color;
                json!({
                    "name": p.name,
                    "pivot": [p.pivot.x, p.pivot.y, p.pivot.z],
                    "material": { "name": p.material.name, "color": [c.r, c.g, c.b, c.a], "metallic": p.material.metallic, "roughness": p.material.roughness },
                    "triangles": mesher::triangle_count(&p.mesh),
                    "mesh": mesh_value(&p.mesh),
                })
            })
            .collect(),
    )
}

pub fn generated_json(g: &Generated) -> String {
    json!({
        "ok": true,
        "understood": g.understood,
        "notes": g.notes,
        "triangles": g.triangles(),
        "program": serde_json::to_value(&g.program).expect("programs serialise"),
        "parts": parts_json(&g.parts),
    })
    .to_string()
}

pub fn error_json(e: &str) -> String {
    json!({ "ok": false, "error": e }).to_string()
}

/// Host entry points with JSON in and out, shared by the wasm and uniffi bindings.
pub mod host {
    use super::*;

    /// `generate(prompt, optionsJson)`.
    pub fn generate_json(prompt: &str, options: &str) -> String {
        match generate(prompt, &GenOptions::from_json(options)) {
            Ok(g) => generated_json(&g),
            Err(e) => error_json(&e.0),
        }
    }

    /// Plans without meshing: `{ok, understood, notes, program}`. Cheap, for live previews of the plan.
    pub fn plan_json(prompt: &str, seed: u32) -> String {
        let p = text::plan(prompt, seed);
        json!({ "ok": true, "understood": p.understood, "notes": p.notes, "program": serde_json::to_value(&p.program).expect("serialise") }).to_string()
    }

    /// Meshes a program written by an LLM, a person or a saved file. Accepts the raw LLM reply too
    /// (code fences and prose around the JSON are ignored).
    pub fn mesh_program_json(program_or_reply: &str, options: &str) -> String {
        let program = match llm::extract_program(program_or_reply) {
            Ok(p) => p,
            Err(e) => return error_json(&e.0),
        };
        match mesh_program(&program, &GenOptions::from_json(options)) {
            Ok(parts) => {
                let g = Generated { program, parts, understood: true, notes: vec![] };
                generated_json(&g)
            }
            Err(e) => error_json(&e.0),
        }
    }

    /// The instructions to give an LLM so it answers with a program.
    pub fn llm_system_prompt() -> String {
        llm::SYSTEM_PROMPT.to_string()
    }

    /// JSON Schema of a program, for constrained decoding (WebLLM, Chrome's Prompt API, Apple's @Generable).
    pub fn program_schema() -> String {
        llm::schema().to_string()
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod predictor_tests {
    use super::*;

    #[test]
    fn zero_retry_resolution_prediction() {
        for (label, prompt, detail, max_t) in [
            ("cup_8k", "a ceramic coffee cup", 64, 8_000),
            ("cup_d128_8k", "a ceramic coffee cup", 128, 8_000),
            ("chair_8k", "a wooden chair", 64, 8_000),
            ("scene_cat_table", "a cat on a table", 64, 8_000),
            ("castle_2k", "a castle", 72, 2_000),
        ] {
            let opts = GenOptions { max_triangles: max_t, detail, seed: 0, quads: true };
            let plan = text::plan(prompt, opts.seed);
            let (parts, attempts) = mesh_program_with_attempts(&plan.program, &opts).unwrap();
            let tris: usize = parts.iter().map(|p| mesher::triangle_count(&p.mesh)).sum();
            assert!(tris <= max_t, "{label}: {tris} > {max_t}");
            assert_eq!(attempts, 1, "{label}: expected 1 attempt (zero retries), got {attempts}");
        }
        // Also verify >= 90% of the 105 lexicon stress prompts hit budget on attempt 1 (remaining ones
        // only retry when thin-wall manifold refinement multiplies triangles past 8k).
        let opts = GenOptions { max_triangles: 8_000, detail: 36, seed: 0, quads: false };
        let prompts = [
            "ceramic mug", "frying pan", "espresso machine", "wine glass", "blender", "toaster", "boba", "croissant",
            "avocado", "watermelon", "ramen", "taco", "ice cream", "pumpkin", "pineapple", "pizza", "burger", "cake",
            "teapot", "cauldron", "laptop", "iphone", "vision pro", "gamepad", "boombox", "turntable", "synthesizer",
            "guitar", "saxophone", "cello", "harp", "drums", "microphone", "headphones", "camera", "microscope",
            "telescope", "drone", "solar panel", "wind turbine", "cybertruck", "fire truck", "school bus", "tractor",
            "tank", "helicopter", "spaceship", "starfighter", "ufo", "pirate ship", "yacht", "submarine", "bicycle",
            "motorcycle", "skateboard", "surfboard", "hoverboard", "jetpack", "lightsaber", "blaster", "dragon",
            "winged dragon", "phoenix", "unicorn", "lion", "wolf", "fox", "bear", "panda", "giraffe", "deer",
            "rabbit", "dolphin", "shark", "octopus", "jellyfish", "eagle", "parrot", "butterfly", "rose", "sunflower",
            "bonsai", "tree", "mushroom", "cactus", "eiffel tower", "statue of liberty", "taj mahal", "colosseum",
            "golden gate", "castle", "lighthouse", "torii", "portal", "astronaut", "wizard", "knight", "samurai",
            "ninja", "mech", "treasure chest", "spellbook", "throne", "cyber-katana", "crystal coffee maker",
        ];
        let mut first_try = 0usize;
        for p in prompts {
            let plan = text::plan(p, opts.seed);
            let (_, attempts) = mesh_program_with_attempts(&plan.program, &opts).unwrap();
            if attempts == 1 {
                first_try += 1;
            }
        }
        assert!(first_try >= 95, "expected >= 95/105 single-attempt hits, got {first_try}/{}", prompts.len());
    }
}
