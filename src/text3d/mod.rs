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

/// Meshes every part of a program within the triangle budget. Validates first.
pub fn mesh_program(program: &Program, opts: &GenOptions) -> Result<Vec<GenPart>, GenError> {
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
    let mut meshes: Vec<Mesh> = Vec::new();
    const ATTEMPTS: usize = 6;
    for attempt in 0..ATTEMPTS {
        meshes.clear();
        for (c, h) in compiled.iter().zip(hs.iter_mut()) {
            let mut grid = mesher::Grid::covering(&c.aabb, *h);
            if grid.samples() > mesher::MAX_SAMPLES {
                *h *= (grid.samples() as f32 / mesher::MAX_SAMPLES as f32).cbrt() * 1.02;
                grid = mesher::Grid::covering(&c.aabb, *h);
            }
            let mut m = mesher::surface_nets(c, &grid);
            // A wall thinner than a cell puts two sheets in one cell and the mesh stops being manifold
            // (Sculptor's booleans and edit tools need it to be). Finer cells fix that; on the last budget
            // attempt the repair pass is used instead so the budget always holds.
            let mut fine = *h;
            for _ in 0..3 {
                if attempt + 1 == ATTEMPTS || mesher::is_manifold(&m) {
                    break;
                }
                fine *= 0.7;
                let g = mesher::Grid::covering(&c.aabb, fine);
                if g.samples() > mesher::MAX_SAMPLES {
                    break;
                }
                m = mesher::surface_nets(c, &g);
            }
            if !mesher::is_manifold(&m) {
                mesher::make_manifold(&mut m);
            }
            meshes.push(m);
        }
        let tris: usize = meshes.iter().map(|m| m.faces.len() * 2).sum();
        if tris <= opts.max_triangles {
            break;
        }
        let f = (tris as f32 / opts.max_triangles as f32).sqrt() * 1.08;
        for h in hs.iter_mut() {
            *h *= f;
        }
    }
    let tris: usize = meshes.iter().map(|m| m.faces.len() * 2).sum();
    if tris > opts.max_triangles {
        return Err(GenError(format!("couldn't fit the model in {} triangles", opts.max_triangles)));
    }
    let mut out = Vec::new();
    for ((part, mut mesh), _) in program.parts.iter().zip(meshes).zip(compiled.iter()) {
        if mesh.positions.is_empty() {
            continue;
        }
        let b = mesh.positions.iter().fold(Aabb::EMPTY, |b, p| b.union(&Aabb::new(*p, *p)));
        let pivot = Vec3::new(b.center().x, b.center().y, b.min.z);
        for p in mesh.positions.iter_mut() {
            *p -= pivot;
        }
        if !opts.quads {
            mesher::triangulate(&mut mesh);
        }
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
    Ok(out)
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
