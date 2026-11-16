# Text to 3D (`src/text3d`)

Words or an LLM's shape program in, closed and manifold meshes out (quad-dominant), on every platform the engine builds for. No GPU and no network. Design and research: sculptor.web `docs/text-to-3d-design.md` and `docs/text-to-3d-research.md`.

| File | What it does |
|---|---|
| `program.rs` | The shape program: JSON, Sculptor space (mm, Z up), parts with colour and finish, an SDF/CSG node tree, validation limits |
| `sdf.rs` | Compiles a program and evaluates the signed distance (Quilez formulas, smooth booleans); `tight_aabb` and `summit` sample the field for layout |
| `mesher.rs` | Surface Nets with per-component cells (Dual Marching Cubes rules, asymptotic decider), Newton projection, manifold check and repair, triangulation |
| `text.rs` | The built-in planner: counts, colours, materials, sizes, dimensions, relations, accessories, layout |
| `templates.rs` | About 85 object templates, a parametric quadruped and bird (70 names), archetype families, the fallback lump |
| `llm.rs` | `SYSTEM_PROMPT`, `schema()` (JSON Schema), `extract_program` (tolerant, validating parser for LLM replies) |
| `mod.rs` | `generate`, `mesh_program`, `GenOptions` (`maxTriangles`, `detail`, `seed`, `quads`), and the JSON `host` API |

Bindings, all JSON strings in and out:

- wasm (`web` feature): `textTo3d(prompt, options)`, `planText(prompt, seed)`, `meshProgram(programOrReply, options)`, `programPrompt()`, `programSchema()`.
- uniffi (`ffi` feature): `textTo3d(prompt:optionsJson:)`, `meshProgram(programOrReply:optionsJson:)`, `programPrompt()`, `programSchema()`.

Try it:

```sh
cargo run --release --no-default-features --example text3d -- "a cat with a red hat on a table" out.obj
cargo run --release --example text3d_shot -- sheet.png "a castle" "a dragon"     # engine-rendered contact sheet
cargo test --release --no-default-features --lib text3d
```

Screenshots: `docs/screenshots/text3d/`.
