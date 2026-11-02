# Status

Remote: github.com/DanielAja/sculpt-engine (private). `main` and each milestone branch are pushed after a merge; never force-pushed.

## Milestones

| | State |
|---|---|
| Phase 0 | Done (`DESIGN.md`). |
| E1 core | Done, merged. 19 unit tests, including the byte-for-byte re-encode of the golden `project.sculpt`. Benchmarks below. |
| E1b dyntopo | Not started (split out of E1 so the renderer isn't blocked). |
| E2 renderer | Done, merged. wgpu 30 on Metal: PBR (GGX, multiscatter), CPU-baked IBL (SH9 plus prefiltered cube plus BRDF LUT), shadow map with rotated Poisson PCF and a ground shadow catcher, half-res SSAO, AgX / ACES / Studio tone maps, 4x MSAA, grid. `Session` is the one host API (open, pick, sculpt, undo, save). Native winit demo. 2 golden image tests and 24 unit tests. |
| E3 wasm | Done, merged. `web` feature, `WebEngine` (wasm-bindgen) over `Session`, demo page `web/` (pointer, pen pressure, undo, open `.sculpt`, look picker). Runs on WebGPU and on WebGL2 (`?webgl`). |
| E4 to E7 | Not started. |

### E1 benchmarks (M-class Mac, shared machine, load 15 to 28, `cargo bench --bench core`)

| Bench | Time | Budget (DESIGN.md §8, desktop) |
|---|---|---|
| `SculptMesh` from a 1 M triangle mesh | 18.7 ms | |
| BVH build, 100 k triangles | 12.3 ms | |
| BVH build, 1 M triangles | 168 ms | < 60 ms: over. Build is single-threaded; parallel subtree build is the fix. |
| BVH refit, 1 M | 4.0 ms | Dabs don't refit: the BVH "slack" only triggers a refit after drift of a quarter of the radius. |
| Ray pick, 1 M | 1.4 µs | |
| Dab, small footprint (~1.2 k vertices) | 0.10 to 0.18 ms | |
| Dab, large footprint (~20 k vertices) | 1.8 to 2.9 ms | < 1 ms per 10 k: about 1.1 to 1.4 ms per 10 k, so slightly over (under load). |
| `.sculpt` encode, 1 M triangles (88 MB JSON) | 246 ms | |
| `.sculpt` decode, 1 M triangles | 135 ms | < 300 ms: met. |

### E2 frame time (Apple M4, 1920x1080, offscreen with a GPU wait per frame, `viewer --bench`)

| Scene | Full (SSAO, shadows) | No SSAO | Studio (no SSAO, no shadows) |
|---|---|---|---|
| 261 k triangles | 2.06 ms | 1.32 ms | 0.87 ms |
| 1.05 M triangles | 2.94 ms | 2.09 ms | 1.61 ms |

The shadow map is re-rendered only when geometry or placement changes, so orbiting doesn't pay for it.

Screenshots (all reviewed): `docs/screenshots/e2/window-project.png` (live Metal window, golden project), `docs/screenshots/e2/sphere-stroke.png` (scripted draw and inflate strokes with X symmetry), `tests/golden/*.png`.

Bugs found by looking at the renders: prepass normals picked up the view translation through the normal matrix's w (a hard SSAO band across the sphere), and forced smooth display smeared flat objects (the engine now shows each mesh with its own `smoothShading`, as Sculptor does).

Known gaps in E2: host overlays (gizmo, edit cage) are not drawn yet (the line pipeline exists; the API is E4); modifiers are not evaluated when the engine opens a project by itself (hosts pass evaluated meshes); flat shaded meshes re-upload every corner per dab.

### E3 web numbers (headless Chrome, Apple M4)

| | WebGPU | WebGL2 (ANGLE, Metal) |
|---|---|---|
| wasm size | 2,194,161 bytes raw, 889,777 gzip, 693,316 brotli (budget: 3 MB raw, 1 MB brotli) | same module |
| Startup (GPU, IBL bake, sphere, BVH) | 177 ms | 214 ms |
| Stroke on 65 k triangles (begin, 41 moves, end incl. `setMesh` JSON) | 110 ms (~2.7 ms per move) | 91 ms (~2.2 ms per move) |
| Frame rate | vsync bound (60 fps headless) | vsync bound |

Screenshots (reviewed): `docs/screenshots/e3/webgpu.png`, `docs/screenshots/e3/webgpu-sculpt.png`, `docs/screenshots/e3/webgl2-sculpt.png` (identical images).

Found by the WebGL2 run: GLSL ES can't read depth textures with `textureLoad`, so the SSAO prepass now packs an octahedral normal and a split linear depth (about 22 bits) into one `rgba16float` target. This is used on every tier.

Build and run: `~/code/ios27-duo-readiness/plans/heavy_gate.sh && web/build.sh`, serve `web/` (any static server with the `application/wasm` type), open `/` or `/?webgl`. `node tools/browser-shot.mjs <url> out.png [--sculpt] [--headed]` takes the screenshots. Tools installed for E3: `wasm-bindgen-cli 0.2.130` (cargo install), `binaryen 133` (Homebrew).

For the Sculptor web port (`~/code/sculptor.web`): `WebEngine` covers the `Renderer` seam (draw, `pick`) and the brush calls of `ModelEngine` (`sculptBegin`, `sculptMove`, `sculptEnd` returning the `setMesh` command to `apply`). Mapping in E7.

## Dream Builder coordination (visionOS option (a))

For the Dream Builder agent (`~/code/dreambuilder.vision`). RealityKit keeps presentation, physics and interaction. The engine plugs in behind the two protocols in `Packages/DreamCore/Sources/DreamCore/Engine.swift`; nothing in Dream Builder changes until E4 ships the xcframework.

- **`SnapshotRenderer`**: this is the real Rust piece. Planned Swift wrapper (E4, in `apple/`): `SculptEngineSnapshotRenderer: SnapshotRenderer` calls the engine's offscreen render (Metal, raster today, path traced in E6) and returns PNG data.
  - Units: the engine works in Sculptor space (millimetres, Z up). Dream Builder passes RealityKit metres, Y up, so the wrapper multiplies each object's world matrix by the fixed `Y-up metres -> Z-up mm` matrix (the inverse of `RecipeMesher`'s `toRealityKit`), and converts `eye` and `target` the same way.
  - Input: `RenderPart` buffers (positions, normals, `UInt32` indices) plus `SculptorCore.Material` map directly to the engine's render objects, with no conversion beyond the matrix.
- **`GeometryEngine`**: `build([Part]) -> BuiltObject` keeps using SculptorCore's primitives. Porting the primitives to Rust would duplicate Swift that already works, so the engine conformer (E4) wraps `SculptorCoreEngine`. It adds engine work only where it pays: sculpting (if Dream Builder ever exposes it) and mesh processing on large meshes.
- **No CompositorServices path** is planned (DESIGN.md §9, option (b) deferred).
- **Asks of the Dream Builder agent:** none yet. When E4 lands, the xcframework goes at `apple/SculptEngine.xcframework` with a SwiftPM binary target, and I'll note the exact `Package.swift` lines here.

## Resume

```sh
cd ~/code/sculpt-engine && export PATH=$HOME/.cargo/bin:$PATH
/Users/daniel/code/ios27-duo-readiness/plans/heavy_gate.sh && cargo test -j 4
/Users/daniel/code/ios27-duo-readiness/plans/heavy_gate.sh && cargo run -j 4 --release --example viewer -- tests/fixtures/project.sculpt   # live window
./target/release/examples/viewer --stroke --shot out.png                      # headless
./target/release/examples/viewer --bench [--big]                              # frame time
```

Rust 1.99.1 was installed with rustup (`--no-modify-path`), so put `$HOME/.cargo/bin` on PATH first.
