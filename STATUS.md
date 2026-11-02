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
| E4 Apple | Done, merged. `ffi` feature (uniffi 0.32): `EngineView` (Metal layer, Android window or offscreen) plus `renderSnapshot` for Dream Builder. `apple/build-xcframework.sh` gives `SculptEngine.xcframework` (iOS, iOS sim, macOS, visionOS, visionOS sim; arm64) and the SwiftPM package `apple/SculptEngine`. iPad SwiftUI sample runs on the simulator; visionOS offscreen snapshot verified in the visionOS simulator. |
| E4.5 host scene | Next: the web port's top gaps (below). |
| E5 to E7, E1b | Not started. |

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

### E4 results

- Host scene API added for every binding (`Session`, `EngineView`, `WebEngine`): `upsertObject(id, meshJson, world[16], color[4], metallic, roughness)`, `removeObject`, `setHighlight`, `frameAll`, `hit(x, y)` (id and world point), `project(x, y, z)` (screen pixels and depth). Input is validated (bad meshes, non-finite matrices refused).
- `apple/Sample` (SwiftUI, `UIView` with a `CAMetalLayer`, `CADisplayLink` at up to 120 Hz, touch and Pencil pressure): Metal on the iOS 27 simulator GPU, 60 fps (the simulator's cap), a scripted stroke on the ball returns a 23 KB `setMesh` command. Screenshot `docs/screenshots/e4/ipad-sample.png` (reviewed).
- visionOS: `apple/SnapshotCheck/run.sh` builds a Swift CLI against the xros-simulator slice and runs it with `simctl spawn` in a throwaway Apple Vision Pro simulator (deleted afterwards). `renderSnapshot` returns a PNG: `docs/screenshots/e4/visionos-snapshot.png` (reviewed). It took 29 s in the simulator (software-heavy GPU and shader compilation there); measure on a device before relying on it.
- Found on the iOS simulator: its Metal device allows 15 inter-stage shader variables, below wgpu's default 16, so device creation failed. The engine now asks for wgpu's downlevel limits on every tier.
- Simulator slices are arm64 only (Apple Silicon Macs). Apps must set `EXCLUDED_ARCHS[sdk=iphonesimulator*] = x86_64` (the sample does).
- Static library slices are about 93 MB each with debug info; the linker strips them to a few MB in an app (not yet measured in an archive).

## Answers for the Sculptor web port (from its STATUS.md gap list, 2026-11-02)

Planned next, in this order (milestone E4.5):

1. **Per-object setters without re-upload:** `setTransform(id, world[16])`, `setMaterial(id, color[4], metallic, roughness)`, `setVisible(id, bool)`. Add and remove already exist as of E4: `upsertObject(id, meshJson, world, color, metallic, roughness)` and `removeObject(id)` (pull `core/sculpt-engine` to `e4-apple` or later).
2. **Binary mesh upload:** `upsertObjectBuffers(id, positions: Float32Array, faceSizes: Uint32Array, faceIndices: Uint32Array, smooth, world, color, metallic, roughness)` and `setMeshBuffers(id, positions, faceSizes, faceIndices, smooth)`, keeping n-gon faces so a stroke's `setMesh` keeps them. The JSON forms stay for convenience.
3. **Overlays:** selection outline (silhouette pass on highlighted objects), edit cage (vertex dots, edges and face tint from host buffers or computed from the object's mesh with a selection mask), gizmo lines from host line lists, drawn after tone mapping so colours match the UI.
4. **Brush symmetry on Y and Z:** `setBrush(kind, radius, strength, symX, symY, symZ)` (the core already supports all three).
5. **`sculptEnd` label:** the result carries the label, and `sculptEnd` gets a variant that returns `{"label": "Sculpt", "command": {...}}` so the host can pass it to `apply` without `applyGroup`.
6. **Modifiers:** later. Refusing to sculpt objects with enabled modifiers is the right call until then.

## Dream Builder coordination (visionOS option (a))

For the Dream Builder agent (`~/code/dreambuilder.vision`). RealityKit keeps presentation, physics and interaction. The engine plugs in behind the two protocols in `Packages/DreamCore/Sources/DreamCore/Engine.swift`; nothing in Dream Builder changes until E4 ships the xcframework.

- **`SnapshotRenderer`**: this is the real Rust piece. Planned Swift wrapper (E4, in `apple/`): `SculptEngineSnapshotRenderer: SnapshotRenderer` calls the engine's offscreen render (Metal, raster today, path traced in E6) and returns PNG data.
  - Units: the engine works in Sculptor space (millimetres, Z up). Dream Builder passes RealityKit metres, Y up, so the wrapper multiplies each object's world matrix by the fixed `Y-up metres -> Z-up mm` matrix (the inverse of `RecipeMesher`'s `toRealityKit`), and converts `eye` and `target` the same way.
  - Input: `RenderPart` buffers (positions, normals, `UInt32` indices) plus `SculptorCore.Material` map directly to the engine's render objects, with no conversion beyond the matrix.
- **`GeometryEngine`**: `build([Part]) -> BuiltObject` keeps using SculptorCore's primitives. Porting the primitives to Rust would duplicate Swift that already works, so the engine conformer (E4) wraps `SculptorCoreEngine`. It adds engine work only where it pays: sculpting (if Dream Builder ever exposes it) and mesh processing on large meshes.
- **No CompositorServices path** is planned (DESIGN.md §9, option (b) deferred).
- **Ready as of E4.** Build `apple/build-xcframework.sh` (or take a release artifact once the owner sets one up), then add the package by local path:

  ```swift
  // DreamCore/Package.swift (or the app): dependencies
  .package(path: "../../sculpt-engine/apple/SculptEngine"),
  // target dependencies
  .product(name: "SculptEngine", package: "SculptEngine"),
  ```

  A conformer sketch for `SnapshotRenderer`: for each `RenderPart`, pass `positions` (flattened, converted Y-up metres to Z-up millimetres: `(x, -z, y) * 1000`), `indices`, `world` = identity (or the entity's transform converted the same way), the part's `finish` colour, metallic and roughness, and an `OrbitCamera` JSON aimed at `target` from `eye` (distance = |eye - target| in mm, yaw and pitch from the direction). Call `renderSnapshot(objects:cameraJson:width:height:)` off the main actor; it returns PNG `Data`. Flat parts render flat because RecipeMesher duplicates their corners; shared vertices render smooth.
- **Asks of the Dream Builder agent:** none. The 29 s simulator time is not representative; please note a device timing when you first try it.

## Resume

```sh
cd ~/code/sculpt-engine && export PATH=$HOME/.cargo/bin:$PATH
/Users/daniel/code/ios27-duo-readiness/plans/heavy_gate.sh && cargo test -j 4
/Users/daniel/code/ios27-duo-readiness/plans/heavy_gate.sh && cargo run -j 4 --release --example viewer -- tests/fixtures/project.sculpt   # live window
./target/release/examples/viewer --stroke --shot out.png                      # headless
./target/release/examples/viewer --bench [--big]                              # frame time
```

Rust 1.99.1 was installed with rustup (`--no-modify-path`), so put `$HOME/.cargo/bin` on PATH first. Targets installed: wasm32, iOS and simulator, visionOS and simulator, Android arm64 and x86_64.

## Next (E4, Apple)

1. Branch `e4-apple`. Add an `ffi` feature (uniffi 0.32, thiserror, pollster, png, raw-window-handle), `uniffi::setup_scaffolding!()` in `lib.rs`, `pub mod ffi` behind it. `src/ffi.rs` is already written (`EngineView` with `with_metal_layer`, `with_android_window`, `offscreen`, the `Session` calls, `snapshot_png`, and `render_snapshot` for Dream Builder) but has **never been compiled**.
2. `apple/build-xcframework.sh`: `cargo rustc --crate-type staticlib --features ffi` for aarch64-apple-ios, aarch64-apple-ios-sim, aarch64-apple-darwin, aarch64-apple-visionos, aarch64-apple-visionos-sim; uniffi-bindgen Swift; `xcodebuild -create-xcframework` (through the heavy gate, release Xcode only).
3. Sample SwiftUI host (`apple/Sample`, a `UIViewRepresentable` with a `CAMetalLayer` and `CADisplayLink`), built for one iPad simulator of my own (deleted afterwards), plus an offscreen visionOS snapshot check.
4. Update the Dream Builder section here with the exact `Package.swift` binary target lines.
