# sculpt-engine: design

Status: Phase 0, 2026-11-01. Local repo only, no remote.

One Rust crate, `sculpt-engine`, compiled for every platform Sculptor and Dream Builder run on, with the same code and the same API everywhere. It renders (wgpu, physically based), sculpts (BVH, multithreaded brushes, dynamic topology) and reads and writes Sculptor's `.sculpt` format.

## 1. What the study found

Sources read, read only:

| Repo | Commit | Renderer today | Relevant facts |
|---|---|---|---|
| DanielAja/builder (Sculptor iPad) | `802990e` | RealityKit (`RealityView`, `LowLevelMesh`) | ~17,500 lines of portable Swift in `SculptorCore`, `SculptorKit`, `SculptorPortable`: mesh, ops, modifiers, booleans, `SceneCommand`, undo, `.sculpt` codec, exporters. Z up, millimetres. |
| DanielAja/sculptor-android | `c78e1a5` | Filament on a `SurfaceView`, linear tone mapper, IBL, no SSAO, no shadows, FXAA | Runs the same Swift core through a JNI façade (`SculptorEngine`, JSON plus bulk buffers, bridge version 4). Golden `project.sculpt` generated from the iOS code. |
| ~/code/sculptor.web (branch `w1-viewer`, in progress) | `bc97d3b` | three.js (WebGPU, WebGL2 fallback) | Same Swift core compiled to WASM in a worker. Two seams for this engine in `src/engine/types.ts`: `ModelEngine` and `Renderer`. Adds Sculpt-lite brushes (draw, inflate, smooth, flatten) that commit as `setMesh`. Tauri 2 desktop shell. |
| ~/code/dreambuilder.vision (branch `m1-summon`, in progress) | `a36597e` | RealityKit, Shared Space volume by default, optional Full Space | Seams in `DreamCore/Engine.swift`: `GeometryEngine.build([Part]) -> BuiltObject` (plain positions, normals, `UInt32` indices) and `SnapshotRenderer.snapshot(scene, eye, target, w, h) -> PNG`. Budgets: 8,000 triangles per object, 150,000 per scene. |

Three findings shape the design:

1. **Sculptor is a polygon modeller with no sculpt brushes.** `DESIGN.md` §2.14 "Sculpt-lite" was never built on iOS or Android; the web port adds it. The modelling logic (commands, booleans, modifiers, exporters) is already one shared Swift implementation on iOS, Android and web.
2. **Undo is snapshot based.** `Document` keeps up to 200 `UndoEntry { label, sceneBefore, selectionBefore }`; every edit is a `SceneCommand`. A brush stroke is therefore one `SceneCommand.setMesh(ObjectID, Mesh)` labelled "Sculpt", exactly as the web port does. No new delta type is needed in the file or the undo stack.
3. **`.sculpt` is pretty printed, key sorted JSON from Swift's `JSONEncoder`** (format version 1). Android already writes byte identical files because it runs the Swift encoder. The web port's acceptance test is a byte for byte re-encode of the golden `project.sculpt`; this engine adopts the same test.

### 1.1 Scope decision

The engine replaces the **renderer** on every platform and adds **sculpting**. It does **not** reimplement `SceneCommand`, booleans, modifiers or the exporters: those stay in the shared Swift core, which already runs everywhere and is the single source of truth for modelling. Rewriting them would duplicate 17,500 tested lines and invite drift.

The engine does carry its own `.sculpt` codec, because Dream Builder (export to Sculptor, M6), the snapshot renderer and the demos need to open and write projects without the Swift core, and because a stroke's result must serialise exactly like Swift would write it.

Owner question 1: should the Rust engine later absorb the modelling core too? Recommended default: no; revisit only if the Swift for WASM download size or Android Swift toolchain becomes a real problem.

## 2. Crate and targets

```
sculpt-engine/
  Cargo.toml              one crate; features: render (wgpu), ffi (uniffi), wasm (wasm-bindgen), threads
  src/
    math.rs               glam re-exports, OrbitCamera (Sculptor's exact maths and JSON)
    mesh.rs               SculptMesh: triangles, adjacency, normals, n-gon face map
    bvh.rs                triangle BVH: build, refit, ray pick, sphere query
    brush.rs              dab maths (draw, inflate, smooth, flatten), symmetry, rayon
    dyntopo.rs            edge split and collapse inside the brush radius
    stroke.rs             SculptSession: begin, move, end -> setMesh command, cancel
    format/               .sculpt codec (serde types + Swift compatible pretty writer), repair
    render/               wgpu renderer: device, passes, IBL bake, shadows, SSAO, tone map, accumulation, path tracer
    ffi.rs                uniffi interface (Swift, Kotlin)
    web.rs                wasm-bindgen interface
  examples/viewer.rs      native demo window (winit)
  web/                    demo page
  apple/                  xcframework build script, sample SwiftUI host
  android/                cargo-ndk build script, sample Kotlin host
  benches/                criterion
  tests/                  round trip, goldens
```

| Platform | Rust target | Packaging | Bindings |
|---|---|---|---|
| iOS, iPadOS | `aarch64-apple-ios`, `aarch64-apple-ios-sim` | static lib in `SculptEngine.xcframework` | uniffi Swift |
| macOS | `aarch64-apple-darwin`, `x86_64-apple-darwin` | same xcframework | uniffi Swift |
| visionOS | `aarch64-apple-visionos`, `aarch64-apple-visionos-sim` (prebuilt std ships with rustup 1.99) | same xcframework | uniffi Swift |
| Android | `aarch64-linux-android` (plus `x86_64-linux-android` for the emulator) | `libsculpt_engine.so` in an AAR | uniffi Kotlin (JNA) |
| Web | `wasm32-unknown-unknown` | wasm-bindgen ES module | wasm-bindgen |
| Windows, Linux | native | rlib linked into a Tauri plugin | Rust API, Tauri commands |

uniffi generates Swift and Kotlin from one Rust definition, so Apple and Android see the same object model. Hot paths (vertex buffers, overlay buffers) cross as byte buffers, not per element calls.

## 3. GPU layer

wgpu (30.x) everywhere: Metal on Apple, Vulkan on Android and Linux, DX12 on Windows, WebGPU in browsers, WebGL2 as the browser fallback.

The renderer has two capability tiers, chosen at startup from `Adapter::get_downlevel_capabilities`:

- **Full** (Metal, Vulkan, DX12, WebGPU): compute shaders and storage buffers. Enables GPU SSAO with a compute blur, the path tracer, and GPU skinning of brush previews.
- **Downlevel** (WebGL2, old Android GLES): raster only, uniform buffers only, no compute. SSAO runs as a fragment pass, the path tracer is unavailable, and quick render uses progressive raster accumulation.

All shaders are WGSL, validated by naga at build time in a unit test for both tiers.

## 4. Renderer

Scene space is Sculptor's: millimetres, right handed, Z up. Colours: `ColorRGBA` is sRGB, converted to linear in the shader. Materials map 1:1 from Sculptor's `Material` (base colour, metallic, roughness).

### 4.1 Interactive frame (sculpting at 60 to 120 fps)

| Pass | Detail | Cost note |
|---|---|---|
| Shadow | One directional key light, 2048² depth map (1024² on mobile) fitted to scene bounds. Sculptor scenes are tabletop sized, so cascades buy nothing; PCF 3x3 with a rotated Poisson kernel. Cached: re-rendered only when geometry or the light changes. | 0 ms on still frames |
| Depth and normal prepass | Only when SSAO is on; view space normals in `rg16snorm` (octahedral). | Doubles vertex work, so off on mobile while a stroke is active |
| SSAO | Half resolution, 12 samples, hemisphere in view space, 4x4 noise, bilateral blur. | ~0.6 ms at 1080p half res on M2 |
| Forward PBR | Cook Torrance GGX, Smith height correlated visibility, Schlick Fresnel; multiscatter energy compensation from the BRDF LUT. Diffuse from 9 coefficient spherical harmonics, specular from a prefiltered cube map (GGX importance sampled, 128² with 6 mips), split sum BRDF LUT (64², `rg16float`). Key light shadowed. Flat or smooth normals per object, as the `smoothShading` flag says. | Main cost |
| MSAA | 4x on the forward pass. Cheap on tile based mobile GPUs (resolve stays in tile memory) and it never ghosts, unlike TAA, which smears a moving sculpt surface. TAA is not used in the interactive mode. | |
| Tone map | AgX by default (handles the saturated clay swatches without hue shifts), ACES fitted as an option, plus a "Studio" preset that reproduces today's Android look (linear, no SSAO) so the switch is not a visual surprise. Output to an sRGB surface. | Fullscreen pass |
| Overlays | Grid, selection outline, and host supplied line, point and triangle lists (edit cage, gizmo), drawn unlit after tone mapping. Gizmo logic stays in the hosts. | |

The IBL (SH9, prefiltered cube, BRDF LUT) is computed by the same Rust functions on the CPU, either baked into the binary for the default studio environment or computed at load for a user HDR. That avoids compute shaders in the downlevel tier and keeps startup fast: the baked studio maps are about 200 KB.

### 4.2 Quick realistic render

Triggered by the host (`render_quality(Final)`) or automatically when the camera and scene have been idle for 300 ms (configurable). Output is either the viewport itself or an offscreen image returned as RGBA8 or PNG.

1. **Progressive raster accumulation** (every tier). Each frame jitters the projection by a Halton (2,3) sub pixel offset, rotates the shadow PCF kernel and SSAO kernel, and samples a jittered light direction inside the sun's solid angle (soft shadows). Frames accumulate in an `rgba16float` (WebGL2) or `rgba32float` (full tier) target. Converges in 32 to 64 frames, well under a second on desktop. Any input resets it.
2. **Path traced preview** (full tier only). A compute shader path tracer over the same triangle BVH the brushes use, flattened into storage buffers: GGX and Lambert BSDF from the same material parameters, environment importance sampling from the IBL, next event estimation for the key light, Russian roulette after 3 bounces, 1 sample per pixel per dispatch, accumulated. Hardware ray queries are not used: wgpu exposes them only experimentally and only on some backends, while the compute BVH runs on every full tier backend. An edge aware à trous denoiser (albedo and normal guided) runs on the accumulated image for the first 16 samples, so the preview looks clean quickly.
3. **Fallback.** Downlevel tier or a mesh above the path tracer budget (Section 8) uses the progressive raster. The host always gets an image; `quality_info()` says which path produced it.

## 5. Mesh core

### 5.1 Representation

`SculptMesh` is an **indexed triangle mesh** with structure of arrays: positions, normals, triangle indices, and vertex to triangle adjacency in compressed sparse row form (rebuilt after topology changes, patched locally by dyntopo). It also keeps the map from each triangle back to its source n-gon face.

Why not half edge: Sculptor's format and every host already use indexed faces; indexed triangles upload to the GPU with no conversion; brush passes are cache friendly linear sweeps; and dyntopo's local edge split and collapse is straightforward with per vertex triangle lists (the SculptGL and Sculptris approach). Half edge would add a conversion on every load and save and a pointer chasing layout to the hot path.

Sculptor meshes have n-gon faces (fan triangulated, convex by construction). Two cases:

- **Displacement only strokes** (draw, inflate, smooth, flatten without dyntopo) move vertices and keep every face exactly. `end()` writes the original faces with new positions. Projects remain identical in structure.
- **Dyntopo strokes** change topology, so the object becomes triangles from that stroke on. The host shows a one time notice ("Detail editing turns this object into triangles"). Owner question 2: acceptable? Recommended default: yes, notice shown once per object.

### 5.2 BVH

Binned SAH build (12 bins) over triangles, flattened into a node array with 4 triangles per leaf; built in parallel with rayon on native.

- **Refit** after each dab (bottom up, only dirty nodes): O(touched) and keeps picking exact during a stroke.
- **Rebuild** when topology changes or when the refit surface area grows 2x past the build's (quality decay).
- **Queries:** closest ray hit (picking, brush placement), sphere query returning vertices (brush footprint, stamp array to dedupe), and the flattened layout is what the path tracer uploads.

### 5.3 Brushes

The brush maths matches the web port's Sculpt-lite exactly, so strokes give the same mesh on every platform and the web port's stroke hash tests carry over:

- falloff `f = (1 - (d/r)²)²` inside radius `r`;
- **draw**: move along the footprint's weighted average normal by `s·f·r·0.1`;
- **inflate**: move along each vertex normal by `s·f·r·0.1`;
- **smooth**: move toward the average of edge neighbours by `|s|·f`;
- **flatten**: move toward the footprint's weighted average plane by `|s|·f`;
- **symmetry**: mirrored dab centres across the object's local X, Y and Z planes;
- parameters: `{ kind, radius (world mm), strength (-1 to 1), symmetry: [bool; 3] }`, plus `pressure` (multiplies strength) and `spacing` (dab spacing as a fraction of the radius, default 0.25).

**Determinism and threads:** every dab reads the positions before the dab and writes into a scratch array (Jacobi style), then swaps. Results do not depend on thread count or scheduling. On native the footprint is split into chunks of 1,024 vertices and processed with rayon when it holds more than 4,096 vertices (below that, threads cost more than they save). On the web the default build is single threaded; a `threads` build uses wasm threads through `wasm-bindgen-rayon` and needs `SharedArrayBuffer`, so the page must be served with COOP and COEP headers (the Tauri webview and a configured host can; arbitrary embeds cannot). Owner question 3: ship the threaded web build? Recommended default: yes where cross origin isolation is available, single threaded otherwise.

**Dyntopo:** per dab, inside the footprint, split edges longer than `4/3·L` and collapse edges shorter than `4/5·L`, where `L` is the detail size (a fraction of the brush radius, or a constant screen size). Collapses that would flip a triangle or create a non manifold edge are refused. Normals and BVH are patched locally; the GPU buffer grows by doubling.

### 5.4 Strokes and undo

```
begin(object_id, mesh, world_matrix)      host passes the object (or the engine already has it)
move(points or rays, pressure)            dabs; the viewport shows the working mesh; nothing enters history
end() -> SetMeshCommand                   {"setMesh":{"_0":{"rawValue":"<uuid>"},"_1":<Mesh JSON>}}
cancel()                                  restores the start mesh
```

`end()` returns the exact `SceneCommand` JSON the Swift core accepts, so each host passes it to its existing `apply` with the label "Sculpt" and gets one undo step through the existing `Document`. Inside a stroke the engine keeps sparse per vertex deltas (index, before) so `cancel()` and live preview cost O(touched), not a mesh copy.

Hosts without the Swift core (Dream Builder, demos) use the engine's own `History`, which stores the same entries: label plus the before mesh (copy on write `Arc`), with the 200 entry limit and a byte budget.

## 6. `.sculpt` format

Rust types mirror `ProjectFile`, `Scene`, `SceneObject`, `Mesh` (flat positions, faces as index arrays), `Transform` (position, quaternion `[x, y, z, w]`, scale), `Material`, `ColorRGBA`, `ModifierEntry` (Swift's synthesized enum shape `{"mirror":{"_0":{...}}}`), `Selection` and `cameraState` (base64 of the `OrbitCamera` JSON).

- **Writer:** a small Swift compatible pretty printer: two space indent, `"key" : value`, keys sorted, empty arrays as `[\n\n<indent>]`, floats printed as Swift's `Float.description` (shortest round trip digits, exponent form outside 1e-4 to 2^24, `.0` dropped), `nan`, `inf` and `-inf` as strings, slashes not escaped.
- **Reader:** accepts everything Swift accepts (flat or nested positions, keyed or bare faces, missing `smoothShading`, numeric or ISO 8601 dates, unknown keys ignored), rejects newer `formatVersion` with the same three errors, and applies Swift's repair rules (drop bad faces, zero non finite positions, reset non finite transforms, clamp modifiers and materials, drop invalid parents).
- **Acceptance test:** decode then encode the golden `project.sculpt` and compare bytes.

## 7. Host integration

The host owns the window or layer and the frame clock. The engine never creates windows on mobile or web.

| Host | Surface handed to the engine | Frame clock |
|---|---|---|
| iOS, iPadOS, macOS | `CAMetalLayer` pointer from a `UIView`/`NSView` subclass (`layerClass`) | `CADisplayLink` (ProMotion 120 Hz) |
| visionOS | none for presentation (Section 9); offscreen only | on demand |
| Android | `ANativeWindow` from the `SurfaceView`'s `Surface` (JNI `ANativeWindow_fromSurface`) | `Choreographer` |
| Web | `HTMLCanvasElement` (or `OffscreenCanvas` in a worker) | `requestAnimationFrame` |
| Tauri (Windows, Linux, macOS) | the web build inside the webview for the interactive viewport; the native crate as a Tauri plugin for offscreen quick renders and threaded sculpting | webview `requestAnimationFrame` |

Tauri note: drawing natively under or over a webview is fragile on every OS (z order, resize, input routing). The web build already runs in the webview at near native speed on WebGPU (WebView2, WKWebView). Linux WebKitGTK has no WebGPU yet, so it gets the WebGL2 tier, and the native plugin's path tracer covers quick renders there.

API shape (same names in Swift, Kotlin, JS and Rust):

```
Engine::new(options) -> Engine                                  device and queue, shared by all viewports
Engine::create_viewport(surface, width, height, scale) -> Viewport
Viewport.resize(width, height, scale)
Viewport.set_camera(OrbitCamera)                                Sculptor's camera JSON shape; hosts keep their camera code
Viewport.set_scene(render list)                                 objects: id, triangles, world matrix, material, smooth flag
Viewport.update_object(id, positions | full mesh)               partial upload after a dab
Viewport.set_overlays(lines, points, triangles)                 edit cage, gizmo
Viewport.set_settings(RenderSettings)                           tone mapper, SSAO, shadows, environment, quality
Viewport.render() -> FrameInfo                                  called from the host's frame clock
Viewport.pick(x, y) -> Hit?                                     BVH ray pick, screen pixels
Engine.render_offscreen(scene, camera, width, height, quality) -> RGBA8 or PNG
Engine.open_project(bytes) / project_bytes(...)                 .sculpt codec
SculptSession::begin / move / end / cancel                      Section 5.4
```

Input: the host turns gestures into camera changes (its existing `OrbitCamera` code) and into `SculptSession.move` calls with screen positions and pressure. The engine converts screen positions to rays with the same camera maths, so picking matches what is drawn.

## 8. Performance budgets

Reference devices: iPad with M2 (120 Hz), iPhone 15 (A16, 60 Hz), Pixel 8 class Android (Tensor G3 or Snapdragon 8 Gen 2, 60 to 120 Hz), M-class Mac and a mid range Windows GPU, Chrome on the same Mac for web.

| Budget | Desktop native | iPad M2 | Phone (A16, Pixel 8) | Web WebGPU | Web WebGL2 |
|---|---|---|---|---|---|
| Sculpt mesh, interactive | 5 M triangles | 2 M | 1 M | 1 M | 500 k |
| Frame time, interactive | 8.3 ms | 8.3 ms (120 Hz) | 16.6 ms | 16.6 ms | 16.6 ms |
| Dab, 1 M triangle mesh, 10 k vertex footprint | < 1 ms | < 2 ms | < 3 ms | < 6 ms (1 thread) | < 6 ms |
| Input to photon during a stroke | 2 frames | 2 frames | 2 frames | 2 frames | 2 frames |
| BVH build, 1 M triangles | < 60 ms | < 120 ms | < 200 ms | < 400 ms | < 400 ms |
| Refit after a dab | < 0.3 ms | < 0.5 ms | < 1 ms | < 1 ms | < 1 ms |
| Raster accumulation to 64 frames, 1080p | < 0.6 s | < 1 s | < 2 s | < 1.5 s | < 3 s |
| Path traced preview, 1080p, 64 spp, 1 M tris | < 3 s | < 6 s | < 12 s | < 8 s | n/a |
| `.sculpt` decode, 1 M triangles | < 300 ms | < 500 ms | < 800 ms | < 1 s | < 1 s |

Sizes: wasm (WebGPU and WebGL2, release, `wasm-opt -Oz`) under 3 MB raw and 1 MB brotli; Android `.so` under 8 MB stripped per ABI; each xcframework slice under 15 MB static (the linker strips most of it).

Benchmarks (criterion, `benches/`): `bvh_build/{100k,1m}`, `bvh_refit`, `ray_pick`, `dab/{draw,inflate,smooth,flatten}/{small,large}` single and multithreaded, `dyntopo_dab`, `sculpt_decode` and `sculpt_encode` on a 1 M triangle project. Frame time is measured with a headless wgpu harness using timestamp queries where the adapter supports them, otherwise wall clock with `device.poll(Wait)`. Each milestone report lists the numbers measured on this Mac.

## 9. visionOS

A custom engine can present immersive content on visionOS only through **CompositorServices** (`CompositorLayer` in an `ImmersiveSpace`, Metal, per eye drawables, rasterization rate maps for foveation). wgpu has no CompositorServices backend, and CompositorServices is not available in the Shared Space at all: windows and volumes are composited by RealityKit only.

Options:

**(a) RealityKit presents; the Rust engine generates geometry, sculpts and makes realistic stills.** Dream Builder keeps RealityKit for presentation, physics, `ManipulationComponent`, grounding shadows, passthrough lighting and spatial audio. The engine plugs into the two existing seams: `GeometryEngine` (meshes and hulls into `LowLevelMesh` buffers, no copy beyond one memcpy) and `SnapshotRenderer` (offscreen wgpu Metal render, raster or path traced, to PNG). Sculpting in Dream Builder, if added, runs the engine's brushes and uploads into `LowLevelMesh` each dab.

- Works in the Shared Space volume, which is Dream Builder's default, and in the simulator.
- Keeps everything visionOS does for free: foveation, reprojection, system lighting estimation, occlusion with the room, comfort features, accessibility.
- No second render path to maintain on visionOS.
- Cost: the live view on visionOS looks like RealityKit, not like the engine's renderer, so the "realistic" look appears only in snapshots. Custom shading is limited to what `ShaderGraphMaterial` and `PhysicallyBasedMaterial` offer.

**(b) A Metal interop path into CompositorServices.** Wrap each frame's CompositorServices color and depth textures as wgpu textures (`create_texture_from_hal`), render both eyes with wgpu, present through `cp_frame`.

- Full control of the look in an immersive space.
- Only works in a Full Space, not in the default volume.
- wgpu cannot use rasterization rate maps (no foveation, so a large GPU cost at visionOS resolutions), has no vertex amplification (two full passes for stereo), and depends on wgpu-hal internals that change between releases.
- Physics, hands and the room would have to be bridged from ARKit data by hand; RealityKit's physics would no longer drive what is seen.

**Recommendation: (a).** It matches Dream Builder's plan (RealityKit owns presentation and physics), works where the app actually runs (the Shared Space volume), and costs nothing extra. Revisit (b) only if a future Full Space "studio" mode needs the engine's exact look live, and then prototype it as a thin Metal layer that consumes the engine's mesh and material data rather than running wgpu inside CompositorServices.

## 10. Testing

- **Unit:** mesh adjacency and normals, n-gon face map round trip, BVH (every triangle reachable, refit bounds contain all vertices, ray hits agree with brute force on random rays), brushes (each kind against a hand computed case, symmetry, Jacobi determinism across thread counts), dyntopo (manifold preserved, no flipped triangles, Euler characteristic kept on closed meshes), stroke begin, move, end and cancel.
- **Format:** byte for byte re-encode of the golden `project.sculpt`; Swift float formatting edge cases; repair rules; newer version rejected.
- **Render goldens:** headless render of fixed scenes (clay sphere, metal and rough spheres, golden project) compared with stored PNGs: mean absolute error under 1.5/255 and at most 0.5% of pixels off by more than 8/255. Goldens are stored per backend (Metal on this Mac first), since rasterisers differ in the last bit.
- **Benchmarks:** criterion, Section 8.
- **Demos:** screenshots of each demo in `docs/screenshots/eN/`, reviewed by eye.

## 11. Milestones

Each milestone is a local branch `eN-...`, merged into `main` with `--no-ff` once its tests pass.

| | Scope | Done when |
|---|---|---|
| **E1** | `SculptMesh`, BVH, brushes with symmetry and rayon, strokes and history, `.sculpt` codec. Pure Rust. | Unit tests pass; golden `project.sculpt` re-encodes byte for byte; criterion benchmarks recorded. |
| **E2** | wgpu renderer: PBR, baked IBL, shadow map with PCF, SSAO, AgX and ACES, MSAA, overlays (grid); native macOS demo window (winit) that opens a `.sculpt` and sculpts with the mouse. | Golden image tests pass on Metal; demo screenshots reviewed; frame times recorded. |
| **E3** | wasm build (WebGPU and WebGL2), wasm-bindgen API, demo page. | Demo runs in Chrome on WebGPU and with WebGPU disabled (WebGL2); wasm size recorded. |
| **E4** | Apple xcframework (iOS, iPadOS, macOS, visionOS, simulators) with uniffi Swift bindings; sample SwiftUI host view (`CAMetalLayer`). | Sample builds for every slice and runs in one iPad simulator; visionOS snapshot renders offscreen. |
| **E5** | Android `.so` (arm64, x86_64) with uniffi Kotlin; sample Kotlin host on a `SurfaceView`. | Sample runs on the emulator; `.so` size recorded. |
| **E6** | Quick realistic render: progressive raster accumulation, compute path tracer, denoiser, fallback logic, offscreen PNG. | Path tracer converges to a golden within tolerance; timings recorded. |
| **E7** | Migration guides for Sculptor iOS, Android, web and Tauri, and Dream Builder. | One guide per app in `docs/migration/`, each mapping today's renderer calls to the engine API. |

## 12. Risks

- **wgpu churn.** wgpu releases break APIs often; the crate pins one version and wraps it behind `render/`.
- **WebGL2 tier** has no compute: no path tracer, slower SSAO. Acceptable as a fallback.
- **wasm threads** need cross origin isolation (Section 5.3).
- **visionOS** presentation stays RealityKit (Section 9).
- **Golden images** differ by GPU vendor; goldens are per backend, and the tolerance is set for ordinary rasteriser differences only.
- **Mobile thermals:** SSAO and the prepass switch off during strokes on phones; budgets are for sustained, not burst, frame rates.

## 13. Owner questions

1. Should the engine later replace the Swift modelling core? Default: no (Section 1.1).
2. Dyntopo turns an object into triangles; acceptable with a one time notice? Default: yes.
3. Ship the threaded web build where cross origin isolation exists? Default: yes.
4. Adding sculpt brushes to Sculptor iOS and Android (they have none today) is a product change; the engine makes it cheap. Default: offer it as a new tool once E4 and E5 land.
