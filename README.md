# sculpt-engine

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust Edition 2024](https://img.shields.io/badge/rust-edition%202024-orange.svg)](Cargo.toml)

`sculpt-engine` is the open source Rust 3D geometry, sculpting, SDF meshing, Gaussian Splat, and `wgpu` rendering engine shared by **[Sculptor](https://ajadigital.co/app/sculptor.html)** and **[Dream Builder](https://ajadigital.co/app/dreambuilder.html)**.

One crate compiles to iPad, iPhone, Mac, Apple Vision Pro, Android, WebAssembly, Windows, and Linux. Every mesh generator produces closed, watertight, outward-facing 2-manifold geometry in right-handed Z-up millimetres ready for 3D printing and real-time sculpting.

Explore the interactive 3D benchmark visualizer at **[ajadigital.co/rust-engine.html](https://ajadigital.co/rust-engine.html)**.

## Headline benchmark gains

On a quiet Apple M4 (4P + 4E, `rustc 1.99.1`, `--release`), every hot path in `sculpt-engine` is benchmarked in Criterion against standard baselines with output equivalence verified before timing:

- **92.8x faster SDF meshing**: 128^3 Text to 3D sparse octree Surface Nets drops from `9.080 ms` to **`97.81 us`** (1% of an 8.33 ms 120 Hz frame).
- **52.0x faster 1M triangle BVH build**: parallel Morton HLBVH drops from `249.7 ms` (sort baseline) and `82.40 ms` (partition baseline) to **`4.798 ms`**, with **`408.0 ns`** ray picks.
- **13.2x faster project JSON I/O**: 88 MB `.sculpt` decode reaches **`17.8 GB/s`** (`4.951 ms` vs `65.43 ms` typed `serde_json`), and streaming encode reaches **`13.7 GB/s`** (`6.394 ms` vs `83.82 ms`).
- **21.4x faster `.splat` encode and 20.7x smaller Gaussian Splats**: ARM NEON `.splat` decode runs at **`182 GB/s`** (`5.04 us` for 16k splats), radix depth sort processes 500k splats in **`427.7 us`** (`15.0x` faster), and `.dsplat` v2 packs each splat into **12 bytes** (vs 248 B SH3 `.ply`).
- **37.4 us brush dabs and 340 fps at 1.05M triangles**: zero-allocation sculpt dabs with stamped normal updates and **`2.94 ms`** full-frame rendering at 1080p with SSAO and shadows.

---

## Run the benchmarks yourself

Do not take our numbers on faith. Every benchmark in this README lives directly in [`benches/core.rs`](benches/core.rs) with built-in equivalence assertions, so you can verify and profile them on your own CPU in a couple of minutes:

```bash
git clone https://github.com/DanielAja/sculpt-engine.git
cd sculpt-engine
cargo test --release --lib
cargo bench --bench core -- vs_industry_standard
```

You can also run individual benchmark groups:

```bash
cargo bench --bench core -- sculpt_format
cargo bench --bench core -- text3d
cargo bench --bench core -- img3d_and_splat
cargo bench --bench core -- ibl
```

If you run `cargo bench` on x86_64 (AVX2/AVX-512), AMD Ryzen, Apple M1/M2/M3/M4, Snapdragon ARM, or Linux/Windows desktops, please share your Criterion results in [GitHub Issues](https://github.com/DanielAja/sculpt-engine/issues) or open a pull request with optimizations or cross-architecture SIMD improvements.

---

## Full Criterion table (Apple M4)

Measured on a quiet Apple M4 (4P + 4E, macOS 27.0, `rustc 1.99.1`, `--release` with `opt-level = 3` and thin LTO) using identical inputs and output equivalence checks before timing:

| Workload | `sculpt-engine` | Reference baseline | Speedup |
| --- | --- | --- | --- |
| Text to 3D SDF meshing (128^3 grid) | **97.81 us** | 9.080 ms (full-grid Surface Nets) | **92.8x** |
| 1M triangle BVH build (parallel Morton HLBVH) | **4.798 ms** | 82.40 ms partition / 249.7 ms sort | **17.2x / 52.0x** |
| 1M triangle ray pick (SIMD FMA slab test) | **408.0 ns** | 913.0 ns (v1 scalar) | **2.24x** |
| 88 MB `.sculpt` project JSON decode | **4.951 ms** (17.8 GB/s) | 65.43 ms typed / 79.5 ms `serde_json::Value` | **13.2x / 16.1x** |
| 88 MB `.sculpt` project JSON encode | **6.394 ms** (13.7 GB/s) | 83.82 ms `serde_json` | **13.1x** |
| 16k `.splat` encode | **5.57 us** | 119.38 us scalar | **21.4x** |
| 16k `.splat` decode (ARM NEON) | **5.04 us** (182 GB/s) | 54.28 us scalar | **10.8x** |
| 16k `.dsplat` v2 compact encode | **5.47 us** (12 B/splat) | 60.75 us scalar / 248 B SH3 `.ply` | **11.1x** (20.7x smaller) |
| 500k Gaussian Splat depth sort (radix) | **427.7 us** | 6.406 ms std sort / 1.288 ms radix v1 | **15.0x / 3.01x** |
| Brush dab (small footprint, ~2.5k verts) | **37.4 us** | 75.8 us (v1) | **2.03x** |
| 1.05M triangle full frame (1080p, SSAO + shadows) | **2.94 ms** (340 fps) | 8.33 ms 120 Hz budget | **65% headroom** |

---

## Architecture and modules

Coordinate convention throughout the crate: right-handed **Z-up**, millimetres (`X` right, `Y` forward, `Z` up). Shapes and extrusions rest cleanly on `Z = 0` with counter-clockwise outward winding.

| Module | Purpose |
| --- | --- |
| `src/shapes.rs` | 38 built-in watertight parametric shapes across 6 categories (`Basics`, `Rounded & Organic`, `Polyhedra & Gems`, `Architecture`, `Mechanical & Hardware`, `Containers & Everyday`) plus `extrude_profile_2d` (freehand Douglas-Peucker simplification, Chaikin smoothing, 2-opt self-crossing untangling, concave ear-clipping triangulation, miter-clamped bevels, and taper). |
| `src/text3d/` | Offline, deterministic text-to-3D pipeline: natural-language planner (`text.rs`, `lexicon.rs`, `compose.rs`, `templates.rs`), JSON shape program schema and LLM prompt (`program.rs`, `llm.rs`), precomputed SDF evaluator (`sdf.rs`), and sparse octree Surface Nets mesher (`mesher.rs`) with guaranteed manifold output. |
| `src/img3d.rs` | On-device image-to-3D mesh reconstruction from RGBA8 plus optional depth maps (item, lathe, and relief modes). |
| `src/splat.rs` | 3D Gaussian Splat pipeline: image-to-splat, mesh-to-splat, natural-language prompt editing, O(N) radix depth sorting, `.splat` (32 B/splat), `.ply` import/export, and the `.dsplat` v2 compact codec (12 B/splat). |
| `src/bvh.rs` | Parallel Morton-code HLBVH builder, fast refit for sculpting, SIMD FMA ray casting, and zero-allocation sphere vertex queries. |
| `src/brush.rs` | Nomad-grade sculpt brushes (`draw`, `clay`, `inflate`, `crease`, `pinch`, `grab`, `smooth`, `flatten`, `mask`), 4 falloff curves, 3-axis and radial symmetry (up to 16x), and reusable stroke scratch buffers. |
| `src/remesh.rs` | Voxel / SDF remeshing that rebuilds arbitrary or damaged meshes into uniform-density watertight quad meshes. |
| `src/draw3d.rs` | Freehand 3D stroke-to-watertight-tube generator (Catmull-Rom smoothing, rotation-minimizing Bishop frames, round/square/ribbon profiles, tapers, hemispherical/flat caps, and 3-axis symmetry). |
| `src/mesh.rs` | Half-edge-free indexed `SculptMesh` with parallel topology build and stamped local normal updates. |
| `src/stroke.rs` | Sculpt stroke recording with memory-bounded undo/redo history. |
| `src/format/` | `.sculpt` v1 project encoder and single-pass SIMD-accelerated streaming JSON decoder compatible byte-for-byte with Sculptor on iPad, Android, and Web. |
| `src/render/` | `wgpu` 30 PBR renderer (`Metal`, `Vulkan`, `DX12`, `WebGPU`, `WebGL2`): GGX multiscatter PBR, CPU-baked IBL (`ibl.rs`), shadow maps with Poisson PCF, half-res SSAO, AgX/ACES/Linear tone mapping, and 4x MSAA. |
| `src/session.rs` | High-level stateful viewport session combining `Renderer`, `OrbitCamera`, `Bvh`, sculpt strokes, voxel remesh, and interactive Draw 3D. |
| `src/ffi.rs` | UniFFI 0.32 bindings for Apple Swift and Android Kotlin (`EngineView`, `render_snapshot`, shapes, extrude, text-to-3D, image-to-3D, splats, and Draw 3D). |
| `src/web.rs` | `wasm-bindgen` bindings for browsers (`WebEngine` plus worker-safe free functions). |

See [`DESIGN.md`](DESIGN.md) for deeper architectural notes and [`STATUS.md`](STATUS.md) for optimization logs.

---

## Feature flags

Configured in [`Cargo.toml`](Cargo.toml):

- `default = ["threads", "render"]`
- `threads`: enables `rayon` multi-threading across BVH builds, SDF meshing, topology, splat sorting, and IBL baking.
- `render`: enables `wgpu` 30 rendering (`src/render/`, `src/session.rs`) with per-platform backend trimming (Metal on Apple, Vulkan/GLES on Android and Linux, DX12/Vulkan on Windows, WebGPU/WebGL2 on `wasm32`).
- `ffi`: enables `render` plus UniFFI 0.32 exports (`src/ffi.rs`) for Swift and Kotlin.
- `uniffi-cli`: builds the local `uniffi-bindgen` binary (`tools/uniffi-bindgen.rs`) for generating Swift and Kotlin bindings.
- `web`: enables `render` plus `wasm-bindgen` exports (`src/web.rs`) for `wasm32-unknown-unknown`.

---

## Cross-platform build instructions

### Native Rust (host tests and benchmarks)

Requires Rust 1.85+ (Rust 2024 edition).

```bash
cargo test --release --lib
cargo check --all-targets --features ffi
cargo bench --bench core
```

### WebAssembly (browser / WebGPU + WebGL2)

Requires `wasm32-unknown-unknown`, `wasm-bindgen-cli`, and `binaryen` (`wasm-opt`):

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130 --locked
./web/build.sh
```

Output lands in `web/pkg/` (`sculpt_engine.js` and `sculpt_engine_bg.wasm`, ~3.21 MB raw / ~986 KB brotli with all 38 shapes, text-to-3D, image-to-3D, Gaussian Splats, and the full WebGPU/WebGL2 renderer included).

### Apple XCFramework and SwiftPM (iOS, macOS, visionOS)

Builds a 5-slice `arm64` static XCFramework (`aarch64-apple-ios`, `aarch64-apple-ios-sim`, `aarch64-apple-darwin`, `aarch64-apple-visionos`, `aarch64-apple-visionos-sim`, ~26 MB per slice) and regenerates `apple/SculptEngine/Sources/SculptEngine/SculptEngine.swift`:

```bash
./apple/build-xcframework.sh
```

Consume via Swift Package Manager from `apple/SculptEngine`:

```swift
dependencies: [
    .package(path: "../sculpt-engine/apple/SculptEngine")
]
```

### Android NDK `.so` and UniFFI Kotlin bindings

Requires Android NDK (defaults to `$ANDROID_NDK_HOME`) and Rust Android targets (`aarch64-linux-android`, optional `x86_64-linux-android` for emulators):

```bash
rustup target add aarch64-linux-android x86_64-linux-android
ANDROID_NDK_HOME=/path/to/ndk ./android/build-android.sh ./android/out
```

Generates `libsculpt_engine.so` and `uniffi/sculpt_engine/sculpt_engine.kt`.

### Desktop Tauri (Linux, Windows, macOS CPU acceleration)

Desktop shells that already render in a WebView can link `sculpt-engine` with multi-threaded native CPU acceleration and zero GPU/windowing system link dependencies:

```toml
[dependencies]
sculpt-engine = { path = "../core/sculpt-engine", default-features = false, features = ["threads"] }
```

This unlocks full native Rayon parallelism and SIMD for `shapes`, `text3d`, `img3d`, `splat`, `draw3d`, `remesh`, `bvh`, and `brush`.

---

## Quick usage examples

### Rust

```rust
use sculpt_engine::shapes::{extrude_profile_2d_command_json, generate_shape_command_json};
use sculpt_engine::splat::{mesh_to_splat, SplatCloud};
use sculpt_engine::text3d::host::generate_json;

// 1. Generate any of the 38 built-in watertight shapes (Z-up, mm)
let gear_cmd = generate_shape_command_json("gear", r#"{"size": 26.0}"#)?;

// 2. Extrude a 2D polygon or freehand outline with bevel and taper
let l_shape = [0.0f32, 0.0, 24.0, 0.0, 24.0, 8.0, 8.0, 8.0, 8.0, 22.0, 0.0, 22.0];
let extruded_cmd = extrude_profile_2d_command_json(
    &l_shape,
    r#"{"mode": "points", "height": 14.0, "bevel": 1.2, "taper": 0.9}"#,
)?;

// 3. Offline text to 3D
let mug_json = generate_json("a red mug", r#"{"maxTriangles": 8000}"#);

// 4. Pack a Gaussian Splat cloud into 12 B/splat .dsplat v2
let cloud: SplatCloud = mesh_to_splat(
    &[0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0],
    &[0, 1, 2],
    None,
    [0.9, 0.4, 0.3, 1.0],
    false,
    1024,
)?;
let compact_bytes: Vec<u8> = cloud.to_compact_bytes();
```

### Swift (UniFFI via `SculptEngine` SwiftPM)

```swift
import SculptEngine

let catalogJson = shapeCatalogJson()
let gearCmd = try generateShape(id: "gear", paramsJson: #"{"size":24}"#)
let extrudedCmd = try extrudeProfile2d(
    pointsXy: [0, 0, 20, 0, 20, 10, 8, 10, 8, 20, 0, 20],
    optionsJson: #"{"mode":"points","height":16,"bevel":1.0,"taper":0.95}"#
)
let mugJson = textTo3d(prompt: "a red mug", optionsJson: #"{"maxTriangles":8000}"#)
let compactSplat = try splatToCompact(splatBytes: rawSplatData)
```

### Kotlin (UniFFI on Android)

```kotlin
import uniffi.sculpt_engine.extrudeProfile2d
import uniffi.sculpt_engine.generateShape
import uniffi.sculpt_engine.shapeCatalogJson
import uniffi.sculpt_engine.splatToCompact
import uniffi.sculpt_engine.textTo3d

val catalogJson = shapeCatalogJson()
val gearCmd = generateShape("gear", """{"size":24}""")
val extrudedCmd = extrudeProfile2d(
    listOf(0f, 0f, 20f, 0f, 20f, 10f, 8f, 10f, 8f, 20f, 0f, 20f),
    """{"mode":"points","height":16,"bevel":1.0,"taper":0.95}"""
)
val mugJson = textTo3d("a red mug", """{"maxTriangles":8000}""")
val compactSplat = splatToCompact(rawSplatBytes)
```

### TypeScript / WebAssembly

```ts
import init, {
  extrudeProfile2d,
  generateShape,
  shapeCatalogJson,
  splatToCompact,
  textTo3d,
} from './sculpt_engine.js';

await init();

const catalog = JSON.parse(shapeCatalogJson());
const gearCmd = JSON.parse(generateShape('gear', JSON.stringify({ size: 24 })));
const extrudedCmd = JSON.parse(
  extrudeProfile2d(
    new Float32Array([0, 0, 20, 0, 20, 10, 8, 10, 8, 20, 0, 20]),
    JSON.stringify({ mode: 'points', height: 16, bevel: 1.0, taper: 0.95 }),
  ),
);
const mug = JSON.parse(textTo3d('a red mug', JSON.stringify({ maxTriangles: 8000 })));
const compact = splatToCompact(rawSplatBytes);
```

---

## Projects using `sculpt-engine`

- **[Sculptor](https://ajadigital.co/app/sculptor.html)** ([launch in browser](https://ajadigital.co/sculptor/app/)): Friendly 3D modelling, sculpting, CAD import, and 3D print export for iPad, iPhone, Mac, Android, Windows, Linux, and the web.
- **[Dream Builder](https://ajadigital.co/app/dreambuilder.html)**: Spatial 3D sculpting and Gaussian Splats at true room scale for Apple Vision Pro, iOS AR, and Android XR.

---

## Attribution and license

- Algorithm references and academic prior art used in the SDF and Surface Nets pipeline are documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- `sculpt-engine` is open source under the **MIT License**. See [`LICENSE`](LICENSE) for the full text.
