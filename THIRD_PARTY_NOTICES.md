# Third-party notices and attribution: text to 3D

This file credits the work that Sculptor's and Dream Builder's text to 3D draws on (sculpt-engine `src/text3d`, sculptor.web `src/text3d`, SculptorCore `TextTo3D`, DreamCore `TextTo3D.swift`). The same file is kept in each of those repos. The full survey with citations is in sculptor.web `docs/text-to-3d-research.md`.

No third-party source files, model weights or datasets are included. Where an algorithm or formula was adapted, it was rewritten here; the licence of the original is listed so the credit is clear.

## Adapted formulas and algorithms (in the code)

| Work | Author(s) | Licence | Used for | Where |
|---|---|---|---|---|
| [Distance functions](https://iquilezles.org/articles/distfunctions/), [2D distance functions](https://iquilezles.org/articles/distfunctions2d/), [smooth minimum](https://iquilezles.org/articles/smin/) | Inigo Quilez | MIT (as published with the code samples; Copyright (c) Inigo Quilez) | Sphere, rounded box, rounded cylinder, capped cone, torus, capsule, ellipsoid, pyramid, polygon, extrusion and revolution formulas; polynomial smooth min and max | `sculpt-engine/src/text3d/sdf.rs` |
| Surface Nets: "Constrained Elastic Surface Nets" (MERL TR99-24, 1998) | Sarah F. F. Gibson | published algorithm | Isosurface meshing of the shape programs | `sculpt-engine/src/text3d/mesher.rs` |
| [Smooth Voxel Terrain, part 2](https://0fps.net/2012/07/12/smooth-voxel-terrain-part-2/) | Mikola Lysenko | article; reference code MIT | Naive Surface Nets formulation | same |
| [fast-surface-nets-rs](https://github.com/bonsairobo/fast-surface-nets-rs) | Duncan Fairbanks | MIT OR Apache-2.0 | Read as a reference for the quad winding and grid layout (not a dependency; no code copied) | same |
| "Dual Marching Cubes" (IEEE Visualization 2004) | Gregory M. Nielson | published algorithm | One vertex per surface component in a cell, which keeps thin walls manifold | same |
| "The asymptotic decider" (IEEE Visualization 1991) | Gregory M. Nielson, Bernd Hamann | published algorithm | Resolving saddle faces consistently between neighbouring cells | same |

## Ideas and prior art (no code used)

| Project | Licence | What we took |
|---|---|---|
| [CSGNet](https://github.com/Hippogriff/CSGNet) (Sharma et al., CVPR 2018) | MIT | Shapes as CSG programs |
| [ShapeAssembly](https://github.com/rkjones4/ShapeAssembly) (Jones et al., SIGGRAPH Asia 2020) | all rights reserved (idea only) | Part-based shape programs with symmetry |
| [CADmium](https://github.com/chandar-lab/CADmium) (2025) | MIT | LLMs writing JSON shape sequences instead of code |
| [OpenSCAD](https://github.com/openscad/openscad) | GPL-2.0 (idea only) | A small CSG vocabulary LLMs already know |
| [fogleman/sdf](https://github.com/fogleman/sdf) | MIT | A compact SDF API with smooth booleans |
| [libfive](https://github.com/libfive/libfive) | MPL-2.0 (core) | F-rep modelling; reference for future sharp-edge meshing |
| [Infinigen](https://github.com/princeton-vl/infinigen) | BSD-3-Clause | Seeded procedural generators |
| [BlenderLLM](https://github.com/FreedomIntelligence/BlenderLLM) | Apache-2.0 | Small local LLMs emitting primitives plus modifiers |
| [LLaMA-Mesh](https://github.com/nv-tlabs/LLaMA-Mesh) (Wang et al., 2024) | NVIDIA non-commercial (idea only) | 3D as text for LLMs |
| [Holodeck](https://github.com/allenai/Holodeck), [LayoutGPT](https://github.com/UCSB-AI/LayoutGPT) | Apache-2.0, MIT | Spatial relations turned into a layout |
| [PartCrafter](https://github.com/wgsxm/PartCrafter), [Roblox Cube / CubePart](https://github.com/Roblox/cube) | MIT; research-only (idea only) | Part-aware output: one object per part |
| [SPAR3D](https://github.com/Stability-AI/stable-point-aware-3d), [TRELLIS](https://github.com/microsoft/TRELLIS) | Stability community; MIT | An editable intermediate before the mesh (here, the program) |
| [Shap-E](https://github.com/openai/shap-e), [Point-E](https://github.com/openai/point-e), [TripoSR](https://github.com/VAST-AI-Research/TripoSR), [TripoSG](https://github.com/VAST-AI-Research/TripoSG) | MIT | Surveyed as optional future backends; not used |
| Dream Builder M5 shape grammar (AJA Digital) | ours | Archetype families for unknown nouns; the noun seeds the proportions |

## Optional runtimes the apps can call (not bundled)

| Runtime | Terms | Notes |
|---|---|---|
| [Chrome built-in AI, Prompt API](https://developer.chrome.com/docs/ai/prompt-api) (Gemini Nano) | Google's terms and Generative AI prohibited use policy | Used only if the person's browser provides it and they tick the box; the model is the browser's, not ours |
| [Apple Foundation Models](https://developer.apple.com/documentation/foundationmodels) | Apple SDK terms | On-device model on Apple platforms |
| [WebLLM](https://github.com/mlc-ai/web-llm) with [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B) | Apache-2.0 | Planned follow-up; not included |

## Projects reviewed and deliberately not used

Their licences don't fit a free, open, worldwide app: Hunyuan3D 1, 2, 2.1, Omni and Part (Tencent licence excludes the EU, UK and South Korea), Stable Fast 3D and SPAR3D (Stability community licence, revenue cap), MeshAnything and FastMesh (S-Lab non-commercial), PartPacker, PartField and LLaMA-Mesh code (NVIDIA non-commercial), OpenLRM and Zero123++ weights (CC-BY-NC), CAD-Recode, Text2CAD and MeshCoder weights (CC-BY-NC or NC-SA), Wonder3D weights and Era3D code (AGPL), SuGaR, 2DGS and Gaussian Opacity Fields (Inria non-commercial), and ManifoldPlus (no licence).
