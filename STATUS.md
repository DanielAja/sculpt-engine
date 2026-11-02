# Status (paused 2026-11-01)

Paused on the coordinator's request, so the owner can use the Mac's build capacity.

## Where things stand

- `main`: Phase 0 done. `DESIGN.md` is committed (`94d9bf2`).
- `e1-core` (this branch, WIP commit): E1 source is written but **has never been compiled or tested**. The first `cargo test` was waiting at the heavy gate when the pause came, and I stopped it before it compiled anything. Expect a round of compile fixes.
  - `src/format/`: `.sculpt` codec. `json.rs` is a Swift `JSONEncoder` compatible pretty writer (float rules checked against a real Swift probe). `de.rs` holds the serde types and Swift's repair rules. `mod.rs` has the types, `encode`, `set_mesh_command` and the tests (byte for byte re-encode of `tests/fixtures/project.sculpt`, versions, repair).
  - `src/mesh.rs`: `SculptMesh` (n-gon polys, fan triangles, CSR adjacency, Newell normals as Swift computes them, local normal update).
  - `src/bvh.rs`: binned SAH BVH, refit, ray pick, sphere query, "slack" to skip a refit on every dab; tested against brute force.
  - `src/brush.rs`: draw, inflate, smooth and flatten (the web port's Sculpt-lite maths), symmetry, Jacobi updates, rayon above 4,096 vertices; includes a determinism test across thread counts.
  - `src/stroke.rs`: `SculptTarget` (begin, `dab_points`, `drag_to`, end returning `setMesh` JSON, cancel) and `History` (200 steps plus a byte budget).
  - `benches/core.rs`: criterion benches (BVH build and refit, ray pick, dabs, 1 M triangle encode and decode).
- Not started in E1: dyntopo (`src/dyntopo.rs`, DESIGN.md §5.3).

## Installed on this Mac

- Rust 1.99.1 via the official rustup installer (`--no-modify-path`, minimal profile plus clippy and rustfmt), in `~/.cargo` and `~/.rustup`. Shell profiles were **not** changed, so prefix commands with `export PATH=$HOME/.cargo/bin:$PATH`.

## Resume

```sh
cd ~/code/sculpt-engine && git checkout e1-core
export PATH=$HOME/.cargo/bin:$PATH
/Users/daniel/code/ios27-duo-readiness/plans/heavy_gate.sh && cargo test -j 4
# fix compile errors and test failures, then:
/Users/daniel/code/ios27-duo-readiness/plans/heavy_gate.sh && cargo bench -j 4 --bench core
```

Next steps, in order:

1. Get `cargo test` green and record the bench numbers against the budgets in DESIGN.md §8.
2. Add dyntopo with its tests: manifold kept, no flipped triangles, Euler characteristic kept.
3. Merge `e1-core` into `main` with `--no-ff`.
4. Move on to E2: the wgpu 30 renderer and the winit macOS demo.

Commits use `git -c user.name=DanielAja -c user.email=51513641+DanielAja@users.noreply.github.com commit` with the session trailer. Work stays local, with no remote.
