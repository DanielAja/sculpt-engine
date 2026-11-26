#!/bin/zsh
# Builds the wasm module into web/pkg and prints its sizes. Run from the repo root, through the heavy gate:
#   ~/code/ios27-duo-readiness/plans/heavy_gate.sh && web/build.sh
set -e
export PATH=$HOME/.cargo/bin:$PATH
cargo build -j 4 --lib --target wasm32-unknown-unknown --no-default-features --features web --profile wasm
wasm-bindgen --target web --no-typescript --out-dir web/pkg target/wasm32-unknown-unknown/wasm/sculpt_engine.wasm
# Measured on the raw wasm-bindgen output: base 2,626,745 B / 825,781 br  ->  2,610,457 B / 818,816 br (-0.6% raw, -0.8% br, -1.4% gz).
wasm-opt -Oz --converge --inlining-optimizing --merge-blocks --simplify-locals --vacuum --dce --reorder-functions --reorder-locals --reorder-globals --low-memory-unused --zero-filled-memory --strip-debug --strip-producers --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals \
  web/pkg/sculpt_engine_bg.wasm -o web/pkg/sculpt_engine_bg.wasm
raw=$(stat -f %z web/pkg/sculpt_engine_bg.wasm)
br=$(brotli -c -q 11 web/pkg/sculpt_engine_bg.wasm | wc -c | tr -d ' ')
gz=$(gzip -9 -c web/pkg/sculpt_engine_bg.wasm | wc -c | tr -d ' ')
echo "wasm: $raw bytes raw, $gz gzip, $br brotli"
