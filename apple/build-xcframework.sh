#!/bin/zsh
# Builds apple/SculptEngine.xcframework (iOS, iOS simulator, macOS, visionOS, visionOS simulator; arm64)
# and the Swift bindings in apple/SculptEngine/Sources/SculptEngine. Run from the repo root:
#   ~/code/ios27-duo-readiness/plans/heavy_gate.sh && apple/build-xcframework.sh
set -euo pipefail
export PATH=$HOME/.cargo/bin:$PATH
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer   # release Xcode only, never a beta
targets=(aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin aarch64-apple-visionos aarch64-apple-visionos-sim)
for t in $targets; do
  echo "== $t"
  cargo rustc -j 4 --lib --release --target $t --features ffi --crate-type staticlib
done

# Bindings come from the host build's library metadata (same code, same API on every slice).
cargo build -j 4 --release --lib --features ffi
gen=target/uniffi-swift
rm -rf $gen && mkdir -p $gen
cargo run -j 4 --release --features uniffi-cli --bin uniffi-bindgen -- \
  generate --library target/release/libsculpt_engine.dylib --language swift --out-dir $gen
cp $gen/sculpt_engine.swift apple/SculptEngine/Sources/SculptEngine/SculptEngine.swift

headers=target/xcframework-headers
rm -rf $headers && mkdir -p $headers
cp $gen/sculpt_engineFFI.h $headers/
cp $gen/sculpt_engineFFI.modulemap $headers/module.modulemap

out=apple/SculptEngine.xcframework
rm -rf $out
args=()
for t in $targets; do args+=(-library target/$t/release/libsculpt_engine.a -headers $headers); done
xcodebuild -create-xcframework $args -output $out
for t in $targets; do printf '%-28s %s\n' $t "$(du -h target/$t/release/libsculpt_engine.a | cut -f1)"; done
