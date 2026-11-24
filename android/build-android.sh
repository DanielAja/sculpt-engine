#!/bin/zsh
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
NDK="${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/27.2.12479018}"
BIN="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin"
OUT="${1:-../../../dreambuilder.vision/android-xr/app/src/main}"
mkdir -p "$OUT/jniLibs/arm64-v8a" "$OUT/java"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$BIN/aarch64-linux-android34-clang"
export CC_aarch64_linux_android="$BIN/aarch64-linux-android34-clang"
export AR_aarch64_linux_android="$BIN/llvm-ar"
cargo build -j 4 --lib --release --target aarch64-linux-android --features ffi
cp target/aarch64-linux-android/release/libsculpt_engine.so "$OUT/jniLibs/arm64-v8a/libsculpt_engine.so"
cargo build -j 4 --release --lib --features ffi
cargo run -j 4 --release --features uniffi-cli --bin uniffi-bindgen -- generate --library target/release/libsculpt_engine.dylib --language kotlin --out-dir "$OUT/java"
du -h "$OUT/jniLibs/arm64-v8a/libsculpt_engine.so"
