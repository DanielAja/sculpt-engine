#!/bin/zsh
# Builds the snapshot check for the visionOS simulator and runs it in a throwaway simulator of our own.
# Needs apple/SculptEngine.xcframework. Run from the repo root through the heavy gate.
set -euo pipefail
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
slice=apple/SculptEngine.xcframework/xros-arm64-simulator
out=target/snapshot-check && mkdir -p $out
sdk=$(xcrun --sdk xrsimulator --show-sdk-path)
xcrun --sdk xrsimulator swiftc -O -target arm64-apple-xros2.0-simulator -sdk $sdk \
  -I $slice/Headers -Xcc -fmodule-map-file=$slice/Headers/module.modulemap \
  apple/SnapshotCheck/main.swift apple/SculptEngine/Sources/SculptEngine/SculptEngine.swift \
  $slice/libsculpt_engine.a -framework Metal -framework QuartzCore -framework CoreGraphics \
  -o $out/snapshot-check
runtime=$(xcrun simctl list runtimes | awk '/visionOS/ {print $NF}' | tail -1)
udid=$(xcrun simctl create sculpt-engine-snapshot "Apple Vision Pro" $runtime)
trap 'xcrun simctl shutdown $udid >/dev/null 2>&1 || true; xcrun simctl delete $udid' EXIT
xcrun simctl boot $udid
xcrun simctl spawn $udid $PWD/$out/snapshot-check $PWD/docs/screenshots/e4/visionos-snapshot.png
