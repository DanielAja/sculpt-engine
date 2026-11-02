// swift-tools-version: 6.0
// The shared Rust engine for Apple platforms: the xcframework plus its generated Swift bindings.
// Build the xcframework first: apple/build-xcframework.sh (from the repo root).
import PackageDescription

let package = Package(
    name: "SculptEngine",
    platforms: [.iOS(.v18), .macOS(.v15), .visionOS(.v2)],
    products: [.library(name: "SculptEngine", targets: ["SculptEngine"])],
    targets: [
        .binaryTarget(name: "sculpt_engineFFI", path: "../SculptEngine.xcframework"),
        .target(
            name: "SculptEngine",
            dependencies: ["sculpt_engineFFI"],
            linkerSettings: [.linkedFramework("Metal"), .linkedFramework("QuartzCore"), .linkedFramework("CoreGraphics"), .linkedFramework("Foundation")]
        ),
    ]
)
