// visionOS offscreen snapshot check (Dream Builder's SnapshotRenderer path, DESIGN.md section 9 option (a)).
// Built for the visionOS simulator by apple/SnapshotCheck/run.sh and run with `simctl spawn`.
// Renders a cube and a sphere-ish octahedron from loose triangle buffers, writes the PNG to argv[1].
import Foundation

/// Flat shaded cube the way Dream Builder's RecipeMesher builds flat parts: each face has its own corners,
/// so the engine's per-vertex normals come out flat. (Shared corners would be shaded smooth.)
func cube(_ s: Float) -> ([Float], [UInt32]) {
    let c: [[Float]] = [[-s,-s,-s],[s,-s,-s],[s,s,-s],[-s,s,-s],[-s,-s,s],[s,-s,s],[s,s,s],[-s,s,s]]
    let quads: [[Int]] = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    var p: [Float] = [], idx: [UInt32] = []
    for q in quads {
        let base = UInt32(p.count / 3)
        for v in q { p += c[v] }
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    }
    return (p, idx)
}

func translation(_ x: Float, _ y: Float, _ z: Float) -> [Float] {
    [1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]
}

let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "snapshot.png"
let (cp, ci) = cube(10)
let objects = [
    SnapshotObject(positions: cp, indices: ci, world: translation(-14, 0, 10), baseColor: [0.16, 0.62, 0.56, 1], metallic: 0, roughness: 0.6),
    SnapshotObject(positions: cp, indices: ci, world: translation(14, 0, 10), baseColor: [0.91, 0.71, 0.30, 1], metallic: 1, roughness: 0.25),
]
let camera = #"{"distance":110,"fovDegrees":45,"isOrthographic":false,"pitchDegrees":25,"target":[0,0,10],"yawDegrees":30}"#
let start = Date()
do {
    let png = try renderSnapshot(objects: objects, cameraJson: camera, width: 640, height: 400)
    try png.write(to: URL(fileURLWithPath: out))
    print("snapshot ok: \(png.count) bytes in \(Int(Date().timeIntervalSince(start) * 1000)) ms -> \(out)")
} catch {
    print("snapshot failed: \(error)")
    exit(1)
}
