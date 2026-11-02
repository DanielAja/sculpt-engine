import SwiftUI

/// Sample host for the shared Rust engine: the app owns the layer and the frame clock, the engine draws
/// and sculpts. Launch arguments: `-demoStroke` runs a scripted stroke after loading (for screenshots).
@main
struct SculptSampleApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @State private var brush = "draw"
    @State private var radius: Float = 6
    @State private var symmetry = true
    @State private var status = ""
    @State private var host = EngineHost()

    var body: some View {
        ZStack(alignment: .top) {
            EngineViewRepresentable(host: host).ignoresSafeArea()
            HStack(spacing: 12) {
                Picker("Brush", selection: $brush) {
                    Text("Draw").tag("draw"); Text("Inflate").tag("inflate")
                    Text("Smooth").tag("smooth"); Text("Flatten").tag("flatten")
                }
                .pickerStyle(.segmented).frame(maxWidth: 360)
                Slider(value: $radius, in: 1...20) { Text("Radius") }.frame(maxWidth: 160)
                Toggle("X symmetry", isOn: $symmetry).fixedSize()
                Button("Undo") { host.engine?.undo() }
                Button("Redo") { host.engine?.redo() }
                Text(host.status).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            .padding(12)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
            .padding()
        }
        .onChange(of: [brush, "\(radius)", "\(symmetry)"], initial: true) {
            host.engine?.setBrush(kind: brush, radius: radius, strength: 0.6, symmetryX: symmetry)
            host.brush = (brush, radius, symmetry)
        }
    }
}
