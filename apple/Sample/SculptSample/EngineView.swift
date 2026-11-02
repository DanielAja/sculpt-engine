import SwiftUI
import UIKit
import QuartzCore
import SculptEngine

/// Holds the engine for SwiftUI (created by the view once it has a layer).
@Observable @MainActor
final class EngineHost {
    var engine: EngineView?
    var status = "starting"
    var brush: (String, Float, Bool) = ("draw", 6, true)
}

struct EngineViewRepresentable: UIViewRepresentable {
    let host: EngineHost
    func makeUIView(context: Context) -> MetalHostView { MetalHostView(host: host) }
    func updateUIView(_ view: MetalHostView, context: Context) {}
}

/// A UIView backed by a CAMetalLayer that the engine renders into.
final class MetalHostView: UIView {
    override class var layerClass: AnyClass { CAMetalLayer.self }
    private let host: EngineHost
    private var link: CADisplayLink?
    private var mode: Mode = .none
    private var last: CGPoint = .zero
    private var frames = 0
    private var since = CACurrentMediaTime()
    private enum Mode { case none, sculpt, orbit }

    init(host: EngineHost) {
        self.host = host
        super.init(frame: .zero)
        isMultipleTouchEnabled = false
    }
    required init?(coder: NSCoder) { fatalError("not used") }

    private var pixelSize: (UInt32, UInt32) {
        let s = window?.screen.scale ?? 2
        return (UInt32(max(1, bounds.width * s)), UInt32(max(1, bounds.height * s)))
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        guard window != nil, host.engine == nil, bounds.width > 0 else { return }
        start()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        layer.contentsScale = window?.screen.scale ?? 2
        if host.engine == nil, window != nil { start() }
        let (w, h) = pixelSize
        host.engine?.resize(width: w, height: h)
    }

    private func start() {
        let (w, h) = pixelSize
        let ptr = UInt64(UInt(bitPattern: Unmanaged.passUnretained(layer).toOpaque()))
        do {
            let e = try EngineView.withMetalLayer(layer: ptr, width: w, height: h)
            if let url = Bundle.main.url(forResource: "project", withExtension: "sculpt") {
                try e.openProject(bytes: try Data(contentsOf: url))
            }
            let b = host.brush
            e.setBrush(kind: b.0, radius: b.1, strength: 0.6, symmetryX: b.2)
            host.engine = e
            host.status = e.backend()
        } catch {
            host.status = "engine error: \(error)"
            return
        }
        let link = CADisplayLink(target: self, selector: #selector(step))
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 60, maximum: 120, preferred: 120)
        link.add(to: .main, forMode: .common)
        self.link = link
        if ProcessInfo.processInfo.arguments.contains("-demoStroke") {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak self] in self?.demoStroke() }
        }
    }

    @objc private func step() {
        host.engine?.render()
        frames += 1
        let now = CACurrentMediaTime()
        if now - since > 1, let e = host.engine {
            host.status = "\(e.backend()) | \(String(format: "%.1f", (now - since) * 1000 / Double(frames))) ms/frame" + demoNote
            frames = 0; since = now
        }
    }

    /// A stroke across the ball (the smooth object right of the cube), as a finger would drive it.
    private var demoNote = ""
    private func demoStroke() {
        guard let e = host.engine else { return }
        let (w, h) = pixelSize
        e.setBrush(kind: "draw", radius: 5, strength: 1, symmetryX: false)
        let (x0, y0) = (Float(w) * 0.58, Float(h) * 0.46)
        guard e.sculptBegin(x: x0, y: y0, pressure: 1) else { demoNote = " | demo stroke missed"; return }
        for i in 0...40 { e.sculptMove(x: x0 + Float(w) * 0.005 * Float(i), y: y0 + Float(h) * 0.0015 * Float(i), pressure: 1) }
        demoNote = " | stroke: setMesh \((e.sculptEnd() ?? "").count) bytes"
    }

    private func point(_ t: UITouch) -> (Float, Float, Float) {
        let p = t.location(in: self)
        let s = window?.screen.scale ?? 2
        let pressure = t.type == .pencil ? Float(t.force / max(t.maximumPossibleForce, 1)) : 1
        return (Float(p.x * s), Float(p.y * s), pressure)
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let t = touches.first, let e = host.engine else { return }
        let (x, y, pr) = point(t)
        mode = e.sculptBegin(x: x, y: y, pressure: pr) ? .sculpt : .orbit
        last = t.location(in: self)
    }
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let t = touches.first, let e = host.engine else { return }
        switch mode {
        case .sculpt:
            let (x, y, pr) = point(t)
            e.sculptMove(x: x, y: y, pressure: pr)
        case .orbit:
            let p = t.location(in: self)
            e.orbit(dx: Float(p.x - last.x), dy: Float(p.y - last.y))
            last = p
        case .none: break
        }
    }
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        // A Swift host with SculptorCore would pass this command to Document.perform (label "Sculpt").
        if mode == .sculpt { _ = host.engine?.sculptEnd() }
        mode = .none
    }
    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        if mode == .sculpt { host.engine?.sculptCancel() }
        mode = .none
    }
}
