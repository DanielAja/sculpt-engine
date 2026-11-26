// Fullscreen passes: SSAO (half resolution), its blur, and tone mapping onto the output.
// Each entry point uses its own group 1 binding numbers so one module serves all three pipelines.

struct Frame {
    view_proj: mat4x4f,
    view: mat4x4f,
    proj: mat4x4f,
    inv_proj: mat4x4f,
    light_vp: mat4x4f,
    cam_pos: vec4f,
    light_dir: vec4f,
    light_color: vec4f,
    background: vec4f,
    viewport: vec4f,
    params: vec4f,
    params2: vec4f,
    jitter: vec4f,
    sh: array<vec4f, 9>,
    debug: vec4f,         // x: 1 shows raw SSAO
}

@group(0) @binding(0) var<uniform> F: Frame;
@group(1) @binding(0) var pre: texture_2d<f32>;   // octahedral normal (rg), view depth hi + lo (ba)
@group(1) @binding(2) var ssao_raw: texture_2d<f32>;
@group(1) @binding(3) var hdr: texture_2d<f32>;

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn oct_decode(e: vec2f) -> vec3f {
    var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) { n = vec3f((1.0 - abs(n.yx)) * select(vec2f(-1.0), vec2f(1.0), n.xy >= vec2f(0.0)), n.z); }
    return normalize(n);
}

/// View space position of a full resolution pixel from an already-loaded prepass texel; z = 0 marks background.
fn view_pos_from(px: vec2i, t: vec4f, inv_full: vec2f) -> vec3f {
    let d = t.b + t.a;
    if (d <= 0.0) { return vec3f(0.0); }
    let uv = (vec2f(px) + 0.5) * inv_full;
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let q = F.inv_proj * vec4f(ndc, 1.0, 1.0);
    let p = q.xyz / q.w;
    if (F.proj[3][3] > 0.5) { return vec3f(p.xy, -d); }   // orthographic: parallel rays
    return p * (d / -p.z);
}

fn view_pos(px: vec2i, inv_full: vec2f) -> vec3f {
    return view_pos_from(px, textureLoad(pre, px, 0), inv_full);
}

fn hash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

// Precomputed golden-angle hemisphere kernel (cos(phi)*r, sin(phi)*r, z) * scale for 12 SSAO taps;
// eliminates 24 trig + 12 sqrt calls per half-res fragment.
const SSAO_KERNEL: array<vec3f, 12> = array<vec3f, 12>(
    vec3f(0.1473990, 0.0000000, 0.0278125),
    vec3f(-0.1111195, 0.1017946, 0.0399501),
    vec3f(0.0143431, -0.1634335, 0.0567853),
    vec3f(0.1133834, 0.1478884, 0.0808268),
    vec3f(-0.2126246, -0.0376101, 0.1145833),
    vec3f(0.2113886, -0.1344685, 0.1605635),
    vec3f(-0.0745396, 0.2772851, 0.2212760),
    vec3f(-0.1482020, -0.2853527, 0.2992296),
    vec3f(0.3267351, 0.1193225, 0.3969329),
    vec3f(-0.3298772, 0.1361693, 0.5168945),
    vec3f(0.1407429, -0.3007612, 0.6616233),
    vec3f(0.0682273, 0.2175175, 0.8336277),
);

@fragment
fn fs_ssao(@builtin(position) frag: vec4f) -> @location(0) vec4f {
    let full = F.viewport.xy;
    let inv_full = F.viewport.zw;
    let px = vec2i(frag.xy * 2.0);
    let center = textureLoad(pre, px, 0);
    let p = view_pos_from(px, center, inv_full);
    if (p.z >= 0.0) { return vec4f(1.0); }
    let n = oct_decode(center.rg);
    let radius = F.params2.w;
    let bias = radius * 0.05;
    // Random rotation per pixel (and per frame while accumulating) around the normal.
    let ang = (hash(frag.xy) + F.jitter.w * 0.618) * 6.2831853;
    var t = normalize(cross(n, select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.x) > 0.9)));
    let b = cross(n, t);
    t = t * cos(ang) + b * sin(ang);
    let b2 = cross(n, t);
    let tr = t * radius;
    let br = b2 * radius;
    let nr = n * radius;
    var occ = 0.0;
    for (var k = 0; k < 12; k++) {
        let ks = SSAO_KERNEL[k];
        let s = p + tr * ks.x + br * ks.y + nr * ks.z;
        let c = F.proj * vec4f(s, 1.0);
        let inv_w = 1.0 / c.w;
        let suv = vec2f(c.x * inv_w * 0.5 + 0.5, 0.5 - c.y * inv_w * 0.5);
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) { continue; }
        let sp = view_pos(vec2i(suv * full), inv_full);
        if (sp.z >= 0.0) { continue; }
        let range = smoothstep(0.0, 1.0, radius / max(abs(p.z - sp.z), 1e-3));
        occ += select(0.0, 1.0, sp.z >= s.z + bias) * range;
    }
    return vec4f(1.0 - occ * (1.0 / 12.0));
}

@fragment
fn fs_blur(@builtin(position) frag: vec4f) -> @location(0) vec4f {
    let size = vec2i(textureDimensions(ssao_raw));
    var s = 0.0;
    for (var y = -2; y < 2; y++) {
        for (var x = -2; x < 2; x++) {
            s += textureLoad(ssao_raw, clamp(vec2i(frag.xy) + vec2i(x, y), vec2i(0), size - 1), 0).r;
        }
    }
    return vec4f(s / 16.0);
}

// AgX (Troy Sobotka's curve, Benjamin Wrensch's polynomial fit), default look.
fn agx(c0: vec3f) -> vec3f {
    let inset = mat3x3f(vec3f(0.842479, 0.0423282, 0.0423756), vec3f(0.0784336, 0.878468, 0.0784336), vec3f(0.0792237, 0.0791661, 0.879142));
    let outset = mat3x3f(vec3f(1.19688, -0.0528969, -0.0529716), vec3f(-0.0980209, 1.15190, -0.0980435), vec3f(-0.0990297, -0.0989612, 1.15107));
    var c = inset * c0;
    c = clamp(log2(max(c, vec3f(1e-10))), vec3f(-12.47393), vec3f(4.026069));
    c = (c + 12.47393) / 16.5;
    let x2 = c * c;
    let x4 = x2 * x2;
    c = 15.5 * x4 * x2 - 40.14 * x4 * c + 31.96 * x4 - 6.868 * x2 * c + 0.4298 * x2 + 0.1191 * c - 0.00232;
    // Back from AgX's sRGB-like display encoding to linear, so the output stage encodes once.
    c = outset * c;
    return pow(max(c, vec3f(0.0)), vec3f(2.2));
}

// ACES fitted (Stephen Hill).
fn aces(c0: vec3f) -> vec3f {
    let i = mat3x3f(vec3f(0.59719, 0.07600, 0.02840), vec3f(0.35458, 0.90834, 0.13383), vec3f(0.04823, 0.01566, 0.83777));
    let o = mat3x3f(vec3f(1.60475, -0.10208, -0.00327), vec3f(-0.53108, 1.10813, -0.07276), vec3f(-0.07367, -0.00605, 1.07602));
    let v = i * c0;
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return clamp(o * (a / b), vec3f(0.0), vec3f(1.0));
}

fn srgb_encode(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

@fragment
fn fs_tonemap(@builtin(position) frag: vec4f) -> @location(0) vec4f {
    let s = textureLoad(hdr, vec2i(frag.xy), 0);
    // Premultiplied scene over the flat background: background x (1 - a) + tonemap(rgb / a) x a.
    var col = F.background.rgb * (1.0 - s.a);
    if (s.a > 1e-4) {
        let lin = s.rgb / s.a * F.params.x;
        var m: vec3f;
        let op = i32(F.params2.x + 0.5);
        if (op == 0) { m = agx(lin); } else if (op == 1) { m = aces(lin); } else { m = clamp(lin, vec3f(0.0), vec3f(1.0)); }
        col += m * s.a;
    }
    if (F.params2.y > 0.5) { col = srgb_encode(clamp(col, vec3f(0.0), vec3f(1.0))); }
    return vec4f(col, 1.0);
}
