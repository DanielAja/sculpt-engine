// Scene passes. Z up, millimetres. Every pipeline uses only the bindings its entry points touch:
// shadow and prepass use group 0 binding 0 (frame) and group 1 (object); the main pass uses all of group 0.

struct Frame {
    view_proj: mat4x4f,
    view: mat4x4f,
    proj: mat4x4f,
    inv_proj: mat4x4f,
    light_vp: mat4x4f,
    cam_pos: vec4f,
    light_dir: vec4f,     // toward the light
    light_color: vec4f,   // linear rgb x intensity
    background: vec4f,    // linear rgb
    viewport: vec4f,      // w, h, 1/w, 1/h
    params: vec4f,        // exposure, ssao on, shadows on, prefiltered max mip
    params2: vec4f,       // tonemap, encode sRGB, ground z, ssao radius
    jitter: vec4f,        // clip space sub pixel offset (xy), shadow kernel rotation (z), frame index (w)
    sh: array<vec4f, 9>,
    debug: vec4f,         // x: 1 shows raw SSAO
}

struct Object {
    model: mat4x4f,
    normal_mat: mat4x4f,
    base_color: vec4f,    // linear
    mr: vec4f,            // metallic, roughness, highlight, 0
}

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_cmp: sampler_comparison;
@group(0) @binding(3) var env_cube: texture_cube<f32>;
@group(0) @binding(4) var lin: sampler;
@group(0) @binding(5) var brdf_lut: texture_2d<f32>;
@group(0) @binding(6) var ssao_tex: texture_2d<f32>;
@group(1) @binding(0) var<uniform> O: Object;

struct VIn {
    @location(0) pos: vec3f,
    @location(1) nrm: vec3f,
    @location(2) mask: f32,
}

const PI: f32 = 3.14159265;

fn jittered(clip: vec4f) -> vec4f {
    return vec4f(clip.xy + F.jitter.xy * clip.w, clip.zw);
}

// ---- shadow (depth only, from the key light) ----

@vertex
fn vs_shadow(v: VIn) -> @builtin(position) vec4f {
    return F.light_vp * O.model * vec4f(v.pos, 1.0);
}

// ---- prepass: view space normal and depth for SSAO, in one rgba16float target that every tier can read:
// octahedral normal (rg) and linear view depth split into a half float and its remainder (ba).

struct PreOut {
    @builtin(position) clip: vec4f,
    @location(0) vn: vec3f,
    @location(1) depth: f32,
}

@vertex
fn vs_prepass(v: VIn) -> PreOut {
    var o: PreOut;
    let world = O.model * vec4f(v.pos, 1.0);
    o.clip = F.view_proj * world;
    // normal_mat's bottom row is -translation, so its w is not 0: drop it before the view transform.
    o.vn = (F.view * vec4f((O.normal_mat * vec4f(v.nrm, 0.0)).xyz, 0.0)).xyz;
    o.depth = -(F.view * world).z;
    return o;
}

fn oct_encode(n: vec3f) -> vec2f {
    let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z >= 0.0) { return p; }
    return (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
}

@fragment
fn fs_prepass(i: PreOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    var n = normalize(i.vn);
    if (!front) { n = -n; }
    let hi = unpack2x16float(pack2x16float(vec2f(i.depth, 0.0))).x;
    return vec4f(oct_encode(n), hi, i.depth - hi);
}

// ---- main PBR pass ----

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) n: vec3f,
    @location(2) mask: f32,
}

@vertex
fn vs_main(v: VIn) -> VOut {
    var o: VOut;
    let world = O.model * vec4f(v.pos, 1.0);
    o.world = world.xyz;
    o.n = (O.normal_mat * vec4f(v.nrm, 0.0)).xyz;
    o.mask = v.mask;
    o.clip = jittered(F.view_proj * world);
    return o;
}

const SHADOW_DISC: array<vec2f, 12> = array<vec2f, 12>(
    vec2f(-0.326, -0.406), vec2f(-0.840, -0.074), vec2f(-0.696, 0.457), vec2f(-0.203, 0.621),
    vec2f(0.962, -0.195), vec2f(0.473, -0.480), vec2f(0.519, 0.767), vec2f(0.185, -0.893),
    vec2f(0.507, 0.064), vec2f(0.896, 0.412), vec2f(-0.322, -0.933), vec2f(-0.792, -0.598),
);

fn shadow_at(world: vec3f, n: vec3f) -> f32 {
    if (F.params.z < 0.5) { return 1.0; }
    // Normal offset scaled by the texel size in world units keeps acne away without peter panning.
    let dim = vec2f(textureDimensions(shadow_map));
    let texel = 1.0 / dim;
    let texel_world = 2.0 * texel.x / F.light_vp[0][0];
    let p = F.light_vp * vec4f(world + n * (texel_world * 1.5), 1.0);
    let uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || p.z > 1.0) { return 1.0; }
    let a = F.jitter.z;
    let ca = cos(a);
    let sa = sin(a);
    let step = texel * 2.0;
    let rot_step = mat2x2f(ca * step.x, sa * step.y, -sa * step.x, ca * step.y);
    let ref_z = p.z - 0.0015;
    var s = 0.0;
    // 12 tap rotated Poisson disc (radius ~2 texels).
    for (var k = 0; k < 12; k++) {
        s += textureSampleCompareLevel(shadow_map, shadow_cmp, uv + rot_step * SHADOW_DISC[k], ref_z);
    }
    return s * (1.0 / 12.0);
}

// SH basis constants are pre-multiplied into `F.sh` on the CPU.
fn sh_irradiance(n: vec3f) -> vec3f {
    let x = n.x; let y = n.y; let z = n.z;
    var c = F.sh[0].xyz;
    c += F.sh[1].xyz * y + F.sh[2].xyz * z + F.sh[3].xyz * x;
    c += F.sh[4].xyz * (x * y) + F.sh[5].xyz * (y * z);
    c += F.sh[6].xyz * (3.0 * z * z - 1.0) + F.sh[7].xyz * (x * z);
    c += F.sh[8].xyz * (x * x - y * y);
    return max(c, vec3f(0.0));
}

fn ssao_at(frag: vec4f) -> f32 {
    if (F.params.y < 0.5) { return 1.0; }
    let uv = frag.xy * F.viewport.zw;
    return textureSampleLevel(ssao_tex, lin, uv, 0.0).r;
}

fn d_ggx(nh: f32, a2: f32) -> f32 {
    let d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn v_smith(nv: f32, nl: f32, a2: f32) -> f32 {
    let one_m_a2 = 1.0 - a2;
    let gv = nl * sqrt(nv * nv * one_m_a2 + a2);
    let gl = nv * sqrt(nl * nl * one_m_a2 + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

@fragment
fn fs_main(i: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    let ao = ssao_at(i.clip);
    if (F.debug.x > 0.5) { return vec4f(vec3f(ao), 1.0); }
    var n = normalize(i.n);
    if (!front) { n = -n; }
    let v = normalize(F.cam_pos.xyz - i.world);
    let l = F.light_dir.xyz;
    let base = O.base_color.rgb * mix(1.0, 0.32, clamp(i.mask, 0.0, 1.0));
    let metal = O.mr.x;
    let rough = clamp(O.mr.y, 0.045, 1.0);
    let a = rough * rough;
    let nv = max(dot(n, v), 1e-4);
    let nl = dot(n, l);
    let f0 = mix(vec3f(0.04), base, metal);
    let diffuse = base * (1.0 - metal);

    let lut = textureSampleLevel(brdf_lut, lin, vec2f(nv, rough), 0.0).rg;
    // Multiple scattering energy compensation (Fdez-Aguera): rough metals stay bright.
    let ess = lut.x + lut.y;
    let energy = 1.0 + f0 * (1.0 / max(ess, 1e-3) - 1.0);

    var direct = vec3f(0.0);
    // Skip direct BRDF and 12-tap shadow PCF on surfaces facing away from the key light.
    if (nl > 0.0) {
        let h = normalize(v + l);
        let nh = max(dot(n, h), 0.0);
        let vh = max(dot(v, h), 0.0);
        let om = 1.0 - vh;
        let om2 = om * om;
        let f = f0 + (1.0 - f0) * (om2 * om2 * om);
        let a2 = a * a;
        let spec_direct = d_ggx(nh, a2) * v_smith(nv, nl, a2) * f * energy;
        let shadow = shadow_at(i.world, n);
        direct = (diffuse * (1.0 / PI) * (1.0 - f) + spec_direct) * F.light_color.rgb * (nl * shadow);
    }

    let r = reflect(-v, n);
    let pre = textureSampleLevel(env_cube, lin, r, rough * F.params.w).rgb;
    // Horizon occlusion: reflections that point into the surface fade out.
    let horizon = clamp(1.0 + dot(r, n), 0.0, 1.0);
    let spec_ibl = pre * (f0 * lut.x + lut.y) * energy * (horizon * horizon);
    let ambient = (diffuse * sh_irradiance(n) + spec_ibl) * ao;

    var c = direct + ambient;
    // Selection highlight: a soft fresnel rim in Sculptor's accent orange.
    if (O.mr.z > 0.0) {
        let om = 1.0 - nv;
        c += vec3f(1.0, 0.45, 0.1) * O.mr.z * (om * om * om * 0.6);
    }
    return vec4f(c, 1.0);
}

// ---- ground: a shadow catcher under the scene, darkening the background (premultiplied, rgb 0) ----

struct GOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
}

@vertex
fn vs_ground(@builtin(vertex_index) vi: u32) -> GOut {
    // One big quad (two triangles) centred under the camera target.
    let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
    let extent = 4000.0;
    let w = vec3f(corners[vi] * extent, F.params2.z);
    var o: GOut;
    o.world = w;
    o.clip = jittered(F.view_proj * vec4f(w, 1.0));
    return o;
}

@fragment
fn fs_ground(i: GOut) -> @location(0) vec4f {
    let s = shadow_at(i.world, vec3f(0.0, 0.0, 1.0));
    let ao = ssao_at(i.clip);
    let dark = clamp((1.0 - s) * 0.45 + (1.0 - ao) * 0.6, 0.0, 0.8);
    return vec4f(0.0, 0.0, 0.0, dark);
}

// ---- lines: grid and host overlays, unlit, premultiplied ----

struct LIn {
    @location(0) pos: vec3f,
    @location(1) color: vec4f,
}
struct LOut {
    @builtin(position) clip: vec4f,
    @location(0) color: vec4f,
}

@vertex
fn vs_line(v: LIn) -> LOut {
    var o: LOut;
    o.clip = jittered(F.view_proj * vec4f(v.pos, 1.0));
    o.color = v.color;
    return o;
}

@fragment
fn fs_line(i: LOut) -> @location(0) vec4f {
    return vec4f(i.color.rgb * i.color.a, i.color.a);
}
