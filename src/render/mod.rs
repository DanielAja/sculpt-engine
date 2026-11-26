//! wgpu renderer: shadow map, SSAO, physically based forward pass with IBL, MSAA, tone mapping.
//! See DESIGN.md section 4. The host owns the surface; `Renderer::render` draws into any texture view of
//! the renderer's output format (a surface frame, or an offscreen texture for snapshots and tests).

pub mod ibl;

use crate::camera::OrbitCamera;
use crate::format::{Material, ProjectFile};
use crate::mesh::SculptMesh;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec3A, Vec4};
use std::sync::{Arc, OnceLock};
use wgpu::util::DeviceExt;

/// What the adapter can do (DESIGN.md section 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Compute shaders and storage buffers (Metal, Vulkan, DX12, WebGPU).
    Full,
    /// Raster only (WebGL2, old GLES).
    Downlevel,
}

pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub tier: Tier,
}

impl Gpu {
    /// Picks an adapter (compatible with `surface` when given) and opens a device.
    pub async fn new(instance: wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<Arc<Gpu>, String> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: surface, ..Default::default() })
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let caps = adapter.get_downlevel_capabilities();
        let tier = if caps.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) { Tier::Full } else { Tier::Downlevel };
        // Ask for the modest downlevel limits (the iOS simulator's Metal device, for one, is below wgpu's
        // defaults: 15 inter-stage variables); the shaders stay well inside them.
        let limits = if tier == Tier::Full { wgpu::Limits::downlevel_defaults() } else { wgpu::Limits::downlevel_webgl2_defaults() };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor { label: Some("sculpt-engine"), required_limits: limits.using_resolution(adapter.limits()), ..Default::default() })
            .await
            .map_err(|e| format!("no GPU device: {e}"))?;
        Ok(Arc::new(Gpu { instance, adapter, device, queue, tier }))
    }

    /// Headless device for snapshots and tests.
    pub async fn headless() -> Result<Arc<Gpu>, String> {
        Gpu::new(wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env()), None).await
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tonemap {
    AgX = 0,
    Aces = 1,
    /// Android's current look (linear, clamped); pair with `ssao: false` for the "Studio" preset.
    Linear = 2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderSettings {
    pub tonemap: Tonemap,
    pub exposure: f32,
    pub ssao: bool,
    pub shadows: bool,
    pub grid: bool,
    /// sRGB.
    pub background: [f32; 3],
    /// Key light intensity (linear multiplier on white).
    pub key_light: f32,
    pub ssao_radius_mm: f32,
    /// Shows raw SSAO instead of shading (diagnostics).
    pub debug_ao: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        RenderSettings { tonemap: Tonemap::AgX, exposure: 1.0, ssao: true, shadows: true, grid: true, background: [0.93, 0.93, 0.94], key_light: 2.6, ssao_radius_mm: 6.0, debug_ao: false }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameU {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    light_vp: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    light_dir: [f32; 4],
    light_color: [f32; 4],
    background: [f32; 4],
    viewport: [f32; 4],
    params: [f32; 4],
    params2: [f32; 4],
    jitter: [f32; 4],
    sh: [[f32; 4]; 9],
    debug: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ObjectU {
    model: [[f32; 4]; 4],
    normal_mat: [[f32; 4]; 4],
    base_color: [f32; 4],
    mr: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    pub mask: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

struct GpuObject {
    id: String,
    vbuf: wgpu::Buffer,
    ibuf: Option<wgpu::Buffer>,
    count: u32,
    /// Smooth objects share vertices with the mesh, so a dab can upload just the changed range.
    smooth: bool,
    ubuf: wgpu::Buffer,
    bind: wgpu::BindGroup,
    world: Mat4,
    /// Cached inverse-transpose of `world` so material/highlight updates never invert a 4x4 matrix.
    normal_mat: [[f32; 4]; 4],
    material: Material,
    highlight: bool,
    visible: bool,
    local_bounds: (Vec3, Vec3),
    bounds: (Vec3, Vec3),
}

/// Process-wide cached studio IBL (procedural and deterministic, so multiple `Renderer` instances
/// pay CPU prefiltering cost only once).
struct BakedStudioIbl {
    sh_packed: [[f32; 4]; 9],
    env_bytes: Vec<u8>,
    lut_bytes: Vec<u8>,
}

fn baked_studio_ibl() -> &'static BakedStudioIbl {
    static CACHE: OnceLock<BakedStudioIbl> = OnceLock::new();
    CACHE.get_or_init(|| {
        const SH_BASIS_SCALE: [f32; 9] = [
            0.282095, 0.488603, 0.488603, 0.488603, 1.092548, 1.092548, 0.315392, 1.092548, 0.546274,
        ];
        let sh = ibl::sh9(ibl::studio);
        let sh_packed: [[f32; 4]; 9] = std::array::from_fn(|k| (sh[k] * SH_BASIS_SCALE[k]).extend(0.0).to_array());

        let cube = ibl::prefilter(&ibl::studio);
        let total_texels: usize = (0..ibl::CUBE_MIPS).map(|m| {
            let s = (ibl::CUBE_SIZE >> m).max(1) as usize;
            s * s * 6
        }).sum();
        let mut env_bytes = Vec::with_capacity(total_texels * 8);
        // Layer major then mip, as wgpu's TextureDataOrder::LayerMajor expects.
        for face in 0..6 {
            for mip in &cube {
                for t in &mip[face] {
                    for c in t {
                        env_bytes.extend_from_slice(&ibl::f16(*c).to_le_bytes());
                    }
                }
            }
        }

        let lut_raw = ibl::brdf_lut();
        let mut lut_bytes = Vec::with_capacity(lut_raw.len() * 8);
        for v in &lut_raw {
            for h in [ibl::f16(v[0]), ibl::f16(v[1]), 0u16, 0x3c00u16] {
                lut_bytes.extend_from_slice(&h.to_le_bytes());
            }
        }
        BakedStudioIbl { sh_packed, env_bytes, lut_bytes }
    })
}

/// Size dependent targets.
struct Targets {
    size: (u32, u32),
    hdr_msaa: wgpu::TextureView,
    hdr: wgpu::TextureView,
    depth_msaa: wgpu::TextureView,
    pre_normal: wgpu::TextureView,
    pre_depth: wgpu::TextureView,
    ssao_raw: wgpu::TextureView,
    ssao: wgpu::TextureView,
    scene_bind: wgpu::BindGroup,
    ssao_bind: wgpu::BindGroup,
    blur_bind: wgpu::BindGroup,
    tone_bind: wgpu::BindGroup,
}

pub const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const AO: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
pub const MSAA: u32 = 4;

pub struct Renderer {
    pub gpu: Arc<Gpu>,
    pub format: wgpu::TextureFormat,
    pub settings: RenderSettings,
    frame_buf: wgpu::Buffer,
    frame_bind: wgpu::BindGroup,
    l_scene: wgpu::BindGroupLayout,
    l_object: wgpu::BindGroupLayout,
    l_ssao: wgpu::BindGroupLayout,
    l_blur: wgpu::BindGroupLayout,
    l_tone: wgpu::BindGroupLayout,
    p_shadow: wgpu::RenderPipeline,
    p_prepass: wgpu::RenderPipeline,
    p_main: wgpu::RenderPipeline,
    p_ground: wgpu::RenderPipeline,
    p_line: wgpu::RenderPipeline,
    p_ssao: wgpu::RenderPipeline,
    p_blur: wgpu::RenderPipeline,
    /// Primary tone map pipeline for `self.format` (avoids per-frame HashMap lookups).
    p_tone_primary: wgpu::RenderPipeline,
    /// Secondary tone map pipelines for alternate output formats (e.g. RGBA8 snapshots).
    p_tone_extra: std::collections::HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    post: wgpu::ShaderModule,
    lo_tone: wgpu::PipelineLayout,
    shadow_view: wgpu::TextureView,
    shadow_cmp: wgpu::Sampler,
    linear: wgpu::Sampler,
    env_view: wgpu::TextureView,
    lut_view: wgpu::TextureView,
    /// Pre-packed, basis-scaled SH irradiance coefficients ready for `FrameU`.
    sh_packed: [[f32; 4]; 9],
    grid: (wgpu::Buffer, u32),
    objects: Vec<GpuObject>,
    targets: Option<Targets>,
    /// Re-render the shadow map only when geometry, placement or the light changes.
    shadow_dirty: bool,
    /// Cached scene bounds and derived shadow frustum (`light_vp`, `ground_z`), invalidated on scene changes.
    bounds_dirty: bool,
    cached_bounds: (Vec3, Vec3),
    cached_light_vp: [[f32; 4]; 4],
    cached_ground_z: f32,
    /// Cached linear background conversion so `srgb_to_linear` (`powf`) only runs when background changes.
    last_bg_srgb: [f32; 3],
    last_bg_linear: [f32; 3],
    /// Reusable vertex staging buffer for upserts and partial GPU uploads during sculpt strokes.
    vertex_staging: Vec<Vertex>,
    /// Sub pixel jitter and kernel rotation, for progressive accumulation (E6); zero when interactive.
    pub jitter: [f32; 4],
}

fn mat(m: Mat4) -> [[f32; 4]; 4] {
    m.to_cols_array_2d()
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn tex(device: &wgpu::Device, label: &str, size: (u32, u32), format: wgpu::TextureFormat, samples: u32, usage: wgpu::TextureUsages) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size.0.max(1), height: size.1.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn bgl_entry(binding: u32, ty: wgpu::BindingType, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: vis, ty, count: None }
}
fn uniform_ty() -> wgpu::BindingType {
    wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }
}
fn tex_ty(sample: wgpu::TextureSampleType, dim: wgpu::TextureViewDimension) -> wgpu::BindingType {
    wgpu::BindingType::Texture { sample_type: sample, view_dimension: dim, multisampled: false }
}

impl Renderer {
    pub fn new(gpu: Arc<Gpu>, format: wgpu::TextureFormat) -> Renderer {
        let d = &gpu.device;
        let vf = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let fs = wgpu::ShaderStages::FRAGMENT;
        let scene_src = wgpu::ShaderModuleDescriptor { label: Some("scene"), source: wgpu::ShaderSource::Wgsl(include_str!("scene.wgsl").into()) };
        let scene = d.create_shader_module(scene_src);
        let post = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("post"), source: wgpu::ShaderSource::Wgsl(include_str!("post.wgsl").into()) });

        let l_frame = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("frame"), entries: &[bgl_entry(0, uniform_ty(), vf)] });
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let unfilt = wgpu::TextureSampleType::Float { filterable: false };
        let d2 = wgpu::TextureViewDimension::D2;
        let l_scene = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene"),
            entries: &[
                bgl_entry(0, uniform_ty(), vf),
                bgl_entry(1, tex_ty(wgpu::TextureSampleType::Depth, d2), fs),
                bgl_entry(2, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison), fs),
                bgl_entry(3, tex_ty(float, wgpu::TextureViewDimension::Cube), fs),
                bgl_entry(4, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), fs),
                bgl_entry(5, tex_ty(float, d2), fs),
                bgl_entry(6, tex_ty(float, d2), fs),
            ],
        });
        let l_object = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("object"), entries: &[bgl_entry(0, uniform_ty(), vf)] });
        let l_ssao = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao in"),
            entries: &[bgl_entry(0, tex_ty(unfilt, d2), fs)],
        });
        let l_blur = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("blur in"), entries: &[bgl_entry(2, tex_ty(unfilt, d2), fs)] });
        let l_tone = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("tone in"), entries: &[bgl_entry(3, tex_ty(unfilt, d2), fs)] });

        let layout = |groups: &[&wgpu::BindGroupLayout]| {
            let g: Vec<Option<&wgpu::BindGroupLayout>> = groups.iter().map(|g| Some(*g)).collect();
            d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &g, immediate_size: 0 })
        };
        let vlayout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32],
        };
        let llayout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
        };
        let premul = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
        };
        let depth_state = |write: bool, bias: wgpu::DepthBiasState| wgpu::DepthStencilState {
            format: DEPTH,
            depth_write_enabled: Some(write),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias,
        };
        let pipe = |label: &str,
                    layout: &wgpu::PipelineLayout,
                    module: &wgpu::ShaderModule,
                    vs: &str,
                    fs: Option<&str>,
                    buffers: &[Option<wgpu::VertexBufferLayout>],
                    targets: &[Option<wgpu::ColorTargetState>],
                    topology: wgpu::PrimitiveTopology,
                    depth: Option<wgpu::DepthStencilState>,
                    samples: u32| {
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState { module, entry_point: Some(vs), compilation_options: Default::default(), buffers },
                primitive: wgpu::PrimitiveState { topology, ..Default::default() },
                depth_stencil: depth,
                multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
                fragment: fs.map(|fs| wgpu::FragmentState { module, entry_point: Some(fs), compilation_options: Default::default(), targets }),
                multiview_mask: None,
                cache: None,
            })
        };
        let tri = wgpu::PrimitiveTopology::TriangleList;
        let hdr_opaque = [Some(wgpu::ColorTargetState { format: HDR, blend: None, write_mask: wgpu::ColorWrites::ALL })];
        let hdr_blend = [Some(wgpu::ColorTargetState { format: HDR, blend: Some(premul), write_mask: wgpu::ColorWrites::ALL })];
        let lo_fo = layout(&[&l_frame, &l_object]);
        let lo_so = layout(&[&l_scene, &l_object]);
        let lo_s = layout(&[&l_scene]);
        let p_shadow = pipe(
            "shadow",
            &lo_fo,
            &scene,
            "vs_shadow",
            None,
            &[Some(vlayout.clone())],
            &[],
            tri,
            Some(depth_state(true, wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 })),
            1,
        );
        let p_prepass = pipe(
            "prepass",
            &lo_fo,
            &scene,
            "vs_prepass",
            Some("fs_prepass"),
            &[Some(vlayout.clone())],
            &[Some(wgpu::ColorTargetState { format: HDR, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            tri,
            Some(depth_state(true, Default::default())),
            1,
        );
        let p_main = pipe("pbr", &lo_so, &scene, "vs_main", Some("fs_main"), &[Some(vlayout)], &hdr_opaque, tri, Some(depth_state(true, Default::default())), MSAA);
        let p_ground = pipe("ground", &lo_s, &scene, "vs_ground", Some("fs_ground"), &[], &hdr_blend, tri, Some(depth_state(false, Default::default())), MSAA);
        let p_line = pipe("lines", &lo_s, &scene, "vs_line", Some("fs_line"), &[Some(llayout)], &hdr_blend, wgpu::PrimitiveTopology::LineList, Some(depth_state(false, Default::default())), MSAA);
        let ao_target = [Some(wgpu::ColorTargetState { format: AO, blend: None, write_mask: wgpu::ColorWrites::ALL })];
        let p_ssao = pipe("ssao", &layout(&[&l_frame, &l_ssao]), &post, "vs_full", Some("fs_ssao"), &[], &ao_target, tri, None, 1);
        // Group 0 is unused by the blur; an empty layout keeps the group numbers aligned with the shader.
        let l_empty = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("empty"), entries: &[] });
        let p_blur = pipe("blur", &layout(&[&l_empty, &l_blur]), &post, "vs_full", Some("fs_blur"), &[], &ao_target, tri, None, 1);
        let lo_tone = layout(&[&l_frame, &l_tone]);
        let p_tone_primary = tone_pipeline(d, &lo_tone, &post, format);

        let frame_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame"),
            size: std::mem::size_of::<FrameU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &l_frame, entries: &[wgpu::BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() }] });

        let shadow_size = if gpu.tier == Tier::Full { 2048 } else { 1024 };
        let shadow_view = tex(d, "shadow", (shadow_size, shadow_size), DEPTH, 1, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let shadow_cmp = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow cmp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let linear = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // IBL from the procedural studio (baked once per process, uploaded to this device).
        let baked = baked_studio_ibl();
        let env = d.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("env"),
                size: wgpu::Extent3d { width: ibl::CUBE_SIZE, height: ibl::CUBE_SIZE, depth_or_array_layers: 6 },
                mip_level_count: ibl::CUBE_MIPS,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &baked.env_bytes,
        );
        let env_view = env.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default() });
        let lut_view = d
            .create_texture_with_data(
                &gpu.queue,
                &wgpu::TextureDescriptor {
                    label: Some("brdf lut"),
                    size: wgpu::Extent3d { width: ibl::LUT_SIZE, height: ibl::LUT_SIZE, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &baked.lut_bytes,
            )
            .create_view(&Default::default());

        let grid = grid_lines(d);
        let settings = RenderSettings::default();
        let last_bg_srgb = settings.background;
        let last_bg_linear = last_bg_srgb.map(srgb_to_linear);
        Renderer {
            format,
            settings,
            frame_buf,
            frame_bind,
            l_scene,
            l_object,
            l_ssao,
            l_blur,
            l_tone,
            p_shadow,
            p_prepass,
            p_main,
            p_ground,
            p_line,
            p_ssao,
            p_blur,
            p_tone_primary,
            p_tone_extra: std::collections::HashMap::new(),
            post,
            lo_tone,
            shadow_view,
            shadow_cmp,
            linear,
            env_view,
            lut_view,
            sh_packed: baked.sh_packed,
            grid,
            objects: Vec::new(),
            targets: None,
            shadow_dirty: true,
            bounds_dirty: true,
            cached_bounds: (Vec3::splat(-10.0), Vec3::splat(10.0)),
            cached_light_vp: mat(Mat4::IDENTITY),
            cached_ground_z: -10.0,
            last_bg_srgb,
            last_bg_linear,
            vertex_staging: Vec::new(),
            jitter: [0.0; 4],
            gpu,
        }
    }

    fn ensure_targets(&mut self, size: (u32, u32)) {
        if self.targets.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        let d = &self.gpu.device;
        let ra = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let rt = ra | wgpu::TextureUsages::TEXTURE_BINDING;
        let half = ((size.0 / 2).max(1), (size.1 / 2).max(1));
        let hdr_msaa = tex(d, "hdr msaa", size, HDR, MSAA, ra);
        let hdr = tex(d, "hdr", size, HDR, 1, rt | wgpu::TextureUsages::COPY_SRC);
        let depth_msaa = tex(d, "depth msaa", size, DEPTH, MSAA, ra);
        let pre_normal = tex(d, "pre normal", size, HDR, 1, rt);
        let pre_depth = tex(d, "pre depth", size, DEPTH, 1, ra);
        let ssao_raw = tex(d, "ssao raw", half, AO, 1, rt);
        let ssao = tex(d, "ssao", half, AO, 1, rt);
        fn e(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(view) }
        }
        let scene_bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene"),
            layout: &self.l_scene,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.frame_buf.as_entire_binding() },
                e(1, &self.shadow_view),
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.shadow_cmp) },
                e(3, &self.env_view),
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.linear) },
                e(5, &self.lut_view),
                e(6, &ssao),
            ],
        });
        let ssao_bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.l_ssao, entries: &[e(0, &pre_normal)] });
        let blur_bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.l_blur, entries: &[e(2, &ssao_raw)] });
        let tone_bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.l_tone, entries: &[e(3, &hdr)] });
        self.targets = Some(Targets { size, hdr_msaa, hdr, depth_msaa, pre_normal, pre_depth, ssao_raw, ssao, scene_bind, ssao_bind, blur_bind, tone_bind });
    }

    // ---- scene ----

    pub fn clear(&mut self) {
        self.objects.clear();
        self.shadow_dirty = true;
        self.bounds_dirty = true;
    }

    /// Adds or replaces an object from a sculpt mesh (evaluated, modifiers applied, by the host).
    pub fn upsert(&mut self, id: &str, mesh: &SculptMesh, world: Mat4, material: &Material) {
        let d = &self.gpu.device;
        let existing_idx = self.objects.iter().position(|x| x.id == id);

        let (vbuf, ibuf, count) = if mesh.smooth_shading {
            fill_smooth_vertices(&mut self.vertex_staging, mesh, 0..mesh.positions.len());
            let vbytes = bytemuck::cast_slice(&self.vertex_staging);
            // Reuse existing GPU vertex buffer when byte size matches to avoid VRAM churn on live edits.
            let reuse_vbuf = existing_idx
                .map(|idx| self.objects[idx].vbuf.size() == vbytes.len() as u64)
                .unwrap_or(false);
            let vbuf = if reuse_vbuf {
                let idx = existing_idx.unwrap();
                self.gpu.queue.write_buffer(&self.objects[idx].vbuf, 0, vbytes);
                None
            } else {
                Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(id),
                    contents: vbytes,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }))
            };
            // `mesh.tris` is `&[[u32; 3]]` (tightly packed Pod): zero-copy upload, no flattened Vec<u32>!
            let ibytes: &[u8] = bytemuck::cast_slice(&mesh.tris);
            let ibuf = Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(id),
                contents: ibytes,
                usage: wgpu::BufferUsages::INDEX,
            }));
            (vbuf, ibuf, (mesh.tris.len() * 3) as u32)
        } else {
            fill_flat_vertices(&mut self.vertex_staging, mesh);
            let vbytes = bytemuck::cast_slice(&self.vertex_staging);
            let reuse_vbuf = existing_idx
                .map(|idx| self.objects[idx].vbuf.size() == vbytes.len() as u64)
                .unwrap_or(false);
            let vbuf = if reuse_vbuf {
                let idx = existing_idx.unwrap();
                self.gpu.queue.write_buffer(&self.objects[idx].vbuf, 0, vbytes);
                None
            } else {
                Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(id),
                    contents: vbytes,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }))
            };
            (vbuf, None, self.vertex_staging.len() as u32)
        };

        let local_bounds = local_bounds(&mesh.positions);
        let bounds = if world == Mat4::IDENTITY { local_bounds } else { world_bounds(&mesh.positions, world) };
        let normal_mat = compute_normal_mat(world);

        if let Some(idx) = existing_idx {
            let o = &mut self.objects[idx];
            if let Some(vb) = vbuf {
                o.vbuf = vb;
            }
            o.ibuf = ibuf;
            o.count = count;
            o.smooth = mesh.smooth_shading;
            o.world = world;
            o.normal_mat = normal_mat;
            o.material = material.clone();
            o.highlight = false;
            o.visible = true;
            o.local_bounds = local_bounds;
            o.bounds = bounds;
            write_object_u(&self.gpu.queue, o);
        } else {
            let ubuf = d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(id),
                size: std::mem::size_of::<ObjectU>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(id),
                layout: &self.l_object,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }],
            });
            let o = GpuObject {
                id: id.into(),
                vbuf: vbuf.unwrap(),
                ibuf,
                count,
                smooth: mesh.smooth_shading,
                ubuf,
                bind,
                world,
                normal_mat,
                material: material.clone(),
                highlight: false,
                visible: true,
                local_bounds,
                bounds,
            };
            write_object_u(&self.gpu.queue, &o);
            self.objects.push(o);
        }
        self.shadow_dirty = true;
        self.bounds_dirty = true;
    }

    /// Uploads moved vertices after a dab. Smooth meshes upload only the changed index range.
    pub fn update_vertices(&mut self, id: &str, mesh: &SculptMesh, dirty: &[u32]) {
        let Some(i) = self.objects.iter().position(|x| x.id == id) else { return };
        if dirty.is_empty() {
            return;
        }
        // Single-pass min/max over `dirty` (moved vertices + 1-ring whose normals changed).
        let mut lo = u32::MAX;
        let mut hi = 0u32;
        for &idx in dirty {
            lo = lo.min(idx);
            hi = hi.max(idx);
        }
        let (lo, hi) = (lo as usize, hi as usize);
        let o = &mut self.objects[i];
        if o.smooth && hi < mesh.positions.len() && (o.vbuf.size() as usize) == mesh.positions.len() * std::mem::size_of::<Vertex>() {
            // Reuse vertex staging buffer across dabs to eliminate per-dab heap allocations.
            fill_smooth_vertices(&mut self.vertex_staging, mesh, lo..hi + 1);
            self.gpu.queue.write_buffer(&o.vbuf, (lo * std::mem::size_of::<Vertex>()) as u64, bytemuck::cast_slice(&self.vertex_staging));
        } else {
            // Flat shaded meshes re-upload every corner via the reusable staging buffer.
            fill_flat_vertices(&mut self.vertex_staging, mesh);
            self.gpu.queue.write_buffer(&o.vbuf, 0, bytemuck::cast_slice(&self.vertex_staging));
        }
        o.local_bounds = local_bounds(&mesh.positions);
        // Transform the 8 AABB corners (same as `set_transform`) instead of transforming every vertex.
        o.bounds = if o.world == Mat4::IDENTITY { o.local_bounds } else { box_world_bounds(o.local_bounds, o.world) };
        self.shadow_dirty = true;
        self.bounds_dirty = true;
    }

    pub fn remove(&mut self, id: &str) {
        let before = self.objects.len();
        self.objects.retain(|o| o.id != id);
        if self.objects.len() != before {
            self.shadow_dirty = true;
            self.bounds_dirty = true;
        }
    }

    pub fn set_transform(&mut self, id: &str, world: Mat4) -> bool {
        let Some(i) = self.objects.iter().position(|x| x.id == id) else { return false };
        if self.objects[i].world == world {
            return true;
        }
        let o = &mut self.objects[i];
        o.world = world;
        o.normal_mat = compute_normal_mat(world);
        o.bounds = if world == Mat4::IDENTITY { o.local_bounds } else { box_world_bounds(o.local_bounds, world) };
        write_object_u(&self.gpu.queue, o);
        self.shadow_dirty = true;
        self.bounds_dirty = true;
        true
    }

    pub fn set_material(&mut self, id: &str, material: &Material) -> bool {
        let Some(i) = self.objects.iter().position(|x| x.id == id) else { return false };
        let o = &mut self.objects[i];
        if o.material == *material {
            return true;
        }
        o.material = material.clone();
        write_object_u(&self.gpu.queue, o);
        true
    }

    pub fn set_visible(&mut self, id: &str, visible: bool) -> bool {
        let Some(i) = self.objects.iter().position(|x| x.id == id) else { return false };
        if self.objects[i].visible != visible {
            self.objects[i].visible = visible;
            self.shadow_dirty = true;
            self.bounds_dirty = true;
        }
        true
    }

    pub fn set_highlight(&mut self, id: &str, on: bool) {
        if let Some(i) = self.objects.iter().position(|x| x.id == id) {
            if self.objects[i].highlight != on {
                self.objects[i].highlight = on;
                write_object_u(&self.gpu.queue, &self.objects[i]);
            }
        }
    }

    /// Loads every visible object of a project (base meshes; modifiers are the host's job).
    pub fn set_project(&mut self, p: &ProjectFile) {
        self.clear();
        for o in p.scene.objects.iter().filter(|o| o.is_visible) {
            let world = world_matrix(p, &o.id);
            self.upsert(&o.id, &SculptMesh::from_mesh(&o.mesh), world, &o.material);
        }
    }

    pub fn scene_bounds(&self) -> (Vec3, Vec3) {
        if !self.bounds_dirty {
            return self.cached_bounds;
        }
        compute_scene_bounds(&self.objects)
    }

    fn refresh_scene_cache(&mut self) {
        if !self.bounds_dirty {
            return;
        }
        let (bmin, bmax) = compute_scene_bounds(&self.objects);
        self.cached_bounds = (bmin, bmax);
        self.cached_ground_z = bmin.z.min(0.0);
        // Light frustum around the scene and the ground near it.
        let center = (bmin + bmax) * 0.5;
        let radius = ((bmax - bmin).length() * 0.5).max(10.0) * 1.4;
        let l = ibl::KEY_DIR_NORM;
        let up = if l.z.abs() > 0.99 { Vec3::Y } else { Vec3::Z };
        let lview = look_at_rh(center + l * (radius * 2.0), center, up);
        let lproj = ortho_rh(radius, radius * 4.0);
        self.cached_light_vp = mat(lproj * lview);
        self.bounds_dirty = false;
    }

    // ---- frame ----

    fn frame_uniforms(&mut self, cam: &OrbitCamera, size: (u32, u32), format: wgpu::TextureFormat) -> (FrameU, Mat4) {
        self.refresh_scene_cache();
        let s = &self.settings;
        if self.last_bg_srgb != s.background {
            self.last_bg_srgb = s.background;
            self.last_bg_linear = s.background.map(srgb_to_linear);
        }
        let aspect = size.0 as f32 / size.1.max(1) as f32;
        let view = cam.render_view_matrix();
        let proj = cam.projection_matrix(aspect);
        let view_proj = proj * view;
        let inv_proj = fast_inv_proj(proj);
        let cam_pos = view_eye_from_matrix(view);
        let encode = !format.is_srgb();
        let bg = self.last_bg_linear;
        let u = FrameU {
            view_proj: mat(view_proj),
            view: mat(view),
            proj: mat(proj),
            inv_proj: mat(inv_proj),
            light_vp: self.cached_light_vp,
            cam_pos: cam_pos.extend(1.0).to_array(),
            light_dir: ibl::KEY_DIR_NORM.extend(0.0).to_array(),
            light_color: [s.key_light, s.key_light * 0.97, s.key_light * 0.93, 0.0],
            background: [bg[0], bg[1], bg[2], 1.0],
            viewport: [size.0 as f32, size.1 as f32, 1.0 / size.0 as f32, 1.0 / size.1 as f32],
            params: [s.exposure, s.ssao as u32 as f32, s.shadows as u32 as f32, (ibl::CUBE_MIPS - 1) as f32],
            params2: [s.tonemap as u32 as f32, encode as u32 as f32, self.cached_ground_z, s.ssao_radius_mm],
            jitter: self.jitter,
            sh: self.sh_packed,
            debug: [s.debug_ao as u32 as f32, 0.0, 0.0, 0.0],
        };
        (u, view_proj)
    }

    fn draw_objects<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, frustum: Option<&Frustum>) {
        for o in &self.objects {
            if !o.visible {
                continue;
            }
            if let Some(f) = frustum {
                if !f.intersects_aabb(o.bounds.0, o.bounds.1) {
                    continue;
                }
            }
            pass.set_bind_group(1, &o.bind, &[]);
            pass.set_vertex_buffer(0, o.vbuf.slice(..));
            match &o.ibuf {
                Some(ib) => {
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..o.count, 0, 0..1);
                }
                None => pass.draw(0..o.count, 0..1),
            }
        }
    }

    /// Draws one frame into `target` (a view of `self.format`, `size` pixels).
    pub fn render(&mut self, target: &wgpu::TextureView, size: (u32, u32), cam: &OrbitCamera) {
        self.render_as(target, self.format, size, cam)
    }

    /// Draws one frame into a view of any color format.
    pub fn render_as(&mut self, target: &wgpu::TextureView, format: wgpu::TextureFormat, size: (u32, u32), cam: &OrbitCamera) {
        self.ensure_targets(size);
        if format != self.format && !self.p_tone_extra.contains_key(&format) {
            let p = tone_pipeline(&self.gpu.device, &self.lo_tone, &self.post, format);
            self.p_tone_extra.insert(format, p);
        }
        let (u, view_proj) = self.frame_uniforms(cam, size, format);
        self.gpu.queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&u));
        let frustum = Frustum::from_view_proj(view_proj, self.jitter[0].abs(), self.jitter[1].abs());
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        let t = self.targets.as_ref().unwrap();
        let clear_depth = |view, store| Some(wgpu::RenderPassDepthStencilAttachment { view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store }), stencil_ops: None });
        let color = |view, resolve, clear: wgpu::Color, store| wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: resolve,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store },
        };
        fn pass_desc<'a>(label: &'a str, colors: &'a [Option<wgpu::RenderPassColorAttachment<'a>>], depth: Option<wgpu::RenderPassDepthStencilAttachment<'a>>) -> wgpu::RenderPassDescriptor<'a> {
            wgpu::RenderPassDescriptor { label: Some(label), color_attachments: colors, depth_stencil_attachment: depth, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None }
        }

        let rendered_shadow = self.settings.shadows && self.shadow_dirty;
        if rendered_shadow {
            let mut p = enc.begin_render_pass(&pass_desc("shadow", &[], clear_depth(&self.shadow_view, wgpu::StoreOp::Store)));
            p.set_pipeline(&self.p_shadow);
            p.set_bind_group(0, &self.frame_bind, &[]);
            self.draw_objects(&mut p, None);
        }
        if self.settings.ssao {
            {
                let c = [Some(color(&t.pre_normal, None, wgpu::Color::TRANSPARENT, wgpu::StoreOp::Store))];
                // `pre_depth` is never sampled after prepass (depth is packed in `pre_normal.ba`); discard tile memory.
                let mut p = enc.begin_render_pass(&pass_desc("prepass", &c, clear_depth(&t.pre_depth, wgpu::StoreOp::Discard)));
                p.set_pipeline(&self.p_prepass);
                p.set_bind_group(0, &self.frame_bind, &[]);
                self.draw_objects(&mut p, Some(&frustum));
            }
            {
                let c = [Some(color(&t.ssao_raw, None, wgpu::Color::WHITE, wgpu::StoreOp::Store))];
                let mut p = enc.begin_render_pass(&pass_desc("ssao", &c, None));
                p.set_pipeline(&self.p_ssao);
                p.set_bind_group(0, &self.frame_bind, &[]);
                p.set_bind_group(1, &t.ssao_bind, &[]);
                p.draw(0..3, 0..1);
            }
            {
                let c = [Some(color(&t.ssao, None, wgpu::Color::WHITE, wgpu::StoreOp::Store))];
                let mut p = enc.begin_render_pass(&pass_desc("ssao blur", &c, None));
                p.set_pipeline(&self.p_blur);
                p.set_bind_group(1, &t.blur_bind, &[]);
                p.draw(0..3, 0..1);
            }
        }
        {
            // Both `hdr_msaa` (resolved into `t.hdr`) and `depth_msaa` are transient MSAA buffers:
            // `StoreOp::Discard` keeps them in TBDR tile memory and avoids writing ~100 MB/frame to DRAM.
            let c = [Some(color(&t.hdr_msaa, Some(&t.hdr), wgpu::Color::TRANSPARENT, wgpu::StoreOp::Discard))];
            let mut p = enc.begin_render_pass(&pass_desc("main", &c, clear_depth(&t.depth_msaa, wgpu::StoreOp::Discard)));
            p.set_bind_group(0, &t.scene_bind, &[]);
            p.set_pipeline(&self.p_ground);
            p.draw(0..6, 0..1);
            if self.settings.grid {
                p.set_pipeline(&self.p_line);
                p.set_vertex_buffer(0, self.grid.0.slice(..));
                p.draw(0..self.grid.1, 0..1);
            }
            p.set_pipeline(&self.p_main);
            self.draw_objects(&mut p, Some(&frustum));
        }
        {
            let c = [Some(color(target, None, wgpu::Color::BLACK, wgpu::StoreOp::Store))];
            let mut p = enc.begin_render_pass(&pass_desc("tonemap", &c, None));
            let tone = if format == self.format { &self.p_tone_primary } else { &self.p_tone_extra[&format] };
            p.set_pipeline(tone);
            p.set_bind_group(0, &self.frame_bind, &[]);
            p.set_bind_group(1, &t.tone_bind, &[]);
            p.draw(0..3, 0..1);
        }
        self.gpu.queue.submit([enc.finish()]);
        if rendered_shadow {
            self.shadow_dirty = false;
        }
    }

    /// Renders offscreen and reads back tightly packed RGBA8 (sRGB) pixels, whatever `self.format` is.
    pub fn render_rgba(&mut self, size: (u32, u32), cam: &OrbitCamera) -> Vec<u8> {
        let d = &self.gpu.device;
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let texture = d.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.render_as(&texture.create_view(&Default::default()), format, size, cam);
        read_rgba8(&self.gpu, &texture, size)
    }
}

fn tone_pipeline(d: &wgpu::Device, layout: &wgpu::PipelineLayout, post: &wgpu::ShaderModule, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("tonemap"),
        layout: Some(layout),
        vertex: wgpu::VertexState { module: post, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: post,
            entry_point: Some("fs_tonemap"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Copies an RGBA8 texture to the CPU (waits for the GPU).
pub fn read_rgba8(gpu: &Gpu, texture: &wgpu::Texture, size: (u32, u32)) -> Vec<u8> {
    let row = size.0 * 4;
    let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: (padded * size.1) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(size.1) } },
        wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map readback"));
    gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("poll");
    let data = slice.get_mapped_range().expect("mapped range");
    let mut out = Vec::with_capacity((row * size.1) as usize);
    for y in 0..size.1 as usize {
        out.extend_from_slice(&data[y * padded as usize..y * padded as usize + row as usize]);
    }
    out
}

/// Right handed view matrix (glam 0.34 has no `look_at_rh` on `Mat4`).
fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let f = (center - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    Mat4::from_cols(
        glam::Vec4::new(s.x, u.x, -f.x, 0.0),
        glam::Vec4::new(s.y, u.y, -f.y, 0.0),
        glam::Vec4::new(s.z, u.z, -f.z, 0.0),
        glam::Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

/// Symmetric orthographic projection, right handed, depth 0 (near = 0) to 1 (far).
fn ortho_rh(half: f32, far: f32) -> Mat4 {
    Mat4::from_cols(
        glam::Vec4::new(1.0 / half, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, 1.0 / half, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, -1.0 / far, 0.0),
        glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
    )
}

#[inline]
fn compute_normal_mat(world: Mat4) -> [[f32; 4]; 4] {
    if world == Mat4::IDENTITY {
        Mat4::IDENTITY.to_cols_array_2d()
    } else {
        world.inverse().transpose().to_cols_array_2d()
    }
}

fn write_object_u(queue: &wgpu::Queue, o: &GpuObject) {
    let c = o.material.base_color;
    let u = ObjectU {
        model: mat(o.world),
        normal_mat: o.normal_mat,
        base_color: [srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b), c.a],
        mr: [o.material.metallic, o.material.roughness, if o.highlight { 1.0 } else { 0.0 }, 0.0],
    };
    queue.write_buffer(&o.ubuf, 0, bytemuck::bytes_of(&u));
}

fn compute_scene_bounds(objects: &[GpuObject]) -> (Vec3, Vec3) {
    let mut b = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for o in objects {
        if !o.visible {
            continue;
        }
        b.0 = b.0.min(o.bounds.0);
        b.1 = b.1.max(o.bounds.1);
    }
    if b.0.x > b.1.x { (Vec3::splat(-10.0), Vec3::splat(10.0)) } else { b }
}

/// Analytic inverse for standard right-handed WebGPU perspective and orthographic projections
/// (avoids a full 4x4 LU inversion per frame), with exact fallback to `Mat4::inverse`.
#[inline]
fn fast_inv_proj(p: Mat4) -> Mat4 {
    let c = p.to_cols_array_2d();
    let (x, y, z, zn) = (c[0][0], c[1][1], c[2][2], c[3][2]);
    // Standard perspective: col2.w == -1, col3.w == 0, off-diagonals 0.
    if c[2][3] == -1.0 && c[3][3] == 0.0 && x != 0.0 && y != 0.0 && zn != 0.0 {
        let inv_zn = 1.0 / zn;
        return Mat4::from_cols(
            Vec4::new(1.0 / x, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0 / y, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, inv_zn),
            Vec4::new(0.0, 0.0, -1.0, z * inv_zn),
        );
    }
    // Standard orthographic: col2.w == 0, col3.w == 1, off-diagonals 0.
    if c[2][3] == 0.0 && c[3][3] == 1.0 && x != 0.0 && y != 0.0 && z != 0.0 {
        let inv_z = 1.0 / z;
        return Mat4::from_cols(
            Vec4::new(1.0 / x, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0 / y, 0.0, 0.0),
            Vec4::new(0.0, 0.0, inv_z, 0.0),
            Vec4::new(0.0, 0.0, -zn * inv_z, 1.0),
        );
    }
    p.inverse()
}

/// Extracts world-space camera eye directly from an orthonormal view matrix (zero trig calls).
#[inline]
fn view_eye_from_matrix(v: Mat4) -> Vec3 {
    let t = v.w_axis.truncate();
    Vec3::new(-v.x_axis.truncate().dot(t), -v.y_axis.truncate().dot(t), -v.z_axis.truncate().dot(t))
}

/// Six clip-space frustum planes (WebGPU depth 0..1) extracted via Gribb-Hartmann.
struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    #[inline]
    fn from_view_proj(vp: Mat4, jx: f32, jy: f32) -> Self {
        let r0 = vp.row(0);
        let r1 = vp.row(1);
        let r2 = vp.row(2);
        let r3 = vp.row(3);
        let sx = 1.0 + jx;
        let sy = 1.0 + jy;
        Self {
            planes: [
                r3 * sx + r0, // left:   x >= -w*(1+jx)
                r3 * sx - r0, // right:  x <=  w*(1+jx)
                r3 * sy + r1, // bottom: y >= -w*(1+jy)
                r3 * sy - r1, // top:    y <=  w*(1+jy)
                r2,           // near:   z >= 0
                r3 - r2,      // far:    z <= w
            ],
        }
    }

    /// True if the world-space AABB `[lo, hi]` intersects or lies inside the frustum.
    #[inline]
    fn intersects_aabb(&self, lo: Vec3, hi: Vec3) -> bool {
        if lo.x > hi.x {
            return false;
        }
        for p in &self.planes {
            let px = if p.x >= 0.0 { hi.x } else { lo.x };
            let py = if p.y >= 0.0 { hi.y } else { lo.y };
            let pz = if p.z >= 0.0 { hi.z } else { lo.z };
            if p.x * px + p.y * py + p.z * pz + p.w < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Packs smooth vertices into a reusable staging buffer with hoisted mask check (zero per-vertex bounds checks).
fn fill_smooth_vertices(dst: &mut Vec<Vertex>, mesh: &SculptMesh, range: std::ops::Range<usize>) {
    dst.clear();
    let count = range.end.saturating_sub(range.start);
    dst.reserve(count);
    let pos = &mesh.positions[range.clone()];
    let nrm = &mesh.normals[range.clone()];
    if mesh.masks.len() >= range.end {
        let msk = &mesh.masks[range];
        for ((p, n), &mask) in pos.iter().zip(nrm).zip(msk) {
            dst.push(Vertex { pos: p.to_array(), nrm: n.to_array(), mask });
        }
    } else if mesh.masks.is_empty() {
        for (p, n) in pos.iter().zip(nrm) {
            dst.push(Vertex { pos: p.to_array(), nrm: n.to_array(), mask: 0.0 });
        }
    } else {
        for (offset, (p, n)) in pos.iter().zip(nrm).enumerate() {
            let mask = mesh.masks.get(range.start + offset).copied().unwrap_or(0.0);
            dst.push(Vertex { pos: p.to_array(), nrm: n.to_array(), mask });
        }
    }
}

/// Packs flat-shaded vertices directly into a reusable staging buffer without intermediate position/normal/mask Vecs.
fn fill_flat_vertices(dst: &mut Vec<Vertex>, mesh: &SculptMesh) {
    dst.clear();
    dst.reserve(mesh.tris.len() * 3);
    let positions = &mesh.positions;
    let masks = &mesh.masks;
    let full_masks = masks.len() >= positions.len();
    let mut t = 0usize;
    for p in 0..mesh.poly_count() {
        let row = mesh.polys.row(p);
        let n = match row.len() {
            3 => {
                let p0 = positions[row[0] as usize];
                let p1 = positions[row[1] as usize];
                let p2 = positions[row[2] as usize];
                (p1 - p0).cross(p2 - p0).normalize_or_zero().to_array()
            }
            4 => {
                let p0 = positions[row[0] as usize];
                let p1 = positions[row[1] as usize];
                let p2 = positions[row[2] as usize];
                let p3 = positions[row[3] as usize];
                (p2 - p0).cross(p3 - p1).normalize_or_zero().to_array()
            }
            _ => {
                let mut acc = Vec3::ZERO;
                let mut a = positions[row[row.len() - 1] as usize];
                for &idx in row {
                    let b = positions[idx as usize];
                    acc.x += (a.y - b.y) * (a.z + b.z);
                    acc.y += (a.z - b.z) * (a.x + b.x);
                    acc.z += (a.x - b.x) * (a.y + b.y);
                    a = b;
                }
                acc.normalize_or_zero().to_array()
            }
        };
        let fan_tris = row.len().saturating_sub(2);
        if full_masks {
            for tri in &mesh.tris[t..t + fan_tris] {
                for &v in tri {
                    let vi = v as usize;
                    dst.push(Vertex { pos: positions[vi].to_array(), nrm: n, mask: masks[vi] });
                }
            }
        } else {
            for tri in &mesh.tris[t..t + fan_tris] {
                for &v in tri {
                    let vi = v as usize;
                    dst.push(Vertex { pos: positions[vi].to_array(), nrm: n, mask: masks.get(vi).copied().unwrap_or(0.0) });
                }
            }
        }
        t += fan_tris;
    }
}

/// 4-way unrolled SIMD AABB computation for high-throughput sculpt dab updates.
fn local_bounds(positions: &[Vec3]) -> (Vec3, Vec3) {
    let mut lo0 = Vec3A::splat(f32::INFINITY);
    let mut hi0 = Vec3A::splat(f32::NEG_INFINITY);
    let mut lo1 = lo0;
    let mut hi1 = hi0;
    let chunks = positions.chunks_exact(4);
    let rem = chunks.remainder();
    for c in chunks {
        let p0 = Vec3A::from(c[0]);
        let p1 = Vec3A::from(c[1]);
        let p2 = Vec3A::from(c[2]);
        let p3 = Vec3A::from(c[3]);
        lo0 = lo0.min(p0).min(p1);
        hi0 = hi0.max(p0).max(p1);
        lo1 = lo1.min(p2).min(p3);
        hi1 = hi1.max(p2).max(p3);
    }
    let mut lo = lo0.min(lo1);
    let mut hi = hi0.max(hi1);
    for &p in rem {
        let pa = Vec3A::from(p);
        lo = lo.min(pa);
        hi = hi.max(pa);
    }
    (Vec3::from(lo), Vec3::from(hi))
}

fn box_world_bounds((lo, hi): (Vec3, Vec3), world: Mat4) -> (Vec3, Vec3) {
    if lo.x > hi.x {
        return (lo, hi);
    }
    let mut b = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for x in [lo.x, hi.x] {
        for y in [lo.y, hi.y] {
            for z in [lo.z, hi.z] {
                let w = world.transform_point3(Vec3::new(x, y, z));
                b.0 = b.0.min(w);
                b.1 = b.1.max(w);
            }
        }
    }
    b
}

fn world_bounds(positions: &[Vec3], world: Mat4) -> (Vec3, Vec3) {
    let mut b = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for p in positions {
        let w = world.transform_point3(*p);
        b.0 = b.0.min(w);
        b.1 = b.1.max(w);
    }
    b
}

/// Local to world through the parent chain (Sculptor's `Scene.worldMatrix`).
pub fn world_matrix(p: &ProjectFile, id: &str) -> Mat4 {
    let objs = &p.scene.objects;
    let Some(mut o) = objs.iter().find(|o| o.id == id) else { return Mat4::IDENTITY };
    let mut m = o.transform.matrix();
    let mut steps = 0;
    while let Some(pid) = &o.parent {
        let Some(po) = objs.iter().find(|x| &x.id == pid) else { break };
        m = po.transform.matrix() * m;
        o = po;
        steps += 1;
        if steps > objs.len() {
            break;
        }
    }
    m
}

/// A 200 mm ground grid: 10 mm minor and 50 mm major lines, red X and green Y axes (Blender colours).
fn grid_lines(d: &wgpu::Device) -> (wgpu::Buffer, u32) {
    let mut v = Vec::new();
    let half = 100.0f32;
    let mut line = |a: [f32; 3], b: [f32; 3], c: [f32; 4]| {
        v.push(LineVertex { pos: a, color: c });
        v.push(LineVertex { pos: b, color: c });
    };
    for i in -10..=10 {
        let x = i as f32 * 10.0;
        let c = if i == 0 { None } else if i % 5 == 0 { Some([0.25, 0.25, 0.27, 0.35]) } else { Some([0.3, 0.3, 0.32, 0.18]) };
        if let Some(c) = c {
            line([x, -half, 0.0], [x, half, 0.0], c);
            line([-half, x, 0.0], [half, x, 0.0], c);
        }
    }
    line([-half, 0.0, 0.0], [half, 0.0, 0.0], [0.85, 0.2, 0.25, 0.7]);
    line([0.0, -half, 0.0], [0.0, half, 0.0], [0.45, 0.75, 0.15, 0.7]);
    let buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("grid"), contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::VERTEX });
    (buf, v.len() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::uv_sphere;

    #[test]
    fn fast_inv_proj_and_view_eye_match_exact() {
        for ortho in [false, true] {
            let cam = OrbitCamera {
                target: [12.0, -5.0, 8.0],
                distance: 135.0,
                yaw_degrees: 42.0,
                pitch_degrees: 27.0,
                fov_degrees: 50.0,
                is_orthographic: ortho,
            };
            let proj = cam.projection_matrix(16.0 / 9.0);
            let fast = fast_inv_proj(proj);
            let exact = proj.inverse();
            for (a, b) in fast.to_cols_array().iter().zip(exact.to_cols_array().iter()) {
                assert!((a - b).abs() < 1e-4 * (1.0 + b.abs()), "ortho={ortho}: {a} vs {b}");
            }
            let view = cam.render_view_matrix();
            let eye_fast = view_eye_from_matrix(view);
            let eye_exact = cam.render_eye();
            assert!((eye_fast - eye_exact).length() < 1e-2, "ortho={ortho}: {eye_fast} vs {eye_exact}");
        }
    }

    #[test]
    fn frustum_culling_accepts_visible_and_rejects_offscreen() {
        let cam = OrbitCamera { target: [0.0, 0.0, 0.0], distance: 100.0, yaw_degrees: 0.0, pitch_degrees: 0.0, ..Default::default() };
        let vp = cam.view_projection(1.5);
        let f = Frustum::from_view_proj(vp, 0.0, 0.0);
        assert!(f.intersects_aabb(Vec3::splat(-5.0), Vec3::splat(5.0)));
        // Far behind camera (-Y is toward camera at y=-100 looking +Y).
        assert!(!f.intersects_aabb(Vec3::new(-5.0, -500.0, -5.0), Vec3::new(5.0, -400.0, 5.0)));
        // Far to the right outside FOV.
        assert!(!f.intersects_aabb(Vec3::new(500.0, -5.0, -5.0), Vec3::new(510.0, 5.0, 5.0)));
    }

    #[test]
    fn flat_vertex_packing_matches_flat_buffers_masked() {
        let mesh = SculptMesh::from_mesh(&uv_sphere(10.0, 12, 8));
        let (p, n, m) = mesh.flat_buffers_masked();
        let mut staged = Vec::new();
        fill_flat_vertices(&mut staged, &mesh);
        assert_eq!(staged.len(), p.len());
        for i in 0..p.len() {
            assert_eq!(staged[i].pos, p[i].to_array());
            assert!((Vec3::from(staged[i].nrm) - n[i]).length() < 1e-5, "normal mismatch at {i}");
            assert_eq!(staged[i].mask, m[i]);
        }
    }
}
