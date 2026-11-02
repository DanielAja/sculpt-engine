//! wgpu renderer: shadow map, SSAO, physically based forward pass with IBL, MSAA, tone mapping.
//! See DESIGN.md section 4. The host owns the surface; `Renderer::render` draws into any texture view of
//! the renderer's output format (a surface frame, or an offscreen texture for snapshots and tests).

pub mod ibl;

use crate::camera::OrbitCamera;
use crate::format::{Material, ProjectFile};
use crate::mesh::SculptMesh;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::sync::Arc;
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
        let limits = if tier == Tier::Full { wgpu::Limits::default() } else { wgpu::Limits::downlevel_webgl2_defaults() };
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
    material: Material,
    highlight: bool,
    bounds: (Vec3, Vec3),
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
    /// Tone map pipelines per output format (the surface's, plus RGBA8 for snapshots).
    p_tone: std::collections::HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    post: wgpu::ShaderModule,
    lo_tone: wgpu::PipelineLayout,
    shadow_view: wgpu::TextureView,
    shadow_cmp: wgpu::Sampler,
    linear: wgpu::Sampler,
    env_view: wgpu::TextureView,
    lut_view: wgpu::TextureView,
    sh: [Vec3; 9],
    grid: (wgpu::Buffer, u32),
    objects: Vec<GpuObject>,
    targets: Option<Targets>,
    /// Re-render the shadow map only when geometry, placement or the light changes.
    shadow_dirty: bool,
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
            entries: &[bgl_entry(0, tex_ty(unfilt, d2), fs), bgl_entry(1, tex_ty(wgpu::TextureSampleType::Depth, d2), fs)],
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
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
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
        let p_tone = std::collections::HashMap::from([(format, tone_pipeline(d, &lo_tone, &post, format))]);

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

        // IBL from the procedural studio.
        let sh = ibl::sh9(ibl::studio);
        let cube = ibl::prefilter(&ibl::studio);
        let mut bytes = Vec::new();
        // Layer major then mip, as wgpu's TextureDataOrder::LayerMajor expects.
        for face in 0..6 {
            for mip in &cube {
                for t in &mip[face] {
                    for c in t {
                        bytes.extend_from_slice(&ibl::f16(*c).to_le_bytes());
                    }
                }
            }
        }
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
            &bytes,
        );
        let env_view = env.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default() });
        let lut: Vec<u8> = ibl::brdf_lut().iter().flat_map(|v| [ibl::f16(v[0]), ibl::f16(v[1]), 0, 0x3c00]).flat_map(|h| h.to_le_bytes()).collect();
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
                &lut,
            )
            .create_view(&Default::default());

        let grid = grid_lines(d);
        Renderer {
            format,
            settings: RenderSettings::default(),
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
            p_tone,
            post,
            lo_tone,
            shadow_view,
            shadow_cmp,
            linear,
            env_view,
            lut_view,
            sh,
            grid,
            objects: Vec::new(),
            targets: None,
            shadow_dirty: true,
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
        let pre_depth = tex(d, "pre depth", size, DEPTH, 1, rt);
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
        let ssao_bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.l_ssao, entries: &[e(0, &pre_normal), e(1, &pre_depth)] });
        let blur_bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.l_blur, entries: &[e(2, &ssao_raw)] });
        let tone_bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.l_tone, entries: &[e(3, &hdr)] });
        self.targets = Some(Targets { size, hdr_msaa, hdr, depth_msaa, pre_normal, pre_depth, ssao_raw, ssao, scene_bind, ssao_bind, blur_bind, tone_bind });
    }

    // ---- scene ----

    pub fn clear(&mut self) {
        self.objects.clear();
        self.shadow_dirty = true;
    }

    /// Adds or replaces an object from a sculpt mesh (evaluated, modifiers applied, by the host).
    pub fn upsert(&mut self, id: &str, mesh: &SculptMesh, world: Mat4, material: &Material) {
        let d = &self.gpu.device;
        let (vbuf, ibuf, count) = if mesh.smooth_shading {
            let v: Vec<Vertex> = mesh.positions.iter().zip(&mesh.normals).map(|(p, n)| Vertex { pos: p.to_array(), nrm: n.to_array() }).collect();
            let idx: Vec<u32> = mesh.tris.iter().flatten().copied().collect();
            (
                d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(id), contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST }),
                Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(id), contents: bytemuck::cast_slice(&idx), usage: wgpu::BufferUsages::INDEX })),
                idx.len() as u32,
            )
        } else {
            let v = flat_vertices(mesh);
            (
                d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(id), contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST }),
                None,
                v.len() as u32,
            )
        };
        let ubuf = d.create_buffer(&wgpu::BufferDescriptor { label: Some(id), size: std::mem::size_of::<ObjectU>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(id), layout: &self.l_object, entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }] });
        let bounds = world_bounds(&mesh.positions, world);
        let o = GpuObject { id: id.into(), vbuf, ibuf, count, smooth: mesh.smooth_shading, ubuf, bind, world, material: material.clone(), highlight: false, bounds };
        self.write_object(&o);
        match self.objects.iter().position(|x| x.id == id) {
            Some(i) => self.objects[i] = o,
            None => self.objects.push(o),
        }
        self.shadow_dirty = true;
    }

    fn write_object(&self, o: &GpuObject) {
        let c = o.material.base_color;
        let u = ObjectU {
            model: mat(o.world),
            normal_mat: mat(o.world.inverse().transpose()),
            base_color: [srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b), c.a],
            mr: [o.material.metallic, o.material.roughness, if o.highlight { 1.0 } else { 0.0 }, 0.0],
        };
        self.gpu.queue.write_buffer(&o.ubuf, 0, bytemuck::bytes_of(&u));
    }

    /// Uploads moved vertices after a dab. Smooth meshes upload only the changed index range.
    pub fn update_vertices(&mut self, id: &str, mesh: &SculptMesh, dirty: &[u32]) {
        let Some(i) = self.objects.iter().position(|x| x.id == id) else { return };
        let o = &mut self.objects[i];
        if dirty.is_empty() {
            return;
        }
        // `dirty` holds moved vertices and the ring whose normals changed (SculptTarget::take_dirty).
        let (lo, hi) = (*dirty.iter().min().unwrap() as usize, *dirty.iter().max().unwrap() as usize);
        if o.smooth && hi < mesh.positions.len() && (o.vbuf.size() as usize) == mesh.positions.len() * std::mem::size_of::<Vertex>() {
            // shortcut: uploads the whole min..max index range; split into runs if far apart footprints
            // (symmetry on meshes with scattered vertex order) make the range large.
            let v: Vec<Vertex> = (lo..=hi).map(|k| Vertex { pos: mesh.positions[k].to_array(), nrm: mesh.normals[k].to_array() }).collect();
            self.gpu.queue.write_buffer(&o.vbuf, (lo * std::mem::size_of::<Vertex>()) as u64, bytemuck::cast_slice(&v));
        } else {
            // shortcut: flat shaded meshes re-upload every corner; sculpting normally runs on smooth meshes.
            let v = flat_vertices(mesh);
            self.gpu.queue.write_buffer(&o.vbuf, 0, bytemuck::cast_slice(&v));
        }
        o.bounds = world_bounds(&mesh.positions, o.world);
        self.shadow_dirty = true;
    }

    pub fn set_highlight(&mut self, id: &str, on: bool) {
        if let Some(i) = self.objects.iter().position(|x| x.id == id) {
            self.objects[i].highlight = on;
            self.write_object(&self.objects[i]);
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
        let mut b = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for o in &self.objects {
            b.0 = b.0.min(o.bounds.0);
            b.1 = b.1.max(o.bounds.1);
        }
        if b.0.x > b.1.x { (Vec3::splat(-10.0), Vec3::splat(10.0)) } else { b }
    }

    // ---- frame ----

    fn frame_uniforms(&self, cam: &OrbitCamera, size: (u32, u32), format: wgpu::TextureFormat) -> FrameU {
        let s = &self.settings;
        let aspect = size.0 as f32 / size.1.max(1) as f32;
        let view = cam.render_view_matrix();
        let proj = cam.projection_matrix(aspect);
        let (bmin, bmax) = self.scene_bounds();
        let ground_z = bmin.z.min(0.0);
        // Light frustum around the scene and the ground near it.
        let center = (bmin + bmax) * 0.5;
        let radius = ((bmax - bmin).length() * 0.5).max(10.0) * 1.4;
        let l = ibl::KEY_DIR.normalize();
        let up = if l.z.abs() > 0.99 { Vec3::Y } else { Vec3::Z };
        let lview = look_at_rh(center + l * radius * 2.0, center, up);
        let lproj = ortho_rh(radius, radius * 4.0);
        let mut sh = [[0.0; 4]; 9];
        for k in 0..9 {
            sh[k] = self.sh[k].extend(0.0).to_array();
        }
        let encode = !format.is_srgb();
        let bg = s.background.map(srgb_to_linear);
        FrameU {
            view_proj: mat(proj * view),
            view: mat(view),
            proj: mat(proj),
            inv_proj: mat(proj.inverse()),
            light_vp: mat(lproj * lview),
            cam_pos: cam.render_eye().extend(1.0).to_array(),
            light_dir: l.extend(0.0).to_array(),
            light_color: [s.key_light, s.key_light * 0.97, s.key_light * 0.93, 0.0],
            background: [bg[0], bg[1], bg[2], 1.0],
            viewport: [size.0 as f32, size.1 as f32, 1.0 / size.0 as f32, 1.0 / size.1 as f32],
            params: [s.exposure, s.ssao as u32 as f32, s.shadows as u32 as f32, (ibl::CUBE_MIPS - 1) as f32],
            params2: [s.tonemap as u32 as f32, encode as u32 as f32, ground_z, s.ssao_radius_mm],
            jitter: self.jitter,
            sh,
            debug: [s.debug_ao as u32 as f32, 0.0, 0.0, 0.0],
        }
    }

    fn draw_objects<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        for o in &self.objects {
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
        if !self.p_tone.contains_key(&format) {
            let p = tone_pipeline(&self.gpu.device, &self.lo_tone, &self.post, format);
            self.p_tone.insert(format, p);
        }
        let u = self.frame_uniforms(cam, size, format);
        self.gpu.queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&u));
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        let t = self.targets.as_ref().unwrap();
        let clear_depth = |view| Some(wgpu::RenderPassDepthStencilAttachment { view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None });
        let color = |view, resolve, clear: wgpu::Color| wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: resolve,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
        };
        fn pass_desc<'a>(label: &'a str, colors: &'a [Option<wgpu::RenderPassColorAttachment<'a>>], depth: Option<wgpu::RenderPassDepthStencilAttachment<'a>>) -> wgpu::RenderPassDescriptor<'a> {
            wgpu::RenderPassDescriptor { label: Some(label), color_attachments: colors, depth_stencil_attachment: depth, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None }
        }

        if self.settings.shadows && self.shadow_dirty {
            let mut p = enc.begin_render_pass(&pass_desc("shadow", &[], clear_depth(&self.shadow_view)));
            p.set_pipeline(&self.p_shadow);
            p.set_bind_group(0, &self.frame_bind, &[]);
            self.draw_objects(&mut p);
        }
        if self.settings.ssao {
            {
                let c = [Some(color(&t.pre_normal, None, wgpu::Color::TRANSPARENT))];
                let mut p = enc.begin_render_pass(&pass_desc("prepass", &c, clear_depth(&t.pre_depth)));
                p.set_pipeline(&self.p_prepass);
                p.set_bind_group(0, &self.frame_bind, &[]);
                self.draw_objects(&mut p);
            }
            {
                let c = [Some(color(&t.ssao_raw, None, wgpu::Color::WHITE))];
                let mut p = enc.begin_render_pass(&pass_desc("ssao", &c, None));
                p.set_pipeline(&self.p_ssao);
                p.set_bind_group(0, &self.frame_bind, &[]);
                p.set_bind_group(1, &t.ssao_bind, &[]);
                p.draw(0..3, 0..1);
            }
            {
                let c = [Some(color(&t.ssao, None, wgpu::Color::WHITE))];
                let mut p = enc.begin_render_pass(&pass_desc("ssao blur", &c, None));
                p.set_pipeline(&self.p_blur);
                p.set_bind_group(1, &t.blur_bind, &[]);
                p.draw(0..3, 0..1);
            }
        }
        {
            let c = [Some(color(&t.hdr_msaa, Some(&t.hdr), wgpu::Color::TRANSPARENT))];
            let mut p = enc.begin_render_pass(&pass_desc("main", &c, clear_depth(&t.depth_msaa)));
            p.set_bind_group(0, &t.scene_bind, &[]);
            p.set_pipeline(&self.p_ground);
            p.draw(0..6, 0..1);
            if self.settings.grid {
                p.set_pipeline(&self.p_line);
                p.set_vertex_buffer(0, self.grid.0.slice(..));
                p.draw(0..self.grid.1, 0..1);
            }
            p.set_pipeline(&self.p_main);
            self.draw_objects(&mut p);
        }
        {
            let c = [Some(color(target, None, wgpu::Color::BLACK))];
            let mut p = enc.begin_render_pass(&pass_desc("tonemap", &c, None));
            p.set_pipeline(&self.p_tone[&format]);
            p.set_bind_group(0, &self.frame_bind, &[]);
            p.set_bind_group(1, &t.tone_bind, &[]);
            p.draw(0..3, 0..1);
        }
        self.gpu.queue.submit([enc.finish()]);
        self.shadow_dirty = false;
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

fn flat_vertices(mesh: &SculptMesh) -> Vec<Vertex> {
    let (p, n) = mesh.flat_buffers();
    p.iter().zip(&n).map(|(p, n)| Vertex { pos: p.to_array(), nrm: n.to_array() }).collect()
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
