//! On-device free Image-to-3D reconstruction (pure Rust, zero GPU, zero cloud).
//!
//! Converts any RGBA8 photo (plus optional depth map from LiDAR / ARKit / DepthAnything) into a
//! watertight, smooth, manifold 3D mesh in Sculptor / Dream Builder `text3d` JSON format:
//! - `item` (default): automatic foreground segmentation + exact linear-time 2D Euclidean Distance
//!   Transform (Felzenszwalb-Huttenlocher) + smooth 3D rounded bilateral inflation SDF -> Surface Nets
//!   -> volume-preserving Taubin smoothing.
//! - `lathe`: extracts the vertical silhouette radius profile and revolves it 360 degrees for cups,
//!   bottles, vases, lamps, and pottery.
//! - `relief`: depth/luminance sculpted 3D bas-relief slab with a clean watertight flat back.

use crate::format::{Color, Material, Mesh};
use crate::text3d::mesher::{make_manifold, quads_manifold, triangle_count, Grid, EDGES};
use crate::text3d::{error_json, parse_color, parts_json, GenPart};
use glam::{Vec3, Vec3A};
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Image3DMode {
    Item,
    Lathe,
    Relief,
}

impl Image3DMode {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "lathe" | "revolve" | "spin" | "vase" | "cup" => Self::Lathe,
            "relief" | "height" | "heightmap" | "plaque" | "emboss" => Self::Relief,
            _ => Self::Item,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Image3DOptions {
    pub mode: Image3DMode,
    pub name: String,
    pub max_triangles: usize,
    pub resolution: u32,
    pub depth_scale: f32,
    pub smooth_iters: u32,
    pub quads: bool,
    pub color_override: Option<Color>,
    pub metallic: f32,
    pub roughness: f32,
}

impl Default for Image3DOptions {
    fn default() -> Self {
        Self {
            mode: Image3DMode::Item,
            name: "Photo Object".into(),
            max_triangles: 12_000,
            resolution: 56,
            depth_scale: 0.34,
            smooth_iters: 2,
            quads: false,
            color_override: None,
            metallic: 0.05,
            roughness: 0.45,
        }
    }
}

impl Image3DOptions {
    pub fn from_json(text: &str) -> Self {
        let mut o = Self::default();
        if let Ok(v) = serde_json::from_str::<Value>(text) {
            if let Some(s) = v.get("mode").and_then(Value::as_str) {
                o.mode = Image3DMode::parse(s);
            }
            if let Some(s) = v.get("name").and_then(Value::as_str) {
                let t = s.trim();
                if !t.is_empty() {
                    o.name = t.chars().take(60).collect();
                }
            }
            if let Some(n) = v.get("maxTriangles").and_then(Value::as_u64) {
                o.max_triangles = (n as usize).clamp(400, 200_000);
            }
            if let Some(n) = v.get("resolution").and_then(Value::as_u64) {
                o.resolution = (n as u32).clamp(20, 96);
            }
            if let Some(d) = v.get("depthScale").and_then(Value::as_f64) {
                o.depth_scale = (d as f32).clamp(0.08, 1.2);
            }
            if let Some(s) = v.get("smoothIters").and_then(Value::as_u64) {
                o.smooth_iters = (s as u32).clamp(0, 6);
            }
            if let Some(b) = v.get("quads").and_then(Value::as_bool) {
                o.quads = b;
            }
            if let Some(c) = v.get("color").and_then(Value::as_str).and_then(parse_color) {
                o.color_override = Some(c);
            }
            if let Some(m) = v.get("metallic").and_then(Value::as_f64) {
                o.metallic = (m as f32).clamp(0.0, 1.0);
            }
            if let Some(r) = v.get("roughness").and_then(Value::as_f64) {
                o.roughness = (r as f32).clamp(0.04, 1.0);
            }
        }
        o
    }
}

/// Working downscaled image representation (<= 128px on longest side).
pub(crate) struct WorkImage {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<[u8; 4]>,
    pub depth: Option<Vec<f32>>,
}

pub(crate) fn downsample_image(
    width: u32,
    height: u32,
    rgba: &[u8],
    depth: Option<&[f32]>,
    max_dim: usize,
) -> Result<WorkImage, String> {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 {
        return Err("image dimensions must be non-zero".into());
    }
    let expected = w.checked_mul(h).and_then(|n| n.checked_mul(4)).ok_or("image dimensions overflow")?;
    if rgba.len() < expected {
        return Err(format!("rgba buffer too small: got {}, need {expected}", rgba.len()));
    }
    let has_depth = depth.is_some_and(|d| d.len() >= w * h);
    let longest = w.max(h);
    let (tw, th) = if longest <= max_dim {
        (w, h)
    } else {
        let scale = max_dim as f32 / longest as f32;
        (((w as f32 * scale).round() as usize).max(4), ((h as f32 * scale).round() as usize).max(4))
    };

    if (tw, th) == (w, h) {
        // Identity scale: every box is exactly one pixel, so this is a straight copy (no box loop).
        let rgba_px: Vec<[u8; 4]> = rgba[..expected].chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
        let od = if has_depth {
            depth.map(|d| d[..w * h].iter().map(|&v| if v.is_finite() { v } else { 0.0 }).collect())
        } else {
            None
        };
        return Ok(WorkImage { w, h, rgba: rgba_px, depth: od });
    }
    let mut out = vec![[0u8; 4]; tw * th];
    let mut out_depth = if has_depth { Some(vec![0f32; tw * th]) } else { None };
    let sx = w as f32 / tw as f32;
    let sy = h as f32 / th as f32;
    // Exact fixed-point reciprocals for c in 1..=16: `(sum * RECIP16[c]) >> 16 == sum / c` for sum <= 255*c.
    const RECIP16: [u32; 17] = [0, 65536, 32768, 21846, 16384, 13108, 10923, 9363, 8192, 7282, 6554, 5958, 5462, 5042, 4682, 4370, 4096];
    // Source column span of every output column, stack-allocated for <= 128px working width.
    let mut stack_spans = [(0usize, 0usize); 128];
    let heap_spans: Vec<(usize, usize)>;
    let xspans: &[(usize, usize)] = if tw <= 128 {
        for (x, sp) in stack_spans[..tw].iter_mut().enumerate() {
            let x0 = ((x as f32) * sx).floor() as usize;
            let x1 = (((x + 1) as f32) * sx).ceil().min(w as f32) as usize;
            *sp = (x0, x1.max(x0 + 1).min(w));
        }
        &stack_spans[..tw]
    } else {
        heap_spans = (0..tw)
            .map(|x| {
                let x0 = ((x as f32) * sx).floor() as usize;
                let x1 = (((x + 1) as f32) * sx).ceil().min(w as f32) as usize;
                (x0, x1.max(x0 + 1).min(w))
            })
            .collect();
        &heap_spans
    };
    let depth_src = depth.filter(|_| has_depth);
    let do_row = |y: usize, orow: &mut [[u8; 4]], drow: Option<&mut [f32]>| {
        let y0 = ((y as f32) * sy).floor() as usize;
        let y1 = (((y + 1) as f32) * sy).ceil().min(h as f32) as usize;
        let y1 = y1.max(y0 + 1).min(h);
        let dy = y1 - y0;
        let row_stride = w * 4;
        if dy == 2 {
            let r0 = &rgba[y0 * row_stride..(y0 + 1) * row_stride];
            let r1 = &rgba[(y0 + 1) * row_stride..(y0 + 2) * row_stride];
            for (x, o) in orow.iter_mut().enumerate() {
                let (x0, x1) = xspans[x];
                let dx = x1 - x0;
                let off = x0 * 4;
                if dx == 2 {
                    let a = &r0[off..off + 8];
                    let b = &r1[off..off + 8];
                    *o = [
                        ((a[0] as u32 + a[4] as u32 + b[0] as u32 + b[4] as u32) >> 2) as u8,
                        ((a[1] as u32 + a[5] as u32 + b[1] as u32 + b[5] as u32) >> 2) as u8,
                        ((a[2] as u32 + a[6] as u32 + b[2] as u32 + b[6] as u32) >> 2) as u8,
                        ((a[3] as u32 + a[7] as u32 + b[3] as u32 + b[7] as u32) >> 2) as u8,
                    ];
                } else if dx == 3 {
                    let a = &r0[off..off + 12];
                    let b = &r1[off..off + 12];
                    const R6: u32 = RECIP16[6];
                    *o = [
                        (((a[0] as u32 + a[4] as u32 + a[8] as u32 + b[0] as u32 + b[4] as u32 + b[8] as u32) * R6) >> 16) as u8,
                        (((a[1] as u32 + a[5] as u32 + a[9] as u32 + b[1] as u32 + b[5] as u32 + b[9] as u32) * R6) >> 16) as u8,
                        (((a[2] as u32 + a[6] as u32 + a[10] as u32 + b[2] as u32 + b[6] as u32 + b[10] as u32) * R6) >> 16) as u8,
                        (((a[3] as u32 + a[7] as u32 + a[11] as u32 + b[3] as u32 + b[7] as u32 + b[11] as u32) * R6) >> 16) as u8,
                    ];
                } else {
                    let (mut s0, mut s1, mut s2, mut s3) = (0u32, 0u32, 0u32, 0u32);
                    for px in r0[off..x1 * 4].chunks_exact(4) {
                        s0 += px[0] as u32; s1 += px[1] as u32; s2 += px[2] as u32; s3 += px[3] as u32;
                    }
                    for px in r1[off..x1 * 4].chunks_exact(4) {
                        s0 += px[0] as u32; s1 += px[1] as u32; s2 += px[2] as u32; s3 += px[3] as u32;
                    }
                    let c = (2 * dx).max(1);
                    if c <= 16 {
                        let r = RECIP16[c];
                        *o = [((s0 * r) >> 16) as u8, ((s1 * r) >> 16) as u8, ((s2 * r) >> 16) as u8, ((s3 * r) >> 16) as u8];
                    } else {
                        let cu = c as u32;
                        *o = [(s0 / cu) as u8, (s1 / cu) as u8, (s2 / cu) as u8, (s3 / cu) as u8];
                    }
                }
            }
        } else {
            for (x, o) in orow.iter_mut().enumerate() {
                let (x0, x1) = xspans[x];
                let (mut s0, mut s1, mut s2, mut s3) = (0u32, 0u32, 0u32, 0u32);
                for py in y0..y1 {
                    for px in rgba[(py * w + x0) * 4..(py * w + x1) * 4].chunks_exact(4) {
                        s0 += px[0] as u32;
                        s1 += px[1] as u32;
                        s2 += px[2] as u32;
                        s3 += px[3] as u32;
                    }
                }
                let c = (dy * (x1 - x0)).max(1);
                if c <= 16 {
                    let r = RECIP16[c];
                    *o = [((s0 * r) >> 16) as u8, ((s1 * r) >> 16) as u8, ((s2 * r) >> 16) as u8, ((s3 * r) >> 16) as u8];
                } else {
                    let cu = c as u32;
                    *o = [(s0 / cu) as u8, (s1 / cu) as u8, (s2 / cu) as u8, (s3 / cu) as u8];
                }
            }
        }
        if let (Some(d), Some(drow)) = (depth_src, drow) {
            for (x, o) in drow.iter_mut().enumerate() {
                let (x0, x1) = xspans[x];
                let mut dsum = 0.0f32;
                for py in y0..y1 {
                    for &v in &d[py * w + x0..py * w + x1] {
                        if v.is_finite() {
                            dsum += v;
                        }
                    }
                }
                *o = dsum / (dy * (x1 - x0)).max(1) as f32;
            }
        }
    };
    #[cfg(feature = "threads")]
    if tw * th >= 65_536 {
        use rayon::prelude::*;
        match out_depth.as_mut() {
            Some(od) => out
                .par_chunks_mut(tw)
                .zip(od.par_chunks_mut(tw))
                .enumerate()
                .for_each(|(y, (orow, drow))| do_row(y, orow, Some(drow))),
            None => out.par_chunks_mut(tw).enumerate().for_each(|(y, orow)| do_row(y, orow, None)),
        }
        return Ok(WorkImage { w: tw, h: th, rgba: out, depth: out_depth });
    }
    match out_depth.as_mut() {
        Some(od) => {
            for (y, (orow, drow)) in out.chunks_mut(tw).zip(od.chunks_mut(tw)).enumerate() {
                do_row(y, orow, Some(drow));
            }
        }
        None => {
            for (y, orow) in out.chunks_mut(tw).enumerate() {
                do_row(y, orow, None);
            }
        }
    }
    Ok(WorkImage { w: tw, h: th, rgba: out, depth: out_depth })
}

/// Extracts a clean foreground boolean mask and dominant sRGB color.
/// Supports alpha-transparent cutouts directly, or border-seeded flood-fill background keying for opaque photos.
pub(crate) fn segment_foreground(img: &WorkImage) -> (Vec<bool>, Color) {
    let n = img.w * img.h;
    let transparent_count = img.rgba.iter().filter(|p| p[3] < 200).count();
    let mut mask;

    if transparent_count * 25 >= n {
        // Meaningful alpha transparency present: use alpha channel.
        mask = img.rgba.iter().map(|p| p[3] >= 96).collect();
    } else {
        // Multi-corner & perimeter background model: handles solid, dark, bright, AND gradient
        // backgrounds (such as Wikipedia reference photos) by sampling all 4 corners + perimeter mean.
        let corner_avg = |x0: usize, y0: usize| -> [f32; 3] {
            let mut s = [0.0f32; 3];
            let mut c = 0.0f32;
            for dy in 0..3 {
                for dx in 0..3 {
                    let x = (x0 + dx).min(img.w - 1);
                    let y = (y0 + dy).min(img.h - 1);
                    let p = img.rgba[y * img.w + x];
                    s[0] += p[0] as f32;
                    s[1] += p[1] as f32;
                    s[2] += p[2] as f32;
                    c += 1.0;
                }
            }
            [s[0] / c, s[1] / c, s[2] / c]
        };
        let corners = [
            corner_avg(0, 0),
            corner_avg(img.w.saturating_sub(3), 0),
            corner_avg(0, img.h.saturating_sub(3)),
            corner_avg(img.w.saturating_sub(3), img.h.saturating_sub(3)),
        ];
        let mut br = 0.0f32;
        let mut bg = 0.0f32;
        let mut bb = 0.0f32;
        let mut bcnt = 0.0f32;
        for x in 0..img.w {
            for &y in &[0, img.h - 1] {
                let p = img.rgba[y * img.w + x];
                br += p[0] as f32;
                bg += p[1] as f32;
                bb += p[2] as f32;
                bcnt += 1.0;
            }
        }
        for y in 1..img.h.saturating_sub(1) {
            for &x in &[0, img.w - 1] {
                let p = img.rgba[y * img.w + x];
                br += p[0] as f32;
                bg += p[1] as f32;
                bb += p[2] as f32;
                bcnt += 1.0;
            }
        }
        let inv_bcnt = 1.0 / bcnt.max(1.0);
        let mean = [br * inv_bcnt, bg * inv_bcnt, bb * inv_bcnt];
        // Distinct background models only (a solid backdrop makes the mean and all four corners identical).
        let mut models = [mean; 5];
        let mut n_models = 1;
        for &cm in &corners {
            if !models[..n_models].contains(&cm) {
                models[n_models] = cm;
                n_models += 1;
            }
        }
        let models = &models[..n_models];
        let min_bg_dist_sq = |p: [u8; 4]| -> f32 {
            let mut d = color_dist_sq(p, models[0]);
            for &m in &models[1..] {
                d = d.min(color_dist_sq(p, m));
            }
            d
        };

        // Compute border color spread on just the top and bottom rows first so we know `thresh_sq`
        // and `loose_sq` before classifying the full image into a compact 1-byte `cls` table.
        let (w, h) = (img.w, img.h);
        let mut var = 0.0f32;
        for &p in &img.rgba[..w] {
            var += min_bg_dist_sq(p);
        }
        for &p in &img.rgba[(h - 1) * w..h * w] {
            var += min_bg_dist_sq(p);
        }
        let std_dev = (var / (2.0 * w as f32).max(1.0)).sqrt();
        let thresh = (std_dev * 1.55 + 22.0).clamp(24.0, 90.0);
        let thresh_sq = thresh * thresh;
        let grad_lim = (thresh * 0.42) * (thresh * 0.42);
        let loose_sq = thresh_sq * 2.1;

        // Classify every pixel into `0 = bg (<= thresh_sq)`, `1 = loose (<= loose_sq)`, `2 = fg`:
        // 4x smaller than `Vec<f32>` (fits in L1) and NEON-accelerated for solid backdrops (`n_models == 1`).
        let mut cls = vec![2u8; n];
        let any_loose = if n_models == 1 {
            classify_bg_1model(&img.rgba, models[0], thresh_sq, loose_sq, &mut cls)
        } else {
            let mut loose = false;
            for (c, &p) in cls.iter_mut().zip(img.rgba.iter()) {
                let d = min_bg_dist_sq(p);
                let v = (d > thresh_sq) as u8 + (d > loose_sq) as u8;
                loose |= v == 1;
                *c = v;
            }
            loose
        };

        // Border-seeded flood-fill directly on `mask` (true = foreground, false = background).
        // First flood all strictly-background (`cls == 0`) regions via fast scanline spans (~3 us).
        // Only if any pixel fell in the loose gradient band (`any_loose`), run a tiny secondary BFS
        // starting strictly from the few `cls == 1` pixels bordering the flooded region.
        mask = vec![true; n];
        scanline_flood_bg(&cls, &mut mask, w, h);
        if any_loose {
            let mut q: Vec<u32> = Vec::with_capacity(256);
            for y in 0..h {
                let row = y * w;
                for x in 0..w {
                    let idx = row + x;
                    if cls[idx] != 1 || !mask[idx] {
                        continue;
                    }
                    let px = img.rgba[idx];
                    let adj_ok = |nidx: usize| -> bool {
                        if mask[nidx] {
                            return false;
                        }
                        let cur = img.rgba[nidx];
                        color_dist_sq(px, [cur[0] as f32, cur[1] as f32, cur[2] as f32]) <= grad_lim
                    };
                    if (x > 0 && adj_ok(idx - 1))
                        || (x + 1 < w && adj_ok(idx + 1))
                        || (y > 0 && adj_ok(idx - w))
                        || (y + 1 < h && adj_ok(idx + w))
                    {
                        mask[idx] = false;
                        q.push(idx as u32);
                    }
                }
            }
            let mut head = 0usize;
            while head < q.len() {
                let idx = q[head] as usize;
                head += 1;
                let (x, y) = (idx % w, idx / w);
                let mut cur_f_opt: Option<[f32; 3]> = None;
                let mut try_nbr = |nidx: usize| {
                    if !mask[nidx] {
                        return;
                    }
                    let c = cls[nidx];
                    if c == 0 {
                        mask[nidx] = false;
                        q.push(nidx as u32);
                    } else if c == 1 {
                        let cur_f = *cur_f_opt.get_or_insert_with(|| {
                            let cur = img.rgba[idx];
                            [cur[0] as f32, cur[1] as f32, cur[2] as f32]
                        });
                        if color_dist_sq(img.rgba[nidx], cur_f) <= grad_lim {
                            mask[nidx] = false;
                            q.push(nidx as u32);
                        }
                    }
                };
                if x > 0 { try_nbr(idx - 1); }
                if x + 1 < w { try_nbr(idx + 1); }
                if y > 0 { try_nbr(idx - w); }
                if y + 1 < h { try_nbr(idx + w); }
            }
        }
    }

    // Morphological close (dilate then erode) using one ping-pong buffer and stack rolling row sums.
    let mut morph_tmp = vec![false; n];
    morph_pass_into(&mask, &mut morph_tmp, img.w, img.h, true);
    morph_pass_into(&morph_tmp, &mut mask, img.w, img.h, false);

    // Guard against empty or full-bleed masks.
    let fg_count = mask.iter().filter(|&&b| b).count();
    if fg_count * 40 < n || fg_count * 100 >= n * 98 {
        // Fall back to a clean centered soft-ellipse subject mask.
        let cx = (img.w as f32 - 1.0) * 0.5;
        let cy = (img.h as f32 - 1.0) * 0.5;
        let inv_rx = 1.0 / (img.w as f32 * 0.42).max(1.0);
        let inv_ry = 1.0 / (img.h as f32 * 0.42).max(1.0);
        for y in 0..img.h {
            let dy = (y as f32 - cy) * inv_ry;
            let dy2 = dy * dy;
            let row = &mut mask[y * img.w..(y + 1) * img.w];
            for (x, m) in row.iter_mut().enumerate() {
                let dx = (x as f32 - cx) * inv_rx;
                *m = dx * dx + dy2 <= 1.0;
            }
        }
    }

    // Clear 1px border so the silhouette always closes cleanly inside the domain.
    mask[..img.w].fill(false);
    mask[(img.h - 1) * img.w..img.h * img.w].fill(false);
    for y in 1..img.h.saturating_sub(1) {
        mask[y * img.w] = false;
        mask[y * img.w + (img.w - 1)] = false;
    }

    // Dominant vibrant sRGB color of foreground pixels (integer min/max + 256-entry reciprocal LUT).
    const INV_U8: [f32; 256] = {
        let mut t = [0.0f32; 256];
        let mut i = 1usize;
        while i < 256 {
            t[i] = 1.0 / i as f32;
            i += 1;
        }
        t
    };
    let mut r_sum = 0.0f32;
    let mut g_sum = 0.0f32;
    let mut b_sum = 0.0f32;
    let mut w_sum = 0.0f32;
    for (&fg, &p) in mask.iter().zip(img.rgba.iter()) {
        if !fg {
            continue;
        }
        let (r8, g8, b8) = (p[0], p[1], p[2]);
        let maxc = r8.max(g8).max(b8);
        let minc = r8.min(g8).min(b8);
        let sat = (maxc - minc) as f32 * INV_U8[maxc as usize];
        let weight = 0.35 + sat * 1.4;
        r_sum += r8 as f32 * weight;
        g_sum += g8 as f32 * weight;
        b_sum += b8 as f32 * weight;
        w_sum += weight;
    }
    let color = if w_sum > 1e-4 {
        let inv = 1.0 / (w_sum * 255.0);
        Color {
            r: (r_sum * inv).clamp(0.0, 1.0),
            g: (g_sum * inv).clamp(0.0, 1.0),
            b: (b_sum * inv).clamp(0.0, 1.0),
            a: 1.0,
        }
    } else {
        Color { r: 0.78, g: 0.76, b: 0.73, a: 1.0 }
    };

    (mask, color)
}

/// Exact 4-connected scanline span flood-fill marking border-reachable `cls == 0` pixels as
/// background (`mask[i] = false`). Visits each background pixel once with ~40x fewer pushes than per-pixel BFS.
fn scanline_flood_bg(cls: &[u8], mask: &mut [bool], w: usize, h: usize) {
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(256);
    let seed_span = |x: usize, y: usize, stack: &mut Vec<(usize, usize)>| {
        let idx = y * w + x;
        if mask[idx] && cls[idx] == 0 {
            stack.push((x, y));
        }
    };
    for x in 0..w {
        seed_span(x, 0, &mut stack);
        seed_span(x, h - 1, &mut stack);
    }
    for y in 1..h.saturating_sub(1) {
        seed_span(0, y, &mut stack);
        seed_span(w - 1, y, &mut stack);
    }
    while let Some((sx, y)) = stack.pop() {
        let row = y * w;
        if !mask[row + sx] || cls[row + sx] != 0 {
            continue;
        }
        let mut x0 = sx;
        while x0 > 0 && mask[row + x0 - 1] && cls[row + x0 - 1] == 0 {
            x0 -= 1;
        }
        let mut x1 = sx + 1;
        while x1 < w && mask[row + x1] && cls[row + x1] == 0 {
            x1 += 1;
        }
        mask[row + x0..row + x1].fill(false);
        for ny in [y.wrapping_sub(1), y + 1] {
            if ny >= h {
                continue;
            }
            let nrow = ny * w;
            let mut in_run = false;
            for x in x0..x1 {
                let bg = mask[nrow + x] && cls[nrow + x] == 0;
                if bg && !in_run {
                    stack.push((x, ny));
                    in_run = true;
                } else if !bg {
                    in_run = false;
                }
            }
        }
    }
}

#[inline(always)]
fn color_dist_sq(p: [u8; 4], m: [f32; 3]) -> f32 {
    let dr = p[0] as f32 - m[0];
    let dg = p[1] as f32 - m[1];
    let db = p[2] as f32 - m[2];
    0.299 * dr * dr + 0.587 * dg * dg + 0.114 * db * db
}

/// Classifies every RGBA pixel against a single background color model into
/// `0 = bg (<= thresh_sq)`, `1 = loose (<= loose_sq)`, `2 = fg`, returning true if any pixel has class `1`.
/// Uses 8-wide NEON `vld4_u8` on AArch64.
fn classify_bg_1model(rgba: &[[u8; 4]], m: [f32; 3], thresh_sq: f32, loose_sq: f32, cls: &mut [u8]) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n = rgba.len();
        let chunks = n & !7;
        let ptr = rgba.as_ptr() as *const u8;
        unsafe {
            let mr = vdupq_n_f32(m[0]);
            let mg = vdupq_n_f32(m[1]);
            let mb = vdupq_n_f32(m[2]);
            let wr = vdupq_n_f32(0.299);
            let wg = vdupq_n_f32(0.587);
            let wb = vdupq_n_f32(0.114);
            let vt = vdupq_n_f32(thresh_sq);
            let vl = vdupq_n_f32(loose_sq);
            let v1 = vdupq_n_u32(1);
            let mut v_any_loose = vdupq_n_u32(0);
            let mut i = 0usize;
            while i < chunks {
                let px = vld4_u8(ptr.add(i * 4));
                let r16 = vmovl_u8(px.0);
                let g16 = vmovl_u8(px.1);
                let b16 = vmovl_u8(px.2);
                // Low 4 pixels
                let dr0 = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16))), mr);
                let dg0 = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16))), mg);
                let db0 = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16))), mb);
                let d0 = vfmaq_f32(vfmaq_f32(vmulq_f32(wr, vmulq_f32(dr0, dr0)), wg, vmulq_f32(dg0, dg0)), wb, vmulq_f32(db0, db0));
                let gt_t0 = vcgtq_f32(d0, vt);
                let gt_l0 = vcgtq_f32(d0, vl);
                v_any_loose = vorrq_u32(v_any_loose, vbicq_u32(gt_t0, gt_l0));
                let c0 = vaddq_u32(vandq_u32(gt_t0, v1), vandq_u32(gt_l0, v1));
                // High 4 pixels
                let dr1 = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(r16))), mr);
                let dg1 = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(g16))), mg);
                let db1 = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(b16))), mb);
                let d1 = vfmaq_f32(vfmaq_f32(vmulq_f32(wr, vmulq_f32(dr1, dr1)), wg, vmulq_f32(dg1, dg1)), wb, vmulq_f32(db1, db1));
                let gt_t1 = vcgtq_f32(d1, vt);
                let gt_l1 = vcgtq_f32(d1, vl);
                v_any_loose = vorrq_u32(v_any_loose, vbicq_u32(gt_t1, gt_l1));
                let c1 = vaddq_u32(vandq_u32(gt_t1, v1), vandq_u32(gt_l1, v1));
                let packed16 = vcombine_u16(vmovn_u32(c0), vmovn_u32(c1));
                vst1_u8(cls.as_mut_ptr().add(i), vmovn_u16(packed16));
                i += 8;
            }
            let mut any_loose = vmaxvq_u32(v_any_loose) != 0;
            for k in chunks..n {
                let d = color_dist_sq(rgba[k], m);
                let v = (d > thresh_sq) as u8 + (d > loose_sq) as u8;
                any_loose |= v == 1;
                cls[k] = v;
            }
            any_loose
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut any_loose = false;
        for (c, &p) in cls.iter_mut().zip(rgba.iter()) {
            let d = color_dist_sq(p, m);
            let v = (d > thresh_sq) as u8 + (d > loose_sq) as u8;
            any_loose |= v == 1;
            *c = v;
        }
        any_loose
    }
}

#[inline]
fn hsum_row(src: &[bool], out: &mut [u8]) {
    let w = src.len();
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let ptr = src.as_ptr() as *const u8;
        let optr = out.as_mut_ptr();
        let mut x = 1usize;
        let end = w.saturating_sub(1);
        unsafe {
            while x + 16 <= end {
                let a = vld1q_u8(ptr.add(x - 1));
                let b = vld1q_u8(ptr.add(x));
                let c = vld1q_u8(ptr.add(x + 1));
                vst1q_u8(optr.add(x), vaddq_u8(vaddq_u8(a, b), c));
                x += 16;
            }
        }
        while x < end {
            out[x] = src[x - 1] as u8 + src[x] as u8 + src[x + 1] as u8;
            x += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for x in 1..w - 1 {
        out[x] = src[x - 1] as u8 + src[x] as u8 + src[x + 1] as u8;
    }
}

#[inline]
fn vthresh_row(r0: &[u8], r1: &[u8], r2: &[u8], dst: &mut [bool], thr: u8) {
    let w = dst.len();
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let end = w.saturating_sub(1);
        let mut x = 1usize;
        unsafe {
            let vthr = vdupq_n_u8(thr);
            let vone = vdupq_n_u8(1);
            let dptr = dst.as_mut_ptr() as *mut u8;
            while x + 16 <= end {
                let s = vaddq_u8(
                    vaddq_u8(vld1q_u8(r0.as_ptr().add(x)), vld1q_u8(r1.as_ptr().add(x))),
                    vld1q_u8(r2.as_ptr().add(x)),
                );
                // `vcgeq_u8` produces 0xFF or 0x00; masking with 1 yields valid `bool` byte values (0 or 1).
                let b = vandq_u8(vcgeq_u8(s, vthr), vone);
                vst1q_u8(dptr.add(x), b);
                x += 16;
            }
        }
        while x < end {
            dst[x] = (r0[x] + r1[x] + r2[x]) >= thr;
            x += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for x in 1..w - 1 {
        dst[x] = (r0[x] + r1[x] + r2[x]) >= thr;
    }
}

/// Separable 3x3 morphological pass writing into `dst` with a 3-row rolling horizontal-sum ring
/// buffer on the stack for `w <= 128` (zero temporary heap allocations) and 16-wide NEON on AArch64.
fn morph_pass_into(src: &[bool], dst: &mut [bool], w: usize, h: usize, dilate: bool) {
    if w < 3 || h < 3 {
        dst.copy_from_slice(src);
        return;
    }
    let thr: u8 = if dilate { 3 } else { 6 };
    dst[..w].copy_from_slice(&src[..w]);
    dst[(h - 1) * w..h * w].copy_from_slice(&src[(h - 1) * w..h * w]);

    if w <= 128 {
        let mut ring = [[0u8; 128]; 3];
        hsum_row(&src[..w], &mut ring[0][..w]);
        hsum_row(&src[w..2 * w], &mut ring[1][..w]);
        for y in 1..h - 1 {
            let next_slot = (y + 1) % 3;
            hsum_row(&src[(y + 1) * w..(y + 2) * w], &mut ring[next_slot][..w]);
            let r0 = &ring[(y - 1) % 3][..w];
            let r1 = &ring[y % 3][..w];
            let r2 = &ring[next_slot][..w];
            let out = &mut dst[y * w..(y + 1) * w];
            out[0] = src[y * w];
            out[w - 1] = src[y * w + w - 1];
            vthresh_row(r0, r1, r2, out, thr);
        }
    } else {
        let mut ring = [vec![0u8; w], vec![0u8; w], vec![0u8; w]];
        hsum_row(&src[..w], &mut ring[0]);
        hsum_row(&src[w..2 * w], &mut ring[1]);
        for y in 1..h - 1 {
            let next_slot = (y + 1) % 3;
            hsum_row(&src[(y + 1) * w..(y + 2) * w], &mut ring[next_slot]);
            let (s0, s1, s2) = ((y - 1) % 3, y % 3, next_slot);
            let out = &mut dst[y * w..(y + 1) * w];
            out[0] = src[y * w];
            out[w - 1] = src[y * w + w - 1];
            vthresh_row(&ring[s0], &ring[s1], &ring[s2], out, thr);
        }
    }
}

#[cfg(test)]
fn morph_pass(src: &[bool], w: usize, h: usize, dilate: bool) -> Vec<bool> {
    let mut dst = vec![false; src.len()];
    morph_pass_into(src, &mut dst, w, h, dilate);
    dst
}

/// Precomputed `0.5 / d` lookup table for `d in 1..256`: replaces every floating-point division in
/// `edt_1d` with a single L1 table load + multiply.
const INV_2D: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 1usize;
    while i < 256 {
        t[i] = 0.5 / (i as f32);
        i += 1;
    }
    t
};

const EDT_INF: f32 = 1e12;
const EDT_FAR_U16: u16 = 30_000;

enum EdtScratch {
    Stack {
        vi: [u16; 128],
        vf: [f32; 128],
        vh: [f32; 128],
        z: [f32; 129],
        bg_row: [f32; 128],
    },
    Heap {
        vi: Vec<u16>,
        vf: Vec<f32>,
        vh: Vec<f32>,
        z: Vec<f32>,
        bg_row: Vec<f32>,
    },
}

impl EdtScratch {
    #[inline]
    fn new(w: usize) -> Self {
        if w <= 128 {
            Self::Stack {
                vi: [0; 128],
                vf: [0.0; 128],
                vh: [0.0; 128],
                z: [0.0; 129],
                bg_row: [0.0; 128],
            }
        } else {
            Self::Heap {
                vi: vec![0; w],
                vf: vec![0.0; w],
                vh: vec![0.0; w],
                z: vec![0.0; w + 1],
                bg_row: vec![0.0; w],
            }
        }
    }

    #[inline]
    fn slices(&mut self, w: usize) -> (&mut [u16], &mut [f32], &mut [f32], &mut [f32], &mut [f32]) {
        match self {
            Self::Stack { vi, vf, vh, z, bg_row } => (&mut vi[..w], &mut vf[..w], &mut vh[..w], &mut z[..w + 1], &mut bg_row[..w]),
            Self::Heap { vi, vf, vh, z, bg_row } => (vi, vf, vh, z, bg_row),
        }
    }
}

/// Exact O(N) 1D squared Euclidean Distance Transform (Felzenszwalb & Huttenlocher, 2012) over a
/// row of `u16` vertical distances (`>= EDT_FAR_U16` denotes unreachable columns). Division-free
/// via `INV_2D` and reuses caller-provided stack scratch across all rows (zero heap allocs).
#[inline]
fn edt_1d_u16_inner(col_d: &[u16], out: &mut [f32], vi: &mut [u16], vf: &mut [f32], vh: &mut [f32], z: &mut [f32]) {
    let n = col_d.len();
    assert!(out.len() >= n && vi.len() >= n && vf.len() >= n && vh.len() >= n && z.len() > n);
    let mut first = 0usize;
    while first < n && col_d[first] >= EDT_FAR_U16 {
        first += 1;
    }
    if first == n {
        out[..n].fill(EDT_INF);
        return;
    }
    // SAFETY: all slice lengths >= n (and z.len() > n) asserted above; k stays in 0..n and dq in 1..n.
    unsafe {
        let f0 = {
            let d = *col_d.get_unchecked(first) as f32;
            d * d
        };
        let q0 = first as f32;
        let mut k = 0usize;
        *vi.get_unchecked_mut(0) = first as u16;
        *vf.get_unchecked_mut(0) = f0 + q0 * q0;
        *vh.get_unchecked_mut(0) = f0;
        *z.get_unchecked_mut(0) = f32::NEG_INFINITY;
        *z.get_unchecked_mut(1) = f32::INFINITY;

        let last = n - 1;
        for q in (first + 1)..n {
            let dc = *col_d.get_unchecked(q);
            if dc >= EDT_FAR_U16 {
                continue;
            }
            // An interior zero flanked by zeros on both sides can never be the nearest site for any
            // non-zero pixel (and zero pixels are never read by `signed_distance_2d`).
            if dc == 0 && *col_d.get_unchecked(q - 1) == 0 && q < last && *col_d.get_unchecked(q + 1) == 0 {
                continue;
            }
            let qf = q as f32;
            let fq = (dc as f32) * (dc as f32);
            let fqq = fq + qf * qf;
            loop {
                let dq = q - *vi.get_unchecked(k) as usize;
                let inv = if dq < 256 { *INV_2D.get_unchecked(dq) } else { 0.5 / dq as f32 };
                let s = (fqq - *vf.get_unchecked(k)) * inv;
                if s <= *z.get_unchecked(k) {
                    if k == 0 {
                        *vi.get_unchecked_mut(0) = q as u16;
                        *vf.get_unchecked_mut(0) = fqq;
                        *vh.get_unchecked_mut(0) = fq;
                        *z.get_unchecked_mut(0) = f32::NEG_INFINITY;
                        *z.get_unchecked_mut(1) = f32::INFINITY;
                        break;
                    }
                    k -= 1;
                } else {
                    k += 1;
                    *vi.get_unchecked_mut(k) = q as u16;
                    *vf.get_unchecked_mut(k) = fqq;
                    *vh.get_unchecked_mut(k) = fq;
                    *z.get_unchecked_mut(k) = s;
                    *z.get_unchecked_mut(k + 1) = f32::INFINITY;
                    break;
                }
            }
        }
        k = 0;
        for q in 0..n {
            let qf = q as f32;
            while *z.get_unchecked(k + 1) < qf {
                k += 1;
            }
            let d = qf - *vi.get_unchecked(k) as f32;
            *out.get_unchecked_mut(q) = d * d + *vh.get_unchecked(k);
        }
    }
}

/// Exact O(W*H) 2D signed distance field in pixel units: negative inside foreground, positive outside.
/// Branchless 8-wide auto-vectorized `u16` column sweep computes both `to_fg` and `to_bg` vertical
/// distances simultaneously into one combined buffer, then runs the division-free 1D row pass with
/// stack-allocated scratch while skipping `to_bg` on all-background rows and `to_fg` on all-foreground rows.
pub(crate) fn signed_distance_2d(mask: &[bool], w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    // Single allocation for both column distance grids (overwritten completely during the forward sweep).
    let mut col_both = vec![0u16; n * 2];
    let (col_fg, col_bg) = col_both.split_at_mut(n);

    let mut stack_flags = [0u8; 128];
    let mut heap_flags: Vec<u8>;
    let row_flags: &mut [u8] = if h <= 128 {
        &mut stack_flags[..h]
    } else {
        heap_flags = vec![0u8; h];
        &mut heap_flags
    };

    let mut stack_rfg = [EDT_FAR_U16; 128];
    let mut stack_rbg = [EDT_FAR_U16; 128];
    let mut heap_rfg: Vec<u16>;
    let mut heap_rbg: Vec<u16>;
    let (run_fg, run_bg): (&mut [u16], &mut [u16]) = if w <= 128 {
        (&mut stack_rfg[..w], &mut stack_rbg[..w])
    } else {
        heap_rfg = vec![EDT_FAR_U16; w];
        heap_rbg = vec![EDT_FAR_U16; w];
        (&mut heap_rfg, &mut heap_rbg)
    };

    let safe_add = h < 30_000;
    for y in 0..h {
        let m = &mask[y * w..(y + 1) * w];
        let cfg = &mut col_fg[y * w..(y + 1) * w];
        let cbg = &mut col_bg[y * w..(y + 1) * w];
        let any_fg = m.iter().any(|&b| b);
        let any_bg = m.iter().any(|&b| !b);
        row_flags[y] = (any_fg as u8) | ((any_bg as u8) << 1);
        if safe_add {
            edt_col_fwd_row(m, run_fg, run_bg, cfg, cbg);
        } else {
            for x in 0..w {
                let fg = m[x];
                let rf = if fg { 0 } else { run_fg[x].saturating_add(1).min(EDT_FAR_U16) };
                let rb = if fg { run_bg[x].saturating_add(1).min(EDT_FAR_U16) } else { 0 };
                run_fg[x] = rf;
                run_bg[x] = rb;
                cfg[x] = rf;
                cbg[x] = rb;
            }
        }
    }

    run_fg.fill(EDT_FAR_U16);
    run_bg.fill(EDT_FAR_U16);
    for y in (0..h).rev() {
        let m = &mask[y * w..(y + 1) * w];
        let cfg = &mut col_fg[y * w..(y + 1) * w];
        let cbg = &mut col_bg[y * w..(y + 1) * w];
        if safe_add {
            edt_col_bwd_row(m, run_fg, run_bg, cfg, cbg);
        } else {
            for x in 0..w {
                let fg = m[x];
                let rf = if fg { 0 } else { run_fg[x].saturating_add(1).min(EDT_FAR_U16) };
                let rb = if fg { run_bg[x].saturating_add(1).min(EDT_FAR_U16) } else { 0 };
                run_fg[x] = rf;
                run_bg[x] = rb;
                cfg[x] = cfg[x].min(rf);
                cbg[x] = cbg[x].min(rb);
            }
        }
    }

    let mut sdf = vec![0.0f32; n];
    let col_fg: &[u16] = col_fg;
    let col_bg: &[u16] = col_bg;
    let row_flags: &[u8] = row_flags;
    let do_row = |y: usize, orow: &mut [f32], scr: &mut EdtScratch| {
        let (vi, vf, vh, z, bg_row) = scr.slices(w);
        let flags = row_flags[y];
        let r0 = y * w;
        let r1 = r0 + w;
        if flags == 2 {
            // All background: only distance to foreground is needed.
            edt_1d_u16_inner(&col_fg[r0..r1], orow, vi, vf, vh, z);
            for v in orow.iter_mut() {
                *v = (v.sqrt() - 0.5).max(0.0);
            }
        } else if flags == 1 {
            // All foreground: only distance to background is needed.
            edt_1d_u16_inner(&col_bg[r0..r1], orow, vi, vf, vh, z);
            for v in orow.iter_mut() {
                *v = -(v.sqrt() - 0.5).max(0.0);
            }
        } else {
            let m = &mask[r0..r1];
            edt_1d_u16_inner(&col_fg[r0..r1], orow, vi, vf, vh, z);
            edt_1d_u16_inner(&col_bg[r0..r1], bg_row, vi, vf, vh, z);
            for x in 0..w {
                let fg = m[x];
                let d2 = if fg { bg_row[x] } else { orow[x] };
                let d = (d2.sqrt() - 0.5).max(0.0);
                orow[x] = if fg { -d } else { d };
            }
        }
    };

    #[cfg(feature = "threads")]
    if n >= 8_192 {
        use rayon::prelude::*;
        let rows_per_chunk = 16usize;
        sdf.par_chunks_mut(w * rows_per_chunk).enumerate().for_each_init(
            || EdtScratch::new(w),
            |scr, (chunk_idx, chunk)| {
                let y0 = chunk_idx * rows_per_chunk;
                for (dy, orow) in chunk.chunks_exact_mut(w).enumerate() {
                    do_row(y0 + dy, orow, scr);
                }
            },
        );
        return sdf;
    }
    let mut scr = EdtScratch::new(w);
    for (y, orow) in sdf.chunks_exact_mut(w).enumerate() {
        do_row(y, orow, &mut scr);
    }
    sdf
}

#[inline]
fn edt_col_fwd_row(m: &[bool], run_fg: &mut [u16], run_bg: &mut [u16], cfg: &mut [u16], cbg: &mut [u16]) {
    let w = m.len();
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let chunks = w & !7;
        let mp = m.as_ptr() as *const u8;
        let mut x = 0usize;
        unsafe {
            let vzero = vdupq_n_u16(0);
            let vone = vdupq_n_u16(1);
            while x < chunks {
                let m16 = vmovl_u8(vld1_u8(mp.add(x)));
                let is_bg = vceqq_u16(m16, vzero); // 0xFFFF where !fg, 0x0000 where fg
                let rf = vandq_u16(is_bg, vaddq_u16(vld1q_u16(run_fg.as_ptr().add(x)), vone));
                let rb = vbicq_u16(vaddq_u16(vld1q_u16(run_bg.as_ptr().add(x)), vone), is_bg);
                vst1q_u16(run_fg.as_mut_ptr().add(x), rf);
                vst1q_u16(run_bg.as_mut_ptr().add(x), rb);
                vst1q_u16(cfg.as_mut_ptr().add(x), rf);
                vst1q_u16(cbg.as_mut_ptr().add(x), rb);
                x += 8;
            }
        }
        for x in chunks..w {
            let fg = m[x];
            let rf = if fg { 0 } else { run_fg[x] + 1 };
            let rb = if fg { run_bg[x] + 1 } else { 0 };
            run_fg[x] = rf;
            run_bg[x] = rb;
            cfg[x] = rf;
            cbg[x] = rb;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for x in 0..w {
        let fg = m[x];
        let rf = if fg { 0 } else { run_fg[x] + 1 };
        let rb = if fg { run_bg[x] + 1 } else { 0 };
        run_fg[x] = rf;
        run_bg[x] = rb;
        cfg[x] = rf;
        cbg[x] = rb;
    }
}

#[inline]
fn edt_col_bwd_row(m: &[bool], run_fg: &mut [u16], run_bg: &mut [u16], cfg: &mut [u16], cbg: &mut [u16]) {
    let w = m.len();
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let chunks = w & !7;
        let mp = m.as_ptr() as *const u8;
        let mut x = 0usize;
        unsafe {
            let vzero = vdupq_n_u16(0);
            let vone = vdupq_n_u16(1);
            while x < chunks {
                let m16 = vmovl_u8(vld1_u8(mp.add(x)));
                let is_bg = vceqq_u16(m16, vzero);
                let rf = vandq_u16(is_bg, vaddq_u16(vld1q_u16(run_fg.as_ptr().add(x)), vone));
                let rb = vbicq_u16(vaddq_u16(vld1q_u16(run_bg.as_ptr().add(x)), vone), is_bg);
                vst1q_u16(run_fg.as_mut_ptr().add(x), rf);
                vst1q_u16(run_bg.as_mut_ptr().add(x), rb);
                vst1q_u16(cfg.as_mut_ptr().add(x), vminq_u16(vld1q_u16(cfg.as_ptr().add(x)), rf));
                vst1q_u16(cbg.as_mut_ptr().add(x), vminq_u16(vld1q_u16(cbg.as_ptr().add(x)), rb));
                x += 8;
            }
        }
        for x in chunks..w {
            let fg = m[x];
            let rf = if fg { 0 } else { run_fg[x] + 1 };
            let rb = if fg { run_bg[x] + 1 } else { 0 };
            run_fg[x] = rf;
            run_bg[x] = rb;
            cfg[x] = cfg[x].min(rf);
            cbg[x] = cbg[x].min(rb);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for x in 0..w {
        let fg = m[x];
        let rf = if fg { 0 } else { run_fg[x] + 1 };
        let rb = if fg { run_bg[x] + 1 } else { 0 };
        run_fg[x] = rf;
        run_bg[x] = rb;
        cfg[x] = cfg[x].min(rf);
        cbg[x] = cbg[x].min(rb);
    }
}

/// Bilinear sample of a 2D scalar buffer from precomputed `(row0, row1, x0, x1, tx, ty)`.
#[inline(always)]
fn sample_bilinear_pre(row0: &[f32], row1: &[f32], x0: usize, x1: usize, tx: f32, ty: f32) -> f32 {
    let om_tx = 1.0 - tx;
    let a = row0[x0] * om_tx + row0[x1] * tx;
    let b = row1[x0] * om_tx + row1[x1] * tx;
    a + (b - a) * ty
}

/// Builds a watertight 3D `GenPart` mesh from an RGBA8 image (+ optional per-pixel depth in `[0..1]` where larger = closer).
pub fn build_image_part(
    width: u32,
    height: u32,
    rgba: &[u8],
    depth: Option<&[f32]>,
    opts: &Image3DOptions,
) -> Result<GenPart, String> {
    // Cap working 2D image to match target voxel grid detail (eliminates excess 2D EDT work).
    let work_cap = ((opts.resolution as usize) * 2).clamp(64, 128);
    let work = downsample_image(width, height, rgba, depth, work_cap)?;
    let (mask, dominant) = segment_foreground(&work);
    let color = opts.color_override.unwrap_or(dominant);

    // Find tight foreground bounding box in pixel coordinates.
    let (mut min_x, mut min_y) = (work.w, work.h);
    let (mut max_x, mut max_y) = (0usize, 0usize);
    for y in 0..work.h {
        for x in 0..work.w {
            if mask[y * work.w + x] {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    if min_x >= max_x || min_y >= max_y {
        return Err("could not find foreground subject in image".into());
    }

    let sd2d = signed_distance_2d(&mask, work.w, work.h);
    // Compute normalized height/luminance map (0..1) for subtle surface relief (unused in Lathe mode).
    let lum = if opts.mode == Image3DMode::Lathe {
        Vec::new()
    } else {
        let mut l = vec![0.0f32; work.w * work.h];
        if let Some(ref d) = work.depth {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for (&fg, &dv) in mask.iter().zip(d.iter()) {
                if fg && dv.is_finite() {
                    lo = lo.min(dv);
                    hi = hi.max(dv);
                }
            }
            let inv_span = 1.0 / (hi - lo).max(1e-4);
            for (o, &dv) in l.iter_mut().zip(d.iter()) {
                *o = ((dv - lo) * inv_span).clamp(0.0, 1.0);
            }
        } else {
            const WR: f32 = 0.299 / 255.0;
            const WG: f32 = 0.587 / 255.0;
            const WB: f32 = 0.114 / 255.0;
            for (o, p) in l.iter_mut().zip(work.rgba.iter()) {
                *o = WR * p[0] as f32 + WG * p[1] as f32 + WB * p[2] as f32;
            }
        }
        // Smooth luminance slightly in-place so micro-noise doesn't pit the 3D surface (zero extra allocs).
        box_blur_2d_in_place(&mut l, work.w, work.h);
        l
    };

    let max_in_dist = sd2d.iter().fold(0.0f32, |m, &d| m.max(-d)).max(1.0);

    // Map foreground into Sculptor coordinates: mm, Z-up, longest side = 160 mm.
    let crop_w = (max_x - min_x + 1) as f32;
    let crop_h = (max_y - min_y + 1) as f32;
    let longest_px = crop_w.max(crop_h).max(1.0);
    let target_mm = 160.0f32;
    let mm_per_px = target_mm / longest_px;
    let width_mm = crop_w * mm_per_px;
    let height_mm = crop_h * mm_per_px;

    // Precompute lathe radius profile if in Lathe mode.
    let mut lathe_r = Vec::new();
    if opts.mode == Image3DMode::Lathe {
        lathe_r.resize(work.h, 0.0f32);
        for y in 0..work.h {
            let row = &mask[y * work.w..(y + 1) * work.w];
            if let Some(left) = row.iter().position(|&b| b) {
                let right = row.iter().rposition(|&b| b).unwrap_or(left);
                lathe_r[y] = ((right - left + 1) as f32 * 0.5) * mm_per_px;
            }
        }
        // Smooth lathe profile slightly to eliminate stair-stepping.
        let orig = lathe_r.clone();
        for y in 1..work.h.saturating_sub(1) {
            lathe_r[y] = 0.25 * orig[y - 1] + 0.5 * orig[y] + 0.25 * orig[y + 1];
        }
    }

    let max_depth_mm = (max_in_dist * mm_per_px * 1.15).clamp(10.0, target_mm * 0.45) * (opts.depth_scale / 0.34);

    // Predict resolution directly from triangle budget so attempt 1 succeeds without a throwaway mesh.
    let area_ratio = ((crop_w * crop_h) / (longest_px * longest_px)).clamp(0.2, 1.0);
    let budget_res = ((opts.max_triangles as f32 / (4.4 * area_ratio)).sqrt()).floor() as u32;
    let mut res = opts.resolution.min(budget_res.max(16));
    let mut mesh = Mesh { positions: vec![], faces: vec![], smooth_shading: true };

    // Reusable 3D field & row-sign bitmask scratch buffers shared across probe and attempt iterations.
    let mut field_buf: Vec<f32> = Vec::new();
    let mut sign_buf: Vec<u128> = Vec::new();

    let w_m1 = (work.w - 1).max(1) as f32;
    let h_m1 = (work.h - 1).max(1) as f32;
    let cx_px = (min_x + max_x) as f32 * 0.5;
    let inv_mm_per_px = 1.0 / mm_per_px;
    let inv_max_in = 1.0 / max_in_dist;
    let fx_lo = -0.05 * w_m1;
    let fx_hi = 1.05 * w_m1;
    let fy_lo = -0.05 * h_m1;
    let fy_hi = 1.05 * h_m1;
    let lathe_max_r = if opts.mode == Image3DMode::Lathe {
        lathe_r.iter().copied().fold(10.0f32, f32::max)
    } else {
        0.0
    };

    let build_field = |res: u32, field: &mut Vec<f32>, signs: &mut Vec<u128>| -> Grid {
        let h = target_mm / res as f32;
        let pad = 2.0 * h;
        let (hx, hy, hz) = match opts.mode {
            Image3DMode::Lathe => (lathe_max_r + pad, lathe_max_r + pad, height_mm + 2.0 * pad),
            Image3DMode::Item | Image3DMode::Relief => (width_mm * 0.5 + pad, max_depth_mm + pad, height_mm + 2.0 * pad),
        };
        let origin = Vec3::new(-hx, -hy, -pad);
        let nx = ((2.0 * hx / h).ceil() as usize + 1).clamp(8, 128);
        let ny = ((2.0 * hy / h).ceil() as usize + 1).clamp(8, 96);
        let nz = ((hz / h).ceil() as usize + 1).clamp(8, 128);
        let grid = Grid { origin, h, n: [nx, ny, nz] };

        let total = nx * ny * nz;
        field.resize(total, 0.0);
        signs.resize(ny * nz, 0);

        // Precompute X-axis bilinear lookup coordinates ONCE for all `nz` slices.
        let mut x0_tab = [0usize; 128];
        let mut x1_tab = [0usize; 128];
        let mut tx_tab = [0.0f32; 128];
        let mut in_x_tab = [false; 128];
        if opts.mode != Image3DMode::Lathe {
            for i in 0..nx {
                let x_mm = origin.x + i as f32 * h;
                let fx = cx_px + x_mm * inv_mm_per_px;
                if fx >= fx_lo && fx <= fx_hi {
                    in_x_tab[i] = true;
                    let fxc = fx.clamp(0.0, w_m1);
                    let x0 = fxc.floor() as usize;
                    x0_tab[i] = x0;
                    x1_tab[i] = (x0 + 1).min(work.w - 1);
                    tx_tab[i] = fxc - x0 as f32;
                }
            }
        }
        let x_mm_tab: [f32; 128] = std::array::from_fn(|i| origin.x + i as f32 * h);
        let x2_tab: [f32; 128] = std::array::from_fn(|i| x_mm_tab[i] * x_mm_tab[i]);
        let slice_len = nx * ny;

        // Precompute (sd_mm, half_thick/front_depth) per (i, k) pixel column ONCE per slice and emit
        // both the `f32` field and packed `u128` row sign bitmasks in a single vectorized pass.
        let eval_slice = |k: usize, slice: &mut [f32], s_rows: &mut [u128]| {
            let z_mm = origin.z + k as f32 * h;
            let py = max_y as f32 - z_mm * inv_mm_per_px;
            let z_cap_sd = (-z_mm).max(z_mm - height_mm);

            match opts.mode {
                Image3DMode::Lathe => {
                    let py_clamped = py.clamp(0.0, h_m1);
                    let y0 = py_clamped.floor() as usize;
                    let y1 = (y0 + 1).min(work.h - 1);
                    let t = py_clamped - y0 as f32;
                    let prof_r = lathe_r[y0] * (1.0 - t) + lathe_r[y1] * t;
                    let oz = z_cap_sd.max(0.0);
                    let oz2 = oz * oz;
                    for j in 0..ny {
                        let y_mm = origin.y + j as f32 * h;
                        let y2 = y_mm * y_mm;
                        let row = &mut slice[j * nx..(j + 1) * nx];
                        let mut bits = 0u128;
                        for (i, cell) in row.iter_mut().enumerate() {
                            let radial_sd = (x2_tab[i] + y2).sqrt() - prof_r;
                            let ox = radial_sd.max(0.0);
                            let d = (ox * ox + oz2).sqrt() + radial_sd.max(z_cap_sd).min(0.0);
                            *cell = d;
                            bits |= ((d < 0.0) as u128) << i;
                        }
                        s_rows[j] = bits;
                    }
                }
                Image3DMode::Item => {
                    if py < fy_lo || py > fy_hi {
                        slice.fill(100.0);
                        s_rows.fill(0);
                        return;
                    }
                    let fyc = py.clamp(0.0, h_m1);
                    let y0 = fyc.floor() as usize;
                    let y1 = (y0 + 1).min(work.h - 1);
                    let ty = fyc - y0 as f32;
                    let sd_r0 = &sd2d[y0 * work.w..(y0 + 1) * work.w];
                    let sd_r1 = &sd2d[y1 * work.w..(y1 + 1) * work.w];
                    let lum_r0 = &lum[y0 * work.w..(y0 + 1) * work.w];
                    let lum_r1 = &lum[y1 * work.w..(y1 + 1) * work.w];

                    let mut col_sd = [100.0f32; 128];
                    let mut col_sd2 = [10_000.0f32; 128];
                    let mut col_ht = [0.0f32; 128];
                    for i in 0..nx {
                        if in_x_tab[i] {
                            let (x0, x1, tx) = (x0_tab[i], x1_tab[i], tx_tab[i]);
                            let sd_px = sample_bilinear_pre(sd_r0, sd_r1, x0, x1, tx, ty);
                            let sd_mm = sd_px * mm_per_px;
                            col_sd[i] = sd_mm;
                            col_sd2[i] = sd_mm * sd_mm;
                            if sd_mm < 0.0 {
                                let lum_v = sample_bilinear_pre(lum_r0, lum_r1, x0, x1, tx, ty);
                                let in_ratio = ((-sd_px) * inv_max_in).clamp(0.0, 1.0);
                                col_ht[i] = max_depth_mm * in_ratio.sqrt() * (0.80 + 0.24 * lum_v);
                            }
                        }
                    }
                    fill_item_slice(slice, s_rows, nx, ny, origin.y, h, &col_sd, &col_sd2, &col_ht);
                }
                Image3DMode::Relief => {
                    if py < fy_lo || py > fy_hi {
                        slice.fill(100.0);
                        s_rows.fill(0);
                        return;
                    }
                    let fyc = py.clamp(0.0, h_m1);
                    let y0 = fyc.floor() as usize;
                    let y1 = (y0 + 1).min(work.h - 1);
                    let ty = fyc - y0 as f32;
                    let sd_r0 = &sd2d[y0 * work.w..(y0 + 1) * work.w];
                    let sd_r1 = &sd2d[y1 * work.w..(y1 + 1) * work.w];
                    let lum_r0 = &lum[y0 * work.w..(y0 + 1) * work.w];
                    let lum_r1 = &lum[y1 * work.w..(y1 + 1) * work.w];
                    let back = -max_depth_mm * 0.22;
                    let mut col_sd = [100.0f32; 128];
                    let mut col_front = [0.0f32; 128];
                    for i in 0..nx {
                        if in_x_tab[i] {
                            let (x0, x1, tx) = (x0_tab[i], x1_tab[i], tx_tab[i]);
                            let sd_px = sample_bilinear_pre(sd_r0, sd_r1, x0, x1, tx, ty);
                            col_sd[i] = sd_px * mm_per_px;
                            let lum_v = sample_bilinear_pre(lum_r0, lum_r1, x0, x1, tx, ty);
                            col_front[i] = max_depth_mm * (0.20 + 0.80 * lum_v);
                        }
                    }
                    let (csd, cfr) = (&col_sd[..nx], &col_front[..nx]);
                    for j in 0..ny {
                        let y_mm = origin.y + j as f32 * h;
                        let row = &mut slice[j * nx..(j + 1) * nx];
                        let mut bits = 0u128;
                        for (i, ((o, &sd_mm), &front)) in row.iter_mut().zip(csd).zip(cfr).enumerate() {
                            let dy = (back - y_mm).max(y_mm - front);
                            let os = sd_mm.max(0.0);
                            let oy = dy.max(0.0);
                            let d = (os * os + oy * oy).sqrt() + sd_mm.max(dy).min(0.0);
                            let val = if sd_mm >= 99.0 { 100.0 } else { d };
                            *o = val;
                            bits |= ((val < 0.0) as u128) << i;
                        }
                        s_rows[j] = bits;
                    }
                }
            }
        };

        #[cfg(feature = "threads")]
        if total >= 262_144 {
            use rayon::prelude::*;
            field
                .par_chunks_mut(slice_len)
                .zip(signs.par_chunks_mut(ny))
                .enumerate()
                .for_each(|(k, (slice, s_rows))| eval_slice(k, slice, s_rows));
        } else {
            for (k, (slice, s_rows)) in field.chunks_mut(slice_len).zip(signs.chunks_mut(ny)).enumerate() {
                eval_slice(k, slice, s_rows);
            }
        }
        #[cfg(not(feature = "threads"))]
        for (k, (slice, s_rows)) in field.chunks_mut(slice_len).zip(signs.chunks_mut(ny)).enumerate() {
            eval_slice(k, slice, s_rows);
        }

        grid
    };

    // Surface-nets quad counts scale with res^2, so when the starting resolution would blow the budget, one
    // cheap half-resolution field predicts by how much; the loop below still verifies on the real field.
    // For `Item` mode, only the sign bitmask (`sd_mm < 0 && y_abs < col_ht`) is needed for `count_crossings_bits`,
    // avoiding all `sqrt` and `field_buf` writes during the probe pass!
    if res >= 28 {
        let probe = res / 2;
        let probe_n = if opts.mode == Image3DMode::Item {
            let h = target_mm / probe as f32;
            let pad = 2.0 * h;
            let (hx, hy, hz) = (width_mm * 0.5 + pad, max_depth_mm + pad, height_mm + 2.0 * pad);
            let origin = Vec3::new(-hx, -hy, -pad);
            let nx = ((2.0 * hx / h).ceil() as usize + 1).clamp(8, 128);
            let ny = ((2.0 * hy / h).ceil() as usize + 1).clamp(8, 96);
            let nz = ((hz / h).ceil() as usize + 1).clamp(8, 128);
            sign_buf.resize(ny * nz, 0);
            let mut x0_tab = [0usize; 128];
            let mut x1_tab = [0usize; 128];
            let mut tx_tab = [0.0f32; 128];
            let mut in_x_tab = [false; 128];
            for i in 0..nx {
                let fx = cx_px + (origin.x + i as f32 * h) * inv_mm_per_px;
                if fx >= fx_lo && fx <= fx_hi {
                    in_x_tab[i] = true;
                    let fxc = fx.clamp(0.0, w_m1);
                    let x0 = fxc.floor() as usize;
                    x0_tab[i] = x0;
                    x1_tab[i] = (x0 + 1).min(work.w - 1);
                    tx_tab[i] = fxc - x0 as f32;
                }
            }
            for (k, s_rows) in sign_buf.chunks_mut(ny).enumerate() {
                let z_mm = origin.z + k as f32 * h;
                let py = max_y as f32 - z_mm * inv_mm_per_px;
                if py < fy_lo || py > fy_hi {
                    s_rows.fill(0);
                    continue;
                }
                let fyc = py.clamp(0.0, h_m1);
                let y0 = fyc.floor() as usize;
                let y1 = (y0 + 1).min(work.h - 1);
                let ty = fyc - y0 as f32;
                let sd_r0 = &sd2d[y0 * work.w..(y0 + 1) * work.w];
                let sd_r1 = &sd2d[y1 * work.w..(y1 + 1) * work.w];
                let lum_r0 = &lum[y0 * work.w..(y0 + 1) * work.w];
                let lum_r1 = &lum[y1 * work.w..(y1 + 1) * work.w];
                let mut col_ht = [-1.0f32; 128];
                let mut any_in = false;
                for i in 0..nx {
                    if in_x_tab[i] {
                        let (x0, x1, tx) = (x0_tab[i], x1_tab[i], tx_tab[i]);
                        let sd_px = sample_bilinear_pre(sd_r0, sd_r1, x0, x1, tx, ty);
                        if sd_px < 0.0 {
                            let lum_v = sample_bilinear_pre(lum_r0, lum_r1, x0, x1, tx, ty);
                            let in_ratio = ((-sd_px) * inv_max_in).clamp(0.0, 1.0);
                            col_ht[i] = max_depth_mm * in_ratio.sqrt() * (0.80 + 0.24 * lum_v);
                            any_in = true;
                        }
                    }
                }
                if !any_in {
                    s_rows.fill(0);
                    continue;
                }
                for (j, sr) in s_rows.iter_mut().enumerate() {
                    let y_abs = (origin.y + j as f32 * h).abs();
                    let mut bits = 0u128;
                    for i in 0..nx {
                        bits |= ((y_abs < col_ht[i]) as u128) << i;
                    }
                    *sr = bits;
                }
            }
            [nx, ny, nz]
        } else {
            build_field(probe, &mut field_buf, &mut sign_buf).n
        };
        let scale = (res as f32 / probe as f32).powi(2);
        let est = count_crossings_bits(&sign_buf, probe_n) as f32 * 2.0 * scale;
        if est > opts.max_triangles as f32 {
            let ratio = (opts.max_triangles as f32 / est).sqrt() * 0.92;
            res = ((res as f32) * ratio).floor().max(14.0).min((res - 1) as f32) as u32;
        }
    }

    for _attempt in 0..4 {
        let grid = build_field(res, &mut field_buf, &mut sign_buf);
        // One quad per crossing sample edge: the triangle count is known from the bitmasks in <1 us, so an
        // over-budget resolution is rejected without extracting, smoothing and checking a throwaway mesh.
        let predicted = count_crossings_bits(&sign_buf, grid.n) * 2;
        let mut tris = predicted;
        if predicted <= opts.max_triangles || res <= 14 {
            let (mut pos_a, quads) = surface_nets_flat(&field_buf, &sign_buf, &grid, predicted / 2);
            tris = quads.len() * 2;
            if tris <= opts.max_triangles || res <= 14 {
                // `taubin_smooth_quads_a` mutates positions while `quads_are_manifold` only inspects
                // quad topology: run them in parallel under `threads` so manifold validation is free!
                let nv = pos_a.len();
                let do_smooth = opts.smooth_iters > 0 && nv > 0;
                #[cfg(feature = "threads")]
                let already_manifold = if do_smooth && quads.len() >= 1_024 {
                    let (_, m) = rayon::join(
                        || taubin_smooth_quads_a(&mut pos_a, &quads, opts.smooth_iters),
                        || quads_are_manifold(&quads, nv),
                    );
                    m
                } else {
                    if do_smooth {
                        taubin_smooth_quads_a(&mut pos_a, &quads, opts.smooth_iters);
                    }
                    quads_are_manifold(&quads, nv)
                };
                #[cfg(not(feature = "threads"))]
                let already_manifold = {
                    if do_smooth {
                        taubin_smooth_quads_a(&mut pos_a, &quads, opts.smooth_iters);
                    }
                    quads_are_manifold(&quads, nv)
                };
                // Convert flat quads into Mesh faces once (zero intermediate throwaway Vec<Vec<u32>> allocations).
                let faces = quads_to_faces(&quads, &pos_a, opts.quads);
                // Recenter X/Y and sit bottom cleanly at Z = 0 while converting Vec3A -> Vec3 in one pass.
                let (mut lo, mut hi) = (Vec3A::splat(f32::INFINITY), Vec3A::splat(f32::NEG_INFINITY));
                for &p in &pos_a {
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
                let offset = Vec3A::new((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, lo.z);
                let positions: Vec<Vec3> = pos_a.iter().map(|&p| Vec3::from(p - offset)).collect();
                mesh = Mesh { positions, faces, smooth_shading: true };
                if !already_manifold {
                    make_manifold(&mut mesh);
                }
                break;
            }
        }
        let ratio = (opts.max_triangles as f32 / tris as f32).sqrt() * 0.92;
        res = ((res as f32) * ratio).floor().max(14.0).min((res - 1) as f32) as u32;
    }

    if mesh.faces.is_empty() {
        return Err("image reconstruction produced an empty mesh".into());
    }

    Ok(GenPart {
        name: opts.name.clone(),
        material: Material {
            name: opts.name.clone(),
            base_color: color,
            metallic: opts.metallic,
            roughness: opts.roughness,
        },
        pivot: Vec3::ZERO,
        mesh,
    })
}

/// Fills one Z-slice of `Item`-mode SDF values and extracts packed `u128` row sign bitmasks.
/// On AArch64 uses NEON 4-wide `vsqrtq_f32` / `vbslq_f32` and extracts 4 sign bits in one reduction.
#[inline]
fn fill_item_slice(
    slice: &mut [f32],
    s_rows: &mut [u128],
    nx: usize,
    ny: usize,
    origin_y: f32,
    h: f32,
    col_sd: &[f32; 128],
    col_sd2: &[f32; 128],
    col_ht: &[f32; 128],
) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        debug_assert!(nx <= 128 && slice.len() >= nx * ny && s_rows.len() >= ny);
        let vzero = unsafe { vdupq_n_f32(0.0) };
        let v99 = unsafe { vdupq_n_f32(99.0) };
        let v100 = unsafe { vdupq_n_f32(100.0) };
        let bit_weights: [u32; 4] = [1, 2, 4, 8];
        let vw = unsafe { vld1q_u32(bit_weights.as_ptr()) };
        let chunks = nx & !3;
        for j in 0..ny {
            let y_abs = (origin_y + j as f32 * h).abs();
            let y2 = y_abs * y_abs;
            let vy_abs = unsafe { vdupq_n_f32(y_abs) };
            let vy2 = unsafe { vdupq_n_f32(y2) };
            let row = &mut slice[j * nx..(j + 1) * nx];
            let mut bits = 0u128;
            let mut i = 0usize;
            while i < chunks {
                unsafe {
                    let vsd = vld1q_f32(col_sd.as_ptr().add(i));
                    let vsd2 = vld1q_f32(col_sd2.as_ptr().add(i));
                    let vht = vld1q_f32(col_ht.as_ptr().add(i));
                    let vout = vsqrtq_f32(vaddq_f32(vsd2, vy2));
                    let vdy = vsubq_f32(vy_abs, vht);
                    let vin = vbslq_f32(vcgtq_f32(vdy, vzero), vdy, vmaxq_f32(vdy, vsd));
                    let vd = vbslq_f32(vcgeq_f32(vsd, vzero), vout, vin);
                    let vfinal = vbslq_f32(vcgeq_f32(vsd, v99), v100, vd);
                    vst1q_f32(row.as_mut_ptr().add(i), vfinal);
                    let vneg = vcltzq_f32(vfinal);
                    let b4 = vaddvq_u32(vandq_u32(vneg, vw));
                    bits |= (b4 as u128) << i;
                }
                i += 4;
            }
            while i < nx {
                let sd_mm = col_sd[i];
                let ht = col_ht[i];
                let outside = (col_sd2[i] + y2).sqrt();
                let dy = y_abs - ht;
                let inside = if dy > 0.0 { dy } else { dy.max(sd_mm) };
                let d = if sd_mm >= 0.0 { outside } else { inside };
                let val = if sd_mm >= 99.0 { 100.0 } else { d };
                row[i] = val;
                bits |= ((val < 0.0) as u128) << i;
                i += 1;
            }
            s_rows[j] = bits;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let (csd, csd2, cht) = (&col_sd[..nx], &col_sd2[..nx], &col_ht[..nx]);
        for j in 0..ny {
            let y_abs = (origin_y + j as f32 * h).abs();
            let y2 = y_abs * y_abs;
            let row = &mut slice[j * nx..(j + 1) * nx];
            let mut bits = 0u128;
            for i in 0..nx {
                let sd_mm = csd[i];
                let outside = (csd2[i] + y2).sqrt();
                let dy = y_abs - cht[i];
                let inside = if dy > 0.0 { dy } else { dy.max(sd_mm) };
                let d = if sd_mm >= 0.0 { outside } else { inside };
                let val = if sd_mm >= 99.0 { 100.0 } else { d };
                row[i] = val;
                bits |= ((val < 0.0) as u128) << i;
            }
            s_rows[j] = bits;
        }
    }
}

/// Host JSON entry point matching `text3d::host::generate_json` schema.
pub fn image_to_3d_json(
    width: u32,
    height: u32,
    rgba: &[u8],
    depth: Option<&[f32]>,
    options_json: &str,
) -> String {
    let opts = Image3DOptions::from_json(options_json);
    match build_image_part(width, height, rgba, depth, &opts) {
        Ok(part) => {
            let tris = triangle_count(&part.mesh);
            let parts = vec![part];
            json!({
                "ok": true,
                "understood": true,
                "notes": [format!("Reconstructed {} from photo ({:?}, {} tris)", opts.name, opts.mode, tris)],
                "triangles": tris,
                "program": { "name": opts.name, "realScale": 1.0, "parts": [] },
                "parts": parts_json(&parts),
            })
            .to_string()
        }
        Err(e) => error_json(&e),
    }
}

/// In-place separable 3-tap horizontal + vertical box blur using a 3-row stack ring buffer for
/// `w <= 128` (zero temporary heap allocations).
fn box_blur_2d_in_place(buf: &mut [f32], w: usize, h: usize) {
    if w < 3 || h < 3 {
        return;
    }
    const INV3: f32 = 1.0 / 3.0;
    #[inline]
    fn hblur_row(src: &[f32], dst: &mut [f32]) {
        let w = src.len();
        dst[0] = src[0];
        dst[w - 1] = src[w - 1];
        for x in 1..(w - 1) {
            dst[x] = (src[x - 1] + src[x] + src[x + 1]) * INV3;
        }
    }
    if w <= 128 {
        let mut ring = [[0.0f32; 128]; 3];
        hblur_row(&buf[..w], &mut ring[0][..w]);
        buf[..w].copy_from_slice(&ring[0][..w]);
        hblur_row(&buf[w..2 * w], &mut ring[1][..w]);
        for y in 1..(h - 1) {
            let next_slot = (y + 1) % 3;
            hblur_row(&buf[(y + 1) * w..(y + 2) * w], &mut ring[next_slot][..w]);
            let r0 = &ring[(y - 1) % 3][..w];
            let r1 = &ring[y % 3][..w];
            let r2 = &ring[next_slot][..w];
            let out = &mut buf[y * w..(y + 1) * w];
            for x in 0..w {
                out[x] = (r0[x] + r1[x] + r2[x]) * INV3;
            }
        }
        buf[(h - 1) * w..h * w].copy_from_slice(&ring[(h - 1) % 3][..w]);
    } else {
        let mut tmp = buf.to_vec();
        for y in 0..h {
            hblur_row(&buf[y * w..(y + 1) * w], &mut tmp[y * w..(y + 1) * w]);
        }
        buf[..w].copy_from_slice(&tmp[..w]);
        buf[(h - 1) * w..h * w].copy_from_slice(&tmp[(h - 1) * w..h * w]);
        for y in 1..(h - 1) {
            let (r0, r1, r2) = (&tmp[(y - 1) * w..y * w], &tmp[y * w..(y + 1) * w], &tmp[(y + 1) * w..(y + 2) * w]);
            let out = &mut buf[y * w..(y + 1) * w];
            for x in 0..w {
                out[x] = (r0[x] + r1[x] + r2[x]) * INV3;
            }
        }
    }
}

/// Zero-heap-indirection volume-preserving Taubin smoothing directly in-place on `Vec3A` positions.
/// Topology is fixed, so vertex valence (and its reciprocal) is computed once for all passes;
/// opposing diagonal sums `p0+p2` and `p1+p3` are shared across both adjacent corners of each quad.
fn taubin_smooth_quads_a(pos: &mut [Vec3A], quads: &[[u32; 4]], iters: u32) {
    let n = pos.len();
    if n == 0 || quads.is_empty() {
        return;
    }
    let mut max_v = 0u32;
    for q in quads {
        max_v = max_v.max(q[0]).max(q[1]).max(q[2]).max(q[3]);
    }
    assert!((max_v as usize) < n, "quad vertex index out of bounds");

    let mut inv_cnt = vec![0.0f32; n];
    for q in quads {
        // SAFETY: `max_v < n == inv_cnt.len()` verified above.
        unsafe {
            *inv_cnt.get_unchecked_mut(q[0] as usize) += 2.0;
            *inv_cnt.get_unchecked_mut(q[1] as usize) += 2.0;
            *inv_cnt.get_unchecked_mut(q[2] as usize) += 2.0;
            *inv_cnt.get_unchecked_mut(q[3] as usize) += 2.0;
        }
    }
    for c in inv_cnt.iter_mut() {
        if *c > 0.0 {
            *c = c.recip();
        }
    }
    let mut sum = vec![Vec3A::ZERO; n];
    for _ in 0..iters {
        for &factor in &[0.42f32, -0.44f32] {
            for q in quads {
                let (i0, i1, i2, i3) = (q[0] as usize, q[1] as usize, q[2] as usize, q[3] as usize);
                // SAFETY: `max_v < n == pos.len() == sum.len()` verified above; each `sum` mutation
                // completes before the next so no aliasing `&mut` references coexist.
                unsafe {
                    let p0 = *pos.get_unchecked(i0);
                    let p1 = *pos.get_unchecked(i1);
                    let p2 = *pos.get_unchecked(i2);
                    let p3 = *pos.get_unchecked(i3);
                    let p02 = p0 + p2;
                    let p13 = p1 + p3;
                    *sum.get_unchecked_mut(i0) += p13;
                    *sum.get_unchecked_mut(i1) += p02;
                    *sum.get_unchecked_mut(i2) += p13;
                    *sum.get_unchecked_mut(i3) += p02;
                }
            }
            // Fuse clearing `sum` into the update pass while each cache line is already hot in L1.
            for ((p, s), &ic) in pos.iter_mut().zip(sum.iter_mut()).zip(inv_cnt.iter()) {
                if ic > 0.0 {
                    *p += (*s * ic - *p) * factor;
                }
                *s = Vec3A::ZERO;
            }
        }
    }
}

/// True when the flat quad list is a closed 2-manifold (every directed edge once, with its reverse once).
/// Uses a single 16-byte-aligned `[u32; 4]` zero-sentinel per-vertex forward-edge table (since `hi > lo >= 0`,
/// `(hi << 2) | bit != 0`, eliminating any separate `cnt` array or CSR/sorting pass); falls back to
/// `quads_manifold` on any vertex with > 4 forward neighbors.
fn quads_are_manifold(quads: &[[u32; 4]], num_vertices: usize) -> bool {
    if quads.is_empty() {
        return false;
    }
    let mut max_v = 0u32;
    for q in quads {
        max_v = max_v.max(q[0]).max(q[1]).max(q[2]).max(q[3]);
    }
    if (max_v as usize) >= num_vertices {
        return false;
    }
    let mut slots = vec![[0u32; 4]; num_vertices];
    #[inline(always)]
    unsafe fn insert_edge(slots: &mut [[u32; 4]], a: u32, b: u32) -> u8 {
        if a == b {
            return 2;
        }
        let (lo, hi, bit) = if a < b { (a as usize, b, 1u32) } else { (b as usize, a, 2u32) };
        let row = unsafe { slots.get_unchecked_mut(lo) };
        let e0 = row[0];
        if e0 == 0 { row[0] = (hi << 2) | bit; return 0; }
        if (e0 >> 2) == hi { if (e0 & bit) != 0 { return 2; } row[0] = e0 | bit; return 0; }
        let e1 = row[1];
        if e1 == 0 { row[1] = (hi << 2) | bit; return 0; }
        if (e1 >> 2) == hi { if (e1 & bit) != 0 { return 2; } row[1] = e1 | bit; return 0; }
        let e2 = row[2];
        if e2 == 0 { row[2] = (hi << 2) | bit; return 0; }
        if (e2 >> 2) == hi { if (e2 & bit) != 0 { return 2; } row[2] = e2 | bit; return 0; }
        let e3 = row[3];
        if e3 == 0 { row[3] = (hi << 2) | bit; return 0; }
        if (e3 >> 2) == hi { if (e3 & bit) != 0 { return 2; } row[3] = e3 | bit; return 0; }
        1 // overflow -> fallback
    }
    for q in quads {
        // SAFETY: `max_v < num_vertices == slots.len()` verified above.
        let r = unsafe {
            insert_edge(&mut slots, q[0], q[1])
                | insert_edge(&mut slots, q[1], q[2])
                | insert_edge(&mut slots, q[2], q[3])
                | insert_edge(&mut slots, q[3], q[0])
        };
        if r != 0 {
            if (r & 2) != 0 {
                return false;
            }
            return quads_manifold(quads, num_vertices);
        }
    }
    for row in &slots {
        for &e in row {
            if e != 0 && (e & 3) != 3 {
                return false;
            }
        }
    }
    true
}

/// Boxed `Mesh` faces from flat quads: the quads themselves, or two triangles each split along the shorter
/// diagonal. Keeps allocation single-threaded below 16k quads to avoid cross-magazine free contention.
fn quads_to_faces(quads: &[[u32; 4]], positions: &[Vec3A], keep_quads: bool) -> Vec<Vec<u32>> {
    if keep_quads {
        #[cfg(feature = "threads")]
        if quads.len() >= 16_384 {
            use rayon::prelude::*;
            return quads.par_iter().map(|q| q.to_vec()).collect();
        }
        return quads.iter().map(|q| q.to_vec()).collect();
    }
    let mut max_v = 0u32;
    for q in quads {
        max_v = max_v.max(q[0]).max(q[1]).max(q[2]).max(q[3]);
    }
    assert!(quads.is_empty() || (max_v as usize) < positions.len(), "quad vertex index out of bounds");

    #[cfg(feature = "threads")]
    if quads.len() >= 16_384 {
        use rayon::prelude::*;
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); quads.len() * 2];
        out.par_chunks_mut(2).zip(quads.par_iter()).for_each(|(o, q)| {
            let (a, b, c, d) = (q[0], q[1], q[2], q[3]);
            // SAFETY: `max_v < positions.len()` verified above.
            let (pa, pb, pc, pd) = unsafe {
                (*positions.get_unchecked(a as usize), *positions.get_unchecked(b as usize), *positions.get_unchecked(c as usize), *positions.get_unchecked(d as usize))
            };
            if (pa - pc).length_squared() <= (pb - pd).length_squared() {
                o[0] = vec![a, b, c];
                o[1] = vec![a, c, d];
            } else {
                o[0] = vec![a, b, d];
                o[1] = vec![b, c, d];
            }
        });
        return out;
    }
    let mut out = Vec::with_capacity(quads.len() * 2);
    for q in quads {
        let (a, b, c, d) = (q[0], q[1], q[2], q[3]);
        // SAFETY: `max_v < positions.len()` verified above.
        let (pa, pb, pc, pd) = unsafe {
            (*positions.get_unchecked(a as usize), *positions.get_unchecked(b as usize), *positions.get_unchecked(c as usize), *positions.get_unchecked(d as usize))
        };
        if (pa - pc).length_squared() <= (pb - pd).length_squared() {
            out.push(vec![a, b, c]);
            out.push(vec![a, c, d]);
        } else {
            out.push(vec![a, b, d]);
            out.push(vec![b, c, d]);
        }
    }
    out
}

/// Compile-time crossing edge bitmask for every 8-bit corner sign mask (zero runtime OnceLock/LazyLock).
const CROSS_TABLE: [u16; 256] = {
    let mut out = [0u16; 256];
    let mut m = 0usize;
    while m < 256 {
        let mut bits = 0u16;
        let mut e = 0usize;
        while e < 12 {
            let (a, b) = EDGES[e];
            if (((m >> a) ^ (m >> b)) & 1) != 0 {
                bits |= 1u16 << e;
            }
            e += 1;
        }
        out[m] = bits;
        m += 1;
    }
    out
};

/// Compile-time per-mask `[sum_base_x, sum_base_y, sum_base_z, 1.0 / edge_count]`.
/// Because `EDGES[0..4]` are +X, `EDGES[4..8]` are +Y, and `EDGES[8..12]` are +Z, each crossing edge
/// adds its scalar `t` directly into register `sx`, `sy`, or `sz` (zero stack array spills).
const MASK_BASE: [[f32; 4]; 256] = {
    let mut out = [[0.0f32; 4]; 256];
    let mut m = 1usize;
    while m < 255 {
        let bits = CROSS_TABLE[m];
        let mut sx = 0.0f32;
        let mut sy = 0.0f32;
        let mut sz = 0.0f32;
        let mut cnt = 0u32;
        let mut e = 0usize;
        while e < 12 {
            if (bits & (1u16 << e)) != 0 {
                let a = EDGES[e].0;
                sx += (a & 1) as f32;
                sy += ((a >> 1) & 1) as f32;
                sz += ((a >> 2) & 1) as f32;
                cnt += 1;
            }
            e += 1;
        }
        out[m] = [sx, sy, sz, if cnt > 0 { 1.0 / cnt as f32 } else { 0.0 }];
        m += 1;
    }
    out
};

/// Bitmask for the lowest `count` bits (`count in 1..=128`), shift-overflow safe when `count == 128`.
#[inline(always)]
fn low_bits_u128(count: usize) -> u128 {
    debug_assert!(count >= 1 && count <= 128);
    u128::MAX >> (128 - count)
}

/// Exact crossing edge count directly from packed `u128` row sign bitmasks (<1 us for the whole grid).
fn count_crossings_bits(signs: &[u128], n: [usize; 3]) -> usize {
    let [nx, ny, nz] = n;
    if nx < 3 || ny < 3 || nz < 3 {
        return 0;
    }
    let x_mask = low_bits_u128(nx - 1);
    let inner_mask = low_bits_u128(nx - 2) << 1;
    let mut total = 0u32;
    for k in 0..nz {
        let slab = &signs[k * ny..(k + 1) * ny];
        if k >= 1 && k + 1 < nz {
            // +X edges (j in 1..ny-1, i in 0..nx-1)
            for j in 1..ny - 1 {
                let s = slab[j];
                total += ((s ^ (s >> 1)) & x_mask).count_ones();
            }
            // +Y edges (j in 0..ny-1, i in 1..nx-1)
            for j in 0..ny - 1 {
                total += ((slab[j] ^ slab[j + 1]) & inner_mask).count_ones();
            }
        }
        if k + 1 < nz {
            // +Z edges (j in 1..ny-1, i in 1..nx-1)
            let next_slab = &signs[(k + 1) * ny..(k + 2) * ny];
            for j in 1..ny - 1 {
                total += ((slab[j] ^ next_slab[j]) & inner_mask).count_ones();
            }
        }
    }
    total as usize
}

/// Bitmask-accelerated flat-quad Naive Surface Nets extractor:
/// - `u128` row sign bitmasks locate EVERY active crossing cell across a row in ~10 bitwise cycles,
///   completely skipping the ~90% empty/interior cells without loading any floats or branching.
/// - Compile-time `CROSS_TABLE` and `MASK_BASE` accumulate `(sx, sy, sz)` purely in registers.
/// - Only touched cells are cleared between Z layers; emits `Vec3A` positions directly for Taubin smoothing.
fn surface_nets_flat(field: &[f32], signs: &[u128], grid: &Grid, expected_quads: usize) -> (Vec<Vec3A>, Vec<[u32; 4]>) {
    let [nx, ny, nz] = grid.n;
    let h = grid.h;
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);
    let sy = nx;
    let sz = nx * ny;
    assert!(field.len() >= nx * ny * nz && signs.len() >= ny * nz);
    let cell_mask = low_bits_u128(cx);

    // Find tight Z range of interior samples from the row sign bitmasks (touches `ny*nz` u128s, not `nx*ny*nz` floats).
    let (mut klo, mut khi) = (usize::MAX, 0usize);
    for k in 0..nz {
        let slab = &signs[k * ny..(k + 1) * ny];
        if slab.iter().any(|&s| s != 0) {
            klo = klo.min(k);
            khi = k;
        }
    }
    if klo == usize::MAX {
        return (Vec::new(), Vec::new());
    }
    let (ck0, ck1) = (klo.saturating_sub(1), khi.min(cz - 1));

    let origin = Vec3A::from(grid.origin);
    let cap = expected_quads.max(64);
    let mut positions: Vec<Vec3A> = Vec::with_capacity(cap + 16);
    let mut quads: Vec<[u32; 4]> = Vec::with_capacity(cap + 16);
    let layer_len = cx * cy;
    let mut prev = vec![u32::MAX; layer_len];
    let mut cur = vec![u32::MAX; layer_len];
    let mut written_prev: Vec<usize> = Vec::with_capacity(256);
    let mut written_cur: Vec<usize> = Vec::with_capacity(256);

    for k in ck0..=ck1 {
        let kz = k * sz;
        let s_k0 = &signs[k * ny..(k + 1) * ny];
        let s_k1 = &signs[(k + 1) * ny..(k + 2) * ny];
        let kf = k as f32;
        for j in 0..cy {
            let s00 = s_k0[j];
            let s10 = s_k0[j + 1];
            let s01 = s_k1[j];
            let s11 = s_k1[j + 1];
            let any = (s00 | (s00 >> 1)) | (s10 | (s10 >> 1)) | (s01 | (s01 >> 1)) | (s11 | (s11 >> 1));
            if (any & cell_mask) == 0 {
                continue;
            }
            let all = (s00 & (s00 >> 1)) & (s10 & (s10 >> 1)) & (s01 & (s01 >> 1)) & (s11 & (s11 >> 1));
            let mut active = (any & !all) & cell_mask;
            if active == 0 {
                continue;
            }
            let b00 = kz + j * sy;
            let b10 = b00 + sy;
            let b01 = b00 + sz;
            let b11 = b10 + sz;
            let row = j * cx;
            let jf = j as f32;

            while active != 0 {
                let i = active.trailing_zeros() as usize;
                active &= active - 1;
                let mask = (((s00 >> i) & 3)
                    | (((s10 >> i) & 3) << 2)
                    | (((s01 >> i) & 3) << 4)
                    | (((s11 >> i) & 3) << 6)) as usize;
                // SAFETY: `i < cx = nx - 1`, `j < cy = ny - 1`, `k < cz = nz - 1`, `field.len() >= nx*ny*nz`.
                let v = unsafe {
                    [
                        *field.get_unchecked(b00 + i),
                        *field.get_unchecked(b00 + i + 1),
                        *field.get_unchecked(b10 + i),
                        *field.get_unchecked(b10 + i + 1),
                        *field.get_unchecked(b01 + i),
                        *field.get_unchecked(b01 + i + 1),
                        *field.get_unchecked(b11 + i),
                        *field.get_unchecked(b11 + i + 1),
                    ]
                };
                let mb = MASK_BASE[mask];
                let (mut sx, mut sy_acc, mut sz_acc) = (mb[0], mb[1], mb[2]);
                let bits = CROSS_TABLE[mask];
                // X-edges: (0,1), (2,3), (4,5), (6,7)
                let mut bx = bits & 0x00F;
                while bx != 0 {
                    let e = bx.trailing_zeros() as usize;
                    bx &= bx - 1;
                    let a = e << 1;
                    let va = v[a];
                    sx += (va / (va - v[a + 1])).clamp(0.0, 1.0);
                }
                // Y-edges: (0,2), (1,3), (4,6), (5,7)
                let mut by = (bits >> 4) & 0x00F;
                while by != 0 {
                    let e = by.trailing_zeros() as usize;
                    by &= by - 1;
                    let a = (e & 1) | ((e & 2) << 1);
                    let va = v[a];
                    sy_acc += (va / (va - v[a + 2])).clamp(0.0, 1.0);
                }
                // Z-edges: (0,4), (1,5), (2,6), (3,7)
                let mut bz = bits >> 8;
                while bz != 0 {
                    let a = bz.trailing_zeros() as usize;
                    bz &= bz - 1;
                    let va = v[a];
                    sz_acc += (va / (va - v[a + 4])).clamp(0.0, 1.0);
                }
                let local = Vec3A::new(sx, sy_acc, sz_acc) * mb[3];
                let id = positions.len() as u32;
                positions.push(origin + (Vec3A::new(i as f32, jf, kf) + local) * h);
                let me = row + i;
                // SAFETY: `j < cy`, `i < cx` -> `me < layer_len == cur.len() == prev.len()`.
                unsafe {
                    *cur.get_unchecked_mut(me) = id;
                }
                written_cur.push(me);
                let inside = mask & 1 != 0;
                let mut push = |q: [u32; 4]| {
                    if q[0] != u32::MAX && q[1] != u32::MAX && q[3] != u32::MAX {
                        quads.push(if inside { q } else { [q[0], q[3], q[2], q[1]] });
                    }
                };
                // +X edge (corner 0 -> 1), +Y edge (0 -> 2), +Z edge (0 -> 4); same cell order as the full scan.
                // SAFETY: guarded by `j >= 1`, `i >= 1`, `k >= 1`, and `me = j * cx + i < cx * cy`.
                unsafe {
                    if j >= 1 && k >= 1 && (mask ^ (mask >> 1)) & 1 != 0 {
                        push([*prev.get_unchecked(me - cx), *prev.get_unchecked(me), id, *cur.get_unchecked(me - cx)]);
                    }
                    if k >= 1 && i >= 1 && (mask ^ (mask >> 2)) & 1 != 0 {
                        push([*prev.get_unchecked(me - 1), *cur.get_unchecked(me - 1), id, *prev.get_unchecked(me)]);
                    }
                    if i >= 1 && j >= 1 && (mask ^ (mask >> 4)) & 1 != 0 {
                        push([*cur.get_unchecked(me - cx - 1), *cur.get_unchecked(me - cx), id, *cur.get_unchecked(me - 1)]);
                    }
                }
            }
        }
        // Reset only the cells written in `prev` before swapping `cur` -> `prev`.
        for &idx in &written_prev {
            unsafe {
                *prev.get_unchecked_mut(idx) = u32::MAX;
            }
        }
        written_prev.clear();
        std::mem::swap(&mut prev, &mut cur);
        std::mem::swap(&mut written_prev, &mut written_cur);
    }
    (positions, quads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text3d::mesher::is_manifold;

    fn make_red_circle_image(size: u32, transparent_bg: bool) -> Vec<u8> {
        let mut rgba = vec![0u8; (size * size * 4) as usize];
        let c = size as f32 * 0.5;
        let r = size as f32 * 0.34;
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - c;
                let dy = y as f32 - c;
                let i = ((y * size + x) * 4) as usize;
                if dx * dx + dy * dy <= r * r {
                    let shade = (1.0 - (dx * dx + dy * dy).sqrt() / r * 0.35).clamp(0.4, 1.0);
                    rgba[i] = (225.0 * shade) as u8;
                    rgba[i + 1] = (48.0 * shade) as u8;
                    rgba[i + 2] = (42.0 * shade) as u8;
                    rgba[i + 3] = 255;
                } else if transparent_bg {
                    rgba[i + 3] = 0;
                } else {
                    rgba[i] = 240;
                    rgba[i + 1] = 242;
                    rgba[i + 2] = 245;
                    rgba[i + 3] = 255;
                }
            }
        }
        rgba
    }

    #[test]
    fn item_mode_produces_watertight_manifold_mesh_with_dominant_color() {
        let rgba = make_red_circle_image(96, false);
        let opts = Image3DOptions { mode: Image3DMode::Item, quads: false, ..Default::default() };
        let part = build_image_part(96, 96, &rgba, None, &opts).expect("reconstruct item");
        assert!(is_manifold(&part.mesh), "image-to-3d mesh must be watertight manifold");
        assert!(triangle_count(&part.mesh) >= 300);
        assert!(triangle_count(&part.mesh) <= opts.max_triangles);
        // Dominant color should be strongly reddish.
        assert!(part.material.base_color.r > 0.65);
        assert!(part.material.base_color.r > part.material.base_color.b + 0.35);
    }

    #[test]
    fn lathe_and_relief_modes_are_watertight() {
        let rgba = make_red_circle_image(64, true);
        for mode in [Image3DMode::Lathe, Image3DMode::Relief] {
            let opts = Image3DOptions { mode, resolution: 40, ..Default::default() };
            let part = build_image_part(64, 64, &rgba, None, &opts).expect("reconstruct mode");
            assert!(is_manifold(&part.mesh), "mode {mode:?} must be watertight");
            assert!(triangle_count(&part.mesh) > 100);
        }
    }

    #[test]
    fn signed_distance_matches_brute_force() {
        let (w, h) = (23usize, 17usize);
        let mut mask = vec![false; w * h];
        for y in 4..13 {
            for x in 5..18 {
                mask[y * w + x] = (x as i32 - 11).pow(2) + (y as i32 - 8).pow(2) <= 20;
            }
        }
        let sdf = signed_distance_2d(&mask, w, h);
        for y in 0..h {
            for x in 0..w {
                let me = mask[y * w + x];
                let mut best = f32::INFINITY;
                for yy in 0..h {
                    for xx in 0..w {
                        if mask[yy * w + xx] != me {
                            let d = (((x as i32 - xx as i32).pow(2) + (y as i32 - yy as i32).pow(2)) as f32).sqrt();
                            best = best.min(d);
                        }
                    }
                }
                let want = (best - 0.5).max(0.0);
                let want = if me { -want } else { want };
                assert!((sdf[y * w + x] - want).abs() < 1e-3, "({x},{y}) {} vs {want}", sdf[y * w + x]);
            }
        }
    }

    #[test]
    fn morph_pass_matches_naive_3x3() {
        let (w, h) = (19usize, 13usize);
        let mask: Vec<bool> = (0..w * h).map(|i| (i * 2654435761usize >> 7) % 3 != 0).collect();
        for dilate in [true, false] {
            let got = morph_pass(&mask, w, h, dilate);
            for y in 0..h {
                for x in 0..w {
                    let mut want = mask[y * w + x];
                    if x >= 1 && y >= 1 && x + 1 < w && y + 1 < h {
                        let mut hits = 0;
                        for dy in 0..3 {
                            for dx in 0..3 {
                                hits += mask[(y + dy - 1) * w + x + dx - 1] as u32;
                            }
                        }
                        want = if dilate { hits >= 3 } else { hits >= 6 };
                    }
                    assert_eq!(got[y * w + x], want);
                }
            }
        }
    }

    #[test]
    fn image_128_hits_triangle_budget_with_flat_manifold_quads() {
        let rgba = make_red_circle_image(128, false);
        let opts = Image3DOptions { quads: false, ..Default::default() };
        let part = build_image_part(128, 128, &rgba, None, &opts).expect("reconstruct");
        let tris = triangle_count(&part.mesh);
        assert!(is_manifold(&part.mesh));
        assert!(tris <= opts.max_triangles && tris * 100 >= opts.max_triangles * 40, "tris {tris}");
    }

    #[test]
    fn image_to_3d_json_matches_text3d_schema() {
        let rgba = make_red_circle_image(64, true);
        let json_str = image_to_3d_json(64, 64, &rgba, None, r#"{"mode":"item","name":"Apple","maxTriangles":6000}"#);
        let v: Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["understood"], true);
        assert_eq!(v["parts"][0]["name"], "Apple");
        assert!(v["triangles"].as_u64().unwrap() > 100);
        assert!(v["triangles"].as_u64().unwrap() <= 6000);
    }

    #[test]
    fn quads_are_manifold_validates_closed_and_rejects_broken_meshes() {
        // Closed unit cube (6 outward-wound quads, 8 vertices).
        let cube = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ];
        assert!(quads_are_manifold(&cube, 8));
        // Missing a face -> open boundary -> non-manifold.
        assert!(!quads_are_manifold(&cube[..5], 8));
        // Reversed face -> duplicate directed edges -> non-manifold.
        let mut flipped = cube;
        flipped[0].reverse();
        assert!(!quads_are_manifold(&flipped, 8));
        // Degenerate edge (a == b).
        assert!(!quads_are_manifold(&[[0, 0, 1, 2]], 3));
    }
}
