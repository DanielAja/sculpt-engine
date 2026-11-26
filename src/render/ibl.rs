//! Image based lighting, computed on the CPU so the downlevel (WebGL2) tier needs no compute shaders:
//! a procedural studio environment, its irradiance as 9 spherical harmonics, a GGX prefiltered cube map
//! and the split sum BRDF lookup table. Z up.

use glam::{Vec3, Vec3A};
use std::f32::consts::PI;

/// Direction toward the key light (shared by the shadow casting sun and the environment's softbox).
pub const KEY_DIR: Vec3 = Vec3::new(0.35, -0.45, 1.0);
// Pre-normalized unit vectors for `KEY_DIR` and the rim softbox (`(-0.6, 0.9, 0.5)`) so `studio()`
// and `Renderer::frame_uniforms` never call `sqrt` per environment sample or per frame.
pub(crate) const KEY_DIR_NORM: Vec3 = Vec3::new(0.304_347_82, -0.391_304_34, 0.869_565_2);
const RIM_DIR_NORM: Vec3 = Vec3::new(-0.503_508_8, 0.755_263_2, 0.419_590_68);

/// Fast accurate `x^0.6` on `[0, 1]` via `t = x^0.25` and a degree-5 minimax polynomial (max err 1.05e-4).
#[inline(always)]
fn fast_pow_06(x: f32) -> f32 {
    let t = x.sqrt().sqrt();
    t * (-0.010_480_6 + t * (0.410_499 + t * (0.937_502_8 + t * (-0.468_361_1 + t * 0.130_900_9))))
}

/// Fast accurate `x^0.4` on `[0, 1]` via `t = x^0.25` and a degree-5 minimax polynomial (max err 4.9e-4).
#[inline(always)]
fn fast_pow_04(x: f32) -> f32 {
    let t = x.sqrt().sqrt();
    t * (0.099_666_1 + t * (1.672_982_2 + t * (-1.637_460_1 + t * (1.272_507 - t * 0.407_946_7))))
}

/// Soft studio: a cool bright ceiling, neutral horizon, darker warm floor, a large key softbox (front
/// right, high, matching the shadow casting key light) and a rim softbox behind. Linear radiance.
#[inline]
pub fn studio(d: Vec3) -> Vec3 {
    let up = d.z;
    let sky = Vec3::new(0.80, 0.84, 0.92);
    let horizon = Vec3::new(0.62, 0.62, 0.62);
    let floor = Vec3::new(0.22, 0.20, 0.18);
    let mut c = if up >= 0.0 {
        horizon.lerp(sky, fast_pow_06(up.min(1.0)))
    } else {
        horizon.lerp(floor, fast_pow_04((-up).min(1.0)))
    };
    let soft = |unit_center: Vec3, inv_size: f32, threshold: f32, power: f32| -> f32 {
        let cos = d.dot(unit_center);
        let t = ((cos - threshold) * inv_size).clamp(0.0, 1.0);
        power * t * t * (3.0 - 2.0 * t)
    };
    c += Vec3::splat(soft(KEY_DIR_NORM, 1.0 / 0.12, 0.88, 6.0));
    c += Vec3::new(0.9, 0.95, 1.0) * soft(RIM_DIR_NORM, 1.0 / 0.06, 0.94, 3.0);
    c
}

/// Irradiance / pi as 9 SH coefficients (bands 0 to 2), cosine lobe already convolved.
pub fn sh9(env: impl Fn(Vec3) -> Vec3) -> [Vec3; 9] {
    const NT: usize = 64;
    const NP: usize = 128;
    const DTHETA_DPHI: f32 = (PI / NT as f32) * (2.0 * PI / NP as f32);
    let mut cos_phi = [0.0f32; NP];
    let mut sin_phi = [0.0f32; NP];
    for j in 0..NP {
        let phi = (j as f32 + 0.5) * (2.0 * PI / NP as f32);
        let (sp, cp) = phi.sin_cos();
        cos_phi[j] = cp;
        sin_phi[j] = sp;
    }
    let mut c = [Vec3A::ZERO; 9];
    for i in 0..NT {
        let theta = (i as f32 + 0.5) * (PI / NT as f32);
        let (sin_t, cos_t) = theta.sin_cos();
        let w = sin_t * DTHETA_DPHI;
        let b2 = 0.488603 * cos_t;
        let b6 = 0.315392 * (3.0 * cos_t * cos_t - 1.0);
        for j in 0..NP {
            let x = sin_t * cos_phi[j];
            let y = sin_t * sin_phi[j];
            let z = cos_t;
            let l = Vec3A::from(env(Vec3::new(x, y, z)) * w);
            c[0] += l * 0.282095;
            c[1] += l * (0.488603 * y);
            c[2] += l * b2;
            c[3] += l * (0.488603 * x);
            c[4] += l * (1.092548 * x * y);
            c[5] += l * (1.092548 * y * z);
            c[6] += l * b6;
            c[7] += l * (1.092548 * x * z);
            c[8] += l * (0.546274 * (x * x - y * y));
        }
    }
    // Cosine lobe convolution (A0 = pi, A1 = 2pi/3, A2 = pi/4), then / pi for Lambert.
    let a = [1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25];
    std::array::from_fn(|k| Vec3::from(c[k] * a[k]))
}

pub fn sh_basis(d: Vec3) -> [f32; 9] {
    let (x, y, z) = (d.x, d.y, d.z);
    [
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
}

pub fn eval_sh(c: &[Vec3; 9], d: Vec3) -> Vec3 {
    sh_basis(d).iter().zip(c).map(|(b, c)| *c * *b).sum()
}

/// wgpu cube face directions: +X, -X, +Y, -Y, +Z, -Z; `u`, `v` in -1..1, v down.
#[inline]
pub fn cube_dir(face: usize, u: f32, v: f32) -> Vec3 {
    match face {
        0 => Vec3::new(1.0, -v, -u),
        1 => Vec3::new(-1.0, -v, u),
        2 => Vec3::new(u, 1.0, v),
        3 => Vec3::new(u, -1.0, -v),
        4 => Vec3::new(u, -v, 1.0),
        _ => Vec3::new(-u, -v, -1.0),
    }
    .normalize()
}

#[inline]
fn van_der_corput(i: u32) -> f32 {
    i.reverse_bits() as f32 * 2.328_306_4e-10
}

pub const CUBE_SIZE: u32 = 64;
pub const CUBE_MIPS: u32 = 6;
const PREFILTER_SAMPLES: u32 = 96;

/// Precomputes tangent-space reflected light directions `[lx, ly, lz, normalized_weight]` for a
/// roughness level `a`. Because `(t, b, n)` is orthonormal, `n.dot(h) = hz`, `l = t*(2*hz*hx) +
/// b*(2*hz*hy) + n*(2*hz^2 - 1)`, and `nl = 2*hz^2 - 1` depend only on `(i, a)` — identical for
/// every pixel on all 6 faces of the mip.
fn precompute_mip_samples(a: f32, samples: u32) -> Vec<[f32; 4]> {
    let a2_m1 = a * a - 1.0;
    let inv_n = 2.0 * PI / samples as f32;
    let mut out = Vec::with_capacity(samples as usize);
    let mut total_w = 0.0f32;
    for i in 0..samples {
        let phi = i as f32 * inv_n;
        let xi_y = van_der_corput(i);
        let cos_t = ((1.0 - xi_y) / (1.0 + a2_m1 * xi_y)).sqrt();
        let nl = 2.0 * cos_t * cos_t - 1.0;
        if nl > 0.0 {
            let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
            let (sp, cp) = phi.sin_cos();
            let k = 2.0 * cos_t * sin_t;
            out.push([cp * k, sp * k, nl, nl]);
            total_w += nl;
        }
    }
    let inv_w = 1.0 / total_w.max(1e-6);
    for s in &mut out {
        s[3] *= inv_w;
    }
    out
}

fn prefilter_rows(
    env: &(dyn Fn(Vec3) -> Vec3 + Sync),
    face: usize,
    size: u32,
    y0: u32,
    y1: u32,
    samples: &[[f32; 4]],
    out: &mut [[f32; 4]],
) {
    let inv_size = 2.0 / size as f32;
    let s = size as usize;
    if samples.is_empty() {
        for y in y0..y1 {
            let v = (y as f32 + 0.5) * inv_size - 1.0;
            let row = &mut out[((y - y0) as usize) * s..((y - y0) as usize + 1) * s];
            for (x, px) in row.iter_mut().enumerate() {
                let u = (x as f32 + 0.5) * inv_size - 1.0;
                let c = env(cube_dir(face, u, v));
                *px = [c.x, c.y, c.z, 1.0];
            }
        }
        return;
    }
    for y in y0..y1 {
        let v = (y as f32 + 0.5) * inv_size - 1.0;
        let row = &mut out[((y - y0) as usize) * s..((y - y0) as usize + 1) * s];
        for (x, px) in row.iter_mut().enumerate() {
            let u = (x as f32 + 0.5) * inv_size - 1.0;
            let n = cube_dir(face, u, v);
            let up = if n.z.abs() < 0.999 { Vec3::Z } else { Vec3::X };
            let t = up.cross(n).normalize();
            let b = n.cross(t);
            let mut sum = Vec3A::ZERO;
            for &[lx, ly, lz, w] in samples {
                let l = t * lx + b * ly + n * lz;
                sum += Vec3A::from(env(l)) * w;
            }
            *px = [sum.x, sum.y, sum.z, 1.0];
        }
    }
}

/// GGX prefiltered cube: `[mip][face]` RGBA f32 texels, roughness = mip / (mips - 1).
pub fn prefilter(env: &(dyn Fn(Vec3) -> Vec3 + Sync)) -> Vec<Vec<Vec<[f32; 4]>>> {
    let mip_samples: Vec<Vec<[f32; 4]>> = (0..CUBE_MIPS)
        .map(|mip| {
            if mip == 0 {
                Vec::new()
            } else {
                let rough = mip as f32 / (CUBE_MIPS - 1) as f32;
                let a = (rough * rough).max(1e-4);
                precompute_mip_samples(a, PREFILTER_SAMPLES)
            }
        })
        .collect();

    #[cfg(feature = "threads")]
    {
        use rayon::prelude::*;
        let mut cube: Vec<Vec<Vec<[f32; 4]>>> = (0..CUBE_MIPS)
            .map(|mip| {
                let size = (CUBE_SIZE >> mip).max(1) as usize;
                (0..6).map(|_| vec![[0.0f32; 4]; size * size]).collect()
            })
            .collect();
        // Split every face into row bands (8 rows on mip 1/2) so all cores stay evenly saturated.
        let mut slices: Vec<(u32, usize, u32, u32, &mut [[f32; 4]])> = Vec::with_capacity(64);
        for (mip_idx, faces) in cube.iter_mut().enumerate() {
            let mip = mip_idx as u32;
            let size = (CUBE_SIZE >> mip).max(1);
            let band = if mip == 1 { 8 } else if mip == 2 { 8 } else { size };
            for (face, buf) in faces.iter_mut().enumerate() {
                let mut y = 0u32;
                let mut rest = buf.as_mut_slice();
                while y < size {
                    let y1 = (y + band).min(size);
                    let count = ((y1 - y) * size) as usize;
                    let (head, tail) = rest.split_at_mut(count);
                    slices.push((mip, face, y, y1, head));
                    rest = tail;
                    y = y1;
                }
            }
        }
        slices.into_par_iter().for_each(|(mip, face, y0, y1, dst)| {
            let size = (CUBE_SIZE >> mip).max(1);
            prefilter_rows(env, face, size, y0, y1, &mip_samples[mip as usize], dst);
        });
        cube
    }
    #[cfg(not(feature = "threads"))]
    {
        (0..CUBE_MIPS)
            .map(|mip| {
                let size = (CUBE_SIZE >> mip).max(1);
                (0..6)
                    .map(|face| {
                        let mut out = vec![[0.0f32; 4]; (size * size) as usize];
                        prefilter_rows(env, face, size, 0, size, &mip_samples[mip as usize], &mut out);
                        out
                    })
                    .collect()
            })
            .collect()
    }
}

pub const LUT_SIZE: u32 = 32;
const LUT_SAMPLES: usize = 128;

fn brdf_lut_row_into(
    y: u32,
    row: &mut [[f32; 2]],
    sin_phi: &[f32; LUT_SAMPLES],
    xi_y: &[f32; LUT_SAMPLES],
    nv_vx: &[(f32, f32); LUT_SIZE as usize],
) {
    const INV_N: f32 = 1.0 / LUT_SAMPLES as f32;
    let rough = (y as f32 + 0.5) / LUT_SIZE as f32;
    let a = rough * rough;
    let a2_m1 = a * a - 1.0;
    let k = a * 0.5;
    let one_minus_k = 1.0 - k;

    // Half-vectors depend only on `(y, i)` (n = +Z -> h.x = sin(phi)*sin_t, h.z = cos_t; h.y is unused
    // because v.y == 0). Precompute once per row instead of `LUT_SIZE` times.
    let mut hx = [0.0f32; LUT_SAMPLES];
    let mut hz = [0.0f32; LUT_SAMPLES];
    for i in 0..LUT_SAMPLES {
        let xy = xi_y[i];
        let cos_t = ((1.0 - xy) / (1.0 + a2_m1 * xy)).sqrt();
        let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
        hx[i] = sin_phi[i] * sin_t;
        hz[i] = cos_t;
    }

    for (x, out) in row.iter_mut().enumerate() {
        let (nv, vx) = nv_vx[x];
        let g_v = nv / (nv * one_minus_k + k);
        let (mut sa, mut sb) = (0.0f32, 0.0f32);
        for i in 0..LUT_SAMPLES {
            let hxi = hx[i];
            let hzi = hz[i];
            let vh = vx * hxi + nv * hzi;
            let nl = 2.0 * vh * hzi - nv;
            if nl > 0.0 {
                let nh = hzi.max(0.0);
                let vh = vh.max(0.0);
                let g = g_v * (nl / (nl * one_minus_k + k));
                let gv = g * vh / (nh * nv).max(1e-6);
                let om = 1.0 - vh;
                let om2 = om * om;
                let fc = om2 * om2 * om;
                let b = fc * gv;
                sa += gv - b;
                sb += b;
            }
        }
        *out = [sa * INV_N, sb * INV_N];
    }
}

/// Split sum BRDF table (scale, bias) for NdotV (x) and roughness (y), Smith GGX for IBL (k = a^2 / 2).
pub fn brdf_lut() -> Vec<[f32; 2]> {
    let mut sin_phi = [0.0f32; LUT_SAMPLES];
    let mut xi_y = [0.0f32; LUT_SAMPLES];
    for i in 0..LUT_SAMPLES {
        sin_phi[i] = (2.0 * PI * i as f32 / LUT_SAMPLES as f32).sin();
        xi_y[i] = van_der_corput(i as u32);
    }
    let nv_vx: [(f32, f32); LUT_SIZE as usize] = std::array::from_fn(|x| {
        let nv = ((x as f32 + 0.5) / LUT_SIZE as f32).max(1e-3);
        (nv, (1.0 - nv * nv).sqrt())
    });
    let mut out = vec![[0.0f32; 2]; (LUT_SIZE * LUT_SIZE) as usize];
    for (y, row) in out.chunks_exact_mut(LUT_SIZE as usize).enumerate() {
        brdf_lut_row_into(y as u32, row, &sin_phi, &xi_y, &nv_vx);
    }
    out
}

/// IEEE half from f32 (round to nearest even, saturating to inf, flushing tiny values to zero).
pub fn f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return sign | 0x7e00; // NaN
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - exp);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    let h = ((exp as u32) << 10) | (man >> 13);
    let round = man & 0x1fff;
    let h = if round > 0x1000 || (round == 0x1000 && h & 1 == 1) { h + 1 } else { h };
    sign | h as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sh_reproduces_constant_and_linear_environments() {
        // Constant radiance L: irradiance / pi = L everywhere.
        let c = sh9(|_| Vec3::splat(2.0));
        assert!((eval_sh(&c, Vec3::Z) - Vec3::splat(2.0)).length() < 1e-2);
        // Radiance max(z, 0): E/pi straight up is 2/3, straight down 0 (approximately, band limited).
        let c = sh9(|d| Vec3::splat(d.z.max(0.0)));
        assert!((eval_sh(&c, Vec3::Z).x - 2.0 / 3.0).abs() < 0.05);
        assert!(eval_sh(&c, -Vec3::Z).x.abs() < 0.08);
    }

    #[test]
    fn lut_is_bounded_and_f16_round_trips() {
        let l = brdf_lut();
        assert!(l.iter().all(|v| v[0] >= 0.0 && v[0] + v[1] <= 1.05));
        // Smooth surface seen head on reflects almost everything.
        let top = l[(LUT_SIZE - 1) as usize];
        assert!(top[0] + top[1] > 0.9, "{top:?}");
        assert_eq!(f16(1.0), 0x3c00);
        assert_eq!(f16(-2.0), 0xc000);
        assert_eq!(f16(65504.0), 0x7bff);
        assert_eq!(f16(1e9), 0x7c00);
        assert_eq!(f16(0.0), 0);
    }

    #[test]
    fn cube_faces_point_the_right_way() {
        assert!((cube_dir(0, 0.0, 0.0) - Vec3::X).length() < 1e-6);
        assert!((cube_dir(4, 0.0, 0.0) - Vec3::Z).length() < 1e-6);
        // +Z face: u right is +X, v down is -Y.
        assert!(cube_dir(4, 1.0, 0.0).x > 0.0 && cube_dir(4, 0.0, 1.0).y < 0.0);
    }
}
