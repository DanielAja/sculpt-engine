//! Image based lighting, computed on the CPU so the downlevel (WebGL2) tier needs no compute shaders:
//! a procedural studio environment, its irradiance as 9 spherical harmonics, a GGX prefiltered cube map
//! and the split sum BRDF lookup table. Z up.

use glam::{Vec2, Vec3};
use std::f32::consts::PI;

/// Soft studio: a cool bright ceiling, neutral horizon, darker warm floor, a large key softbox (front
/// right, high, matching the shadow casting key light) and a rim softbox behind. Linear radiance.
pub fn studio(d: Vec3) -> Vec3 {
    let up = d.z;
    let sky = Vec3::new(0.80, 0.84, 0.92);
    let horizon = Vec3::new(0.62, 0.62, 0.62);
    let floor = Vec3::new(0.22, 0.20, 0.18);
    let mut c = if up >= 0.0 { horizon.lerp(sky, up.powf(0.6)) } else { horizon.lerp(floor, (-up).powf(0.4)) };
    let soft = |center: Vec3, size: f32, power: f32| -> f32 {
        let cos = d.dot(center.normalize());
        let t = ((cos - (1.0 - size)) / size).clamp(0.0, 1.0);
        power * t * t * (3.0 - 2.0 * t)
    };
    c += Vec3::splat(soft(KEY_DIR, 0.12, 6.0));
    c += Vec3::new(0.9, 0.95, 1.0) * soft(Vec3::new(-0.6, 0.9, 0.5), 0.06, 3.0);
    c
}

/// Direction toward the key light (shared by the shadow casting sun and the environment's softbox).
pub const KEY_DIR: Vec3 = Vec3::new(0.35, -0.45, 1.0);

/// Irradiance / pi as 9 SH coefficients (bands 0 to 2), cosine lobe already convolved.
pub fn sh9(env: impl Fn(Vec3) -> Vec3) -> [Vec3; 9] {
    let (nt, np) = (64, 128);
    let mut c = [Vec3::ZERO; 9];
    for i in 0..nt {
        let theta = (i as f32 + 0.5) / nt as f32 * PI;
        let w = theta.sin() * (PI / nt as f32) * (2.0 * PI / np as f32);
        for j in 0..np {
            let phi = (j as f32 + 0.5) / np as f32 * 2.0 * PI;
            let d = Vec3::new(theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos());
            let l = env(d) * w;
            for (k, b) in sh_basis(d).iter().enumerate() {
                c[k] += l * *b;
            }
        }
    }
    // Cosine lobe convolution (A0 = pi, A1 = 2pi/3, A2 = pi/4), then / pi for Lambert.
    let a = [1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25];
    for k in 0..9 {
        c[k] *= a[k];
    }
    c
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

fn hammersley(i: u32, n: u32) -> Vec2 {
    Vec2::new(i as f32 / n as f32, i.reverse_bits() as f32 * 2.328_306_4e-10)
}

fn ggx_sample(xi: Vec2, n: Vec3, a: f32) -> Vec3 {
    let phi = 2.0 * PI * xi.x;
    let cos_t = ((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y)).sqrt();
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let h = Vec3::new(phi.cos() * sin_t, phi.sin() * sin_t, cos_t);
    let up = if n.z.abs() < 0.999 { Vec3::Z } else { Vec3::X };
    let t = up.cross(n).normalize();
    let b = n.cross(t);
    (t * h.x + b * h.y + n * h.z).normalize()
}

pub const CUBE_SIZE: u32 = 64;
pub const CUBE_MIPS: u32 = 6;

/// GGX prefiltered cube: `[mip][face]` RGBA f32 texels, roughness = mip / (mips - 1).
pub fn prefilter(env: &(dyn Fn(Vec3) -> Vec3 + Sync)) -> Vec<Vec<Vec<[f32; 4]>>> {
    (0..CUBE_MIPS)
        .map(|mip| {
            let size = (CUBE_SIZE >> mip).max(1);
            let rough = mip as f32 / (CUBE_MIPS - 1) as f32;
            let a = (rough * rough).max(1e-4);
            let samples = if mip == 0 { 1 } else { 96 };
            (0..6)
                .map(|face| {
                    let mut out = Vec::with_capacity((size * size) as usize);
                    for y in 0..size {
                        for x in 0..size {
                            let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                            let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                            let n = cube_dir(face, u, v);
                            let c = if samples == 1 {
                                env(n)
                            } else {
                                let (mut sum, mut w) = (Vec3::ZERO, 0.0);
                                for i in 0..samples {
                                    let h = ggx_sample(hammersley(i, samples), n, a);
                                    let l = 2.0 * n.dot(h) * h - n;
                                    let nl = n.dot(l);
                                    if nl > 0.0 {
                                        sum += env(l) * nl;
                                        w += nl;
                                    }
                                }
                                sum / w.max(1e-6)
                            };
                            out.push([c.x, c.y, c.z, 1.0]);
                        }
                    }
                    out
                })
                .collect()
        })
        .collect()
}

pub const LUT_SIZE: u32 = 32;

/// Split sum BRDF table (scale, bias) for NdotV (x) and roughness (y), Smith GGX for IBL (k = a^2 / 2).
pub fn brdf_lut() -> Vec<[f32; 2]> {
    let n = 128;
    let mut out = Vec::with_capacity((LUT_SIZE * LUT_SIZE) as usize);
    for y in 0..LUT_SIZE {
        let rough = (y as f32 + 0.5) / LUT_SIZE as f32;
        let a = rough * rough;
        for x in 0..LUT_SIZE {
            let nv = ((x as f32 + 0.5) / LUT_SIZE as f32).max(1e-3);
            let v = Vec3::new((1.0 - nv * nv).sqrt(), 0.0, nv);
            let (mut sa, mut sb) = (0.0, 0.0);
            for i in 0..n {
                let h = ggx_sample(hammersley(i, n), Vec3::Z, a);
                let l = 2.0 * v.dot(h) * h - v;
                let (nl, nh, vh) = (l.z.max(0.0), h.z.max(0.0), v.dot(h).max(0.0));
                if nl > 0.0 {
                    let k = a / 2.0;
                    let g = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));
                    let gv = g * vh / (nh * nv).max(1e-6);
                    let fc = (1.0 - vh).powi(5);
                    sa += (1.0 - fc) * gv;
                    sb += fc * gv;
                }
            }
            out.push([sa / n as f32, sb / n as f32]);
        }
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
