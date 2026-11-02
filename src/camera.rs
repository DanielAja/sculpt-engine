//! Sculptor's `OrbitCamera` (Sculptor/Viewport/OrbitCamera.swift, ported as Android's OrbitCamera.kt).
//! Same maths, so a host's camera, the engine's picture and the engine's picking agree.
//! Z up, millimetres. yaw 0 / pitch 0 looks from -Y toward +Y ("Front"); pitch 90 looks straight down.

use glam::{Mat4, Vec3, Vec4};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrbitCamera {
    pub target: [f32; 3],
    pub distance: f32,
    pub yaw_degrees: f32,
    pub pitch_degrees: f32,
    pub fov_degrees: f32,
    pub is_orthographic: bool,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        OrbitCamera { target: [0.0, 0.0, 10.0], distance: 120.0, yaw_degrees: 35.0, pitch_degrees: 25.0, fov_degrees: 45.0, is_orthographic: false }
    }
}

pub const MIN_DISTANCE: f32 = 5.0;
pub const MAX_DISTANCE: f32 = 5000.0;

impl OrbitCamera {
    pub fn target(&self) -> Vec3 {
        Vec3::from(self.target)
    }
    pub fn backward(&self) -> Vec3 {
        let (y, p) = (self.yaw_degrees.to_radians(), self.pitch_degrees.to_radians());
        Vec3::new(p.cos() * y.sin(), -p.cos() * y.cos(), p.sin())
    }
    pub fn right(&self) -> Vec3 {
        let y = self.yaw_degrees.to_radians();
        Vec3::new(y.cos(), y.sin(), 0.0)
    }
    pub fn up(&self) -> Vec3 {
        let (y, p) = (self.yaw_degrees.to_radians(), self.pitch_degrees.to_radians());
        Vec3::new(-p.sin() * y.sin(), p.sin() * y.cos(), p.cos())
    }
    pub fn forward(&self) -> Vec3 {
        -self.backward()
    }
    pub fn eye(&self) -> Vec3 {
        self.target() + self.backward() * self.distance
    }
    pub fn tan_half_fov(&self) -> f32 {
        (self.fov_degrees.clamp(1.0, 170.0) * std::f32::consts::PI / 360.0).tan()
    }
    pub fn half_height_at_target(&self) -> f32 {
        self.distance * self.tan_half_fov()
    }
    fn ortho_backoff(&self) -> f32 {
        self.distance * 20.0 + 10_000.0
    }
    pub fn near_clip(&self) -> f32 {
        if self.is_orthographic { 0.1 } else { (self.distance * 0.002).max(0.05) }
    }
    pub fn far_clip(&self) -> f32 {
        let f = self.distance * 50.0 + 20_000.0;
        if self.is_orthographic { self.ortho_backoff() + f } else { f }
    }
    pub fn render_eye(&self) -> Vec3 {
        if self.is_orthographic { self.eye() + self.backward() * self.ortho_backoff() } else { self.eye() }
    }

    pub fn view_matrix(&self) -> Mat4 {
        self.view(self.eye())
    }
    pub fn render_view_matrix(&self) -> Mat4 {
        self.view(self.render_eye())
    }
    fn view(&self, e: Vec3) -> Mat4 {
        let (r, u, b) = (self.right(), self.up(), self.backward());
        Mat4::from_cols(Vec4::new(r.x, u.x, b.x, 0.0), Vec4::new(r.y, u.y, b.y, 0.0), Vec4::new(r.z, u.z, b.z, 0.0), Vec4::new(-r.dot(e), -u.dot(e), -b.dot(e), 1.0))
    }

    /// Right handed, clip depth 0 to 1, as the iPad app builds it.
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        let a = aspect.max(0.0001);
        if self.is_orthographic {
            let hh = self.half_height_at_target();
            let (w, h) = (2.0 * hh * a, 2.0 * hh);
            let (near, far) = (-self.ortho_backoff(), self.far_clip());
            let z = 1.0 / (near - far);
            Mat4::from_cols(Vec4::new(2.0 / w, 0.0, 0.0, 0.0), Vec4::new(0.0, 2.0 / h, 0.0, 0.0), Vec4::new(0.0, 0.0, z, 0.0), Vec4::new(0.0, 0.0, z * near, 1.0))
        } else {
            let y = 1.0 / (self.fov_degrees.to_radians() * 0.5).tan();
            let x = y / a;
            let (n, f) = (self.near_clip(), self.far_clip());
            let z = f / (n - f);
            Mat4::from_cols(Vec4::new(x, 0.0, 0.0, 0.0), Vec4::new(0.0, y, 0.0, 0.0), Vec4::new(0.0, 0.0, z, -1.0), Vec4::new(0.0, 0.0, z * n, 0.0))
        }
    }

    /// The view used for rendering (orthographic pulls the eye back; picking uses `ray`).
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.render_view_matrix()
    }

    /// Ray through `px`, `py` in 0..1 (top left origin): (origin, unit direction).
    pub fn ray(&self, px: f32, py: f32, aspect: f32) -> (Vec3, Vec3) {
        let a = aspect.max(0.0001);
        let (nx, ny) = (px * 2.0 - 1.0, 1.0 - py * 2.0);
        if self.is_orthographic {
            let hh = self.half_height_at_target();
            let on_plane = self.target() + self.right() * (nx * hh * a) + self.up() * (ny * hh);
            return (on_plane + self.backward() * (self.distance + self.ortho_backoff()), self.forward());
        }
        let t = self.tan_half_fov();
        (self.eye(), (self.forward() + self.right() * (nx * t * a) + self.up() * (ny * t)).normalize())
    }

    pub fn orbit(&self, d_yaw: f32, d_pitch: f32) -> OrbitCamera {
        OrbitCamera { yaw_degrees: (self.yaw_degrees + d_yaw).rem_euclid(360.0), pitch_degrees: (self.pitch_degrees + d_pitch).clamp(-90.0, 90.0), ..*self }
    }
    pub fn pan(&self, dx: f32, dy: f32, viewport_height: f32) -> OrbitCamera {
        if viewport_height <= 0.0 {
            return *self;
        }
        let k = 2.0 * self.half_height_at_target() / viewport_height;
        OrbitCamera { target: (self.target() + (-self.right() * dx + self.up() * dy) * k).to_array(), ..*self }
    }
    /// factor > 1 zooms in.
    pub fn zoom(&self, factor: f32) -> OrbitCamera {
        if !(factor > 0.0) || !factor.is_finite() {
            return *self;
        }
        OrbitCamera { distance: (self.distance / factor).clamp(MIN_DISTANCE, MAX_DISTANCE), ..*self }
    }
    /// Frames a bounding box from the current direction.
    pub fn fitted(&self, min: Vec3, max: Vec3, aspect: f32) -> OrbitCamera {
        let radius = ((max - min).length() * 0.5).max(1.0);
        let a = aspect.max(0.0001);
        let t = self.tan_half_fov();
        let d = if self.is_orthographic { radius * 1.15 / (t * a.min(1.0)) } else { radius * 1.15 / t.atan().min((t * a).atan()).sin() };
        OrbitCamera { target: ((min + max) * 0.5).to_array(), distance: d.clamp(MIN_DISTANCE, MAX_DISTANCE), ..*self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_shape_matches_sculptor() {
        let c: OrbitCamera = serde_json::from_str(r#"{"distance":140,"fovDegrees":45,"isOrthographic":false,"pitchDegrees":25,"target":[0,0,10],"yawDegrees":35}"#).unwrap();
        assert_eq!(c.distance, 140.0);
        assert_eq!(c.target, [0.0, 0.0, 10.0]);
    }

    #[test]
    fn centre_ray_hits_the_target_and_projects_to_the_centre() {
        let c = OrbitCamera::default();
        let (o, d) = c.ray(0.5, 0.5, 1.5);
        assert!((o + d * c.distance - c.target()).length() < 1e-3);
        let clip = c.view_projection(1.5) * c.target().extend(1.0);
        assert!((clip.x / clip.w).abs() < 1e-5 && (clip.y / clip.w).abs() < 1e-5);
        // A point a quarter of the way across the screen projects back to x = 0.25.
        let (o, d) = c.ray(0.25, 0.5, 1.5);
        let p = c.view_projection(1.5) * (o + d * 50.0).extend(1.0);
        assert!(((p.x / p.w + 1.0) * 0.5 - 0.25).abs() < 1e-4);
        let front = OrbitCamera { yaw_degrees: 0.0, pitch_degrees: 0.0, ..c };
        assert!((front.forward() - Vec3::Y).length() < 1e-6);
    }
}
