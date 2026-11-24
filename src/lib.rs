//! sculpt-engine: the shared render and sculpt engine for Sculptor and Dream Builder.
//! See DESIGN.md for the architecture.

pub mod brush;
pub mod bvh;
pub mod camera;
pub mod draw3d;
pub mod format;
pub mod mesh;
pub mod remesh;
pub mod text3d;
#[cfg(feature = "render")]
pub mod render;
#[cfg(feature = "render")]
pub mod session;
#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub mod web;
pub mod stroke;
#[cfg(feature = "ffi")]
pub mod ffi;
#[cfg(feature = "ffi")]
uniffi::setup_scaffolding!();

pub use glam;
