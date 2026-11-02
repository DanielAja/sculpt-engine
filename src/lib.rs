//! sculpt-engine: the shared render and sculpt engine for Sculptor and Dream Builder.
//! See DESIGN.md for the architecture.

pub mod brush;
pub mod bvh;
pub mod camera;
pub mod format;
pub mod mesh;
#[cfg(feature = "render")]
pub mod render;
#[cfg(feature = "render")]
pub mod session;
#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub mod web;
pub mod stroke;

pub use glam;
