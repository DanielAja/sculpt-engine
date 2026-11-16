//! Shape programs: the open, LLM-friendly description of a model that every text-to-3D path produces.
//!
//! A program is a list of coloured parts. Each part is a tree of signed-distance primitives combined with
//! (optionally smooth) CSG operators, in Sculptor's space: millimetres, right handed, Z up, the ground at z = 0.
//! Programs are plain JSON, so a rule-based planner, a small on-device LLM or a person can write one, and
//! the engine turns it into a sculptable mesh. The idea of describing shapes as SDF/CSG programs follows
//! Inigo Quilez's distance-function articles, CSGNet and ShapeAssembly; see THIRD_PARTY_NOTICES.md.

use serde::{Deserialize, Serialize};

pub const PROGRAM_VERSION: u32 = 1;

/// Limits applied to every program, whoever wrote it (an LLM reply is untrusted input).
pub const MAX_PARTS: usize = 32;
pub const MAX_NODES: usize = 400;
pub const MAX_DEPTH: usize = 16;
pub const MAX_POINTS: usize = 64;
pub const MAX_ARRAY: u32 = 32;
pub const MAX_SIZE_MM: f32 = 5000.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Program {
    #[serde(default = "version_one")]
    pub version: u32,
    #[serde(default)]
    pub name: String,
    pub parts: Vec<Part>,
    /// Multiply by this for life size. Programs are modelled at tabletop or print scale (a toy car is
    /// 100 mm long); Dream Builder uses this to summon a real-sized car.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub real_scale: Option<f32>,
}

fn version_one() -> u32 {
    PROGRAM_VERSION
}

/// One coloured piece. Becomes one Sculptor object (or one RealityKit material group in Dream Builder).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(default)]
    pub name: String,
    /// sRGB hex, "#rrggbb". Missing or invalid means Sculptor's clay colour.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(default)]
    pub metallic: f32,
    #[serde(default = "half")]
    pub roughness: f32,
    pub shape: Node,
}

fn half() -> f32 {
    0.5
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

/// A shape plus its placement. Placement order: stretch, uniform scale, rotate (Euler degrees, X then Y then Z,
/// intrinsic, like Sculptor's `Transform.eulerDegrees`), then translate by `at`. `mirror` also places a copy
/// reflected across the parent's plane through the origin (Sculptor's Mirror modifier is its symmetry).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Node {
    #[serde(flatten)]
    pub shape: Shape,
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub at: [f32; 3],
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub rot: [f32; 3],
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub scale: f32,
    #[serde(default = "one3", skip_serializing_if = "is_one3")]
    pub stretch: [f32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mirror: Option<Axis>,
    /// Wall thickness in mm: hollows the shape and keeps its outer surface.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub hollow: f32,
}

fn one() -> f32 {
    1.0
}
fn one3() -> [f32; 3] {
    [1.0; 3]
}
fn is_zero(v: &f32) -> bool {
    *v == 0.0
}
fn is_one(v: &f32) -> bool {
    *v == 1.0
}
fn is_zero3(v: &[f32; 3]) -> bool {
    *v == [0.0; 3]
}
fn is_one3(v: &[f32; 3]) -> bool {
    *v == [1.0; 3]
}

/// Primitives are centred on their local origin; round things have their axis along Z.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Shape {
    Sphere { r: f32 },
    /// Full extents; `round` is an edge radius that keeps the outer size.
    #[serde(rename = "box")]
    Cuboid {
        size: [f32; 3],
        #[serde(default, skip_serializing_if = "is_zero")]
        round: f32,
    },
    Cylinder {
        r: f32,
        h: f32,
        #[serde(default, skip_serializing_if = "is_zero")]
        round: f32,
    },
    /// Capped cone: radius `r1` at the bottom, `r2` at the top.
    Cone {
        r1: f32,
        #[serde(default)]
        r2: f32,
        h: f32,
    },
    /// Ring in the XY plane: `major` is the ring radius, `minor` the tube radius.
    Torus { major: f32, minor: f32 },
    Capsule { from: [f32; 3], to: [f32; 3], r: f32 },
    /// Full extents.
    Ellipsoid { size: [f32; 3] },
    /// Square base of side `base`, apex `h` above it.
    Pyramid { base: f32, h: f32 },
    /// Profile of `[radius, z]` points from bottom to top, revolved around Z (vases, bottles, chess pieces).
    Lathe { profile: Vec<[f32; 2]> },
    /// Closed outline of `[x, y]` points extruded `h` along Z (stars, hearts, signs).
    Extrude { outline: Vec<[f32; 2]>, h: f32 },
    /// `smooth` is a blend radius in mm (0 = sharp).
    Union {
        children: Vec<Node>,
        #[serde(default, skip_serializing_if = "is_zero")]
        smooth: f32,
    },
    /// The first child minus all the others.
    Difference {
        children: Vec<Node>,
        #[serde(default, skip_serializing_if = "is_zero")]
        smooth: f32,
    },
    Intersection {
        children: Vec<Node>,
        #[serde(default, skip_serializing_if = "is_zero")]
        smooth: f32,
    },
    /// `count` copies of `child`, each moved by `offset` from the previous one.
    Array { child: Box<Node>, count: u32, offset: [f32; 3] },
}

impl Node {
    pub fn new(shape: Shape) -> Node {
        Node { shape, at: [0.0; 3], rot: [0.0; 3], scale: 1.0, stretch: [1.0; 3], mirror: None, hollow: 0.0 }
    }
    pub fn at(mut self, x: f32, y: f32, z: f32) -> Node {
        self.at = [self.at[0] + x, self.at[1] + y, self.at[2] + z];
        self
    }
    pub fn rot(mut self, x: f32, y: f32, z: f32) -> Node {
        self.rot = [x, y, z];
        self
    }
    pub fn mirror(mut self, axis: Axis) -> Node {
        self.mirror = Some(axis);
        self
    }
    pub fn hollow(mut self, wall: f32) -> Node {
        self.hollow = wall;
        self
    }
    pub fn scaled(mut self, s: f32) -> Node {
        self.scale *= s;
        self
    }
    pub fn stretched(mut self, x: f32, y: f32, z: f32) -> Node {
        self.stretch = [self.stretch[0] * x, self.stretch[1] * y, self.stretch[2] * z];
        self
    }

    pub(crate) fn children(&self) -> &[Node] {
        match &self.shape {
            Shape::Union { children, .. } | Shape::Difference { children, .. } | Shape::Intersection { children, .. } => children,
            Shape::Array { child, .. } => std::slice::from_ref(child.as_ref()),
            _ => &[],
        }
    }
}

impl Part {
    pub fn new(name: &str, color: &str, shape: Node) -> Part {
        Part { name: name.into(), color: Some(color.into()), metallic: 0.0, roughness: 0.5, shape }
    }
    pub fn finish(mut self, metallic: f32, roughness: f32) -> Part {
        self.metallic = metallic;
        self.roughness = roughness;
        self
    }
}

/// Why a program was refused.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramError(pub String);

impl std::fmt::Display for ProgramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ProgramError {}

fn err<T>(s: impl Into<String>) -> Result<T, ProgramError> {
    Err(ProgramError(s.into()))
}

impl Program {
    pub fn from_json(text: &str) -> Result<Program, ProgramError> {
        let p: Program = serde_json::from_str(text).map_err(|e| ProgramError(format!("not a shape program: {e}")))?;
        p.validate()?;
        Ok(p)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("programs always serialise")
    }

    /// Checks the limits and that every number is finite and in range. Hosts call this on anything an LLM wrote.
    pub fn validate(&self) -> Result<(), ProgramError> {
        if self.version > PROGRAM_VERSION {
            return err(format!("program version {} is newer than this engine ({PROGRAM_VERSION})", self.version));
        }
        if self.parts.is_empty() {
            return err("a program needs at least one part");
        }
        if let Some(s) = self.real_scale {
            if !(s.is_finite() && (0.001..=1000.0).contains(&s)) {
                return err("realScale must be between 0.001 and 1000");
            }
        }
        if self.parts.len() > MAX_PARTS {
            return err(format!("too many parts ({} > {MAX_PARTS})", self.parts.len()));
        }
        let mut nodes = 0;
        for p in &self.parts {
            if !(0.0..=1.0).contains(&p.metallic) || !(0.0..=1.0).contains(&p.roughness) {
                return err(format!("part \"{}\": metallic and roughness must be 0 to 1", p.name));
            }
            check_node(&p.shape, 1, &mut nodes)?;
        }
        Ok(())
    }
}

fn finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite() && x.abs() <= MAX_SIZE_MM * 4.0)
}

fn positive(what: &str, v: f32) -> Result<(), ProgramError> {
    if v.is_finite() && v > 0.0 && v <= MAX_SIZE_MM {
        Ok(())
    } else {
        err(format!("{what} must be between 0 and {MAX_SIZE_MM} mm (got {v})"))
    }
}

fn check_node(n: &Node, depth: usize, count: &mut usize) -> Result<(), ProgramError> {
    *count += 1;
    if *count > MAX_NODES {
        return err(format!("too many shapes (more than {MAX_NODES})"));
    }
    if depth > MAX_DEPTH {
        return err(format!("shapes nested too deeply (more than {MAX_DEPTH})"));
    }
    if !finite(&n.at) || !finite(&n.rot) {
        return err("a position or rotation is not a usable number");
    }
    positive("scale", n.scale)?;
    for s in n.stretch {
        if !(s.is_finite() && (0.05..=20.0).contains(&s)) {
            return err("stretch must be between 0.05 and 20");
        }
    }
    if !(n.hollow.is_finite() && n.hollow >= 0.0 && n.hollow <= MAX_SIZE_MM) {
        return err("hollow must be a wall thickness in mm");
    }
    match &n.shape {
        Shape::Sphere { r } => positive("sphere r", *r)?,
        Shape::Cuboid { size, round } => {
            for s in size {
                positive("box size", *s)?;
            }
            if !(round.is_finite() && *round >= 0.0) {
                return err("box round must be 0 or more");
            }
        }
        Shape::Cylinder { r, h, round } => {
            positive("cylinder r", *r)?;
            positive("cylinder h", *h)?;
            if !(round.is_finite() && *round >= 0.0) {
                return err("cylinder round must be 0 or more");
            }
        }
        Shape::Cone { r1, r2, h } => {
            positive("cone h", *h)?;
            if !(r1.is_finite() && r2.is_finite() && *r1 >= 0.0 && *r2 >= 0.0 && r1.max(*r2) > 0.0 && r1.max(*r2) <= MAX_SIZE_MM) {
                return err("cone radii must be 0 or more and not both 0");
            }
        }
        Shape::Torus { major, minor } => {
            positive("torus major", *major)?;
            positive("torus minor", *minor)?;
        }
        Shape::Capsule { from, to, r } => {
            positive("capsule r", *r)?;
            if !finite(from) || !finite(to) {
                return err("capsule ends must be usable numbers");
            }
        }
        Shape::Ellipsoid { size } => {
            for s in size {
                positive("ellipsoid size", *s)?;
            }
        }
        Shape::Pyramid { base, h } => {
            positive("pyramid base", *base)?;
            positive("pyramid h", *h)?;
        }
        Shape::Lathe { profile } => {
            if profile.len() < 2 || profile.len() > MAX_POINTS {
                return err(format!("a lathe profile needs 2 to {MAX_POINTS} points"));
            }
            if profile.iter().any(|p| !finite(p) || p[0] < 0.0) {
                return err("lathe points are [radius, z] with radius 0 or more");
            }
        }
        Shape::Extrude { outline, h } => {
            positive("extrude h", *h)?;
            if outline.len() < 3 || outline.len() > MAX_POINTS {
                return err(format!("an outline needs 3 to {MAX_POINTS} points"));
            }
            if outline.iter().any(|p| !finite(p)) {
                return err("outline points must be usable numbers");
            }
        }
        Shape::Union { children, smooth } | Shape::Difference { children, smooth } | Shape::Intersection { children, smooth } => {
            if children.is_empty() {
                return err("a union, difference or intersection needs children");
            }
            if !(smooth.is_finite() && *smooth >= 0.0 && *smooth <= MAX_SIZE_MM) {
                return err("smooth must be a blend radius in mm");
            }
        }
        Shape::Array { count, offset, .. } => {
            if *count == 0 || *count > MAX_ARRAY {
                return err(format!("array count must be 1 to {MAX_ARRAY}"));
            }
            if !finite(offset) {
                return err("array offset must be usable numbers");
            }
        }
    }
    for c in n.children() {
        check_node(c, depth + 1, count)?;
    }
    Ok(())
}

// Builders used by the planner's templates. Sizes are full extents in mm unless named r (radius).
pub fn sphere(r: f32) -> Node {
    Node::new(Shape::Sphere { r })
}
pub fn cuboid(x: f32, y: f32, z: f32) -> Node {
    Node::new(Shape::Cuboid { size: [x, y, z], round: 0.0 })
}
pub fn rbox(x: f32, y: f32, z: f32, round: f32) -> Node {
    Node::new(Shape::Cuboid { size: [x, y, z], round })
}
pub fn cylinder(r: f32, h: f32) -> Node {
    Node::new(Shape::Cylinder { r, h, round: 0.0 })
}
pub fn rcylinder(r: f32, h: f32, round: f32) -> Node {
    Node::new(Shape::Cylinder { r, h, round })
}
pub fn cone(r1: f32, r2: f32, h: f32) -> Node {
    Node::new(Shape::Cone { r1, r2, h })
}
pub fn torus(major: f32, minor: f32) -> Node {
    Node::new(Shape::Torus { major, minor })
}
pub fn capsule(from: [f32; 3], to: [f32; 3], r: f32) -> Node {
    Node::new(Shape::Capsule { from, to, r })
}
pub fn ellipsoid(x: f32, y: f32, z: f32) -> Node {
    Node::new(Shape::Ellipsoid { size: [x, y, z] })
}
pub fn pyramid(base: f32, h: f32) -> Node {
    Node::new(Shape::Pyramid { base, h })
}
pub fn lathe(profile: &[(f32, f32)]) -> Node {
    Node::new(Shape::Lathe { profile: profile.iter().map(|&(r, z)| [r, z]).collect() })
}
pub fn extrude(outline: &[(f32, f32)], h: f32) -> Node {
    Node::new(Shape::Extrude { outline: outline.iter().map(|&(x, y)| [x, y]).collect(), h })
}
pub fn union(smooth: f32, children: Vec<Node>) -> Node {
    Node::new(Shape::Union { children, smooth })
}
pub fn difference(smooth: f32, children: Vec<Node>) -> Node {
    Node::new(Shape::Difference { children, smooth })
}
pub fn intersection(smooth: f32, children: Vec<Node>) -> Node {
    Node::new(Shape::Intersection { children, smooth })
}
pub fn array(child: Node, count: u32, offset: [f32; 3]) -> Node {
    Node::new(Shape::Array { child: Box::new(child), count, offset })
}
