//! Noun templates for the built-in planner. All original, built from SDF primitives in millimetres,
//! Z up, standing on z = 0 and centred on the Z axis; fronts face -Y, animals face +X.
//! Sizes are tabletop or print scale; the second value returned is the factor to life size.

use super::program::{Axis::*, *};
use super::text::Rng;

type Parts = Vec<Part>;

const BLACK: &str = "#1e1e20";
const WHITE: &str = "#f4f4f2";
const WOOD: &str = "#a0703f";
const DARK_WOOD: &str = "#6b4423";
const METAL: &str = "#a9adb3";
const GOLD: &str = "#e6b84d";
const GLASS: &str = "#cfe8f2";
const LEAF: &str = "#4caf50";
const SKIN: &str = "#f1c7a5";

fn p(name: &str, color: &str, n: Node) -> Part {
    Part::new(name, color, n)
}
fn metal(name: &str, color: &str, n: Node) -> Part {
    Part::new(name, color, n).finish(1.0, 0.3)
}
fn glassy(name: &str, n: Node) -> Part {
    Part::new(name, GLASS, n).finish(0.0, 0.05)
}
/// Four copies at (±x, ±y).
fn four(n: Node) -> Node {
    union(0.0, vec![n.mirror(X)]).mirror(Y)
}
fn eyes(r: f32, x: f32, y: f32, z: f32) -> Part {
    p("Eyes", BLACK, sphere(r).at(x, y, z).mirror(X)).finish(0.0, 0.2)
}
/// Eyes on a creature facing +X (mirrored across Y).
fn side_eyes(r: f32, x: f32, y: f32, z: f32) -> Part {
    p("Eyes", BLACK, sphere(r).at(x, y, z).mirror(Y)).finish(0.0, 0.2)
}
fn ring_of(n: usize, radius: f32, z: f32, f: impl Fn(f32) -> Node) -> Vec<Node> {
    (0..n)
        .map(|i| {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            f(a.to_degrees()).at(radius * a.cos(), radius * a.sin(), z)
        })
        .collect()
}

const NAMES: &[&str] = &[
    "mug", "cup", "glass", "tumbler", "bowl", "plate", "vase", "bottle", "jar", "pot", "flower pot", "bucket", "teapot",
    "table", "desk", "chair", "stool", "bench", "bed", "shelf", "bookshelf", "sofa", "couch", "lamp", "candle", "book",
    "phone", "tree", "pine tree", "christmas tree", "mushroom", "flower", "cactus", "rock", "stone", "cloud", "mountain",
    "snowman", "house", "tower", "castle", "tent", "igloo", "car", "truck", "bus", "boat", "sailboat", "ship", "rocket",
    "plane", "airplane", "ufo", "robot", "person", "man", "woman", "kid", "child", "teddy bear", "teddy", "duck", "fish",
    "snake", "turtle", "frog", "apple", "pear", "banana", "carrot", "pumpkin", "egg", "donut", "doughnut", "cake",
    "cupcake", "ice cream", "burger", "hamburger", "lollipop", "heart", "star", "crown", "ring", "gem", "diamond",
    "crystal", "trophy", "sword", "hammer", "key", "pawn", "chess pawn", "ghost", "planet", "hat", "top hat", "dice",
    "die", "coin", "barrel", "chest", "treasure chest", "umbrella", "balloon", "sphere", "ball", "orb", "cube", "box",
    "block", "cylinder", "cone", "torus", "capsule", "pill", "pyramid", "disc", "disk", "tube", "pipe", "ellipsoid",
    "dragon", "bird", "eagle", "owl", "parrot", "penguin", "chicken", "swan", "flamingo",
];

pub fn known(name: &str) -> bool {
    NAMES.contains(&name) || species(name).is_some()
}

/// The template for `name`, or None. Returns the parts and the factor to life size.
pub fn build(name: &str, r: &mut Rng) -> Option<(Parts, f32)> {
    if let Some(s) = species(name) {
        return Some((quadruped(&s, r), s.real));
    }
    if let Some(b) = bird_kind(name) {
        return Some((bird(&b, r), 3.0));
    }
    let v = |r: &mut Rng, x: f32| r.v(x);
    let out: (Parts, f32) = match name {
        "mug" | "cup" => {
            let (rad, h) = (v(r, 40.0), v(r, 95.0));
            let c = r.pick(&["#d63a2f", "#2f6fd6", "#f5c842", "#4caf50", "#f4f4f2"]);
            (vec![p("Mug", c, union(0.0, vec![
                difference(0.0, vec![rcylinder(rad, h, 3.0).at(0.0, 0.0, h / 2.0), cylinder(rad - 5.0, h).at(0.0, 0.0, h / 2.0 + 6.0)]),
                torus(h * 0.23, 5.0).rot(90.0, 0.0, 0.0).at(rad + 2.0, 0.0, h * 0.52),
            ])).finish(0.0, 0.35)], 1.0)
        }
        "glass" | "tumbler" => (vec![glassy("Glass", difference(0.0, vec![cone(30.0, 36.0, 100.0).at(0.0, 0.0, 50.0), cone(27.0, 33.0, 100.0).at(0.0, 0.0, 56.0)]))], 1.0),
        "bowl" => (vec![p("Bowl", r.pick(&["#f4f4f2", "#2f6fd6", "#f2735f"]), difference(1.5, vec![
            lathe(&[(0.0, 0.0), (30.0, 0.0), (55.0, 25.0), (62.0, 55.0), (0.0, 55.0)]),
            lathe(&[(0.0, 6.0), (26.0, 6.0), (52.0, 28.0), (58.0, 62.0), (0.0, 62.0)]),
        ])).finish(0.0, 0.3)], 2.0),
        "plate" => (vec![p("Plate", WHITE, lathe(&[(0.0, 0.0), (70.0, 0.0), (100.0, 10.0), (100.0, 13.0), (70.0, 5.0), (0.0, 5.0)])).finish(0.0, 0.25)], 2.5),
        "vase" => {
            let w = v(r, 1.0);
            (vec![p("Vase", r.pick(&["#1fa39a", "#2f6fd6", "#f2735f", "#ecebe6"]), difference(0.0, vec![
                lathe(&[(0.0, 0.0), (35.0 * w, 0.0), (50.0 * w, 60.0), (38.0 * w, 140.0), (22.0 * w, 175.0), (30.0 * w, 195.0), (0.0, 195.0)]),
                cylinder(18.0 * w, 120.0).at(0.0, 0.0, 200.0),
            ])).finish(0.0, 0.2)], 2.0)
        }
        "bottle" => (vec![
            glassy("Bottle", lathe(&[(0.0, 0.0), (35.0, 0.0), (37.0, 5.0), (37.0, 150.0), (14.0, 190.0), (12.0, 230.0), (0.0, 230.0)])),
            p("Cap", "#d63a2f", rcylinder(13.0, 16.0, 2.0).at(0.0, 0.0, 236.0)),
        ], 1.2),
        "jar" => (vec![glassy("Jar", rcylinder(45.0, 110.0, 8.0).at(0.0, 0.0, 55.0)), metal("Lid", GOLD, rcylinder(47.0, 16.0, 3.0).at(0.0, 0.0, 114.0))], 1.0),
        "pot" | "flower pot" => (vec![
            p("Pot", "#c8643b", difference(0.0, vec![cone(45.0, 60.0, 90.0).at(0.0, 0.0, 45.0), cone(38.0, 54.0, 90.0).at(0.0, 0.0, 52.0)])).finish(0.0, 0.85),
            p("Soil", "#4a3020", cylinder(54.0, 6.0).at(0.0, 0.0, 80.0)).finish(0.0, 1.0),
        ], 2.0),
        "bucket" => (vec![
            metal("Bucket", METAL, difference(0.0, vec![cone(55.0, 65.0, 110.0).at(0.0, 0.0, 55.0), cone(51.0, 61.0, 110.0).at(0.0, 0.0, 60.0)])),
            metal("Handle", "#6f7377", intersection(0.0, vec![torus(63.0, 3.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 105.0), cuboid(200.0, 20.0, 100.0).at(0.0, 0.0, 155.0)])),
        ], 2.5),
        "teapot" => (vec![p("Teapot", WHITE, union(6.0, vec![
            ellipsoid(110.0, 110.0, 85.0).at(0.0, 0.0, 42.0),
            capsule([45.0, 0.0, 35.0], [78.0, 0.0, 72.0], 7.0),
            torus(24.0, 5.5).rot(90.0, 0.0, 0.0).at(-56.0, 0.0, 48.0),
            cylinder(30.0, 8.0).at(0.0, 0.0, 84.0),
            sphere(9.0).at(0.0, 0.0, 94.0),
        ])).finish(0.0, 0.2)], 1.5),
        "table" | "desk" => {
            let (w, d, h) = (v(r, 160.0), v(r, 100.0), v(r, 76.0));
            (vec![p("Table", WOOD, union(0.0, vec![
                rbox(w, d, 8.0, 2.0).at(0.0, 0.0, h - 4.0),
                four(cuboid(8.0, 8.0, h - 6.0).at(w / 2.0 - 10.0, d / 2.0 - 10.0, (h - 6.0) / 2.0)),
            ])).finish(0.0, 0.7)], 10.0)
        }
        "chair" => (vec![p("Chair", r.pick(&[WOOD, "#d63a2f", "#2f6fd6"]), union(0.0, vec![
            rbox(50.0, 50.0, 6.0, 1.5).at(0.0, 0.0, 45.0),
            four(cylinder(2.8, 44.0).at(21.0, 21.0, 22.0)),
            rbox(50.0, 5.0, 46.0, 1.5).at(0.0, 22.5, 70.0),
        ])).finish(0.0, 0.6)], 10.0),
        "stool" => (vec![p("Stool", WOOD, union(0.0, vec![rcylinder(20.0, 5.0, 1.5).at(0.0, 0.0, 50.0), four(cylinder(2.2, 49.0).at(11.0, 11.0, 24.5))])).finish(0.0, 0.7)], 10.0),
        "bench" => (vec![p("Bench", WOOD, union(0.0, vec![rbox(120.0, 35.0, 5.0, 1.0).at(0.0, 0.0, 40.0), cuboid(5.0, 30.0, 38.0).at(50.0, 0.0, 19.0).mirror(X)])).finish(0.0, 0.7)], 12.0),
        "bed" => (vec![
            p("Frame", WOOD, union(0.0, vec![rbox(60.0, 110.0, 14.0, 2.0).at(0.0, 0.0, 13.0), rbox(60.0, 4.0, 40.0, 2.0).at(0.0, 55.0, 26.0)])).finish(0.0, 0.7),
            p("Mattress", WHITE, rbox(56.0, 104.0, 10.0, 4.0).at(0.0, -1.0, 24.0)).finish(0.0, 0.9),
            p("Pillow", "#9fc5e8", rbox(40.0, 18.0, 7.0, 3.5).at(0.0, 40.0, 31.0)).finish(0.0, 0.9),
        ], 20.0),
        "shelf" | "bookshelf" => (vec![
            p("Shelf", WOOD, union(0.0, vec![
                cuboid(4.0, 30.0, 120.0).at(38.0, 0.0, 60.0).mirror(X),
                array(cuboid(76.0, 30.0, 4.0).at(0.0, 0.0, 2.0), 4, [0.0, 0.0, 38.6]),
                cuboid(80.0, 2.0, 120.0).at(0.0, 14.0, 60.0),
            ])).finish(0.0, 0.7),
            p("Books", "#d63a2f", array(rbox(6.0, 22.0, 28.0, 0.8).at(-31.0, -2.0, 18.0), 8, [8.0, 0.0, 0.0])).finish(0.0, 0.6),
        ], 15.0),
        "sofa" | "couch" => (vec![
            p("Sofa", r.pick(&["#2f6fd6", "#8b5a2b", "#4caf50", "#9e9e9e"]), union(2.0, vec![
                rbox(120.0, 50.0, 22.0, 6.0).at(0.0, 0.0, 15.0),
                rbox(120.0, 14.0, 34.0, 6.0).at(0.0, 20.0, 40.0),
                rbox(14.0, 50.0, 30.0, 6.0).at(56.0, 0.0, 26.0).mirror(X),
            ])).finish(0.0, 0.9),
            p("Cushions", "#f4ecd6", rbox(50.0, 38.0, 8.0, 4.0).at(26.0, -4.0, 29.0).mirror(X)).finish(0.0, 0.9),
        ], 15.0),
        "lamp" => (vec![
            metal("Stand", "#3a3a3d", union(0.0, vec![rcylinder(30.0, 6.0, 2.0).at(0.0, 0.0, 3.0), cylinder(3.0, 110.0).at(0.0, 0.0, 60.0)])),
            p("Shade", r.pick(&["#f4ecd6", "#f5c842", "#f48fb1"]), cone(40.0, 22.0, 45.0).hollow(3.0).at(0.0, 0.0, 122.0)).finish(0.0, 0.8),
            p("Bulb", "#fff6c2", sphere(9.0).at(0.0, 0.0, 108.0)).finish(0.0, 0.1),
        ], 4.0),
        "candle" => (vec![
            p("Candle", "#f4ecd6", cylinder(15.0, 70.0).at(0.0, 0.0, 35.0)).finish(0.0, 0.6),
            p("Wick", BLACK, cylinder(0.9, 6.0).at(0.0, 0.0, 72.0)),
            p("Flame", "#f28c28", union(4.0, vec![sphere(4.5).at(0.0, 0.0, 79.0), cone(4.5, 0.0, 13.0).at(0.0, 0.0, 86.0)])).finish(0.0, 0.2),
        ], 1.0),
        "book" => (vec![
            p("Cover", r.pick(&["#d63a2f", "#2f6fd6", "#4caf50", "#8e44ad"]), rbox(60.0, 84.0, 14.0, 1.0).at(0.0, 0.0, 7.0)).finish(0.0, 0.6),
            p("Pages", "#f4ecd6", cuboid(57.0, 80.0, 11.0).at(2.5, 0.0, 7.0)).finish(0.0, 0.9),
        ], 2.5),
        "phone" => (vec![p("Phone", BLACK, rbox(36.0, 8.0, 74.0, 5.0).at(0.0, 0.0, 37.0)).finish(0.3, 0.3), glassy("Screen", rbox(32.0, 1.0, 66.0, 3.0).at(0.0, -4.0, 37.0))], 2.0),
        "tree" => {
            let mut blobs = vec![sphere(v(r, 40.0)).at(0.0, 0.0, 100.0)];
            for i in 0..4 {
                let a = (i as f32 * 90.0 + r.range(0.0, 60.0)).to_radians();
                blobs.push(sphere(v(r, 27.0)).at(28.0 * a.cos(), 28.0 * a.sin(), v(r, 82.0)));
            }
            (vec![p("Trunk", "#7a4b2a", cone(12.0, 8.0, 70.0).at(0.0, 0.0, 35.0)).finish(0.0, 0.9), p("Leaves", LEAF, union(12.0, blobs)).finish(0.0, 0.8)], 40.0)
        }
        "pine tree" | "christmas tree" => (vec![
            p("Trunk", "#7a4b2a", cylinder(7.0, 28.0).at(0.0, 0.0, 14.0)).finish(0.0, 0.9),
            p("Needles", "#2e7d4f", union(0.0, vec![cone(45.0, 4.0, 60.0).at(0.0, 0.0, 55.0), cone(36.0, 3.0, 50.0).at(0.0, 0.0, 85.0), cone(26.0, 0.0, 42.0).at(0.0, 0.0, 113.0)])).finish(0.0, 0.8),
        ], 40.0),
        "mushroom" => {
            let spots: Vec<Node> = [(0.0f32, 90.0f32), (40.0, 45.0), (160.0, 50.0), (280.0, 40.0), (100.0, 25.0), (220.0, 22.0), (340.0, 28.0)]
                .iter()
                .map(|&(az, el)| {
                    let (a, e) = (az.to_radians(), el.to_radians());
                    sphere(6.0).at(40.0 * e.cos() * a.cos(), 40.0 * e.cos() * a.sin(), 45.0 + 25.0 * e.sin())
                })
                .collect();
            (vec![
                p("Stem", "#f4ecd6", lathe(&[(0.0, 0.0), (15.0, 0.0), (11.0, 30.0), (12.0, 48.0), (0.0, 48.0)])).finish(0.0, 0.8),
                p("Cap", "#d63a2f", difference(0.0, vec![ellipsoid(84.0, 84.0, 52.0).at(0.0, 0.0, 45.0), cuboid(100.0, 100.0, 40.0).at(0.0, 0.0, 25.0)])).finish(0.0, 0.4),
                p("Spots", WHITE, union(0.0, spots)),
            ], 1.0)
        }
        "flower" => {
            let petals = ring_of(6, 15.0, 95.0, |deg| ellipsoid(24.0, 12.0, 5.0).rot(0.0, 0.0, deg));
            (vec![
                p("Stem", LEAF, union(2.0, vec![capsule([0.0, 0.0, 0.0], [0.0, 0.0, 92.0], 2.8), ellipsoid(30.0, 4.0, 12.0).rot(0.0, -30.0, 0.0).at(14.0, 0.0, 40.0)])).finish(0.0, 0.7),
                p("Petals", r.pick(&["#f48fb1", "#d63a2f", "#f5c842", "#9b6ad6", WHITE]), union(2.0, petals)).finish(0.0, 0.6),
                p("Centre", "#f5a623", sphere(8.0).at(0.0, 0.0, 96.0)).finish(0.0, 0.7),
            ], 2.5)
        }
        "cactus" => (vec![
            p("Cactus", "#3f9a59", union(4.0, vec![
                capsule([0.0, 0.0, 20.0], [0.0, 0.0, 125.0], 16.0),
                capsule([0.0, 0.0, 70.0], [28.0, 0.0, 70.0], 8.5),
                capsule([28.0, 0.0, 70.0], [28.0, 0.0, 102.0], 8.5),
                capsule([0.0, 0.0, 85.0], [-24.0, 0.0, 85.0], 7.5),
                capsule([-24.0, 0.0, 85.0], [-24.0, 0.0, 108.0], 7.5),
            ])).finish(0.0, 0.7),
            p("Pot", "#c8643b", difference(0.0, vec![cone(22.0, 28.0, 32.0).at(0.0, 0.0, 16.0), cylinder(23.0, 10.0).at(0.0, 0.0, 33.0)])).finish(0.0, 0.85),
        ], 2.0),
        "rock" | "stone" => (vec![p("Rock", "#8d8d88", union(10.0, (0..4).map(|_| {
            let h = r.range(25.0, 45.0);
            ellipsoid(r.range(40.0, 70.0), r.range(35.0, 60.0), h).rot(0.0, 0.0, r.range(0.0, 180.0)).at(r.range(-15.0, 15.0), r.range(-12.0, 12.0), h * 0.5)
        }).collect())).finish(0.0, 0.95)], 3.0),
        "cloud" => (vec![p("Cloud", WHITE, union(10.0, (0..5).map(|i| {
            let x = (i as f32 - 2.0) * 24.0;
            sphere(r.range(22.0, 32.0) * (1.0 - (i as f32 - 2.0).abs() * 0.12)).at(x, r.range(-6.0, 6.0), 30.0 + r.range(-4.0, 8.0))
        }).collect())).finish(0.0, 0.9)], 10.0),
        "mountain" => (vec![
            p("Mountain", "#7d7f7d", union(10.0, vec![cone(80.0, 0.0, 110.0).at(0.0, 0.0, 55.0), cone(55.0, 0.0, 70.0).at(60.0, 10.0, 35.0), cone(50.0, 0.0, 62.0).at(-55.0, -10.0, 31.0)])).finish(0.0, 0.95),
            p("Snow", WHITE, intersection(0.0, vec![cone(81.0, 0.0, 110.0).at(0.0, 0.0, 55.0), cuboid(200.0, 200.0, 40.0).at(0.0, 0.0, 102.0)])).finish(0.0, 0.7),
        ], 500.0),
        "snowman" => (vec![
            p("Snow", "#f4f7fb", union(6.0, vec![sphere(40.0).at(0.0, 0.0, 40.0), sphere(29.0).at(0.0, 0.0, 100.0), sphere(20.0).at(0.0, 0.0, 142.0)])).finish(0.0, 0.8),
            p("Nose", "#f28c28", cone(4.0, 0.0, 18.0).rot(90.0, 0.0, 0.0).at(0.0, -26.0, 142.0)).finish(0.0, 0.6),
            eyes(3.0, 7.0, -17.0, 150.0),
            p("Buttons", BLACK, union(0.0, vec![sphere(3.5).at(0.0, -28.0, 104.0), sphere(3.5).at(0.0, -27.0, 92.0)])),
            p("Hat", BLACK, union(0.0, vec![cylinder(18.0, 4.0).at(0.0, 0.0, 160.0), cylinder(12.0, 22.0).at(0.0, 0.0, 172.0)])).finish(0.0, 0.6),
        ], 10.0),
        "house" => (vec![
            p("Walls", r.pick(&["#f4ecd6", "#9fc5e8", "#f7b98f", WHITE]), cuboid(80.0, 70.0, 60.0).at(0.0, 0.0, 30.0)).finish(0.0, 0.8),
            p("Roof", r.pick(&["#b3202f", "#6b4423", "#2f6fd6"]), union(0.0, vec![
                extrude(&[(-48.0, -2.0), (48.0, -2.0), (0.0, 36.0)], 78.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 60.0),
                cuboid(10.0, 10.0, 25.0).at(22.0, 15.0, 85.0),
            ])).finish(0.0, 0.7),
            p("Door", DARK_WOOD, cuboid(16.0, 2.0, 28.0).at(0.0, -35.0, 14.0)),
            glassy("Windows", cuboid(14.0, 2.0, 14.0).at(24.0, -35.0, 38.0).mirror(X)),
        ], 80.0),
        "tower" => (vec![
            p("Tower", "#8d8d88", union(0.0, vec![cylinder(25.0, 140.0).at(0.0, 0.0, 70.0)])).finish(0.0, 0.9),
            p("Roof", "#2f6fd6", cone(32.0, 0.0, 50.0).at(0.0, 0.0, 165.0)).finish(0.0, 0.6),
            p("Door", DARK_WOOD, cuboid(14.0, 6.0, 24.0).at(0.0, -23.0, 12.0)),
        ], 100.0),
        "castle" => (vec![
            p("Walls", "#9a9a95", union(0.0, vec![
                cuboid(70.0, 70.0, 60.0).at(0.0, 0.0, 30.0),
                array(cuboid(9.0, 8.0, 9.0).at(-30.0, 31.0, 64.0), 5, [15.0, 0.0, 0.0]).mirror(Y),
                array(cuboid(8.0, 9.0, 9.0).at(31.0, -30.0, 64.0), 5, [0.0, 15.0, 0.0]).mirror(X),
                four(cylinder(15.0, 90.0).at(38.0, 38.0, 45.0)),
            ])).finish(0.0, 0.9),
            p("Roofs", "#b3202f", four(cone(18.0, 0.0, 30.0).at(38.0, 38.0, 105.0))).finish(0.0, 0.6),
            p("Gate", DARK_WOOD, rbox(20.0, 4.0, 30.0, 4.0).at(0.0, -35.0, 15.0)),
        ], 100.0),
        "tent" => (vec![
            p("Tent", r.pick(&["#f28c28", "#4caf50", "#2f6fd6"]), extrude(&[(-40.0, 0.0), (40.0, 0.0), (0.0, 55.0)], 90.0).rot(90.0, 0.0, 0.0)).finish(0.0, 0.8),
            p("Door", "#3a2a1a", extrude(&[(-14.0, 0.0), (14.0, 0.0), (0.0, 28.0)], 2.0).rot(90.0, 0.0, 0.0).at(0.0, -45.0, 0.0)),
        ], 30.0),
        "igloo" => (vec![p("Igloo", "#e8f4fb", difference(0.0, vec![
            union(4.0, vec![sphere(50.0), cylinder(17.0, 36.0).rot(90.0, 0.0, 0.0).at(0.0, -48.0, 0.0)]),
            cuboid(200.0, 200.0, 100.0).at(0.0, 0.0, -50.0),
            cylinder(11.0, 50.0).rot(90.0, 0.0, 0.0).at(0.0, -58.0, 0.0),
        ])).finish(0.0, 0.5)], 40.0),
        "car" => {
            let c = r.pick(&["#d63a2f", "#2f6fd6", "#f5c842", "#4caf50", WHITE]);
            let l = v(r, 100.0);
            (vec![
                p("Body", c, union(4.0, vec![rbox(l, 48.0, 22.0, 6.0).at(0.0, 0.0, 22.0), rbox(l * 0.56, 44.0, 22.0, 8.0).at(-5.0, 0.0, 40.0)])).finish(0.2, 0.3),
                glassy("Windows", rbox(l * 0.5, 45.0, 13.0, 5.0).at(-5.0, 0.0, 43.0)),
                p("Wheels", BLACK, four(cylinder(11.0, 8.0).rot(90.0, 0.0, 0.0).at(l * 0.32, 23.0, 11.0))).finish(0.0, 0.9),
                p("Lights", "#fff3b0", sphere(3.5).at(l / 2.0 - 1.0, 14.0, 25.0).mirror(Y)).finish(0.0, 0.1),
            ], 40.0)
        }
        "truck" => (vec![
            p("Cab", r.pick(&["#d63a2f", "#2f6fd6", "#f28c28"]), rbox(40.0, 50.0, 40.0, 5.0).at(42.0, 0.0, 34.0)).finish(0.2, 0.4),
            p("Bed", "#6f7377", difference(0.0, vec![rbox(80.0, 52.0, 22.0, 2.0).at(-20.0, 0.0, 25.0), cuboid(74.0, 46.0, 22.0).at(-20.0, 0.0, 32.0)])).finish(0.6, 0.4),
            glassy("Windows", rbox(41.0, 46.0, 12.0, 3.0).at(44.0, 0.0, 44.0)),
            p("Wheels", BLACK, union(0.0, vec![cylinder(12.0, 9.0).rot(90.0, 0.0, 0.0).at(38.0, 25.0, 12.0).mirror(Y), cylinder(12.0, 9.0).rot(90.0, 0.0, 0.0).at(-35.0, 25.0, 12.0).mirror(Y)])).finish(0.0, 0.9),
        ], 50.0),
        "bus" => (vec![
            p("Bus", "#f5c842", rbox(130.0, 46.0, 48.0, 8.0).at(0.0, 0.0, 34.0)).finish(0.1, 0.4),
            glassy("Windows", array(rbox(14.0, 47.0, 14.0, 2.0).at(-48.0, 0.0, 44.0), 6, [18.0, 0.0, 0.0])),
            p("Wheels", BLACK, four(cylinder(12.0, 8.0).rot(90.0, 0.0, 0.0).at(42.0, 22.0, 12.0))).finish(0.0, 0.9),
        ], 80.0),
        "boat" | "sailboat" | "ship" => (vec![
            p("Hull", r.pick(&["#8b5a2b", "#d63a2f", "#2f6fd6", WHITE]), difference(0.0, vec![ellipsoid(120.0, 44.0, 50.0).at(0.0, 0.0, 25.0), cuboid(140.0, 60.0, 40.0).at(0.0, 0.0, 45.0)])).finish(0.0, 0.5),
            p("Deck", "#d2b48c", ellipsoid(112.0, 38.0, 4.0).at(0.0, 0.0, 24.0)).finish(0.0, 0.8),
            p("Mast", WOOD, cylinder(2.0, 80.0).at(15.0, 0.0, 64.0)),
            p("Sail", WHITE, extrude(&[(0.0, 0.0), (0.0, 62.0), (40.0, 0.0)], 1.5).rot(90.0, 0.0, 0.0).at(17.0, 0.0, 34.0)).finish(0.0, 0.9),
        ], 50.0),
        "rocket" => (vec![
            p("Body", WHITE, lathe(&[(0.0, 0.0), (16.0, 0.0), (18.0, 20.0), (18.0, 100.0), (0.0, 100.0)])).finish(0.3, 0.3),
            p("Nose", "#d63a2f", cone(18.0, 0.0, 40.0).at(0.0, 0.0, 120.0)).finish(0.2, 0.3),
            p("Fins", "#d63a2f", union(0.0, vec![
                extrude(&[(0.0, 0.0), (22.0, 0.0), (0.0, 40.0)], 3.0).rot(90.0, 0.0, 0.0).at(16.0, 0.0, 0.0).mirror(X),
                union(0.0, vec![extrude(&[(0.0, 0.0), (22.0, 0.0), (0.0, 40.0)], 3.0).rot(90.0, 0.0, 0.0).at(16.0, 0.0, 0.0).mirror(X)]).rot(0.0, 0.0, 90.0),
            ])).finish(0.2, 0.3),
            glassy("Window", sphere(7.0).at(0.0, -15.0, 75.0)),
        ], 200.0),
        "plane" | "airplane" => (vec![
            p("Body", WHITE, union(2.0, vec![
                capsule([-60.0, 0.0, 25.0], [60.0, 0.0, 25.0], 11.0),
                extrude(&[(0.0, 0.0), (18.0, 0.0), (22.0, 28.0), (12.0, 28.0)], 3.0).rot(90.0, 0.0, 0.0).at(-66.0, 0.0, 28.0),
            ])).finish(0.2, 0.3),
            p("Wings", "#2f6fd6", union(0.0, vec![rbox(30.0, 150.0, 4.0, 2.0).at(5.0, 0.0, 23.0), rbox(16.0, 52.0, 3.0, 1.0).at(-56.0, 0.0, 30.0)])).finish(0.2, 0.3),
            glassy("Windows", array(sphere(3.0).at(-30.0, -10.0, 29.0), 8, [9.0, 0.0, 0.0]).mirror(Y)),
        ], 200.0),
        "ufo" => (vec![
            metal("Saucer", "#c9ccd1", ellipsoid(110.0, 110.0, 24.0).at(0.0, 0.0, 22.0)),
            glassy("Dome", difference(0.0, vec![sphere(28.0).at(0.0, 0.0, 28.0), cuboid(70.0, 70.0, 34.0).at(0.0, 0.0, 11.0)])),
            p("Lights", "#f5c842", union(0.0, ring_of(8, 47.0, 20.0, |_| sphere(4.5)))).finish(0.0, 0.1),
        ], 100.0),
        "robot" => (vec![
            metal("Body", "#c9ccd1", union(0.0, vec![
                rbox(50.0, 36.0, 56.0, 6.0).at(0.0, 0.0, 60.0),
                rbox(38.0, 32.0, 30.0, 6.0).at(0.0, 0.0, 106.0),
                capsule([31.0, 0.0, 80.0], [37.0, 0.0, 50.0], 7.0).mirror(X),
                rbox(16.0, 20.0, 34.0, 4.0).at(13.0, 0.0, 17.0).mirror(X),
                capsule([0.0, 0.0, 120.0], [0.0, 0.0, 134.0], 1.5),
            ])),
            p("Eyes", "#33c3e0", sphere(4.5).at(9.0, -15.0, 110.0).mirror(X)).finish(0.0, 0.1),
            p("Antenna", "#d63a2f", sphere(4.0).at(0.0, 0.0, 137.0)).finish(0.0, 0.2),
        ], 12.0),
        "person" | "man" | "woman" | "kid" | "child" => (vec![
            p("Skin", SKIN, union(3.0, vec![sphere(14.0).at(0.0, 0.0, 128.0), capsule([0.0, 0.0, 108.0], [0.0, 0.0, 118.0], 6.0)])).finish(0.0, 0.6),
            p("Shirt", r.pick(&["#2f6fd6", "#d63a2f", "#4caf50", "#f5c842"]), union(4.0, vec![capsule([0.0, 0.0, 62.0], [0.0, 0.0, 96.0], 17.0), capsule([22.0, 0.0, 100.0], [28.0, 0.0, 64.0], 6.0).mirror(X)])).finish(0.0, 0.8),
            p("Trousers", "#1f2f6b", capsule([8.0, 0.0, 6.0], [8.0, 0.0, 58.0], 7.5).mirror(X)).finish(0.0, 0.8),
            p("Hair", r.pick(&["#3a2a1a", "#e6b84d", "#7a4b2a", BLACK]), difference(0.0, vec![sphere(15.5).at(0.0, 2.0, 130.0), cuboid(40.0, 30.0, 30.0).at(0.0, -14.0, 118.0)])).finish(0.0, 0.8),
            eyes(2.0, 5.0, -13.0, 131.0),
        ], 13.0),
        "teddy bear" | "teddy" => (vec![
            p("Fur", "#b07a45", union(5.0, vec![
                ellipsoid(56.0, 48.0, 64.0).at(0.0, 0.0, 40.0),
                sphere(24.0).at(0.0, 0.0, 92.0),
                sphere(9.0).at(17.0, 0.0, 112.0).mirror(X),
                capsule([24.0, -4.0, 62.0], [31.0, -14.0, 40.0], 9.0).mirror(X),
                capsule([15.0, -6.0, 12.0], [17.0, -22.0, 11.0], 11.0).mirror(X),
            ])).finish(0.0, 0.95),
            p("Snout", "#e8cfa8", ellipsoid(20.0, 13.0, 14.0).at(0.0, -20.0, 87.0)).finish(0.0, 0.9),
            eyes(2.8, 8.0, -21.0, 98.0),
            p("Nose", BLACK, sphere(3.5).at(0.0, -27.0, 90.0)).finish(0.0, 0.2),
        ], 3.0),
        "duck" => (vec![
            p("Body", "#f5c842", union(6.0, vec![ellipsoid(60.0, 42.0, 36.0).at(0.0, 0.0, 20.0), sphere(16.0).at(24.0, 0.0, 48.0), cone(10.0, 0.0, 16.0).rot(0.0, -60.0, 0.0).at(-30.0, 0.0, 28.0)])).finish(0.0, 0.4),
            p("Beak", "#f28c28", ellipsoid(18.0, 11.0, 5.0).at(41.0, 0.0, 45.0)).finish(0.0, 0.4),
            side_eyes(2.6, 35.0, 9.0, 53.0),
        ], 4.0),
        "fish" => (vec![
            p("Fish", r.pick(&["#f28c28", "#33c3e0", "#f5c842", "#d6338f"]), union(3.0, vec![
                ellipsoid(70.0, 20.0, 40.0).at(0.0, 0.0, 30.0),
                extrude(&[(0.0, 0.0), (-24.0, -18.0), (-24.0, 18.0)], 4.0).rot(90.0, 0.0, 0.0).at(-28.0, 0.0, 30.0),
                extrude(&[(-10.0, 0.0), (12.0, 0.0), (-4.0, 14.0)], 3.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 46.0),
            ])).finish(0.1, 0.3),
            side_eyes(3.5, 22.0, 8.0, 36.0),
        ], 3.0),
        "snake" => {
            let pts: Vec<[f32; 3]> = (0..12).map(|i| [i as f32 * 12.0 - 66.0, 14.0 * (i as f32 * 0.9).sin(), 7.0]).collect();
            let mut segs: Vec<Node> = pts.windows(2).map(|w| capsule(w[0], w[1], 7.0)).collect();
            let head = pts[pts.len() - 1];
            segs.push(ellipsoid(24.0, 18.0, 13.0).at(head[0] + 8.0, head[1], 8.0));
            (vec![p("Snake", "#4caf50", union(3.0, segs)).finish(0.0, 0.4), side_eyes(2.2, head[0] + 14.0, head[1] + 6.0, 12.0)], 5.0)
        }
        "turtle" => (vec![
            p("Shell", "#3f7a3a", difference(0.0, vec![ellipsoid(80.0, 64.0, 50.0).at(0.0, 0.0, 14.0), cuboid(100.0, 100.0, 30.0).at(0.0, 0.0, -1.0)])).finish(0.0, 0.5),
            p("Skin", "#8bc34a", union(4.0, vec![
                ellipsoid(76.0, 60.0, 12.0).at(0.0, 0.0, 10.0),
                sphere(13.0).at(46.0, 0.0, 18.0),
                four(ellipsoid(18.0, 14.0, 10.0).at(24.0, 24.0, 6.0)),
            ])).finish(0.0, 0.7),
            side_eyes(2.4, 55.0, 6.0, 22.0),
        ], 5.0),
        "frog" => (vec![
            p("Frog", "#6abf4b", union(6.0, vec![
                ellipsoid(54.0, 48.0, 34.0).at(0.0, 0.0, 18.0),
                sphere(9.0).at(14.0, 12.0, 36.0).mirror(Y),
                ellipsoid(34.0, 14.0, 12.0).at(-10.0, 24.0, 6.0).mirror(Y),
            ])).finish(0.0, 0.4),
            side_eyes(4.0, 20.0, 12.0, 40.0),
        ], 3.0),
        "apple" => (vec![
            p("Apple", r.pick(&["#d63a2f", "#8bc34a", "#f5c842"]), difference(6.0, vec![sphere(32.0).at(0.0, 0.0, 31.0), sphere(9.0).at(0.0, 0.0, 64.0)])).finish(0.0, 0.3),
            p("Stem", "#6b4423", capsule([0.0, 0.0, 54.0], [3.0, 0.0, 70.0], 2.2)),
            p("Leaf", LEAF, ellipsoid(18.0, 6.0, 3.0).rot(0.0, -30.0, 0.0).at(11.0, 0.0, 67.0)),
        ], 1.0),
        "pear" => (vec![p("Pear", "#c5d94a", union(16.0, vec![sphere(28.0).at(0.0, 0.0, 28.0), sphere(17.0).at(0.0, 0.0, 64.0)])).finish(0.0, 0.4), p("Stem", "#6b4423", capsule([0.0, 0.0, 78.0], [2.0, 0.0, 92.0], 2.0))], 1.0),
        "banana" => {
            let pts: Vec<[f32; 3]> = (0..7).map(|i| { let a = (i as f32 / 6.0 - 0.5) * 1.6; [70.0 * a.sin(), 0.0, 14.0 + 40.0 * (1.0 - a.cos())] }).collect();
            let segs: Vec<Node> = pts.windows(2).enumerate().map(|(i, w)| capsule(w[0], w[1], 13.0 - (i as f32 - 2.5).abs() * 2.2)).collect();
            (vec![p("Banana", "#f5d742", union(4.0, segs)).finish(0.0, 0.5)], 1.0)
        }
        "carrot" => (vec![
            p("Carrot", "#f28c28", cone(3.0, 17.0, 100.0).rot(0.0, 90.0, 0.0).at(0.0, 0.0, 17.0)).finish(0.0, 0.6),
            p("Leaves", LEAF, union(2.0, vec![capsule([-50.0, 0.0, 17.0], [-75.0, 0.0, 30.0], 3.0), capsule([-50.0, 0.0, 17.0], [-74.0, 8.0, 14.0], 3.0), capsule([-50.0, 0.0, 17.0], [-73.0, -8.0, 22.0], 3.0)])),
        ], 1.0),
        "pumpkin" => (vec![
            p("Pumpkin", "#f28c28", union(6.0, ring_of(8, 16.0, 26.0, |deg| ellipsoid(44.0, 26.0, 52.0).rot(0.0, 0.0, deg)))).finish(0.0, 0.5),
            p("Stem", "#4a7a32", cylinder(5.0, 16.0).rot(10.0, 0.0, 0.0).at(0.0, 0.0, 55.0)),
        ], 3.0),
        "egg" => (vec![p("Egg", "#f4ecd6", lathe(&[(0.0, 0.0), (14.0, 1.5), (21.0, 10.0), (22.0, 24.0), (19.0, 40.0), (12.0, 52.0), (0.0, 56.0)])).finish(0.0, 0.4)], 1.0),
        "donut" | "doughnut" => (vec![
            p("Dough", "#d9a05b", torus(30.0, 14.0).at(0.0, 0.0, 14.0)).finish(0.0, 0.6),
            p("Icing", r.pick(&["#f48fb1", "#5a3420", WHITE]), intersection(0.0, vec![torus(30.0, 15.2).at(0.0, 0.0, 14.0), cuboid(120.0, 120.0, 20.0).at(0.0, 0.0, 26.0)])).finish(0.0, 0.3),
        ], 1.0),
        "cake" => (vec![
            p("Cake", "#f4ecd6", union(0.0, vec![rcylinder(60.0, 40.0, 3.0).at(0.0, 0.0, 20.0), rcylinder(44.0, 32.0, 3.0).at(0.0, 0.0, 56.0)])).finish(0.0, 0.6),
            p("Icing", "#f48fb1", union(0.0, vec![torus(60.0, 3.5).at(0.0, 0.0, 40.0), torus(44.0, 3.0).at(0.0, 0.0, 72.0), rcylinder(44.5, 3.0, 1.0).at(0.0, 0.0, 71.0)])).finish(0.0, 0.3),
            p("Cherry", "#b3202f", sphere(7.0).at(0.0, 0.0, 79.0)).finish(0.0, 0.2),
        ], 2.0),
        "cupcake" => (vec![
            p("Case", "#f48fb1", cone(24.0, 31.0, 30.0).at(0.0, 0.0, 15.0)).finish(0.0, 0.6),
            p("Frosting", "#f4ecd6", union(6.0, vec![ellipsoid(70.0, 70.0, 24.0).at(0.0, 0.0, 34.0), sphere(20.0).at(0.0, 0.0, 46.0), cone(11.0, 0.0, 18.0).at(0.0, 0.0, 66.0)])).finish(0.0, 0.5),
            p("Cherry", "#d63a2f", sphere(6.0).at(0.0, 0.0, 76.0)).finish(0.0, 0.2),
        ], 1.0),
        "ice cream" => (vec![
            p("Cone", "#d9a05b", cone(2.0, 22.0, 75.0).at(0.0, 0.0, 37.5)).finish(0.0, 0.8),
            p("Scoop", r.pick(&["#f48fb1", "#98e2c6", "#f4ecd6", "#5a3420"]), union(3.0, vec![sphere(24.0).at(0.0, 0.0, 86.0), sphere(20.0).at(0.0, 0.0, 116.0)])).finish(0.0, 0.5),
        ], 1.0),
        "burger" | "hamburger" => (vec![
            p("Bun", "#d9a05b", union(0.0, vec![rcylinder(45.0, 14.0, 6.0).at(0.0, 0.0, 7.0), intersection(0.0, vec![ellipsoid(92.0, 92.0, 60.0).at(0.0, 0.0, 34.0), cuboid(100.0, 100.0, 40.0).at(0.0, 0.0, 54.0)])])).finish(0.0, 0.6),
            p("Patty", "#5a3420", rcylinder(47.0, 12.0, 5.0).at(0.0, 0.0, 20.0)).finish(0.0, 0.8),
            p("Cheese", "#f5c842", cuboid(84.0, 84.0, 2.5).rot(0.0, 0.0, 45.0).at(0.0, 0.0, 27.5)).finish(0.0, 0.4),
            p("Lettuce", LEAF, rcylinder(48.0, 3.0, 1.2).at(0.0, 0.0, 31.0)).finish(0.0, 0.6),
        ], 1.0),
        "lollipop" => (vec![p("Stick", WHITE, cylinder(2.5, 90.0).at(0.0, 0.0, 45.0)), p("Candy", r.pick(&["#f48fb1", "#d63a2f", "#9b6ad6"]), rcylinder(28.0, 10.0, 5.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 112.0)).finish(0.0, 0.15)], 1.0),
        "heart" => {
            let pts: Vec<(f32, f32)> = (0..40).map(|i| {
                let t = i as f32 / 40.0 * std::f32::consts::TAU;
                (16.0 * t.sin().powi(3) * 2.6, (13.0 * t.cos() - 5.0 * (2.0 * t).cos() - 2.0 * (3.0 * t).cos() - (4.0 * t).cos()) * 2.6 + 44.0)
            }).collect();
            (vec![p("Heart", "#d63a2f", extrude(&pts, 20.0).rot(90.0, 0.0, 0.0)).finish(0.0, 0.3)], 1.0)
        }
        "star" => {
            let pts: Vec<(f32, f32)> = (0..10).map(|i| { let a = std::f32::consts::FRAC_PI_2 + i as f32 * std::f32::consts::PI / 5.0; let r = if i % 2 == 0 { 50.0 } else { 21.0 }; (r * a.cos(), r * a.sin() + 41.0) }).collect();
            (vec![p("Star", "#f5c842", extrude(&pts, 14.0).rot(90.0, 0.0, 0.0)).finish(0.3, 0.3)], 1.0)
        }
        "crown" => (vec![
            metal("Crown", GOLD, union(0.0, {
                let mut v = vec![difference(0.0, vec![cylinder(32.0, 22.0).at(0.0, 0.0, 11.0), cylinder(28.5, 30.0).at(0.0, 0.0, 14.0)])];
                v.extend(ring_of(8, 30.5, 26.0, |_| cone(5.0, 0.0, 18.0)));
                v
            })),
            p("Gems", "#d63a2f", union(0.0, ring_of(8, 32.0, 11.0, |_| sphere(3.0)))).finish(0.0, 0.05),
        ], 3.0),
        "ring" => (vec![metal("Band", GOLD, torus(20.0, 3.5).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 24.0)), p("Gem", "#33c3e0", union(0.0, vec![cone(0.0, 7.0, 7.0).at(0.0, 0.0, 49.0), cone(7.0, 4.0, 3.0).at(0.0, 0.0, 54.0)])).finish(0.0, 0.03)], 0.5),
        "gem" | "diamond" | "crystal" => (vec![p("Gem", r.pick(&["#33c3e0", "#d6338f", "#4caf50", "#9b6ad6"]), union(0.0, vec![cone(0.0, 30.0, 32.0).at(0.0, 0.0, 16.0), cone(30.0, 18.0, 12.0).at(0.0, 0.0, 38.0)])).finish(0.0, 0.03)], 0.3),
        "trophy" => (vec![metal("Trophy", GOLD, union(0.0, vec![
            difference(0.0, vec![lathe(&[(0.0, 0.0), (28.0, 0.0), (28.0, 8.0), (8.0, 12.0), (6.0, 40.0), (10.0, 46.0), (30.0, 60.0), (34.0, 95.0), (0.0, 95.0)]), cylinder(30.0, 40.0).at(0.0, 0.0, 100.0)]),
            torus(13.0, 3.0).rot(90.0, 0.0, 0.0).at(36.0, 0.0, 78.0).mirror(X),
        ]))], 3.0),
        "sword" => (vec![
            metal("Blade", "#d9dde2", extrude(&[(-4.5, 0.0), (4.5, 0.0), (4.5, 92.0), (0.0, 104.0), (-4.5, 92.0)], 2.2).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 40.0)),
            metal("Guard", GOLD, rbox(38.0, 7.0, 7.0, 2.5).at(0.0, 0.0, 38.0)),
            p("Grip", "#6b4423", cylinder(3.5, 26.0).at(0.0, 0.0, 22.0)).finish(0.0, 0.8),
            metal("Pommel", GOLD, sphere(5.5).at(0.0, 0.0, 7.0)),
        ], 8.0),
        "hammer" => (vec![p("Handle", WOOD, rcylinder(5.0, 110.0, 2.0).at(0.0, 0.0, 55.0)).finish(0.0, 0.7), metal("Head", "#6f7377", rbox(62.0, 22.0, 22.0, 2.0).at(0.0, 0.0, 115.0))], 2.5),
        "key" => (vec![metal("Key", GOLD, union(0.0, vec![
            difference(0.0, vec![cylinder(16.0, 4.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 80.0), cylinder(8.0, 6.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 80.0)]),
            cylinder(3.5, 66.0).at(0.0, 0.0, 33.0),
            cuboid(10.0, 4.0, 6.0).at(6.0, 0.0, 4.0),
            cuboid(8.0, 4.0, 5.0).at(5.0, 0.0, 16.0),
        ]))], 1.0),
        "pawn" | "chess pawn" => (vec![p("Pawn", r.pick(&[WHITE, BLACK]), union(3.0, vec![
            lathe(&[(0.0, 0.0), (22.0, 0.0), (22.0, 6.0), (16.0, 10.0), (15.0, 14.0), (8.0, 40.0), (13.0, 44.0), (13.0, 47.0), (6.0, 49.0), (0.0, 49.0)]),
            sphere(11.0).at(0.0, 0.0, 58.0),
        ])).finish(0.0, 0.25)], 1.0),
        "ghost" => (vec![
            p("Ghost", WHITE, union(6.0, {
                let mut v = vec![sphere(30.0).at(0.0, 0.0, 72.0), cylinder(30.0, 50.0).at(0.0, 0.0, 47.0)];
                v.extend(ring_of(6, 22.0, 18.0, |_| sphere(10.0)));
                v
            })).finish(0.0, 0.5),
            p("Face", BLACK, union(0.0, vec![ellipsoid(8.0, 4.0, 12.0).at(10.0, -27.0, 76.0).mirror(X), ellipsoid(10.0, 4.0, 7.0).at(0.0, -29.0, 60.0)])),
        ], 12.0),
        "planet" => (vec![
            p("Planet", r.pick(&["#f28c28", "#2f6fd6", "#e6b84d", "#9b6ad6"]), sphere(40.0).at(0.0, 0.0, 40.0)).finish(0.0, 0.6),
            p("Rings", "#e8dcc0", torus(62.0, 4.0).stretched(1.0, 1.0, 0.35).rot(20.0, 0.0, 0.0).at(0.0, 0.0, 40.0)).finish(0.0, 0.5),
        ], 1.0),
        "hat" | "top hat" => (vec![
            p("Hat", BLACK, union(0.0, vec![rcylinder(40.0, 3.0, 1.0).at(0.0, 0.0, 1.5), cylinder(25.0, 52.0).at(0.0, 0.0, 28.0)])).finish(0.0, 0.6),
            p("Band", "#d63a2f", cylinder(25.6, 8.0).at(0.0, 0.0, 8.0)),
        ], 3.0),
        "dice" | "die" => (vec![p("Die", WHITE, difference(0.0, vec![
            rbox(40.0, 40.0, 40.0, 5.0).at(0.0, 0.0, 20.0),
            sphere(4.0).at(0.0, -21.0, 20.0),
            sphere(4.0).at(21.0, 9.0, 29.0), sphere(4.0).at(21.0, -9.0, 11.0),
            sphere(4.0).at(-10.0, -10.0, 41.0), sphere(4.0).at(0.0, 0.0, 41.0), sphere(4.0).at(10.0, 10.0, 41.0),
        ])).finish(0.0, 0.3)], 0.5),
        "coin" => (vec![metal("Coin", GOLD, rcylinder(25.0, 4.0, 1.0).at(0.0, 0.0, 2.0))], 1.0),
        "barrel" => (vec![
            p("Barrel", WOOD, lathe(&[(0.0, 0.0), (30.0, 0.0), (36.0, 35.0), (30.0, 70.0), (0.0, 70.0)])).finish(0.0, 0.8),
            metal("Hoops", "#6f7377", union(0.0, vec![torus(33.2, 2.0).at(0.0, 0.0, 17.5), torus(33.2, 2.0).at(0.0, 0.0, 52.5)])),
        ], 10.0),
        "chest" | "treasure chest" => (vec![
            p("Chest", WOOD, union(0.0, vec![rbox(80.0, 50.0, 40.0, 2.0).at(0.0, 0.0, 20.0), intersection(0.0, vec![cylinder(25.0, 80.0).rot(0.0, 90.0, 0.0).at(0.0, 0.0, 40.0), cuboid(90.0, 60.0, 30.0).at(0.0, 0.0, 55.0)])])).finish(0.0, 0.8),
            metal("Trim", GOLD, union(0.0, vec![cuboid(6.0, 52.0, 42.0).at(30.0, 0.0, 20.0).mirror(X), cuboid(10.0, 4.0, 12.0).at(0.0, -26.0, 40.0)])),
        ], 8.0),
        "umbrella" => (vec![
            p("Canopy", r.pick(&["#d63a2f", "#2f6fd6", "#f5c842"]), difference(0.0, vec![sphere(60.0).at(0.0, 0.0, 70.0), sphere(57.0).at(0.0, 0.0, 66.0), cuboid(140.0, 140.0, 80.0).at(0.0, 0.0, 70.0)])).finish(0.0, 0.5),
            p("Pole", BLACK, union(0.0, vec![cylinder(2.0, 130.0).at(0.0, 0.0, 65.0), torus(7.0, 2.0).rot(90.0, 0.0, 0.0).at(7.0, 0.0, 2.0)])),
        ], 8.0),
        "balloon" => (vec![
            p("Balloon", r.pick(&["#d63a2f", "#2f6fd6", "#f5c842", "#f48fb1"]), union(2.0, vec![ellipsoid(50.0, 50.0, 62.0).at(0.0, 0.0, 100.0), cone(4.0, 1.0, 6.0).at(0.0, 0.0, 68.0)])).finish(0.0, 0.15),
            p("String", WHITE, cylinder(0.8, 66.0).at(0.0, 0.0, 33.0)),
        ], 3.0),
        "sphere" | "ball" | "orb" => (vec![p("Sphere", r.pick(&["#d63a2f", "#2f6fd6", "#f5c842", "#4caf50"]), sphere(25.0).at(0.0, 0.0, 25.0)).finish(0.0, 0.4)], 1.0),
        "cube" | "box" | "block" => (vec![p("Cube", "#e9c9a8", cuboid(40.0, 40.0, 40.0).at(0.0, 0.0, 20.0)).finish(0.0, 0.6)], 1.0),
        "cylinder" => (vec![p("Cylinder", "#e9c9a8", cylinder(20.0, 50.0).at(0.0, 0.0, 25.0)).finish(0.0, 0.6)], 1.0),
        "tube" | "pipe" => (vec![p("Tube", METAL, difference(0.0, vec![cylinder(20.0, 60.0).at(0.0, 0.0, 30.0), cylinder(15.0, 70.0).at(0.0, 0.0, 30.0)])).finish(0.8, 0.3)], 1.0),
        "cone" => (vec![p("Cone", "#e9c9a8", cone(25.0, 0.0, 50.0).at(0.0, 0.0, 25.0)).finish(0.0, 0.6)], 1.0),
        "torus" => (vec![p("Torus", "#e9c9a8", torus(25.0, 10.0).at(0.0, 0.0, 10.0)).finish(0.0, 0.6)], 1.0),
        "capsule" | "pill" => (vec![p("Capsule", "#e9c9a8", capsule([0.0, 0.0, 15.0], [0.0, 0.0, 55.0], 15.0)).finish(0.0, 0.6)], 1.0),
        "pyramid" => (vec![p("Pyramid", "#e6c27a", pyramid(60.0, 45.0).at(0.0, 0.0, 22.5)).finish(0.0, 0.8)], 1.0),
        "disc" | "disk" => (vec![p("Disc", "#e9c9a8", rcylinder(30.0, 6.0, 1.5).at(0.0, 0.0, 3.0)).finish(0.0, 0.6)], 1.0),
        "ellipsoid" => (vec![p("Ellipsoid", "#e9c9a8", ellipsoid(60.0, 40.0, 30.0).at(0.0, 0.0, 15.0)).finish(0.0, 0.6)], 1.0),
        "dragon" => {
            let s = Species { horns: true, wings: true, color: "#3f9a59", ..species("dinosaur").unwrap() };
            return Some((quadruped(&s, r), 30.0));
        }
        _ => return None,
    };
    Some(out)
}

#[derive(Clone, Copy, PartialEq)]
enum Ear {
    None,
    Pointy,
    Floppy,
    Long,
    Round,
    Big,
}

/// A four-legged creature, length along X, facing +X.
#[derive(Clone, Copy)]
struct Species {
    len: f32,
    girth: f32,
    leg: f32,
    head: f32,
    snout: f32,
    neck: f32,
    ear: Ear,
    tail: f32,
    /// -1 hangs down, 0 straight back, 1 up.
    tail_up: f32,
    horns: bool,
    horn: bool,
    trunk: bool,
    wings: bool,
    color: &'static str,
    real: f32,
}

const Q: Species = Species { len: 70.0, girth: 34.0, leg: 30.0, head: 18.0, snout: 10.0, neck: 12.0, ear: Ear::Pointy, tail: 30.0, tail_up: 0.6, horns: false, horn: false, trunk: false, wings: false, color: "#b98250", real: 10.0 };

fn species(name: &str) -> Option<Species> {
    Some(match name {
        "cat" | "kitten" | "kitty" => Species { len: 60.0, girth: 30.0, leg: 28.0, head: 18.0, snout: 0.0, neck: 10.0, tail: 40.0, tail_up: 0.55, color: "#e8a25a", real: 6.0, ..Q },
        "dog" | "puppy" | "doggy" => Species { snout: 14.0, neck: 16.0, ear: Ear::Floppy, tail: 26.0, tail_up: 0.8, real: 8.0, ..Q },
        "wolf" => Species { snout: 18.0, neck: 16.0, tail: 34.0, tail_up: -0.3, color: "#8d8d88", real: 12.0, ..Q },
        "fox" => Species { len: 66.0, girth: 28.0, snout: 18.0, neck: 14.0, ear: Ear::Pointy, tail: 52.0, tail_up: 0.1, color: "#e06a2b", real: 8.0, ..Q },
        "horse" | "pony" => Species { len: 110.0, girth: 42.0, leg: 70.0, head: 18.0, snout: 30.0, neck: 55.0, tail: 45.0, tail_up: -0.8, color: "#8b5a2b", real: 20.0, ..Q },
        "unicorn" => Species { len: 110.0, girth: 42.0, leg: 70.0, head: 18.0, snout: 30.0, neck: 55.0, tail: 45.0, tail_up: -0.8, horn: true, color: "#f4f0fb", real: 20.0, ..Q },
        "zebra" | "donkey" => Species { len: 105.0, girth: 42.0, leg: 62.0, head: 18.0, snout: 28.0, neck: 45.0, ear: Ear::Long, tail: 40.0, tail_up: -0.8, color: if name == "zebra" { WHITE } else { "#8d8d88" }, real: 18.0, ..Q },
        "cow" | "bull" | "ox" => Species { len: 110.0, girth: 56.0, leg: 46.0, head: 22.0, snout: 14.0, neck: 18.0, ear: Ear::Round, tail: 40.0, tail_up: -0.9, horns: true, color: "#f2efe9", real: 20.0, ..Q },
        "pig" | "piggy" | "boar" => Species { len: 80.0, girth: 52.0, leg: 22.0, head: 22.0, snout: 9.0, neck: 4.0, tail: 8.0, tail_up: 0.6, color: "#f4a7b9", real: 12.0, ..Q },
        "sheep" | "lamb" => Species { len: 75.0, girth: 54.0, leg: 30.0, head: 16.0, snout: 8.0, neck: 12.0, ear: Ear::Floppy, tail: 8.0, color: "#f5f2ea", real: 12.0, ..Q },
        "goat" => Species { len: 78.0, girth: 40.0, leg: 40.0, head: 16.0, snout: 14.0, neck: 24.0, horns: true, tail: 8.0, color: "#e8dcc0", real: 12.0, ..Q },
        "lion" => Species { len: 95.0, girth: 45.0, leg: 40.0, head: 27.0, snout: 9.0, neck: 18.0, ear: Ear::Round, tail: 50.0, tail_up: -0.2, color: "#d9a441", real: 20.0, ..Q },
        "tiger" | "leopard" | "cheetah" | "panther" => Species { len: 100.0, girth: 40.0, leg: 38.0, head: 22.0, snout: 9.0, neck: 14.0, ear: Ear::Round, tail: 55.0, tail_up: -0.1, color: if name == "panther" { BLACK } else { "#e7832f" }, real: 20.0, ..Q },
        "bear" | "panda" => Species { len: 90.0, girth: 56.0, leg: 34.0, head: 25.0, snout: 10.0, neck: 12.0, ear: Ear::Round, tail: 4.0, color: if name == "panda" { WHITE } else { "#7a4b2a" }, real: 20.0, ..Q },
        "elephant" => Species { len: 110.0, girth: 72.0, leg: 50.0, head: 32.0, snout: 0.0, neck: 8.0, ear: Ear::Big, tail: 22.0, tail_up: -0.9, trunk: true, color: "#9aa0a6", real: 30.0, ..Q },
        "giraffe" => Species { len: 80.0, girth: 36.0, leg: 90.0, head: 14.0, snout: 16.0, neck: 95.0, tail: 30.0, tail_up: -0.8, horns: true, color: "#e8b04b", real: 30.0, ..Q },
        "rabbit" | "bunny" | "hare" => Species { len: 50.0, girth: 34.0, leg: 14.0, head: 16.0, snout: 0.0, neck: 14.0, ear: Ear::Long, tail: 4.0, color: "#d9cfc4", real: 4.0, ..Q },
        "mouse" | "rat" | "hamster" => Species { len: 40.0, girth: 24.0, leg: 10.0, head: 12.0, snout: 8.0, neck: 6.0, ear: Ear::Round, tail: if name == "hamster" { 2.0 } else { 46.0 }, tail_up: 0.0, color: "#a7a0a0", real: 2.0, ..Q },
        "deer" | "reindeer" | "moose" => Species { len: 90.0, girth: 36.0, leg: 62.0, head: 16.0, snout: 16.0, neck: 40.0, horns: true, tail: 8.0, color: "#a8703c", real: 18.0, ..Q },
        "camel" | "llama" | "alpaca" => Species { len: 95.0, girth: 42.0, leg: 66.0, head: 15.0, snout: 16.0, neck: 60.0, tail: 18.0, tail_up: -0.8, color: "#d2b48c", real: 20.0, ..Q },
        "dinosaur" | "dino" | "dragon lizard" => Species { len: 120.0, girth: 46.0, leg: 44.0, head: 20.0, snout: 18.0, neck: 60.0, ear: Ear::None, tail: 90.0, tail_up: -0.35, color: "#5fa35a", real: 50.0, ..Q },
        "lizard" | "crocodile" | "alligator" => Species { len: 110.0, girth: 30.0, leg: 12.0, head: 16.0, snout: 30.0, neck: 0.0, ear: Ear::None, tail: 90.0, tail_up: 0.0, color: "#5f8a3a", real: 10.0, ..Q },
        "animal" | "pet" | "creature" | "beast" => Q,
        _ => return None,
    })
}

fn quadruped(s: &Species, r: &mut Rng) -> Parts {
    let (l, g, lg, hr) = (r.v(s.len), r.v(s.girth), r.v(s.leg), r.v(s.head));
    let bz = lg + g * 0.4;
    let mut body = vec![ellipsoid(l, g, g * 0.85).at(0.0, 0.0, bz)];
    let lr = g * 0.16;
    body.push(four(capsule([l * 0.32, g * 0.26, bz], [l * 0.32, g * 0.26, lr], lr)));
    let head = [l * 0.5 + hr * 0.35, 0.0, bz + g * 0.25 + s.neck];
    if s.neck > 0.0 {
        body.push(capsule([l * 0.36, 0.0, bz + g * 0.1], [head[0] - hr * 0.3, 0.0, head[2] - hr * 0.3], g * 0.2));
    }
    body.push(sphere(hr).at(head[0], 0.0, head[2]));
    if s.snout > 0.0 {
        body.push(ellipsoid(hr * 0.7 + s.snout, hr * 1.05, hr * 0.85).at(head[0] + hr * 0.45 + s.snout * 0.4, 0.0, head[2] - hr * 0.25));
    }
    match s.ear {
        Ear::Pointy => body.push(cone(hr * 0.36, 0.0, hr * 0.75).rot(-12.0, 0.0, 0.0).at(head[0] - hr * 0.1, hr * 0.55, head[2] + hr * 0.85).mirror(Y)),
        Ear::Floppy => body.push(ellipsoid(hr * 0.5, hr * 0.28, hr * 1.15).at(head[0] - hr * 0.15, hr * 0.92, head[2] - hr * 0.15).mirror(Y)),
        Ear::Long => body.push(capsule([head[0] - hr * 0.2, hr * 0.35, head[2] + hr * 0.6], [head[0] - hr * 0.45, hr * 0.5, head[2] + hr * 2.3], hr * 0.26).mirror(Y)),
        Ear::Round => body.push(sphere(hr * 0.33).at(head[0] - hr * 0.15, hr * 0.7, head[2] + hr * 0.75).mirror(Y)),
        Ear::Big => body.push(ellipsoid(hr * 0.3, hr * 1.4, hr * 1.7).at(head[0] - hr * 0.55, hr * 0.95, head[2]).mirror(Y)),
        Ear::None => {}
    }
    if s.tail > 0.0 {
        let t = r.v(s.tail);
        let start = [-l * 0.46, 0.0, bz + g * 0.15];
        let dir = glam::Vec2::new(-1.0, s.tail_up).normalize() * t;
        body.push(capsule(start, [start[0] + dir.x, 0.0, (start[2] + dir.y).max(g * 0.08)], (g * 0.07).max(1.5)));
    }
    if s.trunk {
        body.push(capsule([head[0] + hr * 0.8, 0.0, head[2] - hr * 0.2], [head[0] + hr * 1.25, 0.0, lg * 0.35], hr * 0.24));
    }
    let mut parts = vec![p("Body", s.color, union(g * 0.16, body)).finish(0.0, 0.8)];
    if s.horns || s.horn {
        let h = if s.horn {
            cone(hr * 0.16, 0.0, hr * 1.0).rot(0.0, 30.0, 0.0).at(head[0] + hr * 0.4, 0.0, head[2] + hr * 1.1)
        } else {
            cone(hr * 0.15, 0.0, hr * 0.75).rot(-15.0, 0.0, 0.0).at(head[0] - hr * 0.1, hr * 0.38, head[2] + hr * 1.0).mirror(Y)
        };
        parts.push(p("Horns", if s.horn { GOLD } else { "#f4ecd6" }, h).finish(if s.horn { 1.0 } else { 0.0 }, 0.4));
    }
    if s.wings {
        parts.push(p("Wings", "#2e7d4f", extrude(&[(l * 0.2, 0.0), (-l * 0.25, 0.0), (-l * 0.15, l * 0.55), (l * 0.05, l * 0.7)], (g * 0.12).max(4.0)).rot(55.0, 0.0, 0.0).at(0.0, g * 0.3, bz + g * 0.42).mirror(Y)).finish(0.0, 0.6));
    }
    parts.push(side_eyes(hr * 0.14, head[0] + hr * 0.78, hr * 0.42, head[2] + hr * 0.25));
    parts
}

#[derive(Clone, Copy)]
struct BirdKind {
    len: f32,
    upright: bool,
    neck: f32,
    leg: f32,
    color: &'static str,
    head: &'static str,
    beak: &'static str,
}

fn bird_kind(name: &str) -> Option<BirdKind> {
    let b = BirdKind { len: 50.0, upright: false, neck: 0.0, leg: 12.0, color: "#2f6fd6", head: "#2f6fd6", beak: "#f5a623" };
    Some(match name {
        "bird" | "sparrow" | "robin" | "pigeon" | "bluebird" => b,
        "eagle" | "hawk" | "falcon" => BirdKind { len: 70.0, color: "#6b4423", head: WHITE, beak: "#f5c842", ..b },
        "owl" => BirdKind { len: 45.0, upright: true, color: "#8b5a2b", head: "#8b5a2b", beak: "#f5a623", ..b },
        "parrot" => BirdKind { len: 55.0, upright: true, color: "#d63a2f", head: "#4caf50", beak: "#f4ecd6", ..b },
        "penguin" => BirdKind { len: 60.0, upright: true, leg: 4.0, color: BLACK, head: BLACK, beak: "#f28c28", ..b },
        "chicken" | "hen" | "rooster" => BirdKind { len: 55.0, leg: 16.0, color: WHITE, head: WHITE, beak: "#f5a623", ..b },
        "swan" | "goose" => BirdKind { len: 70.0, neck: 40.0, leg: 4.0, color: WHITE, head: WHITE, beak: "#f28c28", ..b },
        "flamingo" => BirdKind { len: 55.0, neck: 45.0, leg: 70.0, color: "#f48fb1", head: "#f48fb1", beak: BLACK, ..b },
        _ => return None,
    })
}

fn bird(b: &BirdKind, r: &mut Rng) -> Parts {
    let l = r.v(b.len);
    let bz = b.leg + l * 0.35;
    let (body, head_c) = if b.upright {
        (ellipsoid(l * 0.7, l * 0.62, l).at(0.0, 0.0, b.leg + l * 0.5), [l * 0.08, 0.0, b.leg + l * 1.0])
    } else {
        (ellipsoid(l, l * 0.55, l * 0.6).at(0.0, 0.0, bz), [l * 0.42, 0.0, bz + l * 0.3 + b.neck])
    };
    let hr = l * if b.upright { 0.3 } else { 0.24 };
    let mut bits = vec![body, ellipsoid(l * 0.6, l * 0.12, l * 0.36).rot(0.0, -10.0, 0.0).at(-l * 0.05, l * 0.28, if b.upright { b.leg + l * 0.55 } else { bz + l * 0.05 }).mirror(Y)];
    if b.neck > 0.0 {
        bits.push(capsule([l * 0.35, 0.0, bz + l * 0.1], [head_c[0], 0.0, head_c[2] - hr * 0.5], l * 0.1));
    }
    if !b.upright {
        bits.push(extrude(&[(0.0, -l * 0.18), (0.0, l * 0.18), (-l * 0.35, 0.0)], l * 0.06).rot(0.0, -15.0, 0.0).at(-l * 0.48, 0.0, bz + l * 0.05));
    }
    let mut parts = vec![p("Body", b.color, union(l * 0.08, bits)).finish(0.0, 0.7)];
    parts.push(p("Head", b.head, sphere(hr).at(head_c[0], 0.0, head_c[2])).finish(0.0, 0.7));
    parts.push(p("Beak", b.beak, cone(hr * 0.35, 0.0, hr * 0.9).rot(0.0, 90.0, 0.0).at(head_c[0] + hr * 1.15, 0.0, head_c[2] - hr * 0.1)).finish(0.0, 0.4));
    if b.leg > 5.0 {
        parts.push(p("Legs", "#f5a623", capsule([0.0, l * 0.12, bz - l * 0.2], [0.0, l * 0.12, 1.0], (l * 0.035).max(1.2)).mirror(Y)));
    }
    parts.push(side_eyes(hr * 0.16, head_c[0] + hr * 0.75, hr * 0.5, head_c[2] + hr * 0.2));
    parts
}

/// Archetype keywords for nouns without a template (as in Dream Builder's M5 shape grammar).
const FAMILIES: &[(&str, &[&str])] = &[
    ("animal", &["otter", "badger", "beaver", "wombat", "koala", "kangaroo", "hippo", "rhino", "squirrel", "hedgehog", "raccoon", "skunk", "ferret", "mole", "puma", "jaguar", "lynx", "buffalo", "bison", "yak", "antelope", "gazelle", "meerkat", "weasel", "chipmunk", "sloth", "tapir", "armadillo", "possum", "opossum", "mammal", "critter"]),
    ("bird", &["crow", "raven", "seagull", "gull", "pelican", "stork", "heron", "toucan", "peacock", "turkey", "ostrich", "emu", "kiwi", "hummingbird", "finch", "canary", "magpie", "woodpecker", "vulture", "phoenix", "griffin"]),
    ("car", &["vehicle", "van", "jeep", "taxi", "cab", "wagon", "cart", "ambulance", "racecar", "suv", "limo", "minivan", "buggy", "kart"]),
    ("house", &["building", "home", "cabin", "cottage", "hut", "shed", "barn", "church", "palace", "school", "shop", "store", "hotel", "temple", "mansion", "villa", "doghouse", "garage"]),
    ("tree", &["bush", "shrub", "oak", "maple", "birch", "willow", "palm", "hedge", "plant", "sapling"]),
    ("flower", &["rose", "tulip", "daisy", "sunflower", "lily", "orchid", "poppy", "daffodil", "lotus", "blossom", "bloom"]),
    ("jar", &["container", "can", "tin", "canister", "pitcher", "jug", "urn", "flask", "thermos", "kettle"]),
    ("chest", &["crate", "trunk", "case", "suitcase", "toolbox", "package", "parcel", "present", "gift", "basket"]),
    ("sphere", &["globe", "marble", "pearl", "bead", "bubble", "orange", "lemon", "lime", "peach", "plum", "cherry", "grape", "tomato", "onion", "melon", "berry", "moon", "sun", "earth"]),
    ("sword", &["knife", "dagger", "blade", "axe", "spear", "lance", "wand", "staff", "stick", "pole", "rod", "cane", "broom", "bat", "club", "baton", "scepter", "sceptre", "pencil", "pen", "paddle", "oar", "flute"]),
    ("robot", &["android", "cyborg", "mech", "droid", "machine", "golem", "statue", "figure", "doll", "puppet"]),
    ("boat", &["canoe", "kayak", "yacht", "raft", "ferry", "submarine", "sub", "dinghy", "gondola"]),
    ("plane", &["jet", "aircraft", "glider", "helicopter", "drone", "spaceship", "spacecraft", "shuttle"]),
    ("cupcake", &["muffin", "pastry", "dessert", "cookie", "pie", "tart", "pudding", "sweet", "candy", "treat"]),
    ("tower", &["lighthouse", "skyscraper", "windmill", "chimney", "column", "pillar", "obelisk", "monument", "spire"]),
];

pub fn family(word: &str, r: &mut Rng) -> Option<(Parts, f32, &'static str)> {
    let fam = FAMILIES.iter().find(|(_, ws)| ws.contains(&word)).map(|(f, _)| *f)?;
    let (parts, real) = if fam == "animal" {
        let s = Species {
            len: r.range(55.0, 110.0),
            girth: r.range(28.0, 55.0),
            leg: r.range(14.0, 50.0),
            head: r.range(14.0, 24.0),
            snout: r.range(0.0, 20.0),
            neck: r.range(4.0, 30.0),
            ear: [Ear::Pointy, Ear::Round, Ear::Floppy][(r.next() % 3) as usize],
            tail: r.range(4.0, 50.0),
            tail_up: r.range(-0.8, 0.8),
            color: r.pick(&["#8b5a2b", "#b98250", "#9e9e9e", "#d9a441", "#6b4423", "#e8dcc0"]),
            ..Q
        };
        (quadruped(&s, r), 12.0)
    } else if fam == "bird" {
        let b = BirdKind { len: r.range(40.0, 70.0), upright: r.unit() < 0.3, neck: 0.0, leg: r.range(6.0, 18.0), color: r.pick(&["#2f6fd6", "#1e1e20", "#8b5a2b", WHITE, "#d63a2f"]), head: r.pick(&["#2f6fd6", "#1e1e20", WHITE]), beak: "#f5a623" };
        (bird(&b, r), 3.0)
    } else {
        build(fam, r)?
    };
    Some((parts, real, fam))
}

/// A seeded clay lump: something to sculpt from when the words mean nothing to the planner.
pub fn blob(r: &mut Rng) -> Parts {
    let mut v = vec![sphere(r.range(26.0, 34.0)).at(0.0, 0.0, 30.0)];
    for _ in 0..4 {
        let a = r.range(0.0, std::f32::consts::TAU);
        let d = r.range(12.0, 24.0);
        v.push(sphere(r.range(12.0, 22.0)).at(d * a.cos(), d * a.sin(), r.range(20.0, 46.0)));
    }
    vec![p("Clay", "#e9c9a8", union(12.0, v)).finish(0.0, 0.6)]
}
