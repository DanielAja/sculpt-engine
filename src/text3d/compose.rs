//! Generalisation layer for the built-in planner: more nouns, compound words, feature modifiers and an
//! expressive fallback, all offline, deterministic and made of the same SDF/CSG primitives as `templates.rs`.
//!
//! ```text
//! "cybernetic shark"  ──▶ resolve("shark") + attach_modifiers(robotic)   -> a shark with a visor and antenna
//! "hoverboard"        ──▶ alias table                                     -> a board with glowing thrusters
//! "firetruck"         ──▶ compound table / suffix root ("truck")          -> the truck template
//! "zorbak"            ──▶ fallback_sculpture(seeded by the word)          -> an original multi-feature sculpture
//! ```
//!
//! Same conventions as `templates.rs`: millimetres, Z up, standing on z = 0, centred on the Z axis, fronts face -Y,
//! animals and vehicles face +X. Features are kept chunky (at least ~3 % of the object) so they survive the
//! mesher at Dream Builder's 8,000 triangle budget and stay closed and manifold.

use super::program::{Axis::*, *};
use super::sdf::{summit, tight_aabb, Aabb, Compiled};
use super::templates;
use super::text::Rng;
use glam::Vec3;

type Parts = Vec<Part>;

const BLACK: &str = "#1e1e20";
const WHITE: &str = "#f4f4f2";
const WOOD: &str = "#a0703f";
const DARK_WOOD: &str = "#6b4423";
const METAL: &str = "#a9adb3";
const DARK_METAL: &str = "#4a4d55";
const GOLD: &str = "#e6b84d";
const LEAF: &str = "#4caf50";
const RED: &str = "#d63a2f";
const BLUE: &str = "#2f6fd6";
const CYAN: &str = "#33d6ff";
const BONE: &str = "#efe6d2";

fn p(name: &str, color: &str, n: Node) -> Part {
    Part::new(name, color, n)
}
fn metal(name: &str, color: &str, n: Node) -> Part {
    Part::new(name, color, n).finish(1.0, 0.3)
}
fn glassy(name: &str, color: &str, n: Node) -> Part {
    Part::new(name, color, n).finish(0.0, 0.05)
}
fn glow(name: &str, color: &str, n: Node) -> Part {
    Part::new(name, color, n).finish(0.0, 0.15)
}
fn four(n: Node) -> Node {
    union(0.0, vec![n.mirror(X)]).mirror(Y)
}
fn cap(a: [f32; 3], b: [f32; 3], r: f32) -> Node {
    capsule(a, b, r)
}
/// Capsules between successive points with one radius per segment.
fn chain(pts: &[[f32; 3]], radii: &[f32]) -> Vec<Node> {
    pts.windows(2).zip(radii).map(|(w, &r)| capsule(w[0], w[1], r)).collect()
}
/// A flat plate in the XZ plane (outline is [x, z]), `t` thick along Y.
fn xz(outline: &[(f32, f32)], t: f32) -> Node {
    extrude(outline, t).rot(90.0, 0.0, 0.0)
}
/// Rotates and moves a node as a unit around its own origin.
fn tilt(n: Node, rx: f32, ry: f32, rz: f32, at: [f32; 3]) -> Node {
    union(0.0, vec![n]).rot(rx, ry, rz).at(at[0], at[1], at[2])
}
fn ring_of(n: usize, radius: f32, z: f32, f: impl Fn(f32) -> Node) -> Vec<Node> {
    (0..n)
        .map(|i| {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            f(a.to_degrees()).at(radius * a.cos(), radius * a.sin(), z)
        })
        .collect()
}
/// Radius of a lathe profile at height `z`.
fn lathe_r(profile: &[(f32, f32)], z: f32) -> f32 {
    for w in profile.windows(2) {
        let ((r0, z0), (r1, z1)) = (w[0], w[1]);
        if (z0 <= z && z <= z1) || (z1 <= z && z <= z0) {
            let t = if (z1 - z0).abs() < 1e-4 { 0.0 } else { (z - z0) / (z1 - z0) };
            return r0 + (r1 - r0) * t;
        }
    }
    0.0
}

// ───────────────────────────── names, aliases and compound words ─────────────────────────────

/// canonical name, every word that means it. Canonical names are either built here or are `templates.rs` names.
const ALIASES: &[(&str, &[&str])] = &[
    ("guitar", &["guitar", "ukulele", "banjo", "mandolin", "lute", "bass guitar", "electric guitar", "acoustic guitar"]),
    ("piano", &["piano", "grand piano", "synthesizer", "synth", "keytar", "organ", "harpsichord"]),
    ("violin", &["violin", "fiddle", "viola", "cello"]),
    ("drum", &["drum", "snare", "snare drum", "bongo", "tambourine", "drums"]),
    ("microphone", &["microphone", "mic", "mike", "karaoke"]),
    ("headset", &["headset", "headphone", "headphones", "earphone", "earphones", "earmuffs", "earbuds"]),
    ("laptop", &["laptop", "computer", "notebook", "macbook", "pc", "netbook", "chromebook"]),
    ("camera", &["camera", "dslr", "camcorder", "webcam"]),
    ("speaker", &["speaker", "loudspeaker", "boombox", "subwoofer", "amp", "amplifier", "jukebox"]),
    ("controller", &["controller", "gamepad", "joypad", "console", "joystick"]),
    ("watch", &["watch", "wristwatch", "smartwatch"]),
    ("bicycle", &["bicycle", "bike", "cycle", "bmx", "tricycle", "unicycle"]),
    ("motorcycle", &["motorcycle", "motorbike", "scooter", "moped", "chopper", "harley"]),
    ("helicopter", &["helicopter", "chopper heli", "copter", "helicopters", "gyrocopter"]),
    ("submarine", &["submarine", "sub", "u-boat", "uboat", "submersible", "bathyscaphe"]),
    ("train", &["train", "locomotive", "engine", "steam engine", "tram", "trolley", "railway"]),
    ("skateboard", &["skateboard", "longboard", "skate", "penny board"]),
    ("hoverboard", &["hoverboard", "hover board", "airboard", "jetboard"]),
    ("snowboard", &["snowboard", "snow board", "ski", "skis", "surfboard", "surf board", "wakeboard"]),
    ("ufo", &["saucer", "flying saucer", "alien ship", "mothership", "alien craft"]),
    ("spaceship", &["spaceship", "space ship", "starship", "star ship", "spacecraft", "space craft", "shuttle", "space shuttle", "starfighter", "starcruiser", "spaceplane", "interceptor"]),
    ("whale", &["whale", "orca", "humpback", "dolphin", "porpoise", "narwhal", "beluga"]),
    ("shark", &["shark", "megalodon", "hammerhead", "jaws"]),
    ("octopus", &["octopus", "squid", "kraken", "cuttlefish", "tentacle monster", "cthulhu"]),
    ("butterfly", &["butterfly", "moth", "dragonfly", "ladybug", "ladybird", "firefly", "bee", "wasp", "bug", "insect", "fairy"]),
    ("spider", &["spider", "tarantula", "arachnid", "crab", "scorpion", "ant", "beetle"]),
    ("trex", &["trex", "rex", "tyrannosaurus", "tyrannosaur", "raptor", "velociraptor", "allosaurus"]),
    ("dinosaur", &["triceratops", "stegosaurus", "brontosaurus", "brachiosaurus", "diplodocus", "ankylosaurus", "pterodactyl"]),
    ("jellyfish", &["jellyfish", "jelly", "medusa", "sea jelly", "anemone", "squid jelly"]),
    ("pizza", &["pizza", "flatbread", "calzone", "quiche"]),
    ("taco", &["taco", "burrito", "tortilla", "quesadilla", "fajita", "nacho"]),
    ("hot dog", &["hot dog", "hotdog", "sausage", "frankfurter", "bratwurst", "wiener"]),
    ("sushi", &["sushi", "nigiri", "maki", "sashimi", "onigiri", "dumpling"]),
    ("croissant", &["croissant", "crescent", "baguette", "bagel", "pretzel", "bread", "loaf", "pastry roll"]),
    ("cookie", &["cookie", "biscuit", "cracker", "brownie", "macaron"]),
    ("watermelon", &["watermelon", "water melon", "melon slice"]),
    ("strawberry", &["strawberry", "raspberry", "blackberry", "mulberry"]),
    ("pineapple", &["pineapple", "ananas"]),
    ("axe", &["axe", "hatchet", "tomahawk", "cleaver", "pickaxe", "ax"]),
    ("battleaxe", &["battleaxe", "battle axe", "double axe", "labrys"]),
    ("shield", &["shield", "buckler", "aegis"]),
    ("bow", &["bow", "longbow", "archery", "crossbow", "bow and arrow"]),
    ("staff", &["staff", "scepter", "sceptre", "quarterstaff", "crook", "caduceus"]),
    ("wand", &["wand", "magic wand", "magicwand"]),
    ("helmet", &["helmet", "helm", "knight helmet", "armet", "visor", "hardhat", "hard hat"]),
    ("potion", &["potion", "elixir", "flask potion", "philter", "tonic", "serum", "vial"]),
    ("lightsaber", &["lightsaber", "light saber", "saber", "laser sword", "energy sword", "beam sword"]),
    ("blaster", &["blaster", "raygun", "ray gun", "laser gun", "phaser", "laser pistol", "zapper"]),
    ("drone", &["drone", "quadcopter", "multicopter", "uav", "quadrotor"]),
    ("telescope", &["telescope", "spyglass", "observatory scope", "binoculars"]),
    ("hourglass", &["hourglass", "sandglass", "egg timer", "sand timer"]),
    ("anchor", &["anchor", "grapnel"]),
    ("compass", &["compass", "sundial", "astrolabe"]),
    ("skull", &["skull", "skeleton head", "cranium", "skeleton", "bones"]),
    ("shoe", &["shoe", "sneaker", "sneakers", "trainer", "trainers", "boot", "sho", "loafer", "sandal", "slipper", "footwear", "heel"]),
    ("backpack", &["backpack", "rucksack", "knapsack", "schoolbag", "satchel", "bag", "handbag", "purse"]),
    ("shades", &["shades", "sunglass", "sunglasses", "eyeglass", "eyeglasses", "goggles", "spectacles", "eyewear"]),
    ("lighthouse", &["lighthouse", "light house", "beacon"]),
    ("windmill", &["windmill", "wind mill", "turbine", "wind turbine"]),
    ("bridge", &["bridge", "viaduct", "aqueduct", "overpass"]),
    ("volcano", &["volcano", "eruption", "geyser"]),
    ("palm tree", &["palm tree", "palmtree", "coconut tree", "palm", "coconut palm"]),
    ("bonsai", &["bonsai", "topiary", "houseplant", "potted plant", "zen tree"]),
    ("throne", &["throne", "royal chair", "king chair", "kings chair"]),
    ("armchair", &["armchair", "arm chair", "recliner", "loveseat", "easy chair", "lounge chair"]),
    ("toaster", &["toaster", "microwave", "oven", "kitchen appliance", "blender"]),
    ("coffee maker", &["coffee maker", "coffeemaker", "coffee machine", "espresso machine", "espresso", "kettle machine"]),
    ("clock", &["clock", "alarm clock", "alarmclock", "wall clock", "timepiece", "stopwatch"]),
    ("tree house", &["tree house", "treehouse"]),
    ("skyscraper", &["skyscraper", "sky scraper", "high rise", "highrise", "office tower", "apartment tower", "city tower"]),
    ("sunflower", &["sunflower", "sun flower"]),
    ("race car", &["race car", "racecar", "racing car", "sports car", "sportscar", "formula one", "f1 car", "supercar"]),
    ("water bottle", &["water bottle", "waterbottle", "sports bottle", "canteen", "drink bottle", "gym bottle"]),
];

/// Roots for compounds whose pieces are not simply "something + noun".
const COMPOUNDS: &[(&str, &str)] = &[
    ("firetruck", "truck"), ("pickup", "truck"), ("seahorse", "horse"), ("bookcase", "bookshelf"), ("houseboat", "boat"),
    ("rowboat", "boat"), ("speedboat", "boat"), ("lifeboat", "boat"), ("warship", "ship"), ("airship", "ship"),
    ("catfish", "fish"), ("goldfish", "fish"), ("swordfish", "fish"), ("starfish", "fish"), ("snowflake", "star"),
    ("pancake", "cake"), ("cheesecake", "cake"), ("shortcake", "cake"), ("fruitcake", "cake"), ("moonrock", "rock"),
    ("windowsill", "shelf"), ("workbench", "bench"), ("sandcastle", "castle"), ("rainbowcloud", "cloud"),
    ("teacup", "cup"), ("eggcup", "cup"), ("butterfly", "butterfly"), ("bathtub", "bucket"), ("dollhouse", "house"),
    ("pigeon", "bird"), ("hummingbird", "bird"), ("eaglet", "eagle"), ("kitty", "cat"), ("doggo", "dog"),
];

fn alias(w: &str) -> Option<&'static str> {
    let found = ALIASES.iter().find(|(_, a)| a.contains(&w)).map(|(c, _)| *c);
    // Plural-only aliases ("goggles", "bones") also answer to their singular.
    found.or_else(|| ALIASES.iter().find(|(_, a)| a.iter().any(|x| x.len() > 3 && x.strip_suffix('s') == Some(w))).map(|(c, _)| *c))
}

/// A word the planner can build directly: an alias here or a `templates.rs` noun.
fn known_root(w: &str) -> Option<String> {
    if let Some(c) = alias(w) {
        return Some(c.to_string());
    }
    templates::known(w).then(|| w.to_string())
}

const SHORT_ROOTS: &[&str] = &["cup", "dog", "cat", "pot", "hat", "bed", "bus", "car", "key", "pig", "cow", "box", "jar"];

/// Maps any word to something the planner can build: aliases, compound words ("firetruck" -> truck,
/// "cheesecake" -> cake) and stems ("houseboat"), or None for words that mean nothing to it.
pub fn resolve(word: &str) -> Option<String> {
    if let Some(c) = alias(word) {
        return Some(c.to_string());
    }
    if templates::known(word) {
        return Some(word.to_string());
    }
    if let Some((_, root)) = COMPOUNDS.iter().find(|(c, _)| *c == word) {
        return known_root(root);
    }
    if !word.is_ascii() || word.len() < 6 {
        return None;
    }
    // Words with a rough archetype of their own (wombat, cottage...) keep it.
    if templates::family(word, &mut Rng::new(word, 0)).is_some() {
        return None;
    }
    let n = word.len();
    // The head of an English compound is its last piece: "bluebird", "teacup", "lighthouse", "pancake".
    for k in (3..=n - 3).rev() {
        let tail = &word[n - k..];
        if k == 3 && !SHORT_ROOTS.contains(&tail) {
            continue;
        }
        if let Some(c) = known_root(tail) {
            return Some(c);
        }
    }
    // Stems at the front: "elephantine", "sharknado".
    for k in (5..=n - 3).rev() {
        if let Some(c) = known_root(&word[..k]) {
            return Some(c);
        }
    }
    None
}

/// Two-word names ("palm tree", "coffee maker") and run-together pairs ("space" + "ship").
pub fn resolve_phrase(first: &str, second: &str) -> Option<String> {
    let two = format!("{first} {second}");
    if let Some(c) = alias(&two) {
        return Some(c.to_string());
    }
    let joined = format!("{first}{second}");
    if let Some(c) = alias(&joined) {
        return Some(c.to_string());
    }
    COMPOUNDS.iter().find(|(c, _)| *c == joined).and_then(|(_, r)| known_root(r))
}

pub fn known(name: &str) -> bool {
    resolve(name).is_some()
}

/// Every canonical name this module builds itself (used by tests).
pub fn own_names() -> Vec<&'static str> {
    ALIASES.iter().map(|(c, _)| *c).filter(|c| build_own(c, &mut Rng::new(c, 0)).is_some()).collect()
}

/// Builds `name` (any alias or compound) from a template here or in `templates.rs`.
pub fn build_generalized(name: &str, r: &mut Rng) -> Option<(Parts, f32)> {
    let canon = resolve(name)?;
    build_own(&canon, r).or_else(|| templates::build(&canon, r))
}

// ───────────────────────────── object templates ─────────────────────────────

/// Parts with a top-level mirror are wrapped in a plain union so layout can move them: a mirror reflects across
/// the parent's origin plane, so shifting `at` on a mirrored node would move the reflection the other way.
fn build_own(name: &str, r: &mut Rng) -> Option<(Parts, f32)> {
    let (mut parts, real) = build_raw(name, r)?;
    for p in parts.iter_mut() {
        if p.shape.mirror.is_some() {
            let inner = std::mem::replace(&mut p.shape, sphere(1.0));
            p.shape = union(0.0, vec![inner]);
        }
    }
    Some((parts, real))
}

fn build_raw(name: &str, r: &mut Rng) -> Option<(Parts, f32)> {
    Some(match name {
        "guitar" => (stringed(r, false), 3.0),
        "violin" => (stringed(r, true), 3.0),
        "piano" => {
            let c = r.pick(&["#2a2a2e", "#6b4423", "#f4f4f2"]);
            (vec![
                p("Cabinet", c, union(0.0, vec![rbox(150.0, 50.0, 104.0, 4.0).at(0.0, 10.0, 66.0), rbox(150.0, 30.0, 8.0, 2.0).at(0.0, -22.0, 66.0)])).finish(0.0, 0.25),
                p("White Keys", "#f7f4ea", cuboid(138.0, 24.0, 6.0).at(0.0, -32.0, 72.0)).finish(0.0, 0.3),
                p("Black Keys", BLACK, array(cuboid(6.0, 14.0, 6.0).at(-60.0, -26.0, 77.0), 7, [20.0, 0.0, 0.0])).finish(0.0, 0.3),
                p("Legs", DARK_WOOD, union(0.0, vec![rbox(14.0, 14.0, 20.0, 2.0).at(66.0, 0.0, 10.0).mirror(X), rbox(14.0, 14.0, 20.0, 2.0).at(66.0, 24.0, 10.0).mirror(X)])).finish(0.0, 0.5),
                metal("Pedals", GOLD, union(0.0, vec![cuboid(12.0, 14.0, 4.0).at(0.0, -22.0, 8.0), cuboid(12.0, 14.0, 4.0).at(16.0, -22.0, 8.0), cuboid(12.0, 14.0, 4.0).at(-16.0, -22.0, 8.0)])),
                p("Bench", DARK_WOOD, union(0.0, vec![rbox(66.0, 26.0, 8.0, 3.0).at(0.0, -72.0, 38.0), cuboid(8.0, 8.0, 36.0).at(26.0, -72.0, 18.0).mirror(X)])).finish(0.0, 0.5),
            ], 12.0)
        }
        "drum" => {
            let c = r.pick(&[RED, BLUE, "#2a2a2e", "#f5c842"]);
            (vec![
                p("Shell", c, rcylinder(50.0, 54.0, 3.0).at(0.0, 0.0, 44.0)).finish(0.2, 0.3),
                p("Heads", WHITE, union(0.0, vec![cylinder(47.0, 4.0).at(0.0, 0.0, 72.0), cylinder(47.0, 4.0).at(0.0, 0.0, 17.0)])).finish(0.0, 0.7),
                metal("Rims", "#c9ccd1", union(0.0, vec![torus(50.0, 3.8).at(0.0, 0.0, 71.0), torus(50.0, 3.8).at(0.0, 0.0, 17.0)])),
                metal("Lugs", METAL, cuboid(7.0, 7.0, 40.0).at(52.0, 0.0, 44.0).mirror(X)),
                p("Sticks", WOOD, union(0.0, vec![cap([-62.0, -22.0, 82.0], [60.0, 28.0, 82.0], 4.2), cap([-60.0, 30.0, 82.0], [62.0, -20.0, 82.0], 4.2)])).finish(0.0, 0.6),
            ], 3.0)
        }
        "microphone" => (vec![
            metal("Stand Base", DARK_METAL, cylinder(46.0, 8.0).at(0.0, 0.0, 4.0)),
            metal("Stand Pole", DARK_METAL, cap([0.0, 0.0, 8.0], [0.0, 0.0, 122.0], 4.5)),
            metal("Mic Body", "#3a3a3d", tilt(cone(9.0, 15.0, 70.0).at(0.0, 0.0, 35.0), 18.0, 0.0, 0.0, [0.0, 0.0, 122.0])),
            metal("Mic Head", "#c9ccd1", tilt(sphere(21.0).at(0.0, 0.0, 86.0), 18.0, 0.0, 0.0, [0.0, 0.0, 122.0])),
            p("Mic Ring", RED, tilt(torus(17.0, 4.0).at(0.0, 0.0, 70.0), 18.0, 0.0, 0.0, [0.0, 0.0, 122.0])).finish(0.0, 0.4),
        ], 1.5),
        "headset" => {
            let c = r.pick(&[BLACK, WHITE, RED, BLUE, "#8e44ad"]);
            (vec![
                p("Band", c, intersection(0.0, vec![torus(60.0, 6.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 60.0), cuboid(260.0, 60.0, 100.0).at(0.0, 0.0, 110.0)])).finish(0.0, 0.5),
                p("Ear Cups", c, rcylinder(30.0, 26.0, 7.0).rot(0.0, 90.0, 0.0).at(60.0, 0.0, 60.0).mirror(X)).finish(0.1, 0.4),
                p("Cushions", "#3a3a3d", torus(23.0, 8.0).rot(0.0, 90.0, 0.0).at(44.0, 0.0, 60.0).mirror(X)).finish(0.0, 0.9),
            ], 1.5)
        }
        "laptop" => {
            (vec![
                metal("Base", "#b8bcc4", rbox(150.0, 100.0, 9.0, 3.0).at(0.0, 0.0, 4.5)),
                p("Keys", "#2a2a2e", array(array(cuboid(9.0, 9.0, 3.0).at(-55.0, -4.0, 10.0), 12, [10.0, 0.0, 0.0]), 4, [0.0, 10.0, 0.0])).finish(0.0, 0.5),
                p("Trackpad", "#9aa0a8", cuboid(44.0, 30.0, 2.0).at(0.0, -32.0, 9.5)).finish(0.5, 0.3),
                metal("Lid", "#b8bcc4", rbox(150.0, 6.0, 100.0, 2.5).rot(-15.0, 0.0, 0.0).at(0.0, 63.0, 57.0)),
                glassy("Screen", "#1c2a4a", cuboid(136.0, 3.0, 86.0).rot(-15.0, 0.0, 0.0).at(0.0, 61.0, 57.6)),
            ], 3.0)
        }
        "camera" => (vec![
            p("Body", "#2b2b30", rbox(130.0, 58.0, 82.0, 10.0).at(0.0, 0.0, 41.0)).finish(0.3, 0.45),
            metal("Top Plate", "#c9ccd1", union(0.0, vec![rbox(130.0, 58.0, 10.0, 5.0).at(0.0, 0.0, 80.0), rbox(46.0, 36.0, 20.0, 6.0).at(0.0, 0.0, 92.0)])),
            p("Lens Barrel", "#17171a", cylinder(32.0, 38.0).rot(90.0, 0.0, 0.0).at(8.0, -43.0, 46.0)).finish(0.4, 0.4),
            metal("Lens Ring", "#c9ccd1", torus(30.0, 4.0).rot(90.0, 0.0, 0.0).at(8.0, -60.0, 46.0)),
            glassy("Lens Glass", "#243a66", cylinder(24.0, 6.0).rot(90.0, 0.0, 0.0).at(8.0, -61.0, 46.0)),
            p("Buttons", RED, union(0.0, vec![cylinder(8.0, 7.0).at(46.0, -6.0, 88.0), cylinder(9.0, 7.0).at(-46.0, -6.0, 86.0)])).finish(0.0, 0.4),
            p("Grip", "#17171a", rbox(26.0, 30.0, 70.0, 8.0).at(-48.0, -30.0, 40.0)).finish(0.0, 0.9),
        ], 2.5),
        "speaker" => {
            let c = r.pick(&["#2a2a2e", "#6b4423", "#f4f4f2"]);
            (vec![
                p("Cabinet", c, rbox(86.0, 76.0, 140.0, 8.0).at(0.0, 0.0, 70.0)).finish(0.0, 0.6),
                p("Woofer", "#3a3a3d", cylinder(31.0, 10.0).rot(90.0, 0.0, 0.0).at(0.0, -38.0, 50.0)).finish(0.0, 0.8),
                metal("Woofer Ring", "#c9ccd1", torus(31.0, 4.5).rot(90.0, 0.0, 0.0).at(0.0, -40.0, 50.0)),
                p("Woofer Cone", "#1e1e20", sphere(15.0).at(0.0, -40.0, 50.0)).finish(0.0, 0.5),
                p("Tweeter", "#3a3a3d", cylinder(15.0, 8.0).rot(90.0, 0.0, 0.0).at(0.0, -38.0, 110.0)).finish(0.0, 0.8),
                metal("Tweeter Ring", GOLD, torus(15.0, 3.6).rot(90.0, 0.0, 0.0).at(0.0, -40.0, 110.0)),
            ], 3.0)
        }
        "controller" => {
            let c = r.pick(&[WHITE, "#2a2a2e", "#2f6fd6"]);
            (vec![
                p("Body", c, union(10.0, vec![ellipsoid(100.0, 58.0, 26.0).at(0.0, 0.0, 16.0), ellipsoid(46.0, 70.0, 32.0).at(46.0, -14.0, 16.0).mirror(X)])).finish(0.0, 0.45),
                p("D-Pad", "#2a2a2e", union(0.0, vec![cuboid(9.0, 28.0, 7.0), cuboid(28.0, 9.0, 7.0)]).at(-34.0, 8.0, 29.0)),
                p("Buttons", r.pick(&[RED, "#4caf50", "#f5c842"]), union(0.0, vec![sphere(6.5).at(36.0, 18.0, 28.0), sphere(6.5).at(48.0, 6.0, 28.0), sphere(6.5).at(24.0, 6.0, 28.0), sphere(6.5).at(36.0, -6.0, 28.0)])).finish(0.0, 0.4),
                p("Sticks", "#2a2a2e", union(0.0, vec![ellipsoid(20.0, 20.0, 14.0).at(-18.0, -16.0, 31.0), ellipsoid(20.0, 20.0, 14.0).at(18.0, -16.0, 31.0)])).finish(0.0, 0.8),
            ], 3.0)
        }
        "watch" => {
            let c = r.pick(&[BLACK, "#8b5a2b", RED, BLUE]);
            (vec![
                p("Strap", c, difference(0.0, vec![cylinder(40.0, 24.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 40.0), cylinder(33.0, 30.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 40.0)])).finish(0.0, 0.8),
                metal("Case", GOLD, rcylinder(27.0, 12.0, 3.0).at(0.0, 0.0, 83.0)),
                p("Dial", WHITE, cylinder(22.0, 4.0).at(0.0, 0.0, 89.0)).finish(0.0, 0.3),
                p("Hands", BLACK, union(0.0, vec![cuboid(4.5, 20.0, 3.0).at(0.0, 8.0, 92.0), cuboid(17.0, 4.5, 3.0).rot(0.0, 0.0, 30.0).at(5.0, 3.0, 92.5)])),
                metal("Crown", GOLD, cylinder(5.0, 10.0).rot(0.0, 90.0, 0.0).at(30.0, 0.0, 83.0)),
            ], 1.0)
        }
        "bicycle" => {
            let c = r.pick(&[RED, BLUE, "#4caf50", "#f5c842", BLACK]);
            let bb = [0.0, 0.0, 36.0];
            let (rear, front) = ([-66.0, 0.0, 58.0], [66.0, 0.0, 58.0]);
            let (seat, s2, ht, hb) = ([-16.0, 0.0, 108.0], [-14.0, 0.0, 100.0], [46.0, 0.0, 104.0], [52.0, 0.0, 84.0]);
            let mut frame = chain(&[bb, seat], &[4.2]);
            frame.extend(chain(&[s2, ht], &[3.8]));
            frame.extend(chain(&[bb, hb], &[4.5]));
            frame.extend(chain(&[bb, rear], &[3.4]));
            frame.extend(chain(&[s2, rear], &[3.2]));
            frame.extend(chain(&[hb, ht], &[5.0]));
            (vec![
                p("Tires", BLACK, union(0.0, vec![torus(52.0, 6.5).rot(90.0, 0.0, 0.0).at(-66.0, 0.0, 58.5), torus(52.0, 6.5).rot(90.0, 0.0, 0.0).at(66.0, 0.0, 58.5)])).finish(0.0, 0.9),
                metal("Rims", "#c9ccd1", union(0.0, vec![torus(45.0, 2.8).rot(90.0, 0.0, 0.0).at(-66.0, 0.0, 58.5), torus(45.0, 2.8).rot(90.0, 0.0, 0.0).at(66.0, 0.0, 58.5), cylinder(7.0, 12.0).rot(90.0, 0.0, 0.0).at(-66.0, 0.0, 58.5), cylinder(7.0, 12.0).rot(90.0, 0.0, 0.0).at(66.0, 0.0, 58.5)])),
                metal("Frame", c, union(2.0, frame)),
                metal("Fork", c, union(0.0, vec![cap(hb, front, 3.6)])),
                p("Handlebar", BLACK, union(2.0, vec![cap(ht, [44.0, 0.0, 120.0], 3.4), cap([44.0, -32.0, 122.0], [44.0, 32.0, 122.0], 3.4)])).finish(0.0, 0.7),
                p("Saddle", BLACK, ellipsoid(36.0, 16.0, 9.0).at(-18.0, 0.0, 114.0)).finish(0.0, 0.8),
                metal("Pedals", DARK_METAL, union(0.0, vec![cylinder(14.0, 5.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 36.0), cuboid(22.0, 12.0, 5.0).at(14.0, -22.0, 28.0), cuboid(22.0, 12.0, 5.0).at(-14.0, 22.0, 44.0)])),
            ], 15.0)
        }
        "motorcycle" => {
            let c = r.pick(&[RED, BLACK, BLUE, "#f5c842"]);
            (vec![
                p("Tires", BLACK, union(0.0, vec![torus(34.0, 12.0).rot(90.0, 0.0, 0.0).at(-58.0, 0.0, 46.0), torus(34.0, 12.0).rot(90.0, 0.0, 0.0).at(58.0, 0.0, 46.0)])).finish(0.0, 0.9),
                metal("Hubs", "#c9ccd1", union(0.0, vec![cylinder(17.0, 22.0).rot(90.0, 0.0, 0.0).at(-58.0, 0.0, 46.0), cylinder(17.0, 22.0).rot(90.0, 0.0, 0.0).at(58.0, 0.0, 46.0)])),
                p("Tank", c, ellipsoid(66.0, 38.0, 34.0).at(10.0, 0.0, 86.0)).finish(0.4, 0.3),
                p("Seat", "#2a2a2e", ellipsoid(66.0, 30.0, 14.0).at(-32.0, 0.0, 80.0)).finish(0.0, 0.8),
                metal("Engine", DARK_METAL, union(2.0, vec![cuboid(46.0, 30.0, 34.0).at(2.0, 0.0, 52.0), cylinder(11.0, 26.0).rot(0.0, 20.0, 0.0).at(-4.0, 0.0, 76.0)])),
                metal("Fork", METAL, union(0.0, vec![cap([34.0, 11.0, 102.0], [58.0, 11.0, 46.0], 4.5).mirror(Y)])),
                metal("Swingarm", DARK_METAL, union(0.0, vec![cap([-58.0, 11.0, 46.0], [-26.0, 11.0, 66.0], 4.5).mirror(Y)])),
                p("Handlebar", BLACK, cap([30.0, -38.0, 112.0], [30.0, 38.0, 112.0], 3.6)).finish(0.0, 0.6),
                glow("Headlight", "#fff6c2", sphere(14.0).at(46.0, 0.0, 96.0)),
                metal("Exhaust", "#c9ccd1", cap([-2.0, 20.0, 34.0], [-84.0, 20.0, 50.0], 7.0)),
            ], 12.0)
        }
        "helicopter" => {
            let c = r.pick(&[RED, "#2f6fd6", "#f5c842", "#4caf50"]);
            (vec![
                p("Body", c, union(6.0, vec![ellipsoid(112.0, 54.0, 58.0).at(0.0, 0.0, 56.0), cone(15.0, 6.0, 100.0).rot(0.0, -90.0, 0.0).at(-96.0, 0.0, 66.0), xz(&[(-134.0, 66.0), (-152.0, 108.0), (-138.0, 108.0), (-122.0, 66.0)], 7.0)])).finish(0.2, 0.35),
                glassy("Cockpit", "#7cc4f2", ellipsoid(54.0, 46.0, 40.0).at(34.0, 0.0, 62.0)),
                metal("Rotor", DARK_METAL, union(0.0, vec![cylinder(6.0, 16.0).at(0.0, 0.0, 94.0), cuboid(160.0, 11.0, 5.0).at(0.0, 0.0, 103.0), cuboid(11.0, 160.0, 5.0).at(0.0, 0.0, 103.0)])),
                metal("Tail Rotor", DARK_METAL, union(0.0, vec![cuboid(6.0, 5.0, 42.0).at(-146.0, 10.0, 80.0)])),
                metal("Skids", DARK_METAL, union(0.0, vec![cap([-44.0, 28.0, 8.0], [54.0, 28.0, 8.0], 4.2).mirror(Y), cap([-20.0, 24.0, 12.0], [-14.0, 16.0, 36.0], 3.6).mirror(Y), cap([26.0, 24.0, 12.0], [20.0, 16.0, 36.0], 3.6).mirror(Y)])),
            ], 15.0)
        }
        "submarine" => {
            let ports: Vec<Node> = [-34.0f32, 6.0, 46.0].iter().map(|&x| { let y = 28.0 * (1.0 - (x / 95.0) * (x / 95.0)).max(0.0).sqrt(); cylinder(7.5, 6.0).rot(90.0, 0.0, 0.0).at(x, y, 58.0).mirror(Y) }).collect();
            (vec![
                p("Hull", "#f2c230", ellipsoid(190.0, 58.0, 58.0).at(0.0, 0.0, 58.0)).finish(0.3, 0.35),
                p("Tower", "#e0b020", union(3.0, vec![rcylinder(15.0, 28.0, 4.0).at(8.0, 0.0, 98.0), cap([10.0, 0.0, 108.0], [10.0, 0.0, 132.0], 3.4), cap([10.0, 0.0, 132.0], [22.0, 0.0, 132.0], 3.4)])).finish(0.3, 0.35),
                glassy("Portholes", "#7cc4f2", union(0.0, ports)),
                metal("Propeller", DARK_METAL, union(0.0, vec![cuboid(6.0, 12.0, 46.0).at(-104.0, 0.0, 58.0), cuboid(6.0, 46.0, 12.0).at(-104.0, 0.0, 58.0), sphere(8.0).at(-100.0, 0.0, 58.0)])),
                p("Fins", "#c9921a", union(0.0, vec![xz(&[(-76.0, 80.0), (-98.0, 108.0), (-106.0, 108.0), (-96.0, 76.0)], 6.0), extrude(&[(-70.0, 6.0), (-100.0, 36.0), (-108.0, 36.0), (-96.0, 6.0)], 6.0).at(0.0, 0.0, 58.0).mirror(Y)])).finish(0.2, 0.4),
            ], 20.0)
        }
        "train" => {
            let c = r.pick(&[RED, "#2f6fd6", "#2e7d4f", BLACK]);
            (vec![
                p("Boiler", c, union(2.0, vec![rcylinder(26.0, 100.0, 4.0).rot(0.0, 90.0, 0.0).at(30.0, 0.0, 58.0), rbox(48.0, 54.0, 62.0, 3.0).at(-54.0, 0.0, 68.0)])).finish(0.3, 0.4),
                p("Chassis", "#2a2a2e", rbox(160.0, 54.0, 12.0, 3.0).at(0.0, 0.0, 24.0)).finish(0.3, 0.6),
                p("Roof", "#2a2a2e", rbox(56.0, 60.0, 7.0, 2.0).at(-54.0, 0.0, 102.0)).finish(0.0, 0.7),
                p("Smokebox", BLACK, union(0.0, vec![cylinder(26.0, 5.0).rot(0.0, 90.0, 0.0).at(82.0, 0.0, 58.0), cylinder(8.0, 28.0).at(62.0, 0.0, 90.0), cone(8.0, 13.0, 8.0).at(62.0, 0.0, 108.0), sphere(10.0).at(22.0, 0.0, 86.0)])),
                glow("Headlight", "#fff6c2", sphere(6.5).at(86.0, 0.0, 66.0)),
                metal("Wheels", GOLD, array(cylinder(15.0, 10.0).rot(90.0, 0.0, 0.0).at(-56.0, 30.0, 15.0), 4, [38.0, 0.0, 0.0]).mirror(Y)),
                p("Track", "#8d8d88", union(0.0, vec![cuboid(240.0, 12.0, 8.0).at(0.0, 30.0, -4.0).mirror(Y), cuboid(240.0, 84.0, 5.0).at(0.0, 0.0, -10.5)])).finish(0.0, 0.95),
            ], 25.0)
        }
        "skateboard" => (vec![
            p("Deck", r.pick(&[RED, BLUE, "#8e44ad", "#f5c842"]), union(1.5, vec![rbox(150.0, 50.0, 7.0, 3.0).at(0.0, 0.0, 26.0), rbox(34.0, 50.0, 7.0, 3.0).rot(0.0, -22.0, 0.0).at(80.0, 0.0, 31.0).mirror(X)])).finish(0.0, 0.5),
            p("Grip Stripe", BLACK, cuboid(90.0, 14.0, 2.0).at(0.0, 0.0, 30.0)).finish(0.0, 0.95),
            metal("Trucks", METAL, union(0.0, vec![cuboid(12.0, 40.0, 9.0).at(54.0, 0.0, 17.0).mirror(X), cylinder(3.6, 62.0).rot(90.0, 0.0, 0.0).at(54.0, 0.0, 12.0).mirror(X)])),
            p("Wheels", "#f4f4f2", union(0.0, vec![cylinder(12.0, 13.0).rot(90.0, 0.0, 0.0).at(54.0, 28.0, 12.0).mirror(X).mirror(Y)])).finish(0.0, 0.5),
        ], 8.0),
        "hoverboard" => {
            let c = r.pick(&["#f4f4f2", "#2a2a2e", "#8e44ad", "#2f6fd6"]);
            (vec![
                p("Board", c, rbox(180.0, 58.0, 12.0, 6.0).at(0.0, 0.0, 44.0)).finish(0.5, 0.25),
                glow("Neon Trim", CYAN, union(0.0, vec![cuboid(150.0, 8.0, 3.0).at(0.0, 0.0, 51.0), cuboid(8.0, 52.0, 3.0).at(60.0, 0.0, 51.0).mirror(X)])),
                metal("Thrusters", DARK_METAL, union(0.0, vec![cylinder(23.0, 18.0).at(52.0, 0.0, 29.0).mirror(X)])),
                glow("Thruster Glow", CYAN, union(0.0, vec![cylinder(19.0, 4.0).at(52.0, 0.0, 18.0).mirror(X), torus(26.0, 3.5).at(52.0, 0.0, 10.0).mirror(X)])),
            ], 8.0)
        }
        "snowboard" => (vec![
            p("Board", r.pick(&[BLUE, "#8e44ad", "#d63a2f", "#33c3e0"]), union(1.5, vec![rbox(150.0, 56.0, 7.0, 3.0).at(0.0, 0.0, 14.0), rbox(46.0, 56.0, 7.0, 3.0).rot(0.0, -18.0, 0.0).at(88.0, 0.0, 19.0).mirror(X)])).finish(0.0, 0.4),
            p("Bindings", "#2a2a2e", union(0.0, vec![rbox(40.0, 44.0, 9.0, 3.0).at(34.0, 0.0, 22.0).mirror(X), rbox(9.0, 44.0, 28.0, 3.0).at(50.0, 0.0, 34.0).mirror(X)])).finish(0.0, 0.7),
            p("Straps", RED, union(0.0, vec![cuboid(10.0, 50.0, 6.0).at(30.0, 0.0, 30.0).mirror(X)])).finish(0.0, 0.7),
        ], 8.0),
        "spaceship" => {
            let c = r.pick(&[WHITE, "#c9ccd1", "#e8dcc0"]);
            let wing = r.pick(&[RED, BLUE, "#f5a623"]);
            (vec![
                p("Hull", c, union(6.0, vec![ellipsoid(160.0, 50.0, 44.0).at(0.0, 0.0, 36.0), cone(19.0, 0.0, 60.0).rot(0.0, 90.0, 0.0).at(90.0, 0.0, 36.0)])).finish(0.6, 0.3),
                glassy("Cockpit", "#33c3e0", ellipsoid(52.0, 32.0, 28.0).at(34.0, 0.0, 52.0)),
                p("Wings", wing, extrude(&[(34.0, 12.0), (-30.0, 84.0), (-56.0, 84.0), (-50.0, 12.0)], 7.0).at(0.0, 0.0, 28.0).mirror(Y)).finish(0.5, 0.35),
                p("Tail Fin", wing, xz(&[(-34.0, 52.0), (-64.0, 104.0), (-80.0, 104.0), (-72.0, 52.0)], 7.0)).finish(0.5, 0.35),
                metal("Engines", DARK_METAL, union(0.0, vec![cylinder(12.0, 26.0).rot(0.0, 90.0, 0.0).at(-80.0, 0.0, 36.0), cylinder(9.0, 22.0).rot(0.0, 90.0, 0.0).at(-78.0, 28.0, 28.0).mirror(Y)])),
                glow("Exhaust Glow", "#ff9f2e", union(0.0, vec![cone(11.0, 0.0, 34.0).rot(0.0, -90.0, 0.0).at(-110.0, 0.0, 36.0), cone(8.0, 0.0, 26.0).rot(0.0, -90.0, 0.0).at(-102.0, 28.0, 28.0).mirror(Y)])),
            ], 20.0)
        }
        "whale" => {
            let body = "#4a6fa5";
            let slab = intersection(0.0, vec![ellipsoid(174.0, 80.0, 76.0).at(0.0, 0.0, 50.0), cuboid(260.0, 120.0, 26.0).at(0.0, 0.0, 25.0)]);
            (vec![
                p("Body", body, union(12.0, vec![ellipsoid(170.0, 78.0, 74.0).at(0.0, 0.0, 50.0), cone(30.0, 11.0, 96.0).rot(0.0, -90.0, 0.0).at(-108.0, 0.0, 56.0)])).finish(0.0, 0.5),
                p("Belly", "#e8f0f8", slab).finish(0.0, 0.5),
                p("Fluke", body, extrude(&[(0.0, -6.0), (-16.0, 8.0), (-8.0, 40.0), (-32.0, 62.0), (-38.0, 46.0), (-34.0, 18.0), (-44.0, 0.0)], 9.0).at(-160.0, 0.0, 64.0).mirror(Y)).finish(0.0, 0.5),
                p("Flippers", "#3d5d8c", ellipsoid(62.0, 18.0, 9.0).rot(0.0, 0.0, 0.0).at(26.0, 44.0, 24.0).rot(0.0, 0.0, 0.0).mirror(Y)).finish(0.0, 0.5),
                p("Eyes", BLACK, sphere(5.5).at(58.0, 38.0, 56.0).mirror(Y)).finish(0.0, 0.2),
            ], 40.0)
        }
        "shark" => {
            let body = "#7a8794";
            (vec![
                p("Body", body, union(8.0, vec![ellipsoid(150.0, 46.0, 44.0).at(0.0, 0.0, 48.0), cone(15.0, 0.0, 44.0).rot(0.0, 90.0, 0.0).at(80.0, 0.0, 46.0), cone(16.0, 6.0, 64.0).rot(0.0, -90.0, 0.0).at(-96.0, 0.0, 50.0)])).finish(0.0, 0.4),
                p("Belly", WHITE, intersection(0.0, vec![ellipsoid(152.0, 48.0, 46.0).at(0.0, 0.0, 48.0), cuboid(260.0, 100.0, 22.0).at(0.0, 0.0, 14.0)])).finish(0.0, 0.5),
                p("Fins", body, union(0.0, vec![xz(&[(12.0, 66.0), (-24.0, 112.0), (-40.0, 66.0)], 8.0), xz(&[(-112.0, 50.0), (-138.0, 100.0), (-126.0, 54.0), (-142.0, 8.0), (-112.0, 44.0)], 7.0), extrude(&[(34.0, 10.0), (-8.0, 58.0), (-18.0, 58.0), (-4.0, 10.0)], 6.0).rot(-18.0, 0.0, 0.0).at(0.0, 0.0, 34.0).mirror(Y)])).finish(0.0, 0.4),
                p("Eyes", BLACK, sphere(4.2).at(52.0, 18.0, 58.0).mirror(Y)).finish(0.0, 0.2),
                p("Teeth", WHITE, array(cone(3.4, 0.0, 10.0).rot(180.0, 0.0, 0.0).at(52.0, 14.0, 41.0), 5, [6.0, 0.0, 0.0]).mirror(Y)),
            ], 30.0)
        }
        "octopus" => {
            let c = r.pick(&["#8e44ad", "#d6338f", "#e06a2b", "#d63a2f"]);
            let legs: Vec<Node> = (0..8)
                .map(|k| {
                    let a = (k as f32 * 45.0 + 22.0).to_radians();
                    let pt = |d: f32, z: f32| [d * a.cos(), d * a.sin(), z];
                    union(5.0, chain(&[pt(18.0, 52.0), pt(44.0, 26.0), pt(70.0, 11.0), pt(88.0, 20.0)], &[13.0, 10.5, 8.5]))
                })
                .collect();
            (vec![
                p("Head", c, ellipsoid(76.0, 70.0, 84.0).at(0.0, 0.0, 82.0)).finish(0.0, 0.5),
                p("Tentacles", c, union(6.0, legs)).finish(0.0, 0.5),
                p("Eyes", WHITE, sphere(10.5).at(24.0, -30.0, 90.0).mirror(X)).finish(0.0, 0.2),
                p("Pupils", BLACK, sphere(5.5).at(24.0, -39.0, 90.0).mirror(X)).finish(0.0, 0.2),
            ], 10.0)
        }
        "butterfly" => {
            let wing = r.pick(&["#f28c28", "#2f6fd6", "#9b6ad6", "#d6338f", "#f5c842"]);
            (vec![
                p("Body", "#3a2a20", union(2.0, vec![cap([0.0, 0.0, 14.0], [0.0, 0.0, 84.0], 6.5), sphere(9.5).at(0.0, -1.0, 94.0)])).finish(0.0, 0.8),
                p("Antennae", "#3a2a20", union(0.0, vec![cap([3.0, -2.0, 98.0], [16.0, -4.0, 128.0], 2.6).mirror(X)])).finish(0.0, 0.8),
                p("Wings", wing, union(2.0, vec![ellipsoid(76.0, 7.0, 70.0).rot(0.0, -24.0, 0.0).at(42.0, 2.0, 80.0).mirror(X), ellipsoid(54.0, 7.0, 50.0).rot(0.0, 22.0, 0.0).at(32.0, 2.0, 40.0).mirror(X)])).finish(0.0, 0.6),
                p("Wing Spots", WHITE, union(0.0, vec![ellipsoid(24.0, 9.0, 20.0).rot(0.0, -24.0, 0.0).at(52.0, 2.0, 86.0).mirror(X), ellipsoid(16.0, 9.0, 14.0).rot(0.0, 22.0, 0.0).at(34.0, 2.0, 40.0).mirror(X)])).finish(0.0, 0.6),
            ], 3.0)
        }
        "spider" => {
            let c = r.pick(&["#2a2a2e", "#4a2f1c", "#3b2a4f"]);
            let legs: Vec<Node> = [22.0f32, 58.0, 100.0, 138.0]
                .iter()
                .map(|&deg| {
                    let a = deg.to_radians();
                    let pt = |d: f32, z: f32| [d * a.cos(), d * a.sin(), z];
                    union(4.0, chain(&[pt(14.0, 32.0), pt(44.0, 56.0), pt(72.0, 6.0)], &[4.6, 4.0]))
                })
                .collect();
            (vec![
                p("Abdomen", c, sphere(27.0).at(-30.0, 0.0, 36.0)).finish(0.0, 0.5),
                p("Head", c, sphere(18.0).at(14.0, 0.0, 34.0)).finish(0.0, 0.5),
                p("Legs", c, four(union(4.0, legs))).finish(0.0, 0.6),
                glow("Eyes", RED, union(0.0, vec![sphere(4.5).at(28.0, 7.0, 40.0).mirror(Y), sphere(3.2).at(26.0, 13.0, 36.0).mirror(Y)])),
                p("Marking", RED, ellipsoid(18.0, 14.0, 6.0).at(-30.0, 0.0, 62.0)).finish(0.0, 0.5),
            ], 8.0)
        }
        "trex" => {
            let c = r.pick(&["#5b8c3a", "#6b8e23", "#7a8f3a", "#8a6f3a"]);
            (vec![
                p("Body", c, union(10.0, vec![
                    ellipsoid(86.0, 52.0, 66.0).at(0.0, 0.0, 84.0),
                    cap([20.0, 0.0, 96.0], [44.0, 0.0, 118.0], 17.0),
                    ellipsoid(60.0, 36.0, 36.0).at(62.0, 0.0, 128.0),
                    ellipsoid(40.0, 28.0, 26.0).at(86.0, 0.0, 122.0),
                    cap([-28.0, 0.0, 82.0], [-76.0, 0.0, 70.0], 18.0),
                    cap([-76.0, 0.0, 70.0], [-124.0, 0.0, 58.0], 10.5),
                    cap([-124.0, 0.0, 58.0], [-156.0, 0.0, 52.0], 6.0),
                    ellipsoid(46.0, 30.0, 66.0).at(-6.0, 24.0, 52.0).mirror(Y),
                    cap([-6.0, 24.0, 34.0], [10.0, 24.0, 12.0], 9.0).mirror(Y),
                    ellipsoid(44.0, 20.0, 13.0).at(18.0, 24.0, 6.5).mirror(Y),
                    cap([36.0, 17.0, 86.0], [50.0, 17.0, 72.0], 4.6).mirror(Y),
                ])).finish(0.0, 0.6),
                p("Teeth", WHITE, array(cone(4.0, 0.0, 11.0).rot(180.0, 0.0, 0.0).at(70.0, 12.0, 112.0), 5, [7.0, 0.0, 0.0]).mirror(Y)),
                p("Eyes", "#f5c842", sphere(5.2).at(70.0, 15.0, 136.0).mirror(Y)).finish(0.0, 0.2),
                p("Belly", "#d9c88a", intersection(0.0, vec![ellipsoid(88.0, 54.0, 68.0).at(0.0, 0.0, 84.0), cuboid(60.0, 80.0, 20.0).at(34.0, 0.0, 52.0)])).finish(0.0, 0.7),
            ], 12.0)
        }
        "jellyfish" => {
            let c = r.pick(&["#f48fb1", "#7cc4f2", "#c7b3eb", "#98e2c6"]);
            let swim: Vec<Node> = (0..8)
                .map(|k| {
                    let a = (k as f32 * 45.0).to_radians();
                    let (x, y) = (40.0 * a.cos(), 40.0 * a.sin());
                    union(3.0, chain(&[[x, y, 64.0], [x * 1.1 + 6.0, y * 1.1, 38.0], [x * 0.9, y * 0.9 - 6.0, 14.0]], &[4.2, 3.8]))
                })
                .collect();
            let arms: Vec<Node> = (0..3)
                .map(|k| {
                    let a = (k as f32 * 120.0 + 30.0).to_radians();
                    let (x, y) = (12.0 * a.cos(), 12.0 * a.sin());
                    union(3.0, chain(&[[x, y, 62.0], [x * 1.8, y * 1.8, 34.0], [x * 1.2, y * 1.2, 10.0]], &[8.0, 6.5]))
                })
                .collect();
            (vec![
                p("Bell", c, union(4.0, vec![intersection(0.0, vec![ellipsoid(112.0, 112.0, 112.0).at(0.0, 0.0, 66.0), cuboid(130.0, 130.0, 60.0).at(0.0, 0.0, 96.0)]), torus(52.0, 7.0).at(0.0, 0.0, 66.0)])).finish(0.0, 0.1),
                p("Tentacles", c, union(0.0, swim)).finish(0.0, 0.2),
                p("Oral Arms", "#ffffff", union(0.0, arms)).finish(0.0, 0.2),
            ], 6.0)
        }
        "pizza" => {
            let wedge = extrude(&[(0.0, 0.0), (120.0, -42.0), (120.0, 42.0)], 30.0).at(0.0, 0.0, 8.0);
            let tops: Vec<Node> = (0..9)
                .filter_map(|i| {
                    let deg = i as f32 * 41.0 + 12.0;
                    let a = deg.to_radians();
                    let wrapped = ((deg + 180.0) % 360.0) - 180.0;
                    if wrapped.abs() < 28.0 {
                        return None;
                    }
                    let d = if i % 2 == 0 { 34.0 } else { 48.0 };
                    Some(cylinder(9.5, 5.0).at(d * a.cos(), d * a.sin(), 16.5))
                })
                .collect();
            (vec![
                p("Crust", "#d9a35a", difference(0.0, vec![rcylinder(78.0, 12.0, 5.0).at(0.0, 0.0, 6.0), wedge.clone()])).finish(0.0, 0.8),
                p("Cheese", "#f5c842", difference(0.0, vec![cylinder(66.0, 4.0).at(0.0, 0.0, 14.0), wedge])).finish(0.0, 0.5),
                p("Pepperoni", "#b3202f", union(0.0, tops)).finish(0.0, 0.6),
                p("Basil", LEAF, union(0.0, vec![ellipsoid(18.0, 10.0, 4.0).rot(0.0, 0.0, 40.0).at(-10.0, 26.0, 17.0), ellipsoid(18.0, 10.0, 4.0).rot(0.0, 0.0, -30.0).at(-34.0, -16.0, 17.0)])).finish(0.0, 0.7),
            ], 4.0)
        }
        "taco" => {
            let shell = intersection(0.0, vec![
                difference(0.0, vec![cylinder(48.0, 110.0).rot(90.0, 0.0, 0.0), cylinder(40.0, 120.0).rot(90.0, 0.0, 0.0)]),
                cuboid(130.0, 130.0, 48.0).at(0.0, 0.0, -24.0),
            ]).at(0.0, 0.0, 48.0);
            (vec![
                p("Shell", "#e8b84a", shell).finish(0.0, 0.7),
                p("Meat", "#6b3f23", ellipsoid(62.0, 96.0, 30.0).at(0.0, 0.0, 34.0)).finish(0.0, 0.8),
                p("Lettuce", "#6cc24a", union(3.0, vec![ellipsoid(76.0, 98.0, 14.0).at(0.0, 0.0, 50.0), sphere(11.0).at(-22.0, -34.0, 54.0), sphere(11.0).at(20.0, 10.0, 56.0), sphere(11.0).at(-14.0, 40.0, 54.0)])).finish(0.0, 0.7),
                p("Tomato", RED, union(0.0, vec![rbox(12.0, 12.0, 9.0, 3.0).at(-10.0, -26.0, 63.0), rbox(12.0, 12.0, 9.0, 3.0).at(14.0, 0.0, 65.0), rbox(12.0, 12.0, 9.0, 3.0).at(-8.0, 28.0, 63.0), rbox(12.0, 12.0, 9.0, 3.0).at(10.0, 40.0, 61.0)])).finish(0.0, 0.5),
                p("Cheese", "#f5c842", union(0.0, vec![cuboid(26.0, 14.0, 6.0).rot(0.0, 0.0, 20.0).at(-14.0, -6.0, 67.0), cuboid(24.0, 14.0, 6.0).rot(0.0, 0.0, -25.0).at(12.0, 22.0, 66.0)])).finish(0.0, 0.6),
            ], 3.0)
        }
        "hot dog" => {
            let zig: Vec<[f32; 3]> = (0..9).map(|i| [-52.0 + i as f32 * 13.0, if i % 2 == 0 { 7.0 } else { -7.0 }, 52.0]).collect();
            (vec![
                p("Bun", "#e0a458", difference(0.0, vec![ellipsoid(136.0, 58.0, 48.0).at(0.0, 0.0, 26.0), cap([-70.0, 0.0, 42.0], [70.0, 0.0, 42.0], 10.0)])).finish(0.0, 0.8),
                p("Sausage", "#b5482f", cap([-66.0, 0.0, 42.0], [66.0, 0.0, 42.0], 12.5)).finish(0.0, 0.5),
                p("Mustard", "#f5c842", union(2.0, chain(&zig, &[3.6; 8]))).finish(0.0, 0.4),
            ], 3.0)
        }
        "sushi" => (vec![
            p("Rice", WHITE, union(0.0, vec![rbox(66.0, 34.0, 24.0, 10.0).at(-36.0, 0.0, 12.0), cylinder(20.0, 32.0).at(44.0, 0.0, 16.0)])).finish(0.0, 0.8),
            p("Salmon", "#f2735f", union(0.0, vec![rbox(76.0, 40.0, 13.0, 5.0).at(-36.0, 0.0, 29.0), cylinder(9.5, 33.0).at(44.0, 0.0, 16.5)])).finish(0.0, 0.4),
            p("Nori", "#1f3a2a", union(0.0, vec![difference(0.0, vec![cylinder(25.0, 32.0).at(44.0, 0.0, 16.0), cylinder(19.0, 40.0).at(44.0, 0.0, 16.0)]), cuboid(14.0, 44.0, 22.0).at(-36.0, 0.0, 15.0)])).finish(0.0, 0.9),
            p("Wasabi", "#7cb342", sphere(9.0).at(6.0, -34.0, 9.0)).finish(0.0, 0.7),
        ], 3.0),
        "croissant" => {
            let pieces: Vec<Node> = (-3..=3)
                .map(|i| {
                    let a = i as f32 * 0.4;
                    let s = 1.0 - 0.16 * (i as f32).abs();
                    ellipsoid(42.0 * s.max(0.5), 48.0 * s, 40.0 * s).rot(0.0, 0.0, -a.to_degrees()).at(66.0 * a.sin(), 66.0 * a.cos(), 20.0 * s)
                })
                .collect();
            (vec![p("Croissant", "#d9993f", union(5.0, pieces)).finish(0.0, 0.7)], 3.0)
        }
        "cookie" => (vec![
            p("Cookie", "#d2a05a", difference(0.0, vec![union(3.0, vec![rcylinder(62.0, 16.0, 7.0).at(0.0, 0.0, 8.0), sphere(10.0).at(48.0, -34.0, 8.0), sphere(10.0).at(-50.0, 30.0, 8.0)]), sphere(26.0).at(54.0, 34.0, 18.0)])).finish(0.0, 0.85),
            p("Chips", "#4a2b1a", union(0.0, vec![ellipsoid(16.0, 14.0, 9.0).at(-20.0, 18.0, 16.0), ellipsoid(15.0, 13.0, 9.0).at(8.0, 30.0, 16.0), ellipsoid(16.0, 13.0, 9.0).at(26.0, -10.0, 16.0), ellipsoid(15.0, 14.0, 9.0).at(-32.0, -16.0, 16.0), ellipsoid(14.0, 14.0, 9.0).at(0.0, -26.0, 16.0), ellipsoid(14.0, 12.0, 9.0).at(-4.0, 2.0, 16.0)])).finish(0.0, 0.5),
        ], 3.0),
        "watermelon" => {
            let half = |r: f32, t: f32, up: f32| {
                intersection(0.0, vec![cylinder(r, t).rot(90.0, 0.0, 0.0), cuboid(2.0 * r + 8.0, t + 4.0, r + 4.0).at(0.0, 0.0, -(r + 4.0) / 2.0 + up)]).at(0.0, 0.0, 70.0)
            };
            (vec![
                p("Rind", "#2e8b3d", half(70.0, 28.0, 0.0)).finish(0.0, 0.5),
                p("White Rind", "#e8f2d0", half(65.0, 29.0, 0.5)).finish(0.0, 0.6),
                p("Flesh", "#e5383b", half(60.0, 30.0, 1.0)).finish(0.0, 0.35),
                p("Seeds", BLACK, union(0.0, vec![ellipsoid(8.0, 5.0, 13.0).at(-26.0, 15.0, 44.0).mirror(Y), ellipsoid(8.0, 5.0, 13.0).at(2.0, 15.0, 36.0).mirror(Y), ellipsoid(8.0, 5.0, 13.0).at(26.0, 15.0, 46.0).mirror(Y), ellipsoid(8.0, 5.0, 13.0).at(-8.0, 15.0, 58.0).mirror(Y), ellipsoid(8.0, 5.0, 13.0).at(16.0, 15.0, 62.0).mirror(Y)])).finish(0.0, 0.3),
            ], 2.5)
        }
        "strawberry" => {
            let prof = [(0.0, 0.0), (9.0, 2.0), (21.0, 14.0), (31.0, 32.0), (36.0, 52.0), (32.0, 70.0), (22.0, 82.0), (0.0, 86.0)];
            let seeds: Vec<Node> = (0..18)
                .map(|i| {
                    let z = 12.0 + i as f32 * 3.9;
                    let a = i as f32 * 2.399963;
                    let rr = lathe_r(&prof, z) + 0.6;
                    ellipsoid(5.6, 5.6, 7.5).at(rr * a.cos(), rr * a.sin(), z)
                })
                .collect();
            let leaves = ring_of(6, 16.0, 86.0, |deg| ellipsoid(36.0, 13.0, 6.0).rot(0.0, 14.0, deg));
            (vec![
                p("Berry", "#d6253a", lathe(&prof)).finish(0.0, 0.35),
                p("Seeds", "#f5d442", union(0.0, seeds)).finish(0.0, 0.5),
                p("Leaves", LEAF, union(2.0, leaves)).finish(0.0, 0.6),
                p("Stem", "#2e7d4f", cylinder(4.5, 18.0).at(0.0, 0.0, 96.0)).finish(0.0, 0.6),
            ], 1.5)
        }
        "pineapple" => {
            let prof = [(0.0, 0.0), (24.0, 0.0), (32.0, 14.0), (34.0, 40.0), (31.0, 68.0), (21.0, 86.0), (0.0, 92.0)];
            let scales: Vec<Node> = (0..8)
                .map(|k| {
                    let a = (k as f32 * 45.0).to_radians();
                    array(sphere(6.0).at(33.5 * a.cos(), 33.5 * a.sin(), 20.0), 4, [0.0, 0.0, 18.0])
                })
                .collect();
            let leaves = ring_of(9, 7.0, 108.0, |deg| cone(9.0, 0.0, 56.0).rot(0.0, 22.0, deg));
            (vec![
                p("Fruit", "#e8b233", lathe(&prof)).finish(0.0, 0.6),
                p("Scales", "#b8791f", union(0.0, scales)).finish(0.0, 0.7),
                p("Crown", "#3f8f3f", union(3.0, leaves)).finish(0.0, 0.6),
            ], 2.5)
        }
        "axe" => (vec![
            p("Handle", WOOD, union(0.0, vec![cylinder(5.5, 150.0).at(0.0, 0.0, 75.0), rcylinder(7.0, 14.0, 2.0).at(0.0, 0.0, 8.0)])).finish(0.0, 0.7),
            metal("Blade", "#c9ccd1", union(0.0, vec![xz(&[(4.0, 146.0), (30.0, 160.0), (46.0, 134.0), (52.0, 104.0), (30.0, 90.0), (4.0, 110.0)], 9.0), rbox(16.0, 12.0, 34.0, 2.0).at(-5.0, 0.0, 128.0)])),
        ], 3.0),
        "battleaxe" => (vec![
            p("Handle", DARK_WOOD, union(0.0, vec![cylinder(6.0, 170.0).at(0.0, 0.0, 85.0), rcylinder(8.0, 16.0, 2.0).at(0.0, 0.0, 8.0), sphere(8.0).at(0.0, 0.0, 174.0)])).finish(0.0, 0.7),
            metal("Blades", "#c9ccd1", union(0.0, vec![xz(&[(4.0, 160.0), (36.0, 184.0), (60.0, 160.0), (66.0, 130.0), (36.0, 108.0), (4.0, 124.0)], 9.0).mirror(X)])),
            metal("Hub", GOLD, rcylinder(11.0, 34.0, 3.0).at(0.0, 0.0, 142.0)),
        ], 3.0),
        "shield" => {
            let outline = [(-42.0, 60.0), (42.0, 60.0), (42.0, 22.0), (34.0, -10.0), (18.0, -38.0), (0.0, -54.0), (-18.0, -38.0), (-34.0, -10.0), (-42.0, 22.0)];
            let inner: Vec<(f32, f32)> = outline.iter().map(|&(x, z)| (x * 0.82, (z - 3.0) * 0.82 + 3.0)).collect();
            let c = r.pick(&[BLUE, RED, "#2e7d4f", "#8e44ad"]);
            (vec![
                p("Face", c, xz(&outline, 9.0).at(0.0, 0.0, 58.0)).finish(0.0, 0.5),
                metal("Rim", GOLD, difference(0.0, vec![xz(&outline, 13.0), xz(&inner, 16.0)]).at(0.0, 0.0, 58.0)),
                metal("Boss", "#c9ccd1", ellipsoid(26.0, 14.0, 26.0).at(0.0, -5.0, 76.0)),
                metal("Emblem", GOLD, union(0.0, vec![cuboid(11.0, 12.0, 58.0).at(0.0, -2.0, 62.0), cuboid(58.0, 12.0, 11.0).at(0.0, -2.0, 76.0)])),
            ], 2.5)
        }
        "bow" => (vec![
            p("Limbs", WOOD, intersection(0.0, vec![torus(74.0, 6.5).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 74.0), cuboid(60.0, 30.0, 170.0).at(54.0, 0.0, 74.0)])).finish(0.0, 0.6),
            p("String", "#f4ecd6", cap([31.0, 0.0, 142.0], [31.0, 0.0, 6.0], 2.4)).finish(0.0, 0.9),
            p("Arrow", "#8d6a4a", union(0.0, vec![cap([-26.0, 0.0, 74.0], [92.0, 0.0, 74.0], 3.0), cone(7.0, 0.0, 18.0).rot(0.0, 90.0, 0.0).at(100.0, 0.0, 74.0), cuboid(20.0, 2.5, 12.0).at(-18.0, 0.0, 74.0)])).finish(0.0, 0.7),
        ], 3.0),
        "staff" => {
            let orb = r.pick(&["#7cc4f2", "#b36bff", "#7dff9e", "#ff7b7b"]);
            (vec![
                p("Staff", DARK_WOOD, union(0.0, vec![cylinder(5.5, 150.0).at(0.0, 0.0, 75.0), cone(7.0, 4.0, 10.0).at(0.0, 0.0, 6.0)])).finish(0.0, 0.8),
                metal("Claw", GOLD, union(0.0, ring_of(4, 11.0, 168.0, |deg| cap([0.0, 0.0, -22.0], [6.0, 0.0, 14.0], 3.6).rot(0.0, 0.0, deg)))),
                glow("Orb", orb, sphere(15.0).at(0.0, 0.0, 168.0)),
            ], 3.0)
        }
        "wand" => (vec![
            p("Wand", "#2a2a2e", cone(7.5, 4.0, 110.0).at(0.0, 0.0, 55.0)).finish(0.0, 0.5),
            metal("Grip Bands", GOLD, union(0.0, vec![cylinder(8.6, 6.0).at(0.0, 0.0, 14.0), cylinder(7.8, 6.0).at(0.0, 0.0, 30.0)])),
            glow("Star", "#ffe14d", extrude(&[(0.0, 22.0), (6.5, 7.0), (22.0, 7.0), (10.0, -3.0), (14.0, -18.0), (0.0, -9.0), (-14.0, -18.0), (-10.0, -3.0), (-22.0, 7.0), (-6.5, 7.0)], 8.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 132.0)),
        ], 2.0),
        "helmet" => {
            let c = r.pick(&["#a9adb3", "#8a8f97", "#b08d57"]);
            (vec![
                metal("Dome", c, difference(0.0, vec![ellipsoid(104.0, 114.0, 116.0).at(0.0, 0.0, 58.0), cuboid(76.0, 34.0, 9.0).at(0.0, -55.0, 72.0)])),
                metal("Rim", GOLD, torus(54.0, 5.0).at(0.0, 0.0, 9.0)),
                p("Crest", r.pick(&[RED, BLUE, "#2e7d4f"]), ellipsoid(18.0, 104.0, 44.0).at(0.0, 0.0, 116.0)).finish(0.0, 0.7),
                metal("Cheek Guards", c, rbox(9.0, 40.0, 42.0, 3.0).at(50.0, -14.0, 36.0).mirror(X)),
                metal("Nose Guard", c, cuboid(11.0, 8.0, 46.0).at(0.0, -57.0, 62.0)),
            ], 3.0)
        }
        "potion" => {
            let c = r.pick(&["#b36bff", "#ff5c8a", "#4de1ff", "#7dff9e"]);
            let prof = [(0.0, 0.0), (28.0, 0.0), (40.0, 12.0), (46.0, 34.0), (44.0, 52.0), (30.0, 70.0), (14.0, 82.0), (12.0, 104.0), (15.0, 110.0), (0.0, 110.0)];
            (vec![
                glassy("Flask", c, lathe(&prof)),
                p("Cork", "#c8a06a", union(0.0, vec![cylinder(11.0, 16.0).at(0.0, 0.0, 116.0), sphere(2.0).at(0.0, 0.0, 124.0)])).finish(0.0, 0.9),
                p("Label", "#f4ecd6", cylinder(48.0, 16.0).at(0.0, 0.0, 36.0).hollow(0.0)).finish(0.0, 0.9),
                glow("Sparkles", "#ffffff", union(0.0, vec![sphere(5.0).at(30.0, 0.0, 142.0), sphere(4.0).at(-26.0, 10.0, 132.0), sphere(3.5).at(8.0, -28.0, 150.0)])),
            ], 1.5)
        }
        "lightsaber" => {
            let b = r.pick(&["#3ec6ff", "#4cff6a", "#ff3b3b", "#b36bff"]);
            (vec![
                metal("Hilt", DARK_METAL, union(0.0, vec![cylinder(8.0, 66.0).at(0.0, 0.0, 33.0), array(torus(9.0, 2.6).at(0.0, 0.0, 14.0), 4, [0.0, 0.0, 13.0]), cylinder(11.0, 8.0).at(0.0, 0.0, 70.0)])),
                metal("Emitter", "#c9ccd1", cylinder(10.0, 6.0).at(0.0, 0.0, 77.0)),
                p("Button", RED, sphere(3.6).at(0.0, -8.0, 40.0)),
                glow("Blade", b, cap([0.0, 0.0, 80.0], [0.0, 0.0, 180.0], 6.2)),
            ], 3.0)
        }
        "blaster" => (vec![
            metal("Body", r.pick(&["#c9ccd1", "#2f6fd6", "#d63a2f"]), union(4.0, vec![rbox(110.0, 22.0, 38.0, 8.0).at(0.0, 0.0, 66.0), rbox(24.0, 20.0, 52.0, 6.0).rot(0.0, 14.0, 0.0).at(-28.0, 0.0, 30.0)])),
            metal("Barrel", DARK_METAL, union(0.0, vec![cylinder(8.0, 56.0).rot(0.0, 90.0, 0.0).at(78.0, 0.0, 70.0), torus(10.0, 3.2).rot(0.0, 90.0, 0.0).at(100.0, 0.0, 70.0)])),
            metal("Scope", DARK_METAL, cylinder(6.5, 46.0).rot(0.0, 90.0, 0.0).at(6.0, 0.0, 92.0)),
            glow("Energy Cell", CYAN, cylinder(7.0, 40.0).rot(0.0, 90.0, 0.0).at(-6.0, 12.0, 70.0)),
        ], 2.0),
        "drone" => (vec![
            p("Body", r.pick(&["#f4f4f2", "#2a2a2e", "#d63a2f"]), union(6.0, vec![ellipsoid(66.0, 58.0, 28.0).at(0.0, 0.0, 40.0), rbox(32.0, 32.0, 12.0, 4.0).at(0.0, 0.0, 50.0)])).finish(0.3, 0.4),
            p("Arms", "#2a2a2e", four(cap([0.0, 0.0, 42.0], [54.0, 54.0, 44.0], 5.0))).finish(0.2, 0.5),
            metal("Motors", DARK_METAL, four(cylinder(11.0, 18.0).at(54.0, 54.0, 49.0))),
            p("Propellers", "#3a3a3d", four(cylinder(32.0, 5.0).at(54.0, 54.0, 61.0))).finish(0.0, 0.4),
            p("Legs", "#2a2a2e", four(cap([24.0, 24.0, 36.0], [32.0, 32.0, 6.0], 3.8))).finish(0.0, 0.6),
            glassy("Camera", "#1c2a4a", sphere(14.0).at(0.0, -22.0, 28.0)),
        ], 3.0),
        "telescope" => {
            let u = Vec3::new(55.0f32.to_radians().sin(), 0.0, 55.0f32.to_radians().cos());
            let c = Vec3::new(0.0, 0.0, 128.0);
            let at = |s: f32| -> [f32; 3] { let q = c + u * s; [q.x, q.y, q.z] };
            let foot = |deg: f32| -> [f32; 3] { let a = deg.to_radians(); [62.0 * a.cos(), 62.0 * a.sin(), 2.0] };
            (vec![
                p("Tube", "#f4f4f2", cylinder(17.0, 120.0).rot(0.0, 55.0, 0.0).at(c.x, c.y, c.z)).finish(0.2, 0.35),
                metal("Lens Hood", BLACK, cylinder(21.0, 16.0).rot(0.0, 55.0, 0.0).at(at(66.0)[0], 0.0, at(66.0)[2])),
                metal("Eyepiece", DARK_METAL, cylinder(8.0, 28.0).rot(0.0, 55.0, 0.0).at(at(-68.0)[0], 0.0, at(-68.0)[2])),
                metal("Finder", GOLD, cylinder(6.5, 46.0).rot(0.0, 55.0, 0.0).at(at(20.0)[0] - 8.0, 0.0, at(20.0)[2] + 24.0)),
                metal("Tripod", "#6f7377", union(2.0, vec![cylinder(11.0, 12.0).at(0.0, 0.0, 112.0), cap([0.0, 0.0, 112.0], foot(90.0), 4.4), cap([0.0, 0.0, 112.0], foot(210.0), 4.4), cap([0.0, 0.0, 112.0], foot(330.0), 4.4)])),
            ], 4.0)
        }
        "hourglass" => {
            let prof = [(0.0, 10.0), (30.0, 10.0), (34.0, 24.0), (33.0, 40.0), (24.0, 54.0), (8.0, 64.0), (5.0, 67.0), (8.0, 70.0), (24.0, 80.0), (33.0, 94.0), (34.0, 110.0), (30.0, 124.0), (0.0, 124.0)];
            (vec![
                glassy("Glass", "#cfe8f2", lathe(&prof).hollow(5.0)),
                p("Sand", "#e6b84d", union(0.0, vec![cone(26.0, 2.0, 26.0).at(0.0, 0.0, 28.0), cone(3.0, 22.0, 20.0).at(0.0, 0.0, 108.0), cap([0.0, 0.0, 70.0], [0.0, 0.0, 44.0], 2.6)])).finish(0.0, 0.9),
                p("Frame", WOOD, union(0.0, vec![rcylinder(48.0, 10.0, 3.0).at(0.0, 0.0, 5.0), rcylinder(48.0, 10.0, 3.0).at(0.0, 0.0, 129.0), four(cap([32.0, 32.0, 8.0], [32.0, 32.0, 126.0], 4.4))])).finish(0.0, 0.7),
            ], 2.0)
        }
        "anchor" => (vec![metal("Anchor", "#4a5160", union(3.0, vec![
            cap([0.0, 0.0, 8.0], [0.0, 0.0, 138.0], 6.0),
            intersection(0.0, vec![torus(46.0, 7.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 56.0), cuboid(130.0, 30.0, 52.0).at(0.0, 0.0, 30.0)]),
            cone(14.0, 0.0, 30.0).rot(0.0, -30.0, 0.0).at(-52.0, 0.0, 68.0),
            cone(14.0, 0.0, 30.0).rot(0.0, 30.0, 0.0).at(52.0, 0.0, 68.0),
            torus(14.0, 4.6).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 152.0),
            cap([-34.0, 0.0, 120.0], [34.0, 0.0, 120.0], 5.4),
        ]))], 20.0),
        "compass" => {
            let needle_rot = r.range(0.0, 360.0);
            (vec![
                metal("Case", GOLD, union(2.0, vec![rcylinder(58.0, 20.0, 6.0).at(0.0, 0.0, 10.0), torus(12.0, 4.0).rot(90.0, 0.0, 0.0).at(0.0, 68.0, 10.0), cylinder(7.0, 10.0).at(0.0, 62.0, 24.0)])),
                p("Face", "#f4ecd6", cylinder(49.0, 4.0).at(0.0, 0.0, 20.5)).finish(0.0, 0.4),
                p("Ticks", BLACK, union(0.0, ring_of(4, 40.0, 23.5, |deg| cuboid(11.0, 5.0, 3.0).rot(0.0, 0.0, deg)))).finish(0.0, 0.5),
                p("Needle Red", RED, extrude(&[(-7.0, 0.0), (0.0, 38.0), (7.0, 0.0)], 5.0).rot(0.0, 0.0, needle_rot).at(0.0, 0.0, 25.0)).finish(0.0, 0.4),
                p("Needle White", WHITE, extrude(&[(-7.0, 0.0), (0.0, -38.0), (7.0, 0.0)], 5.0).rot(0.0, 0.0, needle_rot).at(0.0, 0.0, 25.0)).finish(0.0, 0.4),
                metal("Pivot", "#c9ccd1", sphere(6.0).at(0.0, 0.0, 27.0)),
            ], 1.0)
        }
        "skull" => {
            let sockets = sphere(13.5).at(21.0, -46.0, 62.0).mirror(X);
            (vec![
                p("Skull", BONE, difference(0.0, vec![
                    union(8.0, vec![ellipsoid(94.0, 102.0, 88.0).at(0.0, 0.0, 70.0), ellipsoid(76.0, 46.0, 34.0).at(0.0, -24.0, 46.0), rbox(54.0, 50.0, 30.0, 10.0).at(0.0, -14.0, 20.0)]),
                    sockets,
                    ellipsoid(12.0, 18.0, 22.0).at(0.0, -52.0, 40.0),
                ])).finish(0.0, 0.7),
                p("Teeth", "#f7f3e6", array(cuboid(7.0, 7.0, 13.0).at(-21.0, -39.0, 24.0), 6, [8.4, 0.0, 0.0])).finish(0.0, 0.5),
            ], 2.0)
        }
        "shoe" => {
            let upper = r.pick(&[RED, BLUE, "#2a2a2e", "#f5c842", "#4caf50"]);
            let sole = [(-62.0, -18.0), (-66.0, 0.0), (-62.0, 18.0), (-30.0, 25.0), (20.0, 29.0), (52.0, 27.0), (72.0, 12.0), (74.0, -6.0), (62.0, -23.0), (20.0, -27.0), (-30.0, -22.0)];
            (vec![
                p("Sole", WHITE, extrude(&sole, 15.0).at(0.0, 0.0, 7.5)).finish(0.0, 0.7),
                p("Upper", upper, difference(0.0, vec![
                    union(10.0, vec![ellipsoid(80.0, 54.0, 46.0).at(36.0, 0.0, 36.0), ellipsoid(68.0, 52.0, 80.0).at(-34.0, 0.0, 54.0), ellipsoid(70.0, 48.0, 50.0).at(0.0, 0.0, 40.0)]),
                    ellipsoid(36.0, 32.0, 44.0).at(-42.0, 0.0, 94.0),
                ])).finish(0.0, 0.6),
                p("Toe Cap", WHITE, ellipsoid(42.0, 54.0, 34.0).at(56.0, 0.0, 26.0)).finish(0.0, 0.6),
                p("Laces", WHITE, array(cap([0.0, -17.0, 0.0], [0.0, 17.0, 0.0], 3.4).rot(0.0, -28.0, 0.0).at(10.0, 0.0, 60.0), 4, [-9.0, 0.0, 6.0])).finish(0.0, 0.8),
            ], 3.0)
        }
        "backpack" => {
            let c = r.pick(&[RED, BLUE, "#2e7d4f", "#f5a623", "#8e44ad"]);
            (vec![
                p("Pack", c, rbox(96.0, 56.0, 126.0, 22.0).at(0.0, 0.0, 63.0)).finish(0.0, 0.8),
                p("Pocket", c, rbox(72.0, 24.0, 52.0, 10.0).at(0.0, -34.0, 34.0)).finish(0.0, 0.8),
                p("Flap", "#2a2a2e", rbox(94.0, 60.0, 28.0, 10.0).at(0.0, -2.0, 112.0)).finish(0.0, 0.8),
                metal("Zipper", GOLD, cuboid(70.0, 5.0, 6.0).at(0.0, -47.0, 52.0)),
                p("Handle", "#2a2a2e", torus(14.0, 3.6).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 132.0)).finish(0.0, 0.8),
                metal("Buckle", GOLD, cuboid(14.0, 6.0, 14.0).at(0.0, -34.0, 110.0)),
            ], 3.0)
        }
        "shades" => {
            let c = r.pick(&[BLACK, "#8b5a2b", RED, "#f5c842"]);
            (vec![
                glassy("Lenses", "#1c1f2a", rcylinder(26.0, 6.0, 2.5).rot(90.0, 0.0, 0.0).at(32.0, 0.0, 34.0).mirror(X)),
                p("Frames", c, union(0.0, vec![torus(26.0, 4.6).rot(90.0, 0.0, 0.0).at(32.0, 0.0, 34.0).mirror(X), cap([-8.0, 0.0, 40.0], [8.0, 0.0, 40.0], 4.0)])).finish(0.0, 0.4),
                p("Temples", c, union(0.0, vec![cap([58.0, 0.0, 40.0], [60.0, 92.0, 40.0], 3.8).mirror(X), cap([60.0, 92.0, 40.0], [60.0, 102.0, 28.0], 3.8).mirror(X)])).finish(0.0, 0.4),
            ], 1.0)
        }
        "lighthouse" => {
            let prof = [(0.0, 0.0), (36.0, 0.0), (36.0, 10.0), (24.0, 130.0), (0.0, 130.0)];
            let band = |z0: f32, z1: f32| lathe(&[(0.0, z0), (lathe_r(&prof, z0) + 1.0, z0), (lathe_r(&prof, z1) + 1.0, z1), (0.0, z1)]);
            (vec![
                p("Tower", WHITE, lathe(&prof)).finish(0.0, 0.7),
                p("Stripes", RED, union(0.0, vec![band(24.0, 48.0), band(72.0, 96.0), band(114.0, 130.0)])).finish(0.0, 0.7),
                metal("Gallery", "#3a3a3d", union(0.0, vec![rcylinder(33.0, 6.0, 2.0).at(0.0, 0.0, 132.0), torus(30.0, 2.8).at(0.0, 0.0, 144.0)])),
                glassy("Lamp Room", "#cfe8f2", cylinder(17.0, 28.0).at(0.0, 0.0, 149.0)),
                glow("Light", "#fff3a0", sphere(12.0).at(0.0, 0.0, 149.0)),
                p("Roof", RED, union(0.0, vec![cone(23.0, 0.0, 26.0).at(0.0, 0.0, 176.0), sphere(4.0).at(0.0, 0.0, 192.0)])).finish(0.0, 0.5),
                p("Door", "#3a2a20", rbox(14.0, 6.0, 26.0, 3.0).at(0.0, -35.0, 18.0)).finish(0.0, 0.7),
                p("Rocks", "#8d8d88", union(8.0, vec![ellipsoid(110.0, 90.0, 18.0).at(0.0, 0.0, 2.0), ellipsoid(40.0, 34.0, 24.0).at(46.0, 20.0, 8.0), ellipsoid(34.0, 30.0, 20.0).at(-44.0, -22.0, 6.0)])).finish(0.0, 0.95),
            ], 40.0)
        }
        "windmill" => {
            let sail = |deg: f32| {
                union(0.0, vec![cuboid(8.0, 6.0, 150.0), cuboid(26.0, 3.5, 62.0).at(15.0, -1.0, 42.0), cuboid(26.0, 3.5, 62.0).at(-15.0, -1.0, -42.0)]).rot(0.0, deg, 0.0)
            };
            (vec![
                p("Tower", "#e8dcc0", lathe(&[(0.0, 0.0), (44.0, 0.0), (28.0, 112.0), (0.0, 112.0)])).finish(0.0, 0.85),
                p("Cap", DARK_WOOD, cone(34.0, 0.0, 36.0).at(0.0, 0.0, 130.0)).finish(0.0, 0.8),
                p("Door", "#3a2a20", rbox(18.0, 6.0, 30.0, 4.0).at(0.0, -43.0, 15.0)).finish(0.0, 0.8),
                p("Windows", "#7cc4f2", rbox(12.0, 6.0, 14.0, 3.0).at(0.0, -31.0, 62.0)).finish(0.0, 0.3),
                p("Sails", WOOD, union(0.0, vec![sail(20.0), sail(110.0)]).at(0.0, -34.0, 104.0)).finish(0.0, 0.8),
                metal("Hub", GOLD, sphere(10.0).at(0.0, -34.0, 104.0)),
            ], 60.0)
        }
        "bridge" => (vec![
            p("Arch", "#a9a396", union(0.0, vec![difference(0.0, vec![cuboid(210.0, 46.0, 64.0).at(0.0, 0.0, 32.0), cylinder(40.0, 60.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 0.0), cuboid(260.0, 100.0, 4.0).at(0.0, 0.0, -2.0)]), cuboid(220.0, 8.0, 10.0).at(0.0, 18.0, 69.0).mirror(Y)])).finish(0.0, 0.9),
            p("Water", "#3a8fd6", cuboid(260.0, 90.0, 8.0).at(0.0, 0.0, -4.0)).finish(0.0, 0.05),
        ], 60.0),
        "volcano" => {
            let prof = [(0.0, 0.0), (100.0, 0.0), (64.0, 50.0), (42.0, 82.0), (34.0, 92.0), (20.0, 84.0), (0.0, 82.0)];
            (vec![
                p("Mountain", "#6b5a4e", lathe(&prof)).finish(0.0, 0.95),
                glow("Lava", "#ff6a1a", union(3.0, vec![cylinder(21.0, 4.0).at(0.0, 0.0, 83.0), cap([26.0, 0.0, 88.0], [48.0, 4.0, 60.0], 7.0), cap([48.0, 4.0, 60.0], [68.0, 0.0, 30.0], 8.0), cap([-24.0, 8.0, 86.0], [-44.0, 14.0, 62.0], 6.0)])),
                p("Smoke", "#9a9a9a", union(6.0, vec![sphere(22.0).at(0.0, 0.0, 120.0), sphere(18.0).at(14.0, 6.0, 146.0), sphere(15.0).at(-4.0, 0.0, 168.0)])).finish(0.0, 1.0),
            ], 120.0)
        }
        "palm tree" => {
            let trunk: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [4.0, 0.0, 36.0], [12.0, 0.0, 72.0], [24.0, 0.0, 106.0], [38.0, 0.0, 136.0]];
            let fronds = ring_of(7, 34.0, 142.0, |deg| ellipsoid(92.0, 18.0, 6.0).rot(0.0, 26.0, deg));
            (vec![
                p("Trunk", "#8a5a2b", union(4.0, chain(&trunk, &[11.0, 9.5, 8.5, 7.5]))).finish(0.0, 0.9),
                p("Fronds", "#3f9d4a", union(3.0, fronds).at(38.0, 0.0, 0.0)).finish(0.0, 0.7),
                p("Coconuts", "#5a3a22", union(0.0, vec![sphere(9.0).at(48.0, 8.0, 132.0), sphere(9.0).at(32.0, 12.0, 132.0), sphere(9.0).at(40.0, -10.0, 130.0)])).finish(0.0, 0.8),
                p("Sand", "#e8d28a", ellipsoid(150.0, 150.0, 14.0).at(10.0, 0.0, 0.0)).finish(0.0, 1.0),
            ], 40.0)
        }
        "bonsai" => {
            let pads = vec![
                union(6.0, vec![ellipsoid(70.0, 62.0, 28.0).at(-8.0, 0.0, 128.0), sphere(20.0).at(14.0, 8.0, 138.0)]),
                union(6.0, vec![ellipsoid(54.0, 50.0, 24.0).at(38.0, -4.0, 98.0), sphere(15.0).at(48.0, 6.0, 106.0)]),
                union(6.0, vec![ellipsoid(46.0, 42.0, 22.0).at(-36.0, 6.0, 92.0)]),
            ];
            (vec![
                p("Pot", "#8b4a2b", rbox(120.0, 76.0, 28.0, 7.0).at(0.0, 0.0, 14.0)).finish(0.0, 0.85),
                p("Soil", "#3a2a1c", cuboid(106.0, 62.0, 4.0).at(0.0, 0.0, 29.0)).finish(0.0, 1.0),
                p("Trunk", "#6b4423", union(4.0, chain(&[[0.0, 0.0, 28.0], [-6.0, 0.0, 52.0], [6.0, 0.0, 76.0], [-4.0, 0.0, 104.0], [-8.0, 0.0, 120.0]], &[10.0, 8.0, 7.0, 6.0]).into_iter().chain(chain(&[[2.0, 0.0, 70.0], [34.0, 0.0, 88.0]], &[4.4])).chain(chain(&[[-2.0, 0.0, 66.0], [-34.0, 0.0, 84.0]], &[4.4])).collect())).finish(0.0, 0.9),
                p("Foliage", "#2e7d4f", union(0.0, pads)).finish(0.0, 0.8),
            ], 8.0)
        }
        "throne" => (vec![
            metal("Frame", GOLD, union(2.0, vec![
                rbox(76.0, 66.0, 24.0, 5.0).at(0.0, 0.0, 44.0),
                rbox(82.0, 14.0, 150.0, 6.0).at(0.0, 30.0, 95.0),
                rbox(12.0, 62.0, 34.0, 4.0).at(38.0, -2.0, 70.0).mirror(X),
                cone(26.0, 0.0, 36.0).at(0.0, 30.0, 188.0),
                sphere(8.0).at(36.0, 30.0, 174.0).mirror(X),
                cuboid(10.0, 10.0, 34.0).at(32.0, -26.0, 16.0).mirror(X).mirror(Y),
            ])),
            p("Cushion", RED, rbox(64.0, 54.0, 12.0, 5.0).at(0.0, -2.0, 62.0)).finish(0.0, 0.9),
            p("Back Pad", RED, rbox(56.0, 6.0, 90.0, 4.0).at(0.0, 21.0, 112.0)).finish(0.0, 0.9),
            p("Steps", "#8d8d88", union(0.0, vec![rbox(130.0, 100.0, 8.0, 2.0).at(0.0, 0.0, -4.0), rbox(104.0, 82.0, 8.0, 2.0).at(0.0, 4.0, 4.0)])).finish(0.0, 0.9),
        ], 12.0),
        "armchair" => {
            let c = r.pick(&["#8b5a2b", "#2f6fd6", "#4caf50", "#9e9e9e", "#b3202f"]);
            (vec![
                p("Chair", c, union(2.0, vec![rbox(78.0, 66.0, 26.0, 8.0).at(0.0, 0.0, 30.0), rbox(78.0, 18.0, 86.0, 9.0).at(0.0, 25.0, 66.0), rbox(18.0, 66.0, 48.0, 8.0).at(30.0, 0.0, 44.0).mirror(X)])).finish(0.0, 0.9),
                p("Cushion", "#f4ecd6", rbox(48.0, 48.0, 14.0, 6.0).at(0.0, -2.0, 50.0)).finish(0.0, 0.9),
                p("Legs", DARK_WOOD, cone(6.0, 4.0, 16.0).at(32.0, 26.0, 8.0).mirror(X).mirror(Y)).finish(0.0, 0.6),
            ], 15.0)
        }
        "toaster" => {
            let c = r.pick(&["#c9ccd1", RED, "#f4ecd6", "#2f6fd6"]);
            (vec![
                metal("Body", c, difference(0.0, vec![rbox(116.0, 64.0, 74.0, 16.0).at(0.0, 0.0, 41.0), cuboid(80.0, 11.0, 20.0).at(0.0, 15.0, 78.0).mirror(Y)])),
                p("Toast", "#e0b070", rbox(72.0, 8.0, 36.0, 8.0).at(0.0, 15.0, 80.0).mirror(Y)).finish(0.0, 0.9),
                p("Lever", BLACK, cuboid(8.0, 12.0, 20.0).at(60.0, -4.0, 56.0)).finish(0.0, 0.5),
                p("Dial", BLACK, cylinder(9.0, 8.0).rot(90.0, 0.0, 0.0).at(-34.0, -34.0, 40.0)).finish(0.0, 0.5),
                p("Feet", BLACK, cylinder(6.0, 6.0).at(48.0, 22.0, 3.0).mirror(X).mirror(Y)).finish(0.0, 0.9),
            ], 4.0)
        }
        "coffee maker" => (vec![
            p("Base", "#2a2a2e", rbox(106.0, 96.0, 12.0, 4.0).at(0.0, 0.0, 6.0)).finish(0.2, 0.5),
            p("Column", "#2a2a2e", rbox(94.0, 28.0, 150.0, 6.0).at(0.0, 34.0, 81.0)).finish(0.2, 0.5),
            p("Head", "#3a3a3d", rbox(94.0, 74.0, 32.0, 8.0).at(0.0, 10.0, 138.0)).finish(0.2, 0.5),
            glassy("Carafe", "#5a3420", union(0.0, vec![cylinder(31.0, 56.0).at(0.0, -6.0, 40.0), torus(20.0, 5.0).rot(90.0, 0.0, 0.0).at(34.0, -6.0, 42.0)])),
            p("Lid", BLACK, cylinder(33.0, 8.0).at(0.0, -6.0, 72.0)).finish(0.0, 0.5),
            p("Button", RED, sphere(6.0).at(0.0, 18.0, 160.0)),
        ], 6.0),
        "clock" => {
            let c = r.pick(&[RED, "#2f6fd6", "#f5c842", "#c9ccd1"]);
            (vec![
                metal("Body", c, union(0.0, vec![rcylinder(48.0, 36.0, 8.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 56.0), cone(8.0, 5.0, 14.0).at(30.0, 0.0, 6.0).mirror(X)])),
                p("Face", "#f7f4ea", cylinder(39.0, 4.0).rot(90.0, 0.0, 0.0).at(0.0, -18.0, 56.0)).finish(0.0, 0.3),
                p("Hands", BLACK, union(0.0, vec![cuboid(6.0, 4.0, 32.0).at(0.0, 0.0, 14.0).rot(0.0, 35.0, 0.0), cuboid(5.0, 4.0, 24.0).at(0.0, 0.0, 10.0).rot(0.0, -80.0, 0.0), sphere(5.0)]).at(0.0, -22.0, 56.0)),
                p("Ticks", BLACK, union(0.0, vec![cuboid(6.0, 4.0, 8.0).at(0.0, -21.0, 90.0), cuboid(6.0, 4.0, 8.0).at(0.0, -21.0, 22.0), cuboid(8.0, 4.0, 6.0).at(34.0, -21.0, 56.0).mirror(X)])),
                metal("Bells", GOLD, union(0.0, vec![intersection(0.0, vec![sphere(18.0).at(28.0, 0.0, 106.0), cuboid(60.0, 60.0, 40.0).at(28.0, 0.0, 116.0)]).mirror(X), cap([0.0, 0.0, 100.0], [0.0, 0.0, 118.0], 3.4), sphere(5.0).at(0.0, 0.0, 120.0)])),
            ], 3.0)
        }
        "tree house" => (vec![
            p("Trunk", "#7a4b2a", cone(24.0, 17.0, 200.0).at(0.0, 52.0, 100.0)).finish(0.0, 0.9),
            p("Canopy", LEAF, union(14.0, vec![sphere(52.0).at(0.0, 50.0, 220.0), sphere(40.0).at(40.0, 40.0, 196.0), sphere(40.0).at(-40.0, 44.0, 200.0), sphere(34.0).at(0.0, 20.0, 236.0)])).finish(0.0, 0.8),
            p("Platform", WOOD, rbox(116.0, 104.0, 7.0, 2.0).at(0.0, 2.0, 78.0)).finish(0.0, 0.8),
            p("Cabin", "#b98250", rbox(76.0, 70.0, 52.0, 2.0).at(0.0, -2.0, 108.0)).finish(0.0, 0.8),
            p("Roof", RED, pyramid(100.0, 40.0).at(0.0, -2.0, 154.0)).finish(0.0, 0.6),
            p("Door", "#3a2a20", cuboid(20.0, 4.0, 32.0).at(0.0, -38.0, 100.0)).finish(0.0, 0.8),
            p("Window", "#fff3a0", cuboid(16.0, 4.0, 16.0).at(26.0, -38.0, 112.0)).finish(0.0, 0.2),
            p("Supports", DARK_WOOD, union(0.0, vec![cap([0.0, 48.0, 40.0], [42.0, -18.0, 72.0], 5.0), cap([0.0, 48.0, 40.0], [-42.0, -18.0, 72.0], 5.0)])).finish(0.0, 0.9),
        ], 40.0),
        "skyscraper" => {
            let c = r.pick(&["#5a7fa6", "#6b7a8f", "#4a6a7a", "#7a8794"]);
            (vec![
                glassy("Tower", c, union(0.0, vec![rbox(56.0, 56.0, 210.0, 3.0).at(0.0, 0.0, 105.0), rbox(44.0, 44.0, 42.0, 3.0).at(0.0, 0.0, 231.0), rbox(30.0, 30.0, 32.0, 3.0).at(0.0, 0.0, 268.0)])),
                metal("Floors", "#c9ccd1", array(rbox(60.0, 60.0, 7.0, 1.5).at(0.0, 0.0, 20.0), 10, [0.0, 0.0, 20.0])),
                metal("Spire", DARK_METAL, union(0.0, vec![cap([0.0, 0.0, 280.0], [0.0, 0.0, 330.0], 3.6), cone(8.0, 3.0, 14.0).at(0.0, 0.0, 286.0)])),
                p("Base", "#8d8d88", rbox(86.0, 86.0, 10.0, 2.0).at(0.0, 0.0, 5.0)).finish(0.0, 0.9),
            ], 100.0)
        }
        "sunflower" => {
            let petals: Vec<Node> = (0..14)
                .map(|i| {
                    let a = i as f32 / 14.0 * std::f32::consts::TAU;
                    ellipsoid(42.0, 7.0, 17.0).rot(0.0, -a.to_degrees(), 0.0).at(38.0 * a.cos(), -5.0, 138.0 + 38.0 * a.sin())
                })
                .collect();
            (vec![
                p("Stem", LEAF, union(2.0, vec![cap([0.0, 0.0, 0.0], [0.0, 2.0, 126.0], 5.0), ellipsoid(44.0, 8.0, 20.0).rot(0.0, -28.0, 0.0).at(24.0, 0.0, 48.0), ellipsoid(40.0, 8.0, 18.0).rot(0.0, 28.0, 0.0).at(-22.0, 0.0, 78.0)])).finish(0.0, 0.7),
                p("Disc", "#5a3a22", union(0.0, vec![cylinder(30.0, 16.0).rot(90.0, 0.0, 0.0).at(0.0, -4.0, 138.0), ellipsoid(58.0, 22.0, 58.0).at(0.0, -4.0, 138.0)])).finish(0.0, 0.9),
                p("Petals", "#f5c842", union(0.0, petals)).finish(0.0, 0.6),
            ], 3.0)
        }
        "race car" => {
            let c = r.pick(&[RED, "#f5c842", BLUE, "#2e7d4f", WHITE]);
            let wheel = |x: f32| cylinder(20.0, 22.0).rot(90.0, 0.0, 0.0).at(x, 38.0, 20.0).mirror(Y);
            (vec![
                p("Body", c, union(8.0, vec![ellipsoid(170.0, 52.0, 26.0).at(0.0, 0.0, 28.0), ellipsoid(70.0, 28.0, 20.0).at(-30.0, 0.0, 44.0), cone(18.0, 4.0, 60.0).rot(0.0, 90.0, 0.0).at(92.0, 0.0, 26.0)])).finish(0.4, 0.3),
                glassy("Cockpit", "#1c2a4a", ellipsoid(40.0, 24.0, 18.0).at(-8.0, 0.0, 50.0)),
                p("Wings", BLACK, union(0.0, vec![rbox(20.0, 104.0, 5.0, 2.0).at(108.0, 0.0, 12.0), rbox(30.0, 94.0, 6.0, 2.0).at(-92.0, 0.0, 66.0), cuboid(8.0, 6.0, 40.0).at(-88.0, 14.0, 44.0).mirror(Y)])).finish(0.0, 0.5),
                p("Tires", BLACK, union(0.0, vec![wheel(-62.0), wheel(66.0)])).finish(0.0, 0.9),
                p("Number", WHITE, cylinder(11.0, 3.0).at(20.0, 0.0, 40.0)).finish(0.0, 0.4),
            ], 28.0)
        }
        "water bottle" => {
            let c = r.pick(&["#33c3e0", "#4caf50", "#d63a2f", "#8e44ad", "#c9ccd1"]);
            (vec![
                glassy("Bottle", c, lathe(&[(0.0, 0.0), (26.0, 0.0), (31.0, 6.0), (31.0, 106.0), (28.0, 126.0), (17.0, 138.0), (0.0, 138.0)])),
                p("Cap", "#2a2a2e", union(0.0, vec![cylinder(19.0, 14.0).at(0.0, 0.0, 144.0), cone(12.0, 8.0, 14.0).at(0.0, 0.0, 158.0), torus(14.0, 4.0).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 170.0)])).finish(0.0, 0.5),
                p("Label", WHITE, cylinder(31.6, 30.0).at(0.0, 0.0, 66.0)).finish(0.0, 0.6),
                p("Label Stripe", "#2f6fd6", cylinder(32.2, 8.0).at(0.0, 0.0, 66.0)).finish(0.0, 0.6),
            ], 2.0)
        }
        _ => return None,
    })
}

/// Guitar or violin: a figure-eight body, neck and head.
fn stringed(r: &mut Rng, violin: bool) -> Parts {
    let s = if violin { 0.8 } else { 1.0 };
    let wood = if violin { "#8b3a1e" } else { r.pick(&["#c8843c", "#a0522d", "#d9a441", "#6b4423"]) };
    let mut parts = vec![
        p("Body", wood, union(9.0, vec![ellipsoid(84.0 * s, 32.0, 76.0), ellipsoid(62.0 * s, 32.0, 56.0).at(0.0, 0.0, 46.0)]).at(0.0, 0.0, 40.0)).finish(0.0, 0.3),
        p("Neck", DARK_WOOD, union(0.0, vec![cuboid(15.0, 10.0, 104.0).at(0.0, 0.0, 152.0), rbox(22.0, 8.0, 36.0, 3.0).at(0.0, 0.0, 214.0)])).finish(0.0, 0.5),
        p("Fingerboard", BLACK, cuboid(12.0, 4.0, 100.0).at(0.0, -6.0, 152.0)).finish(0.0, 0.5),
        metal("Tuning Pegs", GOLD, union(0.0, vec![sphere(4.6).at(15.0, 0.0, 204.0).mirror(X), sphere(4.6).at(15.0, 0.0, 214.0).mirror(X), sphere(4.6).at(15.0, 0.0, 224.0).mirror(X)])),
        p("Bridge", BLACK, cuboid(34.0 * s, 7.0, 8.0).at(0.0, -17.0, 20.0)).finish(0.0, 0.5),
    ];
    if violin {
        parts.push(p("F-Holes", BLACK, union(0.0, vec![ellipsoid(6.0, 5.0, 28.0).at(15.0, -16.0, 58.0).mirror(X)])));
        parts.push(p("Chin Rest", BLACK, ellipsoid(30.0, 22.0, 12.0).at(-18.0, -12.0, 12.0)));
        parts.push(p("Scroll", wood, sphere(10.0).at(0.0, 0.0, 236.0)).finish(0.0, 0.3));
    } else {
        parts.push(p("Sound Hole", BLACK, cylinder(15.0, 5.0).rot(90.0, 0.0, 0.0).at(0.0, -15.0, 60.0)));
        parts.push(p("Pickguard", WHITE, ellipsoid(34.0, 5.0, 24.0).at(20.0, -15.5, 36.0)).finish(0.0, 0.5));
    }
    parts
}

// ───────────────────────────── feature modifiers ─────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Wheels,
    Legs,
    Wings,
    Thrusters,
    Horns,
    Spikes,
    Robotic,
    Antenna,
    Crystal,
    Glow,
    Cute,
    Armor,
    Crown,
    Halo,
    Pedestal,
}

/// (words, kind, ambiguous): ambiguous words are also nouns ("a crown", "a jet"), so they only modify when
/// another noun follows them.
const KIND_WORDS: &[(&[&str], Kind, bool)] = &[
    (&["wheeled", "wheels", "wheel"], Kind::Wheels, false),
    (&["legged", "legs", "walking", "walker", "strider"], Kind::Legs, false),
    (&["winged", "wings", "flying", "flies"], Kind::Wings, false),
    (&["thruster", "thrusters", "jet", "jets", "booster", "boosters", "jetpowered", "rocketpowered"], Kind::Thrusters, true),
    (&["horned", "horns", "horn", "antlered", "antlers", "demonic", "devil"], Kind::Horns, false),
    (&["spiky", "spiked", "spikes", "spike", "spiny", "thorny"], Kind::Spikes, false),
    (&["robotic", "cyber", "cybernetic", "mecha", "mechanical", "futuristic", "techno", "steampunk"], Kind::Robotic, false),
    (&["antenna", "antennas", "antennae", "antennaed"], Kind::Antenna, true),
    (&["crystal", "crystals", "crystalline", "gemstone"], Kind::Crystal, true),
    (&["glowing", "glow", "glows", "neon", "luminous", "bioluminescent", "radiant"], Kind::Glow, false),
    (&["cute", "chibi", "kawaii", "adorable"], Kind::Cute, false),
    (&["armored", "armoured", "armor", "armour", "plated"], Kind::Armor, false),
    (&["crowned", "crown"], Kind::Crown, true),
    (&["halo", "haloed"], Kind::Halo, true),
    (&["pedestal", "podium", "plinth"], Kind::Pedestal, true),
];

/// Things that already fly or float: "flying" does not add wings to them.
const AIRBORNE: &[&str] = &[
    "ufo", "saucer", "spaceship", "ship", "plane", "airplane", "rocket", "helicopter", "drone", "jet", "shuttle", "balloon", "carpet",
    "bird", "eagle", "owl", "parrot", "swan", "duck", "dragon", "butterfly", "hoverboard", "cloud", "kite", "angel", "fairy", "bee",
    "penguin", "flamingo", "chicken",
];

fn stop_word(w: &str) -> bool {
    matches!(w, "a" | "an" | "the" | "with" | "and" | "of" | "my" | "some" | "that" | "is" | "in" | "on" | "very" | "really" | "super" | "little" | "tiny" | "small" | "big" | "large" | "huge")
}

fn kind_of(w: &str) -> Option<(Kind, bool)> {
    KIND_WORDS.iter().find(|(ws, ..)| ws.contains(&w)).map(|(_, k, amb)| (*k, *amb))
}

/// An adjective-like modifier ("winged", "robotic") that should never be taken for the noun.
pub fn is_modifier_word(w: &str) -> bool {
    kind_of(w).is_some_and(|(_, amb)| !amb)
}

/// Which feature modifiers the words ask for, in the order they are built. With `as_feature` (the words after
/// "with": "a cat with wings and a halo") every keyword counts, otherwise ambiguous ones need a noun after them.
pub fn detect(words: &[String], as_feature: bool) -> Vec<Kind> {
    let last = words.iter().rposition(|w| !stop_word(w) && w.parse::<f32>().is_err());
    let mut out: Vec<Kind> = Vec::new();
    for (i, w) in words.iter().enumerate() {
        let w = w.as_str();
        let Some((k, amb)) = kind_of(w) else { continue };
        if amb && !as_feature && Some(i) == last {
            continue;
        }
        if as_feature && k == Kind::Crown {
            continue; // `templates.rs` has a real crown that accessorise() fits to the head
        }
        if k == Kind::Wings && w == "flying" && words.iter().any(|x| AIRBORNE.contains(&x.as_str())) {
            continue;
        }
        if k == Kind::Cute && as_feature {
            continue;
        }
        if !out.contains(&k) {
            out.push(k);
        }
    }
    const ORDER: [Kind; 15] = [
        Kind::Wheels, Kind::Legs, Kind::Wings, Kind::Thrusters, Kind::Horns, Kind::Spikes, Kind::Robotic, Kind::Antenna,
        Kind::Crystal, Kind::Glow, Kind::Cute, Kind::Armor, Kind::Crown, Kind::Halo, Kind::Pedestal,
    ];
    ORDER.iter().copied().filter(|k| out.contains(k)).collect()
}

/// Measurements of what is built so far, so features can be fitted to it.
struct Ctx {
    bb: Aabb,
    cs: Vec<Compiled>,
    /// Rotation about Z that turns the local frame (front +X, sides ±Y) into the object's own.
    yaw: f32,
    l: f32,
    w: f32,
    h: f32,
    md: f32,
    cx: f32,
    cy: f32,
    top: Vec3,
    hw: f32,
}

fn measure(parts: &[Part]) -> Ctx {
    let cs: Vec<Compiled> = parts.iter().map(|p| Compiled::new(&p.shape)).collect();
    let bb = cs.iter().fold(Aabb::EMPTY, |b, c| b.union(&tight_aabb(c)));
    let s = bb.size();
    let c = bb.center();
    let yaw = if s.x > s.y * 1.25 { 0.0 } else { -90.0 };
    let (l, w) = if yaw == 0.0 { (s.x, s.y) } else { (s.y, s.x) };
    let md = s.max_element().max(1.0);
    let (top, hw) = summit(&cs).unwrap_or((Vec3::new(c.x, c.y, bb.max.z), s.x));
    Ctx { bb, cs, yaw, l, w, h: s.z, md, cx: c.x, cy: c.y, top, hw: hw.clamp(0.1 * md, 0.7 * md) }
}

fn lift(parts: &mut [Part], dz: f32) {
    for p in parts {
        p.shape.at[2] += dz;
    }
}

/// Top surface at (x, y), searching down from `hi`.
fn top_z(cs: &[Compiled], x: f32, y: f32, hi: f32, lo: f32) -> Option<f32> {
    let step = ((hi - lo) / 200.0).max(0.2);
    let mut z = hi;
    while z >= lo {
        let q = Vec3::new(x, y, z);
        if cs.iter().any(|c| c.eval(q) < 0.0) {
            return Some(z + step * 0.5);
        }
        z -= step;
    }
    None
}

/// A group in the local frame placed at `at` and turned by the object's yaw.
fn place(children: Vec<Node>, at: Vec3, yaw: f32) -> Node {
    union(0.0, children).rot(0.0, 0.0, yaw).at(at.x, at.y, at.z)
}

/// Attaches real 3D features for the modifier words ("winged", "horned", "wheeled", "robotic", "crystal", "cute",
/// "armored", "crowned", "halo", "antenna", "jet", "pedestal"...) found in `words`. Returns how many it attached.
pub fn attach_modifiers(parts: &mut Vec<Part>, words: &[String], rng: &mut Rng) -> usize {
    attach(parts, &detect(words, false), rng)
}

/// Same for the words after "with": "a cat with wings", "a robot with a halo and antenna".
pub fn attach_features(parts: &mut Vec<Part>, words: &[String], rng: &mut Rng) -> usize {
    attach(parts, &detect(words, true), rng)
}

fn has(parts: &[Part], needle: &str) -> bool {
    parts.iter().any(|p| p.name.to_lowercase().contains(needle))
}

fn attach(parts: &mut Vec<Part>, kinds: &[Kind], rng: &mut Rng) -> usize {
    let mut done = 0;
    for &kind in kinds {
        if parts.len() + 4 > MAX_PARTS - 6 || parts.is_empty() {
            break;
        }
        let c = measure(parts);
        if c.bb.is_empty() {
            break;
        }
        let before = parts.len();
        add_kind(parts, kind, &c, rng);
        if parts.len() > before || kind == Kind::Crystal {
            done += 1;
        }
    }
    done
}

fn add_kind(parts: &mut Vec<Part>, kind: Kind, c: &Ctx, rng: &mut Rng) {
    let (md, hw) = (c.md, c.hw);
    let foot = Vec3::new(c.cx, c.cy, 0.0);
    let min_z = c.bb.min.z;
    match kind {
        Kind::Wheels => {
            if has(parts, "wheel") || has(parts, "tire") {
                return;
            }
            let rad = (0.11 * md).clamp(4.0, 0.3 * c.h.max(8.0));
            let tw = (0.5 * rad).max(0.03 * md);
            lift(parts, 1.2 * rad);
            let (x, y) = (0.3 * c.l, 0.5 * c.w);
            let wheel = |x: f32| rcylinder(rad, tw, tw * 0.3).rot(90.0, 0.0, 0.0).at(x, y, min_z + rad - foot.z).mirror(Y);
            let hub = |x: f32| cylinder(0.45 * rad, tw + 0.04 * md).rot(90.0, 0.0, 0.0).at(x, y, min_z + rad - foot.z).mirror(Y);
            let tires = place(vec![wheel(x), wheel(-x)], foot, c.yaw);
            let hubs = place(vec![hub(x), hub(-x)], foot, c.yaw);
            parts.push(p("Wheels", "#2a2a2e", tires).finish(0.0, 0.9));
            parts.push(metal("Hubcaps", METAL, hubs));
        }
        Kind::Legs => {
            if has(parts, "leg") {
                return;
            }
            let lg = (0.28 * c.h).max(0.1 * md);
            let rl = (0.045 * md).max(0.1 * c.w.min(c.l));
            lift(parts, lg);
            let (x, y) = (0.28 * c.l, 0.3 * c.w);
            let top = min_z + lg + 0.25 * c.h;
            let leg = union(2.0, vec![cap([x, y, top], [x * 1.15, y * 1.3, min_z + rl], rl), sphere(rl * 1.5).at(x * 1.15, y * 1.3, min_z + rl * 1.2)]);
            let legs = place(vec![union(0.0, vec![leg.mirror(X)]).mirror(Y)], foot, c.yaw);
            let mech = parts[0].metallic > 0.5;
            parts.push(Part { name: "Legs".into(), color: parts[0].color.clone(), metallic: if mech { 1.0 } else { 0.0 }, roughness: 0.5, shape: legs });
        }
        Kind::Wings => {
            if has(parts, "wing") || has(parts, "rotor") {
                return;
            }
            let s = (0.62 * md).max(30.0);
            let t = (0.035 * md).max(2.5);
            let poly: Vec<(f32, f32)> = [(0.10, 0.0), (0.14, 0.40), (0.04, 0.78), (-0.14, 1.0), (-0.20, 0.74), (-0.17, 0.60), (-0.30, 0.58), (-0.28, 0.40), (-0.20, 0.28), (-0.26, 0.14), (-0.18, 0.0)]
                .iter()
                .map(|&(x, y)| (x * s, y * s))
                .collect();
            let root = Vec3::new(-0.05 * c.l, 0.2 * c.w, 0.64 * c.h + min_z - foot.z);
            let wing = extrude(&poly, t).rot(38.0, 0.0, 0.0).at(root.x, root.y, root.z).mirror(Y);
            let col = rng.pick(&[WHITE, "#f7e7c6", "#e9d5ff", "#cfe8f2"]);
            parts.push(p("Wings", col, place(vec![wing], foot, c.yaw)).finish(0.0, 0.5));
        }
        Kind::Thrusters => {
            if has(parts, "thruster") {
                return;
            }
            let rn = (0.07 * md).max(3.0);
            let (x, y, z) = (-0.5 * c.l, 0.22 * c.w, 0.45 * c.h + min_z - foot.z);
            let nozzle = cone(0.62 * rn, rn, 2.0 * rn).rot(0.0, -90.0, 0.0).at(x - rn * 0.4, y, z).mirror(Y);
            let flame = cone(0.8 * rn, 0.0, 3.6 * rn).rot(0.0, -90.0, 0.0).at(x - rn * 2.6, y, z).mirror(Y);
            parts.push(metal("Thrusters", DARK_METAL, place(vec![nozzle], foot, c.yaw)));
            parts.push(glow("Thruster Flames", "#ff9f2e", place(vec![flame], foot, c.yaw)));
        }
        Kind::Horns => {
            if has(parts, "horn") || has(parts, "antler") {
                return;
            }
            let k = (0.55 * hw).clamp(0.07 * md, 0.25 * md);
            let rr = (0.2 * k).max(0.022 * md);
            let b0 = [0.0, 0.28 * hw, -0.1 * hw];
            let b1 = [0.0, 0.28 * hw + 0.45 * k, -0.1 * hw + 0.65 * k];
            let b2 = [0.0, b1[1] - 0.12 * k, b1[2] + 0.75 * k];
            let horn = union(0.0, vec![cap(b0, b1, rr), cap(b1, b2, 0.78 * rr), cone(0.74 * rr, 0.0, 0.9 * k).at(b2[0], b2[1], b2[2] + 0.4 * k)]).mirror(Y);
            parts.push(p("Horns", BONE, place(vec![horn], c.top, c.yaw)).finish(0.0, 0.5));
        }
        Kind::Spikes => {
            if has(parts, "spike") {
                return;
            }
            let rs = (0.04 * md).max(2.0);
            let hs = 0.15 * md;
            let mut cones = Vec::new();
            for i in 0..5 {
                let lx = (-0.36 + 0.18 * i as f32) * c.l;
                let (a, b) = (c.yaw.to_radians().cos(), c.yaw.to_radians().sin());
                let (wx, wy) = (c.cx + lx * a, c.cy + lx * b);
                if let Some(z) = top_z(&c.cs, wx, wy, c.bb.max.z + 1.0, c.bb.min.z) {
                    cones.push(cone(rs, 0.0, hs).at(wx, wy, z + 0.32 * hs));
                }
            }
            if !cones.is_empty() {
                parts.push(p("Spikes", "#3a3a3d", union(0.0, cones)).finish(0.3, 0.4));
            }
        }
        Kind::Robotic => {
            if !has(parts, "visor") {
                let vis = ellipsoid(0.42 * hw, 0.8 * hw, 0.28 * hw).at(0.34 * hw, 0.0, -0.42 * hw);
                parts.push(glow("Visor", CYAN, place(vec![vis], c.top, c.yaw)));
            }
            if !has(parts, "antenna") {
                let top = 0.55 * hw + 0.04 * md;
                let ra = (0.022 * md).max(1.6);
                let mast = cap([-0.05 * hw, 0.0, -0.1 * hw], [-0.05 * hw, 0.0, top], ra);
                parts.push(metal("Antenna", DARK_METAL, place(vec![mast], c.top, c.yaw)));
                parts.push(glow("Antenna Light", RED, place(vec![sphere(2.2 * ra).at(-0.05 * hw, 0.0, top + ra)], c.top, c.yaw)));
            }
            if parts[0].metallic < 0.5 {
                parts[0].metallic = 0.8;
                parts[0].roughness = 0.35;
            }
        }
        Kind::Antenna => {
            if has(parts, "antenna") {
                return;
            }
            let top = 0.6 * hw + 0.05 * md;
            let ra = (0.022 * md).max(1.6);
            let mast = cap([0.0, 0.0, -0.1 * hw], [0.0, 0.0, top], ra);
            parts.push(metal("Antenna", DARK_METAL, place(vec![mast], c.top, c.yaw)));
            parts.push(glow("Antenna Light", RED, place(vec![sphere(2.4 * ra).at(0.0, 0.0, top + ra)], c.top, c.yaw)));
        }
        Kind::Crystal => {
            let shard = |len: f32, rad: f32, tilt_deg: f32, az: f32, off: f32| {
                let a = az.to_radians();
                let t = tilt_deg.to_radians();
                let axis = Vec3::new(t.sin() * a.cos(), t.sin() * a.sin(), t.cos());
                let base = Vec3::new(off * a.cos(), off * a.sin(), -0.1 * hw);
                let ctr = base + axis * (0.5 * len);
                union(0.0, vec![cylinder(rad, len), cone(rad, 0.0, 0.7 * rad * 2.0).at(0.0, 0.0, 0.5 * len + 0.7 * rad)]).rot(0.0, tilt_deg, az).at(ctr.x, ctr.y, ctr.z)
            };
            let l0 = (0.7 * hw).max(0.12 * md);
            let rc = (0.2 * l0).max(0.02 * md);
            let nodes = vec![
                shard(l0 * 1.5, rc * 1.3, 0.0, 0.0, 0.0),
                shard(l0, rc, 28.0, 20.0, 0.25 * hw),
                shard(l0 * 0.9, rc, 30.0, 140.0, 0.25 * hw),
                shard(l0 * 0.8, rc * 0.9, 34.0, 250.0, 0.25 * hw),
                shard(l0 * 0.6, rc * 0.8, 40.0, 320.0, 0.3 * hw),
            ];
            parts.push(glassy("Crystals", "#8ff0ff", place(nodes, c.top, c.yaw)));
            parts[0].roughness = 0.1;
            if parts[0].color.is_some() {
                parts[0].color = Some("#9be7ff".into());
            }
        }
        Kind::Glow => {
            if has(parts, "glow") {
                return;
            }
            let ro = (0.05 * md).max(2.0);
            let rr = 0.5 * c.l.max(c.w) + 0.14 * md;
            let col = rng.pick(&["#39ffb0", "#ff3df2", "#4de1ff", "#ffe14d"]);
            let orbs: Vec<Node> = (0..3)
                .map(|i| {
                    let a = (i as f32 * 120.0 + 30.0).to_radians();
                    sphere(ro * (1.0 - 0.15 * i as f32)).at(rr * a.cos(), rr * a.sin(), min_z + (0.3 + 0.28 * i as f32) * c.h - foot.z)
                })
                .collect();
            parts.push(glow("Glow Orbs", col, place(orbs, foot, 0.0)));
        }
        Kind::Cute => {
            let big = has(parts, "eye");
            if !big {
                let er = 0.17 * hw;
                let eyes = sphere(er).at(0.4 * hw, 0.27 * hw, -0.42 * hw).mirror(Y);
                let shine = sphere(0.42 * er).at(0.4 * hw + 0.6 * er, 0.27 * hw - 0.1 * er, -0.42 * hw + 0.45 * er).mirror(Y);
                parts.push(p("Cute Eyes", BLACK, place(vec![eyes], c.top, c.yaw)).finish(0.0, 0.1));
                parts.push(p("Eye Shine", WHITE, place(vec![shine], c.top, c.yaw)).finish(0.0, 0.1));
            }
            let blush = ellipsoid(0.1 * hw, 0.22 * hw, 0.12 * hw).at(0.4 * hw, 0.42 * hw, -0.66 * hw).mirror(Y);
            parts.push(p("Blush", "#ff8fb1", place(vec![blush], c.top, c.yaw)).finish(0.0, 0.8));
        }
        Kind::Armor => {
            if has(parts, "armor") {
                return;
            }
            let mut plates = Vec::new();
            let (a, b) = (c.yaw.to_radians().cos(), c.yaw.to_radians().sin());
            for i in 0..5 {
                let lx = (-0.34 + 0.17 * i as f32) * c.l;
                let (wx, wy) = (c.cx + lx * a, c.cy + lx * b);
                if let Some(z) = top_z(&c.cs, wx, wy, c.bb.max.z + 1.0, c.bb.min.z) {
                    plates.push(ellipsoid(0.2 * c.l, 0.6 * c.w, 0.12 * c.h.max(0.1 * md)).at(lx, 0.0, z - min_z * 0.0 - 0.02 * c.h));
                }
            }
            let pauldron = sphere((0.13 * md).max(3.0)).at(0.0, 0.5 * c.w, 0.72 * c.h + min_z).mirror(Y);
            plates.push(pauldron);
            parts.push(metal("Armor", "#b8bcc4", place(plates, Vec3::new(c.cx, c.cy, 0.0), c.yaw)));
        }
        Kind::Crown => {
            if has(parts, "crown") {
                return;
            }
            let rb = 0.5 * hw;
            let band = cylinder(rb, 0.22 * hw).at(0.0, 0.0, 0.1 * hw);
            let tips = ring_of(5, 0.78 * rb, 0.34 * hw, |_| cone(0.2 * rb, 0.0, 0.5 * hw));
            let mut nodes = vec![band];
            nodes.extend(tips);
            parts.push(metal("Crown", GOLD, place(nodes, c.top, c.yaw)));
        }
        Kind::Halo => {
            if has(parts, "halo") {
                return;
            }
            let maj = (0.55 * hw).max(0.1 * md);
            let minr = (0.07 * hw).max(0.022 * md);
            parts.push(glow("Halo", "#ffe680", place(vec![torus(maj, minr).at(0.0, 0.0, 0.6 * hw + 0.05 * md)], c.top, c.yaw)));
        }
        Kind::Pedestal => {
            if has(parts, "pedestal") {
                return;
            }
            let d = c.l.max(c.w).max(0.5 * md);
            let (h1, h2) = (0.07 * md, 0.07 * md);
            lift(parts, h1 + h2);
            let base = union(0.0, vec![rcylinder(0.78 * d, h1, 0.3 * h1).at(c.cx, c.cy, min_z + h1 / 2.0 - (h1 + h2)), rcylinder(0.64 * d, h2, 0.3 * h2).at(c.cx, c.cy, min_z + h1 + h2 / 2.0 - (h1 + h2))]);
            parts.push(p("Pedestal", "#e8e6df", base).finish(0.0, 0.3));
        }
    }
}

// ───────────────────────────── fallback sculpture ─────────────────────────────

/// For words that mean nothing to the planner: not a lump but an original, symmetric multi-feature sculpture
/// (sculpted core, side fins or wings, a ring, a crest and feet), seeded by the word so it is always the same.
/// It stays one part (one colour) so the host can treat it as one object to sculpt from.
pub fn fallback_sculpture(r: &mut Rng) -> Parts {
    let colour = r.pick(&["#e9a8c8", "#8fd3c8", "#f0b86e", "#9db4ee", "#c3a6e6", "#e98f7e", "#a8d58a", "#e8d27a"]);
    let style = r.next() % 3;
    let hc = r.range(86.0, 120.0);
    let rc = r.range(24.0, 34.0);
    let mut nodes: Vec<Node> = Vec::new();
    // Core.
    match style {
        0 => {
            let prof = [(0.0, 0.0), (rc * 0.9, 0.0), (rc * 1.2, hc * 0.22), (rc * 0.85, hc * 0.5), (rc * 1.1, hc * 0.74), (rc * 0.6, hc * 0.94), (0.0, hc)];
            nodes.push(lathe(&prof));
        }
        1 => nodes.push(ellipsoid(rc * 2.2, rc * 2.0, hc).at(0.0, 0.0, hc / 2.0)),
        _ => {
            nodes.push(cone(rc * 1.2, rc * 0.5, hc * 0.55).at(0.0, 0.0, hc * 0.275));
            nodes.push(sphere(rc * 1.05).at(0.0, 0.0, hc * 0.62));
        }
    }
    // Side fins, mirrored.
    let pairs = 1 + (r.next() % 2) as usize;
    for k in 0..pairs {
        let z = hc * (0.35 + 0.28 * k as f32) + r.range(-4.0, 4.0);
        let span = r.range(0.9, 1.5) * rc;
        let tiltx = r.range(10.0, 40.0);
        nodes.push(ellipsoid(r.range(0.5, 0.9) * rc, 2.0 * span + rc * 0.9, rc * 0.32).rot(tiltx, 0.0, 0.0).at(0.0, rc * 0.7 + span * 0.35, z).mirror(Y));
    }
    // A ring around the core.
    if r.unit() < 0.7 {
        nodes.push(torus(rc * 1.15, (rc * 0.17).max(3.2)).at(0.0, 0.0, hc * r.range(0.4, 0.62)));
    }
    // Crest on top.
    let prongs = 3 + (r.next() % 3) as usize;
    let crest_len = rc * r.range(1.1, 1.8);
    nodes.extend(ring_of(prongs, rc * 0.34, hc * 0.9, |deg| cone(rc * 0.3, 0.0, crest_len).rot(0.0, 14.0, deg)));
    // Feet.
    nodes.extend(ring_of(3, rc * 0.85, rc * 0.25, |_| sphere(rc * 0.42)));
    vec![p("Sculpture", colour, union(rc * 0.3, nodes)).finish(0.0, 0.5)]
}

// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::{generate, GenOptions};
    use super::*;
    use std::collections::HashMap;

    fn closed(m: &crate::format::Mesh) -> bool {
        let mut e: HashMap<(u32, u32), i32> = HashMap::new();
        for f in &m.faces {
            for k in 0..f.len() {
                *e.entry((f[k], f[(k + 1) % f.len()])).or_default() += 1;
            }
        }
        e.iter().all(|(&(a, b), &n)| n == 1 && e.get(&(b, a)) == Some(&1))
    }

    fn db() -> GenOptions {
        GenOptions { max_triangles: 8_000, quads: false, ..GenOptions::default() }
    }

    fn check(prompt: &str) -> super::super::Generated {
        let g = generate(prompt, &db()).unwrap_or_else(|e| panic!("{prompt}: {e}"));
        assert!(g.understood, "{prompt}: not understood, notes {:?}", g.notes);
        assert!(g.triangles() > 100 && g.triangles() <= 8_000, "{prompt}: {} triangles", g.triangles());
        for p in &g.parts {
            assert!(closed(&p.mesh), "{prompt}: part {} is not closed", p.name);
            assert!(p.mesh.positions.iter().all(|v| v.is_finite()));
        }
        g
    }

    #[test]
    fn everyday_objects_are_multi_part_and_watertight() {
        for w in ["a guitar", "a spaceship", "a dinosaur", "headphones", "a laptop", "a pizza"] {
            let g = check(w);
            assert!(g.parts.len() >= 2, "{w}: {} parts", g.parts.len());
            let g2 = generate(w, &GenOptions { detail: 40, ..GenOptions::default() }).unwrap();
            assert!(g2.parts.iter().all(|p| closed(&p.mesh)), "{w}: not closed at detail 40");
        }
        let g = check("two headphones");
        assert!(g.parts.len() >= 6, "plural headphones make several sets: {:?}", g.parts.iter().map(|p| &p.name).collect::<Vec<_>>());
    }

    #[test]
    fn compound_words_and_stems_resolve() {
        for (w, root) in [
            ("hoverboard", "hoverboard"), ("lighthouse", "lighthouse"), ("treehouse", "tree house"), ("racecar", "race car"),
            ("skyscraper", "skyscraper"), ("snowboard", "snowboard"), ("sunflower", "sunflower"), ("armchair", "armchair"),
            ("battleaxe", "battleaxe"), ("waterbottle", "water bottle"), ("spaceship", "spaceship"), ("headphones", "headset"),
            ("firetruck", "truck"), ("cheesecake", "cake"), ("seahorse", "horse"), ("teacup", "cup"), ("houseboat", "boat"),
        ] {
            assert_eq!(resolve(w).as_deref(), Some(root), "{w}");
        }
        assert_eq!(resolve_phrase("palm", "tree").as_deref(), Some("palm tree"));
        assert_eq!(resolve_phrase("coffee", "maker").as_deref(), Some("coffee maker"));
        assert!(resolve("zorblax").is_none() && resolve("zorbak").is_none() && resolve("flibbertigibbet").is_none());
        for prompt in ["a hoverboard", "a lighthouse", "a treehouse", "cybernetic shark", "medieval helmet", "alien guitar", "crystal jellyfish", "floating spaceship", "a firetruck"] {
            let g = check(prompt);
            assert!(g.parts.len() >= 2, "{prompt}");
        }
    }

    #[test]
    fn feature_modifiers_attach_real_parts() {
        let names = |prompt: &str| -> Vec<String> { check(prompt).parts.iter().map(|p| p.name.clone()).collect() };
        let n = names("a winged robot");
        assert!(n.iter().any(|x| x.ends_with("Wings")), "{n:?}");
        let n = names("a horned helmet");
        assert!(n.iter().any(|x| x.ends_with("Horns")), "{n:?}");
        let n = names("a robotic dragon");
        assert!(n.iter().any(|x| x.ends_with("Visor")) && n.iter().any(|x| x.ends_with("Antenna")), "{n:?}");
        let n = names("a wheeled cat");
        assert!(n.iter().any(|x| x.ends_with("Wheels")), "{n:?}");
        let n = names("a spiky helmet");
        assert!(n.iter().any(|x| x.ends_with("Spikes")), "{n:?}");
        let n = names("a cat with wings");
        assert!(n.iter().any(|x| x.ends_with("Wings")), "{n:?}");
        let n = names("a crowned frog with a halo");
        assert!(n.iter().any(|x| x.ends_with("Crown")) && n.iter().any(|x| x.ends_with("Halo")), "{n:?}");
        for w in ["a legged car", "a crystal robot", "a cute robot", "an armored horse", "a pedestal vase", "a jet duck", "a glowing mug"] {
            check(w);
        }
        // Nouns that are also modifiers stay nouns when alone.
        let n = names("a crown");
        assert!(!n.iter().any(|x| x.ends_with("Crown") && x != "Crown"), "{n:?}");
        let g = check("a flying saucer");
        assert!(!g.parts.iter().any(|p| p.name.ends_with("Wings")));
    }

    #[test]
    fn every_template_is_watertight_within_the_8k_budget() {
        let names = own_names();
        assert!(names.len() >= 60, "{} templates", names.len());
        for w in &names {
            let g = check(&format!("a {w}"));
            let p = super::super::text::plan(&format!("three {w}"), 0);
            assert!(p.program.validate().is_ok(), "three {w}: {:?}", p.program.validate());
            assert!(g.parts.len() >= 1);
        }
    }

    #[test]
    fn invented_words_get_an_expressive_sculpture_but_stay_not_understood() {
        let g = generate("a zorbak", &GenOptions { detail: 48, ..GenOptions::default() }).unwrap();
        assert!(!g.understood && g.notes[0].contains("zorbak"));
        assert!(g.triangles() > 300);
        assert!(g.parts.iter().all(|p| closed(&p.mesh)));
        let again = generate("a zorbak", &GenOptions { detail: 48, ..GenOptions::default() }).unwrap();
        assert_eq!(g, again, "deterministic");
        let other = generate("a quuxlen", &GenOptions { detail: 48, ..GenOptions::default() }).unwrap();
        assert_ne!(g.program, other.program, "another word, another sculpture");
        for w in ["zorbak", "quuxlen", "flibbertigibbet", "snorglefrump"] {
            let g = generate(&format!("a {w}"), &db()).unwrap();
            assert!(g.triangles() <= 8_000 && g.parts.iter().all(|p| closed(&p.mesh)), "{w}");
        }
    }
}
