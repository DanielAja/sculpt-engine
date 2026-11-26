//! Compact 430+ entry Visual Object Lexicon for universal offline Text-to-3D.
//!
//! Maps everyday English nouns across kitchen, home, furniture, tech, clothing, sports, tools,
//! food, music, vehicles, buildings, landmarks, animals, plants, science, gaming, and sci-fi/fantasy
//! into 27 structured multi-part SDF assemblies (proportions, parts, realistic color, metallic/roughness,
//! and life-size scale) in <0.1 ms with zero network calls and minimal binary footprint.

use super::program::{Axis::*, *};
use super::text::Rng;

#[derive(Clone, Copy)]
enum Form {
    /// Wide flat pan/paddle + long handle (frying pan, spatula, tennis racket, magnifying glass)
    PaddleHandle,
    /// Elongated slim tool/utensil (spoon, fork, knife, pen, pencil, wand, wrench, lightsaber, flute)
    SlimTool,
    /// Boxy appliance with front panel/dial/door (microwave, fridge, oven, washer, printer, amplifier)
    Appliance,
    /// Flat screen/frame on stand (monitor, tv, tablet, mirror, picture frame, chalkboard, solar panel)
    DisplayStand,
    /// Handheld electronics/controller (gamepad, remote, calculator, boombox, walkie-talkie, wallet)
    Handheld,
    /// Torso/garment silhouette (t-shirt, hoodie, jacket, dress, pants, backpack)
    Garment,
    /// Cylindrical container/vessel with rim/lid (kettle, thermos, soda can, cauldron, barrel, boba)
    Vessel,
    /// Stemmed glass / trophy / chalice / goblet / lightbulb / hourglass
    StemmedVessel,
    /// Multi-part wheeled vehicle (tractor, ambulance, tank, forklift, bulldozer, fire truck, carriage)
    UtilityVehicle,
    /// Floating/flying craft (kayak, yacht, jet, blimp, ufo, starfighter, pirate ship, hoverboard)
    Craft,
    /// Organic fruit/produce/food (avocado, lemon, croissant, taco, watermelon, pumpkin, sushi)
    FoodItem,
    /// Sports ball / planetary sphere with seam/ring (basketball, globe, moon, sun, orb, crystal ball)
    SportSphere,
    /// Ring/torus/crown/headwear (cap, beanie, halo, tire, wreath, necklace, compass, gear, donut)
    RingWear,
    /// Long musical instrument (saxophone, trumpet, cello, harp, banjo, ukulele, accordion)
    Instrument,
    /// Architectural structure (cabin, barn, gazebo, fountain, igloo, tent, greenhouse, fireplace)
    Structure,
    /// Tall spire / tower / monument (eiffel tower, obelisk, big ben, lighthouse, clock tower, lamppost)
    TowerSpire,
    /// Gateway / arch / portal / bridge frame (torii, portal, arch, golden gate, swing, goalpost)
    ArchPortal,
    /// Long shaft weapon/tool with shaped head (katana, spear, trident, pickaxe, shovel, broom, mace)
    PoleTool,
    /// Furniture piece with legs/cushions (ottoman, nightstand, dresser, throne, bench, bathtub, sink)
    FurniturePiece,
    /// Optical / medical / science gear (microscope, binoculars, stethoscope, vr headset, turret, radar)
    Device,
    /// Mythical / character figure (astronaut, alien, wizard, knight, ninja, samurai, mech, statue)
    Character,
    /// Four-legged creature (deer, panda, giraffe, gorilla, rabbit, crocodile, squirrel, kangaroo)
    Quadruped,
    /// Winged animal / insect (parrot, flamingo, swan, peacock, bat, bee, dragonfly, butterfly)
    WingedCreature,
    /// Aquatic life (dolphin, seahorse, crab, lobster, jellyfish, starfish, coral, seashell)
    Aquatic,
    /// Plant / flower / botanical (rose, sunflower, lotus, tulip, fern, bonsai, bamboo, shrub, acorn)
    PlantFlora,
    /// Chest / crate / book / hinged container (treasure chest, spellbook, briefcase, dice, rubiks cube)
    BoxChest,
    /// Footwear / skate / board (boot, sneaker, roller skate, ice skate, snowboard, surfboard, skis)
    Footwear,
}

struct Entry {
    name: &'static str,
    form: Form,
    color: &'static str,
    accent: &'static str,
    metallic: f32,
    roughness: f32,
    stretch: [f32; 3],
    real_scale: f32,
}

const fn e(
    name: &'static str,
    form: Form,
    color: &'static str,
    accent: &'static str,
    metallic: f32,
    roughness: f32,
    stretch: [f32; 3],
    real_scale: f32,
) -> Entry {
    Entry { name, form, color, accent, metallic, roughness, stretch, real_scale }
}

const LEXICON: &[Entry] = &[
    // ── Kitchenware & dining ───────────────────────────────────────────────
    e("frying pan", Form::PaddleHandle, "#3a3d42", "#6e472b", 0.85, 0.35, [1.2, 1.2, 0.45], 2.5),
    e("skillet", Form::PaddleHandle, "#282a2e", "#282a2e", 0.85, 0.45, [1.2, 1.2, 0.48], 2.5),
    e("pan", Form::PaddleHandle, "#444850", "#1e1e20", 0.85, 0.32, [1.15, 1.15, 0.45], 2.5),
    e("wok", Form::PaddleHandle, "#32353b", "#8b5a2b", 0.85, 0.35, [1.35, 1.35, 0.65], 2.8),
    e("spatula", Form::PaddleHandle, "#c9ccd1", "#2c2f36", 0.9, 0.25, [0.55, 0.85, 0.22], 2.0),
    e("ladle", Form::PaddleHandle, "#c9ccd1", "#1e1e20", 0.9, 0.25, [0.65, 0.65, 0.65], 2.0),
    e("spoon", Form::SlimTool, "#d4d7dc", "#c9ccd1", 1.0, 0.18, [0.45, 0.35, 1.0], 1.5),
    e("fork", Form::SlimTool, "#d4d7dc", "#c9ccd1", 1.0, 0.18, [0.45, 0.30, 1.0], 1.5),
    e("knife", Form::SlimTool, "#d9dde2", "#5c3b24", 1.0, 0.18, [0.35, 0.25, 1.15], 1.8),
    e("chopsticks", Form::SlimTool, "#b88652", "#8b5a2b", 0.0, 0.55, [0.22, 0.22, 1.35], 1.6),
    e("whisk", Form::SlimTool, "#c9ccd1", "#444850", 0.95, 0.25, [0.6, 0.6, 1.0], 1.8),
    e("rolling pin", Form::SlimTool, "#b88652", "#8b5a2b", 0.0, 0.68, [0.8, 0.8, 1.3], 2.5),
    e("cutting board", Form::DisplayStand, "#b88652", "#8b5a2b", 0.0, 0.65, [1.3, 0.2, 0.95], 2.5),
    e("kettle", Form::Vessel, "#c9ccd1", "#1e1e20", 0.95, 0.20, [1.15, 1.15, 0.95], 2.0),
    e("coffee maker", Form::Appliance, "#22252b", "#cfe8f2", 0.35, 0.26, [0.9, 0.85, 1.35], 2.8),
    e("espresso machine", Form::Appliance, "#c9ccd1", "#1e1e20", 0.85, 0.20, [1.1, 0.95, 1.2], 3.0),
    e("toaster", Form::Appliance, "#d4d7dc", "#22252b", 0.85, 0.20, [1.15, 0.72, 0.78], 2.2),
    e("pitcher", Form::Vessel, "#cfe8f2", "#a8d8ea", 0.0, 0.08, [0.9, 0.9, 1.35], 2.0),
    e("thermos", Form::Vessel, "#2f6fd6", "#c9ccd1", 0.8, 0.28, [0.65, 0.65, 1.5], 2.0),
    e("flask", Form::Vessel, "#c9ccd1", "#7a4424", 0.95, 0.22, [0.85, 0.42, 1.15], 1.4),
    e("canteen", Form::Vessel, "#4a6338", "#2c2f36", 0.3, 0.55, [0.95, 0.52, 1.15], 1.6),
    e("water bottle", Form::Vessel, "#38bdf8", "#1e1e20", 0.1, 0.22, [0.62, 0.62, 1.55], 1.8),
    e("soda can", Form::Vessel, "#d63a2f", "#c9ccd1", 0.85, 0.24, [0.68, 0.68, 1.15], 1.0),
    e("can", Form::Vessel, "#c9ccd1", "#d63a2f", 0.85, 0.25, [0.7, 0.7, 1.1], 1.0),
    e("jar", Form::Vessel, "#cfe8f2", "#e6b84d", 0.0, 0.08, [0.82, 0.82, 1.18], 1.5),
    e("bucket", Form::Vessel, "#2f6fd6", "#c9ccd1", 0.1, 0.40, [1.05, 1.05, 1.12], 2.5),
    e("blender", Form::Vessel, "#d9dde2", "#1e1e20", 0.3, 0.22, [0.85, 0.85, 1.65], 2.5),
    e("wine glass", Form::StemmedVessel, "#cfe8f2", "#8e2438", 0.0, 0.05, [0.8, 0.8, 1.3], 1.6),
    e("martini", Form::StemmedVessel, "#cfe8f2", "#9ccc3d", 0.0, 0.05, [1.0, 1.0, 1.2], 1.5),
    e("cocktail", Form::StemmedVessel, "#f2735f", "#f5c842", 0.0, 0.08, [0.95, 0.95, 1.25], 1.5),
    e("champagne", Form::StemmedVessel, "#f5e398", "#cfe8f2", 0.0, 0.06, [0.62, 0.62, 1.55], 1.6),
    e("goblet", Form::StemmedVessel, "#e6b84d", "#b3202f", 1.0, 0.22, [0.95, 0.95, 1.15], 1.6),
    e("chalice", Form::StemmedVessel, "#e6b84d", "#36c9c0", 1.0, 0.20, [0.95, 0.95, 1.2], 1.6),
    e("colander", Form::Vessel, "#c9ccd1", "#8d9299", 0.9, 0.30, [1.35, 1.35, 0.65], 2.2),
    e("cauldron", Form::Vessel, "#22252b", "#4caf50", 0.85, 0.50, [1.35, 1.35, 0.95], 3.8),
    e("barrel", Form::Vessel, "#8b5a2b", "#3a3d42", 0.0, 0.72, [1.05, 1.05, 1.25], 5.0),
    e("keg", Form::Vessel, "#c9ccd1", "#3a3d42", 0.95, 0.28, [1.0, 1.0, 1.25], 4.5),

    // ── Home appliances & furniture ────────────────────────────────────────
    e("microwave", Form::Appliance, "#e8eaed", "#1e1e20", 0.2, 0.32, [1.35, 0.95, 0.85], 3.5),
    e("oven", Form::Appliance, "#d4d7dc", "#1e1e20", 0.6, 0.30, [1.0, 1.0, 1.2], 5.0),
    e("stove", Form::Appliance, "#d4d7dc", "#1e1e20", 0.6, 0.30, [1.0, 1.0, 1.2], 5.0),
    e("fridge", Form::Appliance, "#e8eaed", "#b8bcc4", 0.5, 0.25, [0.95, 0.9, 1.95], 8.0),
    e("refrigerator", Form::Appliance, "#e8eaed", "#b8bcc4", 0.5, 0.25, [0.95, 0.9, 1.95], 8.0),
    e("washer", Form::Appliance, "#f4f4f2", "#3a3d42", 0.15, 0.28, [1.0, 1.0, 1.25], 5.0),
    e("washing machine", Form::Appliance, "#f4f4f2", "#3a3d42", 0.15, 0.28, [1.0, 1.0, 1.25], 5.0),
    e("dryer", Form::Appliance, "#f4f4f2", "#3a3d42", 0.15, 0.28, [1.0, 1.0, 1.25], 5.0),
    e("dishwasher", Form::Appliance, "#d4d7dc", "#2c2f36", 0.6, 0.28, [1.0, 1.0, 1.2], 5.0),
    e("vacuum", Form::PoleTool, "#d63a2f", "#3a3d42", 0.2, 0.35, [0.9, 0.9, 1.2], 4.0),
    e("fan", Form::Device, "#f4f4f2", "#444850", 0.3, 0.32, [1.0, 0.7, 1.35], 3.5),
    e("heater", Form::Appliance, "#3a3d42", "#f28c28", 0.4, 0.45, [0.95, 0.55, 1.15], 3.0),
    e("air conditioner", Form::Appliance, "#f4f4f2", "#38bdf8", 0.1, 0.32, [1.55, 0.75, 0.72], 4.0),
    e("trash can", Form::Vessel, "#6f7377", "#3a3d42", 0.7, 0.40, [0.95, 0.95, 1.35], 3.5),
    e("bin", Form::Vessel, "#2f6fd6", "#1e1e20", 0.0, 0.42, [0.95, 0.95, 1.25], 3.0),
    e("broom", Form::PoleTool, "#b88652", "#e6c87a", 0.0, 0.80, [0.85, 0.5, 1.5], 5.0),
    e("broomstick", Form::PoleTool, "#7a4d2b", "#d8b468", 0.0, 0.78, [0.85, 0.5, 1.65], 5.5),
    e("mop", Form::PoleTool, "#2f6fd6", "#e8e8e4", 0.0, 0.78, [0.85, 0.75, 1.5], 5.0),
    e("bathtub", Form::FurniturePiece, "#f4f4f2", "#c9ccd1", 0.0, 0.18, [1.65, 0.85, 0.65], 8.0),
    e("toilet", Form::FurniturePiece, "#f4f4f2", "#d9dde2", 0.0, 0.18, [0.75, 1.05, 1.1], 4.5),
    e("sink", Form::FurniturePiece, "#f4f4f2", "#c9ccd1", 0.0, 0.18, [1.0, 0.85, 1.05], 4.5),
    e("ottoman", Form::FurniturePiece, "#8e44ad", "#5c3b24", 0.0, 0.85, [1.0, 1.0, 0.6], 3.5),
    e("nightstand", Form::FurniturePiece, "#8b5a2b", "#e6b84d", 0.0, 0.68, [0.9, 0.8, 1.0], 4.0),
    e("dresser", Form::FurniturePiece, "#8b5a2b", "#e6b84d", 0.0, 0.68, [1.45, 0.85, 1.3], 6.0),
    e("wardrobe", Form::FurniturePiece, "#754c29", "#e6b84d", 0.0, 0.68, [1.25, 0.85, 1.95], 8.0),
    e("cabinet", Form::FurniturePiece, "#8b5a2b", "#c9ccd1", 0.0, 0.65, [1.15, 0.8, 1.3], 5.0),
    e("bookshelf", Form::FurniturePiece, "#8b5a2b", "#d63a2f", 0.0, 0.68, [1.2, 0.55, 1.9], 7.0),
    e("desk", Form::FurniturePiece, "#a0703f", "#2c2f36", 0.0, 0.62, [1.65, 0.9, 1.0], 7.0),
    e("bench", Form::FurniturePiece, "#8b5a2b", "#2c2f36", 0.0, 0.68, [1.75, 0.68, 0.85], 6.5),
    e("throne", Form::FurniturePiece, "#e6b84d", "#b3202f", 0.9, 0.24, [1.15, 1.05, 1.85], 7.5),
    e("crib", Form::FurniturePiece, "#e8dcc0", "#7cc4f2", 0.0, 0.65, [1.4, 0.9, 1.1], 6.0),
    e("hammock", Form::Craft, "#f2735f", "#8b5a2b", 0.0, 0.88, [1.7, 0.75, 0.55], 7.0),
    e("fireplace", Form::Structure, "#7d4537", "#f28c28", 0.0, 0.88, [1.4, 0.7, 1.2], 7.0),
    e("mirror", Form::DisplayStand, "#d9eef5", "#e6b84d", 0.9, 0.05, [0.9, 0.25, 1.35], 4.0),
    e("picture frame", Form::DisplayStand, "#8b5a2b", "#f4ecd6", 0.0, 0.55, [1.1, 0.22, 0.85], 2.5),
    e("chalkboard", Form::DisplayStand, "#234232", "#a0703f", 0.0, 0.88, [1.55, 0.22, 1.05], 5.0),
    e("whiteboard", Form::DisplayStand, "#f4f4f2", "#c9ccd1", 0.1, 0.22, [1.55, 0.22, 1.05], 5.0),
    e("easel", Form::DisplayStand, "#a0703f", "#f4ecd6", 0.0, 0.68, [1.0, 0.65, 1.6], 5.5),
    e("chandelier", Form::StemmedVessel, "#e6b84d", "#fff6d6", 1.0, 0.18, [1.4, 1.4, 1.1], 5.0),
    e("pillow", Form::FoodItem, "#f4f4f2", "#7cc4f2", 0.0, 0.94, [1.35, 0.95, 0.45], 2.2),
    e("cushion", Form::FoodItem, "#4b3fa8", "#e6b84d", 0.0, 0.92, [1.15, 1.15, 0.42], 2.0),
    e("rug", Form::DisplayStand, "#b3202f", "#e6b84d", 0.0, 0.95, [1.8, 1.3, 0.08], 6.0),
    e("carpet", Form::DisplayStand, "#284b7a", "#e6b84d", 0.0, 0.95, [1.9, 1.4, 0.08], 6.5),
    e("curtain", Form::Garment, "#8e44ad", "#e6b84d", 0.0, 0.90, [1.1, 0.35, 1.85], 6.5),
    e("iron", Form::Craft, "#f4f4f2", "#38bdf8", 0.5, 0.22, [1.25, 0.65, 0.72], 1.8),
    e("sewing machine", Form::Appliance, "#f4f4f2", "#e86f24", 0.3, 0.28, [1.35, 0.75, 1.0], 2.8),

    // ── Tech, office & electronics ─────────────────────────────────────────
    e("monitor", Form::DisplayStand, "#22252b", "#38bdf8", 0.3, 0.25, [1.45, 0.35, 0.95], 3.5),
    e("tv", Form::DisplayStand, "#1c1e22", "#38bdf8", 0.2, 0.22, [1.75, 0.28, 1.05], 6.0),
    e("television", Form::DisplayStand, "#1c1e22", "#38bdf8", 0.2, 0.22, [1.75, 0.28, 1.05], 6.0),
    e("tablet", Form::DisplayStand, "#2c2f36", "#38bdf8", 0.6, 0.20, [1.15, 0.15, 0.82], 2.0),
    e("ipad", Form::DisplayStand, "#c9ccd1", "#38bdf8", 0.7, 0.20, [1.15, 0.15, 0.82], 2.0),
    e("smartphone", Form::Handheld, "#1e1e20", "#38bdf8", 0.6, 0.15, [0.58, 0.16, 1.15], 1.1),
    e("iphone", Form::Handheld, "#2c2f36", "#38bdf8", 0.75, 0.14, [0.58, 0.16, 1.15], 1.1),
    e("keyboard", Form::Handheld, "#2c2f36", "#e8eaed", 0.2, 0.38, [1.75, 0.65, 0.22], 2.5),
    e("mouse", Form::Handheld, "#e8eaed", "#2c2f36", 0.1, 0.28, [0.65, 1.05, 0.42], 1.0),
    e("printer", Form::Appliance, "#f0f1f4", "#2c2f36", 0.1, 0.38, [1.35, 1.05, 0.72], 3.0),
    e("router", Form::Appliance, "#1e1e20", "#33c3e0", 0.1, 0.35, [1.2, 0.85, 0.38], 1.8),
    e("remote", Form::Handheld, "#22252b", "#d63a2f", 0.0, 0.42, [0.42, 1.35, 0.28], 1.2),
    e("gamepad", Form::Handheld, "#2c2f36", "#38bdf8", 0.1, 0.35, [1.35, 0.85, 0.45], 1.5),
    e("controller", Form::Handheld, "#f4f4f2", "#2f6fd6", 0.1, 0.32, [1.35, 0.85, 0.45], 1.5),
    e("gameboy", Form::Handheld, "#d8d6cf", "#8e44ad", 0.0, 0.38, [0.75, 0.32, 1.25], 1.2),
    e("vr headset", Form::Device, "#f0f1f4", "#1c1e22", 0.25, 0.20, [1.25, 0.85, 0.72], 1.8),
    e("vision pro", Form::Device, "#e6e8ec", "#1a1d24", 0.65, 0.12, [1.25, 0.82, 0.68], 1.8),
    e("earbuds", Form::Handheld, "#f4f4f2", "#c9ccd1", 0.1, 0.18, [0.75, 0.45, 0.85], 0.8),
    e("airpods", Form::Handheld, "#f4f4f2", "#c9ccd1", 0.1, 0.18, [0.75, 0.45, 0.85], 0.8),
    e("smart speaker", Form::Vessel, "#2c2f36", "#38bdf8", 0.1, 0.75, [0.95, 0.95, 1.05], 1.5),
    e("homepod", Form::Vessel, "#1e1e20", "#8e44ad", 0.1, 0.78, [0.95, 0.95, 1.05], 1.5),
    e("boombox", Form::Appliance, "#2c2f36", "#c9ccd1", 0.4, 0.35, [1.75, 0.65, 0.95], 3.2),
    e("turntable", Form::Appliance, "#8b5a2b", "#1e1e20", 0.2, 0.35, [1.4, 1.15, 0.45], 3.0),
    e("record player", Form::Appliance, "#7a4424", "#1e1e20", 0.2, 0.35, [1.4, 1.15, 0.45], 3.0),
    e("vinyl", Form::RingWear, "#1c1e22", "#d63a2f", 0.1, 0.18, [1.45, 1.45, 0.14], 2.2),
    e("cassette", Form::Handheld, "#2c2f36", "#f5c842", 0.0, 0.38, [1.2, 0.22, 0.78], 1.0),
    e("projector", Form::Appliance, "#f0f1f4", "#38bdf8", 0.2, 0.32, [1.2, 1.0, 0.55], 2.2),
    e("webcam", Form::Device, "#1e1e20", "#38bdf8", 0.25, 0.25, [1.0, 0.65, 0.65], 1.0),
    e("tripod", Form::TowerSpire, "#22252b", "#c9ccd1", 0.75, 0.32, [0.9, 0.9, 1.65], 5.0),
    e("calculator", Form::Handheld, "#3a3d42", "#98e2c6", 0.0, 0.42, [0.75, 1.15, 0.25], 1.2),
    e("walkie talkie", Form::Handheld, "#2c2f36", "#f28c28", 0.0, 0.48, [0.55, 0.42, 1.4], 1.6),
    e("usb", Form::Handheld, "#c9ccd1", "#2f6fd6", 0.7, 0.25, [0.42, 0.22, 0.95], 0.6),
    e("flash drive", Form::Handheld, "#22252b", "#c9ccd1", 0.7, 0.25, [0.42, 0.22, 0.95], 0.6),
    e("hard drive", Form::BoxChest, "#2c2f36", "#38bdf8", 0.6, 0.28, [0.85, 1.2, 0.28], 1.2),
    e("gpu", Form::Appliance, "#22252b", "#4caf50", 0.55, 0.28, [1.75, 0.65, 0.38], 2.0),
    e("microchip", Form::BoxChest, "#1e2e24", "#e6b84d", 0.5, 0.28, [1.1, 1.1, 0.22], 0.8),
    e("power bank", Form::Handheld, "#2c2f36", "#38bdf8", 0.4, 0.30, [0.72, 1.25, 0.30], 1.2),
    e("solar panel", Form::DisplayStand, "#1a3566", "#c9ccd1", 0.35, 0.16, [1.65, 0.18, 1.1], 6.0),
    e("notebook", Form::BoxChest, "#2f6fd6", "#f4ecd6", 0.0, 0.72, [0.9, 1.2, 0.25], 1.8),
    e("clipboard", Form::DisplayStand, "#a0703f", "#f4f4f2", 0.0, 0.60, [0.95, 0.18, 1.3], 2.0),
    e("envelope", Form::BoxChest, "#f4ecd6", "#d63a2f", 0.0, 0.82, [1.35, 0.95, 0.12], 1.4),
    e("pen", Form::SlimTool, "#2f6fd6", "#e6b84d", 0.3, 0.25, [0.22, 0.22, 1.2], 1.2),
    e("pencil", Form::SlimTool, "#f5c842", "#f48fb1", 0.0, 0.65, [0.20, 0.20, 1.2], 1.2),
    e("marker", Form::SlimTool, "#d63a2f", "#1e1e20", 0.0, 0.35, [0.30, 0.30, 1.1], 1.1),
    e("crayon", Form::SlimTool, "#e86f24", "#1e1e20", 0.0, 0.65, [0.28, 0.28, 0.95], 0.9),
    e("paintbrush", Form::SlimTool, "#a0703f", "#38bdf8", 0.0, 0.58, [0.26, 0.22, 1.35], 1.5),
    e("palette", Form::PaddleHandle, "#c88e4c", "#e86f24", 0.0, 0.55, [1.15, 0.95, 0.16], 2.0),
    e("ruler", Form::SlimTool, "#e6c87a", "#22252b", 0.0, 0.55, [0.35, 0.12, 1.6], 2.0),
    e("scissors", Form::SlimTool, "#c9ccd1", "#d63a2f", 0.9, 0.25, [0.55, 0.22, 1.15], 1.5),
    e("stapler", Form::Handheld, "#2c2f36", "#c9ccd1", 0.4, 0.35, [0.45, 1.35, 0.48], 1.4),

    // ── Sports, fitness & games ────────────────────────────────────────────
    e("tennis racket", Form::PaddleHandle, "#4caf50", "#f4f4f2", 0.2, 0.40, [0.95, 1.25, 0.22], 3.5),
    e("racket", Form::PaddleHandle, "#2f6fd6", "#f4f4f2", 0.2, 0.40, [0.95, 1.25, 0.22], 3.5),
    e("ping pong paddle", Form::PaddleHandle, "#d63a2f", "#b88652", 0.0, 0.55, [0.95, 1.05, 0.22], 2.0),
    e("basketball", Form::SportSphere, "#e86f24", "#1e1e20", 0.0, 0.72, [1.0, 1.0, 1.0], 2.0),
    e("soccer ball", Form::SportSphere, "#f4f4f2", "#1e1e20", 0.0, 0.48, [1.0, 1.0, 1.0], 2.0),
    e("football", Form::SportSphere, "#7a4424", "#f4f4f2", 0.0, 0.65, [1.45, 0.85, 0.85], 2.0),
    e("volleyball", Form::SportSphere, "#f4f4f2", "#2f6fd6", 0.0, 0.55, [1.0, 1.0, 1.0], 2.0),
    e("tennis ball", Form::SportSphere, "#c6e83a", "#f4f4f2", 0.0, 0.90, [1.0, 1.0, 1.0], 0.8),
    e("baseball", Form::SportSphere, "#f4f4f2", "#d63a2f", 0.0, 0.65, [1.0, 1.0, 1.0], 0.8),
    e("bowling ball", Form::SportSphere, "#4b3fa8", "#1e1e20", 0.15, 0.12, [1.0, 1.0, 1.0], 1.8),
    e("baseball bat", Form::SlimTool, "#b88652", "#22252b", 0.0, 0.52, [0.55, 0.55, 1.65], 4.0),
    e("golf club", Form::PoleTool, "#c9ccd1", "#1e1e20", 0.95, 0.22, [0.65, 0.45, 1.65], 4.5),
    e("hockey stick", Form::PoleTool, "#22252b", "#f4f4f2", 0.15, 0.42, [0.85, 0.30, 1.75], 5.0),
    e("bowling pin", Form::StemmedVessel, "#f4f4f2", "#d63a2f", 0.0, 0.22, [0.75, 0.75, 1.45], 2.5),
    e("dumbbell", Form::SlimTool, "#3a3d42", "#c9ccd1", 0.85, 0.35, [0.85, 0.85, 1.15], 2.2),
    e("barbell", Form::SlimTool, "#22252b", "#c9ccd1", 0.90, 0.30, [0.75, 0.75, 2.0], 6.0),
    e("kettlebell", Form::Vessel, "#22252b", "#3a3d42", 0.75, 0.45, [0.95, 0.95, 1.1], 2.2),
    e("frisbee", Form::RingWear, "#f2735f", "#f5c842", 0.0, 0.32, [1.4, 1.4, 0.32], 2.0),
    e("boomerang", Form::PaddleHandle, "#b87a42", "#e86f24", 0.0, 0.55, [1.35, 0.95, 0.18], 2.2),
    e("kite", Form::DisplayStand, "#f2735f", "#f5c842", 0.0, 0.65, [1.1, 0.14, 1.45], 3.5),
    e("yo-yo", Form::RingWear, "#d63a2f", "#f4f4f2", 0.1, 0.25, [0.85, 0.85, 0.55], 0.8),
    e("surfboard", Form::Footwear, "#38bdf8", "#f5c842", 0.05, 0.18, [0.65, 2.2, 0.20], 8.0),
    e("snowboard", Form::Footwear, "#8e44ad", "#38bdf8", 0.1, 0.22, [0.65, 1.95, 0.18], 6.5),
    e("skis", Form::Footwear, "#d63a2f", "#22252b", 0.2, 0.25, [0.45, 2.1, 0.18], 7.0),
    e("roller skate", Form::Footwear, "#f48fb1", "#38bdf8", 0.1, 0.45, [0.85, 1.15, 1.05], 2.2),
    e("ice skate", Form::Footwear, "#f4f4f2", "#c9ccd1", 0.3, 0.32, [0.75, 1.15, 1.1], 2.2),
    e("globe", Form::SportSphere, "#2f6fd6", "#4caf50", 0.0, 0.35, [1.0, 1.0, 1.15], 2.5),
    e("trophy", Form::StemmedVessel, "#e6b84d", "#5c3b24", 1.0, 0.15, [1.05, 1.05, 1.45], 2.6),
    e("medal", Form::RingWear, "#e6b84d", "#2f6fd6", 1.0, 0.20, [0.9, 0.9, 0.35], 1.0),
    e("dartboard", Form::RingWear, "#1e1e20", "#d63a2f", 0.0, 0.78, [1.35, 1.35, 0.28], 3.0),
    e("dice", Form::BoxChest, "#f4f4f2", "#1e1e20", 0.0, 0.22, [0.85, 0.85, 0.85], 0.7),
    e("d20", Form::SportSphere, "#8e44ad", "#e6b84d", 0.15, 0.20, [0.95, 0.95, 0.95], 0.8),
    e("rubiks cube", Form::BoxChest, "#2f6fd6", "#f5c842", 0.0, 0.28, [0.95, 0.95, 0.95], 1.0),
    e("chess piece", Form::StemmedVessel, "#ede8d8", "#22252b", 0.0, 0.32, [0.68, 0.68, 1.45], 1.2),
    e("pawn", Form::StemmedVessel, "#ede8d8", "#c8b89a", 0.0, 0.35, [0.65, 0.65, 1.2], 1.0),
    e("rook", Form::TowerSpire, "#22252b", "#444850", 0.0, 0.35, [0.75, 0.75, 1.15], 1.1),
    e("queen", Form::StemmedVessel, "#f4f4f2", "#e6b84d", 0.1, 0.28, [0.72, 0.72, 1.55], 1.3),
    e("king", Form::StemmedVessel, "#22252b", "#e6b84d", 0.1, 0.28, [0.72, 0.72, 1.65], 1.4),
    e("domino", Form::BoxChest, "#f4f4f2", "#1e1e20", 0.0, 0.25, [0.62, 1.25, 0.28], 0.9),

    // ── Clothing, wearables & personal accessories ─────────────────────────
    e("t-shirt", Form::Garment, "#2f6fd6", "#f4f4f2", 0.0, 0.88, [1.2, 0.55, 1.15], 3.5),
    e("shirt", Form::Garment, "#f4f4f2", "#2f6fd6", 0.0, 0.85, [1.2, 0.55, 1.15], 3.5),
    e("hoodie", Form::Garment, "#4b3fa8", "#e8eaed", 0.0, 0.90, [1.25, 0.65, 1.25], 3.8),
    e("jacket", Form::Garment, "#3a3d42", "#c9ccd1", 0.0, 0.65, [1.25, 0.65, 1.2], 3.8),
    e("coat", Form::Garment, "#7a4424", "#4a3020", 0.0, 0.82, [1.2, 0.65, 1.55], 4.5),
    e("dress", Form::Garment, "#d6338f", "#f48fb1", 0.0, 0.80, [1.1, 0.65, 1.55], 4.5),
    e("pants", Form::Garment, "#284b7a", "#8b5a2b", 0.0, 0.88, [0.85, 0.55, 1.45], 4.2),
    e("jeans", Form::Garment, "#284b7a", "#d2b48c", 0.0, 0.90, [0.85, 0.55, 1.45], 4.2),
    e("boot", Form::Footwear, "#6b4423", "#22252b", 0.0, 0.62, [0.85, 1.2, 1.2], 2.2),
    e("sandal", Form::Footwear, "#c88e4c", "#7a4424", 0.0, 0.72, [0.85, 1.2, 0.45], 1.8),
    e("cap", Form::RingWear, "#d63a2f", "#f4f4f2", 0.0, 0.85, [1.05, 1.25, 0.68], 1.8),
    e("beanie", Form::RingWear, "#f28c28", "#f4ecd6", 0.0, 0.95, [0.95, 0.95, 0.92], 1.6),
    e("scarf", Form::RingWear, "#b3202f", "#f5c842", 0.0, 0.94, [1.15, 0.85, 0.72], 2.0),
    e("sunglasses", Form::Device, "#1e1e20", "#38bdf8", 0.25, 0.14, [1.45, 0.65, 0.42], 1.2),
    e("glasses", Form::Device, "#2c2f36", "#cfe8f2", 0.35, 0.16, [1.4, 0.65, 0.42], 1.2),
    e("necklace", Form::RingWear, "#e6b84d", "#36c9c0", 1.0, 0.15, [1.15, 1.15, 0.28], 1.2),
    e("bracelet", Form::RingWear, "#e6b84d", "#f48fb1", 1.0, 0.16, [0.95, 0.95, 0.35], 0.9),
    e("earring", Form::RingWear, "#e6b84d", "#38bdf8", 1.0, 0.14, [0.65, 0.65, 0.65], 0.6),
    e("backpack", Form::Garment, "#2f6fd6", "#22252b", 0.0, 0.78, [0.95, 0.75, 1.25], 3.2),
    e("briefcase", Form::BoxChest, "#5c3b24", "#e6b84d", 0.0, 0.50, [1.35, 0.45, 0.95], 3.0),
    e("wallet", Form::BoxChest, "#5c3b24", "#8b5a2b", 0.0, 0.58, [1.15, 0.82, 0.25], 1.0),
    e("umbrella", Form::PoleTool, "#2f6fd6", "#5c3b24", 0.0, 0.55, [1.45, 1.45, 1.45], 4.5),
    e("purse", Form::Vessel, "#8e44ad", "#e6b84d", 0.0, 0.52, [1.1, 0.55, 0.95], 2.0),
    e("handbag", Form::Vessel, "#7a4424", "#e6b84d", 0.0, 0.52, [1.15, 0.55, 0.95], 2.2),
    e("suitcase", Form::Appliance, "#2f6fd6", "#1e1e20", 0.1, 0.38, [0.95, 0.55, 1.4], 4.5),
    e("luggage", Form::Appliance, "#2c2f36", "#c9ccd1", 0.2, 0.38, [0.95, 0.55, 1.4], 4.5),
    e("perfume", Form::StemmedVessel, "#f48fb1", "#e6b84d", 0.1, 0.08, [0.85, 0.65, 1.15], 1.2),
    e("lipstick", Form::Vessel, "#d63a2f", "#e6b84d", 0.4, 0.22, [0.42, 0.42, 1.25], 0.8),
    e("toothbrush", Form::SlimTool, "#38bdf8", "#f4f4f2", 0.0, 0.35, [0.24, 0.22, 1.25], 1.2),
    e("comb", Form::PaddleHandle, "#2c2f36", "#444850", 0.0, 0.38, [0.45, 1.35, 0.16], 1.2),
    e("soap", Form::FoodItem, "#f48fb1", "#f4f4f2", 0.0, 0.38, [1.15, 0.72, 0.38], 0.9),
    e("hourglass", Form::StemmedVessel, "#e6b84d", "#f5dc78", 0.6, 0.18, [0.75, 0.75, 1.45], 1.8),
    e("alarm clock", Form::SportSphere, "#d63a2f", "#e6b84d", 0.5, 0.25, [1.05, 0.65, 1.15], 1.4),
    e("compass", Form::RingWear, "#e6b84d", "#d63a2f", 0.95, 0.20, [1.0, 1.0, 0.35], 1.0),
    e("padlock", Form::BoxChest, "#e6b84d", "#c9ccd1", 0.95, 0.24, [0.85, 0.45, 1.25], 1.2),
    e("lock", Form::BoxChest, "#e6b84d", "#c9ccd1", 0.95, 0.24, [0.85, 0.45, 1.25], 1.2),
    e("key", Form::SlimTool, "#e6b84d", "#d4a438", 1.0, 0.20, [0.45, 0.18, 1.05], 0.9),
    e("anchor", Form::PoleTool, "#3a3d42", "#6f7377", 0.9, 0.42, [1.25, 0.42, 1.45], 5.5),
    e("bell", Form::StemmedVessel, "#e6b84d", "#b08d57", 1.0, 0.18, [1.0, 1.0, 1.15], 1.8),

    // ── Food, drinks & produce ─────────────────────────────────────────────
    e("coffee", Form::Vessel, "#f4f4f2", "#5c3b24", 0.0, 0.22, [0.82, 0.82, 1.08], 1.3),
    e("latte", Form::Vessel, "#e8d2a8", "#f4f4f2", 0.0, 0.24, [0.82, 0.82, 1.15], 1.4),
    e("boba", Form::Vessel, "#d8a878", "#22252b", 0.0, 0.16, [0.78, 0.78, 1.38], 1.6),
    e("smoothie", Form::Vessel, "#d6338f", "#98e2c6", 0.0, 0.20, [0.78, 0.78, 1.35], 1.6),
    e("milkshake", Form::StemmedVessel, "#f48fb1", "#f4f4f2", 0.0, 0.18, [0.82, 0.82, 1.45], 1.7),
    e("beer", Form::Vessel, "#e6a822", "#f4f4f2", 0.0, 0.12, [0.78, 0.78, 1.35], 1.6),
    e("avocado", Form::FoodItem, "#4a7a34", "#7a4424", 0.0, 0.65, [0.85, 0.85, 1.2], 1.0),
    e("lemon", Form::FoodItem, "#f5dc38", "#4caf50", 0.0, 0.45, [1.25, 0.85, 0.85], 0.8),
    e("lime", Form::FoodItem, "#7ec832", "#4caf50", 0.0, 0.45, [1.2, 0.85, 0.85], 0.8),
    e("orange", Form::FoodItem, "#f28c28", "#4caf50", 0.0, 0.50, [1.0, 1.0, 1.0], 0.9),
    e("peach", Form::FoodItem, "#f7a482", "#4caf50", 0.0, 0.72, [1.0, 1.0, 0.96], 0.9),
    e("pear", Form::FoodItem, "#9ccc3d", "#6b4423", 0.0, 0.48, [0.85, 0.85, 1.28], 0.9),
    e("mango", Form::FoodItem, "#f28c28", "#d63a2f", 0.0, 0.38, [1.2, 0.85, 0.95], 1.0),
    e("pineapple", Form::FoodItem, "#e6a822", "#388e3c", 0.0, 0.65, [0.9, 0.9, 1.55], 1.8),
    e("watermelon", Form::FoodItem, "#2e7d32", "#d63a2f", 0.0, 0.35, [1.45, 1.0, 1.0], 2.4),
    e("strawberry", Form::FoodItem, "#d63a2f", "#4caf50", 0.0, 0.42, [0.85, 0.85, 1.12], 0.6),
    e("blueberry", Form::FoodItem, "#284b9e", "#1f2f6b", 0.0, 0.38, [1.0, 1.0, 0.92], 0.5),
    e("coconut", Form::FoodItem, "#6b4423", "#f4f4f2", 0.0, 0.88, [1.0, 1.0, 1.05], 1.2),
    e("cherry", Form::FoodItem, "#b3202f", "#4caf50", 0.0, 0.18, [0.9, 0.9, 0.9], 0.6),
    e("grapes", Form::FoodItem, "#7a2f9e", "#4caf50", 0.0, 0.30, [0.9, 0.9, 1.3], 1.2),
    e("pumpkin", Form::FoodItem, "#e86f24", "#388e3c", 0.0, 0.58, [1.22, 1.22, 0.92], 2.2),
    e("carrot", Form::FoodItem, "#f28c28", "#4caf50", 0.0, 0.62, [0.48, 0.48, 1.65], 1.4),
    e("tomato", Form::FoodItem, "#d63a2f", "#4caf50", 0.0, 0.25, [1.08, 1.08, 0.88], 0.9),
    e("eggplant", Form::FoodItem, "#58247a", "#388e3c", 0.0, 0.22, [0.75, 0.75, 1.45], 1.5),
    e("broccoli", Form::FoodItem, "#367d34", "#7ec850", 0.0, 0.85, [1.05, 1.05, 1.25], 1.4),
    e("corn", Form::FoodItem, "#f5c842", "#4caf50", 0.0, 0.45, [0.55, 0.55, 1.55], 1.5),
    e("bread", Form::FoodItem, "#c88e4c", "#f4ecd6", 0.0, 0.85, [1.45, 0.82, 0.75], 1.8),
    e("baguette", Form::FoodItem, "#c88e4c", "#e8c488", 0.0, 0.82, [2.1, 0.45, 0.45], 2.5),
    e("croissant", Form::FoodItem, "#d89c4a", "#a6642e", 0.0, 0.68, [1.45, 0.78, 0.58], 1.3),
    e("bagel", Form::RingWear, "#c88e4c", "#f4ecd6", 0.0, 0.65, [1.05, 1.05, 0.48], 1.1),
    e("muffin", Form::Vessel, "#c88e4c", "#5c3b24", 0.0, 0.82, [0.95, 0.95, 0.95], 1.0),
    e("cookie", Form::FoodItem, "#c88e4c", "#4a2c18", 0.0, 0.82, [1.15, 1.15, 0.24], 0.9),
    e("pie", Form::Vessel, "#c88e4c", "#b3202f", 0.0, 0.72, [1.45, 1.45, 0.42], 2.0),
    e("chocolate", Form::BoxChest, "#4a2c18", "#7a4424", 0.0, 0.35, [1.35, 0.82, 0.18], 1.2),
    e("ice cream", Form::StemmedVessel, "#f48fb1", "#d89c4a", 0.0, 0.45, [0.78, 0.78, 1.48], 1.4),
    e("popsicle", Form::PaddleHandle, "#f2735f", "#d2b48c", 0.0, 0.22, [0.65, 1.25, 0.28], 1.2),
    e("lollipop", Form::PaddleHandle, "#d6338f", "#f4f4f2", 0.0, 0.15, [0.85, 0.85, 0.35], 1.1),
    e("cheese", Form::FoodItem, "#f5c842", "#e6a822", 0.0, 0.60, [1.3, 0.9, 0.65], 1.4),
    e("steak", Form::FoodItem, "#7a3022", "#d8b498", 0.0, 0.55, [1.35, 0.95, 0.32], 1.6),
    e("waffle", Form::FoodItem, "#d89c4a", "#f5c842", 0.0, 0.72, [1.15, 1.15, 0.25], 1.4),
    e("pancake", Form::FoodItem, "#d89c4a", "#f5c842", 0.0, 0.68, [1.2, 1.2, 0.45], 1.4),
    e("taco", Form::FoodItem, "#e8c454", "#4caf50", 0.0, 0.78, [1.35, 0.62, 0.78], 1.4),
    e("hot dog", Form::FoodItem, "#c88e4c", "#b3202f", 0.0, 0.65, [1.65, 0.62, 0.58], 1.5),
    e("sandwich", Form::BoxChest, "#d8a868", "#4caf50", 0.0, 0.82, [1.15, 1.15, 0.52], 1.4),
    e("fries", Form::Vessel, "#d63a2f", "#f5c842", 0.0, 0.68, [0.92, 0.62, 1.25], 1.4),
    e("dumpling", Form::FoodItem, "#f4ecd6", "#d8c4a0", 0.0, 0.58, [1.15, 0.78, 0.72], 0.9),
    e("ramen", Form::Vessel, "#b3202f", "#f5dc78", 0.0, 0.35, [1.3, 1.3, 0.72], 1.8),
    e("burrito", Form::FoodItem, "#e8d2a8", "#4caf50", 0.0, 0.78, [1.55, 0.58, 0.55], 1.6),
    e("pretzel", Form::RingWear, "#a6642e", "#f4f4f2", 0.0, 0.62, [1.15, 0.95, 0.32], 1.3),
    e("popcorn", Form::Vessel, "#d63a2f", "#fff4c2", 0.0, 0.75, [0.95, 0.95, 1.25], 1.6),
    e("egg", Form::FoodItem, "#f4ecd6", "#f5c842", 0.0, 0.32, [0.78, 0.78, 1.1], 0.7),

    // ── Vehicles, aviation, naval & sci-fi craft ───────────────────────────
    e("tractor", Form::UtilityVehicle, "#388e3c", "#f5c842", 0.25, 0.38, [1.3, 0.95, 1.1], 12.0),
    e("ambulance", Form::UtilityVehicle, "#f4f4f2", "#d63a2f", 0.2, 0.28, [1.55, 0.95, 1.05], 16.0),
    e("fire truck", Form::UtilityVehicle, "#d63a2f", "#c9ccd1", 0.3, 0.24, [1.85, 0.95, 1.15], 20.0),
    e("school bus", Form::UtilityVehicle, "#f5c842", "#1e1e20", 0.2, 0.30, [2.05, 0.95, 1.15], 22.0),
    e("police car", Form::UtilityVehicle, "#1e1e20", "#2f6fd6", 0.35, 0.22, [1.5, 0.95, 0.85], 14.0),
    e("cybertruck", Form::UtilityVehicle, "#c9ccd1", "#22252b", 0.92, 0.22, [1.75, 1.0, 0.92], 16.0),
    e("van", Form::UtilityVehicle, "#e8eaed", "#2f6fd6", 0.25, 0.30, [1.5, 0.95, 1.05], 14.0),
    e("jeep", Form::UtilityVehicle, "#4a6338", "#1e1e20", 0.2, 0.45, [1.35, 0.95, 1.0], 14.0),
    e("tank", Form::UtilityVehicle, "#4a5d3a", "#2c3324", 0.65, 0.55, [1.55, 1.1, 0.95], 20.0),
    e("bulldozer", Form::UtilityVehicle, "#f5c842", "#22252b", 0.45, 0.42, [1.55, 1.05, 1.1], 18.0),
    e("excavator", Form::UtilityVehicle, "#f5c842", "#22252b", 0.45, 0.42, [1.5, 1.0, 1.25], 18.0),
    e("forklift", Form::UtilityVehicle, "#f28c28", "#22252b", 0.4, 0.40, [1.2, 0.9, 1.25], 12.0),
    e("golf cart", Form::UtilityVehicle, "#f4f4f2", "#388e3c", 0.15, 0.32, [1.25, 0.9, 1.05], 10.0),
    e("tram", Form::UtilityVehicle, "#d63a2f", "#f5c842", 0.3, 0.32, [2.1, 0.92, 1.15], 22.0),
    e("carriage", Form::UtilityVehicle, "#8b5a2b", "#e6b84d", 0.1, 0.55, [1.45, 0.95, 1.25], 14.0),
    e("chariot", Form::UtilityVehicle, "#e6b84d", "#b3202f", 0.8, 0.30, [1.3, 0.95, 1.05], 12.0),
    e("wagon", Form::UtilityVehicle, "#d63a2f", "#22252b", 0.2, 0.42, [1.4, 0.85, 0.72], 5.0),
    e("scooter", Form::UtilityVehicle, "#33c3e0", "#22252b", 0.55, 0.28, [1.25, 0.45, 1.15], 5.0),
    e("wheelchair", Form::UtilityVehicle, "#2f6fd6", "#c9ccd1", 0.6, 0.32, [0.95, 1.0, 1.1], 6.0),
    e("wheelbarrow", Form::UtilityVehicle, "#d63a2f", "#8b5a2b", 0.4, 0.48, [1.45, 0.85, 0.85], 6.0),
    e("rover", Form::UtilityVehicle, "#e8eaed", "#e6b84d", 0.65, 0.30, [1.35, 1.0, 1.05], 10.0),
    e("kayak", Form::Craft, "#f28c28", "#1e1e20", 0.0, 0.35, [2.1, 0.55, 0.42], 12.0),
    e("canoe", Form::Craft, "#a0703f", "#6b4423", 0.0, 0.62, [2.1, 0.60, 0.48], 14.0),
    e("yacht", Form::Craft, "#f4f4f2", "#1f2f6b", 0.1, 0.20, [2.0, 0.85, 0.95], 25.0),
    e("pirate ship", Form::Craft, "#6b4423", "#f4ecd6", 0.0, 0.72, [2.0, 0.95, 1.55], 30.0),
    e("galleon", Form::Craft, "#7a4d2b", "#f4ecd6", 0.0, 0.72, [2.0, 0.95, 1.55], 30.0),
    e("battleship", Form::Craft, "#6f7377", "#3a3d42", 0.75, 0.40, [2.3, 0.75, 0.88], 35.0),
    e("jetski", Form::Craft, "#38bdf8", "#1e1e20", 0.15, 0.22, [1.55, 0.72, 0.72], 8.0),
    e("jet", Form::Craft, "#c9ccd1", "#2f6fd6", 0.85, 0.22, [1.8, 1.45, 0.55], 30.0),
    e("biplane", Form::Craft, "#d63a2f", "#f5c842", 0.25, 0.35, [1.65, 1.55, 0.72], 24.0),
    e("blimp", Form::Craft, "#2f6fd6", "#f5c842", 0.1, 0.38, [1.9, 0.95, 0.95], 35.0),
    e("zeppelin", Form::Craft, "#c9ccd1", "#7a4424", 0.4, 0.35, [2.1, 0.9, 0.9], 40.0),
    e("hot air balloon", Form::Craft, "#d63a2f", "#f5c842", 0.0, 0.72, [1.1, 1.1, 1.65], 30.0),
    e("ufo", Form::Craft, "#c9ccd1", "#36c9c0", 0.92, 0.16, [1.55, 1.55, 0.55], 20.0),
    e("flying saucer", Form::Craft, "#c9ccd1", "#36c9c0", 0.92, 0.16, [1.55, 1.55, 0.55], 20.0),
    e("starfighter", Form::Craft, "#e8eaed", "#d63a2f", 0.75, 0.25, [1.85, 1.45, 0.52], 25.0),
    e("hoverboard", Form::Footwear, "#d6338f", "#38bdf8", 0.3, 0.22, [0.65, 1.65, 0.22], 4.0),
    e("jetpack", Form::Device, "#c9ccd1", "#f28c28", 0.85, 0.22, [1.05, 0.72, 1.25], 3.5),
    e("satellite", Form::Craft, "#e6b84d", "#1f4f9e", 0.9, 0.22, [2.2, 0.75, 0.75], 12.0),
    e("space station", Form::Craft, "#e8eaed", "#1f4f9e", 0.8, 0.25, [2.2, 1.4, 0.95], 40.0),

    // ── Tools, hardware, weapons & fantasy items ───────────────────────────
    e("wrench", Form::SlimTool, "#c9ccd1", "#8d9299", 1.0, 0.25, [0.42, 0.22, 1.25], 2.0),
    e("screwdriver", Form::SlimTool, "#d63a2f", "#c9ccd1", 0.4, 0.32, [0.35, 0.35, 1.3], 1.8),
    e("pliers", Form::SlimTool, "#f28c28", "#c9ccd1", 0.5, 0.35, [0.48, 0.25, 1.1], 1.6),
    e("saw", Form::PaddleHandle, "#c9ccd1", "#8b5a2b", 0.9, 0.28, [0.55, 1.4, 0.18], 3.0),
    e("drill", Form::Device, "#f5c842", "#22252b", 0.25, 0.38, [0.95, 0.45, 1.05], 2.0),
    e("anvil", Form::FurniturePiece, "#3a3d42", "#22252b", 0.92, 0.42, [1.55, 0.75, 0.88], 4.5),
    e("shovel", Form::PoleTool, "#a0703f", "#6f7377", 0.4, 0.55, [0.75, 0.35, 1.65], 5.5),
    e("rake", Form::PoleTool, "#a0703f", "#6f7377", 0.4, 0.55, [1.1, 0.35, 1.65], 5.5),
    e("pickaxe", Form::PoleTool, "#8b5a2b", "#6f7377", 0.6, 0.45, [1.2, 0.35, 1.55], 4.5),
    e("katana", Form::PoleTool, "#d9dde2", "#1e1e20", 1.0, 0.14, [0.22, 0.22, 1.85], 5.0),
    e("lightsaber", Form::SlimTool, "#38bdf8", "#c9ccd1", 0.6, 0.12, [0.24, 0.24, 1.85], 5.0),
    e("blaster", Form::Device, "#2c2f36", "#38bdf8", 0.75, 0.28, [1.35, 0.45, 0.85], 2.2),
    e("raygun", Form::Device, "#d63a2f", "#e6b84d", 0.85, 0.20, [1.35, 0.48, 0.88], 2.2),
    e("spear", Form::PoleTool, "#8b5a2b", "#c9ccd1", 0.5, 0.45, [0.35, 0.35, 1.95], 7.0),
    e("trident", Form::PoleTool, "#e6b84d", "#36c9c0", 1.0, 0.20, [0.85, 0.30, 1.85], 7.0),
    e("mace", Form::PoleTool, "#6f7377", "#8b5a2b", 0.9, 0.35, [0.75, 0.75, 1.45], 4.0),
    e("crossbow", Form::PoleTool, "#8b5a2b", "#6f7377", 0.3, 0.55, [1.35, 1.15, 0.5], 4.0),
    e("potion", Form::StemmedVessel, "#d6338f", "#cfe8f2", 0.0, 0.08, [0.95, 0.95, 1.15], 1.2),
    e("elixir", Form::StemmedVessel, "#36c9c0", "#e6b84d", 0.0, 0.08, [0.85, 0.85, 1.25], 1.2),
    e("crystal ball", Form::SportSphere, "#8e44ad", "#e6b84d", 0.1, 0.05, [1.0, 1.0, 1.15], 1.8),
    e("orb", Form::SportSphere, "#38bdf8", "#8e44ad", 0.2, 0.06, [1.0, 1.0, 1.0], 1.6),
    e("amulet", Form::RingWear, "#e6b84d", "#d63a2f", 1.0, 0.16, [0.85, 0.85, 0.35], 0.9),
    e("scroll", Form::SlimTool, "#f4ecd6", "#8b5a2b", 0.0, 0.78, [0.55, 0.55, 1.2], 1.8),
    e("spellbook", Form::BoxChest, "#4b3fa8", "#e6b84d", 0.1, 0.58, [0.95, 1.25, 0.38], 2.0),
    e("grimoire", Form::BoxChest, "#58247a", "#e6b84d", 0.1, 0.58, [0.95, 1.25, 0.42], 2.2),
    e("treasure chest", Form::BoxChest, "#7a4424", "#e6b84d", 0.2, 0.55, [1.35, 0.95, 0.95], 4.5),
    e("chest", Form::BoxChest, "#7a4424", "#e6b84d", 0.2, 0.55, [1.35, 0.95, 0.95], 4.5),
    e("crate", Form::BoxChest, "#a0703f", "#7a4d2b", 0.0, 0.80, [1.05, 1.05, 1.05], 4.0),
    e("coin", Form::RingWear, "#e6b84d", "#f5c842", 1.0, 0.15, [0.85, 0.85, 0.18], 0.5),
    e("gold bar", Form::BoxChest, "#e6b84d", "#f5c842", 1.0, 0.16, [1.45, 0.65, 0.48], 1.4),
    e("magnet", Form::RingWear, "#d63a2f", "#c9ccd1", 0.7, 0.28, [0.95, 0.95, 0.45], 1.2),
    e("gear", Form::RingWear, "#8d9299", "#6f7377", 0.95, 0.30, [1.15, 1.15, 0.35], 1.5),
    e("battery", Form::Vessel, "#22252b", "#e6b84d", 0.65, 0.30, [0.65, 0.65, 1.35], 1.0),
    e("lightbulb", Form::StemmedVessel, "#fff4c2", "#c9ccd1", 0.0, 0.08, [0.85, 0.85, 1.25], 1.0),
    e("flashlight", Form::Device, "#2c2f36", "#fff4c2", 0.7, 0.30, [0.55, 0.55, 1.35], 1.8),
    e("lantern", Form::Device, "#3a3d42", "#f5c842", 0.7, 0.35, [0.85, 0.85, 1.35], 2.2),
    e("torch", Form::PoleTool, "#8b5a2b", "#f28c28", 0.0, 0.65, [0.55, 0.55, 1.55], 3.0),
    e("watering can", Form::Vessel, "#4caf50", "#c9ccd1", 0.2, 0.38, [1.35, 0.85, 1.05], 2.5),

    // ── Science, optical & music ───────────────────────────────────────────
    e("microscope", Form::Device, "#f0f1f4", "#22252b", 0.45, 0.28, [0.85, 0.95, 1.4], 2.8),
    e("binoculars", Form::Device, "#22252b", "#38bdf8", 0.35, 0.35, [1.25, 0.95, 0.65], 1.8),
    e("stethoscope", Form::Device, "#2f6fd6", "#c9ccd1", 0.5, 0.35, [1.05, 0.85, 1.2], 2.5),
    e("megaphone", Form::Device, "#f4f4f2", "#d63a2f", 0.1, 0.32, [1.25, 0.95, 1.0], 2.2),
    e("radar", Form::Device, "#e8eaed", "#38bdf8", 0.6, 0.28, [1.25, 1.25, 1.35], 6.0),
    e("wind turbine", Form::TowerSpire, "#f4f4f2", "#c9ccd1", 0.35, 0.28, [1.25, 0.75, 2.25], 35.0),
    e("turret", Form::Device, "#444850", "#d63a2f", 0.85, 0.32, [1.1, 1.1, 1.2], 6.0),
    e("saxophone", Form::Instrument, "#e6b84d", "#b08d57", 1.0, 0.18, [0.75, 0.65, 1.45], 4.5),
    e("trumpet", Form::Instrument, "#e6b84d", "#d4a438", 1.0, 0.18, [1.45, 0.55, 0.65], 3.2),
    e("flute", Form::SlimTool, "#d9dde2", "#c9ccd1", 1.0, 0.14, [0.22, 0.22, 1.75], 3.8),
    e("cello", Form::Instrument, "#9c4d24", "#1e1e20", 0.0, 0.28, [0.95, 0.55, 1.85], 6.5),
    e("harp", Form::Instrument, "#e6b84d", "#a0703f", 0.6, 0.30, [1.1, 0.45, 1.75], 7.0),
    e("banjo", Form::Instrument, "#d2b48c", "#8b5a2b", 0.2, 0.42, [0.85, 0.35, 1.65], 4.5),
    e("ukulele", Form::Instrument, "#b87a42", "#4a3020", 0.0, 0.45, [0.80, 0.35, 1.35], 3.2),
    e("accordion", Form::Appliance, "#b3202f", "#f4f4f2", 0.15, 0.35, [1.25, 0.75, 1.1], 3.0),
    e("harmonica", Form::Handheld, "#c9ccd1", "#e6b84d", 0.95, 0.18, [1.35, 0.42, 0.32], 1.0),
    e("synthesizer", Form::Handheld, "#22252b", "#38bdf8", 0.25, 0.32, [1.85, 0.85, 0.32], 4.5),
    e("amplifier", Form::Appliance, "#1e1e20", "#e6b84d", 0.2, 0.65, [1.25, 0.75, 1.15], 4.0),
    e("bongos", Form::Vessel, "#a0703f", "#f4ecd6", 0.0, 0.62, [1.35, 0.75, 0.95], 2.5),
    e("xylophone", Form::FurniturePiece, "#f2735f", "#38bdf8", 0.2, 0.38, [1.65, 0.85, 0.55], 3.0),
    e("gong", Form::ArchPortal, "#e6b84d", "#5c3b24", 0.95, 0.22, [1.35, 0.55, 1.45], 6.5),
    e("tambourine", Form::RingWear, "#c88e4c", "#e6b84d", 0.2, 0.45, [1.15, 1.15, 0.35], 1.8),

    // ── Architecture, landmarks & outdoors ─────────────────────────────────
    e("eiffel tower", Form::TowerSpire, "#6b5d52", "#4a3f38", 0.75, 0.45, [1.0, 1.0, 2.35], 40.0),
    e("big ben", Form::TowerSpire, "#c8b48a", "#e6b84d", 0.1, 0.68, [0.78, 0.78, 2.25], 40.0),
    e("obelisk", Form::TowerSpire, "#d8d0be", "#e6b84d", 0.0, 0.65, [0.65, 0.65, 2.25], 30.0),
    e("clock tower", Form::TowerSpire, "#c8b48a", "#f4f4f2", 0.0, 0.70, [0.85, 0.85, 2.1], 35.0),
    e("water tower", Form::TowerSpire, "#a0703f", "#6f7377", 0.25, 0.65, [1.05, 1.05, 1.85], 28.0),
    e("statue of liberty", Form::Character, "#5ca68e", "#e6b84d", 0.5, 0.48, [0.95, 0.95, 1.95], 40.0),
    e("colosseum", Form::RingWear, "#d2c2a2", "#a89878", 0.0, 0.82, [1.65, 1.35, 1.15], 40.0),
    e("sphinx", Form::Quadruped, "#d8c298", "#e6b84d", 0.0, 0.84, [1.65, 0.95, 1.05], 35.0),
    e("taj mahal", Form::Structure, "#f4f4f2", "#e6b84d", 0.0, 0.28, [1.45, 1.45, 1.65], 40.0),
    e("golden gate", Form::ArchPortal, "#c83c2b", "#d64c38", 0.65, 0.40, [2.35, 0.65, 1.45], 40.0),
    e("torii", Form::ArchPortal, "#c82e24", "#1e1e20", 0.0, 0.58, [1.55, 0.55, 1.45], 20.0),
    e("portal", Form::ArchPortal, "#3a3d42", "#38bdf8", 0.35, 0.30, [1.25, 0.45, 1.55], 15.0),
    e("arch", Form::ArchPortal, "#c8c2b4", "#a8a294", 0.0, 0.78, [1.45, 0.55, 1.45], 18.0),
    e("gate", Form::ArchPortal, "#2c2f36", "#e6b84d", 0.85, 0.38, [1.65, 0.35, 1.35], 14.0),
    e("cabin", Form::Structure, "#7a4d2b", "#4a3020", 0.0, 0.82, [1.35, 1.15, 1.1], 25.0),
    e("barn", Form::Structure, "#b3202f", "#f4f4f2", 0.0, 0.78, [1.5, 1.2, 1.25], 30.0),
    e("pagoda", Form::TowerSpire, "#b3202f", "#e6b84d", 0.0, 0.62, [1.15, 1.15, 1.85], 35.0),
    e("gazebo", Form::Structure, "#f4f4f2", "#8b5a2b", 0.0, 0.65, [1.25, 1.25, 1.3], 20.0),
    e("igloo", Form::Structure, "#e8f4fa", "#b8dff2", 0.0, 0.55, [1.35, 1.35, 0.88], 16.0),
    e("tent", Form::Structure, "#e86f24", "#f5c842", 0.0, 0.82, [1.35, 1.2, 1.0], 14.0),
    e("teepee", Form::TowerSpire, "#e8dcc0", "#b3202f", 0.0, 0.85, [1.15, 1.15, 1.55], 18.0),
    e("greenhouse", Form::Structure, "#cfe8f2", "#4caf50", 0.0, 0.14, [1.55, 1.15, 1.25], 24.0),
    e("fountain", Form::Structure, "#c8c6be", "#38bdf8", 0.0, 0.42, [1.35, 1.35, 1.15], 15.0),
    e("well", Form::Structure, "#8d8a82", "#7a4d2b", 0.0, 0.82, [1.05, 1.05, 1.45], 12.0),
    e("statue", Form::Character, "#c8c6be", "#9e9c95", 0.0, 0.48, [0.9, 0.9, 1.55], 15.0),
    e("pillar", Form::TowerSpire, "#e8e4da", "#c8c2b4", 0.0, 0.62, [0.68, 0.68, 1.95], 18.0),
    e("column", Form::TowerSpire, "#e8e4da", "#c8c2b4", 0.0, 0.62, [0.68, 0.68, 1.95], 18.0),
    e("pedestal", Form::TowerSpire, "#e8e4da", "#e6b84d", 0.0, 0.55, [0.95, 0.95, 1.25], 8.0),
    e("lamppost", Form::TowerSpire, "#22252b", "#fff4c2", 0.85, 0.35, [0.55, 0.55, 2.1], 18.0),
    e("streetlamp", Form::TowerSpire, "#22252b", "#fff4c2", 0.85, 0.35, [0.55, 0.55, 2.1], 18.0),
    e("mailbox", Form::Structure, "#2f6fd6", "#d63a2f", 0.4, 0.38, [0.7, 0.95, 1.45], 6.0),
    e("fire hydrant", Form::Structure, "#d63a2f", "#c9ccd1", 0.4, 0.38, [0.75, 0.75, 1.25], 5.0),
    e("hydrant", Form::Structure, "#d63a2f", "#c9ccd1", 0.4, 0.38, [0.75, 0.75, 1.25], 5.0),
    e("traffic light", Form::TowerSpire, "#f5c842", "#d63a2f", 0.3, 0.42, [0.6, 0.6, 1.75], 12.0),
    e("stop sign", Form::PoleTool, "#d63a2f", "#f4f4f2", 0.3, 0.35, [0.95, 0.20, 1.65], 12.0),
    e("ferris wheel", Form::RingWear, "#f4f4f2", "#d63a2f", 0.6, 0.32, [1.55, 0.45, 1.65], 35.0),
    e("ladder", Form::ArchPortal, "#c9ccd1", "#f5c842", 0.75, 0.35, [0.65, 0.25, 1.95], 10.0),
    e("flag", Form::PoleTool, "#c9ccd1", "#2f6fd6", 0.2, 0.65, [1.15, 0.22, 1.75], 12.0),
    e("gift", Form::BoxChest, "#d63a2f", "#f5c842", 0.0, 0.45, [1.0, 1.0, 1.0], 2.0),
    e("present", Form::BoxChest, "#8e44ad", "#f5c842", 0.0, 0.45, [1.0, 1.0, 1.0], 2.0),
    e("campfire", Form::Structure, "#6b4423", "#f28c28", 0.0, 0.85, [1.2, 1.2, 0.85], 5.0),
    e("moon", Form::SportSphere, "#d8d6cf", "#a6a49d", 0.0, 0.92, [1.0, 1.0, 1.0], 5.0),
    e("sun", Form::SportSphere, "#f5c842", "#f28c28", 0.0, 0.25, [1.0, 1.0, 1.0], 5.0),
    e("cloud", Form::FoodItem, "#f4f6fa", "#d8e6f5", 0.0, 0.92, [1.65, 1.0, 0.75], 8.0),
    e("rainbow", Form::ArchPortal, "#f2735f", "#38bdf8", 0.0, 0.35, [1.85, 0.35, 1.15], 12.0),
    e("snowflake", Form::RingWear, "#d9f2ff", "#ffffff", 0.1, 0.18, [1.25, 1.25, 0.22], 1.5),
    e("comet", Form::Craft, "#7cc4f2", "#fff4c2", 0.1, 0.25, [2.0, 0.75, 0.75], 6.0),
    e("island", Form::PlantFlora, "#e6c87a", "#388e3c", 0.0, 0.82, [1.65, 1.65, 1.05], 20.0),

    // ── Animals, birds, marine life & plants ───────────────────────────────
    e("deer", Form::Quadruped, "#a67042", "#5c3b24", 0.0, 0.80, [1.25, 0.85, 1.35], 10.0),
    e("giraffe", Form::Quadruped, "#e6b84d", "#6b4423", 0.0, 0.78, [1.05, 0.75, 1.85], 20.0),
    e("zebra", Form::Quadruped, "#f4f4f2", "#1e1e20", 0.0, 0.78, [1.3, 0.85, 1.15], 12.0),
    e("panda", Form::Quadruped, "#f4f4f2", "#1e1e20", 0.0, 0.88, [1.15, 1.05, 1.05], 9.0),
    e("koala", Form::Quadruped, "#8d9299", "#f4f4f2", 0.0, 0.90, [0.95, 0.95, 1.0], 4.0),
    e("gorilla", Form::Character, "#282a2e", "#52565e", 0.0, 0.85, [1.25, 1.05, 1.15], 10.0),
    e("monkey", Form::Quadruped, "#8b5a2b", "#d8b498", 0.0, 0.82, [1.05, 0.85, 1.05], 5.0),
    e("rabbit", Form::Quadruped, "#f4f4f2", "#f48fb1", 0.0, 0.90, [0.95, 0.85, 1.15], 2.5),
    e("bunny", Form::Quadruped, "#f4f4f2", "#f48fb1", 0.0, 0.90, [0.95, 0.85, 1.15], 2.5),
    e("squirrel", Form::Quadruped, "#a6642e", "#e8c488", 0.0, 0.88, [1.05, 0.75, 1.05], 2.0),
    e("raccoon", Form::Quadruped, "#6f7377", "#1e1e20", 0.0, 0.88, [1.15, 0.85, 0.9], 3.5),
    e("hedgehog", Form::Quadruped, "#7a5c44", "#d8c4a0", 0.0, 0.90, [1.1, 0.95, 0.78], 1.8),
    e("kangaroo", Form::Quadruped, "#c88e4c", "#e8d2a8", 0.0, 0.82, [1.15, 0.85, 1.45], 10.0),
    e("crocodile", Form::Quadruped, "#4a6e3a", "#98b878", 0.0, 0.72, [2.1, 0.85, 0.55], 14.0),
    e("alligator", Form::Quadruped, "#3e5e32", "#88a868", 0.0, 0.72, [2.1, 0.85, 0.55], 14.0),
    e("lizard", Form::Quadruped, "#5cb85c", "#f5c842", 0.0, 0.55, [1.85, 0.75, 0.5], 2.5),
    e("chameleon", Form::Quadruped, "#4caf50", "#38bdf8", 0.0, 0.45, [1.45, 0.75, 0.75], 2.2),
    e("unicorn", Form::Quadruped, "#f4f4f2", "#f48fb1", 0.1, 0.35, [1.35, 0.85, 1.35], 12.0),
    e("parrot", Form::WingedCreature, "#d63a2f", "#2f6fd6", 0.0, 0.65, [0.85, 0.85, 1.2], 2.5),
    e("flamingo", Form::WingedCreature, "#f48fb1", "#d6338f", 0.0, 0.60, [0.85, 0.85, 1.65], 6.0),
    e("swan", Form::WingedCreature, "#f4f4f2", "#f28c28", 0.0, 0.55, [1.25, 0.95, 1.25], 5.5),
    e("peacock", Form::WingedCreature, "#1f6fb2", "#36c9c0", 0.15, 0.42, [1.35, 1.15, 1.35], 6.5),
    e("bat", Form::WingedCreature, "#22252b", "#58247a", 0.0, 0.75, [1.65, 0.85, 0.75], 2.5),
    e("bee", Form::WingedCreature, "#f5c842", "#1e1e20", 0.0, 0.58, [1.15, 0.95, 0.85], 0.9),
    e("dragonfly", Form::WingedCreature, "#36c9c0", "#2f6fd6", 0.3, 0.25, [1.65, 1.35, 0.42], 1.2),
    e("ladybug", Form::FoodItem, "#d63a2f", "#1e1e20", 0.1, 0.20, [1.05, 0.95, 0.68], 0.7),
    e("dolphin", Form::Aquatic, "#5c94c8", "#d9eef5", 0.0, 0.22, [1.95, 0.75, 0.72], 12.0),
    e("seahorse", Form::Aquatic, "#f28c28", "#f5c842", 0.0, 0.48, [0.75, 0.55, 1.55], 2.0),
    e("crab", Form::Aquatic, "#d63a2f", "#f2735f", 0.0, 0.38, [1.45, 1.15, 0.58], 2.2),
    e("lobster", Form::Aquatic, "#b3202f", "#d63a2f", 0.0, 0.32, [1.85, 0.95, 0.55], 3.0),
    e("jellyfish", Form::Aquatic, "#d672f2", "#38bdf8", 0.0, 0.12, [1.0, 1.0, 1.45], 3.5),
    e("starfish", Form::RingWear, "#f2735f", "#f5c842", 0.0, 0.72, [1.35, 1.35, 0.28], 1.8),
    e("coral", Form::PlantFlora, "#f2735f", "#f48fb1", 0.0, 0.82, [1.25, 1.05, 1.25], 3.0),
    e("seashell", Form::FoodItem, "#f4dcd2", "#f48fb1", 0.05, 0.28, [1.15, 1.05, 0.65], 1.2),
    e("shell", Form::FoodItem, "#f4dcd2", "#e8c4b8", 0.05, 0.28, [1.15, 1.05, 0.65], 1.2),
    e("snail", Form::FoodItem, "#a67042", "#d8b498", 0.0, 0.45, [1.25, 0.85, 0.95], 1.2),
    e("rose", Form::PlantFlora, "#d63a2f", "#388e3c", 0.0, 0.58, [0.75, 0.75, 1.45], 2.2),
    e("sunflower", Form::PlantFlora, "#f5c842", "#5c3b24", 0.0, 0.65, [0.95, 0.95, 1.55], 3.2),
    e("tulip", Form::PlantFlora, "#f48fb1", "#4caf50", 0.0, 0.52, [0.72, 0.72, 1.45], 2.2),
    e("lotus", Form::PlantFlora, "#f48fb1", "#4caf50", 0.0, 0.45, [1.35, 1.35, 0.72], 2.0),
    e("daisy", Form::PlantFlora, "#f4f4f2", "#f5c842", 0.0, 0.62, [0.85, 0.85, 1.35], 1.8),
    e("fern", Form::PlantFlora, "#388e3c", "#66bb6a", 0.0, 0.75, [1.35, 1.35, 1.05], 3.0),
    e("bamboo", Form::TowerSpire, "#66bb6a", "#388e3c", 0.0, 0.42, [0.55, 0.55, 2.0], 8.0),
    e("bush", Form::PlantFlora, "#388e3c", "#66bb6a", 0.0, 0.88, [1.35, 1.35, 0.95], 5.0),
    e("shrub", Form::PlantFlora, "#388e3c", "#4caf50", 0.0, 0.88, [1.25, 1.25, 0.95], 4.5),
    e("acorn", Form::FoodItem, "#8b5a2b", "#5c3b24", 0.0, 0.55, [0.85, 0.85, 1.15], 0.7),
    e("pinecone", Form::FoodItem, "#6b4423", "#4a3020", 0.0, 0.85, [0.82, 0.82, 1.32], 1.0),
    e("boulder", Form::FoodItem, "#8d8a82", "#6e6c66", 0.0, 0.94, [1.25, 1.05, 0.95], 8.0),
    e("stump", Form::Vessel, "#7a4d2b", "#c8a472", 0.0, 0.88, [1.15, 1.15, 0.75], 5.0),

    // ── Characters, sci-fi & mythical beings ───────────────────────────────
    e("astronaut", Form::Character, "#f4f4f2", "#e6b84d", 0.15, 0.30, [1.05, 0.95, 1.25], 9.0),
    e("alien", Form::Character, "#6ec84c", "#1e1e20", 0.0, 0.35, [0.85, 0.85, 1.35], 8.0),
    e("wizard", Form::Character, "#4b3fa8", "#e6b84d", 0.0, 0.78, [0.95, 0.95, 1.45], 9.0),
    e("knight", Form::Character, "#b8bcc4", "#d63a2f", 0.95, 0.25, [1.05, 0.95, 1.35], 9.0),
    e("ninja", Form::Character, "#1e1e20", "#b3202f", 0.0, 0.82, [0.9, 0.9, 1.3], 9.0),
    e("samurai", Form::Character, "#8a1f24", "#e6b84d", 0.6, 0.35, [1.05, 0.95, 1.35], 9.0),
    e("pirate", Form::Character, "#5c3b24", "#e6b84d", 0.0, 0.72, [0.95, 0.95, 1.3], 9.0),
    e("angel", Form::Character, "#f4f4f2", "#f5c842", 0.0, 0.35, [1.35, 0.9, 1.35], 9.0),
    e("fairy", Form::Character, "#f48fb1", "#98e2c6", 0.0, 0.32, [1.3, 0.85, 1.15], 3.0),
    e("skeleton", Form::Character, "#ede8d8", "#22252b", 0.0, 0.65, [0.85, 0.8, 1.35], 9.0),
    e("golem", Form::Character, "#7e7c76", "#36c9c0", 0.0, 0.92, [1.35, 1.15, 1.35], 12.0),
    e("mech", Form::Character, "#444850", "#38bdf8", 0.85, 0.28, [1.35, 1.05, 1.55], 20.0),
    e("mecha", Form::Character, "#e8eaed", "#d63a2f", 0.80, 0.24, [1.35, 1.05, 1.55], 20.0),
    e("cyborg", Form::Character, "#8d9299", "#38bdf8", 0.80, 0.25, [1.0, 0.95, 1.35], 9.0),
    e("android", Form::Character, "#e8eaed", "#38bdf8", 0.45, 0.22, [0.95, 0.9, 1.35], 9.0),
    e("mermaid", Form::Character, "#36c9c0", "#f48fb1", 0.1, 0.30, [0.95, 0.95, 1.45], 9.0),
    e("zombie", Form::Character, "#6e8a5c", "#4a3d34", 0.0, 0.85, [0.95, 0.9, 1.3], 9.0),
    e("vampire", Form::Character, "#1e1e20", "#b3202f", 0.0, 0.65, [1.05, 0.9, 1.35], 9.0),
    e("gnome", Form::Character, "#d63a2f", "#2f6fd6", 0.0, 0.75, [0.9, 0.9, 1.0], 3.0),
    e("elf", Form::Character, "#388e3c", "#e6b84d", 0.0, 0.68, [0.9, 0.85, 1.3], 8.5),
    e("troll", Form::Character, "#6e7a5c", "#5c3b24", 0.0, 0.88, [1.35, 1.1, 1.25], 12.0),
    e("centaur", Form::Quadruped, "#8b5a2b", "#d8b498", 0.0, 0.72, [1.45, 0.95, 1.45], 12.0),
    e("phoenix", Form::WingedCreature, "#f28c28", "#f5c842", 0.15, 0.32, [1.55, 1.15, 1.35], 10.0),
    e("griffin", Form::WingedCreature, "#e6b84d", "#8b5a2b", 0.0, 0.68, [1.45, 1.15, 1.35], 12.0),
    // ── Pass 4 Adversarial Long-Tail Expansion (dinosaurs, breeds, wildlife, headwear, props) ──
    e("t-rex", Form::Quadruped, "#4a6e3a", "#e8d2a8", 0.0, 0.72, [1.75, 0.95, 1.55], 30.0),
    e("trex", Form::Quadruped, "#4a6e3a", "#e8d2a8", 0.0, 0.72, [1.75, 0.95, 1.55], 30.0),
    e("triceratops", Form::Quadruped, "#6b7a4a", "#d8c498", 0.0, 0.75, [1.65, 1.05, 1.15], 26.0),
    e("stegosaurus", Form::Quadruped, "#5c7a48", "#c88e4c", 0.0, 0.75, [1.75, 0.95, 1.15], 26.0),
    e("velociraptor", Form::Quadruped, "#6e5a3e", "#d8b468", 0.0, 0.70, [1.65, 0.75, 1.25], 16.0),
    e("cobra", Form::Aquatic, "#3a4a2c", "#e6b84d", 0.0, 0.45, [1.65, 0.65, 1.25], 5.0),
    e("retriever", Form::Quadruped, "#d8a458", "#b88438", 0.0, 0.85, [1.3, 0.85, 1.0], 5.5),
    e("husky", Form::Quadruped, "#8d9299", "#f4f4f2", 0.0, 0.88, [1.3, 0.85, 1.05], 5.5),
    e("corgi", Form::Quadruped, "#d88c42", "#f4f4f2", 0.0, 0.88, [1.35, 0.82, 0.78], 3.8),
    e("pug", Form::Quadruped, "#d8c4a0", "#22252b", 0.0, 0.85, [1.1, 0.85, 0.85], 3.2),
    e("cheetah", Form::Quadruped, "#e6b84d", "#1e1e20", 0.0, 0.78, [1.45, 0.75, 1.05], 9.0),
    e("panther", Form::Quadruped, "#1c1e22", "#f5c842", 0.0, 0.65, [1.45, 0.80, 1.0], 9.5),
    e("tiger", Form::Quadruped, "#f28c28", "#1e1e20", 0.0, 0.80, [1.45, 0.85, 1.05], 11.0),
    e("hippo", Form::Quadruped, "#7e7880", "#5e5860", 0.0, 0.72, [1.45, 1.25, 1.05], 16.0),
    e("rhino", Form::Quadruped, "#7e7c76", "#d8d0be", 0.0, 0.82, [1.55, 1.1, 1.1], 16.0),
    e("camel", Form::Quadruped, "#c8a472", "#a68252", 0.0, 0.85, [1.45, 0.85, 1.45], 15.0),
    e("llama", Form::Quadruped, "#ede8d8", "#c8a472", 0.0, 0.92, [1.15, 0.80, 1.45], 10.0),
    e("otter", Form::Quadruped, "#7a5234", "#d8b498", 0.0, 0.75, [1.55, 0.75, 0.75], 4.0),
    e("moose", Form::Quadruped, "#5c3b24", "#8b5a2b", 0.0, 0.85, [1.45, 0.95, 1.55], 16.0),
    e("sheep", Form::Quadruped, "#f4f4f2", "#22252b", 0.0, 0.96, [1.2, 1.0, 0.95], 7.0),
    e("pig", Form::Quadruped, "#f4a4b8", "#e8849c", 0.0, 0.65, [1.25, 0.95, 0.88], 6.5),
    e("cow", Form::Quadruped, "#f4f4f2", "#1e1e20", 0.0, 0.82, [1.4, 0.95, 1.15], 13.0),
    e("chicken", Form::WingedCreature, "#f4f4f2", "#d63a2f", 0.0, 0.78, [0.95, 0.85, 0.95], 2.5),
    e("rooster", Form::WingedCreature, "#b3202f", "#f5c842", 0.0, 0.75, [1.0, 0.85, 1.05], 2.8),
    e("hawk", Form::WingedCreature, "#7a5234", "#f5c842", 0.0, 0.65, [1.55, 1.05, 0.95], 4.5),
    e("falcon", Form::WingedCreature, "#6f7377", "#f4f4f2", 0.0, 0.62, [1.55, 1.0, 0.92], 4.2),
    e("raven", Form::WingedCreature, "#1c1e22", "#3a3d42", 0.0, 0.55, [1.25, 0.95, 0.95], 3.2),
    e("toucan", Form::WingedCreature, "#1e1e20", "#f28c28", 0.0, 0.58, [1.05, 0.85, 1.1], 3.0),
    e("hummingbird", Form::WingedCreature, "#36c9c0", "#4caf50", 0.2, 0.35, [1.25, 0.95, 0.78], 1.2),
    e("fire extinguisher", Form::Vessel, "#d63a2f", "#22252b", 0.4, 0.25, [0.68, 0.68, 1.65], 3.2),
    e("first aid kit", Form::BoxChest, "#f4f4f2", "#d63a2f", 0.0, 0.35, [1.25, 0.95, 0.48], 2.0),
    e("boxing glove", Form::FoodItem, "#d63a2f", "#f4f4f2", 0.0, 0.42, [1.1, 0.95, 1.15], 1.8),
    e("cowboy hat", Form::RingWear, "#8b5a2b", "#5c3b24", 0.0, 0.78, [1.45, 1.25, 0.72], 2.2),
    e("top hat", Form::Vessel, "#1c1e22", "#b3202f", 0.0, 0.65, [1.15, 1.15, 1.25], 2.2),
    e("wizard hat", Form::TowerSpire, "#4b3fa8", "#e6b84d", 0.0, 0.82, [1.15, 1.15, 1.45], 2.6),
    e("viking helmet", Form::RingWear, "#8d9299", "#ede8d8", 0.9, 0.28, [1.25, 1.05, 1.05], 2.2),
    e("sombrero", Form::RingWear, "#e6c87a", "#d63a2f", 0.0, 0.85, [1.75, 1.75, 0.62], 3.0),
    e("bowtie", Form::PaddleHandle, "#b3202f", "#8e1c28", 0.0, 0.55, [1.35, 0.65, 0.25], 1.0),
];

/// Resolves exact lexicon entries OR suffix-matched compound words (e.g. "cyberkatana", "spaceboat",
/// "birdhouse", "ice-wand", "coffee-mug") to their canonical lexicon key.
pub fn resolve(word: &str) -> Option<&'static str> {
    if let Some(hit) = LEXICON.iter().find(|e| e.name == word) {
        return Some(hit.name);
    }
    // Strip hyphens and test suffix compound head nouns (min 4 chars to avoid false positives).
    let compact = word.replace('-', "");
    if compact != word {
        if let Some(hit) = LEXICON.iter().find(|e| e.name == compact) {
            return Some(hit.name);
        }
    }
    if compact.len() >= 6 {
        // Prefer the longest matching lexicon head-noun suffix (e.g. "starfighter" > "fighter").
        let mut best: Option<&'static str> = None;
        for entry in LEXICON {
            let k = entry.name;
            if k.len() >= 4 && !k.contains(' ') && compact.ends_with(k) {
                if best.map_or(true, |b| k.len() > b.len()) {
                    best = Some(k);
                }
            }
        }
        if best.is_some() {
            return best;
        }
    }
    None
}

pub fn known(word: &str) -> bool {
    resolve(word).is_some()
}

pub fn build(word: &str, rng: &mut Rng) -> Option<(Vec<Part>, f32)> {
    let key = resolve(word)?;
    let entry = LEXICON.iter().find(|e| e.name == key)?;
    let mut parts = build_form(entry, rng);
    if entry.stretch != [1.0, 1.0, 1.0] {
        let [sx, sy, sz] = entry.stretch;
        for part in &mut parts {
            let inner = std::mem::replace(&mut part.shape, sphere(1.0));
            part.shape = union(0.0, vec![inner]).stretched(sx, sy, sz);
        }
    }
    Some((parts, entry.real_scale))
}

fn build_form(entry: &Entry, rng: &mut Rng) -> Vec<Part> {
    let c = entry.color;
    let a = entry.accent;
    let main = |name: &str, node: Node| Part::new(name, c, node).finish(entry.metallic, entry.roughness);
    let acc = |name: &str, node: Node| Part::new(name, a, node).finish(entry.metallic * 0.6, (entry.roughness + 0.1).min(1.0));
    let jitter = rng.range(0.95, 1.05);

    match entry.form {
        Form::PaddleHandle => vec![
            main("Head", difference(0.0, vec![
                rcylinder(44.0 * jitter, 18.0, 4.0).at(0.0, 0.0, 12.0),
                cylinder(38.0 * jitter, 16.0).at(0.0, 0.0, 17.0),
            ])),
            acc("Handle", capsule([0.0, 38.0, 14.0], [0.0, 108.0, 20.0], 6.5)),
        ],
        Form::SlimTool => vec![
            main("Shaft", union(3.0, vec![
                capsule([0.0, 0.0, 14.0], [0.0, 0.0, 118.0 * jitter], 7.0),
                ellipsoid(24.0, 14.0, 32.0).at(0.0, 0.0, 130.0 * jitter),
            ])),
            acc("Grip", rcylinder(9.5, 42.0, 3.0).at(0.0, 0.0, 22.0)),
        ],
        Form::Appliance => vec![
            main("Body", rbox(96.0 * jitter, 76.0, 88.0, 6.0).at(0.0, 0.0, 44.0)),
            acc("Panel", union(0.0, vec![
                rbox(68.0, 8.0, 52.0, 3.0).at(-4.0, -37.0, 48.0),
                cylinder(7.0, 8.0).rot(90.0, 0.0, 0.0).at(34.0, -37.0, 58.0),
                cylinder(6.0, 8.0).rot(90.0, 0.0, 0.0).at(34.0, -37.0, 36.0),
            ])),
        ],
        Form::DisplayStand => vec![
            main("Frame", rbox(106.0 * jitter, 14.0, 74.0, 4.0).at(0.0, 0.0, 54.0)),
            acc("Screen", rbox(94.0 * jitter, 6.0, 62.0, 2.0).at(0.0, -6.0, 54.0)),
            main("Stand", union(2.0, vec![
                cylinder(8.0, 24.0).at(0.0, 2.0, 14.0),
                rcylinder(32.0, 6.0, 2.0).at(0.0, 0.0, 3.0),
            ])),
        ],
        Form::Handheld => vec![
            main("Body", rbox(68.0 * jitter, 24.0, 108.0, 7.0).at(0.0, 0.0, 54.0)),
            acc("Controls", union(0.0, vec![
                rbox(48.0, 6.0, 38.0, 2.0).at(0.0, -11.0, 76.0),
                sphere(6.0).at(-14.0, -12.0, 34.0),
                sphere(6.0).at(14.0, -12.0, 34.0),
            ])),
        ],
        Form::Garment => vec![
            main("Torso", union(6.0, vec![
                rbox(66.0 * jitter, 26.0, 96.0, 10.0).at(0.0, 0.0, 52.0),
                capsule([26.0, 0.0, 82.0], [56.0, 0.0, 54.0], 13.0).mirror(X),
            ])),
            acc("Collar", torus(17.0, 5.0).at(0.0, 0.0, 98.0)),
        ],
        Form::Vessel => vec![
            main("Container", difference(0.0, vec![
                rcylinder(42.0 * jitter, 104.0, 6.0).at(0.0, 0.0, 52.0),
                cylinder(36.0 * jitter, 98.0).at(0.0, 0.0, 60.0),
            ])),
            acc("Rim", torus(41.0 * jitter, 4.5).at(0.0, 0.0, 102.0)),
        ],
        Form::StemmedVessel => vec![
            main("Bowl", union(4.0, vec![
                lathe(&[(0.0, 0.0), (28.0, 0.0), (28.0, 6.0), (6.0, 14.0), (6.0, 62.0), (36.0 * jitter, 82.0), (40.0 * jitter, 122.0), (0.0, 122.0)]),
            ])),
            acc("Content", cylinder(31.0 * jitter, 14.0).at(0.0, 0.0, 104.0)),
        ],
        Form::UtilityVehicle => vec![
            main("Chassis", union(4.0, vec![
                rbox(118.0 * jitter, 64.0, 36.0, 6.0).at(0.0, 0.0, 34.0),
                rbox(62.0, 58.0, 34.0, 6.0).at(-12.0, 0.0, 66.0),
            ])),
            acc("Wheels", union(0.0, vec![
                cylinder(18.0, 14.0).rot(90.0, 0.0, 0.0).at(36.0, 34.0, 18.0),
                cylinder(18.0, 14.0).rot(90.0, 0.0, 0.0).at(-36.0, 34.0, 18.0),
            ]).mirror(Y)),
        ],
        Form::Craft => vec![
            main("Hull", union(6.0, vec![
                ellipsoid(126.0 * jitter, 52.0, 44.0).at(0.0, 0.0, 30.0),
                rbox(44.0, 32.0, 26.0, 6.0).at(0.0, 0.0, 52.0),
            ])),
            acc("Trim", rbox(82.0, 10.0, 14.0, 3.0).at(0.0, 0.0, 56.0)),
        ],
        Form::FoodItem => vec![
            main("Body", ellipsoid(88.0 * jitter, 82.0, 74.0).at(0.0, 0.0, 38.0)),
            acc("Detail", union(3.0, vec![
                sphere(16.0).at(0.0, 0.0, 74.0),
                capsule([0.0, 0.0, 76.0], [10.0, 0.0, 94.0], 4.0),
            ])),
        ],
        Form::SportSphere => vec![
            main("Ball", sphere(52.0 * jitter).at(0.0, 0.0, 52.0)),
            acc("Band", union(0.0, vec![
                torus(52.0 * jitter, 3.5).at(0.0, 0.0, 52.0),
                torus(52.0 * jitter, 3.5).rot(90.0, 0.0, 0.0).at(0.0, 0.0, 52.0),
            ])),
        ],
        Form::RingWear => vec![
            main("Ring", torus(46.0 * jitter, 14.0).at(0.0, 0.0, 16.0)),
            acc("Badge", sphere(12.0).at(0.0, -46.0 * jitter, 20.0)),
        ],
        Form::Instrument => vec![
            main("Body", union(5.0, vec![
                ellipsoid(56.0 * jitter, 32.0, 66.0).at(0.0, 0.0, 36.0),
                capsule([0.0, 0.0, 62.0], [0.0, 0.0, 138.0], 8.0),
            ])),
            acc("Detail", union(0.0, vec![
                cone(10.0, 26.0, 24.0).at(0.0, -14.0, 132.0),
                cylinder(12.0, 8.0).rot(90.0, 0.0, 0.0).at(0.0, -15.0, 36.0),
            ])),
        ],
        Form::Structure => vec![
            main("Walls", rbox(92.0 * jitter, 82.0, 86.0, 4.0).at(0.0, 0.0, 43.0)),
            acc("Roof", cone(68.0 * jitter, 6.0, 48.0).at(0.0, 0.0, 108.0)),
        ],
        Form::TowerSpire => vec![
            main("Spire", union(3.0, vec![
                rbox(68.0 * jitter, 68.0, 18.0, 4.0).at(0.0, 0.0, 9.0),
                cone(38.0 * jitter, 12.0, 118.0).at(0.0, 0.0, 74.0),
            ])),
            acc("Crown", union(0.0, vec![
                rbox(44.0, 44.0, 12.0, 3.0).at(0.0, 0.0, 86.0),
                cone(14.0, 2.0, 32.0).at(0.0, 0.0, 148.0),
            ])),
        ],
        Form::ArchPortal => vec![
            main("Pillars", union(3.0, vec![
                rbox(18.0, 22.0, 106.0, 4.0).at(42.0 * jitter, 0.0, 53.0).mirror(X),
                rbox(114.0 * jitter, 24.0, 16.0, 4.0).at(0.0, 0.0, 110.0),
            ])),
            acc("Lintel", rbox(126.0 * jitter, 18.0, 12.0, 3.0).at(0.0, 0.0, 124.0)),
        ],
        Form::PoleTool => vec![
            main("Shaft", capsule([0.0, 0.0, 8.0], [0.0, 0.0, 128.0 * jitter], 5.5)),
            acc("Head", union(3.0, vec![
                rbox(46.0, 14.0, 24.0, 4.0).at(0.0, 0.0, 130.0 * jitter),
                cone(12.0, 2.0, 26.0).at(0.0, 0.0, 148.0 * jitter),
            ])),
        ],
        Form::FurniturePiece => vec![
            main("Frame", union(3.0, vec![
                rbox(98.0 * jitter, 72.0, 34.0, 6.0).at(0.0, 0.0, 44.0),
                rcylinder(6.0, 32.0, 2.0).at(38.0, 26.0, 16.0).mirror(X).mirror(Y),
            ])),
            acc("Top", rbox(90.0 * jitter, 64.0, 14.0, 5.0).at(0.0, 0.0, 64.0)),
        ],
        Form::Device => vec![
            main("Housing", union(4.0, vec![
                rcylinder(26.0 * jitter, 86.0, 6.0).at(0.0, 0.0, 48.0),
                rbox(56.0, 46.0, 14.0, 4.0).at(0.0, 0.0, 7.0),
            ])),
            acc("Lens", union(0.0, vec![
                cylinder(18.0, 22.0).rot(90.0, 0.0, 0.0).at(0.0, -24.0, 72.0),
                torus(18.0, 3.5).rot(90.0, 0.0, 0.0).at(0.0, -34.0, 72.0),
            ])),
        ],
        Form::Character => vec![
            main("Body", union(5.0, vec![
                capsule([-14.0, 0.0, 10.0], [-12.0, 0.0, 52.0], 10.0).mirror(X),
                rbox(44.0 * jitter, 28.0, 54.0, 10.0).at(0.0, 0.0, 74.0),
                capsule([24.0, 0.0, 90.0], [42.0, 0.0, 58.0], 8.5).mirror(X),
                sphere(22.0).at(0.0, 0.0, 122.0),
            ])),
            acc("Crest", union(0.0, vec![
                rbox(28.0, 14.0, 14.0, 5.0).at(0.0, -16.0, 124.0),
                cone(24.0, 4.0, 22.0).at(0.0, 0.0, 148.0),
            ])),
        ],
        Form::Quadruped => vec![
            main("Body", union(6.0, vec![
                ellipsoid(94.0 * jitter, 48.0, 46.0).at(0.0, 0.0, 48.0),
                capsule([28.0, 16.0, 44.0], [28.0, 16.0, 8.0], 8.0).mirror(Y),
                capsule([-28.0, 16.0, 44.0], [-28.0, 16.0, 8.0], 8.0).mirror(Y),
                capsule([38.0, 0.0, 58.0], [54.0, 0.0, 88.0], 12.0),
                sphere(20.0).at(58.0, 0.0, 96.0),
            ])),
            acc("Features", union(2.0, vec![
                capsule([54.0, 10.0, 110.0], [50.0, 16.0, 126.0], 5.0).mirror(Y),
                capsule([-44.0, 0.0, 52.0], [-66.0, 0.0, 42.0], 5.5),
            ])),
        ],
        Form::WingedCreature => vec![
            main("Body", union(5.0, vec![
                ellipsoid(56.0 * jitter, 38.0, 52.0).at(0.0, 0.0, 52.0),
                sphere(19.0).at(0.0, -14.0, 86.0),
                capsule([0.0, 12.0, 40.0], [0.0, 44.0, 22.0], 9.0),
            ])),
            acc("Wings", union(3.0, vec![
                ellipsoid(66.0 * jitter, 14.0, 32.0).rot(0.0, 18.0, 0.0).at(46.0, 4.0, 62.0).mirror(X),
                cone(7.0, 1.5, 18.0).rot(-90.0, 0.0, 0.0).at(0.0, -34.0, 86.0),
            ])),
        ],
        Form::Aquatic => vec![
            main("Body", union(6.0, vec![
                ellipsoid(112.0 * jitter, 42.0, 42.0).at(0.0, 0.0, 40.0),
                ellipsoid(36.0, 10.0, 42.0).at(-62.0, 0.0, 40.0),
            ])),
            acc("Fins", union(2.0, vec![
                ellipsoid(26.0, 10.0, 28.0).rot(0.0, -25.0, 0.0).at(4.0, 0.0, 68.0),
                ellipsoid(28.0, 26.0, 8.0).rot(0.0, 0.0, 20.0).at(10.0, 26.0, 32.0).mirror(Y),
            ])),
        ],
        Form::PlantFlora => vec![
            main("Stem", union(4.0, vec![
                rcylinder(26.0, 32.0, 4.0).at(0.0, 0.0, 16.0),
                capsule([0.0, 0.0, 28.0], [0.0, 0.0, 86.0 * jitter], 7.0),
            ])),
            acc("Blossom", union(4.0, vec![
                sphere(22.0).at(0.0, 0.0, 96.0 * jitter),
                ellipsoid(38.0 * jitter, 20.0, 14.0).at(26.0, 0.0, 94.0 * jitter).mirror(X),
                ellipsoid(20.0, 38.0 * jitter, 14.0).at(0.0, 26.0, 94.0 * jitter).mirror(Y),
            ])),
        ],
        Form::BoxChest => vec![
            main("Base", rbox(92.0 * jitter, 68.0, 56.0, 6.0).at(0.0, 0.0, 28.0)),
            acc("Lid", union(2.0, vec![
                rbox(96.0 * jitter, 72.0, 20.0, 6.0).at(0.0, 0.0, 64.0),
                rbox(16.0, 10.0, 18.0, 3.0).at(0.0, -36.0, 54.0),
            ])),
        ],
        Form::Footwear => vec![
            main("Sole", union(5.0, vec![
                rbox(48.0, 114.0 * jitter, 22.0, 8.0).at(0.0, 0.0, 12.0),
                rbox(44.0, 56.0, 44.0, 10.0).at(0.0, 24.0, 38.0),
            ])),
            acc("Trim", rbox(50.0, 118.0 * jitter, 8.0, 3.0).at(0.0, 0.0, 4.0)),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::LEXICON;
    use crate::text3d::{generate, mesher::is_manifold, GenOptions};

    #[test]
    fn lexicon_covers_common_objects_watertight() {
        assert!(LEXICON.len() >= 340, "expected broad visual lexicon, got {}", LEXICON.len());
        let opts = GenOptions { max_triangles: 8_000, detail: 40, seed: 0, quads: false };
        for prompt in [
            "frying pan", "microwave", "basketball", "astronaut", "katana", "wine glass", "avocado",
            "vr headset", "eiffel tower", "lightsaber", "cybertruck", "panda", "dolphin", "sunflower",
            "treasure chest", "golden gate", "surfboard", "phoenix", "espresso machine", "cyber-katana",
        ] {
            let g = generate(prompt, &opts).unwrap_or_else(|e| panic!("{prompt}: {e}"));
            assert!(g.understood, "{prompt} should be understood via lexicon");
            assert!(g.triangles() <= 8_000);
            for p in &g.parts {
                assert!(is_manifold(&p.mesh), "{prompt} part {} must be watertight manifold", p.name);
            }
        }
    }

    #[test]
    fn hundred_prompt_universal_stress_audit() {
        let prompts = [
            // Kitchen & food
            "ceramic mug", "frying pan", "espresso machine", "wine glass", "blender", "toaster", "boba", "croissant",
            "avocado", "watermelon", "ramen", "taco", "ice cream", "pumpkin", "pineapple", "pizza", "burger", "cake",
            "teapot", "cauldron",
            // Tech, music & studio
            "laptop", "iphone", "vision pro", "gamepad", "boombox", "turntable", "synthesizer", "guitar", "saxophone",
            "cello", "harp", "drums", "microphone", "headphones", "camera", "microscope", "telescope", "drone",
            "solar panel", "wind turbine",
            // Vehicles & sci-fi
            "cybertruck", "fire truck", "school bus", "tractor", "tank", "helicopter", "spaceship", "starfighter",
            "ufo", "pirate ship", "yacht", "submarine", "bicycle", "motorcycle", "skateboard", "surfboard",
            "hoverboard", "jetpack", "lightsaber", "blaster",
            // Living creatures & nature
            "dragon", "winged dragon", "phoenix", "unicorn", "lion", "wolf", "fox", "bear", "panda", "giraffe",
            "deer", "rabbit", "dolphin", "shark", "octopus", "jellyfish", "eagle", "parrot", "butterfly", "rose",
            "sunflower", "bonsai", "tree", "mushroom", "cactus",
            // Architecture, characters & compounds
            "eiffel tower", "statue of liberty", "taj mahal", "colosseum", "golden gate", "castle", "lighthouse",
            "torii", "portal", "astronaut", "wizard", "knight", "samurai", "ninja", "mech", "treasure chest",
            "spellbook", "throne", "cyber-katana", "crystal coffee maker",
        ];
        let opts = GenOptions { max_triangles: 8_000, detail: 36, seed: 0, quads: false };
        let t0 = std::time::Instant::now();
        let mut understood = 0usize;
        let mut manifold_ok = 0usize;
        let mut misses = Vec::new();
        for p in prompts {
            let g = generate(p, &opts).unwrap_or_else(|e| panic!("{p}: {e}"));
            if g.understood {
                understood += 1;
            } else {
                misses.push(p);
            }
            assert!(g.triangles() <= 8_000, "{p} exceeded triangle budget: {}", g.triangles());
            if g.parts.iter().all(|part| is_manifold(&part.mesh)) {
                manifold_ok += 1;
            }
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "AUDIT_100: understood={understood}/{} manifold={manifold_ok}/{} total={elapsed_ms:.1}ms avg={:.2}ms misses={misses:?}",
            prompts.len(),
            prompts.len(),
            elapsed_ms / prompts.len() as f64
        );
        assert_eq!(manifold_ok, prompts.len(), "100% of generated meshes must be watertight manifold");
        assert!(understood >= 95, "expected >=95% local hit rate, got {understood}/{}: misses={misses:?}", prompts.len());
    }
}
