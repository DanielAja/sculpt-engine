//! The built-in planner: plain English to a shape program, offline, instant and deterministic.
//!
//! "two small red mugs next to a teapot", "a cat with a hat on a table", "a gold robot 120 mm tall".
//! It reads counts, colours, materials, sizes, dimensions and three relations (on top of, under, next to /
//! and / commas, plus "with <accessory>"), then builds each noun from a template. Nouns it has no template
//! for fall back to an archetype family (like Dream Builder's M5 shape grammar: the noun seeds the
//! proportions, so "a wombat" always looks the same) and finally to a seeded blob.

use super::program::*;
use super::sdf::{summit, tight_aabb, Aabb, Compiled};
use super::compose;
use super::templates;

/// SplitMix64, seeded from the noun so a word always gives the same model.
pub struct Rng(u64);

impl Rng {
    pub fn new(word: &str, seed: u32) -> Rng {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in word.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        Rng(h ^ ((seed as u64) << 32 | seed as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }
    /// `x` varied by up to 10 %.
    pub fn v(&mut self, x: f32) -> f32 {
        x * self.range(0.9, 1.1)
    }
    pub fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[(self.next() % items.len() as u64) as usize]
    }
}

pub struct Plan {
    pub program: Program,
    pub understood: bool,
    pub notes: Vec<String>,
}

/// One built object before layout: parts standing on z = 0, centred on the Z axis.
struct Object {
    name: String,
    parts: Vec<Part>,
    real_scale: f32,
}

/// "Cat" + "Body" -> "Cat Body"; a part already named after its object keeps its name.
fn prefixed(object: &str, part: &str) -> String {
    if part.is_empty() || part == object || part.starts_with(&format!("{object} ")) {
        object.to_string()
    } else {
        format!("{object} {part}")
    }
}

const COLORS: &[(&str, &str)] = &[
    ("red", "#d63a2f"), ("crimson", "#b3202f"), ("maroon", "#7a1f2b"), ("orange", "#f28c28"), ("yellow", "#f5c842"),
    ("gold", "#e6b84d"), ("golden", "#e6b84d"), ("lime", "#9ccc3d"), ("green", "#4caf50"), ("olive", "#808a3a"),
    ("mint", "#98e2c6"), ("teal", "#1fa39a"), ("turquoise", "#36c9c0"), ("cyan", "#33c3e0"), ("sky", "#7cc4f2"),
    ("blue", "#2f6fd6"), ("navy", "#1f2f6b"), ("indigo", "#4b3fa8"), ("purple", "#8e44ad"), ("violet", "#9b6ad6"),
    ("lavender", "#c7b3eb"), ("magenta", "#d6338f"), ("pink", "#f48fb1"), ("coral", "#f2735f"), ("peach", "#f7b98f"),
    ("brown", "#8b5a2b"), ("tan", "#d2b48c"), ("beige", "#e8dcc0"), ("cream", "#f4ecd6"), ("white", "#f4f4f2"),
    ("silver", "#c9ccd1"), ("grey", "#9e9e9e"), ("gray", "#9e9e9e"), ("black", "#1e1e20"), ("rainbow", "#e85d9c"),
];

/// (word, colour or "" to keep the colour, metallic, roughness)
const MATERIALS: &[(&str, &str, f32, f32)] = &[
    ("gold", "#e6b84d", 1.0, 0.25), ("golden", "#e6b84d", 1.0, 0.25), ("silver", "#c9ccd1", 1.0, 0.25),
    ("metal", "#a9adb3", 1.0, 0.3), ("metallic", "", 1.0, 0.3), ("steel", "#a9adb3", 1.0, 0.3), ("iron", "#6f7377", 1.0, 0.45),
    ("chrome", "#d9dde2", 1.0, 0.08), ("copper", "#c27a4a", 1.0, 0.3), ("bronze", "#b08d57", 1.0, 0.35),
    ("wood", "#a0703f", 0.0, 0.75), ("wooden", "#a0703f", 0.0, 0.75), ("stone", "#8d8d88", 0.0, 0.9),
    ("marble", "#ecebe6", 0.0, 0.25), ("glass", "#cfe8f2", 0.0, 0.05), ("plastic", "", 0.0, 0.35), ("clay", "", 0.0, 0.85),
    ("rubber", "#333336", 0.0, 0.9), ("ice", "#cdeefa", 0.0, 0.1), ("icy", "#cdeefa", 0.0, 0.1),
    ("chocolate", "#5a3420", 0.0, 0.5), ("shiny", "", 0.0, 0.12), ("matte", "", 0.0, 0.9),
];

const NUMBERS: &[(&str, u32)] = &[
    ("a", 1), ("an", 1), ("one", 1), ("single", 1), ("two", 2), ("pair", 2), ("couple", 2), ("three", 3), ("few", 3),
    ("four", 4), ("several", 4), ("five", 5), ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9), ("ten", 10),
    ("eleven", 11), ("twelve", 12), ("dozen", 12),
];

#[derive(Default, Clone)]
struct Mods {
    count: Option<u32>,
    plural: bool,
    stacked: bool,
    color: Option<String>,
    material: Option<(String, f32, f32)>,
    scale: f32,
    stretch: [f32; 3],
    /// Target size in mm and whether it is a height.
    dimension: Option<(f32, bool)>,
}

fn word_index<T: Copy>(table: &[(&str, T)], w: &str) -> Option<T> {
    table.iter().find(|(k, _)| *k == w).map(|(_, v)| *v)
}

pub fn singular(w: &str) -> String {
    let irregular = [("mice", "mouse"), ("geese", "goose"), ("people", "person"), ("men", "man"), ("women", "woman"),
        ("children", "child"), ("teeth", "tooth"), ("feet", "foot"), ("knives", "knife"), ("leaves", "leaf"), ("wolves", "wolf"),
        ("loaves", "loaf"), ("dice", "dice"), ("fish", "fish"), ("sheep", "sheep"), ("deer", "deer"), ("glasses", "glass"),
        ("cactuses", "cactus"), ("cacti", "cactus"), ("octopi", "octopus")];
    if let Some((_, s)) = irregular.iter().find(|(p, _)| *p == w) {
        return s.to_string();
    }
    if w.len() > 4 && w.ends_with("ies") {
        return format!("{}y", &w[..w.len() - 3]);
    }
    for suf in ["ches", "shes", "sses", "xes", "zes", "oes"] {
        if w.len() > suf.len() + 1 && w.ends_with(suf) {
            return w[..w.len() - 2].to_string();
        }
    }
    if w.len() > 3 && w.ends_with('s') && !w.ends_with("ss") && !w.ends_with("us") && !w.ends_with("is") {
        return w[..w.len() - 1].to_string();
    }
    w.to_string()
}

fn tokens(prompt: &str) -> Vec<String> {
    let mut out = Vec::new();
    let lower = prompt.to_lowercase();
    for raw in lower.split(|c: char| !(c.is_alphanumeric() || c == '.' || c == '#')) {
        let t = raw.trim_matches('.');
        if t.is_empty() {
            continue;
        }
        // "30mm" -> "30", "mm"
        let split = t.find(|c: char| !(c.is_ascii_digit() || c == '.'));
        match split {
            Some(i) if i > 0 && t[..i].parse::<f32>().is_ok() => {
                out.push(t[..i].to_string());
                out.push(t[i..].to_string());
            }
            _ => out.push(t.to_string()),
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Rel {
    /// The next phrase is a new object beside this one.
    Beside,
    /// This one sits on the next.
    On,
    /// The next sits on this one.
    Under,
}

/// Splits the token stream into phrases and the relation from each phrase to the next.
fn phrases(t: &[String]) -> Vec<(Vec<String>, Rel)> {
    let mut out = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    let mut i = 0;
    let push = |out: &mut Vec<(Vec<String>, Rel)>, cur: &mut Vec<String>, r: Rel| {
        if cur.iter().any(|w| !is_filler(w)) {
            out.push((std::mem::take(cur), r));
        } else {
            cur.clear();
        }
    };
    while i < t.len() {
        let w = t[i].as_str();
        let next = t.get(i + 1).map(|s| s.as_str()).unwrap_or("");
        let next2 = t.get(i + 2).map(|s| s.as_str()).unwrap_or("");
        match w {
            "and" | "plus" | "beside" | "besides" => {
                push(&mut out, &mut cur, Rel::Beside);
                i += 1;
            }
            "next" | "close" if next == "to" => {
                push(&mut out, &mut cur, Rel::Beside);
                i += 2;
            }
            "on" | "onto" | "upon" | "atop" | "above" | "over" => {
                // "on top of", "on a", "sitting on"
                push(&mut out, &mut cur, Rel::On);
                i += if next == "top" && next2 == "of" { 3 } else { 1 };
            }
            "sitting" | "standing" | "resting" | "stacked" if matches!(next, "on" | "upon" | "atop") => {
                push(&mut out, &mut cur, Rel::On);
                i += 2;
            }
            "under" | "underneath" | "below" | "beneath" | "supporting" => {
                push(&mut out, &mut cur, Rel::Under);
                i += 1;
            }
            "by" if !cur.is_empty() => {
                push(&mut out, &mut cur, Rel::Beside);
                i += 1;
            }
            _ => {
                cur.push(t[i].clone());
                i += 1;
            }
        }
    }
    push(&mut out, &mut cur, Rel::Beside);
    out
}

fn is_filler(w: &str) -> bool {
    matches!(
        w,
        "the" | "of" | "some" | "please" | "make" | "me" | "build" | "create" | "draw" | "sculpt" | "model" | "i" | "want"
            | "need" | "give" | "show" | "put" | "add" | "cute" | "nice" | "simple" | "3d" | "shaped" | "like" | "that"
            | "is" | "it" | "its" | "very" | "really" | "super" | "with" | "in" | "row" | "lots" | "bunch" | "group"
            | "set" | "made" | "out" | "toy" | "little" | "kind" | "sort" | "thing" | "object" | "tiny" | "small"
            | "big" | "large" | "huge" | "giant" | "tall" | "short" | "wide" | "fat" | "thin" | "long" | "flat"
            | "light" | "dark" | "pale" | "bright" | "deep" | "and" | "to" | "a" | "an" | "my" | "for" | "who" | "which"
    )
}

fn read_mods(words: &[String], m: &mut Mods) {
    m.scale = 1.0;
    m.stretch = [1.0; 3];
    let mut shade: Option<&str> = None;
    let mut i = 0;
    while i < words.len() {
        let w = words[i].as_str();
        if let Ok(n) = w.parse::<f32>() {
            let unit = words.get(i + 1).map(|s| s.as_str()).unwrap_or("");
            let mm = match unit {
                "mm" | "millimeter" | "millimeters" | "millimetre" | "millimetres" => Some(n),
                "cm" | "centimeter" | "centimeters" | "centimetre" | "centimetres" => Some(n * 10.0),
                "m" | "meter" | "meters" | "metre" | "metres" => Some(n * 1000.0),
                "in" | "inch" | "inches" => Some(n * 25.4),
                _ => None,
            };
            if let Some(mm) = mm {
                let tall = matches!(words.get(i + 2).map(|s| s.as_str()), Some("tall" | "high" | "height"));
                if mm.is_finite() && mm > 0.0 {
                    m.dimension = Some((mm.clamp(1.0, MAX_SIZE_MM), tall));
                }
                i += 2;
                continue;
            }
            if n >= 1.0 && n <= 24.0 && n.fract() == 0.0 {
                m.count = Some(n as u32);
            }
            i += 1;
            continue;
        }
        if let Some(n) = word_index(NUMBERS, w) {
            if m.count.is_none() || n > 1 {
                m.count = Some(n);
            }
        }
        match w {
            "stack" | "stacked" | "tower" | "pile" => m.stacked = true,
            "tiny" | "mini" | "miniature" => m.scale *= 0.4,
            "small" | "little" => m.scale *= 0.65,
            "big" | "large" => m.scale *= 1.5,
            "huge" | "giant" | "enormous" | "massive" => m.scale *= 2.4,
            "tall" => m.stretch[2] *= 1.4,
            "short" | "squat" => m.stretch[2] *= 0.7,
            "flat" => m.stretch[2] *= 0.5,
            "wide" | "fat" | "chubby" | "round" => {
                m.stretch[0] *= 1.3;
                m.stretch[1] *= 1.3;
            }
            "thin" | "skinny" | "slim" | "narrow" => {
                m.stretch[0] *= 0.72;
                m.stretch[1] *= 0.72;
            }
            "long" => m.stretch[0] *= 1.5,
            "light" | "pale" | "pastel" => shade = Some("light"),
            "dark" | "deep" => shade = Some("dark"),
            _ => {}
        }
        if let Some((_, c, metal, rough)) = MATERIALS.iter().find(|(k, ..)| *k == w) {
            m.material = Some((c.to_string(), *metal, *rough));
        }
        if let Some(c) = word_index(COLORS, w) {
            m.color = Some(shade_color(c, shade.take()));
        }
        i += 1;
    }
}

fn shade_color(hex: &str, shade: Option<&str>) -> String {
    let Some(c) = super::parse_color(hex) else { return hex.to_string() };
    let (r, g, b) = match shade {
        Some("light") => (c.r + (1.0 - c.r) * 0.45, c.g + (1.0 - c.g) * 0.45, c.b + (1.0 - c.b) * 0.45),
        Some("dark") => (c.r * 0.55, c.g * 0.55, c.b * 0.55),
        _ => (c.r, c.g, c.b),
    };
    let h = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", h(r), h(g), h(b))
}

/// The noun of a phrase: the rightmost known word or two-word name, else the last content word.
fn find_noun(words: &[String]) -> (Option<String>, Option<String>) {
    for i in (0..words.len()).rev() {
        if i > 0 {
            let two = format!("{} {}", words[i - 1], singular(&words[i]));
            if templates::known(&two) {
                return (Some(two), None);
            }
            if let Some(c) = compose::resolve_phrase(&words[i - 1], &singular(&words[i])) {
                return (Some(c), None);
            }
            if let Some(lex) = super::lexicon::resolve(&two) {
                return (Some(lex.to_string()), None);
            }
        }
        let s = singular(&words[i]);
        if templates::known(&s) {
            return (Some(s), None);
        }
        if let Some(c) = compose::resolve(&s) {
            return (Some(c), None);
        }
        if let Some(lex) = super::lexicon::resolve(&s) {
            return (Some(lex.to_string()), None);
        }
    }
    let unknown = words
        .iter()
        .rev()
        .find(|w| {
            !is_filler(w)
                && w.parse::<f32>().is_err()
                && word_index(COLORS, w).is_none()
                && !MATERIALS.iter().any(|(k, ..)| k == &w.as_str())
                && word_index(NUMBERS, w).is_none()
                && !compose::is_modifier_word(w)
                && !matches!(w.as_str(), "mm" | "cm" | "m" | "in" | "inch" | "inches" | "tall" | "high" | "stacked" | "stack" | "tower" | "pile")
        })
        .map(|w| singular(w));
    (None, unknown)
}

fn aabb(parts: &[Part]) -> Aabb {
    parts.iter().fold(Aabb::EMPTY, |b, p| b.union(&tight_aabb(&Compiled::new(&p.shape))))
}

/// Wraps a part so `f` applies around the program origin.
fn wrap(part: &mut Part, f: impl FnOnce(Node) -> Node) {
    let inner = std::mem::replace(&mut part.shape, sphere(1.0));
    part.shape = f(union(0.0, vec![inner]));
}

fn translate(parts: &mut [Part], x: f32, y: f32, z: f32) {
    for p in parts {
        p.shape.at = [p.shape.at[0] + x, p.shape.at[1] + y, p.shape.at[2] + z];
    }
}

fn scale_parts(parts: &mut [Part], s: f32, stretch: [f32; 3]) {
    if s == 1.0 && stretch == [1.0; 3] {
        return;
    }
    for p in parts {
        wrap(p, |n| n.stretched(stretch[0], stretch[1], stretch[2]).scaled(s));
    }
}

struct Built {
    obj: Object,
    understood: bool,
    note: Option<String>,
}

fn build_noun(noun: Option<&str>, unknown: Option<&str>, seed: u32) -> Built {
    if let Some(n) = noun {
        let mut r = Rng::new(n, seed);
        if let Some((parts, real)) = templates::build(n, &mut r) {
            return Built { obj: Object { name: title(n), parts, real_scale: real }, understood: true, note: None };
        }
        if let Some((parts, real)) = compose::build_generalized(n, &mut r) {
            return Built { obj: Object { name: title(n), parts, real_scale: real }, understood: true, note: None };
        }
        if let Some((parts, real)) = super::lexicon::build(n, &mut r) {
            return Built { obj: Object { name: title(n), parts, real_scale: real }, understood: true, note: None };
        }
    }
    let word = unknown.unwrap_or("shape");
    let mut r = Rng::new(word, seed);
    if let Some((parts, real)) = compose::build_generalized(word, &mut r) {
        return Built { obj: Object { name: title(word), parts, real_scale: real }, understood: true, note: None };
    }
    if let Some((parts, real)) = super::lexicon::build(word, &mut r) {
        return Built { obj: Object { name: title(word), parts, real_scale: real }, understood: true, note: None };
    }
    let mut r = Rng::new(word, seed);
    if let Some((parts, real, family)) = templates::family(word, &mut r) {
        return Built {
            obj: Object { name: title(word), parts, real_scale: real },
            understood: true,
            note: Some(format!(
                "I haven't learned \"{word}\" yet, so I shaped it like {} {family}.",
                if family.starts_with(['a', 'e', 'i', 'o', 'u']) { "an" } else { "a" }
            )),
        };
    }
    Built {
        obj: Object { name: title(word), parts: compose::fallback_sculpture(&mut r), real_scale: 1.0 },
        understood: false,
        note: Some(format!(
            "I don't know \"{word}\" yet, so this is an original sculpture to sculpt from. Try shapes and colours (\"a blue sphere on a box\"), or turn on AI planning."
        )),
    }
}

fn title(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn apply_mods(obj: &mut Object, m: &Mods) {
    if let Some((c, metal, rough)) = &m.material {
        let p = &mut obj.parts[0];
        p.metallic = *metal;
        p.roughness = *rough;
        if !c.is_empty() && m.color.is_none() {
            p.color = Some(c.clone());
        }
    }
    if let Some(c) = &m.color {
        obj.parts[0].color = Some(c.clone());
    }
    let mut s = m.scale;
    if let Some((mm, tall)) = m.dimension {
        let size = aabb(&obj.parts).size() * glam::Vec3::from_array(m.stretch);
        let now = if tall { size.z } else { size.max_element() };
        if now > 0.0 {
            s = mm / now;
        }
    }
    scale_parts(&mut obj.parts, s, m.stretch);
}

/// Lays `objs` out as one object: each sits centred on top of the previous one.
fn stack(mut objs: Vec<Object>) -> Object {
    if objs.len() > 1 {
        for o in objs.iter_mut() {
            for p in o.parts.iter_mut() {
                p.name = prefixed(&o.name, &p.name);
            }
        }
    }
    let mut out = objs.remove(0);
    for mut o in objs {
        let below = aabb(&out.parts);
        let me = aabb(&o.parts);
        translate(&mut o.parts, below.center().x - me.center().x, below.center().y - me.center().y, below.max.z - me.min.z);
        out.name = format!("{} on {}", o.name, out.name);
        out.parts.extend(o.parts);
    }
    out
}

/// Puts `acc` on top of `host`'s highest feature (a cat's head, a snowman's head), sized to fit it.
fn accessorise(host: &mut Object, mut acc: Object) {
    let compiled: Vec<Compiled> = host.parts.iter().map(|p| Compiled::new(&p.shape)).collect();
    let hb = aabb(&host.parts);
    let (top, width) = summit(&compiled).unwrap_or((glam::Vec3::new(hb.center().x, hb.center().y, hb.max.z), hb.size().x));
    let ab = aabb(&acc.parts);
    let target = width.max(hb.size().max_element() * 0.12) * 1.1;
    let s = if ab.size().max_element() > 0.0 { target / ab.size().x.max(ab.size().y) } else { 1.0 };
    scale_parts(&mut acc.parts, s.clamp(0.05, 20.0), [1.0; 3]);
    let ab = aabb(&acc.parts);
    // Sink it a little so it rests on the curve of the head rather than touching at one point.
    translate(&mut acc.parts, top.x - ab.center().x, top.y - ab.center().y, top.z - ab.min.z - ab.size().z * 0.08);
    for p in acc.parts.iter_mut() {
        p.name = prefixed(&acc.name, &p.name);
    }
    host.parts.extend(acc.parts);
}

pub fn plan(prompt: &str, seed: u32) -> Plan {
    let t = tokens(prompt);
    let mut notes = Vec::new();
    let mut understood = true;
    // Stacks of objects; each stack is listed top to bottom as read.
    let mut stacks: Vec<Vec<Object>> = vec![vec![]];
    let mut put_next_on_top = false;
    let list = phrases(&t);
    let mut real = None;
    for (words, rel) in list.iter() {
        // "with" splits an accessory off the phrase: "a cat with a red hat".
        let (main, acc_words) = match words.iter().position(|w| w == "with" || w == "wearing" || w == "holding") {
            Some(i) => (&words[..i], Some(&words[i + 1..])),
            None => (&words[..], None),
        };
        let mut m = Mods::default();
        read_mods(main, &mut m);
        let (noun, unknown) = find_noun(main);
        m.plural = main.iter().any(|w| singular(w) != *w && noun.as_deref().is_some_and(|n| n.ends_with(&singular(w))));
        let b = build_noun(noun.as_deref(), unknown.as_deref(), seed);
        understood &= b.understood;
        if let Some(n) = b.note {
            notes.push(n);
        }
        let mut obj = b.obj;
        real.get_or_insert(obj.real_scale);
        let mut mod_rng = Rng::new(&format!("{}:features", obj.name), seed);
        compose::attach_modifiers(&mut obj.parts, main, &mut mod_rng);
        if let Some(aw) = acc_words {
            let mut am = Mods::default();
            read_mods(aw, &mut am);
            let (an, au) = find_noun(aw);
            // "a cat with wings", "a robot with a halo": real 3D features, not an accessory object.
            let features = compose::attach_features(&mut obj.parts, aw, &mut mod_rng);
            if features == 0 && (an.is_some() || au.is_some()) {
                let ab = build_noun(an.as_deref(), au.as_deref(), seed);
                if let Some(n) = ab.note {
                    notes.push(n);
                }
                let mut acc = ab.obj;
                am.scale = 1.0;
                am.stretch = [1.0; 3];
                am.dimension = None;
                apply_mods(&mut acc, &am);
                accessorise(&mut obj, acc);
            }
        }
        apply_mods(&mut obj, &m);
        let count = m.count.unwrap_or(if m.plural { 3 } else { 1 }).clamp(1, 12);
        let copies: Vec<Object> =
            (0..count).map(|_| Object { name: obj.name.clone(), parts: obj.parts.clone(), real_scale: obj.real_scale }).collect();
        if count > 1 {
            notes.push(format!("Made {count} of them{}.", if m.stacked { ", stacked" } else { " in a row" }));
        }
        let cur = stacks.last_mut().unwrap();
        if m.stacked {
            let mut s = copies;
            s.reverse();
            let tower = stack(s);
            if put_next_on_top { cur.insert(0, tower) } else { cur.push(tower) }
        } else {
            let mut it = copies.into_iter();
            let first = it.next().unwrap();
            if put_next_on_top { cur.insert(0, first) } else { cur.push(first) }
            for extra in it {
                stacks.push(vec![extra]);
            }
        }
        // On: this one sits on the next, so keep reading into the same stack (with copies, the last one does).
        // Under: the next one goes on top of this stack. Beside: start a new stack.
        put_next_on_top = *rel == Rel::Under;
        if *rel == Rel::Beside {
            stacks.push(vec![]);
        }
    }
    let mut built: Vec<Object> = stacks
        .into_iter()
        .filter(|s| !s.is_empty())
        .map(|mut s| {
            s.reverse(); // bottom first
            stack(s)
        })
        .collect();
    if built.is_empty() {
        let mut r = Rng::new("shape", seed);
        built.push(Object { name: "Shape".into(), parts: templates::blob(&mut r), real_scale: 1.0 });
        understood = false;
        notes.push("Describe something to make, like \"a red mug\" or \"a cat on a box\".".into());
    }
    // A row along X, centred, with a gap that scales with the objects.
    let boxes: Vec<Aabb> = built.iter().map(|o| aabb(&o.parts)).collect();
    let gap = boxes.iter().map(|b| b.size().x.max(b.size().y)).fold(0.0f32, f32::max) * 0.15 + 8.0;
    let total: f32 = boxes.iter().map(|b| b.size().x).sum::<f32>() + gap * (boxes.len() as f32 - 1.0);
    let mut x = -total / 2.0;
    let many = built.len() > 1;
    let mut parts = Vec::new();
    let mut names = Vec::new();
    for (o, b) in built.into_iter().zip(boxes) {
        let mut ps = o.parts;
        translate(&mut ps, x - b.min.x, -b.center().y, -b.min.z);
        x += b.size().x + gap;
        let stacked = o.name.contains(" on ");
        let single = ps.len() == 1;
        for p in ps.iter_mut() {
            if single && !stacked {
                p.name = o.name.clone(); // one part: it is the object ("Flibbertigibbet", not "Clay")
            } else if (many && !stacked) || p.name.is_empty() {
                p.name = prefixed(&o.name, &p.name);
            }
        }
        names.push(o.name);
        parts.extend(ps);
    }
    parts.truncate(MAX_PARTS);
    let name = prompt.trim().chars().take(80).collect::<String>();
    let program = Program { version: PROGRAM_VERSION, name: if name.is_empty() { names.join(", ") } else { name }, parts, real_scale: real };
    Plan { program, understood, notes }
}
