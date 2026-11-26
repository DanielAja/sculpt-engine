//! Pure-Rust Voice & Cloud AI planner (`EngineVoice` + `PlanCompiler` + `PlanBuild`).
//!
//! Faithfully mirrors `SculptorKit/Cloud` and `SculptorKit/Planning` so all Android voice,
//! Gemini Nano, cloud provider (Anthropic / OpenAI / Gemini / Custom), consent, and quick-parser
//! workflows run 100% natively in `libsculpt_engine` with zero Swift dependency.

use crate::bridge::{
    commands::{apply_command_value, make_primitive_mesh},
    evaluate_object_mesh, ok_envelope, scene_world_matrix, BoundingBox, BridgeError, EngineDoc,
};
use crate::format::Scene;
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Mutex, OnceLock};

const CONSENT_VERSION: i32 = 1;
const MAX_STEPS: usize = 12;
const DEFAULT_SIZE: f64 = 20.0;
const GAP: f32 = 2.0;

const TOOL_NAME: &str = "submit_build_plan";
const TOOL_DESCRIPTION: &str = "Submit the modelling plan for Sculptor to build. Call this exactly once with the complete plan.";

const PLAN_INSTRUCTIONS: &str = "You are the modelling assistant inside Sculptor, a 3D modelling app. Convert the user's description into a short plan of modelling steps.
Units are millimetres. The ground plane is z = 0 and objects rest on it; x is right, y is away from the viewer, z is up. For addPrimitive, x and y place the object's centre and z is the height of its base above the ground (leave z empty to sit on the ground). When a relation is used, x, y and z are small extra offsets. Sizes: width is X, depth is Y, height is Z; for a sphere give width only; for a torus width is the outer diameter and height is the tube thickness. Typical small objects are 10 to 100 mm.
Prefer few steps (at most 12). Build compound things from primitives:
- mug or cup: cylinder \"Body\" 40 wide 60 high; cylinder \"Hollow\" 34 wide 60 high, relation centeredOn Body with z = 4; boolean difference target Body tool Hollow; torus \"Handle\" width 30 height 8, relation rightOf Body with x = -10 and z = 26; rotate Handle 90 degrees with x = 1 so it stands upright.
- table: cube \"Top\" 100 wide 60 deep 5 high at z = 45; four cylinder legs \"Leg 1\"..\"Leg 4\" 6 wide 45 high at the corners (x = +-44, y = +-24).
- snowman: spheres \"Bottom\" 40, \"Middle\" 30 onTopOf Bottom, \"Head\" 20 onTopOf Middle.
- house: cube \"Base\" 40; cone \"Roof\" 50 wide 50 deep 25 high onTopOf Base.
- tower: one cube plus an array modifier with amount = number of floors and relation onTopOf (stacked).
Name objects with short nouns (Body, Handle, Leg 1). Use relations (onTopOf, leftOf, rightOf, inFrontOf, behind, inside, centeredOn) with relativeTo instead of guessing coordinates when possible. Use target \"last\" for the most recently created object. Set colours with setColor on the object (colour name or #hex; finish matte, glossy or metal). Rotations are about the up axis unless exactly one of x, y, z is 1 to pick that axis. Modifiers: subdivision (amount = levels 1-3) makes things smooth and round, solidify (amount = wall thickness mm) makes things hollow, array (amount = count) repeats, mirror mirrors on X. For rename, put the new name in tool. Objects already in the scene can be targeted by name. Do not describe; only produce steps.";

const TOOL_OUTPUT_NOTE: &str = "Always answer by calling the submit_build_plan tool exactly once with the complete plan (a summary and 1 to 12 steps). Leave out fields that don't apply to a step. Do not reply with plain text.";
const JSON_OUTPUT_NOTE: &str = "Reply with a single JSON object with \"summary\" (string) and \"steps\" (array of 1 to 12 step objects). Use only the documented field names and enum values; use null (or leave the field out) when a field doesn't apply. Output JSON only, no prose, no code fences.";

// ── BuildPlan data types ──

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BuildStep {
    pub action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relative_to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rotation_degrees: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale_factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub amount: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boolean_op: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<i32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildPlan {
    pub summary: String,
    pub steps: Vec<BuildStep>,
}

// ── Color vocabulary ──

struct ColorEntry {
    name: &'static str,
    hex: &'static str,
    aliases: &'static [&'static str],
}

const COLORS: &[ColorEntry] = &[
    ColorEntry { name: "Red", hex: "#E63946", aliases: &["scarlet", "crimson"] },
    ColorEntry { name: "Blue", hex: "#3A86FF", aliases: &[] },
    ColorEntry { name: "Green", hex: "#4CAF50", aliases: &[] },
    ColorEntry { name: "Yellow", hex: "#F6D55C", aliases: &[] },
    ColorEntry { name: "Orange", hex: "#F4A261", aliases: &[] },
    ColorEntry { name: "Purple", hex: "#8E44AD", aliases: &["violet"] },
    ColorEntry { name: "Pink", hex: "#F8A5C2", aliases: &[] },
    ColorEntry { name: "White", hex: "#F5F5F5", aliases: &["snow"] },
    ColorEntry { name: "Black", hex: "#222222", aliases: &[] },
    ColorEntry { name: "Grey", hex: "#8D8D8D", aliases: &["gray"] },
    ColorEntry { name: "Brown", hex: "#8B5A2B", aliases: &["chocolate", "wood", "wooden"] },
    ColorEntry { name: "Gold", hex: "#D4AF37", aliases: &["golden"] },
    ColorEntry { name: "Silver", hex: "#C0C0C0", aliases: &["chrome", "steel"] },
    ColorEntry { name: "Teal", hex: "#2A9D8F", aliases: &[] },
    ColorEntry { name: "Cyan", hex: "#00BCD4", aliases: &["aqua"] },
    ColorEntry { name: "Navy", hex: "#1D3557", aliases: &["navy blue"] },
    ColorEntry { name: "Lime", hex: "#A7C957", aliases: &["lime green"] },
    ColorEntry { name: "Coral", hex: "#FF6B60", aliases: &[] },
    ColorEntry { name: "Beige", hex: "#E8D8B8", aliases: &["cream"] },
    ColorEntry { name: "Tan", hex: "#D2B48C", aliases: &[] },
    ColorEntry { name: "Maroon", hex: "#800000", aliases: &["burgundy"] },
    ColorEntry { name: "Olive", hex: "#808000", aliases: &[] },
    ColorEntry { name: "Turquoise", hex: "#40E0D0", aliases: &[] },
    ColorEntry { name: "Magenta", hex: "#FF00FF", aliases: &["fuchsia"] },
    ColorEntry { name: "Lavender", hex: "#B8A1E3", aliases: &["lilac"] },
    ColorEntry { name: "Mint", hex: "#98E2C6", aliases: &[] },
    ColorEntry { name: "Sky Blue", hex: "#87CEEB", aliases: &["sky", "light blue"] },
    ColorEntry { name: "Dark Green", hex: "#1B5E20", aliases: &["forest green"] },
    ColorEntry { name: "Clay", hex: "#E9C9A8", aliases: &[] },
];

fn parse_hex_rgba(hex: &str) -> Option<[f32; 4]> {
    let h = hex.strip_prefix('#').unwrap_or(hex);
    if (h.len() != 6 && h.len() != 8) || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let r = u8::from_str_radix(&h[0..2], 16).ok()? as f32 / 255.0;
    let g = u8::from_str_radix(&h[2..4], 16).ok()? as f32 / 255.0;
    let b = u8::from_str_radix(&h[4..6], 16).ok()? as f32 / 255.0;
    let a = if h.len() == 8 { u8::from_str_radix(&h[6..8], 16).ok()? as f32 / 255.0 } else { 1.0 };
    Some([r, g, b, a])
}

fn parse_color(text: &str) -> Option<(String, [f32; 4])> {
    let key = text.trim().to_ascii_lowercase();
    if key.is_empty() { return None; }
    for e in COLORS {
        if e.name.to_ascii_lowercase() == key || e.aliases.iter().any(|&a| a == key) {
            return parse_hex_rgba(e.hex).map(|rgba| (e.name.to_string(), rgba));
        }
    }
    let hex_cand = key.strip_prefix('#').unwrap_or(&key);
    if let Some(rgba) = parse_hex_rgba(hex_cand) {
        return Some((format!("#{}", hex_cand.to_ascii_uppercase()), rgba));
    }
    None
}

fn find_color_in_text(lower: &str) -> Option<String> {
    // Check #hex
    let bytes = lower.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'#' && i + 7 <= bytes.len() && bytes[i + 1..i + 7].iter().all(|b| b.is_ascii_hexdigit()) {
            let end = if i + 9 <= bytes.len() && bytes[i + 7..i + 9].iter().all(|b| b.is_ascii_hexdigit()) { i + 9 } else { i + 7 };
            return Some(lower[i..end].to_string());
        }
    }
    let mut vocab: Vec<(&str, &str)> = Vec::new();
    for e in COLORS {
        vocab.push((e.name, e.name));
        for &a in e.aliases {
            vocab.push((a, e.name));
        }
    }
    vocab.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    let mut best: Option<(usize, String)> = None;
    for (word, canon) in vocab {
        let w = word.to_ascii_lowercase();
        if let Some(pos) = find_whole_word(lower, &w) {
            if best.as_ref().is_none_or(|(bp, _)| pos < *bp) {
                best = Some((pos, canon.to_ascii_lowercase()));
            }
        }
    }
    best.map(|(_, c)| c)
}

fn find_whole_word(haystack: &str, needle: &str) -> Option<usize> {
    let hb = haystack.as_bytes();
    let nb = needle.as_bytes();
    if nb.is_empty() || nb.len() > hb.len() { return None; }
    let mut start = 0;
    while let Some(rel) = haystack[start..].find(needle) {
        let pos = start + rel;
        let end = pos + nb.len();
        let left_ok = pos == 0 || !hb[pos - 1].is_ascii_alphanumeric();
        let right_ok = end == hb.len() || !hb[end].is_ascii_alphanumeric();
        if left_ok && right_ok {
            return Some(pos);
        }
        start = pos + 1;
    }
    None
}

// ── Providers, Consent, URL validation, Pretty model names ──

const ALL_PROVIDERS: &[&str] = &["onDevice", "quickParser", "anthropic", "openAI", "gemini", "custom"];

fn is_cloud(p: &str) -> bool {
    matches!(p, "anthropic" | "openAI" | "gemini" | "custom")
}

fn brand_name(p: &str) -> Option<&'static str> {
    match p {
        "anthropic" => Some("Anthropic"),
        "openAI" => Some("OpenAI"),
        "gemini" => Some("Google Gemini"),
        _ => None,
    }
}

fn suggested_models(p: &str) -> &'static [&'static str] {
    match p {
        "anthropic" => &["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5", "claude-opus-5-5", "claude-fable-5-1"],
        "openAI" => &["gpt-5", "gpt-5-mini"],
        "gemini" => &["gemini-2.5-flash", "gemini-2.5-pro"],
        _ => &[],
    }
}

fn key_url(p: &str) -> Option<&'static str> {
    match p {
        "anthropic" => Some("https://console.anthropic.com/settings/keys"),
        "openAI" => Some("https://platform.openai.com/api-keys"),
        "gemini" => Some("https://aistudio.google.com/apikey"),
        _ => None,
    }
}

fn privacy_url(p: &str) -> Option<&'static str> {
    match p {
        "anthropic" => Some("https://www.anthropic.com/legal/privacy"),
        "openAI" => Some("https://openai.com/policies/privacy-policy/"),
        "gemini" => Some("https://policies.google.com/privacy"),
        _ => None,
    }
}

fn parse_url_parts(text: &str) -> Option<(String, String, String)> {
    let mut t = text.trim().to_string();
    while t.ends_with('/') { t.pop(); }
    let (scheme, rest) = t.split_once("://")?;
    let scheme_l = scheme.to_ascii_lowercase();
    if scheme_l != "http" && scheme_l != "https" { return None; }
    let authority = rest.split('/').next()?;
    if authority.is_empty() { return None; }
    let host_port = authority.rsplit('@').next().unwrap_or(authority);
    let host = if host_port.starts_with('[') {
        let end = host_port.find(']')?;
        &host_port[1..end]
    } else {
        host_port.split(':').next()?
    };
    if host.is_empty() || host.contains(char::is_whitespace) { return None; }
    let host_owned = host.to_string();
    Some((t, scheme_l, host_owned))
}

fn is_local_host(host: &str) -> bool {
    let h = host.trim_matches(|c| c == '[' || c == ']').to_ascii_lowercase();
    matches!(h.as_str(), "localhost" | "127.0.0.1" | "::1" | "10.0.2.2")
        || h.ends_with(".localhost")
        || (h.ends_with(".local") && h.len() > ".local".len())
}

fn allows_cleartext(scheme: &str, host: &str) -> bool {
    scheme == "https" || (scheme == "http" && is_local_host(host))
}

fn recipient_for_selection(provider: &str, base_url: Option<&str>) -> Option<String> {
    match provider {
        "anthropic" => Some("api.anthropic.com".to_string()),
        "openAI" => Some("api.openai.com".to_string()),
        "gemini" => Some("generativelanguage.googleapis.com".to_string()),
        "custom" => base_url.and_then(parse_url_parts).map(|(_, _, h)| h),
        _ => None,
    }
}

fn pretty_model_name(id: &str, display_names: &HashMap<String, String>) -> String {
    if let Some(known) = display_names.get(id) {
        if !known.trim().is_empty() { return known.clone(); }
    }
    if !id.starts_with("claude-") {
        return id.to_string();
    }
    let mut parts: Vec<&str> = id.split('-').collect();
    if let Some(&last) = parts.last() {
        if last.len() == 8 && last.bytes().all(|b| b.is_ascii_digit()) {
            parts.pop();
        }
    }
    if parts.len() < 3
        || parts[1].is_empty()
        || !parts[1].chars().all(|c| c.is_ascii_alphabetic())
        || !parts[2..].iter().all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
    {
        return id.to_string();
    }
    let mut fam = parts[1].chars();
    let family = match fam.next() {
        Some(c) => format!("{}{}", c.to_ascii_uppercase(), fam.as_str()),
        None => String::new(),
    };
    let ver = parts[2..].join(".");
    format!("Claude {family} {ver}")
}

fn cloud_label(provider: &str, model: Option<&str>, display_names: &HashMap<String, String>) -> (Option<String>, String) {
    if !is_cloud(provider) {
        return (None, String::new());
    }
    let model_name = model
        .filter(|m| !m.is_empty())
        .map(|m| pretty_model_name(m, display_names))
        .or_else(|| brand_name(provider).map(|s| s.to_string()));
    let shown = model_name.clone().unwrap_or_else(|| "Custom".to_string());
    let prov_label = if provider == "custom" { "Custom" } else { brand_name(provider).unwrap_or(provider) };
    (model_name, format!("{shown} · {prov_label}"))
}

// ── Lenient BuildPlan JSON parser + Sanitizer + StepDisplay ──

const VALID_ACTIONS: &[&str] = &["addPrimitive", "move", "rotate", "scale", "setColor", "addModifier", "boolean", "duplicate", "rename", "delete"];
const VALID_SHAPES: &[&str] = &["cube", "sphere", "cylinder", "cone", "torus", "plane", "blob"];
const VALID_RELATIONS: &[&str] = &["onTopOf", "leftOf", "rightOf", "inFrontOf", "behind", "inside", "centeredOn"];
const VALID_FINISHES: &[&str] = &["matte", "glossy", "metal"];
const VALID_MODIFIERS: &[&str] = &["mirror", "subdivision", "array", "solidify"];
const VALID_BOOLEANS: &[&str] = &["union", "difference", "intersect"];

fn norm_key(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

fn match_enum(v: Option<&Value>, allowed: &[&'static str]) -> Option<String> {
    let raw = parse_opt_str(v)?;
    let k = norm_key(&raw);
    allowed.iter().find(|&&a| norm_key(a) == k).map(|&s| s.to_string())
}

fn parse_opt_str(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => {
            let t = s.trim();
            if t.is_empty() || t.eq_ignore_ascii_case("null") { None } else { Some(t.to_string()) }
        }
        Value::Number(n) => Some(format_step_number(n.as_f64()?)),
        _ => None,
    }
}

fn parse_opt_num(v: Option<&Value>) -> Option<f64> {
    let n = match v? {
        Value::Number(x) => x.as_f64()?,
        Value::String(s) => {
            let cleaned = s.to_ascii_lowercase().replace("mm", "");
            cleaned.trim().parse::<f64>().ok()?
        }
        _ => return None,
    };
    (n.is_finite() && n.abs() <= 1e6).then_some(n)
}

fn extract_json_from_text(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return Some(v);
    }
    // Strip markdown fences or extract first balanced {...} or [...]
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{' || b == b'[')?;
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc { esc = false; } else if b == b'\\' { esc = true; } else if b == b'"' { in_str = false; }
            continue;
        }
        match b {
            b'"' => in_str = true,
            x if x == open => depth += 1,
            x if x == close => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str::<Value>(&text[start..=i]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_plan_from_value(value: &Value) -> Result<BuildPlan, &'static str> {
    let mut root = value.clone();
    if root.is_array() {
        root = json!({"steps": root});
    }
    for wrapper in ["plan", "buildPlan", "build_plan", "input", "arguments"] {
        if root.get("steps").is_none() {
            if let Some(inner) = root.get(wrapper).cloned() {
                if let Some(s) = inner.as_str() {
                    if let Some(parsed) = extract_json_from_text(s) {
                        root = parsed;
                        continue;
                    }
                }
                root = inner;
            }
        }
    }
    let obj = root.as_object().ok_or("The answer wasn't a JSON object.")?;
    let raw_steps = obj.get("steps").and_then(|x| x.as_array()).ok_or("The answer had no steps.")?;
    let mut steps = Vec::new();
    for sv in raw_steps {
        let Some(o) = sv.as_object() else { continue };
        let Some(action) = match_enum(o.get("action"), VALID_ACTIONS) else { continue };
        steps.push(BuildStep {
            action,
            target: parse_opt_str(o.get("target")),
            shape: match_enum(o.get("shape"), VALID_SHAPES),
            width: parse_opt_num(o.get("width")),
            height: parse_opt_num(o.get("height")),
            depth: parse_opt_num(o.get("depth")),
            x: parse_opt_num(o.get("x")),
            y: parse_opt_num(o.get("y")),
            z: parse_opt_num(o.get("z")),
            relation: match_enum(o.get("relation"), VALID_RELATIONS),
            relative_to: parse_opt_str(o.get("relativeTo")),
            rotation_degrees: parse_opt_num(o.get("rotationDegrees")),
            scale_factor: parse_opt_num(o.get("scaleFactor")),
            color: parse_opt_str(o.get("color")),
            finish: match_enum(o.get("finish"), VALID_FINISHES),
            modifier: match_enum(o.get("modifier"), VALID_MODIFIERS),
            amount: parse_opt_num(o.get("amount")),
            boolean_op: match_enum(o.get("booleanOp"), VALID_BOOLEANS),
            tool: parse_opt_str(o.get("tool")),
            count: parse_opt_num(o.get("count")).map(|n| n.round() as i32),
        });
    }
    if steps.is_empty() {
        return Err("None of the steps could be understood.");
    }
    let summary = obj.get("summary").and_then(|x| x.as_str()).unwrap_or("").to_string();
    Ok(BuildPlan { summary, steps })
}

fn parse_plan_from_text(text: &str) -> Result<BuildPlan, &'static str> {
    let val = extract_json_from_text(text).ok_or("The answer wasn't valid JSON.")?;
    parse_plan_from_value(&val)
}

fn sanitize_plan(plan: BuildPlan) -> (BuildPlan, Vec<BridgeError>) {
    let mut warnings = Vec::new();
    let mut steps = Vec::new();
    let clamp_sz = |v: Option<f64>| v.map(|x| x.abs().clamp(1.0, 500.0));
    let clamp_pos = |v: Option<f64>| v.map(|x| x.clamp(-1000.0, 1000.0));
    let has_name = |s: &Option<String>| s.as_deref().is_some_and(|x| !x.trim().is_empty());

    for (i, orig) in plan.steps.into_iter().enumerate() {
        let orig_action = orig.action.clone();
        let mut s = orig;
        s.width = clamp_sz(s.width);
        s.height = clamp_sz(s.height);
        s.depth = clamp_sz(s.depth);
        s.x = clamp_pos(s.x);
        s.y = clamp_pos(s.y);
        s.z = clamp_pos(s.z);
        if let Some(c) = s.count { s.count = Some(c.clamp(1, 50)); }
        if let Some(a) = s.amount { s.amount = a.is_finite().then(|| a.clamp(0.0, 1000.0)); }
        if let Some(r) = s.rotation_degrees { s.rotation_degrees = r.is_finite().then(|| r % 360.0); }
        if let Some(f) = s.scale_factor { if !f.is_finite() { s.scale_factor = None; } }
        let keep = match s.action.as_str() {
            "addPrimitive" => {
                let ok = s.shape.is_some();
                if ok && s.width.is_none() && s.height.is_none() && s.depth.is_none() {
                    s.width = Some(DEFAULT_SIZE);
                }
                ok
            }
            "move" => s.relation.is_some() || s.x.is_some() || s.y.is_some() || s.z.is_some(),
            "rotate" => s.rotation_degrees.unwrap_or(0.0) != 0.0,
            "scale" => {
                if let Some(f) = s.scale_factor { s.scale_factor = Some(f.clamp(0.05, 20.0)); }
                s.scale_factor.is_some_and(|f| f != 1.0)
            }
            "setColor" => {
                if let Some(ref c) = s.color {
                    if parse_color(c).is_none() {
                        s.color = find_color_in_text(&c.to_ascii_lowercase());
                    }
                }
                s.color.is_some() || s.finish.is_some()
            }
            "addModifier" => s.modifier.is_some(),
            "boolean" => {
                if s.boolean_op.is_none() { s.boolean_op = Some("union".to_string()); }
                has_name(&s.tool)
            }
            "rename" => has_name(&s.tool) || has_name(&s.relative_to),
            "duplicate" | "delete" => true,
            _ => false,
        };
        if keep {
            steps.push(s);
        } else {
            warnings.push(BridgeError::with_args("plan.skippedStep", vec![(i + 1).to_string(), orig_action]));
        }
    }
    if steps.len() > MAX_STEPS {
        warnings.push(BridgeError::with_args("plan.keptFirstSteps", vec![MAX_STEPS.to_string()]));
        steps.truncate(MAX_STEPS);
    }
    (BuildPlan { summary: plan.summary.trim().to_string(), steps }, warnings)
}

fn is_pronoun(name: Option<&str>) -> bool {
    let Some(n) = name else { return true };
    if n.is_empty() { return true; }
    matches!(
        n.to_ascii_lowercase().as_str(),
        "last" | "it" | "that" | "this" | "them" | "previous" | "last one" | "that one" | "this one" | "the last one" | "the previous one" | "one"
    )
}

fn format_step_number(v: f64) -> String {
    if !v.is_finite() { return format!("{v}"); }
    let r = (v * 10.0).round() / 10.0;
    if (r - r.round()).abs() < 1e-9 {
        format!("{}", r.round() as i64)
    } else {
        format!("{r:.1}")
    }
}

fn primitive_size(shape: &str, w: Option<f64>, h: Option<f64>, d: Option<f64>) -> Vec3 {
    let cl = |v: f64| v.clamp(0.1, 5000.0) as f32;
    match shape {
        "sphere" => {
            let s = cl(w.or(h).or(d).unwrap_or(DEFAULT_SIZE));
            Vec3::splat(s)
        }
        "torus" => {
            if let Some(outer) = w.or(d) {
                let tube = h.unwrap_or(outer / 3.0).min(outer / 2.0);
                Vec3::new(cl(outer), cl(outer), cl(tube))
            } else {
                Vec3::new(32.0, 32.0, cl(h.unwrap_or(8.0)))
            }
        }
        "plane" => {
            let ww = w.unwrap_or(DEFAULT_SIZE);
            Vec3::new(cl(ww), cl(d.unwrap_or(ww)), 0.0)
        }
        _ => {
            let ww = w.or(d).or(h).unwrap_or(DEFAULT_SIZE);
            Vec3::new(cl(ww), cl(d.unwrap_or(ww)), cl(h.unwrap_or(ww)))
        }
    }
}

fn rotation_axis(x: Option<f64>, y: Option<f64>, z: Option<f64>) -> Vec3 {
    let v = Vec3::new(x.unwrap_or(0.0) as f32, y.unwrap_or(0.0) as f32, z.unwrap_or(0.0) as f32);
    let nz = [v.x, v.y, v.z].iter().filter(|&&c| c != 0.0).count();
    if nz == 1 { v.normalize() } else { Vec3::Z }
}

fn step_display_json(s: &BuildStep) -> Value {
    let ref_name = |opt: Option<&str>| -> Option<String> {
        if is_pronoun(opt) { None } else { opt.map(|x| x.to_string()) }
    };
    let mut d = serde_json::Map::new();
    d.insert("action".into(), json!(s.action));
    if let Some(who) = ref_name(s.target.as_deref()) { d.insert("who".into(), json!(who)); }

    match s.action.as_str() {
        "addPrimitive" => {
            if let Some(ref shape) = s.shape {
                d.insert("shape".into(), json!(shape));
                if s.width.is_some() || s.height.is_some() || s.depth.is_some() {
                    let sz = primitive_size(shape, s.width, s.height, s.depth);
                    let nums: Vec<String> = match shape.as_str() {
                        "sphere" => vec![format_step_number(sz.x as f64)],
                        "torus" => vec![format_step_number(sz.x as f64), format_step_number(sz.z as f64)],
                        "plane" => vec![format_step_number(sz.x as f64), format_step_number(sz.y as f64)],
                        _ => vec![format_step_number(sz.x as f64), format_step_number(sz.y as f64), format_step_number(sz.z as f64)],
                    };
                    d.insert("size".into(), json!(nums));
                }
                if let Some(ref t) = s.target {
                    if !t.is_empty() && !t.eq_ignore_ascii_case(shape) && !t.eq_ignore_ascii_case("last") {
                        d.insert("name".into(), json!(t));
                    }
                }
                if let Some(ref rel) = s.relation {
                    d.insert("relation".into(), json!(rel));
                    if let Some(r) = ref_name(s.relative_to.as_deref().or(Some("last"))) {
                        d.insert("reference".into(), json!(r));
                    }
                } else if let Some(z) = s.z {
                    if z > 0.0 { d.insert("elevation".into(), json!(format_step_number(z))); }
                }
            }
        }
        "move" => {
            if let Some(ref rel) = s.relation {
                d.insert("relation".into(), json!(rel));
                if let Some(r) = ref_name(s.relative_to.as_deref().or(Some("last"))) { d.insert("reference".into(), json!(r)); }
            } else if let Some(ref rto) = s.relative_to {
                if let Some(r) = ref_name(Some(rto)) { d.insert("reference".into(), json!(r)); }
                let same = if is_pronoun(Some(rto)) { is_pronoun(s.target.as_deref()) } else { Some(rto) == s.target.as_ref() };
                d.insert("sameReference".into(), json!(same));
                let mut offsets = Vec::new();
                if let Some(x) = s.x { if x != 0.0 { offsets.push(json!({"direction": if x > 0.0 { "right" } else { "left" }, "millimetres": format_step_number(x.abs())})); } }
                if let Some(y) = s.y { if y != 0.0 { offsets.push(json!({"direction": if y > 0.0 { "back" } else { "forward" }, "millimetres": format_step_number(y.abs())})); } }
                if let Some(z) = s.z { if z != 0.0 { offsets.push(json!({"direction": if z > 0.0 { "up" } else { "down" }, "millimetres": format_step_number(z.abs())})); } }
                d.insert("offsets".into(), Value::Array(offsets));
            } else {
                let coords: Vec<String> = [s.x, s.y, s.z].iter().map(|v| v.map(format_step_number).unwrap_or_else(|| "·".into())).collect();
                d.insert("coordinates".into(), json!(coords));
            }
        }
        "rotate" => {
            let ax = rotation_axis(s.x, s.y, s.z);
            let axis_name = if ax.x.abs() > 0.5 { "X" } else if ax.y.abs() > 0.5 { "Y" } else { "Z" };
            d.insert("axis".into(), json!(axis_name));
            d.insert("degrees".into(), json!(format_step_number(s.rotation_degrees.unwrap_or(0.0))));
        }
        "scale" => {
            d.insert("factor".into(), json!(format_step_number(s.scale_factor.unwrap_or(1.0))));
        }
        "setColor" => {
            if let Some(ref c) = s.color {
                let disp = parse_color(c).map(|(n, _)| n.to_ascii_lowercase()).unwrap_or_else(|| c.clone());
                d.insert("color".into(), json!(disp));
            }
            if let Some(ref fin) = s.finish {
                if fin != "matte" || s.color.is_none() { d.insert("finish".into(), json!(fin)); }
            }
        }
        "addModifier" => {
            if let Some(ref m) = s.modifier {
                d.insert("modifier".into(), json!(m));
                match m.as_str() {
                    "subdivision" => { d.insert("level".into(), json!(s.amount.unwrap_or(2.0).round().clamp(1.0, 3.0) as i32)); }
                    "array" => {
                        let cnt = s.amount.or(s.count.map(|c| c as f64)).unwrap_or(2.0).round().clamp(2.0, 50.0) as i32;
                        d.insert("count".into(), json!(cnt));
                        d.insert("stacked".into(), json!(s.relation.as_deref() == Some("onTopOf")));
                    }
                    "solidify" => { d.insert("thickness".into(), json!(format_step_number(s.amount.unwrap_or(2.0)))); }
                    _ => {}
                }
            }
        }
        "boolean" => {
            if let Some(t) = ref_name(s.tool.as_deref().or(Some("?"))) { d.insert("tool".into(), json!(t)); }
            d.insert("booleanOp".into(), json!(s.boolean_op.as_deref().unwrap_or("union")));
        }
        "duplicate" => {
            let copies = (s.count.unwrap_or(2).clamp(1, 51) - 1).max(1);
            d.insert("copies".into(), json!(copies));
        }
        "rename" => {
            d.insert("newName".into(), json!(s.tool.as_deref().or(s.relative_to.as_deref()).unwrap_or("…")));
        }
        _ => {}
    }
    Value::Object(d)
}

// ── Deterministic KeywordPlanner (Quick Parser) ──

fn parse_word_number(w: &str) -> Option<f64> {
    if let Ok(n) = w.parse::<f64>() { return Some(n); }
    match w {
        "one" | "a" | "an" => Some(1.0),
        "two" | "pair" | "couple" => Some(2.0),
        "three" => Some(3.0),
        "four" => Some(4.0),
        "five" => Some(5.0),
        "six" => Some(6.0),
        "seven" => Some(7.0),
        "eight" => Some(8.0),
        "nine" => Some(9.0),
        "ten" => Some(10.0),
        _ => None,
    }
}

fn extract_mm_dimension(clause: &str) -> Option<f64> {
    let toks: Vec<&str> = clause.split_whitespace().collect();
    for (i, &t) in toks.iter().enumerate() {
        let stripped = t.trim_end_matches("mm").trim_end_matches("cm");
        if stripped.len() < t.len() {
            if let Ok(v) = stripped.parse::<f64>() {
                return Some(if t.ends_with("cm") { v * 10.0 } else { v });
            }
        }
        if let Ok(v) = t.parse::<f64>() {
            let next = toks.get(i + 1).copied().unwrap_or("");
            if next == "mm" || next == "millimetres" || next == "millimeters" {
                return Some(v);
            }
            if next == "cm" || next == "centimetres" || next == "centimeters" {
                return Some(v * 10.0);
            }
        }
    }
    None
}

fn quick_plan(request: &str, scene_names: &[String]) -> Result<(BuildPlan, Vec<BridgeError>), BridgeError> {
    let raw = request.trim();
    if raw.is_empty() {
        return Err(BridgeError::new("plan.nothingRecognised"));
    }
    let normalized: String = raw
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '#' || c == '.' || c == '-' || c.is_whitespace() { c } else { ' ' })
        .collect();
    let norm = normalized.split_whitespace().collect::<Vec<_>>().join(" ");

    // Split clauses on " and then ", " then ", " and " (keeping compound phrases intact)
    let mut clauses: Vec<String> = Vec::new();
    for part in norm.split(" and then ") {
        for sub in part.split(" then ") {
            let s = sub.trim();
            if !s.is_empty() { clauses.push(s.to_string()); }
        }
    }

    let mut steps = Vec::<BuildStep>::new();
    let mut warnings = Vec::<BridgeError>::new();
    let mut last_created = false;

    for clause in &clauses {
        let color = find_color_in_text(clause);
        let finish = if find_whole_word(clause, "glossy").is_some() || find_whole_word(clause, "shiny").is_some() {
            Some("glossy".to_string())
        } else if find_whole_word(clause, "metal").is_some() || find_whole_word(clause, "metallic").is_some() {
            Some("metal".to_string())
        } else if find_whole_word(clause, "matte").is_some() {
            Some("matte".to_string())
        } else {
            None
        };

        // 1. Compound templates (mug, table, snowman, house, tower, donut on plate)
        if let Some((tmpl_steps, color_targets)) = match_template(clause) {
            steps.extend(tmpl_steps);
            if color.is_some() || finish.is_some() {
                for t in color_targets {
                    steps.push(BuildStep {
                        action: "setColor".into(),
                        target: Some(t.to_string()),
                        color: color.clone(),
                        finish: finish.clone(),
                        ..Default::default()
                    });
                }
            }
            last_created = true;
            continue;
        }

        // 2. Primitive shapes
        let shape = [
            ("cube", &["cube", "cubes", "box", "boxes", "block", "blocks"][..]),
            ("sphere", &["sphere", "spheres", "ball", "balls"][..]),
            ("cylinder", &["cylinder", "cylinders", "tube", "pillar", "rod"][..]),
            ("cone", &["cone", "cones", "pyramid"][..]),
            ("torus", &["torus", "donut", "doughnut", "ring"][..]),
            ("plane", &["plane", "floor", "plate", "sheet"][..]),
            ("blob", &["blob", "organic"][..]),
        ]
        .iter()
        .find_map(|&(canon, words)| words.iter().any(|&w| find_whole_word(clause, w).is_some()).then_some(canon));

        if let Some(sh) = shape {
            let dim = extract_mm_dimension(clause);
            let stacked = find_whole_word(clause, "stacked").is_some() || find_whole_word(clause, "stack").is_some();
            let count = clause
                .split_whitespace()
                .filter_map(parse_word_number)
                .map(|n| n.round() as i32)
                .find(|&n| (2..=20).contains(&n));

            steps.push(BuildStep {
                action: "addPrimitive".into(),
                shape: Some(sh.to_string()),
                width: Some(dim.unwrap_or(DEFAULT_SIZE)),
                ..Default::default()
            });
            if color.is_some() || finish.is_some() {
                steps.push(BuildStep {
                    action: "setColor".into(),
                    target: Some("last".into()),
                    color: color.clone(),
                    finish: finish.clone(),
                    ..Default::default()
                });
            }
            if let Some(cnt) = count {
                if stacked || find_whole_word(clause, "row").is_some() || find_whole_word(clause, "line").is_some() || cnt > 1 {
                    steps.push(BuildStep {
                        action: "addModifier".into(),
                        target: Some("last".into()),
                        modifier: Some("array".into()),
                        amount: Some(cnt as f64),
                        count: Some(cnt),
                        relation: if stacked { Some("onTopOf".into()) } else { None },
                        ..Default::default()
                    });
                }
            }
            last_created = true;
            continue;
        }

        // 3. Target resolution against scene names or pronoun
        let matched_scene_target = scene_names
            .iter()
            .find(|n| find_whole_word(clause, &n.to_ascii_lowercase()).is_some())
            .cloned();
        let target = matched_scene_target.or_else(|| if last_created { Some("last".into()) } else { None });

        // 4. Scale (bigger / smaller / double / half)
        if find_whole_word(clause, "bigger").is_some() || find_whole_word(clause, "larger").is_some() || find_whole_word(clause, "double").is_some() {
            steps.push(BuildStep { action: "scale".into(), target: target.or(Some("last".into())), scale_factor: Some(1.5), ..Default::default() });
            continue;
        }
        if find_whole_word(clause, "smaller").is_some() || find_whole_word(clause, "shrink").is_some() || find_whole_word(clause, "half").is_some() {
            steps.push(BuildStep { action: "scale".into(), target: target.or(Some("last".into())), scale_factor: Some(0.5), ..Default::default() });
            continue;
        }

        // 5. Rotate
        if find_whole_word(clause, "rotate").is_some() || find_whole_word(clause, "turn").is_some() {
            let deg = clause.split_whitespace().find_map(|w| w.parse::<f64>().ok()).unwrap_or(90.0);
            steps.push(BuildStep { action: "rotate".into(), target: target.or(Some("last".into())), rotation_degrees: Some(deg), ..Default::default() });
            continue;
        }

        // 6. Delete / Duplicate / Modifiers / Color on existing object
        if find_whole_word(clause, "delete").is_some() || find_whole_word(clause, "remove").is_some() {
            steps.push(BuildStep { action: "delete".into(), target: target.or(Some("last".into())), ..Default::default() });
            continue;
        }
        if find_whole_word(clause, "duplicate").is_some() || find_whole_word(clause, "copy").is_some() {
            steps.push(BuildStep { action: "duplicate".into(), target: target.or(Some("last".into())), count: Some(2), ..Default::default() });
            continue;
        }
        if find_whole_word(clause, "mirror").is_some() {
            steps.push(BuildStep { action: "addModifier".into(), target: target.or(Some("last".into())), modifier: Some("mirror".into()), ..Default::default() });
            continue;
        }
        if find_whole_word(clause, "smooth").is_some() || find_whole_word(clause, "subdivide").is_some() {
            steps.push(BuildStep { action: "addModifier".into(), target: target.or(Some("last".into())), modifier: Some("subdivision".into()), amount: Some(2.0), ..Default::default() });
            continue;
        }
        if find_whole_word(clause, "hollow").is_some() || find_whole_word(clause, "solidify").is_some() {
            steps.push(BuildStep { action: "addModifier".into(), target: target.or(Some("last".into())), modifier: Some("solidify".into()), amount: Some(2.0), ..Default::default() });
            continue;
        }
        if color.is_some() || finish.is_some() {
            steps.push(BuildStep {
                action: "setColor".into(),
                target: target.or(Some("last".into())),
                color,
                finish,
                ..Default::default()
            });
            continue;
        }

        warnings.push(BridgeError::with_args("plan.skippedClause", vec![clause.clone()]));
    }

    if steps.is_empty() {
        return Err(BridgeError::new("plan.nothingRecognised"));
    }
    let (clean, mut san_w) = sanitize_plan(BuildPlan { summary: raw.to_string(), steps });
    warnings.append(&mut san_w);
    Ok((clean, warnings))
}

fn match_template(clause: &str) -> Option<(Vec<BuildStep>, Vec<&'static str>)> {
    if (find_whole_word(clause, "donut").is_some() || find_whole_word(clause, "doughnut").is_some())
        && (find_whole_word(clause, "plate").is_some() || find_whole_word(clause, "dish").is_some())
    {
        return Some((
            vec![
                BuildStep { action: "addPrimitive".into(), target: Some("Plate".into()), shape: Some("cylinder".into()), width: Some(70.0), height: Some(3.0), depth: Some(70.0), ..Default::default() },
                BuildStep { action: "addPrimitive".into(), target: Some("Donut".into()), shape: Some("torus".into()), width: Some(40.0), height: Some(14.0), relation: Some("onTopOf".into()), relative_to: Some("Plate".into()), ..Default::default() },
                BuildStep { action: "setColor".into(), target: Some("Plate".into()), color: Some("white".into()), finish: Some("glossy".into()), ..Default::default() },
                BuildStep { action: "setColor".into(), target: Some("Donut".into()), color: Some("tan".into()), ..Default::default() },
            ],
            vec!["Donut"],
        ));
    }
    if ["mug", "mugs", "cup", "cups", "teacup", "coffee cup"].iter().any(|&w| find_whole_word(clause, w).is_some()) {
        return Some((
            vec![
                BuildStep { action: "addPrimitive".into(), target: Some("Body".into()), shape: Some("cylinder".into()), width: Some(40.0), height: Some(60.0), depth: Some(40.0), ..Default::default() },
                BuildStep { action: "addPrimitive".into(), target: Some("Hollow".into()), shape: Some("cylinder".into()), width: Some(34.0), height: Some(60.0), depth: Some(34.0), z: Some(4.0), relation: Some("centeredOn".into()), relative_to: Some("Body".into()), ..Default::default() },
                BuildStep { action: "boolean".into(), target: Some("Body".into()), boolean_op: Some("difference".into()), tool: Some("Hollow".into()), ..Default::default() },
                BuildStep { action: "addPrimitive".into(), target: Some("Handle".into()), shape: Some("torus".into()), width: Some(30.0), height: Some(8.0), x: Some(-10.0), z: Some(26.0), relation: Some("rightOf".into()), relative_to: Some("Body".into()), ..Default::default() },
                BuildStep { action: "rotate".into(), target: Some("Handle".into()), x: Some(1.0), rotation_degrees: Some(90.0), ..Default::default() },
            ],
            vec!["Body", "Handle"],
        ));
    }
    if ["table", "tables", "desk", "desks"].iter().any(|&w| find_whole_word(clause, w).is_some()) {
        return Some((
            vec![
                BuildStep { action: "addPrimitive".into(), target: Some("Top".into()), shape: Some("cube".into()), width: Some(100.0), height: Some(5.0), depth: Some(60.0), z: Some(45.0), ..Default::default() },
                BuildStep { action: "addPrimitive".into(), target: Some("Leg 1".into()), shape: Some("cylinder".into()), width: Some(6.0), height: Some(45.0), depth: Some(6.0), x: Some(-44.0), y: Some(-24.0), ..Default::default() },
                BuildStep { action: "addPrimitive".into(), target: Some("Leg 2".into()), shape: Some("cylinder".into()), width: Some(6.0), height: Some(45.0), depth: Some(6.0), x: Some(44.0), y: Some(-24.0), ..Default::default() },
                BuildStep { action: "addPrimitive".into(), target: Some("Leg 3".into()), shape: Some("cylinder".into()), width: Some(6.0), height: Some(45.0), depth: Some(6.0), x: Some(-44.0), y: Some(24.0), ..Default::default() },
                BuildStep { action: "addPrimitive".into(), target: Some("Leg 4".into()), shape: Some("cylinder".into()), width: Some(6.0), height: Some(45.0), depth: Some(6.0), x: Some(44.0), y: Some(24.0), ..Default::default() },
            ],
            vec!["Top"],
        ));
    }
    if ["snowman", "snowmen"].iter().any(|&w| find_whole_word(clause, w).is_some()) {
        return Some((
            vec![
                BuildStep { action: "addPrimitive".into(), target: Some("Bottom".into()), shape: Some("sphere".into()), width: Some(40.0), ..Default::default() },
                BuildStep { action: "addPrimitive".into(), target: Some("Middle".into()), shape: Some("sphere".into()), width: Some(30.0), z: Some(-4.0), relation: Some("onTopOf".into()), relative_to: Some("Bottom".into()), ..Default::default() },
                BuildStep { action: "addPrimitive".into(), target: Some("Head".into()), shape: Some("sphere".into()), width: Some(20.0), z: Some(-3.0), relation: Some("onTopOf".into()), relative_to: Some("Middle".into()), ..Default::default() },
                BuildStep { action: "setColor".into(), target: Some("Bottom".into()), color: Some("white".into()), ..Default::default() },
                BuildStep { action: "setColor".into(), target: Some("Middle".into()), color: Some("white".into()), ..Default::default() },
                BuildStep { action: "setColor".into(), target: Some("Head".into()), color: Some("white".into()), ..Default::default() },
            ],
            vec!["Bottom", "Middle", "Head"],
        ));
    }
    if ["house", "houses", "cabin", "hut", "cottage"].iter().any(|&w| find_whole_word(clause, w).is_some()) {
        return Some((
            vec![
                BuildStep { action: "addPrimitive".into(), target: Some("Base".into()), shape: Some("cube".into()), width: Some(40.0), ..Default::default() },
                BuildStep { action: "addPrimitive".into(), target: Some("Roof".into()), shape: Some("cone".into()), width: Some(50.0), height: Some(25.0), depth: Some(50.0), relation: Some("onTopOf".into()), relative_to: Some("Base".into()), ..Default::default() },
                BuildStep { action: "setColor".into(), target: Some("Roof".into()), color: Some("maroon".into()), ..Default::default() },
            ],
            vec!["Base"],
        ));
    }
    if ["tower", "towers", "skyscraper"].iter().any(|&w| find_whole_word(clause, w).is_some()) {
        return Some((
            vec![
                BuildStep { action: "addPrimitive".into(), target: Some("Tower".into()), shape: Some("cube".into()), width: Some(20.0), ..Default::default() },
                BuildStep { action: "addModifier".into(), target: Some("Tower".into()), relation: Some("onTopOf".into()), modifier: Some("array".into()), amount: Some(5.0), count: Some(5), ..Default::default() },
            ],
            vec!["Tower"],
        ));
    }
    None
}

// ── Cloud Request Builder, Exchange Ladder, Response Parser & HTTP Status Mapping ──

#[derive(Clone)]
struct AnthropicOpts {
    force_tool: bool,
    strict: bool,
    refusal_fallbacks: bool,
}

#[derive(Clone)]
enum ExchangeKind {
    Anthropic { key: String, model: String, opts: AnthropicOpts, attempt: usize },
    OpenAI { provider: String, key: Option<String>, model: String, base_url: String, formats: Vec<&'static str>, index: usize },
    Gemini { key: String, model: String },
}

struct HeldExchange {
    kind: ExchangeKind,
    transcript: String,
    scene_text: String,
    label: String,
    last_model: Option<String>,
}

static EXCHANGE_NEXT: AtomicI64 = AtomicI64::new(1);
static EXCHANGES: OnceLock<Mutex<HashMap<i64, HeldExchange>>> = OnceLock::new();

fn exchanges() -> &'static Mutex<HashMap<i64, HeldExchange>> {
    EXCHANGES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn supports_forced_tool(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    let rejects = ["claude-opus-5-5", "claude-fable-5-1", "claude-mythos"];
    if rejects.iter().any(|&r| m.starts_with(r)) { return false; }
    let known = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5", "claude-fable-5", "claude-opus-4", "claude-sonnet-4", "claude-3"];
    known.iter().any(|&k| m == k || m.starts_with(&format!("{k}-")) || m.starts_with(&format!("{k}@")))
}

fn wants_refusal_fallbacks(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    m.starts_with("claude-opus-5") || m.starts_with("claude-fable-5")
}

fn build_plan_schema(flavor: &str) -> Value {
    let ty = |t: &str| if flavor == "gemini" { t.to_ascii_uppercase() } else { t.to_string() };
    let fields: &[(&str, &str, Option<&[&str]>, bool)] = &[
        ("action", "string", Some(VALID_ACTIONS), true),
        ("target", "string", None, false),
        ("shape", "string", Some(VALID_SHAPES), false),
        ("width", "number", None, false),
        ("height", "number", None, false),
        ("depth", "number", None, false),
        ("x", "number", None, false),
        ("y", "number", None, false),
        ("z", "number", None, false),
        ("relation", "string", Some(VALID_RELATIONS), false),
        ("relativeTo", "string", None, false),
        ("rotationDegrees", "number", None, false),
        ("scaleFactor", "number", None, false),
        ("color", "string", None, false),
        ("finish", "string", Some(VALID_FINISHES), false),
        ("modifier", "string", Some(VALID_MODIFIERS), false),
        ("amount", "number", None, false),
        ("booleanOp", "string", Some(VALID_BOOLEANS), false),
        ("tool", "string", None, false),
        ("count", "integer", None, false),
    ];
    let mut props = serde_json::Map::new();
    for &(name, kind, en, req) in fields {
        let mut base = serde_json::Map::new();
        base.insert("type".into(), json!(ty(kind)));
        if let Some(vals) = en { base.insert("enum".into(), json!(vals)); }
        let val = if req || flavor == "anthropic" {
            Value::Object(base)
        } else if flavor == "gemini" {
            base.insert("nullable".into(), json!(true));
            Value::Object(base)
        } else {
            json!({"anyOf": [Value::Object(base), {"type": "null"}]})
        };
        props.insert(name.into(), val);
    }
    let req_fields: Vec<&str> = if flavor == "openAIStrict" {
        fields.iter().map(|f| f.0).collect()
    } else {
        vec!["action"]
    };
    let mut step = json!({"type": ty("object"), "properties": props, "required": req_fields});
    if flavor != "gemini" {
        step["additionalProperties"] = json!(false);
    }
    let mut plan = json!({
        "type": ty("object"),
        "properties": {
            "summary": {"type": ty("string")},
            "steps": {"type": ty("array"), "items": step}
        },
        "required": ["summary", "steps"]
    });
    if flavor != "gemini" {
        plan["additionalProperties"] = json!(false);
    }
    plan
}

fn cloud_error(provider: &str, reason: &str, detail: Option<&str>) -> BridgeError {
    let mut args = vec![provider.to_string()];
    if let Some(d) = detail { args.push(d.to_string()); }
    BridgeError::with_args(format!("cloud.{reason}"), args)
}

fn error_message_from_body(body: &str) -> String {
    let Ok(v) = serde_json::from_str::<Value>(body) else { return String::new() };
    if let Some(m) = v.get("error").and_then(|e| e.get("message")).and_then(|x| x.as_str()) { return m.to_string(); }
    if let Some(m) = v.get("error").and_then(|x| x.as_str()) { return m.to_string(); }
    if let Some(m) = v.get("message").and_then(|x| x.as_str()) { return m.to_string(); }
    if let Some(m) = v.get(0).and_then(|e| e.get("error")).and_then(|e| e.get("message")).and_then(|x| x.as_str()) { return m.to_string(); }
    String::new()
}

fn map_http_status(status: i32, body: &str, provider: &str, model: Option<&str>) -> BridgeError {
    let message = error_message_from_body(body);
    let prefix: String = body.chars().take(2000).collect();
    let lower = format!("{message} {prefix}").to_ascii_lowercase();
    let key_problem = lower.contains("api key not valid")
        || lower.contains("api_key_invalid")
        || lower.contains("invalid_api_key")
        || lower.contains("invalid x-api-key")
        || lower.contains("incorrect api key")
        || lower.contains("authentication_error");
    let model_problem = lower.contains("model_not_found")
        || lower.contains("not_found_error")
        || (lower.contains("model") && (lower.contains("not found") || lower.contains("does not exist")));
    match status {
        401 => cloud_error(provider, "invalidKey", None),
        402 => cloud_error(provider, "quotaExceeded", None),
        403 => if key_problem { cloud_error(provider, "invalidKey", None) } else { cloud_error(provider, "permissionDenied", None) },
        404 => {
            if provider == "custom" && !model_problem {
                cloud_error(provider, "notFoundAtAddress", None)
            } else {
                cloud_error(provider, "modelNotFound", Some(model.unwrap_or("")))
            }
        }
        408 => cloud_error(provider, "timeout", None),
        429 => {
            if lower.contains("insufficient_quota") || lower.contains("billing") || lower.contains("credit") {
                cloud_error(provider, "quotaExceeded", None)
            } else {
                cloud_error(provider, "rateLimited", None)
            }
        }
        400 | 422 => {
            if key_problem {
                cloud_error(provider, "invalidKey", None)
            } else if model_problem {
                cloud_error(provider, "modelNotFound", Some(model.unwrap_or("")))
            } else if lower.contains("credit balance") || lower.contains("billing") {
                cloud_error(provider, "quotaExceeded", None)
            } else {
                let d: String = message.chars().take(200).collect();
                cloud_error(provider, "badRequest", Some(&d))
            }
        }
        503 | 529 => cloud_error(provider, "overloaded", None),
        500..=599 => cloud_error(provider, "server", Some(&status.to_string())),
        _ => {
            let d = if message.is_empty() { format!("HTTP {status}") } else { message.chars().take(200).collect() };
            cloud_error(provider, "badRequest", Some(&d))
        }
    }
}

fn parse_cloud_plan(kind: &ExchangeKind, body: &str) -> Result<BuildPlan, BridgeError> {
    let provider = match kind {
        ExchangeKind::Anthropic { .. } => "anthropic",
        ExchangeKind::OpenAI { provider, .. } => provider.as_str(),
        ExchangeKind::Gemini { .. } => "gemini",
    };
    let v: Value = serde_json::from_str(body).map_err(|_| cloud_error(provider, "badResponse", Some("not JSON")))?;
    match kind {
        ExchangeKind::Anthropic { .. } => {
            let blocks = v.get("content").and_then(|x| x.as_array()).cloned().unwrap_or_default();
            if let Some(input) = blocks
                .iter()
                .find(|b| b.get("type").and_then(|x| x.as_str()) == Some("tool_use") && b.get("name").and_then(|x| x.as_str()) == Some(TOOL_NAME))
                .or_else(|| blocks.iter().find(|b| b.get("type").and_then(|x| x.as_str()) == Some("tool_use")))
                .and_then(|b| b.get("input"))
            {
                return parse_plan_from_value(input).map_err(|r| cloud_error("anthropic", "badResponse", Some(r)));
            }
            if v.get("stop_reason").and_then(|x| x.as_str()) == Some("refusal") {
                return Err(cloud_error("anthropic", "refused", None));
            }
            let text: String = blocks
                .iter()
                .filter_map(|b| (b.get("type").and_then(|x| x.as_str()) == Some("text")).then(|| b.get("text").and_then(|x| x.as_str())).flatten())
                .collect::<Vec<_>>()
                .join("\n");
            if let Ok(p) = parse_plan_from_text(&text) { return Ok(p); }
            if v.get("stop_reason").and_then(|x| x.as_str()) == Some("max_tokens") {
                return Err(cloud_error("anthropic", "badResponse", Some("the answer was cut off")));
            }
            Err(cloud_error("anthropic", "badResponse", Some("no plan in the answer")))
        }
        ExchangeKind::OpenAI { provider, .. } => {
            let Some(msg) = v.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("message")) else {
                return Err(cloud_error(provider, "badResponse", Some("no choices in the answer")));
            };
            if msg.get("refusal").and_then(|x| x.as_str()).is_some_and(|s| !s.is_empty()) {
                return Err(cloud_error(provider, "refused", None));
            }
            let mut text = msg.get("content").and_then(|x| x.as_str()).unwrap_or("").to_string();
            if text.is_empty() {
                if let Some(parts) = msg.get("content").and_then(|x| x.as_array()) {
                    text = parts.iter().filter_map(|p| p.get("text").and_then(|x| x.as_str())).collect();
                }
            }
            if text.is_empty() {
                if let Some(args) = msg.get("tool_calls").and_then(|t| t.get(0)).and_then(|t| t.get("function")).and_then(|f| f.get("arguments")).and_then(|x| x.as_str()) {
                    text = args.to_string();
                }
            }
            if text.is_empty() {
                let finish = v.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("finish_reason")).and_then(|x| x.as_str());
                return Err(cloud_error(provider, "badResponse", Some(if finish == Some("length") { "the answer was cut off" } else { "empty answer" })));
            }
            parse_plan_from_text(&text).map_err(|r| cloud_error(provider, "badResponse", Some(r)))
        }
        ExchangeKind::Gemini { .. } => {
            if v.get("promptFeedback").and_then(|p| p.get("blockReason")).is_some() {
                return Err(cloud_error("gemini", "refused", None));
            }
            let Some(cand) = v.get("candidates").and_then(|c| c.get(0)) else {
                return Err(cloud_error("gemini", "badResponse", Some("no candidates in the answer")));
            };
            let parts = cand.get("content").and_then(|c| c.get("parts")).and_then(|x| x.as_array()).cloned().unwrap_or_default();
            let text: String = parts
                .iter()
                .filter(|p| p.get("thought").and_then(|x| x.as_bool()) != Some(true))
                .filter_map(|p| p.get("text").and_then(|x| x.as_str()))
                .collect();
            if text.is_empty() {
                return match cand.get("finishReason").and_then(|x| x.as_str()) {
                    Some("SAFETY" | "PROHIBITED_CONTENT" | "BLOCKLIST" | "RECITATION") => Err(cloud_error("gemini", "refused", None)),
                    Some("MAX_TOKENS") => Err(cloud_error("gemini", "badResponse", Some("the answer was cut off"))),
                    _ => Err(cloud_error("gemini", "badResponse", Some("empty answer"))),
                };
            }
            parse_plan_from_text(&text).map_err(|r| cloud_error("gemini", "badResponse", Some(r)))
        }
    }
}

fn is_openai_chat_model(id: &str) -> bool {
    let m = id.to_ascii_lowercase();
    if !(m.starts_with("gpt-") || m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4") || m.starts_with("chatgpt")) {
        return false;
    }
    let excluded = ["embedding", "tts", "whisper", "dall-e", "audio", "realtime", "transcribe", "image", "moderation", "search", "instruct"];
    !excluded.iter().any(|&e| m.contains(e))
}

fn plan_response_fields(plan: &BuildPlan, warnings: &[BridgeError], engine: &str, engine_label: Option<&str>) -> serde_json::Map<String, Value> {
    let mut m = serde_json::Map::new();
    m.insert("plan".into(), serde_json::to_value(plan).unwrap_or(json!({})));
    m.insert("engine".into(), json!(engine));
    m.insert(
        "warnings".into(),
        Value::Array(warnings.iter().map(|w| json!({"key": w.key, "args": w.args})).collect()),
    );
    m.insert("steps".into(), Value::Array(plan.steps.iter().map(step_display_json).collect()));
    if let Some(l) = engine_label {
        m.insert("engineLabel".into(), json!(l));
    }
    m
}

// ── Pure VoiceNative.call entry point ──

pub fn call(name: &str, args_json: &str, payload: Option<&str>) -> String {
    let a: Value = serde_json::from_str(if args_json.is_empty() { "{}" } else { args_json }).unwrap_or(json!({}));
    match dispatch(name, &a, payload) {
        Ok(fields) => ok_envelope(fields),
        Err(e) => e.to_envelope(),
    }
}

fn display_names_map(v: Option<&Value>) -> HashMap<String, String> {
    let mut m = HashMap::new();
    if let Some(obj) = v.and_then(|x| x.as_object()) {
        for (k, val) in obj {
            if let Some(s) = val.as_str() { m.insert(k.clone(), s.to_string()); }
        }
    }
    m
}

fn parse_selection(v: Option<&Value>) -> Result<(String, Option<String>, Option<String>), BridgeError> {
    let o = v.and_then(|x| x.as_object()).ok_or_else(|| BridgeError::with_args("engine.badJSON", vec!["provider".into()]))?;
    let p = o.get("provider").and_then(|x| x.as_str()).filter(|p| ALL_PROVIDERS.contains(p))
        .ok_or_else(|| BridgeError::with_args("engine.badJSON", vec!["provider".into()]))?
        .to_string();
    let model = o.get("model").and_then(|x| x.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string());
    let base = o.get("baseURL").and_then(|x| x.as_str()).and_then(parse_url_parts).map(|(u, _, _)| u);
    Ok((p, model, base))
}

fn dispatch(name: &str, a: &Value, payload: Option<&str>) -> Result<serde_json::Map<String, Value>, BridgeError> {
    let mut out = serde_json::Map::new();
    match name {
        "quickPlan" => {
            let req = a.get("request").and_then(|x| x.as_str()).unwrap_or("");
            let names: Vec<String> = a
                .get("sceneNames")
                .and_then(|x| x.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let (plan, warnings) = quick_plan(req, &names)?;
            return Ok(plan_response_fields(&plan, &warnings, "keyword", None));
        }
        "parsePlanText" => {
            let raw = parse_plan_from_text(payload.unwrap_or(""))
                .map_err(|r| BridgeError::with_args("plan.badAnswer", vec![r.to_string()]))?;
            let (plan, fixes) = sanitize_plan(raw);
            if plan.steps.is_empty() {
                return Err(BridgeError::with_args("plan.badAnswer", vec!["None of the steps could be understood.".into()]));
            }
            return Ok(plan_response_fields(&plan, &fixes, "appleIntelligence", None));
        }
        "display" => {
            let plan = parse_plan_from_value(a.get("plan").unwrap_or(&Value::Null))
                .map_err(|r| BridgeError::with_args("plan.badAnswer", vec![r.to_string()]))?;
            out.insert("steps".into(), Value::Array(plan.steps.iter().map(step_display_json).collect()));
            Ok(out)
        }
        "prompt" => {
            let scene_text = a.get("sceneText").and_then(|x| x.as_str()).unwrap_or("(empty scene)");
            let req = a.get("request").and_then(|x| x.as_str()).unwrap_or("");
            out.insert("system".into(), json!(format!("{PLAN_INSTRUCTIONS}\n\n{JSON_OUTPUT_NOTE}")));
            out.insert("user".into(), json!(format!("Scene now:\n{scene_text}\n\nUser: {req}")));
            Ok(out)
        }
        "providers" => {
            let list: Vec<Value> = ALL_PROVIDERS
                .iter()
                .map(|&p| {
                    json!({
                        "id": p,
                        "cloud": is_cloud(p),
                        "brand": brand_name(p),
                        "requiresKey": matches!(p, "anthropic" | "openAI" | "gemini"),
                        "suggestedModels": suggested_models(p),
                        "keyURL": key_url(p),
                        "privacyURL": privacy_url(p),
                        "recipient": recipient_for_selection(p, None),
                    })
                })
                .collect();
            out.insert("providers".into(), Value::Array(list));
            out.insert("consentVersion".into(), json!(CONSENT_VERSION));
            Ok(out)
        }
        "modelInfo" => {
            let (p, model, base) = parse_selection(a.get("selection"))?;
            let names = display_names_map(a.get("displayNames"));
            let (mname, label) = cloud_label(&p, model.as_deref(), &names);
            out.insert("name".into(), json!(mname));
            out.insert("label".into(), json!(label));
            out.insert("recipient".into(), json!(recipient_for_selection(&p, base.as_deref())));
            Ok(out)
        }
        "prettyModel" => {
            let id = a.get("id").and_then(|x| x.as_str()).unwrap_or("");
            let names = display_names_map(a.get("displayNames"));
            out.insert("name".into(), json!(pretty_model_name(id, &names)));
            Ok(out)
        }
        "validateBaseURL" => {
            let text = a.get("text").and_then(|x| x.as_str()).unwrap_or("");
            let (url, scheme, host) = parse_url_parts(text).ok_or_else(|| cloud_error("custom", "invalidBaseURL", None))?;
            out.insert("url".into(), json!(url));
            out.insert("host".into(), json!(host));
            out.insert("cleartextAllowed".into(), json!(allows_cleartext(&scheme, &host)));
            Ok(out)
        }
        "consentCovers" => {
            let (p, _, base) = parse_selection(a.get("selection"))?;
            if !is_cloud(&p) {
                out.insert("covers".into(), json!(true));
                return Ok(out);
            }
            let covers = (|| {
                let rec = a.get("record")?.as_object()?;
                let ver = rec.get("version")?.as_i64()? as i32;
                let rec_host = rec.get("recipient")?.as_str()?;
                let target_host = recipient_for_selection(&p, base.as_deref())?;
                Some(ver >= CONSENT_VERSION && rec_host.eq_ignore_ascii_case(&target_host))
            })()
            .unwrap_or(false);
            out.insert("covers".into(), json!(covers));
            Ok(out)
        }
        "compileStandalone" => {
            let plan = parse_plan_from_value(a.get("plan").unwrap_or(&Value::Null))
                .map_err(|r| BridgeError::with_args("plan.badAnswer", vec![r.to_string()]))?;
            let mut empty = Scene::default();
            compile_plan_against_scene(&plan, &mut empty)?;
            Ok(out)
        }
        "cloudBegin" => {
            let (p, model_opt, base_opt) = parse_selection(a.get("selection"))?;
            let key = a.get("key").and_then(|x| x.as_str()).unwrap_or("").trim().to_string();
            let req = a.get("request").and_then(|x| x.as_str()).unwrap_or("").to_string();
            let scene_text = a.get("sceneText").and_then(|x| x.as_str()).unwrap_or("(empty scene)").to_string();
            let names = display_names_map(a.get("displayNames"));
            if matches!(p.as_str(), "anthropic" | "openAI" | "gemini") && key.is_empty() {
                return Err(cloud_error(&p, "missingKey", None));
            }
            let model = model_opt
                .clone()
                .or_else(|| suggested_models(&p).first().map(|s| s.to_string()))
                .ok_or_else(|| cloud_error(&p, "badRequest", Some("enter a model ID in AI Models")))?;
            let (_, label) = cloud_label(&p, Some(&model), &names);
            let kind = match p.as_str() {
                "anthropic" => ExchangeKind::Anthropic {
                    key,
                    opts: AnthropicOpts {
                        force_tool: supports_forced_tool(&model),
                        strict: true,
                        refusal_fallbacks: wants_refusal_fallbacks(&model),
                    },
                    model,
                    attempt: 0,
                },
                "openAI" => ExchangeKind::OpenAI {
                    provider: "openAI".into(),
                    key: Some(key),
                    model,
                    base_url: "https://api.openai.com/v1".into(),
                    formats: vec!["jsonSchema"],
                    index: 0,
                },
                "custom" => {
                    let base = base_opt.ok_or_else(|| cloud_error("custom", "invalidBaseURL", None))?;
                    ExchangeKind::OpenAI {
                        provider: "custom".into(),
                        key: (!key.is_empty()).then_some(key),
                        model,
                        base_url: base,
                        formats: vec!["jsonSchema", "jsonObject", "none"],
                        index: 0,
                    }
                }
                "gemini" => {
                    let clean_m = model.strip_prefix("models/").unwrap_or(&model).to_string();
                    ExchangeKind::Gemini { key, model: clean_m }
                }
                _ => return Err(cloud_error(&p, "badRequest", Some("not a cloud provider"))),
            };
            let handle = EXCHANGE_NEXT.fetch_add(1, Ordering::Relaxed);
            exchanges().lock().unwrap().insert(handle, HeldExchange { kind, transcript: req, scene_text, label, last_model: None });
            out.insert("handle".into(), json!(handle));
            Ok(out)
        }
        "cloudNext" => {
            let h = a.get("handle").and_then(|x| x.as_i64()).unwrap_or(0);
            let mut map = exchanges().lock().unwrap();
            let held = map.get_mut(&h).ok_or_else(|| BridgeError::new("engine.badHandle"))?;
            let user_turn = format!("Scene now:\n{}\n\nUser: {}", held.scene_text, held.transcript);
            let req = match &held.kind {
                ExchangeKind::Anthropic { key, model, opts, .. } => {
                    held.last_model = Some(model.clone());
                    let mut tool = json!({
                        "name": TOOL_NAME,
                        "description": TOOL_DESCRIPTION,
                        "input_schema": build_plan_schema("anthropic"),
                    });
                    if opts.strict { tool["strict"] = json!(true); }
                    let tc = if opts.force_tool {
                        json!({"type": "tool", "name": TOOL_NAME})
                    } else {
                        json!({"type": "auto", "disable_parallel_tool_use": true})
                    };
                    let mut body = json!({
                        "model": model,
                        "max_tokens": 16000,
                        "system": format!("{PLAN_INSTRUCTIONS}\n\n{TOOL_OUTPUT_NOTE}"),
                        "messages": [{"role": "user", "content": user_turn}],
                        "tools": [tool],
                        "tool_choice": tc,
                    });
                    let mut headers = vec![
                        ("Accept", "application/json".to_string()),
                        ("Content-Type", "application/json".to_string()),
                        ("anthropic-version", "2023-06-01".to_string()),
                        ("x-api-key", key.clone()),
                    ];
                    if opts.refusal_fallbacks {
                        body["fallbacks"] = json!("default");
                        headers.push(("anthropic-beta", "server-side-fallback-2026-07-01".to_string()));
                    }
                    json!({
                        "url": "https://api.anthropic.com/v1/messages",
                        "method": "POST",
                        "headers": headers,
                        "body": body.to_string(),
                        "timeout": 90.0,
                    })
                }
                ExchangeKind::OpenAI { key, model, base_url, formats, index, .. } => {
                    held.last_model = Some(model.clone());
                    let fmt = formats[*index];
                    let note = if fmt == "jsonSchema" {
                        JSON_OUTPUT_NOTE.to_string()
                    } else {
                        format!("{JSON_OUTPUT_NOTE}\nJSON Schema:\n{}", build_plan_schema("anthropic"))
                    };
                    let mut body = json!({
                        "model": model,
                        "messages": [
                            {"role": "system", "content": format!("{PLAN_INSTRUCTIONS}\n\n{note}")},
                            {"role": "user", "content": user_turn},
                        ]
                    });
                    if fmt == "jsonSchema" {
                        body["response_format"] = json!({
                            "type": "json_schema",
                            "json_schema": {"name": "build_plan", "strict": true, "schema": build_plan_schema("openAIStrict")}
                        });
                    } else if fmt == "jsonObject" {
                        body["response_format"] = json!({"type": "json_object"});
                    }
                    let mut headers = vec![
                        ("Accept", "application/json".to_string()),
                        ("Content-Type", "application/json".to_string()),
                    ];
                    if let Some(k) = key {
                        headers.push(("Authorization", format!("Bearer {k}")));
                    }
                    json!({
                        "url": format!("{base_url}/chat/completions"),
                        "method": "POST",
                        "headers": headers,
                        "body": body.to_string(),
                        "timeout": 90.0,
                    })
                }
                ExchangeKind::Gemini { key, model } => {
                    held.last_model = Some(model.clone());
                    let body = json!({
                        "systemInstruction": {"parts": [{"text": format!("{PLAN_INSTRUCTIONS}\n\n{JSON_OUTPUT_NOTE}")}]},
                        "contents": [{"role": "user", "parts": [{"text": user_turn}]}],
                        "generationConfig": {"responseMimeType": "application/json", "responseSchema": build_plan_schema("gemini")}
                    });
                    json!({
                        "url": format!("https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"),
                        "method": "POST",
                        "headers": [
                            ["Accept", "application/json"],
                            ["Content-Type", "application/json"],
                            ["x-goog-api-key", key],
                        ],
                        "body": body.to_string(),
                        "timeout": 90.0,
                    })
                }
            };
            out.insert("request".into(), req);
            Ok(out)
        }
        "cloudReceive" => {
            let h = a.get("handle").and_then(|x| x.as_i64()).unwrap_or(0);
            let status = a.get("status").and_then(|x| x.as_i64()).unwrap_or(0) as i32;
            let body = payload.unwrap_or("");
            let mut map = exchanges().lock().unwrap();
            let held = map.get_mut(&h).ok_or_else(|| BridgeError::new("engine.badHandle"))?;
            let provider = match &held.kind {
                ExchangeKind::Anthropic { .. } => "anthropic".to_string(),
                ExchangeKind::OpenAI { provider, .. } => provider.clone(),
                ExchangeKind::Gemini { .. } => "gemini".to_string(),
            };
            if !(200..300).contains(&status) {
                let err = map_http_status(status, body, &provider, held.last_model.as_deref());
                // Retry ladder on 400
                if err.key == "cloud.badRequest" {
                    match &mut held.kind {
                        ExchangeKind::Anthropic { opts, attempt, .. } if *attempt < 2 => {
                            let msg = error_message_from_body(body).to_ascii_lowercase();
                            let relaxed = if msg.contains("tool_choice") && opts.force_tool {
                                opts.force_tool = false; true
                            } else if msg.contains("fallback") && opts.refusal_fallbacks {
                                opts.refusal_fallbacks = false; true
                            } else if (msg.contains("strict") || msg.contains("schema")) && opts.strict {
                                opts.strict = false; true
                            } else { false };
                            if relaxed {
                                *attempt += 1;
                                out.insert("retry".into(), json!(true));
                                return Ok(out);
                            }
                        }
                        ExchangeKind::OpenAI { formats, index, .. } if *index + 1 < formats.len() => {
                            *index += 1;
                            out.insert("retry".into(), json!(true));
                            return Ok(out);
                        }
                        _ => {}
                    }
                }
                return Err(err);
            }
            let raw = parse_cloud_plan(&held.kind, body)?;
            let (plan, fixes) = sanitize_plan(raw);
            if plan.steps.is_empty() {
                return Err(cloud_error(&provider, "badResponse", Some("no usable steps")));
            }
            Ok(plan_response_fields(&plan, &fixes, "cloud", Some(&held.label)))
        }
        "cloudFree" => {
            let h = a.get("handle").and_then(|x| x.as_i64()).unwrap_or(0);
            exchanges().lock().unwrap().remove(&h);
            Ok(out)
        }
        "modelsRequest" => {
            let p = a.get("provider").and_then(|x| x.as_str()).unwrap_or("");
            let (_, _, base_opt) = parse_selection(a.get("selection")).unwrap_or((p.to_string(), None, None));
            let key = a.get("key").and_then(|x| x.as_str()).unwrap_or("").trim();
            if matches!(p, "anthropic" | "openAI" | "gemini") && key.is_empty() {
                return Err(cloud_error(p, "missingKey", None));
            }
            let req = match p {
                "anthropic" => json!({
                    "url": "https://api.anthropic.com/v1/models?limit=1000",
                    "method": "GET",
                    "headers": [["Accept", "application/json"], ["anthropic-version", "2023-06-01"], ["x-api-key", key]],
                    "body": null,
                    "timeout": 20.0,
                }),
                "openAI" => json!({
                    "url": "https://api.openai.com/v1/models",
                    "method": "GET",
                    "headers": [["Accept", "application/json"], ["Authorization", format!("Bearer {key}")]],
                    "body": null,
                    "timeout": 20.0,
                }),
                "gemini" => json!({
                    "url": "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
                    "method": "GET",
                    "headers": [["Accept", "application/json"], ["x-goog-api-key", key]],
                    "body": null,
                    "timeout": 20.0,
                }),
                "custom" => {
                    let base = base_opt.ok_or_else(|| cloud_error("custom", "invalidBaseURL", None))?;
                    let mut headers = vec![vec!["Accept", "application/json"]];
                    let auth = format!("Bearer {key}");
                    let mut hdrs_val: Vec<Value> = headers.drain(..).map(|h| json!(h)).collect();
                    if !key.is_empty() { hdrs_val.push(json!(["Authorization", auth])); }
                    json!({
                        "url": format!("{base}/models"),
                        "method": "GET",
                        "headers": hdrs_val,
                        "body": null,
                        "timeout": 20.0,
                    })
                }
                _ => return Err(cloud_error(p, "badRequest", Some("not a cloud provider"))),
            };
            out.insert("request".into(), req);
            Ok(out)
        }
        "parseModels" => {
            let p = a.get("provider").and_then(|x| x.as_str()).unwrap_or("");
            let status = a.get("status").and_then(|x| x.as_i64()).unwrap_or(0) as i32;
            let body = payload.unwrap_or("");
            if !(200..300).contains(&status) {
                return Err(map_http_status(status, body, p, None));
            }
            let v: Value = serde_json::from_str(body).map_err(|_| cloud_error(p, "badResponse", Some("not JSON")))?;
            let mut display_names = serde_json::Map::new();
            let models: Vec<String> = match p {
                "anthropic" => {
                    let items = v.get("data").and_then(|x| x.as_array()).ok_or_else(|| cloud_error(p, "badResponse", Some("no model list")))?;
                    for item in items {
                        if let (Some(id), Some(dn)) = (item.get("id").and_then(|x| x.as_str()), item.get("display_name").and_then(|x| x.as_str())) {
                            if !dn.is_empty() { display_names.insert(id.into(), json!(dn)); }
                        }
                    }
                    items.iter().filter_map(|i| i.get("id").and_then(|x| x.as_str()).map(|s| s.to_string())).collect()
                }
                "openAI" | "custom" => {
                    let items = v.get("data").or_else(|| v.get("models")).and_then(|x| x.as_array())
                        .ok_or_else(|| cloud_error(p, "badResponse", Some("no model list")))?;
                    let mut ids: Vec<String> = items
                        .iter()
                        .filter_map(|i| i.get("id").or_else(|| i.get("name")).and_then(|x| x.as_str()).map(|s| s.to_string()))
                        .collect();
                    if p == "openAI" {
                        ids.retain(|id| is_openai_chat_model(id));
                        ids.sort_by(|a, b| b.cmp(a));
                    }
                    ids
                }
                "gemini" => {
                    let items = v.get("models").and_then(|x| x.as_array()).ok_or_else(|| cloud_error(p, "badResponse", Some("no model list")))?;
                    items
                        .iter()
                        .filter_map(|i| {
                            let name = i.get("name").and_then(|x| x.as_str())?;
                            let methods = i.get("supportedGenerationMethods").and_then(|x| x.as_array());
                            let ok = methods.is_none_or(|m| m.iter().any(|x| x.as_str() == Some("generateContent")));
                            if !ok { return None; }
                            let id = name.strip_prefix("models/").unwrap_or(name);
                            (!id.to_ascii_lowercase().contains("embedding")).then(|| id.to_string())
                        })
                        .collect()
                }
                _ => return Err(cloud_error(p, "badRequest", Some("not a cloud provider"))),
            };
            out.insert("models".into(), json!(models));
            out.insert("displayNames".into(), Value::Object(display_names));
            Ok(out)
        }
        "statusError" => {
            let p = a.get("provider").and_then(|x| x.as_str()).unwrap_or("custom");
            let status = a.get("status").and_then(|x| x.as_i64()).unwrap_or(500) as i32;
            let model = a.get("model").and_then(|x| x.as_str());
            Err(map_http_status(status, payload.unwrap_or(""), p, model))
        }
        _ => Err(BridgeError::with_args("engine.unknownCall", vec![name.to_string()])),
    }
}

// ── Document-bound Voice PlanCompiler & PlanBuild (`doc_voice_call`) ──

pub struct PlanBuildState {
    plan: BuildPlan,
    resolver: StepResolver,
    created: Vec<String>,
}

struct StepResolver {
    created_by_name: HashMap<String, String>,
    created_order: Vec<String>,
}

impl StepResolver {
    fn new() -> Self {
        Self { created_by_name: HashMap::new(), created_order: Vec::new() }
    }

    fn resolve(&self, name: Option<&str>, scene: &Scene) -> Result<String, BridgeError> {
        let raw = name.unwrap_or("last").trim();
        if raw.is_empty() || is_pronoun(Some(raw)) {
            if let Some(id) = self.created_order.last() {
                if scene.objects.iter().any(|o| &o.id == id) { return Ok(id.clone()); }
            }
            if let Some(o) = scene.objects.last() { return Ok(o.id.clone()); }
            return Err(BridgeError::with_args("plan.unknownTarget", vec![raw.to_string()]));
        }
        let lower = raw.to_ascii_lowercase();
        if let Some(id) = self.created_by_name.get(&lower) {
            if scene.objects.iter().any(|o| &o.id == id) { return Ok(id.clone()); }
        }
        if let Some(o) = scene.objects.iter().rev().find(|o| o.name.eq_ignore_ascii_case(raw)) {
            return Ok(o.id.clone());
        }
        Err(BridgeError::with_args("plan.unknownTarget", vec![raw.to_string()]))
    }

    fn world_bounds(scene: &Scene, id: &str) -> BoundingBox {
        let Some(o) = scene.objects.iter().find(|o| o.id == id) else { return BoundingBox::EMPTY };
        let eval = evaluate_object_mesh(o);
        BoundingBox::from_mesh(&eval, scene_world_matrix(scene, id))
    }

    fn placement_center(rel: &str, size: Vec3, ref_b: BoundingBox) -> Vec3 {
        let c = ref_b.center();
        let rest_z = ref_b.min.z + size.z * 0.5;
        match rel {
            "onTopOf" => Vec3::new(c.x, c.y, ref_b.max.z + size.z * 0.5),
            "leftOf" => Vec3::new(ref_b.min.x - size.x * 0.5 - GAP, c.y, rest_z),
            "rightOf" => Vec3::new(ref_b.max.x + size.x * 0.5 + GAP, c.y, rest_z),
            "inFrontOf" => Vec3::new(c.x, ref_b.min.y - size.y * 0.5 - GAP, rest_z),
            "behind" => Vec3::new(c.x, ref_b.max.y + size.y * 0.5 + GAP, rest_z),
            "inside" => c,
            _ => Vec3::new(c.x, c.y, rest_z), // centeredOn
        }
    }

    fn run_step(&mut self, step: &BuildStep, index: usize, scene: &mut Scene) -> Result<Vec<String>, BridgeError> {
        let mut made = Vec::new();
        let exec = |scene: &mut Scene, cmd: Value| -> Result<Vec<String>, BridgeError> {
            let out = apply_command_value(scene, &cmd, None)?;
            Ok(out.created)
        };
        match step.action.as_str() {
            "addPrimitive" => {
                let Some(ref shape) = step.shape else {
                    return Err(BridgeError::with_args("plan.missingShape", vec![(index + 1).to_string()]));
                };
                let size = primitive_size(shape, step.width, step.height, step.depth);
                let offset = Vec3::new(step.x.unwrap_or(0.0) as f32, step.y.unwrap_or(0.0) as f32, step.z.unwrap_or(0.0) as f32);
                let implicit_ref = if step.relative_to.is_none() { self.resolve(Some("last"), scene).ok() } else { None };
                let pos = if let Some(ref rel) = step.relation {
                    if step.relative_to.is_some() || implicit_ref.is_some() {
                        let rid = match implicit_ref {
                            Some(id) => id,
                            None => self.resolve(step.relative_to.as_deref(), scene)?,
                        };
                        Self::placement_center(rel, size, Self::world_bounds(scene, &rid)) + offset
                    } else {
                        Vec3::new(offset.x, offset.y, offset.z + size.z * 0.5)
                    }
                } else if let Some(ref rname) = step.relative_to {
                    let rid = self.resolve(Some(rname), scene)?;
                    Self::world_bounds(scene, &rid).center() + offset
                } else {
                    Vec3::new(offset.x, offset.y, offset.z + size.z * 0.5)
                };
                let segs = if shape == "cube" || shape == "plane" { 1 } else { 32 };
                let rngs = if shape == "sphere" || shape == "torus" { 16 } else { 1 };
                let name = step.target.as_deref().filter(|s| !s.trim().is_empty() && !is_pronoun(Some(s)));
                let mut body = json!({
                    "_0": {"kind": shape, "size": [size.x, size.y, size.z], "segments": segs, "rings": rngs},
                    "position": [pos.x, pos.y, pos.z],
                });
                if let Some(n) = name { body["name"] = json!(n); }
                let created = exec(scene, json!({"addPrimitive": body}))?;
                if let Some(id) = created.last().cloned() {
                    if let Some(n) = name { self.created_by_name.insert(n.to_ascii_lowercase(), id.clone()); }
                    self.created_order.push(id.clone());
                    if let Some(deg) = step.rotation_degrees {
                        if deg.is_finite() && (deg % 360.0).abs() > 1e-4 {
                            let piv = Self::world_bounds(scene, &id).center();
                            let q = Quat::from_axis_angle(Vec3::Z, (deg as f32).to_radians());
                            exec(scene, json!({"rotate": {"_0": [{"rawValue": id}], "by": [q.x, q.y, q.z, q.w], "pivot": [piv.x, piv.y, piv.z]}}))?;
                        }
                    }
                }
                made.extend(created);
            }
            "move" => {
                let id = self.resolve(step.target.as_deref(), scene)?;
                let b = Self::world_bounds(scene, &id);
                let off = Vec3::new(step.x.unwrap_or(0.0) as f32, step.y.unwrap_or(0.0) as f32, step.z.unwrap_or(0.0) as f32);
                let delta = if let Some(ref rel) = step.relation {
                    let rid = self.resolve(step.relative_to.as_deref().or(Some("last")), scene)?;
                    if rid == id { Vec3::ZERO } else { Self::placement_center(rel, b.size(), Self::world_bounds(scene, &rid)) + off - b.center() }
                } else if let Some(ref rto) = step.relative_to {
                    let rid = self.resolve(Some(rto), scene)?;
                    if rid == id { off } else { Self::world_bounds(scene, &rid).center() + off - b.center() }
                } else {
                    let c = b.center();
                    let t = Vec3::new(
                        step.x.map(|v| v as f32).unwrap_or(c.x),
                        step.y.map(|v| v as f32).unwrap_or(c.y),
                        step.z.map(|v| v as f32 + b.size().z * 0.5).unwrap_or(c.z),
                    );
                    t - c
                };
                if delta.length() > 1e-6 {
                    exec(scene, json!({"translate": {"_0": [{"rawValue": id}], "by": [delta.x, delta.y, delta.z]}}))?;
                }
            }
            "rotate" => {
                let id = self.resolve(step.target.as_deref(), scene)?;
                let ax = rotation_axis(step.x, step.y, step.z);
                let rad = (step.rotation_degrees.unwrap_or(0.0) as f32).to_radians();
                let q = Quat::from_axis_angle(ax, rad);
                let piv = Self::world_bounds(scene, &id).center();
                exec(scene, json!({"rotate": {"_0": [{"rawValue": id}], "by": [q.x, q.y, q.z, q.w], "pivot": [piv.x, piv.y, piv.z]}}))?;
            }
            "scale" => {
                let id = self.resolve(step.target.as_deref(), scene)?;
                let f = step.scale_factor.unwrap_or(1.0) as f32;
                let piv = Self::world_bounds(scene, &id).center();
                exec(scene, json!({"scale": {"_0": [{"rawValue": id}], "by": [f, f, f], "pivot": [piv.x, piv.y, piv.z]}}))?;
            }
            "setColor" => {
                let id = self.resolve(step.target.as_deref(), scene)?;
                let (cname, rgba) = step.color.as_deref().and_then(parse_color).unwrap_or_else(|| ("Clay".into(), [233.0 / 255.0, 201.0 / 255.0, 168.0 / 255.0, 1.0]));
                let (metallic, roughness) = match step.finish.as_deref() {
                    Some("glossy") => (0.0, 0.15),
                    Some("metal") => (1.0, 0.25),
                    _ => (0.0, 0.6),
                };
                exec(scene, json!({
                    "setMaterial": {
                        "_0": [{"rawValue": id}],
                        "_1": {"name": cname, "baseColor": {"r": rgba[0], "g": rgba[1], "b": rgba[2], "a": rgba[3]}, "metallic": metallic, "roughness": roughness}
                    }
                }))?;
            }
            "addModifier" => {
                let id = self.resolve(step.target.as_deref(), scene)?;
                let mod_val = match step.modifier.as_deref() {
                    Some("mirror") => json!({"mirror": {"_0": {"axis": 0, "merge": true, "mergeDistance": 0.01}}}),
                    Some("subdivision") => json!({"subdivision": {"_0": {"levels": step.amount.unwrap_or(2.0).round().clamp(1.0, 3.0) as i64}}}),
                    Some("solidify") => json!({"solidify": {"_0": {"thickness": step.amount.unwrap_or(2.0), "offset": -1.0}}}),
                    Some("array") => {
                        let cnt = step.amount.or(step.count.map(|c| c as f64)).unwrap_or(2.0).round().clamp(2.0, 50.0) as i64;
                        let rel = if step.relation.as_deref() == Some("onTopOf") { [0.0, 0.0, 1.0] } else { [1.1, 0.0, 0.0] };
                        json!({"array": {"_0": {"count": cnt, "relativeOffset": rel, "constantOffset": [0.0, 0.0, 0.0], "useRelative": true}}})
                    }
                    _ => return Ok(made),
                };
                exec(scene, json!({"addModifier": {"_0": {"rawValue": id}, "_1": mod_val}}))?;
            }
            "boolean" => {
                let target_id = self.resolve(step.target.as_deref(), scene)?;
                let Some(ref tool_name) = step.tool else { return Err(BridgeError::new("plan.missingTool")); };
                let tool_id = self.resolve(Some(tool_name), scene)?;
                let op = step.boolean_op.as_deref().unwrap_or("union");
                exec(scene, json!({"boolean": {"_0": op, "target": {"rawValue": target_id}, "tool": {"rawValue": tool_id}, "deleteTool": true}}))?;
            }
            "duplicate" => {
                let id = self.resolve(step.target.as_deref(), scene)?;
                let copies = (step.count.unwrap_or(2).clamp(1, 51) - 1).max(1);
                let sz = Self::world_bounds(scene, &id).size();
                let step_x = (sz.x + GAP).max(10.0);
                for k in 1..=copies {
                    let c = exec(scene, json!({"duplicateObjects": {"_0": [{"rawValue": id}], "offset": [step_x * k as f32, 0.0, 0.0]}}))?;
                    for cid in &c { self.created_order.push(cid.clone()); }
                    made.extend(c);
                }
            }
            "rename" => {
                let id = self.resolve(step.target.as_deref(), scene)?;
                if let Some(new_name) = step.tool.as_deref().or(step.relative_to.as_deref()) {
                    exec(scene, json!({"renameObject": {"_0": {"rawValue": id}, "_1": new_name}}))?;
                    self.created_by_name.insert(new_name.to_ascii_lowercase(), id);
                }
            }
            "delete" => {
                let id = self.resolve(step.target.as_deref(), scene)?;
                exec(scene, json!({"deleteObjects": {"_0": [{"rawValue": id}]}}))?;
            }
            _ => {}
        }
        Ok(made)
    }
}

fn compile_plan_against_scene(plan: &BuildPlan, scratch: &mut Scene) -> Result<(), BridgeError> {
    if plan.steps.is_empty() {
        return Err(BridgeError::new("plan.nothingRecognised"));
    }
    let mut resolver = StepResolver::new();
    for (i, step) in plan.steps.iter().enumerate() {
        // For fast compile validation of booleans, validate target/tool resolution without running heavy SDF mesh extraction
        if step.action == "boolean" {
            let _ = resolver.resolve(step.target.as_deref(), scratch)?;
            let Some(ref tool_name) = step.tool else { return Err(BridgeError::new("plan.missingTool")); };
            let _ = resolver.resolve(Some(tool_name), scratch)?;
            continue;
        }
        resolver.run_step(step, i, scratch)?;
    }
    Ok(())
}

pub fn doc_voice_call(doc: &mut EngineDoc, name: &str, args_json: &str) -> String {
    let a: Value = serde_json::from_str(if args_json.is_empty() { "{}" } else { args_json }).unwrap_or(json!({}));
    let res: Result<serde_json::Map<String, Value>, BridgeError> = (|| {
        let mut out = serde_json::Map::new();
        match name {
            "compile" => {
                let plan = parse_plan_from_value(a.get("plan").unwrap_or(&Value::Null))
                    .map_err(|r| BridgeError::with_args("plan.badAnswer", vec![r.to_string()]))?;
                let mut scratch = doc.scene.clone();
                compile_plan_against_scene(&plan, &mut scratch)?;
                Ok(out)
            }
            "planBegin" => {
                let plan = parse_plan_from_value(a.get("plan").unwrap_or(&Value::Null))
                    .map_err(|r| BridgeError::with_args("plan.badAnswer", vec![r.to_string()]))?;
                if doc.plan_build.take().is_some() {
                    doc.end_undo_group();
                }
                let mut scratch = doc.scene.clone();
                compile_plan_against_scene(&plan, &mut scratch)?;
                let label = a.get("label").and_then(|x| x.as_str()).unwrap_or("Voice").to_string();
                let count = plan.steps.len();
                doc.begin_undo_group(label);
                doc.plan_build = Some(PlanBuildState { plan, resolver: StepResolver::new(), created: Vec::new() });
                out.insert("steps".into(), json!(count));
                Ok(out)
            }
            "planStep" => {
                let index = a.get("index").and_then(|x| x.as_i64()).unwrap_or(-1) as usize;
                let mut pb = doc.plan_build.take().ok_or_else(|| BridgeError::new("engine.noBuild"))?;
                if index >= pb.plan.steps.len() {
                    doc.plan_build = Some(pb);
                    return Err(BridgeError::with_args("engine.badJSON", vec!["index".into()]));
                }
                let step = pb.plan.steps[index].clone();
                let mut next = doc.scene.clone();
                let made = match pb.resolver.run_step(&step, index, &mut next) {
                    Ok(m) => m,
                    Err(e) => {
                        doc.plan_build = Some(pb);
                        return Err(e);
                    }
                };
                doc.record_scene_change("Voice".into(), next, &made, None);
                pb.created.extend(made.clone());
                doc.plan_build = Some(pb);
                out.insert("created".into(), json!(made));
                Ok(out)
            }
            "planFinish" => {
                let made = if let Some(pb) = doc.plan_build.take() {
                    doc.end_undo_group();
                    pb.created
                } else {
                    Vec::new()
                };
                out.insert("created".into(), json!(made));
                Ok(out)
            }
            _ => Err(BridgeError::with_args("engine.unknownCall", vec![name.to_string()])),
        }
    })();
    match res {
        Ok(m) => ok_envelope(m),
        Err(e) => e.to_envelope(),
    }
}

#[allow(dead_code)]
fn _keep_prim_import() {
    let _ = (make_primitive_mesh, Mat4::IDENTITY, HashSet::<usize>::new());
}
