//! Using any LLM as a planner: the prompt that makes it write a shape program, the JSON Schema for
//! constrained decoding, and a defensive parser for its reply.
//!
//! The pattern (an LLM writes a small CAD or shape program instead of a mesh) follows CADmium, CAD-Recode,
//! BlenderLLM, LLaMA-Mesh's "3D as text" and Holodeck/LayoutGPT for layouts; we only borrow the idea, no code
//! or weights. Recommended free, permissive local models: Qwen3 0.6B to 8B and Qwen2.5 1.5B or 7B Coder
//! (Apache-2.0), Phi-4-mini (MIT), via WebLLM in the browser, or Apple's on-device Foundation Models.

use super::program::{Program, ProgramError};
use serde_json::{json, Value};

pub const SYSTEM_PROMPT: &str = r##"You design 3D models for Sculptor. Reply with ONE JSON object and nothing else: a shape program.

Units are millimetres. Z is up and the ground is z = 0, so models stand on it. X is width, Y is depth.
Program: {"version":1,"name":"<what it is>","parts":[Part,...]}   (1 to 12 parts)
Part: {"name":"Body","color":"#rrggbb","metallic":0..1,"roughness":0..1,"shape":Node}
  Use one part per colour or material (a mug body and its handle can share a part).
Node: a shape plus optional placement fields:
  "at":[x,y,z] (centre position), "rot":[xDeg,yDeg,zDeg], "scale":s, "stretch":[sx,sy,sz],
  "mirror":"x"|"y"|"z" (also adds a copy reflected across that plane; use for pairs of legs, eyes, wings),
  "hollow":wallThickness
Shapes ("type"), all centred on their own origin, round ones with their axis along Z:
  {"type":"sphere","r":R}
  {"type":"box","size":[x,y,z],"round":edgeRadius}
  {"type":"cylinder","r":R,"h":H,"round":edgeRadius}
  {"type":"cone","r1":bottomRadius,"r2":topRadius,"h":H}
  {"type":"torus","major":ringRadius,"minor":tubeRadius}        (ring lies flat in XY)
  {"type":"capsule","from":[x,y,z],"to":[x,y,z],"r":R}          (limbs, handles, stems)
  {"type":"ellipsoid","size":[x,y,z]}
  {"type":"pyramid","base":B,"h":H}
  {"type":"lathe","profile":[[radius,z],...]}                     (bottom to top, revolved around Z: vases, bottles)
  {"type":"extrude","outline":[[x,y],...],"h":H}                  (flat outline pushed up Z: stars, letters, signs)
  {"type":"union","children":[Node,...],"smooth":blendRadius}     (smooth > 0 melts shapes together like clay)
  {"type":"difference","children":[base,cut,...],"smooth":r}      (first child minus the rest: holes, hollows)
  {"type":"intersection","children":[Node,...]}
  {"type":"array","child":Node,"count":N,"offset":[dx,dy,dz]}     (N copies, each moved by offset)
Rules: realistic sizes (a mug is about 95 mm tall), parts touch or overlap so the model holds together,
at most 60 shapes in total, simple recognisable forms over detail, bright friendly colours.

Example, "a red mug":
{"version":1,"name":"mug","parts":[{"name":"Mug","color":"#d63a2f","roughness":0.35,"shape":{"type":"union","children":[
 {"type":"difference","children":[{"type":"cylinder","r":40,"h":95,"round":3,"at":[0,0,47.5]},{"type":"cylinder","r":35,"h":95,"at":[0,0,54]}]},
 {"type":"torus","major":22,"minor":5,"rot":[90,0,0],"at":[42,0,50]}]}}]}

Example, "a snowman":
{"version":1,"name":"snowman","parts":[
 {"name":"Snow","color":"#f4f7fb","roughness":0.8,"shape":{"type":"union","smooth":6,"children":[
  {"type":"sphere","r":40,"at":[0,0,40]},{"type":"sphere","r":29,"at":[0,0,100]},{"type":"sphere","r":20,"at":[0,0,142]}]}},
 {"name":"Nose","color":"#f28c28","shape":{"type":"cone","r1":4,"r2":0,"h":18,"rot":[-90,0,0],"at":[0,-26,142]}},
 {"name":"Eyes","color":"#1e1e20","shape":{"type":"sphere","r":3,"at":[7,-17,150],"mirror":"x"}}]}
"##;

/// JSON Schema (draft 2020-12 subset that WebLLM, Chrome's Prompt API and most JSON-mode servers accept).
/// Recursion is expressed with `$defs`; hosts that can't handle `$ref` can pass `schema_flat()` instead.
pub fn schema() -> Value {
    let num = json!({ "type": "number" });
    let v3 = json!({ "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 });
    let v2s = json!({ "type": "array", "items": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 }, "minItems": 2, "maxItems": 64 });
    let node_ref = json!({ "$ref": "#/$defs/node" });
    let kids = json!({ "type": "array", "items": node_ref, "minItems": 1, "maxItems": 24 });
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["version", "name", "parts"],
        "properties": {
            "version": { "const": 1 },
            "name": { "type": "string" },
            "parts": { "type": "array", "minItems": 1, "maxItems": 12, "items": {
                "type": "object", "required": ["name", "color", "shape"],
                "properties": {
                    "name": { "type": "string" },
                    "color": { "type": "string", "pattern": "^#[0-9a-fA-F]{6}$" },
                    "metallic": num, "roughness": num, "shape": node_ref
                }
            }}
        },
        "$defs": { "node": {
            "type": "object", "required": ["type"],
            "properties": {
                "type": { "enum": ["sphere", "box", "cylinder", "cone", "torus", "capsule", "ellipsoid", "pyramid", "lathe", "extrude", "union", "difference", "intersection", "array"] },
                "r": num, "h": num, "r1": num, "r2": num, "major": num, "minor": num, "round": num, "base": num,
                "size": v3, "from": v3, "to": v3, "profile": v2s, "outline": v2s,
                "children": kids, "smooth": num, "child": node_ref, "count": { "type": "integer" }, "offset": v3,
                "at": v3, "rot": v3, "scale": num, "stretch": v3, "mirror": { "enum": ["x", "y", "z"] }, "hollow": num
            }
        }}
    })
}

/// Finds and validates the program in an LLM reply: tolerates code fences, prose before or after,
/// and a bare part list. Never trusts it beyond `Program::validate`.
pub fn extract_program(reply: &str) -> Result<Program, ProgramError> {
    let text = reply.trim();
    let start = text.find('{').ok_or_else(|| ProgramError("the reply has no JSON object".into()))?;
    let end = text.rfind('}').ok_or_else(|| ProgramError("the reply's JSON is cut off".into()))?;
    if end <= start {
        return Err(ProgramError("the reply's JSON is cut off".into()));
    }
    let body = &text[start..=end];
    let mut v: Value = serde_json::from_str(body).map_err(|e| ProgramError(format!("the reply isn't valid JSON: {e}")))?;
    // Small models sometimes answer with a single part or a bare node; wrap them.
    if v.get("parts").is_none() {
        if v.get("shape").is_some() {
            v = json!({ "version": 1, "name": v.get("name").cloned().unwrap_or(json!("model")), "parts": [v] });
        } else if v.get("type").is_some() {
            v = json!({ "version": 1, "name": "model", "parts": [{ "name": "Model", "shape": v }] });
        }
    }
    let p: Program = serde_json::from_value(v).map_err(|e| ProgramError(format!("not a shape program: {e}")))?;
    p.validate()?;
    Ok(p)
}
