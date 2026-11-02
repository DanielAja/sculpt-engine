//! Decoding: serde types in Swift's Codable shapes, then conversion with Swift's repair rules.

use super::*;
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::Deserialize;

/// A Float as Swift decodes it: a number, or "nan" / "inf" / "-inf".
#[derive(Clone, Copy)]
struct Fl(f32);

struct FlVisitor;
impl<'de> Visitor<'de> for FlVisitor {
    type Value = Fl;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a number")
    }
    fn visit_f64<E>(self, v: f64) -> Result<Fl, E> {
        Ok(Fl(v as f32))
    }
    fn visit_i64<E>(self, v: i64) -> Result<Fl, E> {
        Ok(Fl(v as f32))
    }
    fn visit_u64<E>(self, v: u64) -> Result<Fl, E> {
        Ok(Fl(v as f32))
    }
    fn visit_str<E: de::Error>(self, v: &str) -> Result<Fl, E> {
        match v {
            "nan" => Ok(Fl(f32::NAN)),
            "inf" => Ok(Fl(f32::INFINITY)),
            "-inf" => Ok(Fl(f32::NEG_INFINITY)),
            _ => Err(E::custom("not a float")),
        }
    }
}
impl<'de> Deserialize<'de> for Fl {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Fl, D::Error> {
        d.deserialize_any(FlVisitor)
    }
}

/// One element of `positions`: a scalar (flat form) or an `[x, y, z]` triple (older nested form).
enum PosElem {
    Scalar(f32),
    Triple([f32; 3]),
}
impl<'de> Deserialize<'de> for PosElem {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<PosElem, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = PosElem;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a number or [x, y, z]")
            }
            fn visit_f64<E>(self, v: f64) -> Result<PosElem, E> {
                Ok(PosElem::Scalar(v as f32))
            }
            fn visit_i64<E>(self, v: i64) -> Result<PosElem, E> {
                Ok(PosElem::Scalar(v as f32))
            }
            fn visit_u64<E>(self, v: u64) -> Result<PosElem, E> {
                Ok(PosElem::Scalar(v as f32))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<PosElem, E> {
                FlVisitor.visit_str(v).map(|f| PosElem::Scalar(f.0))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut s: A) -> Result<PosElem, A::Error> {
                let mut t = [0f32; 3];
                for x in &mut t {
                    *x = s.next_element::<Fl>()?.ok_or_else(|| de::Error::custom("short vector"))?.0;
                }
                if s.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom("long vector"));
                }
                Ok(PosElem::Triple(t))
            }
        }
        d.deserialize_any(V)
    }
}

/// A face: a bare index array, or the keyed `{"indices": [...]}` form.
struct FaceDe(Vec<i64>);
impl<'de> Deserialize<'de> for FaceDe {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<FaceDe, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = FaceDe;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a face")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut s: A) -> Result<FaceDe, A::Error> {
                let mut v = Vec::with_capacity(s.size_hint().unwrap_or(4));
                while let Some(i) = s.next_element::<i64>()? {
                    v.push(i);
                }
                Ok(FaceDe(v))
            }
            fn visit_map<A: de::MapAccess<'de>>(self, mut m: A) -> Result<FaceDe, A::Error> {
                let mut out = None;
                while let Some(k) = m.next_key::<String>()? {
                    if k == "indices" {
                        out = Some(m.next_value::<Vec<i64>>()?);
                    } else {
                        m.next_value::<de::IgnoredAny>()?;
                    }
                }
                out.map(FaceDe).ok_or_else(|| de::Error::missing_field("indices"))
            }
        }
        d.deserialize_any(V)
    }
}

/// A date: ISO 8601 text, or seconds since 2001-01-01 (Swift's reference date).
struct DateDe(String);
impl<'de> Deserialize<'de> for DateDe {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<DateDe, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = DateDe;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a date")
            }
            fn visit_f64<E>(self, v: f64) -> Result<DateDe, E> {
                Ok(DateDe(iso_from_unix((v + 978_307_200.0).floor() as i64)))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<DateDe, E> {
                self.visit_f64(v as f64)
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<DateDe, E> {
                self.visit_f64(v as f64)
            }
            fn visit_str<E: de::Error>(self, s: &str) -> Result<DateDe, E> {
                normalize_iso(s).map(DateDe).ok_or_else(|| E::custom("bad date"))
            }
        }
        d.deserialize_any(V)
    }
}

/// `YYYY-MM-DDTHH:MM:SS[.fff]Z` -> whole seconds, as Swift re-encodes it.
// shortcut: only UTC "Z" dates are accepted (all Sculptor writers emit UTC); add offset parsing if a file with "+hh:mm" shows up.
fn normalize_iso(s: &str) -> Option<String> {
    let b = s.as_bytes();
    if b.len() < 20 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' || !s.ends_with('Z') {
        return None;
    }
    let digits = |r: std::ops::Range<usize>| b[r].iter().all(u8::is_ascii_digit);
    if !(digits(0..4) && digits(5..7) && digits(8..10) && digits(11..13) && digits(14..16) && digits(17..19)) {
        return None;
    }
    let rest = &s[19..s.len() - 1];
    if !(rest.is_empty() || (rest.starts_with('.') && rest.len() > 1 && rest[1..].bytes().all(|c| c.is_ascii_digit()))) {
        return None;
    }
    Some(format!("{}Z", &s[..19]))
}

pub(crate) fn iso_from_unix(t: i64) -> String {
    let (days, secs) = (t.div_euclid(86_400), t.rem_euclid(86_400));
    // Civil from days (Howard Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", secs / 3600, secs / 60 % 60, secs % 60)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Header {
    format_version: i64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProjectDe {
    scene: SceneDe,
    selection: SelectionDe,
    camera_state: Option<String>,
    created_at: DateDe,
    modified_at: DateDe,
    app_version: String,
}

#[derive(Deserialize)]
struct SceneDe {
    objects: Vec<ObjectDe>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct IdDe {
    raw_value: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ObjectDe {
    id: IdDe,
    name: String,
    mesh: MeshDe,
    transform: TransformDe,
    material: MaterialDe,
    modifiers: Vec<EntryDe>,
    parent: Option<IdDe>,
    is_visible: bool,
    is_locked: bool,
    primitive_kind: Option<PrimitiveKind>,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
enum PrimitiveKind {
    Cube,
    Sphere,
    Cylinder,
    Cone,
    Torus,
    Plane,
    Blob,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MeshDe {
    positions: Vec<PosElem>,
    faces: Vec<FaceDe>,
    smooth_shading: Option<bool>,
}

#[derive(Deserialize)]
struct TransformDe {
    position: [Fl; 3],
    rotation: [Fl; 4],
    scale: [Fl; 3],
}

#[derive(Deserialize)]
struct ColorDe {
    r: Fl,
    g: Fl,
    b: Fl,
    a: Fl,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MaterialDe {
    name: String,
    base_color: ColorDe,
    metallic: Fl,
    roughness: Fl,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EntryDe {
    id: String,
    modifier: ModifierDe,
    is_enabled: bool,
}

#[derive(Deserialize)]
struct Wrap<T> {
    #[serde(rename = "_0")]
    v: T,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
enum ModifierDe {
    Mirror(Wrap<MirrorDe>),
    Subdivision(Wrap<SubdivisionDe>),
    Array(Wrap<ArrayDe>),
    Solidify(Wrap<SolidifyDe>),
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MirrorDe {
    axis: u8,
    merge: bool,
    merge_distance: Fl,
}
#[derive(Deserialize)]
struct SubdivisionDe {
    levels: i64,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ArrayDe {
    count: i64,
    relative_offset: [Fl; 3],
    constant_offset: [Fl; 3],
    use_relative: bool,
}
#[derive(Deserialize)]
struct SolidifyDe {
    thickness: Fl,
    offset: Fl,
}

#[derive(Deserialize)]
struct SelectionDe {
    objects: Vec<IdDe>,
    active: Option<IdDe>,
    elements: ElementsDe,
}
#[derive(Deserialize)]
struct ElementsDe {
    vertices: Vec<i64>,
    edges: Vec<EdgeDe>,
    faces: Vec<i64>,
}
#[derive(Deserialize)]
struct EdgeDe {
    a: i64,
    b: i64,
}

/// Swift's UUID decoder: 8-4-4-4-12 hex, written back uppercase.
fn uuid(id: IdDe) -> Result<String, FormatError> {
    uuid_str(&id.raw_value)
}
fn uuid_str(s: &str) -> Result<String, FormatError> {
    let ok = s.len() == 36
        && s.char_indices().all(|(i, c)| if [8, 13, 18, 23].contains(&i) { c == '-' } else { c.is_ascii_hexdigit() });
    if ok { Ok(s.to_ascii_uppercase()) } else { Err(FormatError::NotAProject) }
}

fn v3(a: [Fl; 3]) -> Vec3 {
    Vec3::new(a[0].0, a[1].0, a[2].0)
}

fn sorted_unique<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v.dedup();
    v
}

pub(super) fn decode(bytes: &[u8]) -> Result<ProjectFile, FormatError> {
    let header: Header = serde_json::from_slice(bytes).map_err(|_| FormatError::NotAProject)?;
    if header.format_version > CURRENT_FORMAT_VERSION {
        return Err(FormatError::NewerVersion(header.format_version));
    }
    if header.format_version != CURRENT_FORMAT_VERSION {
        return Err(FormatError::UnsupportedVersion(header.format_version));
    }
    let p: ProjectDe = serde_json::from_slice(bytes).map_err(|_| FormatError::NotAProject)?;
    let mut objects = Vec::with_capacity(p.scene.objects.len());
    for o in p.scene.objects {
        objects.push(SceneObject {
            id: uuid(o.id)?,
            name: o.name,
            mesh: mesh(o.mesh)?,
            transform: Transform {
                position: v3(o.transform.position),
                rotation: glam::Quat::from_xyzw(o.transform.rotation[0].0, o.transform.rotation[1].0, o.transform.rotation[2].0, o.transform.rotation[3].0),
                scale: v3(o.transform.scale),
            },
            material: Material {
                name: o.material.name,
                base_color: Color { r: o.material.base_color.r.0, g: o.material.base_color.g.0, b: o.material.base_color.b.0, a: o.material.base_color.a.0 },
                metallic: o.material.metallic.0,
                roughness: o.material.roughness.0,
            },
            modifiers: o.modifiers.into_iter().map(entry).collect::<Result<_, _>>()?,
            parent: o.parent.map(uuid).transpose()?,
            is_visible: o.is_visible,
            is_locked: o.is_locked,
            primitive_kind: o.primitive_kind.map(|k| {
                match k {
                    PrimitiveKind::Cube => "cube",
                    PrimitiveKind::Sphere => "sphere",
                    PrimitiveKind::Cylinder => "cylinder",
                    PrimitiveKind::Cone => "cone",
                    PrimitiveKind::Torus => "torus",
                    PrimitiveKind::Plane => "plane",
                    PrimitiveKind::Blob => "blob",
                }
                .to_string()
            }),
        });
    }
    let mut scene = Scene { objects };
    repair_scene(&mut scene);
    let mut sel_objects = p.selection.objects.into_iter().map(uuid).collect::<Result<Vec<_>, _>>()?;
    sel_objects = sorted_unique(sel_objects);
    let e = p.selection.elements;
    Ok(ProjectFile {
        format_version: CURRENT_FORMAT_VERSION,
        scene,
        selection: Selection {
            objects: sel_objects,
            active: p.selection.active.map(uuid).transpose()?,
            elements: MeshSelection {
                vertices: sorted_unique(e.vertices),
                edges: sorted_unique(e.edges.into_iter().map(|e| (e.a, e.b)).collect()),
                faces: sorted_unique(e.faces),
            },
        },
        camera_state: p.camera_state,
        created_at: p.created_at.0,
        modified_at: p.modified_at.0,
        app_version: p.app_version,
    })
}

fn entry(e: EntryDe) -> Result<ModifierEntry, FormatError> {
    let modifier = match e.modifier {
        ModifierDe::Mirror(Wrap { v }) => {
            if v.axis > 2 {
                return Err(FormatError::NotAProject);
            }
            Modifier::Mirror { axis: v.axis as i64, merge: v.merge, merge_distance: v.merge_distance.0 }
        }
        ModifierDe::Subdivision(Wrap { v }) => Modifier::Subdivision { levels: v.levels },
        ModifierDe::Array(Wrap { v }) => Modifier::Array {
            count: v.count,
            relative_offset: v3(v.relative_offset),
            constant_offset: v3(v.constant_offset),
            use_relative: v.use_relative,
        },
        ModifierDe::Solidify(Wrap { v }) => Modifier::Solidify { thickness: v.thickness.0, offset: v.offset.0 },
    };
    Ok(ModifierEntry { id: uuid_str(&e.id)?, modifier, is_enabled: e.is_enabled })
}

/// Swift's `Mesh(from:)` plus `ProjectFile.repaired(_ mesh:)`.
fn mesh(m: MeshDe) -> Result<Mesh, FormatError> {
    let mut positions = Vec::new();
    match m.positions.first() {
        None => {}
        Some(PosElem::Scalar(_)) => {
            if m.positions.len() % 3 != 0 {
                return Err(FormatError::NotAProject);
            }
            let mut flat = Vec::with_capacity(m.positions.len());
            for e in &m.positions {
                match e {
                    PosElem::Scalar(f) => flat.push(*f),
                    PosElem::Triple(_) => return Err(FormatError::NotAProject),
                }
            }
            positions = flat.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect();
        }
        Some(PosElem::Triple(_)) => {
            for e in &m.positions {
                match e {
                    PosElem::Triple(t) => positions.push(Vec3::from_array(*t)),
                    PosElem::Scalar(_) => return Err(FormatError::NotAProject),
                }
            }
        }
    }
    for p in &mut positions {
        if !p.is_finite() {
            *p = Vec3::ZERO;
        }
    }
    let n = positions.len() as i64;
    let faces = m
        .faces
        .into_iter()
        .filter(|f| {
            let f = &f.0;
            f.len() >= 3 && f.iter().all(|&i| i >= 0 && i < n) && {
                let mut s = f.clone();
                s.sort_unstable();
                s.windows(2).all(|w| w[0] != w[1])
            }
        })
        .map(|f| f.0.into_iter().map(|i| i as u32).collect())
        .collect();
    Ok(Mesh { positions, faces, smooth_shading: m.smooth_shading.unwrap_or(false) })
}
