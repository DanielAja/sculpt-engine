//! Parallel reader for the two big mesh arrays of a project (`positions`, `faces`).
//!
//! The document is scanned once for those arrays (strings are skipped, so a name can never be mistaken for a
//! key). Every big one is cut out and replaced by an index marker (`"@p": k` / `"@f": k`); the small stub that
//! remains goes through the normal serde path, and the arrays are then parsed from their byte ranges by all
//! threads with a strict JSON grammar and the same repairs as `mesh`.
//!
//! This path only ever *accepts*: whatever it does not fully understand (a string number like `"nan"`,
//! negative or fractional indices, odd nesting, a duplicate marker, any serde error) makes `decode` return
//! `None`, and the caller re-reads the untouched document with the serde path, which owns all error reporting.

use super::*;
use std::ops::Range;

/// Arrays shorter than this are left in the stub: serde reads them faster than the cut-out costs.
const MIN_ARRAY: usize = 16 * 1024;
/// Bytes per parallel work item (tiny in tests so that small documents still split into many pieces).
const CHUNK: usize = if cfg!(test) { 4096 } else { 1 << 20 };

/// Where `mesh` gets the arrays that the stub replaced by markers.
pub(super) trait Ext {
    fn positions(&mut self, k: u32) -> Option<Vec<Vec3>>;
    fn faces(&mut self, k: u32, n: usize) -> Option<Vec<Vec<u32>>>;
}

/// The serde path: documents never carry markers.
pub(super) struct NoExt;
impl Ext for NoExt {
    fn positions(&mut self, _: u32) -> Option<Vec<Vec3>> {
        None
    }
    fn faces(&mut self, _: u32, _: usize) -> Option<Vec<Vec<u32>>> {
        None
    }
}

struct Arrays<'a> {
    bytes: &'a [u8],
    pos: Vec<Range<usize>>,
    faces: Vec<Range<usize>>,
    used_p: Vec<bool>,
    used_f: Vec<bool>,
    failed: bool,
}

impl Ext for Arrays<'_> {
    fn positions(&mut self, k: u32) -> Option<Vec<Vec3>> {
        let (r, used) = (self.pos.get(k as usize)?.clone(), &mut self.used_p[k as usize]);
        if std::mem::replace(used, true) {
            self.failed = true; // marker referenced twice
            return None;
        }
        let out = parse_positions(&self.bytes[r]);
        self.failed |= out.is_none();
        out
    }
    fn faces(&mut self, k: u32, n: usize) -> Option<Vec<Vec<u32>>> {
        let (r, used) = (self.faces.get(k as usize)?.clone(), &mut self.used_f[k as usize]);
        if std::mem::replace(used, true) {
            self.failed = true;
            return None;
        }
        let out = parse_faces(&self.bytes[r], n);
        self.failed |= out.is_none();
        out
    }
}

pub(super) fn decode(bytes: &[u8]) -> Option<ProjectFile> {
    let (stub, pos, faces) = extract(bytes)?;
    if pos.is_empty() && faces.is_empty() {
        return None;
    }
    let p: ProjectDe = serde_json::from_slice(&stub).ok()?;
    if p.format_version != CURRENT_FORMAT_VERSION {
        return None;
    }
    let mut ext = Arrays { bytes, used_p: vec![false; pos.len()], used_f: vec![false; faces.len()], pos, faces, failed: false };
    let out = convert(p, &mut ext).ok()?;
    // Every marker must have been claimed by exactly one mesh.
    (!ext.failed && ext.used_p.iter().all(|&u| u) && ext.used_f.iter().all(|&u| u)).then_some(out)
}

#[inline]
fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\r' | b'\t')
}

#[inline]
fn skip_ws(b: &[u8], p: &mut usize) {
    while *p < b.len() && is_ws(b[*p]) {
        *p += 1;
    }
}

/// First `needle` in `h` (eight bytes per step).
fn find_byte(h: &[u8], needle: u8) -> Option<usize> {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    let rep = LO.wrapping_mul(needle as u64);
    let mut chunks = h.chunks_exact(8);
    let mut at = 0;
    for c in &mut chunks {
        let w = u64::from_le_bytes(c.try_into().unwrap()) ^ rep;
        let z = w.wrapping_sub(LO) & !w & HI;
        if z != 0 {
            return Some(at + (z.trailing_zeros() / 8) as usize);
        }
        at += 8;
    }
    chunks.remainder().iter().position(|&b| b == needle).map(|i| at + i)
}

/// Runs `f(start..end)` over `from..len` in `CHUNK` pieces and returns the first `Some` in document order.
fn find_first(len: usize, from: usize, f: impl Fn(Range<usize>) -> Option<usize> + Sync + Send) -> Option<usize> {
    let range = |c: usize| from + c * CHUNK..(from + (c + 1) * CHUNK).min(len);
    let nc = (len - from).div_ceil(CHUNK);
    #[cfg(feature = "threads")]
    if nc > 1 {
        use rayon::prelude::*;
        return (0..nc).into_par_iter().map(|c| range(c)).find_map_first(|r| f(r));
    }
    (0..nc).find_map(|c| f(range(c)))
}

/// Cuts the big arrays out of `bytes`: the stub document, plus the byte range inside the `[ ]` of each.
fn extract(bytes: &[u8]) -> Option<(Vec<u8>, Vec<Range<usize>>, Vec<Range<usize>>)> {
    let n = bytes.len();
    let mut stub = Vec::new();
    let (mut pos, mut faces) = (Vec::new(), Vec::new());
    let mut copied = 0usize;
    let mut i = 0usize;
    while i < n {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        let key_start = i;
        let mut j = i + 1;
        loop {
            match *bytes.get(j)? {
                b'\\' => j += 2,
                b'"' => break,
                _ => j += 1,
            }
        }
        let key = &bytes[i + 1..j];
        i = j + 1;
        let is_pos = key == b"positions";
        if !is_pos && key != b"faces" {
            continue;
        }
        let mut k = i;
        skip_ws(bytes, &mut k);
        if bytes.get(k) != Some(&b':') {
            continue;
        }
        k += 1;
        skip_ws(bytes, &mut k);
        if bytes.get(k) != Some(&b'[') {
            continue;
        }
        let body = k + 1;
        let mut f = body;
        skip_ws(bytes, &mut f);
        let first = *bytes.get(f)?;
        // Flat numbers for `positions`, nested arrays for `faces` (the selection's `faces` is flat integers).
        if !(if is_pos { first == b'-' || first.is_ascii_digit() } else { first == b'[' }) {
            continue;
        }
        let end = if is_pos {
            find_first(n, body, |r| find_byte(&bytes[r.clone()], b']').map(|x| r.start + x))?
        } else {
            // The outer array closes at the first `]` whose next non-blank byte is another `]`.
            find_first(n, body, |r| {
                let mut at = r.start;
                while let Some(x) = find_byte(&bytes[at..r.end], b']') {
                    let mut q = at + x + 1;
                    skip_ws(bytes, &mut q);
                    if bytes.get(q) == Some(&b']') {
                        return Some(q);
                    }
                    at += x + 1;
                }
                None
            })?
        };
        if end - body < MIN_ARRAY {
            continue;
        }
        stub.extend_from_slice(&bytes[copied..key_start]);
        let (list, tag) = if is_pos { (&mut pos, "p") } else { (&mut faces, "f") };
        stub.extend_from_slice(format!("\"@{tag}\":{}", list.len()).as_bytes());
        list.push(body..end);
        copied = end + 1;
        i = end + 1;
    }
    stub.extend_from_slice(&bytes[copied..]);
    Some((stub, pos, faces))
}

/// Splits `body` into pieces that each start where `start_of(next byte)` says a record starts: the cut is the
/// first such byte at or after every multiple of `CHUNK`. Returns the piece start offsets (first is 0).
fn split_points(body: &[u8], mut cut_after: impl FnMut(&[u8]) -> Option<usize>) -> Vec<usize> {
    let mut starts = vec![0usize];
    let mut from = CHUNK;
    while from < body.len() {
        match cut_after(&body[from..]) {
            Some(off) if from + off < body.len() => {
                let at = from + off;
                if at > *starts.last().unwrap() {
                    starts.push(at);
                }
                from = at + CHUNK;
            }
            _ => break,
        }
    }
    starts
}

/// Runs `f` on every piece (in parallel) and gives the results back in order; `None` if any piece failed.
fn map_pieces<T: Send>(body: &[u8], starts: &[usize], f: impl Fn(&[u8], bool) -> Option<T> + Sync + Send) -> Option<Vec<T>> {
    let piece = |c: usize| {
        let end = starts.get(c + 1).copied().unwrap_or(body.len());
        f(&body[starts[c]..end], c + 1 == starts.len())
    };
    #[cfg(feature = "threads")]
    if starts.len() > 1 {
        use rayon::prelude::*;
        return (0..starts.len()).into_par_iter().map(piece).collect();
    }
    (0..starts.len()).map(piece).collect()
}

const POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// One strict JSON number at `*p`, as the f64 serde would hand to `visit_f64` / widen from an integer.
#[inline]
fn parse_number(b: &[u8], p: &mut usize) -> Option<f64> {
    let start = *p;
    let mut i = start;
    let neg = b.get(i) == Some(&b'-');
    i += neg as usize;
    let (mut mant, mut digits, mut e10) = (0u64, 0u32, 0i32);
    let mut slow = false;
    match *b.get(i)? {
        b'0' => {
            i += 1;
            if b.get(i).is_some_and(u8::is_ascii_digit) {
                return None; // no leading zeros in JSON
            }
        }
        b'1'..=b'9' => {
            while let Some(&d) = b.get(i) {
                if !d.is_ascii_digit() {
                    break;
                }
                if digits < 18 {
                    mant = mant * 10 + (d - b'0') as u64;
                    digits += 1;
                } else {
                    slow = true;
                }
                i += 1;
            }
        }
        _ => return None,
    }
    if b.get(i) == Some(&b'.') {
        i += 1;
        let fs = i;
        while let Some(&d) = b.get(i) {
            if !d.is_ascii_digit() {
                break;
            }
            if digits < 18 {
                mant = mant * 10 + (d - b'0') as u64;
                digits += (mant != 0) as u32;
                e10 -= 1;
            } else {
                slow = true;
            }
            i += 1;
        }
        if i == fs {
            return None;
        }
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        i += 1;
        let eneg = match b.get(i) {
            Some(b'-') => {
                i += 1;
                true
            }
            Some(b'+') => {
                i += 1;
                false
            }
            _ => false,
        };
        let es = i;
        let mut ev = 0i32;
        while let Some(&d) = b.get(i) {
            if !d.is_ascii_digit() {
                break;
            }
            ev = (ev * 10 + (d - b'0') as i32).min(100_000);
            i += 1;
        }
        if i == es {
            return None;
        }
        e10 += if eneg { -ev } else { ev };
    }
    *p = i;
    let fast = !slow && mant <= (1 << 53) && (-22..=22).contains(&e10);
    let v = if fast {
        // Both factors are exact doubles, so the one multiplication / division rounds correctly.
        let m = mant as f64;
        let a = if e10 < 0 { m / POW10[(-e10) as usize] } else { m * POW10[e10 as usize] };
        if neg { -a } else { a } // `-0` stays negative zero
    } else {
        std::str::from_utf8(&b[start..i]).ok()?.parse::<f64>().ok()?
    };
    v.is_finite().then_some(v)
}

/// `a, b, c, ...` (flat numbers) into floats; `last` pieces must not end in a comma, others must.
fn parse_floats(b: &[u8], last: bool) -> Option<Vec<f32>> {
    let mut out = Vec::with_capacity(b.len() / 8 + 1);
    let mut p = 0usize;
    skip_ws(b, &mut p);
    if p == b.len() {
        return (!last).then_some(out);
    }
    loop {
        out.push(parse_number(b, &mut p)? as f32);
        skip_ws(b, &mut p);
        if p == b.len() {
            return last.then_some(out);
        }
        if b[p] != b',' {
            return None;
        }
        p += 1;
        skip_ws(b, &mut p);
        if p == b.len() {
            return (!last).then_some(out);
        }
    }
}

/// The flat `[x, y, z, x, y, z, ...]` body into vertices (same repair for non-finite as `mesh`).
fn parse_positions(body: &[u8]) -> Option<Vec<Vec3>> {
    let starts = split_points(body, |rest| find_byte(rest, b',').map(|c| c + 1));
    let parts = map_pieces(body, &starts, parse_floats)?;
    let total: usize = parts.iter().map(Vec::len).sum();
    if total == 0 || total % 3 != 0 {
        return None;
    }
    let mut out: Vec<Vec3> = Vec::with_capacity(total / 3);
    let mut offsets = Vec::with_capacity(parts.len());
    let mut acc = 0usize;
    for p in &parts {
        offsets.push(acc);
        acc += p.len();
    }
    let dst = SendPtr(out.as_mut_ptr() as *mut f32);
    let copy = |(part, &off): (&Vec<f32>, &usize)| {
        // SAFETY: `Vec3` is three packed f32; parts write disjoint ranges of the `total` floats reserved above.
        unsafe { std::ptr::copy_nonoverlapping(part.as_ptr(), dst.get().add(off), part.len()) };
    };
    #[cfg(feature = "threads")]
    {
        use rayon::prelude::*;
        parts.par_iter().zip(offsets.par_iter()).for_each(copy);
    }
    #[cfg(not(feature = "threads"))]
    parts.iter().zip(offsets.iter()).for_each(copy);
    // SAFETY: all `total / 3` vertices were initialised by the copies.
    unsafe { out.set_len(total / 3) };
    Some(out)
}

struct SendPtr<T>(*mut T);
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

#[inline]
fn parse_uint(b: &[u8], p: &mut usize) -> Option<u32> {
    let start = *p;
    let mut v = 0u64;
    while let Some(&d) = b.get(*p) {
        if !d.is_ascii_digit() {
            break;
        }
        v = v * 10 + (d - b'0') as u64;
        *p += 1;
        if *p - start > 10 {
            return None;
        }
    }
    let len = *p - start;
    if len == 0 || (len > 1 && b[start] == b'0') || v > u32::MAX as u64 {
        return None;
    }
    Some(v as u32)
}

#[inline]
fn face_unique(f: &[u32]) -> bool {
    match f.len() {
        3 => f[0] != f[1] && f[1] != f[2] && f[0] != f[2],
        4 => f[0] != f[1] && f[0] != f[2] && f[0] != f[3] && f[1] != f[2] && f[1] != f[3] && f[2] != f[3],
        _ => {
            let mut s = f.to_vec();
            s.sort_unstable();
            s.windows(2).all(|w| w[0] != w[1])
        }
    }
}

/// `[a, b, c], [d, e, f], ...`: keeps the faces `mesh` keeps (3+ distinct indices below `n`).
fn parse_face_piece(b: &[u8], last: bool, n: usize) -> Option<Vec<Vec<u32>>> {
    let mut out = Vec::with_capacity(b.len() / 40 + 1);
    let mut tmp: Vec<u32> = Vec::with_capacity(8);
    let mut p = 0usize;
    skip_ws(b, &mut p);
    if p == b.len() {
        return (!last).then_some(out);
    }
    loop {
        if b.get(p) != Some(&b'[') {
            return None;
        }
        p += 1;
        skip_ws(b, &mut p);
        tmp.clear();
        if b.get(p) == Some(&b']') {
            p += 1;
        } else {
            loop {
                tmp.push(parse_uint(b, &mut p)?);
                skip_ws(b, &mut p);
                match *b.get(p)? {
                    b',' => {
                        p += 1;
                        skip_ws(b, &mut p);
                    }
                    b']' => {
                        p += 1;
                        break;
                    }
                    _ => return None,
                }
            }
        }
        if tmp.len() >= 3 && tmp.iter().all(|&i| (i as usize) < n) && face_unique(&tmp) {
            out.push(tmp.clone());
        }
        skip_ws(b, &mut p);
        if p == b.len() {
            return last.then_some(out);
        }
        if b[p] != b',' {
            return None;
        }
        p += 1;
        skip_ws(b, &mut p);
        if p == b.len() {
            return (!last).then_some(out);
        }
    }
}

fn parse_faces(body: &[u8], n: usize) -> Option<Vec<Vec<u32>>> {
    // Faces are flat (`[` only ever opens a face), so any `[` is a safe place to cut.
    let starts = split_points(body, |rest| find_byte(rest, b'['));
    let mut parts = map_pieces(body, &starts, |b, last| parse_face_piece(b, last, n))?;
    let total: usize = parts.iter().map(Vec::len).sum();
    let mut out: Vec<Vec<u32>> = Vec::with_capacity(total);
    let mut offsets = Vec::with_capacity(parts.len());
    let mut acc = 0usize;
    for p in &parts {
        offsets.push(acc);
        acc += p.len();
    }
    let dst = SendPtr(out.as_mut_ptr());
    let mv = |(part, &off): (&mut Vec<Vec<u32>>, &usize)| {
        // SAFETY: parts move their `Vec` headers into disjoint slots of the reserved capacity, then forget them.
        unsafe {
            std::ptr::copy_nonoverlapping(part.as_ptr(), dst.get().add(off), part.len());
            part.set_len(0);
        }
    };
    #[cfg(feature = "threads")]
    {
        use rayon::prelude::*;
        parts.par_iter_mut().zip(offsets.par_iter()).for_each(mv);
    }
    #[cfg(not(feature = "threads"))]
    parts.iter_mut().zip(offsets.iter()).for_each(mv);
    // SAFETY: all `total` slots were initialised by the moves.
    unsafe { out.set_len(total) };
    Some(out)
}
