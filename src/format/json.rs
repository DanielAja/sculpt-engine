//! A JSON writer that produces the same bytes as Swift's `JSONEncoder` with
//! `[.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]` and the `nan` / `inf` / `-inf` string strategy.

/// A JSON value tree. Object keys are sorted when written.
#[derive(Debug, Clone, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Int(i64),
    F32(f32),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(&'static str, J)>),
}

impl J {
    pub fn pretty(&self) -> String {
        let mut s = String::new();
        write(self, &mut s, Some(0));
        s
    }
    pub fn compact(&self) -> String {
        let mut s = String::new();
        write(self, &mut s, None);
        s
    }
}

use std::fmt::Write as _;

const SPACES: &str = "                                ";

pub(crate) fn indent(out: &mut String, n: usize) {
    let mut rem = n * 2;
    while rem > 0 {
        let chunk = rem.min(SPACES.len());
        out.push_str(&SPACES[..chunk]);
        rem -= chunk;
    }
}

#[inline]
pub(crate) fn write_u32(mut v: u32, out: &mut String) {
    if v == 0 {
        out.push('0');
        return;
    }
    let mut buf = [0u8; 10];
    let mut i = buf.len();
    while v > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    // SAFETY: digits are pure ASCII.
    out.push_str(unsafe { std::str::from_utf8_unchecked(&buf[i..]) });
}

#[inline]
pub(crate) fn write_i64(v: i64, out: &mut String) {
    if v == 0 {
        out.push('0');
        return;
    }
    let mut u = v.unsigned_abs();
    let mut buf = [0u8; 21];
    let mut i = buf.len();
    while u > 0 {
        i -= 1;
        buf[i] = b'0' + (u % 10) as u8;
        u /= 10;
    }
    if v < 0 {
        i -= 1;
        buf[i] = b'-';
    }
    // SAFETY: '-' and digits are pure ASCII.
    out.push_str(unsafe { std::str::from_utf8_unchecked(&buf[i..]) });
}

pub(crate) fn write_value(v: &J, out: &mut String, level: Option<usize>) {
    match v {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Int(i) => write_i64(*i, out),
        J::F32(f) => swift_float(*f, out),
        J::Str(s) => string(s, out),
        J::Arr(items) => container(out, level, '[', ']', items.len(), |out, i, l| write_value(&items[i], out, l)),
        J::Obj(fields) => {
            let mut sorted: Vec<&(&str, J)> = fields.iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(b.0));
            let sep = if level.is_some() { " : " } else { ":" };
            container(out, level, '{', '}', sorted.len(), |out, i, l| {
                string(sorted[i].0, out);
                out.push_str(sep);
                write_value(&sorted[i].1, out, l);
            })
        }
    }
}

fn write(v: &J, out: &mut String, level: Option<usize>) {
    write_value(v, out, level);
}

pub(crate) fn container(out: &mut String, level: Option<usize>, open: char, close: char, n: usize, mut item: impl FnMut(&mut String, usize, Option<usize>)) {
    out.push(open);
    match level {
        None => {
            for i in 0..n {
                if i > 0 {
                    out.push(',');
                }
                item(out, i, None);
            }
        }
        Some(l) => {
            // Swift writes empty containers as "[\n\n<indent>]".
            out.push('\n');
            if n == 0 {
                out.push('\n');
            }
            for i in 0..n {
                indent(out, l + 1);
                item(out, i, Some(l + 1));
                out.push_str(if i + 1 < n { ",\n" } else { "\n" });
            }
            indent(out, l);
        }
    }
    out.push(close);
}

pub(crate) fn string(s: &str, out: &mut String) {
    out.push('"');
    let bytes = s.as_bytes();
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        let esc = match b {
            b'"' => Some("\\\""),
            b'\\' => Some("\\\\"),
            b'\n' => Some("\\n"),
            b'\r' => Some("\\r"),
            b'\t' => Some("\\t"),
            0x08 => Some("\\b"),
            0x0C => Some("\\f"),
            0..=0x1F => {
                if start < i {
                    out.push_str(&s[start..i]);
                }
                let _ = write!(out, "\\u{:04x}", b);
                start = i + 1;
                continue;
            }
            _ => None,
        };
        if let Some(e) = esc {
            if start < i {
                out.push_str(&s[start..i]);
            }
            out.push_str(e);
            start = i + 1;
        }
    }
    if start < s.len() {
        out.push_str(&s[start..]);
    }
    out.push('"');
}

struct StackBuf {
    buf: [u8; 40],
    len: usize,
}
impl std::fmt::Write for StackBuf {
    #[inline]
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let b = s.as_bytes();
        let n = b.len().min(self.buf.len() - self.len);
        self.buf[self.len..self.len + n].copy_from_slice(&b[..n]);
        self.len += n;
        Ok(())
    }
}

/// Swift's `Float.description` with a trailing ".0" dropped (what `JSONEncoder` writes):
/// shortest round trip digits, decimal for 1e-4 <= |x| <= 2^24, otherwise `1.5e-05` / `3e+08`.
#[inline]
pub fn swift_float(f: f32, out: &mut String) {
    if f.is_nan() {
        return out.push_str("\"nan\"");
    }
    if f.is_infinite() {
        return out.push_str(if f > 0.0 { "\"inf\"" } else { "\"-inf\"" });
    }
    if f == 0.0 {
        return out.push_str(if f.is_sign_negative() { "-0" } else { "0" });
    }
    // Rust's `{:e}` is the shortest round trip form too: "1.6777218e7", "-5e-5".
    // Formatted into a stack buffer to avoid any per-float heap allocations.
    let mut sb = StackBuf { buf: [0u8; 40], len: 0 };
    let _ = write!(sb, "{f:e}");
    let bytes = &sb.buf[..sb.len];
    let mut e_pos = 0;
    while e_pos < bytes.len() && bytes[e_pos] != b'e' {
        e_pos += 1;
    }
    let mut m_start = 0;
    let neg = if bytes[0] == b'-' {
        m_start = 1;
        true
    } else {
        false
    };
    let mut digits = [0u8; 16];
    let mut dlen = 0usize;
    for &b in &bytes[m_start..e_pos] {
        if b != b'.' {
            digits[dlen] = b;
            dlen += 1;
        }
    }
    let mut exp: i32 = 0;
    let mut exp_neg = false;
    let mut ei = e_pos + 1;
    if ei < bytes.len() && bytes[ei] == b'-' {
        exp_neg = true;
        ei += 1;
    } else if ei < bytes.len() && bytes[ei] == b'+' {
        ei += 1;
    }
    while ei < bytes.len() {
        exp = exp * 10 + (bytes[ei] - b'0') as i32;
        ei += 1;
    }
    if exp_neg {
        exp = -exp;
    }
    if neg {
        out.push('-');
    }
    // SAFETY: `digits[..dlen]` are ASCII digits.
    let dstr = unsafe { std::str::from_utf8_unchecked(&digits[..dlen]) };
    let a = f.abs();
    if (1e-4..=16_777_216.0).contains(&a) {
        // Decimal: point after (exp + 1) digits.
        let point = exp + 1;
        if point <= 0 {
            out.push_str("0.");
            for _ in 0..-point {
                out.push('0');
            }
            out.push_str(dstr);
        } else if point as usize >= dlen {
            out.push_str(dstr);
            for _ in 0..(point as usize - dlen) {
                out.push('0');
            }
        } else {
            out.push_str(&dstr[..point as usize]);
            out.push('.');
            out.push_str(&dstr[point as usize..]);
        }
    } else {
        out.push(digits[0] as char);
        if dlen > 1 {
            out.push('.');
            out.push_str(&dstr[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        let ea = exp.unsigned_abs();
        if ea < 10 {
            out.push('0');
            out.push((b'0' + ea as u8) as char);
        } else {
            write_u32(ea, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_match_swift() {
        // Expected strings are what Swift's JSONEncoder wrote for the same Float values (probe, 2026-11-01).
        let cases: &[(f32, &str)] = &[
            (0.0, "0"), (-0.0, "-0"), (1.0, "1"), (10.0, "10"), (0.1, "0.1"), (0.0001, "0.0001"),
            (0.00001, "1e-05"), (1.5e-5, "1.5e-05"), (123456.7, "123456.7"), (1e7, "10000000"),
            (16777216.0, "16777216"), (16777218.0, "1.6777218e+07"), (3e8, "3e+08"), (1e-10, "1e-10"),
            (f32::NAN, "\"nan\""), (f32::INFINITY, "\"inf\""), (f32::NEG_INFINITY, "\"-inf\""),
            (0.56078434, "0.56078434"), (f32::from_bits(1), "1e-45"), (f32::MAX, "3.4028235e+38"),
            (99999.99, "99999.99"), (0.001234, "0.001234"), (-10.5, "-10.5"),
        ];
        for (f, want) in cases {
            let mut s = String::new();
            swift_float(*f, &mut s);
            assert_eq!(&s, want, "for {f:e}");
        }
    }

    #[test]
    fn pretty_matches_swift_layout() {
        let v = J::Obj(vec![
            ("o", J::Obj(vec![])),
            ("e", J::Arr(vec![])),
            ("s", J::Str("a/b\"c\u{01}é".into())),
            ("n", J::Obj(vec![("x", J::Arr(vec![J::Arr(vec![]), J::Arr(vec![J::Int(1)])]))])),
        ]);
        let want = "{\n  \"e\" : [\n\n  ],\n  \"n\" : {\n    \"x\" : [\n      [\n\n      ],\n      [\n        1\n      ]\n    ]\n  },\n  \"o\" : {\n\n  },\n  \"s\" : \"a/b\\\"c\\u0001é\"\n}";
        assert_eq!(v.pretty(), want);
        assert_eq!(J::Arr(vec![J::Int(1), J::Bool(true)]).compact(), "[1,true]");
    }
}
