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

fn indent(out: &mut String, n: usize) {
    for _ in 0..n {
        out.push_str("  ");
    }
}

fn write(v: &J, out: &mut String, level: Option<usize>) {
    match v {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Int(i) => out.push_str(&i.to_string()),
        J::F32(f) => swift_float(*f, out),
        J::Str(s) => string(s, out),
        J::Arr(items) => container(out, level, '[', ']', items.len(), |out, i, l| write(&items[i], out, l)),
        J::Obj(fields) => {
            let mut sorted: Vec<&(&str, J)> = fields.iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(b.0));
            let sep = if level.is_some() { " : " } else { ":" };
            container(out, level, '{', '}', sorted.len(), |out, i, l| {
                string(sorted[i].0, out);
                out.push_str(sep);
                write(&sorted[i].1, out, l);
            })
        }
    }
}

fn container(out: &mut String, level: Option<usize>, open: char, close: char, n: usize, mut item: impl FnMut(&mut String, usize, Option<usize>)) {
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

fn string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Swift's `Float.description` with a trailing ".0" dropped (what `JSONEncoder` writes):
/// shortest round trip digits, decimal for 1e-4 <= |x| <= 2^24, otherwise `1.5e-05` / `3e+08`.
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
    let sci = format!("{:e}", f);
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let (neg, mant) = mant.strip_prefix('-').map_or((false, mant), |m| (true, m));
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    if neg {
        out.push('-');
    }
    let a = f.abs();
    if (1e-4..=16_777_216.0).contains(&a) {
        // Decimal: point after (exp + 1) digits.
        let point = exp + 1;
        if point <= 0 {
            out.push_str("0.");
            for _ in 0..-point {
                out.push('0');
            }
            out.push_str(&digits);
        } else if point as usize >= digits.len() {
            out.push_str(&digits);
            for _ in 0..(point as usize - digits.len()) {
                out.push('0');
            }
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push_str(&format!("e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs()));
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
