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
    // SAFETY: `swift_float_into` only appends ASCII.
    swift_float_into(f, unsafe { out.as_mut_vec() })
}

/// "00" .. "99" back to back, indexed by `2 * value`.
const DIGIT_PAIRS: &[u8; 200] = b"0001020304050607080910111213141516171819\
2021222324252627282930313233343536373839\
4041424344454647484950515253545556575859\
6061626364656667686970717273747576777879\
8081828384858687888990919293949596979899";

#[inline(always)]
unsafe fn write_pair(p: *mut u8, val: u32) {
    debug_assert!(val < 100);
    unsafe {
        std::ptr::copy_nonoverlapping(DIGIT_PAIRS.as_ptr().add((val as usize) * 2), p, 2);
    }
}

/// Writes the decimal digits of `v` directly to `p` (caller must ensure at least 10 writable bytes)
/// and returns the advanced pointer.
#[inline(always)]
pub(crate) unsafe fn write_u32_ptr(p: *mut u8, v: u32) -> *mut u8 {
    unsafe {
        if v < 10_000 {
            if v < 100 {
                if v < 10 {
                    *p = b'0' + v as u8;
                    return p.add(1);
                }
                write_pair(p, v);
                return p.add(2);
            }
            let hi = v / 100;
            let lo = v - hi * 100;
            let n = if hi >= 10 {
                write_pair(p, hi);
                2
            } else {
                *p = b'0' + hi as u8;
                1
            };
            write_pair(p.add(n), lo);
            return p.add(n + 2);
        }
        if v < 100_000_000 {
            let hi = v / 10_000;
            let lo = v - hi * 10_000;
            let p = if hi < 100 {
                if hi < 10 {
                    *p = b'0' + hi as u8;
                    p.add(1)
                } else {
                    write_pair(p, hi);
                    p.add(2)
                }
            } else {
                let h1 = hi / 100;
                let h2 = hi - h1 * 100;
                let n = if h1 >= 10 {
                    write_pair(p, h1);
                    2
                } else {
                    *p = b'0' + h1 as u8;
                    1
                };
                write_pair(p.add(n), h2);
                p.add(n + 2)
            };
            let l1 = lo / 100;
            let l2 = lo - l1 * 100;
            write_pair(p, l1);
            write_pair(p.add(2), l2);
            return p.add(4);
        }
        let hi = v / 100_000_000;
        let lo = v - hi * 100_000_000;
        let p = if hi >= 10 {
            write_pair(p, hi);
            p.add(2)
        } else {
            *p = b'0' + hi as u8;
            p.add(1)
        };
        let a = lo / 1_000_000;
        let r1 = lo - a * 1_000_000;
        let b = r1 / 10_000;
        let r2 = r1 - b * 10_000;
        let c = r2 / 100;
        let d = r2 - c * 100;
        write_pair(p, a);
        write_pair(p.add(2), b);
        write_pair(p.add(4), c);
        write_pair(p.add(6), d);
        p.add(8)
    }
}

/// Appends the decimal digits of `v` to `out` (two digits per step, no heap, no `fmt`).
#[inline]
pub(crate) fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.reserve(10);
    let len = out.len();
    // SAFETY: 10 bytes reserved above; `write_u32_ptr` writes 1..=10 ASCII digits.
    unsafe {
        let end = write_u32_ptr(out.as_mut_ptr().add(len), v);
        out.set_len(end as usize - out.as_ptr() as usize);
    }
}

/// `n` levels of two space indentation.
#[inline]
pub(crate) fn push_indent(out: &mut Vec<u8>, n: usize) {
    let mut rem = n * 2;
    while rem > 0 {
        let chunk = rem.min(SPACES.len());
        out.extend_from_slice(&SPACES.as_bytes()[..chunk]);
        rem -= chunk;
    }
}

const FLOAT_POW5_INV_BITCOUNT: i32 = 59;
const FLOAT_POW5_BITCOUNT: i32 = 61;

#[inline(always)]
const fn pow5bits(e: i32) -> i32 {
    (((e as u32) * 1217359) >> 19) as i32 + 1
}

const FLOAT_POW5_INV_SPLIT: [u64; 31] = {
    let mut t = [0u64; 31];
    let mut q = 0usize;
    let mut p5 = 1u128;
    while q < 31 {
        let b = pow5bits(q as i32) - 1;
        let shift = (b + FLOAT_POW5_INV_BITCOUNT) as u32;
        let div = if shift < 128 {
            (1u128 << shift) / p5
        } else {
            let half = 1u128 << (shift - 1);
            let q0 = half / p5;
            let r0 = half % p5;
            q0 * 2 + (r0 * 2) / p5
        };
        t[q] = if q == 0 { div as u64 } else { (div + 1) as u64 };
        p5 *= 5;
        q += 1;
    }
    t
};

const FLOAT_POW5_SPLIT: [u64; 47] = {
    let mut t = [0u64; 47];
    let mut i = 0usize;
    let mut p5 = 1u128;
    while i < 47 {
        let b = pow5bits(i as i32);
        t[i] = if b <= FLOAT_POW5_BITCOUNT {
            (p5 << (FLOAT_POW5_BITCOUNT - b) as u32) as u64
        } else {
            (p5 >> (b - FLOAT_POW5_BITCOUNT) as u32) as u64
        };
        p5 *= 5;
        i += 1;
    }
    t
};

#[inline(always)]
fn pow5_factor_32(mut v: u32) -> u32 {
    let mut count = 0;
    loop {
        let q = v / 5;
        if v - q * 5 != 0 {
            return count;
        }
        v = q;
        count += 1;
    }
}

#[inline(always)]
fn multiple_of_power_of_5_32(v: u32, p: i32) -> bool {
    pow5_factor_32(v) >= p as u32
}

#[inline(always)]
fn multiple_of_power_of_2_32(v: u32, p: i32) -> bool {
    (v & ((1u32 << p) - 1)) == 0
}

#[inline(always)]
fn log10_pow2(e: i32) -> i32 {
    (((e as u32) * 78913) >> 18) as i32
}

#[inline(always)]
fn log10_pow5(e: i32) -> i32 {
    (((e as u32) * 732923) >> 20) as i32
}

#[inline(always)]
fn mul_pow5_inv_div_pow2(m: u32, q: usize, j: i32) -> u32 {
    let mul = FLOAT_POW5_INV_SPLIT[q];
    (((m as u128) * (mul as u128)) >> (j as u32)) as u32
}

#[inline(always)]
fn mul_pow5_div_pow2(m: u32, i: usize, j: i32) -> u32 {
    let mul = FLOAT_POW5_SPLIT[i];
    (((m as u128) * (mul as u128)) >> (j as u32)) as u32
}

#[inline(always)]
fn decimal_length9(v: u32) -> u32 {
    if v >= 100_000_000 { 9 }
    else if v >= 10_000_000 { 8 }
    else if v >= 1_000_000 { 7 }
    else if v >= 100_000 { 6 }
    else if v >= 10_000 { 5 }
    else if v >= 1_000 { 4 }
    else if v >= 100 { 3 }
    else if v >= 10 { 2 }
    else { 1 }
}

#[cold]
#[inline(never)]
unsafe fn swift_tie_fallback(f: f32, p: *mut u8) -> *mut u8 {
    let mut tmp = Vec::with_capacity(24);
    swift_float_slow(f, &mut tmp);
    unsafe {
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), p, tmp.len());
        p.add(tmp.len())
    }
}

/// Writes `olength` digits of `output` at `p` (1..=9 digits) and returns `p.add(olength)`.
#[inline(always)]
unsafe fn write_digits9(p: *mut u8, mut output: u32, olength: u32) -> *mut u8 {
    unsafe {
        let mut i = olength as usize;
        while output >= 100 {
            let c = (output % 100) * 2;
            output /= 100;
            i -= 2;
            std::ptr::copy_nonoverlapping(DIGIT_PAIRS.as_ptr().add(c as usize), p.add(i), 2);
        }
        if output >= 10 {
            let c = output * 2;
            std::ptr::copy_nonoverlapping(DIGIT_PAIRS.as_ptr().add(c as usize), p, 2);
        } else {
            *p = b'0' + output as u8;
        }
        p.add(olength as usize)
    }
}

/// Writes Swift's `Float.description` (with trailing `.0` dropped) directly to `p`.
/// Caller must ensure at least 24 writable bytes at `p`. Returns the advanced pointer.
#[inline]
pub(crate) unsafe fn write_swift_f32_ptr(mut p: *mut u8, f: f32) -> *mut u8 {
    unsafe {
        let bits = f.to_bits();
        let sign = (bits >> 31) != 0;
        let ieee_mantissa = bits & 0x7f_ffff;
        let ieee_exponent = (bits >> 23) & 0xff;

        if ieee_exponent == 0xff {
            return if ieee_mantissa != 0 {
                std::ptr::copy_nonoverlapping(b"\"nan\"".as_ptr(), p, 5);
                p.add(5)
            } else if sign {
                std::ptr::copy_nonoverlapping(b"\"-inf\"".as_ptr(), p, 6);
                p.add(6)
            } else {
                std::ptr::copy_nonoverlapping(b"\"inf\"".as_ptr(), p, 5);
                p.add(5)
            };
        }
        if ieee_exponent == 0 && ieee_mantissa == 0 {
            if sign {
                *p = b'-';
                *p.add(1) = b'0';
                return p.add(2);
            }
            *p = b'0';
            return p.add(1);
        }

        let (e2, m2) = if ieee_exponent == 0 {
            (1 - 127 - 23 - 2, ieee_mantissa)
        } else {
            (ieee_exponent as i32 - 127 - 23 - 2, (1 << 23) | ieee_mantissa)
        };
        let even = (m2 & 1) == 0;
        let accept_bounds = even;

        let mv = 4 * m2;
        let mp = 4 * m2 + 2;
        let mm_shift = u32::from(ieee_mantissa != 0 || ieee_exponent <= 1);
        let mm = 4 * m2 - 1 - mm_shift;

        let mut vr: u32;
        let mut vp: u32;
        let mut vm: u32;
        let e10: i32;
        let mut vm_is_trailing_zeros = false;
        let mut vr_is_trailing_zeros = false;
        let mut last_removed_digit: u8 = 0;

        if e2 >= 0 {
            let q = log10_pow2(e2);
            e10 = q;
            let k = FLOAT_POW5_INV_BITCOUNT + pow5bits(q) - 1;
            let i = -e2 + q + k;
            vr = mul_pow5_inv_div_pow2(mv, q as usize, i);
            vp = mul_pow5_inv_div_pow2(mp, q as usize, i);
            vm = mul_pow5_inv_div_pow2(mm, q as usize, i);
            if q != 0 && (vp - 1) / 10 <= vm / 10 {
                let l = FLOAT_POW5_INV_BITCOUNT + pow5bits(q - 1) - 1;
                last_removed_digit = (mul_pow5_inv_div_pow2(mv, (q - 1) as usize, -e2 + q - 1 + l) % 10) as u8;
            }
            if q <= 9 {
                if mv % 5 == 0 {
                    vr_is_trailing_zeros = multiple_of_power_of_5_32(mv, q);
                } else if accept_bounds {
                    vm_is_trailing_zeros = multiple_of_power_of_5_32(mm, q);
                } else {
                    vp -= u32::from(multiple_of_power_of_5_32(mp, q));
                }
            }
        } else {
            let q = log10_pow5(-e2);
            e10 = q + e2;
            let i = -e2 - q;
            let k = pow5bits(i) - FLOAT_POW5_BITCOUNT;
            let mut j = q - k;
            vr = mul_pow5_div_pow2(mv, i as usize, j);
            vp = mul_pow5_div_pow2(mp, i as usize, j);
            vm = mul_pow5_div_pow2(mm, i as usize, j);
            if q != 0 && (vp - 1) / 10 <= vm / 10 {
                j = q - 1 - (pow5bits(i + 1) - FLOAT_POW5_BITCOUNT);
                last_removed_digit = (mul_pow5_div_pow2(mv, (i + 1) as usize, j) % 10) as u8;
            }
            if q <= 1 {
                vr_is_trailing_zeros = true;
                if accept_bounds {
                    vm_is_trailing_zeros = mm_shift == 1;
                } else {
                    vp -= 1;
                }
            } else if q < 31 {
                vr_is_trailing_zeros = multiple_of_power_of_2_32(mv, q - 1);
            }
        }

        let mut removed = 0i32;
        let output: u32 = if vm_is_trailing_zeros || vr_is_trailing_zeros {
            while vp / 10 > vm / 10 {
                vm_is_trailing_zeros &= vm - (vm / 10) * 10 == 0;
                vr_is_trailing_zeros &= last_removed_digit == 0;
                let vr_div = vr / 10;
                last_removed_digit = (vr - vr_div * 10) as u8;
                vr = vr_div;
                vp /= 10;
                vm /= 10;
                removed += 1;
            }
            if vm_is_trailing_zeros {
                while vm % 10 == 0 {
                    vr_is_trailing_zeros &= last_removed_digit == 0;
                    let vr_div = vr / 10;
                    last_removed_digit = (vr - vr_div * 10) as u8;
                    vr = vr_div;
                    vm /= 10;
                    removed += 1;
                }
            }
            if vr_is_trailing_zeros && last_removed_digit == 5 && vr % 2 == 0 {
                // Exact decimal tie: Swift / Rust `{:e}` rounds half-up (away from zero).
                return swift_tie_fallback(f, p);
            }
            vr + u32::from((vr == vm && (!accept_bounds || !vm_is_trailing_zeros)) || last_removed_digit >= 5)
        } else {
            while vp / 10 > vm / 10 {
                let vr_div = vr / 10;
                last_removed_digit = (vr - vr_div * 10) as u8;
                vr = vr_div;
                vp /= 10;
                vm /= 10;
                removed += 1;
            }
            vr + u32::from(vr == vm || last_removed_digit >= 5)
        };

        let exp = e10 + removed;
        let olength = decimal_length9(output);
        let sci_exp = exp + olength as i32 - 1;

        if sign {
            *p = b'-';
            p = p.add(1);
        }

        // Plain decimal for 1e-4 <= |f| <= 16_777_216.0 (2^24), matching Swift's Float.description.
        let abs_bits = bits & 0x7fff_ffff;
        if (0x38d1_b717..=0x4b80_0000).contains(&abs_bits) {
            let point = sci_exp + 1;
            if point <= 0 {
                *p = b'0';
                *p.add(1) = b'.';
                p = p.add(2);
                let zeros = (-point) as usize;
                std::ptr::write_bytes(p, b'0', zeros);
                p = p.add(zeros);
                return write_digits9(p, output, olength);
            } else if point as u32 >= olength {
                p = write_digits9(p, output, olength);
                let zeros = point as usize - olength as usize;
                std::ptr::write_bytes(p, b'0', zeros);
                return p.add(zeros);
            } else {
                // Split `output` at `point`: write all digits starting at p+1, then shift integer part left by 1 and insert '.'.
                let pt = point as usize;
                let end = write_digits9(p.add(1), output, olength);
                std::ptr::copy(p.add(1), p, pt);
                *p.add(pt) = b'.';
                return end;
            }
        }

        // Scientific notation: `d[.ddd]e[+-]dd`
        let end = write_digits9(p.add(1), output, olength);
        *p = *p.add(1);
        let mut cur = if olength > 1 {
            *p.add(1) = b'.';
            end
        } else {
            p.add(1)
        };
        *cur = b'e';
        *cur.add(1) = if sci_exp < 0 { b'-' } else { b'+' };
        cur = cur.add(2);
        let ea = sci_exp.unsigned_abs();
        if ea < 10 {
            *cur = b'0';
            *cur.add(1) = b'0' + ea as u8;
            cur.add(2)
        } else {
            write_pair(cur, ea) ;
            cur.add(2)
        }
    }
}

/// Byte buffer version of [`swift_float`].
#[inline]
pub fn swift_float_into(f: f32, out: &mut Vec<u8>) {
    out.reserve(24);
    let len = out.len();
    // SAFETY: 24 bytes reserved above; `write_swift_f32_ptr` writes at most 24 ASCII bytes.
    unsafe {
        let end = write_swift_f32_ptr(out.as_mut_ptr().add(len), f);
        out.set_len(end as usize - out.as_ptr() as usize);
    }
}

#[cold]
#[inline(never)]
fn swift_float_slow(f: f32, out: &mut Vec<u8>) {
    if f.is_nan() {
        return out.extend_from_slice(b"\"nan\"");
    }
    if f.is_infinite() {
        return out.extend_from_slice(if f > 0.0 { b"\"inf\"" } else { b"\"-inf\"" });
    }
    if f == 0.0 {
        return out.extend_from_slice(if f.is_sign_negative() { b"-0" } else { b"0" });
    }
    // Rust's `{:e}` is the shortest round trip form too: "1.6777218e7", "-5e-5".
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
        out.push(b'-');
    }
    let dstr = &digits[..dlen];
    let a = f.abs();
    if (1e-4..=16_777_216.0).contains(&a) {
        // Decimal: point after (exp + 1) digits.
        let point = exp + 1;
        if point <= 0 {
            out.extend_from_slice(b"0.");
            for _ in 0..-point {
                out.push(b'0');
            }
            out.extend_from_slice(dstr);
        } else if point as usize >= dlen {
            out.extend_from_slice(dstr);
            for _ in 0..(point as usize - dlen) {
                out.push(b'0');
            }
        } else {
            out.extend_from_slice(&dstr[..point as usize]);
            out.push(b'.');
            out.extend_from_slice(&dstr[point as usize..]);
        }
    } else {
        out.push(digits[0]);
        if dlen > 1 {
            out.push(b'.');
            out.extend_from_slice(&dstr[1..]);
        }
        out.push(b'e');
        out.push(if exp < 0 { b'-' } else { b'+' });
        let ea = exp.unsigned_abs();
        if ea < 10 {
            out.push(b'0');
            out.push(b'0' + ea as u8);
        } else {
            push_u32(out, ea);
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

    /// The `zmij` fast path must agree with the `{:e}` reference on every float class.
    #[test]
    fn fast_float_matches_reference() {
        let mut x = 0x2545F4914F6CDD1Du64;
        let check = |f: f32| {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            swift_float_into(f, &mut a);
            swift_float_slow(f, &mut b);
            assert_eq!(String::from_utf8(a).unwrap(), String::from_utf8(b).unwrap(), "for {f:e} ({:#x})", f.to_bits());
        };
        for _ in 0..400_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            check(f32::from_bits((x >> 16) as u32)); // every exponent
            check(((x >> 40) as f32 / 16_777_216.0 - 0.5) * 200.0); // typical vertex coordinates
            check((x >> 44) as f32 / 1024.0); // short decimals / integers
            check((x >> 40) as f32 / (1u32 << ((x >> 8) % 24)) as f32); // dyadic values: exact ties
            check((x >> 38) as f32 / (1u32 << ((x >> 8) % 26)) as f32);
        }
        for e in -12..=12 {
            for m in [1.0f32, 1.5, 9.999, 5.0, 2.5] {
                let v = m * 10f32.powi(e);
                check(v);
                check(-v);
            }
        }
        for v in [1e-4f32, 9.999999e-5, 16_777_216.0, 16_777_218.0, 1e7, 1.0, -1.0, 123456.7, 0.0, -0.0, f32::NAN, f32::INFINITY, f32::MIN_POSITIVE] {
            check(v);
        }
    }

    #[test]
    fn push_u32_matches_display() {
        for v in [0u32, 1, 9, 10, 99, 100, 101, 999, 1000, 65535, 1_000_000, 4_294_967_295, 123_456_789] {
            let mut b = Vec::new();
            push_u32(&mut b, v);
            assert_eq!(String::from_utf8(b).unwrap(), v.to_string());
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
