//! Decoding: serde types in Swift's Codable shapes, then conversion with Swift's repair rules.

use super::*;
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::Deserialize;

mod fast {
    use super::*;
    use std::ops::Range;

    /// Arrays shorter than this are left in the stub: serde reads them faster than the cut-out costs.
    const MIN_ARRAY: usize = 16 * 1024;
    /// Bytes per parallel work item (tiny in small tests so they still split into many pieces).
    #[inline(always)]
    fn chunk_size(len: usize) -> usize {
        if cfg!(test) && len < 1_000_000 { 4096 } else { 256 * 1024 }
    }

    /// Where `mesh` gets the arrays that the stub replaced by markers.
    pub(super) trait Ext {
        fn positions(&mut self, k: u32) -> Option<(Vec<Vec3>, bool)>;
        fn faces(&mut self, k: u32, n: usize) -> Option<Vec<Vec<u32>>>;
    }

    /// The serde path: documents never carry markers.
    pub(super) struct NoExt;
    impl Ext for NoExt {
        fn positions(&mut self, _: u32) -> Option<(Vec<Vec3>, bool)> {
            None
        }
        fn faces(&mut self, _: u32, _: usize) -> Option<Vec<Vec<u32>>> {
            None
        }
    }

    struct PreParsed<'a> {
        pos: Vec<Option<(Vec<Vec3>, bool)>>,
        faces: Vec<Option<(Vec<Vec<u32>>, u32)>>,
        used_p: Vec<bool>,
        used_f: Vec<bool>,
        failed: bool,
        _marker: std::marker::PhantomData<&'a [u8]>,
    }

    impl Ext for PreParsed<'_> {
        fn positions(&mut self, k: u32) -> Option<(Vec<Vec3>, bool)> {
            let idx = k as usize;
            if idx >= self.pos.len() || std::mem::replace(&mut self.used_p[idx], true) {
                self.failed = true;
                return None;
            }
            let out = self.pos[idx].take();
            self.failed |= out.is_none();
            out
        }
        fn faces(&mut self, k: u32, n: usize) -> Option<Vec<Vec<u32>>> {
            let idx = k as usize;
            if idx >= self.faces.len() || std::mem::replace(&mut self.used_f[idx], true) {
                self.failed = true;
                return None;
            }
            let Some((mut fs, max_idx)) = self.faces[idx].take() else {
                self.failed = true;
                return None;
            };
            if (max_idx as usize) >= n {
                fs.retain(|f| f.iter().all(|&i| (i as usize) < n));
            }
            Some(fs)
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
        let np = pos.len();
        let nf = faces.len();
        #[cfg(feature = "threads")]
        let (parsed_pos, parsed_faces) = rayon::join(
            || pos.into_iter().map(|r| parse_positions(&bytes[r])).collect::<Vec<_>>(),
            || faces.into_iter().map(|r| parse_faces_unchecked(&bytes[r])).collect::<Vec<_>>(),
        );
        #[cfg(not(feature = "threads"))]
        let (parsed_pos, parsed_faces) = (
            pos.into_iter().map(|r| parse_positions(&bytes[r])).collect::<Vec<_>>(),
            faces.into_iter().map(|r| parse_faces_unchecked(&bytes[r])).collect::<Vec<_>>(),
        );
        let mut ext = PreParsed {
            pos: parsed_pos,
            faces: parsed_faces,
            used_p: vec![false; np],
            used_f: vec![false; nf],
            failed: false,
            _marker: std::marker::PhantomData,
        };
        let out = convert(p, &mut ext).ok()?;
        (!ext.failed && ext.used_p.iter().all(|&u| u) && ext.used_f.iter().all(|&u| u)).then_some(out)
    }

    #[inline(always)]
    fn is_ws(b: u8) -> bool {
        matches!(b, b' ' | b'\n' | b'\r' | b'\t')
    }

    /// Fast JSON whitespace skip: 1-branch exit for compact tokens, 8-byte SWAR space skip for pretty indents.
    #[inline(always)]
    fn skip_ws(b: &[u8], mut p: usize) -> usize {
        let len = b.len();
        if p >= len {
            return p;
        }
        if b[p] > b' ' {
            return p;
        }
        if b[p] == b'\n' {
            p += 1;
        }
        if p + 8 <= len {
            // SAFETY: `p + 8 <= len`.
            let w = unsafe { (b.as_ptr().add(p) as *const u64).read_unaligned().to_le() };
            let xor = w ^ 0x2020_2020_2020_2020;
            if xor != 0 {
                p += (xor.trailing_zeros() >> 3) as usize;
                // SAFETY: `p < len` because `idx < 8` and `p_old + 8 <= len`.
                if unsafe { *b.get_unchecked(p) } > b' ' {
                    return p;
                }
            } else {
                p += 8;
                if p + 8 <= len {
                    // SAFETY: `p + 8 <= len`.
                    let w2 = unsafe { (b.as_ptr().add(p) as *const u64).read_unaligned().to_le() };
                    let xor2 = w2 ^ 0x2020_2020_2020_2020;
                    if xor2 != 0 {
                        p += (xor2.trailing_zeros() >> 3) as usize;
                        if unsafe { *b.get_unchecked(p) } > b' ' {
                            return p;
                        }
                    } else {
                        p += 8;
                    }
                }
            }
        }
        while p + 8 <= len {
            // SAFETY: `p + 8 <= len`.
            let w = unsafe { (b.as_ptr().add(p) as *const u64).read_unaligned().to_le() };
            let xor = w ^ 0x2020_2020_2020_2020;
            if xor == 0 {
                p += 8;
                continue;
            }
            let idx = (xor.trailing_zeros() >> 3) as usize;
            p += idx;
            // SAFETY: `p < len` because `idx < 8` and `p_old + 8 <= len`.
            if unsafe { *b.get_unchecked(p) } > b' ' {
                return p;
            }
            break;
        }
        while p < len && is_ws(b[p]) {
            p += 1;
        }
        p
    }

    /// First `needle` in `h` (64-byte NEON on aarch64, 8-byte SWAR elsewhere).
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[inline]
    fn find_byte(h: &[u8], needle: u8) -> Option<usize> {
        use std::arch::aarch64::*;
        let len = h.len();
        let ptr = h.as_ptr();
        let mut at = 0usize;
        unsafe {
            let v_needle = vdupq_n_u8(needle);
            while at + 64 <= len {
                let v0 = vceqq_u8(vld1q_u8(ptr.add(at)), v_needle);
                let v1 = vceqq_u8(vld1q_u8(ptr.add(at + 16)), v_needle);
                let v2 = vceqq_u8(vld1q_u8(ptr.add(at + 32)), v_needle);
                let v3 = vceqq_u8(vld1q_u8(ptr.add(at + 48)), v_needle);
                let any = vorrq_u8(vorrq_u8(v0, v1), vorrq_u8(v2, v3));
                if vmaxvq_u8(any) != 0 {
                    for (k, v) in [(0usize, v0), (16, v1), (32, v2), (48, v3)] {
                        if vmaxvq_u8(v) != 0 {
                            let lo = vgetq_lane_u64(vreinterpretq_u64_u8(v), 0);
                            if lo != 0 {
                                return Some(at + k + (lo.trailing_zeros() >> 3) as usize);
                            }
                            let hi = vgetq_lane_u64(vreinterpretq_u64_u8(v), 1);
                            return Some(at + k + 8 + (hi.trailing_zeros() >> 3) as usize);
                        }
                    }
                }
                at += 64;
            }
            while at + 16 <= len {
                let v = vceqq_u8(vld1q_u8(ptr.add(at)), v_needle);
                if vmaxvq_u8(v) != 0 {
                    let lo = vgetq_lane_u64(vreinterpretq_u64_u8(v), 0);
                    if lo != 0 {
                        return Some(at + (lo.trailing_zeros() >> 3) as usize);
                    }
                    let hi = vgetq_lane_u64(vreinterpretq_u64_u8(v), 1);
                    return Some(at + 8 + (hi.trailing_zeros() >> 3) as usize);
                }
                at += 16;
            }
        }
        h[at..].iter().position(|&b| b == needle).map(|i| at + i)
    }

    #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
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

    /// Finds the closing `]` of the outer `faces` array in `r` (a `]` followed by whitespace then `]`),
    /// scanning 64 bytes per step on aarch64 and skipping `],` branchlessly.
    fn find_faces_end(bytes: &[u8], r: Range<usize>) -> Option<usize> {
        let mut at = r.start;
        let end = r.end;
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        unsafe {
            use std::arch::aarch64::*;
            let ptr = bytes.as_ptr();
            let v_rb = vdupq_n_u8(b']');
            let v_comma = vdupq_n_u8(b',');
            if at + 80 <= end {
                let mut p0 = vld1q_u8(ptr.add(at));
                loop {
                    let p1 = vld1q_u8(ptr.add(at + 16));
                    let p2 = vld1q_u8(ptr.add(at + 32));
                    let p3 = vld1q_u8(ptr.add(at + 48));
                    let p4 = vld1q_u8(ptr.add(at + 64));
                    let n0 = vextq_u8::<1>(p0, p1);
                    let n1 = vextq_u8::<1>(p1, p2);
                    let n2 = vextq_u8::<1>(p2, p3);
                    let n3 = vextq_u8::<1>(p3, p4);
                    let h0 = vbicq_u8(vceqq_u8(p0, v_rb), vceqq_u8(n0, v_comma));
                    let h1 = vbicq_u8(vceqq_u8(p1, v_rb), vceqq_u8(n1, v_comma));
                    let h2 = vbicq_u8(vceqq_u8(p2, v_rb), vceqq_u8(n2, v_comma));
                    let h3 = vbicq_u8(vceqq_u8(p3, v_rb), vceqq_u8(n3, v_comma));
                    let any = vorrq_u8(vorrq_u8(h0, h1), vorrq_u8(h2, h3));
                    if vmaxvq_u8(any) != 0 {
                        for (k, v) in [(0usize, h0), (16, h1), (32, h2), (48, h3)] {
                            if vmaxvq_u8(v) != 0 {
                                let mut lo = vgetq_lane_u64(vreinterpretq_u64_u8(v), 0);
                                while lo != 0 {
                                    let b = (lo.trailing_zeros() >> 3) as usize;
                                    let after = at + k + b + 1;
                                    let q = skip_ws(bytes, after);
                                    if bytes.get(q) == Some(&b']') {
                                        return Some(q);
                                    }
                                    lo &= !(0xffu64 << (b * 8));
                                }
                                let mut hi = vgetq_lane_u64(vreinterpretq_u64_u8(v), 1);
                                while hi != 0 {
                                    let b = (hi.trailing_zeros() >> 3) as usize;
                                    let after = at + k + 8 + b + 1;
                                    let q = skip_ws(bytes, after);
                                    if bytes.get(q) == Some(&b']') {
                                        return Some(q);
                                    }
                                    hi &= !(0xffu64 << (b * 8));
                                }
                            }
                        }
                    }
                    at += 64;
                    if at + 80 > end {
                        break;
                    }
                    p0 = p4;
                }
            }
        }
        while at < end {
            let Some(x) = find_byte(&bytes[at..end], b']') else { break };
            let after = at + x + 1;
            if bytes.get(after) == Some(&b',') {
                at = after + 1;
                continue;
            }
            let q = skip_ws(bytes, after);
            if bytes.get(q) == Some(&b']') {
                return Some(q);
            }
            at = after;
        }
        None
    }

    /// Runs `f(start..end)` over `from..len` in parallel waves and returns the first `Some` in document order.
    fn find_first(len: usize, from: usize, f: impl Fn(Range<usize>) -> Option<usize> + Sync + Send) -> Option<usize> {
        let rem = len - from;
        let chunk = if cfg!(test) && rem < 1_000_000 { 4096 } else { 1024 * 1024 };
        let range = |c: usize| from + c * chunk..(from + (c + 1) * chunk).min(len);
        let nc = rem.div_ceil(chunk);
        #[cfg(feature = "threads")]
        if nc > 1 {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicUsize, Ordering};
            let wave = 64usize;
            let mut base = 0usize;
            while base < nc {
                let end = (base + wave).min(nc);
                let best = AtomicUsize::new(usize::MAX);
                (base..end).into_par_iter().for_each(|c| {
                    let r = range(c);
                    if r.start >= best.load(Ordering::Relaxed) {
                        return;
                    }
                    if let Some(hit) = f(r) {
                        best.fetch_min(hit, Ordering::Relaxed);
                    }
                });
                let ans = best.load(Ordering::Relaxed);
                if ans != usize::MAX {
                    return Some(ans);
                }
                base = end;
            }
            return None;
        }
        (0..nc).find_map(|c| f(range(c)))
    }

    /// Cuts the big arrays out of `bytes`: the stub document, plus the byte range inside the `[ ]` of each.
    fn extract(bytes: &[u8]) -> Option<(Vec<u8>, Vec<Range<usize>>, Vec<Range<usize>>)> {
        let n = bytes.len();
        let mut stub = Vec::with_capacity(2048);
        let (mut pos, mut faces) = (Vec::new(), Vec::new());
        let mut copied = 0usize;
        let mut i = 0usize;
        while i < n {
            let Some(off) = find_byte(&bytes[i..], b'"') else {
                break;
            };
            let key_start = i + off;
            let mut j = key_start + 1;
            loop {
                match *bytes.get(j)? {
                    b'\\' => j += 2,
                    b'"' => break,
                    _ => j += 1,
                }
            }
            let key = &bytes[key_start + 1..j];
            i = j + 1;
            let is_pos = key == b"positions";
            if !is_pos && key != b"faces" {
                continue;
            }
            let mut k = skip_ws(bytes, i);
            if bytes.get(k) != Some(&b':') {
                continue;
            }
            k = skip_ws(bytes, k + 1);
            if bytes.get(k) != Some(&b'[') {
                continue;
            }
            let body = k + 1;
            let f = skip_ws(bytes, body);
            let first = *bytes.get(f)?;
            if !(if is_pos { first == b'-' || first.is_ascii_digit() } else { first == b'[' }) {
                continue;
            }
            let end = if is_pos {
                find_first(n, body, |r| find_byte(&bytes[r.clone()], b']').map(|x| r.start + x))?
            } else {
                find_first(n, body, |r| find_faces_end(bytes, r))?
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

    /// Splits `body` into pieces that each start where `cut_after` says a record starts.
    fn split_points(body: &[u8], mut cut_after: impl FnMut(&[u8]) -> Option<usize>) -> Vec<usize> {
        let chunk = chunk_size(body.len());
        let mut starts = vec![0usize];
        let mut from = chunk;
        while from < body.len() {
            match cut_after(&body[from..]) {
                Some(off) if from + off < body.len() => {
                    let at = from + off;
                    if at > *starts.last().unwrap() {
                        starts.push(at);
                    }
                    from = at + chunk;
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
    const POW10_U64: [u64; 9] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000];

    /// Parses `nd` (1..=8) little-endian ASCII digits from the low bytes of `w` into an integer.
    #[inline(always)]
    fn swar_digits_u64(w: u64, nd: usize) -> u64 {
        let mut val = (w & 0x0f0f_0f0f_0f0f_0f0f) << ((8 - nd) * 8);
        val = (val.wrapping_mul(2561) >> 8) & 0x00ff_00ff_00ff_00ff;
        val = (val.wrapping_mul(6_553_601) >> 16) & 0x0000_ffff_0000_ffff;
        val.wrapping_mul(42_949_672_960_001) >> 32
    }

    #[inline(always)]
    fn non_digit_mask(w: u64) -> u64 {
        (w.wrapping_sub(0x3030_3030_3030_3030) | w.wrapping_add(0x4646_4646_4646_4646) | w) & 0x8080_8080_8080_8080
    }

    #[cold]
    #[inline(never)]
    fn parse_number_slow(slice: &[u8]) -> Option<f64> {
        std::str::from_utf8(slice).ok()?.parse::<f64>().ok()
    }

    #[cold]
    #[inline(never)]
    fn parse_exponent(b: &[u8], mut i: usize) -> Option<(i32, usize)> {
        let len = b.len();
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
        while i < len {
            let d = b[i].wrapping_sub(b'0');
            if d >= 10 {
                break;
            }
            ev = (ev * 10 + d as i32).min(100_000);
            i += 1;
        }
        if i == es {
            return None;
        }
        Some((if eneg { -ev } else { ev }, i))
    }

    /// One strict JSON number at `start`, returning `(value, next_pos, is_f32_finite)`.
    #[inline(always)]
    fn parse_number(b: &[u8], start: usize) -> Option<(f32, usize, bool)> {
        let len = b.len();
        if start + 32 <= len {
            if let Some(res) = parse_number_unchecked(b, start) {
                return Some(res);
            }
        }
        parse_number_bounded(b, start)
    }

    #[inline(always)]
    fn parse_number_unchecked(b: &[u8], start: usize) -> Option<(f32, usize, bool)> {
        // SAFETY: caller ensures `start + 32 <= b.len()`.
        let ptr = b.as_ptr();
        let mut i = start;
        let neg = unsafe { *ptr.add(i) } == b'-';
        i += neg as usize;
        let d0 = unsafe { *ptr.add(i) }.wrapping_sub(b'0');
        if d0 >= 10 {
            return None;
        }
        let mut mant = d0 as u64;
        let mut digits = u32::from(d0 != 0);
        i += 1;
        if d0 == 0 {
            if unsafe { *ptr.add(i) }.wrapping_sub(b'0') < 10 {
                return None;
            }
        } else {
            let d1 = unsafe { *ptr.add(i) }.wrapping_sub(b'0');
            if d1 < 10 {
                mant = mant * 10 + d1 as u64;
                digits += 1;
                i += 1;
                let d2 = unsafe { *ptr.add(i) }.wrapping_sub(b'0');
                if d2 < 10 {
                    mant = mant * 10 + d2 as u64;
                    digits += 1;
                    i += 1;
                    while i < start + 19 {
                        let d = unsafe { *ptr.add(i) }.wrapping_sub(b'0');
                        if d >= 10 {
                            break;
                        }
                        mant = mant * 10 + d as u64;
                        digits += 1;
                        i += 1;
                    }
                    if unsafe { *ptr.add(i) }.wrapping_sub(b'0') < 10 {
                        return None;
                    }
                }
            }
        }

        let mut e10 = 0i32;
        if unsafe { *ptr.add(i) } == b'.' {
            i += 1;
            if digits <= 10 {
                let w = unsafe { (ptr.add(i) as *const u64).read_unaligned().to_le() };
                let non_digit = non_digit_mask(w);
                if non_digit != 0 {
                    let nd = (non_digit.trailing_zeros() >> 3) as usize;
                    if nd == 0 {
                        return None;
                    }
                    mant = mant * POW10_U64[nd] + swar_digits_u64(w, nd);
                    e10 -= nd as i32;
                    i += nd;
                } else {
                    mant = mant * POW10_U64[8] + swar_digits_u64(w, 8);
                    digits = if mant == 0 { 0 } else { digits + 8 };
                    e10 -= 8;
                    i += 8;
                    if digits <= 10 {
                        let w2 = unsafe { (ptr.add(i) as *const u64).read_unaligned().to_le() };
                        let non_digit2 = non_digit_mask(w2);
                        if non_digit2 != 0 {
                            let nd = (non_digit2.trailing_zeros() >> 3) as usize;
                            if nd > 0 {
                                mant = mant * POW10_U64[nd] + swar_digits_u64(w2, nd);
                                e10 -= nd as i32;
                                i += nd;
                            }
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
            } else {
                return None;
            }
        }

        if unsafe { *ptr.add(i) } == b'e' || unsafe { *ptr.add(i) } == b'E' {
            let (ev, next_i) = parse_exponent(b, i + 1)?;
            e10 += ev;
            i = next_i;
        }

        if mant <= (1 << 53) && (-22..=22).contains(&e10) {
            let m = mant as f64;
            let a = if e10 == 0 {
                m
            } else if e10 < 0 {
                m / POW10[(-e10) as usize]
            } else {
                m * POW10[e10 as usize]
            };
            let v = if neg { -a } else { a };
            return Some((v as f32, i, true));
        }
        None
    }

    #[cold]
    #[inline(never)]
    fn parse_number_bounded(b: &[u8], start: usize) -> Option<(f32, usize, bool)> {
        let len = b.len();
        let mut i = start;
        if i >= len {
            return None;
        }
        let neg = b[i] == b'-';
        i += neg as usize;
        if i >= len {
            return None;
        }
        let mut e10 = 0i32;
        let mut slow = false;
        let d0 = b[i].wrapping_sub(b'0');
        if d0 >= 10 {
            return None;
        }
        let mut mant = d0 as u64;
        let mut digits = u32::from(d0 != 0);
        i += 1;
        if i < len {
            let d1 = b[i].wrapping_sub(b'0');
            let is_d1 = d1 < 10;
            if is_d1 && d0 == 0 {
                return None;
            }
            mant = if is_d1 { mant * 10 + d1 as u64 } else { mant };
            digits += u32::from(is_d1);
            i += is_d1 as usize;
            if i < len && b[i].wrapping_sub(b'0') < 10 {
                while i < len {
                    let d = b[i].wrapping_sub(b'0');
                    if d >= 10 {
                        break;
                    }
                    if digits < 18 {
                        mant = mant * 10 + d as u64;
                        digits += 1;
                    } else {
                        slow = true;
                    }
                    i += 1;
                }
            }
        }

        if i < len && b[i] == b'.' {
            i += 1;
            let fs = i;
            'frac: {
                if i + 8 <= len && digits <= 10 {
                    // SAFETY: `i + 8 <= len`.
                    let w = unsafe { (b.as_ptr().add(i) as *const u64).read_unaligned().to_le() };
                    let non_digit = non_digit_mask(w);
                    if non_digit != 0 {
                        let nd = (non_digit.trailing_zeros() >> 3) as usize;
                        if nd == 0 {
                            return None;
                        }
                        mant = mant * POW10_U64[nd] + swar_digits_u64(w, nd);
                        e10 -= nd as i32;
                        i += nd;
                        break 'frac;
                    }
                    mant = mant * POW10_U64[8] + swar_digits_u64(w, 8);
                    digits = if mant == 0 { 0 } else { digits + 8 };
                    e10 -= 8;
                    i += 8;
                    if i + 8 <= len && digits <= 10 {
                        let w2 = unsafe { (b.as_ptr().add(i) as *const u64).read_unaligned().to_le() };
                        let non_digit2 = non_digit_mask(w2);
                        if non_digit2 != 0 {
                            let nd = (non_digit2.trailing_zeros() >> 3) as usize;
                            if nd > 0 {
                                mant = mant * POW10_U64[nd] + swar_digits_u64(w2, nd);
                                e10 -= nd as i32;
                                i += nd;
                            }
                            break 'frac;
                        }
                    }
                }
                while i < len {
                    let d = b[i].wrapping_sub(b'0');
                    if d >= 10 {
                        break;
                    }
                    if digits < 18 {
                        mant = mant * 10 + d as u64;
                        digits += u32::from(mant != 0);
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
        }

        if i < len && (b[i] == b'e' || b[i] == b'E') {
            let (ev, next_i) = parse_exponent(b, i + 1)?;
            e10 += ev;
            i = next_i;
        }

        let fast = !slow && mant <= (1 << 53) && (-22..=22).contains(&e10);
        let v = if fast {
            let m = mant as f64;
            let a = if e10 < 0 { m / POW10[(-e10) as usize] } else { m * POW10[e10 as usize] };
            if neg { -a } else { a }
        } else {
            parse_number_slow(&b[start..i])?
        };
        if !v.is_finite() {
            return None;
        }
        let f = v as f32;
        Some((f, i, fast || f.is_finite()))
    }

    #[derive(Clone, Copy)]
    struct Indent {
        len: usize,
        mask: u128,
        pat: u128,
    }

    const EMPTY_INDENT: Indent = Indent { len: 0, mask: 0, pat: 0 };

    #[inline]
    fn sniff_indent_at(b: &[u8], nl: usize) -> Indent {
        if b.get(nl) != Some(&b'\n') {
            return EMPTY_INDENT;
        }
        let mut s = nl + 1;
        while s < b.len() && b[s] == b' ' {
            s += 1;
        }
        let total = s - nl;
        if total == 0 || total > 16 || s >= b.len() || is_ws(b[s]) {
            return EMPTY_INDENT;
        }
        let pat_bytes = b"\n                ";
        let mut raw = [0u8; 16];
        raw[..total].copy_from_slice(&pat_bytes[..total]);
        let pat = u128::from_le_bytes(raw);
        let mask = if total == 16 { u128::MAX } else { (1u128 << (total * 8)) - 1 };
        Indent { len: total, mask, pat }
    }

    #[cold]
    #[inline(never)]
    fn skip_ws_cold(b: &[u8], p: usize) -> usize {
        skip_ws(b, p)
    }

    #[inline(always)]
    fn skip_ws_ind_fast<const HAS_IND: bool>(b: &[u8], at: usize, ind: Indent) -> usize {
        if HAS_IND {
            // SAFETY: caller ensures `at + 16 <= b.len()` and `ind.len <= 16`.
            let w = unsafe { (b.as_ptr().add(at) as *const u128).read_unaligned().to_le() };
            let dst = at + ind.len;
            if (w & ind.mask) == ind.pat && unsafe { *b.get_unchecked(dst) } > b' ' {
                return dst;
            }
        } else if unsafe { *b.get_unchecked(at) } > b' ' {
            return at;
        }
        skip_ws_cold(b, at)
    }

    #[inline(always)]
    fn skip_ws_ind(b: &[u8], at: usize, ind: Indent) -> usize {
        if ind.len > 0 && at + 16 <= b.len() {
            let w = unsafe { (b.as_ptr().add(at) as *const u128).read_unaligned().to_le() };
            let dst = at + ind.len;
            if (w & ind.mask) == ind.pat && unsafe { *b.get_unchecked(dst) } > b' ' {
                return dst;
            }
        }
        skip_ws(b, at)
    }

    /// Sniffs `\n` + spaces indentation after the first comma.
    #[inline]
    fn sniff_indent(b: &[u8], start: usize) -> Indent {
        let Some(off) = find_byte(&b[start..b.len().min(start + 128)], b',') else {
            return EMPTY_INDENT;
        };
        sniff_indent_at(b, start + off + 1)
    }

    #[cold]
    #[inline(never)]
    fn advance_after_float_cold(b: &[u8], next: usize) -> Option<Result<usize, bool>> {
        let len = b.len();
        let p = skip_ws(b, next);
        if p == len {
            return Some(Err(true));
        }
        if b[p] != b',' {
            return None;
        }
        let p2 = skip_ws(b, p + 1);
        if p2 == len { Some(Err(false)) } else { Some(Ok(p2)) }
    }

    /// Advances past `,` and following whitespace using the sniffed `\n + spaces` pattern when possible.
    #[inline(always)]
    fn advance_after_float<const HAS_IND: bool>(b: &[u8], next: usize, ind: Indent) -> Option<Result<usize, bool>> {
        let len = b.len();
        if next < len && unsafe { *b.get_unchecked(next) } == b',' {
            let p = if HAS_IND && next + 17 <= len {
                skip_ws_ind_fast::<true>(b, next + 1, ind)
            } else {
                skip_ws_ind(b, next + 1, ind)
            };
            return if p == len { Some(Err(false)) } else { Some(Ok(p)) };
        }
        advance_after_float_cold(b, next)
    }

    struct RawOut<T> {
        vec: Vec<T>,
        ptr: *mut T,
        end: *mut T,
    }

    impl<T> RawOut<T> {
        #[inline(always)]
        fn with_capacity(cap: usize) -> Self {
            let mut vec = Vec::<T>::with_capacity(cap);
            let ptr = vec.as_mut_ptr();
            let end = unsafe { ptr.add(vec.capacity()) };
            Self { vec, ptr, end }
        }

        #[inline(always)]
        fn push(&mut self, v: T) {
            if self.ptr == self.end {
                self.grow();
            }
            unsafe {
                self.ptr.write(v);
                self.ptr = self.ptr.add(1);
            }
        }

        #[cold]
        #[inline(never)]
        fn grow(&mut self) {
            let len = unsafe { self.ptr.offset_from(self.vec.as_ptr()) as usize };
            unsafe { self.vec.set_len(len) };
            self.vec.reserve(4096);
            self.ptr = unsafe { self.vec.as_mut_ptr().add(len) };
            self.end = unsafe { self.vec.as_mut_ptr().add(self.vec.capacity()) };
        }

        #[inline(always)]
        fn finish(mut self) -> Vec<T> {
            let len = unsafe { self.ptr.offset_from(self.vec.as_ptr()) as usize };
            unsafe { self.vec.set_len(len) };
            self.vec
        }
    }

    /// `a, b, c, ...` (flat numbers) into `Vec<f32>` plus `all_finite` flag.
    fn parse_floats_seq(b: &[u8], last: bool) -> Option<(Vec<f32>, bool)> {
        let len = b.len();
        let mut out = RawOut::<f32>::with_capacity(len / 16 + 16);
        let mut all_finite = true;
        let p = skip_ws(b, 0);
        if p == len {
            return (!last).then_some((out.finish(), all_finite));
        }
        let ind = sniff_indent(b, p);
        if ind.len > 0 {
            parse_floats_tail::<true>(b, p, last, ind, &mut out, &mut all_finite)?;
        } else {
            parse_floats_tail::<false>(b, p, last, ind, &mut out, &mut all_finite)?;
        }
        Some((out.finish(), all_finite))
    }

    #[inline(always)]
    fn parse_floats_tail<const HAS_IND: bool>(
        b: &[u8],
        mut p: usize,
        last: bool,
        ind: Indent,
        out: &mut RawOut<f32>,
        all_finite: &mut bool,
    ) -> Option<()> {
        let len = b.len();
        let safe_end = len.saturating_sub(48);
        while p < safe_end {
            let (f, next, finite) = match parse_number_unchecked(b, p) {
                Some(r) => r,
                None => parse_number_bounded(b, p)?,
            };
            *all_finite &= finite;
            out.push(f);
            if next < safe_end && unsafe { *b.get_unchecked(next) } == b',' {
                p = skip_ws_ind_fast::<HAS_IND>(b, next + 1, ind);
                continue;
            }
            match advance_after_float::<HAS_IND>(b, next, ind)? {
                Ok(np) => p = np,
                Err(ended_no_comma) => return (ended_no_comma == last).then_some(()),
            }
        }
        loop {
            let (f, next, finite) = parse_number(b, p)?;
            *all_finite &= finite;
            out.push(f);
            match advance_after_float::<HAS_IND>(b, next, ind)? {
                Ok(np) => p = np,
                Err(ended_no_comma) => return (ended_no_comma == last).then_some(()),
            }
        }
    }

    /// The flat `[x, y, z, x, y, z, ...]` body into vertices (plus `all_finite` flag).
    fn parse_positions(body: &[u8]) -> Option<(Vec<Vec3>, bool)> {
        let starts = split_points(body, |rest| find_byte(rest, b',').map(|c| c + 1));
        let parts = map_pieces(body, &starts, parse_floats_seq)?;
        let total: usize = parts.iter().map(|(a, _)| a.len()).sum();
        if total == 0 || total % 3 != 0 {
            return None;
        }
        let all_finite = parts.iter().all(|&(_, f)| f);
        let mut out: Vec<Vec3> = Vec::with_capacity(total / 3);
        let mut offsets = Vec::with_capacity(parts.len());
        let mut acc = 0usize;
        for (a, _) in &parts {
            offsets.push(acc);
            acc += a.len();
        }
        let dst = SendPtr(out.as_mut_ptr() as *mut f32);
        let copy = |((a, _), &off): (&(Vec<f32>, bool), &usize)| unsafe {
            std::ptr::copy_nonoverlapping(a.as_ptr(), dst.get().add(off), a.len());
        };
        #[cfg(feature = "threads")]
        {
            use rayon::prelude::*;
            parts.par_iter().zip(offsets.par_iter()).for_each(copy);
        }
        #[cfg(not(feature = "threads"))]
        parts.iter().zip(offsets.iter()).for_each(copy);
        unsafe { out.set_len(total / 3) };
        Some((out, all_finite))
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

    #[inline(always)]
    fn parse_uint_fast(b: &[u8], p: usize) -> Option<(u32, usize)> {
        // SAFETY: caller ensures `p + 8 <= b.len()`.
        let w = unsafe { (b.as_ptr().add(p) as *const u64).read_unaligned().to_le() };
        let non_digit = non_digit_mask(w);
        if non_digit == 0 {
            return None;
        }
        let nd = (non_digit.trailing_zeros() >> 3) as usize;
        if nd == 0 {
            return None;
        }
        if (w & 0xff) == b'0' as u64 {
            return (nd == 1).then_some((0, p + 1));
        }
        Some((swar_digits_u64(w, nd) as u32, p + nd))
    }

    #[inline(always)]
    fn parse_uint(b: &[u8], p: usize) -> Option<(u32, usize)> {
        let len = b.len();
        if p >= len {
            return None;
        }
        if p + 8 <= len {
            if let Some(res) = parse_uint_fast(b, p) {
                return Some(res);
            }
        }
        parse_uint_slow(b, p)
    }

    #[cold]
    #[inline(never)]
    fn parse_uint_slow(b: &[u8], mut p: usize) -> Option<(u32, usize)> {
        let len = b.len();
        let d0 = b[p].wrapping_sub(b'0');
        if d0 >= 10 {
            return None;
        }
        let mut v = d0 as u64;
        p += 1;
        if d0 == 0 {
            if p < len && b[p].is_ascii_digit() {
                return None;
            }
            return Some((0, p));
        }
        let mut digits = 1usize;
        while p < len {
            let d = b[p].wrapping_sub(b'0');
            if d >= 10 {
                break;
            }
            digits += 1;
            if digits > 10 {
                return None;
            }
            v = v * 10 + d as u64;
            p += 1;
        }
        (v <= u32::MAX as u64).then_some((v as u32, p))
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

    #[inline(always)]
    fn skip_comma_ws(b: &[u8], p: usize, ind: Indent) -> Option<usize> {
        if p < b.len() && b[p] == b',' {
            return Some(skip_ws_ind(b, p + 1, ind));
        }
        let pw = skip_ws(b, p);
        (b.get(pw) == Some(&b',')).then(|| skip_ws_ind(b, pw + 1, ind))
    }

    #[inline(always)]
    fn sniff_face_indents(b: &[u8], p: usize) -> (Indent, Indent) {
        if b.get(p) != Some(&b'[') {
            return (EMPTY_INDENT, EMPTY_INDENT);
        }
        let in_ind = sniff_indent_at(b, p + 1);
        let out_ind = match find_byte(&b[p..b.len().min(p + 128)], b']') {
            Some(rb) => {
                let mut nl = p + rb;
                while nl > p && b[nl - 1] == b' ' {
                    nl -= 1;
                }
                if nl > p && b[nl - 1] == b'\n' {
                    sniff_indent_at(b, nl - 1)
                } else {
                    EMPTY_INDENT
                }
            }
            None => EMPTY_INDENT,
        };
        (in_ind, out_ind)
    }

    #[cold]
    #[inline(never)]
    fn parse_ngon_face(
        b: &[u8],
        mut p: usize,
        out: &mut RawOut<Vec<u32>>,
        tmp: &mut Vec<u32>,
        max_idx: &mut u32,
    ) -> Result<Option<(usize, bool)>, ()> {
        let len = b.len();
        tmp.clear();
        if b.get(p) == Some(&b']') {
            p += 1;
        } else {
            loop {
                let (u, np) = parse_uint(b, p).ok_or(())?;
                tmp.push(u);
                p = skip_ws(b, np);
                match *b.get(p).ok_or(())? {
                    b',' => p = skip_ws(b, p + 1),
                    b']' => {
                        p += 1;
                        break;
                    }
                    _ => return Err(()),
                }
            }
        }
        if tmp.len() >= 3 && face_unique(tmp) {
            for &u in tmp.iter() {
                *max_idx = (*max_idx).max(u);
            }
            out.push(tmp.clone());
        }
        p = skip_ws(b, p);
        if p == len {
            return Ok(None);
        }
        if b[p] != b',' {
            return Err(());
        }
        let np = skip_ws(b, p + 1);
        Ok(if np == len { Some((np, true)) } else { Some((np, false)) })
    }

    #[inline(always)]
    fn parse_one_face_unchecked<const HAS_IND: bool>(
        b: &[u8],
        mut p: usize,
        in_ind: Indent,
        out_ind: Indent,
        out: &mut RawOut<Vec<u32>>,
        tmp: &mut Vec<u32>,
        max_idx: &mut u32,
    ) -> Result<Option<(usize, bool)>, ()> {
        // SAFETY: caller ensures `p + 192 <= b.len()`.
        if unsafe { *b.get_unchecked(p) } != b'[' {
            return Err(());
        }
        p = skip_ws_ind_fast::<HAS_IND>(b, p + 1, in_ind);
        'fast_unchecked: {
            let Some((u0, p0)) = parse_uint_fast(b, p) else { break 'fast_unchecked };
            if unsafe { *b.get_unchecked(p0) } != b',' { break 'fast_unchecked; }
            let p1s = skip_ws_ind_fast::<HAS_IND>(b, p0 + 1, in_ind);
            let Some((u1, p1)) = parse_uint_fast(b, p1s) else { break 'fast_unchecked };
            if unsafe { *b.get_unchecked(p1) } != b',' { break 'fast_unchecked; }
            let p2s = skip_ws_ind_fast::<HAS_IND>(b, p1 + 1, in_ind);
            let Some((u2, p2)) = parse_uint_fast(b, p2s) else { break 'fast_unchecked };
            let p2w = if unsafe { *b.get_unchecked(p2) } == b',' {
                p2
            } else {
                skip_ws_ind_fast::<HAS_IND>(b, p2, out_ind)
            };
            match unsafe { *b.get_unchecked(p2w) } {
                b']' => {
                    p = p2w + 1;
                    if u0 != u1 && u1 != u2 && u0 != u2 {
                        *max_idx = (*max_idx).max(u0).max(u1).max(u2);
                        out.push(vec![u0, u1, u2]);
                    }
                }
                b',' => {
                    let p3s = skip_ws_ind_fast::<HAS_IND>(b, p2w + 1, in_ind);
                    let Some((u3, p3)) = parse_uint_fast(b, p3s) else { break 'fast_unchecked };
                    let p3w = skip_ws_ind_fast::<HAS_IND>(b, p3, out_ind);
                    if unsafe { *b.get_unchecked(p3w) } != b']' {
                        break 'fast_unchecked;
                    }
                    p = p3w + 1;
                    if u0 != u1 && u0 != u2 && u0 != u3 && u1 != u2 && u1 != u3 && u2 != u3 {
                        *max_idx = (*max_idx).max(u0).max(u1).max(u2).max(u3);
                        out.push(vec![u0, u1, u2, u3]);
                    }
                }
                _ => break 'fast_unchecked,
            }
            if unsafe { *b.get_unchecked(p) } == b',' {
                let np = skip_ws_ind_fast::<HAS_IND>(b, p + 1, out_ind);
                return Ok(Some((np, false)));
            }
            let len = b.len();
            let pw = skip_ws_cold(b, p);
            if pw == len {
                return Ok(None);
            }
            if b[pw] != b',' {
                return Err(());
            }
            let np = skip_ws_cold(b, pw + 1);
            return Ok(if np == len { Some((np, true)) } else { Some((np, false)) });
        }
        parse_ngon_face(b, p, out, tmp, max_idx)
    }

    #[cold]
    #[inline(never)]
    fn parse_one_face_bounded(
        b: &[u8],
        mut p: usize,
        in_ind: Indent,
        out_ind: Indent,
        out: &mut RawOut<Vec<u32>>,
        tmp: &mut Vec<u32>,
        max_idx: &mut u32,
    ) -> Result<Option<(usize, bool)>, ()> {
        let len = b.len();
        if b.get(p) != Some(&b'[') {
            return Err(());
        }
        p = skip_ws_ind(b, p + 1, in_ind);
        'fast_face: {
            let Some((u0, p0)) = parse_uint(b, p) else { break 'fast_face };
            let Some(p1s) = skip_comma_ws(b, p0, in_ind) else { break 'fast_face };
            let Some((u1, p1)) = parse_uint(b, p1s) else { break 'fast_face };
            let Some(p2s) = skip_comma_ws(b, p1, in_ind) else { break 'fast_face };
            let Some((u2, p2)) = parse_uint(b, p2s) else { break 'fast_face };
            let p2w = skip_ws_ind(b, p2, out_ind);
            match b.get(p2w) {
                Some(b']') => {
                    p = p2w + 1;
                    if u0 != u1 && u1 != u2 && u0 != u2 {
                        *max_idx = (*max_idx).max(u0).max(u1).max(u2);
                        out.push(vec![u0, u1, u2]);
                    }
                }
                Some(b',') => {
                    let p3s = skip_ws_ind(b, p2w + 1, in_ind);
                    let Some((u3, p3)) = parse_uint(b, p3s) else { break 'fast_face };
                    let p3w = skip_ws_ind(b, p3, out_ind);
                    if b.get(p3w) != Some(&b']') {
                        break 'fast_face;
                    }
                    p = p3w + 1;
                    if u0 != u1 && u0 != u2 && u0 != u3 && u1 != u2 && u1 != u3 && u2 != u3 {
                        *max_idx = (*max_idx).max(u0).max(u1).max(u2).max(u3);
                        out.push(vec![u0, u1, u2, u3]);
                    }
                }
                _ => break 'fast_face,
            }
            if p < len && b[p] == b',' {
                let np = skip_ws_ind(b, p + 1, out_ind);
                return Ok(if np == len { Some((np, true)) } else { Some((np, false)) });
            }
            p = skip_ws(b, p);
            if p == len {
                return Ok(None);
            }
            if b[p] != b',' {
                return Err(());
            }
            let np = skip_ws_ind(b, p + 1, out_ind);
            return Ok(if np == len { Some((np, true)) } else { Some((np, false)) });
        }

        parse_ngon_face(b, p, out, tmp, max_idx)
    }

    fn parse_face_piece_inner<const HAS_IND: bool>(
        b: &[u8],
        mut p: usize,
        last: bool,
        in_ind: Indent,
        out_ind: Indent,
        out: &mut RawOut<Vec<u32>>,
        tmp: &mut Vec<u32>,
        max_idx: &mut u32,
    ) -> Option<()> {
        let safe_end = b.len().saturating_sub(192);
        while p < safe_end {
            match parse_one_face_unchecked::<HAS_IND>(b, p, in_ind, out_ind, out, tmp, max_idx).ok()? {
                None => return last.then_some(()),
                Some((_, true)) => return (!last).then_some(()),
                Some((np, false)) => p = np,
            }
        }
        loop {
            match parse_one_face_bounded(b, p, in_ind, out_ind, out, tmp, max_idx).ok()? {
                None => return last.then_some(()),
                Some((_, true)) => return (!last).then_some(()),
                Some((np, false)) => p = np,
            }
        }
    }

    /// `[a, b, c], [d, e, f], ...`: keeps 3+ distinct-index faces and returns `(faces, max_index)`.
    fn parse_face_piece(b: &[u8], last: bool) -> Option<(Vec<Vec<u32>>, u32)> {
        let len = b.len();
        let mut out = RawOut::<Vec<u32>>::with_capacity(len / 72 + 1);
        let mut tmp: Vec<u32> = Vec::with_capacity(8);
        let mut max_idx = 0u32;
        let p = skip_ws(b, 0);
        if p == len {
            return (!last).then_some((out.finish(), max_idx));
        }
        let (in_ind, out_ind) = sniff_face_indents(b, p);
        if in_ind.len > 0 && out_ind.len > 0 {
            parse_face_piece_inner::<true>(b, p, last, in_ind, out_ind, &mut out, &mut tmp, &mut max_idx)?;
        } else {
            parse_face_piece_inner::<false>(b, p, last, in_ind, out_ind, &mut out, &mut tmp, &mut max_idx)?;
        }
        Some((out.finish(), max_idx))
    }

    fn parse_faces_unchecked(body: &[u8]) -> Option<(Vec<Vec<u32>>, u32)> {
        let starts = split_points(body, |rest| find_byte(rest, b'['));
        let mut parts = map_pieces(body, &starts, parse_face_piece)?;
        let total: usize = parts.iter().map(|(a, _)| a.len()).sum();
        let max_idx = parts.iter().map(|&(_, m)| m).max().unwrap_or(0);
        let mut out: Vec<Vec<u32>> = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(parts.len());
        let mut acc = 0usize;
        for (a, _) in &parts {
            offsets.push(acc);
            acc += a.len();
        }
        let dst = SendPtr(out.as_mut_ptr());
        let mv = |((a, _), &off): (&mut (Vec<Vec<u32>>, u32), &usize)| unsafe {
            let alen = a.len();
            std::ptr::copy_nonoverlapping(a.as_ptr(), dst.get().add(off), alen);
            a.set_len(0);
        };
        #[cfg(feature = "threads")]
        {
            use rayon::prelude::*;
            parts.par_iter_mut().zip(offsets.par_iter()).for_each(mv);
        }
        #[cfg(not(feature = "threads"))]
        parts.iter_mut().zip(offsets.iter()).for_each(mv);
        unsafe { out.set_len(total) };
        Some((out, max_idx))
    }
}

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

/// Streams `positions` directly into `Vec<Vec3>` (accepts both flat `[x, y, z, ...]` and nested `[[x, y, z], ...]`).
struct PositionsDe(Vec<Vec3>);
impl<'de> Deserialize<'de> for PositionsDe {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<PositionsDe, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = PositionsDe;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a sequence of numbers or [x, y, z] triples")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut s: A) -> Result<PositionsDe, A::Error> {
                let mut out = Vec::with_capacity(s.size_hint().unwrap_or(0));
                let Some(first) = s.next_element::<PosElem>()? else {
                    return Ok(PositionsDe(out));
                };
                match first {
                    PosElem::Scalar(x) => {
                        let y = s.next_element::<Fl>()?.ok_or_else(|| de::Error::custom("odd position count"))?.0;
                        let z = s.next_element::<Fl>()?.ok_or_else(|| de::Error::custom("odd position count"))?.0;
                        out.push(Vec3::new(x, y, z));
                        while let Some(x) = s.next_element::<Fl>()? {
                            let y = s.next_element::<Fl>()?.ok_or_else(|| de::Error::custom("odd position count"))?.0;
                            let z = s.next_element::<Fl>()?.ok_or_else(|| de::Error::custom("odd position count"))?.0;
                            out.push(Vec3::new(x.0, y, z));
                        }
                    }
                    PosElem::Triple(t) => {
                        out.push(Vec3::from_array(t));
                        while let Some(t) = s.next_element::<[Fl; 3]>()? {
                            out.push(Vec3::new(t[0].0, t[1].0, t[2].0));
                        }
                    }
                }
                Ok(PositionsDe(out))
            }
        }
        d.deserialize_seq(V)
    }
}

/// One element of `positions`: used only to sniff flat vs nested on element 0.
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
    format_version: i64,
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

/// `positions` / `faces` are only absent in a stub document, where `fast` replaced a big array by an
/// index (`"@p": k` / `"@f": k`); `mesh` rejects a mesh that has neither the array nor its marker.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MeshDe {
    #[serde(default)]
    positions: Option<PositionsDe>,
    #[serde(default)]
    faces: Option<Vec<FaceDe>>,
    smooth_shading: Option<bool>,
    #[serde(default, rename = "@p")]
    ext_p: Option<u32>,
    #[serde(default, rename = "@f")]
    ext_f: Option<u32>,
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

/// Documents at least this big try the parallel mesh array reader first.
const FAST_MIN: usize = 256 * 1024;

pub(super) fn decode(bytes: &[u8]) -> Result<ProjectFile, FormatError> {
    decode_with(bytes, FAST_MIN)
}

/// `fast_min`: smallest document the array fast path is tried on (tests use 0). The fast path only ever
/// returns a project it fully understood; every doubt falls back to the serde path, which owns all errors.
fn decode_with(bytes: &[u8], fast_min: usize) -> Result<ProjectFile, FormatError> {
    if bytes.len() >= fast_min {
        if let Some(p) = fast::decode(bytes) {
            return Ok(p);
        }
    }
    decode_serde(bytes)
}

fn decode_serde(bytes: &[u8]) -> Result<ProjectFile, FormatError> {
    // Single-pass parse for valid projects; fall back to header check only on error so newer/older
    // format versions still report the exact version error even if their schema changed.
    let p: ProjectDe = match serde_json::from_slice::<ProjectDe>(bytes) {
        Ok(p) => {
            if p.format_version > CURRENT_FORMAT_VERSION {
                return Err(FormatError::NewerVersion(p.format_version));
            }
            if p.format_version != CURRENT_FORMAT_VERSION {
                return Err(FormatError::UnsupportedVersion(p.format_version));
            }
            p
        }
        Err(_) => {
            let header: Header = serde_json::from_slice(bytes).map_err(|_| FormatError::NotAProject)?;
            if header.format_version > CURRENT_FORMAT_VERSION {
                return Err(FormatError::NewerVersion(header.format_version));
            }
            if header.format_version != CURRENT_FORMAT_VERSION {
                return Err(FormatError::UnsupportedVersion(header.format_version));
            }
            return Err(FormatError::NotAProject);
        }
    };
    convert(p, &mut fast::NoExt)
}

fn convert(p: ProjectDe, ext: &mut impl fast::Ext) -> Result<ProjectFile, FormatError> {
    let mut objects = Vec::with_capacity(p.scene.objects.len());
    for o in p.scene.objects {
        objects.push(SceneObject {
            id: uuid(o.id)?,
            name: o.name,
            mesh: mesh(o.mesh, &mut *ext)?,
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

#[inline]
fn face_indices_unique(f: &[i64]) -> bool {
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

/// Swift's `Mesh(from:)` plus `ProjectFile.repaired(_ mesh:)`.
fn mesh(m: MeshDe, ext: &mut impl fast::Ext) -> Result<Mesh, FormatError> {
    let (mut positions, all_finite) = match (m.positions, m.ext_p) {
        (Some(p), None) => (p.0, false),
        (None, Some(k)) => ext.positions(k).ok_or(FormatError::NotAProject)?,
        _ => return Err(FormatError::NotAProject),
    };
    if !all_finite {
        for p in &mut positions {
            if !p.is_finite() {
                *p = Vec3::ZERO;
            }
        }
    }
    let n = positions.len();
    let faces = match (m.faces, m.ext_f) {
        (Some(fs), None) => {
            let n = n as i64;
            let mut faces = Vec::with_capacity(fs.len());
            for f in fs {
                let idx = &f.0;
                if idx.len() >= 3 && idx.iter().all(|&i| i >= 0 && i < n) && face_indices_unique(idx) {
                    faces.push(idx.iter().map(|&i| i as u32).collect());
                }
            }
            faces
        }
        (None, Some(k)) => ext.faces(k, n).ok_or(FormatError::NotAProject)?,
        _ => return Err(FormatError::NotAProject),
    };
    Ok(Mesh { positions, faces, smooth_shading: m.smooth_shading.unwrap_or(false) })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN: &str = include_str!("../../tests/fixtures/project.sculpt");

    fn sphere_project(objs: &[(u32, u32)]) -> ProjectFile {
        let mut p = ProjectFile::decode(GOLDEN.as_bytes()).unwrap();
        let base = p.scene.objects[0].clone();
        p.scene.objects = objs
            .iter()
            .enumerate()
            .map(|(i, &(seg, rings))| {
                let mut o = base.clone();
                o.id = format!("0A1B2C3D-0000-4000-8000-{:012X}", i + 1);
                o.name = format!("ob\\\"positions\" : [1,2,3] {i}"); // strings never act as keys
                o.parent = None;
                o.mesh = crate::mesh::uv_sphere(3.0 + i as f32, seg, rings);
                for (k, v) in o.mesh.positions.iter_mut().enumerate() {
                    *v *= 1.0 + (k % 13) as f32 * 1.7e-3 - 0.01; // varied digits, some negative zero-ish values
                }
                o
            })
            .collect();
        p.selection = Selection::default();
        p
    }

    /// The array fast path (forced on) must give exactly what the serde path gives.
    fn same(bytes: &[u8]) {
        let slow = decode_with(bytes, usize::MAX);
        let fast = decode_with(bytes, 0);
        assert_eq!(fast, slow);
    }

    #[test]
    fn fast_path_matches_serde_path() {
        same(GOLDEN.as_bytes());
        // Mixed small and big meshes, pretty and compact text.
        let p = sphere_project(&[(4, 3), (150, 90), (6, 4), (120, 80)]);
        let text = p.encode();
        let fast = fast::decode(text.as_bytes()).expect("big arrays take the fast path");
        assert_eq!(fast, p);
        same(text.as_bytes());
        let compact: String = {
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            serde_json::to_string(&v).unwrap()
        };
        same(compact.as_bytes());
        // Repairs: out of range, repeated and short faces, non finite numbers.
        let mut dirty = sphere_project(&[(150, 90)]);
        let n = dirty.scene.objects[0].mesh.positions.len() as u32;
        dirty.scene.objects[0].mesh.faces.splice(5..5, [vec![0, 1, n + 7], vec![2, 2, 3], vec![4, 5], vec![6, 7, 8, 8]]);
        let dirty_text = dirty.encode();
        let fast = fast::decode(dirty_text.as_bytes()).expect("fast");
        assert_eq!(fast.scene.objects[0].mesh.faces.len(), dirty.scene.objects[0].mesh.faces.len() - 4);
        same(dirty_text.as_bytes());
    }

    #[test]
    fn fast_path_declines_what_it_does_not_understand() {
        let text = sphere_project(&[(150, 90)]).encode();
        // Offsets of the first face index and of the first position number.
        let faces_key = text.find("\"faces\" : [").unwrap();
        let first_idx = faces_key + text[faces_key..].find("[\n").unwrap() + 2; // after the outer `[`
        let first_idx = first_idx + text[first_idx..].find("[\n").unwrap() + 2;
        let first_idx = first_idx + text[first_idx..].find(|c: char| c.is_ascii_digit()).unwrap();
        let pos_key = text.find("\"positions\" : [").unwrap();
        let first_num = pos_key + text[pos_key..].find(|c: char| c == '-' || c.is_ascii_digit()).unwrap();
        let insert = |at: usize, s: &str| format!("{}{}{}", &text[..at], s, &text[at..]);
        let cases: Vec<(&str, String, bool)> = vec![
            ("nan string", insert(first_num, "\"nan\", "), true),
            ("missing comma", insert(first_num, "1 "), true),
            ("negative index", insert(first_idx, "-"), true),
            ("leading zero", insert(first_idx, "0"), true),
            ("float index", insert(first_idx + 1, ".0"), true),
            ("double comma", insert(first_num, ", "), true),
            ("truncated", text[..text.len() / 2].to_string(), true),
            ("extra marker", text.replacen("\"smoothShading\"", "\"@f\" : 0, \"smoothShading\"", 1), true),
            ("odd float count", insert(first_num, "5, "), true),
            ("valid control", text.clone(), false),
        ];
        for (name, doc, declines) in &cases {
            let slow = decode_with(doc.as_bytes(), usize::MAX);
            assert_eq!(decode_with(doc.as_bytes(), 0), slow, "{name}");
            assert_eq!(fast::decode(doc.as_bytes()).is_none(), *declines, "{name}");
        }
    }
}
