//! Triangle BVH: Morton (LBVH) build for big meshes, binned SAH for small ones, refit, closest ray hit and sphere (brush footprint) queries.
//!
//! Between refits, moved vertices may leave their node boxes. Instead of refitting after every dab, the
//! tree tracks `slack`: the largest distance any vertex has moved since the last refit. Queries grow
//! every box by `slack`, so they stay exact; `refit` runs once the slack is large compared to the brush.

use glam::{Vec3, Vec3A, Vec4};

/// 32 bytes, laid out for upload to the GPU path tracer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Node {
    pub min: [f32; 3],
    /// Interior: index of the left child (the right one follows). Leaf: first entry in `tri_ids`.
    pub left_first: u32,
    pub max: [f32; 3],
    /// Triangle count; 0 for interior nodes.
    pub count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub t: f32,
    pub tri: u32,
    pub point: Vec3,
}

#[derive(Debug, Clone, Default)]
pub struct Bvh {
    pub nodes: Vec<Node>,
    pub tri_ids: Vec<u32>,
    pub slack: f32,
}

const LEAF: usize = 4;
const BINS: usize = 12;

#[derive(Clone, Copy)]
struct Aabb {
    min: Vec3A,
    max: Vec3A,
}
impl Aabb {
    const EMPTY: Aabb = Aabb { min: Vec3A::splat(f32::INFINITY), max: Vec3A::splat(f32::NEG_INFINITY) };
    #[inline(always)]
    fn grow(&mut self, p: Vec3A) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
    #[inline(always)]
    fn union(&mut self, b: &Aabb) {
        self.min = self.min.min(b.min);
        self.max = self.max.max(b.max);
    }
    #[inline(always)]
    fn area(&self) -> f32 {
        let d = (self.max - self.min).max(Vec3A::ZERO);
        d.x * d.y + d.y * d.z + d.z * d.x
    }
}

/// Per triangle build record, 48 bytes: box, centroid and (in the unused centroid lane) the triangle id.
#[derive(Clone, Copy)]
struct TriInfo {
    bb: Aabb,
    cent: Vec4,
}
impl TriInfo {
    #[inline(always)]
    fn new(bb: Aabb, id: u32) -> TriInfo {
        let c = (bb.min + bb.max) * 0.5;
        TriInfo { bb, cent: Vec4::new(c.x, c.y, c.z, f32::from_bits(id)) }
    }
    #[inline(always)]
    fn id(&self) -> u32 {
        self.cent.w.to_bits()
    }
}

#[inline(always)]
fn tri_box(p: &[Vec3], t: &[u32; 3]) -> Aabb {
    let a = Vec3A::from(p[t[0] as usize]);
    let b = Vec3A::from(p[t[1] as usize]);
    let c = Vec3A::from(p[t[2] as usize]);
    Aabb { min: a.min(b).min(c), max: a.max(b).max(c) }
}

/// Box of a node's triangle boxes plus box of their centroids, handed down from the parent's partition pass.
#[derive(Clone, Copy)]
struct Bounds {
    bb: Aabb,
    cb: Aabb,
}
impl Bounds {
    const EMPTY: Bounds = Bounds { bb: Aabb::EMPTY, cb: Aabb::EMPTY };
    #[inline(always)]
    fn add(&mut self, ti: &TriInfo) {
        self.bb.union(&ti.bb);
        self.cb.grow(Vec3A::from_vec4(ti.cent));
    }
    fn of_recs(recs: &[TriInfo]) -> Bounds {
        let mut b = Bounds::EMPTY;
        for r in recs {
            b.add(r);
        }
        b
    }
}

/// Where a node splits: centroid axis, last bin that goes left and the binning transform.
#[derive(Clone, Copy)]
struct Plan {
    axis: usize,
    best: usize,
    min_c: f32,
    scale: f32,
}

type Bins = [(Aabb, u32); BINS];

fn bin_recs(recs: &[TriInfo], axis: usize, min_c: f32, scale: f32) -> Bins {
    let mut bins = [(Aabb::EMPTY, 0u32); BINS];
    for ti in recs {
        let b = &mut bins[(((ti.cent[axis] - min_c) * scale) as usize).min(BINS - 1)];
        b.0.union(&ti.bb);
        b.1 += 1;
    }
    bins
}

/// Binned SAH split of a node whose bounds are known; `None` makes it a leaf. `bin` fills the bins for the
/// chosen axis (over ids or over records).
fn plan_split(count: usize, b: &Bounds, bin: impl FnOnce(usize, f32, f32) -> Bins) -> Option<Plan> {
    if count <= LEAF {
        return None;
    }
    let ext = b.cb.max - b.cb.min;
    let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
    if ext[axis] <= 0.0 {
        return None; // all centroids coincide: keep a (rare) fat leaf
    }
    let min_c = b.cb.min[axis];
    let scale = BINS as f32 / ext[axis];
    let bins = bin(axis, min_c, scale);
    let mut right_area = [0f32; BINS];
    let mut acc = (Aabb::EMPTY, 0u32);
    for i in (1..BINS).rev() {
        acc.0.union(&bins[i].0);
        acc.1 += bins[i].1;
        right_area[i] = acc.0.area() * acc.1 as f32;
    }
    let mut best = (f32::INFINITY, 0);
    let mut left = (Aabb::EMPTY, 0u32);
    for i in 0..BINS - 1 {
        left.0.union(&bins[i].0);
        left.1 += bins[i].1;
        let cost = left.0.area() * left.1 as f32 + right_area[i + 1];
        if left.1 > 0 && (left.1 as usize) < count && cost < best.0 {
            best = (cost, i);
        }
    }
    if best.0 >= b.bb.area() * count as f32 && count <= 16 {
        return None;
    }
    Some(Plan { axis, best: best.1, min_c, scale })
}

/// In place record partition (one streaming pass) that also accumulates both children's bounds.
fn partition_recs(recs: &mut [TriInfo], p: &Plan) -> Option<(usize, Bounds, Bounds)> {
    let count = recs.len();
    let (mut i, mut j) = (0usize, count);
    let (mut lb, mut rb) = (Bounds::EMPTY, Bounds::EMPTY);
    while i < j {
        let r = recs[i];
        let bin = (((r.cent[p.axis] - p.min_c) * p.scale) as usize).min(BINS - 1);
        if bin <= p.best {
            lb.add(&r);
            i += 1;
        } else {
            rb.add(&r);
            j -= 1;
            recs.swap(i, j);
        }
    }
    (i != 0 && i != count).then_some((i, lb, rb))
}

/// Builds the subtree over `recs` (reordering them) with leaves pointing at `first_base + position`.
/// Records stay contiguous per node, so every pass streams through memory instead of gathering.
fn build_seq(recs: &mut [TriInfo], first_base: u32, root: Bounds) -> Vec<Node> {
    let n = recs.len();
    let mut nodes = Vec::with_capacity(2 * n / LEAF + 1);
    nodes.push(Node { left_first: 0, count: n as u32, ..Default::default() });
    let mut stack = Vec::with_capacity(64);
    stack.push((0usize, 0usize, n, root));
    while let Some((ni, rel_first, count, b)) = stack.pop() {
        nodes[ni].min = b.bb.min.to_array();
        nodes[ni].max = b.bb.max.to_array();
        let slice = &mut recs[rel_first..rel_first + count];
        let split = plan_split(count, &b, |axis, min_c, scale| bin_recs(slice, axis, min_c, scale)).and_then(|p| partition_recs(slice, &p));
        let Some((i, lb, rb)) = split else {
            nodes[ni].left_first = first_base + rel_first as u32;
            nodes[ni].count = count as u32;
            continue;
        };
        let l = nodes.len();
        nodes.push(Node::default());
        nodes.push(Node::default());
        nodes[ni].left_first = l as u32;
        nodes[ni].count = 0;
        stack.push((l + 1, rel_first + i, count - i, rb));
        stack.push((l, rel_first, i, lb));
    }
    nodes
}

// ---- Morton (LBVH) build for big meshes -------------------------------------------------------------------
//
// 1. bounds of the positions, 2. Morton keys + per chunk histograms of their top bits, 3. one counting-sort
// scatter by those top bits, 4. the top of the tree is split on bucket boundaries alone (from the bucket starts,
// no data pass), 5. each subtree task radix-sorts its own (cache resident) key range, builds a binary radix tree
// (split at the highest differing key bit) with leaf boxes from the positions, and improves it with tree
// rotations on the way up, 6. every task lays its nodes out depth first behind the top nodes.
// Keys are `morton << (64 - 3 * bits) | tri_id`: equal codes still split (on the id bits) and the low bits give
// the triangle back.

/// Below this many triangles the (sequential) binned SAH build is used: cheap at that size and the best tree.
const LBVH_MIN: usize = 4096;

/// Raw pointer that may cross threads; every user writes a disjoint range.
struct SyncPtr<T>(*mut T);
impl<T> Clone for SyncPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SyncPtr<T> {}
// SAFETY: only used to write disjoint index ranges from different tasks (see each use).
unsafe impl<T> Send for SyncPtr<T> {}
// SAFETY: as above.
unsafe impl<T> Sync for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    /// A method (not field access) so closures capture the whole `Send + Sync` wrapper.
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

#[cfg(feature = "threads")]
fn par_map<R: Send>(n: usize, f: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    use rayon::prelude::*;
    (0..n).into_par_iter().map(f).collect()
}
#[cfg(not(feature = "threads"))]
fn par_map<R: Send>(n: usize, f: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    (0..n).map(f).collect()
}

fn sort_keys(keys: &mut [u64]) {
    #[cfg(feature = "threads")]
    if keys.len() >= 1 << 15 {
        use rayon::prelude::*;
        keys.par_sort_unstable();
        return;
    }
    keys.sort_unstable();
}

/// Stable LSD radix sort of `keys` on bits `lo_bit..hi_bit` (digits of at most 11 bits, fewer for short arrays).
fn radix_sort(keys: &mut [u64], lo_bit: u32, hi_bit: u32) {
    let m = keys.len();
    if m < 2 || hi_bit <= lo_bit {
        return;
    }
    let bits = hi_bit - lo_bit;
    let max_digit = (usize::BITS - m.leading_zeros()).saturating_sub(1).clamp(6, 11);
    let passes = bits.div_ceil(max_digit);
    let digit = bits.div_ceil(passes);
    let nb = 1usize << digit;
    let mask = (nb - 1) as u64;
    let mut hist = vec![0u32; passes as usize * nb];
    for &k in keys.iter() {
        let k = k >> lo_bit;
        for p in 0..passes {
            hist[p as usize * nb + ((k >> (p * digit)) & mask) as usize] += 1;
        }
    }
    let mut tmp = vec![0u64; m];
    let mut in_tmp = false;
    for p in 0..passes {
        let h = &mut hist[p as usize * nb..(p as usize + 1) * nb];
        let shift = lo_bit + p * digit;
        // A digit every key shares changes nothing.
        let k0 = if in_tmp { tmp[0] } else { keys[0] };
        if h[((k0 >> shift) & mask) as usize] as usize == m {
            continue;
        }
        let mut run = 0u32;
        for c in h.iter_mut() {
            let v = *c;
            *c = run;
            run += v;
        }
        let (src, dst): (&[u64], &mut [u64]) = if in_tmp { (&tmp, keys) } else { (keys, &mut tmp) };
        for &k in src {
            let o = &mut h[((k >> shift) & mask) as usize];
            dst[*o as usize] = k;
            *o += 1;
        }
        in_tmp = !in_tmp;
    }
    if in_tmp {
        keys.copy_from_slice(&tmp);
    }
}

/// Spreads the low 21 bits of `v` to every third bit.
#[inline(always)]
fn spread3(v: u32) -> u64 {
    let mut x = (v & 0x1f_ffff) as u64;
    x = (x | x << 32) & 0x001f_0000_0000_ffff;
    x = (x | x << 16) & 0x001f_0000_ff00_00ff;
    x = (x | x << 8) & 0x100f_00f0_0f00_f00f;
    x = (x | x << 4) & 0x10c3_0c30_c30c_30c3;
    x = (x | x << 2) & 0x1249_2492_4924_9249;
    x
}

/// First index in `keys` (non-empty, sorted, sharing every bit above `d`) whose bit `d` is set.
#[inline(always)]
fn split_at_bit(keys: &[u64], d: u32) -> usize {
    keys.partition_point(|&k| (k >> d) & 1 == 0)
}

/// A subtree built by one task: its root goes into `slot` of the top nodes, it covers keys `lo..hi`, which still
/// need sorting on bits `code_shift..sort_hi` (none when `sort_hi == 0`). A `big` task is a single bucket over
/// the task size, which builds its subtree in parallel itself.
struct Task {
    slot: u32,
    lo: u32,
    hi: u32,
    sort_hi: u32,
    big: bool,
}

/// Splits the top of the tree (ranges over `task_max` triangles) into `top`, collecting the subtree tasks.
struct TopSplit<'a> {
    keys: &'a [u64],
    /// `starts[b]` = first key of bucket `b`; one extra entry = n.
    starts: &'a [u32],
    top_shift: u32,
    task_max: usize,
    top: Vec<Node>,
    tasks: Vec<Task>,
}
impl TopSplit<'_> {
    /// Bucket holding key position `i`.
    #[inline(always)]
    fn bucket(&self, i: usize) -> usize {
        self.starts[1..].partition_point(|&e| e as usize <= i)
    }
    /// Unsorted ranges always start and end on bucket boundaries (`sorted` ones split on their keys); a single
    /// bucket that is too big for one task becomes a `big` task.
    fn split(&mut self, ni: usize, lo: usize, hi: usize, sorted: bool) {
        let (bf, bl) = if sorted { (0, 0) } else { (self.bucket(lo), self.bucket(hi - 1)) };
        let big = !sorted && bf == bl && hi - lo > self.task_max;
        if hi - lo <= self.task_max || big {
            let sort_hi = if sorted { 0 } else { self.top_shift + (usize::BITS - (bf ^ bl).leading_zeros()) };
            self.tasks.push(Task { slot: ni as u32, lo: lo as u32, hi: hi as u32, sort_hi, big });
            return;
        }
        let mid = if sorted {
            let d = 63 - (self.keys[lo] ^ self.keys[hi - 1]).leading_zeros();
            lo + split_at_bit(&self.keys[lo..hi], d)
        } else {
            let d = usize::BITS - 1 - (bf ^ bl).leading_zeros();
            self.starts[(bl >> d) << d] as usize
        };
        let l = self.top.len();
        self.top.push(Node::default());
        self.top.push(Node::default());
        self.top[ni].left_first = l as u32;
        self.split(l, lo, mid, sorted);
        self.split(l + 1, mid, hi, sorted);
    }
}

#[inline(always)]
fn half_area(min: Vec3A, max: Vec3A) -> f32 {
    let d = max - min;
    d.x * d.y + d.y * d.z + d.z * d.x
}

/// Tree rotation (Kensler 2008) at interior node `ni`: swaps a child with a grandchild under the other child, or
/// two grandchildren under different children, when that shrinks the children's boxes (the only boxes, so the
/// only SAH terms, that change). Returns the child that got new children, if any. Moved subtrees keep their
/// nodes, so children may end up before their parent here; the depth first layout restores the order.
#[inline]
fn rotate(nodes: &mut [Node], ni: usize) -> Option<usize> {
    let l = nodes[ni].left_first as usize;
    let (ln, rn) = (nodes[l], nodes[l + 1]);
    let (lmn, lmx) = node_box(&ln);
    let (rmn, rmx) = node_box(&rn);
    let (la, ra) = (half_area(lmn, lmx), half_area(rmn, rmx));
    // (gain, first node to swap, second node to swap)
    let mut best = (0.0f32, 0usize, 0usize);
    let grand = |n: &Node| {
        let g = n.left_first as usize;
        (g, node_box(&nodes[g]), node_box(&nodes[g + 1]))
    };
    let (lg, rg) = ((ln.count == 0).then(|| grand(&ln)), (rn.count == 0).then(|| grand(&rn)));
    // A child swapped with a grandchild under the other child.
    for (c_area, g, s) in [(ra, rg, (l, lmn, lmx)), (la, lg, (l + 1, rmn, rmx))] {
        let Some((g, a, b)) = g else { continue };
        for (gi, keep) in [(g, b), (g + 1, a)] {
            let gain = c_area - half_area(s.1.min(keep.0), s.2.max(keep.1));
            if gain > best.0 {
                best = (gain, gi, s.0);
            }
        }
    }
    // A grandchild under each child swapped with each other.
    if let (Some((lg, la0, la1)), Some((rg, ra0, ra1))) = (lg, rg) {
        for (x, xs, xk) in [(lg, la0, la1), (lg + 1, la1, la0)] {
            for (y, ys, yk) in [(rg, ra0, ra1), (rg + 1, ra1, ra0)] {
                let gain = la + ra - half_area(xk.0.min(ys.0), xk.1.max(ys.1)) - half_area(yk.0.min(xs.0), yk.1.max(xs.1));
                if gain > best.0 {
                    best = (gain, x, y);
                }
            }
        }
    }
    if best.0 <= 0.0 {
        return None;
    }
    nodes.swap(best.1, best.2);
    for c in [l, l + 1] {
        if nodes[c].count == 0 {
            let gl = nodes[c].left_first as usize;
            let (amn, amx) = node_box(&nodes[gl]);
            let (bmn, bmx) = node_box(&nodes[gl + 1]);
            nodes[c].min = amn.min(bmn).to_array();
            nodes[c].max = amx.max(bmx).to_array();
        }
    }
    // After a child / grandchild swap the other child has new children: worth another look.
    match best.2 {
        s if s == l => Some(l + 1),
        s if s == l + 1 => Some(l),
        _ => None,
    }
}

/// Binary radix tree over one task's sorted keys. Node 0 is the root; leaves point at `base + position`.
struct SubBuild<'a> {
    keys: &'a [u64],
    /// `delta[i]` = leading zeros of `keys[i] ^ keys[i + 1]`: a range splits after its (unique) smallest one.
    delta: Vec<u8>,
    ids: &'a [u32],
    positions: &'a [Vec3],
    tris: &'a [[u32; 3]],
    base: u32,
    nodes: Vec<Node>,
}
impl SubBuild<'_> {
    fn node(&mut self, ni: usize, lo: usize, hi: usize) -> Aabb {
        let count = hi - lo;
        if count <= LEAF {
            let mut bb = Aabb::EMPTY;
            for &t in &self.ids[lo..hi] {
                bb.union(&tri_box(self.positions, &self.tris[t as usize]));
            }
            self.nodes[ni] = Node { min: bb.min.to_array(), left_first: self.base + lo as u32, max: bb.max.to_array(), count: count as u32 };
            return bb;
        }
        let v = (self.keys[lo] ^ self.keys[hi - 1]).leading_zeros() as u8;
        let mid = lo + 1 + self.delta[lo..hi - 1].iter().position(|&x| x == v).unwrap_or(0);
        let l = self.nodes.len();
        self.nodes.push(Node::default());
        self.nodes.push(Node::default());
        let mut bb = self.node(l, lo, mid);
        bb.union(&self.node(l + 1, mid, hi));
        self.nodes[ni] = Node { min: bb.min.to_array(), left_first: l as u32, max: bb.max.to_array(), count: 0 };
        // Rotations below this size barely change the tree and cost a third of the build.
        if count > 16
            && let Some(c) = rotate(&mut self.nodes, ni)
        {
            rotate(&mut self.nodes, c);
        }
        bb
    }
}

/// What every subtree task shares. Subtrees go straight into the final node array: each one bump-allocates its
/// block behind the top nodes (blocks land in completion order; the array has no gaps either way).
struct TaskCx<'a> {
    keys: SyncPtr<u64>,
    ids: SyncPtr<u32>,
    out: SyncPtr<Node>,
    next: std::sync::atomic::AtomicUsize,
    positions: &'a [Vec3],
    tris: &'a [[u32; 3]],
    code_shift: u32,
    id_mask: u64,
    task_max: usize,
}
// SAFETY: the pointers are only used for disjoint ranges: each task's own keys / ids, its slot and the node block
// it allocated from `next` (see `run` and `emit`).
unsafe impl Sync for TaskCx<'_> {}
impl TaskCx<'_> {
    /// Sorts the task's keys, writes its triangle ids and its subtree (root into `t.slot`).
    fn run(&self, t: &Task) {
        let (lo, m) = (t.lo as usize, (t.hi - t.lo) as usize);
        // SAFETY: tasks cover disjoint ranges of `0..n`, and no one else touches the arrays meanwhile.
        let (k, ids) = unsafe { (std::slice::from_raw_parts_mut(self.keys.get().add(lo), m), std::slice::from_raw_parts_mut(self.ids.get().add(lo), m)) };
        if t.big && m >= 1 << 16 {
            sort_keys(k);
        } else {
            radix_sort(k, self.code_shift, t.sort_hi);
        }
        for (d, &key) in ids.iter_mut().zip(k.iter()) {
            *d = (key & self.id_mask) as u32;
        }
        if !t.big {
            self.emit(&sub_build(k, ids, self.positions, self.tris, t.lo), t.slot, None);
            return;
        }
        // Split this bucket on its (now sorted) keys into tasks of its own, built in parallel under its own top.
        let mut ts = TopSplit { keys: k, starts: &[], top_shift: 0, task_max: self.task_max, top: vec![Node::default()], tasks: Vec::new() };
        ts.split(0, 0, m, true);
        let TopSplit { top, tasks, .. } = ts;
        let mut pos = vec![0u32; top.len()];
        self.emit(&top, t.slot, Some(&mut pos));
        par_map(tasks.len(), |j| {
            let st = &tasks[j];
            let (a, b) = (st.lo as usize, st.hi as usize);
            self.emit(&sub_build(&k[a..b], &ids[a..b], self.positions, self.tris, t.lo + st.lo), pos[st.slot as usize], None);
        });
        // SAFETY: every subtree under this top is complete, and only this task writes these top nodes.
        unsafe { fix_top_boxes(self.out.get(), &top, &pos) };
    }

    /// Writes `src` (root first, local child indices, any order) depth first: the root into `slot`, the rest into
    /// a fresh block. `pos` receives every node's final index.
    fn emit(&self, src: &[Node], slot: u32, mut pos: Option<&mut [u32]>) {
        let out = self.out.get();
        let mut next = self.next.fetch_add(src.len() - 1, std::sync::atomic::Ordering::Relaxed) as u32;
        let mut stack = Vec::with_capacity(64);
        stack.push((0u32, slot));
        while let Some((q, p)) = stack.pop() {
            let mut nd = src[q as usize];
            // (`left_first == 0` on an interior node: a subtree slot of a top, filled in by that subtree.)
            if nd.count == 0 && nd.left_first != 0 {
                stack.push((nd.left_first + 1, next + 1));
                stack.push((nd.left_first, next));
                nd.left_first = next;
                next += 2;
            }
            if let Some(pos) = pos.as_deref_mut() {
                pos[q as usize] = p;
            }
            // SAFETY: `p` is the slot or inside the block reserved above; the array has room for every node.
            unsafe { out.add(p as usize).write(nd) };
        }
    }
}

/// Boxes of the interior top nodes (final index `pos[i]`) from their children, bottom up. Top nodes with
/// `left_first == 0` are subtree roots, already complete.
///
/// SAFETY: `out` must hold every node reachable from the top nodes, and nobody else may access them meanwhile.
unsafe fn fix_top_boxes(out: *mut Node, top: &[Node], pos: &[u32]) {
    for ni in (0..top.len()).rev() {
        if top[ni].left_first == 0 {
            continue;
        }
        // SAFETY: see the function contract.
        unsafe {
            let nd = &mut *out.add(pos[ni] as usize);
            let l = nd.left_first as usize;
            let (amn, amx) = node_box(&*out.add(l));
            let (bmn, bmx) = node_box(&*out.add(l + 1));
            nd.min = amn.min(bmn).to_array();
            nd.max = amx.max(bmx).to_array();
        }
    }
}

/// Builds one task's subtree (root first, local indices; rotations leave children before parents sometimes).
fn sub_build(keys: &[u64], ids: &[u32], positions: &[Vec3], tris: &[[u32; 3]], base: u32) -> Vec<Node> {
    let delta = keys.windows(2).map(|w| (w[0] ^ w[1]).leading_zeros() as u8).collect();
    let mut sb = SubBuild { keys, delta, ids, positions, tris, base, nodes: Vec::with_capacity(keys.len() * 2 / 3 + 2) };
    sb.nodes.push(Node::default());
    sb.node(0, 0, keys.len());
    sb.nodes
}

fn build_lbvh(positions: &[Vec3], tris: &[[u32; 3]]) -> Bvh {
    let n = tris.len();
    #[cfg(feature = "threads")]
    let threads = rayon::current_num_threads();
    #[cfg(not(feature = "threads"))]
    let threads = 1usize;
    let chunk = n.div_ceil(threads * 2).max(4096);
    let chunks = n.div_ceil(chunk);
    let range = |c: usize| c * chunk..((c + 1) * chunk).min(n);

    // 1. Bounds of the positions (they contain every centroid).
    let pchunk = positions.len().div_ceil(threads * 2).max(8192);
    let pbs = par_map(positions.len().div_ceil(pchunk), |c| {
        let mut b = Aabb::EMPTY;
        for &p in &positions[c * pchunk..((c + 1) * pchunk).min(positions.len())] {
            b.grow(p.into());
        }
        b
    });
    let mut cb = Aabb::EMPTY;
    for b in &pbs {
        cb.union(b);
    }

    // 2. Keys and per chunk histograms of the top bits.
    let id_bits = (u32::BITS - (n as u32 - 1).leading_zeros()).max(1);
    // Bits per axis: a grid well finer than the triangles (cheaper to sort than the full 64 bit key).
    let bits = ((64 - id_bits) / 3).min(id_bits.div_ceil(3) + 5).min(21);
    let top_bits = (usize::BITS - n.leading_zeros()).saturating_sub(8).clamp(6, 12).min(3 * bits);
    let nb = 1usize << top_bits;
    // Centroids are kept doubled (min + max), so the grid spans twice the bounds.
    let (cmin, ext) = (cb.min * 2.0, (cb.max - cb.min).max_element() * 2.0);
    let qmax = ((1u32 << bits) - 1) as f32;
    let scale = if ext > 0.0 { qmax / ext } else { 0.0 };
    let code_shift = 64 - 3 * bits;
    let top_shift = 64 - top_bits;
    let mut keys: Vec<u64> = Vec::with_capacity(n);
    let keys_ptr = SyncPtr(keys.as_mut_ptr());
    let hists = par_map(chunks, |c| {
        let mut h = vec![0u32; nb];
        let out = keys_ptr.get();
        for i in range(c) {
            let b = tri_box(positions, &tris[i]);
            let q = ((b.min + b.max - cmin) * scale).clamp(Vec3A::ZERO, Vec3A::splat(qmax));
            let code = spread3(q.x as u32) << 2 | spread3(q.y as u32) << 1 | spread3(q.z as u32);
            let k = code << code_shift | i as u64;
            h[(k >> top_shift) as usize] += 1;
            // SAFETY: `i < n` = capacity, and every chunk writes only its own indices.
            unsafe { out.add(i).write(k) };
        }
        h
    });
    // SAFETY: the chunks above covered `0..n`.
    unsafe { keys.set_len(n) };

    // 3. Counting sort scatter on the top bits (stable: equal codes keep ascending ids).
    let mut offs = vec![0u32; chunks * nb];
    let mut starts = vec![0u32; nb + 1];
    let mut run = 0u32;
    for b in 0..nb {
        starts[b] = run;
        for (c, h) in hists.iter().enumerate() {
            offs[c * nb + b] = run;
            run += h[b];
        }
    }
    starts[nb] = run;
    drop(hists);
    let mut sorted: Vec<u64> = Vec::with_capacity(n);
    let sorted_ptr = SyncPtr(sorted.as_mut_ptr());
    let offs_ptr = SyncPtr(offs.as_mut_ptr());
    par_map(chunks, |c| {
        // SAFETY: each chunk takes its own row of offsets.
        let offs = unsafe { std::slice::from_raw_parts_mut(offs_ptr.get().add(c * nb), nb) };
        let out = sorted_ptr.get();
        for &k in &keys[range(c)] {
            let o = &mut offs[(k >> top_shift) as usize];
            // SAFETY: the prefix sums give every (chunk, bucket) pair its own slots inside `0..n`.
            unsafe { out.add(*o as usize).write(k) };
            *o += 1;
        }
    });
    // SAFETY: the offsets cover `0..n` exactly once.
    unsafe { sorted.set_len(n) };
    drop(keys);
    let mut keys = sorted;

    // 4. Top of the tree on bucket boundaries.
    let task_max = (n / (threads * 4)).clamp(512, 8192);
    let mut ts = TopSplit { keys: &keys, starts: &starts, top_shift, task_max, top: vec![Node::default()], tasks: Vec::new() };
    ts.split(0, 0, n, false);
    let TopSplit { top, tasks, .. } = ts;

    // 5. Subtrees, written into place (at most 2n - 1 nodes: every leaf holds a triangle).
    let mut tri_ids: Vec<u32> = Vec::with_capacity(n);
    let mut nodes: Vec<Node> = Vec::with_capacity(2 * n);
    nodes.extend_from_slice(&top);
    let cx = TaskCx {
        keys: SyncPtr(keys.as_mut_ptr()),
        ids: SyncPtr(tri_ids.as_mut_ptr()),
        out: SyncPtr(nodes.as_mut_ptr()),
        next: std::sync::atomic::AtomicUsize::new(top.len()),
        positions,
        tris,
        code_shift,
        id_mask: (1u64 << id_bits) - 1,
        task_max,
    };
    par_map(tasks.len(), |j| cx.run(&tasks[j]));
    let total = cx.next.into_inner();
    // SAFETY: the tasks covered `0..n` of the ids and wrote nodes `0..total` (top slots and their blocks).
    unsafe {
        tri_ids.set_len(n);
        nodes.set_len(total);
    }

    // 6. Top boxes.
    let ident: Vec<u32> = (0..top.len() as u32).collect();
    // SAFETY: all subtrees are complete.
    unsafe { fix_top_boxes(nodes.as_mut_ptr(), &top, &ident) };
    Bvh { nodes, tri_ids, slack: 0.0 }
}

/// Fixed-capacity stack with heap overflow for panic-free, zero-alloc BVH traversal. The inline part is left
/// uninitialised: clearing 256 bytes per query costs more than the whole walk of a small tree.
struct LocalStack {
    buf: [std::mem::MaybeUninit<u32>; 64],
    len: usize,
    overflow: Vec<u32>,
}
impl LocalStack {
    #[inline(always)]
    fn new() -> Self {
        LocalStack { buf: [std::mem::MaybeUninit::uninit(); 64], len: 0, overflow: Vec::new() }
    }
    #[inline(always)]
    fn push(&mut self, v: usize) {
        if self.len < 64 {
            self.buf[self.len].write(v as u32);
            self.len += 1;
        } else {
            self.overflow.push(v as u32);
        }
    }
    #[inline(always)]
    fn pop(&mut self) -> Option<usize> {
        if !self.overflow.is_empty() {
            return self.overflow.pop().map(|v| v as usize);
        }
        if self.len > 0 {
            self.len -= 1;
            // SAFETY: slots below `len` were written by `push`.
            Some(unsafe { self.buf[self.len].assume_init() } as usize)
        } else {
            None
        }
    }
}

/// The two 16 byte halves of a node as vectors (lane 3 of each holds `left_first` / `count` bits, ignored).
#[inline(always)]
fn node_box(n: &Node) -> (Vec3A, Vec3A) {
    let p = n as *const Node as *const [f32; 4];
    // SAFETY: `Node` is `repr(C)` and 32 bytes: both reads stay inside it, and every bit pattern is a valid f32.
    unsafe { (Vec3A::from_vec4(Vec4::from_array(p.read())), Vec3A::from_vec4(Vec4::from_array(p.add(1).read()))) }
}

/// Per-ray constants for the slab test.
struct RayCtx {
    #[cfg(target_arch = "aarch64")]
    inv: std::arch::aarch64::float32x4_t,
    #[cfg(target_arch = "aarch64")]
    neg_o_inv: std::arch::aarch64::float32x4_t,
    #[cfg(target_arch = "aarch64")]
    lo_add: std::arch::aarch64::float32x4_t,
    #[cfg(target_arch = "aarch64")]
    hi_add: std::arch::aarch64::float32x4_t,
    /// Scalar copies (the root test only, on aarch64).
    s: (Vec3A, Vec3A, Vec3A),
}
impl RayCtx {
    #[inline(always)]
    fn new(o: Vec3A, d: Vec3A, slack: f32) -> RayCtx {
        let inv = d.recip();
        let neg_o_inv = -o * inv;
        let s_inv = Vec3A::splat(slack) * inv.abs();
        #[cfg(target_arch = "aarch64")]
        {
            // Lane 3 is neutralised: inv = 0 (so the node's integer lane vanishes), -inf / +inf biases.
            let v = |a: Vec3A, w: f32| Vec4::new(a.x, a.y, a.z, w).to_array();
            // SAFETY: loads from 4 element arrays.
            unsafe {
                use std::arch::aarch64::vld1q_f32;
                RayCtx {
                    s: (inv, neg_o_inv, s_inv),
                    inv: vld1q_f32(v(inv, 0.0).as_ptr()),
                    neg_o_inv: vld1q_f32(v(neg_o_inv, 0.0).as_ptr()),
                    lo_add: vld1q_f32(v(-s_inv, f32::NEG_INFINITY).as_ptr()),
                    hi_add: vld1q_f32(v(s_inv, f32::INFINITY).as_ptr()),
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        RayCtx { s: (inv, neg_o_inv, s_inv) }
    }
}

impl Bvh {
    pub fn build(positions: &[Vec3], tris: &[[u32; 3]]) -> Bvh {
        let n = tris.len();
        if n == 0 {
            return Bvh::default();
        }
        if n >= LBVH_MIN {
            // On a pool thread every pass is pushed onto a local queue instead of being injected from outside.
            #[cfg(feature = "threads")]
            return rayon::scope(|_| build_lbvh(positions, tris));
            #[cfg(not(feature = "threads"))]
            return build_lbvh(positions, tris);
        }
        let mut info: Vec<TriInfo> = tris.iter().enumerate().map(|(i, t)| TriInfo::new(tri_box(positions, t), i as u32)).collect();
        let root = Bounds::of_recs(&info);
        let nodes = build_seq(&mut info, 0, root);
        let tri_ids = info.iter().map(TriInfo::id).collect();
        Bvh { nodes, tri_ids, slack: 0.0 }
    }

    /// Recomputes every box bottom up (children always come after their parent) and clears the slack.
    pub fn refit(&mut self, positions: &[Vec3], tris: &[[u32; 3]]) {
        let tri_ids = &self.tri_ids;
        let leaf = |nd: &mut Node| {
            if nd.count == 0 {
                return;
            }
            let mut lo = Vec3A::splat(f32::INFINITY);
            let mut hi = Vec3A::splat(f32::NEG_INFINITY);
            let start = nd.left_first as usize;
            for &t in &tri_ids[start..start + nd.count as usize] {
                let tri = tris[t as usize];
                let a = Vec3A::from(positions[tri[0] as usize]);
                let b = Vec3A::from(positions[tri[1] as usize]);
                let c = Vec3A::from(positions[tri[2] as usize]);
                lo = lo.min(a).min(b).min(c);
                hi = hi.max(a).max(b).max(c);
            }
            nd.min = lo.to_array();
            nd.max = hi.to_array();
        };
        // Leaves are independent: all of them at once, then the (cheap) interior boxes from the bottom up.
        #[cfg(feature = "threads")]
        if self.nodes.len() >= 16384 {
            use rayon::prelude::*;
            self.nodes.par_iter_mut().with_min_len(2048).for_each(leaf);
        } else {
            self.nodes.iter_mut().for_each(leaf);
        }
        #[cfg(not(feature = "threads"))]
        self.nodes.iter_mut().for_each(leaf);
        for ni in (0..self.nodes.len()).rev() {
            let nd = self.nodes[ni];
            if nd.count > 0 {
                continue;
            }
            let l = nd.left_first as usize;
            let (nl, nr) = (&self.nodes[l], &self.nodes[l + 1]);
            let lo = Vec3A::from_array(nl.min).min(Vec3A::from_array(nr.min));
            let hi = Vec3A::from_array(nl.max).max(Vec3A::from_array(nr.max));
            self.nodes[ni].min = lo.to_array();
            self.nodes[ni].max = hi.to_array();
        }
        self.slack = 0.0;
    }

    /// Ray vs both children (`l` and `l + 1`) at once. Returns each child's entry distance, or `INFINITY` for a miss.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn slab2(&self, l: usize, r: &RayCtx, tmax: f32) -> (f32, f32) {
        use std::arch::aarch64::*;
        let pair = &self.nodes[l..l + 2];
        // SAFETY: `pair` holds two 32 byte `repr(C)` nodes = four 16 byte loads, all in bounds. Lane 3 of the node
        // halves holds an integer bit pattern (a tiny finite f32, never NaN/inf); `r.inv` is 0 there so it vanishes,
        // and the lane-3 biases in `lo_add`/`hi_add` (-inf/+inf) keep it out of both reductions.
        unsafe {
            let p = pair.as_ptr() as *const f32;
            let (ml, xl, mr, xr) = (vld1q_f32(p), vld1q_f32(p.add(4)), vld1q_f32(p.add(8)), vld1q_f32(p.add(12)));
            let (inv, noi) = (r.inv, r.neg_o_inv);
            let (t0l, t1l) = (vfmaq_f32(noi, ml, inv), vfmaq_f32(noi, xl, inv));
            let (t0r, t1r) = (vfmaq_f32(noi, mr, inv), vfmaq_f32(noi, xr, inv));
            let lol = vaddq_f32(vminnmq_f32(t0l, t1l), r.lo_add);
            let lor = vaddq_f32(vminnmq_f32(t0r, t1r), r.lo_add);
            let hil = vaddq_f32(vmaxnmq_f32(t0l, t1l), r.hi_add);
            let hir = vaddq_f32(vmaxnmq_f32(t0r, t1r), r.hi_add);
            // Two pairwise steps reduce both children: lanes 0/1 = left/right.
            let lo = vpmaxnmq_f32(lol, lor);
            let lo = vpmaxnmq_f32(lo, lo);
            let hi = vpminnmq_f32(hil, hir);
            let hi = vpminnmq_f32(hi, hi);
            let lo = vmaxnmq_f32(lo, vdupq_n_f32(0.0));
            let hi = vminnmq_f32(hi, vdupq_n_f32(tmax));
            let hit = vcleq_f32(lo, hi);
            let out = vbslq_f32(hit, lo, vdupq_n_f32(f32::INFINITY));
            (vgetq_lane_f32::<0>(out), vgetq_lane_f32::<1>(out))
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    fn slab2(&self, l: usize, r: &RayCtx, tmax: f32) -> (f32, f32) {
        #[inline(always)]
        fn one(n: &Node, r: &RayCtx, tmax: f32) -> f32 {
            let (mn, mx) = node_box(n);
            let (inv, noi, s_inv) = r.s;
            let t0 = mn.mul_add(inv, noi);
            let t1 = mx.mul_add(inv, noi);
            let lo = (t0.min(t1) - s_inv).max_element().max(0.0);
            let hi = (t0.max(t1) + s_inv).min_element().min(tmax);
            if lo <= hi {
                lo
            } else {
                f32::INFINITY
            }
        }
        (one(&self.nodes[l], r, tmax), one(&self.nodes[l + 1], r, tmax))
    }

    #[inline(always)]
    fn slab1(&self, ni: usize, r: &RayCtx, tmax: f32) -> bool {
        let (mn, mx) = node_box(&self.nodes[ni]);
        let (inv, noi, s_inv) = r.s;
        let t0 = mn.mul_add(inv, noi);
        let t1 = mx.mul_add(inv, noi);
        let lo = (t0.min(t1) - s_inv).max_element().max(0.0);
        let hi = (t0.max(t1) + s_inv).min_element().min(tmax);
        lo <= hi
    }

    /// Squared distance from `c` to both children's boxes: `(left, right)`.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn dist2_pair(&self, l: usize, c: Vec3A) -> (f32, f32) {
        use std::arch::aarch64::*;
        let pair = &self.nodes[l..l + 2];
        // SAFETY: as in `slab2`. Lane 3 of the box halves is a tiny finite integer-bit f32 and `c.w` is 0, so the
        // lane contributes ~0 (its square underflows) to the sums.
        unsafe {
            let p = pair.as_ptr() as *const f32;
            let cv = vsetq_lane_f32::<3>(0.0, vld1q_f32(&c as *const Vec3A as *const f32));
            let z = vdupq_n_f32(0.0);
            let d = |mn: float32x4_t, mx: float32x4_t| {
                let d = vmaxq_f32(vmaxq_f32(vsubq_f32(mn, cv), vsubq_f32(cv, mx)), z);
                vmulq_f32(d, d)
            };
            let sl = d(vld1q_f32(p), vld1q_f32(p.add(4)));
            let sr = d(vld1q_f32(p.add(8)), vld1q_f32(p.add(12)));
            let s = vpaddq_f32(sl, sr);
            let s = vpaddq_f32(s, s);
            (vgetq_lane_f32::<0>(s), vgetq_lane_f32::<1>(s))
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    fn dist2_pair(&self, l: usize, c: Vec3A) -> (f32, f32) {
        #[inline(always)]
        fn one(n: &Node, c: Vec3A) -> f32 {
            let (mn, mx) = node_box(n);
            (c.clamp(mn, mx) - c).length_squared()
        }
        (one(&self.nodes[l], c), one(&self.nodes[l + 1], c))
    }

    /// Closest hit along `dir` (need not be unit length; `t` is in units of `dir`). Back faces count.
    pub fn ray(&self, positions: &[Vec3], tris: &[[u32; 3]], o: Vec3, dir: Vec3) -> Option<Hit> {
        if self.nodes.is_empty() || tris.is_empty() {
            return None;
        }
        let (oa, da) = (Vec3A::from(o), Vec3A::from(dir));
        let ctx = RayCtx::new(oa, da, self.slack);
        let mut best: Option<Hit> = None;
        let mut tmax = f32::INFINITY;
        let mut stack = LocalStack::new();
        if self.slab1(0, &ctx, tmax) {
            stack.push(0usize);
        }
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni];
            if n.count > 0 {
                for &t in &self.tri_ids[n.left_first as usize..(n.left_first + n.count) as usize] {
                    let tri = tris[t as usize];
                    let (a, b, c) = (positions[tri[0] as usize], positions[tri[1] as usize], positions[tri[2] as usize]);
                    if let Some(tt) = ray_tri_a(oa, da, a.into(), b.into(), c.into()) {
                        if tt < tmax {
                            tmax = tt;
                            best = Some(Hit { t: tt, tri: t, point: o + dir * tt });
                        }
                    }
                }
                continue;
            }
            let l = n.left_first as usize;
            let (dl, dr) = self.slab2(l, &ctx, tmax);
            let (hl, hr) = (dl < f32::INFINITY, dr < f32::INFINITY);
            // Push the farther child first so the nearer one is visited first.
            if hl && hr {
                if dl <= dr {
                    stack.push(l + 1);
                    stack.push(l);
                } else {
                    stack.push(l);
                    stack.push(l + 1);
                }
            } else if hl {
                stack.push(l);
            } else if hr {
                stack.push(l + 1);
            }
        }
        best
    }

    /// Appends every vertex within `r` of `c` (strictly closer) to `out`, sorted by index, no duplicates.
    /// `stamp` must have one entry per vertex; it is reused across calls to avoid clearing.
    pub fn sphere_vertices(&self, positions: &[Vec3], tris: &[[u32; 3]], c: Vec3, r: f32, stamp: &mut Stamp, out: &mut Vec<u32>) {
        out.clear();
        if self.nodes.is_empty() || !(r > 0.0) {
            return;
        }
        let tag = stamp.next();
        let ca = Vec3A::from(c);
        let reach = r + self.slack;
        let reach2 = reach * reach;
        let r2 = r * r;
        let mut stack = LocalStack::new();
        let (rmn, rmx) = node_box(&self.nodes[0]);
        if (ca.clamp(rmn, rmx) - ca).length_squared() <= reach2 {
            stack.push(0usize);
        }
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni];
            if n.count == 0 {
                let l = n.left_first as usize;
                let (dl, dr) = self.dist2_pair(l, ca);
                if dr <= reach2 {
                    stack.push(l + 1);
                }
                if dl <= reach2 {
                    stack.push(l);
                }
                continue;
            }
            for &t in &self.tri_ids[n.left_first as usize..(n.left_first + n.count) as usize] {
                for &v in &tris[t as usize] {
                    let s = &mut stamp.marks[v as usize];
                    if *s != tag {
                        *s = tag;
                        if (positions[v as usize] - c).length_squared() < r2 {
                            out.push(v);
                        }
                    }
                }
            }
        }
        out.sort_unstable();
    }
}

/// Per vertex "seen" marks that never need clearing (bump the tag instead).
#[derive(Debug, Clone, Default)]
pub struct Stamp {
    pub marks: Vec<u32>,
    tag: u32,
}
impl Stamp {
    pub fn new(n: usize) -> Stamp {
        Stamp { marks: vec![0; n], tag: 0 }
    }
    pub fn resize(&mut self, n: usize) {
        self.marks.resize(n, 0);
    }
    pub fn next(&mut self) -> u32 {
        self.tag = self.tag.wrapping_add(1);
        if self.tag == 0 {
            self.marks.fill(0);
            self.tag = 1;
        }
        self.tag
    }
}

/// Möller-Trumbore, both sides.
#[inline]
pub fn ray_tri(o: Vec3, d: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    ray_tri_a(o.into(), d.into(), a.into(), b.into(), c.into())
}

/// [`ray_tri`] on 16 byte vectors (one NEON / SSE register each).
#[inline(always)]
fn ray_tri_a(o: Vec3A, d: Vec3A, a: Vec3A, b: Vec3A, c: Vec3A) -> Option<f32> {
    let e1 = b - a;
    let e2 = c - a;
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 0.0).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{uv_sphere, SculptMesh};

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
        fn v(&mut self) -> Vec3 {
            Vec3::new(self.f(), self.f(), self.f()) * 2.0 - 1.0
        }
    }

    fn brute_ray(m: &SculptMesh, o: Vec3, d: Vec3) -> Option<f32> {
        m.tris.iter().filter_map(|t| ray_tri(o, d, m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize])).reduce(f32::min)
    }

    #[test]
    fn every_triangle_once_and_boxes_contain() {
        for (seg, rings) in [(64, 32), (128, 64)] {
            let m = SculptMesh::from_mesh(&uv_sphere(10.0, seg, rings));
            let b = Bvh::build(&m.positions, &m.tris);
            let mut ids = b.tri_ids.clone();
            ids.sort_unstable();
            assert_eq!(ids, (0..m.tris.len() as u32).collect::<Vec<_>>());
            let leaves: u32 = b.nodes.iter().map(|n| n.count).sum();
            assert_eq!(leaves as usize, m.tris.len());
            for (ni, n) in b.nodes.iter().enumerate() {
                if n.count == 0 {
                    assert!(n.left_first as usize > ni, "parent {ni} must precede children");
                } else {
                    for &t in &b.tri_ids[n.left_first as usize..(n.left_first + n.count) as usize] {
                        for &v in &m.tris[t as usize] {
                            let p = m.positions[v as usize];
                            assert!(p.cmpge(Vec3::from(n.min)).all() && p.cmple(Vec3::from(n.max)).all());
                        }
                    }
                }
            }
        }
    }

    /// ~100k triangles take the parallel binning / partition passes; refit takes the parallel leaf pass.
    #[test]
    fn big_parallel_build_and_refit_are_valid() {
        let mut m = SculptMesh::from_mesh(&uv_sphere(10.0, 320, 160));
        assert!(m.tris.len() > 2 * PARALLEL_PASS_MIN_TEST);
        let mut b = Bvh::build(&m.positions, &m.tris);
        let check_tree = |m: &SculptMesh, b: &Bvh| {
            let mut ids = b.tri_ids.clone();
            ids.sort_unstable();
            assert!(ids.iter().enumerate().all(|(i, &t)| t as usize == i), "ids must be a permutation");
            let mut leaf_tris = 0u32;
            for (ni, n) in b.nodes.iter().enumerate() {
                let (lo, hi) = (Vec3::from(n.min), Vec3::from(n.max));
                if n.count == 0 {
                    assert!(n.left_first as usize > ni);
                    for c in [n.left_first as usize, n.left_first as usize + 1] {
                        assert!(Vec3::from(b.nodes[c].min).cmpge(lo).all() && Vec3::from(b.nodes[c].max).cmple(hi).all());
                    }
                } else {
                    leaf_tris += n.count;
                    for &t in &b.tri_ids[n.left_first as usize..(n.left_first + n.count) as usize] {
                        for &v in &m.tris[t as usize] {
                            let p = m.positions[v as usize];
                            assert!(p.cmpge(lo).all() && p.cmple(hi).all());
                        }
                    }
                }
            }
            assert_eq!(leaf_tris as usize, m.tris.len());
        };
        check_tree(&m, &b);
        let mut rng = Rng(7);
        let rays = |m: &SculptMesh, b: &Bvh, rng: &mut Rng| {
            for _ in 0..60 {
                let o = rng.v() * 30.0;
                let d = (rng.v() * 5.0 - o).normalize();
                let got = b.ray(&m.positions, &m.tris, o, d).map(|h| h.t);
                let want = brute_ray(m, o, d);
                match (got, want) {
                    (Some(g), Some(w)) => assert!((g - w).abs() < 1e-4, "{g} vs {w}"),
                    (g, w) => assert_eq!(g.is_some(), w.is_some()),
                }
            }
        };
        rays(&m, &b, &mut rng);
        for i in (0..m.positions.len()).step_by(3) {
            m.positions[i] += rng.v() * 0.8;
        }
        b.refit(&m.positions, &m.tris);
        check_tree(&m, &b);
        rays(&m, &b, &mut rng);
    }

    const PARALLEL_PASS_MIN_TEST: usize = 1 << 15;

    #[test]
    fn rays_agree_with_brute_force_before_and_after_moves() {
        let mut m = SculptMesh::from_mesh(&uv_sphere(10.0, 48, 24));
        let mut b = Bvh::build(&m.positions, &m.tris);
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let check = |m: &SculptMesh, b: &Bvh, rng: &mut Rng| {
            for _ in 0..500 {
                let o = rng.v() * 30.0;
                let d = (rng.v() * 5.0 - o).normalize();
                let got = b.ray(&m.positions, &m.tris, o, d).map(|h| h.t);
                let want = brute_ray(m, o, d);
                match (got, want) {
                    (Some(g), Some(w)) => assert!((g - w).abs() < 1e-4, "{g} vs {w}"),
                    (g, w) => assert_eq!(g.is_some(), w.is_some()),
                }
            }
        };
        check(&m, &b, &mut rng);
        // Move some vertices without refitting: slack keeps queries exact.
        let mut moved: f32 = 0.0;
        for i in (0..m.positions.len()).step_by(7) {
            let d = rng.v() * 1.5;
            moved = moved.max(d.length());
            m.positions[i] += d;
        }
        b.slack = moved;
        check(&m, &b, &mut rng);
        b.refit(&m.positions, &m.tris);
        check(&m, &b, &mut rng);
    }

    #[test]
    fn sphere_query_matches_brute_force() {
        let m = SculptMesh::from_mesh(&uv_sphere(10.0, 48, 24));
        let b = Bvh::build(&m.positions, &m.tris);
        let mut stamp = Stamp::new(m.positions.len());
        let mut out = Vec::new();
        let mut rng = Rng(42);
        for _ in 0..50 {
            let c = rng.v() * 11.0;
            let r = 1.0 + rng.f() * 6.0;
            b.sphere_vertices(&m.positions, &m.tris, c, r, &mut stamp, &mut out);
            let want: Vec<u32> = (0..m.positions.len() as u32).filter(|&v| (m.positions[v as usize] - c).length() < r).collect();
            assert_eq!(out, want);
        }
    }
}

