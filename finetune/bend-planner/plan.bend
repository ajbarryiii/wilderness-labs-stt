# plan.bend: the pure batch planner (DESIGN.md "Contract"), with no foreign imports, so LAWS.bend and
# PROOF.bend may import it. planner.bend does the file I/O and the CLI around it.
#
#   samples, pools : List<&2, U32>   (the input files, as lists)
#   Plan.check(pools, n)             None{} when the pool offsets are valid, else Some{message}
#   Plan.batches(b, samples, pools)  the batches, pool by pool, in cut order
#
# A clip is its length in samples and its sequence number. Each pool is sorted by (s, i) and cut
# greedily: a new batch starts before clip x when the batch under way holds c > 0 clips, x.s > 0 and
# c + 1 > b / x.s (the overflow-free form of (c + 1) * x.s > b). Pools are independent, so
# Plan.solve cuts them in parallel over a balanced tree, and Plan.run writes the result into the
# output arrays. LAWS.bend states the laws over these defs; Plan.clips is there only for them.
import Base

# Clips
# -----

type Clip is Data:
  Clip{s: U32, i: U32}

def Clip.s(c: Clip) -> U32:
  match c:
    case Clip{s, i}:
      s

def Clip.i(c: Clip) -> U32:
  match c:
    case Clip{s, i}:
      i

def Clip.le.cmp(c: Cmp, +x: U32, +y: U32) -> Bool:
  match c:
    case LT{}:
      True{}
    case EQ{}:
      U32.is_le(x, y)
    case GT{}:
      False{}

# (a.s, a.i) <= (b.s, b.i), lexicographically: the key DESIGN.md sorts a pool by
def Clip.le(a: Clip, b: Clip) -> Bool:
  match a b:
    case Clip{x, ai} Clip{y, bi}:
      Clip.le.cmp(U32.cmp(x, y), ai, bi)

# The clips of samples, numbered from i.
def clips(ss: List<&2, U32>, +i: U32) -> List<&2, Clip>:
  match ss:
    case Nil{}:
      Nil{}
    case s <> t:
      Clip{s, i} <> clips(t, (i + 1 : U32))

# Sort
# ----

def sort(xs: List<&2, Clip>) -> List<&2, Clip>:
  List.sort(~Clip, ~Clip.le, xs)

# Cut
# ---

def starts.pos(+b: U32, +c: Nat, s: U32, z: Bool) -> Bool:
  match z:
    case True{}:
      False{}
    case False{}:
      Nat.is_gt(1n+c, U32.to_nat(U32.div(b, s)))

# Does a clip of s samples start a new batch, after a batch of c clips? Never after an empty batch,
# never for a zero-sample clip; else when c + 1 > b / s.
def starts(+b: U32, c: Nat, +s: U32) -> Bool:
  match c:
    case 0n:
      False{}
    case 1n+p:
      starts.pos(b, 1n+p, s, U32.is_zero(s))

# A cut in progress: the clips that join the batch under way, and the batches after it.
def Cut() -> Type:
  List<&2, Clip> & List<&2, List<&2, Clip>>

def cut.next(nw: Bool, c: Nat) -> Nat:
  match nw:
    case True{}:
      1n
    case False{}:
      1n+c

def cut.step(x: Clip, nw: Bool, r: Cut()) -> Cut():
  match nw:
    case True{}:
      (cur, rest) = r
      (Nil{}, (x <> cur) <> rest)
    case False{}:
      (cur, rest) = r
      (x <> cur, rest)

# The cut of xs when the batch under way already holds c clips.
def cut.go(+b: U32, xs: List<&2, Clip>, +c: Nat) -> Cut():
  match xs:
    case Nil{}:
      (Nil{}, Nil{})
    case +x <> t:
      +nw = starts(b, c, Clip.s(x))
      cut.step(x, nw, cut.go(b, t, cut.next(nw, c)))

def cut.fin(r: Cut()) -> List<&2, List<&2, Clip>>:
  (cur, rest) = r
  match cur:
    case Nil{}:
      rest
    case h <> t:
      (h <> t) <> rest

# The greedy cut of xs (in the order given) into batches.
def cut(+b: U32, xs: List<&2, Clip>) -> List<&2, List<&2, Clip>>:
  cut.fin(cut.go(b, xs, 0n))

# One pool's batches: sorted by (s, i), then cut.
def pool(+b: U32, xs: List<&2, Clip>) -> List<&2, List<&2, Clip>>:
  cut(b, sort(xs))

# Pools
# -----

# The pool sizes from the offsets (each offset after the first, minus the one before it).
def sizes(ps: List<&2, U32>, +lo: U32) -> List<&2, Nat>:
  match ps:
    case Nil{}:
      Nil{}
    case +h <> t:
      U32.to_nat((h - lo : U32)) <> sizes(t, h)

# A run in progress: its clips, and the samples after it.
def Run() -> Type:
  List<&2, Clip> & List<&2, U32>

def take.cons(x: Clip, r: Run()) -> Run():
  (xs, rest) = r
  (x <> xs, rest)

# The first d samples of ss as clips numbered from i, and the samples after them.
def take(d: Nat, ss: List<&2, U32>, +i: U32) -> Run():
  match d:
    case 0n:
      (Nil{}, ss)
    case 1n+p:
      match ss:
        case Nil{}:
          (Nil{}, Nil{})
        case s <> t:
          take.cons(Clip{s, i}, take(p, t, (i + 1 : U32)))

# r's run, then the runs of sizes ds over r's remaining samples, numbered from i.
def runs(ds: List<&2, Nat>, +i: U32, r: Run()) -> List<&2, List<&2, Clip>>:
  match ds:
    case Nil{}:
      (xs, rest) = r
      [xs]
    case +d <> t:
      (xs, rest) = r
      xs <> runs(t, (i + U32.from_nat(d) : U32), take(d, rest, i))

# The clips of ss cut into consecutive runs of sizes ds, numbered from 0, in one pass.
def split(ds: List<&2, Nat>, ss: List<&2, U32>) -> List<&2, List<&2, Clip>>:
  match ds:
    case Nil{}:
      Nil{}
    case +d <> t:
      runs(t, U32.from_nat(d), take(d, ss, 0))

# The clips of each pool, in sequence order. Assumes Plan.check passed: pools starts at 0.
def pools(ss: List<&2, U32>, ps: List<&2, U32>) -> List<&2, List<&2, Clip>>:
  match ps:
    case Nil{}:
      Nil{}
    case +h <> t:
      split(sizes(t, h), ss)

# The batches of every pool, pool by pool.
def pools.cut(+b: U32, xss: List<&2, List<&2, Clip>>) -> List<&2, List<&2, Clip>>:
  match xss:
    case Nil{}:
      Nil{}
    case xs <> t:
      List.append(&2, List<&2, Clip>, pool(b, xs), pools.cut(b, t))

# The whole plan, sequentially: the specification Plan.solve computes in parallel.
def batches(+b: U32, ss: List<&2, U32>, ps: List<&2, U32>) -> List<&2, List<&2, Clip>>:
  pools.cut(b, pools(ss, ps))

# Validation
# ----------

def check.if(ok: Bool, msg: String, rest: Maybe<&2, String>) -> Maybe<&2, String>:
  match ok:
    case True{}:
      rest
    case False{}:
      Some{msg}

# The offsets after one at prev: none decreases and the last is n.
def check.go(ps: List<&2, U32>, +prev: U32, +n: U32) -> Maybe<&2, String>:
  match ps:
    case Nil{}:
      check.if(U32.is_eq(prev, n), "pools must end at n=" ++ U32.show(n) ++ ", not " ++ U32.show(prev), None{})
    case +h <> t:
      check.if(U32.is_le(prev, h), "pool offsets decrease: " ++ U32.show(h) ++ " after " ++ U32.show(prev),
        check.go(t, h, n))

# None{} when ps is a valid pool offset list for n clips (DESIGN.md "Inputs"), else the reason.
def check(ps: List<&2, U32>, +n: U32) -> Maybe<&2, String>:
  match ps:
    case Nil{}:
      Some{"pools is empty: it needs n_pools + 1 offsets"}
    case +h <> t:
      check.if(U32.is_zero(h), "pools must start at 0, not " ++ U32.show(h), check.go(t, h, n))

# Budget
# ------

# Can acc * 10 + (x - '0') be a u32 digit step? x is a digit and the result fits.
def budget.ok(+acc: U32, +x: U32) -> Bool:
  +digit = Bool.and(U32.is_ge(x, 48), U32.is_le(x, 57))
  +fits = Bool.or(U32.is_lt(acc, 429496729), Bool.and(U32.is_eq(acc, 429496729), U32.is_le(x, 53)))
  Bool.and(digit, fits)

def budget.if(ok: Bool, r: Maybe<&2, U32>) -> Maybe<&2, U32>:
  match ok:
    case True{}:
      r
    case False{}:
      None{}

def budget.go(s: String, +acc: U32) -> Maybe<&2, U32>:
  match s:
    case SNil{}:
      Some{acc}
    case SCon{Chr{+x}, t}:
      budget.if(budget.ok(acc, x), budget.go(t, ((acc * 10 : U32) + (x - 48 : U32) : U32)))

# The budget argument: one or more ASCII digits, at most 4294967295.
def budget(s: String) -> Maybe<&2, U32>:
  match s:
    case SNil{}:
      None{}
    case SCon{h, t}:
      budget.go(SCon{h, t}, 0)

# Parallel driver
# ---------------

type Tree is Data:
  TLeaf{xss: List<&2, List<&2, Clip>>}
  TNode{l: Tree, r: Tree}

def half(n: Nat) -> Nat:
  match n:
    case 0n:
      0n
    case 1n+0n:
      0n
    case 1n+1n+p:
      1n+half(p)

# The k pools xss as a balanced tree of depth d (2^d >= k): halves by count, one pool per leaf.
def tree(+d: Nat, +k: Nat, +xss: List<&2, List<&2, Clip>>) -> Tree:
  match d:
    case 0n:
      TLeaf{xss}
    case 1n+p:
      +h = half(k)
      TNode{tree(p, h, List.take(&2, List<&2, Clip>, xss, h)),
        tree(p, Nat.sub(k, h), List.drop(&2, List<&2, Clip>, xss, h))}

# Each leaf's pools replaced by their batches, the two halves of every node in parallel.
def solve(+b: U32, t: Tree) -> Tree:
  match t:
    case TLeaf{xss}:
      TLeaf{pools.cut(b, xss)}
    case TNode{l, r}:
      x y = solve(b, l) solve(b, r)
      TNode{x, y}

def depth.if(one: Bool, +k: U32) -> Nat:
  match one:
    case True{}:
      0n
    case False{}:
      1n+U32.log2((k - 1 : U32))

# The least d with 2^d >= k.
def depth(+k: U32) -> Nat:
  depth.if(U32.is_le(k, 1), k)

# Output
# ------

# The plan as written: order[0 .. at), and offs[0 .. nb] (offs[0] = 0, offs[k + 1] the end of
# batch k).
type Out is Type:
  Out{order: Array<U32>, offs: Array<U32>, at: U32, nb: U32}

def out.put(+i: U32, o: Out) -> Out:
  match o:
    case Out{order, offs, +at, nb}:
      Out{order[at] <- i, offs, (at + 1 : U32), nb}

def out.close(o: Out) -> Out:
  match o:
    case Out{order, offs, +at, +nb}:
      +k = (nb + 1 : U32)
      Out{order, offs[k] <- at, at, k}

def out.batch(xs: List<&2, Clip>, o: Out) -> Out:
  match xs:
    case Nil{}:
      out.close(o)
    case x <> t:
      out.batch(t, out.put(Clip.i(x), o))

def out.batches(bs: List<&2, List<&2, Clip>>, o: Out) -> Out:
  match bs:
    case Nil{}:
      o
    case xs <> t:
      out.batches(t, out.batch(xs, o))

def out.tree(t: Tree, o: Out) -> Out:
  match t:
    case TLeaf{xss}:
      out.batches(xss, o)
    case TNode{l, r}:
      out.tree(r, out.tree(l, o))

# Arrays
# ------

# Cells i, i - 1, .., i - n of r's array consed onto acc, so the list reads upwards; r holds the
# array and cell i.
def list.go(n: Nat, +i: U32, acc: List<&2, U32>, r: Array<U32> & U32) -> Array<U32> & List<&2, U32>:
  match n:
    case 0n:
      (a, +x) = r
      (a, x <> acc)
    case 1n+p:
      (a, +x) = r
      +j = (i - 1 : U32)
      list.go(p, j, x <> acc, a[j])

def list.if(z: Bool, +n: U32, a: Array<U32>) -> Array<U32> & List<&2, U32>:
  match z:
    case True{}:
      (a, Nil{})
    case False{}:
      +j = (n - 1 : U32)
      list.go(U32.to_nat(j), j, Nil{}, a[j])

# Cells 0 .. n-1 of a, as a list.
def list(+n: U32, a: Array<U32>) -> Array<U32> & List<&2, U32>:
  list.if(U32.is_zero(n), n, a)

# Run
# ---

# The plan of the samples ss and the k + 1 pool offsets ps (validated by Plan.check), written into
# order (n cells) and offs (at least n + 1 cells): the arrays, the clips written and the batches.
def run(+b: U32, ss: List<&2, U32>, ps: List<&2, U32>, +k: Nat, order: Array<U32>, offs: Array<U32>) -> Out:
  +t = tree(depth(U32.from_nat(k)), k, pools(ss, ps))
  out.tree(solve(b, t), Out{order, offs[0] <- 0, 0, 0})
