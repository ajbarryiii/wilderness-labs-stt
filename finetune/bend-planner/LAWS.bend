# LAWS.bend: the planner's laws (DESIGN.md "Laws"), stated over plan.bend, which imports no foreign
# code. PROOF.bend proves them; README.md lists the ones still open and why. The vocabulary below
# is written for the laws and does not reuse the planner's own tests (Plan.starts, Plan.check's
# internals), so a law cannot hold merely by restating the code.
#
# DESIGN.md's laws, per pool xs (one pool's clips) and budget b:
#   1. exact cover   pool_cover    flat(pool(b, xs)) holds each clip of xs exactly as often as xs
#                    pools_cover   the pools are the clips 0 .. n-1, in sequence order, cut into runs
#                    pools_sizes   of the sizes the offsets give: pool k is [ps[k], ps[k + 1])
#   2. budget        pool_budget   every batch has one clip, or every clip x in it has
#                                  x.s = 0 or len <= b / x.s (len * max(s) <= b, division form)
#   3. greedy        pool_sorted   the pool's batches, joined, ascend in (samples, sequence) order
#                    pool_greedy   the first clip x of each next batch breaks the batch before it:
#                                  x.s > 0 and len + 1 > b / x.s
#   and              pool_filled   no batch is empty (the offsets strictly increase)
#                    solve_spec    the parallel driver computes the sequential plan
# The lemmas sort_perm and sort_sorted, and cut_cover, are the halves pool_cover and pool_sorted
# split into.
import Base
import ./u32.bend as W
import ./plan.bend as Plan

# Vocabulary
# ----------

def Holds(b: Bool) -> Data:
  {b == True{} : Bool}

def flat(bs: List<&2, List<&2, Plan.Clip>>) -> List<&2, Plan.Clip>:
  List.concat(&2, Plan.Clip, bs)

def len(xs: List<&2, Plan.Clip>) -> Nat:
  List.length(&2, Plan.Clip, xs)

# a and b are the same clip
def same(a: Plan.Clip, b: Plan.Clip) -> Bool:
  match a b:
    case Plan.Clip{x, i} Plan.Clip{y, j}:
      Bool.and(U32.is_eq(x, y), U32.is_eq(i, j))

def bump(b: Bool, n: Nat) -> Nat:
  match b:
    case True{}:
      1n+n
    case False{}:
      n

# how many times y occurs in xs
def count(+y: Plan.Clip, xs: List<&2, Plan.Clip>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case h <> t:
      bump(same(y, h), count(y, t))

# (a.s, a.i) <= (b.s, b.i), lexicographically
def key.le.cmp(c: Cmp, +i: U32, +j: U32) -> Bool:
  match c:
    case LT{}:
      True{}
    case EQ{}:
      U32.is_le(i, j)
    case GT{}:
      False{}

def key.le(a: Plan.Clip, b: Plan.Clip) -> Bool:
  match a b:
    case Plan.Clip{x, i} Plan.Clip{y, j}:
      key.le.cmp(U32.cmp(x, y), i, j)

# xs ascends in key order from lo
def Sorted.from(+lo: Plan.Clip, xs: List<&2, Plan.Clip>) -> Type:
  match xs:
    case Nil{}:
      Unit
    case +h <> t:
      Holds(key.le(lo, h)) & Sorted.from(h, t)

def Sorted(xs: List<&2, Plan.Clip>) -> Type:
  match xs:
    case Nil{}:
      Unit
    case +h <> t:
      Sorted.from(h, t)

# A clip of s samples is within budget b in a batch of c clips: s = 0 or c <= b / s, which for
# s > 0 is c * s <= b
def fits(+b: U32, +c: Nat, x: Plan.Clip) -> Bool:
  match x:
    case Plan.Clip{+s, i}:
      Bool.or(U32.is_zero(s), Nat.is_le(c, U32.to_nat(U32.div(b, s))))

def fits.all(+b: U32, +c: Nat, xs: List<&2, Plan.Clip>) -> Bool:
  match xs:
    case Nil{}:
      True{}
    case x <> t:
      Bool.and(fits(b, c, x), fits.all(b, c, t))

# A batch is within budget: it has one clip, or every clip fits its length
def budget(+b: U32, +xs: List<&2, Plan.Clip>) -> Bool:
  Bool.or(Nat.is_eq(len(xs), 1n), fits.all(b, len(xs), xs))

def Budgets(+b: U32, bs: List<&2, List<&2, Plan.Clip>>) -> Type:
  match bs:
    case Nil{}:
      Unit
    case +xs <> t:
      Holds(budget(b, xs)) & Budgets(b, t)

# A clip x breaks a batch of c clips: x.s > 0 and c + 1 > b / x.s, which is (c + 1) * x.s > b
def breaks(+b: U32, +c: Nat, x: Plan.Clip) -> Bool:
  match x:
    case Plan.Clip{+s, i}:
      Bool.and(Bool.not(U32.is_zero(s)), Nat.is_gt(1n+c, U32.to_nat(U32.div(b, s))))

def Greedy.next(+b: U32, +c: Nat, bs: List<&2, List<&2, Plan.Clip>>) -> Type:
  match bs:
    case Nil{}:
      Unit
    case Nil{} <> t:
      Unit
    case (x <> xt) <> t:
      Holds(breaks(b, c, x))

# Each batch is broken by the first clip of the batch after it
def Greedy(+b: U32, bs: List<&2, List<&2, Plan.Clip>>) -> Type:
  match bs:
    case Nil{}:
      Unit
    case +xs <> +t:
      Greedy.next(b, len(xs), t) & Greedy(b, t)

# No batch is empty
def Filled(bs: List<&2, List<&2, Plan.Clip>>) -> Type:
  match bs:
    case Nil{}:
      Unit
    case xs <> t:
      Holds(Bool.not(List.is_empty(&2, Plan.Clip, xs))) & Filled(t)

def lens(xss: List<&2, List<&2, Plan.Clip>>) -> List<&2, Nat>:
  match xss:
    case Nil{}:
      Nil{}
    case xs <> t:
      len(xs) <> lens(t)

# The gaps between consecutive offsets, in Nat
def gaps(ps: List<&2, U32>, +lo: U32) -> List<&2, Nat>:
  match ps:
    case Nil{}:
      Nil{}
    case +h <> t:
      Nat.sub(U32.to_nat(h), U32.to_nat(lo)) <> gaps(t, h)

def offsets.gaps(ps: List<&2, U32>) -> List<&2, Nat>:
  match ps:
    case Nil{}:
      Nil{}
    case +h <> t:
      gaps(t, h)

def is_none(m: Maybe<&2, String>) -> Bool:
  match m:
    case None{}:
      True{}
    case Some{x}:
      False{}

# ss has at most 2^32 - 1 samples, as a u32 file does (its length is a u32 count of bytes); past
# that, Valid compares the pools with the length mod 2^32, and the pool laws need not hold
def Small(+ss: List<&2, U32>) -> Type:
  Holds(Nat.is_le(List.length(&2, U32, ss), W.Word.max_nat(32n)))

# The pool offsets pass the planner's validation for the samples ss
def Valid(+ss: List<&2, U32>, +ps: List<&2, U32>) -> Type:
  Holds(is_none(Plan.check(ps, U32.from_nat(List.length(&2, U32, ss)))))

def leaves(t: Plan.Tree) -> List<&2, List<&2, Plan.Clip>>:
  match t:
    case Plan.TLeaf{xss}:
      xss
    case Plan.TNode{l, r}:
      List.append(&2, List<&2, Plan.Clip>, leaves(l), leaves(r))

# Laws: one pool
# --------------

# LAW 1 (exact cover): a pool's batches hold its clips, each exactly as often as the pool does
law pool_cover:
  for +b : U32
  for +y : Plan.Clip
  for +xs: List<&2, Plan.Clip>
  {count(y, flat(Plan.pool(b, xs))) == count(y, xs) : Nat}

# LAW 2 (budget): every batch has one clip, or len * max(samples) <= b (division form, per clip)
law pool_budget:
  for +b : U32
  for +xs: List<&2, Plan.Clip>
  Budgets(b, Plan.pool(b, xs))

# LAW 3a (greedy, order): the batches, joined, ascend in (samples, sequence number) order
law pool_sorted:
  for +b : U32
  for +xs: List<&2, Plan.Clip>
  Sorted(flat(Plan.pool(b, xs)))

# LAW 3b (greedy, cut): adding the next batch's first clip to a batch would break the budget
law pool_greedy:
  for +b : U32
  for +xs: List<&2, Plan.Clip>
  Greedy(b, Plan.pool(b, xs))

# no batch is empty, so the offsets strictly increase
law pool_filled:
  for +b : U32
  for +xs: List<&2, Plan.Clip>
  Filled(Plan.pool(b, xs))

# Laws: the pools
# ---------------

# LAW 1 (exact cover, pools): the pools, joined, are the clips 0 .. n-1 in sequence order
law pools_cover:
  for +ss: List<&2, U32>
  for +ps: List<&2, U32>
  for w  : Small(ss)
  for v  : Valid(ss, ps)
  {flat(Plan.pools(ss, ps)) == Plan.clips(ss, 0) : List<&2, Plan.Clip>}

# LAW 1 (each pool inside its range): pool k holds ps[k + 1] - ps[k] clips, so with pools_cover
# it is exactly the clips [ps[k], ps[k + 1])
law pools_sizes:
  for +ss: List<&2, U32>
  for +ps: List<&2, U32>
  for w  : Small(ss)
  for v  : Valid(ss, ps)
  {lens(Plan.pools(ss, ps)) == offsets.gaps(ps) : List<&2, Nat>}

# the parallel driver (a balanced tree, cut leaf by leaf) computes the sequential plan
law solve_spec:
  for +b  : U32
  for +d  : Nat
  for +k  : Nat
  for +xss: List<&2, List<&2, Plan.Clip>>
  {leaves(Plan.solve(b, Plan.tree(d, k, xss))) == Plan.pools.cut(b, xss) : List<&2, List<&2, Plan.Clip>>}

# Lemmas
# ------

# the cut keeps the clips it is given, in order
law cut_cover:
  for +b : U32
  for +xs: List<&2, Plan.Clip>
  {flat(Plan.cut(b, xs)) == xs : List<&2, Plan.Clip>}

# the sort is a permutation
law sort_perm:
  for +y : Plan.Clip
  for +xs: List<&2, Plan.Clip>
  {count(y, Plan.sort(xs)) == count(y, xs) : Nat}

# the sort ascends in key order
law sort_sorted:
  for +xs: List<&2, Plan.Clip>
  Sorted(Plan.sort(xs))
