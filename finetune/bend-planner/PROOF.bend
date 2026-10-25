# PROOF.bend: all eleven laws of LAWS.bend, with no unsafe or foreign code.
# Both `bend PROOF.bend` and `bend PROOF.bend --verdict` print ALL PROOFS CHECK.
#
#   cut_cover     Cut: cover             the cut keeps its input, in order
#   pool_filled   Cut: no empty batch
#   pool_greedy   Cut: greedy            holds for a cut of any list, sorted or not
#   solve_spec    Parallel driver        the tree and the parallel solve compute pools.cut
#   sort_perm     Sort: permutation      every step of Base's List.sort keeps each count
#   pool_cover    cut_cover + sort_perm
#   sort_sorted   Sort: order            totality of the key order, merges, and enough passes
#   pool_sorted   cut_cover + sort_sorted
#   pools_sizes   validated differences telescope; take/runs consume their requested lengths
#   pool_budget   cut last-clip invariant + sortedness + division antitonicity
#   pools_cover   exact split consumption + bounded U32 conversion
#
# A rewrite `%e : P` with e : {a == b} needs P, with _ at b, to be the goal; the goal becomes P
# with a there. Most lemmas are stated simple == complex so that a rewrite folds the complex side.
import Base
import ./plan.bend as Plan
import ./LAWS.bend as Laws
import ./u32.bend as W
import 0xcd4682caa53da19986acd472c2181473/nat.bend as N

# Cut: cover
# ----------

# the clips of a cut in progress, in order
def joined(r: Plan.Cut()) -> List<&2, Plan.Clip>:
  (cur, rest) = r
  List.append(&2, Plan.Clip, cur, Laws.flat(rest))

def step_cover(-x: Plan.Clip, nw: Bool, r: Plan.Cut()) ->
  {x <> joined(r) == joined(Plan.cut.step(x, nw, r)) : List<&2, Plan.Clip>}:
  match nw:
    case True{}:
      (cur, rest) = r
      {==}
    case False{}:
      (cur, rest) = r
      {==}

def go_cover(+b: U32, xs: List<&2, Plan.Clip>, +c: Nat) ->
  {xs == joined(Plan.cut.go(b, xs, c)) : List<&2, Plan.Clip>}:
  match xs:
    case Nil{}:
      {==}
    case +x <> +t:
      +nw = Plan.starts(b, c, Plan.Clip.s(x))
      %step_cover(x, nw, Plan.cut.go(b, t, Plan.cut.next(nw, c))) : {x <> t == _ : List<&2, Plan.Clip>}
      %go_cover(b, t, Plan.cut.next(nw, c)) : {x <> t == x <> _ : List<&2, Plan.Clip>}
      {==}

def fin_cover(r: Plan.Cut()) -> {joined(r) == Laws.flat(Plan.cut.fin(r)) : List<&2, Plan.Clip>}:
  (cur, rest) = r
  match cur:
    case Nil{}:
      {==}
    case h <> t:
      {==}

def Laws.cut_cover(b, xs):
  %fin_cover(Plan.cut.go(b, xs, 0n)) : {_ == xs : List<&2, Plan.Clip>}
  %go_cover(b, xs, 0n) : {_ == xs : List<&2, Plan.Clip>}
  {==}

# Cut: no empty batch
# -------------------

# the batches a cut in progress has closed are all filled
def Closed.filled(r: Plan.Cut()) -> Type:
  (cur, rest) = r
  Laws.Filled(rest)

def step_filled(-x: Plan.Clip, nw: Bool, r: Plan.Cut()) -> Closed.filled(r) -> Closed.filled(Plan.cut.step(x, nw, r)):
  match nw:
    case True{}:
      (cur, rest) = r
      f => ({==}, f)
    case False{}:
      (cur, rest) = r
      f => f

def go_filled(+b: U32, xs: List<&2, Plan.Clip>, +c: Nat) -> Closed.filled(Plan.cut.go(b, xs, c)):
  match xs:
    case Nil{}:
      Unit{}
    case +x <> +t:
      +nw = Plan.starts(b, c, Plan.Clip.s(x))
      step_filled(x, nw, Plan.cut.go(b, t, Plan.cut.next(nw, c)), go_filled(b, t, Plan.cut.next(nw, c)))

def fin_filled(r: Plan.Cut()) -> Closed.filled(r) -> Laws.Filled(Plan.cut.fin(r)):
  (cur, rest) = r
  match cur:
    case Nil{}:
      f => f
    case h <> t:
      f => ({==}, f)

def cut_filled(+b: U32, +xs: List<&2, Plan.Clip>) -> Laws.Filled(Plan.cut(b, xs)):
  fin_filled(Plan.cut.go(b, xs, 0n), go_filled(b, xs, 0n))

def Laws.pool_filled(b, xs):
  cut_filled(b, Plan.sort(xs))

# Arithmetic and logic
# --------------------

def add_zero(a: Nat) -> {a == Nat.add(a, 0n) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_zero(p) : {1n+p == 1n+_ : Nat}
      {==}

def add_succ(a: Nat, -b: Nat) -> {1n+Nat.add(a, b) == Nat.add(a, 1n+b) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_succ(p, b) : {2n+Nat.add(p, b) == 1n+_ : Nat}
      {==}

# True{} is not False{}
def Bool.disc(b: Bool) -> Type:
  match b:
    case True{}:
      Empty
    case False{}:
      Unit

def false_true(e: {False{} == True{} : Bool}) -> Empty:
  %e : Bool.disc(_)
  Unit{}

# Cut: greedy
# -----------

# a clip that starts a new batch breaks the batch before it
def pos_breaks(+b: U32, +c: Nat, -s: U32, -i: U32, z: Bool) ->
  Laws.Holds(Plan.starts.pos(b, c, s, z)) -> Laws.Holds(Bool.and(Bool.not(z), Nat.is_gt(1n+c, U32.to_nat(U32.div(b, s))))):
  match z:
    case True{}:
      e => Empty.absurd({False{} == True{} : Bool}, false_true(e))
    case False{}:
      e => e

def starts_breaks(+b: U32, c: Nat, +x: Plan.Clip) ->
  Laws.Holds(Plan.starts(b, c, Plan.Clip.s(x))) -> Laws.Holds(Laws.breaks(b, c, x)):
  match c x:
    case 0n Plan.Clip{s, i}:
      e => Empty.absurd(Laws.Holds(Laws.breaks(b, 0n, Plan.Clip{s, i})), false_true(e))
    case 1n+p Plan.Clip{+s, +i}:
      pos_breaks(b, 1n+p, s, i, U32.is_zero(s))

# A cut in progress after a batch of c clips: its closed batches are greedy, and the batch under
# way (c clips, then cur) is broken by the first clip of the first closed batch.
def Inv(+b: U32, +c: Nat, r: Plan.Cut()) -> Type:
  (cur, +rest) = r
  Laws.Greedy(b, rest) & Laws.Greedy.next(b, Nat.add(c, Laws.len(cur)), rest)

def open_greedy(+b: U32, +c: Nat, -x: Plan.Clip, -cur: List<&2, Plan.Clip>, -rest: List<&2, List<&2, Plan.Clip>>,
  ev: Laws.Holds(Laws.breaks(b, c, x)), g: Laws.Greedy(b, rest) & Laws.Greedy.next(b, Nat.add(1n, Laws.len(cur)), rest))
  -> Inv(b, c, (Nil{}, (x <> cur) <> rest)):
  (g1, g2) = g
  %add_zero(c) : (Laws.Greedy.next(b, 1n+Laws.len(cur), rest) & Laws.Greedy(b, rest)) & Laws.Holds(Laws.breaks(b, _, x))
  ((g2, g1), ev)

def join_greedy(+b: U32, +c: Nat, -x: Plan.Clip, -cur: List<&2, Plan.Clip>, -rest: List<&2, List<&2, Plan.Clip>>,
  g: Laws.Greedy(b, rest) & Laws.Greedy.next(b, Nat.add(1n+c, Laws.len(cur)), rest))
  -> Inv(b, c, (x <> cur, rest)):
  %add_succ(c, Laws.len(cur)) : Laws.Greedy(b, rest) & Laws.Greedy.next(b, _, rest)
  g

def step_greedy(+b: U32, +c: Nat, -x: Plan.Clip, nw: Bool, r: Plan.Cut(),
  ev: Laws.Holds(nw) -> Laws.Holds(Laws.breaks(b, c, x)))
  -> Inv(b, Plan.cut.next(nw, c), r) -> Inv(b, c, Plan.cut.step(x, nw, r)):
  match nw:
    case True{}:
      (cur, rest) = r
      g => open_greedy(b, c, x, cur, rest, ev({==}), g)
    case False{}:
      (cur, rest) = r
      g => join_greedy(b, c, x, cur, rest, g)

def go_greedy(+b: U32, xs: List<&2, Plan.Clip>, +c: Nat) -> Inv(b, c, Plan.cut.go(b, xs, c)):
  match xs:
    case Nil{}:
      (Unit{}, Unit{})
    case +x <> +t:
      +nw = Plan.starts(b, c, Plan.Clip.s(x))
      step_greedy(b, c, x, nw, Plan.cut.go(b, t, Plan.cut.next(nw, c)), starts_breaks(b, c, x),
        go_greedy(b, t, Plan.cut.next(nw, c)))

def swap(-A: Type, -B: Type, p: A & B) -> B & A:
  (a, b) = p
  (b, a)

def first(-A: Type, -B: Type, p: A & B) -> A:
  (a, b) = p
  a

def fin_greedy(+b: U32, r: Plan.Cut()) -> Inv(b, 0n, r) -> Laws.Greedy(b, Plan.cut.fin(r)):
  (cur, +rest) = r
  match cur:
    case Nil{}:
      g => first(Laws.Greedy(b, rest), Laws.Greedy.next(b, 0n, rest), g)
    case +h <> +t:
      g => swap(Laws.Greedy(b, rest), Laws.Greedy.next(b, Laws.len(h <> t), rest), g)

def cut_greedy(+b: U32, +xs: List<&2, Plan.Clip>) -> Laws.Greedy(b, Plan.cut(b, xs)):
  fin_greedy(b, Plan.cut.go(b, xs, 0n), go_greedy(b, xs, 0n))

def Laws.pool_greedy(b, xs):
  cut_greedy(b, Plan.sort(xs))

# Lists
# -----

def app_assoc(-A: Data, xs: List<&2, A>, -ys: List<&2, A>, -zs: List<&2, A>) ->
  {List.append(&2, A, xs, List.append(&2, A, ys, zs)) == List.append(&2, A, List.append(&2, A, xs, ys), zs) : List<&2, A>}:
  match xs:
    case Nil{}:
      {==}
    case h <> t:
      %app_assoc(A, t, ys, zs) : {h <> List.append(&2, A, t, List.append(&2, A, ys, zs)) == h <> _ : List<&2, A>}
      {==}

def take_drop(-A: Data, xs: List<&2, A>, n: Nat) ->
  {List.append(&2, A, List.take(&2, A, xs, n), List.drop(&2, A, xs, n)) == xs : List<&2, A>}:
  match xs n:
    case Nil{} _:
      {==}
    case h <> t 0n:
      {==}
    case h <> +t 1n+p:
      %take_drop(A, t, p) : {h <> List.append(&2, A, List.take(&2, A, t, p), List.drop(&2, A, t, p)) == h <> _ : List<&2, A>}
      {==}

# Parallel driver
# ---------------

def cut_app(+b: U32, xs: List<&2, List<&2, Plan.Clip>>, +ys: List<&2, List<&2, Plan.Clip>>) ->
  {List.append(&2, List<&2, Plan.Clip>, Plan.pools.cut(b, xs), Plan.pools.cut(b, ys))
    == Plan.pools.cut(b, List.append(&2, List<&2, Plan.Clip>, xs, ys)) : List<&2, List<&2, Plan.Clip>>}:
  match xs:
    case Nil{}:
      {==}
    case +h <> +t:
      +ph = Plan.pool(b, h)
      +ct = Plan.pools.cut(b, t)
      +cy = Plan.pools.cut(b, ys)
      %cut_app(b, t, ys) : {List.append(&2, List<&2, Plan.Clip>, List.append(&2, List<&2, Plan.Clip>, ph, ct), cy)
        == List.append(&2, List<&2, Plan.Clip>, ph, _) : List<&2, List<&2, Plan.Clip>>}
      %app_assoc(List<&2, Plan.Clip>, ph, ct, cy) : {_ == List.append(&2, List<&2, Plan.Clip>, ph, List.append(&2, List<&2, Plan.Clip>, ct, cy)) : List<&2, List<&2, Plan.Clip>>}
      {==}

def solve_leaves(+b: U32, t: Plan.Tree) -> {Plan.pools.cut(b, Laws.leaves(t)) == Laws.leaves(Plan.solve(b, t)) : List<&2, List<&2, Plan.Clip>>}:
  match t:
    case Plan.TLeaf{xss}:
      {==}
    case Plan.TNode{+l, +r}:
      +ll = Laws.leaves(l)
      +lr = Laws.leaves(r)
      %solve_leaves(b, l) : {Plan.pools.cut(b, List.append(&2, List<&2, Plan.Clip>, ll, lr))
        == List.append(&2, List<&2, Plan.Clip>, _, Laws.leaves(Plan.solve(b, r))) : List<&2, List<&2, Plan.Clip>>}
      %solve_leaves(b, r) : {Plan.pools.cut(b, List.append(&2, List<&2, Plan.Clip>, ll, lr))
        == List.append(&2, List<&2, Plan.Clip>, Plan.pools.cut(b, ll), _) : List<&2, List<&2, Plan.Clip>>}
      %cut_app(b, ll, lr) : {_ == List.append(&2, List<&2, Plan.Clip>, Plan.pools.cut(b, ll), Plan.pools.cut(b, lr)) : List<&2, List<&2, Plan.Clip>>}
      {==}

def tree_leaves(+d: Nat, +k: Nat, +xss: List<&2, List<&2, Plan.Clip>>) -> {xss == Laws.leaves(Plan.tree(d, k, xss)) : List<&2, List<&2, Plan.Clip>>}:
  match d:
    case 0n:
      {==}
    case 1n+p:
      +h = Plan.half(k)
      +tk = List.take(&2, List<&2, Plan.Clip>, xss, h)
      +dr = List.drop(&2, List<&2, Plan.Clip>, xss, h)
      %tree_leaves(p, h, tk) : {xss == List.append(&2, List<&2, Plan.Clip>, _, Laws.leaves(Plan.tree(p, Nat.sub(k, h), dr))) : List<&2, List<&2, Plan.Clip>>}
      %tree_leaves(p, Nat.sub(k, h), dr) : {xss == List.append(&2, List<&2, Plan.Clip>, tk, _) : List<&2, List<&2, Plan.Clip>>}
      %take_drop(List<&2, Plan.Clip>, xss, h) : {_ == List.append(&2, List<&2, Plan.Clip>, tk, dr) : List<&2, List<&2, Plan.Clip>>}
      {==}

def Laws.solve_spec(b, d, k, xss):
  %solve_leaves(b, Plan.tree(d, k, xss)) : {_ == Plan.pools.cut(b, xss) : List<&2, List<&2, Plan.Clip>>}
  %tree_leaves(d, k, xss) : {Plan.pools.cut(b, _) == Plan.pools.cut(b, xss) : List<&2, List<&2, Plan.Clip>>}
  {==}

# Sort: permutation
# -----------------
# Every step of Base's List.sort (singleton runs, passes of pairwise merges, each merge an
# accumulator reversed onto the rest) keeps each clip's count, whatever its fuel, so the proof
# needs no fact about the order: counts move past one another with bump_add and add_bump.

def add_succ.r(a: Nat, -b: Nat) -> {Nat.add(a, 1n+b) == 1n+Nat.add(a, b) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_succ(p, b) : {1n+_ == 2n+Nat.add(p, b) : Nat}
      {==}

def add_assoc(a: Nat, -b: Nat, -c: Nat) -> {Nat.add(a, Nat.add(b, c)) == Nat.add(Nat.add(a, b), c) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_assoc(p, b, c) : {1n+Nat.add(p, Nat.add(b, c)) == 1n+_ : Nat}
      {==}

def bump_add(e: Bool, -n: Nat, -m: Nat) -> {Nat.add(Laws.bump(e, n), m) == Laws.bump(e, Nat.add(n, m)) : Nat}:
  match e:
    case True{}:
      {==}
    case False{}:
      {==}

def bump_add.r(e: Bool, -n: Nat, -m: Nat) -> {Laws.bump(e, Nat.add(n, m)) == Nat.add(Laws.bump(e, n), m) : Nat}:
  match e:
    case True{}:
      {==}
    case False{}:
      {==}

def add_bump(+n: Nat, e: Bool, -m: Nat) -> {Laws.bump(e, Nat.add(n, m)) == Nat.add(n, Laws.bump(e, m)) : Nat}:
  match e:
    case True{}:
      add_succ(n, m)
    case False{}:
      {==}

# the counts of the runs xss, summed
def counts(+y: Plan.Clip, xss: List<&2, List<&2, Plan.Clip>>) -> Nat:
  match xss:
    case Nil{}:
      0n
    case h <> t:
      Nat.add(Laws.count(y, h), counts(y, t))

def Merge() -> Type:
  List<&2, Plan.Clip> & List<&2, Plan.Clip> & List<&2, Plan.Clip>

# the counts of a merge in progress: its accumulator and its two inputs
def count3(+y: Plan.Clip, st: Merge()) -> Nat:
  (acc, xs, ys) = st
  Nat.add(Laws.count(y, acc), Nat.add(Laws.count(y, xs), Laws.count(y, ys)))

def count_app(+y: Plan.Clip, xs: List<&2, Plan.Clip>, +ys: List<&2, Plan.Clip>) ->
  {Nat.add(Laws.count(y, xs), Laws.count(y, ys)) == Laws.count(y, List.append(&2, Plan.Clip, xs, ys)) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case +h <> +t:
      +e = Laws.same(y, h)
      %count_app(y, t, ys) : {Nat.add(Laws.bump(e, Laws.count(y, t)), Laws.count(y, ys)) == Laws.bump(e, _) : Nat}
      bump_add(e, Laws.count(y, t), Laws.count(y, ys))

def count_rev(+y: Plan.Clip, xs: List<&2, Plan.Clip>, +acc: List<&2, Plan.Clip>) ->
  {Nat.add(Laws.count(y, xs), Laws.count(y, acc)) == Laws.count(y, List.reverse.go(&2, Plan.Clip, xs, acc)) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case +h <> +t:
      +e = Laws.same(y, h)
      +ct = Laws.count(y, t)
      +ca = Laws.count(y, acc)
      %count_rev(y, t, h <> acc) : {Nat.add(Laws.bump(e, ct), ca) == _ : Nat}
      %add_bump(ct, e, ca) : {Nat.add(Laws.bump(e, ct), ca) == _ : Nat}
      bump_add(e, ct, ca)

def count_concat(+y: Plan.Clip, xss: List<&2, List<&2, Plan.Clip>>) ->
  {counts(y, xss) == Laws.count(y, List.concat(&2, Plan.Clip, xss)) : Nat}:
  match xss:
    case Nil{}:
      {==}
    case +h <> +t:
      %Equal.sym(Nat, counts(y, t), Laws.count(y, List.concat(&2, Plan.Clip, t)), count_concat(y, t)) : {Nat.add(Laws.count(y, h), _) == Laws.count(y, List.append(&2, Plan.Clip, h, List.concat(&2, Plan.Clip, t))) : Nat}
      count_app(y, h, List.concat(&2, Plan.Clip, t))

def count_step(+y: Plan.Clip, +acc: List<&2, Plan.Clip>, +x: Plan.Clip, +xt: List<&2, Plan.Clip>, +z: Plan.Clip,
  +zt: List<&2, Plan.Clip>, le: Bool) ->
  {Nat.add(Laws.count(y, acc), Nat.add(Laws.count(y, x <> xt), Laws.count(y, z <> zt)))
    == count3(y, List.merge.step(Plan.Clip, acc, x, xt, z, zt, le)) : Nat}:
  match le:
    case False{}:
      +ca = Laws.count(y, acc)
      +ex = Laws.same(y, x)
      +ez = Laws.same(y, z)
      +cx = Laws.count(y, xt)
      +cz = Laws.count(y, zt)
      %add_bump(Laws.bump(ex, cx), ez, cz) : {Nat.add(ca, _) == Nat.add(Laws.bump(ez, ca), Nat.add(Laws.bump(ex, cx), cz)) : Nat}
      %add_bump(ca, ez, Nat.add(Laws.bump(ex, cx), cz)) : {_ == Nat.add(Laws.bump(ez, ca), Nat.add(Laws.bump(ex, cx), cz)) : Nat}
      bump_add.r(ez, ca, Nat.add(Laws.bump(ex, cx), cz))
    case True{}:
      +ca = Laws.count(y, acc)
      +ex = Laws.same(y, x)
      +ez = Laws.same(y, z)
      +cx = Laws.count(y, xt)
      +cz = Laws.count(y, zt)
      +bz = Laws.bump(ez, cz)
      %bump_add.r(ex, cx, bz) : {Nat.add(ca, _) == Nat.add(Laws.bump(ex, ca), Nat.add(cx, bz)) : Nat}
      %add_bump(ca, ex, Nat.add(cx, bz)) : {_ == Nat.add(Laws.bump(ex, ca), Nat.add(cx, bz)) : Nat}
      bump_add.r(ex, ca, Nat.add(cx, bz))

def count_merge(+y: Plan.Clip, fuel: Nat, st: Merge()) ->
  {count3(y, st) == Laws.count(y, List.merge.go(~Plan.Clip, ~Plan.Clip.le, fuel, st)) : Nat}:
  match fuel:
    case 0n:
      (acc0, xs0, ys0) = st
      +acc = {acc0 : List<&2, Plan.Clip>}
      +xs = {xs0 : List<&2, Plan.Clip>}
      +ys = {ys0 : List<&2, Plan.Clip>}
      +ca = Laws.count(y, acc)
      %count_rev(y, acc, List.append(&2, Plan.Clip, xs, ys)) : {Nat.add(ca, Nat.add(Laws.count(y, xs), Laws.count(y, ys))) == _ : Nat}
      %count_app(y, xs, ys) : {Nat.add(ca, Nat.add(Laws.count(y, xs), Laws.count(y, ys))) == Nat.add(ca, _) : Nat}
      {==}
    case 1n+f:
      (acc0, xs, ys) = st
      match xs ys:
        case Nil{} Nil{}:
          count_rev(y, acc0, Nil{})
        case Nil{} +z <> +zt:
          count_rev(y, acc0, z <> zt)
        case +x <> +xt Nil{}:
          +acc = acc0
          +ca = Laws.count(y, acc)
          +cx = Laws.count(y, x <> xt)
          %count_rev(y, acc, x <> xt) : {Nat.add(ca, Nat.add(cx, 0n)) == _ : Nat}
          %add_zero(cx) : {Nat.add(ca, _) == Nat.add(ca, cx) : Nat}
          {==}
        case +x <> +xt +z <> +zt:
          +acc = acc0
          %count_merge(y, f, List.merge.step(Plan.Clip, acc, x, xt, z, zt, Plan.Clip.le(x, z))) : {count3(y, (acc, x <> xt, z <> zt)) == _ : Nat}
          count_step(y, acc, x, xt, z, zt, Plan.Clip.le(x, z))

def count_runs(+y: Plan.Clip, xs: List<&2, Plan.Clip>) ->
  {Laws.count(y, xs) == counts(y, List.sort.runs(Plan.Clip, xs)) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case +h <> +t:
      +e = Laws.same(y, h)
      %count_runs(y, t) : {Laws.bump(e, Laws.count(y, t)) == Nat.add(Laws.bump(e, 0n), _) : Nat}
      bump_add.r(e, 0n, Laws.count(y, t))

def count_pass(+y: Plan.Clip, +n: Nat, runs: List<&2, List<&2, Plan.Clip>>) ->
  {counts(y, runs) == counts(y, List.sort.pass(~Plan.Clip, ~Plan.Clip.le, n, runs)) : Nat}:
  match runs:
    case Nil{}:
      {==}
    case r <> Nil{}:
      {==}
    case +r1 <> +r2 <> +rest:
      +c1 = Laws.count(y, r1)
      +c2 = Laws.count(y, r2)
      +cs = counts(y, rest)
      %count_merge(y, n, (Nil{}, r1, r2)) : {Nat.add(c1, Nat.add(c2, cs))
        == Nat.add(_, counts(y, List.sort.pass(~Plan.Clip, ~Plan.Clip.le, n, rest))) : Nat}
      %count_pass(y, n, rest) : {Nat.add(c1, Nat.add(c2, cs)) == Nat.add(Nat.add(c1, c2), _) : Nat}
      add_assoc(c1, c2, cs)

def count_sortgo(+y: Plan.Clip, fuel: Nat, +n: Nat, runs: List<&2, List<&2, Plan.Clip>>) ->
  {counts(y, runs) == Laws.count(y, List.sort.go(~Plan.Clip, ~Plan.Clip.le, fuel, n, runs)) : Nat}:
  match fuel:
    case 0n:
      count_concat(y, runs)
    case 1n+f:
      match runs:
        case Nil{}:
          {==}
        case +r <> Nil{}:
          %add_zero(Laws.count(y, r)) : {_ == Laws.count(y, r) : Nat}
          {==}
        case +r1 <> +r2 <> +rest:
          +rs = List.sort.pass(~Plan.Clip, ~Plan.Clip.le, n, r1 <> r2 <> rest)
          %count_sortgo(y, f, n, rs) : {counts(y, r1 <> r2 <> rest) == _ : Nat}
          count_pass(y, n, r1 <> r2 <> rest)

def Laws.sort_perm(y, xs):
  match xs:
    case Nil{}:
      {==}
    case +h <> +t:
      +n = List.length(&2, Plan.Clip, h <> t)
      %count_sortgo(y, n, n, List.sort.runs(Plan.Clip, h <> t)) : {_ == Laws.count(y, h <> t) : Nat}
      %count_runs(y, h <> t) : {_ == Laws.count(y, h <> t) : Nat}
      {==}

def Laws.pool_cover(b, y, xs):
  +zs = Plan.sort(xs)
  %Equal.sym(List<&2, Plan.Clip>, Laws.flat(Plan.cut(b, zs)), zs, Laws.cut_cover(b, zs)) : {Laws.count(y, _) == Laws.count(y, xs) : Nat}
  Laws.sort_perm(y, xs)

# Sort: order
# -----------
# Base's List.sort is a bottom-up merge sort: singleton runs, then up to n passes, each merging
# neighbouring runs with fuel n. Its output is sorted because (a) the key order is total (Word.cmp
# flips when its arguments swap), (b) a merge with enough fuel keeps a bound and sortedness, and (c)
# n passes are enough: each pass leaves at most as many runs as half, rounded up, and the runs'
# lengths always sum to n, so each merge has fuel for both of its runs.

def true_false(e: {True{} == False{} : Bool}) -> Empty:
  %e : Bool.disc(Bool.not(_))
  Unit{}

def flip(c: Cmp) -> Cmp:
  match c:
    case LT{}:
      GT{}
    case EQ{}:
      EQ{}
    case GT{}:
      LT{}

def fin_flip(t: Cmp, ab: Bool, bb: Bool) -> {Word.cmp.fin(bb, ab, flip(t)) == flip(Word.cmp.fin(ab, bb, t)) : Cmp}:
  match t:
    case LT{}:
      {==}
    case GT{}:
      {==}
    case EQ{}:
      match ab bb:
        case True{} True{}:
          {==}
        case True{} False{}:
          {==}
        case False{} True{}:
          {==}
        case False{} False{}:
          {==}

def word_flip(+n: Nat, a: Word(n), b: Word(n)) -> {flip(Word.cmp(n, a, b)) == Word.cmp(n, b, a) : Cmp}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match a b:
        case WCon{+ab, +at} WCon{+bb, +bt}:
          %word_flip(p, at, bt) : {flip(Word.cmp.fin(ab, bb, Word.cmp(p, at, bt))) == Word.cmp.fin(bb, ab, _) : Cmp}
          Equal.sym(Cmp, Word.cmp.fin(bb, ab, flip(Word.cmp(p, at, bt))), flip(Word.cmp.fin(ab, bb, Word.cmp(p, at, bt))),
            fin_flip(Word.cmp(p, at, bt), ab, bb))

def u32_flip(x: U32, y: U32) -> {flip(U32.cmp(x, y)) == U32.cmp(y, x) : Cmp}:
  match x y:
    case U32{a} U32{b}:
      word_flip(32n, a, b)

def le_flip(d: Cmp) -> {Cmp.is_le(d) == False{} : Bool} -> Laws.Holds(Cmp.is_le(flip(d))):
  match d:
    case LT{}:
      e => Empty.absurd({False{} == True{} : Bool}, true_false(e))
    case EQ{}:
      e => Empty.absurd({True{} == True{} : Bool}, true_false(e))
    case GT{}:
      e => {==}

def total.cmp(c: Cmp, +i: U32, +j: U32) -> {Plan.Clip.le.cmp(c, i, j) == False{} : Bool} -> Laws.Holds(Laws.key.le.cmp(flip(c), j, i)):
  match c:
    case LT{}:
      e => Empty.absurd(Laws.Holds(Laws.key.le.cmp(GT{}, j, i)), true_false(e))
    case EQ{}:
      %u32_flip(i, j) : {U32.is_le(i, j) == False{} : Bool} -> Laws.Holds(Cmp.is_le(_))
      le_flip(U32.cmp(i, j))
    case GT{}:
      e => {==}

# the key order is total: when x <= y fails, y <= x
def total(x: Plan.Clip, y: Plan.Clip) -> {Plan.Clip.le(x, y) == False{} : Bool} -> Laws.Holds(Laws.key.le(y, x)):
  match x y:
    case Plan.Clip{+s, +i} Plan.Clip{+t, +j}:
      %u32_flip(s, t) : {Plan.Clip.le.cmp(U32.cmp(s, t), i, j) == False{} : Bool} -> Laws.Holds(Laws.key.le.cmp(_, j, i))
      total.cmp(U32.cmp(s, t), i, j)

def key_eq.cmp(c: Cmp, -i: U32, -j: U32) -> {Plan.Clip.le.cmp(c, i, j) == Laws.key.le.cmp(c, i, j) : Bool}:
  match c:
    case LT{}:
      {==}
    case EQ{}:
      {==}
    case GT{}:
      {==}

# the planner's order is the laws' key order
def key_eq(x: Plan.Clip, y: Plan.Clip) -> {Plan.Clip.le(x, y) == Laws.key.le(x, y) : Bool}:
  match x y:
    case Plan.Clip{+s, i} Plan.Clip{+t, j}:
      key_eq.cmp(U32.cmp(s, t), i, j)

def key_le(+x: Plan.Clip, +y: Plan.Clip, e: {Plan.Clip.le(x, y) == True{} : Bool}) -> Laws.Holds(Laws.key.le(x, y)):
  %key_eq(x, y) : {_ == True{} : Bool}
  e

# A merge without the accumulator: List.merge.go(fuel, (acc, xs, ys)) is acc reversed onto it.
def mrg.pick(c: Bool, x: Plan.Clip, y: Plan.Clip, a: List<&2, Plan.Clip>, b: List<&2, Plan.Clip>) -> List<&2, Plan.Clip>:
  match c:
    case True{}:
      x <> a
    case False{}:
      y <> b

def mrg(+fuel: Nat, xs: List<&2, Plan.Clip>, ys: List<&2, Plan.Clip>) -> List<&2, Plan.Clip>:
  match fuel:
    case 0n:
      List.append(&2, Plan.Clip, xs, ys)
    case 1n+f:
      match xs ys:
        case Nil{} _:
          ys
        case x <> xt Nil{}:
          x <> xt
        case +x <> +xt +y <> +yt:
          mrg.pick(Plan.Clip.le(x, y), x, y, mrg(f, xt, y <> yt), mrg(f, x <> xt, yt))

def mrg3(fuel: Nat, st: Merge()) -> List<&2, Plan.Clip>:
  (acc, xs, ys) = st
  List.reverse.go(&2, Plan.Clip, acc, mrg(fuel, xs, ys))

def merge_pick(c: Bool, -f: Nat, -acc: List<&2, Plan.Clip>, -x: Plan.Clip, -xt: List<&2, Plan.Clip>, -y: Plan.Clip,
  -yt: List<&2, Plan.Clip>,
  h1: {mrg3(f, (x <> acc, xt, y <> yt)) == List.merge.go(~Plan.Clip, ~Plan.Clip.le, f, (x <> acc, xt, y <> yt)) : List<&2, Plan.Clip>},
  h2: {mrg3(f, (y <> acc, x <> xt, yt)) == List.merge.go(~Plan.Clip, ~Plan.Clip.le, f, (y <> acc, x <> xt, yt)) : List<&2, Plan.Clip>})
  -> {List.reverse.go(&2, Plan.Clip, acc, mrg.pick(c, x, y, mrg(f, xt, y <> yt), mrg(f, x <> xt, yt)))
    == List.merge.go(~Plan.Clip, ~Plan.Clip.le, f, List.merge.step(Plan.Clip, acc, x, xt, y, yt, c)) : List<&2, Plan.Clip>}:
  match c:
    case True{}:
      h1
    case False{}:
      h2

def merge_rev(+fuel: Nat, st: Merge()) -> {mrg3(fuel, st) == List.merge.go(~Plan.Clip, ~Plan.Clip.le, fuel, st) : List<&2, Plan.Clip>}:
  match fuel:
    case 0n:
      (acc, xs, ys) = st
      {==}
    case 1n+f:
      (acc0, xs, ys) = st
      match xs ys:
        case Nil{} Nil{}:
          {==}
        case Nil{} y <> yt:
          {==}
        case x <> xt Nil{}:
          {==}
        case +x <> +xt +y <> +yt:
          +acc = {acc0 : List<&2, Plan.Clip>}
          merge_pick(Plan.Clip.le(x, y), f, acc, x, xt, y, yt, merge_rev(f, (x <> acc, xt, y <> yt)),
            merge_rev(f, (y <> acc, x <> xt, yt)))

# Nat order
# ---------

def le_zero(b: Nat) -> Laws.Holds(Nat.is_le(0n, b)):
  match b:
    case 0n:
      {==}
    case 1n+p:
      {==}

def le_refl(a: Nat) -> Laws.Holds(Nat.is_le(a, a)):
  match a:
    case 0n:
      {==}
    case 1n+p:
      le_refl(p)

def le_step(x: Nat, y: Nat) -> Laws.Holds(Nat.is_le(x, y)) -> Laws.Holds(Nat.is_le(x, 1n+y)):
  match x y:
    case 0n _:
      e => le_zero(1n+y)
    case 1n+p 0n:
      e => Empty.absurd(Laws.Holds(Nat.is_le(1n+p, 1n)), false_true(e))
    case 1n+p 1n+q:
      le_step(p, q)

def le_trans(a: Nat, b: Nat, c: Nat) -> Laws.Holds(Nat.is_le(a, b)) -> Laws.Holds(Nat.is_le(b, c)) -> Laws.Holds(Nat.is_le(a, c)):
  match a b c:
    case 0n _ _:
      e1 => e2 => le_zero(c)
    case 1n+p 0n _:
      e1 => e2 => Empty.absurd(Laws.Holds(Nat.is_le(1n+p, c)), false_true(e1))
    case 1n+p 1n+q 0n:
      e1 => e2 => Empty.absurd(Laws.Holds(Nat.is_le(1n+p, 0n)), false_true(e2))
    case 1n+p 1n+q 1n+r:
      le_trans(p, q, r)

def le_add(a: Nat, +b: Nat) -> Laws.Holds(Nat.is_le(a, Nat.add(a, b))):
  match a:
    case 0n:
      le_zero(b)
    case 1n+p:
      le_add(p, b)

def le_add_l(+a: Nat, +b: Nat) -> Laws.Holds(Nat.is_le(b, Nat.add(a, b))):
  match a:
    case 0n:
      le_refl(b)
    case 1n+p:
      le_step(b, Nat.add(p, b), le_add_l(p, b))

# Merging sorted runs
# -------------------

# fuel enough for both inputs
def Fuel(+f: Nat, +xs: List<&2, Plan.Clip>, +ys: List<&2, Plan.Clip>) -> Type:
  Laws.Holds(Nat.is_le(Nat.add(Laws.len(xs), Laws.len(ys)), f))

def fuel_r(-f: Nat, +a: Nat, +b: Nat) -> Laws.Holds(Nat.is_le(Nat.add(a, 1n+b), f)) -> Laws.Holds(Nat.is_le(1n+Nat.add(a, b), f)):
  %add_succ.r(a, b) : Laws.Holds(Nat.is_le(Nat.add(a, 1n+b), f)) -> Laws.Holds(Nat.is_le(_, f))
  h => h

def mrg_from.yes(-lo: Plan.Clip, +x: Plan.Clip, +xt: List<&2, Plan.Clip>, +y: Plan.Clip, +yt: List<&2, Plan.Clip>, +g: Nat,
  r1: Fuel(g, xt, y <> yt) -> Laws.Sorted.from(x, xt) -> Laws.Sorted.from(x, y <> yt) -> Laws.Sorted.from(x, mrg(g, xt, y <> yt)),
  e: {Plan.Clip.le(x, y) == True{} : Bool}, fu: Fuel(1n+g, x <> xt, y <> yt), sx: Laws.Sorted.from(lo, x <> xt),
  sy: Laws.Sorted.from(lo, y <> yt)) -> Laws.Sorted.from(lo, x <> mrg(g, xt, y <> yt)):
  (lx, sxt) = sx
  (ly, syt) = sy
  (lx, r1(fu, sxt, (key_le(x, y, e), syt)))

def mrg_from.no(-lo: Plan.Clip, +x: Plan.Clip, +xt: List<&2, Plan.Clip>, +y: Plan.Clip, +yt: List<&2, Plan.Clip>, +g: Nat,
  r2: Fuel(g, x <> xt, yt) -> Laws.Sorted.from(y, x <> xt) -> Laws.Sorted.from(y, yt) -> Laws.Sorted.from(y, mrg(g, x <> xt, yt)),
  e: {Plan.Clip.le(x, y) == False{} : Bool}, fu: Fuel(1n+g, x <> xt, y <> yt), sx: Laws.Sorted.from(lo, x <> xt),
  sy: Laws.Sorted.from(lo, y <> yt)) -> Laws.Sorted.from(lo, y <> mrg(g, x <> xt, yt)):
  (lx, sxt) = sx
  (ly, syt) = sy
  (ly, r2(fuel_r(g, Laws.len(xt), Laws.len(yt), fu), (total(x, y, e), sxt), syt))

def mrg_from.pick(c: Bool, -lo: Plan.Clip, +x: Plan.Clip, +xt: List<&2, Plan.Clip>, +y: Plan.Clip, +yt: List<&2, Plan.Clip>, +g: Nat,
  r1: Fuel(g, xt, y <> yt) -> Laws.Sorted.from(x, xt) -> Laws.Sorted.from(x, y <> yt) -> Laws.Sorted.from(x, mrg(g, xt, y <> yt)),
  r2: Fuel(g, x <> xt, yt) -> Laws.Sorted.from(y, x <> xt) -> Laws.Sorted.from(y, yt) -> Laws.Sorted.from(y, mrg(g, x <> xt, yt)))
  -> {Plan.Clip.le(x, y) == c : Bool} -> Fuel(1n+g, x <> xt, y <> yt) -> Laws.Sorted.from(lo, x <> xt) -> Laws.Sorted.from(lo, y <> yt)
  -> Laws.Sorted.from(lo, mrg.pick(c, x, y, mrg(g, xt, y <> yt), mrg(g, x <> xt, yt))):
  match c:
    case True{}:
      e => fu => sx => sy => mrg_from.yes(lo, x, xt, y, yt, g, r1, e, fu, sx, sy)
    case False{}:
      e => fu => sx => sy => mrg_from.no(lo, x, xt, y, yt, g, r2, e, fu, sx, sy)

# a merge with enough fuel keeps a lower bound shared by its sorted inputs, and sorts
def mrg_from(+f: Nat, -lo: Plan.Clip, xs: List<&2, Plan.Clip>, ys: List<&2, Plan.Clip>) ->
  Fuel(f, xs, ys) -> Laws.Sorted.from(lo, xs) -> Laws.Sorted.from(lo, ys) -> Laws.Sorted.from(lo, mrg(f, xs, ys)):
  match f:
    case 0n:
      match xs ys:
        case Nil{} Nil{}:
          fu => sx => sy => Unit{}
        case Nil{} +y <> +yt:
          fu => Empty.absurd(Laws.Sorted.from(lo, Nil{}) -> Laws.Sorted.from(lo, y <> yt) -> Laws.Sorted.from(lo, y <> yt),
            false_true(fu))
        case +x <> +xt +ys2:
          fu => Empty.absurd(Laws.Sorted.from(lo, x <> xt) -> Laws.Sorted.from(lo, ys2)
            -> Laws.Sorted.from(lo, x <> List.append(&2, Plan.Clip, xt, ys2)), false_true(fu))
    case 1n+g:
      match xs ys:
        case Nil{} _:
          fu => sx => sy => sy
        case x <> xt Nil{}:
          fu => sx => sy => sx
        case +x <> +xt +y <> +yt:
          mrg_from.pick(Plan.Clip.le(x, y), lo, x, xt, y, yt, g, mrg_from(g, x, xt, y <> yt), mrg_from(g, y, x <> xt, yt), {==})

def mrg_sorted.pick(c: Bool, +x: Plan.Clip, +xt: List<&2, Plan.Clip>, +y: Plan.Clip, +yt: List<&2, Plan.Clip>, +g: Nat)
  -> {Plan.Clip.le(x, y) == c : Bool} -> Fuel(1n+g, x <> xt, y <> yt) -> Laws.Sorted(x <> xt) -> Laws.Sorted(y <> yt)
  -> Laws.Sorted(mrg.pick(c, x, y, mrg(g, xt, y <> yt), mrg(g, x <> xt, yt))):
  match c:
    case True{}:
      e => fu => sx => sy => mrg_from(g, x, xt, y <> yt, fu, sx, (key_le(x, y, e), sy))
    case False{}:
      e => fu => sx => sy => mrg_from(g, y, x <> xt, yt, fuel_r(g, Laws.len(xt), Laws.len(yt), fu), (total(x, y, e), sx), sy)

def mrg_sorted(+f: Nat, xs: List<&2, Plan.Clip>, ys: List<&2, Plan.Clip>) ->
  Fuel(f, xs, ys) -> Laws.Sorted(xs) -> Laws.Sorted(ys) -> Laws.Sorted(mrg(f, xs, ys)):
  match f:
    case 0n:
      match xs ys:
        case Nil{} Nil{}:
          fu => sx => sy => Unit{}
        case Nil{} +y <> +yt:
          fu => Empty.absurd(Laws.Sorted(Nil{}) -> Laws.Sorted(y <> yt) -> Laws.Sorted(y <> yt), false_true(fu))
        case +x <> +xt +ys2:
          fu => Empty.absurd(Laws.Sorted(x <> xt) -> Laws.Sorted(ys2) -> Laws.Sorted(x <> List.append(&2, Plan.Clip, xt, ys2)),
            false_true(fu))
    case 1n+g:
      match xs ys:
        case Nil{} _:
          fu => sx => sy => sy
        case x <> xt Nil{}:
          fu => sx => sy => sx
        case +x <> +xt +y <> +yt:
          mrg_sorted.pick(Plan.Clip.le(x, y), x, xt, y, yt, g, {==})

# Lengths
# -------

def len_app(xs: List<&2, Plan.Clip>, +ys: List<&2, Plan.Clip>) ->
  {Nat.add(Laws.len(xs), Laws.len(ys)) == Laws.len(List.append(&2, Plan.Clip, xs, ys)) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case h <> +t:
      %len_app(t, ys) : {Nat.add(Laws.len(h <> t), Laws.len(ys)) == 1n+_ : Nat}
      {==}

def len_pick(c: Bool, +x: Plan.Clip, +xt: List<&2, Plan.Clip>, +y: Plan.Clip, +yt: List<&2, Plan.Clip>, +g: Nat,
  r1: {Nat.add(Laws.len(xt), Laws.len(y <> yt)) == Laws.len(mrg(g, xt, y <> yt)) : Nat},
  r2: {Nat.add(Laws.len(x <> xt), Laws.len(yt)) == Laws.len(mrg(g, x <> xt, yt)) : Nat})
  -> {Nat.add(Laws.len(x <> xt), Laws.len(y <> yt)) == Laws.len(mrg.pick(c, x, y, mrg(g, xt, y <> yt), mrg(g, x <> xt, yt))) : Nat}:
  match c:
    case True{}:
      %r1 : {Nat.add(Laws.len(x <> xt), Laws.len(y <> yt)) == 1n+_ : Nat}
      {==}
    case False{}:
      %r2 : {Nat.add(Laws.len(x <> xt), Laws.len(y <> yt)) == 1n+_ : Nat}
      %add_succ(Laws.len(xt), Laws.len(yt)) : {1n+_ == 2n+Nat.add(Laws.len(xt), Laws.len(yt)) : Nat}
      {==}

def len_mrg(+f: Nat, xs: List<&2, Plan.Clip>, ys: List<&2, Plan.Clip>) ->
  {Nat.add(Laws.len(xs), Laws.len(ys)) == Laws.len(mrg(f, xs, ys)) : Nat}:
  match f:
    case 0n:
      len_app(xs, ys)
    case 1n+g:
      match xs ys:
        case Nil{} _:
          {==}
        case +x <> +xt Nil{}:
          %add_zero(Laws.len(x <> xt)) : {_ == Laws.len(x <> xt) : Nat}
          {==}
        case +x <> +xt +y <> +yt:
          len_pick(Plan.Clip.le(x, y), x, xt, y, yt, g, len_mrg(g, xt, y <> yt), len_mrg(g, x <> xt, yt))

def app_nil(xs: List<&2, Plan.Clip>) -> {xs == List.append(&2, Plan.Clip, xs, Nil{}) : List<&2, Plan.Clip>}:
  match xs:
    case Nil{}:
      {==}
    case h <> +t:
      %app_nil(t) : {h <> t == h <> _ : List<&2, Plan.Clip>}
      {==}

# Passes
# ------

def AllSorted(runs: List<&2, List<&2, Plan.Clip>>) -> Type:
  match runs:
    case Nil{}:
      Unit
    case r <> t:
      Laws.Sorted(r) & AllSorted(t)

def sumlen(runs: List<&2, List<&2, Plan.Clip>>) -> Nat:
  match runs:
    case Nil{}:
      0n
    case r <> t:
      Nat.add(Laws.len(r), sumlen(t))

def nruns(runs: List<&2, List<&2, Plan.Clip>>) -> Nat:
  List.length(&2, List<&2, Plan.Clip>, runs)

def merged(+n: Nat, +r1: List<&2, Plan.Clip>, +r2: List<&2, Plan.Clip>) -> List<&2, Plan.Clip>:
  List.merge.go(~Plan.Clip, ~Plan.Clip.le, n, (Nil{}, r1, r2))

def merged_sorted(+n: Nat, +r1: List<&2, Plan.Clip>, +r2: List<&2, Plan.Clip>) ->
  Fuel(n, r1, r2) -> Laws.Sorted(r1) -> Laws.Sorted(r2) -> Laws.Sorted(merged(n, r1, r2)):
  %merge_rev(n, (Nil{}, r1, r2)) : Fuel(n, r1, r2) -> Laws.Sorted(r1) -> Laws.Sorted(r2) -> Laws.Sorted(_)
  mrg_sorted(n, r1, r2)

def merged_len(+n: Nat, +r1: List<&2, Plan.Clip>, +r2: List<&2, Plan.Clip>) ->
  {Nat.add(Laws.len(r1), Laws.len(r2)) == Laws.len(merged(n, r1, r2)) : Nat}:
  %merge_rev(n, (Nil{}, r1, r2)) : {Nat.add(Laws.len(r1), Laws.len(r2)) == Laws.len(_) : Nat}
  len_mrg(n, r1, r2)

def le_head.go(+l1: Nat, +l2: Nat, +s: Nat) -> Laws.Holds(Nat.is_le(Nat.add(l1, l2), Nat.add(l1, Nat.add(l2, s)))):
  %Equal.sym(Nat, Nat.add(l1, Nat.add(l2, s)), Nat.add(Nat.add(l1, l2), s), add_assoc(l1, l2, s)) : Laws.Holds(Nat.is_le(Nat.add(l1, l2), _))
  le_add(Nat.add(l1, l2), s)


def le_head(+l1: Nat, +l2: Nat, +s: Nat, n: Nat, h: Laws.Holds(Nat.is_le(Nat.add(l1, Nat.add(l2, s)), n))) ->
  Laws.Holds(Nat.is_le(Nat.add(l1, l2), n)):
  +a = Nat.add(l1, l2)
  +whole = Nat.add(l1, Nat.add(l2, s))
  le_trans(a, whole, n, le_head.go(l1, l2, s), h)

def le_tail(+l1: Nat, +l2: Nat, +s: Nat, n: Nat, h: Laws.Holds(Nat.is_le(Nat.add(l1, Nat.add(l2, s)), n))) ->
  Laws.Holds(Nat.is_le(s, n)):
  +mid = Nat.add(l2, s)
  +whole = Nat.add(l1, mid)
  le_trans(s, whole, n, le_trans(s, mid, whole, le_add_l(l2, s), le_add_l(l1, mid)), h)

def pass_ok.cons(+n: Nat, +r1: List<&2, Plan.Clip>, +r2: List<&2, Plan.Clip>, +rest: List<&2, List<&2, Plan.Clip>>,
  ih: AllSorted(rest) -> Laws.Holds(Nat.is_le(sumlen(rest), n)) -> AllSorted(List.sort.pass(~Plan.Clip, ~Plan.Clip.le, n, rest)),
  a: AllSorted(r1 <> r2 <> rest), +h: Laws.Holds(Nat.is_le(sumlen(r1 <> r2 <> rest), n)))
  -> AllSorted(merged(n, r1, r2) <> List.sort.pass(~Plan.Clip, ~Plan.Clip.le, n, rest)):
  (s1, a2) = a
  (s2, a3) = a2
  +l1 = Laws.len(r1)
  +l2 = Laws.len(r2)
  +s = sumlen(rest)
  (merged_sorted(n, r1, r2, le_head(l1, l2, s, n, h), s1, s2), ih(a3, le_tail(l1, l2, s, n, h)))


def pass_ok(+n: Nat, runs: List<&2, List<&2, Plan.Clip>>) ->
  AllSorted(runs) -> Laws.Holds(Nat.is_le(sumlen(runs), n)) -> AllSorted(List.sort.pass(~Plan.Clip, ~Plan.Clip.le, n, runs)):
  match runs:
    case Nil{}:
      a => h => Unit{}
    case r <> Nil{}:
      a => h => a
    case +r1 <> +r2 <> +rest:
      a => h => pass_ok.cons(n, r1, r2, rest, pass_ok(n, rest), a, h)

def pass_sum(+n: Nat, runs: List<&2, List<&2, Plan.Clip>>) ->
  {sumlen(runs) == sumlen(List.sort.pass(~Plan.Clip, ~Plan.Clip.le, n, runs)) : Nat}:
  match runs:
    case Nil{}:
      {==}
    case r <> Nil{}:
      {==}
    case +r1 <> +r2 <> +rest:
      +l1 = Laws.len(r1)
      +l2 = Laws.len(r2)
      +s = sumlen(rest)
      %merged_len(n, r1, r2) : {Nat.add(l1, Nat.add(l2, s))
        == Nat.add(_, sumlen(List.sort.pass(~Plan.Clip, ~Plan.Clip.le, n, rest))) : Nat}
      %pass_sum(n, rest) : {Nat.add(l1, Nat.add(l2, s)) == Nat.add(Nat.add(l1, l2), _) : Nat}
      add_assoc(l1, l2, s)

def pass_len(+n: Nat, runs: List<&2, List<&2, Plan.Clip>>) ->
  Laws.Holds(Nat.is_le(nruns(List.sort.pass(~Plan.Clip, ~Plan.Clip.le, n, runs)), nruns(runs))):
  match runs:
    case Nil{}:
      {==}
    case r <> Nil{}:
      {==}
    case r1 <> r2 <> +rest:
      le_step(nruns(List.sort.pass(~Plan.Clip, ~Plan.Clip.le, n, rest)), nruns(rest), pass_len(n, rest))

def sortgo_ok.sum(+n: Nat, +runs: List<&2, List<&2, Plan.Clip>>, h: Laws.Holds(Nat.is_le(sumlen(runs), n))) ->
  Laws.Holds(Nat.is_le(sumlen(List.sort.pass(~Plan.Clip, ~Plan.Clip.le, n, runs)), n)):
  %pass_sum(n, runs) : Laws.Holds(Nat.is_le(_, n))
  h

def sortgo_ok.len(+f: Nat, +n: Nat, -r1: List<&2, Plan.Clip>, -r2: List<&2, Plan.Clip>, +rest: List<&2, List<&2, Plan.Clip>>,
  k: Laws.Holds(Nat.is_le(nruns(r1 <> r2 <> rest), 2n+f))) ->
  Laws.Holds(Nat.is_le(nruns(List.sort.pass(~Plan.Clip, ~Plan.Clip.le, n, r1 <> r2 <> rest)), 1n+f)):
  le_trans(nruns(List.sort.pass(~Plan.Clip, ~Plan.Clip.le, n, rest)), nruns(rest), f, pass_len(n, rest), k)

def sortgo_one(+r: List<&2, Plan.Clip>, s: Laws.Sorted(r)) -> Laws.Sorted(List.concat(&2, Plan.Clip, [r])):
  %app_nil(r) : Laws.Sorted(_)
  s

# with the runs sorted, their lengths summing to at most n, and at most fuel + 1 of them, the
# passes end in one sorted run
def sortgo_ok(+fuel: Nat, +n: Nat, runs: List<&2, List<&2, Plan.Clip>>) ->
  AllSorted(runs) -> Laws.Holds(Nat.is_le(sumlen(runs), n)) -> Laws.Holds(Nat.is_le(nruns(runs), 1n+fuel))
  -> Laws.Sorted(List.sort.go(~Plan.Clip, ~Plan.Clip.le, fuel, n, runs)):
  match fuel:
    case 0n:
      match runs:
        case Nil{}:
          a => h => k => Unit{}
        case +r <> Nil{}:
          a => h => k => sortgo_one(r, first(Laws.Sorted(r), Unit, a))
        case +r1 <> +r2 <> +rest:
          a => h => k => Empty.absurd(Laws.Sorted(List.concat(&2, Plan.Clip, r1 <> r2 <> rest)), false_true(k))
    case 1n+f:
      match runs:
        case Nil{}:
          a => h => k => Unit{}
        case +r <> Nil{}:
          a => h => k => first(Laws.Sorted(r), Unit, a)
        case +r1 <> +r2 <> +rest:
          a => +h => k => sortgo_ok(f, n, List.sort.pass(~Plan.Clip, ~Plan.Clip.le, n, r1 <> r2 <> rest),
            pass_ok(n, r1 <> r2 <> rest, a, h), sortgo_ok.sum(n, r1 <> r2 <> rest, h), sortgo_ok.len(f, n, r1, r2, rest, k))

def runs_sorted(xs: List<&2, Plan.Clip>) -> AllSorted(List.sort.runs(Plan.Clip, xs)):
  match xs:
    case Nil{}:
      Unit{}
    case h <> t:
      (Unit{}, runs_sorted(t))

def runs_sum(xs: List<&2, Plan.Clip>) -> {Laws.len(xs) == sumlen(List.sort.runs(Plan.Clip, xs)) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case h <> +t:
      %runs_sum(t) : {1n+Laws.len(t) == 1n+_ : Nat}
      {==}

def runs_len(xs: List<&2, Plan.Clip>) -> {Laws.len(xs) == nruns(List.sort.runs(Plan.Clip, xs)) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case h <> +t:
      %runs_len(t) : {1n+Laws.len(t) == 1n+_ : Nat}
      {==}

def sort_sum(+xs: List<&2, Plan.Clip>) -> Laws.Holds(Nat.is_le(sumlen(List.sort.runs(Plan.Clip, xs)), Laws.len(xs))):
  %runs_sum(xs) : Laws.Holds(Nat.is_le(_, Laws.len(xs)))
  le_refl(Laws.len(xs))

def sort_nruns(+xs: List<&2, Plan.Clip>) -> Laws.Holds(Nat.is_le(nruns(List.sort.runs(Plan.Clip, xs)), 1n+Laws.len(xs))):
  %runs_len(xs) : Laws.Holds(Nat.is_le(_, 1n+Laws.len(xs)))
  le_step(Laws.len(xs), Laws.len(xs), le_refl(Laws.len(xs)))

def Laws.sort_sorted(xs):
  match xs:
    case Nil{}:
      Unit{}
    case +h <> +t:
      sortgo_ok(Laws.len(h <> t), Laws.len(h <> t), List.sort.runs(Plan.Clip, h <> t),
        runs_sorted(h <> t), sort_sum(h <> t), sort_nruns(h <> t))

def Laws.pool_sorted(b, xs):
  +zs = Plan.sort(xs)
  %Equal.sym(List<&2, Plan.Clip>, Laws.flat(Plan.cut(b, zs)), zs, Laws.cut_cover(b, zs)) : Laws.Sorted(_)
  Laws.sort_sorted(xs)

# Pools: sizes
# ------------
# Only a lower bound on the sample count is needed for sizes: modulo conversion
# cannot increase a Nat. The statement keeps both original hypotheses.

# Validation's conditional can succeed only through its true branch.
def check_if_ok(ok: Bool, -msg: String, -rest: Maybe<&2, String>) ->
  Laws.Holds(Laws.is_none(Plan.check.if(ok, msg, rest))) -> Laws.Holds(ok):
  match ok:
    case True{}:
      h => {==}
    case False{}:
      h => h

def check_if_rest(ok: Bool, -msg: String, -rest: Maybe<&2, String>) ->
  Laws.Holds(Laws.is_none(Plan.check.if(ok, msg, rest))) -> Laws.Holds(Laws.is_none(rest)):
  match ok:
    case True{}:
      h => h
    case False{}:
      h => Empty.absurd(Laws.Holds(Laws.is_none(rest)), false_true(h))

# A validation certificate with no message strings: adjacent offsets ascend and
# the last has n's natural value. It does not assume the initial offset is zero.
def Offsets(ps: List<&2, U32>, +lo: U32, +n: U32) -> Type:
  match ps:
    case Nil{}:
      {U32.to_nat(lo) == U32.to_nat(n) : Nat}
    case +h <> t:
      Laws.Holds(U32.is_le(lo, h)) & Offsets(t, h, n)

def check_offsets(ps: List<&2, U32>, +lo: U32, +n: U32) ->
  Laws.Holds(Laws.is_none(Plan.check.go(ps, lo, n))) -> Offsets(ps, lo, n):
  match ps:
    case Nil{}:
      h => W.U32.to_nat_eq(lo, n,
        check_if_ok(U32.is_eq(lo, n), "pools must end at n=" ++ U32.show(n) ++ ", not " ++ U32.show(lo), None{}, h))
    case +x <> +t:
      +h =>
        (check_if_ok(U32.is_le(lo, x), "pool offsets decrease: " ++ U32.show(x) ++ " after " ++ U32.show(lo), Plan.check.go(t, x, n), h),
         check_offsets(t, x, n,
           check_if_rest(U32.is_le(lo, x), "pool offsets decrease: " ++ U32.show(x) ++ " after " ++ U32.show(lo), Plan.check.go(t, x, n), h)))

def sum_sizes(ds: List<&2, Nat>) -> Nat:
  match ds:
    case Nil{}:
      0n
    case d <> t:
      Nat.add(d, sum_sizes(t))

# Adjacent no-underflow differences telescope to the final offset.
def sizes_sum(ps: List<&2, U32>, +lo: U32, +n: U32, v: Offsets(ps, lo, n)) ->
  {Nat.add(U32.to_nat(lo), sum_sizes(Plan.sizes(ps, lo))) == U32.to_nat(n) : Nat}:
  match ps:
    case Nil{}:
      %Equal.sym(Nat, Nat.add(U32.to_nat(lo), 0n), U32.to_nat(lo), N.add_zero(U32.to_nat(lo))) : {_ == U32.to_nat(n) : Nat}
      v
    case +x <> +t:
      (+h, vt) = v
      +l = U32.to_nat(lo)
      +r = U32.to_nat(x)
      +d = U32.to_nat(U32.sub(x, lo))
      +s = sum_sizes(Plan.sizes(t, x))
      Equal.trans(Nat, Nat.add(l, Nat.add(d, s)), Nat.add(Nat.add(l, d), s), U32.to_nat(n),
        Equal.sym(Nat, Nat.add(Nat.add(l, d), s), Nat.add(l, Nat.add(d, s)), N.add_assoc(l, d, s)),
        Equal.trans(Nat, Nat.add(Nat.add(l, d), s), Nat.add(r, s), U32.to_nat(n),
          Equal.cong(Nat, Nat, z => Nat.add(z, s), Nat.add(l, d), r,
            Equal.trans(Nat, Nat.add(l, d), Nat.add(l, Nat.sub(r, l)), r,
              Equal.cong(Nat, Nat, z => Nat.add(l, z), d, Nat.sub(r, l), W.U32.to_nat_sub(x, lo, h)),
              N.add_sub_of_le(l, r, W.U32.to_nat_le(lo, x, h)))),
          sizes_sum(t, x, n, vt)))

# The implementation's unsigned differences are the law's Nat gaps.
def sizes_gaps(ps: List<&2, U32>, +lo: U32, +n: U32, v: Offsets(ps, lo, n)) ->
  {Plan.sizes(ps, lo) == Laws.gaps(ps, lo) : List<&2, Nat>}:
  match ps:
    case Nil{}:
      {==}
    case +x <> +t:
      (h, vt) = v
      Equal.trans(List<&2, Nat>, U32.to_nat(U32.sub(x, lo)) <> Plan.sizes(t, x), Nat.sub(U32.to_nat(x), U32.to_nat(lo)) <> Plan.sizes(t, x),
        Nat.sub(U32.to_nat(x), U32.to_nat(lo)) <> Laws.gaps(t, x),
        Equal.cong(Nat, List<&2, Nat>, z => z <> Plan.sizes(t, x), U32.to_nat(U32.sub(x, lo)), Nat.sub(U32.to_nat(x), U32.to_nat(lo)), W.U32.to_nat_sub(x, lo, h)),
        Equal.cong(List<&2, Nat>, List<&2, Nat>, z => Nat.sub(U32.to_nat(x), U32.to_nat(lo)) <> z, Plan.sizes(t, x), Laws.gaps(t, x), sizes_gaps(t, x, n, vt)))

# Taking d available samples produces exactly d clips and consumes d samples.
def run_len(r: Plan.Run()) -> Nat:
  (xs, ss) = r
  Laws.len(xs)

def rest_len(r: Plan.Run()) -> Nat:
  (xs, ss) = r
  List.length(&2, U32, ss)

def take_cons_len(-x: Plan.Clip, r: Plan.Run()) ->
  {1n+run_len(r) == run_len(Plan.take.cons(x, r)) : Nat}:
  (xs, ss) = r
  {==}

def take_cons_rest(-x: Plan.Clip, r: Plan.Run()) ->
  {rest_len(r) == rest_len(Plan.take.cons(x, r)) : Nat}:
  (xs, ss) = r
  {==}

def take_len(+d: Nat, ss: List<&2, U32>, +i: U32) ->
  N.le(d, List.length(&2, U32, ss)) -> {d == run_len(Plan.take(d, ss, i)) : Nat}:
  match d ss:
    case 0n _:
      h => {==}
    case 1n+p Nil{}:
      h => Empty.absurd({1n+p == 0n : Nat}, false_true(h))
    case 1n++p +s <> +t:
      h =>
        %take_cons_len(Plan.Clip{s, i}, Plan.take(p, t, U32.add(i, 1))) : {1n+p == _ : Nat}
        Equal.cong(Nat, Nat, z => 1n+z, p, run_len(Plan.take(p, t, U32.add(i, 1))), take_len(p, t, U32.add(i, 1), h))

def take_rest_len(+d: Nat, ss: List<&2, U32>, +i: U32) ->
  N.le(d, List.length(&2, U32, ss)) -> {List.length(&2, U32, ss) == Nat.add(d, rest_len(Plan.take(d, ss, i))) : Nat}:
  match d ss:
    case 0n _:
      h => {==}
    case 1n+p Nil{}:
      h => Empty.absurd({0n == Nat.add(1n+p, 0n) : Nat}, false_true(h))
    case 1n++p +s <> +t:
      h =>
        %take_cons_rest(Plan.Clip{s, i}, Plan.take(p, t, U32.add(i, 1))) : {1n+List.length(&2, U32, t) == 1n+Nat.add(p, _) : Nat}
        Equal.cong(Nat, Nat, z => 1n+z, List.length(&2, U32, t), Nat.add(p, rest_len(Plan.take(p, t, U32.add(i, 1)))), take_rest_len(p, t, U32.add(i, 1), h))

# After taking d, a bound d + tail <= available leaves tail available.
def take_tail_bound(+d: Nat, +t: List<&2, Nat>, +ss: List<&2, U32>, +i: U32,
  +h: N.le(Nat.add(d, sum_sizes(t)), List.length(&2, U32, ss))) ->
  N.le(sum_sizes(t), rest_len(Plan.take(d, ss, i))):
  +hd = N.le_trans(d, Nat.add(d, sum_sizes(t)), List.length(&2, U32, ss), N.le_add_right(d, sum_sizes(t)), h)
  +r = rest_len(Plan.take(d, ss, i))
  Equal.trans(Bool, Nat.is_le(sum_sizes(t), r), Nat.is_le(Nat.add(d, sum_sizes(t)), Nat.add(d, r)), True{},
    N.add_le_add_iff_left(d, sum_sizes(t), r),
    Equal.trans(Bool, Nat.is_le(Nat.add(d, sum_sizes(t)), Nat.add(d, r)), Nat.is_le(Nat.add(d, sum_sizes(t)), List.length(&2, U32, ss)), True{},
      Equal.cong(Nat, Bool, z => Nat.is_le(Nat.add(d, sum_sizes(t)), z), Nat.add(d, r), List.length(&2, U32, ss),
        Equal.sym(Nat, List.length(&2, U32, ss), Nat.add(d, r), take_rest_len(d, ss, i, hd))), h))

# The already-taken run followed by the requested run lengths is preserved by runs.
def runs_lens(ds: List<&2, Nat>, +i: U32, r: Plan.Run()) ->
  N.le(sum_sizes(ds), rest_len(r)) ->
  {run_len(r) <> ds == Laws.lens(Plan.runs(ds, i, r)) : List<&2, Nat>}:
  match ds:
    case Nil{}:
      (xs, ss) = r
      h => {==}
    case +d <> +t:
      (xs, +ss) = r
      +h =>
        +hd = N.le_trans(d, Nat.add(d, sum_sizes(t)), List.length(&2, U32, ss), N.le_add_right(d, sum_sizes(t)), h)
        %runs_lens(t, U32.add(i, U32.from_nat(d)), Plan.take(d, ss, i), take_tail_bound(d, t, ss, i, h)) :
          {Laws.len(xs) <> d <> t == Laws.len(xs) <> _ : List<&2, Nat>}
        %take_len(d, ss, i, hd) : {Laws.len(xs) <> d <> t == Laws.len(xs) <> _ <> t : List<&2, Nat>}
        {==}

# A split has the requested lengths whenever their total fits its input.
def split_lens(ds: List<&2, Nat>, +ss: List<&2, U32>) ->
  N.le(sum_sizes(ds), List.length(&2, U32, ss)) -> {ds == Laws.lens(Plan.split(ds, ss)) : List<&2, Nat>}:
  match ds:
    case Nil{}:
      h => {==}
    case +d <> +t:
      +h =>
        +hd = N.le_trans(d, Nat.add(d, sum_sizes(t)), List.length(&2, U32, ss), N.le_add_right(d, sum_sizes(t)), h)
        %runs_lens(t, U32.from_nat(d), Plan.take(d, ss, 0), take_tail_bound(d, t, ss, 0, h)) : {d <> t == _ : List<&2, Nat>}
        %take_len(d, ss, 0, hd) : {d <> t == _ <> t : List<&2, Nat>}
        {==}

# The total validated pool size fits the real input length, including modulo conversion.
def sizes_fit(+ps: List<&2, U32>, +lo: U32, +k: Nat, v: Offsets(ps, lo, U32.from_nat(k))) ->
  N.le(sum_sizes(Plan.sizes(ps, lo)), k):
  +s = sum_sizes(Plan.sizes(ps, lo))
  +l = U32.to_nat(lo)
  N.le_trans(s, Nat.add(l, s), k, N.le_add_left(s, l),
    Equal.trans(Bool, Nat.is_le(Nat.add(l, s), k), Nat.is_le(U32.to_nat(U32.from_nat(k)), k), True{},
      Equal.cong(Nat, Bool, z => Nat.is_le(z, k), Nat.add(l, s), U32.to_nat(U32.from_nat(k)), sizes_sum(ps, lo, U32.from_nat(k), v)),
      W.U32.to_nat_from_nat_le(k)))

# Fill the original law verbatim. Small is unnecessary for lengths, but is kept
# in the original interface; it remains essential for exact cover.
def Laws.pools_sizes(ss, ps, w, v):
  match ps:
    case Nil{}:
      Empty.absurd({Nil{} == Nil{} : List<&2, Nat>}, false_true(v))
    case +lo <> +t:
      +k = List.length(&2, U32, ss)
      +vg = check_if_rest(U32.is_zero(lo), "pools must start at 0, not " ++ U32.show(lo), Plan.check.go(t, lo, U32.from_nat(k)), v)
      Equal.trans(List<&2, Nat>, Laws.lens(Plan.split(Plan.sizes(t, lo), ss)), Plan.sizes(t, lo), Laws.gaps(t, lo),
        Equal.sym(List<&2, Nat>, Plan.sizes(t, lo), Laws.lens(Plan.split(Plan.sizes(t, lo), ss)),
          split_lens(Plan.sizes(t, lo), ss, sizes_fit(t, lo, k, check_offsets(t, lo, U32.from_nat(k), vg)))),
        sizes_gaps(t, lo, U32.from_nat(k), check_offsets(t, lo, U32.from_nat(k), vg)))

# Cut: budget
# -----------

# Elementary Boolean introduction/elimination keeps the budget proof independent
# of the planner's own cut predicate.
def and_intro(+a: Bool, b: Bool, ha: Laws.Holds(a), hb: Laws.Holds(b)) -> Laws.Holds(Bool.and(a, b)):
  match a:
    case True{}:
      hb
    case False{}:
      Empty.absurd({False{} == True{} : Bool}, false_true(ha))

def and_left(a: Bool, -b: Bool) -> Laws.Holds(Bool.and(a, b)) -> Laws.Holds(a):
  match a:
    case True{}:
      h => {==}
    case False{}:
      h => h

def or_right(a: Bool, -b: Bool, h: Laws.Holds(b)) -> Laws.Holds(Bool.or(a, b)):
  match a:
    case True{}:
      {==}
    case False{}:
      h

def or_left_true(a: Bool, -b: Bool, h: Laws.Holds(a)) -> Laws.Holds(Bool.or(a, b)):
  match a:
    case True{}:
      {==}
    case False{}:
      Empty.absurd(Laws.Holds(b), false_true(h))

def or_false_left(a: Bool, -b: Bool, z: {a == False{} : Bool}, h: Laws.Holds(Bool.or(a, b))) -> Laws.Holds(b):
  match a:
    case False{}:
      h
    case True{}:
      Empty.absurd(Laws.Holds(b), true_false(z))

# A smaller positive sample count has at least as large a quotient. The target
# may have zero samples only when the source does too; positivity rules that out here.
def fits_word_pos(+b: U32, +c: Nat, +s: U32, +t: U32, +st: Laws.Holds(U32.is_le(s, t)),
  nonzero: {U32.is_zero(s) == False{} : Bool},
  h: Laws.Holds(Bool.or(U32.is_zero(t), Nat.is_le(c, U32.to_nat(U32.div(b, t)))))) ->
  Laws.Holds(Nat.is_le(c, U32.to_nat(U32.div(b, s)))):
  +ps = W.U32.positive_of_nonzero(s, nonzero)
  +nst = W.U32.to_nat_le(s, t, st)
  +pt = N.lt_of_lt_of_le(0n, U32.to_nat(s), U32.to_nat(t), ps, nst)
  +zt = Equal.trans(Bool, U32.is_zero(t), Nat.is_eq(U32.to_nat(t), 0n), False{}, W.U32.zero_nat(t), W.Nat.positive_ne_zero(U32.to_nat(t), pt))
  N.le_trans(c, U32.to_nat(U32.div(b, t)), U32.to_nat(U32.div(b, s)),
    or_false_left(U32.is_zero(t), Nat.is_le(c, U32.to_nat(U32.div(b, t))), zt, h), W.U32.div_antitone_nat(b, s, t, nst, ps))

def fits_word_mono.zero(z: Bool, +b: U32, +c: Nat, +s: U32, +t: U32, st: Laws.Holds(U32.is_le(s, t)),
  e: {U32.is_zero(s) == z : Bool},
  h: Laws.Holds(Bool.or(U32.is_zero(t), Nat.is_le(c, U32.to_nat(U32.div(b, t)))))) ->
  Laws.Holds(Bool.or(z, Nat.is_le(c, U32.to_nat(U32.div(b, s))))):
  match z:
    case True{}:
      {==}
    case False{}:
      fits_word_pos(b, c, s, t, st, e, h)

def fits_mono(+b: U32, +c: Nat, x: Plan.Clip, y: Plan.Clip) ->
  Laws.Holds(U32.is_le(Plan.Clip.s(x), Plan.Clip.s(y))) -> Laws.Holds(Laws.fits(b, c, y)) -> Laws.Holds(Laws.fits(b, c, x)):
  match x y:
    case Plan.Clip{+s, i} Plan.Clip{+t, j}:
      st => h => fits_word_mono.zero(U32.is_zero(s), b, c, s, t, st, {==}, h)

# The lexicographic key order implies weak order of sample counts.
def key_samples.cmp(c: Cmp, -i: U32, -j: U32) -> Laws.Holds(Laws.key.le.cmp(c, i, j)) -> Laws.Holds(Cmp.is_le(c)):
  match c:
    case LT{}:
      h => {==}
    case EQ{}:
      h => {==}
    case GT{}:
      h => h

def key_samples(x: Plan.Clip, y: Plan.Clip) -> Laws.Holds(Laws.key.le(x, y)) ->
  Laws.Holds(U32.is_le(Plan.Clip.s(x), Plan.Clip.s(y))):
  match x y:
    case Plan.Clip{+s, i} Plan.Clip{+t, j}:
      key_samples.cmp(U32.cmp(s, t), i, j)

# If a comparison is not greater, it is at most.
def cmp_not_gt(c: Cmp) -> {Cmp.is_gt(c) == False{} : Bool} -> Laws.Holds(Cmp.is_le(c)):
  match c:
    case LT{}:
      h => {==}
    case EQ{}:
      h => {==}
    case GT{}:
      h => Empty.absurd({False{} == True{} : Bool}, true_false(h))

# A clip that joins a nonempty batch fits the new batch length; a clip joining
# an empty batch gets the singleton exception.
def starts_false.pos(z: Bool, +b: U32, +p: Nat, +s: U32, -i: U32, e: {U32.is_zero(s) == z : Bool}) ->
  {Plan.starts.pos(b, 1n+p, s, z) == False{} : Bool} -> Laws.Holds(Laws.fits(b, 2n+p, Plan.Clip{s, i})):
  match z:
    case True{}:
      h => or_left_true(U32.is_zero(s), Nat.is_le(2n+p, U32.to_nat(U32.div(b, s))), e)
    case False{}:
      h => or_right(U32.is_zero(s), Nat.is_le(2n+p, U32.to_nat(U32.div(b, s))), cmp_not_gt(Nat.cmp(2n+p, U32.to_nat(U32.div(b, s))), h))

def starts_false(+b: U32, c: Nat, x: Plan.Clip) -> {Plan.starts(b, c, Plan.Clip.s(x)) == False{} : Bool} ->
  Laws.Holds(Bool.or(Nat.is_eq(c, 0n), Laws.fits(b, 1n+c, x))):
  match c x:
    case 0n Plan.Clip{s, i}:
      h => {==}
    case 1n+p Plan.Clip{+s, +i}:
      starts_false.pos(U32.is_zero(s), b, p, s, i, {==})

# Only the last clip's fit is required here. c counts the clips before xs in the
# same batch; the singleton exception is exactly c == 0 at the final clip.
def Last(xs: List<&2, Plan.Clip>, +b: U32, +c: Nat) -> Type:
  match xs:
    case Nil{}:
      Unit
    case x <> Nil{}:
      Laws.Holds(Bool.or(Nat.is_eq(c, 0n), Laws.fits(b, 1n+c, x)))
    case x <> +y <> t:
      Last(y <> t, b, 1n+c)

def Lasts(+b: U32, xss: List<&2, List<&2, Plan.Clip>>) -> Type:
  match xss:
    case Nil{}:
      Unit
    case xs <> t:
      Last(xs, b, 0n) & Lasts(b, t)

def LastInv(+b: U32, +c: Nat, r: Plan.Cut()) -> Type:
  (cur, rest) = r
  Last(cur, b, c) & Lasts(b, rest)

# Adding a clip before a nonempty suffix merely advances its prefix count;
# an empty suffix uses the cut guard's singleton-or-fit certificate.
def last_join(+b: U32, +c: Nat, +x: Plan.Clip, cur: List<&2, Plan.Clip>,
  h: Laws.Holds(Bool.or(Nat.is_eq(c, 0n), Laws.fits(b, 1n+c, x))), v: Last(cur, b, 1n+c)) -> Last(x <> cur, b, c):
  match cur:
    case Nil{}:
      h
    case y <> t:
      v

# A newly started batch is either a singleton or inherits its last clip's fit.
def last_new(+b: U32, -x: Plan.Clip, cur: List<&2, Plan.Clip>, v: Last(cur, b, 1n)) -> Last(x <> cur, b, 0n):
  match cur:
    case Nil{}:
      {==}
    case y <> t:
      v

def last_step(nw: Bool, +b: U32, +c: Nat, +x: Plan.Clip, r: Plan.Cut(),
  e: {Plan.starts(b, c, Plan.Clip.s(x)) == nw : Bool}, v: LastInv(b, Plan.cut.next(nw, c), r)) ->
  LastInv(b, c, Plan.cut.step(x, nw, r)):
  match nw:
    case True{}:
      (cur, rest) = r
      (vc, vr) = v
      (Unit{}, (last_new(b, x, cur, vc), vr))
    case False{}:
      (cur, rest) = r
      (vc, vr) = v
      (last_join(b, c, x, cur, starts_false(b, c, x, e), vc), vr)

def last_go(+b: U32, xs: List<&2, Plan.Clip>, +c: Nat) -> LastInv(b, c, Plan.cut.go(b, xs, c)):
  match xs:
    case Nil{}:
      (Unit{}, Unit{})
    case +x <> +t:
      +nw = Plan.starts(b, c, Plan.Clip.s(x))
      last_step(nw, b, c, x, Plan.cut.go(b, t, Plan.cut.next(nw, c)), {==}, last_go(b, t, Plan.cut.next(nw, c)))

def last_fin(+b: U32, r: Plan.Cut(), v: LastInv(b, 0n, r)) -> Lasts(b, Plan.cut.fin(r)):
  match r:
    case Tuple{Nil{}, rest}:
      (vc, vr) = v
      vr
    case Tuple{x <> t, rest}:
      (vc, vr) = v
      (vc, vr)

# With at least one preceding clip, a sorted suffix fits every clip at the
# final combined length whenever its last clip fits.
def last_all(xs: List<&2, Plan.Clip>, +b: U32, +p: Nat, s: Laws.Sorted(xs), v: Last(xs, b, 1n+p)) ->
  Laws.Holds(Laws.fits.all(b, Nat.add(1n+p, Laws.len(xs)), xs)):
  match xs:
    case Nil{}:
      {==}
    case +x <> Nil{}:
      %add_succ(p, 0n) : Laws.Holds(Laws.fits.all(b, 1n+_, [x]))
      %add_zero(p) : Laws.Holds(Laws.fits.all(b, 2n+_, [x]))
      and_intro(Laws.fits(b, 2n+p, x), True{}, v, {==})
    case +x <> +y <> +t:
      (xy, sy) = s
      %add_succ(p, Laws.len(y <> t)) : Laws.Holds(Laws.fits.all(b, 1n+_, x <> y <> t))
      +k = {2n+Nat.add(p, Laws.len(y <> t)) : Nat}
      +ft = last_all(y <> t, b, 1n+p, sy, v)
      +fy = and_left(Laws.fits(b, k, y), Laws.fits.all(b, k, t), ft)
      and_intro(Laws.fits(b, k, x), Laws.fits.all(b, k, y <> t), fits_mono(b, k, x, y, key_samples(x, y, xy), fy), ft)

# The initial suffix either is a singleton or has one preceding clip when
# recursing into its tail, making the final length exactly the batch length.
def last_budget(+b: U32, xs: List<&2, Plan.Clip>, s: Laws.Sorted(xs), v: Last(xs, b, 0n)) -> Laws.Holds(Laws.budget(b, xs)):
  match xs:
    case Nil{}:
      {==}
    case x <> Nil{}:
      {==}
    case +x <> +y <> +t:
      (xy, sy) = s
      +k = {2n+Laws.len(t) : Nat}
      +ft = last_all(y <> t, b, 0n, sy, v)
      +fy = and_left(Laws.fits(b, k, y), Laws.fits.all(b, k, t), ft)
      and_intro(Laws.fits(b, k, x), Laws.fits.all(b, k, y <> t), fits_mono(b, k, x, y, key_samples(x, y, xy), fy), ft)

# Consume a computed pair through a helper parameter, as required by Bend's
# structural matching discipline.
def pair_use(-A: Type, -B: Type, -R: Type, p: A & B, f: A -> B -> R) -> R:
  (a, b) = p
  f(a, b)

# Split a sorted concatenation's certificate without copying its linear pair.
def sorted_of_from(-lo: Plan.Clip, xs: List<&2, Plan.Clip>, s: Laws.Sorted.from(lo, xs)) -> Laws.Sorted(xs):
  match xs:
    case Nil{}:
      Unit{}
    case h <> t:
      (lh, st) = s
      st

def sorted_from_split(xs: List<&2, Plan.Clip>, +lo: Plan.Clip, +ys: List<&2, Plan.Clip>,
  s: Laws.Sorted.from(lo, List.append(&2, Plan.Clip, xs, ys))) -> Laws.Sorted.from(lo, xs) & Laws.Sorted(ys):
  match xs:
    case Nil{}:
      (Unit{}, sorted_of_from(lo, ys, s))
    case +h <> +t:
      (lh, st) = s
      pair_use(Laws.Sorted.from(h, t), Laws.Sorted(ys), Laws.Sorted.from(lo, h <> t) & Laws.Sorted(ys),
        sorted_from_split(t, h, ys, st), pt => sy => ((lh, pt), sy))

def sorted_split(xs: List<&2, Plan.Clip>, +ys: List<&2, Plan.Clip>,
  s: Laws.Sorted(List.append(&2, Plan.Clip, xs, ys))) -> Laws.Sorted(xs) & Laws.Sorted(ys):
  match xs:
    case Nil{}:
      (Unit{}, s)
    case +h <> +t:
      sorted_from_split(t, h, ys, s)

# Each batch's last-clip certificate and the global sorted order imply every
# clip's budget certificate, including zero lengths and oversized singletons.
def lasts_budget(bs: List<&2, List<&2, Plan.Clip>>, +b: U32, s: Laws.Sorted(Laws.flat(bs)), v: Lasts(b, bs)) -> Laws.Budgets(b, bs):
  match bs:
    case Nil{}:
      Unit{}
    case +xs <> +t:
      (vx, vt) = v
      pair_use(Laws.Sorted(xs), Laws.Sorted(Laws.flat(t)), Laws.Budgets(b, xs <> t), sorted_split(xs, Laws.flat(t), s),
        sx => st => (last_budget(b, xs, sx, vx), lasts_budget(t, b, st, vt)))

def Laws.pool_budget(b, xs):
  +zs = Plan.sort(xs)
  lasts_budget(Plan.cut(b, zs), b, Laws.pool_sorted(b, xs), last_fin(b, Plan.cut.go(b, zs, 0n), last_go(b, zs, 0n)))

# Pools: exact cover with the symbolic width bound
# -----------------------------------------------

# The taken clips followed by the remaining samples, starting at j.
def take_join(+j: U32, r: Plan.Run()) -> List<&2, Plan.Clip>:
  (xs, ss) = r
  List.append(&2, Plan.Clip, xs, Plan.clips(ss, j))

def take_join_cons(+j: U32, -x: Plan.Clip, r: Plan.Run()) ->
  {x <> take_join(j, r) == take_join(j, Plan.take.cons(x, r)) : List<&2, Plan.Clip>}:
  (xs, ss) = r
  {==}

# Taking d samples and continuing at i + from_nat(d) preserves every clip and
# its index. No bound is needed: U32 addition and indices both wrap identically.
def take_cover(+d: Nat, ss: List<&2, U32>, +i: U32) ->
  {take_join(U32.add(i, U32.from_nat(d)), Plan.take(d, ss, i)) == Plan.clips(ss, i) : List<&2, Plan.Clip>}:
  match d ss:
    case 0n _:
      %Equal.sym(U32, U32.add(i, 0), i, W.U32.add_zero(i)) : {Plan.clips(ss, _) == Plan.clips(ss, i) : List<&2, Plan.Clip>}
      {==}
    case 1n+p Nil{}:
      {==}
    case 1n++p +s <> +t:
      %take_join_cons(U32.add(i, U32.from_nat(1n+p)), Plan.Clip{s, i}, Plan.take(p, t, U32.add(i, 1))) : {_ == Plan.Clip{s, i} <> Plan.clips(t, U32.add(i, 1)) : List<&2, Plan.Clip>}
      %Equal.sym(U32, U32.add(i, U32.from_nat(1n+p)), U32.add(U32.add(i, 1), U32.from_nat(p)), W.U32.add_from_nat_succ(i, p)) :
        {Plan.Clip{s, i} <> take_join(_, Plan.take(p, t, U32.add(i, 1))) == Plan.Clip{s, i} <> Plan.clips(t, U32.add(i, 1)) : List<&2, Plan.Clip>}
      Equal.cong(List<&2, Plan.Clip>, List<&2, Plan.Clip>, z => Plan.Clip{s, i} <> z,
        take_join(U32.add(U32.add(i, 1), U32.from_nat(p)), Plan.take(p, t, U32.add(i, 1))), Plan.clips(t, U32.add(i, 1)), take_cover(p, t, U32.add(i, 1)))

# Exact consumption leaves precisely the remaining requested count.
def take_tail_eq(+d: Nat, +t: List<&2, Nat>, +ss: List<&2, U32>, +i: U32,
  +e: {Nat.add(d, sum_sizes(t)) == List.length(&2, U32, ss) : Nat}) ->
  {sum_sizes(t) == rest_len(Plan.take(d, ss, i)) : Nat}:
  +hd = N.le_of_add_eq(d, sum_sizes(t), List.length(&2, U32, ss), e)
  N.add_left_cancel(d, sum_sizes(t), rest_len(Plan.take(d, ss, i)),
    Equal.trans(Nat, Nat.add(d, sum_sizes(t)), List.length(&2, U32, ss), Nat.add(d, rest_len(Plan.take(d, ss, i))), e, take_rest_len(d, ss, i, hd)))

# The existing run and exactly all remaining requested runs preserve the input.
def runs_cover(ds: List<&2, Nat>, +i: U32, r: Plan.Run(), e: {sum_sizes(ds) == rest_len(r) : Nat}) ->
  {Laws.flat(Plan.runs(ds, i, r)) == take_join(i, r) : List<&2, Plan.Clip>}:
  match ds:
    case Nil{}:
      match r:
        case Tuple{xs, Nil{}}:
          {==}
        case Tuple{xs, s <> t}:
          Empty.absurd({List.append(&2, Plan.Clip, xs, Nil{}) == List.append(&2, Plan.Clip, xs, Plan.clips(s <> t, i)) : List<&2, Plan.Clip>},
            N.internal_zero_ne_succ(List.length(&2, U32, t), e))
    case +d <> +t:
      match r:
        case Tuple{+xs, +ss}:
          %Equal.sym(List<&2, Plan.Clip>, Laws.flat(Plan.runs(t, U32.add(i, U32.from_nat(d)), Plan.take(d, ss, i))),
            take_join(U32.add(i, U32.from_nat(d)), Plan.take(d, ss, i)), runs_cover(t, U32.add(i, U32.from_nat(d)), Plan.take(d, ss, i), take_tail_eq(d, t, ss, i, e))) :
            {List.append(&2, Plan.Clip, xs, _) == List.append(&2, Plan.Clip, xs, Plan.clips(ss, i)) : List<&2, Plan.Clip>}
          Equal.cong(List<&2, Plan.Clip>, List<&2, Plan.Clip>, z => List.append(&2, Plan.Clip, xs, z),
            take_join(U32.add(i, U32.from_nat(d)), Plan.take(d, ss, i)), Plan.clips(ss, i), take_cover(d, ss, i))

def split_cover(ds: List<&2, Nat>, +ss: List<&2, U32>, e: {sum_sizes(ds) == List.length(&2, U32, ss) : Nat}) ->
  {Laws.flat(Plan.split(ds, ss)) == Plan.clips(ss, 0) : List<&2, Plan.Clip>}:
  match ds:
    case Nil{}:
      match ss:
        case Nil{}:
          {==}
        case s <> t:
          Empty.absurd({Nil{} == Plan.clips(s <> t, 0) : List<&2, Plan.Clip>}, N.internal_zero_ne_succ(List.length(&2, U32, t), e))
    case +d <> +t:
      %Equal.sym(List<&2, Plan.Clip>, Laws.flat(Plan.runs(t, U32.from_nat(d), Plan.take(d, ss, 0))),
        take_join(U32.from_nat(d), Plan.take(d, ss, 0)), runs_cover(t, U32.from_nat(d), Plan.take(d, ss, 0), take_tail_eq(d, t, ss, 0, e))) :
        {_ == Plan.clips(ss, 0) : List<&2, Plan.Clip>}
      %W.U32.zero_add(U32.from_nat(d)) : {take_join(_, Plan.take(d, ss, 0)) == Plan.clips(ss, 0) : List<&2, Plan.Clip>}
      take_cover(d, ss, 0)

# Complete pool-cover proof when conversion preserves the input length.
def pools_cover_roundtrip(+ss: List<&2, U32>, ps: List<&2, U32>,
  roundtrip: {U32.to_nat(U32.from_nat(List.length(&2, U32, ss))) == List.length(&2, U32, ss) : Nat}, v: Laws.Valid(ss, ps)) ->
  {Laws.flat(Plan.pools(ss, ps)) == Plan.clips(ss, 0) : List<&2, Plan.Clip>}:
  match ps:
    case Nil{}:
      Empty.absurd({Nil{} == Plan.clips(ss, 0) : List<&2, Plan.Clip>}, false_true(v))
    case +lo <> +t:
      +vv = {v : Laws.Holds(Laws.is_none(Plan.check(lo <> t, U32.from_nat(List.length(&2, U32, ss)))))}
      +k = List.length(&2, U32, ss)
      +n = U32.from_nat(k)
      +s = sum_sizes(Plan.sizes(t, lo))
      +z = W.U32.to_nat_eq(lo, 0, check_if_ok(U32.is_zero(lo), "pools must start at 0, not " ++ U32.show(lo), Plan.check.go(t, lo, n), vv))
      +vg = check_if_rest(U32.is_zero(lo), "pools must start at 0, not " ++ U32.show(lo), Plan.check.go(t, lo, n), vv)
      split_cover(Plan.sizes(t, lo), ss,
        Equal.trans(Nat, s, Nat.add(U32.to_nat(lo), s), k,
          Equal.cong(Nat, Nat, x => Nat.add(x, s), 0n, U32.to_nat(lo), Equal.sym(Nat, U32.to_nat(lo), 0n, z)),
          Equal.trans(Nat, Nat.add(U32.to_nat(lo), s), U32.to_nat(n), k, sizes_sum(t, lo, n, check_offsets(t, lo, n, vg)), roundtrip)))

# The conversion lemma supplies exact cover for every representable sample count.
def pools_cover_bounded(+ss: List<&2, U32>, +ps: List<&2, U32>,
  bounded: {Nat.is_le(List.length(&2, U32, ss), W.Word.max_nat(32n)) == True{} : Bool}, v: Laws.Valid(ss, ps)) ->
  {Laws.flat(Plan.pools(ss, ps)) == Plan.clips(ss, 0) : List<&2, Plan.Clip>}:
  pools_cover_roundtrip(ss, ps, W.U32.to_nat_from_nat(List.length(&2, U32, ss), bounded), v)

# Returning w directly compares the Small/Holds aliases with an explicit
# equality. Bend 2.0.37 then unfolds max_nat(32n) while comparing their bounds
# and overflows. Equality elimination only exposes w's equation; the motive
# and reflexivity compare identical symbolic expressions without expanding it.
def small_bound(+ss: List<&2, U32>, w: Laws.Small(ss)) ->
  {Nat.is_le(List.length(&2, U32, ss), W.Word.max_nat(32n)) == True{} : Bool}:
  %w : {Nat.is_le(List.length(&2, U32, ss), W.Word.max_nat(32n)) == _ : Bool}
  {==}

def Laws.pools_cover(ss, ps, w, v):
  pools_cover_bounded(ss, ps, small_bound(ss, w), v)
