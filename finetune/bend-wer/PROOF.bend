# PROOF.bend: all nine laws of LAWS.bend, with no unsafe or foreign code.
# Both `bend PROOF.bend` and `bend PROOF.bend --verdict` print ALL PROOFS CHECK.
#
# The proof goes through Spec.cell(xs, ys), the cell of the suffix table of xs against ys, written
# as a plain (exponential) recursion over the two lists with wer.bend's own step. Then:
#
#   top_spec      wer.bend's rows are Spec.cell's, cell by cell, so Wer.edits is Spec.cell's counts
#   cost_spec     a cell's cost is Laws.lev, the independent recurrence
#   total_spec    a cell's counts add up to its cost: each step pays exactly what the cell costs
#   ref_spec      a cell's M + S + D is the length of its ref words, hyp_spec likewise
#   diag_spec     the cell of a list against itself is all matches
#   lev_le_max, lev_script, lev_sym   facts about Laws.lev alone
#
# and the laws follow, through List.reverse, from those.
#
# A rewrite `%e : P` with e : {a == b} needs P, with _ at b, to be the goal; the goal becomes P
# with a there.
import Base
import ./wer.bend as Wer
import ./LAWS.bend as Laws
import ../bend-planner/u32.bend as W

# Logic
# -----

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

def true_false(e: {True{} == False{} : Bool}) -> Empty:
  false_true(Equal.sym(Bool, True{}, False{}, e))

# Nat
# ---

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

def le_refl(a: Nat) -> Laws.Holds(Nat.is_le(a, a)):
  match a:
    case 0n:
      {==}
    case 1n+p:
      le_refl(p)

def le_zero(a: Nat) -> Laws.Holds(Nat.is_le(0n, a)):
  match a:
    case 0n:
      {==}
    case 1n+p:
      {==}

def le_succ(a: Nat, b: Nat) -> Laws.Holds(Nat.is_le(a, b)) -> Laws.Holds(Nat.is_le(a, 1n+b)):
  match a b:
    case 0n _:
      h => {==}
    case 1n+p 0n:
      h => Empty.absurd(Laws.Holds(Nat.is_le(1n+p, 1n)), false_true(h))
    case 1n+p 1n+q:
      le_succ(p, q)

def le_trans(a: Nat, b: Nat, c: Nat) ->
  Laws.Holds(Nat.is_le(a, b)) -> Laws.Holds(Nat.is_le(b, c)) -> Laws.Holds(Nat.is_le(a, c)):
  match a b c:
    case 0n _ _:
      h1 => h2 => le_zero(c)
    case 1n+p 0n _:
      h1 => h2 => Empty.absurd(Laws.Holds(Nat.is_le(1n+p, c)), false_true(h1))
    case 1n+p 1n+q 0n:
      h1 => h2 => Empty.absurd(Laws.Holds(Nat.is_le(1n+p, 0n)), false_true(h2))
    case 1n+p 1n+q 1n+r:
      le_trans(p, q, r)

def min_le_l(+a: Nat, +b: Nat) -> Laws.Holds(Nat.is_le(Nat.min(a, b), a)):
  match a b:
    case 0n _:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      min_le_l(p, q)

def min_le_r(+a: Nat, +b: Nat) -> Laws.Holds(Nat.is_le(Nat.min(a, b), b)):
  match a b:
    case 0n _:
      le_zero(b)
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      min_le_r(p, q)

def min_comm(+a: Nat, +b: Nat) -> {Nat.min(a, b) == Nat.min(b, a) : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      %min_comm(p, q) : {1n+Nat.min(p, q) == 1n+_ : Nat}
      {==}

def is_eq_sound(a: Nat, b: Nat) -> Laws.Holds(Nat.is_eq(a, b)) -> {a == b : Nat}:
  match a b:
    case 0n 0n:
      h => {==}
    case 0n 1n+q:
      h => Empty.absurd({0n == 1n+q : Nat}, false_true(h))
    case 1n+p 0n:
      h => Empty.absurd({1n+p == 0n : Nat}, false_true(h))
    case 1n+p 1n+q:
      h => Equal.cong(Nat, Nat, n => 1n+n, p, q, is_eq_sound(p, q, h))

# min(a, b) is b when it is not a
def min_right(+a: Nat, +b: Nat) -> {Nat.is_eq(Nat.min(a, b), a) == False{} : Bool} -> {Nat.min(a, b) == b : Nat}:
  match a b:
    case 0n _:
      h => Empty.absurd({0n == b : Nat}, true_false(h))
    case 1n+p 0n:
      h => {==}
    case 1n+p 1n+q:
      h => Equal.cong(Nat, Nat, n => 1n+n, Nat.min(p, q), q, min_right(p, q, h))

# h, with x replaced by y
def is_eq_false.rw(-x: Nat, -y: Nat, -b: Nat, e: {x == y : Nat}, h: {Nat.is_eq(x, b) == False{} : Bool}) ->
  {Nat.is_eq(y, b) == False{} : Bool}:
  %e : {Nat.is_eq(_, b) == False{} : Bool}
  h

# min(a, min(b, c)) is c when it is neither a nor b
def min3_last(+a: Nat, +b: Nat, +c: Nat,
  h1: {Nat.is_eq(Nat.min(a, Nat.min(b, c)), a) == False{} : Bool},
  h2: {Nat.is_eq(Nat.min(a, Nat.min(b, c)), b) == False{} : Bool}) -> {Nat.min(a, Nat.min(b, c)) == c : Nat}:
  +e = min_right(a, Nat.min(b, c), h1)
  %Equal.sym(Nat, Nat.min(a, Nat.min(b, c)), Nat.min(b, c), e) : {_ == c : Nat}
  min_right(b, c, is_eq_false.rw(Nat.min(a, Nat.min(b, c)), Nat.min(b, c), b, e, h2))

# The table
# ---------

# The cell of the suffix table of xs against ys: its cost, and the counts of the walk from it.
def Spec.cell(xs: List<&2, U32>, ys: List<&2, U32>) -> Wer.Cell:
  match xs ys:
    case Nil{} Nil{}:
      Wer.Cell{0n, Wer.Counts.zero()}
    case Nil{} y <> yt:
      Wer.Cell.ins(Spec.cell(Nil{}, yt))
    case x <> xt Nil{}:
      Wer.Cell.del(Spec.cell(xt, Nil{}))
    case +x <> +xt +y <> +yt:
      Wer.step(U32.is_eq(x, y), Spec.cell(xt, y <> yt), Spec.cell(xt, yt), Spec.cell(x <> xt, yt))

# The row of xs: its cells against each suffix of ys, longest first.
def Spec.row(+xs: List<&2, U32>, ys: List<&2, U32>) -> List<&2, Wer.Cell>:
  match ys:
    case Nil{}:
      [Spec.cell(xs, Nil{})]
    case +y <> +yt:
      Spec.cell(xs, y <> yt) <> Spec.row(xs, yt)

def base_cons(-y: U32, t: List<&2, U32>) ->
  {Spec.row(Nil{}, y <> t) == Wer.base.cons(Spec.row(Nil{}, t)) : List<&2, Wer.Cell>}:
  match t:
    case Nil{}:
      {==}
    case z <> u:
      {==}

def base_spec(ys: List<&2, U32>) -> {Spec.row(Nil{}, ys) == Wer.base(ys) : List<&2, Wer.Cell>}:
  match ys:
    case Nil{}:
      {==}
    case +y <> +t:
      %base_spec(t) : {Spec.row(Nil{}, y <> t) == Wer.base.cons(_) : List<&2, Wer.Cell>}
      base_cons(y, t)

def row_cons(+x: U32, +xt: List<&2, U32>, +y: U32, yt: List<&2, U32>) ->
  {Spec.row(x <> xt, y <> yt)
    == Wer.row.cons(x, y, Spec.cell(xt, y <> yt), Spec.row(xt, yt), Spec.row(x <> xt, yt)) : List<&2, Wer.Cell>}:
  match yt:
    case Nil{}:
      {==}
    case z <> u:
      {==}

def row_spec(+x: U32, +xt: List<&2, U32>, ys: List<&2, U32>) ->
  {Spec.row(x <> xt, ys) == Wer.row(x, ys, Spec.row(xt, ys)) : List<&2, Wer.Cell>}:
  match ys:
    case Nil{}:
      {==}
    case +y <> +yt:
      %row_spec(x, xt, yt) : {Spec.row(x <> xt, y <> yt)
        == Wer.row.cons(x, y, Spec.cell(xt, y <> yt), Spec.row(xt, yt), _) : List<&2, Wer.Cell>}
      row_cons(x, xt, y, yt)

def top_spec(xs: List<&2, U32>, +ys: List<&2, U32>) -> {Spec.row(xs, ys) == Wer.top(xs, ys) : List<&2, Wer.Cell>}:
  match xs:
    case Nil{}:
      base_spec(ys)
    case +x <> +t:
      %top_spec(t, ys) : {Spec.row(x <> t, ys) == Wer.row(x, ys, _) : List<&2, Wer.Cell>}
      row_spec(x, t, ys)

def first_row(+xs: List<&2, U32>, ys: List<&2, U32>) ->
  {Wer.Cell.counts(Spec.cell(xs, ys)) == Wer.first(Spec.row(xs, ys)) : Wer.Counts}:
  match ys:
    case Nil{}:
      {==}
    case y <> yt:
      {==}

# Wer.edits reads Spec.cell's counts
def edits_spec(+xs: List<&2, U32>, +ys: List<&2, U32>) ->
  {Wer.Cell.counts(Spec.cell(xs, ys)) == Wer.edits(xs, ys) : Wer.Counts}:
  %top_spec(xs, ys) : {Wer.Cell.counts(Spec.cell(xs, ys)) == Wer.first(_) : Wer.Counts}
  first_row(xs, ys)

# Cost
# ----

def cost_ins(c: Wer.Cell) -> {1n+Wer.Cell.cost(c) == Wer.Cell.cost(Wer.Cell.ins(c)) : Nat}:
  match c:
    case Wer.Cell{n, k}:
      {==}

def cost_del(c: Wer.Cell) -> {1n+Wer.Cell.cost(c) == Wer.Cell.cost(Wer.Cell.del(c)) : Nat}:
  match c:
    case Wer.Cell{n, k}:
      {==}

def step_cost(+e: Bool, a: Wer.Cell, g: Wer.Cell, l: Wer.Cell) ->
  {Nat.min(Nat.add(Wer.Cell.cost(g), Wer.differ(e)), Nat.min(1n+Wer.Cell.cost(a), 1n+Wer.Cell.cost(l)))
    == Wer.Cell.cost(Wer.step(e, a, g, l)) : Nat}:
  match a g l:
    case Wer.Cell{u, ku} Wer.Cell{v, kg} Wer.Cell{w, kl}:
      {==}

def unit_differ(e: Bool) -> {Laws.unit(e) == Wer.differ(e) : Nat}:
  match e:
    case True{}:
      {==}
    case False{}:
      {==}

def lev_nil(xs: List<&2, U32>) -> {Laws.len(xs) == Laws.lev(xs, Nil{}) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case x <> t:
      {==}

# a cell's cost is the Levenshtein distance of its suffixes
def cost_spec(xs: List<&2, U32>, ys: List<&2, U32>) -> {Laws.lev(xs, ys) == Wer.Cell.cost(Spec.cell(xs, ys)) : Nat}:
  match xs ys:
    case Nil{} Nil{}:
      {==}
    case Nil{} +y <> +yt:
      %cost_ins(Spec.cell(Nil{}, yt)) : {Laws.lev(Nil{}, y <> yt) == _ : Nat}
      %cost_spec(Nil{}, yt) : {Laws.lev(Nil{}, y <> yt) == 1n+_ : Nat}
      {==}
    case +x <> +xt Nil{}:
      %cost_del(Spec.cell(xt, Nil{})) : {Laws.lev(x <> xt, Nil{}) == _ : Nat}
      %cost_spec(xt, Nil{}) : {Laws.lev(x <> xt, Nil{}) == 1n+_ : Nat}
      %lev_nil(xt) : {Laws.lev(x <> xt, Nil{}) == 1n+_ : Nat}
      {==}
    case +x <> +xt +y <> +yt:
      +e = U32.is_eq(x, y)
      +ca = Spec.cell(xt, y <> yt)
      +cg = Spec.cell(xt, yt)
      +cl = Spec.cell(x <> xt, yt)
      +lv = Laws.lev(x <> xt, y <> yt)
      %step_cost(e, ca, cg, cl) : {lv == _ : Nat}
      %cost_spec(xt, yt) : {lv == Nat.min(Nat.add(_, Wer.differ(e)),
        Nat.min(1n+Wer.Cell.cost(ca), 1n+Wer.Cell.cost(cl))) : Nat}
      %cost_spec(xt, y <> yt) : {lv == Nat.min(Nat.add(Laws.lev(xt, yt), Wer.differ(e)),
        Nat.min(1n+_, 1n+Wer.Cell.cost(cl))) : Nat}
      %cost_spec(x <> xt, yt) : {lv == Nat.min(Nat.add(Laws.lev(xt, yt), Wer.differ(e)),
        Nat.min(1n+Laws.lev(xt, y <> yt), 1n+_)) : Nat}
      %unit_differ(e) : {lv == Nat.min(Nat.add(Laws.lev(xt, yt), _),
        Nat.min(1n+Laws.lev(xt, y <> yt), 1n+Laws.lev(x <> xt, yt))) : Nat}
      {==}

# Counts add up to the cost
# -------------------------

def total_ins(k: Wer.Counts) -> {1n+Laws.total(k) == Laws.total(Wer.Counts.ins(k)) : Nat}:
  match k:
    case Wer.Counts{m, +s, +d, +i}:
      %add_succ(d, i) : {1n+Nat.add(s, Nat.add(d, i)) == Nat.add(s, _) : Nat}
      add_succ(s, Nat.add(d, i))

def total_del(k: Wer.Counts) -> {1n+Laws.total(k) == Laws.total(Wer.Counts.del(k)) : Nat}:
  match k:
    case Wer.Counts{m, +s, +d, +i}:
      add_succ(s, Nat.add(d, i))

def total_diag(e: Bool, k: Wer.Counts) ->
  {Nat.add(Laws.total(k), Wer.differ(e)) == Laws.total(Wer.Counts.diag(e, k)) : Nat}:
  match e k:
    case True{} Wer.Counts{m, +s, +d, +i}:
      Equal.sym(Nat, Nat.add(s, Nat.add(d, i)), Nat.add(Nat.add(s, Nat.add(d, i)), 0n), add_zero(Nat.add(s, Nat.add(d, i))))
    case False{} Wer.Counts{m, +s, +d, +i}:
      +x = Nat.add(s, Nat.add(d, i))
      %add_succ(x, 0n) : {_ == 1n+x : Nat}
      %add_zero(x) : {1n+_ == 1n+x : Nat}
      {==}

def pick_total(+e: Bool, +u: Nat, +g: Nat, +l: Nat, ku: Wer.Counts, kg: Wer.Counts, kl: Wer.Counts, b1: Bool, b2: Bool) ->
  {Nat.is_eq(Nat.min(Nat.add(g, Wer.differ(e)), Nat.min(1n+u, 1n+l)), Nat.add(g, Wer.differ(e))) == b1 : Bool} ->
  {Nat.is_eq(Nat.min(Nat.add(g, Wer.differ(e)), Nat.min(1n+u, 1n+l)), 1n+u) == b2 : Bool} ->
  {u == Laws.total(ku) : Nat} -> {g == Laws.total(kg) : Nat} -> {l == Laws.total(kl) : Nat} ->
  {Laws.total(Wer.pick(b1, b2, e, ku, kg, kl)) == Nat.min(Nat.add(g, Wer.differ(e)), Nat.min(1n+u, 1n+l)) : Nat}:
  match b1 b2:
    case True{} _:
      +dc = Nat.add(g, Wer.differ(e))
      +uc = {1n+u : Nat}
      +c = Nat.min(dc, Nat.min(uc, 1n+l))
      h1 => h2 => hu => hg => hl =>
        %Equal.sym(Nat, c, dc, is_eq_sound(c, dc, h1)) : {Laws.total(Wer.Counts.diag(e, kg)) == _ : Nat}
        %total_diag(e, kg) : {_ == dc : Nat}
        %hg : {Nat.add(_, Wer.differ(e)) == dc : Nat}
        {==}
    case False{} True{}:
      +dc = Nat.add(g, Wer.differ(e))
      +uc = {1n+u : Nat}
      +c = Nat.min(dc, Nat.min(uc, 1n+l))
      h1 => h2 => hu => hg => hl =>
        %Equal.sym(Nat, c, uc, is_eq_sound(c, uc, h2)) : {Laws.total(Wer.Counts.del(ku)) == _ : Nat}
        %total_del(ku) : {_ == uc : Nat}
        %hu : {1n+_ == uc : Nat}
        {==}
    case False{} False{}:
      +dc = Nat.add(g, Wer.differ(e))
      +uc = {1n+u : Nat}
      +c = Nat.min(dc, Nat.min(uc, 1n+l))
      h1 => h2 => hu => hg => hl =>
        %Equal.sym(Nat, c, 1n+l, min3_last(dc, uc, 1n+l, h1, h2)) : {Laws.total(Wer.Counts.ins(kl)) == _ : Nat}
        %total_ins(kl) : {_ == 1n+l : Nat}
        %hl : {1n+_ == 1n+l : Nat}
        {==}

def step_total(+e: Bool, a: Wer.Cell, g: Wer.Cell, l: Wer.Cell) ->
  {Wer.Cell.cost(a) == Laws.total(Wer.Cell.counts(a)) : Nat} ->
  {Wer.Cell.cost(g) == Laws.total(Wer.Cell.counts(g)) : Nat} ->
  {Wer.Cell.cost(l) == Laws.total(Wer.Cell.counts(l)) : Nat} ->
  {Wer.Cell.cost(Wer.step(e, a, g, l)) == Laws.total(Wer.Cell.counts(Wer.step(e, a, g, l))) : Nat}:
  match a g l:
    case Wer.Cell{+u, +ku} Wer.Cell{+v, +kg} Wer.Cell{+w, +kl}:
      +dc = Nat.add(v, Wer.differ(e))
      +uc = {1n+u : Nat}
      +c = Nat.min(dc, Nat.min(uc, 1n+w))
      ha => hg => hl => Equal.sym(Nat, Laws.total(Wer.pick(Nat.is_eq(c, dc), Nat.is_eq(c, uc), e, ku, kg, kl)), c,
        pick_total(e, u, v, w, ku, kg, kl, Nat.is_eq(c, dc), Nat.is_eq(c, uc), {==}, {==}, ha, hg, hl))

def ins_total(c: Wer.Cell) -> {Wer.Cell.cost(c) == Laws.total(Wer.Cell.counts(c)) : Nat} ->
  {Wer.Cell.cost(Wer.Cell.ins(c)) == Laws.total(Wer.Cell.counts(Wer.Cell.ins(c))) : Nat}:
  match c:
    case Wer.Cell{n, +k}:
      h =>
        %total_ins(k) : {1n+n == _ : Nat}
        %h : {1n+n == 1n+_ : Nat}
        {==}

def del_total(c: Wer.Cell) -> {Wer.Cell.cost(c) == Laws.total(Wer.Cell.counts(c)) : Nat} ->
  {Wer.Cell.cost(Wer.Cell.del(c)) == Laws.total(Wer.Cell.counts(Wer.Cell.del(c))) : Nat}:
  match c:
    case Wer.Cell{n, +k}:
      h =>
        %total_del(k) : {1n+n == _ : Nat}
        %h : {1n+n == 1n+_ : Nat}
        {==}

# a cell's counts add up to its cost
def total_spec(xs: List<&2, U32>, ys: List<&2, U32>) ->
  {Wer.Cell.cost(Spec.cell(xs, ys)) == Laws.total(Wer.Cell.counts(Spec.cell(xs, ys))) : Nat}:
  match xs ys:
    case Nil{} Nil{}:
      {==}
    case Nil{} +y <> +yt:
      ins_total(Spec.cell(Nil{}, yt), total_spec(Nil{}, yt))
    case +x <> +xt Nil{}:
      del_total(Spec.cell(xt, Nil{}), total_spec(xt, Nil{}))
    case +x <> +xt +y <> +yt:
      step_total(U32.is_eq(x, y), Spec.cell(xt, y <> yt), Spec.cell(xt, yt), Spec.cell(x <> xt, yt),
        total_spec(xt, y <> yt), total_spec(xt, yt), total_spec(x <> xt, yt))

# S + D + I of Spec.cell is lev
def lev_total(+xs: List<&2, U32>, +ys: List<&2, U32>) ->
  {Laws.total(Wer.Cell.counts(Spec.cell(xs, ys))) == Laws.lev(xs, ys) : Nat}:
  %total_spec(xs, ys) : {_ == Laws.lev(xs, ys) : Nat}
  Equal.sym(Nat, Laws.lev(xs, ys), Wer.Cell.cost(Spec.cell(xs, ys)), cost_spec(xs, ys))

# Wer.counts is Spec.cell's counts, on the reversed lists
def counts_spec(+ref: List<&2, U32>, +hyp: List<&2, U32>) ->
  {Wer.Cell.counts(Spec.cell(Laws.rev(ref), Laws.rev(hyp))) == Wer.counts(ref, hyp) : Wer.Counts}:
  edits_spec(Laws.rev(ref), Laws.rev(hyp))

# Laws: the distance
# ------------------

def Laws.total_is_distance(ref, hyp):
  %counts_spec(ref, hyp) : {Laws.total(_) == Laws.distance(ref, hyp) : Nat}
  lev_total(Laws.rev(ref), Laws.rev(hyp))

# Lists
# -----

def len_rev.go(xs: List<&2, U32>, +acc: List<&2, U32>) ->
  {Nat.add(Laws.len(xs), Laws.len(acc)) == Laws.len(List.reverse.go(&2, U32, xs, acc)) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case +h <> +t:
      %len_rev.go(t, h <> acc) : {1n+Nat.add(Laws.len(t), Laws.len(acc)) == _ : Nat}
      add_succ(Laws.len(t), Laws.len(acc))

def len_rev(+xs: List<&2, U32>) -> {Laws.len(xs) == Laws.len(Laws.rev(xs)) : Nat}:
  %len_rev.go(xs, Nil{}) : {Laws.len(xs) == _ : Nat}
  add_zero(Laws.len(xs))

# Lengths
# -------

# M + S + D, and M + S + I
def rs(k: Wer.Counts) -> Nat:
  match k:
    case Wer.Counts{m, s, d, i}:
      Nat.add(m, Nat.add(s, d))

def hs(k: Wer.Counts) -> Nat:
  match k:
    case Wer.Counts{m, s, d, i}:
      Nat.add(m, Nat.add(s, i))

def rs_eq(k: Wer.Counts) -> {rs(k) == Nat.add(Laws.M(k), Nat.add(Laws.S(k), Laws.D(k))) : Nat}:
  match k:
    case Wer.Counts{m, s, d, i}:
      {==}

def hs_eq(k: Wer.Counts) -> {hs(k) == Nat.add(Laws.M(k), Nat.add(Laws.S(k), Laws.I(k))) : Nat}:
  match k:
    case Wer.Counts{m, s, d, i}:
      {==}

def rs_diag(e: Bool, k: Wer.Counts) -> {1n+rs(k) == rs(Wer.Counts.diag(e, k)) : Nat}:
  match e k:
    case True{} Wer.Counts{m, s, d, i}:
      {==}
    case False{} Wer.Counts{+m, +s, +d, i}:
      add_succ(m, Nat.add(s, d))

def rs_del(k: Wer.Counts) -> {1n+rs(k) == rs(Wer.Counts.del(k)) : Nat}:
  match k:
    case Wer.Counts{+m, +s, +d, i}:
      %add_succ(s, d) : {1n+Nat.add(m, Nat.add(s, d)) == Nat.add(m, _) : Nat}
      add_succ(m, Nat.add(s, d))

def rs_ins(k: Wer.Counts) -> {rs(k) == rs(Wer.Counts.ins(k)) : Nat}:
  match k:
    case Wer.Counts{m, s, d, i}:
      {==}

def hs_diag(e: Bool, k: Wer.Counts) -> {1n+hs(k) == hs(Wer.Counts.diag(e, k)) : Nat}:
  match e k:
    case True{} Wer.Counts{m, s, d, i}:
      {==}
    case False{} Wer.Counts{+m, +s, d, +i}:
      add_succ(m, Nat.add(s, i))

def hs_ins(k: Wer.Counts) -> {1n+hs(k) == hs(Wer.Counts.ins(k)) : Nat}:
  match k:
    case Wer.Counts{+m, +s, d, +i}:
      %add_succ(s, i) : {1n+Nat.add(m, Nat.add(s, i)) == Nat.add(m, _) : Nat}
      add_succ(m, Nat.add(s, i))

def hs_del(k: Wer.Counts) -> {hs(k) == hs(Wer.Counts.del(k)) : Nat}:
  match k:
    case Wer.Counts{m, s, d, i}:
      {==}

def pick_ref(b1: Bool, b2: Bool, +e: Bool, +ku: Wer.Counts, +kg: Wer.Counts, +kl: Wer.Counts, -n: Nat) ->
  {rs(ku) == n : Nat} -> {rs(kg) == n : Nat} -> {rs(kl) == 1n+n : Nat} ->
  {rs(Wer.pick(b1, b2, e, ku, kg, kl)) == 1n+n : Nat}:
  match b1 b2:
    case True{} _:
      hu => hg => hl =>
        %rs_diag(e, kg) : {_ == 1n+n : Nat}
        Equal.cong(Nat, Nat, x => 1n+x, rs(kg), n, hg)
    case False{} True{}:
      hu => hg => hl =>
        %rs_del(ku) : {_ == 1n+n : Nat}
        Equal.cong(Nat, Nat, x => 1n+x, rs(ku), n, hu)
    case False{} False{}:
      hu => hg => hl =>
        %rs_ins(kl) : {_ == 1n+n : Nat}
        hl

def pick_hyp(b1: Bool, b2: Bool, +e: Bool, +ku: Wer.Counts, +kg: Wer.Counts, +kl: Wer.Counts, -n: Nat) ->
  {hs(ku) == 1n+n : Nat} -> {hs(kg) == n : Nat} -> {hs(kl) == n : Nat} ->
  {hs(Wer.pick(b1, b2, e, ku, kg, kl)) == 1n+n : Nat}:
  match b1 b2:
    case True{} _:
      hu => hg => hl =>
        %hs_diag(e, kg) : {_ == 1n+n : Nat}
        Equal.cong(Nat, Nat, x => 1n+x, hs(kg), n, hg)
    case False{} True{}:
      hu => hg => hl =>
        %hs_del(ku) : {_ == 1n+n : Nat}
        hu
    case False{} False{}:
      hu => hg => hl =>
        %hs_ins(kl) : {_ == 1n+n : Nat}
        Equal.cong(Nat, Nat, x => 1n+x, hs(kl), n, hl)

def step_ref(+e: Bool, a: Wer.Cell, g: Wer.Cell, l: Wer.Cell, -n: Nat) ->
  {rs(Wer.Cell.counts(a)) == n : Nat} -> {rs(Wer.Cell.counts(g)) == n : Nat} -> {rs(Wer.Cell.counts(l)) == 1n+n : Nat} ->
  {rs(Wer.Cell.counts(Wer.step(e, a, g, l))) == 1n+n : Nat}:
  match a g l:
    case Wer.Cell{+u, +ku} Wer.Cell{+v, +kg} Wer.Cell{+w, +kl}:
      +dc = Nat.add(v, Wer.differ(e))
      +uc = {1n+u : Nat}
      +c = Nat.min(dc, Nat.min(uc, 1n+w))
      pick_ref(Nat.is_eq(c, dc), Nat.is_eq(c, uc), e, ku, kg, kl, n)

def step_hyp(+e: Bool, a: Wer.Cell, g: Wer.Cell, l: Wer.Cell, -n: Nat) ->
  {hs(Wer.Cell.counts(a)) == 1n+n : Nat} -> {hs(Wer.Cell.counts(g)) == n : Nat} -> {hs(Wer.Cell.counts(l)) == n : Nat} ->
  {hs(Wer.Cell.counts(Wer.step(e, a, g, l))) == 1n+n : Nat}:
  match a g l:
    case Wer.Cell{+u, +ku} Wer.Cell{+v, +kg} Wer.Cell{+w, +kl}:
      +dc = Nat.add(v, Wer.differ(e))
      +uc = {1n+u : Nat}
      +c = Nat.min(dc, Nat.min(uc, 1n+w))
      pick_hyp(Nat.is_eq(c, dc), Nat.is_eq(c, uc), e, ku, kg, kl, n)

def rs_cell_ins(c: Wer.Cell) -> {rs(Wer.Cell.counts(c)) == rs(Wer.Cell.counts(Wer.Cell.ins(c))) : Nat}:
  match c:
    case Wer.Cell{n, k}:
      rs_ins(k)

def rs_cell_del(c: Wer.Cell) -> {1n+rs(Wer.Cell.counts(c)) == rs(Wer.Cell.counts(Wer.Cell.del(c))) : Nat}:
  match c:
    case Wer.Cell{n, k}:
      rs_del(k)

def hs_cell_ins(c: Wer.Cell) -> {1n+hs(Wer.Cell.counts(c)) == hs(Wer.Cell.counts(Wer.Cell.ins(c))) : Nat}:
  match c:
    case Wer.Cell{n, k}:
      hs_ins(k)

def hs_cell_del(c: Wer.Cell) -> {hs(Wer.Cell.counts(c)) == hs(Wer.Cell.counts(Wer.Cell.del(c))) : Nat}:
  match c:
    case Wer.Cell{n, k}:
      hs_del(k)

# every ref word of a cell's walk is matched, substituted or deleted
def ref_spec(xs: List<&2, U32>, ys: List<&2, U32>) -> {rs(Wer.Cell.counts(Spec.cell(xs, ys))) == Laws.len(xs) : Nat}:
  match xs ys:
    case Nil{} Nil{}:
      {==}
    case Nil{} +y <> +yt:
      %rs_cell_ins(Spec.cell(Nil{}, yt)) : {_ == 0n : Nat}
      ref_spec(Nil{}, yt)
    case +x <> +xt Nil{}:
      %rs_cell_del(Spec.cell(xt, Nil{})) : {_ == 1n+Laws.len(xt) : Nat}
      Equal.cong(Nat, Nat, n => 1n+n, rs(Wer.Cell.counts(Spec.cell(xt, Nil{}))), Laws.len(xt), ref_spec(xt, Nil{}))
    case +x <> +xt +y <> +yt:
      step_ref(U32.is_eq(x, y), Spec.cell(xt, y <> yt), Spec.cell(xt, yt), Spec.cell(x <> xt, yt), Laws.len(xt),
        ref_spec(xt, y <> yt), ref_spec(xt, yt), ref_spec(x <> xt, yt))

# every hyp word of a cell's walk is matched, substituted or inserted
def hyp_spec(xs: List<&2, U32>, ys: List<&2, U32>) -> {hs(Wer.Cell.counts(Spec.cell(xs, ys))) == Laws.len(ys) : Nat}:
  match xs ys:
    case Nil{} Nil{}:
      {==}
    case Nil{} +y <> +yt:
      %hs_cell_ins(Spec.cell(Nil{}, yt)) : {_ == 1n+Laws.len(yt) : Nat}
      Equal.cong(Nat, Nat, n => 1n+n, hs(Wer.Cell.counts(Spec.cell(Nil{}, yt))), Laws.len(yt), hyp_spec(Nil{}, yt))
    case +x <> +xt Nil{}:
      %hs_cell_del(Spec.cell(xt, Nil{})) : {_ == 0n : Nat}
      hyp_spec(xt, Nil{})
    case +x <> +xt +y <> +yt:
      step_hyp(U32.is_eq(x, y), Spec.cell(xt, y <> yt), Spec.cell(xt, yt), Spec.cell(x <> xt, yt), Laws.len(yt),
        hyp_spec(xt, y <> yt), hyp_spec(xt, yt), hyp_spec(x <> xt, yt))

def Laws.ref_length(ref, hyp):
  +k = Wer.counts(ref, hyp)
  %rs_eq(k) : {_ == Laws.len(ref) : Nat}
  %counts_spec(ref, hyp) : {rs(_) == Laws.len(ref) : Nat}
  %Equal.sym(Nat, Laws.len(ref), Laws.len(Laws.rev(ref)), len_rev(ref))
    : {rs(Wer.Cell.counts(Spec.cell(Laws.rev(ref), Laws.rev(hyp)))) == _ : Nat}
  ref_spec(Laws.rev(ref), Laws.rev(hyp))

def Laws.hyp_length(ref, hyp):
  +k = Wer.counts(ref, hyp)
  %hs_eq(k) : {_ == Laws.len(hyp) : Nat}
  %counts_spec(ref, hyp) : {hs(_) == Laws.len(hyp) : Nat}
  %Equal.sym(Nat, Laws.len(hyp), Laws.len(Laws.rev(hyp)), len_rev(hyp))
    : {hs(Wer.Cell.counts(Spec.cell(Laws.rev(ref), Laws.rev(hyp)))) == _ : Nat}
  hyp_spec(Laws.rev(ref), Laws.rev(hyp))

# Identical
# ---------

def cmp_refl(n: Nat) -> {EQ{} == Nat.cmp(n, n) : Cmp}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      cmp_refl(p)

def u32_eq_refl(+x: U32) -> {True{} == U32.is_eq(x, x) : Bool}:
  %Equal.sym(Cmp, U32.cmp(x, x), Nat.cmp(U32.to_nat(x), U32.to_nat(x)), W.U32.cmp_to_nat(x, x))
    : {True{} == Cmp.is_eq(_) : Bool}
  %cmp_refl(U32.to_nat(x)) : {True{} == Cmp.is_eq(_) : Bool}
  {==}

def step_same(a: Wer.Cell, l: Wer.Cell, +k: Wer.Counts) ->
  {Wer.Cell{0n, Wer.Counts.diag(True{}, k)} == Wer.step(True{}, a, Wer.Cell{0n, k}, l) : Wer.Cell}:
  match a l:
    case Wer.Cell{u, ku} Wer.Cell{w, kl}:
      {==}

# the cell of a list against itself: no cost, all matches
def diag_spec(xs: List<&2, U32>) ->
  {Wer.Cell{0n, Wer.Counts{Laws.len(xs), 0n, 0n, 0n}} == Spec.cell(xs, xs) : Wer.Cell}:
  match xs:
    case Nil{}:
      {==}
    case +x <> +t:
      +k = {Wer.Counts{Laws.len(t), 0n, 0n, 0n} : Wer.Counts}
      %u32_eq_refl(x) : {Wer.Cell{0n, Wer.Counts{1n+Laws.len(t), 0n, 0n, 0n}}
        == Wer.step(_, Spec.cell(t, x <> t), Spec.cell(t, t), Spec.cell(x <> t, t)) : Wer.Cell}
      %diag_spec(t) : {Wer.Cell{0n, Wer.Counts{1n+Laws.len(t), 0n, 0n, 0n}}
        == Wer.step(True{}, Spec.cell(t, x <> t), _, Spec.cell(x <> t, t)) : Wer.Cell}
      step_same(Spec.cell(t, x <> t), Spec.cell(x <> t, t), k)

def Laws.identical(ws):
  +r = Laws.rev(ws)
  %counts_spec(ws, ws) : {_ == Wer.Counts{Laws.len(ws), 0n, 0n, 0n} : Wer.Counts}
  %diag_spec(r) : {Wer.Cell.counts(_) == Wer.Counts{Laws.len(ws), 0n, 0n, 0n} : Wer.Counts}
  %len_rev(ws) : {Wer.Counts{_, 0n, 0n, 0n} == Wer.Counts{Laws.len(ws), 0n, 0n, 0n} : Wer.Counts}
  {==}

# Bounds
# ------

def le_add_l(+s: Nat, +x: Nat) -> Laws.Holds(Nat.is_le(x, Nat.add(s, x))):
  match s:
    case 0n:
      le_refl(x)
    case 1n+p:
      le_succ(x, Nat.add(p, x), le_add_l(p, x))

def sub_cancel(m: Nat, -a: Nat, -b: Nat) -> {Nat.sub(a, b) == Nat.sub(Nat.add(m, a), Nat.add(m, b)) : Nat}:
  match m:
    case 0n:
      {==}
    case 1n+p:
      sub_cancel(p, a, b)

# |d - i| <= d + i
def gap_le(+d: Nat, +i: Nat) -> Laws.Holds(Nat.is_le(Laws.gap(d, i), Nat.add(d, i))):
  match d i:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      le_refl(q)
    case 1n+p 0n:
      le_refl(Nat.add(1n+p, 0n))
    case 1n+p 1n+q:
      +g = Laws.gap(p, q)
      %add_succ(p, q) : Laws.Holds(Nat.is_le(g, 1n+_))
      le_succ(g, 1n+Nat.add(p, q), le_succ(g, Nat.add(p, q), gap_le(p, q)))

def gap_counts(k: Wer.Counts) ->
  Laws.Holds(Nat.is_le(Laws.gap(Nat.add(Laws.M(k), Nat.add(Laws.S(k), Laws.D(k))), Nat.add(Laws.M(k), Nat.add(Laws.S(k), Laws.I(k)))),
    Laws.total(k))):
  match k:
    case Wer.Counts{+m, +s, +d, +i}:
      +t = Nat.add(s, Nat.add(d, i))
      %sub_cancel(m, Nat.add(s, d), Nat.add(s, i)) : Laws.Holds(Nat.is_le(Nat.add(_, Nat.sub(Nat.add(m, Nat.add(s, i)), Nat.add(m, Nat.add(s, d)))), t))
      %sub_cancel(m, Nat.add(s, i), Nat.add(s, d)) : Laws.Holds(Nat.is_le(Nat.add(Nat.sub(Nat.add(s, d), Nat.add(s, i)), _), t))
      %sub_cancel(s, d, i) : Laws.Holds(Nat.is_le(Nat.add(_, Nat.sub(Nat.add(s, i), Nat.add(s, d))), t))
      %sub_cancel(s, i, d) : Laws.Holds(Nat.is_le(Nat.add(Nat.sub(d, i), _), t))
      le_trans(Laws.gap(d, i), Nat.add(d, i), t, gap_le(d, i), le_add_l(s, Nat.add(d, i)))

def Laws.lower_bound(ref, hyp):
  +k = Wer.counts(ref, hyp)
  %Laws.ref_length(ref, hyp) : Laws.Holds(Nat.is_le(Laws.gap(_, Laws.len(hyp)), Laws.total(k)))
  %Laws.hyp_length(ref, hyp) : Laws.Holds(Nat.is_le(Laws.gap(Nat.add(Laws.M(k), Nat.add(Laws.S(k), Laws.D(k))), _), Laws.total(k)))
  gap_counts(k)

def add_unit_le(+a: Nat, e: Bool) -> Laws.Holds(Nat.is_le(Nat.add(a, Laws.unit(e)), 1n+a)):
  match e:
    case True{}:
      %add_zero(a) : Laws.Holds(Nat.is_le(_, 1n+a))
      le_succ(a, a, le_refl(a))
    case False{}:
      %add_succ(a, 0n) : Laws.Holds(Nat.is_le(_, 1n+a))
      %add_zero(a) : Laws.Holds(Nat.is_le(1n+_, 1n+a))
      le_refl(a)

# lev(xs, ys) <= max(len(xs), len(ys))
def lev_le_max(xs: List<&2, U32>, ys: List<&2, U32>) -> Laws.Holds(Nat.is_le(Laws.lev(xs, ys), Nat.max(Laws.len(xs), Laws.len(ys)))):
  match xs ys:
    case Nil{} _:
      le_refl(Laws.len(ys))
    case x <> xt Nil{}:
      le_refl(Laws.len(xt))
    case +x <> +xt +y <> +yt:
      +dg = Nat.add(Laws.lev(xt, yt), Laws.unit(U32.is_eq(x, y)))
      +mx = Nat.max(Laws.len(x <> xt), Laws.len(y <> yt))
      le_trans(Laws.lev(x <> xt, y <> yt), dg, mx,
        min_le_l(dg, Nat.min(1n+Laws.lev(xt, y <> yt), 1n+Laws.lev(x <> xt, yt))),
        le_trans(dg, 1n+Laws.lev(xt, yt), mx, add_unit_le(Laws.lev(xt, yt), U32.is_eq(x, y)), lev_le_max(xt, yt)))

def Laws.upper_bound(ref, hyp):
  +mx = Nat.max(Laws.len(ref), Laws.len(hyp))
  %Equal.sym(Nat, Laws.total(Wer.counts(ref, hyp)), Laws.distance(ref, hyp), Laws.total_is_distance(ref, hyp))
    : Laws.Holds(Nat.is_le(_, mx))
  %Equal.sym(Nat, Laws.len(ref), Laws.len(Laws.rev(ref)), len_rev(ref))
    : Laws.Holds(Nat.is_le(Laws.distance(ref, hyp), Nat.max(_, Laws.len(hyp))))
  %Equal.sym(Nat, Laws.len(hyp), Laws.len(Laws.rev(hyp)), len_rev(hyp))
    : Laws.Holds(Nat.is_le(Laws.distance(ref, hyp), Nat.max(Laws.len(Laws.rev(ref)), _)))
  lev_le_max(Laws.rev(ref), Laws.rev(hyp))

# Optimality
# ----------

def lev_drop(+a: U32, +r: List<&2, U32>, h: List<&2, U32>) -> Laws.Holds(Nat.is_le(Laws.lev(a <> r, h), 1n+Laws.lev(r, h))):
  match h:
    case Nil{}:
      %lev_nil(r) : Laws.Holds(Nat.is_le(1n+Laws.len(r), 1n+_))
      le_refl(Laws.len(r))
    case +y <> +ht:
      +dg = Nat.add(Laws.lev(r, ht), Laws.unit(U32.is_eq(a, y)))
      +up = {1n+Laws.lev(r, y <> ht) : Nat}
      +lf = {1n+Laws.lev(a <> r, ht) : Nat}
      le_trans(Nat.min(dg, Nat.min(up, lf)), Nat.min(up, lf), up, min_le_r(dg, Nat.min(up, lf)), min_le_l(up, lf))

def lev_add(+b: U32, r: List<&2, U32>, +h: List<&2, U32>) -> Laws.Holds(Nat.is_le(Laws.lev(r, b <> h), 1n+Laws.lev(r, h))):
  match r:
    case Nil{}:
      le_refl(Laws.len(h))
    case +x <> +rt:
      +dg = Nat.add(Laws.lev(rt, h), Laws.unit(U32.is_eq(x, b)))
      +up = {1n+Laws.lev(rt, b <> h) : Nat}
      +lf = {1n+Laws.lev(x <> rt, h) : Nat}
      le_trans(Nat.min(dg, Nat.min(up, lf)), Nat.min(up, lf), lf, min_le_r(dg, Nat.min(up, lf)), min_le_r(up, lf))

# a kept word costs nothing: lev(r, h) + unit(w = w) <= cost when lev(r, h) <= cost
def keep_le(+w: U32, +r: List<&2, U32>, +h: List<&2, U32>, +c: Nat, e: Laws.Holds(Nat.is_le(Laws.lev(r, h), c))) ->
  Laws.Holds(Nat.is_le(Nat.add(Laws.lev(r, h), Laws.unit(U32.is_eq(w, w))), c)):
  %u32_eq_refl(w) : Laws.Holds(Nat.is_le(Nat.add(Laws.lev(r, h), Laws.unit(_)), c))
  %add_zero(Laws.lev(r, h)) : Laws.Holds(Nat.is_le(_, c))
  e

# no script costs less than the distance of the words it reads and writes
def lev_script(ops: List<&2, Laws.Op>) -> Laws.Holds(Nat.is_le(Laws.lev(Laws.ref_of(ops), Laws.hyp_of(ops)), Laws.cost(ops))):
  match ops:
    case Nil{}:
      {==}
    case Laws.Keep{+w} <> +t:
      +r = Laws.ref_of(t)
      +h = Laws.hyp_of(t)
      +dg = Nat.add(Laws.lev(r, h), Laws.unit(U32.is_eq(w, w)))
      +ct = Laws.cost(t)
      le_trans(Laws.lev(w <> r, w <> h), dg, ct,
        min_le_l(dg, Nat.min(1n+Laws.lev(r, w <> h), 1n+Laws.lev(w <> r, h))),
        keep_le(w, r, h, ct, lev_script(t)))
    case Laws.Subst{+a, +b} <> +t:
      +r = Laws.ref_of(t)
      +h = Laws.hyp_of(t)
      +dg = Nat.add(Laws.lev(r, h), Laws.unit(U32.is_eq(a, b)))
      +ct = {1n+Laws.cost(t) : Nat}
      le_trans(Laws.lev(a <> r, b <> h), dg, ct,
        min_le_l(dg, Nat.min(1n+Laws.lev(r, b <> h), 1n+Laws.lev(a <> r, h))),
        le_trans(dg, 1n+Laws.lev(r, h), ct, add_unit_le(Laws.lev(r, h), U32.is_eq(a, b)), lev_script(t)))
    case Laws.Drop{+a} <> +t:
      +r = Laws.ref_of(t)
      +h = Laws.hyp_of(t)
      le_trans(Laws.lev(a <> r, h), 1n+Laws.lev(r, h), 1n+Laws.cost(t), lev_drop(a, r, h), lev_script(t))
    case Laws.Add{+b} <> +t:
      +r = Laws.ref_of(t)
      +h = Laws.hyp_of(t)
      le_trans(Laws.lev(r, b <> h), 1n+Laws.lev(r, h), 1n+Laws.cost(t), lev_add(b, r, h), lev_script(t))

# A script reversed reads and writes the reversed words, at the same cost.
def ref_of_rev(ops: List<&2, Laws.Op>, +acc: List<&2, Laws.Op>) ->
  {List.reverse.go(&2, U32, Laws.ref_of(ops), Laws.ref_of(acc))
    == Laws.ref_of(List.reverse.go(&2, Laws.Op, ops, acc)) : List<&2, U32>}:
  match ops:
    case Nil{}:
      {==}
    case Laws.Keep{+w} <> t:
      ref_of_rev(t, Laws.Keep{w} <> acc)
    case Laws.Subst{+a, +b} <> t:
      ref_of_rev(t, Laws.Subst{a, b} <> acc)
    case Laws.Drop{+a} <> t:
      ref_of_rev(t, Laws.Drop{a} <> acc)
    case Laws.Add{+b} <> t:
      ref_of_rev(t, Laws.Add{b} <> acc)

def hyp_of_rev(ops: List<&2, Laws.Op>, +acc: List<&2, Laws.Op>) ->
  {List.reverse.go(&2, U32, Laws.hyp_of(ops), Laws.hyp_of(acc))
    == Laws.hyp_of(List.reverse.go(&2, Laws.Op, ops, acc)) : List<&2, U32>}:
  match ops:
    case Nil{}:
      {==}
    case Laws.Keep{+w} <> t:
      hyp_of_rev(t, Laws.Keep{w} <> acc)
    case Laws.Subst{+a, +b} <> t:
      hyp_of_rev(t, Laws.Subst{a, b} <> acc)
    case Laws.Drop{+a} <> t:
      hyp_of_rev(t, Laws.Drop{a} <> acc)
    case Laws.Add{+b} <> t:
      hyp_of_rev(t, Laws.Add{b} <> acc)

def cost_rev(ops: List<&2, Laws.Op>, +acc: List<&2, Laws.Op>) ->
  {Nat.add(Laws.cost(ops), Laws.cost(acc)) == Laws.cost(List.reverse.go(&2, Laws.Op, ops, acc)) : Nat}:
  match ops:
    case Nil{}:
      {==}
    case Laws.Keep{+w} <> +t:
      cost_rev(t, Laws.Keep{w} <> acc)
    case Laws.Subst{+a, +b} <> +t:
      %cost_rev(t, Laws.Subst{a, b} <> acc) : {1n+Nat.add(Laws.cost(t), Laws.cost(acc)) == _ : Nat}
      add_succ(Laws.cost(t), Laws.cost(acc))
    case Laws.Drop{+a} <> +t:
      %cost_rev(t, Laws.Drop{a} <> acc) : {1n+Nat.add(Laws.cost(t), Laws.cost(acc)) == _ : Nat}
      add_succ(Laws.cost(t), Laws.cost(acc))
    case Laws.Add{+b} <> +t:
      %cost_rev(t, Laws.Add{b} <> acc) : {1n+Nat.add(Laws.cost(t), Laws.cost(acc)) == _ : Nat}
      add_succ(Laws.cost(t), Laws.cost(acc))

def Laws.optimal(ops):
  +ro = List.reverse(&2, Laws.Op, ops)
  +r = Laws.ref_of(ops)
  +h = Laws.hyp_of(ops)
  %Equal.sym(Nat, Laws.total(Wer.counts(r, h)), Laws.distance(r, h), Laws.total_is_distance(r, h))
    : Laws.Holds(Nat.is_le(_, Laws.cost(ops)))
  %Equal.sym(List<&2, U32>, Laws.rev(r), Laws.ref_of(ro), ref_of_rev(ops, Nil{}))
    : Laws.Holds(Nat.is_le(Laws.lev(_, Laws.rev(h)), Laws.cost(ops)))
  %Equal.sym(List<&2, U32>, Laws.rev(h), Laws.hyp_of(ro), hyp_of_rev(ops, Nil{}))
    : Laws.Holds(Nat.is_le(Laws.lev(Laws.ref_of(ro), _), Laws.cost(ops)))
  %Equal.sym(Nat, Laws.cost(ops), Laws.cost(ro),
    Equal.trans(Nat, Laws.cost(ops), Nat.add(Laws.cost(ops), 0n), Laws.cost(ro), add_zero(Laws.cost(ops)), cost_rev(ops, Nil{})))
    : Laws.Holds(Nat.is_le(Laws.lev(Laws.ref_of(ro), Laws.hyp_of(ro)), _))
  lev_script(ro)

# Symmetry
# --------

def nat_eq_sym(a: Nat, b: Nat) -> {Cmp.is_eq(Nat.cmp(a, b)) == Cmp.is_eq(Nat.cmp(b, a)) : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      nat_eq_sym(p, q)

def u32_eq_sym(+x: U32, +y: U32) -> {U32.is_eq(x, y) == U32.is_eq(y, x) : Bool}:
  +nx = U32.to_nat(x)
  +ny = U32.to_nat(y)
  %Equal.sym(Cmp, U32.cmp(x, y), Nat.cmp(nx, ny), W.U32.cmp_to_nat(x, y)) : {Cmp.is_eq(_) == U32.is_eq(y, x) : Bool}
  %Equal.sym(Cmp, U32.cmp(y, x), Nat.cmp(ny, nx), W.U32.cmp_to_nat(y, x)) : {Cmp.is_eq(Nat.cmp(nx, ny)) == Cmp.is_eq(_) : Bool}
  nat_eq_sym(nx, ny)

def lev_sym(xs: List<&2, U32>, ys: List<&2, U32>) -> {Laws.lev(xs, ys) == Laws.lev(ys, xs) : Nat}:
  match xs ys:
    case Nil{} Nil{}:
      {==}
    case Nil{} y <> yt:
      {==}
    case x <> xt Nil{}:
      {==}
    case +x <> +xt +y <> +yt:
      +dg = Nat.add(Laws.lev(xt, yt), Laws.unit(U32.is_eq(x, y)))
      +up = {1n+Laws.lev(xt, y <> yt) : Nat}
      +lf = {1n+Laws.lev(x <> xt, yt) : Nat}
      +lv = Nat.min(dg, Nat.min(up, lf))
      %lev_sym(xt, yt) : {lv == Nat.min(Nat.add(_, Laws.unit(U32.is_eq(y, x))),
        Nat.min(1n+Laws.lev(yt, x <> xt), 1n+Laws.lev(y <> yt, xt))) : Nat}
      %lev_sym(x <> xt, yt) : {lv == Nat.min(Nat.add(Laws.lev(xt, yt), Laws.unit(U32.is_eq(y, x))),
        Nat.min(1n+_, 1n+Laws.lev(y <> yt, xt))) : Nat}
      %lev_sym(xt, y <> yt) : {lv == Nat.min(Nat.add(Laws.lev(xt, yt), Laws.unit(U32.is_eq(y, x))),
        Nat.min(lf, 1n+_)) : Nat}
      %u32_eq_sym(x, y) : {lv == Nat.min(Nat.add(Laws.lev(xt, yt), Laws.unit(_)), Nat.min(lf, up)) : Nat}
      %min_comm(up, lf) : {lv == Nat.min(dg, _) : Nat}
      {==}

def Laws.symmetric(ref, hyp):
  %Equal.sym(Nat, Laws.total(Wer.counts(ref, hyp)), Laws.distance(ref, hyp), Laws.total_is_distance(ref, hyp))
    : {_ == Laws.total(Wer.counts(hyp, ref)) : Nat}
  %Equal.sym(Nat, Laws.total(Wer.counts(hyp, ref)), Laws.distance(hyp, ref), Laws.total_is_distance(hyp, ref))
    : {Laws.distance(ref, hyp) == _ : Nat}
  lev_sym(Laws.rev(ref), Laws.rev(hyp))

# Parallel driver
# ---------------

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

def map_app(xs: List<&2, Wer.Utt>, +ys: List<&2, Wer.Utt>) ->
  {List.append(&2, Wer.Counts, Wer.map(xs), Wer.map(ys)) == Wer.map(List.append(&2, Wer.Utt, xs, ys)) : List<&2, Wer.Counts>}:
  match xs:
    case Nil{}:
      {==}
    case +h <> +t:
      %map_app(t, ys) : {Wer.Utt.counts(h) <> List.append(&2, Wer.Counts, Wer.map(t), Wer.map(ys)) == Wer.Utt.counts(h) <> _ : List<&2, Wer.Counts>}
      {==}

def leaves(t: Wer.Tree) -> List<&2, Wer.Utt>:
  match t:
    case Wer.TLeaf{us}:
      us
    case Wer.TNode{l, r}:
      List.append(&2, Wer.Utt, leaves(l), leaves(r))

def solve_leaves(t: Wer.Tree) -> {Wer.map(leaves(t)) == Wer.solve.tree(t) : List<&2, Wer.Counts>}:
  match t:
    case Wer.TLeaf{us}:
      {==}
    case Wer.TNode{+l, +r}:
      +ll = leaves(l)
      +lr = leaves(r)
      %solve_leaves(l) : {Wer.map(List.append(&2, Wer.Utt, ll, lr))
        == List.append(&2, Wer.Counts, _, Wer.solve.tree(r)) : List<&2, Wer.Counts>}
      %solve_leaves(r) : {Wer.map(List.append(&2, Wer.Utt, ll, lr))
        == List.append(&2, Wer.Counts, Wer.map(ll), _) : List<&2, Wer.Counts>}
      Equal.sym(List<&2, Wer.Counts>, List.append(&2, Wer.Counts, Wer.map(ll), Wer.map(lr)),
        Wer.map(List.append(&2, Wer.Utt, ll, lr)), map_app(ll, lr))

def tree_leaves(+d: Nat, +k: Nat, +us: List<&2, Wer.Utt>) -> {us == leaves(Wer.tree(d, k, us)) : List<&2, Wer.Utt>}:
  match d:
    case 0n:
      {==}
    case 1n+p:
      +h = Wer.half(k)
      +tk = List.take(&2, Wer.Utt, us, h)
      +dr = List.drop(&2, Wer.Utt, us, h)
      %tree_leaves(p, h, tk) : {us == List.append(&2, Wer.Utt, _, leaves(Wer.tree(p, Nat.sub(k, h), dr))) : List<&2, Wer.Utt>}
      %tree_leaves(p, Nat.sub(k, h), dr) : {us == List.append(&2, Wer.Utt, tk, _) : List<&2, Wer.Utt>}
      %take_drop(Wer.Utt, us, h) : {_ == List.append(&2, Wer.Utt, tk, dr) : List<&2, Wer.Utt>}
      {==}

def Laws.solve_spec(d, k, us):
  %solve_leaves(Wer.tree(d, k, us)) : {_ == Wer.map(us) : List<&2, Wer.Counts>}
  %tree_leaves(d, k, us) : {Wer.map(_) == Wer.map(us) : List<&2, Wer.Counts>}
  {==}
