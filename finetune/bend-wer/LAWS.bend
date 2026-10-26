# LAWS.bend: the laws of the edit counts, stated over wer.bend, which imports no foreign code.
# PROOF.bend proves them all. The vocabulary below is written for the laws: the distance is the
# textbook Levenshtein recurrence, with its own unit costs, not wer.bend's table, so a law cannot
# hold merely by restating the code.
#
#   total_is_distance   S + D + I is the Levenshtein distance (the value of edit_counts' table)
#   ref_length          M + S + D = len(ref)
#   hyp_length          M + S + I = len(hyp)
#   identical           ref == hyp gives M = len(ref) and S = D = I = 0
#   lower_bound         |len(ref) - len(hyp)| <= S + D + I
#   upper_bound         S + D + I <= max(len(ref), len(hyp))
#   optimal             no alignment of ref and hyp costs less than S + D + I
#   symmetric           S + D + I is the same with ref and hyp swapped
#   solve_spec          the parallel driver computes each utterance's counts, in order
#
# None of them fixes the tie-break between alignments of equal cost: S, D and I individually
# are pinned by the fidelity tests against edit_counts (tests/), not here.
import Base
import ./wer.bend as Wer

# Vocabulary
# ----------

def Holds(b: Bool) -> Data:
  {b == True{} : Bool}

def len(xs: List<&2, U32>) -> Nat:
  List.length(&2, U32, xs)

def rev(xs: List<&2, U32>) -> List<&2, U32>:
  List.reverse(&2, U32, xs)

# The cost of putting word x against word y: 0 if they are equal, else 1.
def unit(b: Bool) -> Nat:
  match b:
    case True{}:
      0n
    case False{}:
      1n

# The Levenshtein distance by its recurrence on the first words: the least cost of turning xs
# into ys by substitutions, deletions and insertions, each costing 1.
def lev(xs: List<&2, U32>, ys: List<&2, U32>) -> Nat:
  match xs ys:
    case Nil{} _:
      len(ys)
    case x <> xt Nil{}:
      1n+len(xt)
    case +x <> +xt +y <> +yt:
      Nat.min(Nat.add(lev(xt, yt), unit(U32.is_eq(x, y))),
        Nat.min(1n+lev(xt, y <> yt), 1n+lev(x <> xt, yt)))

# The distance of ref and hyp by the recurrence on their last words, which is how edit_counts
# fills its table: its cost[i][j] is lev(rev(ref[:i]), rev(hyp[:j])).
def distance(ref: List<&2, U32>, hyp: List<&2, U32>) -> Nat:
  lev(rev(ref), rev(hyp))

def M(k: Wer.Counts) -> Nat:
  match k:
    case Wer.Counts{m, s, d, i}:
      m

def S(k: Wer.Counts) -> Nat:
  match k:
    case Wer.Counts{m, s, d, i}:
      s

def D(k: Wer.Counts) -> Nat:
  match k:
    case Wer.Counts{m, s, d, i}:
      d

def I(k: Wer.Counts) -> Nat:
  match k:
    case Wer.Counts{m, s, d, i}:
      i

# S + D + I
def total(k: Wer.Counts) -> Nat:
  match k:
    case Wer.Counts{m, s, d, i}:
      Nat.add(s, Nat.add(d, i))

# |a - b|
def gap(+a: Nat, +b: Nat) -> Nat:
  Nat.add(Nat.sub(a, b), Nat.sub(b, a))

# An alignment, as the edit script of its columns: a word kept, a word replaced by another, a
# ref word dropped, a hyp word inserted.
type Op is Data:
  Keep{w: U32}
  Subst{a: U32, b: U32}
  Drop{a: U32}
  Add{b: U32}

# The ref words a script reads, in order
def ref_of(ops: List<&2, Op>) -> List<&2, U32>:
  match ops:
    case Nil{}:
      Nil{}
    case Keep{w} <> t:
      w <> ref_of(t)
    case Subst{a, b} <> t:
      a <> ref_of(t)
    case Drop{a} <> t:
      a <> ref_of(t)
    case Add{b} <> t:
      ref_of(t)

# The hyp words a script writes, in order
def hyp_of(ops: List<&2, Op>) -> List<&2, U32>:
  match ops:
    case Nil{}:
      Nil{}
    case Keep{w} <> t:
      w <> hyp_of(t)
    case Subst{a, b} <> t:
      b <> hyp_of(t)
    case Drop{a} <> t:
      hyp_of(t)
    case Add{b} <> t:
      b <> hyp_of(t)

# A script's cost: 1 for every column but a kept word (a Subst of a word by itself still costs 1)
def cost(ops: List<&2, Op>) -> Nat:
  match ops:
    case Nil{}:
      0n
    case Keep{w} <> t:
      cost(t)
    case _ <> t:
      1n+cost(t)

# Laws: one utterance
# -------------------

# S + D + I is the distance the table computes
law total_is_distance:
  for +ref: List<&2, U32>
  for +hyp: List<&2, U32>
  {total(Wer.counts(ref, hyp)) == distance(ref, hyp) : Nat}

# every ref word is matched, substituted or deleted
law ref_length:
  for +ref: List<&2, U32>
  for +hyp: List<&2, U32>
  {Nat.add(M(Wer.counts(ref, hyp)), Nat.add(S(Wer.counts(ref, hyp)), D(Wer.counts(ref, hyp)))) == len(ref) : Nat}

# every hyp word is matched, substituted or inserted
law hyp_length:
  for +ref: List<&2, U32>
  for +hyp: List<&2, U32>
  {Nat.add(M(Wer.counts(ref, hyp)), Nat.add(S(Wer.counts(ref, hyp)), I(Wer.counts(ref, hyp)))) == len(hyp) : Nat}

# a hypothesis equal to its reference has no errors
law identical:
  for +ws: List<&2, U32>
  {Wer.counts(ws, ws) == Wer.Counts{len(ws), 0n, 0n, 0n} : Wer.Counts}

law lower_bound:
  for +ref: List<&2, U32>
  for +hyp: List<&2, U32>
  Holds(Nat.is_le(gap(len(ref), len(hyp)), total(Wer.counts(ref, hyp))))

law upper_bound:
  for +ref: List<&2, U32>
  for +hyp: List<&2, U32>
  Holds(Nat.is_le(total(Wer.counts(ref, hyp)), Nat.max(len(ref), len(hyp))))

# every alignment of the words it reads and writes costs at least S + D + I
law optimal:
  for +ops: List<&2, Op>
  Holds(Nat.is_le(total(Wer.counts(ref_of(ops), hyp_of(ops))), cost(ops)))

law symmetric:
  for +ref: List<&2, U32>
  for +hyp: List<&2, U32>
  {total(Wer.counts(ref, hyp)) == total(Wer.counts(hyp, ref)) : Nat}

# Laws: the batch
# ---------------

# the parallel driver (halving by count, d times) computes each utterance's counts in order
law solve_spec:
  for +d : Nat
  for +k : Nat
  for +us: List<&2, Wer.Utt>
  {Wer.solve(d, k, us) == Wer.map(us) : List<&2, Wer.Counts>}
