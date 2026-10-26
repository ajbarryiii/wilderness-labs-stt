# wer.bend: the project's word edit counts (finetune/whisper-ternary/wer.py edit_counts), with no
# foreign imports, so LAWS.bend and PROOF.bend may import it. scorer.bend does the file I/O and the
# CLI around it.
#
#   Wer.counts(ref, hyp)        Counts{m, s, d, i}: matches, substitutions, deletions, insertions
#   Wer.check(offsets, n)       None{} when the utterance offsets are valid for n word ids, else Some{message}
#   Wer.batch(ids, offsets)     the Counts of every utterance, in order, solved in parallel
#
# edit_counts fills the Levenshtein table over prefixes, cost[i][j] = d(ref[:i], hyp[:j]), and walks
# back from cost[n][m], preferring the diagonal (match or substitution), then a deletion, then an
# insertion. Reversing both word lists turns prefixes into suffixes: the same table, read from the
# other corner. So Wer.edits fills the table over suffixes, row by row from the last word up, and
# each cell holds its cost and the counts of the walk that starts there; a cell's walk takes one
# step to a cell already filled, so the walk is computed with the table, in one pass, keeping two
# rows. Wer.counts reverses the lists and reads the corner cell.
import Base

# Counts
# ------

type Counts is Data:
  Counts{m: Nat, s: Nat, d: Nat, i: Nat}

def Counts.zero() -> Counts:
  Counts{0n, 0n, 0n, 0n}

# One diagonal step: a match when the words are equal, else a substitution.
def Counts.diag(eq: Bool, k: Counts) -> Counts:
  match eq k:
    case True{} Counts{m, s, d, i}:
      Counts{1n+m, s, d, i}
    case False{} Counts{m, s, d, i}:
      Counts{m, 1n+s, d, i}

def Counts.del(k: Counts) -> Counts:
  match k:
    case Counts{m, s, d, i}:
      Counts{m, s, 1n+d, i}

def Counts.ins(k: Counts) -> Counts:
  match k:
    case Counts{m, s, d, i}:
      Counts{m, s, d, 1n+i}

# The table
# ---------

# A cell of the table: its cost, and the counts of the walk from it to the far corner.
type Cell is Data:
  Cell{c: Nat, k: Counts}

def Cell.cost(x: Cell) -> Nat:
  match x:
    case Cell{c, k}:
      c

def Cell.counts(x: Cell) -> Counts:
  match x:
    case Cell{c, k}:
      k

def Cell.ins(x: Cell) -> Cell:
  match x:
    case Cell{c, k}:
      Cell{1n+c, Counts.ins(k)}

def Cell.del(x: Cell) -> Cell:
  match x:
    case Cell{c, k}:
      Cell{1n+c, Counts.del(k)}

# The substitution cost of a pair of words that are (True) or are not (False) equal.
def differ(eq: Bool) -> Nat:
  match eq:
    case True{}:
      0n
    case False{}:
      1n

def pick.up(up: Bool, ku: Counts, kl: Counts) -> Counts:
  match up:
    case True{}:
      Counts.del(ku)
    case False{}:
      Counts.ins(kl)

# The step edit_counts takes: the diagonal when it costs the cell's cost, else the deletion when it
# does, else the insertion.
def pick(dg: Bool, up: Bool, eq: Bool, ku: Counts, kg: Counts, kl: Counts) -> Counts:
  match dg:
    case True{}:
      Counts.diag(eq, kg)
    case False{}:
      pick.up(up, ku, kl)

# The cell for words x, y (equal when eq) from the cells below it (deletion), diagonally below
# (match or substitution) and to its right (insertion).
def step(+eq: Bool, up: Cell, dg: Cell, lf: Cell) -> Cell:
  match up dg lf:
    case Cell{+u, ku} Cell{+g, kg} Cell{+l, kl}:
      +dc = Nat.add(g, differ(eq))
      +uc = {1n+u : Nat}
      +c = Nat.min(dc, Nat.min(uc, 1n+l))
      Cell{c, pick(Nat.is_eq(c, dc), Nat.is_eq(c, uc), eq, ku, kg, kl)}

def base.cons(r: List<&2, Cell>) -> List<&2, Cell>:
  match r:
    case Nil{}:
      Nil{}
    case +h <> t:
      Cell.ins(h) <> (h <> t)

# The bottom row (no ref words left), one cell per suffix of ys, longest first.
def base(ys: List<&2, U32>) -> List<&2, Cell>:
  match ys:
    case Nil{}:
      [Cell{0n, Counts.zero()}]
    case y <> t:
      base.cons(base(t))

def row.cons(+x: U32, +y: U32, b0: Cell, rest: List<&2, Cell>, r: List<&2, Cell>) -> List<&2, Cell>:
  match rest r:
    case +b1 <> bt +l <> rt:
      step(U32.is_eq(x, y), b0, b1, l) <> (l <> rt)
    case _ _:
      Nil{}

# The row for ref word x over ys, from the row below it (one cell per suffix of ys, longest first).
def row(+x: U32, ys: List<&2, U32>, below: List<&2, Cell>) -> List<&2, Cell>:
  match ys below:
    case Nil{} b <> bt:
      [Cell.del(b)]
    case +y <> yt b0 <> +rest:
      row.cons(x, y, b0, rest, row(x, yt, rest))
    case _ _:
      Nil{}

# The top row of the table of xs against ys.
def top(xs: List<&2, U32>, +ys: List<&2, U32>) -> List<&2, Cell>:
  match xs:
    case Nil{}:
      base(ys)
    case x <> t:
      row(x, ys, top(t, ys))

def first(r: List<&2, Cell>) -> Counts:
  match r:
    case Nil{}:
      Counts.zero()
    case h <> t:
      Cell.counts(h)

# The counts of the walk from the corner of the suffix table of xs against ys.
def edits(xs: List<&2, U32>, ys: List<&2, U32>) -> Counts:
  first(top(xs, ys))

# edit_counts(ref, hyp), with the matches.
def counts(ref: List<&2, U32>, hyp: List<&2, U32>) -> Counts:
  edits(List.reverse(&2, U32, ref), List.reverse(&2, U32, hyp))

# Utterances
# ----------

type Utt is Data:
  Utt{ref: List<&2, U32>, hyp: List<&2, U32>}

def Utt.counts(u: Utt) -> Counts:
  match u:
    case Utt{r, h}:
      counts(r, h)

def map(us: List<&2, Utt>) -> List<&2, Counts>:
  match us:
    case Nil{}:
      Nil{}
    case u <> t:
      Utt.counts(u) <> map(t)

def half(n: Nat) -> Nat:
  match n:
    case 0n:
      0n
    case 1n+0n:
      0n
    case 1n+1n+p:
      1n+half(p)

type Tree is Data:
  TLeaf{us: List<&2, Utt>}
  TNode{l: Tree, r: Tree}

# The k utterances us as a balanced tree of depth d: halved by count at each level.
def tree(+d: Nat, +k: Nat, +us: List<&2, Utt>) -> Tree:
  match d:
    case 0n:
      TLeaf{us}
    case 1n+p:
      +h = half(k)
      TNode{tree(p, h, List.take(&2, Utt, us, h)), tree(p, Nat.sub(k, h), List.drop(&2, Utt, us, h))}

# The counts of each leaf's utterances, in order, the two halves of every node in parallel.
def solve.tree(t: Tree) -> List<&2, Counts>:
  match t:
    case TLeaf{us}:
      map(us)
    case TNode{l, r}:
      a b = solve.tree(l) solve.tree(r)
      List.append(&2, Counts, a, b)

# The counts of the k utterances us, split d times. The tree is built first: in bend 2.0.37 a
# parallel call whose arguments still had to be cut (List.take, List.drop) kept only two cores busy.
def solve(+d: Nat, +k: Nat, +us: List<&2, Utt>) -> List<&2, Counts>:
  solve.tree(tree(d, k, us))

# Segments
# --------

# The gaps between consecutive offsets.
def sizes(os: List<&2, U32>, +lo: U32) -> List<&2, Nat>:
  match os:
    case Nil{}:
      Nil{}
    case +h <> t:
      U32.to_nat((h - lo : U32)) <> sizes(t, h)

# A segment in progress: its words, and the words after it.
def Seg() -> Type:
  List<&2, U32> & List<&2, U32>

def take.cons(+x: U32, r: Seg()) -> Seg():
  (xs, rest) = r
  (x <> xs, rest)

# The first d words of ws, and the words after them.
def take(d: Nat, ws: List<&2, U32>) -> Seg():
  match d:
    case 0n:
      (Nil{}, ws)
    case 1n+p:
      match ws:
        case Nil{}:
          (Nil{}, Nil{})
        case +w <> t:
          take.cons(w, take(p, t))

# r's words, then the segments of sizes ds over r's remaining words, in one pass.
def segs(ds: List<&2, Nat>, r: Seg()) -> List<&2, List<&2, U32>>:
  match ds:
    case Nil{}:
      (xs, rest) = r
      [xs]
    case +d <> t:
      (xs, rest) = r
      xs <> segs(t, take(d, rest))

# Consecutive segments, paired as (ref, hyp); an unpaired last segment is dropped.
def pairs(ss: List<&2, List<&2, U32>>) -> List<&2, Utt>:
  match ss:
    case r <> (h <> t):
      Utt{r, h} <> pairs(t)
    case _:
      Nil{}

def utts.split(ds: List<&2, Nat>, ws: List<&2, U32>) -> List<&2, Utt>:
  match ds:
    case Nil{}:
      Nil{}
    case +d <> t:
      pairs(segs(t, take(d, ws)))

# The utterances of the word ids ws and the offsets os (validated by Wer.check): utterance u has
# ref ws[os[2u] .. os[2u + 1]) and hyp ws[os[2u + 1] .. os[2u + 2]).
def utts(ws: List<&2, U32>, os: List<&2, U32>) -> List<&2, Utt>:
  match os:
    case Nil{}:
      Nil{}
    case +h <> t:
      utts.split(sizes(t, h), ws)

# Validation
# ----------

def check.if(ok: Bool, msg: String, rest: Maybe<&2, String>) -> Maybe<&2, String>:
  match ok:
    case True{}:
      rest
    case False{}:
      Some{msg}

# The message is only built when the check fails.
def check.order(ok: Bool, +h: U32, +prev: U32, rest: Maybe<&2, String>) -> Maybe<&2, String>:
  match ok:
    case True{}:
      rest
    case False{}:
      Some{"offsets decrease: " ++ U32.show(h) ++ " after " ++ U32.show(prev)}

def check.end(ok: Bool, +prev: U32, +n: U32) -> Maybe<&2, String>:
  match ok:
    case True{}:
      None{}
    case False{}:
      Some{"offsets must end at n=" ++ U32.show(n) ++ ", not " ++ U32.show(prev)}

# The offsets after one at prev, with an odd count so far when odd: none decreases, the last is
# n, and there is an odd number of them in all.
def check.go(ps: List<&2, U32>, +prev: U32, +n: U32, odd: Bool) -> Maybe<&2, String>:
  match ps:
    case Nil{}:
      check.if(odd, "offsets holds an even number of entries: it needs 2 * utterances + 1",
        check.end(U32.is_eq(prev, n), prev, n))
    case +h <> t:
      check.order(U32.is_le(prev, h), h, prev, check.go(t, h, n, Bool.not(odd)))

def check.first(ok: Bool, +h: U32, rest: Maybe<&2, String>) -> Maybe<&2, String>:
  match ok:
    case True{}:
      rest
    case False{}:
      Some{"offsets must start at 0, not " ++ U32.show(h)}

# None{} when os is a valid offset list for n word ids (2k + 1 offsets for k utterances, from 0
# to n, none decreasing), else the reason.
def check(os: List<&2, U32>, +n: U32) -> Maybe<&2, String>:
  match os:
    case Nil{}:
      Some{"offsets is empty: it needs 2 * utterances + 1 entries"}
    case +h <> t:
      check.first(U32.is_zero(h), h, check.go(t, h, n, True{}))

# Output
# ------

# The S, D, I of each counts into cells at, at + 1, at + 2, ..
def out(ks: List<&2, Counts>, a: Array<U32>, +at: U32) -> Array<U32>:
  match ks:
    case Nil{}:
      a
    case Counts{m, s, d, i} <> t:
      a[at] <- U32.from_nat(s)
      a[(at + 1 : U32)] <- U32.from_nat(d)
      a[(at + 2 : U32)] <- U32.from_nat(i)
      out(t, a, (at + 3 : U32))

def depth.if(one: Bool, +k: U32) -> Nat:
  match one:
    case True{}:
      0n
    case False{}:
      1n+U32.log2((k - 1 : U32))

# The least d with 2^d >= k.
def depth(+k: U32) -> Nat:
  depth.if(U32.is_le(k, 1), k)

def batch.go(+us: List<&2, Utt>) -> List<&2, Counts>:
  +k = List.length(&2, Utt, us)
  solve(depth(U32.from_nat(k)), k, us)

# The counts of every utterance of the word ids ws and the validated offsets os, in order.
def batch(ws: List<&2, U32>, os: List<&2, U32>) -> List<&2, Counts>:
  batch.go(utts(ws, os))
