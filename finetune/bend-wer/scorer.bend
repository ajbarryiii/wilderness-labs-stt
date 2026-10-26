# scorer.bend: the CLI around wer.bend.
#
#   bend scorer.bend -o build/bend-wer
#   build/bend-wer IDS OFFSETS COUNTS [--threads N]   (bend 2.0.37; +RTS .. -RTS from 2.0.38)
#
# IDS holds the word ids of every utterance, ref then hyp, as little-endian u32; OFFSETS the 2k + 1
# offsets into it (utterance u: ref [o[2u], o[2u + 1]), hyp [o[2u + 1], o[2u + 2])). COUNTS gets
# 3k u32: S, D, I of each utterance. Reads both inputs with bend-planner's U32s effects, validates
# everything, and only then opens and writes COUNTS. Malformed input exits 1 with a message on
# stderr and writes nothing; a wrong argument count exits 2. This file imports foreign code, so it
# holds no laws: they live in LAWS.bend, which imports wer.bend only.
import Base
import ./wer.bend as Wer
import ../bend-planner/plan.bend as Plan

# Effects (bend-planner's effs/u32io.c and .js)
# -------

def U32s.read(file: File, offset: U32, n: U32, a: Array<U32>) ->
  IO(File & (Array<U32> & Result<&1, &1, U32 & String, U32>)):
  import "../bend-planner/effs/u32io.c"
  import "../bend-planner/effs/u32io.js"

def U32s.write(file: File, n: U32, a: Array<U32>) ->
  IO(File & (Array<U32> & Result<&1, &1, U32 & String, Unit>)):
  import "../bend-planner/effs/u32io.c"
  import "../bend-planner/effs/u32io.js"

def fail(-A: Type, msg: String) -> IO(A):
  IO.die(A, 1, "bend-wer: " ++ msg)

# Reading
# -------

def load.full(+path: String, +n: U32, a: Array<U32>, ok: Bool) -> IO(Array<U32> & U32):
  match ok:
    case True{}:
      IO.pure(Array<U32> & U32, (a, n))
    case False{}:
      fail(Array<U32> & U32, path ++ ": short read")

def load.got(+path: String, +n: U32, m: File & (Array<U32> & Result<&1, &1, U32 & String, U32>)) ->
  IO(Array<U32> & U32):
  (f, ar) = m
  (a, r) = ar
  do IO<Array<U32> & U32>:
    +got : U32 <- IO.pass(U32, r)
    u : Unit <- File.close(f)
    load.full(path, n, a, U32.is_eq(got, n))

def load.check(+path: String, f: File, +size: U32, ok: Bool) -> IO(Array<U32> & U32):
  match ok:
    case True{}:
      +n = U32.shrn(size, 2n)
      do IO<Array<U32> & U32>:
        m : File & (Array<U32> & Result<&1, &1, U32 & String, U32>) <-
          U32s.read(f, 0, n, Array.new(U32, Wer.depth(n), 0))
        load.got(path, n, m)
    case False{}:
      fail(Array<U32> & U32, path ++ ": length " ++ U32.show(size) ++ " is not a multiple of 4")

def load.sized(+path: String, m: File & Result<&1, &1, U32 & String, U32>) -> IO(Array<U32> & U32):
  (f, r) = m
  do IO<Array<U32> & U32>:
    +size : U32 <- IO.pass(U32, r)
    load.check(path, f, size, U32.is_zero(U32.and(size, 3)))

def load.opened(+path: String, r: Result<&1, &1, U32 & String, File>) -> IO(Array<U32> & U32):
  match r:
    case Fail{(code, msg)}:
      fail(Array<U32> & U32, path ++ ": " ++ msg)
    case Done{f}:
      do IO<Array<U32> & U32>:
        m : File & Result<&1, &1, U32 & String, U32> <- File.size(f)
        load.sized(path, m)

# The u32 array a file holds, and its length.
def load(+path: String) -> IO(Array<U32> & U32):
  do IO<Array<U32> & U32>:
    r : Result<&1, &1, U32 & String, File> <- File.open(path, "r")
    load.opened(path, r)

# Writing
# -------

def save.done(m: File & (Array<U32> & Result<&1, &1, U32 & String, Unit>)) -> IO(Unit):
  (f, ar) = m
  (a, r) = ar
  do IO<Unit>:
    u : Unit <- IO.pass(Unit, r)
    File.close(f)

def save.opened(+path: String, +n: U32, a: Array<U32>, r: Result<&1, &1, U32 & String, File>) -> IO(Unit):
  match r:
    case Fail{(code, msg)}:
      fail(Unit, path ++ ": " ++ msg)
    case Done{f}:
      do IO<Unit>:
        m : File & (Array<U32> & Result<&1, &1, U32 & String, Unit>) <- U32s.write(f, n, a)
        save.done(m)

# Cells 0 .. n-1 of a into the file at path.
def save(+path: String, +n: U32, a: Array<U32>) -> IO(Unit):
  do IO<Unit>:
    r : Result<&1, &1, U32 & String, File> <- File.open(path, "w")
    save.opened(path, n, a, r)

# Run
# ---

# k utterances: 3k cells of S, D, I.
def score.if(ws: List<&2, U32>, +os: List<&2, U32>, +k: U32, cp: String, bad: Maybe<&2, String>) -> IO(Unit):
  match bad:
    case Some{msg}:
      fail(Unit, msg)
    case None{}:
      +n = (k * 3 : U32)
      save(cp, n, Wer.out(Wer.batch(ws, os), Array.new(U32, Wer.depth(n), 0), 0))

def score.lists(+n: U32, +m: U32, cp: String, wl: Array<U32> & List<&2, U32>, ol: Array<U32> & List<&2, U32>) ->
  IO(Unit):
  (wa, ws) = wl
  (oa, +os) = ol
  score.if(ws, os, U32.shrn((m - 1 : U32), 1n), cp, Wer.check(os, n))

def score.offsets(cp: String, wn: Array<U32> & U32, on: Array<U32> & U32) -> IO(Unit):
  (wa, +n) = wn
  (oa, +m) = on
  score.lists(n, m, cp, Plan.list(n, wa), Plan.list(m, oa))

def score.ids(op: String, cp: String, wn: Array<U32> & U32) -> IO(Unit):
  do IO<Unit>:
    on : Array<U32> & U32 <- load(op)
    score.offsets(cp, wn, on)

def args(xs: List<String>) -> IO(Unit):
  match xs:
    case _ <> ip <> op <> cp <> Nil{}:
      do IO<Unit>:
        wn : Array<U32> & U32 <- load(ip)
        score.ids(op, cp, wn)
    case _:
      IO.die(Unit, 2, "usage: bend-wer IDS OFFSETS COUNTS")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    args(xs)
