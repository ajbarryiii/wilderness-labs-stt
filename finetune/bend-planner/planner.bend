# planner.bend: the CLI around plan.bend (DESIGN.md "CLI").
#
#   bend planner.bend -o build/planner
#   build/planner SAMPLES POOLS BUDGET ORDER OFFSETS [--threads N]   (bend 2.0.37; +RTS .. -RTS from 2.0.38)
#
# Reads the two little-endian u32 files into arrays with the U32s effects (effs/u32io.c), validates
# everything, and only then opens and writes ORDER and OFFSETS. Malformed input exits 1 with a
# message on stderr and writes nothing; a wrong argument count exits 2. This file imports foreign
# code, so it holds no laws: they live in LAWS.bend, which imports plan.bend only.
import Base
import ./plan.bend as Plan

# Effects
# -------

def U32s.read(file: File, offset: U32, n: U32, a: Array<U32>) ->
  IO(File & (Array<U32> & Result<&1, &1, U32 & String, U32>)):
  import "./effs/u32io.c"
  import "./effs/u32io.js"

def U32s.write(file: File, n: U32, a: Array<U32>) ->
  IO(File & (Array<U32> & Result<&1, &1, U32 & String, Unit>)):
  import "./effs/u32io.c"
  import "./effs/u32io.js"

def fail(-A: Type, msg: String) -> IO(A):
  IO.die(A, 1, "planner: " ++ msg)

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
          U32s.read(f, 0, n, Array.new(U32, Plan.depth(n), 0))
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

def emit(op: String, fp: String, o: Plan.Out) -> IO(Unit):
  match o:
    case Plan.Out{order, offs, +at, +nb}:
      do IO<Unit>:
        save(op, at, order)
        save(fp, (nb + 1 : U32), offs)

def plan.if(+b: U32, ss: List<&2, U32>, +ps: List<&2, U32>, +n: U32, op: String, fp: String,
  bad: Maybe<&2, String>) -> IO(Unit):
  match bad:
    case Some{msg}:
      fail(Unit, msg)
    case None{}:
      +d = Plan.depth((n + 1 : U32))
      emit(op, fp, Plan.run(b, ss, ps, Nat.sub(List.length(&2, U32, ps), 1n),
        Array.new(U32, d, 0), Array.new(U32, d, 0)))

def plan.lists(+b: U32, +n: U32, op: String, fp: String, sl: Array<U32> & List<&2, U32>,
  pl: Array<U32> & List<&2, U32>) -> IO(Unit):
  (sa, ss) = sl
  (pa, +ps) = pl
  plan.if(b, ss, ps, n, op, fp, Plan.check(ps, n))

def plan.pools(+b: U32, sn: Array<U32> & U32, op: String, fp: String, pn: Array<U32> & U32) -> IO(Unit):
  (sa, +n) = sn
  (pa, +k) = pn
  plan.lists(b, n, op, fp, Plan.list(n, sa), Plan.list(k, pa))

def plan.samples(+b: U32, pp: String, op: String, fp: String, sn: Array<U32> & U32) -> IO(Unit):
  do IO<Unit>:
    pn : Array<U32> & U32 <- load(pp)
    plan.pools(b, sn, op, fp, pn)

def plan.budget(sp: String, pp: String, op: String, fp: String, mb: Maybe<&2, U32>, +bs: String) -> IO(Unit):
  match mb:
    case None{}:
      fail(Unit, "budget " ++ bs ++ " is not a decimal u32")
    case Some{+b}:
      do IO<Unit>:
        sn : Array<U32> & U32 <- load(sp)
        plan.samples(b, pp, op, fp, sn)

def args(xs: List<String>) -> IO(Unit):
  match xs:
    case _ <> sp <> pp <> +bs <> op <> fp <> Nil{}:
      plan.budget(sp, pp, op, fp, Plan.budget(bs), bs)
    case _:
      IO.die(Unit, 2, "usage: planner SAMPLES POOLS BUDGET ORDER OFFSETS")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    args(xs)
