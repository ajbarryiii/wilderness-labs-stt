# Pure word/natural arithmetic. The Nat library is pinned by content hash.
import Base
import 0xcd4682caa53da19986acd472c2181473/nat.bend as N

# The natural value of a low bit followed by a natural-valued tail.
def Nat.bit(b: Bool, n: Nat) -> Nat:
  match b:
    case False{}:
      Nat.double(n)
    case True{}:
      1n+Nat.double(n)

# Binary comparison agrees with comparison of the natural values.
law Nat.cmp_bit:
  for x: Nat
  for y: Nat
  for +a: Bool
  for +b: Bool
  {Nat.cmp(Nat.bit(a, x), Nat.bit(b, y)) == Word.cmp.fin(a, b, Nat.cmp(x, y)) : Cmp}

def Nat.cmp_bit(x, y, a, b):
  match x y:
    case 0n 0n:
      match a b:
        case False{} False{}:
          {==}
        case False{} True{}:
          {==}
        case True{} False{}:
          {==}
        case True{} True{}:
          {==}
    case 0n 1n+q:
      match a b:
        case False{} False{}:
          {==}
        case False{} True{}:
          {==}
        case True{} False{}:
          {==}
        case True{} True{}:
          {==}
    case 1n+p 0n:
      match a b:
        case False{} False{}:
          {==}
        case False{} True{}:
          {==}
        case True{} False{}:
          {==}
        case True{} True{}:
          {==}
    case 1n+p 1n+q:
      match a b:
        case False{} False{}:
          Nat.cmp_bit(p, q, False{}, False{})
        case False{} True{}:
          Nat.cmp_bit(p, q, False{}, True{})
        case True{} False{}:
          Nat.cmp_bit(p, q, True{}, False{})
        case True{} True{}:
          Nat.cmp_bit(p, q, True{}, True{})

# Conversion commutes with adding a low bit.
def Word.to_nat_con(-n: Nat, b: Bool, -w: Word(n)) ->
  {Nat.bit(b, Word.to_nat(n, w)) == Word.to_nat(1n+n, WCon{b, w}) : Nat}:
  match b:
    case False{}:
      {==}
    case True{}:
      {==}

# Word comparison is exactly comparison of unsigned natural values, at every width.
law Word.cmp_to_nat:
  for +n: Nat
  for a: Word(n)
  for b: Word(n)
  {Word.cmp(n, a, b) == Nat.cmp(Word.to_nat(n, a), Word.to_nat(n, b)) : Cmp}

def Word.cmp_to_nat(n, a, b):
  match n:
    case 0n:
      {==}
    case 1n+p:
      match a b:
        case WCon{+ab, +at} WCon{+bb, +bt}:
          %Word.to_nat_con(p, ab, at) : {Word.cmp.fin(ab, bb, Word.cmp(p, at, bt)) == Nat.cmp(_, Word.to_nat(1n+p, WCon{bb, bt})) : Cmp}
          %Word.to_nat_con(p, bb, bt) : {Word.cmp.fin(ab, bb, Word.cmp(p, at, bt)) == Nat.cmp(Nat.bit(ab, Word.to_nat(p, at)), _) : Cmp}
          %Equal.sym(Cmp, Nat.cmp(Nat.bit(ab, Word.to_nat(p, at)), Nat.bit(bb, Word.to_nat(p, bt))), Word.cmp.fin(ab, bb, Nat.cmp(Word.to_nat(p, at), Word.to_nat(p, bt))), Nat.cmp_bit(Word.to_nat(p, at), Word.to_nat(p, bt), ab, bb)) : {Word.cmp.fin(ab, bb, Word.cmp(p, at, bt)) == _ : Cmp}
          %Word.cmp_to_nat(p, at, bt) : {Word.cmp.fin(ab, bb, Word.cmp(p, at, bt)) == Word.cmp.fin(ab, bb, _) : Cmp}
          {==}

# U32 comparison agrees with Nat comparison without any overflow hypotheses.
law U32.cmp_to_nat:
  for a: U32
  for b: U32
  {U32.cmp(a, b) == Nat.cmp(U32.to_nat(a), U32.to_nat(b)) : Cmp}

def U32.cmp_to_nat(a, b):
  match a b:
    case U32{x} U32{y}:
      Word.cmp_to_nat(32n, x, y)

# Unsigned order is transitive; no arithmetic occurs in this statement.
law U32.le_trans:
  for +a: U32
  for +b: U32
  for +c: U32
  for ab: {U32.is_le(a, b) == True{} : Bool}
  for bc: {U32.is_le(b, c) == True{} : Bool}
  {U32.is_le(a, c) == True{} : Bool}

def U32.le_trans(a, b, c, ab, bc):
  %Equal.sym(Cmp, U32.cmp(a, c), Nat.cmp(U32.to_nat(a), U32.to_nat(c)), U32.cmp_to_nat(a, c)) : {Cmp.is_le(_) == True{} : Bool}
  N.le_trans(U32.to_nat(a), U32.to_nat(b), U32.to_nat(c),
    Equal.trans(Bool, Nat.is_le(U32.to_nat(a), U32.to_nat(b)), U32.is_le(a, b), True{},
      Equal.sym(Bool, U32.is_le(a, b), Nat.is_le(U32.to_nat(a), U32.to_nat(b)),
        Equal.cong(Cmp, Bool, Cmp.is_le, U32.cmp(a, b), Nat.cmp(U32.to_nat(a), U32.to_nat(b)), U32.cmp_to_nat(a, b))), ab),
    Equal.trans(Bool, Nat.is_le(U32.to_nat(b), U32.to_nat(c)), U32.is_le(b, c), True{},
      Equal.sym(Bool, U32.is_le(b, c), Nat.is_le(U32.to_nat(b), U32.to_nat(c)),
        Equal.cong(Cmp, Bool, Cmp.is_le, U32.cmp(b, c), Nat.cmp(U32.to_nat(b), U32.to_nat(c)), U32.cmp_to_nat(b, c))), bc))

# A bit as a natural, independent of the machine representation.
def Bool.to_nat(b: Bool) -> Nat:
  match b:
    case False{}:
      0n
    case True{}:
      1n

# The low bit and outgoing carry in a subtraction with incoming carry c
# (c = True means no incoming borrow), exactly as Base's Word.adc computes them.
def Bool.sub_low(a: Bool, b: Bool, c: Bool) -> Bool:
  Pair.fst(Bool, Bool, Bool.full_add(a, Bool.not(b), c))

def Bool.sub_carry(a: Bool, b: Bool, c: Bool) -> Bool:
  Pair.snd(Bool, Bool, Bool.full_add(a, Bool.not(b), c))

def Cmp.sub_ok(c: Bool, d: Cmp) -> Bool:
  match c:
    case False{}:
      Cmp.is_gt(d)
    case True{}:
      Cmp.is_ge(d)

# A carry allows equality; an incoming borrow requires a strict inequality.
def Nat.sub_ok(c: Bool, a: Nat, b: Nat) -> Data:
  {Cmp.sub_ok(c, Nat.cmp(a, b)) == True{} : Bool}

# Subtraction with an optional incoming borrow, in Nat (never wrapping).
def Nat.sub_carry(c: Bool, a: Nat, b: Nat) -> Nat:
  Nat.sub(a, Nat.add(Bool.to_nat(Bool.not(c)), b))

# A valid binary subtraction has a valid subtraction of the high tails.
def Cmp.sub_tail(d: Cmp, a: Bool, b: Bool, c: Bool) ->
  {Cmp.sub_ok(c, Word.cmp.fin(a, b, d)) == True{} : Bool} ->
  {Cmp.sub_ok(Bool.sub_carry(a, b, c), d) == True{} : Bool}:
  match d a b c:
    case LT{} False{} False{} False{}:
      h => Empty.absurd({False{} == True{} : Bool}, N.internal_false_ne_true(h))
    case LT{} False{} False{} True{}:
      h => Empty.absurd({False{} == True{} : Bool}, N.internal_false_ne_true(h))
    case LT{} False{} True{} False{}:
      h => Empty.absurd({False{} == True{} : Bool}, N.internal_false_ne_true(h))
    case LT{} False{} True{} True{}:
      h => Empty.absurd({False{} == True{} : Bool}, N.internal_false_ne_true(h))
    case LT{} True{} False{} False{}:
      h => Empty.absurd({False{} == True{} : Bool}, N.internal_false_ne_true(h))
    case LT{} True{} False{} True{}:
      h => Empty.absurd({False{} == True{} : Bool}, N.internal_false_ne_true(h))
    case LT{} True{} True{} False{}:
      h => Empty.absurd({False{} == True{} : Bool}, N.internal_false_ne_true(h))
    case LT{} True{} True{} True{}:
      h => Empty.absurd({False{} == True{} : Bool}, N.internal_false_ne_true(h))
    case EQ{} False{} False{} False{}:
      h => Empty.absurd({False{} == True{} : Bool}, N.internal_false_ne_true(h))
    case EQ{} False{} False{} True{}:
      h => {==}
    case EQ{} False{} True{} False{}:
      h => Empty.absurd({False{} == True{} : Bool}, N.internal_false_ne_true(h))
    case EQ{} False{} True{} True{}:
      h => Empty.absurd({False{} == True{} : Bool}, N.internal_false_ne_true(h))
    case EQ{} True{} False{} False{}:
      h => {==}
    case EQ{} True{} False{} True{}:
      h => {==}
    case EQ{} True{} True{} False{}:
      h => Empty.absurd({False{} == True{} : Bool}, N.internal_false_ne_true(h))
    case EQ{} True{} True{} True{}:
      h => {==}
    case GT{} False{} False{} False{}:
      h => {==}
    case GT{} False{} False{} True{}:
      h => {==}
    case GT{} False{} True{} False{}:
      h => {==}
    case GT{} False{} True{} True{}:
      h => {==}
    case GT{} True{} False{} False{}:
      h => {==}
    case GT{} True{} False{} True{}:
      h => {==}
    case GT{} True{} True{} False{}:
      h => {==}
    case GT{} True{} True{} True{}:
      h => {==}

def Nat.sub_tail(+x: Nat, +y: Nat, +a: Bool, +b: Bool, +c: Bool, h: Nat.sub_ok(c, Nat.bit(a, x), Nat.bit(b, y))) ->
  Nat.sub_ok(Bool.sub_carry(a, b, c), x, y):
  Cmp.sub_tail(Nat.cmp(x, y), a, b, c,
    Equal.trans(Bool, Cmp.sub_ok(c, Word.cmp.fin(a, b, Nat.cmp(x, y))), Cmp.sub_ok(c, Nat.cmp(Nat.bit(a, x), Nat.bit(b, y))), True{},
      Equal.cong(Cmp, Bool, d => Cmp.sub_ok(c, d), Word.cmp.fin(a, b, Nat.cmp(x, y)), Nat.cmp(Nat.bit(a, x), Nat.bit(b, y)),
        Equal.sym(Cmp, Nat.cmp(Nat.bit(a, x), Nat.bit(b, y)), Word.cmp.fin(a, b, Nat.cmp(x, y)), Nat.cmp_bit(x, y, a, b))), h))

# Binary subtraction agrees with Nat subtraction when its high-tail borrow is valid.
law Nat.sub_bit:
  for x: Nat
  for y: Nat
  for +a: Bool
  for +b: Bool
  for +c: Bool
  for h: Nat.sub_ok(Bool.sub_carry(a, b, c), x, y)
  {Nat.bit(Bool.sub_low(a, b, c), Nat.sub_carry(Bool.sub_carry(a, b, c), x, y))
    == Nat.sub_carry(c, Nat.bit(a, x), Nat.bit(b, y)) : Nat}

def Nat.sub_bit(x, y, a, b, c, h):
  match x y:
    case 0n 0n:
      match a b c:
        case False{} False{} False{}:
          Empty.absurd({Nat.bit(Bool.sub_low(False{}, False{}, False{}), Nat.sub_carry(Bool.sub_carry(False{}, False{}, False{}), 0n, 0n)) == Nat.sub_carry(False{}, Nat.bit(False{}, 0n), Nat.bit(False{}, 0n)) : Nat}, N.internal_false_ne_true(h))
        case False{} False{} True{}:
          {==}
        case False{} True{} False{}:
          Empty.absurd({Nat.bit(Bool.sub_low(False{}, True{}, False{}), Nat.sub_carry(Bool.sub_carry(False{}, True{}, False{}), 0n, 0n)) == Nat.sub_carry(False{}, Nat.bit(False{}, 0n), Nat.bit(True{}, 0n)) : Nat}, N.internal_false_ne_true(h))
        case False{} True{} True{}:
          Empty.absurd({Nat.bit(Bool.sub_low(False{}, True{}, True{}), Nat.sub_carry(Bool.sub_carry(False{}, True{}, True{}), 0n, 0n)) == Nat.sub_carry(True{}, Nat.bit(False{}, 0n), Nat.bit(True{}, 0n)) : Nat}, N.internal_false_ne_true(h))
        case True{} False{} False{}:
          {==}
        case True{} False{} True{}:
          {==}
        case True{} True{} False{}:
          Empty.absurd({Nat.bit(Bool.sub_low(True{}, True{}, False{}), Nat.sub_carry(Bool.sub_carry(True{}, True{}, False{}), 0n, 0n)) == Nat.sub_carry(False{}, Nat.bit(True{}, 0n), Nat.bit(True{}, 0n)) : Nat}, N.internal_false_ne_true(h))
        case True{} True{} True{}:
          {==}
    case 0n 1n++q:
      match a b c:
        case False{} False{} False{}:
          Empty.absurd({Nat.bit(Bool.sub_low(False{}, False{}, False{}), Nat.sub_carry(Bool.sub_carry(False{}, False{}, False{}), 0n, 1n+q)) == Nat.sub_carry(False{}, Nat.bit(False{}, 0n), Nat.bit(False{}, 1n+q)) : Nat}, N.internal_false_ne_true(h))
        case False{} False{} True{}:
          Empty.absurd({Nat.bit(Bool.sub_low(False{}, False{}, True{}), Nat.sub_carry(Bool.sub_carry(False{}, False{}, True{}), 0n, 1n+q)) == Nat.sub_carry(True{}, Nat.bit(False{}, 0n), Nat.bit(False{}, 1n+q)) : Nat}, N.internal_false_ne_true(h))
        case False{} True{} False{}:
          Empty.absurd({Nat.bit(Bool.sub_low(False{}, True{}, False{}), Nat.sub_carry(Bool.sub_carry(False{}, True{}, False{}), 0n, 1n+q)) == Nat.sub_carry(False{}, Nat.bit(False{}, 0n), Nat.bit(True{}, 1n+q)) : Nat}, N.internal_false_ne_true(h))
        case False{} True{} True{}:
          Empty.absurd({Nat.bit(Bool.sub_low(False{}, True{}, True{}), Nat.sub_carry(Bool.sub_carry(False{}, True{}, True{}), 0n, 1n+q)) == Nat.sub_carry(True{}, Nat.bit(False{}, 0n), Nat.bit(True{}, 1n+q)) : Nat}, N.internal_false_ne_true(h))
        case True{} False{} False{}:
          Empty.absurd({Nat.bit(Bool.sub_low(True{}, False{}, False{}), Nat.sub_carry(Bool.sub_carry(True{}, False{}, False{}), 0n, 1n+q)) == Nat.sub_carry(False{}, Nat.bit(True{}, 0n), Nat.bit(False{}, 1n+q)) : Nat}, N.internal_false_ne_true(h))
        case True{} False{} True{}:
          Empty.absurd({Nat.bit(Bool.sub_low(True{}, False{}, True{}), Nat.sub_carry(Bool.sub_carry(True{}, False{}, True{}), 0n, 1n+q)) == Nat.sub_carry(True{}, Nat.bit(True{}, 0n), Nat.bit(False{}, 1n+q)) : Nat}, N.internal_false_ne_true(h))
        case True{} True{} False{}:
          Empty.absurd({Nat.bit(Bool.sub_low(True{}, True{}, False{}), Nat.sub_carry(Bool.sub_carry(True{}, True{}, False{}), 0n, 1n+q)) == Nat.sub_carry(False{}, Nat.bit(True{}, 0n), Nat.bit(True{}, 1n+q)) : Nat}, N.internal_false_ne_true(h))
        case True{} True{} True{}:
          Empty.absurd({Nat.bit(Bool.sub_low(True{}, True{}, True{}), Nat.sub_carry(Bool.sub_carry(True{}, True{}, True{}), 0n, 1n+q)) == Nat.sub_carry(True{}, Nat.bit(True{}, 0n), Nat.bit(True{}, 1n+q)) : Nat}, N.internal_false_ne_true(h))
    case 1n++p 0n:
      match a b c:
        case False{} False{} False{}:
          %Equal.sym(Nat, Nat.sub(p, 0n), p, N.sub_zero(p)) : {Nat.bit(True{}, _) == Nat.sub_carry(False{}, Nat.bit(False{}, 1n+p), Nat.bit(False{}, 0n)) : Nat}
          {==}
        case False{} False{} True{}:
          {==}
        case False{} True{} False{}:
          %Equal.sym(Nat, Nat.sub(p, 0n), p, N.sub_zero(p)) : {Nat.bit(False{}, _) == Nat.sub_carry(False{}, Nat.bit(False{}, 1n+p), Nat.bit(True{}, 0n)) : Nat}
          %Equal.sym(Nat, Nat.sub(Nat.double(p), 0n), Nat.double(p), N.sub_zero(Nat.double(p))) : {Nat.bit(False{}, p) == _ : Nat}
          {==}
        case False{} True{} True{}:
          %Equal.sym(Nat, Nat.sub(p, 0n), p, N.sub_zero(p)) : {Nat.bit(True{}, _) == Nat.sub_carry(True{}, Nat.bit(False{}, 1n+p), Nat.bit(True{}, 0n)) : Nat}
          {==}
        case True{} False{} False{}:
          {==}
        case True{} False{} True{}:
          {==}
        case True{} True{} False{}:
          %Equal.sym(Nat, Nat.sub(p, 0n), p, N.sub_zero(p)) : {Nat.bit(True{}, _) == Nat.sub_carry(False{}, Nat.bit(True{}, 1n+p), Nat.bit(True{}, 0n)) : Nat}
          {==}
        case True{} True{} True{}:
          {==}
    case 1n++p 1n++q:
      match a b c:
        case False{} False{} False{}:
          Nat.sub_bit(p, q, False{}, False{}, False{}, h)
        case False{} False{} True{}:
          Nat.sub_bit(p, q, False{}, False{}, True{}, h)
        case False{} True{} False{}:
          Nat.sub_bit(p, q, False{}, True{}, False{}, h)
        case False{} True{} True{}:
          Nat.sub_bit(p, q, False{}, True{}, True{}, h)
        case True{} False{} False{}:
          Nat.sub_bit(p, q, True{}, False{}, False{}, h)
        case True{} False{} True{}:
          Nat.sub_bit(p, q, True{}, False{}, True{}, h)
        case True{} True{} False{}:
          Nat.sub_bit(p, q, True{}, True{}, False{}, h)
        case True{} True{} True{}:
          Nat.sub_bit(p, q, True{}, True{}, True{}, h)

# No-underflow subtraction with an optional incoming borrow, at every word width.
law Word.to_nat_sub_carry:
  for +n: Nat
  for +a: Word(n)
  for +b: Word(n)
  for +c: Bool
  for +h: Nat.sub_ok(c, Word.to_nat(n, a), Word.to_nat(n, b))
  {Word.to_nat(n, Word.adc(n, a, b, True{}, c))
    == Nat.sub_carry(c, Word.to_nat(n, a), Word.to_nat(n, b)) : Nat}

def Word.to_nat_sub_carry(n, a, b, c, h):
  match n:
    case 0n:
      match c:
        case False{}:
          Empty.absurd({0n == 0n : Nat}, N.internal_false_ne_true(h))
        case True{}:
          {==}
    case 1n+p:
      match a b c:
        case WCon{False{}, +at} WCon{False{}, +bt} False{}:
          %Equal.sym(Nat, Word.to_nat(p, Word.adc(p, at, bt, True{}, False{})), Nat.sub_carry(False{}, Word.to_nat(p, at), Word.to_nat(p, bt)), Word.to_nat_sub_carry(p, at, bt, False{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), False{}, False{}, False{}, h))) :
            {Nat.bit(True{}, _) == Nat.sub_carry(False{}, Nat.bit(False{}, Word.to_nat(p, at)), Nat.bit(False{}, Word.to_nat(p, bt))) : Nat}
          Nat.sub_bit(Word.to_nat(p, at), Word.to_nat(p, bt), False{}, False{}, False{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), False{}, False{}, False{}, h))
        case WCon{False{}, +at} WCon{False{}, +bt} True{}:
          %Equal.sym(Nat, Word.to_nat(p, Word.adc(p, at, bt, True{}, True{})), Nat.sub_carry(True{}, Word.to_nat(p, at), Word.to_nat(p, bt)), Word.to_nat_sub_carry(p, at, bt, True{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), False{}, False{}, True{}, h))) :
            {Nat.bit(False{}, _) == Nat.sub_carry(True{}, Nat.bit(False{}, Word.to_nat(p, at)), Nat.bit(False{}, Word.to_nat(p, bt))) : Nat}
          Nat.sub_bit(Word.to_nat(p, at), Word.to_nat(p, bt), False{}, False{}, True{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), False{}, False{}, True{}, h))
        case WCon{False{}, +at} WCon{True{}, +bt} False{}:
          %Equal.sym(Nat, Word.to_nat(p, Word.adc(p, at, bt, True{}, False{})), Nat.sub_carry(False{}, Word.to_nat(p, at), Word.to_nat(p, bt)), Word.to_nat_sub_carry(p, at, bt, False{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), False{}, True{}, False{}, h))) :
            {Nat.bit(False{}, _) == Nat.sub_carry(False{}, Nat.bit(False{}, Word.to_nat(p, at)), Nat.bit(True{}, Word.to_nat(p, bt))) : Nat}
          Nat.sub_bit(Word.to_nat(p, at), Word.to_nat(p, bt), False{}, True{}, False{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), False{}, True{}, False{}, h))
        case WCon{False{}, +at} WCon{True{}, +bt} True{}:
          %Equal.sym(Nat, Word.to_nat(p, Word.adc(p, at, bt, True{}, False{})), Nat.sub_carry(False{}, Word.to_nat(p, at), Word.to_nat(p, bt)), Word.to_nat_sub_carry(p, at, bt, False{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), False{}, True{}, True{}, h))) :
            {Nat.bit(True{}, _) == Nat.sub_carry(True{}, Nat.bit(False{}, Word.to_nat(p, at)), Nat.bit(True{}, Word.to_nat(p, bt))) : Nat}
          Nat.sub_bit(Word.to_nat(p, at), Word.to_nat(p, bt), False{}, True{}, True{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), False{}, True{}, True{}, h))
        case WCon{True{}, +at} WCon{False{}, +bt} False{}:
          %Equal.sym(Nat, Word.to_nat(p, Word.adc(p, at, bt, True{}, True{})), Nat.sub_carry(True{}, Word.to_nat(p, at), Word.to_nat(p, bt)), Word.to_nat_sub_carry(p, at, bt, True{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), True{}, False{}, False{}, h))) :
            {Nat.bit(False{}, _) == Nat.sub_carry(False{}, Nat.bit(True{}, Word.to_nat(p, at)), Nat.bit(False{}, Word.to_nat(p, bt))) : Nat}
          Nat.sub_bit(Word.to_nat(p, at), Word.to_nat(p, bt), True{}, False{}, False{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), True{}, False{}, False{}, h))
        case WCon{True{}, +at} WCon{False{}, +bt} True{}:
          %Equal.sym(Nat, Word.to_nat(p, Word.adc(p, at, bt, True{}, True{})), Nat.sub_carry(True{}, Word.to_nat(p, at), Word.to_nat(p, bt)), Word.to_nat_sub_carry(p, at, bt, True{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), True{}, False{}, True{}, h))) :
            {Nat.bit(True{}, _) == Nat.sub_carry(True{}, Nat.bit(True{}, Word.to_nat(p, at)), Nat.bit(False{}, Word.to_nat(p, bt))) : Nat}
          Nat.sub_bit(Word.to_nat(p, at), Word.to_nat(p, bt), True{}, False{}, True{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), True{}, False{}, True{}, h))
        case WCon{True{}, +at} WCon{True{}, +bt} False{}:
          %Equal.sym(Nat, Word.to_nat(p, Word.adc(p, at, bt, True{}, False{})), Nat.sub_carry(False{}, Word.to_nat(p, at), Word.to_nat(p, bt)), Word.to_nat_sub_carry(p, at, bt, False{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), True{}, True{}, False{}, h))) :
            {Nat.bit(True{}, _) == Nat.sub_carry(False{}, Nat.bit(True{}, Word.to_nat(p, at)), Nat.bit(True{}, Word.to_nat(p, bt))) : Nat}
          Nat.sub_bit(Word.to_nat(p, at), Word.to_nat(p, bt), True{}, True{}, False{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), True{}, True{}, False{}, h))
        case WCon{True{}, +at} WCon{True{}, +bt} True{}:
          %Equal.sym(Nat, Word.to_nat(p, Word.adc(p, at, bt, True{}, True{})), Nat.sub_carry(True{}, Word.to_nat(p, at), Word.to_nat(p, bt)), Word.to_nat_sub_carry(p, at, bt, True{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), True{}, True{}, True{}, h))) :
            {Nat.bit(False{}, _) == Nat.sub_carry(True{}, Nat.bit(True{}, Word.to_nat(p, at)), Nat.bit(True{}, Word.to_nat(p, bt))) : Nat}
          Nat.sub_bit(Word.to_nat(p, at), Word.to_nat(p, bt), True{}, True{}, True{}, Nat.sub_tail(Word.to_nat(p, at), Word.to_nat(p, bt), True{}, True{}, True{}, h))

# Unsigned subtraction converts to truncated Nat subtraction only when b <= a;
# this hypothesis rules out the machine subtraction's wraparound.
law U32.to_nat_sub:
  for +a: U32
  for +b: U32
  for h: {U32.is_le(b, a) == True{} : Bool}
  {U32.to_nat(U32.sub(a, b)) == Nat.sub(U32.to_nat(a), U32.to_nat(b)) : Nat}

def U32.to_nat_sub(a, b, h):
  match a b:
    case U32{+x} U32{+y}:
      Word.to_nat_sub_carry(32n, x, y, True{},
        Equal.trans(Bool, Nat.is_ge(Word.to_nat(32n, x), Word.to_nat(32n, y)), Nat.is_le(Word.to_nat(32n, y), Word.to_nat(32n, x)), True{},
          N.is_ge_eq_is_le(Word.to_nat(32n, x), Word.to_nat(32n, y)),
          Equal.trans(Bool, Nat.is_le(Word.to_nat(32n, y), Word.to_nat(32n, x)), U32.is_le(U32{y}, U32{x}), True{},
            Equal.cong(Cmp, Bool, Cmp.is_le, Nat.cmp(Word.to_nat(32n, y), Word.to_nat(32n, x)), Word.cmp(32n, y, x),
              Equal.sym(Cmp, Word.cmp(32n, y, x), Nat.cmp(Word.to_nat(32n, y), Word.to_nat(32n, x)), Word.cmp_to_nat(32n, y, x))), h)))

# The largest natural representable at width n (2^n - 1), without machine arithmetic.
def Word.max_nat(n: Nat) -> Nat:
  match n:
    case 0n:
      0n
    case 1n+p:
      1n+Nat.double(Word.max_nat(p))

# Every word lies within the unsigned range of its width.
law Word.to_nat_bound:
  for +n: Nat
  for a: Word(n)
  {Nat.is_le(Word.to_nat(n, a), Word.max_nat(n)) == True{} : Bool}

def Word.to_nat_bound(n, a):
  match n:
    case 0n:
      {==}
    case 1n+p:
      match a:
        case WCon{False{}, +t}:
          N.le_trans(Nat.double(Word.to_nat(p, t)), Nat.double(Word.max_nat(p)), 1n+Nat.double(Word.max_nat(p)),
            N.double_le_double(Word.to_nat(p, t), Word.max_nat(p), Word.to_nat_bound(p, t)),
            N.le_succ(Nat.double(Word.max_nat(p))))
        case WCon{True{}, +t}:
          N.double_le_double(Word.to_nat(p, t), Word.max_nat(p), Word.to_nat_bound(p, t))

# Doubling preserves and reflects the entire comparison, including equality.
law Nat.cmp_double:
  for x: Nat
  for y: Nat
  {Nat.cmp(x, y) == Nat.cmp(Nat.double(x), Nat.double(y)) : Cmp}

def Nat.cmp_double(x, y):
  match x y:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      Nat.cmp_double(p, q)

# Increment agrees with Nat successor strictly below the word's maximum.
law Word.to_nat_inc:
  for +n: Nat
  for +a: Word(n)
  for h: {Nat.is_lt(Word.to_nat(n, a), Word.max_nat(n)) == True{} : Bool}
  {Word.to_nat(n, Word.inc(n, a)) == 1n+Word.to_nat(n, a) : Nat}

def Word.to_nat_inc(n, a, h):
  match n:
    case 0n:
      Empty.absurd({0n == 1n : Nat}, N.internal_false_ne_true(h))
    case 1n+p:
      match a:
        case WCon{False{}, t}:
          {==}
        case WCon{True{}, +t}:
          Equal.cong(Nat, Nat, Nat.double, Word.to_nat(p, Word.inc(p, t)), 1n+Word.to_nat(p, t),
            Word.to_nat_inc(p, t,
              Equal.trans(Bool, Nat.is_lt(Word.to_nat(p, t), Word.max_nat(p)), Nat.is_lt(Nat.double(Word.to_nat(p, t)), Nat.double(Word.max_nat(p))), True{},
                Equal.cong(Cmp, Bool, Cmp.is_lt, Nat.cmp(Word.to_nat(p, t), Word.max_nat(p)), Nat.cmp(Nat.double(Word.to_nat(p, t)), Nat.double(Word.max_nat(p))), Nat.cmp_double(Word.to_nat(p, t), Word.max_nat(p))), h)))

# Expose a U32's word without forcing its 32-bit value in a dependent match.
def U32.word(a: U32) -> Word(32n):
  match a:
    case U32{w}:
      w

# Word and wrapper conversion agree.
def U32.word_nat(a: U32) -> {Word.to_nat(32n, U32.word(a)) == U32.to_nat(a) : Nat}:
  match a:
    case U32{w}:
      {==}

# Every U32 converts to a natural in [0, 2^32 - 1], using the symbolic width bound.
law U32.to_nat_bound:
  for +a: U32
  {Nat.is_le(U32.to_nat(a), Word.max_nat(32n)) == True{} : Bool}

def U32.to_nat_bound(a):
  %U32.word_nat(a) : {Nat.is_le(_, Word.max_nat(32n)) == True{} : Bool}
  Word.to_nat_bound(32n, U32.word(a))

# Exposing a word also commutes with increment.
def U32.inc_nat_word(a: U32) -> {Word.to_nat(32n, Word.inc(32n, U32.word(a))) == U32.to_nat(U32.inc(a)) : Nat}:
  match a:
    case U32{w}:
      {==}

# Increment converts to successor only when it cannot wrap.
law U32.to_nat_inc:
  for +a: U32
  for h: {Nat.is_lt(U32.to_nat(a), Word.max_nat(32n)) == True{} : Bool}
  {U32.to_nat(U32.inc(a)) == 1n+U32.to_nat(a) : Nat}

def U32.to_nat_inc(a, h):
  Equal.trans(Nat, U32.to_nat(U32.inc(a)), Word.to_nat(32n, Word.inc(32n, U32.word(a))), 1n+U32.to_nat(a),
    Equal.sym(Nat, Word.to_nat(32n, Word.inc(32n, U32.word(a))), U32.to_nat(U32.inc(a)), U32.inc_nat_word(a)),
    Equal.trans(Nat, Word.to_nat(32n, Word.inc(32n, U32.word(a))), 1n+Word.to_nat(32n, U32.word(a)), 1n+U32.to_nat(a),
      Word.to_nat_inc(32n, U32.word(a),
        Equal.trans(Bool, Nat.is_lt(Word.to_nat(32n, U32.word(a)), Word.max_nat(32n)), Nat.is_lt(U32.to_nat(a), Word.max_nat(32n)), True{},
          Equal.cong(Nat, Bool, x => Nat.is_lt(x, Word.max_nat(32n)), Word.to_nat(32n, U32.word(a)), U32.to_nat(a), U32.word_nat(a)), h)),
      Equal.cong(Nat, Nat, x => 1n+x, Word.to_nat(32n, U32.word(a)), U32.to_nat(a), U32.word_nat(a))))

# The natural value of a zero word is zero at every width.
law Word.to_nat_zero:
  for n: Nat
  {Word.to_nat(n, Word.zero(n)) == 0n : Nat}

def Word.to_nat_zero(n):
  match n:
    case 0n:
      {==}
    case 1n+p:
      Equal.cong(Nat, Nat, Nat.double, Word.to_nat(p, Word.zero(p)), 0n, Word.to_nat_zero(p))

# Word conversion by repeated increment, exactly the construction used by U32.from_nat.
def Word.from_nat(m: Nat, +n: Nat) -> Word(n):
  match m:
    case 0n:
      Word.zero(n)
    case 1n+p:
      Word.inc(n, Word.from_nat(p, n))

# Within the width's bound, conversion from Nat loses no information.
law Word.to_nat_from_nat:
  for +m: Nat
  for +n: Nat
  for +h: {Nat.is_le(m, Word.max_nat(n)) == True{} : Bool}
  {Word.to_nat(n, Word.from_nat(m, n)) == m : Nat}

def Word.to_nat_from_nat(m, n, h):
  match m:
    case 0n:
      Word.to_nat_zero(n)
    case 1n++p:
      +hp = N.le_of_lt(p, Word.max_nat(n), N.lt_of_succ_le(p, Word.max_nat(n), h))
      Equal.trans(Nat, Word.to_nat(n, Word.inc(n, Word.from_nat(p, n))), 1n+Word.to_nat(n, Word.from_nat(p, n)), 1n+p,
        Word.to_nat_inc(n, Word.from_nat(p, n),
          Equal.trans(Bool, Nat.is_lt(Word.to_nat(n, Word.from_nat(p, n)), Word.max_nat(n)), Nat.is_lt(p, Word.max_nat(n)), True{},
            Equal.cong(Nat, Bool, x => Nat.is_lt(x, Word.max_nat(n)), Word.to_nat(n, Word.from_nat(p, n)), p, Word.to_nat_from_nat(p, n, hp)),
            N.lt_of_succ_le(p, Word.max_nat(n), h))),
        Equal.cong(Nat, Nat, x => 1n+x, Word.to_nat(n, Word.from_nat(p, n)), p, Word.to_nat_from_nat(p, n, hp)))

# U32.from_nat is the width-32 instance of Word.from_nat, including wrapping.
law U32.from_nat_word:
  for +n: Nat
  {U32{Word.from_nat(n, 32n)} == U32.from_nat(n) : U32}

def U32.from_nat_word(n):
  match n:
    case 0n:
      {==}
    case 1n++p:
      %U32.from_nat_word(p) : {U32{Word.inc(32n, Word.from_nat(p, 32n))} == U32.inc(_) : U32}
      {==}

# Conversion is an inverse below 2^32, stated using the width's symbolic maximum
# so the proof never expands a 4-billion-constructor Peano literal.
law U32.to_nat_from_nat:
  for +n: Nat
  for h: {Nat.is_le(n, Word.max_nat(32n)) == True{} : Bool}
  {U32.to_nat(U32.from_nat(n)) == n : Nat}

def U32.to_nat_from_nat(n, h):
  %U32.from_nat_word(n) : {U32.to_nat(_) == n : Nat}
  Word.to_nat_from_nat(n, 32n, h)

# A carry into unsigned addition is the successor of the same addition without it,
# modulo the width (so no overflow hypothesis is needed).
law Word.adc_carry:
  for +n: Nat
  for a: Word(n)
  for b: Word(n)
  {Word.adc(n, a, b, False{}, True{}) == Word.inc(n, Word.adc(n, a, b, False{}, False{})) : Word(n)}

def Word.adc_carry(n, a, b):
  match n:
    case 0n:
      {==}
    case 1n+p:
      match a b:
        case WCon{False{}, at} WCon{False{}, bt}:
          {==}
        case WCon{False{}, +at} WCon{True{}, +bt}:
          Equal.cong(Word(p), Word(1n+p), w => WCon{False{}, w},
            Word.adc(p, at, bt, False{}, True{}), Word.inc(p, Word.adc(p, at, bt, False{}, False{})), Word.adc_carry(p, at, bt))
        case WCon{True{}, +at} WCon{False{}, +bt}:
          Equal.cong(Word(p), Word(1n+p), w => WCon{False{}, w},
            Word.adc(p, at, bt, False{}, True{}), Word.inc(p, Word.adc(p, at, bt, False{}, False{})), Word.adc_carry(p, at, bt))
        case WCon{True{}, at} WCon{True{}, bt}:
          {==}

# Unsigned addition commutes with increment on its right input, modulo the width.
law Word.add_inc:
  for +n: Nat
  for +a: Word(n)
  for +b: Word(n)
  {Word.add(n, a, Word.inc(n, b)) == Word.inc(n, Word.add(n, a, b)) : Word(n)}

def Word.add_inc(n, a, b):
  match n:
    case 0n:
      {==}
    case 1n+p:
      match a b:
        case WCon{False{}, at} WCon{False{}, bt}:
          {==}
        case WCon{True{}, +at} WCon{False{}, +bt}:
          Equal.cong(Word(p), Word(1n+p), w => WCon{False{}, w},
            Word.adc(p, at, bt, False{}, True{}), Word.inc(p, Word.adc(p, at, bt, False{}, False{})), Word.adc_carry(p, at, bt))
        case WCon{False{}, +at} WCon{True{}, +bt}:
          Equal.cong(Word(p), Word(1n+p), w => WCon{False{}, w},
            Word.add(p, at, Word.inc(p, bt)), Word.inc(p, Word.add(p, at, bt)), Word.add_inc(p, at, bt))
        case WCon{True{}, +at} WCon{True{}, +bt}:
          Equal.cong(Word(p), Word(1n+p), w => WCon{True{}, w},
            Word.add(p, at, Word.inc(p, bt)), Word.adc(p, at, bt, False{}, True{}),
            Equal.trans(Word(p), Word.add(p, at, Word.inc(p, bt)), Word.inc(p, Word.add(p, at, bt)), Word.adc(p, at, bt, False{}, True{}),
              Word.add_inc(p, at, bt), Equal.sym(Word(p), Word.adc(p, at, bt, False{}, True{}), Word.inc(p, Word.add(p, at, bt)), Word.adc_carry(p, at, bt))))

# Zero is an identity for addition at every width.
law Word.add_zero:
  for +n: Nat
  for +a: Word(n)
  {Word.add(n, a, Word.zero(n)) == a : Word(n)}

def Word.add_zero(n, a):
  match n:
    case 0n:
      match a:
        case WNil{}:
          {==}
    case 1n+p:
      match a:
        case WCon{False{}, +t}:
          Equal.cong(Word(p), Word(1n+p), w => WCon{False{}, w}, Word.add(p, t, Word.zero(p)), t, Word.add_zero(p, t))
        case WCon{True{}, +t}:
          Equal.cong(Word(p), Word(1n+p), w => WCon{True{}, w}, Word.add(p, t, Word.zero(p)), t, Word.add_zero(p, t))

# Adding one agrees with increment, including wraparound.
law U32.add_one:
  for +a: U32
  {U32.add(a, 1) == U32.inc(a) : U32}

def U32.add_one(a):
  match a:
    case U32{+w}:
      Equal.cong(Word(32n), U32, x => U32{x}, Word.add(32n, w, Word.inc(32n, Word.zero(32n))), Word.inc(32n, w),
        Equal.trans(Word(32n), Word.add(32n, w, Word.inc(32n, Word.zero(32n))), Word.inc(32n, Word.add(32n, w, Word.zero(32n))), Word.inc(32n, w),
          Word.add_inc(32n, w, Word.zero(32n)),
          Equal.cong(Word(32n), Word(32n), x => Word.inc(32n, x), Word.add(32n, w, Word.zero(32n)), w, Word.add_zero(32n, w))))

# Addition commutes with successor on its right input; arithmetic is modulo 2^32.
law U32.add_inc:
  for +a: U32
  for +b: U32
  {U32.add(a, U32.inc(b)) == U32.inc(U32.add(a, b)) : U32}

def U32.add_inc(a, b):
  match a b:
    case U32{+x} U32{+y}:
      Equal.cong(Word(32n), U32, w => U32{w}, Word.add(32n, x, Word.inc(32n, y)), Word.inc(32n, Word.add(32n, x, y)), Word.add_inc(32n, x, y))

# Adding zero changes no U32.
law U32.add_zero:
  for +a: U32
  {U32.add(a, 0) == a : U32}

def U32.add_zero(a):
  match a:
    case U32{+w}:
      Equal.cong(Word(32n), U32, x => U32{x}, Word.add(32n, w, Word.zero(32n)), w, Word.add_zero(32n, w))

# Adding a converted Nat agrees with that many increments, even across wraparound.
# This step form lets clients induct on the number of increments without assuming
# an unbounded U32 addition law.
law U32.add_from_nat_succ:
  for +a: U32
  for +n: Nat
  {U32.add(a, U32.from_nat(1n+n)) == U32.add(U32.add(a, 1), U32.from_nat(n)) : U32}

def U32.add_from_nat_succ(a, n):
  Equal.trans(U32, U32.add(a, U32.inc(U32.from_nat(n))), U32.inc(U32.add(a, U32.from_nat(n))), U32.add(U32.add(a, 1), U32.from_nat(n)),
    U32.add_inc(a, U32.from_nat(n)),
    Equal.trans(U32, U32.inc(U32.add(a, U32.from_nat(n))), U32.inc(U32.add(U32.from_nat(n), a)), U32.add(U32.add(a, 1), U32.from_nat(n)),
      Equal.cong(U32, U32, U32.inc, U32.add(a, U32.from_nat(n)), U32.add(U32.from_nat(n), a), U32.add_comm(a, U32.from_nat(n))),
      Equal.trans(U32, U32.inc(U32.add(U32.from_nat(n), a)), U32.add(U32.from_nat(n), U32.inc(a)), U32.add(U32.add(a, 1), U32.from_nat(n)),
        Equal.sym(U32, U32.add(U32.from_nat(n), U32.inc(a)), U32.inc(U32.add(U32.from_nat(n), a)), U32.add_inc(U32.from_nat(n), a)),
        Equal.trans(U32, U32.add(U32.from_nat(n), U32.inc(a)), U32.add(U32.inc(a), U32.from_nat(n)), U32.add(U32.add(a, 1), U32.from_nat(n)),
          U32.add_comm(U32.from_nat(n), U32.inc(a)),
          Equal.cong(U32, U32, x => U32.add(x, U32.from_nat(n)), U32.inc(a), U32.add(a, 1), Equal.sym(U32, U32.add(a, 1), U32.inc(a), U32.add_one(a)))))))

# Increment may wrap, but its natural value never exceeds the natural successor.
law Word.to_nat_inc_le:
  for +n: Nat
  for +a: Word(n)
  {Nat.is_le(Word.to_nat(n, Word.inc(n, a)), 1n+Word.to_nat(n, a)) == True{} : Bool}

def Word.to_nat_inc_le(n, a):
  match n:
    case 0n:
      {==}
    case 1n+p:
      match a:
        case WCon{False{}, +t}:
          N.le_refl(1n+Nat.double(Word.to_nat(p, t)))
        case WCon{True{}, +t}:
          N.double_le_double(Word.to_nat(p, Word.inc(p, t)), 1n+Word.to_nat(p, t), Word.to_nat_inc_le(p, t))

# Wrapping increment cannot increase the natural value by more than one.
law U32.to_nat_inc_le:
  for +a: U32
  {Nat.is_le(U32.to_nat(U32.inc(a)), 1n+U32.to_nat(a)) == True{} : Bool}

def U32.to_nat_inc_le(a):
  %U32.inc_nat_word(a) : {Nat.is_le(_, 1n+U32.to_nat(a)) == True{} : Bool}
  %U32.word_nat(a) : {Nat.is_le(Word.to_nat(32n, Word.inc(32n, U32.word(a))), 1n+_) == True{} : Bool}
  Word.to_nat_inc_le(32n, U32.word(a))

# Conversion to U32 and back never exceeds the input Nat; no bound is needed.
law U32.to_nat_from_nat_le:
  for +n: Nat
  {Nat.is_le(U32.to_nat(U32.from_nat(n)), n) == True{} : Bool}

def U32.to_nat_from_nat_le(n):
  match n:
    case 0n:
      {==}
    case 1n++p:
      N.le_trans(U32.to_nat(U32.inc(U32.from_nat(p))), 1n+U32.to_nat(U32.from_nat(p)), 1n+p,
        U32.to_nat_inc_le(U32.from_nat(p)), U32.to_nat_from_nat_le(p))

# A true U32 weak comparison also compares its exact natural values.
law U32.to_nat_le:
  for +a: U32
  for +b: U32
  for h: {U32.is_le(a, b) == True{} : Bool}
  {Nat.is_le(U32.to_nat(a), U32.to_nat(b)) == True{} : Bool}

def U32.to_nat_le(a, b, h):
  %U32.cmp_to_nat(a, b) : {Cmp.is_le(_) == True{} : Bool}
  h

# A true U32 equality test gives equality of its exact natural values.
law U32.to_nat_eq:
  for +a: U32
  for +b: U32
  for h: {U32.is_eq(a, b) == True{} : Bool}
  {U32.to_nat(a) == U32.to_nat(b) : Nat}

def U32.to_nat_eq(a, b, h):
  N.eq_of_is_eq(U32.to_nat(a), U32.to_nat(b),
    Equal.trans(Bool, Nat.is_eq(U32.to_nat(a), U32.to_nat(b)), U32.is_eq(a, b), True{},
      Equal.sym(Bool, U32.is_eq(a, b), Nat.is_eq(U32.to_nat(a), U32.to_nat(b)),
        Equal.cong(Cmp, Bool, Cmp.is_eq, U32.cmp(a, b), Nat.cmp(U32.to_nat(a), U32.to_nat(b)), U32.cmp_to_nat(a, b))), h))

# Adding above a low bit doubles the added high part.
law Nat.bit_add:
  for c: Bool
  for +x: Nat
  for +y: Nat
  {Nat.bit(c, Nat.add(x, y)) == Nat.add(Nat.bit(c, x), Nat.double(y)) : Nat}

def Nat.bit_add(c, x, y):
  match c:
    case False{}:
      N.double_add(x, y)
    case True{}:
      Equal.cong(Nat, Nat, z => 1n+z, Nat.double(Nat.add(x, y)), Nat.add(Nat.double(x), Nat.double(y)), N.double_add(x, y))

# One beyond the width's maximum is its modulus, as a Nat (no U32 overflow).
def Word.modulus(n: Nat) -> Nat:
  1n+Word.max_nat(n)

# Exact value of a shift, including its outgoing high bit.
def Word.Shift(+n: Nat, c: Bool, w: Word(n), r: Bool & Word(n)) -> Data:
  (hi, out) = r
  {Nat.bit(c, Word.to_nat(n, w)) == Nat.add(Word.to_nat(n, out), Nat.mul(Bool.to_nat(hi), Word.modulus(n))) : Nat}

def Word.shift_cons(+p: Nat, +c: Bool, +b: Bool, +w: Word(p), r: Bool & Word(p)) ->
  Word.Shift(p, b, w, r) -> Word.Shift(1n+p, c, WCon{b, w}, Word.shl.out.con(p, c, r)):
  (+hi, +out) = r
  h =>
    %Word.to_nat_con(p, b, w) : {Nat.bit(c, _) == Nat.add(Word.to_nat(1n+p, WCon{c, out}), Nat.mul(Bool.to_nat(hi), Word.modulus(1n+p))) : Nat}
    %Word.to_nat_con(p, c, out) : {Nat.bit(c, Nat.bit(b, Word.to_nat(p, w))) == Nat.add(_, Nat.mul(Bool.to_nat(hi), Word.modulus(1n+p))) : Nat}
    +v = Word.to_nat(p, out)
    +k = Nat.mul(Bool.to_nat(hi), Word.modulus(p))
    Equal.trans(Nat, Nat.bit(c, Nat.bit(b, Word.to_nat(p, w))), Nat.bit(c, Nat.add(v, k)), Nat.add(Nat.bit(c, v), Nat.mul(Bool.to_nat(hi), Word.modulus(1n+p))),
      Equal.cong(Nat, Nat, z => Nat.bit(c, z), Nat.bit(b, Word.to_nat(p, w)), Nat.add(v, k), h),
      Equal.trans(Nat, Nat.bit(c, Nat.add(v, k)), Nat.add(Nat.bit(c, v), Nat.double(k)), Nat.add(Nat.bit(c, v), Nat.mul(Bool.to_nat(hi), Word.modulus(1n+p))),
        Nat.bit_add(c, v, k),
        Equal.cong(Nat, Nat, z => Nat.add(Nat.bit(c, v), z), Nat.double(k), Nat.mul(Bool.to_nat(hi), Word.modulus(1n+p)),
          Equal.sym(Nat, Nat.mul(Bool.to_nat(hi), Nat.double(Word.modulus(p))), Nat.double(k), N.mul_double(Bool.to_nat(hi), Word.modulus(p))))))

# Word.shl.out is an exact binary shift with a separately reported overflow bit.
law Word.shift_spec:
  for +n: Nat
  for +c: Bool
  for +w: Word(n)
  Word.Shift(n, c, w, Word.shl.out(n, c, w))

def Word.shift_spec(n, c, w):
  match n:
    case 0n:
      match c:
        case False{}:
          {==}
        case True{}:
          {==}
    case 1n+p:
      match w:
        case WCon{+b, +t}:
          Word.shift_cons(p, c, b, t, Word.shl.out(p, b, t), Word.shift_spec(p, b, t))

# Any natural plus a positive modulus exceeds that modulus's predecessor.
def Nat.add_modulus_not_le(+x: Nat, +m: Nat) -> {Nat.is_le(Nat.add(x, 1n+m), m) == False{} : Bool}:
  Equal.trans(Bool, Nat.is_le(Nat.add(x, 1n+m), m), Nat.is_le(1n+Nat.add(x, m), m), False{},
    Equal.cong(Nat, Bool, z => Nat.is_le(z, m), Nat.add(x, 1n+m), 1n+Nat.add(x, m), N.add_succ(x, m)),
    N.not_le_of_lt(m, 1n+Nat.add(x, m), N.lt_of_le_of_lt(m, Nat.add(x, m), 1n+Nat.add(x, m), N.le_add_left(m, x), N.lt_succ_self(Nat.add(x, m)))))

# A shift whose exact natural value fits the width has no outgoing overflow bit.
def Word.shift_bounded(+n: Nat, +c: Bool, +w: Word(n), r: Bool & Word(n),
  h: Word.Shift(n, c, w, r), bound: N.le(Nat.bit(c, Word.to_nat(n, w)), Word.max_nat(n))) ->
  {Pair.fst(Bool, Word(n), r) == False{} : Bool} &
  {Word.to_nat(n, Pair.snd(Bool, Word(n), r)) == Nat.bit(c, Word.to_nat(n, w)) : Nat}:
  match r:
    case Tuple{False{}, +out}:
      ({==}, Equal.sym(Nat, Nat.bit(c, Word.to_nat(n, w)), Word.to_nat(n, out),
        Equal.trans(Nat, Nat.bit(c, Word.to_nat(n, w)), Nat.add(Word.to_nat(n, out), 0n), Word.to_nat(n, out), h, N.add_zero(Word.to_nat(n, out)))))
    case Tuple{True{}, +out}:
      +v = Word.to_nat(n, out)
      +m = Word.max_nat(n)
      +hh = Equal.trans(Nat, Nat.bit(c, Word.to_nat(n, w)), Nat.add(v, Nat.mul(1n, 1n+m)), Nat.add(v, 1n+m), h,
        Equal.cong(Nat, Nat, z => Nat.add(v, z), Nat.mul(1n, 1n+m), 1n+m, N.one_mul(1n+m)))
      Empty.absurd(({True{} == False{} : Bool} & {v == Nat.bit(c, Word.to_nat(n, w)) : Nat}),
        N.internal_false_ne_true(Equal.trans(Bool, False{}, Nat.is_le(Nat.add(v, 1n+m), m), True{},
          Equal.sym(Bool, Nat.is_le(Nat.add(v, 1n+m), m), False{}, Nat.add_modulus_not_le(v, m)),
          Equal.trans(Bool, Nat.is_le(Nat.add(v, 1n+m), m), Nat.is_le(Nat.bit(c, Word.to_nat(n, w)), m), True{},
            Equal.cong(Nat, Bool, z => Nat.is_le(z, m), Nat.add(v, 1n+m), Nat.bit(c, Word.to_nat(n, w)), Equal.sym(Nat, Nat.bit(c, Word.to_nat(n, w)), Nat.add(v, 1n+m), hh)), bound))))

# A fixed low bit preserves weak natural order.
def Nat.bit_mono(c: Bool, +x: Nat, +y: Nat, h: N.le(x, y)) -> N.le(Nat.bit(c, x), Nat.bit(c, y)):
  match c:
    case False{}:
      N.double_le_double(x, y, h)
    case True{}:
      N.double_le_double(x, y, h)

# Increasing the width cannot reduce its representable natural range.
law Word.max_nat_mono:
  for +n: Nat
  for +m: Nat
  for h: N.le(n, m)
  N.le(Word.max_nat(n), Word.max_nat(m))

def Word.max_nat_mono(n, m, h):
  match n m:
    case 0n _:
      N.zero_le(Word.max_nat(m))
    case 1n+p 0n:
      Empty.absurd(N.le(Word.max_nat(1n+p), 0n), N.internal_false_ne_true(h))
    case 1n++p 1n++q:
      N.double_le_double(Word.max_nat(p), Word.max_nat(q), Word.max_nat_mono(p, q, h))

# The converse conversion of a weak comparison, useful when proving machine guards.
law U32.of_nat_le:
  for +a: U32
  for +b: U32
  for h: N.le(U32.to_nat(a), U32.to_nat(b))
  {U32.is_le(a, b) == True{} : Bool}

def U32.of_nat_le(a, b, h):
  %Equal.sym(Cmp, U32.cmp(a, b), Nat.cmp(U32.to_nat(a), U32.to_nat(b)), U32.cmp_to_nat(a, b)) : {Cmp.is_le(_) == True{} : Bool}
  h

# The machine >= guard is the reverse natural <= guard.
law U32.ge_nat:
  for +a: U32
  for +b: U32
  {U32.is_ge(a, b) == Nat.is_le(U32.to_nat(b), U32.to_nat(a)) : Bool}

def U32.ge_nat(a, b):
  %Equal.sym(Cmp, U32.cmp(a, b), Nat.cmp(U32.to_nat(a), U32.to_nat(b)), U32.cmp_to_nat(a, b)) : {Cmp.is_ge(_) == Nat.is_le(U32.to_nat(b), U32.to_nat(a)) : Bool}
  N.is_ge_eq_is_le(U32.to_nat(a), U32.to_nat(b))

def Cmp.bit_lt(c: Cmp, a: Bool) -> {Cmp.is_lt(c) == True{} : Bool} -> {Cmp.is_lt(Word.cmp.fin(a, False{}, c)) == True{} : Bool}:
  match c:
    case LT{}:
      h => {==}
    case EQ{}:
      h => Empty.absurd({Cmp.is_lt(Word.cmp.fin(a, False{}, EQ{})) == True{} : Bool}, N.internal_false_ne_true(h))
    case GT{}:
      h => Empty.absurd({False{} == True{} : Bool}, N.internal_false_ne_true(h))

# If r < b then either low bit of r lies strictly below twice b.
def Nat.bit_lt_double(+c: Bool, +r: Nat, +b: Nat, h: N.lt(r, b)) -> N.lt(Nat.bit(c, r), Nat.double(b)):
  %Equal.sym(Cmp, Nat.cmp(Nat.bit(c, r), Nat.bit(False{}, b)), Word.cmp.fin(c, False{}, Nat.cmp(r, b)), Nat.cmp_bit(r, b, c, False{})) : {Cmp.is_lt(_) == True{} : Bool}
  Cmp.bit_lt(Nat.cmp(r, b), c, h)

# Subtracting b from a value in [b, 2*b) leaves a value below b.
def Nat.sub_lt_double(+s: Nat, +b: Nat, ge: N.le(b, s), lt: N.lt(s, Nat.double(b))) -> N.lt(Nat.sub(s, b), b):
  %Equal.sym(Bool, Nat.is_lt(Nat.sub(s, b), b), Nat.is_lt(Nat.add(b, Nat.sub(s, b)), Nat.add(b, b)), N.add_lt_add_iff_left(b, Nat.sub(s, b), b)) : {_ == True{} : Bool}
  %Equal.sym(Nat, Nat.add(b, Nat.sub(s, b)), s, N.add_sub_of_le(b, s, ge)) : {Nat.is_lt(_, Nat.add(b, b)) == True{} : Bool}
  %N.double_eq_add(b) : {Nat.is_lt(s, _) == True{} : Bool}
  lt

# Move a high addend across the binary low-bit constructor.
def Nat.bit_add_high(c: Bool, +x: Nat, +y: Nat) ->
  {Nat.add(Nat.double(x), Nat.bit(c, y)) == Nat.bit(c, Nat.add(x, y)) : Nat}:
  match c:
    case False{}:
      Equal.sym(Nat, Nat.double(Nat.add(x, y)), Nat.add(Nat.double(x), Nat.double(y)), N.double_add(x, y))
    case True{}:
      Equal.trans(Nat, Nat.add(Nat.double(x), 1n+Nat.double(y)), 1n+Nat.add(Nat.double(x), Nat.double(y)), 1n+Nat.double(Nat.add(x, y)),
        N.add_succ(Nat.double(x), Nat.double(y)),
        Equal.cong(Nat, Nat, z => 1n+z, Nat.add(Nat.double(x), Nat.double(y)), Nat.double(Nat.add(x, y)),
          Equal.sym(Nat, Nat.double(Nat.add(x, y)), Nat.add(Nat.double(x), Nat.double(y)), N.double_add(x, y))))

# Doubling the quotient and shifting the remainder shifts the represented dividend.
def Nat.div_shift_eq(+q: Nat, +b: Nat, +r: Nat, +c: Bool) ->
  {Nat.add(Nat.mul(Nat.double(q), b), Nat.bit(c, r)) == Nat.bit(c, Nat.add(Nat.mul(q, b), r)) : Nat}:
  %N.double_mul(q, b) : {Nat.add(_, Nat.bit(c, r)) == Nat.bit(c, Nat.add(Nat.mul(q, b), r)) : Nat}
  Nat.bit_add_high(c, Nat.mul(q, b), r)

# A division result represents the dividend exactly, with remainder below b.
def U32.DivSpec(+m: Nat, a: Word(m), +b: U32, qr: Word(m) & U32) -> Type:
  (q, +r) = qr
  {Nat.add(Nat.mul(Word.to_nat(m, q), U32.to_nat(b)), U32.to_nat(r)) == Word.to_nat(m, a) : Nat} & N.lt(U32.to_nat(r), U32.to_nat(b))

# The remainder in a division certificate never exceeds the represented dividend.
def U32.div_rem_le(+m: Nat, +a: Word(m), +b: U32, qr: Word(m) & U32, v: U32.DivSpec(m, a, b, qr)) ->
  N.le(U32.to_nat(Pair.snd(Word(m), U32, qr)), Word.to_nat(m, a)):
  (+q, +r) = qr
  (e, h) = v
  %e : {Nat.is_le(U32.to_nat(r), _) == True{} : Bool}
  N.le_add_left(U32.to_nat(r), Nat.mul(Word.to_nat(m, q), U32.to_nat(b)))

# Raising a quotient by one and subtracting its positive divisor leaves the
# represented dividend unchanged, provided the remainder subtraction cannot underflow.
def Nat.div_sub_eq(+q: Nat, +b: Nat, +s: Nat, h: N.le(b, s)) ->
  {Nat.add(Nat.mul(1n+q, b), Nat.sub(s, b)) == Nat.add(Nat.mul(q, b), s) : Nat}:
  +p = Nat.mul(q, b)
  +r = Nat.sub(s, b)
  Equal.trans(Nat, Nat.add(Nat.add(b, p), r), Nat.add(b, Nat.add(p, r)), Nat.add(p, s), N.add_assoc(b, p, r),
    Equal.trans(Nat, Nat.add(b, Nat.add(p, r)), Nat.add(p, Nat.add(b, r)), Nat.add(p, s), N.add_left_comm(b, p, r),
      Equal.cong(Nat, Nat, z => Nat.add(p, z), Nat.add(b, r), s, N.add_sub_of_le(b, s, h))))

# The two arms of one restoring-division step preserve its exact value and
# restore a remainder below the divisor.
def U32.div_fin_spec(+p: Nat, +c: Bool, +a: Word(p), +b: U32, +q: Word(p), +s: U32, g: Bool,
  guard: {U32.is_ge(s, b) == g : Bool},
  e: {Nat.add(Nat.mul(Nat.double(Word.to_nat(p, q)), U32.to_nat(b)), U32.to_nat(s)) == Nat.bit(c, Word.to_nat(p, a)) : Nat},
  bound: N.lt(U32.to_nat(s), Nat.double(U32.to_nat(b)))) ->
  U32.DivSpec(1n+p, WCon{c, a}, b, U32.divmod.go.fin(p, q, s, b, g)):
  match g:
    case False{}:
      (Equal.trans(Nat, Nat.add(Nat.mul(Nat.double(Word.to_nat(p, q)), U32.to_nat(b)), U32.to_nat(s)), Nat.bit(c, Word.to_nat(p, a)), Word.to_nat(1n+p, WCon{c, a}), e, Word.to_nat_con(p, c, a)),
        N.lt_of_not_le(U32.to_nat(b), U32.to_nat(s),
          Equal.trans(Bool, Nat.is_le(U32.to_nat(b), U32.to_nat(s)), U32.is_ge(s, b), False{},
            Equal.sym(Bool, U32.is_ge(s, b), Nat.is_le(U32.to_nat(b), U32.to_nat(s)), U32.ge_nat(s, b)), guard)))
    case True{}:
      +bv = U32.to_nat(b)
      +sv = U32.to_nat(s)
      +qv = Word.to_nat(p, q)
      +ge = Equal.trans(Bool, Nat.is_le(bv, sv), U32.is_ge(s, b), True{}, Equal.sym(Bool, U32.is_ge(s, b), Nat.is_le(bv, sv), U32.ge_nat(s, b)), guard)
      +sub = U32.to_nat_sub(s, b, U32.of_nat_le(b, s, ge))
      (Equal.trans(Nat, Nat.add(Nat.mul(1n+Nat.double(qv), bv), U32.to_nat(U32.sub(s, b))), Nat.add(Nat.mul(1n+Nat.double(qv), bv), Nat.sub(sv, bv)), Word.to_nat(1n+p, WCon{c, a}),
        Equal.cong(Nat, Nat, z => Nat.add(Nat.mul(1n+Nat.double(qv), bv), z), U32.to_nat(U32.sub(s, b)), Nat.sub(sv, bv), sub),
        Equal.trans(Nat, Nat.add(Nat.mul(1n+Nat.double(qv), bv), Nat.sub(sv, bv)), Nat.add(Nat.mul(Nat.double(qv), bv), sv), Word.to_nat(1n+p, WCon{c, a}),
          Nat.div_sub_eq(Nat.double(qv), bv, sv, ge),
          Equal.trans(Nat, Nat.add(Nat.mul(Nat.double(qv), bv), sv), Nat.bit(c, Word.to_nat(p, a)), Word.to_nat(1n+p, WCon{c, a}), e, Word.to_nat_con(p, c, a)))),
       Equal.trans(Bool, Nat.is_lt(U32.to_nat(U32.sub(s, b)), bv), Nat.is_lt(Nat.sub(sv, bv), bv), True{},
         Equal.cong(Nat, Bool, z => Nat.is_lt(z, bv), U32.to_nat(U32.sub(s, b)), Nat.sub(sv, bv), sub), Nat.sub_lt_double(sv, bv, ge, bound)))

# The shift result needed by one U32 division step: no high overflow and an exact value.
def U32.Shifted(c: Bool, r: U32, ts: Bool & Word(32n)) -> Type:
  (top, w) = ts
  {top == False{} : Bool} & {Word.to_nat(32n, w) == Nat.bit(c, U32.to_nat(r)) : Nat}

def U32.div_shift_step(+p: Nat, +c: Bool, +a: Word(p), +b: U32, +q: Word(p), +r: U32,
  ts: Bool & Word(32n), shifted: U32.Shifted(c, r, ts), v: U32.DivSpec(p, a, b, (q, r))) ->
  U32.DivSpec(1n+p, WCon{c, a}, b, U32.divmod.go.shl(p, q, b, ts)):
  match ts:
    case Tuple{False{}, +out}:
      (z, +se) = shifted
      (+e, hr) = v
      +qv = Word.to_nat(p, q)
      +bv = U32.to_nat(b)
      +rv = U32.to_nat(r)
      +sv = Word.to_nat(32n, out)
      U32.div_fin_spec(p, c, a, b, q, U32{out}, U32.is_ge(U32{out}, b), {==},
        Equal.trans(Nat, Nat.add(Nat.mul(Nat.double(qv), bv), sv), Nat.add(Nat.mul(Nat.double(qv), bv), Nat.bit(c, rv)), Nat.bit(c, Word.to_nat(p, a)),
          Equal.cong(Nat, Nat, z => Nat.add(Nat.mul(Nat.double(qv), bv), z), sv, Nat.bit(c, rv), se),
          Equal.trans(Nat, Nat.add(Nat.mul(Nat.double(qv), bv), Nat.bit(c, rv)), Nat.bit(c, Nat.add(Nat.mul(qv, bv), rv)), Nat.bit(c, Word.to_nat(p, a)),
            Nat.div_shift_eq(qv, bv, rv, c), Equal.cong(Nat, Nat, z => Nat.bit(c, z), Nat.add(Nat.mul(qv, bv), rv), Word.to_nat(p, a), e))),
        Equal.trans(Bool, Nat.is_lt(sv, Nat.double(bv)), Nat.is_lt(Nat.bit(c, rv), Nat.double(bv)), True{},
          Equal.cong(Nat, Bool, z => Nat.is_lt(z, Nat.double(bv)), sv, Nat.bit(c, rv), se), Nat.bit_lt_double(c, rv, bv, hr)))
    case Tuple{True{}, out}:
      (h, e) = shifted
      Empty.absurd(U32.DivSpec(1n+p, WCon{c, a}, b, U32.divmod.go.shl(p, q, b, (True{}, out))),
        N.internal_false_ne_true(Equal.sym(Bool, True{}, False{}, h)))

# At a prefix of at most 32 bits, the shifted remainder fits in a U32.
# This rules out the overflow arm using the dividend-prefix bound, even for
# divisors above 2^31; it does not assume that twice the divisor fits a U32.
def U32.div_shift_bound(+p: Nat, +c: Bool, +a: Word(p), +b: U32, +q: Word(p), +r: U32,
  width: N.le(1n+p, 32n), v: U32.DivSpec(p, a, b, (q, r))) -> N.le(Nat.bit(c, U32.to_nat(r)), Word.max_nat(32n)):
  N.le_trans(Nat.bit(c, U32.to_nat(r)), Nat.bit(c, Word.to_nat(p, a)), Word.max_nat(32n),
    Nat.bit_mono(c, U32.to_nat(r), Word.to_nat(p, a), U32.div_rem_le(p, a, b, (q, r), v)),
    N.le_trans(Nat.bit(c, Word.to_nat(p, a)), Word.max_nat(1n+p), Word.max_nat(32n),
      Equal.trans(Bool, Nat.is_le(Nat.bit(c, Word.to_nat(p, a)), Word.max_nat(1n+p)), Nat.is_le(Word.to_nat(1n+p, WCon{c, a}), Word.max_nat(1n+p)), True{},
        Equal.cong(Nat, Bool, z => Nat.is_le(z, Word.max_nat(1n+p)), Nat.bit(c, Word.to_nat(p, a)), Word.to_nat(1n+p, WCon{c, a}), Word.to_nat_con(p, c, a)),
        Word.to_nat_bound(1n+p, WCon{c, a})),
      Word.max_nat_mono(1n+p, 32n, width)))

# Expose the word of a remainder before performing the shift, without matching
# a U32 inside a proposition containing the closed width-32 natural modulus.
def U32.rec_word(+p: Nat, +c: Bool, +b: U32, +q: Word(p), r: U32) ->
  {U32.divmod.go.shl(p, q, b, Word.shl.out(32n, c, U32.word(r))) == U32.divmod.go.rec(p, c, b, (q, r)) : Word(1n+p) & U32}:
  match r:
    case U32{w}:
      {==}

def U32.shift_bound_word(+c: Bool, +r: U32, +limit: Nat, h: N.le(Nat.bit(c, U32.to_nat(r)), limit)) ->
  N.le(Nat.bit(c, Word.to_nat(32n, U32.word(r))), limit):
  Equal.trans(Bool, Nat.is_le(Nat.bit(c, Word.to_nat(32n, U32.word(r))), limit), Nat.is_le(Nat.bit(c, U32.to_nat(r)), limit), True{},
    Equal.cong(Nat, Bool, z => Nat.is_le(Nat.bit(c, z), limit), Word.to_nat(32n, U32.word(r)), U32.to_nat(r), U32.word_nat(r)), h)

def U32.shifted_from_word(+c: Bool, +r: U32, ts: Bool & Word(32n),
  v: {Pair.fst(Bool, Word(32n), ts) == False{} : Bool} &
    {Word.to_nat(32n, Pair.snd(Bool, Word(32n), ts)) == Nat.bit(c, Word.to_nat(32n, U32.word(r))) : Nat}) -> U32.Shifted(c, r, ts):
  match ts:
    case Tuple{top, +out}:
      (z, e) = v
      (z, Equal.trans(Nat, Word.to_nat(32n, out), Nat.bit(c, Word.to_nat(32n, U32.word(r))), Nat.bit(c, U32.to_nat(r)), e,
        Equal.cong(Nat, Nat, x => Nat.bit(c, x), Word.to_nat(32n, U32.word(r)), U32.to_nat(r), U32.word_nat(r))))

def U32.div_rec_spec(+p: Nat, +c: Bool, +a: Word(p), +b: U32, qr: Word(p) & U32,
  width: N.le(1n+p, 32n), v: U32.DivSpec(p, a, b, qr)) ->
  U32.DivSpec(1n+p, WCon{c, a}, b, U32.divmod.go.rec(p, c, b, qr)):
  match qr:
    case Tuple{+q, +r}:
      (+e, +hr) = v
      %U32.rec_word(p, c, b, q, r) : U32.DivSpec(1n+p, WCon{c, a}, b, _)
      U32.div_shift_step(p, c, a, b, q, r, Word.shl.out(32n, c, U32.word(r)),
        U32.shifted_from_word(c, r, Word.shl.out(32n, c, U32.word(r)),
          Word.shift_bounded(32n, c, U32.word(r), Word.shl.out(32n, c, U32.word(r)), Word.shift_spec(32n, c, U32.word(r)),
            U32.shift_bound_word(c, r, Word.max_nat(32n), U32.div_shift_bound(p, c, a, b, q, r, width, (e, hr))))), (e, hr))

# Restoring division is exact for every dividend prefix up to 32 bits and every
# positive U32 divisor. Its invariant explicitly includes the strict remainder bound.
law U32.divmod_spec:
  for +m: Nat
  for +a: Word(m)
  for +b: U32
  for +width: N.le(m, 32n)
  for positive: N.lt(0n, U32.to_nat(b))
  U32.DivSpec(m, a, b, U32.divmod.go(m, a, b))

def U32.divmod_spec(m, a, b, width, positive):
  match m a:
    case 0n WNil{}:
      ({==}, positive)
    case 1n+p WCon{+c, +hi}:
      U32.div_rec_spec(p, c, hi, b, U32.divmod.go(p, hi, b), width,
        U32.divmod_spec(p, hi, b, N.le_trans(p, 1n+p, 32n, N.le_succ(p), width), positive))

# A positive natural has a false zero-equality test.
def Nat.positive_ne_zero(n: Nat, h: N.lt(0n, n)) -> {Nat.is_eq(n, 0n) == False{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({True{} == False{} : Bool}, N.internal_false_ne_true(h))
    case 1n+p:
      {==}

# Machine zero testing agrees with natural zero testing.
law U32.zero_nat:
  for +a: U32
  {U32.is_zero(a) == Nat.is_eq(U32.to_nat(a), 0n) : Bool}

def U32.zero_nat(a):
  Equal.cong(Cmp, Bool, Cmp.is_eq, U32.cmp(a, 0), Nat.cmp(U32.to_nat(a), 0n), U32.cmp_to_nat(a, 0))

# Exact quotient/remainder with a strict remainder bound determines the Nat quotient.
def Nat.div_from_spec(+q: Nat, +r: Nat, +a: Nat, b: Nat,
  e: {Nat.add(Nat.mul(q, b), r) == a : Nat}, hr: N.lt(r, b), positive: N.lt(0n, b)) -> {q == Nat.div(a, b) : Nat}:
  match b:
    case 0n:
      Empty.absurd({q == Nat.div(a, 0n) : Nat}, N.internal_false_ne_true(positive))
    case 1n++bp:
      %e : {q == Nat.div(_, 1n+bp) : Nat}
      %N.add_comm(r, Nat.mul(q, 1n+bp)) : {q == Nat.div(_, 1n+bp) : Nat}
      %Equal.sym(Nat, Nat.div(Nat.add(r, Nat.mul(q, 1n+bp)), 1n+bp), Nat.add(Nat.div(r, 1n+bp), q), N.add_mul_div_right(r, q, bp)) : {q == _ : Nat}
      %Equal.sym(Nat, Nat.div(r, 1n+bp), 0n, N.div_eq_zero_of_le(r, bp, N.le_of_lt_succ(r, bp, hr))) : {q == Nat.add(_, q) : Nat}
      {==}

# The word-level division certificate determines the U32 quotient's natural value.
def U32.div_nat_result(+a: Word(32n), +b: U32, qr: Word(32n) & U32,
  positive: N.lt(0n, U32.to_nat(b)), v: U32.DivSpec(32n, a, b, qr)) ->
  {U32.to_nat(U32.div.fin(qr)) == Nat.div(Word.to_nat(32n, a), U32.to_nat(b)) : Nat}:
  (+q, +r) = qr
  (e, hr) = v
  Nat.div_from_spec(Word.to_nat(32n, q), U32.to_nat(r), Word.to_nat(32n, a), U32.to_nat(b), e, hr, positive)

# Expose the wrappers used by Base's public division definition.
def U32.div_word(a: U32, b: U32) ->
  {U32.div.if(U32.word(a), b, U32.is_zero(b)) == U32.div(a, b) : U32}:
  match a b:
    case U32{x} U32{y}:
      {==}

def U32.div_if_nonzero(+a: Word(32n), +b: U32, z: Bool, h: {z == False{} : Bool}) ->
  {U32.div.fin(U32.divmod.go(32n, a, b)) == U32.div.if(a, b, z) : U32}:
  match z:
    case False{}:
      {==}
    case True{}:
      Empty.absurd({U32.div.fin(U32.divmod.go(32n, a, b)) == 0 : U32},
        N.internal_false_ne_true(Equal.sym(Bool, True{}, False{}, h)))

# U32 division agrees with Nat division for a positive divisor. Base separately
# defines division by zero to be zero; positivity is explicit here.
law U32.to_nat_div:
  for +a: U32
  for +b: U32
  for +positive: N.lt(0n, U32.to_nat(b))
  {U32.to_nat(U32.div(a, b)) == Nat.div(U32.to_nat(a), U32.to_nat(b)) : Nat}

def U32.to_nat_div(a, b, positive):
  %U32.div_word(a, b) : {U32.to_nat(_) == Nat.div(U32.to_nat(a), U32.to_nat(b)) : Nat}
  %U32.div_if_nonzero(U32.word(a), b, U32.is_zero(b),
    Equal.trans(Bool, U32.is_zero(b), Nat.is_eq(U32.to_nat(b), 0n), False{}, U32.zero_nat(b), Nat.positive_ne_zero(U32.to_nat(b), positive))) :
    {U32.to_nat(_) == Nat.div(U32.to_nat(a), U32.to_nat(b)) : Nat}
  %U32.word_nat(a) : {U32.to_nat(U32.div.fin(U32.divmod.go(32n, U32.word(a), b))) == Nat.div(_, U32.to_nat(b)) : Nat}
  U32.div_nat_result(U32.word(a), b, U32.divmod.go(32n, U32.word(a), b), positive,
    U32.divmod_spec(32n, U32.word(a), b, N.le_refl(32n), positive))

# Increasing a positive Nat divisor cannot increase the quotient.
law Nat.div_antitone:
  for +a: Nat
  for +s: Nat
  for +t: Nat
  for st: N.le(s, t)
  for positive: N.lt(0n, s)
  N.le(Nat.div(a, t), Nat.div(a, s))

def Nat.div_antitone(a, s, t, st, positive):
  match s t:
    case 0n _:
      Empty.absurd(N.le(Nat.div(a, t), Nat.div(a, 0n)), N.internal_false_ne_true(positive))
    case 1n+sp 0n:
      Empty.absurd(N.le(Nat.div(a, 0n), Nat.div(a, 1n+sp)), N.internal_false_ne_true(st))
    case 1n++sp 1n++tp:
      +q = Nat.div(a, 1n+tp)
      %Equal.sym(Bool, Nat.is_le(q, Nat.div(a, 1n+sp)), Nat.is_le(Nat.mul(q, 1n+sp), a), N.le_div_iff_mul_le(q, a, sp)) : {_ == True{} : Bool}
      N.le_trans(Nat.mul(q, 1n+sp), Nat.mul(q, 1n+tp), a,
        N.mul_le_mul_left(1n+sp, 1n+tp, q, st), N.div_mul_le_self(a, 1n+tp))

# A strict machine comparison also compares its exact natural values.
law U32.to_nat_lt:
  for +a: U32
  for +b: U32
  for h: {U32.is_lt(a, b) == True{} : Bool}
  N.lt(U32.to_nat(a), U32.to_nat(b))

def U32.to_nat_lt(a, b, h):
  %U32.cmp_to_nat(a, b) : {Cmp.is_lt(_) == True{} : Bool}
  h

# Natural-value form of antitonicity, shared by the machine-order wrapper.
def U32.div_antitone_nat(+a: U32, +s: U32, +t: U32,
  +st: N.le(U32.to_nat(s), U32.to_nat(t)), +positive: N.lt(0n, U32.to_nat(s))) ->
  N.le(U32.to_nat(U32.div(a, t)), U32.to_nat(U32.div(a, s))):
  +pt = N.lt_of_lt_of_le(0n, U32.to_nat(s), U32.to_nat(t), positive, st)
  %Equal.sym(Nat, U32.to_nat(U32.div(a, t)), Nat.div(U32.to_nat(a), U32.to_nat(t)), U32.to_nat_div(a, t, pt)) : {Nat.is_le(_, U32.to_nat(U32.div(a, s))) == True{} : Bool}
  %Equal.sym(Nat, U32.to_nat(U32.div(a, s)), Nat.div(U32.to_nat(a), U32.to_nat(s)), U32.to_nat_div(a, s, positive)) : {Nat.is_le(Nat.div(U32.to_nat(a), U32.to_nat(t)), _) == True{} : Bool}
  Nat.div_antitone(U32.to_nat(a), U32.to_nat(s), U32.to_nat(t), st, positive)

# Increasing a positive U32 divisor cannot increase the quotient. Requiring
# 0 < s is essential because Base defines a / 0 = 0.
law U32.div_antitone:
  for +a: U32
  for +s: U32
  for +t: U32
  for st: {U32.is_le(s, t) == True{} : Bool}
  for positive: {U32.is_lt(0, s) == True{} : Bool}
  {U32.is_le(U32.div(a, t), U32.div(a, s)) == True{} : Bool}

def U32.div_antitone(a, s, t, st, positive):
  U32.of_nat_le(U32.div(a, t), U32.div(a, s),
    U32.div_antitone_nat(a, s, t, U32.to_nat_le(s, t, st), U32.to_nat_lt(0, s, positive)))

# A false zero test supplies the positive-natural hypothesis needed for division.
def Nat.positive_of_zero_false(n: Nat, h: {Nat.is_eq(n, 0n) == False{} : Bool}) -> N.lt(0n, n):
  match n:
    case 0n:
      Empty.absurd({False{} == True{} : Bool}, N.internal_false_ne_true(Equal.sym(Bool, True{}, False{}, h)))
    case 1n+p:
      {==}

# A nonzero U32 has a strictly positive exact natural value.
law U32.positive_of_nonzero:
  for +a: U32
  for h: {U32.is_zero(a) == False{} : Bool}
  N.lt(0n, U32.to_nat(a))

def U32.positive_of_nonzero(a, h):
  Nat.positive_of_zero_false(U32.to_nat(a), Equal.trans(Bool, Nat.is_eq(U32.to_nat(a), 0n), U32.is_zero(a), False{},
    Equal.sym(Bool, U32.is_zero(a), Nat.is_eq(U32.to_nat(a), 0n), U32.zero_nat(a)), h))

# Zero is also a left identity for U32 addition, including wrapping values.
law U32.zero_add:
  for +a: U32
  {U32.add(0, a) == a : U32}

def U32.zero_add(a):
  Equal.trans(U32, U32.add(0, a), U32.add(a, 0), a, U32.add_comm(0, a), U32.add_zero(a))
