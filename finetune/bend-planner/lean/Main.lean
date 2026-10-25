import BendModel

open BendModel

def modelRow (x y : Nat) : Array Nat := Id.run do
  let a := U32.literal x
  let b := U32.literal y
  let nat := Word.toNat (n := 32)
  let k := nat (U32.mod b (U32.literal 65))
  return #[nat a, nat b, nat (U32.add a b), nat (U32.sub a b), nat (U32.mul a b),
    nat (U32.div a b), nat (U32.mod a b), bit (U32.eq a b), bit (U32.ne a b),
    bit (U32.lt a b), bit (U32.le a b), bit (U32.gt a b), bit (U32.ge a b),
    nat (U32.inc a), nat (Word.not a), nat (Word.map₂ (· && ·) a b),
    nat (Word.map₂ (· || ·) a b), nat (Word.map₂ (· ^^ ·) a b),
    nat (Word.shl a), nat (Word.shr a), nat (U32.shlN a k), nat (U32.shrN a k),
    nat (U32.min a b), nat (U32.max a b),
    bit (U32.isZero a), bit (U32.isEven a),
    U32.log2 a,
    nat a, nat b, nat (U32.literal (nat a)), nat (U32.literal (nat a + nat b)),
    nat (U32.literal (nat a + 1)), nat (U32.literal (nat a * 65536))]

-- Independent stdlib UInt32/Nat oracle. Large shifts explicitly use Bend's contract.
def uintRow (x y : Nat) : Array Nat := Id.run do
  let a := UInt32.ofNat x
  let b := UInt32.ofNat y
  let nat := UInt32.toNat
  let k := nat (b % 65)
  return #[nat a, nat b, nat (a+b), nat (a-b), nat (a*b), nat (a/b), nat (a%b),
    bit (a == b), bit (a != b), bit (a < b), bit (a ≤ b), bit (a > b), bit (a ≥ b),
    nat (a+1), nat (~~~a), nat (a &&& b), nat (a ||| b), nat (a ^^^ b),
    nat (a <<< 1), nat (a >>> 1), (if k < 32 then nat (a <<< k.toUInt32) else 0),
    (if k < 32 then nat (a >>> k.toUInt32) else 0), nat (min a b), nat (max a b),
    bit (a == 0), bit ((a &&& 1) == 0), (nat a).log2,
    nat a, nat b, nat (UInt32.ofNat (nat a)), nat (UInt32.ofNat (nat a+nat b)),
    nat (UInt32.ofNat (nat a+1)), nat (UInt32.ofNat (nat a*65536))]

def main (args : List String) : IO UInt32 := do
  let [path] := args | throw (IO.userError "usage: fidelity pairs.tsv")
  let input ← IO.FS.Handle.mk path .read
  let out ← IO.getStdout
  let err ← IO.getStderr
  let mut count := 0
  repeat
    let line ← input.getLine
    if line.isEmpty then break
    let [sa,sb] := (line.trimAscii.toString.splitOn "\t") |
      throw (IO.userError s!"bad input line {count}")
    let some a := sa.toNat? | throw (IO.userError "bad a")
    let some b := sb.toNat? | throw (IO.userError "bad b")
    let row := modelRow a b
    let oracle := uintRow a b
    if row != oracle then
      throw (IO.userError s!"structural model vs UInt32 at ({a},{b}): {row} != {oracle}")
    out.putStrLn (String.intercalate "\t" (row.toList.map toString))
    count := count + 1
    if count % 10000 == 0 then err.putStrLn s!"Lean checked {count} rows"
  err.putStrLn s!"Lean checked {count} rows, structural model and UInt32 agree"
  return 0
