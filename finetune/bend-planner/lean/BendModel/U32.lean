import BendModel.Word
set_option linter.unusedSimpArgs false

namespace BendModel
abbrev U32 := Word 32
namespace U32

abbrev toNat (a : U32) : Nat := Word.toNat a
abbrev literal (n : Nat) : U32 := Word.ofNat 32 n
abbrev inc (a : U32) : U32 := Word.inc a
abbrev add (a b : U32) : U32 := Word.add a b
abbrev sub (a b : U32) : U32 := Word.sub a b
abbrev mul (a b : U32) : U32 := Word.mul a b
abbrev cmp (a b : U32) : Ordering := Word.cmp a b
def eq (a b : U32) : Bool := cmp a b == .eq
def ne (a b : U32) : Bool := !(eq a b)
def lt (a b : U32) : Bool := cmp a b == .lt
def le (a b : U32) : Bool := !(cmp a b == .gt)
def gt (a b : U32) : Bool := cmp a b == .gt
def ge (a b : U32) : Bool := !(cmp a b == .lt)
def isZero (a : U32) : Bool := eq a (literal 0)
def min (a b : U32) : U32 := if lt a b then a else b
def max (a b : U32) : U32 := if lt a b then b else a
def isEven (a : U32) : Bool := isZero (Word.map₂ (· && ·) a (literal 1))

def fromNat : Nat → U32
  | 0 => Word.zero 32
  | n+1 => inc (fromNat n)

def divStep (a₀ : Bool) (b : U32) (qr : Word m × U32) : Word (m+1) × U32 :=
  let (q,r) := qr
  let (t,s) := Word.shlOut a₀ r
  if t || ge s b then (Word.cons true q, sub s b)
  else (Word.cons false q, s)

def divmodGo (a : Word m) (b : U32) : Word m × U32 :=
  match a with
  | .nil => (.nil, literal 0)
  | .cons a₀ hi => divStep a₀ b (divmodGo hi b)

def div (a b : U32) : U32 :=
  if isZero b then literal 0 else (divmodGo a b).1

def mod (a b : U32) : U32 :=
  if isZero b then a else (divmodGo a b).2

def toUInt32 (a : U32) : UInt32 := UInt32.ofBitVec a.toBitVec

def shlN (a : U32) : Nat → U32
  | 0 => a
  | n+1 => Word.shl (shlN a n)

def shrN (a : U32) : Nat → U32
  | 0 => a
  | n+1 => Word.shr (shrN a n)

def log2Go : Nat → U32 → Nat → Nat
  | 0, _, b => b
  | k+1, n, b =>
    let s := toNat (shlN (literal 1) k)
    let t := s * bit (ne (shrN n s) (literal 0))
    log2Go k (shrN n t) (b+t)

def log2 (a : U32) : Nat := log2Go 5 a 0

theorem eq_spec (a b : U32) : eq a b = decide (toNat a = toNat b) := by
  apply Bool.eq_iff_iff.mpr
  simp [eq, cmp, Word.cmp_spec, toNat]
theorem lt_spec (a b : U32) : lt a b = decide (toNat a < toNat b) := by
  apply Bool.eq_iff_iff.mpr
  simp [lt, cmp, Word.cmp_spec, toNat, Nat.compare_eq_lt]
theorem le_spec (a b : U32) : le a b = decide (toNat a ≤ toNat b) := by
  apply Bool.eq_iff_iff.mpr
  simp [le, cmp, Word.cmp_spec, toNat, Nat.compare_eq_gt]
theorem gt_spec (a b : U32) : gt a b = decide (toNat b < toNat a) := by
  apply Bool.eq_iff_iff.mpr
  simp [gt, cmp, Word.cmp_spec, toNat, Nat.compare_eq_gt]
theorem ge_spec (a b : U32) : ge a b = decide (toNat b ≤ toNat a) := by
  apply Bool.eq_iff_iff.mpr
  simp [ge, cmp, Word.cmp_spec, toNat, Nat.compare_eq_lt]

theorem isZero_spec (a : U32) : isZero a = decide (toNat a = 0) := by
  simp [isZero, eq_spec, toNat, literal, Word.toNat_ofNat]

theorem fromNat_spec (n : Nat) : toNat (fromNat n) = n % 2^32 := by
  induction n with
  | zero => simp [fromNat]
  | succ n ih =>
    simp only [fromNat, inc, toNat, Word.toNat_inc, ih]
    omega

theorem fromNat_eq_literal (n : Nat) : fromNat n = literal n :=
  Word.toNat_injective ((fromNat_spec n).trans (Word.toNat_ofNat 32 n).symm)

theorem toNat_sub_of_le (a b : U32) (h : toNat b ≤ toNat a) :
    toNat (sub a b) = toNat a - toNat b := by
  change Word.toNat b ≤ Word.toNat a at h
  change Word.toNat (Word.sub a b) = Word.toNat a - Word.toNat b
  rw [Word.toNat_sub]
  have ha := Word.toNat_lt a
  have hb := Word.toNat_lt b
  omega

theorem toNat_sub_of_lt (a b : U32) (h : toNat a < toNat b) :
    toNat (sub a b) = 2^32 + toNat a - toNat b := by
  change Word.toNat a < Word.toNat b at h
  change Word.toNat (Word.sub a b) = 2^32 + Word.toNat a - Word.toNat b
  rw [Word.toNat_sub]
  have ha := Word.toNat_lt a
  have hb := Word.toNat_lt b
  omega

-- The high bit of 2*r+a₀ matters. In the overflow branch s<b and wrapping
-- subtraction recovers (2*r+a₀)-b, rather than saturating at zero.
theorem divStep_spec (a₀ : Bool) (b r : U32) (q : Word m)
    (hr : Word.toNat r < Word.toNat b) :
    Word.toNat (divStep a₀ b (q,r)).2 < Word.toNat b ∧
    Word.toNat b * Word.toNat (divStep a₀ b (q,r)).1 +
      Word.toNat (divStep a₀ b (q,r)).2 =
      bit a₀ + 2 * (Word.toNat b * Word.toNat q + Word.toNat r) := by
  have hs := Word.shlOut_spec r a₀
  have hb := Word.toNat_lt b
  have ha : bit a₀ ≤ 1 := by cases a₀ <;> decide
  generalize he : Word.shlOut a₀ r = ts at *
  rcases ts with ⟨t,s⟩
  have hsb := Word.toNat_lt s
  simp only [Prod.fst, Prod.snd] at hs
  unfold divStep
  dsimp only
  rw [he]
  cases t with
  | false =>
    simp only [bit_false, Nat.mul_zero, Nat.add_zero] at hs
    simp only [Bool.false_or]
    by_cases h : Word.toNat b ≤ Word.toNat s
    · have hg : ge s b = true := by rw [ge_spec]; exact decide_eq_true h
      rw [hg]
      simp only [↓reduceIte, Word.toNat, bit_true, Nat.mul_add, Nat.mul_one]
      have hsub := toNat_sub_of_le s b h
      change Word.toNat (Word.sub s b) = Word.toNat s - Word.toNat b at hsub
      rw [hsub]
      rw [Nat.mul_left_comm (Word.toNat b) 2]
      constructor <;> omega
    · have hg : ge s b = false := by rw [ge_spec]; exact decide_eq_false h
      rw [hg]
      simp only [Bool.false_eq_true, ↓reduceIte, Word.toNat, bit_false, Nat.zero_add]
      rw [Nat.mul_left_comm (Word.toNat b) 2]
      constructor <;> omega
  | true =>
    simp only [bit_true, Nat.mul_one] at hs
    simp only [Bool.true_or, ↓reduceIte, Word.toNat, bit_true, Nat.mul_add, Nat.mul_one]
    have hlt : Word.toNat s < Word.toNat b := by omega
    have hsub := toNat_sub_of_lt s b hlt
    change Word.toNat (Word.sub s b) = 2^32 + Word.toNat s - Word.toNat b at hsub
    rw [hsub]
    rw [Nat.mul_left_comm (Word.toNat b) 2]
    constructor <;> omega

theorem divmodGo_spec (a : Word m) (b : U32) (hb : 0 < Word.toNat b) :
    Word.toNat (divmodGo a b).2 < Word.toNat b ∧
    Word.toNat b * Word.toNat (divmodGo a b).1 + Word.toNat (divmodGo a b).2 = Word.toNat a := by
  induction a with
  | nil => simpa [divmodGo, literal, Word.toNat_ofNat, Word.toNat] using hb
  | cons a₀ hi ih =>
    have h := divStep_spec a₀ b (divmodGo hi b).2 (divmodGo hi b).1 ih.1
    simpa only [divmodGo, Word.toNat, Prod.eta, ih.2] using h

theorem divmodGo_quotient (a : Word m) (b : U32) (hb : 0 < Word.toNat b) :
    Word.toNat (divmodGo a b).1 = Word.toNat a / Word.toNat b := by
  have h := divmodGo_spec a b hb
  symm
  apply Nat.div_eq_of_lt_le
  · rw [← h.2, Nat.mul_comm (Word.toNat b)]; omega
  · rw [← h.2, Nat.add_mul, Nat.one_mul, Nat.mul_comm (Word.toNat b)]
    omega

theorem divmodGo_remainder (a : Word m) (b : U32) (hb : 0 < Word.toNat b) :
    Word.toNat (divmodGo a b).2 = Word.toNat a % Word.toNat b := by
  have h := divmodGo_spec a b hb
  rw [divmodGo_quotient a b hb] at h
  have hn := Nat.div_add_mod (Word.toNat a) (Word.toNat b)
  omega

theorem div_spec (a b : U32) : Word.toNat (div a b) = Word.toNat a / Word.toNat b := by
  unfold div
  rw [isZero_spec]
  by_cases hb : Word.toNat b = 0
  · simp [toNat, hb, literal, Word.toNat_ofNat]
  · simp only [toNat, hb, decide_false, Bool.false_eq_true, ↓reduceIte]
    exact divmodGo_quotient a b (by omega)

theorem mod_spec (a b : U32) : Word.toNat (mod a b) = Word.toNat a % Word.toNat b := by
  unfold mod
  rw [isZero_spec]
  by_cases hb : Word.toNat b = 0
  · simp [toNat, hb]
  · simp only [toNat, hb, decide_false, Bool.false_eq_true, ↓reduceIte]
    exact divmodGo_remainder a b (by omega)

theorem div_toBitVec (a b : U32) : (div a b).toBitVec = a.toBitVec / b.toBitVec := by
  apply BitVec.eq_of_toNat_eq
  exact (div_spec a b).trans (BitVec.toNat_udiv (x := a.toBitVec) (y := b.toBitVec)).symm

theorem mod_toBitVec (a b : U32) : (mod a b).toBitVec = a.toBitVec % b.toBitVec := by
  apply BitVec.eq_of_toNat_eq
  exact (mod_spec a b).trans (BitVec.toNat_umod (x := a.toBitVec) (y := b.toBitVec)).symm

end U32
end BendModel
