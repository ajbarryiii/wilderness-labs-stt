import BendModel.U32
set_option linter.unusedSimpArgs false

namespace BendModel.U32

@[simp] theorem toUInt32_toNat (a : U32) : (toUInt32 a).toNat = Word.toNat a := rfl

theorem add_toUInt32 (a b : U32) : toUInt32 (add a b) = toUInt32 a + toUInt32 b := by
  apply UInt32.toNat_inj.mp
  simp only [toUInt32_toNat, UInt32.toNat_add]
  exact Word.toNat_add a b

theorem sub_toUInt32 (a b : U32) : toUInt32 (sub a b) = toUInt32 a - toUInt32 b := by
  apply UInt32.toNat_inj.mp
  simp only [toUInt32_toNat, UInt32.toNat_sub]
  exact Word.toNat_sub a b

theorem mul_toUInt32 (a b : U32) : toUInt32 (mul a b) = toUInt32 a * toUInt32 b := by
  apply UInt32.toNat_inj.mp
  simp only [toUInt32_toNat, UInt32.toNat_mul]
  exact Word.toNat_mul a b

theorem div_toUInt32 (a b : U32) : toUInt32 (div a b) = toUInt32 a / toUInt32 b := by
  apply UInt32.toNat_inj.mp
  simp only [toUInt32_toNat, UInt32.toNat_div]
  exact div_spec a b

theorem mod_toUInt32 (a b : U32) : toUInt32 (mod a b) = toUInt32 a % toUInt32 b := by
  apply UInt32.toNat_inj.mp
  simp only [toUInt32_toNat, UInt32.toNat_mod]
  exact mod_spec a b

theorem fromNat_toUInt32 (n : Nat) : toUInt32 (fromNat n) = UInt32.ofNat n := by
  apply UInt32.toNat_inj.mp
  simp only [toUInt32_toNat, UInt32.toNat_ofNat']
  exact fromNat_spec n

theorem literal_toUInt32 (n : Nat) : toUInt32 (literal n) = UInt32.ofNat n := by
  rw [← fromNat_eq_literal]
  exact fromNat_toUInt32 n

def parseLiteral (n : Nat) : Option U32 := if n < 2^32 then some (literal n) else none

theorem parseLiteral_ok (n : Nat) (h : n < 2^32) :
    parseLiteral n = some (literal n) ∧ Word.toNat (literal n) = n := by
  simp [parseLiteral, h, literal, Word.toNat_ofNat, Nat.mod_eq_of_lt h]

theorem parseLiteral_reject (n : Nat) (h : 2^32 ≤ n) : parseLiteral n = none := by
  simp [parseLiteral, show ¬n < 2^32 by omega]

-- The installed parser admits Nat literals only through 2^32-1. Above 256
-- it expands them to U32.to_nat of the same digits.
def parseNatLiteral (n : Nat) : Option Nat :=
  if n < 2^32 then some (if n ≤ 256 then n else Word.toNat (literal n)) else none

theorem parseNatLiteral_ok (n : Nat) (h : n < 2^32) : parseNatLiteral n = some n := by
  simp [parseNatLiteral, h, literal, Word.toNat_ofNat, Nat.mod_eq_of_lt h]

theorem parseNatLiteral_reject (n : Nat) (h : 2^32 ≤ n) : parseNatLiteral n = none := by
  simp [parseNatLiteral, show ¬n < 2^32 by omega]

theorem eq_toUInt32 (a b : U32) : eq a b = (toUInt32 a == toUInt32 b) := by
  rw [eq_spec]
  apply Bool.eq_iff_iff.mpr
  simp [← UInt32.toNat_inj, toUInt32_toNat, toNat]

theorem ne_toUInt32 (a b : U32) : ne a b = (toUInt32 a != toUInt32 b) := by
  rw [ne, eq_toUInt32]
  rfl

theorem lt_toUInt32 (a b : U32) : lt a b = decide (toUInt32 a < toUInt32 b) := by
  rw [lt_spec]
  rfl
theorem le_toUInt32 (a b : U32) : le a b = decide (toUInt32 a ≤ toUInt32 b) := by
  rw [le_spec]
  rfl
theorem gt_toUInt32 (a b : U32) : gt a b = decide (toUInt32 b < toUInt32 a) := by
  rw [gt_spec]
  rfl
theorem ge_toUInt32 (a b : U32) : ge a b = decide (toUInt32 b ≤ toUInt32 a) := by
  rw [ge_spec]
  rfl

theorem inc_toUInt32 (a : U32) : toUInt32 (inc a) = toUInt32 a + 1 := by
  apply UInt32.toNat_inj.mp
  simp only [toUInt32_toNat, UInt32.toNat_add, UInt32.toNat_one]
  exact Word.toNat_inc a

theorem not_toUInt32 (a : U32) : toUInt32 (Word.not a) = ~~~toUInt32 a := by
  apply UInt32.toNat_inj.mp
  simp only [toUInt32_toNat, UInt32.toNat_not]
  exact Word.toNat_not a

theorem and_toUInt32 (a b : U32) :
    toUInt32 (Word.map₂ (· && ·) a b) = toUInt32 a &&& toUInt32 b := by
  apply UInt32.toBitVec_inj.mp
  change (Word.map₂ (· && ·) a b).toBitVec = a.toBitVec &&& b.toBitVec
  exact Word.toBitVec_and a b

theorem or_toUInt32 (a b : U32) :
    toUInt32 (Word.map₂ (· || ·) a b) = toUInt32 a ||| toUInt32 b := by
  apply UInt32.toBitVec_inj.mp
  change (Word.map₂ (· || ·) a b).toBitVec = a.toBitVec ||| b.toBitVec
  exact Word.toBitVec_or a b

theorem xor_toUInt32 (a b : U32) :
    toUInt32 (Word.map₂ (· ^^ ·) a b) = toUInt32 a ^^^ toUInt32 b := by
  apply UInt32.toBitVec_inj.mp
  change (Word.map₂ (· ^^ ·) a b).toBitVec = a.toBitVec ^^^ b.toBitVec
  exact Word.toBitVec_xor a b

theorem min_spec (a b : U32) : Word.toNat (min a b) = Nat.min (Word.toNat a) (Word.toNat b) := by
  unfold min
  rw [lt_spec]
  split <;> simp_all [toNat, Nat.min_def] <;> omega

theorem max_spec (a b : U32) : Word.toNat (max a b) = Nat.max (Word.toNat a) (Word.toNat b) := by
  unfold max
  rw [lt_spec]
  split <;> simp_all [toNat, Nat.max_def] <;> omega

theorem shlN_toBitVec (a : U32) (n : Nat) : (shlN a n).toBitVec = a.toBitVec <<< n := by
  induction n with
  | zero => simp [shlN]
  | succ n ih => simp only [shlN, Word.toBitVec_shl, ih, BitVec.shiftLeft_add]

theorem shrN_toBitVec (a : U32) (n : Nat) : (shrN a n).toBitVec = a.toBitVec >>> n := by
  induction n with
  | zero => simp [shrN]
  | succ n ih => simp only [shrN, Word.toBitVec_shr, ih, BitVec.shiftRight_add]

theorem shlN_large (a : U32) (n : Nat) (h : 32 ≤ n) : toUInt32 (shlN a n) = 0 := by
  apply UInt32.toBitVec_inj.mp
  change (shlN a n).toBitVec = 0
  rw [shlN_toBitVec]
  exact BitVec.shiftLeft_eq_zero (by omega)

theorem shrN_large (a : U32) (n : Nat) (h : 32 ≤ n) : toUInt32 (shrN a n) = 0 := by
  apply UInt32.toBitVec_inj.mp
  change (shrN a n).toBitVec = 0
  rw [shrN_toBitVec]
  exact BitVec.ushiftRight_eq_zero (by omega)

theorem shlN_small (a : U32) (n : Nat) (h : n < 32) :
    toUInt32 (shlN a n) = toUInt32 a <<< UInt32.ofNat n := by
  apply UInt32.toNat_inj.mp
  change (shlN a n).toBitVec.toNat = _
  rw [shlN_toBitVec, BitVec.toNat_shiftLeft, UInt32.toNat_shiftLeft,
    UInt32.toNat_ofNat', toUInt32_toNat]
  have hn : n % 2^32 % 32 = n := by omega
  rw [hn]
  rfl

theorem shrN_small (a : U32) (n : Nat) (h : n < 32) :
    toUInt32 (shrN a n) = toUInt32 a >>> UInt32.ofNat n := by
  apply UInt32.toNat_inj.mp
  change (shrN a n).toBitVec.toNat = _
  rw [shrN_toBitVec, BitVec.toNat_ushiftRight, UInt32.toNat_shiftRight,
    UInt32.toNat_ofNat', toUInt32_toNat]
  have hn : n % 2^32 % 32 = n := by omega
  rw [hn]
  rfl

-- Unlike Bend's Nat-count shifts, UInt32 masks the count by 31. The
-- BitVec/Nat shifts above are the faithful comparison, including counts ≥32.
theorem uint32_shiftLeft_exact (a : U32) (n : Nat) :
    (toUInt32 a <<< UInt32.ofNat n).toNat = Word.toNat a <<< (n % 32) % 2^32 := by
  rw [UInt32.toNat_shiftLeft, UInt32.toNat_ofNat', toUInt32_toNat]
  congr 2
  omega

theorem uint32_shiftRight_exact (a : U32) (n : Nat) :
    (toUInt32 a >>> UInt32.ofNat n).toNat = Word.toNat a >>> (n % 32) := by
  rw [UInt32.toNat_shiftRight, UInt32.toNat_ofNat', toUInt32_toNat]
  congr 1
  omega

example : toUInt32 (shlN (literal 1) 32) = 0 ∧ ((1 : UInt32) <<< 32) = 1 := by
  constructor
  · exact shlN_large _ _ (by decide)
  · decide

end BendModel.U32
