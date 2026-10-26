import BendModel.Log2
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! The 41 statements below are independent translations of u32.bend, in source
order. Boolean order hypotheses are retained as Boolean equalities. Extra helper
theorems explicitly identify stronger results rather than assuming them. -/
namespace BendModel.Phase2

def natDouble : Nat → Nat
  | 0 => 0
  | n+1 => 2 + natDouble n

@[simp] theorem natDouble_spec (n : Nat) : natDouble n = 2*n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [natDouble, ih]; omega

def natBit (b : Bool) (n : Nat) : Nat := bit b + natDouble n
def subLow (a b c : Bool) : Bool := (Word.fullAdd a (!b) c).1
def subCarry (a b c : Bool) : Bool := (Word.fullAdd a (!b) c).2
def subOk (c : Bool) (a b : Nat) : Prop :=
  (if c then decide (b ≤ a) else decide (b < a)) = true
def natSubCarry (c : Bool) (a b : Nat) : Nat := a - (bit (!c) + b)

def maxNat : Nat → Nat
  | 0 => 0
  | n+1 => 1 + natDouble (maxNat n)

def wordFromNat (m n : Nat) : Word n :=
  match m with
  | 0 => Word.zero n
  | m+1 => Word.inc (wordFromNat m n)

def Shift (c : Bool) (w : Word n) (r : Bool × Word n) : Prop :=
  natBit c w.toNat = r.2.toNat + bit r.1 * (1 + maxNat n)

def DivSpec (a : Word m) (b : U32) (qr : Word m × U32) : Prop :=
  qr.1.toNat * Word.toNat b + Word.toNat qr.2 = a.toNat ∧
    decide (Word.toNat qr.2 < Word.toNat b) = true

theorem maxNat_succ_eq_pow (n : Nat) : maxNat n + 1 = 2^n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [maxNat, natDouble_spec, Nat.pow_succ]; omega

theorem maxNat_eq_pow_sub_one (n : Nat) : maxNat n = 2^n - 1 := by
  have := maxNat_succ_eq_pow n
  omega

theorem maxNat_32_literal : maxNat 32 = 4294967295 := by
  rw [maxNat_eq_pow_sub_one]
theorem maxNat_32_pow : maxNat 32 = 2^32 - 1 := maxNat_eq_pow_sub_one 32

theorem wordFromNat_mod (m n : Nat) : (wordFromNat m n).toNat = m % 2^n := by
  induction m with
  | zero => simp [wordFromNat]
  | succ m ih => simp [wordFromNat, Word.toNat_inc, ih, Nat.mod_add_mod]

theorem adc_mod (a b : Word n) (c : Bool) :
    (Word.adc a b false c).toNat = (a.toNat+b.toNat+bit c) % 2^n := by
  have h := congrArg (fun x => x % 2^n) (Word.adcFull_spec a b c)
  simp only [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (Word.toNat_lt _)] at h
  exact h.symm

-- 01 Nat.cmp_bit
theorem nat_cmp_bit (x y : Nat) (a b : Bool) :
    compare (natBit a x) (natBit b y) = Word.cmpFin a b (compare x y) := by
  cases h : compare x y with
  | lt =>
    have h := Nat.compare_eq_lt.mp h
    cases a <;> cases b <;> simp only [Word.cmpFin] <;>
      apply Nat.compare_eq_lt.mpr <;> simp [natBit, bit] <;> omega
  | eq =>
    have h := Nat.compare_eq_eq.mp h
    cases a <;> cases b <;> simp [natBit, bit, Word.cmpFin, h, Nat.compare_eq_ite_lt] <;> omega
  | gt =>
    have h := Nat.compare_eq_gt.mp h
    cases a <;> cases b <;> simp only [Word.cmpFin] <;>
      apply Nat.compare_eq_gt.mpr <;> simp [natBit, bit] <;> omega

-- 02 Word.cmp_to_nat
theorem word_cmp_to_nat (a b : Word n) : Word.cmp a b = compare a.toNat b.toNat := Word.cmp_spec a b
-- 03 U32.cmp_to_nat
theorem u32_cmp_to_nat (a b : U32) : U32.cmp a b = compare (Word.toNat a) (Word.toNat b) := Word.cmp_spec a b
-- 04 U32.le_trans
theorem u32_le_trans (a b c : U32) (ab : U32.le a b = true) (bc : U32.le b c = true) :
    U32.le a c = true := by
  simp only [U32.le_spec, decide_eq_true_eq] at *
  exact Nat.le_trans ab bc

-- 05 Nat.sub_bit
theorem nat_sub_bit (x y : Nat) (a b c : Bool)
    (h : subOk (subCarry a b c) x y) :
    natBit (subLow a b c) (natSubCarry (subCarry a b c) x y) =
      natSubCarry c (natBit a x) (natBit b y) := by
  cases a <;> cases b <;> cases c <;>
    simp [subOk, subLow, subCarry, Word.fullAdd, natBit, natSubCarry, bit] at h ⊢ <;> omega

-- 06 Word.to_nat_sub_carry
theorem word_to_nat_sub_carry (a b : Word n) (c : Bool)
    (h : subOk c a.toNat b.toNat) :
    (Word.adc a b true c).toNat = natSubCarry c a.toNat b.toNat := by
  have hs := Word.adcFull_spec a (Word.not b) c
  rw [Word.toNat_not] at hs
  have ha := Word.toNat_lt a
  have hb := Word.toNat_lt b
  have hp := Nat.two_pow_pos n
  have hr := Word.toNat_lt (Word.adcFull a (Word.not b) c).2
  cases ho : (Word.adcFull a (Word.not b) c).1 <;> cases c <;>
    simp only [ho, bit_true, bit_false, Nat.mul_zero, Nat.mul_one] at hs <;>
    simp [subOk, natSubCarry, bit] at h ⊢ <;>
    change (Word.adcFull a (Word.not b) _).2.toNat = _ <;> omega

-- 07 U32.to_nat_sub
theorem u32_to_nat_sub (a b : U32) (h : U32.le b a = true) :
    Word.toNat (U32.sub a b) = Word.toNat a - Word.toNat b := by
  apply U32.toNat_sub_of_le
  simpa only [U32.le_spec, decide_eq_true_eq] using h

-- 08 Word.to_nat_bound
theorem word_to_nat_bound (a : Word n) : decide (a.toNat ≤ maxNat n) = true := by
  have := Word.toNat_lt a
  rw [maxNat_eq_pow_sub_one]
  simp only [decide_eq_true_eq]
  omega

-- 09 Nat.cmp_double
theorem nat_cmp_double (x y : Nat) : compare x y = compare (natDouble x) (natDouble y) := by
  simp [Nat.compare_eq_ite_lt]

-- 10 Word.to_nat_inc
theorem word_to_nat_inc (a : Word n) (h : decide (a.toNat < maxNat n) = true) :
    (Word.inc a).toNat = 1+a.toNat := by
  have hp := maxNat_succ_eq_pow n
  simp only [decide_eq_true_eq] at h
  rw [Word.toNat_inc, Nat.mod_eq_of_lt (by omega)]
  omega
-- 11 U32.to_nat_bound
theorem u32_to_nat_bound (a : U32) : decide (Word.toNat a ≤ maxNat 32) = true := word_to_nat_bound a
-- 12 U32.to_nat_inc
theorem u32_to_nat_inc (a : U32) (h : decide (Word.toNat a < maxNat 32) = true) :
    Word.toNat (U32.inc a) = 1+Word.toNat a := word_to_nat_inc a h
-- 13 Word.to_nat_zero
theorem word_to_nat_zero (n : Nat) : (Word.zero n).toNat = 0 := Word.toNat_zero n
-- 14 Word.to_nat_from_nat
theorem word_to_nat_from_nat (m n : Nat) (h : decide (m ≤ maxNat n) = true) :
    (wordFromNat m n).toNat = m := by
  rw [wordFromNat_mod]
  have := maxNat_succ_eq_pow n
  simp only [decide_eq_true_eq] at h
  exact Nat.mod_eq_of_lt (by omega)
-- 15 U32.from_nat_word
theorem u32_from_nat_word (n : Nat) : wordFromNat n 32 = U32.fromNat n := by
  apply Word.toNat_injective
  exact (wordFromNat_mod n 32).trans (U32.fromNat_spec n).symm
-- 16 U32.to_nat_from_nat
theorem u32_to_nat_from_nat (n : Nat) (h : decide (n ≤ maxNat 32) = true) :
    Word.toNat (U32.fromNat n) = n := by
  rw [← u32_from_nat_word]
  exact word_to_nat_from_nat n 32 h
-- 17 Word.adc_carry
theorem word_adc_carry (a b : Word n) :
    Word.adc a b false true = Word.inc (Word.adc a b false false) := by
  apply Word.toNat_injective
  simp only [adc_mod, Word.toNat_inc, bit_true, bit_false, Nat.add_zero, Nat.mod_add_mod]
-- 18 Word.add_inc
theorem word_add_inc (a b : Word n) : Word.add a (Word.inc b) = Word.inc (Word.add a b) := by
  apply Word.toNat_injective
  simp only [Word.toNat_add, Word.toNat_inc, Nat.add_mod_mod, Nat.mod_add_mod, Nat.add_assoc]
-- 19 Word.add_zero
theorem word_add_zero (a : Word n) : Word.add a (Word.zero n) = a := by
  apply Word.toNat_injective
  simp [Word.toNat_add, Nat.mod_eq_of_lt (Word.toNat_lt a)]
-- 20 U32.add_one
theorem u32_add_one (a : U32) : U32.add a (U32.literal 1) = U32.inc a := by
  apply Word.toNat_injective
  simp [U32.add, U32.literal, Word.toNat_add, Word.toNat_inc, Word.toNat_ofNat]
-- 21 U32.add_inc
theorem u32_add_inc (a b : U32) : U32.add a (U32.inc b) = U32.inc (U32.add a b) := word_add_inc a b
-- 22 U32.add_zero
theorem u32_add_zero (a : U32) : U32.add a (U32.literal 0) = a := by
  apply Word.toNat_injective
  simp [U32.add, U32.literal, Word.toNat_add, Word.toNat_ofNat, Nat.mod_eq_of_lt (Word.toNat_lt a)]
-- 23 U32.add_from_nat_succ
theorem u32_add_from_nat_succ (a : U32) (n : Nat) :
    U32.add a (U32.fromNat (1+n)) = U32.add (U32.add a (U32.literal 1)) (U32.fromNat n) := by
  apply Word.toNat_injective
  simp only [U32.add, Word.toNat_add, U32.fromNat_spec, U32.toNat,
    U32.literal, Word.toNat_ofNat]
  rw [Nat.add_comm 1 n]
  simp only [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]
  congr 1
  omega
-- 24 Word.to_nat_inc_le
theorem word_to_nat_inc_le (a : Word n) : decide ((Word.inc a).toNat ≤ 1+a.toNat) = true := by
  simp only [decide_eq_true_eq, Word.toNat_inc]
  exact Nat.le_trans (Nat.mod_le _ _) (by omega)
-- 25 U32.to_nat_inc_le
theorem u32_to_nat_inc_le (a : U32) : decide (Word.toNat (U32.inc a) ≤ 1+Word.toNat a) = true := word_to_nat_inc_le a
-- 26 U32.to_nat_from_nat_le
theorem u32_to_nat_from_nat_le (n : Nat) : decide (Word.toNat (U32.fromNat n) ≤ n) = true := by
  change decide (U32.toNat (U32.fromNat n) ≤ n) = true
  simp only [U32.fromNat_spec]
  exact decide_eq_true (Nat.mod_le _ _)
-- 27 U32.to_nat_le
theorem u32_to_nat_le (a b : U32) (h : U32.le a b = true) :
    decide (Word.toNat a ≤ Word.toNat b) = true := by simpa [U32.le_spec, U32.toNat] using h
-- 28 U32.to_nat_eq
theorem u32_to_nat_eq (a b : U32) (h : U32.eq a b = true) : Word.toNat a = Word.toNat b := by
  simpa [U32.eq_spec, U32.toNat] using h
-- 29 Nat.bit_add
theorem nat_bit_add (c : Bool) (x y : Nat) : natBit c (x+y) = natBit c x + natDouble y := by
  simp [natBit, Nat.mul_add, Nat.add_assoc]
-- 30 Word.shift_spec
theorem word_shift_spec (c : Bool) (w : Word n) : Shift c w (Word.shlOut c w) := by
  unfold Shift natBit
  simp only [natDouble_spec]
  rw [show 1+maxNat n = 2^n by have := maxNat_succ_eq_pow n; omega, Nat.mul_comm]
  simpa [Nat.mul_comm] using Word.shlOut_spec w c
-- 31 Word.max_nat_mono
theorem word_max_nat_mono (n m : Nat) (h : decide (n ≤ m) = true) :
    decide (maxNat n ≤ maxNat m) = true := by
  simp only [decide_eq_true_eq] at *
  have hp := Nat.pow_le_pow_right (by decide : 0 < 2) h
  rw [maxNat_eq_pow_sub_one, maxNat_eq_pow_sub_one]
  omega
-- 32 U32.of_nat_le
theorem u32_of_nat_le (a b : U32) (h : decide (Word.toNat a ≤ Word.toNat b) = true) : U32.le a b = true := by
  simpa [U32.le_spec, U32.toNat] using h
-- 33 U32.ge_nat
theorem u32_ge_nat (a b : U32) : U32.ge a b = decide (Word.toNat b ≤ Word.toNat a) := U32.ge_spec a b
-- 34 U32.divmod_spec (the stated width hypothesis is retained here)
theorem u32_divmod_spec (a : Word m) (b : U32) (width : decide (m ≤ 32) = true)
    (positive : decide (0 < Word.toNat b) = true) : DivSpec a b (U32.divmodGo a b) := by
  have h := U32.divmodGo_spec a b (of_decide_eq_true positive)
  exact ⟨by simpa [Nat.mul_comm] using h.2, decide_eq_true h.1⟩
-- 35 U32.zero_nat
theorem u32_zero_nat (a : U32) : U32.isZero a = decide (Word.toNat a = 0) := U32.isZero_spec a
-- 36 U32.to_nat_div (every divisor, including zero)
theorem u32_to_nat_div (a b : U32) :
    Word.toNat (U32.div a b) = Word.toNat a / Word.toNat b := U32.div_spec a b
-- 37 Nat.div_antitone
theorem nat_div_antitone (a s t : Nat) (st : decide (s ≤ t) = true)
    (positive : decide (0 < s) = true) : decide (a/t ≤ a/s) = true := by
  simp only [decide_eq_true_eq] at *
  exact Nat.div_le_div_left st positive
-- 38 U32.to_nat_lt
theorem u32_to_nat_lt (a b : U32) (h : U32.lt a b = true) :
    decide (Word.toNat a < Word.toNat b) = true := by simpa [U32.lt_spec, U32.toNat] using h
-- 39 U32.div_antitone
theorem u32_div_antitone (a s t : U32) (st : U32.le s t = true)
    (positive : U32.lt (U32.literal 0) s = true) : U32.le (U32.div a t) (U32.div a s) = true := by
  simp only [U32.le_spec, U32.lt_spec, U32.toNat, U32.literal, Word.toNat_ofNat] at *
  simp only [decide_eq_true_eq] at *
  rw [U32.div_spec, U32.div_spec]
  apply of_decide_eq_true
  exact nat_div_antitone _ _ _ (decide_eq_true st) (decide_eq_true (by simpa using positive))
-- 40 U32.positive_of_nonzero
theorem u32_positive_of_nonzero (a : U32) (h : U32.isZero a = false) :
    decide (0 < Word.toNat a) = true := by
  simp only [U32.isZero_spec, decide_eq_false_iff_not, U32.toNat] at h
  exact decide_eq_true (by omega)
-- 41 U32.zero_add
theorem u32_zero_add (a : U32) : U32.add (U32.literal 0) a = a := by
  apply Word.toNat_injective
  simp [U32.add, U32.literal, Word.toNat_add, Word.toNat_ofNat, Nat.mod_eq_of_lt (Word.toNat_lt a)]

-- Stronger forms established independently of the Bend proof's restrictions.
theorem u32_divmod_spec_without_width (a : Word m) (b : U32)
    (positive : decide (0 < Word.toNat b) = true) : DivSpec a b (U32.divmodGo a b) := by
  have h := U32.divmodGo_spec a b (of_decide_eq_true positive)
  exact ⟨by simpa [Nat.mul_comm] using h.2, decide_eq_true h.1⟩

theorem u32_to_nat_div_without_positive (a b : U32) :
    Word.toNat (U32.div a b) = Word.toNat a / Word.toNat b := U32.div_spec a b

theorem word_adc_carry_general (a b : Word n) (f : Bool) :
    Word.adc a b f true = Word.inc (Word.adc a b f false) := by
  cases f with
  | false => exact word_adc_carry a b
  | true => simpa [Word.adc] using word_adc_carry a (Word.not b)

theorem u32_le_iff (a b : U32) : U32.le a b = true ↔ Word.toNat a ≤ Word.toNat b := by
  simp [U32.le_spec, U32.toNat]

theorem u32_lt_iff (a b : U32) : U32.lt a b = true ↔ Word.toNat a < Word.toNat b := by
  simp [U32.lt_spec, U32.toNat]

theorem u32_eq_iff (a b : U32) : U32.eq a b = true ↔ Word.toNat a = Word.toNat b := by
  simp [U32.eq_spec, U32.toNat]

theorem u32_positive_iff (a : U32) : U32.isZero a = false ↔ 0 < Word.toNat a := by
  simp [U32.isZero_spec, U32.toNat]
  omega

theorem small_bounds_equivalent (n : Nat) :
    decide (n ≤ 4294967295) = decide (n ≤ maxNat 32) := by
  rw [maxNat_32_literal]

theorem maxNat_mono_iff (n m : Nat) : maxNat n ≤ maxNat m ↔ n ≤ m := by
  have hn := maxNat_succ_eq_pow n
  have hm := maxNat_succ_eq_pow m
  constructor
  · intro h
    exact (Nat.pow_le_pow_iff_right (by decide : 1 < 2)).mp (by omega)
  · intro h
    exact of_decide_eq_true (word_max_nat_mono n m (decide_eq_true h))

-- Positivity really is necessary for antitonicity with Base's /0 convention.
theorem div_antitone_without_positive_counterexample :
    (0 : Nat) ≤ 1 ∧ ¬ ((1 : Nat)/1 ≤ 1/0) := by decide

-- Valid-borrow and non-wrapping hypotheses cannot simply be dropped in general.
theorem sub_bit_without_hypothesis_counterexample :
    natBit (subLow false false false)
        (natSubCarry (subCarry false false false) 0 0) ≠
      natSubCarry false (natBit false 0) (natBit false 0) := by decide

theorem inc_without_bound_counterexample :
    (Word.inc (Word.ofNat 1 1)).toNat ≠ 1+(Word.ofNat 1 1).toNat := by decide

end BendModel.Phase2
