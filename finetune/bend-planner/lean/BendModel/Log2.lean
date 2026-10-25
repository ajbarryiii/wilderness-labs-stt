import BendModel.Refinement
set_option linter.unusedSimpArgs false

namespace BendModel.U32

theorem shrN_toNat (a : U32) (n : Nat) :
    Word.toNat (shrN a n) = Word.toNat a / 2^n := by
  have h := congrArg BitVec.toNat (shrN_toBitVec a n)
  simpa [BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow] using h

theorem shlN_one_toNat (n : Nat) (h : n < 32) :
    Word.toNat (shlN (literal 1) n) = 2^n := by
  have he := congrArg BitVec.toNat (shlN_toBitVec (literal 1) n)
  simp only [Word.toNat_toBitVec, BitVec.toNat_shiftLeft, literal,
    Word.toNat_ofNat, Nat.shiftLeft_eq] at he
  have hp : 2^n < 2^32 := Nat.pow_lt_pow_right (by decide) h
  simpa [Nat.mod_eq_of_lt hp] using he

def natLog2Go : Nat → Nat → Nat → Nat
  | 0, _, b => b
  | k+1, n, b =>
    let s := 2^k
    let t := s * bit (decide (n / 2^s ≠ 0))
    natLog2Go k (n / 2^t) (b+t)

theorem log2Go_nat (k : Nat) (a : U32) (b : Nat) (h : k ≤ 5) :
    log2Go k a b = natLog2Go k (Word.toNat a) b := by
  induction k generalizing a b with
  | zero => rfl
  | succ k ih =>
    simp only [log2Go, natLog2Go]
    have hs := shlN_one_toNat k (by omega)
    change toNat (shlN (literal 1) k) = 2^k at hs
    rw [hs]
    simp only [ne, eq_spec, toNat, literal, Word.toNat_ofNat, Nat.zero_mod,
      shrN_toNat, Nat.reducePow, ← decide_not]
    rw [ih _ _ (by omega), shrN_toNat]

theorem natLog2Go_five (x : Nat) (h : x < 2^32) : natLog2Go 5 x 0 = x.log2 := by
  by_cases hz : x = 0
  · subst x; decide
  · symm
    apply (Nat.log2_eq_iff hz).mpr
    simp only [natLog2Go, Nat.reducePow, bit, decide_eq_true_eq]
    repeat' (split <;> simp_all only [↓reduceIte, Nat.reduceAdd,
      Nat.reduceMul, Nat.reducePow, Nat.div_one])
    all_goals constructor <;>
      (try simp_all only [Nat.div_div_eq_div_mul, Nat.reduceMul]) <;> omega

theorem log2_spec (a : U32) : log2 a = (Word.toNat a).log2 := by
  rw [log2, log2Go_nat 5 a 0 (by decide)]
  exact natLog2Go_five _ (Word.toNat_lt a)

theorem log2_toUInt32 (a : U32) : log2 a = (UInt32.log2 (toUInt32 a)).toNat :=
  log2_spec a

end BendModel.U32
