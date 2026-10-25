import Std
set_option linter.unusedSimpArgs false

/-! Independent structural transcription of the Word part of installed Bend 2.0.37 Base.
    Heads are LOW bits. No Bend proof is used here. -/
namespace BendModel

def bit (b : Bool) : Nat := if b then 1 else 0
@[simp] theorem bit_false : bit false = 0 := rfl
@[simp] theorem bit_true : bit true = 1 := rfl

inductive Word : Nat → Type where
  | nil : Word 0
  | cons {n : Nat} (head : Bool) (tail : Word n) : Word (n+1)
  deriving Repr, DecidableEq

namespace Word

def toNat : Word n → Nat
  | .nil => 0
  | .cons b t => bit b + 2 * toNat t

def zero : (n : Nat) → Word n
  | 0 => .nil
  | n+1 => .cons false (zero n)

def not : Word n → Word n
  | .nil => .nil
  | .cons b t => .cons (!b) (not t)

def map₂ (f : Bool → Bool → Bool) : Word n → Word n → Word n
  | .nil, .nil => .nil
  | .cons a ax, .cons b bt => .cons (f a b) (map₂ f ax bt)

def fullAdd (a b c : Bool) : Bool × Bool :=
  if a then (!(b ^^ c), b || c) else (b ^^ c, b && c)

-- Base's f=False branch; the extra final carry is only proof instrumentation.
def adcFull : Word n → Word n → Bool → Bool × Word n
  | .nil, .nil, c => (c, .nil)
  | .cons a ax, .cons b bt, c =>
    let (s,k) := fullAdd a b c
    let (out,t) := adcFull ax bt k
    (out, .cons s t)

def adc (a b : Word n) (f c : Bool) : Word n :=
  (adcFull a (if f then not b else b) c).2

def add (a b : Word n) : Word n := adc a b false false
def sub (a b : Word n) : Word n := adc a b true true

def inc : Word n → Word n
  | .nil => .nil
  | .cons false t => .cons true t
  | .cons true t => .cons false (inc t)

def shlPut (c : Bool) : Word n → Word n
  | .nil => .nil
  | .cons b t => .cons c (shlPut b t)

def shl (w : Word n) : Word n := shlPut false w

def shlOut (c : Bool) : Word n → Bool × Word n
  | .nil => (c, .nil)
  | .cons b t =>
    let (hi,t₂) := shlOut b t
    (hi, .cons c t₂)

def shrPad : Word n → Word (n+1)
  | .nil => .cons false .nil
  | .cons b t => .cons b (shrPad t)

def shr : Word n → Word n
  | .nil => .nil
  | .cons _ t => shrPad t

def cmpFin (a b : Bool) (t : Ordering) : Ordering :=
  match t with
  | .lt => .lt
  | .gt => .gt
  | .eq => if a == b then .eq else if a then .gt else .lt

def cmp : Word n → Word n → Ordering
  | .nil, .nil => .eq
  | .cons a ax, .cons b bt => cmpFin a b (cmp ax bt)

def mulGo (a : Word m) (b acc : Word n) : Word n :=
  match a with
  | .nil => acc
  | .cons false ax => mulGo ax (shl b) acc
  | .cons true ax => mulGo ax (shl b) (add acc b)

def mul (a b : Word n) : Word n := mulGo a b (zero n)

-- Efficient bit decoder for input literals, independently proved below.
def ofNat : (n : Nat) → Nat → Word n
  | 0, _ => .nil
  | n+1, x => .cons (x % 2 == 1) (ofNat n (x / 2))

theorem toNat_lt (w : Word n) : toNat w < 2^n := by
  induction w with
  | nil => decide
  | @cons n b t ih =>
    simp only [toNat, Nat.pow_succ]
    cases b <;> simp only [bit, Bool.false_eq_true, ↓reduceIte] <;> omega

@[simp] theorem toNat_zero (n : Nat) : toNat (zero n) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp [zero, toNat, bit, ih]

theorem toNat_injective {a b : Word n} (h : toNat a = toNat b) : a = b := by
  induction a with
  | nil => cases b; rfl
  | cons a ax ih =>
    cases b with
    | cons b bt =>
      simp only [toNat] at h
      have ht : toNat ax = toNat bt := by cases a <;> cases b <;> simp [bit] at h <;> omega
      have hb : a = b := by cases a <;> cases b <;> simp_all [bit]
      cases hb
      exact congrArg (cons a) (ih ht)

def toBitVec (w : Word n) : BitVec n := BitVec.ofNatLT (toNat w) (toNat_lt w)

@[simp] theorem toNat_toBitVec (w : Word n) : w.toBitVec.toNat = w.toNat := rfl

theorem adcFull_spec (a b : Word n) (c : Bool) :
    toNat a + toNat b + bit c =
      toNat (adcFull a b c).2 + 2^n * bit (adcFull a b c).1 := by
  induction a generalizing c with
  | nil => cases b; simp [adcFull, toNat]
  | @cons n a ax ih =>
    cases b with
    | cons b bt =>
      have h := ih bt (fullAdd a b c).2
      cases a <;> cases b <;> cases c <;>
        simp [adcFull, fullAdd, toNat, bit, Nat.pow_succ] at h ⊢ <;>
        simp only [Nat.mul_right_comm (2^n) 2] <;> omega

theorem toNat_add (a b : Word n) :
    toNat (add a b) = (toNat a + toNat b) % 2^n := by
  have h := adcFull_spec a b false
  simp only [bit, Bool.false_eq_true, ↓reduceIte, Nat.add_zero] at h
  rw [h, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (toNat_lt _)]
  rfl

theorem toNat_not (a : Word n) : toNat (not a) = 2^n - 1 - toNat a := by
  induction a with
  | nil => rfl
  | @cons n b t ih =>
    have ht := toNat_lt t
    have hp := Nat.two_pow_pos n
    cases b <;> simp [not, toNat, bit, ih, Nat.pow_succ] <;> omega

theorem toNat_sub (a b : Word n) :
    toNat (sub a b) = (2^n - toNat b + toNat a) % 2^n := by
  have h := adcFull_spec a (not b) true
  rw [toNat_not] at h
  have hb := toNat_lt b
  have hp := Nat.two_pow_pos n
  have he : toNat a + (2^n - 1 - toNat b) + bit true =
      2^n - toNat b + toNat a := by simp [bit]; omega
  rw [he] at h
  rw [h, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (toNat_lt _)]
  rfl

theorem shlOut_spec (w : Word n) (c : Bool) :
    bit c + 2 * toNat w =
      toNat (shlOut c w).2 + 2^n * bit (shlOut c w).1 := by
  induction w generalizing c with
  | nil => simp [shlOut, toNat]
  | @cons n b t ih =>
    have h := ih b
    simp only [shlOut, toNat, Nat.pow_succ]
    rw [Nat.mul_right_comm (2^n) 2]
    omega

theorem shlOut_snd (w : Word n) (c : Bool) : (shlOut c w).2 = shlPut c w := by
  induction w generalizing c with
  | nil => rfl
  | cons b t ih => simp [shlOut, shlPut, ih]

theorem toNat_shl (w : Word n) : toNat (shl w) = (2 * toNat w) % 2^n := by
  have h := shlOut_spec w false
  simp only [bit, Bool.false_eq_true, ↓reduceIte, Nat.zero_add] at h
  rw [h, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (toNat_lt _)]
  rw [shlOut_snd]
  rfl

theorem toNat_shrPad (w : Word n) : toNat (shrPad w) = toNat w := by
  induction w with
  | nil => rfl
  | cons b t ih => simp [shrPad, toNat, ih]

theorem toNat_shr (w : Word n) : toNat (shr w) = toNat w / 2 := by
  cases w with
  | nil => rfl
  | cons b t =>
    simp [shr, toNat_shrPad, toNat]
    cases b <;> simp [bit, Nat.add_div]

theorem toNat_inc (w : Word n) : toNat (inc w) = (toNat w + 1) % 2^n := by
  induction w with
  | nil => rfl
  | @cons n b t ih =>
    have ht := toNat_lt t
    cases b with
    | false =>
      simp [inc, toNat, bit]
      rw [Nat.mod_eq_of_lt]
      · omega
      · simp only [Nat.pow_succ]; omega
    | true =>
      simp only [inc, toNat, bit, Bool.false_eq_true, ↓reduceIte, Nat.zero_add, ih]
      have he : 1 + 2 * toNat t + 1 = 2 * (toNat t + 1) := by omega
      rw [he, Nat.pow_succ, Nat.mul_comm (2^n) 2, Nat.mul_mod_mul_left]

theorem cmp_spec (a b : Word n) : cmp a b = compare (toNat a) (toNat b) := by
  induction a with
  | nil => cases b; rfl
  | cons a ax ih =>
    cases b with
    | cons b bt =>
      rw [cmp, ih]
      cases h : compare (toNat ax) (toNat bt) with
      | lt =>
        have h := Nat.compare_eq_lt.mp h
        cases a <;> cases b <;> simp only [cmpFin] <;>
          symm <;> apply Nat.compare_eq_lt.mpr <;> simp [toNat, bit] <;> omega
      | eq =>
        have h := Nat.compare_eq_eq.mp h
        cases a <;> cases b <;> simp [cmpFin, toNat, bit, Nat.compare_eq_ite_lt, h] <;> omega
      | gt =>
        have h := Nat.compare_eq_gt.mp h
        cases a <;> cases b <;> simp only [cmpFin] <;>
          symm <;> apply Nat.compare_eq_gt.mpr <;> simp [toNat, bit] <;> omega

theorem toNat_ofNat (n x : Nat) : toNat (ofNat n x) = x % 2^n := by
  induction n generalizing x with
  | zero => simp [ofNat, toNat, Nat.mod_one]
  | succ n ih =>
    simp only [ofNat, toNat, ih, Nat.pow_succ]
    have hx := Nat.mod_add_div x 2
    have hr := Nat.mod_lt x (by decide : 0 < 2)
    have hb : bit (x % 2 == 1) = x % 2 := by
      simp only [bit, beq_iff_eq]
      split <;> omega
    rw [hb, Nat.mul_comm (2^n) 2]
    have hm := Nat.mul_mod_mul_left 2 (x/2) (2^n)
    have hl := Nat.mod_lt (x/2) (Nat.two_pow_pos n)
    have hd : x = x % 2 + 2 * (x/2) := by omega
    conv => rhs; rw [hd, Nat.add_mod, hm]
    rw [Nat.mod_eq_of_lt (by have := Nat.two_pow_pos n; omega : x % 2 < 2 * 2^n)]
    have hs : x % 2 + 2 * ((x/2) % 2^n) < 2 * 2^n := by omega
    exact (Nat.mod_eq_of_lt hs).symm

@[simp] theorem toBitVec_add (a b : Word n) : (add a b).toBitVec = a.toBitVec + b.toBitVec := by
  apply BitVec.eq_of_toNat_eq
  exact toNat_add a b

@[simp] theorem toBitVec_sub (a b : Word n) : (sub a b).toBitVec = a.toBitVec - b.toBitVec := by
  apply BitVec.eq_of_toNat_eq
  exact (toNat_sub a b).trans (BitVec.toNat_sub a.toBitVec b.toBitVec).symm

theorem toNat_mulGo (a : Word m) (b acc : Word n) :
    toNat (mulGo a b acc) = (toNat acc + toNat a * toNat b) % 2^n := by
  induction a generalizing b acc with
  | nil => simp [mulGo, toNat, Nat.mod_eq_of_lt (toNat_lt acc)]
  | cons h t ih =>
    cases h <;> simp [mulGo, toNat, bit, ih, toNat_shl, toNat_add,
      Nat.mul_add, Nat.add_mul, Nat.add_mod, Nat.mul_assoc, Nat.mul_comm,
      Nat.mul_left_comm, Nat.add_assoc] <;>
      simp only [Nat.mod_mod, Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]

theorem toNat_mul (a b : Word n) : toNat (mul a b) = (toNat a * toNat b) % 2^n := by
  simp [mul, toNat_mulGo]

@[simp] theorem toBitVec_mul (a b : Word n) : (mul a b).toBitVec = a.toBitVec * b.toBitVec := by
  apply BitVec.eq_of_toNat_eq
  exact toNat_mul a b

@[simp] theorem toBitVec_not (a : Word n) : (not a).toBitVec = ~~~a.toBitVec := by
  apply BitVec.eq_of_toNat_eq
  exact (toNat_not a).trans (BitVec.toNat_not (x := a.toBitVec)).symm

@[simp] theorem toBitVec_cons (b : Bool) (t : Word n) :
    (cons b t).toBitVec = t.toBitVec.concat b := by
  apply BitVec.eq_of_toNat_eq
  simp only [toNat_toBitVec, toNat, BitVec.toNat_concat]
  cases b <;> simp [bit, Nat.mul_comm, Nat.add_comm]

@[simp] theorem toBitVec_and (a b : Word n) :
    (map₂ (· && ·) a b).toBitVec = a.toBitVec &&& b.toBitVec := by
  induction a with
  | nil => cases b; rfl
  | cons h t ih => cases b with
    | cons j s => simp [map₂, ih]

@[simp] theorem toBitVec_or (a b : Word n) :
    (map₂ (· || ·) a b).toBitVec = a.toBitVec ||| b.toBitVec := by
  induction a with
  | nil => cases b; rfl
  | cons h t ih => cases b with
    | cons j s => simp [map₂, ih]

@[simp] theorem toBitVec_xor (a b : Word n) :
    (map₂ (· ^^ ·) a b).toBitVec = a.toBitVec ^^^ b.toBitVec := by
  induction a with
  | nil => cases b; rfl
  | cons h t ih => cases b with
    | cons j s => simp [map₂, ih]

@[simp] theorem toBitVec_shl (a : Word n) : (shl a).toBitVec = a.toBitVec <<< 1 := by
  apply BitVec.eq_of_toNat_eq
  simp only [toNat_toBitVec, toNat_shl, BitVec.toNat_shiftLeft, Nat.shiftLeft_eq]
  simp [Nat.mul_comm]

@[simp] theorem toBitVec_shr (a : Word n) : (shr a).toBitVec = a.toBitVec >>> 1 := by
  apply BitVec.eq_of_toNat_eq
  simp [toNat_shr, BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow]

end Word
end BendModel
