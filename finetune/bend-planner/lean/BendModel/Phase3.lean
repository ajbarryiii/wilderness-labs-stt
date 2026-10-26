import BendModel.Phase2

/-! The precise proposition-level bridge needed by the Small rewording.
This models LAWS.Holds as equality to True, rather than assuming the condition.
Valid and Result below are arbitrary: the bridge changes only Small's bound. -/
namespace BendModel.Phase3

open Phase2

def Holds (b : Bool) : Prop := b = true

def SmallLiteral (ss : List U32) : Prop :=
  Holds (decide (ss.length ≤ 4294967295))

def SmallSymbolic (ss : List U32) : Prop :=
  Holds (decide (ss.length ≤ maxNat 32))

theorem small_propositions_equal (ss : List U32) :
    SmallLiteral ss = SmallSymbolic ss := by
  exact congrArg Holds (small_bounds_equivalent ss.length)

theorem small_iff (ss : List U32) : SmallLiteral ss ↔ SmallSymbolic ss := by
  rw [small_propositions_equal]

-- Like Bend's equality elimination in small_bound, the proof exposes the
-- equation already carried by Holds. It proves no bound without that evidence.
theorem holds_eliminate (p : Bool) (w : Holds p) : p = true := by
  change p = true at w
  cases w
  rfl

theorem small_bound (ss : List U32) (w : SmallSymbolic ss) :
    decide (ss.length ≤ maxNat 32) = true :=
  holds_eliminate _ w

theorem cover_contract_equivalent
    (Valid Result : List U32 → List U32 → Prop) :
    (∀ ss ps, SmallLiteral ss → Valid ss ps → Result ss ps) ↔
    (∀ ss ps, SmallSymbolic ss → Valid ss ps → Result ss ps) := by
  constructor
  · intro h ss ps small valid
    exact h ss ps ((small_iff ss).mpr small) valid
  · intro h ss ps small valid
    exact h ss ps ((small_iff ss).mp small) valid

-- The width-33 mutation is a strictly weaker assumption, not a restatement.
theorem width33_changes_condition :
    ¬ Holds (decide (4294967296 ≤ maxNat 32)) ∧
      Holds (decide (4294967296 ≤ maxNat 33)) := by
  rw [maxNat_eq_pow_sub_one, maxNat_eq_pow_sub_one]
  unfold Holds
  decide

theorem width33_roundtrip_counterexample :
    Holds (decide (4294967296 ≤ maxNat 33)) ∧
      U32.toNat (U32.fromNat 4294967296) ≠ 4294967296 := by
  rw [maxNat_eq_pow_sub_one, U32.fromNat_spec]
  unfold Holds
  decide

end BendModel.Phase3
