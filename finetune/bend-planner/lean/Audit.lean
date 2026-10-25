import BendModel
import Lean

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let names := env.constants.fold (init := #[]) fun acc name info =>
    if (`BendModel).isPrefixOf name then acc.push (name, info) else acc
  let allowed := #[``propext, ``Quot.sound, ``Classical.choice]
  let mut theorems : Nat := 0
  for (name, info) in names do
    if info.isUnsafe then
      throwError "Unsafe declaration in model: {name}"
    if info.isTheorem then theorems := theorems + 1
    for ax in (← collectAxioms name) do
      unless allowed.contains ax do
        throwError "Unapproved axiom {ax} in {name}"
  logInfo m!"Audited {names.size} BendModel declarations, including {theorems} theorems. No unsafe declarations or axioms beyond propext, Quot.sound, Classical.choice."
