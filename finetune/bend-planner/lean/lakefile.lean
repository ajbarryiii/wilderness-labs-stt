import Lake
open Lake DSL

package bendLeanReview where
  moreLeanArgs := #["-DwarningAsError=true"]

lean_lib BendModel

@[default_target]
lean_exe fidelity where
  root := `Main
