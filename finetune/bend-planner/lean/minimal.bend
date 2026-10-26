import Base

def max_nat(n: Nat) -> Nat:
  match n:
    case 0n:
      0n
    case 1n+p:
      1n+Nat.double(max_nat(p))

def Holds(b: Bool) -> Type:
  {b == True{} : Bool}

def unwrap(+n: Nat, h: Holds(Nat.is_le(n, max_nat(32n)))) ->
  {Nat.is_le(n, max_nat(32n)) == True{} : Bool}:
  h
