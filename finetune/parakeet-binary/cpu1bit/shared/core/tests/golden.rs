//! The shared quantizer and the oracle against the x86 LocalFlow golden vectors (M8).

#[test]
fn quantizer_matches_x86_localflow() {
    shared::golden::check_quantizer(&shared::golden::Golden::load());
}

#[test]
fn oracle_matches_x86_localflow() {
    shared::golden::check_oracle(&shared::golden::Golden::load());
}
