//! The contract (shared::contract_tests!) for each B-NEON variant, default tuning.
mod b_neon {
    shared::contract_tests!(bneon::BNeon);
}
mod b_neon_1bit_tile {
    shared::contract_tests!(bneon::BNeon1BitTile);
}
mod b_neon_pre_expanded {
    shared::contract_tests!(bneon::BNeonPreExpanded);
}
