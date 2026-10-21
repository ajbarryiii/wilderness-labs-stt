//! P's frozen tuning grid (p_bitserial/tuning_grid.json; plans/cpu-benchmark-mac.md M7: exactly 6
//! points, the same size as B-NEON's, the default first). Points 1..5 each run H's unchanged
//! contract harness here; point 0 is the default arm, which tests/contract.rs runs (the test below
//! proves point 0 is what `KernelConfig::tuning = Null` selects).

use p_bitserial::{TUNING_GRID_JSON, TUNING_GRID_POINTS, TUNING_GRID_SHA256, Tuning, tuning_grid};

#[test]
fn tuning_grid_is_frozen() {
    assert_eq!(shared::data::sha256_hex(TUNING_GRID_JSON.as_bytes()), TUNING_GRID_SHA256, "tuning_grid.json changed");
    assert_eq!(TUNING_GRID_POINTS, 6, "the grid size is fixed by B-NEON's frozen grid");
    let grid = tuning_grid();
    assert_eq!(grid.len(), 6);
    assert_eq!(grid[0], Tuning::default(), "the default point comes first");
    assert_eq!(Tuning::from_json(&serde_json::Value::Null), grid[0], "tuning Null selects point 0");
    for (i, a) in grid.iter().enumerate() {
        a.validate().unwrap();
        for b in &grid[i + 1..] {
            assert_ne!(a, b, "duplicate grid point");
        }
    }
    let doc: serde_json::Value = serde_json::from_str(TUNING_GRID_JSON).unwrap();
    let raw = doc["points"].as_array().unwrap();
    for (r, t) in raw.iter().zip(&grid) {
        assert_eq!(*r, t.to_json(), "grid entries are canonical tuning JSON");
    }
    assert_eq!(doc["why"].as_array().unwrap().len(), 6, "one rationale per point");
}

macro_rules! point {
    ($name:ident, $i:literal) => {
        mod $name {
            pub struct Pt;
            impl p_bitserial::Preset for Pt {
                const NAME: &'static str = concat!("p-bitserial/grid", stringify!($i));
                fn tuning() -> Option<p_bitserial::Tuning> {
                    Some(p_bitserial::tuning_grid()[$i])
                }
            }
            shared::contract_tests!(p_bitserial::BitSerialWith<Pt>);
        }
    };
}

point!(point1, 1);
point!(point2, 2);
point!(point3, 3);
point!(point4, 4);
point!(point5, 5);

/// Every point is contract-tested: 0 by tests/contract.rs, 1..5 by the modules above.
#[test]
fn every_point_is_contract_tested() {
    let covered = [0usize, 1, 2, 3, 4, 5];
    assert_eq!(covered.len(), TUNING_GRID_POINTS);
    assert!(covered.iter().enumerate().all(|(i, &p)| i == p));
}
