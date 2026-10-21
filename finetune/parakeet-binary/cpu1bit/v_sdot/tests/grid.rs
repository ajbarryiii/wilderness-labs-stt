//! H's unchanged contract suite at every non-default point of V's frozen tuning grid
//! (`v_sdot/grid.json`, 6 points; point 0 = the default, covered by tests/contract.rs), so
//! whichever point the coordinator's tuning stage selects has passed the contract.

macro_rules! preset {
    ($name:ident, $idx:expr) => {
        mod $name {
            pub struct Pt;
            impl v_sdot::Preset for Pt {
                const NAME: &'static str = concat!("v-sdot/", stringify!($name));
                fn tuning() -> serde_json::Value {
                    v_sdot::grid()[$idx].clone()
                }
            }
            shared::contract_tests!(v_sdot::VSdotWith<Pt>);
        }
    };
}

preset!(p1_n8_w7, 1);
preset!(p2_n8_w6, 2);
preset!(p3_n7_w7, 3);
preset!(p4_n7_w6, 4);
preset!(p5_n6_w6, 5);
