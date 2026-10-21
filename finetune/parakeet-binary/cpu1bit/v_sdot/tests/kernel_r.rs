//! V's microkernel contract, for every frame-part count R = 1..=16 (H's contract shapes reach only
//! some R): over a 4-row group and `chunks` 32-input chunks of the v-layout, with frame-part `r`
//! read at `a + r * lda`, the kernel's accumulators give exactly `d = -2 acc - U = Σ_k w_k q_k`.
//! Values: random s8, all -128, all +127, alternating; weights random, all +1, all -1.
#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::*;
use shared::rng::SplitMix64;
use v_sdot::kernel::asm::dot_asm;
use v_sdot::layout;

fn check<const R: usize>() {
    let mut rng = SplitMix64::new(R as u64);
    for &chunks in &[1usize, 2, 5, 32] {
        let k = 32 * chunks;
        let lda = k + 48; // rows strided, not contiguous
        let rb = k / 8;
        for wpat in 0..3 {
            let codes: Vec<u8> = (0..4 * rb)
                .map(|_| match wpat {
                    0 => rng.next_u64() as u8,
                    1 => 0xFF,
                    _ => 0x00,
                })
                .collect();
            let main = layout::build_main(&codes, rb, 4, k, 1, chunks);
            for apat in 0..4 {
                let q: Vec<i8> = (0..R * lda)
                    .map(|j| match apat {
                        0 => rng.next_u64() as i8,
                        1 => -128,
                        2 => 127,
                        _ => if j % 2 == 0 { -128 } else { 127 },
                    })
                    .collect();
                let acc = unsafe { dot_asm::<R>(main.as_ptr(), q.as_ptr(), lda, chunks) };
                for r in 0..R {
                    let row = &q[r * lda..r * lda + k];
                    let u: i32 = row.iter().map(|&v| v as i32).sum();
                    let mut got = [0i32; 4];
                    unsafe { vst1q_s32(got.as_mut_ptr(), acc[r]) };
                    for nl in 0..4 {
                        let want: i32 = row
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| (2 * layout::code_bit(&codes, rb, nl, i) as i32 - 1) * v as i32)
                            .sum();
                        assert_eq!(-2 * got[nl] - u, want, "R={R} chunks={chunks} w={wpat} a={apat} r={r} row={nl}");
                    }
                }
            }
        }
    }
}

#[test]
fn every_frame_part_count_is_exact() {
    check::<1>();
    check::<2>();
    check::<3>();
    check::<4>();
    check::<5>();
    check::<6>();
    check::<7>();
    check::<8>();
    check::<9>();
    check::<10>();
    check::<11>();
    check::<12>();
    check::<13>();
    check::<14>();
    check::<15>();
    check::<16>();
}

#[test]
fn layout_round_trips_with_padding() {
    let mut rng = SplitMix64::new(3);
    for &(n, k) in &[(1usize, 1usize), (5, 63), (4, 32), (130, 1031), (17, 4096)] {
        let rb = k.div_ceil(8);
        let mut codes: Vec<u8> = (0..n * rb).map(|_| rng.next_u64() as u8).collect();
        if k % 8 != 0 {
            for o in 0..n {
                codes[o * rb + rb - 1] &= (1u8 << (k % 8)) - 1;
            }
        }
        let (groups, chunks) = (n.div_ceil(4), k.div_ceil(32));
        let main = layout::build_main(&codes, rb, n, k, groups, chunks);
        assert_eq!(main.len(), groups * chunks * 16);
        for o in 0..groups * 4 {
            for i in 0..chunks * 32 {
                let want = if o < n && i < k { layout::code_bit(&codes, rb, o, i) } else { 0 };
                assert_eq!(layout::main_bit(&main, chunks, o, i), want, "n={n} k={k} o={o} i={i}");
            }
        }
    }
}
