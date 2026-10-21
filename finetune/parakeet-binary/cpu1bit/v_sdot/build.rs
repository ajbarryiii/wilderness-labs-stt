//! Generates V's inline-asm dot kernels into OUT_DIR/dot_asm.rs: the production `dot_r1` ..
//! `dot_r16` ({0,-1} weights from the 1-bit v-layout) and, for the dev bench only, controls with
//! the same register plan: `dot_lm_r16` (lane-major SDOT order), `dot_pm_r15` (±1: cmtst + orr),
//! `dot_b01_r15` ({0,1}: and / ushr + and / ushr) and `dot_tile_r15/16` (weights pre-expanded to
//! {0,-1} s8 in memory, 128 bytes per chunk: the "no expansion cost" control).
//!
//! Register plan (R frame-parts, 1 <= R <= 16):
//! - v0 .. v(R-1): i32x4 accumulators, accumulator r = frame-part r, lanes = the group's 4 rows;
//! - v16 .. v22: byte masks 1 << j for j = 0..6 (bit 7 uses `cmlt #0`, no mask);
//! - v24: the chunk's 16 code bytes; v25 .. v28: the 4 expanded weight vectors of a half chunk;
//! - activations: v29, v30, v31, v23 (alternative forms: v15, with v23 = bytes 1).
//!
//! Schedule per half chunk: frame-parts in blocks of 4 sharing base pointer b_i (= a + 4i lda):
//! parts 4i+1, 4i+2, 4i+3 load from [b_i, lda], [b_i, 2 lda], [b_i, 3 lda], part 4i last with a
//! post-index of 16 (every base advances 32 bytes per chunk with no extra instruction). Production
//! order is row-major: each load followed by its frame-part's 4 SDOTs (lanes 0..3). The lane-major
//! order (a block's 4 loads, then its 16 SDOTs lane by lane) measured slower (dev bench, E-cores:
//! 75.5% vs 83.0% of the SDOT peak loop at R = 16, K = 4096).
use std::fmt::Write;

#[derive(Clone, Copy, PartialEq)]
enum Form {
    /// {0,-1}: 0xFF where the code is +1 (one op per 16 weights). Production.
    Z,
    /// ±1 as -w: 0xFF (-1) where the code is +1, 0x01 where it is -1.
    Pm,
    /// {0,1}: 1 where the code is +1.
    B01,
    /// Pre-expanded {0,-1} bytes loaded from memory (no expansion ops).
    Tile,
}

/// Instructions producing weight register `v{dst}` for input quad `j` of the chunk.
fn expand(form: Form, j: usize, dst: usize) -> Vec<String> {
    let mask = 16 + j;
    match form {
        Form::Z if j == 7 => vec![format!("cmlt v{dst}.16b, v24.16b, #0")],
        Form::Z => vec![format!("cmtst v{dst}.16b, v24.16b, v{mask}.16b")],
        Form::Pm if j == 7 => vec![format!("cmlt v{dst}.16b, v24.16b, #0"), format!("orr v{dst}.16b, v{dst}.16b, v23.16b")],
        Form::Pm => vec![format!("cmtst v{dst}.16b, v24.16b, v{mask}.16b"), format!("orr v{dst}.16b, v{dst}.16b, v23.16b")],
        Form::B01 if j == 0 => vec![format!("and v{dst}.16b, v24.16b, v23.16b")],
        Form::B01 if j == 7 => vec![format!("ushr v{dst}.16b, v24.16b, #7")],
        Form::B01 => vec![format!("ushr v{dst}.16b, v24.16b, #{j}"), format!("and v{dst}.16b, v{dst}.16b, v23.16b")],
        Form::Tile => vec![format!("ldr q{dst}, [{{w}}], #16")],
    }
}

/// Activation load of frame-part 4i + o into v{a} (post-index for o = 0, the block's last load).
fn load(i: usize, o: usize, a: usize) -> String {
    if o == 0 { format!("ldr q{a}, [{{b{i}}}], #16") } else { format!("ldr q{a}, [{{b{i}}}, {{k{o}}}]") }
}

fn kernel(r: usize, form: Form, lane_major: bool) -> String {
    let name = match form {
        Form::Z if lane_major => format!("dot_lm_r{r}"),
        Form::Z => format!("dot_r{r}"),
        Form::Pm => format!("dot_pm_r{r}"),
        Form::B01 => format!("dot_b01_r{r}"),
        Form::Tile => format!("dot_tile_r{r}"),
    };
    let ones = matches!(form, Form::Pm | Form::B01);
    assert!(!ones || r <= 15, "alternative forms use v15");
    let aregs: [usize; 4] = if ones { [29, 30, 31, 15] } else { [29, 30, 31, 23] };
    let bases = r.div_ceil(4);
    let need_k = (1..4).map(|o| (0..bases).any(|i| 4 * i + o < r)).collect::<Vec<_>>();
    let mut asm = String::new();
    let mut line = |s: String| {
        writeln!(asm, "            \"{s}\",").unwrap();
    };
    if matches!(form, Form::Z | Form::Pm) {
        for j in 0..7 {
            line(format!("movi v{}.16b, #{}", 16 + j, 1u32 << j));
        }
    }
    if ones {
        line("movi v23.16b, #1".into());
    }
    line("2:".into());
    if form != Form::Tile {
        line("ldr q24, [{w}], #16".into());
    }
    for h in 0..2 {
        for j in 0..4 {
            for s in expand(form, 4 * h + j, 25 + j) {
                line(s);
            }
        }
        for i in 0..bases {
            let rows: Vec<(usize, usize)> = [1usize, 2, 3, 0]
                .iter()
                .filter(|&&o| 4 * i + o < r)
                .enumerate()
                .map(|(slot, &o)| (o, aregs[slot]))
                .collect();
            if lane_major {
                for &(o, a) in &rows {
                    line(load(i, o, a));
                }
                for lane in 0..4 {
                    for &(o, a) in &rows {
                        line(format!("sdot v{}.4s, v{}.16b, v{a}.4b[{lane}]", 4 * i + o, 25 + lane));
                    }
                }
            } else {
                for &(o, a) in &rows {
                    line(load(i, o, a));
                    for lane in 0..4 {
                        line(format!("sdot v{}.4s, v{}.16b, v{a}.4b[{lane}]", 4 * i + o, 25 + lane));
                    }
                }
            }
        }
    }
    line("subs {n}, {n}, #1".into());
    line("b.ne 2b".into());

    let what = match form {
        Form::Z => "{0,-1}",
        Form::Pm => "±1",
        Form::B01 => "{0,1}",
        Form::Tile => "pre-expanded {0,-1} tile, 128 B per chunk",
    };
    let mut f = String::new();
    writeln!(f, "/// Generated ({what}, {r} frame-parts), see build.rs.").unwrap();
    writeln!(f, "/// # Safety\n/// As `dot_asm` with `R = {r}`; `chunks >= 1`.").unwrap();
    writeln!(f, "#[inline(always)]").unwrap();
    writeln!(
        f,
        "pub unsafe fn {name}(w: *const u8, a: *const i8, lda: usize, chunks: usize) -> [int32x4_t; {r}] {{"
    )
    .unwrap();
    writeln!(f, "    unsafe {{").unwrap();
    writeln!(f, "        let mut acc = [vdupq_n_s32(0); {r}];").unwrap();
    for i in 0..bases {
        writeln!(f, "        let b{i} = a.add({} * lda);", 4 * i).unwrap();
    }
    writeln!(f, "        core::arch::asm!(").unwrap();
    f.push_str(&asm);
    writeln!(f, "            w = inout(reg) w => _,").unwrap();
    for i in 0..bases {
        writeln!(f, "            b{i} = inout(reg) b{i} => _,").unwrap();
    }
    for o in 1..4 {
        if need_k[o - 1] {
            writeln!(f, "            k{o} = in(reg) {o} * lda,").unwrap();
        }
    }
    writeln!(f, "            n = inout(reg) chunks => _,").unwrap();
    for row in 0..r {
        writeln!(f, "            inout(\"v{row}\") acc[{row}],").unwrap();
    }
    let mut clobbers: Vec<usize> = (16..32).collect();
    if ones {
        clobbers.push(15);
    }
    for v in clobbers {
        writeln!(f, "            out(\"v{v}\") _,").unwrap();
    }
    writeln!(f, "            options(nostack, readonly),").unwrap();
    writeln!(f, "        );").unwrap();
    writeln!(f, "        acc").unwrap();
    writeln!(f, "    }}").unwrap();
    writeln!(f, "}}\n").unwrap();
    f
}

fn main() {
    let mut out = String::from("// @generated by v_sdot/build.rs; do not edit.\n\n");
    for r in 1..=16 {
        out.push_str(&kernel(r, Form::Z, false));
    }
    out.push_str(&kernel(15, Form::Pm, false));
    out.push_str(&kernel(16, Form::Z, true));
    out.push_str(&kernel(15, Form::B01, false));
    out.push_str(&kernel(15, Form::Tile, false));
    out.push_str(&kernel(16, Form::Tile, false));
    // Dispatcher from a const R to the generated production kernel.
    out.push_str("/// The generated kernel for `R` frame-parts.\n/// # Safety\n/// As the `dot_rN` kernels.\n");
    out.push_str("#[inline(always)]\npub unsafe fn dot_asm<const R: usize>(w: *const u8, a: *const i8, lda: usize, chunks: usize) -> [int32x4_t; R] {\n");
    out.push_str("    unsafe {\n        let mut out = [vdupq_n_s32(0); R];\n        if chunks == 0 { return out; }\n        match R {\n");
    for r in 1..=16 {
        writeln!(
            out,
            "            {r} => {{ let v = dot_r{r}(w, a, lda, chunks); core::ptr::copy_nonoverlapping(v.as_ptr(), out.as_mut_ptr(), {r}); }}"
        )
        .unwrap();
    }
    out.push_str("            _ => panic!(\"no kernel for R\"),\n        }\n        out\n    }\n}\n");
    let dir = std::env::var("OUT_DIR").unwrap();
    std::fs::write(format!("{dir}/dot_asm.rs"), out).unwrap();
    println!("cargo:rerun-if-changed=build.rs");
}
