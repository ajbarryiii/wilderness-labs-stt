# Encoder padding and masking contract

This is the S0 masking contract that DESIGN.md gate 5 checks. `encoder.py` implements it, and
`gates.py` (gate 5) tests it. It holds for every arm, bucket and length variant (fixed, multifunction,
enumerated). G0 has the same graph but only C0's 15 s window.

## Inputs

- `mel`: Float32 `[1, 128, F_b]`, with F_b = 201, 401, 801 or 1501 for the 2, 4, 8 and 15 s buckets.
  F_b = 16000 b // 160 + 1 is the number of STFT frames of a full bucket.
- `mel_length`: Int32 `[1]`. L0 = M = N // 160 is the number of **valid** mel frames of an utterance of
  N samples (NeMo `get_seq_len` with n_fft 512, hop 160, `exact_pad` False).
  - Precondition: 1 ≤ L0 ≤ F_b.
  - C0's preprocessor reports `N // 160 + 1` instead (README "C0 deviates", item 1). A caller that
    feeds C0's preprocessor output to these encoders gets one more valid frame than NeMo.
- Frames `t ≥ L0` of `mel` must be zero. NeMo's feature contract pads after normalization with
  `pad_value` 0, and the reference features already zero frame M. The graph zeroes them anyway (step 1).

## Lengths at every stage

Each stride-2 stage (kernel 3, padding 1) maps L → ⌊(L − 1) / 2⌋ + 1 = ⌈L / 2⌉, in int32 `floor_div`.
The reference computes the same in float and floors. For L = 0 it gives 0.

| Stage | Valid length | Padded length (allocated), 2/4/8/15 s |
| --- | --- | --- |
| mel | L0 = M | 201 / 401 / 801 / 1501 |
| after subsampling conv 0 | L1 = ⌈L0 / 2⌉ | 101 / 201 / 401 / 751 |
| after depthwise stage 1 | L2 = ⌈L1 / 2⌉ | 51 / 101 / 201 / 376 |
| after depthwise stage 2 = encoder | L3 = ⌈L2 / 2⌉ = `encoder_length` | T = 26 / 51 / 101 / 188 |

## Masks

1. **Subsampling.** NeMo's `MaskedConvSequential` masks the input of each of its eight layers with the
   current length. Our graph multiplies by a 0/1 FP16 time mask in four places:
   - the mel input, with L0;
   - the input of depthwise stage 1, with L1;
   - the input of depthwise stage 2, with L2;
   - the final activations, with L3.

   This equals the reference on every frame. The reference's other masks sit before ReLUs and 1×1
   convolutions, and those act per frame: a padded frame they leave non-zero is zeroed again by the
   next mask, before any kernel that mixes frames.
2. **Attention.** valid(t) = t < L3, for t in [0, T).
   - The mask is `bad[i, j] = not (valid(i) and valid(j))`, shape `[1, 1, T, T]`, exactly as the
     reference does it.
   - Scores are `select(bad, −10000, scores)`, then softmax, then `select(bad, 0, attn)`.
   - The **key** part is what protects valid frames: a valid query gets zero weight on padded keys,
     because exp(−10000 − max) underflows to 0 in FP16 as in FP32.
   - The query part zeroes the attention output of padded frames, which then contribute
     `linear_out(0) = 0`.
3. **Convolution module.** After the GLU, `select(not valid, 0, x)` on `[1, 1024, T]`. Then
   `pad(4, 4)` with zeros, then the depthwise k = 9 convolution, with BatchNorm folded into its weight
   and bias. This zeroes padded frames before each depthwise conv. The padded tail and the explicit
   zero padding are therefore both zero, as in NeMo.
4. **Relative positions.** For the padded length T, the position table holds the positions
   T − 1 … −(T − 1): 2T − 1 columns of `linear_pos(pos_emb)`, `[1, 8, 128, 2T − 1]`.
   - `rel_shift` (pad left by 1, view `[1, 8, 2T, T]`, drop the first row, view `[1, 8, T, 2T − 1]`,
     keep the first T columns) gives entry (i, j) the relative position i − j.
   - Indexing is relative to the **padded** length T, as in NeMo. The value at (i, j) depends only
     on i − j, so a valid frame sees the same position terms in every bucket.
   - Each bucket's table is the middle 2T − 1 columns of the 15 s table, computed once in FP32 with
     the FP16-rounded scales and then cast to FP16.
   - Fixed and multifunction models store one constant per bucket. The enumerated-shape model
     slices the 15 s table at offsets [188 − T, 187 + T), computed from the input shape.

## Outputs

- `encoder`: Float32 `[1, 1024, T]`. Only frames t < `encoder_length` are defined. Frames at or
  beyond it hold whatever the per-frame layers make of the masked inputs; they are finite in every
  gated run, but consumers must not read them.
- `encoder_length`: Int32 `[1]`, equal to L3.

## What gate 5 checks

On every clip in every bucket it fits, the encoder runs on `mel` padded to F_b with
`mel_length = M`. Then:
- `encoder_length` must equal ⌈M / 8⌉ (with the stage formula above);
- the valid frames must be finite;
- the valid frames must match the same arm's 15 s output within the gate-4 ceilings (rel ≤ 2e-2,
  abs ≤ 0.25).

The gate counts the boundary-length, silence and impulse clips. Natural clips are reported too.
The boundary crops are placed at each bucket edge (N_e, N_e + 1) and around the stride steps
(M = 8k − 1, 8k, 8k + 1), so every case where L3 changes is covered.
