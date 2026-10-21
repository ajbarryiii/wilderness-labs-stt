Sweep wp7-20261003-184043. Informational (shared Mac M1 Pro, macOS 27; no claims). Loads: 'post-purge load' = the first load after this binary's Core ML cache directory was purged; 'subsequent fresh-process load' = a new process loading the arm afterwards. Whether Core ML specialized or reused a cache in either is not established (no Instruments cache events).

| arm | total ratio vs C0 (2 / 4 / 8 / 15 s) [95% CI] | encoder ms (2 / 4 / 8 / 15 s) | load post-purge / subsequent fresh process (enc ms) | footprint MB |
|---|---|---|---|---|
| C4-ane-multi-vdsp-f2 | 0.40 [0.39, 0.42] / 0.45 [0.43, 0.46] / 0.57 [0.55, 0.58] / 0.85 [0.84, 0.86] | 18.5 / 19.6 / 24.4 / 47.4 | 179623, 172661 / 3985, 3621 | 304 |
| C3-ane-multi-vdsp-f2 | 0.39 [0.38, 0.41] / 0.44 [0.43, 0.45] / 0.57 [0.55, 0.59] / 0.87 [0.85, 0.88] | 17.8 / 18.8 / 23.5 / 49.2 | 184220, 176080 / 4310, 3739 | 253 |
| C6s8-ane-multi-vdsp-f2 | 0.37 [0.36, 0.38] / 0.41 [0.40, 0.42] / 0.55 [0.52, 0.55] / 0.83 [0.82, 0.85] | 15.9 / 16.9 / 21.7 / 44.9 | 486055, 455092 / 4006, 3771 | 265 |
| C4-multi-gpu-vdsp-f2 | 0.75 [0.74, 0.77] / 0.89 [0.88, 0.90] / 1.32 [1.29, 1.34] / 1.63 [1.58, 1.70] | 40.1 / 48.8 / 82.5 / 122.9 | 18439, 17882 / 125, 111 | 204 |
| C3-multi-gpu-vdsp-f2 | 0.76 [0.75, 0.77] / 0.89 [0.89, 0.90] / 1.29 [1.28, 1.32] / 1.62 [1.55, 1.71] | 40.3 / 49.2 / 81.2 / 122.9 | 13951, 14515 / 126, 158 | 205 |
| C6s8-multi-gpu-vdsp-f2 | 5.24 [5.12, 5.48] / 3.65 [3.56, 3.77] / 4.80 [4.57, 4.96] / 5.99 [5.41, 6.19] | 323.6 / 233.3 / 362.5 / 544.0 | 1282749, 1288002 / 828, 822 | 1388 |
| C6s8-multi-vdsp-f2-anchor-wp7 | 0.34 [0.32, 0.35] / 0.39 [0.39, 0.41] / 0.52 [0.50, 0.54] / 0.77 [0.77, 0.79] | 14.3 / 15.6 / 19.7 / 40.1 | 388073, 367637 / 1526, 1521 | 160 |

| arm | WER % | tokens = mp2 FP32 reference | preprocess / decode ms (15 s) | physical calls (64 clips, 1 call each) | C0 encoder post-purge load ms |
|---|---|---|---|---|---|
| C4-ane-multi-vdsp-f2 | 2.71 | 62/64 | 1.8 / 33.4 | native_joint 2135, native_predict 1955 | 27650, 26645 |
| C3-ane-multi-vdsp-f2 | 2.71 | 63/64 | 1.8 / 36.1 | native_joint 2135, native_predict 1954 | 27488, 26304 |
| C6s8-ane-multi-vdsp-f2 | 2.71 | 62/64 | 1.8 / 34.2 | native_joint 2135, native_predict 1955 | 28694, 26566 |
| C4-multi-gpu-vdsp-f2 | 2.71 | 64/64 | 1.8 / 36.2 | native_joint 2141, native_predict 1957 | 37829, 28848 |
| C3-multi-gpu-vdsp-f2 | 2.71 | 64/64 | 1.8 / 33.4 | native_joint 2141, native_predict 1957 | 29265, 29338 |
| C6s8-multi-gpu-vdsp-f2 | 2.71 | 64/64 | 1.9 / 36.5 | native_joint 2141, native_predict 1957 | 29171, 29487 |
| C6s8-multi-vdsp-f2-anchor-wp7 | 2.71 | 62/64 | 1.8 / 34.0 | native_joint 2136, native_predict 1955 | 28010, 26888 |

Full record: SHA-256 67ee60e690cf3d25e1aac26e13eca3a595949d170557c0def05784484536f9f3; 2692 bytes; 504c07c0cdecde864a45e1d0612f59d24896f914/wp7/sweep_table.md
