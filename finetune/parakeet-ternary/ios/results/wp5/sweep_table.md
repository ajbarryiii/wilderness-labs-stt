Sweep wp5-20261003. Informational (shared Mac M1 Pro, macOS 27; no claims). Loads: 'post-purge load' = the first load after this binary's Core ML cache directory was purged; 'subsequent fresh-process load' = a new process loading the arm afterwards. Whether Core ML specialized or reused a cache in either is not established (no Instruments cache events).

| arm | total ratio vs C0 (2 / 4 / 8 / 15 s) [95% CI] | encoder ms (2 / 4 / 8 / 15 s) | load post-purge / subsequent fresh process (enc ms) | footprint MB |
|---|---|---|---|---|
| C1-multi-vdsp-f2 | 0.75 [0.72, 0.76] / 0.86 [0.85, 0.87] / 1.25 [1.24, 1.26] / 1.55 [1.51, 1.61] | 39.4 / 46.6 / 77.8 / 114.5 | 9835 / 1052 | 145 |
| C3-multi-vdsp-f2 | 0.37 [0.36, 0.39] / 0.42 [0.41, 0.44] / 0.54 [0.53, 0.55] / 0.85 [0.83, 0.86] | 16.1 / 17.6 / 21.3 / 45.1 | 78664 / 1255 | 148 |
| C4-multi-vdsp-f2 | 0.38 [0.36, 0.39] / 0.44 [0.42, 0.46] / 0.55 [0.53, 0.55] / 0.83 [0.81, 0.85] | 16.6 / 18.2 / 22.0 / 44.3 | 76157 / 1369 | 153 |
| C6s2-multi-vdsp-f2 | 0.37 [0.36, 0.38] / 0.43 [0.43, 0.45] / 0.72 [0.71, 0.73] / 0.96 [0.95, 0.97] | 16.4 / 18.5 / 35.0 / 56.1 | 492850 / 1611 | 157 |
| C6s4-multi-vdsp-f2 | 0.34 [0.33, 0.36] / 0.41 [0.39, 0.42] / 0.55 [0.55, 0.57] / 0.81 [0.81, 0.82] | 14.4 / 16.2 / 22.1 / 42.3 | 503325 / 1521 | 157 |
| C6s8-multi-vdsp-f2 | 0.33 [0.32, 0.35] / 0.40 [0.38, 0.41] / 0.52 [0.51, 0.55] / 0.79 [0.78, 0.80] | 14.0 / 15.4 / 19.6 / 40.2 | 387655 / 1542 | 157 |
| C6d4-multi-vdsp-f2 | 0.36 [0.35, 0.38] / 0.42 [0.42, 0.44] / 0.55 [0.54, 0.58] / 0.84 [0.83, 0.86] | 16.1 / 17.7 / 22.1 / 43.8 | 145026 / 1318 | 148 |
| C6d8-multi-vdsp-f2 | 0.37 [0.36, 0.38] / 0.43 [0.41, 0.44] / 0.55 [0.53, 0.57] / 0.84 [0.83, 0.85] | 16.1 / 17.6 / 21.5 / 44.1 | 143466 / 1291 | 148 |
| C7-multi-vdsp-f2 | 0.37 [0.37, 0.39] / 0.44 [0.43, 0.46] / 0.57 [0.55, 0.58] / 0.84 [0.83, 0.84] | 16.6 / 18.5 / 22.8 / 43.7 | 89038 / 1380 | 154 |
| C8-multi-vdsp-f2 | 0.39 [0.38, 0.41] / 0.45 [0.44, 0.47] / 0.61 [0.59, 0.62] / 0.92 [0.91, 0.93] | 17.4 / 19.4 / 26.2 / 53.0 | 410627 / 1651 | 167 |
| C4-fixed-vdsp-f2 | 0.82 [0.82, 0.83] / 0.82 [0.82, 0.82] / 0.83 [0.83, 0.84] / 0.83 [0.83, 0.84] | 44.2 / 44.2 / 44.3 / 44.3 | 20041 / 107 | 168 |
| G0-fixed-vdsp-f2 | 0.83 [0.82, 0.83] / 0.83 [0.83, 0.84] / 0.85 [0.84, 0.85] / 0.84 [0.84, 0.85] | 44.6 / 44.6 / 44.6 / 44.6 | 28572 / 108 | 166 |
| C4-multi-c0pre-f0 | 0.65 [0.63, 0.68] / 0.73 [0.72, 0.76] / 0.91 [0.89, 0.94] / 1.22 [1.21, 1.24] | 16.5 / 18.1 / 22.5 / 44.9 | 77989 / 1365 | 85 |
| C4-multi-c0pre-f1 | 0.68 [0.66, 0.70] / 0.74 [0.72, 0.77] / 0.91 [0.88, 0.95] / 1.22 [1.20, 1.26] | 16.5 / 18.2 / 22.5 / 44.9 | 76138 / 1336 | 83 |
| C4-multi-c0pre-f2 | 0.59 [0.58, 0.60] / 0.64 [0.63, 0.65] / 0.72 [0.71, 0.74] / 1.00 [0.98, 1.00] | 16.5 / 18.2 / 22.0 / 44.3 | 76287 / 1351 | 155 |
| C4-multi-vdsp-f0 | 0.44 [0.41, 0.46] / 0.53 [0.50, 0.56] / 0.72 [0.71, 0.76] / 1.08 [1.05, 1.11] | 16.5 / 18.1 / 21.9 / 44.6 | 75658 / 1352 | 73 |
| C4-multi-vdsp-f1 | 0.46 [0.44, 0.48] / 0.53 [0.51, 0.57] / 0.72 [0.70, 0.77] / 1.07 [1.04, 1.11] | 16.4 / 18.0 / 21.8 / 44.5 | 76127 / 1333 | 70 |

| arm | WER % | tokens = mp2 FP32 reference | preprocess / decode ms (15 s) | physical calls (64 clips, 1 call each) | C0 encoder post-purge load ms |
|---|---|---|---|---|---|
| C1-multi-vdsp-f2 | 2.71 | 63/64 | 1.8 / 35.3 | native_joint 2142, native_predict 1958 | 29162 |
| C3-multi-vdsp-f2 | 2.71 | 62/64 | 1.8 / 35.1 | native_joint 2135, native_predict 1955 | 28997 |
| C4-multi-vdsp-f2 | 2.71 | 62/64 | 1.8 / 35.0 | native_joint 2135, native_predict 1955 | 29509 |
| C6s2-multi-vdsp-f2 | 2.71 | 62/64 | 1.8 / 35.3 | native_joint 2136, native_predict 1955 | 29565 |
| C6s4-multi-vdsp-f2 | 2.71 | 62/64 | 1.8 / 35.5 | native_joint 2136, native_predict 1955 | 29129 |
| C6s8-multi-vdsp-f2 | 2.71 | 62/64 | 1.9 / 35.6 | native_joint 2136, native_predict 1955 | 30038 |
| C6d4-multi-vdsp-f2 | 2.71 | 62/64 | 1.8 / 35.4 | native_joint 2136, native_predict 1955 | 30856 |
| C6d8-multi-vdsp-f2 | 2.71 | 62/64 | 1.8 / 36.0 | native_joint 2136, native_predict 1955 | 29361 |
| C7-multi-vdsp-f2 | 2.71 | 62/64 | 1.8 / 35.4 | native_joint 2135, native_predict 1955 | 35416 |
| C8-multi-vdsp-f2 | 2.71 | 62/64 | 1.8 / 35.0 | native_joint 2136, native_predict 1955 | 30488 |
| C4-fixed-vdsp-f2 | 2.71 | 63/64 | 1.8 / 34.4 | native_joint 2135, native_predict 1954 | 30293 |
| G0-fixed-vdsp-f2 | 1.67 | -/- | 1.8 / 37.0 | native_joint 2152, native_predict 1955 | 29801 |
| C4-multi-c0pre-f0 | 2.71 | 62/64 | 16.4 / 58.3 | decoder_model 1954, joint_model 2129 | 29765 |
| C4-multi-c0pre-f1 | 2.71 | 62/64 | 16.0 / 59.8 | fused_model 2129 | 29145 |
| C4-multi-c0pre-f2 | 2.71 | 62/64 | 16.3 / 35.7 | native_joint 2129, native_predict 1954 | 29030 |
| C4-multi-vdsp-f0 | 2.71 | 62/64 | 1.8 / 59.1 | decoder_model 1955, joint_model 2135 | 29205 |
| C4-multi-vdsp-f1 | 2.71 | 62/64 | 1.8 / 59.7 | fused_model 2135 | 29203 |

Full record: SHA-256 02548e1005e6b3b4c2d28545f14ebf4b7a3f1fe948a9e9e029de3df0b7de13a7; 4909 bytes; 504c07c0cdecde864a45e1d0612f59d24896f914/wp5/sweep_table.md
