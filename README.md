# GPU Bug Reproductions

Bug reproductions for data race bugs found in TileLang, Triton, and TVM.
Verified using GPUVerify (2018-03-22) on NVIDIA GeForce MX450 (sm75), CUDA 12.3, WSL2 Ubuntu 22.04.

## Structure
```
gpu-bug-reproductions/
├── tilelang/   (4 issues)
├── triton/     (6 issues)
└── tvm/        (4 issues)
```

## Tool

[GPUVerify](https://github.com/mc-imperial/gpuverify) — a static analyser for verifying race and divergence freedom of CUDA GPU kernels. Where Triton or TVM do not generate `.cu` files directly, kernels are manually translated from Triton IR / TVM TIR for analysis.

---

## Results Summary

### TileLang

| Issue | Bug Description | Reproduced? | GPUVerify | Classification |
|-------|----------------|-------------|-----------|----------------|
| #1257 | Missing `__syncthreads()` after AtomicAdd | ✅ | RACE DETECTED | ✅ True Positive |
| #666  | Shared memory clear before pipelined loops (H100-specific) | ✅ | Verified | ❌ False Negative |
| #1671 | Python `and`/`or` on TVM Expr (compile-time crash) | ❌ version unavailable | N/A | ⚪ Not Applicable |
| #96   | Race in pipelined matmul (shared memory reuse) | ✅ | RACE DETECTED | ✅ True Positive |

### Triton

| Issue | Bug Description | Reproduced? | GPUVerify | Classification |
|-------|----------------|-------------|-----------|----------------|
| #4233 | `scatter_add` WAW/RAW race (non-atomic RMW) | ✅ | RACE DETECTED (2 errors) | ✅ True Positive |
| #4362 | `tl.associative_scan` wrong results with `reverse=True` | ✅ | Verified | ❌ False Negative |
| #4736 | `tl.min` butterfly shuffle WAW/RAW race | ✅ compute-sanitizer | RACE DETECTED (4 errors) | ✅ True Positive |
| #7264 | `tl.sum` butterfly shuffle WAW race | ✅ compute-sanitizer | RACE DETECTED (2 errors) | ✅ True Positive |
| #7402 | `tl.atomic_add` layout mismatch WAW race | ✅ | RACE DETECTED (1 error) | ✅ True Positive |
| #8311 | `warp_specialize` missing producer-consumer barrier | ✅ TTGIR (sm90+ for runtime) | RACE DETECTED (2 errors) | ✅ True Positive |

### TVM

| Issue | Bug Description | Reproduced? | GPUVerify | Classification |
|-------|----------------|-------------|-----------|----------------|
| #7246  | `call_packed` race under parallel schedule (CPU) | ❌ version unavailable | N/A | ⚪ Not Applicable |
| #10210 | Parallel reduction axis WAW race (CPU) | ✅ max error 31.21 | N/A | ⚪ Not Applicable |
| #17072 | CSE pass static cache race (needs 50+ cores) | ❌ hardware insufficient | N/A | ⚪ Not Applicable |
| #17439 | `ThreadSync` before `MergeSharedMemory` — missing barrier | ✅ TIR + GPUVerify | RACE DETECTED (1 error) | ✅ True Positive |

---

## Overall Statistics

| Framework | True Positive | False Negative | Not Applicable | Total |
|-----------|--------------|----------------|----------------|-------|
| TileLang  | 2 | 1 | 1 | 4 |
| Triton    | 5 | 1 | 0 | 6 |
| TVM       | 1 | 0 | 3 | 4 |
| **Total** | **8** | **2** | **4** | **14** |

## Key Findings

- GPUVerify successfully detected races in **8 out of 10 applicable cases**.
- **2 False Negatives**: TileLang #666 (H100-specific async pipeline race beyond GPUVerify's scope) and Triton #4362 (algorithmic ordering bug, not a synchronisation race).
- **4 Not Applicable**: 3 TVM issues are CPU-level races; TileLang #1671 is a compile-time crash with no kernel generated.
- Where Triton/TVM did not generate `.cu` files, kernels were manually translated from Triton IR / TVM TIR following mentor guidance.