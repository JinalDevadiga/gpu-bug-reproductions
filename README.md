# GPU Bug Reproductions

Bug reproductions for data race bugs found in TileLang, Triton, and TVM.
Verified using GPUVerify (2018-03-22) and Faial (gitlab.com/umb-svl/faial)
on NVIDIA GeForce MX450 (sm75), CUDA 12.3, WSL2 Ubuntu 22.04.

## Structure
```
gpu-bug-reproductions/
├── tilelang/   (4 issues)
├── triton/     (6 issues)
└── tvm/        (4 issues)
```

## Tools

### GPUVerify
[GPUVerify](https://github.com/mc-imperial/gpuverify) — a static analyser for verifying race and divergence freedom of CUDA GPU kernels. Where Triton or TVM do not generate `.cu` files directly, kernels are manually translated from Triton IR / TVM TIR for analysis.

### Faial
[Faial](https://gitlab.com/umb-svl/faial) — a static analyser for CUDA kernels based on memory access protocols and SMT solving (Z3). Provides sound, compositional data-race freedom verification. Developed by the Software Verification Lab at UMass Boston.

**To install Faial (Ubuntu 22.04 / WSL2):**
```bash
# 1. Install system dependencies
sudo apt install -y git curl build-essential pkg-config libzstd-dev m4 bubblewrap unzip

# 2. Install opam (OCaml package manager)
bash -c "sh <(curl -fsSL https://opam.ocaml.org/install.sh)"
opam init --disable-sandboxing -y   # --disable-sandboxing required for WSL
eval $(opam env)
echo 'eval $(opam env)' >> ~/.bashrc

# 3. Install dune and build Faial
opam install dune -y
git clone https://gitlab.com/umb-svl/faial.git
cd faial
opam install . --deps-only -y
opam install alcotest ounit -y
dune build && dune install

# 4. Install c-to-json (CUDA parser)
git clone https://gitlab.com/umb-svl/c-to-json.git
cd c-to-json && sudo apt install -y libzstd-dev cmake && make
sudo make install
sudo cp src/cu-to-json /usr/local/bin/cu-to-json
sudo chmod +x /usr/local/bin/cu-to-json

# 5. Run Faial
faial-drf --cu-to-json=/usr/local/bin/cu-to-json your_kernel.cu
```

---

## Results Summary

### TileLang

| Issue | Bug Description | Reproduced? | GPUVerify | Faial | Notes |
|-------|----------------|-------------|-----------|-------|-------|
| #96   | Race in pipelined matmul (shared memory reuse) | ✅ | RACE DETECTED ✅ TP | DRF ❌ FN | Faial cannot reason across loop iterations |
| #666  | Shared memory clear before pipelined loops (H100-specific) | ✅ | Verified ❌ FN | DRF ❌ FN | Hardware async pipeline race — beyond both tools' scope |
| #1257 | Missing `__syncthreads()` after AtomicAdd | ✅ | RACE DETECTED ✅ TP | RACE DETECTED ✅ TP | Both tools agree; Faial used bit-vector logic for XOR index |
| #1671 | Python `and`/`or` on TVM Expr (compile-time crash) | ❌ version unavailable | N/A ⚪ | DRF ⚪ | No buggy kernel generated — not applicable for either tool |

### Triton

| Issue | Bug Description | Reproduced? | GPUVerify | Faial | Notes |
|-------|----------------|-------------|-----------|-------|-------|
| #4233 | `scatter_add` WAW/RAW race (non-atomic RMW) | ✅ | RACE DETECTED ✅ TP | — | Faial pending |
| #4362 | `tl.associative_scan` wrong results with `reverse=True` | ✅ | Verified ❌ FN | — | Faial pending |
| #4736 | `tl.min` butterfly shuffle WAW/RAW race | ✅ compute-sanitizer | RACE DETECTED ✅ TP | — | Faial pending |
| #7264 | `tl.sum` butterfly shuffle WAW race | ✅ compute-sanitizer | RACE DETECTED ✅ TP | — | Faial pending |
| #7402 | `tl.atomic_add` layout mismatch WAW race | ✅ | RACE DETECTED ✅ TP | — | Faial pending |
| #8311 | `warp_specialize` missing producer-consumer barrier | ✅ TTGIR (sm90+ for runtime) | RACE DETECTED ✅ TP | — | Faial pending |

### TVM

| Issue | Bug Description | Reproduced? | GPUVerify | Faial | Notes |
|-------|----------------|-------------|-----------|-------|-------|
| #7246  | `call_packed` race under parallel schedule (CPU) | ❌ version unavailable | N/A ⚪ | N/A ⚪ | CPU race — not applicable |
| #10210 | Parallel reduction axis WAW race (CPU) | ✅ max error 31.21 | N/A ⚪ | N/A ⚪ | CPU race — not applicable |
| #17072 | CSE pass static cache race (needs 50+ cores) | ❌ hardware insufficient | N/A ⚪ | N/A ⚪ | CPU race — not applicable |
| #17439 | `ThreadSync` before `MergeSharedMemory` — missing barrier | ✅ TIR + GPUVerify | RACE DETECTED ✅ TP | — | Faial pending |

> **Note:** Faial analysis is complete for TileLang (4/4 issues). Triton and TVM analysis is pending.

---

## GPUVerify Statistics

| Framework | True Positive | False Negative | Not Applicable | Total |
|-----------|--------------|----------------|----------------|-------|
| TileLang  | 2 | 1 | 1 | 4 |
| Triton    | 5 | 1 | 0 | 6 |
| TVM       | 1 | 0 | 3 | 4 |
| **Total** | **8** | **2** | **4** | **14** |

## Faial Statistics (TileLang Complete, Others Pending)

| Framework | True Positive | False Negative | Not Applicable | Pending | Total |
|-----------|--------------|----------------|----------------|---------|-------|
| TileLang  | 1 | 2 | 1 | 0 | 4 |
| Triton    | 0 | 0 | 0 | 6 | 6 |
| TVM       | 0 | 0 | 3 | 1 | 4 |
| **Total** | **1** | **2** | **4** | **7** | **14** |

---

## Key Findings

### GPUVerify
- Successfully detected races in **8 out of 10 applicable cases**.
- **2 False Negatives**: TileLang #666 (H100-specific async pipeline race) and Triton #4362 (algorithmic ordering bug, not a synchronisation race).
- **4 Not Applicable**: 3 TVM issues are CPU-level races; TileLang #1671 is a compile-time crash.
- Where Triton/TVM did not generate `.cu` files, kernels were manually translated from Triton IR / TVM TIR following mentor guidance.

### Faial (TileLang Results)
- Faial detected **1 out of 3 applicable TileLang races** (True Positive rate: 33%).
- **✅ True Positive — TileLang #1257**: Correctly detected the missing barrier after `atomicAdd`. Faial automatically switched to bit-vector arithmetic to handle the XOR index expression (`threadIdx.x ^ 32`), demonstrating robustness with bitwise operations.
- **❌ False Negative — TileLang #96**: Missed the cross-iteration WAR hazard. Faial's barrier-phase analysis reasons within single loop iterations only and cannot detect races caused by shared memory slot reuse across iterations (`(ko+2)%3`). This is a known design boundary of Faial.
- **❌ False Negative — TileLang #666**: Missed the async pipeline race. This is a hardware-specific race on H100 caused by async pipeline overlap — beyond the scope of any static thread-level race detector, including both Faial and GPUVerify.
- **⚪ Not Applicable — TileLang #1671**: The bug is a compile-time crash; no GPU kernel is generated.