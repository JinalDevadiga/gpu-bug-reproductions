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
opam install dune alcotest ounit -y
git clone https://gitlab.com/umb-svl/faial.git
cd faial
opam install . --deps-only -y
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
| #1257 | Missing `__syncthreads()` after AtomicAdd | ✅ | RACE DETECTED ✅ TP | RACE DETECTED ✅ TP | Both agree; Faial used bit-vector logic for XOR index |
| #1671 | Python `and`/`or` on TVM Expr (compile-time crash) | ❌ version unavailable | N/A ⚪ | DRF ⚪ | No buggy kernel generated — not applicable |

### Triton

| Issue | Bug Description | Reproduced? | GPUVerify | Faial | Notes |
|-------|----------------|-------------|-----------|-------|-------|
| #4233 | `scatter_add` WAW/RAW race (non-atomic RMW) | ✅ | RACE DETECTED ✅ TP | RACE DETECTED ✅ TP | Faial flagged CIDD (data-dependent index) |
| #4362 | `tl.associative_scan` wrong results with `reverse=True` | ✅ | Verified ❌ FN | DRF ❌ FN | Algorithmic bug — not a race, both tools correctly report DRF |
| #4736 | `tl.min` butterfly shuffle WAW/RAW race | ✅ compute-sanitizer | RACE DETECTED ✅ TP | RACE DETECTED ✅ TP | Faial 2 races matches compute-sanitizer count |
| #7264 | `tl.sum` butterfly shuffle WAW race | ✅ compute-sanitizer | RACE DETECTED ✅ TP | RACE DETECTED ✅ TP | Faial 1 race, GPUVerify 2 errors (counting directions) |
| #7402 | `tl.atomic_add` layout mismatch WAW race | ✅ | RACE DETECTED ✅ TP | DRF ❌ FN | Faial treats `write_index` as unconstrained symbolic variable |
| #8311 | `warp_specialize` missing producer-consumer barrier | ✅ TTGIR (sm90+ for runtime) | RACE DETECTED ✅ TP | RACE DETECTED ✅ TP | Exact match — both find 2 races |

### TVM

| Issue | Bug Description | Reproduced? | GPUVerify | Faial | Notes |
|-------|----------------|-------------|-----------|-------|-------|
| #7246  | `call_packed` race under parallel schedule (CPU) | ❌ version unavailable | N/A ⚪ | N/A ⚪ | CPU race — not applicable |
| #10210 | Parallel reduction axis WAW race (CPU) | ✅ max error 31.21 | N/A ⚪ | N/A ⚪ | CPU race — not applicable |
| #17072 | CSE pass static cache race (needs 50+ cores) | ❌ hardware insufficient | N/A ⚪ | N/A ⚪ | CPU race — not applicable |
| #17439 | `ThreadSync` before `MergeSharedMemory` — missing barrier | ✅ TIR + GPUVerify | RACE DETECTED ✅ TP | RACE DETECTED ✅ TP | Faial found 12 races vs GPUVerify 1 error |

---

## GPUVerify Statistics

| Framework | True Positive | False Negative | Not Applicable | Total |
|-----------|--------------|----------------|----------------|-------|
| TileLang  | 2 | 1 | 1 | 4 |
| Triton    | 5 | 1 | 0 | 6 |
| TVM       | 1 | 0 | 3 | 4 |
| **Total** | **8** | **2** | **4** | **14** |

## Faial Statistics

| Framework | True Positive | False Negative | Not Applicable | Total |
|-----------|--------------|----------------|----------------|-------|
| TileLang  | 1 | 2 | 1 | 4 |
| Triton    | 4 | 2 | 0 | 6 |
| TVM       | 1 | 0 | 3 | 4 |
| **Total** | **6** | **4** | **4** | **14** |

---

## Key Findings

### GPUVerify
- Successfully detected races in **8 out of 10 applicable cases** (80%).
- **2 False Negatives**: TileLang #666 (H100-specific async pipeline race) and Triton #4362 (algorithmic ordering bug, not a race).
- **4 Not Applicable**: 3 TVM issues are CPU-level races; TileLang #1671 is a compile-time crash.
- Where Triton/TVM did not generate `.cu` files, kernels were manually translated from Triton IR / TVM TIR following mentor guidance.

### Faial
- Successfully detected races in **6 out of 10 applicable cases** (60%).
- **4 False Negatives**:
  - **TileLang #96**: Cross-iteration WAR hazard — Faial's barrier-phase analysis reasons within single loop iterations only, cannot detect inter-iteration pipeline slot reuse (`(ko+2)%3`).
  - **TileLang #666**: Hardware async pipeline race (H100-specific) — beyond the scope of any static thread-level race detector.
  - **Triton #4362**: Algorithmic ordering bug — not a data race at all; both tools correctly report DRF.
  - **Triton #7402**: WAW race from `atomic_add` layout mismatch — Faial treats `write_index` as an unconstrained symbolic variable and cannot derive the runtime constraint that forces all non-zero threads to index 0.
- **Notable strengths**:
  - **TileLang #1257**: Automatically switched to bit-vector arithmetic to handle XOR index (`threadIdx.x ^ 32`).
  - **Triton #4233**: Correctly flagged a data-dependent index race (CIDD classification) with an appropriate warning.
  - **TVM #17439**: Found 12 races across all loop iteration combinations — more comprehensive than GPUVerify's single representative error.
  - **Triton #4736 / #8311**: Exact match with GPUVerify and compute-sanitizer race counts.