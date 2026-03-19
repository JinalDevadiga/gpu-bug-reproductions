# GPU Bug Reproductions

Bug reproductions for data race bugs found in TileLang, Triton, and TVM.
Verified using GPUVerify (2018-03-22), Faial (gitlab.com/umb-svl/faial),
and Weft (github.com/lightsighter/Weft)
on NVIDIA GeForce MX450 (sm75), CUDA 12.3, WSL2 Ubuntu 22.04.

## Structure
```
gpu-bug-reproductions/
├── tilelang/   (4 issues)
├── triton/     (6 issues)
└── tvm/        (4 issues)
```

---

## Tools

### GPUVerify
[GPUVerify](https://github.com/mc-imperial/gpuverify) — a static analyser
for verifying race and divergence freedom of CUDA GPU kernels. Where Triton
or TVM do not generate `.cu` files directly, kernels are manually translated
from Triton IR / TVM TIR for analysis.

**To install GPUVerify (Ubuntu / WSL2):**
```bash
# Download pre-built binary (2018-03-22 release)
wget https://github.com/mc-imperial/gpuverify/releases/download/2018-03-22/GPUVerifyLinux64.zip
unzip GPUVerifyLinux64.zip
export PATH=$PATH:$(pwd)/GPUVerify

# Run GPUVerify
python GPUVerify.py --cuda --blockDim=256 --gridDim=1 kernel.cu
```

---

### Faial
[Faial](https://gitlab.com/umb-svl/faial) — a static analyser for CUDA
kernels based on memory access protocols and SMT solving (Z3). Provides
sound, compositional data-race freedom verification. Developed by the
Software Verification Lab at UMass Boston (PLDI 2021).

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

# 4. Install c-to-json (CUDA parser for Faial)
git clone https://gitlab.com/umb-svl/c-to-json.git
cd c-to-json && sudo apt install -y libzstd-dev cmake && make
sudo make install
sudo cp src/cu-to-json /usr/local/bin/cu-to-json
sudo chmod +x /usr/local/bin/cu-to-json

# 5. Run Faial
faial-drf --cu-to-json=/usr/local/bin/cu-to-json your_kernel.cu
```

---

### Weft
[Weft](https://github.com/lightsighter/Weft) — a formal verification tool
for GPU kernels that checks shared memory race freedom, deadlock freedom,
and barrier recycling correctness using a happens-before analysis over PTX.
Developed at Stanford / NVIDIA Research (PLDI 2015).

**To install Weft (Ubuntu 22.04 / WSL2):**
```bash
# 1. Clone and build
git clone https://github.com/lightsighter/Weft.git
cd Weft/src
make

# 2. Apply compatibility patch for CUDA 12.x PTX
# (red.shared.add.u32 instruction added in newer CUDA versions)
sed -i 's/if (line.find("add.") != std::string::npos)/if (line.find("add.") != std::string::npos \&\& line.find("red.") == std::string::npos)/' instruction.cc
make

# 3. Compile your CUDA kernel to PTX
# Note: add __launch_bounds__(N) to your kernel signature first
nvcc -ptx your_kernel.cu -o your_kernel.ptx

# 4. Run Weft
./weft -f your_kernel.ptx -t 4 -i -d
```

**Important notes for Weft:**
- Weft reads **PTX files**, not `.cu` source directly
- Your kernel must have `__launch_bounds__(N)` or Weft cannot determine thread count
- Do **not** use `--generate-line-info` with nvcc — the newer debug format crashes Weft's parser
- Weft only analyzes **shared memory** — global memory races are invisible to it
- The `red.shared.add.u32` patch above is required for CUDA 12.x compatibility

---

## Results Summary

### TileLang

| Issue | Bug Description | Reproduced? | GPUVerify | Faial | Weft |
|-------|----------------|-------------|-----------|-------|------|
| #96   | Race in pipelined matmul (shared memory reuse) | ✅ | RACE DETECTED ✅ TP | DRF ❌ FN | No races ❌ FN |
| #666  | Shared memory clear before pipelined loops (H100-specific) | ✅ | Verified ❌ FN | DRF ❌ FN | No races ❌ FN |
| #1257 | Missing `__syncthreads()` after AtomicAdd | ✅ | RACE DETECTED ✅ TP | RACE DETECTED ✅ TP | RACES DETECTED ✅ TP |
| #1671 | Python `and`/`or` on TVM Expr (compile-time crash) | ❌ version unavailable | N/A ⚪ | DRF ⚪ | No races ⚪ |

### Triton

| Issue | Bug Description | Reproduced? | GPUVerify | Faial | Weft |
|-------|----------------|-------------|-----------|-------|------|
| #4233 | `scatter_add` WAW/RAW race (non-atomic RMW) | ✅ | RACE DETECTED ✅ TP | RACE DETECTED ✅ TP | No races ❌ FN |
| #4362 | `tl.associative_scan` wrong results with `reverse=True` | ✅ | Verified ❌ FN | DRF ❌ FN | No races ❌ FN |
| #4736 | `tl.min` butterfly shuffle WAW/RAW race | ✅ compute-sanitizer | RACE DETECTED ✅ TP | RACE DETECTED ✅ TP | No races ❌ FN |
| #7264 | `tl.sum` butterfly shuffle WAW race | ✅ compute-sanitizer | RACE DETECTED ✅ TP | RACE DETECTED ✅ TP | No races ❌ FN |
| #7402 | `tl.atomic_add` layout mismatch WAW race | ✅ | RACE DETECTED ✅ TP | DRF ❌ FN | No races ❌ FN |
| #8311 | `warp_specialize` missing producer-consumer barrier | ✅ TTGIR (sm90+ for runtime) | RACE DETECTED ✅ TP | RACE DETECTED ✅ TP | No races ❌ FN |

### TVM

| Issue | Bug Description | Reproduced? | GPUVerify | Faial | Weft |
|-------|----------------|-------------|-----------|-------|------|
| #7246  | `call_packed` race under parallel schedule (CPU) | ❌ version unavailable | N/A ⚪ | N/A ⚪ | N/A ⚪ |
| #10210 | Parallel reduction axis WAW race (CPU) | ✅ max error 31.21 | N/A ⚪ | N/A ⚪ | N/A ⚪ |
| #17072 | CSE pass static cache race (needs 50+ cores) | ❌ hardware insufficient | N/A ⚪ | N/A ⚪ | N/A ⚪ |
| #17439 | `ThreadSync` before `MergeSharedMemory` — missing barrier | ✅ TIR + GPUVerify | RACE DETECTED ✅ TP | RACE DETECTED ✅ TP | OOM ⚠️ |

---

## Statistics

### GPUVerify

| Framework | True Positive | False Negative | Not Applicable | Total |
|-----------|--------------|----------------|----------------|-------|
| TileLang  | 2 | 1 | 1 | 4 |
| Triton    | 5 | 1 | 0 | 6 |
| TVM       | 1 | 0 | 3 | 4 |
| **Total** | **8** | **2** | **4** | **14** |

### Faial

| Framework | True Positive | False Negative | Not Applicable | Total |
|-----------|--------------|----------------|----------------|-------|
| TileLang  | 1 | 2 | 1 | 4 |
| Triton    | 4 | 2 | 0 | 6 |
| TVM       | 1 | 0 | 3 | 4 |
| **Total** | **6** | **4** | **4** | **14** |

### Weft

| Framework | True Positive | False Negative | Not Applicable | Inconclusive | Total |
|-----------|--------------|----------------|----------------|--------------|-------|
| TileLang  | 1 | 2 | 1 | 0 | 4 |
| Triton    | 0 | 6 | 0 | 0 | 6 |
| TVM       | 0 | 0 | 3 | 1 | 4 |
| **Total** | **1** | **8** | **3** | **1** | **14** |

---

## Key Findings

### GPUVerify
- Successfully detected races in **8 out of 10 applicable cases** (80%).
- **2 False Negatives**: TileLang #666 (H100-specific async pipeline race) and Triton #4362 (algorithmic ordering bug, not a race).
- **4 Not Applicable**: 3 TVM issues are CPU-level races; TileLang #1671 is a compile-time crash.
- Where Triton/TVM did not generate `.cu` files, kernels were manually translated from Triton IR / TVM TIR.

### Faial
- Successfully detected races in **6 out of 10 applicable cases** (60%).
- **4 False Negatives**:
  - **TileLang #96**: Cross-iteration WAR hazard — Faial's barrier-phase analysis reasons within single loop iterations only.
  - **TileLang #666**: Hardware async pipeline race (H100-specific) — beyond any static thread-level tool's scope.
  - **Triton #4362**: Algorithmic ordering bug — not a data race at all.
  - **Triton #7402**: WAW race from `atomic_add` layout mismatch — `write_index` treated as unconstrained symbolic variable.
- **Notable strengths**: Automatic bit-vector arithmetic for XOR indices (#1257); CIDD classification for data-dependent races (#4233); 12 races found vs GPUVerify's 1 for TVM #17439.

### Weft
- Successfully detected races in **1 out of 10 applicable cases** (10%).
- **1 Inconclusive**: TVM #17439 — process killed (OOM) during emulation of 256-thread deeply nested matmul kernel. Demonstrates Weft's scalability limit on modern production kernels.
- **8 False Negatives** falling into four distinct categories:
  - **Cross-iteration address aliasing** (#96, #4736, #7264): Weft's happens-before analysis correctly reasons within each unrolled loop iteration but cannot detect races where shared memory slots alias across iterations. Affects software pipeline and butterfly reduction patterns.
  - **Global memory scope** (#4233, #7402): Weft only analyzes shared memory (`ld.shared`/`st.shared` in PTX). Races on global memory arrays are completely invisible — `Shared Memory Locations: 0` for both.
  - **Missing barrier — vacuously safe** (#8311): With `Dynamic Barrier Instances: 0`, Weft's barrier dependence graph is empty. It correctly finds no barrier violations but misses that the absence of any barrier between producer and consumer writes/reads is itself the bug. Weft was designed for named-barrier (`bar.sync`/`bar.arrive`) warp-specialized kernels, not standard `__syncthreads()` kernels with missing synchronization.
  - **Out-of-scope bugs** (#666, #4362): H100 async pipeline race and algorithmic ordering error — these are also missed by GPUVerify and Faial for the same reasons.
- **Weft-specific findings**:
  - Required a **one-line source patch** to handle `red.shared.add.u32` PTX instructions emitted by CUDA 12.3 (Weft was built in 2015 for PTX 4.x).
  - The `--generate-line-info` nvcc flag crashes Weft's parser on modern PTX debug format — PTX must be compiled without it.
  - Weft's exhaustive emulation approach works well for small, barrier-heavy kernels (e.g. #1257: 64 threads, 1 barrier, completed in 2ms) but hits memory limits on larger production kernels.

### Cross-Tool Comparison

| Bug Category | GPUVerify | Faial | Weft |
|---|---|---|---|
| Missing `__syncthreads()` (shared mem) | ✅ | ✅ | ✅ (if not cross-iteration) |
| Cross-iteration shared mem aliasing | ✅ | ❌ | ❌ |
| Global memory races | ✅ | ✅ | ❌ (shared mem only) |
| Missing barrier between warp groups | ✅ | ✅ | ❌ (vacuously safe) |
| Hardware async pipeline races (H100) | ❌ | ❌ | ❌ |
| Algorithmic ordering bugs | ❌ | ❌ | ❌ |
| Large nested-loop kernels | ✅ | ✅ | ⚠️ OOM risk |