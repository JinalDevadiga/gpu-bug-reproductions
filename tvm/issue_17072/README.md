# Issue #17072 — Race Condition in TIR `ComputationCache` (CSE Pass)

## Source
- **GitHub Issue:** https://github.com/apache/tvm/issues/17072
- **Reported by:** @guillon, Jun 7 2024
- **Reproduced on:** apache-tvm 0.11.1, Python 3.10, Ubuntu/WSL2
- **Status:** Closed (Feb 8, 2025)

## Approach

This bug requires a high core-count machine to trigger reliably at runtime
(reported on a 52-core Intel Xeon Gold 6230R). It is not reproducible at
runtime on low core-count hardware (e.g. MX450 laptop). Source inspection
approach used.

## What is the Bug?

The CSE (Common Subexpression Elimination) TIR pass uses a `static` cache
declared in `src/tir/transforms/common_subexpr_elim_tools.h#L115`:

```cpp
// static cache shared across ALL threads — no synchronisation
static ComputationCache cache_;
```

When multiple Python threads each call `tvm.build()` concurrently, they
all enter the CSE pass and race on read-modify-write operations to this
single shared `cache_`:

- Thread A calls `cache_.find(expr)` (reading)
- Thread B calls `cache_.insert(expr, ...)` simultaneously (writing)
- HashTable is corrupted → **segmentation fault**

## Symptom

```
Segmentation fault
```

## Requirements

- Python 3.9 – 3.11
- `apache-tvm==0.11.1`
- 50+ core CPU for reliable runtime reproduction
- No GPU required

## Setup

```bash
conda activate tvm-bugs    # apache-tvm 0.11.1
```

## How to Run

```bash
python reproduce.py
python reproduce.py 100 100000   # high load — more likely to trigger
```

## Root Cause

`src/tir/transforms/common_subexpr_elim_tools.h#L115` — the
`ComputationCache` is declared `static`, making it a single instance
shared across all threads. The CSE pass was not designed for concurrent use.

---

## Static Analysis Results

### GPUVerify

| Property | Value |
|----------|-------|
| Result | **NOT APPLICABLE** |
| Classification | ⚪ Not Applicable |
| Reason | C++ compiler-level race on a static cache — no CUDA kernel involved. Requires 50+ cores to trigger reliably — not reproducible on this hardware. |

### Faial

| Property | Value |
|----------|-------|
| Result | **NOT APPLICABLE** |
| Classification | ⚪ Not Applicable |
| Reason | Faial analyzes CUDA GPU kernels only. This is a C++ thread race on a static `ComputationCache` inside TVM's compiler pipeline. No CUDA kernel is generated or involved. |

### Weft

| Property | Value |
|----------|-------|
| Result | **NOT APPLICABLE** |
| Classification | ⚪ Not Applicable |
| Reason | Weft analyzes CUDA GPU kernels via PTX files only. This is a C++ compiler-internal race on a static cache — no CUDA kernel or PTX file is generated. Requires 50+ cores to trigger reliably at runtime. |

---

## Tool Comparison Summary

| Tool | Result | Classification | Notes |
|------|--------|----------------|-------|
| GPUVerify | N/A | ⚪ Not Applicable | C++ compiler race — no CUDA kernel |
| Faial | N/A | ⚪ Not Applicable | C++ compiler race — no CUDA kernel |
| Weft | N/A | ⚪ Not Applicable | C++ compiler race — no PTX file to analyze |