# Issue #7246 — Race Condition in `tvm.tir.call_packed()` Under Parallel Schedule

## Source
- **GitHub Issue:** https://github.com/apache/tvm/issues/7246
- **Reproduced on:** apache-tvm 0.11.1, Python 3.10, Ubuntu/WSL2
- **Status:** Closed

## Approach

`apache-tvm==0.8.0` (the buggy version) is not available as a pip wheel —
the oldest available release is `0.9.0`. TIR inspection approach is used to confirm the fix.

## What is the Bug?

`LowerTVMBuiltin` allocates the packed-func argument stack —
`tvm_stack_alloca` for `stack_value` and `stack_tcode` — **once at the
top of the generated function**, outside any loop. When the enclosing loop
is marked `parallel()`, all OS threads share **the same stack memory**.
Each thread reads and writes the same `stack_value` / `stack_tcode` arrays
concurrently — a WAW race:

```
// Buggy TIR (TVM <0.9.0)
stack_value = tvm_stack_alloca("arg_value", 8)   // <- shared by all threads
stack_tcode = tvm_stack_alloca("arg_tcode", 8)   // <- shared by all threads

parallel for (xo, 0, 2) {
  for (yo, 0, 2) {
    // Thread 0 and Thread 1 both write stack_value[0] = &A_tile, etc.
    tvm_call_packed_lowered("tvm.contrib.cblas.matmul",
                            stack_value, stack_tcode, 0, 5)
  }
}
```

## Requirements

- Python 3.9 – 3.11
- `apache-tvm==0.11.1`
- No GPU or BLAS library required

## Setup

```bash
conda activate tvm-bugs    # apache-tvm 0.11.1
```

## How to Run

```bash
python reproduce.py
```

## Root Cause

`src/tir/transforms/lower_tvm_builtin.cc` — the `LowerTVMBuiltin` pass
lifted all `tvm_stack_alloca` nodes to the outermost scope (function body)
regardless of whether they were inside a parallel loop.

---

## Static Analysis Results

### GPUVerify

| Property | Value |
|----------|-------|
| Result | **NOT APPLICABLE** |
| Classification | ⚪ Not Applicable |
| Reason | CPU-level race in TVM's parallel thread pool — no CUDA kernel involved. Buggy version (TVM 0.8.0) not available as pip wheel. |

### Faial

| Property | Value |
|----------|-------|
| Result | **NOT APPLICABLE** |
| Classification | ⚪ Not Applicable |
| Reason | Faial analyzes CUDA GPU kernels only. This is a CPU thread race in TVM's runtime (`tvm_stack_alloca` shared across OS threads). No CUDA kernel is generated for this bug. |

---

## Tool Comparison Summary

| Tool | Result | Classification | Notes |
|------|--------|----------------|-------|
| GPUVerify | N/A | ⚪ Not Applicable | CPU race — no CUDA kernel |
| Faial | N/A | ⚪ Not Applicable | CPU race — no CUDA kernel |