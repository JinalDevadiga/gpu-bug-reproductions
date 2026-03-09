# Issue #7246 — Race Condition in `tvm.tir.call_packed()` Under Parallel Schedule

## Source
- **GitHub Issue:** https://github.com/apache/tvm/issues/7246
- **Reproduced on:** apache-tvm 0.11.1, Python 3.10, Ubuntu/WSL2
- **Status:** Closed

## Approach

`apache-tvm==0.8.0` (the buggy version) is not available as a pip wheel —
the oldest available release is `0.9.0`. TIR inspection approach used —
same as Triton #8311.

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
    // Last writer wins -> wrong matmul args -> wrong result or crash
    tvm_call_packed_lowered("tvm.contrib.cblas.matmul",
                            stack_value, stack_tcode, 0, 5)
  }
}
```

## Observed TIR on TVM 0.11.1

TVM 0.11.1 eliminates the stack allocation entirely, inlining all arguments
as `tvm_stack_make_array` calls constructed inside the parallel loop:

```
for (i.outer: int32, 0, 2) "parallel" {
  for (j.outer: int32, 0, 2) {
    @tir.tvm_call_packed("tvm.contrib.cblas.matmul",
      @tir.tvm_stack_make_array(A_2, ...),   // <- inline, thread-safe
      @tir.tvm_stack_make_array(B_2, ...),   // <- inline, thread-safe
      ...)
  }
}
```

No `tvm_stack_alloca` appears — confirming the buggy code path is no longer
generated on this version.

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