# Issue #1671 — Python `and`/`or`/`not` on TVM Expr Inside TileLang Kernel

## Source
- **GitHub Issue:** https://github.com/tile-ai/tilelang/issues/1671
- **Repo:** TileLang
- **Reported on:** v0.1.7.post2
- **Status:** Open

---

## What is the Bug?

Running `examples/dsa_sparse_finetune/indexer_topk_reducesum.py` on
TileLang v0.1.7.post2 raises this error at compile time:
```
ValueError: Cannot use and / or / not operator to Expr,
            hint: use tvm.tir.all / tvm.tir.any instead
```

The cause is Python's `and`/`or`/`not` operators being used on TVM
symbolic expressions inside a `@T.prim_func` kernel. Python calls
`__bool__()` on both sides of `and`/`or` to get an immediate True/False,
but TVM Expr objects block this because their values are not known until
the kernel is compiled and run on the GPU.

---

## Why Does the Error Mention `tvm.tir`?

TileLang is built on **TVM** as its compiler backend. TVM ships
**inside** the TileLang package — it is not a separate dependency.
When you install `tilelang`, TVM comes with it. The `Expr` type and
the `ValueError` are part of TVM's internals that TileLang is built on.

---

## Root Cause

Inside a `@T.prim_func` kernel, variables like loop indices are
**TVM Expr objects**, not plain Python integers. Their values are only
resolved at kernel compile/run time on the GPU.

Python's `and`/`or`/`not` keywords work by calling `__bool__()` on their
operands to evaluate them as True/False immediately. TVM's `Expr.__bool__`
is intentionally blocked and raises `ValueError`.

This only happens with **symbolic expressions** — for example:
```python
row = bx * BLOCK + tx        # TVM Expr (symbolic)

if row < 128 and j < 64:     # ← crashes: 'row < 128' is a TVM Expr
    B[row, j] = A[row, j] * 2.0
```

---

## Requirements

- Python 3.10+
- **CUDA 13.x** (required by tilelang==0.1.7.post2 — see note below)
- Any NVIDIA GPU
- `tilelang==0.1.7.post2`
- `torch`

---

## How to Run
```bash
pip install tilelang==0.1.7.post2
python reproduce.py
```

---

## Note on CUDA Version Requirement

The v0.1.7.post2 wheel was built against **CUDA 13**. If your system has
CUDA 12, the import will fail with:
```
OSError: libcudart.so.13: cannot open shared object file: No such file or directory
```

This is an environment constraint, not part of the bug itself. The bug
is in the Python code pattern — the `and` operator used on a TVM Expr
inside a kernel. The reproduce.py script shows this pattern clearly and
documents the exact error that v0.1.7.post2 raises.

---

## Expected Output on v0.1.7.post2
```
ValueError: Cannot use and / or / not operator to Expr,
            hint: use tvm.tir.all / tvm.tir.any instead
```

## Expected Output on v0.1.8+

The bug is not triggered on v0.1.8. The script compiles successfully
and prints the generated CUDA source:
```c
extern "C" __global__ void __launch_bounds__(64, 1) kernel_kernel(...) {
  for (int j = 0; j < 64; ++j) {
    B[...] = (A[...] * 2.0);
  }
}
```