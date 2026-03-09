"""
Issue #1671 - Python 'and'/'or'/'not' operators on TVM Expr inside TileLang
GitHub Issue : https://github.com/tile-ai/tilelang/issues/1671
Reported on  : TileLang v0.1.7.post2
Status       : Open

HOW TO REPRODUCE:
    pip install tilelang==0.1.7.post2  (from GitHub releases, not PyPI)
    python reproduce.py

On v0.1.7.post2 this script crashes with:
    ValueError: Cannot use and / or / not operator to Expr,
                hint: use tvm.tir.all / tvm.tir.any instead

On v0.1.8+ the bug is no longer triggered (silently fixed).
"""

import tilelang
import tilelang.language as T

print(f"TileLang version: {tilelang.__version__}")
print()

BLOCK = 64

# Inside @T.prim_func, loop variables like `row` are TVM Expr objects —
# symbolic values unknown at Python parse time. Python's `and` keyword
# calls __bool__() on them to get an immediate True/False, which TVM
# deliberately blocks because a symbolic expression has no concrete bool
# value yet. On v0.1.7.post2 this raises:
#
#   ValueError: Cannot use and / or / not operator to Expr,
#               hint: use tvm.tir.all / tvm.tir.any instead

@T.prim_func
def buggy_kernel(
    A: T.Tensor((128, 64), "float32"),
    B: T.Tensor((128, 64), "float32"),
):
    with T.Kernel(T.ceildiv(128, BLOCK), threads=BLOCK) as (bx,):
        tx = T.get_thread_binding(0)
        row = bx * BLOCK + tx        # TVM Expr, not a plain Python int
        for j in T.serial(64):
            if row < 128 and j < 64: # BUG: Python 'and' on TVM Expr
                B[row, j] = A[row, j] * 2.0

compiled = tilelang.compile(buggy_kernel, out_idx=[1], target="cuda")
print("Generated CUDA source:")
print(compiled.get_kernel_source())