"""
Reproduction of apache/tvm#10210:
  Silent wrong results when parallel() is applied to a reduction axis.

  Multiple threads write to the same output accumulator C[i][j]
  without atomics -> WAW race -> lost updates -> wrong result.
  TVM emits no warning during compilation.

Environment:
  conda activate tvm-bugs   (apache-tvm==0.11.1)
"""
import numpy as np
import tvm
from tvm import te

print(f"TVM version : {tvm.__version__}")

M, N, K = 64, 64, 128
np.random.seed(42)
A_np = np.random.randn(M, K).astype("float32")
B_np = np.random.randn(K, N).astype("float32")
ref  = A_np @ B_np
dev  = tvm.cpu()

def run_gemm(parallel_reduction: bool):
    A_t = te.placeholder((M, K), name="A")
    B_t = te.placeholder((K, N), name="B")
    k   = te.reduce_axis((0, K), name="k")
    C_t = te.compute(
        (M, N),
        lambda i, j: te.sum(A_t[i, k] * B_t[k, j], axis=k),
        name="C",
    )
    s = te.create_schedule(C_t.op)
    if parallel_reduction:
        s[C_t].parallel(C_t.op.reduce_axis[0])   # <- the bug trigger
    func = tvm.build(s, [A_t, B_t, C_t], target="llvm")
    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)
    func(A_nd, B_nd, C_nd)
    return C_nd.numpy()

# sequential baseline
C_seq   = run_gemm(parallel_reduction=False)
err_seq = np.max(np.abs(C_seq - ref))
print(f"\nSequential GEMM  — max error vs numpy: {err_seq:.2e}  "
      f"({'PASS' if err_seq < 1e-3 else 'FAIL'})")

# parallel reduction (buggy) — run multiple times, race is non-deterministic
RUNS   = 10
errors = [np.max(np.abs(run_gemm(parallel_reduction=True) - ref))
          for _ in range(RUNS)]
max_err = max(errors)
min_err = min(errors)
print(f"Parallel-reduction GEMM ({RUNS} runs) — "
      f"max error: {max_err:.4f},  min error: {min_err:.4f}")

if max_err > 1.0:
    print(
        f"\n-> BUG CONFIRMED: parallel reduction race detected.\n"
        f"   max_error={max_err:.4f} >> 1.0 tolerance.\n"
        f"   Multiple threads wrote to the same C[i][j] accumulator "
        f"without synchronisation -> lost updates -> wrong result.\n"
        f"   TVM emitted no warning during compilation."
    )
else:
    print(
        f"\n-> Bug not observed this run (max_error={max_err:.4f}).\n"
        f"   Race is non-deterministic — increase RUNS or tensor dimensions."
    )