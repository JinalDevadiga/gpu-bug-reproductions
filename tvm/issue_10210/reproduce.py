# """
# Reproduction of apache/tvm#10210:
#   Silent wrong results when parallel() is applied to a reduction axis.

#   Multiple threads write to the same output accumulator C[i][j]
#   without atomics -> WAW race -> lost updates -> wrong result.
#   TVM emits no warning during compilation.

# Environment:
#   conda activate tvm-bugs   (apache-tvm==0.11.1)
# """
# import numpy as np
# import tvm
# from tvm import te

# print(f"TVM version : {tvm.__version__}")

# M, N, K = 64, 64, 128
# np.random.seed(42)
# A_np = np.random.randn(M, K).astype("float32")
# B_np = np.random.randn(K, N).astype("float32")
# ref  = A_np @ B_np
# dev  = tvm.cpu()

# def run_gemm(parallel_reduction: bool):
#     A_t = te.placeholder((M, K), name="A")
#     B_t = te.placeholder((K, N), name="B")
#     k   = te.reduce_axis((0, K), name="k")
#     C_t = te.compute(
#         (M, N),
#         lambda i, j: te.sum(A_t[i, k] * B_t[k, j], axis=k),
#         name="C",
#     )
#     s = te.create_schedule(C_t.op)
#     if parallel_reduction:
#         s[C_t].parallel(C_t.op.reduce_axis[0])   # <- the bug trigger
#     # func = tvm.build(s, [A_t, B_t, C_t], target="llvm")
#     func = tvm.build(s, [A_t, B_t, C_t], target="cuda")

#     A_nd = tvm.nd.array(A_np, dev)
#     B_nd = tvm.nd.array(B_np, dev)
#     C_nd = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)
#     func(A_nd, B_nd, C_nd)
#     return C_nd.numpy()

# # sequential baseline
# C_seq   = run_gemm(parallel_reduction=False)
# err_seq = np.max(np.abs(C_seq - ref))
# print(f"\nSequential GEMM  — max error vs numpy: {err_seq:.2e}  "
#       f"({'PASS' if err_seq < 1e-3 else 'FAIL'})")

# # parallel reduction (buggy) — run multiple times, race is non-deterministic
# RUNS   = 10
# errors = [np.max(np.abs(run_gemm(parallel_reduction=True) - ref))
#           for _ in range(RUNS)]
# max_err = max(errors)
# min_err = min(errors)
# print(f"Parallel-reduction GEMM ({RUNS} runs) — "
#       f"max error: {max_err:.4f},  min error: {min_err:.4f}")

# if max_err > 1.0:
#     print(
#         f"\n-> BUG CONFIRMED: parallel reduction race detected.\n"
#         f"   max_error={max_err:.4f} >> 1.0 tolerance.\n"
#         f"   Multiple threads wrote to the same C[i][j] accumulator "
#         f"without synchronisation -> lost updates -> wrong result.\n"
#         f"   TVM emitted no warning during compilation."
#     )
# else:
#     print(
#         f"\n-> Bug not observed this run (max_error={max_err:.4f}).\n"
#         f"   Race is non-deterministic — increase RUNS or tensor dimensions."
#     )

"""
Reproduction of apache/tvm#10210:
  Tree reduction with shared memory is buggy when threadIdx.y is present.

  TVM's CUDA code generator incorrectly assumes the reduction fits within
  a single warp when blockDim.x is not a multiple of warp size (32).
  It skips __syncthreads() between reduction steps inside the
  if (threadIdx.x < N) blocks — causing a shared memory race condition.

  The fix is to add __syncthreads() between each reduction step.

Environment:
  conda activate tvm-bugs   (apache-tvm==0.11.1)
"""
import numpy as np
import tvm
import tvm.testing
from tvm.script import tir as T

print(f"TVM version : {tvm.__version__}")

# Exact reproduction from the GitHub issue
# d1=1, d2=16, d3=26 — blockDim.x=26 which is NOT a multiple of warp size (32)
# This is what triggers the bug — TVM assumes single-warp reduction but it is not
d1, d2, d3 = 1, 16, 26

@T.prim_func
def reduce(a: T.handle, b: T.handle, _d1: T.int32, _d2: T.int32, _d3: T.int32) -> None:
    A = T.match_buffer(a, [1, _d1, _d2, _d3])
    B = T.match_buffer(b, [1, _d1, _d2])
    for i, j, k, l in T.grid(1, _d1, _d2, _d3):
        with T.block("reduce"):
            vi, vj, vk, vl = T.axis.remap("SSSR", [i, j, k, l])
            with T.init():
                B[vi, vj, vk] = 0.
            B[vi, vj, vk] = B[vi, vj, vk] + A[vi, vj, vk, vl]

_, _, p_d1, p_d2, p_d3 = reduce.params
mod = reduce.specialize({p_d1: d1, p_d2: d2, p_d3: d3})

sch = tvm.tir.Schedule(mod)
blk = sch.get_block("reduce")
i, j, k, l = sch.get_loops(blk)

# Bind axes to GPU threads — this is the exact schedule from the issue
sch.bind(i, "blockIdx.x")
sch.bind(j, "threadIdx.z")
sch.bind(k, "threadIdx.y")
sch.bind(l, "threadIdx.x")   # <- blockDim.x = 26, NOT multiple of 32 -> bug!

f = tvm.build(sch.mod["main"], target="cuda")

print("\n=== Generated CUDA source (BUGGY) ===")
print(f.imported_modules[0].get_source())

print("\n=== IR Analysis ===")
src = f.imported_modules[0].get_source()

# Check for missing __syncthreads() inside reduction
if "threadIdx.x) < 8" in src:
    # Count __syncthreads() inside the reduction block
    lines = src.split('\n')
    in_reduction = False
    syncs_inside = 0
    reduction_steps = 0
    for line in lines:
        if "threadIdx.x) < 8" in line:
            in_reduction = True
        if in_reduction and "__syncthreads()" in line:
            syncs_inside += 1
        if in_reduction and "+=" in line and "red_buf" in line:
            reduction_steps += 1
        if in_reduction and "B[" in line and "red_buf" in line:
            in_reduction = False

    if syncs_inside < reduction_steps - 1:
        print(f"[BUG CONFIRMED] Missing __syncthreads() inside butterfly reduction!")
        print(f"  Reduction steps found : {reduction_steps}")
        print(f"  __syncthreads() inside: {syncs_inside}")
        print(f"  Expected              : {reduction_steps - 1}")
        print(f"  blockDim.x = {d3} — NOT a multiple of warp size (32)")
        print(f"  TVM assumes single-warp reduction but threads span multiple warps")
        print(f"  -> Missing barriers -> shared memory race -> wrong results on GPU")
    else:
        print("[INFO] __syncthreads() correctly inserted — bug may be fixed in this version")
else:
    print("[INFO] Reduction pattern not found in generated code")

# Verify numerically
print("\n=== Numerical Verification ===")
dev = tvm.cuda(0)
a_np = np.random.rand(1, d1, d2, d3).astype("float32")
b_np = a_np.sum(axis=-1).astype("float32")
a = tvm.nd.array(a_np, dev)
b = tvm.nd.array(np.zeros_like(b_np), dev)
f(a, b)

max_err = np.max(np.abs(b.numpy() - b_np))
print(f"Max error vs numpy: {max_err:.2e}")
if max_err > 1e-3:
    print("-> BUG CONFIRMED: wrong results due to missing __syncthreads()")
    print(f"   GPU shared memory race in butterfly reduction")
    print(f"   blockDim.x={d3} spans multiple warps — sync required between steps")
else:
    print("-> PASS (bug may be fixed in this TVM version or race did not fire this run)")
    print("   Note: race is non-deterministic — may need multiple runs to observe")