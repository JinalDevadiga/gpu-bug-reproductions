"""
Reproduction of apache/tvm#7246:
  Race condition in tvm.tir.call_packed() under parallel schedule.

  apache-tvm==0.8.0 is not available as a pip wheel (oldest is 0.9.0).
  IR inspection approach used — same as Triton #8311.

Environment:
  conda activate tvm-bugs   (apache-tvm==0.11.1)
"""
import re
import tvm
from tvm import te

print(f"TVM version : {tvm.__version__}")

M, N, K, bn = 8, 8, 8, 4

A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="C",
)

s = te.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
s[C].reorder(xo, yo, xi, yi, k)
s[C].parallel(xo)

def intrin_matmul(m, k_dim, n):
    a  = te.placeholder((m, k_dim), name="a")
    b  = te.placeholder((k_dim, n), name="b")
    k_ = te.reduce_axis((0, k_dim), name="k")
    c  = te.compute(
        (m, n),
        lambda i, j: te.sum(a[i, k_] * b[k_, j], axis=k_),
        name="c",
    )
    a_buf = tvm.tir.decl_buffer(a.shape, a.dtype, name="a_buf",
                                offset_factor=1, strides=[te.var("s1"), 1])
    b_buf = tvm.tir.decl_buffer(b.shape, b.dtype, name="b_buf",
                                offset_factor=1, strides=[te.var("s2"), 1])
    c_buf = tvm.tir.decl_buffer(c.shape, c.dtype, name="c_buf",
                                offset_factor=1, strides=[te.var("s3"), 1])

    def intrin_func(ins, outs):
        return tvm.tir.call_packed(
            "tvm.contrib.cblas.matmul",
            ins[0], ins[1], outs[0], False, False,
        )

    return te.decl_tensor_intrin(c.op, intrin_func,
                                 binds={a: a_buf, b: b_buf, c: c_buf})

s[C].tensorize(xi, intrin_matmul(bn, K, bn))

lowered = tvm.lower(s, [A, B, C], simple_mode=True)
ir_text = str(lowered)

print("\n=== Lowered TIR ===")
print(ir_text)

alloca_pos   = [m.start() for m in re.finditer(r"tvm_stack_alloca", ir_text)]
parallel_pos = ir_text.find("parallel")

print("\n=== IR Analysis ===")
if not alloca_pos:
    print("[INFO] tvm_stack_alloca not present — call_packed inlined as "
          "tvm_stack_make_array inside the parallel loop (safe).")
    print("       Buggy pattern (pre-0.9.0): stack_alloca appears BEFORE "
          "the parallel loop — all threads share the same arg stack.")
elif parallel_pos != -1 and alloca_pos[0] < parallel_pos:
    print("-> BUG CONFIRMED: tvm_stack_alloca appears BEFORE the parallel "
          "loop — all threads share the same packed-func arg stack → WAW race.")
else:
    print("-> FIXED: tvm_stack_alloca appears INSIDE the parallel loop "
          "— each thread has its own private stack copy.")