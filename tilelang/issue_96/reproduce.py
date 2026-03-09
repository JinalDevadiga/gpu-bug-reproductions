import tilelang
import tilelang.language as T


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            # BUG: num_stages > 0 causes non-deterministic race condition
            # on RTX 4070 Ti and A100 when N is large (e.g. 8192)
            # Changing num_stages=0 gives correct results every time
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])
    return main


print("=== Issue #96: Race Condition in Pipelined Matmul ===")
print()
print("NOTE: Actual wrong output requires SM80+ GPU (A100/H100) and N=8192.")
print("      This script shows the generated CUDA code containing the race.")
print()

N = 8192
func = tilelang.compile(matmul(N, N, N, 128, 128, 32), out_idx=[2], target="cuda")

print("--- Generated CUDA kernel source (BUGGY - num_stages=3) ---")
print(func.get_kernel_source())
