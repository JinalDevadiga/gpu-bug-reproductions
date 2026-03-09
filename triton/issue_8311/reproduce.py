import torch
import triton
import triton.language as tl

print(f"Triton version : {triton.__version__}")
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    sm    = props.major * 10 + props.minor
    print(f"GPU            : {props.name}  (sm{sm})")
print()


@triton.jit
def bad_kernel(
    x, y, out,
    SIZE:            tl.constexpr,
    BLOCK_M:         tl.constexpr,
    BLOCK_N:         tl.constexpr,
    BLOCK_K:         tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    desc = tl.make_tensor_descriptor(y, [SIZE, SIZE], [SIZE, 1], [BLOCK_K, BLOCK_N])
    for mo in tl.range(0, SIZE, BLOCK_M):
        for no in tl.range(0, SIZE, BLOCK_N):
            rows = mo + tl.arange(0, BLOCK_M)
            cols = no + tl.arange(0, BLOCK_N)
            acc  = tl.full([BLOCK_M, BLOCK_N], 0.0, tl.float32)
            for ko in tl.range(0, SIZE, BLOCK_K, warp_specialize=WARP_SPECIALIZE):
                ks     = ko + tl.arange(0, BLOCK_K)
                x_tile = tl.load(x + (rows[:, None] * SIZE + ks[None, :]))
                y_tile = desc.load([ko, no])
                acc    = tl.dot(
                    tl.cast(x_tile, tl.float16),
                    tl.cast(y_tile, tl.float16),
                    acc=acc,
                    input_precision='tf32',
                    out_dtype=tl.float32,
                )
            tl.store(out + (rows[:, None] * SIZE + cols[None, :]), tl.cast(acc, tl.float16))


SIZE    = 1024
BLOCK_M = 2
BLOCK_N = 8
BLOCK_K = 512

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
x   = torch.randn((SIZE, SIZE), dtype=torch.float16, device='cuda')
y   = torch.randn((SIZE, SIZE), dtype=torch.float16, device='cuda')
out = torch.empty((SIZE, SIZE), dtype=torch.float16, device='cuda')
ref = torch.matmul(x, y)

# Run buggy kernel and check output
try:
    compiled = bad_kernel[(1,)](
        x, y, out, SIZE, BLOCK_M, BLOCK_N, BLOCK_K, True,
        num_warps=16, num_stages=1
    )
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("PASSED (sm90+ required to trigger bug — see README)")
except AssertionError as e:
    lines = str(e).splitlines()
    print("BUG CONFIRMED:")
    for line in lines[:6]:
        print(f"  {line}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)[:200]}")

# Dump TTGIR to show the {tt.warp_specialize} attribute on the buggy loop
print()
print("--- TTGIR (shows {tt.warp_specialize} attribute on inner loop) ---")
print()
print(compiled.asm['ttgir'])
