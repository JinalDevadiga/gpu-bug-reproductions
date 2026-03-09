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
def reduction_kernel(x_ptr, out_ptr, N: tl.constexpr):
    offsets = tl.arange(0, N)
    x = tl.load(x_ptr + offsets)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr, s)


N = 1024
x   = torch.randn(N, dtype=torch.float32, device='cuda')
out = torch.zeros(1, dtype=torch.float32, device='cuda')

reduction_kernel[(1,)](x, out, N, num_warps=4)

ref = x.sum()
print(f"Kernel result : {out.item():.6f}")
print(f"Reference     : {ref.item():.6f}")
print()

if torch.allclose(out, ref.unsqueeze(0), atol=1e-3):
    print("Numerically CORRECT (race is silent — run with compute-sanitizer to detect)")
else:
    print("Numerically WRONG")

print()
print("NOTE: This bug is a write-write (WAW) data race in shared memory.")
print("      The output may appear numerically correct because the race is")
print("      between threads writing slightly different FP results to the")
print("      same address due to butterfly shuffle accumulation order.")
print("      Run with compute-sanitizer to observe the actual race:")
print()
print("  compute-sanitizer --tool racecheck python reproduce.py")