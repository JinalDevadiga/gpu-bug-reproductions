import torch
import numpy as np
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
def compute_min_distance_coord(input_ptr, coord_ptr, min_cord_idx_ptr,
                                BLOCK_SIZE: tl.constexpr):
    pid            = tl.program_id(0)
    offs_input_row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_coord_row = tl.arange(0, 32)
    offs_coord_idx = tl.arange(0, 8)

    offs_input = offs_input_row[:, None] * 8 + offs_coord_idx[None, :]
    offs_coord = offs_coord_row[:, None] * 8 + offs_coord_idx[None, :]

    inp   = tl.load(input_ptr + offs_input)
    coord = tl.load(coord_ptr + offs_coord)

    diff    = inp[:, None, :] - coord[None, :, :]
    dist_sq = diff * diff

    _, min_coord_idxs = tl.min(
        tl.sum(dist_sq, axis=-1), axis=-1, return_indices=True
    )
    tl.store(min_cord_idx_ptr + offs_input_row, min_coord_idxs.to(tl.int32))


N      = 1 << 20
input  = torch.rand(N, 8,  dtype=torch.float32, device='cuda')
coords = torch.rand(32, 8, dtype=torch.float32, device='cuda')
out    = torch.zeros(N,    dtype=torch.int32,   device='cuda')

grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
compute_min_distance_coord[grid](input, coords, out, BLOCK_SIZE=512)
torch.cuda.synchronize()

inp_np   = input.cpu().numpy()
coord_np = coords.cpu().numpy()
diff_sq  = np.square(inp_np[:, None, :] - coord_np[None, :, :])
ref      = np.argmin(np.sum(diff_sq, axis=-1), axis=-1).astype(np.int32)

correct = np.allclose(ref, out.cpu().numpy())
print(f"Numerically correct: {correct}")
print("(Race is silent — run with: compute-sanitizer --tool=racecheck python reproduce.py)")