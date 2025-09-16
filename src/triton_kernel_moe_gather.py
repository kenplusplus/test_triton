import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=8),
    ],
    key=['num_scatter_tokens', 'hidden_size']
)
@triton.jit
def moe_gather_kernel(
    scatter_tokens_ptr, scatter_tokens_offset_ptr, convergent_tokens_ptr,
    num_scatter_tokens, num_tokens, hidden_size,
    stride_scatter_m, stride_scatter_n,
    stride_offset,
    stride_convergent_m, stride_convergent_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    m = m_start + tl.arange(0, BLOCK_SIZE_M)             # [BLOCK_SIZE_M]
    n = n_start + tl.arange(0, BLOCK_SIZE_N)             # [BLOCK_SIZE_N]

    mask_m = m < num_scatter_tokens
    mask_n = n < hidden_size
    mask_2d = mask_m[:, None] & mask_n[None, :]          # [BLOCK_SIZE_M, BLOCK_SIZE_N]

    # Load offsets from 1D offsets array; use sentinel -1 for OOB
    offsets = tl.load(scatter_tokens_offset_ptr + m, mask=mask_m, other=-1)  # corrected
    offsets_2d = offsets[:, None]  # [BLOCK_SIZE_M, 1]

    # compute addresses for scatter_tokens: scatter_tokens[m, n]
    scatter_offsets = m[:, None] * stride_scatter_m + n[None, :] * stride_scatter_n
    scatter_data = tl.load(scatter_tokens_ptr + scatter_offsets, mask=mask_2d, other=0.0)

    # valid convergent indices: offsets must be >=0 and < num_tokens
    valid_convergent = (offsets_2d >= 0) & (offsets_2d < num_tokens)
    final_mask = mask_2d & valid_convergent

    # convergent memory addresses
    convergent_offsets = offsets_2d * stride_convergent_m + n[None, :] * stride_convergent_n

    # promote to float32 for atomic add
    data_fp32 = tl.cast(scatter_data, tl.float32)
    tl.atomic_add(convergent_tokens_ptr + convergent_offsets, data_fp32, mask=final_mask)


def moe_gather_run(tensor_mapping):
    scatter_tokens = tensor_mapping["scatter_tokens"]
    scatter_tokens_offset = tensor_mapping["scatter_tokens_offset"]
    convergent_tokens = tensor_mapping["convergent_tokens"]

    # Input validation
    assert scatter_tokens.ndim == 2, "scatter_tokens must be 2D"
    assert scatter_tokens_offset.ndim == 1, "scatter_tokens_offset must be 1D"
    assert convergent_tokens.ndim == 2, "convergent_tokens must be 2D"
    num_scatter_tokens, hidden_size = scatter_tokens.shape
    num_tokens, hidden_size_out = convergent_tokens.shape
    assert scatter_tokens_offset.shape[0] == num_scatter_tokens, "Offset size mismatch"
    assert hidden_size == hidden_size_out, "Hidden size mismatch"
    assert scatter_tokens.device == scatter_tokens_offset.device == convergent_tokens.device, "Device mismatch"

    # Ensure dtypes and contiguous layout: accumulate in float32
    scatter_tokens = scatter_tokens.contiguous()
    scatter_tokens_offset = scatter_tokens_offset.contiguous()
    convergent_tokens = convergent_tokens.to(torch.float32).contiguous()

    grid = lambda meta: (
        triton.cdiv(num_scatter_tokens, meta['BLOCK_SIZE_M']),
        triton.cdiv(hidden_size, meta['BLOCK_SIZE_N'])
    )

    moe_gather_kernel[grid](
        scatter_tokens, scatter_tokens_offset, convergent_tokens,
        num_scatter_tokens, num_tokens, hidden_size,
        scatter_tokens.stride(0), scatter_tokens.stride(1),
        scatter_tokens_offset.stride(0),
        convergent_tokens.stride(0), convergent_tokens.stride(1)
    )

    return convergent_tokens


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # small sizes for quick validation
    num_scatter_tokens = 128
    num_tokens = 64
    hidden_size = 32

    scatter_tokens = torch.randn((num_scatter_tokens, hidden_size), device=device, dtype=torch.float32)
    scatter_tokens_offset = torch.randint(0, num_tokens, (num_scatter_tokens,), device=device, dtype=torch.int32)
    convergent_tokens = torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.float32)

    mapping = {
        "scatter_tokens": scatter_tokens,
        "scatter_tokens_offset": scatter_tokens_offset,
        "convergent_tokens": convergent_tokens
    }

    triton_out = moe_gather_run(mapping)
    ref = mapping["convergent_tokens"].clone().index_add_(0, mapping["scatter_tokens_offset"].to(torch.int64), mapping["scatter_tokens"])
    is_close = torch.allclose(triton_out, ref, atol=1e-3, rtol=1e-3)
    print("Allclose:", is_close)
    if not is_close:
        diff = (triton_out - ref).abs()
        max_diff = diff.max().item()
        print("Max diff:", max_diff)
        print("triton_out[0:5]:", triton_out[:5])
        print("ref[0:5]:", ref[:5])
