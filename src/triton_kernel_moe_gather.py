import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        # for 4096 hidden size
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256}, num_warps=4),

        # for large size
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}, num_warps=8),

        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 16}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 512}, num_warps=8),
    ],
    key=['num_scatter_tokens', 'hidden_size'],
    warmup=10,
    rep=100
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
    m = m_start + tl.arange(0, BLOCK_SIZE_M)  # [BLOCK_SIZE_M]
    n = n_start + tl.arange(0, BLOCK_SIZE_N)  # [BLOCK_SIZE_N]

    mask_m = m < num_scatter_tokens
    mask_n = n < hidden_size
    mask_2d = mask_m[:, None] & mask_n[None, :]  # [BLOCK_SIZE_M, BLOCK_SIZE_N]

    # Load offsets
    offsets = tl.load(scatter_tokens_offset_ptr + m, mask=mask_m, other=-1)
    offsets_2d = offsets[:, None]  # [BLOCK_SIZE_M, 1]

    # Early exit for invalid offsets
    valid_offsets = (offsets_2d >= 0) & (offsets_2d < num_tokens)
    final_mask = mask_2d & valid_offsets

    scatter_base = scatter_tokens_ptr + m[:, None] * stride_scatter_m
    scatter_offsets = scatter_base + n[None, :] * stride_scatter_n
    scatter_data = tl.load(scatter_offsets, mask=final_mask, other=0.0)

    convergent_base = convergent_tokens_ptr + offsets_2d * stride_convergent_m
    convergent_offsets = convergent_base + n[None, :] * stride_convergent_n

    # Use atomic add with proper masking
    tl.atomic_add(convergent_offsets, scatter_data, mask=final_mask)


def moe_gather_run(tensor_mapping, warmup=False):
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
    assert scatter_tokens.dtype == torch.bfloat16, "scatter_tokens must be bfloat16"
    assert convergent_tokens.dtype == torch.bfloat16, "convergent_tokens must be bfloat16"

    # Ensure contiguous layout
    scatter_tokens = scatter_tokens.contiguous()
    scatter_tokens_offset = scatter_tokens_offset.contiguous()
    convergent_tokens = convergent_tokens.contiguous()

    grid = lambda meta: (
        triton.cdiv(num_scatter_tokens, meta['BLOCK_SIZE_M']),
        triton.cdiv(hidden_size, meta['BLOCK_SIZE_N'])
    )

    if warmup:
        print("Performing warmup...")

        warmup_scatter = scatter_tokens[:128, :128]
        warmup_offset = scatter_tokens_offset[:128]
        warmup_convergent = convergent_tokens[:64, :128]

        warmup_grid = lambda meta: (
            triton.cdiv(128, meta['BLOCK_SIZE_M']),
            triton.cdiv(128, meta['BLOCK_SIZE_N'])
        )

        moe_gather_kernel[warmup_grid](
            warmup_scatter, warmup_offset, warmup_convergent,
            128, 64, 128,
            warmup_scatter.stride(0), warmup_scatter.stride(1),
            warmup_offset.stride(0),
            warmup_convergent.stride(0), warmup_convergent.stride(1)
        )
        torch.cuda.synchronize()

    moe_gather_kernel[grid](
        scatter_tokens, scatter_tokens_offset, convergent_tokens,
        num_scatter_tokens, num_tokens, hidden_size,
        scatter_tokens.stride(0), scatter_tokens.stride(1),
        scatter_tokens_offset.stride(0),
        convergent_tokens.stride(0), convergent_tokens.stride(1)
    )

    return convergent_tokens

def benchmark_moe_gather(tensor_mapping, num_runs=10):
    """性能基准测试"""
    times = []

    # 预热
    moe_gather_run(tensor_mapping, warmup=True)
    torch.cuda.synchronize()

    for i in range(num_runs):
        convergent_copy = torch.zeros_like(tensor_mapping["convergent_tokens"])
        run_mapping = {
            "scatter_tokens": tensor_mapping["scatter_tokens"],
            "scatter_tokens_offset": tensor_mapping["scatter_tokens_offset"],
            "convergent_tokens": convergent_copy
        }

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        moe_gather_run(run_mapping, warmup=True)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    return times

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 测试多种尺寸
    test_cases = [
        (32768, 4096, 4096),
        (16384, 2048, 4096),
        (8192, 1024, 2048),
    ]

    for num_scatter_tokens, num_tokens, hidden_size in test_cases:
        print(f"\nTesting: {num_scatter_tokens}x{hidden_size} -> {num_tokens}x{hidden_size}")

        scatter_tokens = torch.randn((num_scatter_tokens, hidden_size), device=device, dtype=torch.bfloat16)
        scatter_tokens_offset = torch.randint(0, num_tokens, (num_scatter_tokens,), device=device, dtype=torch.int32)
        convergent_tokens = torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

        mapping = {
            "scatter_tokens": scatter_tokens,
            "scatter_tokens_offset": scatter_tokens_offset,
            "convergent_tokens": convergent_tokens
        }

        # 测试正确性
        triton_out = moe_gather_run(mapping, warmup=False)
        ref = convergent_tokens.clone().index_add_(0, scatter_tokens_offset.to(torch.int64), scatter_tokens)

        is_close = torch.allclose(triton_out, ref, atol=1e-2, rtol=1e-2)
        print(f"Correctness: {is_close}")

        if not is_close:
            diff = (triton_out - ref).abs()
            print(f"Max diff: {diff.max().item():.6f}")

        # 性能测试
        times = benchmark_moe_gather(mapping, num_runs=5)
        print(f"Performance: {sum(times)/len(times):.3f} ms (avg), {min(times):.3f} ms (min)")
