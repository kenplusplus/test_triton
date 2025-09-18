import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        # 移除重复配置，增加更多多样化的块大小组合
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 8}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 128}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256}, num_warps=16, num_stages=4),
        # 新增小尺寸块配置，优化小输入场景
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 8}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16}, num_warps=4, num_stages=1),
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
    # 1D网格优化：单个程序ID分解为m和n维度
    pid = tl.program_id(0)
    grid_m = tl.cdiv(num_scatter_tokens, BLOCK_SIZE_M)
    grid_n = tl.cdiv(hidden_size, BLOCK_SIZE_N)

    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # 计算当前块的起始索引
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    # 生成块内索引并计算全局索引
    m_local = tl.arange(0, BLOCK_SIZE_M)
    n_local = tl.arange(0, BLOCK_SIZE_N)
    m = m_start + m_local  # [BLOCK_SIZE_M]
    n = n_start + n_local  # [BLOCK_SIZE_N]

    # 合并掩码计算，减少中间变量
    mask_m = m < num_scatter_tokens
    mask_n = n < hidden_size

    # 加载偏移量并扩展维度
    offsets = tl.load(scatter_tokens_offset_ptr + m, mask=mask_m, other=-1)
    offsets_2d = offsets[:, None]  # [BLOCK_SIZE_M, 1]

    # 合并所有掩码条件
    final_mask = (mask_m[:, None] & mask_n[None, :] &
                 (offsets_2d >= 0) & (offsets_2d < num_tokens))

    # 计算 scatter 地址并加载数据
    scatter_offsets = m[:, None] * stride_scatter_m + n[None, :] * stride_scatter_n
    scatter_data = tl.load(scatter_tokens_ptr + scatter_offsets, mask=final_mask, other=0.0)

    # 计算 convergent 地址
    convergent_offsets = offsets_2d * stride_convergent_m + n[None, :] * stride_convergent_n

    # 共享内存中间缓冲区优化：减少全局内存原子操作
    smem = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.bfloat16, shared=True)
    smem[m_local[:, None], n_local[None, :]] = scatter_data
    tl.barrier()  # 确保所有线程都完成共享内存写入

    # 使用共享内存中的数据进行原子加法
    tl.atomic_add(
        convergent_tokens_ptr + convergent_offsets,
        smem[m_local[:, None], n_local[None, :]],
        mask=final_mask
    )


def moe_gather_run(tensor_mapping):
    scatter_tokens = tensor_mapping["scatter_tokens"]
    scatter_tokens_offset = tensor_mapping["scatter_tokens_offset"]
    convergent_tokens = tensor_mapping["convergent_tokens"]

    # 输入验证
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

    # 确保连续内存布局，优化访问效率
    scatter_tokens = scatter_tokens.contiguous()
    scatter_tokens_offset = scatter_tokens_offset.contiguous()
    convergent_tokens = convergent_tokens.contiguous()

    # 1D网格配置，减少内核启动开销
    grid = lambda meta: (
        triton.cdiv(num_scatter_tokens, meta['BLOCK_SIZE_M']) *
        triton.cdiv(hidden_size, meta['BLOCK_SIZE_N']),
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

    # 测试参数
    num_scatter_tokens = 32768
    num_tokens = 4096
    hidden_size = 4096

    scatter_tokens = torch.randn((num_scatter_tokens, hidden_size), device=device, dtype=torch.bfloat16)
    scatter_tokens_offset = torch.randint(0, num_tokens, (num_scatter_tokens,), device=device, dtype=torch.int32)
    convergent_tokens = torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    mapping = {
        "scatter_tokens": scatter_tokens,
        "scatter_tokens_offset": scatter_tokens_offset,
        "convergent_tokens": convergent_tokens
    }

    # 性能测试
    import time
    start = time.time()
    triton_out = moe_gather_run(mapping)
    torch.cuda.synchronize()
    triton_time = time.time() - start

    # 参考实现
    start = time.time()
    ref = convergent_tokens.clone().index_add_(0, scatter_tokens_offset.to(torch.int64), scatter_tokens)
    torch.cuda.synchronize()
    ref_time = time.time() - start

    # 验证正确性
    is_close = torch.allclose(triton_out, ref, atol=1e-2, rtol=1e-2)
    print(f"Allclose: {is_close}")
    print(f"Triton time: {triton_time:.4f}s, Reference time: {ref_time:.4f}s, Speedup: {ref_time/triton_time:.2f}x")

    if not is_close:
        diff = (triton_out - ref).abs()
        max_diff = diff.max().item()
        print(f"Max difference: {max_diff}")
