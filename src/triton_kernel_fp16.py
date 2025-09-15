import triton
import triton.language as tl
import torch

@triton.jit
def moe_softmax_topk_pre_softmax_kernel(
    gating_output_ptr, selected_experts_ptr, moe_weights_ptr,
    batch_size, num_experts, topk: tl.constexpr,
    stride_gating_batch, stride_gating_experts,
    stride_experts_batch, stride_experts_topk,
    stride_weights_batch, stride_weights_topk,
    BLOCK_SIZE: tl.constexpr
):
    # Get batch index
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    # Compute memory offsets for the current batch element
    gating_offset = pid * stride_gating_batch
    experts_offset = pid * stride_experts_batch
    weights_offset = pid * stride_weights_batch

    # Load the gating output for this batch element in FP16
    offsets = gating_offset + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < num_experts
    # Load in FP16, use -inf (FP32, auto-converted by Triton) for masked values
    gating = tl.load(gating_output_ptr + offsets, mask=mask, other=-float('inf'))

    # Convert to FP32 for stable softmax computation
    gating_fp32 = gating.to(tl.float32)
    max_val = tl.max(gating_fp32, axis=0)
    exp_gating = tl.exp(gating_fp32 - max_val)
    sum_exp = tl.sum(exp_gating, axis=0)
    softmax_output = exp_gating / sum_exp

    # Find top-k values and indices from softmax_output
    values = tl.where(mask, softmax_output, -float('inf'))
    indices = tl.arange(0, BLOCK_SIZE)
    topk_values = tl.full([topk], -float('inf'), dtype=tl.float32)
    topk_indices = tl.zeros([topk], dtype=tl.int32)

    # Iteratively find the top-k values and indices
    for k in range(topk):
        max_val = tl.max(values, axis=0)
        max_idx = tl.argmax(values, axis=0)
        topk_values = tl.where(tl.arange(0, topk) == k, max_val, topk_values)
        topk_indices = tl.where(tl.arange(0, topk) == k, max_idx, topk_indices)
        values = tl.where(indices == max_idx, -float('inf'), values)

    # Normalize top-k weights in FP32
    weight_sum = tl.sum(topk_values, axis=0)
    moe_weights_fp32 = topk_values / weight_sum

    # Convert weights to FP16 for storage
    moe_weights_fp16 = moe_weights_fp32.to(tl.float16)

    # Store results
    tl.store(selected_experts_ptr + experts_offset + tl.arange(0, topk), topk_indices)
    tl.store(moe_weights_ptr + weights_offset + tl.arange(0, topk), moe_weights_fp16)

@triton.jit
def moe_softmax_topk_post_softmax_kernel(
    gating_output_ptr, selected_experts_ptr, moe_weights_ptr,
    batch_size, num_experts, topk: tl.constexpr,
    stride_gating_batch, stride_gating_experts,
    stride_experts_batch, stride_experts_topk,
    stride_weights_batch, stride_weights_topk,
    BLOCK_SIZE: tl.constexpr
):
    # Get batch index
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    # Offsets for the current batch
    gating_offset = pid * stride_gating_batch
    experts_offset = pid * stride_experts_batch
    weights_offset = pid * stride_weights_batch

    # Load gating output in FP16
    offsets = gating_offset + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < num_experts
    gating = tl.load(gating_output_ptr + offsets, mask=mask, other=-float('inf'))

    # Convert to FP32 for stable top-k selection
    values_fp32 = gating.to(tl.float32)
    values = tl.where(mask, values_fp32, -float('inf'))
    indices = tl.arange(0, BLOCK_SIZE)
    topk_values = tl.full([topk], -float('inf'), dtype=tl.float32)
    topk_indices = tl.zeros([topk], dtype=tl.int32)

    for k in range(topk):
        max_val = tl.max(values, axis=0)
        max_idx = tl.argmax(values, axis=0)
        topk_values = tl.where(tl.arange(0, topk) == k, max_val, topk_values)
        topk_indices = tl.where(tl.arange(0, topk) == k, max_idx, topk_indices)
        values = tl.where(indices == max_idx, -float('inf'), values)

    # Apply softmax to top-k values in FP32
    max_val = tl.max(topk_values, axis=0)
    exp_values = tl.exp(topk_values - max_val)
    sum_exp = tl.sum(exp_values, axis=0)
    moe_weights_fp32 = exp_values / sum_exp

    # Convert weights to FP16 for storage
    moe_weights_fp16 = moe_weights_fp32.to(tl.float16)

    # Store results
    tl.store(selected_experts_ptr + experts_offset + tl.arange(0, topk), topk_indices)
    tl.store(moe_weights_ptr + weights_offset + tl.arange(0, topk), moe_weights_fp16)

def moe_softmax_topk(gating_output: torch.Tensor, topk: int, compute_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """

    Optimizations:
    - Uses FP16 for input/output to reduce memory bandwidth by ~50%.
    - Performs critical computations (softmax, top-k) in FP32 for numerical stability.
    - Optimized for coalesced memory access on CUDA devices.

    """
    # Input validation
    assert gating_output.ndim == 2, "gating_output must be 2D"
    batch_size, num_experts = gating_output.shape
    assert batch_size > 0, "batch_size must be positive"
    assert num_experts > 0, "num_experts must be positive"
    assert topk <= num_experts, f"topk ({topk}) must be <= num_experts ({num_experts})"
    assert compute_mode in ["pre-softmax", "post-softmax"], "Invalid compute_mode"
    assert gating_output.device.type == "cuda", "FP16 optimization requires CUDA device"

    # Convert input to FP16 for memory efficiency
    if gating_output.dtype != torch.float16:
        gating_output = gating_output.to(torch.float16)

    # Allocate output tensors
    selected_experts = torch.empty(batch_size, topk, dtype=torch.int32, device=gating_output.device)
    moe_weights = torch.empty(batch_size, topk, dtype=torch.float16, device=gating_output.device)

    # Grid size
    grid = (batch_size,)

    # Set BLOCK_SIZE to the next power of 2 greater than or equal to num_experts
    BLOCK_SIZE = triton.next_power_of_2(num_experts)

    # Launch appropriate kernel
    if compute_mode == "pre-softmax":
        moe_softmax_topk_pre_softmax_kernel[grid](
            gating_output_ptr=gating_output,
            selected_experts_ptr=selected_experts,
            moe_weights_ptr=moe_weights,
            batch_size=batch_size,
            num_experts=num_experts,
            topk=topk,
            stride_gating_batch=gating_output.stride(0),
            stride_gating_experts=gating_output.stride(1),
            stride_experts_batch=selected_experts.stride(0),
            stride_experts_topk=selected_experts.stride(1),
            stride_weights_batch=moe_weights.stride(0),
            stride_weights_topk=moe_weights.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:  # post-softmax
        moe_softmax_topk_post_softmax_kernel[grid](
            gating_output_ptr=gating_output,
            selected_experts_ptr=selected_experts,
            moe_weights_ptr=moe_weights,
            batch_size=batch_size,
            num_experts=num_experts,
            topk=topk,
            stride_gating_batch=gating_output.stride(0),
            stride_gating_experts=gating_output.stride(1),
            stride_experts_batch=selected_experts.stride(0),
            stride_experts_topk=selected_experts.stride(1),
            stride_weights_batch=moe_weights.stride(0),
            stride_weights_topk=moe_weights.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )

    # Cast selected_experts to torch.int64 to match PyTorch's topk output
    selected_experts = selected_experts.to(torch.int64)

    return selected_experts, moe_weights

# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Test parameters
    batch_size = 128
    num_experts = 32
    topk = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type != "cuda":
        print("Warning: CUDA device not available. FP16 optimizations require CUDA.")
        exit()

    # Generate random input in FP16
    gating_output = torch.randn(batch_size, num_experts, device=device, dtype=torch.float16)

    # Run PyTorch reference implementation
    def reference_moe_softmax_topk(gating_output, topk, compute_mode):
        if compute_mode == "pre-softmax":
            softmax_output = torch.softmax(gating_output, dim=-1)
            moe_weights, selected_experts = torch.topk(softmax_output, topk, dim=-1)
            moe_weights = moe_weights / moe_weights.sum(dim=-1, keepdim=True)
            return selected_experts, moe_weights
        else:  # post-softmax
            topk_output, selected_experts = torch.topk(gating_output, topk, dim=-1)
            softmax_output = torch.softmax(topk_output, dim=-1)
            return selected_experts, softmax_output

    # Test both modes
    for mode in ["pre-softmax", "post-softmax"]:
        print(f"\nTesting {mode} mode (FP16):")

        # Run Triton implementation
        triton_experts, triton_weights = moe_softmax_topk(gating_output, topk, mode)

        # Run PyTorch reference
        ref_experts, ref_weights = reference_moe_softmax_topk(gating_output, topk, mode)

        # Check correctness with relaxed tolerance for FP16
        experts_match = torch.all(triton_experts == ref_experts)
        weights_match = torch.allclose(triton_weights, ref_weights, atol=1e-2, rtol=1e-2)

        print(f"Experts match: {experts_match}")
        print(f"Weights match: {weights_match}")

        print("Triton experts:", triton_experts[0])
        print("Reference experts:", ref_experts[0])
        print("Triton weights:", triton_weights[0])
        print("Reference weights:", ref_weights[0])
