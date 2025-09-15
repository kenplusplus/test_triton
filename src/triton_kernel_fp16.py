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
    # Get batch index from program ID (each thread block processes one batch element)
    pid = tl.program_id(0)
    # Early exit if batch index exceeds valid range
    if pid >= batch_size:
        return

    # Calculate memory offsets for current batch element
    gating_offset = pid * stride_gating_batch
    experts_offset = pid * stride_experts_batch
    weights_offset = pid * stride_weights_batch

    # Generate indices for loading gating values
    offsets = gating_offset + tl.arange(0, BLOCK_SIZE)
    # Create mask to handle cases where BLOCK_SIZE > num_experts
    mask = tl.arange(0, BLOCK_SIZE) < num_experts

    # Load gating values in FP16 format from global memory
    # Use -inf for invalid positions to avoid affecting max calculations
    gating_fp16 = tl.load(gating_output_ptr + offsets, mask=mask, other=-tl.float16('inf'), dtype=tl.float16)

    # Convert to FP32 for stable softmax computation (prevents precision loss)
    gating_fp32 = gating_fp16.to(tl.float32)
    max_val = tl.max(gating_fp32, axis=0)  # Numerical stability for softmax
    exp_gating = tl.exp(gating_fp32 - max_val)
    sum_exp = tl.sum(exp_gating, axis=0) + 1e-10  # Add epsilon to prevent division by zero
    softmax_output = exp_gating / sum_exp

    # Prepare for top-k selection
    values = softmax_output
    indices = tl.arange(0, BLOCK_SIZE)  # Expert indices
    topk_values = tl.full([topk], -tl.float32('inf'), dtype=tl.float32)  # Initialize with -inf
    topk_indices = tl.zeros([topk], dtype=tl.int32)  # Store selected expert indices
    inf = tl.float32('inf')

    # Iteratively select top-k values
    for k in range(topk):
        # Find current maximum value and its index
        max_val = tl.max(values, axis=0)
        max_idx = tl.argmax(values, axis=0)

        # Update top-k arrays at position k using tl.where for thread-safe assignment
        topk_values = tl.where(tl.arange(0, topk) == k, max_val, topk_values)
        topk_indices = tl.where(tl.arange(0, topk) == k, max_idx, topk_indices)

        # Mask out the selected value by setting to -inf so it's not selected again
        values = tl.where(indices == max_idx, -inf, values)

    # Normalize top-k weights
    weight_sum = tl.sum(topk_values, axis=0) + 1e-10  # Prevent division by zero
    moe_weights_fp32 = topk_values / weight_sum

    # Convert weights back to FP16 for storage
    moe_weights_fp16 = moe_weights_fp32.to(tl.float16)
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
    # Get batch index from program ID
    pid = tl.program_id(0)
    # Early exit if batch index is out of bounds
    if pid >= batch_size:
        return

    # Calculate memory offsets for current batch element
    gating_offset = pid * stride_gating_batch
    experts_offset = pid * stride_experts_batch
    weights_offset = pid * stride_weights_batch

    # Generate indices for loading gating values
    offsets = gating_offset + tl.arange(0, BLOCK_SIZE)
    # Create mask to handle cases where BLOCK_SIZE > num_experts
    mask = tl.arange(0, BLOCK_SIZE) < num_experts

    # Load gating values in FP16 format from global memory
    gating_fp16 = tl.load(gating_output_ptr + offsets, mask=mask, other=-tl.float16('inf'), dtype=tl.float16)

    # Convert to FP32 for stable top-k selection
    values_fp32 = gating_fp16.to(tl.float32)
    indices = tl.arange(0, BLOCK_SIZE)  # Expert indices
    topk_values = tl.full([topk], -tl.float32('inf'), dtype=tl.float32)  # Initialize with -inf
    topk_indices = tl.zeros([topk], dtype=tl.int32)  # Store selected expert indices
    inf = tl.float32('inf')

    # Iteratively select top-k values
    for k in range(topk):
        # Find current maximum value and its index
        max_val = tl.max(values_fp32, axis=0)
        max_idx = tl.argmax(values_fp32, axis=0)

        # Update top-k arrays at position k using tl.where for thread-safe assignment
        topk_values = tl.where(tl.arange(0, topk) == k, max_val, topk_values)
        topk_indices = tl.where(tl.arange(0, topk) == k, max_idx, topk_indices)

        # Mask out the selected value by setting to -inf so it's not selected again
        values_fp32 = tl.where(indices == max_idx, -inf, values_fp32)

    # Compute softmax on the selected top-k values in FP32 for stability
    max_val = tl.max(topk_values, axis=0)  # Numerical stability
    exp_values = tl.exp(topk_values - max_val)
    sum_exp = tl.sum(exp_values, axis=0) + 1e-10  # Prevent division by zero
    moe_weights_fp32 = exp_values / sum_exp

    # Convert weights back to FP16 for storage
    moe_weights_fp16 = moe_weights_fp32.to(tl.float16)

    # Store results to output tensors
    tl.store(selected_experts_ptr + experts_offset + tl.arange(0, topk), topk_indices)
    tl.store(moe_weights_ptr + weights_offset + tl.arange(0, topk), moe_weights_fp16)


def moe_softmax_topk(gating_output: torch.Tensor, topk: int, compute_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs FP16-optimized MoE (Mixture of Experts) softmax and top-k expert selection.

    Key optimizations:
    - Uses FP16 for memory storage (reduces bandwidth usage by 50%)
    - Performs critical computations in FP32 to maintain numerical stability
    - Efficient memory access patterns optimized for GPU

    Args:
        gating_output (torch.Tensor): Input tensor of shape [batch_size, num_experts]
            containing gating scores for each expert. Will be converted to FP16.
        topk (int): Number of top experts to select for each batch element.
        compute_mode (str): Either "pre-softmax" (apply softmax first then select top-k)
            or "post-softmax" (select top-k first then apply softmax).

    Returns:
        tuple: (selected_experts, moe_weights)
            - selected_experts: Tensor of shape [batch_size, topk] with indices of selected experts (int64).
            - moe_weights: Tensor of shape [batch_size, topk] with normalized weights (FP16).
    """
    # Input validation
    assert gating_output.ndim == 2, "gating_output must be 2D (batch_size x num_experts)"
    batch_size, num_experts = gating_output.shape
    assert topk <= num_experts, f"topk ({topk}) must be <= num_experts ({num_experts})"
    assert compute_mode in ["pre-softmax", "post-softmax"], "Invalid compute_mode"
    assert gating_output.device.type == "cuda", "FP16 optimization requires CUDA device"

    # Convert input to FP16 for memory efficiency
    if gating_output.dtype != torch.float16:
        gating_output = gating_output.to(torch.float16)

    # Allocate output tensors
    # Expert indices remain as int32/int64 since they don't benefit from FP16
    selected_experts = torch.empty(batch_size, topk, dtype=torch.int32, device=gating_output.device)
    # Weights stored in FP16 to save memory
    moe_weights = torch.empty(batch_size, topk, dtype=torch.float16, device=gating_output.device)

    # Define grid dimensions (one thread block per batch element)
    grid = (batch_size,)

    # Set block size to next power of 2 >= num_experts for efficient memory access
    BLOCK_SIZE = triton.next_power_of_2(num_experts)

    # Launch appropriate kernel based on compute mode
    if compute_mode == "pre-softmax":
        moe_softmax_topk_pre_softmax_kernel[grid](
            gating_output, selected_experts, moe_weights,
            batch_size, num_experts, topk,
            gating_output.stride(0), gating_output.stride(1),
            selected_experts.stride(0), selected_experts.stride(1),
            moe_weights.stride(0), moe_weights.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:  # post-softmax mode
        moe_softmax_topk_post_softmax_kernel[grid](
            gating_output, selected_experts, moe_weights,
            batch_size, num_experts, topk,
            gating_output.stride(0), gating_output.stride(1),
            selected_experts.stride(0), selected_experts.stride(1),
            moe_weights.stride(0), moe_weights.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )

    # Convert expert indices to int64 to match PyTorch's topk output type
    selected_experts = selected_experts.to(torch.int64)

    return selected_experts, moe_weights


# Example usage and validation
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Configuration parameters
    batch_size = 128
    num_experts = 32
    topk = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("Warning: CUDA device not available. FP16 optimizations require CUDA.")
    else:
        print(f"Using device: {device} (FP16 optimization enabled)")

        # Generate random gating output in FP16
        gating_output = torch.randn(batch_size, num_experts, device=device, dtype=torch.float16)

        # Reference implementation using PyTorch operations
        def reference_moe_softmax_topk(gating_output, topk, compute_mode):
            """PyTorch reference implementation for validation"""
            gating_fp16 = gating_output.to(torch.float16)
            if compute_mode == "pre-softmax":
                # Apply softmax first, then select top-k
                softmax_output = torch.softmax(gating_fp16, dim=-1)
                moe_weights, selected_experts = torch.topk(softmax_output, topk, dim=-1)
                # Normalize the selected weights
                moe_weights = moe_weights / (moe_weights.sum(dim=-1, keepdim=True) + 1e-10)
                return selected_experts, moe_weights
            else:  # post-softmax
                # Select top-k first, then apply softmax
                topk_output, selected_experts = torch.topk(gating_fp16, topk, dim=-1)
                softmax_output = torch.softmax(topk_output, dim=-1)
                return selected_experts, softmax_output

        # Test both computation modes
        for mode in ["pre-softmax", "post-softmax"]:
            print(f"\nTesting {mode} mode (FP16):")

            # Run Triton implementation
            triton_experts, triton_weights = moe_softmax_topk(gating_output, topk, mode)

            # Run PyTorch reference implementation
            ref_experts, ref_weights = reference_moe_softmax_topk(gating_output, topk, mode)

            # Check result correctness
            experts_match = torch.all(triton_experts == ref_experts)
            # Use larger tolerance for FP16 comparisons
            weights_match = torch.allclose(triton_weights, ref_weights, atol=1e-3, rtol=1e-3)

            print(f"Experts match: {experts_match}")
            print(f"Weights match: {weights_match}")

            # Print sample results for verification
            print("Triton experts (FP16):", triton_experts[0])
            print("Reference experts (FP16):", ref_experts[0])
            print("Triton weights (FP16):", triton_weights[0])
            print("Reference weights (FP16):", ref_weights[0])
