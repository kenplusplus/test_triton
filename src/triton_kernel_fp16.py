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
    # Get batch index from program ID
    pid = tl.program_id(0)
    # Early exit if batch index is out of bounds
    if pid >= batch_size:
        return

    # Calculate memory offsets for current batch element
    gating_offset = pid * stride_gating_batch
    experts_offset = pid * stride_experts_batch
    weights_offset = pid * stride_weights_batch

    # Load gating output values with bounds checking
    offsets = gating_offset + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < num_experts  # Mask for valid expert indices
    # Use -inf for invalid positions to avoid affecting max calculations
    gating = tl.load(gating_output_ptr + offsets, mask=mask, other=-float('inf'))

    # Convert to FP32 for stable softmax calculation
    gating_fp32 = gating.to(tl.float32)
    max_val = tl.max(gating_fp32, axis=0)  # Numerical stability for softmax
    exp_gating = tl.exp(gating_fp32 - max_val)
    sum_exp = tl.sum(exp_gating, axis=0) + 1e-10  # Add epsilon to prevent division by zero
    softmax_output = exp_gating / sum_exp

    # Prepare for top-k selection
    values = softmax_output
    indices = tl.arange(0, BLOCK_SIZE)  # Expert indices
    topk_values = tl.full([topk], -float('inf'), dtype=tl.float32)  # Initialize with -inf
    topk_indices = tl.zeros([topk], dtype=tl.int32)  # Store selected expert indices
    inf = float('inf')

    # Iteratively select top-k values
    for k in range(topk):
        # Find current maximum value and its index
        max_val = tl.max(values, axis=0)
        max_idx = tl.argmax(values, axis=0)

        # Update top-k arrays at position k using tl.where for reliable assignment
        topk_values = tl.where(tl.arange(0, topk) == k, max_val, topk_values)
        topk_indices = tl.where(tl.arange(0, topk) == k, max_idx, topk_indices)

        # Mask out the selected value by setting to -inf so it's not selected again
        values = tl.where(indices == max_idx, -inf, values)

    # Normalize top-k weights
    weight_sum = tl.sum(topk_values, axis=0) + 1e-10  # Prevent division by zero
    moe_weights = topk_values / weight_sum

    # Store results to output tensors
    tl.store(selected_experts_ptr + experts_offset + tl.arange(0, topk), topk_indices)
    tl.store(moe_weights_ptr + weights_offset + tl.arange(0, topk), moe_weights)

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

    # Load gating output values with bounds checking
    offsets = gating_offset + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < num_experts  # Mask for valid expert indices
    # Use -inf for invalid positions to avoid affecting max calculations
    gating = tl.load(gating_output_ptr + offsets, mask=mask, other=-float('inf'))

    # Prepare for top-k selection
    values = gating
    indices = tl.arange(0, BLOCK_SIZE)  # Expert indices
    topk_values = tl.full([topk], -float('inf'), dtype=tl.float32)  # Initialize with -inf
    topk_indices = tl.zeros([topk], dtype=tl.int32)  # Store selected expert indices
    inf = float('inf')

    # Iteratively select top-k values
    for k in range(topk):
        # Find current maximum value and its index
        max_val = tl.max(values, axis=0)
        max_idx = tl.argmax(values, axis=0)

        # Update top-k arrays at position k using tl.where for reliable assignment
        topk_values = tl.where(tl.arange(0, topk) == k, max_val, topk_values)
        topk_indices = tl.where(tl.arange(0, topk) == k, max_idx, topk_indices)

        # Mask out the selected value by setting to -inf so it's not selected again
        values = tl.where(indices == max_idx, -inf, values)

    # Compute softmax on the selected top-k values
    max_val = tl.max(topk_values, axis=0)  # Numerical stability
    exp_values = tl.exp(topk_values - max_val)
    sum_exp = tl.sum(exp_values, axis=0) + 1e-10  # Prevent division by zero
    moe_weights = exp_values / sum_exp

    # Store results to output tensors
    tl.store(selected_experts_ptr + experts_offset + tl.arange(0, topk), topk_indices)
    tl.store(moe_weights_ptr + weights_offset + tl.arange(0, topk), moe_weights)

def moe_softmax_topk(gating_output: torch.Tensor, topk: int, compute_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs MoE (Mixture of Experts) softmax and top-k expert selection.

    Args:
        gating_output (torch.Tensor): Input tensor of shape [batch_size, num_experts]
            containing gating scores for each expert.
        topk (int): Number of top experts to select for each batch element.
        compute_mode (str): Either "pre-softmax" (apply softmax first then select top-k)
            or "post-softmax" (select top-k first then apply softmax).

    Returns:
        tuple: (selected_experts, moe_weights)
            - selected_experts: Tensor of shape [batch_size, topk] with indices of selected experts.
            - moe_weights: Tensor of shape [batch_size, topk] with normalized weights for selected experts.
    """
    # Input validation
    assert gating_output.ndim == 2, "gating_output must be 2D (batch_size x num_experts)"
    batch_size, num_experts = gating_output.shape
    assert topk <= num_experts, f"topk ({topk}) must be <= num_experts ({num_experts})"
    assert compute_mode in ["pre-softmax", "post-softmax"], "Invalid compute_mode"

    # Allocate output tensors
    selected_experts = torch.empty(batch_size, topk, dtype=torch.int32, device=gating_output.device)
    moe_weights = torch.empty(batch_size, topk, dtype=torch.float32, device=gating_output.device)

    # Ensure input type compatibility
    if gating_output.dtype != torch.float16 and gating_output.dtype != torch.float32:
        gating_output = gating_output.to(torch.float32)

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
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Configuration parameters
    batch_size = 128
    num_experts = 32
    topk = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate random gating output (can test with FP16 or FP32)
    gating_output = torch.randn(batch_size, num_experts, device=device, dtype=torch.float32)
    # gating_output = torch.randn(batch_size, num_experts, device=device, dtype=torch.float16)

    # Reference implementation using PyTorch operations
    def reference_moe_softmax_topk(gating_output, topk, compute_mode):
        if compute_mode == "pre-softmax":
            # Apply softmax first, then select top-k
            softmax_output = torch.softmax(gating_output, dim=-1)
            moe_weights, selected_experts = torch.topk(softmax_output, topk, dim=-1)
            # Normalize the selected weights
            moe_weights = moe_weights / (moe_weights.sum(dim=-1, keepdim=True) + 1e-10)
            return selected_experts, moe_weights
        else:  # post-softmax
            # Select top-k first, then apply softmax
            topk_output, selected_experts = torch.topk(gating_output, topk, dim=-1)
            softmax_output = torch.softmax(topk_output, dim=-1)
            return selected_experts, softmax_output

    # Test both computation modes
    for mode in ["pre-softmax", "post-softmax"]:
        print(f"\nTesting {mode} mode:")

        # Run Triton implementation
        triton_experts, triton_weights = moe_softmax_topk(gating_output, topk, mode)

        # Run PyTorch reference implementation
        ref_experts, ref_weights = reference_moe_softmax_topk(gating_output, topk, mode)

        # Check result correctness
        experts_match = torch.all(triton_experts == ref_experts)
        weights_match = torch.allclose(triton_weights, ref_weights, atol=1e-4)

        print(f"Experts match: {experts_match}")
        print(f"Weights match: {weights_match}")

        # Print mismatches if any
        #if not experts_match or not weights_match:
        print("Triton experts:", triton_experts[0])
        print("Reference experts:", ref_experts[0])
        print("Triton weights:", triton_weights[0])
        print("Reference weights:", ref_weights[0])
