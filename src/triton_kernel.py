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
    # gating_offset: starting address for the row in gating_output
    # experts_offset: starting address for the row in selected_experts
    # weights_offset: starting address for the row in moe_weights
    gating_offset = pid * stride_gating_batch
    experts_offset = pid * stride_experts_batch
    weights_offset = pid * stride_weights_batch

    # Load the gating output for this batch element (one row of shape [num_experts])
    # Use BLOCK_SIZE for the range to ensure compile-time constant size
    # Mask ensures we only load valid indices (up to num_experts) and pad with 0.0 for safety
    offsets = gating_offset + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < num_experts
    gating = tl.load(gating_output_ptr + offsets, mask=mask, other=0.0)

    # Compute softmax over the num_experts dimension
    # Subtract max for numerical stability to prevent overflow in exp
    max_val = tl.max(gating, axis=0)
    exp_gating = tl.exp(gating - max_val)
    sum_exp = tl.sum(exp_gating, axis=0)
    softmax_output = exp_gating / sum_exp

    # Find top-k values and indices from softmax_output
    # Initialize arrays for top-k values and indices
    values = softmax_output
    indices = tl.arange(0, BLOCK_SIZE)
    topk_values = tl.zeros([topk], dtype=tl.float32)
    topk_indices = tl.zeros([topk], dtype=tl.int32)

    # Iteratively find the top-k values and indices
    for k in range(topk):
        # Find max value and index
        max_val = tl.max(values, axis=0)
        max_idx = tl.argmax(values, axis=0)

        # Store in top-k arrays
        topk_values = topk_values + (k == tl.arange(0, topk)) * max_val
        topk_indices = topk_indices + (k == tl.arange(0, topk)) * max_idx

        # Mask out the selected value by setting it to -inf
        values = values + (max_idx != indices) * values - (max_idx == indices) * float('inf')

    # Normalize top-k weights
    weight_sum = tl.sum(topk_values, axis=0)
    moe_weights = topk_values / weight_sum

    # Store results
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
    # Get batch index
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    # Offsets for the current batch
    gating_offset = pid * stride_gating_batch
    experts_offset = pid * stride_experts_batch
    weights_offset = pid * stride_weights_batch

    # Load gating output for this batch, ensuring block size alignment
    offsets = gating_offset + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < num_experts
    gating = tl.load(gating_output_ptr + offsets, mask=mask, other=0.0)

    # Find top-k values and indices
    values = gating
    indices = tl.arange(0, BLOCK_SIZE)
    topk_values = tl.zeros([topk], dtype=tl.float32)
    topk_indices = tl.zeros([topk], dtype=tl.int32)

    for k in range(topk):
        # Find max value and index
        max_val = tl.max(values, axis=0)
        max_idx = tl.argmax(values, axis=0)

        # Store in top-k arrays
        topk_values = topk_values + (k == tl.arange(0, topk)) * max_val
        topk_indices = topk_indices + (k == tl.arange(0, topk)) * max_idx

        # Mask out the selected value by setting it to -inf
        values = values + (max_idx != indices) * values - (max_idx == indices) * float('inf')

    # Compute softmax on top-k values
    max_val = tl.max(topk_values, axis=0)
    exp_values = tl.exp(topk_values - max_val)
    sum_exp = tl.sum(exp_values, axis=0)
    moe_weights = exp_values / sum_exp

    # Store results
    tl.store(selected_experts_ptr + experts_offset + tl.arange(0, topk), topk_indices)
    tl.store(moe_weights_ptr + weights_offset + tl.arange(0, topk), moe_weights)

def moe_softmax_topk(gating_output: torch.Tensor, topk: int, compute_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs MoE softmax and top-k selection in either pre-softmax or post-softmax mode.

    Args:
        gating_output (torch.Tensor): Input tensor of shape [batch_size, num_experts].
        topk (int): Number of experts to select.
        compute_mode (str): Either "pre-softmax" or "post-softmax".

    Returns:
        tuple: (selected_experts, moe_weights)
            - selected_experts: Tensor of shape [batch_size, topk] with expert indices (dtype=torch.int64).
            - moe_weights: Tensor of shape [batch_size, topk] with normalized weights (dtype=torch.float32).
    """
    assert gating_output.ndim == 2, "gating_output must be 2D"
    batch_size, num_experts = gating_output.shape
    assert topk <= num_experts, f"topk ({topk}) must be <= num_experts ({num_experts})"
    assert compute_mode in ["pre-softmax", "post-softmax"], "Invalid compute_mode"

    # Allocate output tensors
    selected_experts = torch.empty(batch_size, topk, dtype=torch.int32, device=gating_output.device)
    moe_weights = torch.empty(batch_size, topk, dtype=torch.float32, device=gating_output.device)

    # Grid size
    grid = (batch_size,)

    # Set BLOCK_SIZE to the next power of 2 greater than or equal to num_experts
    BLOCK_SIZE = triton.next_power_of_2(num_experts)

    # Launch appropriate kernel
    if compute_mode == "pre-softmax":
        moe_softmax_topk_pre_softmax_kernel[grid](
            gating_output, selected_experts, moe_weights,
            batch_size, num_experts, topk,
            gating_output.stride(0), gating_output.stride(1),
            selected_experts.stride(0), selected_experts.stride(1),
            moe_weights.stride(0), moe_weights.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:  # post-softmax
        moe_softmax_topk_post_softmax_kernel[grid](
            gating_output, selected_experts, moe_weights,
            batch_size, num_experts, topk,
            gating_output.stride(0), gating_output.stride(1),
            selected_experts.stride(0), selected_experts.stride(1),
            moe_weights.stride(0), moe_weights.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )

    # Cast selected_experts to torch.int64 to match PyTorch's topk output
    selected_experts = selected_experts.to(torch.int64)

    return selected_experts, moe_weights

# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test parameters
    batch_size = 128
    num_experts = 32
    topk = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate random input
    gating_output = torch.randn(batch_size, num_experts, device=device)

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
        print(f"\nTesting {mode} mode:")

        # Run Triton implementation
        triton_experts, triton_weights = moe_softmax_topk(gating_output, topk, mode)

        # Run PyTorch reference
        ref_experts, ref_weights = reference_moe_softmax_topk(gating_output, topk, mode)

        # Check correctness
        experts_match = torch.allclose(triton_experts, ref_experts, atol=1e-6)
        weights_match = torch.allclose(triton_weights, ref_weights, atol=1e-6)

        print(f"Experts match: {experts_match}")
        print(f"Weights match: {weights_match}")

        if not experts_match or not weights_match:
            print("Triton experts:", triton_experts[0])
            print("Reference experts:", ref_experts[0])
            print("Triton weights:", triton_weights[0])
            print("Reference weights:", ref_weights[0])