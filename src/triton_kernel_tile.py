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
    """
    Triton kernel for pre-softmax MoE with tiling: softmax -> top-k -> normalize.
    Processes one batch element in chunks (tiles) of BLOCK_SIZE to handle large num_experts.
    """
    # Get batch index from program ID (each thread processes one batch element)
    pid = tl.program_id(0)
    # Early exit if batch index exceeds valid range
    if pid >= batch_size:
        return

    # Calculate base memory offsets for current batch element
    gating_offset = pid * stride_gating_batch
    experts_offset = pid * stride_experts_batch
    weights_offset = pid * stride_weights_batch

    # Initialize accumulators for softmax calculation
    global_max = tl.full([1], float('-inf'), dtype=tl.float32)  # For numerical stability
    global_sum_exp = tl.full([1], 0.0, dtype=tl.float32)       # Sum of exponentials

    # Temporary buffer to store top-k candidates across all tiles
    topk_values = tl.zeros([topk], dtype=tl.float32)
    topk_indices = tl.zeros([topk], dtype=tl.int32)

    # First pass: find global maximum value across all experts (for stable softmax)
    for tile_start in range(0, num_experts, BLOCK_SIZE):
        # Calculate tile boundaries with safety check for last tile
        tile_end = tl.minimum(tile_start + BLOCK_SIZE, num_experts)
        tile_size = tile_end - tile_start

        # Calculate memory offsets for this tile
        offsets = gating_offset + tile_start + tl.arange(0, BLOCK_SIZE)
        # Create mask for valid elements in the tile
        mask = tl.arange(0, BLOCK_SIZE) < tile_size

        # Load gating values for this tile (-inf for invalid positions)
        gating = tl.load(gating_output_ptr + offsets, mask=mask, other=float('-inf'))

        # Update global maximum with maximum from current tile
        tile_max = tl.max(gating, axis=0)
        global_max = tl.maximum(global_max, tile_max)

    # Second pass: compute softmax and find top-k values across all tiles
    for tile_start in range(0, num_experts, BLOCK_SIZE):
        # Calculate tile boundaries
        tile_end = tl.minimum(tile_start + BLOCK_SIZE, num_experts)
        tile_size = tile_end - tile_start

        # Calculate memory offsets for this tile
        offsets = gating_offset + tile_start + tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < tile_size

        # Load gating values and compute exponentials for softmax
        gating = tl.load(gating_output_ptr + offsets, mask=mask, other=float('-inf'))
        exp_gating = tl.exp(gating - global_max) * mask  # Apply mask to zero out invalid values
        global_sum_exp += tl.sum(exp_gating, axis=0)     # Accumulate sum of exponentials

        # Calculate softmax values for this tile
        softmax_output = tl.where(mask, exp_gating / global_sum_exp, 0.0)

        # Prepare values and indices for top-k selection within this tile
        values = softmax_output
        indices = tile_start + tl.arange(0, BLOCK_SIZE)  # Absolute expert indices

        # Find topk values in current tile and update global topk candidates
        for _ in range(topk):
            # Find maximum value and its index in current tile
            max_val = tl.max(values, axis=0)
            max_idx = tl.argmax(values, axis=0)

            # Find smallest value in our current topk candidates
            min_topk_val = tl.min(topk_values, axis=0)
            min_topk_idx = tl.argmin(topk_values, axis=0)

            # Update topk if current max is larger than smallest in topk
            if max_val > min_topk_val:
                # Create mask for position to update
                update_mask = tl.arange(0, topk) == min_topk_idx
                # Update values and indices using mask
                topk_values = tl.where(update_mask, max_val, topk_values)
                topk_indices = tl.where(update_mask, max_idx, topk_indices)

            # Mark this value as processed by setting to -infinity
            values = tl.where(indices == max_idx, float('-inf'), values)

    # Normalize top-k weights to sum to 1
    weight_sum = tl.sum(topk_values, axis=0) + 1e-10  # Add epsilon to prevent division by zero
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
    """
    Triton kernel for post-softmax MoE: top-k -> softmax -> normalize.
    Selects top-k experts first, then applies softmax to their scores.
    """
    # Get batch index from program ID
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    # Calculate memory offsets for current batch element
    gating_offset = pid * stride_gating_batch
    experts_offset = pid * stride_experts_batch
    weights_offset = pid * stride_weights_batch

    # Load gating output values with bounds checking
    offsets = gating_offset + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < num_experts  # Mask for valid expert indices
    gating = tl.load(gating_output_ptr + offsets, mask=mask, other=float('-inf'))

    # Prepare for top-k selection
    values = gating
    indices = tl.arange(0, BLOCK_SIZE)  # Expert indices
    topk_values = tl.zeros([topk], dtype=tl.float32)
    topk_indices = tl.zeros([topk], dtype=tl.int32)

    # Iteratively select top-k values
    for k in range(topk):
        # Find current maximum value and its index
        max_val = tl.max(values, axis=0)
        max_idx = tl.argmax(values, axis=0)

        # Update top-k arrays at position k using mask
        topk_values = tl.where(tl.arange(0, topk) == k, max_val, topk_values)
        topk_indices = tl.where(tl.arange(0, topk) == k, max_idx, topk_indices)

        # Mask out the selected value to prevent re-selection
        values = tl.where(indices == max_idx, float('-inf'), values)

    # Compute softmax on the selected top-k values
    max_val = tl.max(topk_values, axis=0)  # For numerical stability
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

    # Generate random gating output
    gating_output = torch.randn(batch_size, num_experts, device=device)

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
