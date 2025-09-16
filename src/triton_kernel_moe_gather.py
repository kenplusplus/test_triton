import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 8}, num_warps=8),
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
    # Get program IDs for the 2D grid
    pid_m = tl.program_id(0)  # Batch dimension (num_scatter_tokens)
    pid_n = tl.program_id(1)  # Hidden dimension (hidden_size)

    # Compute 1D indices for this block
    scatter_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    scatter_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Create masks for valid indices
    scatter_mask_m = scatter_m < num_scatter_tokens
    scatter_mask_n = scatter_n < hidden_size

    # Load corresponding indices from scatter_tokens_offset (1D along batch)
    offset_mask = scatter_mask_m
    indices = tl.load(scatter_tokens_offset_ptr + scatter_m * stride_offset, mask=offset_mask, other=0)

    # Reshape for broadcasting to 2D [BLOCK_SIZE_M, BLOCK_SIZE_N]
    scatter_m_2d = scatter_m[:, None]
    scatter_n_2d = scatter_n[None, :]
    indices_2d = indices[:, None]

    # Compute offsets for scatter_tokens [BLOCK_SIZE_M, BLOCK_SIZE_N]
    scatter_offsets = scatter_m_2d * stride_scatter_m + scatter_n_2d * stride_scatter_n
    scatter_load_mask = scatter_mask_m[:, None] & scatter_mask_n[None, :]
    scatter_data = tl.load(scatter_tokens_ptr + scatter_offsets, mask=scatter_load_mask, other=0.0)

    # Compute offsets for convergent_tokens [BLOCK_SIZE_M, BLOCK_SIZE_N]
    convergent_m_2d = indices_2d
    convergent_n_2d = scatter_n_2d
    convergent_mask = (convergent_m_2d < num_tokens) & scatter_mask_n[None, :]
    convergent_offsets = convergent_m_2d * stride_convergent_m + convergent_n_2d * stride_convergent_n

    # Perform atomic add to accumulate values in convergent_tokens
    tl.atomic_add(convergent_tokens_ptr + convergent_offsets, scatter_data, mask=convergent_mask)

def moe_gather_run(tensor_mapping):
    """
    Performs MoE gather operation using Triton kernel, accumulating scatter_tokens into convergent_tokens
    based on indices in scatter_tokens_offset.

    Args:
        tensor_mapping (dict): Dictionary containing input and output tensors:
            - scatter_tokens: [num_scatter_tokens, hidden_size] tensor of scattered token data
            - scatter_tokens_offset: [num_scatter_tokens] tensor of indices for gathering
            - convergent_tokens: [num_tokens, hidden_size] output tensor for accumulated results

    Returns:
        torch.Tensor: The convergent_tokens tensor with accumulated values
    """
    # Extract tensors from tensor_mapping
    scatter_tokens = tensor_mapping["scatter_tokens"]
    scatter_tokens_offset = tensor_mapping["scatter_tokens_offset"]
    convergent_tokens = tensor_mapping["convergent_tokens"]

    # Validate input dimensions
    assert scatter_tokens.ndim == 2, "scatter_tokens must be 2D"
    assert scatter_tokens_offset.ndim == 1, "scatter_tokens_offset must be 1D"
    assert convergent_tokens.ndim == 2, "convergent_tokens must be 2D"
    num_scatter_tokens, hidden_size = scatter_tokens.shape
    num_tokens, hidden_size_out = convergent_tokens.shape
    assert scatter_tokens_offset.shape[0] == num_scatter_tokens, "Mismatch in scatter_tokens_offset size"
    assert hidden_size == hidden_size_out, "Hidden size mismatch between scatter_tokens and convergent_tokens"
    assert scatter_tokens.device == scatter_tokens_offset.device == convergent_tokens.device, "All tensors must be on the same device"

    # Ensure tensors are in the correct format
    scatter_tokens = scatter_tokens.to(torch.float32).contiguous()
    scatter_tokens_offset = scatter_tokens_offset.to(torch.int32).contiguous()
    convergent_tokens = convergent_tokens.to(torch.float32).contiguous()

    # Define grid for kernel launch
    grid = lambda meta: (
        triton.cdiv(num_scatter_tokens, meta['BLOCK_SIZE_M']),
        triton.cdiv(hidden_size, meta['BLOCK_SIZE_N'])
    )

    # Launch Triton kernel
    moe_gather_kernel[grid](
        scatter_tokens_ptr=scatter_tokens,
        scatter_tokens_offset_ptr=scatter_tokens_offset,
        convergent_tokens_ptr=convergent_tokens,
        num_scatter_tokens=num_scatter_tokens,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        stride_scatter_m=scatter_tokens.stride(0),
        stride_scatter_n=scatter_tokens.stride(1),
        stride_offset=scatter_tokens_offset.stride(0),
        stride_convergent_m=convergent_tokens.stride(0),
        stride_convergent_n=convergent_tokens.stride(1),
    )

    return convergent_tokens

# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test parameters
    num_scatter_tokens = 128
    num_tokens = 64
    hidden_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate random input tensors
    scatter_tokens = torch.randn(num_scatter_tokens, hidden_size, device=device)
    scatter_tokens_offset = torch.randint(0, num_tokens, (num_scatter_tokens,), device=device)
    convergent_tokens = torch.zeros(num_tokens, hidden_size, device=device)

    # Create tensor_mapping
    tensor_mapping = {
        "scatter_tokens": scatter_tokens,
        "scatter_tokens_offset": scatter_tokens_offset,
        "convergent_tokens": convergent_tokens
    }

    # Run PyTorch reference implementation
    def reference_moe_gather(tensor_mapping):
        scatter_tokens = tensor_mapping["scatter_tokens"]
        scatter_tokens_offset = tensor_mapping["scatter_tokens_offset"]
        convergent_tokens = tensor_mapping["convergent_tokens"].clone()
        convergent_tokens.index_add_(0, scatter_tokens_offset, scatter_tokens)
        return convergent_tokens

    # Run Triton implementation
    triton_result = moe_gather_run(tensor_mapping)

    # Run reference implementation
    ref_result = reference_moe_gather(tensor_mapping)

    # Check correctness
    result_match = torch.allclose(triton_result, ref_result, atol=1e-3)
    print(f"Results match: {result_match}")

    # Print sample results
    print("Triton result (first row):", triton_result[0, :5])
    print("Reference result (first row):", ref_result[0, :5])