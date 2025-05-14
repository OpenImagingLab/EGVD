import torch

def verify_tensor_operations(evs_latents):
    B, T, C, H, W = evs_latents.shape
    
    # Store original tensor for comparison
    original_tensor = evs_latents.clone()
    
    use_evs =  evs_latents[:, 0:T-1].clone()
    # Perform the operation: shift elements
    evs_latents[:, 1:T] = use_evs
    
    # Print results for verification
    print("Original Tensor (before operation):")
    print(original_tensor)
    
    print("\nModified Tensor (after operation):")
    print(evs_latents)
    
    # Check if the modified tensor matches the expected result
    print("\nIs the modified tensor correct?")
    print(torch.all(evs_latents[:, 1:T] == original_tensor[:, 0:T-1]))

# Example usage:
# Assuming evs_latents has the shape (B, T, C, H, W)
B, T, C, H, W = 1, 4, 1, 1, 2  # Example dimensions
evs_latents = torch.randn(B, T, C, H, W)  # Random tensor for testing

verify_tensor_operations(evs_latents)
