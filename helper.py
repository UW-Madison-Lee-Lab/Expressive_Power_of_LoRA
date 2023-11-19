import numpy as np
import random, wandb, torch, argparse

def set_seed(seed_value = 123):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)
    
def generate_diag_matrix(D, r1, r2):
    """
    Generate a diagonal matrix of size D x D.
    Diagonal entries from r1 to r2 will be set to 1.
    All other entries will be 0.
    """
    # Create a zero-filled matrix
    mat = torch.zeros(D, D)
    
    # Set the diagonal entries from r1 to r2 to 1
    mat[r1:r2, r1:r2] = torch.eye(r2 - r1)
    
    return mat

def our_construction(target_weight, frozen_weights, rank, log_wandb, atol = 1e-3):
    """
    Our construction of low-rank adapter for matrix approximation

    Args:
        target_weight: D * D matrix
        frozen_weights: a list of D * D matrices
        rank: rank of the approximation
    """
    
    width = target_weight.shape[0]
    depth = len(frozen_weights)
    
    frozen_prod_weight = torch.eye(width)
    frozen_prod_weight_2depth = {(depth - 1): frozen_prod_weight}
    for l in range(1,depth)[::-1]:
        frozen_prod_weight = frozen_prod_weight @ frozen_weights[l]
        frozen_prod_weight_2depth[l-1] = frozen_prod_weight
    frozen_prod_weight = frozen_prod_weight @ frozen_weights[0]
    
    # compute the discrepancy matrix
    discrepancy_weight = target_weight - frozen_prod_weight
    
    # perform SVD on the discrepancy matrix
    _, S, V = torch.svd(discrepancy_weight)
    
    if log_wandb:
        wandb.log({"Singular values": S})
    else:
        print("Singular values:", S)
        
    # compute the lora weight for each frozen matrix/layer
    lora_A, lora_B = [], []
    adapted_prod_weight = torch.eye(width)
    for l in range(depth):
        # compute the lora adapter for the l-th layer
        lora_A_ = torch.inverse(frozen_prod_weight_2depth[l]) @ discrepancy_weight @ V[:, min(rank*l, width):min(rank*(l+1),width)]
        lora_A.append(lora_A_)
        lora_B_ = (V.T @ torch.inverse(adapted_prod_weight))[min(rank*l, width): min(rank*(l+1), width), :].T
        lora_B.append(lora_B_)
        
        adapted_prod_weight = (frozen_weights[l] + lora_A_ @ lora_B_.T) @ adapted_prod_weight
    
    # Check if the manual and automatic outputs match
    match = torch.allclose(adapted_prod_weight, target_weight, atol=atol)
    if (not match) and (rank >= width // depth + int(width % depth != 0)):
        print("WARNING: our construction does not offer exact representation!")
        print("The maximum discrepancy is", (torch.abs(adapted_prod_weight - target_weight).max()/torch.abs(target_weight).max()).item())

    return lora_A, lora_B    

def my_int(value):
    if value == 'inf':
        return float('inf')
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value: {value}")
