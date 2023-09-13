import torch
import numpy as np
import random

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
    mat[r1:r2+1, r1:r2+1] = torch.eye(r2 - r1 + 1)
    
    return mat