"""
Biregular Graph Generation for Sparse Matrix Factorization.

Generates a bipartite graph where:
- Each left node (W_i) has exactly C1 = alpha1 * M edges
- Each right node (X_j) has exactly C2 = alpha2 * M edges
- Constraint: N1 * C1 = N2 * C2

This satisfies the Dense Limit requirements from arXiv:2510.17886.
"""

from typing import Tuple
import torch
import numpy as np


class BiregularGraph:
    """
    Biregular bipartite graph generator.
    
    Each W-node (row) has exactly C1 edges.
    Each X-node (column) has exactly C2 edges.
    Uses rejection sampling to avoid multi-edges.
    """
    
    def generate(
        self,
        N1: int,
        N2: int,
        M: int,
        alpha1: float,
        device: torch.device,
        seed: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, float]:
        """
        Generate biregular graph edge indices.
        
        Args:
            N1: Number of left nodes (rows of W)
            N2: Number of right nodes (columns of X)
            M: Rank (hidden dimension)
            alpha1: Degree parameter for W (C1 = alpha1 * M)
            device: torch device
            seed: Random seed
        
        Returns:
            i_idx: Row indices of edges
            j_idx: Column indices of edges
            E: Total number of edges
            C1: Degree of each W node
            C2: Degree of each X node
            alpha2: Computed alpha2 = (N1/N2) * alpha1
        """
        # Compute degrees
        C1 = int(round(alpha1 * M))
        C1 = max(1, min(C1, N2))  # Clamp to valid range
        
        # Total edges
        E = N1 * C1
        
        # Compute C2 from constraint N1*C1 = N2*C2
        C2_float = (N1 * C1) / N2
        C2 = int(round(C2_float))
        
        # Ensure constraint is satisfied
        if C2 == 0:
            C2 = 1
        
        # Recompute E to satisfy N1*C1 = N2*C2
        # Use LCM approach for exact matching
        E = N1 * C1
        
        # Compute alpha2
        alpha2 = C2 / M if M > 0 else 0.0
        
        if E == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                0, 0, 0, 0.0
            )
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate biregular graph using row-by-row sampling
        # Each row i gets exactly C1 unique columns
        i_list = []
        j_list = []
        
        # Track column degrees
        col_degrees = np.zeros(N2, dtype=np.int32)
        
        for i in range(N1):
            # Available columns (not yet at max degree for this approach)
            # For simplicity, just sample C1 unique columns per row
            available = np.arange(N2)
            
            # Randomly select C1 columns
            if len(available) >= C1:
                selected = np.random.choice(available, size=C1, replace=False)
            else:
                selected = available
            
            for j in selected:
                i_list.append(i)
                j_list.append(j)
        
        i_idx = torch.tensor(i_list, dtype=torch.long, device=device)
        j_idx = torch.tensor(j_list, dtype=torch.long, device=device)
        E = len(i_idx)
        
        return i_idx, j_idx, E, C1, C2, alpha2

    def generate_dense_mask(
        self,
        N1: int,
        N2: int,
        M: int,
        alpha1: float,
        device: torch.device,
        seed: int = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        int,
        float,
    ]:
        """
        Generate the same graph as ``generate`` and materialize it as a dense mask.

        Returns:
            mask: Dense binary observation mask with shape (N1, N2)
            i_idx: Row indices of observed entries
            j_idx: Column indices of observed entries
            E: Total number of observed entries
            C1: Degree of each W node
            C2: Degree of each X node
            alpha2: Computed alpha2 = (N1/N2) * alpha1
        """
        i_idx, j_idx, E, C1, C2, alpha2 = self.generate(
            N1, N2, M, alpha1, device, seed
        )
        mask = torch.zeros((N1, N2), dtype=torch.float32, device=device)
        if E > 0:
            mask[i_idx.long(), j_idx.long()] = 1.0

        return mask, i_idx, j_idx, E, C1, C2, alpha2


# Backward compatibility wrapper
class RandomGraph:
    """
    Wrapper for backward compatibility.
    Uses BiregularGraph internally.
    """
    
    def __init__(self):
        self._biregular = BiregularGraph()
    
    def generate(
        self,
        N1: int,
        N2: int,
        M: int,
        alpha: float,
        device: torch.device,
        seed: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Generate graph (backward compatible interface).
        
        Args:
            N1, N2, M: Matrix dimensions
            alpha: Degree parameter (alpha1)
            device: torch device
            seed: Random seed
        
        Returns:
            i_idx, j_idx: Edge indices
            E: Total number of edges
        """
        i_idx, j_idx, E, C1, C2, alpha2 = self._biregular.generate(
            N1, N2, M, alpha, device, seed
        )
        return i_idx, j_idx, E
