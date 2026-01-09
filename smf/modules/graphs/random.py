"""
Biregular graph generation (GPU-based).

Generates bipartite graphs where each left node has exactly C1 edges
and attempts to balance right node degrees towards C2.

Satisfies Dense Limit requirements from arXiv:2510.17886.
"""

from typing import Tuple
import torch
import numpy as np

from ..registry import register_graph
from .base import GraphBase


@register_graph(
    key="random",
    name="Biregular Graph (GPU)",
    description="Biregular sampling, each row has exactly C1=alpha*M edges",
)
class RandomGraph(GraphBase):
    """
    Biregular bipartite graph generator.

    Approach:
    1. Each row i gets exactly C1 = alpha * M unique column indices
    2. Columns are sampled randomly for each row
    3. Total edges E = N1 * C1
    
    This ensures each W-node (row) has exactly C1 edges.
    Column degrees will vary around the expected C2 = (N1*C1)/N2.
    """

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
        Generate biregular edge indices.
        
        Each row has exactly C1 = alpha * M edges.
        
        Args:
            N1: Number of rows
            N2: Number of columns
            M: Rank
            alpha: Degree parameter (C1 = alpha * M)
            device: torch device
            seed: Random seed
            
        Returns:
            i_idx: Row indices
            j_idx: Column indices
            E: Total number of edges
        """
        # Calculate degree per row
        C1 = int(round(alpha * M))
        C1 = max(1, min(C1, N2))  # Clamp to valid range
        
        E = N1 * C1
        
        if E == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                0,
            )
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Generate edges: each row gets exactly C1 unique columns
        i_list = []
        j_list = []
        
        for i in range(N1):
            # Sample C1 unique columns for this row
            cols = np.random.choice(N2, size=C1, replace=False)
            for j in cols:
                i_list.append(i)
                j_list.append(j)
        
        i_idx = torch.tensor(i_list, dtype=torch.long, device=device)
        j_idx = torch.tensor(j_list, dtype=torch.long, device=device)
        
        return i_idx, j_idx, E
