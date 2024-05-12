"""Curriculum learning module."""
import random
from typing import *
import numpy as np
import torch
import torch.nn as nn

class curriculum_weighting(nn.Module):
    """Curriculum learning module."""

    def __init__(self, num_modes: int = 16, num_features: int = 2,
                 start: float = 0.1, end: float = 0.5):
        """Initialize the `CurriculumWeighting` module

        Args:
            num_modes (int, optional): the number of 
            num_features (int, optional): the number of featueres per resol.
            progress (float): learning progress in [0, 1]
        """
        super().__init__()
        self.num_modes = num_modes # number of axis
        self.num_features = num_features # number of features per axis
        self.start = start
        self.end = end
        
    def forward(self, features: torch.Tensor, progress: float=0.):
        """Curriculum Weighting according to progress in [0, 1].

        Args:
            features (Tensor): positional encodings

        Returns:
            Tensor: weighted positional encodings
        """
        if not self.training or progress >= 1 or progress < 0 or (self.start == self.end):
            return features

        start, end = self.start, self.end
        alpha = (progress - start) / (end - start) * self.num_features
        k = torch.arange(self.num_features, dtype=features.dtype, device=features.device)
        k = torch.cat([k]*self.num_modes, dim=-1).view(1, -1)
        weight = (1-(alpha-k).clamp_(min=0, max=1).mul_(np.pi).cos_())/2

        # TO DEBUG
        # print(f"progress : {progress}")
        # print(f"curriculum weight : {weight.cpu().numpy()}")
        return features * weight


if __name__ == '__main__':
    features = torch.ones(1, 48).float(); features.requires_grad_()
    dummy = 2*torch.ones(1, 48).float()
    density_weight_module = curriculum_weighting(1, 48, start=0.1, end=0.5)

    # progress
    progress = random.uniform(0,1)

    # curriculum weighting
    out = density_weight_module(features, progress)
    
    # backpropagation
    loss = ((out - dummy)**2).sum()
    loss.backward()
    print(f"The gradient in feature : {features.grad}")
