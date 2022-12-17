import torch
import numpy as np
from torch import nn
from torch import Tensor
from einops import rearrange


"""
    Use torch again as the auto-differentiator
    Of course we can choose to use CUDA, but...
"""
class Solver(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h = nn.Parameter(torch.rand(8, 1), requires_grad = True)

    @staticmethod
    def huber_loss(x: Tensor, alpha: float):
        pass

    # pts_c shape: (N, 2), pts_o: (N, 2), weights is of shape (N)
    # this is just a simple model, with 
    def huber_forward(self, pts_c: Tensor, pts_o: Tensor, weights: Tensor):
        # First: re-arrange pts_c to a reasonable structure
        num_points = pts_c.shape[0]
        rare_part = -pts_o[..., None] @ pts_c[:, None, :]        # shape (N, 2, 2)
        padded = torch.cat([pts_c, torch.FloatTensor([1, 0, 0, 0]).unsqueeze(dim = 0).expand(num_points, -1)], dim = -1)
        rolled = torch.roll(padded, shifts = (3, ), dims = (-1))
        front_part = torch.cat([padded.unsqueeze(dim = -2), rolled.unsqueeze(dim = -2)], dim = -2)
        lhs = torch.cat([front_part, rare_part], dim = -1)
        lhs = rearrange(lhs, 'N M C -> (N M) C', M = 2)                         # shape (2N, 8)
        rhs = pts_o.reshape(-1, 1)
        weights = weights.unsqueeze(dim = -1).expand(-1, 2).reshape(-1, 1)      # make [1, 2, 3] -> [[1], [1], [2], [2], [3], [3]] -- shape (2N, 1)
        diff = weights * (lhs @ self.h - rhs)                                   # difference (weighted)
        return Solver.huber_loss(diff)