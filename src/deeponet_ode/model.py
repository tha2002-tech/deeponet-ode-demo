"""DeepONet (Lu et al. 2021): branch net on the input function, trunk net on the query point."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def mlp(in_dim: int, width: int, n_hidden: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """G(f, k)(t) ≈ sum_p branch_p([f; k]) * trunk_p(t) + b.

    Branch input is standardized with `branch_mean`/`branch_std`, stored as buffers so
    they travel with the state_dict.
    """

    def __init__(
        self,
        n_sensors: int = 100,
        branch_extra: int = 1,
        width: int = 128,
        n_hidden: int = 2,
        p: int = 64,
    ) -> None:
        super().__init__()
        branch_dim = n_sensors + branch_extra
        self.branch = mlp(branch_dim, width, n_hidden, p)
        self.trunk = mlp(1, width, n_hidden, p)
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("branch_mean", torch.zeros(branch_dim))
        self.register_buffer("branch_std", torch.ones(branch_dim))

    def set_branch_stats(self, mean: Tensor, std: Tensor) -> None:
        self.branch_mean.copy_(mean)
        self.branch_std.copy_(std.clamp_min(1e-8))

    def forward(self, branch_in: Tensor, t: Tensor) -> Tensor:
        x = (branch_in - self.branch_mean) / self.branch_std
        b = self.branch(x)  # (B, p)
        tr = self.trunk(t)  # (B, Q, p)
        return torch.einsum("bp,bqp->bq", b, tr) + self.bias


def make_branch_input(f_sensors: Tensor, k: Tensor) -> Tensor:
    """Concatenate sensor values and k into the branch input (B, m + 1)."""
    return torch.cat([f_sensors, k.reshape(-1, 1)], dim=1)


def relative_l2(pred: Tensor, target: Tensor) -> Tensor:
    """Mean over the batch of ||pred - target||_2 / ||target||_2."""
    num = torch.linalg.vector_norm(pred - target, dim=1)
    den = torch.linalg.vector_norm(target, dim=1).clamp_min(1e-12)
    return (num / den).mean()
