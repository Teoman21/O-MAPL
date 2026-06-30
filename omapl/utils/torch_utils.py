"""Small PyTorch helpers shared across the codebase."""
from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mlp(in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int = 2,
        activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    """A simple feed-forward MLP with ``n_hidden`` hidden layers."""
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(last, hidden_dim), activation()]
        last = hidden_dim
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: target <- tau * source + (1 - tau) * target."""
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.copy_(sp)


def grad_norm_clip(params: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    if max_norm and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(list(params), max_norm)


def agent_id_onehot(n_agents: int, batch_shape: torch.Size, device) -> torch.Tensor:
    """Return a one-hot agent-id tensor broadcastable to ``[*batch, n, n]``.

    ``batch_shape`` is the shape *up to but not including* the agent axis.
    """
    eye = torch.eye(n_agents, device=device)
    view = (1,) * len(batch_shape) + (n_agents, n_agents)
    expand = tuple(batch_shape) + (n_agents, n_agents)
    return eye.view(view).expand(expand)
