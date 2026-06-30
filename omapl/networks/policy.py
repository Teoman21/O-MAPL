"""Local decentralised policy networks pi_i(a_i | o_i; omega_i).

Two policy families, matching the paper's Implementation Details (App. B.3):

* Discrete (SMACv1/v2): a Categorical distribution computed via softmax over
  *only the available actions* for each agent. Unavailable actions are masked
  to probability zero so the log-likelihood is not penalised for infeasible
  actions.
* Continuous (MAMuJoCo): a diagonal Gaussian (``torch.distributions.Normal``)
  parameterised by a state-dependent mean and a (state-dependent) log-std.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.distributions as D

from ..networks.agent import LocalValueNet

_NEG_INF = -1e10


class CategoricalPolicy(nn.Module):
    def __init__(self, n_agents: int, obs_dim: int, action_dim: int,
                 hidden_dim: int = 256, param_sharing: bool = True,
                 use_agent_id: bool = True):
        super().__init__()
        self.net = LocalValueNet(n_agents, obs_dim, action_dim, hidden_dim,
                                 param_sharing, use_agent_id)

    def logits(self, obs: torch.Tensor,
               avail: Optional[torch.Tensor]) -> torch.Tensor:
        logits = self.net(obs)                          # [*, n, A]
        if avail is not None:
            logits = logits.masked_fill(avail < 0.5, _NEG_INF)
        return logits

    def distribution(self, obs: torch.Tensor,
                     avail: Optional[torch.Tensor] = None) -> D.Categorical:
        return D.Categorical(logits=self.logits(obs, avail))

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor,
                 avail: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-agent log pi(a_i|o_i) -> ``[*, n_agents]``."""
        return self.distribution(obs, avail).log_prob(action.long())

    @torch.no_grad()
    def act(self, obs: torch.Tensor, avail: Optional[torch.Tensor] = None,
            deterministic: bool = False) -> torch.Tensor:
        logits = self.logits(obs, avail)
        if deterministic:
            return logits.argmax(dim=-1)
        return D.Categorical(logits=logits).sample()


class GaussianPolicy(nn.Module):
    def __init__(self, n_agents: int, obs_dim: int, action_dim: int,
                 hidden_dim: int = 256, param_sharing: bool = True,
                 use_agent_id: bool = True,
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # Outputs mean and log_std for every action dimension.
        self.net = LocalValueNet(n_agents, obs_dim, 2 * action_dim, hidden_dim,
                                 param_sharing, use_agent_id)

    def distribution(self, obs: torch.Tensor,
                     avail: Optional[torch.Tensor] = None) -> D.Normal:
        out = self.net(obs)                             # [*, n, 2A]
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return D.Normal(mean, log_std.exp())

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor,
                 avail: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-agent log pi(a_i|o_i) summed over action dims -> ``[*, n]``."""
        return self.distribution(obs).log_prob(action).sum(dim=-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, avail: Optional[torch.Tensor] = None,
            deterministic: bool = False) -> torch.Tensor:
        dist = self.distribution(obs)
        return dist.mean if deterministic else dist.sample()


def build_policy(discrete: bool, **kwargs) -> nn.Module:
    return CategoricalPolicy(**kwargs) if discrete else GaussianPolicy(**kwargs)
