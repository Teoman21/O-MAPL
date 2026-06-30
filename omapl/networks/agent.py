"""Local per-agent value networks: q_i(o_i, a_i) and v_i(o_i).

These implement the *local* functions of the O-MAPL value factorisation
(Section 5 of the paper). Local Q/V take only the local observation o_i so
that decentralised execution is possible; the global Q_tot / V_tot are formed
by the :class:`~omapl.networks.mixer.Mixer`.

We support the standard MARL trick of *parameter sharing*: a single network is
shared across agents, with a one-hot agent id appended to the local input to
keep agents distinguishable. Set ``param_sharing=False`` for independent nets.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..utils.torch_utils import mlp, agent_id_onehot


class LocalValueNet(nn.Module):
    """Shared base producing per-agent outputs from per-agent observations.

    Input  : obs ``[*, n_agents, obs_dim]``
    Output : ``[*, n_agents, out_dim]``
    """

    def __init__(self, n_agents: int, obs_dim: int, out_dim: int,
                 hidden_dim: int = 256, param_sharing: bool = True,
                 use_agent_id: bool = True):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.out_dim = out_dim
        self.param_sharing = param_sharing
        self.use_agent_id = use_agent_id

        in_dim = obs_dim + (n_agents if use_agent_id else 0)
        if param_sharing:
            self.net = mlp(in_dim, hidden_dim, out_dim)
        else:
            self.nets = nn.ModuleList(
                [mlp(in_dim, hidden_dim, out_dim) for _ in range(n_agents)]
            )

    def _augment(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.use_agent_id:
            return obs
        ids = agent_id_onehot(self.n_agents, obs.shape[:-2], obs.device)
        return torch.cat([obs, ids], dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._augment(obs)
        if self.param_sharing:
            return self.net(x)
        outs = [self.nets[i](x[..., i, :]) for i in range(self.n_agents)]
        return torch.stack(outs, dim=-2)


class LocalQNet(nn.Module):
    """Local Q-function q_i(o_i, a_i).

    Discrete : outputs Q-values for every action; ``forward`` gathers the value
               of the taken action (and ``q_values`` returns the full vector).
    Continuous: concatenates the action to the observation and outputs a scalar.
    """

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int,
                 discrete: bool, hidden_dim: int = 256,
                 param_sharing: bool = True, use_agent_id: bool = True):
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim
        if discrete:
            self.net = LocalValueNet(n_agents, obs_dim, action_dim, hidden_dim,
                                     param_sharing, use_agent_id)
        else:
            self.net = LocalValueNet(n_agents, obs_dim + action_dim, 1,
                                     hidden_dim, param_sharing, use_agent_id)

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Discrete only: full Q-vector ``[*, n_agents, action_dim]``."""
        assert self.discrete
        return self.net(obs)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return q_i(o_i, a_i) with shape ``[*, n_agents]``.

        ``action`` is integer indices ``[*, n_agents]`` (discrete) or
        continuous actions ``[*, n_agents, action_dim]`` (continuous).
        """
        if self.discrete:
            qvals = self.net(obs)                       # [*, n, A]
            idx = action.long().unsqueeze(-1)           # [*, n, 1]
            return qvals.gather(-1, idx).squeeze(-1)     # [*, n]
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)                  # [*, n]


class LocalVNet(nn.Module):
    """Local value function v_i(o_i) -> ``[*, n_agents]``."""

    def __init__(self, n_agents: int, obs_dim: int, hidden_dim: int = 256,
                 param_sharing: bool = True, use_agent_id: bool = True):
        super().__init__()
        self.net = LocalValueNet(n_agents, obs_dim, 1, hidden_dim,
                                 param_sharing, use_agent_id)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
