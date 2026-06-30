"""Mixing networks for O-MAPL value factorisation (Eqs. 6-7).

The global value functions are formed as a **single-layer (linear) combination**
of the local functions, with **non-negative** weights produced by hypernetworks
conditioned on the global state (and, for Q, the joint action):

    V_tot(o)   = sum_i w^v_i(o)   * v_i(o_i)      + b^v(o)
    Q_tot(o,a) = sum_i w^q_i(o,a) * q_i(o_i, a_i) + b^q(o,a)

Why linear / non-negative:
  * Linearity in the local inputs (q, v) is what makes the preference loss L
    concave in q and theta and the Extreme-V loss J convex in v
    (Propositions 4.1 / 4.2). A two-layer mixer breaks this.
  * Non-negative weights give monotonic mixing (QMIX-style) and underpin the
    global-local consistency results (Theorems 4.3 / 4.4).

The hypernetworks themselves may be multi-layer MLPs (this does *not* break the
above, which only concerns linearity in q and v). The weights and biases are
returned explicitly via :meth:`q_params` / :meth:`v_params` so the learner can
control which paths receive gradients (see :mod:`omapl.algos.omapl`).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.torch_utils import mlp


class Mixer(nn.Module):
    """QMIX-style single-layer hypernetwork mixer (shared params ``theta``)."""

    def __init__(self, n_agents: int, state_dim: int, action_dim: int,
                 discrete: bool, hidden_dim: int = 64,
                 q_use_action: bool = True):
        super().__init__()
        self.n_agents = n_agents
        self.discrete = discrete
        self.action_dim = action_dim
        self.q_use_action = q_use_action

        # Q-mixer hypernet input: global state (+ joint action).
        joint_act_dim = (n_agents * action_dim) if discrete else (n_agents * action_dim)
        q_in = state_dim + (joint_act_dim if q_use_action else 0)

        # Each hypernetwork outputs the n per-agent weights and a scalar bias.
        self.hyper_w_q = mlp(q_in, hidden_dim, n_agents, n_hidden=1)
        self.hyper_b_q = mlp(q_in, hidden_dim, 1, n_hidden=1)
        self.hyper_w_v = mlp(state_dim, hidden_dim, n_agents, n_hidden=1)
        self.hyper_b_v = mlp(state_dim, hidden_dim, 1, n_hidden=1)

    # ------------------------------------------------------------------ #
    def _joint_action_feat(self, action: torch.Tensor) -> torch.Tensor:
        """Flatten the joint action into a feature vector ``[*, n*A]``."""
        if self.discrete:
            oh = F.one_hot(action.long(), num_classes=self.action_dim).float()
            return oh.flatten(start_dim=-2)             # [*, n*A]
        return action.flatten(start_dim=-2)             # [*, n*A]

    def q_params(self, state: torch.Tensor,
                 action: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(w_q [*, n], b_q [*, 1])`` for the Q-mixer.

        Weights are made non-negative with ``abs`` (QMIX convention)."""
        x = state
        if self.q_use_action:
            assert action is not None, "Q-mixer configured to use the action."
            x = torch.cat([state, self._joint_action_feat(action)], dim=-1)
        w = self.hyper_w_q(x).abs()
        b = self.hyper_b_q(x)
        return w, b

    def v_params(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(w_v [*, n], b_v [*, 1])`` for the V-mixer."""
        w = self.hyper_w_v(state).abs()
        b = self.hyper_b_v(state)
        return w, b

    # ------------------------------------------------------------------ #
    @staticmethod
    def combine(local_vals: torch.Tensor, w: torch.Tensor,
                b: torch.Tensor) -> torch.Tensor:
        """Q_tot/V_tot = sum_i w_i * local_i + b  ->  ``[*]`` (scalar per item)."""
        return (w * local_vals).sum(dim=-1) + b.squeeze(-1)

    def mix_q(self, local_q: torch.Tensor, state: torch.Tensor,
              action: Optional[torch.Tensor]) -> torch.Tensor:
        w, b = self.q_params(state, action)
        return self.combine(local_q, w, b)

    def mix_v(self, local_v: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        w, b = self.v_params(state)
        return self.combine(local_v, w, b)

    def v_value_only(self, local_v: torch.Tensor,
                     state: torch.Tensor) -> torch.Tensor:
        """V_tot with the hypernet weights treated as constants.

        Used by the Extreme-V loss J(psi_v), where only the local value
        parameters psi_v are updated (theta is updated by the preference loss).
        """
        w, b = self.v_params(state)
        return self.combine(local_v, w.detach(), b.detach())


class VDNMixer(nn.Module):
    """Value-Decomposition Network mixer used by the IPL-VDN baseline.

    Global value is the plain sum of local values (unit weights, no bias),
    i.e. ``Q_tot = sum_i q_i`` and ``V_tot = sum_i v_i`` (Sunehag et al., 2017).
    """

    def __init__(self, n_agents: int, *args, **kwargs):
        super().__init__()
        self.n_agents = n_agents

    def q_params(self, state, action):
        b, n = state.shape[0], self.n_agents
        return (torch.ones(b, n, device=state.device),
                torch.zeros(b, 1, device=state.device))

    def v_params(self, state):
        b, n = state.shape[0], self.n_agents
        return (torch.ones(b, n, device=state.device),
                torch.zeros(b, 1, device=state.device))

    @staticmethod
    def combine(local_vals, w, b):
        return (w * local_vals).sum(dim=-1) + b.squeeze(-1)

    def mix_q(self, local_q, state, action):
        return local_q.sum(dim=-1)

    def mix_v(self, local_v, state):
        return local_v.sum(dim=-1)

    def v_value_only(self, local_v, state):
        # No mixer parameters: gradient flows only through the local values.
        return local_v.sum(dim=-1)
