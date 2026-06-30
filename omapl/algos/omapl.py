"""O-MAPL learner: Offline Multi-Agent Preference Learning (Algorithm 1).

One training step performs the three alternating updates of Algorithm 1:

  (4) maximise the preference likelihood  L(psi_q, psi_v, theta)
  (5) minimise the Extreme-V loss         J(psi_v)
  (6) maximise the local weighted-BC loss Psi(omega_i)

Key quantities (Section 5):
  R(o,a,o') = M_theta[q(o,a)] - gamma * M_theta[v(o')]        (inverse soft Bellman)
  L = sum_pairs [ sum_{sigma1} R - logsumexp(sum_{sigma1} R, sum_{sigma2} R) ]
      + reg_coef * sum_P phi(R),     phi(x) = -1/2 x^2 + x     (chi^2 regulariser)
  J = E[ exp((Q_tot - V_tot)/beta) - (Q_tot - V_tot)/beta ] - 1
  Psi = E_{o,a} [ exp((Q_tot - V_tot)/beta) * log pi_i(a_i|o_i) ]

Gradient routing (Algorithm 1):
  * L updates psi_q and theta. The bootstrap term M_theta[v(o')] uses a
    *target* V-net + target mixer, so it carries no gradient (the tau soft
    update of Table 6 tracks it). theta therefore updates through the Q path.
  * J updates psi_v only: Q_tot is detached and the V-mixer weights are treated
    as constants (``v_value_only``).
  * Psi updates omega only: the advantage weight exp((Q_tot - V_tot)/beta) is
    detached.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..networks.agent import LocalQNet, LocalVNet
from ..networks.mixer import Mixer, VDNMixer
from ..networks.policy import build_policy
from ..utils.config import Config
from ..utils.torch_utils import soft_update, hard_update, grad_norm_clip


class OMAPL:
    name = "omapl"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.gamma = cfg.gamma
        self.beta = cfg.beta

        n, od, sd, ad = cfg.n_agents, cfg.obs_dim, cfg.state_dim, cfg.action_dim
        net_kw = dict(n_agents=n, hidden_dim=cfg.agent_hidden_dim,
                      param_sharing=cfg.param_sharing,
                      use_agent_id=cfg.use_agent_id)

        self.q_net = LocalQNet(obs_dim=od, action_dim=ad, discrete=cfg.discrete,
                               **net_kw).to(self.device)
        self.v_net = LocalVNet(obs_dim=od, **net_kw).to(self.device)
        self.policy = build_policy(cfg.discrete, obs_dim=od, action_dim=ad,
                                   **net_kw).to(self.device)
        self.mixer = self._build_mixer().to(self.device)

        # Target networks for the bootstrap term M_theta[v(o')] in R.
        self.v_net_tgt = LocalVNet(obs_dim=od, **net_kw).to(self.device)
        self.mixer_tgt = self._build_mixer().to(self.device)
        hard_update(self.v_net_tgt, self.v_net)
        hard_update(self.mixer_tgt, self.mixer)

        # Optimisers, one per Algorithm-1 update.
        self.opt_q = torch.optim.Adam(
            list(self.q_net.parameters()) + list(self.mixer.parameters()), lr=cfg.lr)
        self.opt_v = torch.optim.Adam(self.v_net.parameters(), lr=cfg.lr)
        self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

    def _build_mixer(self):
        return Mixer(self.cfg.n_agents, self.cfg.state_dim, self.cfg.action_dim,
                     self.cfg.discrete, hidden_dim=self.cfg.mixer_hidden_dim,
                     q_use_action=self.cfg.mixer_q_use_action)

    # ------------------------------------------------------------------ #
    # Value-mixing helpers (operate on full [B, 2, T, ...] batches)
    # ------------------------------------------------------------------ #
    def _local_q(self, obs, actions):
        return self.q_net(obs, actions)                 # [B,2,T,n]

    def _q_tot(self, obs, actions, state):
        return self.mixer.mix_q(self._local_q(obs, actions), state, actions)

    def _v_tot(self, obs, state, target: bool = False):
        net = self.v_net_tgt if target else self.v_net
        mix = self.mixer_tgt if target else self.mixer
        return mix.mix_v(net(obs), state)

    def _avail(self, batch, key):
        return batch.get(key, None)

    # ------------------------------------------------------------------ #
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics.update(self._update_preference(batch))
        metrics.update(self._update_extreme_v(batch))
        metrics.update(self._update_policy(batch))
        soft_update(self.v_net_tgt, self.v_net, self.cfg.tau)
        soft_update(self.mixer_tgt, self.mixer, self.cfg.tau)
        return metrics

    # --- (4) preference loss: updates psi_q and theta ------------------ #
    def _update_preference(self, batch) -> Dict[str, float]:
        mask = batch["mask"]                            # [B,2,T]
        # Q_tot at (o,a): online (grad -> psi_q, theta).
        q_tot = self._q_tot(batch["obs"], batch["actions"], batch["state"])
        # gamma * V_tot(o'): target nets, no gradient; zero past terminals.
        with torch.no_grad():
            v_next = self._v_tot(batch["next_obs"], batch["next_state"], target=True)
        R = q_tot - self.gamma * v_next * (1.0 - batch["done"])   # [B,2,T]

        # Bradley-Terry over per-trajectory reward sums.
        S = (R * mask).sum(dim=2)                        # [B,2]; idx0 = sigma1
        logp = F.log_softmax(S, dim=1)                   # [B,2]
        label = batch["label"]                           # P(sigma1 preferred)
        bt_loss = -(label * logp[:, 0] + (1.0 - label) * logp[:, 1]).mean()

        # chi^2 regulariser phi(x) = -1/2 x^2 + x, added to the maximised obj.
        phi = (-0.5 * R.pow(2) + R)
        reg = (phi * mask).sum() / mask.sum().clamp(min=1.0)
        loss = bt_loss - self.cfg.reg_coef * reg

        self.opt_q.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_clip(list(self.q_net.parameters()) + list(self.mixer.parameters()),
                       self.cfg.grad_clip)
        self.opt_q.step()

        with torch.no_grad():
            pred = (S[:, 0] > S[:, 1]).float()
            tgt = (label > 0.5).float()
            acc = ((pred == tgt) | (label == 0.5)).float().mean()
        return {"loss_pref": float(bt_loss), "reg": float(reg),
                "pref_acc": float(acc), "R_mean": float((R * mask).sum() / mask.sum())}

    # --- (5) Extreme-V loss: updates psi_v ----------------------------- #
    def _update_extreme_v(self, batch) -> Dict[str, float]:
        mask = batch["mask"]
        with torch.no_grad():
            q_tot = self._q_tot(batch["obs"], batch["actions"], batch["state"])
        # V_tot with mixer weights treated as constants -> grad only to psi_v.
        local_v = self.v_net(batch["obs"])
        v_tot = self.mixer.v_value_only(local_v, batch["state"])

        z = (q_tot - v_tot) / self.beta                  # [B,2,T]
        gumbel = torch.exp(z.clamp(max=self.cfg.exp_clip)) - z - 1.0
        loss_v = (gumbel * mask).sum() / mask.sum().clamp(min=1.0)

        self.opt_v.zero_grad(set_to_none=True)
        loss_v.backward()
        grad_norm_clip(self.v_net.parameters(), self.cfg.grad_clip)
        self.opt_v.step()
        return {"loss_v": float(loss_v), "V_mean": float((v_tot * mask).sum() / mask.sum())}

    # --- (6) local weighted-BC loss: updates omega --------------------- #
    def _update_policy(self, batch) -> Dict[str, float]:
        mask = batch["mask"]
        with torch.no_grad():
            q_tot = self._q_tot(batch["obs"], batch["actions"], batch["state"])
            v_tot = self._v_tot(batch["obs"], batch["state"], target=False)
            weight = torch.exp(((q_tot - v_tot) / self.beta).clamp(
                max=np.log(self.cfg.adv_weight_clip)))
            weight = weight.clamp(max=self.cfg.adv_weight_clip)

        logp = self.policy.log_prob(batch["obs"], batch["actions"],
                                    self._avail(batch, "avail"))   # [B,2,T,n]
        logp_sum = logp.sum(dim=-1)                      # [B,2,T] (sum over agents)
        loss_pi = -((weight * logp_sum * mask).sum() / mask.sum().clamp(min=1.0))

        self.opt_pi.zero_grad(set_to_none=True)
        loss_pi.backward()
        grad_norm_clip(self.policy.parameters(), self.cfg.grad_clip)
        self.opt_pi.step()
        return {"loss_pi": float(loss_pi), "adv_weight_mean": float(
            (weight * mask).sum() / mask.sum())}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, obs: torch.Tensor, avail: Optional[torch.Tensor] = None,
            deterministic: bool = True) -> torch.Tensor:
        return self.policy.act(obs, avail, deterministic=deterministic)

    @torch.no_grad()
    def recovered_reward(self, batch) -> torch.Tensor:
        """R(o,a,o') = M_theta[q] - gamma M_theta[v(o')] (Table 7 diagnostic)."""
        q_tot = self._q_tot(batch["obs"], batch["actions"], batch["state"])
        v_next = self._v_tot(batch["next_obs"], batch["next_state"], target=True)
        return q_tot - self.gamma * v_next * (1.0 - batch["done"])

    def state_dict(self):
        return {"q": self.q_net.state_dict(), "v": self.v_net.state_dict(),
                "mixer": self.mixer.state_dict(), "policy": self.policy.state_dict()}

    def load_state_dict(self, sd):
        self.q_net.load_state_dict(sd["q"]); self.v_net.load_state_dict(sd["v"])
        self.mixer.load_state_dict(sd["mixer"]); self.policy.load_state_dict(sd["policy"])
        hard_update(self.v_net_tgt, self.v_net); hard_update(self.mixer_tgt, self.mixer)


class IPL_VDN(OMAPL):
    """IPL-VDN baseline: identical to O-MAPL but the mixer is a plain sum
    (VDN, Sunehag et al. 2017) instead of a learned hypernetwork mixer."""
    name = "ipl_vdn"

    def _build_mixer(self):
        return VDNMixer(self.cfg.n_agents)
