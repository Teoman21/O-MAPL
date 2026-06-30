"""Baseline learners: IIPL, BC.

These share the network modules with O-MAPL but differ in how (or whether)
local values are aggregated and how the policy is trained:

* **IIPL** (Independent Inverse Preference Learning): single-agent IPL applied
  *independently per agent*. There is no mixing network — each agent has its
  own inverse-soft-Bellman reward R_i, its own Bradley-Terry preference loss,
  its own Extreme-V loss, and its own weighted-BC policy.
* **BC** (Behavior Cloning): supervised imitation of the *preferred* trajectory
  of each pair, with no value learning at all.

(**IPL-VDN** lives in :mod:`omapl.algos.omapl` as a one-line subclass of O-MAPL,
since it is exactly O-MAPL with a VDN sum mixer.)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..networks.agent import LocalQNet, LocalVNet
from ..networks.policy import build_policy
from ..utils.config import Config
from ..utils.torch_utils import soft_update, hard_update, grad_norm_clip


class IIPL:
    name = "iipl"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.gamma, self.beta = cfg.gamma, cfg.beta
        n, od, ad = cfg.n_agents, cfg.obs_dim, cfg.action_dim
        net_kw = dict(n_agents=n, hidden_dim=cfg.agent_hidden_dim,
                      param_sharing=cfg.param_sharing, use_agent_id=cfg.use_agent_id)
        self.q_net = LocalQNet(obs_dim=od, action_dim=ad, discrete=cfg.discrete,
                               **net_kw).to(self.device)
        self.v_net = LocalVNet(obs_dim=od, **net_kw).to(self.device)
        self.v_net_tgt = LocalVNet(obs_dim=od, **net_kw).to(self.device)
        self.policy = build_policy(cfg.discrete, obs_dim=od, action_dim=ad,
                                   **net_kw).to(self.device)
        hard_update(self.v_net_tgt, self.v_net)
        self.opt_q = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.opt_v = torch.optim.Adam(self.v_net.parameters(), lr=cfg.lr)
        self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

    def update(self, batch) -> Dict[str, float]:
        m = {}
        mask = batch["mask"]                             # [B,2,T]
        maskn = mask.unsqueeze(-1)                        # [B,2,T,1]
        denom = maskn.sum().clamp(min=1.0)

        # --- per-agent preference loss (updates q) ---
        q_i = self.q_net(batch["obs"], batch["actions"])             # [B,2,T,n]
        with torch.no_grad():
            v_next_i = self.v_net_tgt(batch["next_obs"])             # [B,2,T,n]
        R_i = q_i - self.gamma * v_next_i * (1.0 - batch["done"]).unsqueeze(-1)
        S_i = (R_i * maskn).sum(dim=2)                   # [B,2,n]
        logp = F.log_softmax(S_i, dim=1)                 # [B,2,n]
        label = batch["label"].view(-1, 1)               # [B,1]
        bt = -(label * logp[:, 0] + (1.0 - label) * logp[:, 1]).mean()
        phi = (-0.5 * R_i.pow(2) + R_i)
        reg = (phi * maskn).sum() / denom
        loss_q = bt - self.cfg.reg_coef * reg
        self.opt_q.zero_grad(set_to_none=True)
        loss_q.backward()
        grad_norm_clip(self.q_net.parameters(), self.cfg.grad_clip)
        self.opt_q.step()
        m["loss_pref"] = float(bt); m["reg"] = float(reg)

        # --- per-agent Extreme-V (updates v) ---
        with torch.no_grad():
            q_i_d = self.q_net(batch["obs"], batch["actions"])
        v_i = self.v_net(batch["obs"])
        z = (q_i_d - v_i) / self.beta
        gumbel = torch.exp(z.clamp(max=self.cfg.exp_clip)) - z - 1.0
        loss_v = (gumbel * maskn).sum() / denom
        self.opt_v.zero_grad(set_to_none=True)
        loss_v.backward()
        grad_norm_clip(self.v_net.parameters(), self.cfg.grad_clip)
        self.opt_v.step()
        m["loss_v"] = float(loss_v)

        # --- per-agent weighted BC (updates policy) ---
        with torch.no_grad():
            q_i_d = self.q_net(batch["obs"], batch["actions"])
            v_i_d = self.v_net(batch["obs"])
            w = torch.exp(((q_i_d - v_i_d) / self.beta).clamp(
                max=np.log(self.cfg.adv_weight_clip))).clamp(max=self.cfg.adv_weight_clip)
        logp_pi = self.policy.log_prob(batch["obs"], batch["actions"],
                                       batch.get("avail"))           # [B,2,T,n]
        loss_pi = -((w * logp_pi * maskn).sum() / denom)
        self.opt_pi.zero_grad(set_to_none=True)
        loss_pi.backward()
        grad_norm_clip(self.policy.parameters(), self.cfg.grad_clip)
        self.opt_pi.step()
        m["loss_pi"] = float(loss_pi)

        soft_update(self.v_net_tgt, self.v_net, self.cfg.tau)
        return m

    @torch.no_grad()
    def act(self, obs, avail=None, deterministic=True):
        return self.policy.act(obs, avail, deterministic=deterministic)

    def state_dict(self):
        return {"q": self.q_net.state_dict(), "v": self.v_net.state_dict(),
                "policy": self.policy.state_dict()}

    def load_state_dict(self, sd):
        self.q_net.load_state_dict(sd["q"]); self.v_net.load_state_dict(sd["v"])
        self.policy.load_state_dict(sd["policy"]); hard_update(self.v_net_tgt, self.v_net)


class BC:
    """Behavior cloning of the preferred trajectory in each pair."""
    name = "bc"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.policy = build_policy(
            cfg.discrete, n_agents=cfg.n_agents, obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim, hidden_dim=cfg.agent_hidden_dim,
            param_sharing=cfg.param_sharing, use_agent_id=cfg.use_agent_id
        ).to(self.device)
        self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

    def update(self, batch) -> Dict[str, float]:
        mask = batch["mask"]                             # [B,2,T]
        # Per-trajectory "is preferred" weight: [label, 1 - label].
        label = batch["label"]
        pref_w = torch.stack([label, 1.0 - label], dim=1)            # [B,2]
        w = pref_w.unsqueeze(-1) * mask                  # [B,2,T]
        logp = self.policy.log_prob(batch["obs"], batch["actions"],
                                    batch.get("avail")).sum(dim=-1)  # [B,2,T]
        loss = -((w * logp).sum() / w.sum().clamp(min=1.0))
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_clip(self.policy.parameters(), self.cfg.grad_clip)
        self.opt_pi.step()
        return {"loss_pi": float(loss)}

    @torch.no_grad()
    def act(self, obs, avail=None, deterministic=True):
        return self.policy.act(obs, avail, deterministic=deterministic)

    def state_dict(self):
        return {"policy": self.policy.state_dict()}

    def load_state_dict(self, sd):
        self.policy.load_state_dict(sd["policy"])
