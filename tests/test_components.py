"""Unit tests for the core O-MAPL components.

Run with:  python -m pytest tests/  (or)  python tests/test_components.py
"""
import os
import sys
import copy

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omapl.networks.mixer import Mixer, VDNMixer
from omapl.networks.agent import LocalQNet, LocalVNet
from omapl.algos import build_algo
from omapl.utils.config import Config


def _tiny_cfg(algo="omapl"):
    return Config(algo=algo, n_agents=3, obs_dim=5, state_dim=6, action_dim=4,
                  discrete=True, agent_hidden_dim=16, mixer_hidden_dim=8,
                  batch_size=4, device="cpu")


def _random_batch(cfg, T=7, device="cpu"):
    B, n, od, sd, ad = cfg.batch_size, cfg.n_agents, cfg.obs_dim, cfg.state_dim, cfg.action_dim
    g = torch.Generator().manual_seed(0)
    rand = lambda *s: torch.rand(*s, generator=g)
    batch = {
        "obs": rand(B, 2, T, n, od), "next_obs": rand(B, 2, T, n, od),
        "actions": torch.randint(0, ad, (B, 2, T, n), generator=g).float(),
        "state": rand(B, 2, T, sd), "next_state": rand(B, 2, T, sd),
        "avail": torch.ones(B, 2, T, n, ad),
        "done": torch.zeros(B, 2, T), "mask": torch.ones(B, 2, T),
        "label": (rand(B) > 0.5).float(),
    }
    return {k: v.to(device) for k, v in batch.items()}


def test_mixer_shapes_and_nonneg():
    m = Mixer(n_agents=3, state_dim=6, action_dim=4, discrete=True,
              hidden_dim=8, q_use_action=True)
    state = torch.rand(5, 6)
    action = torch.randint(0, 4, (5, 3))
    local_q = torch.rand(5, 3)
    w_q, b_q = m.q_params(state, action)
    assert w_q.shape == (5, 3) and b_q.shape == (5, 1)
    assert (w_q >= 0).all(), "Q-mixer weights must be non-negative (monotonic)."
    q_tot = m.mix_q(local_q, state, action)
    assert q_tot.shape == (5,)
    print("test_mixer_shapes_and_nonneg: OK")


def test_mixer_linear_in_inputs():
    """Linearity in q underpins the convexity result (Prop. 4.1)."""
    m = Mixer(3, 6, 4, discrete=True, hidden_dim=8, q_use_action=True)
    state = torch.rand(5, 6)
    action = torch.randint(0, 4, (5, 3))
    q1, q2 = torch.rand(5, 3), torch.rand(5, 3)
    a = 0.37
    lhs = m.mix_q(a * q1 + (1 - a) * q2, state, action)
    rhs = a * m.mix_q(q1, state, action) + (1 - a) * m.mix_q(q2, state, action)
    assert torch.allclose(lhs, rhs, atol=1e-5), "Mixing must be linear in q."
    print("test_mixer_linear_in_inputs: OK")


def test_extreme_v_recovers_logsumexp():
    """Minimising the Gumbel/Extreme-V loss drives V -> beta*logsumexp(Q/beta).

    Single 'agent', identity mixing: V_tot = v, Q_tot = q over a discrete set.
    """
    torch.manual_seed(0)
    beta = 1.0
    q = torch.tensor([0.5, 2.0, -1.0, 1.0, 0.3])          # "Q-values" of samples
    target = beta * torch.logsumexp(q / beta - np.log(len(q)), dim=0)  # softmax-mean
    v = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.05)
    for _ in range(3000):
        z = (q - v) / beta
        loss = (torch.exp(z) - z - 1.0).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert abs(v.item() - target.item()) < 0.05, (v.item(), target.item())
    print(f"test_extreme_v_recovers_logsumexp: OK (v={v.item():.3f}, "
          f"target={target.item():.3f})")


def test_gradient_routing():
    """Algorithm 1 routing: preference loss updates only (q, mixer); Extreme-V
    only v; WBC only policy."""
    cfg = _tiny_cfg("omapl")
    algo = build_algo(cfg)
    batch = _random_batch(cfg)

    def snap(mod):
        return [p.detach().clone() for p in mod.parameters()]

    def changed(before, mod):
        return any(not torch.equal(b, p) for b, p in zip(before, mod.parameters()))

    q0, mix0, v0, pi0 = (snap(algo.q_net), snap(algo.mixer),
                         snap(algo.v_net), snap(algo.policy))
    algo._update_preference(batch)
    assert changed(q0, algo.q_net), "preference loss must update q"
    assert changed(mix0, algo.mixer), "preference loss must update mixer (theta)"
    assert not changed(v0, algo.v_net), "preference loss must NOT update v"
    assert not changed(pi0, algo.policy), "preference loss must NOT update policy"

    v0, q1, mix1 = snap(algo.v_net), snap(algo.q_net), snap(algo.mixer)
    algo._update_extreme_v(batch)
    assert changed(v0, algo.v_net), "Extreme-V must update v"
    assert not changed(q1, algo.q_net), "Extreme-V must NOT update q"
    assert not changed(mix1, algo.mixer), "Extreme-V must NOT update mixer"

    pi0, q1, v1 = snap(algo.policy), snap(algo.q_net), snap(algo.v_net)
    algo._update_policy(batch)
    assert changed(pi0, algo.policy), "WBC must update policy"
    assert not changed(q1, algo.q_net) and not changed(v1, algo.v_net)
    print("test_gradient_routing: OK")


def test_all_algos_step():
    for name in ["omapl", "ipl_vdn", "iipl", "bc"]:
        cfg = _tiny_cfg(name)
        algo = build_algo(cfg)
        m = algo.update(_random_batch(cfg))
        assert all(np.isfinite(list(m.values()))), (name, m)
        # decentralised action shape check
        obs = torch.rand(cfg.n_agents, cfg.obs_dim)
        avail = torch.ones(cfg.n_agents, cfg.action_dim)
        a = algo.act(obs, avail)
        assert a.shape == (cfg.n_agents,)
        print(f"test_all_algos_step[{name}]: OK  metrics={ {k: round(v,3) for k,v in m.items()} }")


if __name__ == "__main__":
    test_mixer_shapes_and_nonneg()
    test_mixer_linear_in_inputs()
    test_extreme_v_recovers_logsumexp()
    test_gradient_routing()
    test_all_algos_step()
    print("\nAll component tests passed.")
