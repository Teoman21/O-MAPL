"""Checkpoint/resume correctness.

A preempted HPC job must resume *exactly* where it stopped. This test verifies
that: training 2N steps in one shot produces bit-identical weights to training
N steps, checkpointing, then loading into a fresh learner and training N more.

Run: ``python tests/test_checkpoint.py``  (no pytest needed).
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omapl.algos import build_algo
from omapl.data.preference_dataset import DataSpec, PreferenceDataset
from omapl.utils.checkpoint import load_checkpoint, save_checkpoint
from omapl.utils.config import Config
from omapl.utils.torch_utils import set_seed


def _make_dataset(seed=7, n_traj=12, T=6):
    g = np.random.default_rng(seed)
    N, OD, SD, AD = 3, 5, 6, 4
    spec = DataSpec(n_agents=N, obs_dim=OD, state_dim=SD, action_dim=AD, discrete=True)
    trajs = []
    for _ in range(n_traj):
        trajs.append({
            "obs": g.standard_normal((T, N, OD)).astype(np.float32),
            "next_obs": g.standard_normal((T, N, OD)).astype(np.float32),
            "actions": g.integers(0, AD, (T, N)).astype(np.float32),
            "state": g.standard_normal((T, SD)).astype(np.float32),
            "next_state": g.standard_normal((T, SD)).astype(np.float32),
            "avail": np.ones((T, N, AD), np.float32),
            "next_avail": np.ones((T, N, AD), np.float32),
            "done": np.zeros(T, np.float32),
        })
    pairs = g.integers(0, n_traj, (20, 2))
    labels = g.integers(0, 2, 20).astype(np.float32)
    return PreferenceDataset(trajs, pairs, labels, spec), spec


def _cfg(spec) -> Config:
    return Config(algo="omapl", device="cpu", discrete=True,
                  n_agents=spec.n_agents, obs_dim=spec.obs_dim,
                  state_dim=spec.state_dim, action_dim=spec.action_dim,
                  agent_hidden_dim=32, mixer_hidden_dim=16, batch_size=8)


def _flat_params(algo):
    return torch.cat([p.detach().reshape(-1)
                      for m in vars(algo).values()
                      if isinstance(m, torch.nn.Module)
                      for p in m.parameters()])


def _run(algo, ds, cfg, rng, n_steps):
    for _ in range(n_steps):
        algo.update(ds.sample_batch(cfg.batch_size, device=cfg.device, rng=rng))


def test_resume_is_bit_identical():
    ds, spec = _make_dataset()
    cfg = _cfg(spec)
    N = 15

    # --- Reference: 2N steps straight through -------------------------------
    set_seed(0)
    rng_ref = np.random.default_rng(123)
    ref = build_algo(cfg)
    _run(ref, ds, cfg, rng_ref, 2 * N)
    ref_params = _flat_params(ref)

    # --- Split: N steps, checkpoint, resume into a fresh learner, N more -----
    set_seed(0)                          # identical init to `ref`
    rng_a = np.random.default_rng(123)
    a = build_algo(cfg)
    _run(a, ds, cfg, rng_a, N)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "checkpoint.pt")
        save_checkpoint(path, a, step=N, best=-1.0, rng=rng_a, cfg_dict=cfg.to_dict())

        set_seed(999)                    # deliberately different: must be overridden
        b = build_algo(cfg)
        rng_b = np.random.default_rng(999)
        step, best = load_checkpoint(path, b, rng_b, cfg.device)
        assert step == N, f"restored step {step} != {N}"
        assert best == -1.0
        _run(b, ds, cfg, rng_b, N)

    b_params = _flat_params(b)
    max_diff = (ref_params - b_params).abs().max().item()
    assert max_diff == 0.0, f"resumed weights diverge from reference (max |Δ|={max_diff})"
    print(f"test_resume_is_bit_identical: OK  (max |Δ|={max_diff})")


def test_atomic_write_leaves_no_tmp():
    ds, spec = _make_dataset()
    cfg = _cfg(spec)
    set_seed(1)
    algo = build_algo(cfg)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "checkpoint.pt")
        save_checkpoint(path, algo, step=5, best=0.0,
                        rng=np.random.default_rng(0), cfg_dict=cfg.to_dict())
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp"), "temp file not cleaned up"
    print("test_atomic_write_leaves_no_tmp: OK")


if __name__ == "__main__":
    test_resume_is_bit_identical()
    test_atomic_write_leaves_no_tmp()
    print("\nAll checkpoint tests passed.")
