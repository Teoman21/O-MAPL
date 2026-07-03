"""Training entry point for O-MAPL and baselines.

Usage:
    python -m omapl.train --config configs/synthetic.yaml
    python -m omapl.train --config configs/synthetic.yaml algo=ipl_vdn seed=1
    python -m omapl.train --data_path data/synthetic.pkl algo=omapl env=synthetic
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from typing import Optional

import numpy as np
import torch

from .algos import build_algo
from .data import PreferenceDataset
from .envs import build_env
from .evaluate import evaluate_policy
from .utils.checkpoint import load_checkpoint, save_checkpoint
from .utils.config import Config
from .utils.logger import Logger
from .utils.torch_utils import set_seed

# Set by SIGUSR1/SIGTERM (SLURM sends these before preemption / time limit) so
# the training loop can checkpoint at a safe point and, under SLURM, requeue.
_STOP = {"flag": False}


def _install_signal_handlers() -> None:
    def _handler(signum, frame):  # noqa: ANN001
        _STOP["flag"] = True
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # e.g. not in main thread
            pass


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train O-MAPL / baselines.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.data_path:
        cfg.data_path = args.data_path
    cfg.update_from_overrides(args.overrides)
    return cfg


def fill_dims_from_dataset(cfg: Config, ds: PreferenceDataset) -> None:
    cfg.n_agents = ds.spec.n_agents
    cfg.obs_dim = ds.spec.obs_dim
    cfg.state_dim = ds.spec.state_dim
    cfg.action_dim = ds.spec.action_dim
    cfg.discrete = ds.spec.discrete


def train(cfg: Config) -> None:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
    set_seed(cfg.seed)

    if not cfg.data_path or not os.path.exists(cfg.data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{cfg.data_path}'. Generate one first, e.g.\n"
            f"  python scripts/make_synthetic_data.py --out data/synthetic.pkl")
    ds = PreferenceDataset.load(cfg.data_path)
    fill_dims_from_dataset(cfg, ds)

    algo = build_algo(cfg)
    logger = Logger(cfg.log_dir, f"{cfg.exp_name}-{cfg.algo}-{cfg.task}-s{cfg.seed}")
    rng = np.random.default_rng(cfg.seed)

    # Resume from a full-state checkpoint if one exists (preemptible HPC jobs).
    ckpt_path = os.path.join(logger.dir, "checkpoint.pt")
    start_step, best = 0, -np.inf
    if cfg.resume and os.path.exists(ckpt_path):
        start_step, best = load_checkpoint(ckpt_path, algo, rng, cfg.device)
        print(f"[resume] loaded {ckpt_path}: continuing from step {start_step + 1} "
              f"(best so far {best:.3f})")
    _install_signal_handlers()

    try:
        env = build_env(cfg)
    except Exception as e:  # eval env may be unavailable (e.g. no StarCraft)
        env = None
        print(f"[warn] evaluation env unavailable ({e}); training without eval.")

    if env is not None and cfg.discrete:
        env_dims = (getattr(env, "n_agents", None), getattr(env, "obs_dim", None),
                    getattr(env, "state_dim", None), getattr(env, "action_dim", None))
        ds_dims = (cfg.n_agents, cfg.obs_dim, cfg.state_dim, cfg.action_dim)
        if None not in env_dims and tuple(env_dims) != tuple(ds_dims):
            raise RuntimeError(
                "Eval env dims do not match the offline dataset:\n"
                f"  env  (n_agents,obs,state,act) = {tuple(env_dims)}\n"
                f"  data (n_agents,obs,state,act) = {tuple(ds_dims)}\n"
                "The trained policy is sized from the dataset, so these MUST agree. "
                "This usually means the SMACv2 capability config differs from the "
                "one used to generate the dataset (obs flags / unit counts). "
                "Check omapl/envs/smac_wrapper.py against the dataset's source config.")

    print(f"Training {cfg.algo} on {cfg.task}: n_agents={cfg.n_agents}, "
          f"obs_dim={cfg.obs_dim}, state_dim={cfg.state_dim}, "
          f"action_dim={cfg.action_dim}, discrete={cfg.discrete}, "
          f"|P|={len(ds)} pairs, device={cfg.device}")

    cfg_dict = cfg.to_dict()
    for step in range(start_step + 1, cfg.n_train_steps + 1):
        batch = ds.sample_batch(cfg.batch_size, device=cfg.device, rng=rng)
        metrics = algo.update(batch)

        if step % 200 == 0:
            logger.log(step, metrics, prefix="train/")
            Logger.print(step, metrics)

        if env is not None and step % cfg.eval_every == 0:
            ev = evaluate_policy(algo, env, cfg.eval_episodes, cfg.device)
            logger.log(step, ev, prefix="eval/")
            Logger.print(step, {f"eval_{k}": v for k, v in ev.items()})
            score = ev.get("win_rate", ev["return_mean"])
            if score > best:
                best = score
                torch.save(algo.state_dict(),
                           os.path.join(logger.dir, "best.pt"))

        # Periodic full-state checkpoint, and an immediate one if SLURM has
        # signalled an incoming preemption / time limit.
        if step % cfg.ckpt_every == 0 or _STOP["flag"]:
            save_checkpoint(ckpt_path, algo, step, best, rng, cfg_dict)
        if _STOP["flag"]:
            _handle_preemption(step, ckpt_path, cfg)
            return  # (only reached if not requeued, e.g. a local run)

    torch.save(algo.state_dict(), os.path.join(logger.dir, "final.pt"))
    save_checkpoint(ckpt_path, algo, cfg.n_train_steps, best, rng, cfg_dict)
    logger.close()
    print(f"Done. Best eval score: {best:.3f}. Artifacts in {logger.dir}")


def _handle_preemption(step: int, ckpt_path: str, cfg: Config) -> None:
    """Checkpoint already written; requeue this SLURM job so it resumes."""
    job_id = os.environ.get("SLURM_JOB_ID")
    print(f"[signal] caught preemption/timeout at step {step}; "
          f"checkpoint saved to {ckpt_path}", flush=True)
    if cfg.requeue_on_signal and job_id:
        print(f"[signal] requeueing SLURM job {job_id}", flush=True)
        os.system(f"scontrol requeue {job_id}")
    sys.exit(0)


if __name__ == "__main__":
    train(parse_args())
