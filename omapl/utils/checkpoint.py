"""Resumable checkpointing for preemptible / time-limited (SLURM) training.

An HPC GPU job can be killed at any time — preempted by a higher-priority job
or hitting the partition wall-clock limit. To survive that, training must be
able to stop and resume *exactly* where it left off. This module snapshots the
**entire** learner state, not just the model weights:

  * every ``nn.Module`` on the learner (online q/v/mixer/policy **and** the
    lagging target nets ``*_tgt``) — the targets matter because they trail the
    online nets during training, so restoring them from the online copy (as
    ``algo.load_state_dict`` does for eval) would not be a bit-exact resume;
  * every ``torch.optim.Optimizer`` (Adam's moment buffers);
  * the training step counter, best eval score, and the RNG state of the
    NumPy ``Generator`` used for batch sampling plus the global
    python/NumPy/torch(+CUDA) RNGs — so the sampled batch stream continues
    unchanged.

Writes are **atomic** (temp file + ``os.replace``) so a kill mid-write cannot
corrupt the checkpoint. The collection is generic (it introspects the learner's
attributes), so it works unchanged for O-MAPL, IPL-VDN, IIPL and BC.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _modules(algo) -> Dict[str, nn.Module]:
    return {k: v for k, v in vars(algo).items() if isinstance(v, nn.Module)}


def _optims(algo) -> Dict[str, torch.optim.Optimizer]:
    return {k: v for k, v in vars(algo).items()
            if isinstance(v, torch.optim.Optimizer)}


def _rng_state(rng: Optional[np.random.Generator]) -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if rng is not None:
        state["sampler"] = rng.bit_generator.state
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Dict[str, Any], rng: Optional[np.random.Generator]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(_as_byte_cpu(state["torch"]))
    if rng is not None and "sampler" in state:
        rng.bit_generator.state = state["sampler"]
    if torch.cuda.is_available() and "cuda" in state:
        try:
            torch.cuda.set_rng_state_all([_as_byte_cpu(s) for s in state["cuda"]])
        except Exception:
            pass  # e.g. checkpoint made on a different GPU count; not fatal


def _as_byte_cpu(t: torch.Tensor) -> torch.Tensor:
    # torch RNG state must be a CPU ByteTensor regardless of where it was loaded.
    return t.detach().to("cpu", torch.uint8).contiguous()


def save_checkpoint(path: str, algo, step: int, best: float,
                    rng: Optional[np.random.Generator] = None,
                    cfg_dict: Optional[Dict[str, Any]] = None) -> None:
    """Atomically write the full training state to ``path``."""
    payload = {
        "step": int(step),
        "best": float(best),
        "modules": {k: m.state_dict() for k, m in _modules(algo).items()},
        "optims": {k: o.state_dict() for k, o in _optims(algo).items()},
        "rng": _rng_state(rng),
        "cfg": cfg_dict,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on POSIX: a killed job never sees a half file


def load_checkpoint(path: str, algo, rng: Optional[np.random.Generator] = None,
                    device="cpu") -> Tuple[int, float]:
    """Restore learner + RNG from ``path``. Returns ``(step, best)``.

    ``step`` is the number of steps already completed, so the training loop
    should resume at ``step + 1``.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mods = _modules(algo)
    for k, sd in ckpt["modules"].items():
        if k in mods:
            mods[k].load_state_dict(sd)
    opts = _optims(algo)
    for k, sd in ckpt["optims"].items():
        if k in opts:
            opts[k].load_state_dict(sd)
    if "rng" in ckpt and ckpt["rng"] is not None:
        _restore_rng(ckpt["rng"], rng)
    return int(ckpt["step"]), float(ckpt["best"])
