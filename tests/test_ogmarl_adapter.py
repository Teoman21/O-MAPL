"""Validate the OG-MARL SMACv2 -> trajectory conversion without flashbax/SC2.

Builds a tiny fake OG-MARL experience dict (2 short episodes), runs
experience_to_arrays + trajectories_from_arrays, and checks episode slicing,
the next_* one-step shift, the done mask, available-action masks, and returns.
Run: python3 tests/test_ogmarl_adapter.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omapl.data.ogmarl_adapter import experience_to_arrays
from omapl.data.omiga_adapter import trajectories_from_arrays


def _fake_experience(T, N, obs_dim, state_dim, A, ep_ends):
    """ep_ends: list of timestep indices (inclusive) that end an episode."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(1, T, N, obs_dim)).astype(np.float32)
    actions = rng.integers(0, A, size=(1, T, N)).astype(np.float32)
    rewards = rng.normal(size=(1, T, N)).astype(np.float32)
    state = rng.normal(size=(1, T, state_dim)).astype(np.float32)
    legals = (rng.random((1, T, N, A)) > 0.3).astype(np.float32)
    legals[..., 0] = 1.0  # ensure at least one legal action
    terminals = np.zeros((1, T, N), np.float32)
    truncations = np.zeros((1, T, N), np.float32)
    for i, e in enumerate(ep_ends):
        # alternate: even episodes end as terminal, odd as truncation
        (terminals if i % 2 == 0 else truncations)[0, e, :] = 1.0
    return {
        "observations": obs, "actions": actions, "rewards": rewards,
        "terminals": terminals, "truncations": truncations,
        "infos": {"legals": legals, "state": state},
    }


def test_conversion():
    T, N, obs_dim, state_dim, A = 12, 3, 5, 7, 4
    ep_ends = [4, 11]  # episode 0: steps 0..4 (len 5); episode 1: steps 5..11 (len 7)
    exp = _fake_experience(T, N, obs_dim, state_dim, A, ep_ends)

    arr = experience_to_arrays(exp)
    assert arr["obs"].shape == (T, N, obs_dim)
    assert arr["actions"].shape == (T, N)
    assert arr["avail"].shape == (T, N, A)
    assert arr["state"].shape == (T, state_dim)
    assert arr["rewards"].shape == (T,)
    # episode boundary detection (terminal | truncation)
    assert arr["terminals"].astype(bool).tolist() == [
        bool(t in ep_ends) for t in range(T)]
    # next_obs is a one-step shift (last frame duplicated)
    assert np.allclose(arr["next_obs"][:-1], arr["obs"][1:])
    assert np.allclose(arr["next_obs"][-1], arr["obs"][-1])

    trajs, spec = trajectories_from_arrays(
        obs=arr["obs"], actions=arr["actions"], next_obs=arr["next_obs"],
        state=arr["state"], next_state=arr["next_state"],
        terminals=arr["terminals"], discrete=True,
        avail=arr["avail"], next_avail=arr["next_avail"], rewards=arr["rewards"])

    assert len(trajs) == 2, f"expected 2 episodes, got {len(trajs)}"
    assert len(trajs[0]["actions"]) == 5
    assert len(trajs[1]["actions"]) == 7
    # done is 1 only on the final step of each episode
    assert trajs[0]["done"].tolist() == [0, 0, 0, 0, 1]
    assert trajs[1]["done"].tolist() == [0, 0, 0, 0, 0, 0, 1]
    # within-episode next_obs matches the next within-episode obs
    assert np.allclose(trajs[0]["next_obs"][:-1], trajs[0]["obs"][1:])
    # return = summed reward over the episode
    assert np.isclose(trajs[0]["return"], arr["rewards"][0:5].sum(), atol=1e-5)
    assert np.isclose(trajs[1]["return"], arr["rewards"][5:12].sum(), atol=1e-5)
    # spec inferred correctly
    assert (spec.n_agents, spec.obs_dim, spec.state_dim, spec.action_dim,
            spec.discrete) == (N, obs_dim, state_dim, A, True)
    print("test_conversion: OK")


def test_dataset_build_and_batch():
    """End-to-end: fake -> trajectories by quality -> PreferenceDataset -> batch."""
    from omapl.data.generate_preferences import build_preference_dataset

    T, N, obs_dim, state_dim, A = 30, 3, 5, 7, 4
    ep_ends = list(range(4, T, 5))
    arr = experience_to_arrays(
        _fake_experience(T, N, obs_dim, state_dim, A, ep_ends))
    trajs, spec = trajectories_from_arrays(
        obs=arr["obs"], actions=arr["actions"], next_obs=arr["next_obs"],
        state=arr["state"], next_state=arr["next_state"],
        terminals=arr["terminals"], discrete=True,
        avail=arr["avail"], next_avail=arr["next_avail"], rewards=arr["rewards"])
    # split into fake quality tiers
    third = max(1, len(trajs) // 3)
    tbq = {"Poor": trajs[:third], "Medium": trajs[third:2 * third],
           "Good": trajs[2 * third:]}
    ds = build_preference_dataset(tbq, ["Poor", "Medium", "Good"], 50, spec)
    batch = ds.sample_batch(8, device="cpu")
    assert batch["obs"].shape[:4] == (8, 2, batch["obs"].shape[2], N)
    assert batch["avail"].shape[-1] == A
    assert set(("obs", "actions", "next_obs", "state", "next_state", "done",
                "mask", "avail", "next_avail", "label")).issubset(batch.keys())
    print("test_dataset_build_and_batch: OK")


if __name__ == "__main__":
    test_conversion()
    test_dataset_build_and_batch()
    print("All OG-MARL adapter tests passed.")
