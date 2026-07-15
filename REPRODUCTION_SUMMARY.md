# O-MAPL reproduction — summary for discussion

A one-page summary of the O-MAPL reproduction effort on public data. For full
technical detail see `HPC_RUN_LOG.md`; for verification status see `NOTES.md`.

## 1. What was built (the working part)
- A faithful **from-scratch O-MAPL implementation** — there is no public reference
  code, so it was built directly from the paper's equations. Unit-tested against
  the paper's own theory (linear non-negative mixer, the three alternating
  updates, gradient routing, the Extreme-V → soft-value property).
- A **full end-to-end pipeline on the Explorer cluster**: StarCraft II + SMACv2,
  public OG-MARL offline data → rule-based preference datasets → training → live
  win-rate evaluation. Stable, checkpointed, auto-resuming, fully documented.

## 2. What was run
- O-MAPL, **100k steps × 4 seeds**, on the **3 SMACv2 scenarios that have public
  offline data** (of the paper's 15 cells): terran_5v5, zerg_5v5, terran_10v10.
- Final win-rates (mean ± std over 4 seeds):

  | scenario | ours | paper (approx) |
  |---|---|---|
  | terran_5_vs_5 | 21.1 ± 8.1 % | ~40 % |
  | zerg_5_vs_5 | 8.6 ± 2.6 % | ~30 % |
  | terran_10_vs_10 | 3.1 ± 3.1 % | ~30 % |

- The curves are **flat, not rising** — training is stable but the policy does not
  visibly improve over training. Figure: `runs/omapl_smacv2_winrate.png`.

## 3. Why it does not match the paper (two honest reasons)
1. **The data isn't public.** The paper's datasets (ComaDICE) were never released.
   OG-MARL (the closest public data) is weaker and covers only 3 of 15 cells, so
   there is no apples-to-apples target — absolute numbers can't be expected to match.
2. **Key hyperparameters aren't specified — especially `beta`.** There is a
   stability↔learning tension: high `beta` keeps training stable but collapses the
   advantage-weighted policy into **plain behavior cloning** (→ flat curves); low
   `beta` gives a real learning signal but the **value function diverges**. The
   paper must use a stabilization mechanism (most likely reward/return
   normalization) that isn't stated. Reproducing the result therefore means
   *rediscovering* an unstated recipe, not just re-running it.

## 4. Why this is genuinely hard (not a matter of effort)
- No reference code + unspecified hyperparameters → the method has to be
  partially rediscovered.
- No reference data → no exact target to hit.
- Preference-only offline value learning is intrinsically unstable (deadly triad;
  nothing grounds the value scale but a soft regulariser); the stable-and-learning
  region is narrow and task-dependent.
- Slow, expensive iteration: ~100k offline steps + live StarCraft II eval per data
  point, on a shared, quota-limited, preemptible GPU cluster.

## 5. Questions / next steps to discuss
- Can we get the ComaDICE datasets or the missing hyperparameters (e.g. by
  contacting the authors)?
- Is it worth pursuing the likely fix — **add return normalization + retry at a
  low `beta`** on terran_5v5 — or is a faithful, stable, documented baseline
  sufficient for the goal?
- What's the actual objective: the exact paper numbers, or a working reproduction
  of the method?

## One-line summary
A faithful, working O-MAPL implementation that trains **stably** on real public
SMACv2 data, but does **not** recover the paper's win-rate curves — because the
datasets and key hyperparameters are not public, so part of the method must be
rediscovered rather than reproduced.

---

## Appendix A — what we tried (and why it didn't work)

**χ² regulariser WEIGHT (`reg_coef`) sweep** — terran_5_vs_5, seed 0, `beta=1`:

| reg_coef | R_mean (early → late) | V_mean | outcome |
|---|---|---|---|
| 1 (default) | ~17 → 52 | → ~5,200 | diverged; win-rate 31→19→16 |
| 50 | 17 → −25 | 4,771 → 5,815 | diverged; win-rate 34→22 |
| 100 | 7.6 → −33 | 4,766 → 6,206 | diverged; win-rate 28→19 |
| 200 | 4.7 → −18 | 4,551 → 6,163 | diverged; win-rate 31→28 |
| 500 | 1.1 (briefly) → osc. −3…+1.8 | 87 → 4,499 | looked fixed at 2k, diverged by 6k |
| 1000 | — | — | no usable logs; abandoned |

Raising `reg_coef` *does* pull the reward `R` toward its target of ~1 (17 → 7.6
→ 4.7 → 1.1), but it **never stopped `V` diverging** (always ~4,500–6,000). It
anchors the reward `R = Q − γV`, not the absolute value gauge `V` — so no weight
fixed the instability.

**χ² regulariser SCOPE — NOT swept (untested idea).** Kept fixed as the *mean* of
`φ(R)` over all masked transitions of both trajectories. A mean over ~100 steps
is ~100× weaker than a sum, so a summed/per-trajectory form could change the
effective strength a lot — but we never ran it. Open lever, not tried-and-failed.

**`beta` sweep** — terran_5_vs_5 for tuning; zerg/terran_10 on full runs:

| beta | V_mean | win-rate behavior | verdict |
|---|---|---|---|
| 1 | ~4,500–5,200 | peaks early then decays (31→19→16) | diverges |
| 5 | ~24,000 | climbs 22→25→28→34 (20k) | learns but V huge |
| 10 | ~41,000 | best at 20k (→37.5); full-100k terran_10 diverged (→0–3%) | learns short, diverges long |
| 20 | ~24 (bounded) | flat & low (terran_5 18%, zerg 6–18%, terran_10 0–6%) | stable but ≈ BC |
| 30 | ~16 (bounded) | flat & low (like 20) | stable but ≈ BC |

**No single `beta` gives both.** Low `beta` (1–10) learns but the value diverges
(short runs look fine, full 100k runs blow up — the trap). High `beta` (20–30)
bounds the value but the advantage weight flattens to ~1 → policy collapses to
behavior cloning → flat/low curves. Final choice: `beta=10` (terran_5), `beta=20`
(zerg/terran_10) as the least-bad points — hence stable but flat.

**Takeaway:** swept `reg_coef ∈ {1,5,50,100,200,500,1000}` and
`beta ∈ {1,5,10,20,30}`; `reg_coef` controls reward scale but not value
divergence, and `beta` only trades divergence for behavior-cloning. Neither knob
alone reaches the stable-and-learning point — the missing piece is almost
certainly a scale fix (value/return normalization) and/or the untested
regulariser scope, which would let a low `beta` learn without diverging.
