"""Search objective (minimized). Every physical fall ranks below every survivor, which ranks below
every numerical failure; among falls, lower impulse is better; among survivors, a smaller margin to
the fall thresholds is better.

    fell:      -2 + impulse / 30 N*s           in [-1.987, -1.0]
    survived:  min(tilt margin, height margin) in [0, 1]
    numerical: +2 (invalid; never a successful attack)
"""

from __future__ import annotations

import math

from robustlab.search.space import MAX_IMPULSE_NS

NOMINAL_TRUNK_Z = 0.28  # approximate standing/walking trunk height, m
NUMERICAL_OBJECTIVE = 2.0


def robustness_score(min_up_z: float, min_trunk_z: float, tilt_deg: float, min_z: float) -> float:
    """0 = at a fall threshold, 1 = perfectly level / at nominal height (clipped to [0, 1])."""
    cos_tilt = math.cos(math.radians(tilt_deg))
    tilt_margin = (min_up_z - cos_tilt) / (1.0 - cos_tilt)
    height_margin = (min_trunk_z - min_z) / (NOMINAL_TRUNK_Z - min_z)
    return min(1.0, max(0.0, min(tilt_margin, height_margin)))


def objective(outcome: str, impulse_ns: float, score: float | None) -> float:
    if outcome == "fell":
        return -2.0 + impulse_ns / MAX_IMPULSE_NS
    if outcome == "survived":
        if score is None or not math.isfinite(score):
            raise ValueError("a surviving evaluation needs a finite robustness score")
        return min(1.0, max(0.0, score))
    if outcome == "numerical":
        return NUMERICAL_OBJECTIVE
    raise ValueError(f"unknown outcome {outcome!r}")
