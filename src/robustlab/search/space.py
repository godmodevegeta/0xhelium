"""The M4 search space: one horizontal trunk shove, quantized to what the simulator executes.

External parameters: force_n [10, 150] N at 0.1 N, direction_deg [0, 360) at 0.1° (initial-heading
frame, 0° forward, 90° left), start on the 4 ms grid in [2.000, 3.024) s, duration on the 4 ms grid
in [0.040, 0.200] s. Command, horizon, policy, initial state and scene are fixed by the caller.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from robustlab.experiments.push import PushScenario

SIM_DT = 0.004
FORCE_MIN_N, FORCE_MAX_N, FORCE_STEP_N = 10.0, 150.0, 0.1
DIRECTION_STEP_DEG = 0.1
START_MIN_SUBSTEP, START_SUBSTEPS = 500, 256  # 2.000 .. 3.020 s
DURATION_MIN_SUBSTEPS, DURATION_MAX_SUBSTEPS = 10, 50  # 0.040 .. 0.200 s
DURATION_LEVELS = DURATION_MAX_SUBSTEPS - DURATION_MIN_SUBSTEPS + 1
MAX_IMPULSE_NS = FORCE_MAX_N * DURATION_MAX_SUBSTEPS * SIM_DT  # 30 N*s

# CMA-ES internal box: (x_fx, x_fy) in [-1, 1]^2 scaled by 150 N, (u_start, u_duration) in [0, 1]^2.
CMA_LOWER = (-1.0, -1.0, 0.0, 0.0)
CMA_UPPER = (1.0, 1.0, 1.0, 1.0)


def bounds_record() -> dict:
    return {
        "force_n": [FORCE_MIN_N, FORCE_MAX_N],
        "force_resolution_n": FORCE_STEP_N,
        "direction_deg": [0.0, 360.0],
        "direction_resolution_deg": DIRECTION_STEP_DEG,
        "direction_frame": "initial-heading horizontal plane: 0 deg forward (+x), 90 deg left (+y)",
        "start_s": [START_MIN_SUBSTEP * SIM_DT, (START_MIN_SUBSTEP + START_SUBSTEPS) * SIM_DT],
        "start_substeps": [START_MIN_SUBSTEP, START_MIN_SUBSTEP + START_SUBSTEPS - 1],
        "duration_s": [DURATION_MIN_SUBSTEPS * SIM_DT, DURATION_MAX_SUBSTEPS * SIM_DT],
        "duration_substeps": [DURATION_MIN_SUBSTEPS, DURATION_MAX_SUBSTEPS],
        "time_quantum_s": SIM_DT,
        "max_impulse_ns": MAX_IMPULSE_NS,
    }


def quantize_force(force: float) -> float:
    value = round(round(float(force) / FORCE_STEP_N) * FORCE_STEP_N, 1)
    return min(FORCE_MAX_N, max(FORCE_MIN_N, value))


def quantize_direction(direction_deg: float) -> float:
    value = round(
        round((float(direction_deg) % 360.0) / DIRECTION_STEP_DEG) * DIRECTION_STEP_DEG, 1
    )
    return 0.0 if value >= 360.0 else value + 0.0  # + 0.0 turns -0.0 into 0.0


def clamp_start(substep: int) -> int:
    return min(START_MIN_SUBSTEP + START_SUBSTEPS - 1, max(START_MIN_SUBSTEP, int(substep)))


def clamp_duration(substeps: int) -> int:
    return min(DURATION_MAX_SUBSTEPS, max(DURATION_MIN_SUBSTEPS, int(substeps)))


@dataclass(frozen=True, order=True)
class Candidate:
    force_n: float
    direction_deg: float
    start_substep: int
    duration_substeps: int

    @classmethod
    def make(cls, force_n, direction_deg, start_substep, duration_substeps) -> Candidate:
        """Quantize and clamp to the search space."""
        return cls(
            quantize_force(force_n),
            quantize_direction(direction_deg),
            clamp_start(start_substep),
            clamp_duration(duration_substeps),
        )

    @property
    def start_s(self) -> float:
        return self.start_substep * SIM_DT

    @property
    def duration_s(self) -> float:
        return self.duration_substeps * SIM_DT

    @property
    def impulse_ns(self) -> float:
        return self.force_n * self.duration_substeps * SIM_DT

    @property
    def direction_vector(self) -> tuple[float, float, float]:
        a = math.radians(self.direction_deg)
        return (math.cos(a), math.sin(a), 0.0)

    def push(self) -> PushScenario:
        return PushScenario(self.force_n, self.direction_vector, self.start_s, self.duration_s)

    def key(self) -> tuple:
        return (self.force_n, self.direction_deg, self.start_substep, self.duration_substeps)

    def to_dict(self) -> dict:
        out = asdict(self)
        out.update(start_s=self.start_s, duration_s=self.duration_s, impulse_ns=self.impulse_ns)
        return out


def decode_cma(x) -> Candidate:
    """CMA-ES internal vector -> Candidate. Horizontal force is Cartesian (no 0/360 seam),
    projected radially onto the 10-150 N annulus."""
    x = np.asarray(x, dtype=float)
    fx, fy = FORCE_MAX_N * float(x[0]), FORCE_MAX_N * float(x[1])
    r = math.hypot(fx, fy)
    direction = math.degrees(math.atan2(fy, fx)) if r > 0 else 0.0
    force = min(FORCE_MAX_N, max(FORCE_MIN_N, r))
    u_start = min(1.0, max(0.0, float(x[2])))
    u_duration = min(1.0, max(0.0, float(x[3])))
    start = START_MIN_SUBSTEP + min(START_SUBSTEPS - 1, math.floor(START_SUBSTEPS * u_start))
    duration = DURATION_MIN_SUBSTEPS + min(
        DURATION_LEVELS - 1, math.floor(DURATION_LEVELS * u_duration)
    )
    return Candidate.make(force, direction, start, duration)


def sample_uniform(rng: np.random.Generator) -> Candidate:
    """Random Search prior: uniform force magnitude, direction, start and duration."""
    force = rng.uniform(FORCE_MIN_N, FORCE_MAX_N)
    direction = rng.uniform(0.0, 360.0)
    start = int(rng.integers(START_MIN_SUBSTEP, START_MIN_SUBSTEP + START_SUBSTEPS))
    duration = int(rng.integers(DURATION_MIN_SUBSTEPS, DURATION_MAX_SUBSTEPS + 1))
    return Candidate.make(force, direction, start, duration)
