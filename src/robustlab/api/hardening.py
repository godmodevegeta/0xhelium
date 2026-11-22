"""Policy Hardening API service (M6): attack choices, the precomputed sample candidate, and the
baseline-vs-candidate replay.

Replays re-simulate both controllers on the server on the identical episode (model, reset,
command, push) and compare the outcomes with the candidate record (see `reflex_replay`). Locked
episodes are not replayed here: `HardeningSets.split` refuses them; the Gate API replays them
after the gate result exists. Candidate labels carry the M7 gate status when the gate evaluated
that candidate id.
"""

from __future__ import annotations

from collections.abc import Callable

from robustlab.adapt.harden import DEFAULT_SETS, HardenConfig
from robustlab.adapt.reflex import Calibration
from robustlab.adapt.sets import load_sets
from robustlab.api.jobs import JobManager
from robustlab.api.presets import PresetError, load_preset
from robustlab.api.reflex_replay import paired_reflex_replay

PRESET_DIR = "presets/hardening-candidate-v1"
PRESET_KEY = "preset-v1"
LIMITS = {"max_generations": (1, 30), "popsize": 8, "max_rollouts": 4500}
LABELS = {
    "candidate": "Candidate: frozen policy + reflex (not evaluated by the M7 gate)",
    "baseline": "Frozen baseline controller, same shove",
    "matched": "Matched condition: frozen baseline, no shove",
}


class HardeningService:
    def __init__(self, settings, jobs: JobManager, wall_clock_s: float, workers: int,
                 gate_status: Callable[[str], dict | None] | None = None) -> None:  # fmt: skip
        self.jobs = jobs
        self.gate_status = gate_status
        self.root = settings.root
        self.wall_clock_s = wall_clock_s
        self.workers = workers
        self.sets = load_sets(self.root / DEFAULT_SETS, self.root)
        try:
            self.preset = load_preset(self.root, PRESET_DIR)
            self.preset_record = self.preset.json("candidate.json")
            self.preset_grids = self.preset.json("grids.json")
        except PresetError:
            self.preset = self.preset_record = self.preset_grids = None

    def episodes(self) -> list[dict]:
        out = []
        for split in ("train", "validation_preview"):
            for e in self.sets.split(split):
                out.append({**e.to_dict(), "selectable": split == "train" and e.kind == "attack"
                            and e.confirmed_failure})  # fmt: skip
        return out

    def config(self) -> dict:
        return {
            "sets": self.sets.summary(),
            "episodes": self.episodes(),
            "defaults": {
                "max_generations": 12,
                "popsize": LIMITS["popsize"],
                "max_rollouts": 2000,
                "wall_clock_s": self.wall_clock_s,
                "workers": self.workers,
            },
            "limits": {
                "max_generations": list(LIMITS["max_generations"]),
                "max_rollouts": LIMITS["max_rollouts"],
                "wall_clock_s": self.wall_clock_s,
            },
            "preset": None
            if self.preset is None
            else {
                "key": PRESET_KEY,
                "result_origin": "precomputed",
                "meta": {k: self.preset.meta[k] for k in ("id", "version", "title", "source")},
                "record": self.preset_record,
                "grids": self.preset_grids,
                "gate_status": self._gate_status(self.preset_record["candidate_id"]),
            },
            "runs": [
                {**r, "gate_status": self._gate_status(r.get("candidate_id"))}
                for r in self.jobs.list_hardenings()
            ],
            "semantics": "A candidate is the frozen go1-walk-s0 actor plus a bounded residual "
            "reflex. A hardening run never promotes anything: only the pre-registered M7 gate "
            "(robustlab gate) moves the registry pointer.",
        }

    def _gate_status(self, candidate_id: str | None) -> dict | None:
        if self.gate_status is None or candidate_id is None:
            return None
        return self.gate_status(candidate_id)

    def make_config(self, seed: int, generations: int, focus: str | None) -> HardenConfig:
        return HardenConfig(seed=seed, popsize=LIMITS["popsize"], max_generations=generations,
                            max_rollouts=LIMITS["max_rollouts"], wall_clock_s=self.wall_clock_s,
                            workers=self.workers, focus=focus)  # fmt: skip

    def record_for(self, key: str) -> dict | None:
        if key == PRESET_KEY:
            return self.preset_record
        return self.jobs.hardening_record(key)

    def candidate_label(self, candidate_id: str) -> str:
        status = self._gate_status(candidate_id)
        if status is None:
            return LABELS["candidate"]
        return f"Candidate ({status['label']}): frozen policy + reflex"

    def replay_compute(self, key: str, record: dict, episode_id: str, ghost: str):
        """Raises LookupError for an unknown or locked episode."""
        spec = next((e for split in ("train", "validation_preview")
                     for e in self.sets.split(split) if e.id == episode_id), None)  # fmt: skip
        if spec is None:
            raise LookupError(
                f"unknown episode {episode_id!r} (locked episodes are replayed in the Gate view)"
            )
        recorded = {}
        for split in ("train", "validation_preview"):
            for row in record["measurements"][split].get("comparison") or []:
                recorded[row["id"]] = row
        status = self._gate_status(record["candidate_id"])
        return paired_reflex_replay(
            self.root, spec, record["reflex"]["vector"], Calibration.from_dict(record["calibration"]),
            ghost, mode="hardening", labels={**LABELS, "candidate": self.candidate_label(
                record["candidate_id"])}, recorded=recorded.get(spec.id),
            recorded_commit=record["provenance"]["git"]["commit"],
            extra={"run_id": key, "candidate_id": record["candidate_id"],
                   "candidate_status": record["status"], "gate_status": status},
        )  # fmt: skip
