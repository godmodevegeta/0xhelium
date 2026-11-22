"""Compute the reflex dead-zone calibration from base-policy episodes without pushes and write
`examples/m6/reflex-calibration-v1.json` (deterministic; the integration tests recompute it).

Usage: uv run python scripts/calibrate_reflex.py
"""

from __future__ import annotations

import json
from pathlib import Path

from robustlab.adapt.evaluate import calibrate
from robustlab.experiments.stack import load_stack

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples/m6/reflex-calibration-v1.json"

if __name__ == "__main__":
    stack = load_stack(ROOT, Path("policies/go1-walk-s0.json"), "fall-contact")
    calibration = calibrate(stack)
    OUT.write_text(json.dumps(calibration.to_dict(), indent=1) + "\n")
    print(OUT, calibration.sha256()[:12])
    for name, c, e in zip(calibration.to_dict()["signals"], calibration.center,
                          calibration.half_width, strict=True):  # fmt: skip
        print(f"{name:24s} centre {c:+.4f}  half-width {e:.4f}")
