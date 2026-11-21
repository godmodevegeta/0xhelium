"""Importing robustlab must not pull in accelerator/ML stacks (Requirement 2.3)."""

from __future__ import annotations

import json
import subprocess
import sys

FORBIDDEN_MODULES = ("jax", "brax", "flax", "torch")

_PROBE = """
import json
import sys

import robustlab
import robustlab.cli

robustlab.cli.main(["doctor"])  # also loads NumPy and MuJoCo through the subcommand
forbidden = {forbidden!r}
loaded = sorted(
    name for name in sys.modules if name.split(".", 1)[0] in forbidden
)
print("PROBE_RESULT=" + json.dumps(loaded))
"""


def test_import_leaves_accelerator_stacks_unloaded() -> None:
    probe = _PROBE.format(forbidden=FORBIDDEN_MODULES)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    marker = next(line for line in result.stdout.splitlines() if line.startswith("PROBE_RESULT="))
    loaded = json.loads(marker.removeprefix("PROBE_RESULT="))
    assert loaded == [], f"forbidden modules imported: {loaded}"
