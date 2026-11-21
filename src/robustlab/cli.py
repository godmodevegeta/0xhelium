"""Command-line entry point for `robustlab`.

Heavy dependencies (NumPy, MuJoCo) are imported inside each command, so `--help` stays fast and
importing this module does not load the scientific stack.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

from robustlab import __version__


def _numpy_build_summary(np_module) -> list[tuple[str, str]]:
    """Summarize NumPy's BLAS/LAPACK and SIMD build configuration as (label, value) rows."""
    try:
        config = np_module.show_config(mode="dicts")
    except Exception as exc:  # noqa: BLE001 - diagnostic output must never crash doctor
        return [("numpy config", f"unavailable ({type(exc).__name__}: {exc})")]
    if not isinstance(config, dict):
        return [("numpy config", "unavailable (show_config returned no dict)")]

    rows: list[tuple[str, str]] = []
    deps = config.get("Build Dependencies", {})
    for key in ("blas", "lapack"):
        info = deps.get(key, {})
        if not info:
            rows.append((key, "unknown"))
            continue
        name = info.get("name", "unknown")
        version = info.get("version", "unknown")
        found = info.get("found", "unknown")
        rows.append((key, f"{name} (version {version}, found={found})"))

    simd = config.get("SIMD Extensions", {})
    if simd:
        baseline = ",".join(simd.get("baseline", [])) or "none"
        found = ",".join(simd.get("found", [])) or "none"
        rows.append(("simd", f"baseline={baseline}; found={found}"))
    return rows


def _cmd_doctor(_args: argparse.Namespace) -> int:
    import mujoco
    import numpy as np

    rows: list[tuple[str, str]] = [
        ("robustlab", __version__),
        ("python", f"{platform.python_version()} ({sys.executable})"),
        ("numpy", np.__version__),
        ("mujoco", mujoco.__version__),
        ("platform", platform.platform()),
        ("machine", platform.machine()),
    ]
    rows.extend(_numpy_build_summary(np))

    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}")
    return 0


def _cmd_verify_assets(args: argparse.Namespace) -> int:
    from robustlab.assets import AssetError, verified_paths, verify_assets

    root = Path(args.root)
    issues = verify_assets(root)
    try:
        paths = verified_paths(root)
    except AssetError:
        paths = []
    failed = {issue.path for issue in issues}
    for relpath in paths:
        if relpath not in failed:
            print(f"OK    {relpath}")
    for issue in issues:
        print(f"FAIL  {issue}")
    if issues:
        print(f"verify-assets: {len(issues)} issue(s)", file=sys.stderr)
        return 1
    print(f"verify-assets: all {len(paths)} files match the lock and provenance")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robustlab",
        description="Robustness Studio M0: pretrained Go1 walking in native MuJoCo (CPU).",
    )
    parser.add_argument("--version", action="version", version=f"robustlab {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    doctor = subparsers.add_parser(
        "doctor", help="print Python, NumPy, MuJoCo, platform and NumPy BLAS information"
    )
    doctor.set_defaults(func=_cmd_doctor)

    verify = subparsers.add_parser(
        "verify-assets",
        help="check pinned assets against assets/assets.lock.json and walk.provenance.json",
    )
    verify.add_argument("--root", default=".", help="repository root (default: current dir)")
    verify.set_defaults(func=_cmd_verify_assets)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
