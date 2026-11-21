"""Command-line entry point for `robustlab`.

Heavy dependencies (NumPy, MuJoCo) are imported inside each command, so `--help` stays fast and
importing this module does not load the scientific stack.
"""

from __future__ import annotations

import argparse
import math
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


EXIT_ASSETS, EXIT_INVALID, EXIT_WRITE = 1, 2, 3
MAX_SECONDS = 600.0
MODEL_XML_RELPATH = "assets/go1/walk.xml"


def _parse_command(text: str) -> tuple[float, float, float]:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected vx,vy,yaw (three comma-separated numbers)")
    try:
        values = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number in {text!r}") from exc
    if not all(math.isfinite(v) for v in values):
        raise argparse.ArgumentTypeError("command values must be finite")
    return values  # type: ignore[return-value]


def _load_stack(root: Path, policy_path: Path):
    """Load and cross-check model + policy. Returns (model, ids, policy, actor, load_s)."""
    import time

    from robustlab.assets import locked_sha256
    from robustlab.physics.model import apply_model_parameters, check_model, load_model, resolve_ids
    from robustlab.policy.actor import Actor, load_policy

    start = time.perf_counter()
    policy_rel = Path(policy_path).resolve().relative_to(root.resolve()).as_posix()
    policy = load_policy(policy_path, locked_sha256(root, policy_rel))
    model = load_model(root / MODEL_XML_RELPATH)
    apply_model_parameters(model, policy.model_parameters)
    check_model(model, policy)
    ids = resolve_ids(model)
    actor = Actor(policy)
    return model, ids, policy, actor, time.perf_counter() - start


def _print_summary(manifest: dict, paths) -> None:
    out = manifest["outcome"]
    lat = manifest["latency"]

    def fmt(value, spec=".3f"):
        return "n/a" if value is None else format(value, spec)

    rows = [
        ("run", manifest["run_id"]),
        ("command (vx, vy, yaw)", ", ".join(f"{c:g}" for c in manifest["command"])),
        ("horizon", f"{manifest['horizon_s']:g} s ({manifest['n_ticks']} ticks)"),
        ("criteria set", out.get("criteria_set", "upright")),
        ("upright (45 deg gate)", out["upright_45"]),
        ("upright (60 deg compare)", out["upright_60"]),
        ("max tilt", f"{fmt(out['max_tilt_deg'], '.2f')} deg"),
        ("min trunk z", f"{fmt(out['min_trunk_z'])} m"),
        ("finite / no warnings", f"{out['finite']} / {out['warnings_ok']}"),
    ]
    for key, label, unit in (
        ("forward_distance_m", "forward distance", "m"),
        ("mean_forward_speed", "mean fwd speed 2-10 s", "m/s"),
        ("horizontal_disp_m", "horizontal displacement", "m"),
        ("yaw_drift_deg", "yaw drift", "deg"),
    ):
        if key in out:
            rows.append((label, f"{fmt(out[key])} {unit}"))
    if "touchdowns" in out:
        rows.append(("touchdowns 2-10 s FR,FL,RR,RL", ", ".join(map(str, out["touchdowns"]))))
        rows.append(("duty factor FR,FL,RR,RL", ", ".join(f"{d:.2f}" for d in out["duty_factor"])))
    rows += [
        (
            "tick latency median/p95/max",
            (
                f"{fmt(lat.get('median_ms'))} / {fmt(lat.get('p95_ms'))} / "
                f"{fmt(lat.get('max_ms'))} ms (target {lat['target_ms']:g} ms, "
                f"{lat['measured_ticks']} ticks after {lat['warmup_ticks']} warm-up)"
            ),
        ),
        ("real-time factor", fmt(lat.get("real_time_factor"), ".1f")),
        ("load time (model+policy)", f"{manifest['timing']['model_and_policy_load_s']:.3f} s"),
        (
            "JSON write trajectory/manifest",
            f"{paths.trajectory_write_s:.3f} / {paths.manifest_write_s:.3f} s",
        ),
        ("RESULT", "PASS" if out["passed"] else f"FAIL {out['failed_checks']}"),
        ("trajectory", str(paths.trajectory)),
        ("manifest", str(paths.manifest)),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}")


def _cmd_walk(args: argparse.Namespace) -> int:
    root = Path(args.root)
    from robustlab.assets import AssetError, load_lock, verify_assets

    issues = verify_assets(root)
    if issues:
        for issue in issues:
            print(f"FAIL  {issue}", file=sys.stderr)
        print("walk: asset verification failed; refusing to run", file=sys.stderr)
        return EXIT_ASSETS

    import json

    import mujoco
    import numpy as np

    from robustlab.experiments.trace import build_manifest, write_run
    from robustlab.experiments.walk import EpisodeConfig, EpisodeRunner, latency_summary
    from robustlab.physics.model import ModelError
    from robustlab.policy.actor import PolicyError

    if not (0 < args.seconds <= MAX_SECONDS):
        print(f"walk: --seconds must be in (0, {MAX_SECONDS:g}]", file=sys.stderr)
        return EXIT_INVALID
    try:
        model, ids, policy, actor, load_s = _load_stack(root, Path(args.policy))
    except (PolicyError, ModelError, AssetError, ValueError) as exc:
        print(f"walk: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INVALID
    for value, (lo, hi), name in zip(args.command, policy.command_ranges, ("vx", "vy", "yaw")):
        if not lo <= value <= hi:
            print(
                f"walk: command {name}={value} outside trained range [{lo}, {hi}]", file=sys.stderr
            )
            return EXIT_INVALID

    config = EpisodeConfig(command=args.command, seconds=args.seconds, seed=args.seed)
    runner = EpisodeRunner(model, ids, policy, actor)
    try:
        record = runner.run(config)
    except ValueError as exc:
        print(f"walk: {exc}", file=sys.stderr)
        return EXIT_INVALID

    from robustlab.experiments.walk import WalkCriteria, criteria_dict, evaluate

    criteria = WalkCriteria()
    outcome = evaluate(record, criteria)
    run_dir = Path(args.out)
    lock = load_lock(root)
    provenance = json.loads((root / "assets/go1/walk.provenance.json").read_text())
    manifest = build_manifest(
        run_id=run_dir.name,
        record=record,
        command=args.command,
        seconds=args.seconds,
        seed=args.seed,
        criteria=criteria_dict(criteria),
        outcome=outcome,
        latency=latency_summary(record.tick_ns, record.ctrl_dt),
        load_s=load_s,
        policy_info={
            "path": Path(args.policy).as_posix(),
            "sha256": policy.sha256,
            "run_id": policy.run_id,
            "env_name": policy.env_name,
            "trained_with": policy.versions,
        },
        lock=lock,
        provenance=provenance,
        library_versions={"numpy": np.__version__, "mujoco": mujoco.__version__},
        repo_root=root,
    )
    try:
        paths = write_run(run_dir, record, manifest, overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"walk: {exc} (use --overwrite)", file=sys.stderr)
        return EXIT_WRITE
    except OSError as exc:
        print(f"walk: write failed: {exc}", file=sys.stderr)
        return EXIT_WRITE
    _print_summary(manifest, paths)
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

    walk = subparsers.add_parser(
        "walk", help="run the pretrained policy for one episode and record trajectory + manifest"
    )
    walk.add_argument("--policy", required=True, help="policy JSON listed in the asset lock")
    walk.add_argument(
        "--command",
        required=True,
        type=_parse_command,
        help="body-frame command vx,vy,yaw in m/s, m/s, rad/s, e.g. 1,0,0 (use --command=-1,0,0 for negatives)",
    )
    walk.add_argument("--seconds", required=True, type=float, help="episode length, s")
    walk.add_argument("--out", required=True, help="run directory, e.g. runs/m0/forward")
    walk.add_argument("--seed", type=int, default=0, help="recorded only; M0 has no randomness")
    walk.add_argument("--overwrite", action="store_true", help="replace an existing run")
    walk.add_argument("--root", default=".", help="repository root (default: current dir)")
    walk.set_defaults(func=_cmd_walk)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
