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


def _print_summary(manifest: dict, paths) -> None:
    out = manifest["outcome"]
    lat = manifest["latency"]

    def fmt(value, spec=".3f"):
        return "n/a" if value is None else format(value, spec)

    rows = [
        ("run", manifest["run_id"]),
        ("scene", manifest["scene"]["name"]),
        ("command (vx, vy, yaw)", ", ".join(f"{c:g}" for c in manifest["command"])),
        ("horizon", f"{manifest['horizon_s']:g} s ({manifest['n_ticks']} ticks)"),
        ("criteria set", out.get("criteria_set", "upright")),
        ("upright (45 deg gate)", out["upright_45"]),
        ("upright (60 deg compare)", out["upright_60"]),
        ("max tilt", f"{fmt(out['max_tilt_deg'], '.2f')} deg"),
        ("min trunk z", f"{fmt(out['min_trunk_z'])} m"),
        ("finite / no warnings", f"{out['finite']} / {out['warnings_ok']}"),
    ]
    # Window metrics stop at the horizon and at the last Tick before a fall; label what was used.
    w0, w1 = out.get("window_s", (2.0, 10.0))
    used = f"{w0:g}-{min(w1, out.get('progress_until_s', w1)):.2f} s"
    progress = "until fall" if out.get("fall", {}).get("kind", "none") != "none" else ""
    for key, label, unit in (
        ("forward_distance_m", f"forward distance {progress}".strip(), "m"),
        ("mean_forward_speed", f"mean fwd speed {used}", "m/s"),
        ("horizontal_disp_m", "horizontal displacement", "m"),
        ("yaw_drift_deg", "yaw drift", "deg"),
    ):
        if key in out:
            rows.append((label, f"{fmt(out[key])} {unit}"))
    if "touchdowns" in out:
        rows.append((f"touchdowns {used} FR,FL,RR,RL", ", ".join(map(str, out["touchdowns"]))))
        rows.append(("duty factor FR,FL,RR,RL", ", ".join(f"{d:.2f}" for d in out["duty_factor"])))
    push = manifest["push"]
    if push is None:
        rows.append(("push", "none"))
    else:
        world = ", ".join(f"{v:.1f}" for v in push["world_force_n"])
        rows += [
            (
                "push",
                (
                    f"{push['force_n']:g} N toward {push['direction_heading_frame']} "
                    f"(heading frame), t = {push['start_s']:g} to {push['end_s']:g} s"
                ),
            ),
            ("push world force / impulse", f"[{world}] N / {push['impulse_ns']:g} N*s"),
        ]
    fall = out["fall"]
    if fall["kind"] == "none":
        rows.append(("FELL", "no"))
    else:
        detail = fall["events"][fall["reason"]]["detail"]
        verdict = "yes" if fall["kind"] == "physical" else "NUMERICAL FAILURE (not a fall)"
        rows.append(("FELL", f"{verdict}: {fall['reason']} at t = {fall['time_s']:.3f} s {detail}"))
        later = sorted(
            (e["time_s"], name) for name, e in fall["events"].items() if name != fall["reason"]
        )
        if later:
            rows.append(("later events", ", ".join(f"{n} {t:.3f} s" for t, n in later)))
    if not fall["nonfoot_contact_detection"]:
        rows.append(("non-foot contact detection", "unavailable in this scene"))
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
    from robustlab.assets import AssetError, verify_assets

    issues = verify_assets(root)
    if issues:
        for issue in issues:
            print(f"FAIL  {issue}", file=sys.stderr)
        print("walk: asset verification failed; refusing to run", file=sys.stderr)
        return EXIT_ASSETS

    from robustlab.experiments.push import PushError, PushScenario, parse_direction
    from robustlab.experiments.stack import load_stack
    from robustlab.experiments.trace import write_run
    from robustlab.experiments.walk import EpisodeConfig, EpisodeRunner
    from robustlab.physics.model import ModelError
    from robustlab.physics.scene import SceneError
    from robustlab.policy.actor import PolicyError

    if not (0 < args.seconds <= MAX_SECONDS):
        print(f"walk: --seconds must be in (0, {MAX_SECONDS:g}]", file=sys.stderr)
        return EXIT_INVALID
    push_args = (args.push_force, args.push_direction, args.push_start, args.push_duration)
    push = None
    if any(a is not None for a in push_args):
        if any(a is None for a in push_args):
            print(
                "walk: give all of --push-force, --push-direction, --push-start and "
                "--push-duration, or none",
                file=sys.stderr,
            )
            return EXIT_INVALID
        try:
            push = PushScenario(
                force_n=args.push_force,
                direction=parse_direction(args.push_direction),
                start_s=args.push_start,
                duration_s=args.push_duration,
            )
        except PushError as exc:
            print(f"walk: {exc}", file=sys.stderr)
            return EXIT_INVALID
    try:
        stack = load_stack(root, Path(args.policy), args.scene)
    except (PolicyError, ModelError, AssetError, SceneError, ValueError) as exc:
        print(f"walk: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INVALID
    error = stack.command_error(args.command)
    if error:
        print(f"walk: {error}", file=sys.stderr)
        return EXIT_INVALID

    config = EpisodeConfig(command=args.command, seconds=args.seconds, seed=args.seed, push=push)
    runner = EpisodeRunner(stack.model, stack.ids, stack.policy, stack.actor)
    try:
        record = runner.run(config)
    except ValueError as exc:
        print(f"walk: {exc}", file=sys.stderr)
        return EXIT_INVALID

    from robustlab.experiments.walk import WalkCriteria, evaluate

    criteria = WalkCriteria()
    outcome = evaluate(record, criteria)
    run_dir = Path(args.out)
    manifest = stack.manifest_for(
        record,
        run_id=run_dir.name,
        command=args.command,
        seconds=args.seconds,
        seed=args.seed,
        outcome=outcome,
        criteria=criteria,
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


EXIT_PAIRING = 4


def _cmd_experiment(args: argparse.Namespace) -> int:
    root = Path(args.root)
    from robustlab.assets import AssetError, verify_assets
    from robustlab.experiments.suite import (
        SuiteConfigError,
        check_out_dir,
        format_summary,
        load_suite_config,
        run_suite,
        sampling_record,
    )

    try:
        config = load_suite_config(Path(args.config), trials=args.trials, seed=args.seed)
    except (OSError, SuiteConfigError) as exc:
        print(f"experiment: invalid config: {exc}", file=sys.stderr)
        return EXIT_INVALID
    if args.dry_run:
        sampling = sampling_record(config)
        print(f"experiment {config.name}: {sampling['kind']}, seed {config.seed}")
        print(
            f"rollouts: {2 * config.trials} ({config.trials} nominal + {config.trials} attacked), "
            f"{config.horizon_s:g} s each"
        )
        for i, k in enumerate(sampling["start_substeps"]):
            print(f"trial {i:>4}  start substep {k}  start {k * 0.004:.3f} s")
        return 0

    issues = verify_assets(root)
    if issues:
        for issue in issues:
            print(f"FAIL  {issue}", file=sys.stderr)
        print("experiment: asset verification failed; refusing to run", file=sys.stderr)
        return EXIT_ASSETS
    from robustlab.experiments.stack import load_stack
    from robustlab.physics.model import ModelError
    from robustlab.physics.scene import SceneError
    from robustlab.policy.actor import PolicyError

    try:
        check_out_dir(Path(args.out), args.overwrite)
    except FileExistsError as exc:
        print(f"experiment: {exc} (use --overwrite)", file=sys.stderr)
        return EXIT_WRITE
    try:
        stack = load_stack(root, Path(config.policy), config.scene)
    except (PolicyError, ModelError, AssetError, SceneError, ValueError) as exc:
        print(f"experiment: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INVALID
    try:
        report = run_suite(
            config,
            stack,
            Path(args.out),
            save_trajectories=args.save_trajectories,
            overwrite=args.overwrite,
        )
    except SuiteConfigError as exc:
        print(f"experiment: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except OSError as exc:
        print(f"experiment: write failed: {exc}", file=sys.stderr)
        return EXIT_WRITE
    print(format_summary(report))
    print(f"report           {Path(args.out) / 'experiment.json'}")
    timing = report["timing"]
    print(f"wall time        {timing['wall_s']:.2f} s for {timing['rollouts']} rollouts")
    if not report["aggregates"]["paired"]["pairing_ok_all"]:
        print("experiment: pre-attack trajectories differ between paired runs", file=sys.stderr)
        return EXIT_PAIRING
    return 0


def _cmd_discover(args: argparse.Namespace) -> int:
    root = Path(args.root)
    from robustlab.assets import AssetError, verify_assets

    try:
        from robustlab.search.engine import SearchConfig, SearchConfigError, run_discovery
        from robustlab.search.report import format_discovery, write_discovery
    except ImportError as exc:
        print(f"discover: {exc}; install with: uv sync --extra search", file=sys.stderr)
        return EXIT_INVALID
    try:
        config = SearchConfig(args.algorithm, args.budget, args.seed, args.shrink_budget,
                              args.timeout)  # fmt: skip
    except SearchConfigError as exc:
        print(f"discover: {exc}", file=sys.stderr)
        return EXIT_INVALID
    out = Path(args.out)
    if (out / "discovery.json").exists() and not args.overwrite:
        print(f"discover: {out / 'discovery.json'} exists (use --overwrite)", file=sys.stderr)
        return EXIT_WRITE
    issues = verify_assets(root)
    if issues:
        for issue in issues:
            print(f"FAIL  {issue}", file=sys.stderr)
        return EXIT_ASSETS
    from robustlab.experiments.stack import load_stack

    try:
        stack = load_stack(root, Path(args.policy), "fall-contact")
    except (AssetError, ValueError) as exc:
        print(f"discover: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INVALID
    record = run_discovery(config, stack)
    try:
        write_discovery(out, record)
    except OSError as exc:
        print(f"discover: write failed: {exc}", file=sys.stderr)
        return EXIT_WRITE
    print(format_discovery(record))
    print(f"record           {out / 'discovery.json'}")
    return 0


def _cmd_scenario(args: argparse.Namespace) -> int:
    import json

    root = Path(args.root)
    from robustlab.assets import AssetError, verify_assets
    from robustlab.scenarios.schema import ScenarioError, validate

    try:
        raw = json.loads(Path(args.file).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"scenario: cannot read {args.file}: {exc}", file=sys.stderr)
        return EXIT_INVALID
    from robustlab.experiments.stack import load_stack

    issues = verify_assets(root)
    if issues:
        for issue in issues:
            print(f"FAIL  {issue}", file=sys.stderr)
        return EXIT_ASSETS
    try:
        validated = validate(raw)
    except ScenarioError as exc:
        for error in exc.errors:
            print(f"scenario: invalid {error['field']}: {error['reason']}", file=sys.stderr)
        return EXIT_INVALID
    for adj in validated.adjustments:
        print(
            f"adjusted         {adj['field']}: {adj['requested']} -> {adj['applied']} ({adj['reason']})"
        )
    out = Path(args.out)
    if (out / "scenario_result.json").exists() and not args.overwrite:
        print(f"scenario: {out / 'scenario_result.json'} exists (use --overwrite)", file=sys.stderr)
        return EXIT_WRITE
    try:
        stack = load_stack(root, Path(args.policy), validated.scenario.scene)
    except (AssetError, ValueError) as exc:
        print(f"scenario: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INVALID
    from robustlab.experiments.scenario_run import outcome_line, run_scenario_suite, run_single

    scenario = validated.scenario
    if args.suite:
        try:
            low, high = (float(x) for x in args.interval.split(","))
            result = run_scenario_suite(scenario, stack, out, trials=args.trials,
                                        interval_s=(low, high), seed=args.seed,
                                        method=args.method, adjustments=validated.adjustments,
                                        overwrite=args.overwrite)  # fmt: skip
        except ValueError as exc:
            print(f"scenario: {exc}", file=sys.stderr)
            return EXIT_INVALID
        agg = result["aggregates"]
        print(f"suite            {result['trial_semantics']}")
        for case in ("nominal", "attacked"):
            a = agg[case]
            label = "matched baseline" if case == "nominal" else "scenario (pushed)"
            print(f"{label:<18} falls {a['physical_falls']}/{a['valid_trials']} valid, "
                  f"numerical {a['numerical_failures']}")  # fmt: skip
        print(f"original nominal {outcome_line(result['original'])}")
    else:
        result = run_single(scenario, stack, out, adjustments=validated.adjustments)
        fr = result["model"]["friction"]
        print(f"friction         requested {fr['requested_mu']} configured {fr['configured_mu']:g} "
              f"observed at reset {fr['observed_at_reset']['sliding_mu_values']}")  # fmt: skip
        print(f"total mass       {result['model']['mass']['total_mass_kg']:.3f} kg")
        for name, case in result["cases"].items():
            slip = case["slip"]["mean_slip_speed_mps"]
            print(f"{name:<17}{outcome_line(case)}; progress {case['forward_progress_m']:.3f} m; "
                  f"max tilt {case['max_tilt_deg']:.1f} deg; foot slip "
                  f"{'n/a' if slip is None else f'{slip:.4f} m/s'}")  # fmt: skip
        print(f"primary ghost    {result['primary_baseline']}")
        print(f"results_sha256   {result['results_sha256']}")
    print(f"record           {out / 'scenario_result.json'}")
    return 0


def _cmd_harden(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    from robustlab.assets import AssetError, verify_assets

    issues = verify_assets(root)
    if issues:
        for issue in issues:
            print(f"FAIL  {issue}", file=sys.stderr)
        return EXIT_ASSETS
    from robustlab.adapt.harden import (
        CANDIDATE_NAME,
        HardenConfig,
        HardenConfigError,
        default_workers,
        load_calibration,
        run_hardening,
        write_candidate,
    )
    from robustlab.adapt.sets import SetsError, load_sets
    from robustlab.experiments.stack import load_stack

    out = Path(args.out)
    if (out / CANDIDATE_NAME).exists():
        print(f"harden: {out / CANDIDATE_NAME} exists; candidates are immutable", file=sys.stderr)
        return EXIT_WRITE
    try:
        config = HardenConfig(seed=args.seed, popsize=args.popsize, sigma0=args.sigma0,
                              max_generations=args.generations, max_rollouts=args.max_rollouts,
                              wall_clock_s=args.wall_clock,
                              workers=args.workers or default_workers(), focus=args.focus)  # fmt: skip
        stack = load_stack(root, Path(args.policy), "fall-contact")
        sets = load_sets(root / args.sets, root)
        calibration = load_calibration(root / args.calibration, stack.policy.sha256)
    except (HardenConfigError, SetsError, AssetError, ValueError, OSError) as exc:
        print(f"harden: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INVALID

    def progress(entry: dict) -> None:
        print(f"generation {entry['generation']:>3}  rollouts {entry['rollouts']:>5}  "
              f"best J {entry['best_objective']:.4f} (baseline {entry['baseline_objective']:.4f})  "
              f"attacks survived {entry['best_attack_survived']}/{entry['attacks']}  "
              f"{entry['elapsed_s']:.1f} s", flush=True)  # fmt: skip

    try:
        record = run_hardening(stack, calibration, sets, config, on_progress=progress)
    except HardenConfigError as exc:
        print(f"harden: {exc}", file=sys.stderr)
        return EXIT_INVALID
    path = write_candidate(out, record)
    acc, res = record["accounting"], record["result"]
    print(f"stop             {res['stop_reason']} after {res['generations_completed']} generations")
    print(f"rollouts         {acc['total_rollouts']} (optimization {acc['optimization_rollouts']}, "
          f"baseline {acc['baseline_rollouts']}, references {acc['reference_rollouts']}, "
          f"validation preview {acc['validation_preview_rollouts']})")  # fmt: skip
    print(f"wall             {record['timing']['wall_s']:.1f} s on {config.workers} worker(s)")
    print(f"result           {res['statement']}")
    print(f"candidate        {record['candidate_id']} (status {record['status']})")
    print(f"reproducibility  {record['reproducibility_sha256']}")
    print(f"record           {path}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import os

    try:
        import uvicorn
    except ImportError:
        print("serve: install the server extra: uv sync --extra server", file=sys.stderr)
        return EXIT_INVALID
    os.environ.setdefault("ROBUSTLAB_ROOT", str(Path(args.root).resolve()))
    uvicorn.run(
        "robustlab.api.app:create_app", factory=True, host=args.host, port=args.port, workers=1
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robustlab",
        description=(
            "Robustness Studio: pretrained Go1 walking in native MuJoCo (CPU), with an optional "
            "timed trunk push and substep fall detection."
        ),
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
    walk.add_argument(
        "--scene",
        choices=("fall-contact", "feet-only"),
        default="fall-contact",
        help="fall-contact (default): trunk, hips, thighs and knees can touch the floor; "
        "feet-only: the M0 walk.xml as pinned",
    )
    walk.add_argument("--push-force", type=float, help="push magnitude, N, in [0, 150]")
    walk.add_argument(
        "--push-direction",
        help="where the force points, initial-heading frame: forward|backward|left|right|x,y,z "
        "(right = shove from the robot's left)",
    )
    walk.add_argument("--push-start", type=float, help="push start time, s (multiple of 0.004)")
    walk.add_argument(
        "--push-duration", type=float, help="push duration, s, in [0, 0.5] (multiple of 0.004)"
    )
    walk.add_argument("--root", default=".", help="repository root (default: current dir)")
    walk.set_defaults(func=_cmd_walk)

    experiment = subparsers.add_parser(
        "experiment",
        help="run a paired nominal-vs-shove suite over shove start times and write a report",
    )
    experiment.add_argument("--config", required=True, help="suite config JSON, e.g. examples/m2/")
    experiment.add_argument("--out", required=True, help="experiment directory, e.g. runs/m2/x")
    experiment.add_argument("--trials", type=int, help="override the config's trial count")
    experiment.add_argument("--seed", type=int, help="override the config's seed")
    experiment.add_argument(
        "--save-trajectories",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="write M1-format trajectories per trial (default: on for <= 32 trials)",
    )
    experiment.add_argument("--overwrite", action="store_true", help="replace an existing report")
    experiment.add_argument(
        "--dry-run", action="store_true", help="print sampled start times without simulating"
    )
    experiment.add_argument("--root", default=".", help="repository root (default: current dir)")
    experiment.set_defaults(func=_cmd_experiment)

    discover = subparsers.add_parser(
        "discover",
        help="bounded search for a low-impulse shove that makes the robot fall (needs pycma)",
    )
    discover.add_argument("--algorithm", choices=("random", "cmaes"), required=True)
    discover.add_argument("--budget", type=int, default=64, help="search rollouts (multiple of 8)")
    discover.add_argument("--seed", type=int, default=0)
    discover.add_argument("--shrink-budget", type=int, default=32, help="shrinking rollouts")
    discover.add_argument("--timeout", type=float, default=300.0, help="wall-clock limit, s")
    discover.add_argument("--policy", default="policies/go1-walk-s0.json")
    discover.add_argument("--out", required=True, help="output directory, e.g. runs/m4/x")
    discover.add_argument("--overwrite", action="store_true")
    discover.add_argument("--root", default=".", help="repository root (default: current dir)")
    discover.set_defaults(func=_cmd_discover)

    scenario = subparsers.add_parser(
        "scenario", help="run a multi-axis scenario file (friction, payload, push)"
    )
    scenario.add_argument("--file", required=True, help="scenario JSON (robustlab-scenario v1)")
    scenario.add_argument("--out", required=True)
    scenario.add_argument("--suite", action="store_true", help="vary the push start over trials")
    scenario.add_argument("--trials", type=int, default=16)
    scenario.add_argument("--interval", default="2.0,3.024", help="push start interval low,high s")
    scenario.add_argument("--method", choices=("stratified", "grid"), default="stratified")
    scenario.add_argument("--seed", type=int, default=0)
    scenario.add_argument("--policy", default="policies/go1-walk-s0.json")
    scenario.add_argument("--overwrite", action="store_true")
    scenario.add_argument("--root", default=".")
    scenario.set_defaults(func=_cmd_scenario)

    harden = subparsers.add_parser(
        "harden", help="search a bounded residual reflex around the frozen policy (candidate only)"
    )
    harden.add_argument("--out", required=True, help="output directory for candidate.json")
    harden.add_argument("--seed", type=int, default=0)
    harden.add_argument("--generations", type=int, default=12)
    harden.add_argument("--popsize", type=int, default=8)
    harden.add_argument("--sigma0", type=float, default=0.3)
    harden.add_argument("--max-rollouts", type=int, default=2000, help="optimization rollouts")
    harden.add_argument("--wall-clock", type=float, default=60.0, help="wall-clock cap, s")
    harden.add_argument("--workers", type=int, default=0, help="processes (0 = automatic)")
    harden.add_argument("--focus", default=None, help="one confirmed training attack id")
    harden.add_argument("--sets", default="examples/m6/hardening-sets-v1.json")
    harden.add_argument("--calibration", default="examples/m6/reflex-calibration-v1.json")
    harden.add_argument("--policy", default="policies/go1-walk-s0.json")
    harden.add_argument("--root", default=".")
    harden.set_defaults(func=_cmd_harden)

    serve = subparsers.add_parser("serve", help="start the web app (needs the server extra)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--root", default=".", help="repository root (default: current dir)")
    serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
