"""Suite_Config validation and the committed example configs (Requirement 1)."""

from __future__ import annotations

import copy
import json

import pytest

from conftest import REPO_ROOT
from robustlab.experiments.suite import SuiteConfigError, load_suite_config, sampling_record

EXAMPLES = sorted((REPO_ROOT / "examples" / "m2").glob("*.json"))

BASE = {
    "schema_version": 1,
    "name": "small",
    "policy": "policies/go1-walk-s0.json",
    "scene": "fall-contact",
    "command": [1.0, 0.0, 0.0],
    "horizon_s": 1.0,
    "trials": 4,
    "seed": 7,
    "attack": {
        "kind": "push",
        "force_n": 150.0,
        "direction": "right",
        "duration_s": 0.1,
        "start": {"method": "stratified", "low_s": 0.2, "high_s": 0.264},
    },
}


def _with(path: str, value) -> dict:
    cfg = copy.deepcopy(BASE)
    *parents, last = path.split(".")
    node = cfg
    for key in parents:
        node = node[key]
    if value is KeyError:
        del node[last]
    else:
        node[last] = value
    return cfg


def test_base_config_loads() -> None:
    config = load_suite_config(BASE)
    assert config.low_step == 50 and config.high_step == 66
    assert config.attack.direction == (0.0, -1.0, 0.0)
    assert config.push_at(55).window() == (55, 80)


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        ("extra", 1, "extra"),
        ("attack.start.extra", 1, "attack.start.extra"),
        ("seed", KeyError, "seed"),
        ("schema_version", 2, "schema_version"),
        ("name", "has space", "name"),
        ("scene", "full", "scene"),
        ("command", [1.0, 0.0], "command"),
        ("command", [1.0, "x", 0.0], "command"),
        ("horizon_s", 1.01, "horizon_s"),
        ("horizon_s", 0, "horizon_s"),
        ("trials", 0, "trials"),
        ("trials", 2000, "trials"),
        ("trials", 4.0, "trials"),
        ("trials", True, "trials"),
        ("trials", 5, "trials"),  # 16 substeps do not split into 5 equal strata
        ("trials", 17, "trials"),  # more trials than start substeps
        ("seed", -1, "seed"),
        ("attack.kind", "friction", "attack.kind"),
        ("attack.force_n", 151.0, "attack.force_n"),
        ("attack.force_n", float("nan"), "attack.force_n"),
        ("attack.direction", "up-left", "attack.direction"),
        ("attack.direction", [0, 0, 0], "attack.direction"),
        ("attack.duration_s", 0.6, "attack.duration_s"),
        ("attack.duration_s", 0.101, "attack.duration_s"),
        ("attack.start.method", "random", "attack.start.method"),
        ("attack.start.low_s", 0.201, "attack.start.low_s"),
        ("attack.start.low_s", -0.004, "attack.start.low_s"),
        ("attack.start.high_s", 0.2, "attack.start.high_s"),
        ("attack.start.high_s", 0.2641, "attack.start.high_s"),
        ("attack.start.high_s", 1.0, "horizon_s"),  # latest shove would end after 1 s
    ],
)
def test_invalid_config_names_the_field(path, value, field) -> None:
    with pytest.raises(SuiteConfigError) as info:
        load_suite_config(_with(path, value))
    assert info.value.field == field


def test_grid_allows_uneven_strata_and_overrides() -> None:
    config = load_suite_config(_with("attack.start.method", "grid"), trials=5, seed=99)
    assert config.trials == 5 and config.seed == 99
    assert sampling_record(config)["start_substeps"] == [50, 53, 56, 59, 62]


def test_invalid_json_is_a_config_error(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(SuiteConfigError):
        load_suite_config(path)


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_example_configs_load(path) -> None:
    config = load_suite_config(path)
    assert config.name == path.stem
    assert json.loads(path.read_text())["trials"] == config.trials
    record = sampling_record(config)
    assert len(set(record["start_substeps"])) == config.trials
