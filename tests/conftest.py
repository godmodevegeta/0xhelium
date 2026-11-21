"""Shared paths and fixtures for the M0 test suite."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "policies" / "go1-walk-s0.json"
MODEL_XML = REPO_ROOT / "assets" / "go1" / "walk.xml"
ORACLE_PATH = REPO_ROOT / "tests" / "reference" / "weeble_sim.py"


@pytest.fixture(scope="session")
def policy_spec() -> dict:
    """The raw upstream policy JSON."""
    return json.loads(POLICY_PATH.read_text())


@pytest.fixture(scope="session")
def policy():
    """The validated policy, hash-checked against the asset lock."""
    from robustlab.assets import locked_sha256
    from robustlab.policy.actor import load_policy

    return load_policy(POLICY_PATH, locked_sha256(REPO_ROOT, "policies/go1-walk-s0.json"))


def build_model(policy, scene: str):
    """Compiled scene with the policy's model parameters applied, and its resolved IDs."""
    from robustlab.physics.model import apply_model_parameters, resolve_ids
    from robustlab.physics.scene import build_scene

    model = build_scene(MODEL_XML, scene).model
    apply_model_parameters(model, policy.model_parameters)
    return model, resolve_ids(model)


@pytest.fixture(scope="session")
def oracle():
    """weeble eval/sim.py loaded by file path, so it is never importable as a package module."""
    name = "weeble_sim_oracle"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
