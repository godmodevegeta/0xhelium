"""Model loading, ID resolution and model-parameter application (Requirements 4.1-4.5)."""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import pytest

from conftest import MODEL_XML
from robustlab.physics.model import (
    ModelError,
    actuated_joint_order,
    apply_model_parameters,
    check_model,
    load_model,
    policy_joint_order,
    resolve_ids,
)

SEED = 4


@pytest.fixture(scope="module")
def model(policy_spec) -> mujoco.MjModel:
    loaded = load_model(MODEL_XML)
    apply_model_parameters(loaded, policy_spec["model_parameters"])
    return loaded


def test_model_facts(model, policy) -> None:
    assert (model.nq, model.nv, model.nu) == (19, 18, 12)
    assert model.opt.timestep == 0.004
    assert policy.n_substeps == 5
    check_model(model, policy)


def test_actuator_order_matches_policy_order(model, policy) -> None:
    expected = policy_joint_order(policy.feet_geoms)
    assert policy.joint_names == expected
    assert expected[:3] == ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint")
    assert actuated_joint_order(model) == expected


def test_check_model_rejects_reordered_feet(model, policy) -> None:
    bad = dataclasses.replace(policy, feet_geoms=("FL", "FR", "RR", "RL"))
    with pytest.raises(ModelError, match="feet_geoms"):
        check_model(model, bad)


def test_check_model_rejects_wrong_substeps(model, policy) -> None:
    with pytest.raises(ModelError, match="n_substeps"):
        check_model(model, dataclasses.replace(policy, n_substeps=4))


def test_resolve_ids(model) -> None:
    ids = resolve_ids(model)
    assert model.body(ids.trunk_body).name == "trunk"
    assert model.site(ids.imu_site).name == "imu"
    assert [model.geom(g).name for g in ids.foot_geoms] == ["FR", "FL", "RR", "RL"]
    assert model.geom(ids.floor_geom).name == "floor"
    assert ids.velocimeter_adr == model.sensor_adr[model.sensor("local_linvel").id]
    assert ids.gyro_adr == model.sensor_adr[model.sensor("gyro").id]


def test_resolve_ids_missing_name_raises() -> None:
    xml = """<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/>
      <body name="not_trunk"><freejoint/><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>"""
    with pytest.raises(ModelError, match="geom 'FR' not found in model"):
        resolve_ids(mujoco.MjModel.from_xml_string(xml))


def test_apply_rejects_unknown_array_and_bad_shape(policy_spec) -> None:
    fresh = load_model(MODEL_XML)
    with pytest.raises(ModelError, match="not a model array"):
        apply_model_parameters(fresh, {"arrays": {"no_such_array": [0.0]}})
    with pytest.raises(ModelError, match="shape"):
        apply_model_parameters(fresh, {"arrays": {"dof_damping": [0.0] * 3}})
    with pytest.raises(ModelError, match="MjOption"):
        apply_model_parameters(fresh, {"options": {"no_such_option": 1}})


def _snapshot(model: mujoco.MjModel, params: dict) -> dict:
    snap = {name: getattr(model, name).copy() for name in params["arrays"]}
    snap.update({f"opt.{name}": getattr(model.opt, name) for name in params["options"]})
    return snap


def _randomize(model: mujoco.MjModel, params: dict, rng: np.random.Generator) -> None:
    for name in params["arrays"]:
        target = getattr(model, name)
        target[...] = rng.normal(0.0, 10.0, size=target.shape).astype(target.dtype)
    for name, value in params["options"].items():
        if isinstance(value, float):
            setattr(model.opt, name, float(rng.uniform(1e-4, 0.05)))
        else:
            # Valid enum ranges for integrator/solver/cone; any int for counts and flags.
            upper = {"integrator": 4, "solver": 3, "cone": 2}.get(name, 1 << 15)
            setattr(model.opt, name, int(rng.integers(upper)))


# Feature: m0-walking-go1, Property 2: Applying model parameters is idempotent
def test_property_apply_model_parameters_idempotent(policy_spec) -> None:
    params = policy_spec["model_parameters"]
    model = load_model(MODEL_XML)
    rng = np.random.default_rng(SEED)
    for case in range(120):
        if case > 0:  # case 0 starts from the exported XML values
            _randomize(model, params, rng)
        apply_model_parameters(model, params)
        once = _snapshot(model, params)
        apply_model_parameters(model, params)
        twice = _snapshot(model, params)
        context = f"seed={SEED} case={case}"
        for key, value in once.items():
            np.testing.assert_array_equal(twice[key], value, err_msg=f"{context} {key}")
        for name, value in params["arrays"].items():
            target = getattr(model, name)
            np.testing.assert_array_equal(
                target, np.asarray(value, dtype=target.dtype), err_msg=f"{context} {name}"
            )
        for name, value in params["options"].items():
            assert getattr(model.opt, name) == value, f"{context} opt.{name}"
