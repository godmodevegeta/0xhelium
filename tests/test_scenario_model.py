"""Scenario models: friction configuration, payload mass and inertia, cache (M5 Requirements 2-4)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import POLICY_PATH, REPO_ROOT
from robustlab.experiments.stack import load_stack
from robustlab.physics.scenario_model import (
    ScenarioModelCache,
    friction_report,
    mass_report,
    signature,
)
from robustlab.physics.scene import PAYLOAD_GEOM, Payload, build_scene

XML = REPO_ROOT / "assets/go1/walk.xml"


_STACK = []


def shared_stack():
    """One loaded stack for the module (Hypothesis tests cannot take function-scoped fixtures)."""
    if not _STACK:
        _STACK.append(load_stack(REPO_ROOT, POLICY_PATH, "fall-contact"))
    return _STACK[0]


@pytest.fixture(scope="module")
def stack():
    return shared_stack()


# Feature: m5-openrouter-scenarios, Property 31: configured and resolved friction equal the request
@settings(max_examples=15)
@given(st.floats(0.05, 1.5))
def test_property_friction_configuration(mu) -> None:
    stack = shared_stack()
    entry = ScenarioModelCache(maxsize=2).get(stack, mu, None)
    m0, m = stack.model, entry.model
    floor = entry.ids.floor_geom
    report = friction_report(m, entry.ids, stack.policy, mu)
    assert report["configured_mu"] == mu
    assert all(r[0] == mu for r in report["resolved_foot_floor"].values())
    assert report["observed_at_reset"]["sliding_mu_values"] == [round(mu, 12)]
    assert report["observed_at_reset"]["contacts"] == 4 and report["observed_at_reset"]["dims"] == [
        3
    ]
    # only the floor's sliding coefficient changed
    np.testing.assert_array_equal(m.geom_friction[floor, 1:], m0.geom_friction[floor, 1:])
    others = np.arange(m.ngeom) != floor
    np.testing.assert_array_equal(m.geom_friction[others], m0.geom_friction[others])
    np.testing.assert_array_equal(m.geom_priority, m0.geom_priority)
    np.testing.assert_array_equal(m.geom_condim, m0.geom_condim)


# Feature: m5-openrouter-scenarios, Property 32: payload mass, CoM and positive-definite inertia
@settings(max_examples=20)
@given(st.floats(0.01, 5.0), st.floats(-0.1, 0.1), st.floats(-0.08, 0.08), st.floats(0.0, 0.08))
def test_property_payload_mass_and_inertia(mass, ox, oy, oz) -> None:
    m0 = build_scene(XML, "fall-contact").model
    info = build_scene(XML, "fall-contact", payload=Payload(mass, (ox, oy, oz)))
    m = info.model
    assert m.body_mass.sum() == pytest.approx(m0.body_mass.sum() + mass, abs=1e-12)
    assert (m.nq, m.nv, m.nu, m.njnt, m.nsensor) == (m0.nq, m0.nv, m0.nu, m0.njnt, m0.nsensor)
    for name in ("body_mass", "body_inertia", "body_ipos", "body_pos", "geom_size"):
        n = getattr(m0, name).shape[0]
        np.testing.assert_array_equal(getattr(m, name)[:n], getattr(m0, name), err_msg=name)
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    ids = type("Ids", (), {"trunk_body": trunk})()
    report = mass_report(m, ids)["payload"]
    assert report["positive_definite"] and min(report["composite_inertia_eigenvalues"]) > 0
    assert report["composite_mass_kg"] == pytest.approx(m0.body_mass[trunk] + mass)
    # CoM where requested: payload centre at trunk CoM + offset (checked with MuJoCo kinematics)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    pid = report["body_id"]
    expected = d.xipos[trunk] + d.xmat[trunk].reshape(3, 3) @ np.array([ox, oy, oz])
    np.testing.assert_allclose(d.xipos[pid], expected, atol=1e-12)
    expected_com = (m0.body_mass[trunk] * d.xipos[trunk] + mass * d.xipos[pid]) / (
        m0.body_mass[trunk] + mass
    )
    world = d.xpos[trunk] + d.xmat[trunk].reshape(3, 3) @ np.array(
        report["composite_com_trunk_frame_m"]
    )
    np.testing.assert_allclose(world, expected_com, atol=1e-12)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, PAYLOAD_GEOM)
    assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and m.geom_group[g] == 4


def test_payload_is_not_floor_collidable(stack) -> None:
    from robustlab.physics.fall import nonfoot_floor_mask

    entry = ScenarioModelCache().get(stack, None, Payload(5.0, (0, 0, 0.08)))
    mask = nonfoot_floor_mask(entry.model, entry.ids)
    g = mujoco.mj_name2id(entry.model, mujoco.mjtObj.mjOBJ_GEOM, PAYLOAD_GEOM)
    assert not mask[g] and int(mask.sum()) == 29


def test_cache_keys_reuse_and_lru(stack) -> None:
    cache = ScenarioModelCache(maxsize=3)
    nominal = cache.get(stack, None, None)
    assert nominal.base and nominal.model is stack.model and cache.builds == 0
    assert cache.get(stack, None, Payload(0.0, (0.05, 0, 0))).model is stack.model  # no mass
    a = cache.get(stack, 0.3, Payload(2.0, (0.0, 0.0, 0.04)))
    assert cache.get(stack, 0.3, Payload(2.0, (0.0, 0.0, 0.04))) is a and cache.hits == 1
    b = cache.get(stack, 0.3, Payload(2.0, (0.0, 0.0, 0.05)))  # offset is part of the key
    assert b is not a and b.signature != a.signature
    cache.get(stack, 0.4, None)
    cache.get(stack, 0.5, None)  # evicts `a`
    assert cache.stats()["size"] == 3
    assert cache.get(stack, 0.3, Payload(2.0, (0.0, 0.0, 0.04))) is not a
    sig = signature("fall-contact", "x", 0.3, Payload(2.0, (0.1, -0.02, 0.03)))
    assert sig[:4] == ("fall-contact", 1, "x", 1) and sig[4:] == (0.3, 2.0, 0.1, -0.02, 0.03)


def test_friction_only_and_payload_only_descriptions(stack) -> None:
    cache = ScenarioModelCache()
    fr = cache.get(stack, 0.2, None).scene_info.describe()["modifications"]
    assert set(fr) == {"friction"} and fr["friction"]["sliding_after"] == 0.2
    pl = cache.get(stack, None, Payload(1.0, (0, 0, 0))).scene_info.describe()["modifications"]
    assert set(pl) == {"payload"} and pl["payload"]["box_size_m"] == [0.10, 0.08, 0.04]
    assert "modifications" not in stack.scene_info.describe()
