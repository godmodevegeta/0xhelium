"""Scene_Builder tests (Requirement 2)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import MODEL_XML
from robustlab.physics.model import FOOT_GEOMS, check_model, load_model
from robustlab.physics.scene import SceneError, build_scene

# Derived collision-acceleration arrays that MuJoCo rebuilds when collision flags change.
ALLOWED_DIFFS = {
    "geom_conaffinity",
    "body_conaffinity",
    "body_bvhadr",
    "body_bvhnum",
    "bvh_aabb",
    "bvh_child",
    "bvh_depth",
    "bvh_nodeid",
    "mesh_bvhadr",
}


def _arrays(model) -> dict[str, np.ndarray]:
    out = {}
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except Exception:  # noqa: BLE001, S112 - some attributes are not readable arrays
            continue
        if isinstance(value, np.ndarray):
            out[name] = value
    return out


def _differing(a, b) -> set[str]:
    first, second = _arrays(a), _arrays(b)
    return {
        n
        for n in first
        if first[n].shape != second[n].shape or not np.array_equal(first[n], second[n], True)
    }


def test_feet_only_equals_m0_model() -> None:
    scene = build_scene(MODEL_XML, "feet-only")
    assert _differing(scene.model, load_model(MODEL_XML)) == set()
    assert scene.fall_geoms == () and scene.unchanged_geoms == ()


def test_fall_contact_changes_only_collision_arrays(policy) -> None:
    m0 = load_model(MODEL_XML)
    scene = build_scene(MODEL_XML, "fall-contact")
    model = scene.model
    assert _differing(model, m0) <= ALLOWED_DIFFS
    for name in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "geom_size", "geom_pos"):
        np.testing.assert_array_equal(getattr(model, name), getattr(m0, name), err_msg=name)
    assert model.opt.timestep == m0.opt.timestep
    check_model(model, policy)

    fall_ids = [g["geom_id"] for g in scene.fall_geoms]
    shin_ids = [g["geom_id"] for g in scene.unchanged_geoms]
    assert len(fall_ids) == 29 and len(shin_ids) == 4
    assert {g["class"] for g in scene.unchanged_geoms} == {"calf2"}
    assert {g["body"] for g in scene.unchanged_geoms} == {f"{leg}_calf" for leg in FOOT_GEOMS}
    changed = np.flatnonzero(model.geom_conaffinity != m0.geom_conaffinity).tolist()
    assert changed == sorted(fall_ids)
    assert (model.geom_conaffinity[shin_ids] == 0).all()
    robot = model.geom_bodyid != 0
    assert (model.geom_contype[robot] == 0).all()  # no robot-robot collision pairs
    trunk_geoms = [g for g in scene.fall_geoms if g["body"] == "trunk"]
    assert len(trunk_geoms) == 3


def test_transform_description_is_deterministic() -> None:
    a = build_scene(MODEL_XML, "fall-contact").describe()
    b = build_scene(MODEL_XML, "fall-contact").describe()
    assert a == b
    assert a["name"] == "fall-contact" and a["transform_version"] == 1
    assert a["base_xml_sha256"].startswith("5f9e9cba")
    assert len(a["transform_sha256"]) == 64


def test_unknown_scene_rejected() -> None:
    with pytest.raises(SceneError):
        build_scene(MODEL_XML, "full-collision")
