"""Scenario schema validation (M5 Requirement 1) with Hypothesis properties."""

from __future__ import annotations

import copy
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from robustlab.scenarios.schema import (
    BOUNDS,
    SIM_DT,
    ScenarioError,
    json_schema,
    nominal_dict,
    validate,
)


def doc(**changes) -> dict:
    d = nominal_dict("t")
    for path, value in changes.items():
        node = d
        keys = path.split("__")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return d


def fields(exc: ScenarioError) -> set[str]:
    return {e["field"] for e in exc.errors}


in_bounds = {
    "mu": st.floats(0.05, 1.5),
    "mass": st.floats(0.0, 5.0),
    "ox": st.floats(-0.10, 0.10),
    "oy": st.floats(-0.08, 0.08),
    "oz": st.floats(0.0, 0.08),
    "force": st.floats(0.0, 150.0),
    "direction": st.floats(-1e4, 1e4, allow_nan=False),
    "start": st.floats(0.0, 4.4),
    "duration": st.floats(0.0, 0.5),
}


# Feature: m5-openrouter-scenarios, Property 28: in-bounds scenarios validate; values preserved
@given(**in_bounds)
def test_property_valid_scenarios_round_trip(
    mu, mass, ox, oy, oz, force, direction, start, duration
):
    d = doc(environment__ground_friction_mu=mu)
    d["payload"] = {"mass_kg": mass, "offset_xyz_m": [ox, oy, oz]}
    d["push"] = {"force_n": force, "direction_deg": direction, "start_s": start,
                 "duration_s": duration}  # fmt: skip
    v = validate(d)
    s = v.scenario
    assert s.ground_friction_mu == mu and s.payload_mass_kg == mass
    assert s.payload_offset_xyz_m == (ox, oy, oz)
    # 4 ms quantization, disclosed whenever it changed a value
    q_start, q_dur = round(start / SIM_DT) * SIM_DT, round(duration / SIM_DT) * SIM_DT
    adjusted = {a["field"] for a in v.adjustments}
    if s.push is None:
        assert force == 0 or q_dur < SIM_DT / 2 or "push" in adjusted
        assert "push" in adjusted
    else:
        assert abs(s.push.start_s - q_start) < 1e-9 and abs(s.push.duration_s - q_dur) < 1e-9
        assert abs(s.push.start_s - start) <= SIM_DT / 2 + 1e-9
        assert s.push.start_substep * SIM_DT == pytest.approx(s.push.start_s, abs=1e-9)
        assert 0.0 <= s.push.direction_deg < 360.0
        assert math.isclose(math.cos(math.radians(s.push.direction_deg)),
                            math.cos(math.radians(direction)), abs_tol=1e-6)  # fmt: skip
        if abs(q_start - start) > 1e-9:
            assert "push.start_s" in adjusted
    assert validate(s.to_dict()).scenario == s  # canonical form is a fixed point


bounded_fields = [
    ("environment.ground_friction_mu", "environment__ground_friction_mu"),
    ("payload.mass_kg", "payload__mass_kg"),
]


# Feature: m5-openrouter-scenarios, Property 29: out-of-bounds values are rejected, never clamped
@given(st.sampled_from(bounded_fields), st.floats(allow_nan=False, allow_infinity=False),
       st.integers(0, 2))  # fmt: skip
def test_property_out_of_bounds_rejected(field, value, axis) -> None:
    name, path = field
    lo, hi = BOUNDS[name]
    if lo <= value <= hi:
        return
    with pytest.raises(ScenarioError) as info:
        validate(doc(**{path: value}))
    assert name in fields(info.value)
    off_name = f"payload.offset_xyz_m[{axis}]"
    lo, hi = BOUNDS[off_name]
    if not lo <= value <= hi:
        d = doc()
        d["payload"]["offset_xyz_m"][axis] = value
        with pytest.raises(ScenarioError) as info:
            validate(d)
        assert off_name in fields(info.value)


def test_errors_are_all_reported_with_paths() -> None:
    d = doc(environment__ground_friction_mu=0.01, horizon_s=5.01)
    d["payload"] = {"mass_kg": 6.0, "offset_xyz_m": [0.0, 0.2, -0.01], "extra": 1}
    d["push"] = {"force_n": 151, "direction_deg": "left", "start_s": 4.95, "duration_s": 0.1}
    d["command"] = [2.0, 0.0, 0.0]
    d["code"] = "import os"
    with pytest.raises(ScenarioError) as info:
        validate(d)
    got = fields(info.value)
    for f in ("environment.ground_friction_mu", "horizon_s", "payload.mass_kg", "payload.extra",
              "payload.offset_xyz_m[1]", "payload.offset_xyz_m[2]", "push.force_n",
              "push.direction_deg", "command[0]", "code"):  # fmt: skip
        assert f in got, (f, got)


@pytest.mark.parametrize(
    "value",
    [{"range": [0.1, 0.5]}, {"range": [0.1]}, {"min": 0.1}, float("nan"), True, "0.3", None],
)
def test_ranges_and_bad_types_rejected_in_fixed_mode(value) -> None:
    d = doc()
    d["payload"]["mass_kg"] = value
    with pytest.raises(ScenarioError) as info:
        validate(d)
    assert "payload.mass_kg" in fields(info.value)


def test_push_beyond_horizon_and_zero_push() -> None:
    d = doc()
    d["push"] = {"force_n": 50, "direction_deg": 90, "start_s": 4.95, "duration_s": 0.1}
    with pytest.raises(ScenarioError) as info:
        validate(d)
    assert "push.start_s" in fields(info.value)
    d["push"] = {"force_n": 0, "direction_deg": 90, "start_s": 2.0, "duration_s": 0.1}
    v = validate(d)
    assert v.scenario.push is None and any(a["field"] == "push" for a in v.adjustments)


# Feature: m5-openrouter-scenarios, Property 30: nominal-default identity
@given(st.floats(-0.1, 0.1), st.floats(-0.08, 0.08), st.floats(0.0, 0.08))
def test_property_nominal_default_identity(ox, oy, oz) -> None:
    d = nominal_dict()
    d["payload"]["offset_xyz_m"] = [ox, oy, oz]  # an offset without mass changes nothing
    s = validate(d).scenario
    assert s.is_nominal_physics and s.push is None
    assert s.changed_from_nominal() == []
    assert s.effective_mu() == 0.6


def test_json_schema_lists_every_top_level_field() -> None:
    schema = json_schema()
    assert set(schema["required"]) == set(nominal_dict())
    assert schema["additionalProperties"] is False
    assert copy.deepcopy(nominal_dict())["kind"] == "robustlab-scenario"
