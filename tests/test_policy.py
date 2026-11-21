"""Policy validation and actor unit/property tests (Requirements 5.1-5.5, 6.3, 6.4)."""

from __future__ import annotations

import base64
import copy
import hashlib
import json

import numpy as np
import pytest

from conftest import POLICY_PATH
from robustlab.policy.actor import Actor, PolicyError, load_policy, parse_policy, swish

SEED = 1121


@pytest.fixture(scope="module")
def doc() -> dict:
    return json.loads(POLICY_PATH.read_text())


def _encode(array: np.ndarray) -> str:
    return base64.b64encode(np.asarray(array, "<f4").tobytes()).decode("ascii")


def _decode(layer: dict, name: str) -> np.ndarray:
    raw = base64.b64decode(layer[name])
    return np.frombuffer(raw, "<f4").reshape(layer[f"{name}_shape"]).copy()


def test_normalizer_selected_by_obs_key(doc, policy) -> None:
    assert policy.obs_key == "state"
    np.testing.assert_array_equal(policy.mean, np.float32(doc["normalizer"]["mean"]["state"]))
    np.testing.assert_array_equal(policy.std, np.float32(doc["normalizer"]["std"]["state"]))
    mutated = copy.deepcopy(doc)
    mutated["normalizer"]["mean"]["state"] = [0.25] * 48
    mutated["normalizer"]["mean"]["privileged_state"] = [9.0] * 123
    assert np.all(parse_policy(mutated, "x").mean == np.float32(0.25))


def test_policy_metadata(policy) -> None:
    assert (policy.obs_dim, policy.act_dim) == (48, 12)
    assert [shape for shape in (k.shape for k, _ in policy.layers)] == [
        (48, 512),
        (512, 256),
        (256, 128),
        (128, 24),
    ]
    assert policy.action_scale == 0.5
    assert not policy.mean.flags.writeable


def test_actor_reuses_preallocated_buffers(policy) -> None:
    actor = Actor(policy)
    buffers = [id(actor._x), *map(id, actor._h), id(actor._e), id(actor._logits)]
    rng = np.random.default_rng(SEED)
    first = actor.act(rng.normal(size=48).astype(np.float32))
    second = actor.act(rng.normal(size=48).astype(np.float32))
    assert first is second is actor._action
    assert first.dtype == np.float32 and first.shape == (12,)
    assert [id(actor._x), *map(id, actor._h), id(actor._e), id(actor._logits)] == buffers
    with pytest.raises(ValueError):
        actor.act(np.zeros(47, np.float32))


def _mutations(doc: dict, rng: np.random.Generator):
    """Yield (description, mutated_doc, expected_field) for Property 3."""
    n_layers = len(doc["policy"]["layers"])

    def fresh():
        return copy.deepcopy(doc)

    for value in (0, 2, "1", None, 1.5):
        d = fresh()
        d["schema_version"] = value
        yield f"schema_version={value!r}", d, "schema_version"
    for value in ("relu", "tanh", "Swish", "", None):
        d = fresh()
        d["policy"]["activation"] = value
        yield f"activation={value!r}", d, "policy.activation"
    for _ in range(10):
        i = int(rng.integers(n_layers))
        value = str(rng.choice(["base64-f64-le", "base64-f32-be", "hex", ""]))
        d = fresh()
        d["policy"]["layers"][i]["encoding"] = value
        yield f"layers[{i}].encoding={value!r}", d, f"policy.layers[{i}].encoding"
    for _ in range(15):
        # A correctly encoded kernel whose input width breaks the chain.
        i = int(rng.integers(n_layers))
        layer = doc["policy"]["layers"][i]
        rows, cols = layer["kernel_shape"]
        new_rows = rows + int(rng.choice([-1, 1])) * int(rng.integers(1, 8))
        d = fresh()
        target = d["policy"]["layers"][i]
        target["kernel"] = _encode(rng.normal(size=(new_rows, cols)))
        target["kernel_shape"] = [new_rows, cols]
        yield f"layers[{i}] rows {rows}->{new_rows}", d, f"policy.layers[{i}].kernel_shape"
    for _ in range(10):
        value = int(rng.choice([0, 47, 49, 96, 123]))
        d = fresh()
        d["observation_size"]["state"] = value
        yield f"obs_dim={value}", d, "observation_size.state"
    for _ in range(10):
        value = int(rng.choice([0, 11, 13, 24]))
        d = fresh()
        d["action_size"] = value
        yield f"act_dim={value}", d, "action_size"
    bad_values = (np.nan, np.inf, -np.inf)
    for _ in range(30):
        i = int(rng.integers(n_layers))
        name = str(rng.choice(["kernel", "bias"]))
        d = fresh()
        target = d["policy"]["layers"][i]
        tensor = _decode(target, name)
        flat = tensor.reshape(-1)
        pos = int(rng.choice([0, flat.size - 1, int(rng.integers(flat.size))]))
        flat[pos] = bad_values[int(rng.integers(3))]
        target[name] = _encode(tensor)
        yield f"layers[{i}].{name}[{pos}] non-finite", d, f"policy.layers[{i}].{name}"
    for _ in range(20):
        name = str(rng.choice(["mean", "std"]))
        pos = int(rng.choice([0, 47, int(rng.integers(48))]))
        d = fresh()
        d["normalizer"][name]["state"][pos] = float(bad_values[int(rng.integers(3))])
        yield f"normalizer.{name}[{pos}] non-finite", d, f"normalizer.{name}"


# Feature: m0-walking-go1, Property 3: Malformed policies are rejected
def test_property_malformed_policies_rejected(doc, tmp_path) -> None:
    rng = np.random.default_rng(SEED)
    cases = list(_mutations(doc, rng))
    assert len(cases) >= 100
    for index, (description, mutated, field) in enumerate(cases):
        with pytest.raises(PolicyError) as info:
            parse_policy(mutated, "sha")
        context = f"seed={SEED} case={index} {description}: {info.value}"
        assert info.value.field == field, context

    # File hash differs from the Asset_Lock entry (any expected hash other than the real one).
    for index in range(10):
        wrong = "".join(rng.choice(list("0123456789abcdef"), 64))
        with pytest.raises(PolicyError) as info:
            load_policy(POLICY_PATH, wrong)
        assert info.value.field == "sha256", f"seed={SEED} hash case={index}"
    copy_path = tmp_path / "policy.json"
    copy_path.write_bytes(POLICY_PATH.read_bytes() + b" ")
    with pytest.raises(PolicyError) as info:
        # The tampered copy is checked against the lock's hash of the original file.
        load_policy(copy_path, hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest())
    assert info.value.field == "sha256"


def _swish_buffers(n: int):
    return np.empty(n, np.float32), np.empty(n, np.float32), np.empty(n, bool)


# Feature: m0-walking-go1, Property 4: Swish is finite and accurate
def test_property_swish_finite_and_accurate() -> None:
    rng = np.random.default_rng(SEED)
    boundary = np.float32([-80, 80, 0, -0.0, 1e-30, -1e-30, -1, 1, -40, 40, -79.99999, 79.99999])
    x = np.concatenate([boundary, rng.uniform(-80, 80, 10_000).astype(np.float32)])
    out = np.empty_like(x)
    swish(x, out, *_swish_buffers(x.size))
    assert np.all(np.isfinite(out)), f"seed={SEED}"
    x64 = x.astype(np.float64)
    reference = x64 / (1 + np.exp(-x64))
    # float32 rounding of a few chained operations: a handful of ulps, relative.
    np.testing.assert_allclose(out, reference, rtol=4e-7 * 8, atol=1e-37, err_msg=f"seed={SEED}")


def test_swish_in_place_alias() -> None:
    x = np.float32([-3, 0, 2])
    expected = x.astype(np.float64) / (1 + np.exp(-x.astype(np.float64)))
    swish(x, x, *_swish_buffers(3))
    np.testing.assert_allclose(x, expected, rtol=1e-6)
