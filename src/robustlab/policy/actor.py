"""Validated loading of weeble's JSON Go1 policy and a NumPy actor that reproduces it.

Policy files are JSON only (base64 little-endian float32 tensors). There is no pickle or `.brax`
loading path. Field names follow the upstream file (`policies/go1-walk-s0.json`, schema 1):

- `policy.layers[i].{encoding, kernel, kernel_shape, bias, bias_shape}`, kernel layout input x output
- `normalizer.{enabled, clip, mean[obs_key], std[obs_key]}` with `obs_key = policy.obs_key`
- `observation_size[obs_key]` (obs_dim) and `action_size` (act_dim)

Inference: `x = (obs - mean) / std` in float32, no epsilon, no clip; hidden layers
`swish(x @ W + b)`; linear last layer; action `tanh(logits[:act_dim])` (the deterministic mode of
Brax's tanh-normal head; the scale half of the logits is unused).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robustlab.physics.model import policy_joint_order

SCHEMA_VERSION = 1
ACTIVATIONS = ("swish", "silu")
ENCODING = "base64-f32-le"
OBS_DIM = 48
ACT_DIM = 12
OBSERVATION_FIELDS = (
    ("local_linear_velocity", 3),
    ("gyro", 3),
    ("gravity", 3),
    ("joint_position", 12),
    ("joint_velocity", 12),
    ("last_action", 12),
    ("command", 3),
)


class PolicyError(Exception):
    """A policy file failed validation. `field` names the offending JSON field."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


@dataclass(frozen=True)
class Policy:
    """Validated, read-only policy data. Arrays are float32 and not writeable."""

    sha256: str
    run_id: str
    env_name: str
    obs_key: str
    layers: tuple[tuple[np.ndarray, np.ndarray], ...]
    mean: np.ndarray
    std: np.ndarray
    fields: tuple[tuple[str, int, int], ...]  # (name, offset, size)
    default_pose: np.ndarray
    default_qpos: np.ndarray
    default_qvel: np.ndarray
    action_scale: float
    feet_geoms: tuple[str, ...]
    joint_names: tuple[str, ...]  # derived from feet_geoms; the file lists no joint names
    n_substeps: int
    control_dt: float
    simulation_dt: float
    force_body: str
    floor_geom: str
    sensors: dict[str, str]
    model_parameters: dict[str, Any]
    versions: dict[str, str]

    @property
    def obs_dim(self) -> int:
        return int(self.mean.size)

    @property
    def act_dim(self) -> int:
        return len(self.joint_names)


def _get(doc: Any, field: str) -> Any:
    node = doc
    for part in field.split("."):
        if not isinstance(node, dict) or part not in node:
            raise PolicyError(field, "missing")
        node = node[part]
    return node


def _expect(doc: Any, field: str, allowed: tuple[Any, ...]) -> Any:
    value = _get(doc, field)
    if value not in allowed:
        raise PolicyError(field, f"got {value!r}, expected one of {allowed!r}")
    return value


def _float_array(value: Any, field: str, size: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PolicyError(field, f"not a numeric array ({exc})") from exc
    if array.shape != (size,):
        raise PolicyError(field, f"shape {array.shape}, expected ({size},)")
    if not np.all(np.isfinite(array)):
        raise PolicyError(field, "contains non-finite values")
    return array


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _decode_tensor(layer: dict, name: str, field: str) -> np.ndarray:
    shape = _get(layer, f"{name}_shape")
    if not (isinstance(shape, list) and shape and all(isinstance(d, int) and d > 0 for d in shape)):
        raise PolicyError(f"{field}.{name}_shape", f"invalid shape {shape!r}")
    try:
        raw = base64.b64decode(_get(layer, name), validate=True)
    except (binascii.Error, TypeError, ValueError) as exc:
        raise PolicyError(f"{field}.{name}", f"invalid base64 ({exc})") from exc
    if len(raw) != 4 * math.prod(shape):
        raise PolicyError(f"{field}.{name}", f"{len(raw)} bytes do not match shape {shape}")
    tensor = np.frombuffer(raw, dtype="<f4").astype(np.float32).reshape(shape)
    if not np.all(np.isfinite(tensor)):
        raise PolicyError(f"{field}.{name}", "contains non-finite values")
    return _readonly(tensor)


def parse_policy(doc: Any, sha256: str) -> Policy:
    """Validate a decoded policy document. Raises `PolicyError(field, reason)`."""
    if not isinstance(doc, dict):
        raise PolicyError("<root>", "policy document must be a JSON object")
    _expect(doc, "schema_version", (SCHEMA_VERSION,))
    _expect(doc, "policy.activation", ACTIVATIONS)
    _expect(doc, "policy.kernel_layout", ("input_output",))
    _expect(doc, "policy.output_activation", ("linear",))
    _expect(doc, "policy.distribution", ("tanh_normal",))
    obs_key = _get(doc, "policy.obs_key")
    if not isinstance(obs_key, str):
        raise PolicyError("policy.obs_key", "must be a string")
    _expect(doc, "observation.key", (obs_key,))
    _expect(doc, f"observation_size.{obs_key}", (OBS_DIM,))
    _expect(doc, "observation.size", (OBS_DIM,))
    _expect(doc, "action_size", (ACT_DIM,))
    # Conventions the runner implements; anything else fails closed.
    _expect(doc, "task", ("walk",))
    _expect(doc, "gait", (None,))
    _expect(doc, "control.target", ("default_pose",))
    _expect(doc, "control.clip", (None,))
    _expect(doc, "normalizer.enabled", (True,))
    _expect(doc, "normalizer.clip", (None,))

    raw_layers = _get(doc, "policy.layers")
    if not isinstance(raw_layers, list) or len(raw_layers) < 2:
        raise PolicyError("policy.layers", "expected at least two layers")
    layers = []
    width = OBS_DIM
    for i, layer in enumerate(raw_layers):
        field = f"policy.layers[{i}]"
        if not isinstance(layer, dict):
            raise PolicyError(field, "must be an object")
        if layer.get("encoding") != ENCODING:
            raise PolicyError(
                f"{field}.encoding", f"got {layer.get('encoding')!r}, expected {ENCODING!r}"
            )
        kernel = _decode_tensor(layer, "kernel", field)
        bias = _decode_tensor(layer, "bias", field)
        if kernel.ndim != 2 or kernel.shape[0] != width:
            raise PolicyError(
                f"{field}.kernel_shape", f"{list(kernel.shape)} does not chain from width {width}"
            )
        if bias.shape != (kernel.shape[1],):
            raise PolicyError(f"{field}.bias_shape", f"{list(bias.shape)} != [{kernel.shape[1]}]")
        width = kernel.shape[1]
        layers.append((kernel, bias))
    if width != 2 * ACT_DIM:
        raise PolicyError(
            f"policy.layers[{len(raw_layers) - 1}].kernel_shape",
            f"output width {width}, expected {2 * ACT_DIM} (tanh-normal loc and scale)",
        )

    mean = _float_array(_get(doc, f"normalizer.mean.{obs_key}"), "normalizer.mean", OBS_DIM)
    std = _float_array(_get(doc, f"normalizer.std.{obs_key}"), "normalizer.std", OBS_DIM)
    if np.any(std.astype(np.float32) <= 0):
        raise PolicyError("normalizer.std", "contains non-positive values")

    raw_fields = _get(doc, "observation.fields")
    fields = []
    offset = 0
    expected = OBSERVATION_FIELDS
    if not isinstance(raw_fields, list) or len(raw_fields) != len(expected):
        raise PolicyError("observation.fields", f"expected {len(expected)} fields")
    for i, (entry, (name, size)) in enumerate(zip(raw_fields, expected, strict=True)):
        got = (entry.get("name"), entry.get("offset"), entry.get("size"))
        if got != (name, offset, size):
            raise PolicyError(
                f"observation.fields[{i}]", f"got {got}, expected {(name, offset, size)}"
            )
        fields.append((name, offset, size))
        offset += size

    feet = _get(doc, "feet_geoms")
    if not (isinstance(feet, list) and len(feet) == 4 and all(isinstance(f, str) for f in feet)):
        raise PolicyError("feet_geoms", "expected four foot geom names")
    action_scale = _expect(doc, "control.action_scale", (0.5,))
    _expect(doc, "training_config.action_scale", (action_scale,))

    return Policy(
        sha256=sha256,
        run_id=str(_get(doc, "run_id")),
        env_name=str(_get(doc, "env_name")),
        obs_key=obs_key,
        layers=tuple(layers),
        mean=_readonly(mean.astype(np.float32)),
        std=_readonly(std.astype(np.float32)),
        fields=tuple(fields),
        default_pose=_readonly(
            _float_array(_get(doc, "default_pose"), "default_pose", ACT_DIM).astype(np.float32)
        ),
        default_qpos=_readonly(_float_array(_get(doc, "default_qpos"), "default_qpos", 19)),
        default_qvel=_readonly(_float_array(_get(doc, "default_qvel"), "default_qvel", 18)),
        action_scale=float(action_scale),
        feet_geoms=tuple(feet),
        joint_names=policy_joint_order(feet),
        n_substeps=int(_get(doc, "n_substeps")),
        control_dt=float(_get(doc, "control_dt")),
        simulation_dt=float(_get(doc, "simulation_dt")),
        force_body=str(_get(doc, "force_body")),
        floor_geom=str(_get(doc, "floor_geom")),
        sensors=dict(_get(doc, "sensors")),
        model_parameters=_get(doc, "model_parameters"),
        versions=dict(_get(doc, "versions")),
    )


def load_policy(path: Path, expected_sha256: str) -> Policy:
    """Load and validate a policy JSON file whose SHA-256 must equal `expected_sha256`
    (the Asset_Lock entry). Raises `PolicyError`."""
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise PolicyError("sha256", f"file hash {actual} != expected {expected_sha256}")
    try:
        doc = json.loads(data)
    except json.JSONDecodeError as exc:
        raise PolicyError("<root>", f"invalid JSON ({exc})") from exc
    return parse_policy(doc, actual)


def swish(
    x: np.ndarray,
    out: np.ndarray,
    e: np.ndarray,
    s: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """`x * sigmoid(x)` with a stable sigmoid, using caller-provided float32/bool buffers.

    Same operation order as the upstream reference: `e = exp(-|x|)`,
    `sigmoid = 1/(1+e)` where `x >= 0`, else `e/(1+e)`. `out` may alias `x`.
    """
    np.greater_equal(x, 0, out=mask)
    np.abs(x, out=e)
    np.negative(e, out=e)
    np.exp(e, out=e)
    np.add(1, e, out=s)
    np.divide(e, s, out=e)  # e / (1 + e)
    np.divide(1, s, out=s)  # 1 / (1 + e)
    np.copyto(e, s, where=mask)
    return np.multiply(x, e, out=out)


class Actor:
    """Deterministic NumPy inference. All intermediate arrays are allocated once."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self._mean = policy.mean
        self._std = policy.std
        self._hidden = policy.layers[:-1]
        last_kernel, last_bias = policy.layers[-1]
        act = policy.act_dim
        self._out_kernel = last_kernel[:, :act]  # view, as in the reference implementation
        self._out_bias = last_bias[:act]
        self._x = np.empty(policy.obs_dim, np.float32)
        self._h = [np.empty(b.shape, np.float32) for _, b in self._hidden]
        widest = max(b.size for _, b in self._hidden)
        self._e = np.empty(widest, np.float32)
        self._s = np.empty(widest, np.float32)
        self._mask = np.empty(widest, bool)
        self._logits = np.empty(act, np.float32)
        self._action = np.empty(act, np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return the action for one 48-dim observation. The returned array is an internal
        buffer that the next call overwrites; copy it to keep it."""
        if obs.shape != self._x.shape:
            raise ValueError(f"expected observation shape {self._x.shape}, got {obs.shape}")
        x = self._x
        np.subtract(obs, self._mean, out=x, casting="same_kind", dtype=np.float32)
        np.divide(x, self._std, out=x)
        for (kernel, bias), h in zip(self._hidden, self._h, strict=True):
            n = h.size
            np.matmul(x, kernel, out=h)
            np.add(h, bias, out=h)
            swish(h, h, self._e[:n], self._s[:n], self._mask[:n])
            x = h
        np.matmul(x, self._out_kernel, out=self._logits)
        np.add(self._logits, self._out_bias, out=self._logits)
        return np.tanh(self._logits, out=self._action)
