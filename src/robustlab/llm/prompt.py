"""System prompt and the strict JSON schema the model must answer with (fixed values only)."""

from __future__ import annotations

PROMPT_VERSION = 2
MAX_PROMPT_CHARS = 1000

SYSTEM_PROMPT = """You translate a plain-English description of a disturbance for a walking \
Unitree Go1 quadruped (simulated in MuJoCo) into one test scenario. Reply with JSON only, matching \
the schema. Never include code, URLs or extra fields.

The robot walks forward at the commanded velocity for the horizon. You may set:
- ground_friction_mu: floor sliding friction. Trained value 0.6. Allowed 0.05-1.50. Use null to \
keep 0.6. Ice ~0.05-0.1, wet tile ~0.2-0.3, dry concrete ~0.6-0.8, rubber ~1.0+.
- payload_mass_kg: a rigid box strapped to the trunk, 0-5 kg (0 = none).
- payload_offset_xyz_m: box centre relative to the trunk centre of mass, robot axes: x forward \
[-0.10, 0.10], y left [-0.08, 0.08], z up [0, 0.08] m.
- push: a constant horizontal force on the trunk, or null for none. force_n 0-150 N; \
direction_deg in the robot's starting heading frame: 0 = forward, 90 = toward the robot's left, \
180 = backward, 270 = toward the robot's right (a shove FROM the left points to 270); start_s \
between 2.0 and 4.5; duration_s 0.04-0.5.
- command: [vx, vy, yaw_rate], usually [1.0, 0.0, 0.0]; vx in [-1.5, 1.5] m/s, vy in [-0.8, 0.8] \
m/s, yaw_rate in [-1.2, 1.2] rad/s.

Translate magnitudes faithfully. If the user states a number outside an allowed range (for \
example 40 kg or 2000 N), output the number they asked for anyway: the server checks every value, \
rejects out-of-range ones and tells the user. Never substitute a different value to make it fit. \
Choose values yourself only where the user gave none; a typical shove lasts about 0.1 s. Describe \
anything not representable at all (stairs, slopes, wind, other robots) in `not_representable` and \
leave it out. Keep `rationale` under 40 words."""

_NUM = {"type": "number"}

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "name",
        "rationale",
        "not_representable",
        "command",
        "ground_friction_mu",
        "payload_mass_kg",
        "payload_offset_xyz_m",
        "push",
    ],
    "properties": {
        "name": {"type": "string", "description": "short kebab-case scenario name"},
        "rationale": {"type": "string"},
        "not_representable": {"type": "string", "description": "empty if everything fit"},
        "command": {"type": "array", "items": _NUM, "minItems": 3, "maxItems": 3},
        "ground_friction_mu": {"type": ["number", "null"]},
        "payload_mass_kg": _NUM,
        "payload_offset_xyz_m": {"type": "array", "items": _NUM, "minItems": 3, "maxItems": 3},
        "push": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["force_n", "direction_deg", "start_s", "duration_s"],
            "properties": {
                "force_n": _NUM,
                "direction_deg": _NUM,
                "start_s": _NUM,
                "duration_s": _NUM,
            },
        },
    },
}

EXAMPLES = [
    "Walking on wet tile when someone shoves it from the left at 2.5 seconds",
    "Carrying a 4 kg battery pack mounted high and toward the front",
    "Icy floor, no other disturbance",
    "Heavy backpack on a slippery floor and a strong kick from behind",
]


def to_scenario(proposal: dict, *, horizon_s: float = 5.0, scene: str = "fall-contact") -> dict:
    """Map the model's answer to a robustlab-scenario document (unvalidated)."""
    name = "".join(c if c.isalnum() or c in "-_." else "-" for c in str(proposal.get("name", "")))[
        :64
    ]
    return {
        "schema_version": 1,
        "kind": "robustlab-scenario",
        "name": name.strip("-") or "llm-scenario",
        "command": proposal.get("command"),
        "horizon_s": horizon_s,
        "scene": scene,
        "environment": {"ground_friction_mu": proposal.get("ground_friction_mu")},
        "payload": {
            "mass_kg": proposal.get("payload_mass_kg"),
            "offset_xyz_m": proposal.get("payload_offset_xyz_m"),
        },
        "push": proposal.get("push"),
    }
