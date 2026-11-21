"""Build the simulation scene from the pinned weeble `walk.xml`.

`feet-only` is the M0 model: only the four foot spheres can touch the floor.

`fall-contact` makes the other robot collision geoms floor-collidable so a fall shows up as body
contact instead of the trunk sinking through the ground. Only `conaffinity` changes (0 -> 1); the
floor already has `contype=1`, robot geoms keep `contype=0`, so robot-robot collisions stay off. The
lower shin capsule just above each foot (default class `calf2`) is left non-colliding: during normal
fast walking it dips a few millimetres below the floor in the feet-only scene the policy was trained
in, and making it solid changed nominal walking (see `docs/decision-log/0002-m1-fall-contact-scene.md`).

The pinned XML file is never modified; the transform is applied in memory with `MjSpec`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco

from robustlab.physics.model import FLOOR_GEOM, FOOT_GEOMS, load_model

SCENE_FALL_CONTACT = "fall-contact"
SCENE_FEET_ONLY = "feet-only"
SCENES = (SCENE_FALL_CONTACT, SCENE_FEET_ONLY)
SCENE_TRANSFORM_VERSION = 1
COLLISION_GROUP = 3
UNCHANGED_CLASSES = ("calf2",)


class SceneError(Exception):
    """The scene cannot be built from the given XML."""


@dataclass(frozen=True)
class SceneInfo:
    model: mujoco.MjModel
    name: str
    transform_version: int
    base_xml_sha256: str
    fall_geoms: tuple[dict, ...]  # {"geom_id", "body", "class"} made floor-collidable
    unchanged_geoms: tuple[dict, ...]  # non-foot collision geoms left non-colliding
    transform_sha256: str

    def describe(self) -> dict:
        """JSON-ready description for manifests (no model object)."""
        return {
            "name": self.name,
            "transform_version": self.transform_version,
            "base_xml_sha256": self.base_xml_sha256,
            "fall_geoms": list(self.fall_geoms),
            "unchanged_geoms": list(self.unchanged_geoms),
            "transform_sha256": self.transform_sha256,
        }


def _class_name(geom) -> str:
    cls = geom.classname
    return cls.name if hasattr(cls, "name") else str(cls)


def _transform_sha256(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_scene(xml_path: Path, scene: str) -> SceneInfo:
    """Compile `scene` from `xml_path`. Raises `SceneError` for an unknown scene name."""
    if scene not in SCENES:
        raise SceneError(f"unknown scene {scene!r}; expected one of {SCENES}")
    xml_path = Path(xml_path)
    base_sha = hashlib.sha256(xml_path.read_bytes()).hexdigest()

    fall_geoms: list[dict] = []
    unchanged: list[dict] = []
    if scene == SCENE_FEET_ONLY:
        model = load_model(xml_path)
    else:
        spec = mujoco.MjSpec.from_file(str(xml_path))
        # spec.geoms lists geoms in compiled geom-id order; checked against the model below.
        for gid, geom in enumerate(spec.geoms):
            body = geom.parent.name
            if body == "world" or geom.group != COLLISION_GROUP or geom.name in FOOT_GEOMS:
                continue
            entry = {"geom_id": gid, "body": body, "class": _class_name(geom)}
            if entry["class"] in UNCHANGED_CLASSES:
                unchanged.append(entry)
            else:
                geom.conaffinity = 1
                fall_geoms.append(entry)
        model = spec.compile()
        for entry in fall_geoms + unchanged:
            gid = entry["geom_id"]
            if model.body(int(model.geom_bodyid[gid])).name != entry["body"]:
                raise SceneError(f"geom {gid}: spec order does not match compiled model")
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
        if floor < 0 or not model.geom_contype[floor] & 1:
            raise SceneError("floor geom missing or not collidable with conaffinity bit 1")

    payload = {
        "name": scene,
        "transform_version": SCENE_TRANSFORM_VERSION,
        "base_xml_sha256": base_sha,
        "fall_geoms": fall_geoms,
        "unchanged_geoms": unchanged,
    }
    return SceneInfo(
        model=model,
        name=scene,
        transform_version=SCENE_TRANSFORM_VERSION,
        base_xml_sha256=base_sha,
        fall_geoms=tuple(fall_geoms),
        unchanged_geoms=tuple(unchanged),
        transform_sha256=_transform_sha256(payload),
    )
