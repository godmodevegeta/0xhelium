"""M0 Task 6.6: render the final pose of runs/m0/forward to runs/m0/forward/final_pose.png.

Runs outside any timed loop. Any exception is reported and the render is skipped (no retries,
no backend switching).

Usage: uv run python scripts/render_final_pose.py (after `robustlab walk ... --out runs/m0/forward`)
"""

import json
import sys
import zlib
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/m0/forward/final_pose.png"


def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return len(data).to_bytes(4, "big") + body + zlib.crc32(body).to_bytes(4, "big")

    ihdr = w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw))
    path.write_bytes(png + chunk(b"IEND", b""))


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/go1/walk.xml"))
    data = mujoco.MjData(model)
    trajectory = json.loads((ROOT / "runs/m0/forward/trajectory.json").read_text())
    data.qpos[:] = trajectory["qpos"][-1]
    mujoco.mj_forward(model, data)
    try:
        renderer = mujoco.Renderer(model, 480, 640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = data.qpos[0:3]
        camera.distance, camera.azimuth, camera.elevation = 1.5, 135.0, -20.0
        renderer.update_scene(data, camera)
        image = renderer.render()
        renderer.close()
    except Exception as exc:  # noqa: BLE001 - any error means skip
        print(f"RENDER_SKIPPED {type(exc).__name__}: {exc}")
        return 0
    write_png(OUT, np.ascontiguousarray(image))
    print(f"RENDER_OK {OUT} shape={image.shape} mean={image.mean():.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
