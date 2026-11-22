"""Wall-clock browser screenshots over the Chrome DevTools Protocol (dev tool for milestone docs).

Chrome's `--screenshot` flag uses virtual time, which stalls on pages with a running WebGL
animation loop; this script waits real seconds instead.

    uv run python scripts/screenshot.py URL OUT.png [--width 1280 --height 2400 --wait 12]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def launch(
    width: int, height: int, port: int, headless: bool = True
) -> tuple[subprocess.Popen, str]:
    profile = tempfile.mkdtemp(prefix="chrome-cdp-")
    flags = (["--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
             if headless else ["--no-first-run", "--no-default-browser-check"]) + [
             "--hide-scrollbars", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
             f"--window-size={width},{height}", "about:blank"]  # fmt: skip
    proc = subprocess.Popen([CHROME, *flags], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json") as r:
                pages = [t for t in json.load(r) if t["type"] == "page"]
            if pages:
                return proc, pages[0]["webSocketDebuggerUrl"]
        except OSError:
            pass
        time.sleep(0.1)
    proc.terminate()
    raise RuntimeError("Chrome did not expose a page over CDP")


async def capture(ws_url: str, url: str, width: int, height: int, wait: float) -> bytes:
    async with websockets.connect(ws_url, max_size=2**28) as ws:
        seq = 0

        async def call(method: str, **params):
            nonlocal seq
            seq += 1
            await ws.send(json.dumps({"id": seq, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == seq:
                    return msg.get("result", {})

        await call("Emulation.setDeviceMetricsOverride", width=width, height=height,
                   deviceScaleFactor=1, mobile=False)  # fmt: skip
        await call("Page.navigate", url=url)
        await asyncio.sleep(wait)
        shot = await call("Page.captureScreenshot", format="png")
        return base64.b64decode(shot["data"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("out")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=2400)
    p.add_argument("--wait", type=float, default=12.0)
    p.add_argument("--cdp", type=int, default=9333, help="local DevTools port")
    a = p.parse_args()
    proc, ws_url = launch(a.width, a.height, a.cdp)
    try:
        png = asyncio.run(capture(ws_url, a.url, a.width, a.height, a.wait))
        Path(a.out).write_bytes(png)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
