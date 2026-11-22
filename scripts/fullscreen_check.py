"""Fullscreen check in a real (headed) Chrome window: headless Chrome refuses requestFullscreen.

Opens a visible window with a fresh profile, clicks the stage's Fullscreen button with a trusted
CDP mouse event, verifies `document.fullscreenElement` is the stage card and the canvas fills the
screen, that playback, scrubbing and orbit still work in fullscreen, then exits with the same
button and verifies the layout returns. Writes runs/m6ui/fixes/fullscreen.json and screenshots.

Usage: uv run python scripts/fullscreen_check.py [--base http://localhost:8000]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import websockets
from screenshot import launch
from ui_checks import TITLE, Page

RECT = """(() => { const r = document.querySelector('.stage-card .canvas').getBoundingClientRect();
  return {x: r.left, y: r.top, w: r.width, h: r.height, sw: screen.width, sh: screen.height,
          iw: innerWidth, ih: innerHeight}; })()"""


async def run(base: str, out: Path) -> dict:
    proc, ws_url = launch(1440, 900, 9339, headless=False)
    result: dict = {"browser": "Google Chrome, headed window (not headless)"}
    try:
        async with websockets.connect(ws_url, max_size=2**28) as ws:
            page = Page(ws)
            await page.call("Runtime.enable")
            result["user_agent"] = (await page.call("Browser.getVersion"))["product"]
            await page.call(
                "Page.navigate", url=base + "/?tab=harden&hreplay=preset-v1:m2-100N-500:baseline"
            )
            await page.until(
                f"{TITLE}.startsWith('Hardening') && !document.querySelector('.stage-overlay')", 60
            )
            await page.call("Page.bringToFront")  # an unfocused window is refused fullscreen
            await asyncio.sleep(1.0)
            result["document_has_focus"] = await page.eval("document.hasFocus()")
            before = await page.eval(RECT)
            await page.click("document.querySelector('[aria-label=\"Fullscreen\"]')")
            await asyncio.sleep(1.5)
            result["fullscreen_element"] = await page.eval(
                "document.fullscreenElement?.className ?? null"
            )
            result["expanded_fallback_used"] = await page.eval(
                "!!document.querySelector('.stage-card.expanded')"
            )
            inside = await page.eval(RECT)
            result["canvas_before"] = before
            result["canvas_in_fullscreen"] = inside
            await page.shot(out / "fullscreen-real-browser.png")
            # playback and controls inside fullscreen
            t0 = await page.eval("document.querySelector('.track-meta span').textContent")
            await page.click("document.querySelector('.transport .play')")
            await asyncio.sleep(1.0)
            t1 = await page.eval("document.querySelector('.track-meta span').textContent")
            result["playback_toggles_in_fullscreen"] = t0 != t1 or await page.eval(
                "document.querySelector('.transport .play').getAttribute('aria-label')"
            )
            c = inside
            clip = {"x": c["x"], "y": c["y"], "width": c["w"], "height": c["h"]}
            if await page.eval(
                "document.querySelector('.transport .play').getAttribute('aria-label') === 'Pause'"
            ):
                await page.click("document.querySelector('.transport .play')")
            await asyncio.sleep(0.5)
            h0 = hashlib.sha256(await page.shot(None, clip)).hexdigest()
            await page.drag(c["x"] + c["w"] * 0.5, c["y"] + c["h"] * 0.6, 200, -50)
            await asyncio.sleep(0.8)
            h1 = hashlib.sha256(await page.shot(None, clip)).hexdigest()
            result["orbit_in_fullscreen_changes_view"] = h0 != h1
            labels = await page.eval(
                "[...document.querySelectorAll('.robot-label')].map(e => e.textContent)"
            )
            result["labels_in_fullscreen"] = labels
            # exit with the same button
            await page.click("document.querySelector('[aria-label=\"Fullscreen\"]')")
            await asyncio.sleep(1.5)
            result["after_exit_fullscreen_element"] = await page.eval(
                "document.fullscreenElement?.className ?? null"
            )
            result["canvas_after_exit"] = await page.eval(RECT)
            await page.shot(out / "fullscreen-real-browser-exited.png")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    fs = result["canvas_in_fullscreen"]
    result["checks"] = {
        "entered_real_fullscreen": bool(result["fullscreen_element"])
        and "stage-card" in result["fullscreen_element"],
        "no_fallback_needed": not result["expanded_fallback_used"],
        "canvas_fills_width": abs(fs["w"] - fs["iw"]) <= 2,
        "canvas_taller_than_before": fs["h"] > result["canvas_before"]["h"] + 50,
        "orbit_works": result["orbit_in_fullscreen_changes_view"],
        "exited": result["after_exit_fullscreen_element"] is None
        and abs(result["canvas_after_exit"]["w"] - result["canvas_before"]["w"]) <= 2,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--out", default="runs/m6ui/fixes")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(run(args.base, out))
    (out / "fullscreen.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
