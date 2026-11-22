"""Automated browser checks of the simulation-first workspace (M6), over the Chrome DevTools
Protocol against a running server (`uv run robustlab serve`).

Checks layout at 1440x900, 1280x800 and 390x844 (viewport above the fold, no horizontal overflow,
sample replay on load), that switching modes keeps one viewer canvas, and runs the Scenario
Designer, Failure Discovery and Policy Hardening flows end to end. Writes screenshots and
`ui-checks.json` to --out.

Usage: uv run python scripts/ui_checks.py [--base http://localhost:8000] [--out runs/m6/screens]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import websockets
from screenshot import launch  # scripts/ is on sys.path when run as a script


class Page:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.seq = 0

    async def call(self, method: str, **params):
        self.seq += 1
        await self.ws.send(json.dumps({"id": self.seq, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.seq:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    async def eval(self, expression: str):
        result = await self.call("Runtime.evaluate", expression=expression, returnByValue=True,
                                 awaitPromise=True)  # fmt: skip
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")

    async def until(self, expression: str, timeout: float = 30.0, every: float = 0.25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = await self.eval(expression)
            if value:
                return value
            await asyncio.sleep(every)
        raise TimeoutError(expression)

    async def size(self, width: int, height: int) -> None:
        await self.call("Emulation.setDeviceMetricsOverride", width=width, height=height,
                        deviceScaleFactor=1, mobile=width < 600)  # fmt: skip

    async def shot(self, path: Path) -> None:
        data = await self.call("Page.captureScreenshot", format="png")
        path.write_bytes(base64.b64decode(data["data"]))

    async def click_text(self, text: str, selector: str = "button") -> None:
        found = await self.eval(
            f"""(() => {{ const el = [...document.querySelectorAll({json.dumps(selector)})]
                 .find(e => e.offsetParent !== null && e.textContent.includes({json.dumps(text)}));
                 if (!el || el.disabled) return false; el.click(); return true; }})()"""
        )
        if not found:
            raise RuntimeError(f"no enabled {selector} containing {text!r}")


LAYOUT = """(() => {
  const c = document.querySelector('.replay .canvas');
  const r = c ? c.getBoundingClientRect() : null;
  return { width: innerWidth, height: innerHeight, scroll_width: document.documentElement.scrollWidth,
           canvas: r && { top: Math.round(r.top), bottom: Math.round(r.bottom), left: Math.round(r.left),
                          right: Math.round(r.right), width: Math.round(r.width), height: Math.round(r.height) },
           canvases: document.querySelectorAll('canvas').length,
           title: document.querySelector('.replay-title strong')?.textContent ?? null,
           sample: !!document.querySelector('.replay-title .badge'),
           legend: [...document.querySelectorAll('.legend-overlay > div')].map(d => d.textContent) };
})()"""
REPLAY_TITLE = "(document.querySelector('.replay-title strong')?.textContent ?? '')"


async def run(base: str, out: Path) -> dict:
    results: dict = {"base": base, "layouts": {}, "flows": {}}
    proc, ws_url = launch(1440, 900, 9335)
    try:
        async with websockets.connect(ws_url, max_size=2**28) as ws:
            page = Page(ws)
            await page.call("Page.enable")
            await page.call("Runtime.enable")
            # ---- loading state, then layout at three sizes
            for width, height in ((1440, 900), (1280, 800), (390, 844)):
                await page.size(width, height)
                await page.call("Page.navigate", url=base + "/")
                if width == 1440:
                    await asyncio.sleep(0.6)
                    await page.shot(out / "loading-1440x900.png")
                await page.until("!!document.querySelector('.replay canvas')", 40)
                await page.until("document.querySelector('.replay-title .badge') !== null", 40)
                await asyncio.sleep(2.5)
                layout = await page.eval(LAYOUT)
                c = layout["canvas"]
                layout["checks"] = {
                    "no_horizontal_overflow": layout["scroll_width"] <= layout["width"],
                    "sample_replay_labelled": layout["sample"],
                    "one_canvas": layout["canvases"] == 1,
                    "canvas_fully_above_fold": c["bottom"] <= layout["height"] and c["top"] >= 0,
                    "canvas_near_top": c["top"] <= 0.35 * layout["height"],
                    "stage_wider_than_controls_desktop": width < 1024 or c["width"] >= 0.55 * width,
                }
                results["layouts"][f"{width}x{height}"] = layout
                await page.shot(out / f"workspace-{width}x{height}.png")

            # ---- mode switching keeps one viewer (no remount) at 1440x900
            await page.size(1440, 900)
            await page.call("Page.navigate", url=base + "/")
            await page.until("document.querySelector('.replay-title .badge') !== null", 40)
            await page.eval("document.querySelector('.replay canvas').dataset.mark = 'm6'")
            kept = {}
            for label, name in (("Hidden Phase", "hidden"), ("Stress Test", "live"),
                                ("Failure Discovery", "discovery"), ("Policy Hardening", "hardening"),
                                ("Scenario Designer", "designer")):  # fmt: skip
                await page.click_text(label, "nav button")
                await asyncio.sleep(0.8)
                kept[name] = await page.eval(
                    "document.querySelectorAll('canvas').length === 1 && "
                    "document.querySelector('.replay canvas').dataset.mark === 'm6'"
                )
                await page.shot(out / f"mode-{name}-1440x900.png")
            results["flows"]["mode_switch_keeps_viewer"] = kept

            # ---- Scenario Designer: run a single paired trial, result auto-replays
            started = time.monotonic()
            await page.click_text("Run single paired trial")
            title = await page.until(
                f"({REPLAY_TITLE}).startsWith('Scenario ') && {REPLAY_TITLE}", 60
            )
            await asyncio.sleep(2.0)
            layout = await page.eval(LAYOUT)
            results["flows"]["scenario_single"] = {
                "replay_title": title, "seconds": round(time.monotonic() - started, 2),
                "legend": layout["legend"], "canvas_above_fold": layout["canvas"]["bottom"] <= 900,
                "result_line": await page.eval(
                    "document.querySelector('.result-box')?.textContent ?? null"),
            }  # fmt: skip
            await page.shot(out / "flow-scenario-1440x900.png")

            # ---- Failure Discovery: default CMA-ES search, discovered failure auto-replays
            await page.click_text("Failure Discovery", "nav button")
            await asyncio.sleep(0.5)
            started = time.monotonic()
            await page.click_text("Discover Failure")
            title = await page.until(
                f"({REPLAY_TITLE}).startsWith('Failure Discovery') && {REPLAY_TITLE}", 120
            )
            await asyncio.sleep(2.0)
            results["flows"]["discovery"] = {"replay_title": title,
                                             "seconds": round(time.monotonic() - started, 2)}  # fmt: skip
            await page.shot(out / "flow-discovery-1440x900.png")

            # ---- Policy Hardening: replay the sample candidate, then a small live run
            await page.click_text("Policy Hardening", "nav button")
            await asyncio.sleep(0.5)
            await page.click_text("Replay Baseline vs. Candidate")
            await page.until(f"({REPLAY_TITLE}).startsWith('Policy Hardening')", 60)
            await asyncio.sleep(1.5)
            sync = await page.eval("""(() => {
              const legend = [...document.querySelectorAll('.legend-overlay > div')].map(d => d.textContent);
              const slider = document.querySelector('.controls input[type=range]');
              return { legend, frames: Number(slider.max) + 1, badge: document.querySelector('.replay-head .badge.ok, .replay-head .badge.warn')?.textContent };
            })()""")
            results["flows"]["hardening_replay"] = sync
            await page.shot(out / "flow-hardening-replay-1440x900.png")
            await page.eval("""(() => { const s = [...document.querySelectorAll('select')]
              .find(e => e.offsetParent && [...e.options].some(o => o.value === 'm2-100N-500' && o.text.includes('N @')));
              s.value = 'm2-100N-500'; s.dispatchEvent(new Event('change', { bubbles: true }));
              const g = [...document.querySelectorAll('label.field')].find(l => l.textContent.startsWith('generations')).querySelector('input');
              const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
              set.call(g, '3'); g.dispatchEvent(new Event('input', { bubbles: true })); return true; })()""")
            await asyncio.sleep(0.3)
            started = time.monotonic()
            await page.click_text("Harden Controller")
            await asyncio.sleep(1.5)
            await page.shot(out / "flow-hardening-progress-1440x900.png")
            title = await page.until(
                f"({REPLAY_TITLE}).startsWith('Policy Hardening') && !!document.querySelector('.badge.live') "
                "&& [...document.querySelectorAll('.badge.live')].some(b => b.textContent.includes('Executed now · hd-')) "
                f"&& {REPLAY_TITLE}", 180)  # fmt: skip
            await asyncio.sleep(2.0)
            results["flows"]["hardening_run"] = {
                "replay_title": title, "seconds": round(time.monotonic() - started, 2),
                "status_pill": await page.eval("document.querySelector('.status-pill')?.textContent"),
                "summary": await page.eval("[...document.querySelectorAll('table.compare tr')].map(r => r.textContent)"),
            }  # fmt: skip
            await page.shot(out / "flow-hardening-run-1440x900.png")

            # ---- error state: an unknown episode deep link
            await page.call(
                "Page.navigate", url=base + "/?tab=hardening&hreplay=preset-v1:nope:baseline"
            )
            err = await page.until("document.querySelector('.stage .error')?.textContent", 40)
            results["flows"]["error_state"] = err
            await page.shot(out / "state-error-1440x900.png")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--out", default="runs/m6/screens")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(run(args.base, out))
    (out / "ui-checks.json").write_text(json.dumps(results, indent=1))
    failed = [f"{k}.{c}" for k, v in results["layouts"].items() for c, ok in v["checks"].items()
              if not ok]  # fmt: skip
    failed += [m for m, ok in results["flows"]["mode_switch_keeps_viewer"].items() if not ok]
    print(json.dumps({k: v["checks"] for k, v in results["layouts"].items()}, indent=1))
    print(json.dumps(results["flows"], indent=1)[:4000])
    print("FAILED: " + ", ".join(failed) if failed else "all layout and mode checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
