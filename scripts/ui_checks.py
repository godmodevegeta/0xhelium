"""Automated browser checks of the simulation-first workspace (M6.1), over the Chrome DevTools
Protocol against a running server (`uv run robustlab serve`). Real mouse and keyboard events.

Per viewport (1440x900, 1280x800, 1024x768, 390x844) and per mode:
- no horizontal page overflow;
- desktop: canvas, transport and the mode's primary action fully above the fold;
  mobile: canvas inside the first viewport, >= 16 px side gutters;
- no two visible interactive elements in the hero overlap; adjacent buttons on a row are
  >= 8 px apart; primary buttons >= 40 px high; the play button >= 44x44; form fields are labelled;
- non-chip buttons do not clip their text.
Then (1440x900): one persistent canvas and an unchanged replay across all five modes; camera
orbit, reset view and fullscreen; keyboard focus ring; the Design, Stress, Discover and Harden
flows end to end, including the "previous replay" state while jobs run; an error state; and,
with --nokey-base, the no-OpenRouter fallback. Writes screenshots and ui-checks.json to --out.

Usage: uv run python scripts/ui_checks.py [--base URL] [--nokey-base URL] [--out DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path

import websockets
from screenshot import launch  # scripts/ is on sys.path when run as a script

SIZES = ((1440, 900), (1280, 800), (1024, 768), (390, 844))
MODES = (("design", "Design scenario"), ("stress", "Stress test"), ("discover", "Discover failure"),
         ("harden", "Harden controller"), ("results", "Results"))  # fmt: skip
TITLE = "(document.querySelector('.stage-title strong')?.textContent ?? '')"

GEOMETRY = r"""(() => {
  const vis = (e) => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none' && e.offsetParent !== null; };
  const rect = (e) => { const r = e.getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height,
    r: r.right, b: r.bottom }; };
  const name = (e) => (e.getAttribute('aria-label') || e.textContent || e.placeholder || e.tagName).trim().slice(0, 40);
  const hero = document.querySelector('.hero');
  const items = [...hero.querySelectorAll('button, input, select, textarea, summary, a')]
    .filter(vis).filter(e => !e.closest('.track')).filter(e => !(e.closest('details:not([open])') && e.tagName !== 'SUMMARY'));
  const boxes = items.map(e => ({ el: e, n: name(e), ...rect(e) }));
  const overlaps = [];
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
    const a = boxes[i], b = boxes[j];
    if (a.el.contains(b.el) || b.el.contains(a.el)) continue;
    const ix = Math.min(a.r, b.r) - Math.max(a.x, b.x), iy = Math.min(a.b, b.b) - Math.max(a.y, b.y);
    if (ix > 1 && iy > 1) overlaps.push([a.n, b.n]);
  }
  const buttons = boxes.filter(b => b.el.tagName === 'BUTTON');
  const tight = [];
  for (let i = 0; i < buttons.length; i++) for (let j = 0; j < buttons.length; j++) {
    const a = buttons[i], b = buttons[j];
    if (i === j || b.x < a.r - 1) continue;
    const vo = Math.min(a.b, b.b) - Math.max(a.y, b.y);
    if (vo < 0.5 * Math.min(a.h, b.h)) continue;
    const between = buttons.some(c => c !== a && c !== b && c.x >= a.r - 1 && c.r <= b.x + 1 && Math.min(a.b, c.b) - Math.max(a.y, c.y) > 0);
    const gap = b.x - a.r;
    if (!between && gap < 7.5) tight.push([a.n, b.n, Math.round(gap)]);
  }
  const vgap = [];
  for (const a of buttons) for (const b of buttons) {
    if (a === b) continue;
    const ho = Math.min(a.r, b.r) - Math.max(a.x, b.x);
    const gap = b.y - a.b;
    if (ho > 4 && gap >= 0 && gap < 7.5) vgap.push([a.n, b.n, Math.round(gap)]);
  }
  const primaries = [...document.querySelectorAll('.hero .controls-col button.primary')].filter(vis).map(e => ({ n: name(e), ...rect(e) }));
  const unlabeled = items.filter(e => ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.tagName) && e.type !== 'range')
    .filter(e => !(e.labels && e.labels.length) && !e.getAttribute('aria-label') && !e.getAttribute('aria-labelledby')).map(name);
  const clipped = buttons.filter(b => !b.el.classList.contains('chip-btn') && b.el.scrollWidth > b.el.clientWidth + 1).map(b => b.n);
  const q = (s) => { const e = document.querySelector(s); return e && vis(e) ? rect(e) : null; };
  const col = document.querySelector('.controls-col').getBoundingClientRect();
  const stageCol = document.querySelector('.stage-col').getBoundingClientRect();
  const escapes = [...document.querySelectorAll('.controls-col *')].filter(vis).filter(e => {
    const r = e.getBoundingClientRect(); const box = e.closest('.table-scroll, table');
    return !box && (r.right > col.right + 1 || r.left < col.left - 1); }).map(name).slice(0, 6);
  const colsOverlap = stageCol.left < col.right - 1 && stageCol.top < col.bottom && col.top < stageCol.bottom;
  const play = q('.transport .play');
  return { vw: innerWidth, vh: innerHeight, scroll_w: document.documentElement.scrollWidth,
    canvas: q('.stage-card .canvas'), transport: q('.stage-card .transport'), controls: q('.controls-col'),
    primaries, play, overlaps, tight, vgap, unlabeled, clipped, escapes, colsOverlap,
    canvases: document.querySelectorAll('canvas').length, title: document.querySelector('.stage-title strong')?.textContent };
})()"""


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
                                 awaitPromise=True, userGesture=True)  # fmt: skip
        if "exceptionDetails" in result:
            raise RuntimeError(f"{expression[:80]}: {result['exceptionDetails']}")
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

    async def shot(self, path: Path, clip: dict | None = None) -> bytes:
        params = {"format": "png"}
        if clip:
            params["clip"] = {**clip, "scale": 1}
        data = base64.b64decode((await self.call("Page.captureScreenshot", **params))["data"])
        if path:
            path.write_bytes(data)
        return data

    async def center(self, expression: str) -> tuple[float, float]:
        r = await self.eval(f"(() => {{ const e = {expression}; if (!e) return null; e.scrollIntoView({{block: 'nearest'}});"
                            " const r = e.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; })()")  # fmt: skip
        if not r:
            raise RuntimeError(f"not found: {expression}")
        return r[0], r[1]

    async def mouse_click(self, x: float, y: float) -> None:
        for kind in ("mousePressed", "mouseReleased"):
            await self.call(
                "Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1
            )

    async def click(self, expression: str) -> None:
        await self.mouse_click(*await self.center(expression))

    async def drag(self, x0: float, y0: float, dx: float, dy: float) -> None:
        await self.call(
            "Input.dispatchMouseEvent", type="mousePressed", x=x0, y=y0, button="left", clickCount=1
        )
        for i in range(1, 11):
            await self.call(
                "Input.dispatchMouseEvent",
                type="mouseMoved",
                x=x0 + dx * i / 10,
                y=y0 + dy * i / 10,
                button="left",
                buttons=1,
            )
        await self.call(
            "Input.dispatchMouseEvent",
            type="mouseReleased",
            x=x0 + dx,
            y=y0 + dy,
            button="left",
            clickCount=1,
        )

    async def key(self, key: str, code: str, vk: int) -> None:
        for kind in ("keyDown", "keyUp"):
            await self.call(
                "Input.dispatchKeyEvent", type=kind, key=key, code=code, windowsVirtualKeyCode=vk
            )


def button(text: str, scope: str = "document") -> str:
    return (f"[...{scope}.querySelectorAll('button')].find(e => e.offsetParent !== null && !e.disabled "
            f"&& e.textContent.trim().startsWith({json.dumps(text)}))")  # fmt: skip


def nav(label: str) -> str:
    return button(label, "document.querySelector('nav.nav')")


def judge(g: dict, width: int, height: int) -> dict:
    desktop = width >= 1024
    inside = lambda r: (
        r is not None
        and r["y"] >= 0
        and r["b"] <= height + 0.5
        and r["x"] >= 0
        and r["r"] <= width + 0.5
    )
    checks = {
        "no_horizontal_overflow": g["scroll_w"] <= g["vw"],
        "one_canvas": g["canvases"] == 1,
        "no_overlapping_controls": not g["overlaps"],
        "button_gaps_ge_8px": not g["tight"] and not g["vgap"],
        "primary_height_ge_40": all(p["h"] >= 40 for p in g["primaries"]),
        "single_primary_action": len(g["primaries"]) == 1,
        "play_button_ge_44": g["play"] is not None
        and g["play"]["w"] >= 44
        and g["play"]["h"] >= 44,
        "fields_labelled": not g["unlabeled"],
        "no_clipped_button_text": not g["clipped"],
        "controls_stay_in_column": not g["escapes"],
        "columns_do_not_overlap": not g["colsOverlap"],
    }
    if desktop:
        checks["canvas_above_fold"] = inside(g["canvas"])
        checks["transport_above_fold"] = inside(g["transport"])
        checks["primary_above_fold"] = bool(g["primaries"]) and all(
            inside(p) for p in g["primaries"]
        )
        checks["viewer_about_two_thirds"] = (
            g["canvas"] is not None
            and 0.58 <= g["canvas"]["w"] / (g["canvas"]["w"] + g["controls"]["w"]) <= 0.72
        )
    else:
        checks["canvas_in_first_viewport"] = inside(g["canvas"])
        checks["gutters_ge_16"] = (
            g["canvas"] is not None
            and g["canvas"]["x"] >= 15.5
            and width - g["canvas"]["r"] >= 15.5
        )
    return checks


async def run(base: str, nokey: str | None, out: Path) -> dict:
    results: dict = {"base": base, "layouts": {}, "flows": {}}
    proc, ws_url = launch(1440, 900, 9336)
    try:
        async with websockets.connect(ws_url, max_size=2**28) as ws:
            page = Page(ws)
            await page.call("Page.enable")
            await page.call("Runtime.enable")

            # ---------------------------------------------------------- layouts, every mode
            for width, height in SIZES:
                await page.size(width, height)
                await page.call("Page.navigate", url=base + "/")
                if width == 1440:
                    await asyncio.sleep(0.5)
                    await page.shot(out / "state-loading-1440x900.png")
                await page.until("document.querySelector('.transport .play') !== null", 60)
                await page.until("!document.querySelector('.stage-overlay')", 60)
                for key, label in MODES:
                    await page.click(nav(label))
                    await asyncio.sleep(1.2)
                    await page.eval("window.scrollTo(0, 0)")
                    await asyncio.sleep(0.3)
                    g = await page.eval(GEOMETRY)
                    checks = judge(g, width, height)
                    results["layouts"][f"{key}-{width}x{height}"] = {
                        "checks": checks,
                        "geometry": {
                            k: g[k]
                            for k in (
                                "canvas",
                                "transport",
                                "primaries",
                                "overlaps",
                                "tight",
                                "vgap",
                                "unlabeled",
                                "clipped",
                                "escapes",
                            )
                        },
                    }
                    await page.shot(out / f"{key}-{width}x{height}.png")

            # ---------------------------------------------------------- persistence, camera, keyboard
            await page.size(1440, 900)
            await page.call("Page.navigate", url=base + "/")
            await page.until(
                "!document.querySelector('.stage-overlay') && !!document.querySelector('.transport .play')",
                60,
            )
            focus = []
            for _ in range(4):
                await page.key("Tab", "Tab", 9)
                focus.append(
                    await page.eval(
                        "(() => { const e = document.activeElement; const s = getComputedStyle(e); return [e.tagName, (e.textContent || '').trim().slice(0, 24), s.outlineStyle, s.outlineWidth]; })()"
                    )
                )
            results["flows"]["keyboard_focus"] = focus
            await page.shot(out / "state-keyboard-focus-1440x900.png")
            await page.eval("document.querySelector('canvas').dataset.mark = 'persist'")
            title0 = await page.eval(TITLE)
            kept = {}
            for key, label in MODES:
                await page.click(nav(label))
                await asyncio.sleep(0.6)
                kept[key] = await page.eval(
                    f"document.querySelectorAll('canvas').length === 1 && document.querySelector('canvas').dataset.mark === 'persist' && {TITLE} === {json.dumps(title0)}"
                )
            results["flows"]["mode_switch_keeps_viewer_and_replay"] = kept
            await page.click(nav("Design scenario"))
            await asyncio.sleep(0.5)
            # pause, then orbit: the canvas image must change; reset view must change it again
            if await page.eval(
                "document.querySelector('.transport .play').getAttribute('aria-label') === 'Pause'"
            ):
                await page.click("document.querySelector('.transport .play')")
            await asyncio.sleep(0.6)
            c = await page.eval(
                "(() => { const r = document.querySelector('.stage-card .canvas').getBoundingClientRect(); return {x: r.left, y: r.top, width: r.width, height: r.height}; })()"
            )
            h0 = hashlib.sha256(await page.shot(None, c)).hexdigest()
            await page.drag(c["x"] + c["width"] * 0.5, c["y"] + c["height"] * 0.6, 160, -40)
            await asyncio.sleep(0.8)
            h1 = hashlib.sha256(await page.shot(None, c)).hexdigest()
            await page.click("document.querySelector('[aria-label=\"Reset camera view\"]')")
            await asyncio.sleep(0.8)
            h2 = hashlib.sha256(await page.shot(None, c)).hexdigest()
            await page.click("document.querySelector('[aria-label=\"Fullscreen\"]')")
            await asyncio.sleep(0.8)
            fs = await page.eval(
                "document.fullscreenElement?.className ?? (document.querySelector('.stage-card.expanded') ? 'expanded (fullscreen refused by this browser)' : null)"
            )
            await page.shot(out / "state-fullscreen-1440x900.png")
            await page.eval(
                "document.fullscreenElement ? document.exitFullscreen() : (document.querySelector('.stage-card.expanded') && document.querySelector('[aria-label=\"Exit expanded view\"]').click())"
            )
            await asyncio.sleep(0.5)
            results["flows"]["camera"] = {
                "orbit_changes_view": h1 != h0,
                "reset_changes_view": h2 != h1,
                "fullscreen_element": fs,
            }
            # ---------------------------------------------------------- Design: run scenario
            await page.click(button("icy-heavy-shove"))
            await asyncio.sleep(1.0)
            await page.until(f"!{button('Run scenario')}.disabled", 20)
            started = time.monotonic()
            await page.click(button("Run scenario"))
            title = await page.until(f"{TITLE}.startsWith('Scenario ') && {TITLE}", 60)
            await page.until("!document.querySelector('.stage-overlay')", 30)
            await asyncio.sleep(1.5)
            labels = await page.eval(
                "[...document.querySelectorAll('.robot-label')].map(e => e.textContent)"
            )
            hud = await page.eval(
                "[...document.querySelectorAll('.hud-chip')].map(e => e.textContent)"
            )
            results["flows"]["design"] = {"title": title, "seconds": round(time.monotonic() - started, 2), "labels": labels, "hud": hud,
                                          "result": await page.eval("document.querySelector('.result-box')?.textContent")}  # fmt: skip
            await page.shot(out / "flow-design-1440x900.png")

            # ---------------------------------------------------------- Stress: run 16 trials
            await page.click(nav("Stress test"))
            await asyncio.sleep(0.5)
            started = time.monotonic()
            await page.click(button("Run 16 trials"))
            await asyncio.sleep(0.8)
            prev = await page.eval(
                "[...document.querySelectorAll('.hud-chip.prev')].map(e => e.textContent)"
            )
            await page.shot(out / "state-previous-replay-1440x900.png")
            title = await page.until(f"{TITLE}.startsWith('Stress test') && {TITLE}", 90)
            await asyncio.sleep(1.5)
            results["flows"]["stress"] = {"title": title, "seconds": round(time.monotonic() - started, 2), "previous_label_while_running": prev,
                                          "chips": await page.eval("document.querySelectorAll('.chips .chip').length")}  # fmt: skip
            await page.shot(out / "flow-stress-1440x900.png")

            # ---------------------------------------------------------- Discover: CMA-ES, budget 32
            await page.click(nav("Discover failure"))
            await asyncio.sleep(0.5)
            await page.click(
                button("32", "document.querySelector('[aria-labelledby=\"budget-label\"]')")
            )
            started = time.monotonic()
            await page.click(button("Discover failure", "document.querySelector('.controls-col')"))
            await asyncio.sleep(1.5)
            await page.shot(out / "flow-discover-running-1440x900.png")
            outcome = await page.until(
                f"({TITLE}.startsWith('Discovered failure') && {TITLE}) || (document.querySelector('.controls-col .bad')?.textContent)",
                150,
            )
            await asyncio.sleep(1.5)
            results["flows"]["discover"] = {
                "outcome": outcome,
                "seconds": round(time.monotonic() - started, 2),
            }
            await page.shot(out / "flow-discover-1440x900.png")

            # ---------------------------------------------------------- Harden: replay sample, then a short run
            await page.click(nav("Harden controller"))
            await asyncio.sleep(0.5)
            await page.click(button("Replay baseline vs candidate"))
            title = await page.until(f"{TITLE}.startsWith('Hardening') && {TITLE}", 60)
            await page.until("!document.querySelector('.stage-overlay')", 30)
            await page.eval(
                "(() => { const s = document.querySelector('.track input[type=range]'); const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; set.call(s, '140'); s.dispatchEvent(new Event('input', {bubbles: true})); })()"
            )
            await asyncio.sleep(1.0)
            sem = await page.eval(
                "({labels: [...document.querySelectorAll('.robot-label')].map(e => e.textContent), head: document.querySelector('.stage-title .xsmall').textContent, badge: document.querySelector('.stage-head .badge')?.textContent, falls: [...document.querySelectorAll('.track-fall')].map(e => e.className), push: !!document.querySelector('.track-push'), reflex: document.querySelectorAll('.track-reflex').length, page_text_promoted: /promoted (policy|controller)|is now live|improved policy/i.test(document.querySelector('.hero').textContent.replace('never a promoted policy', ''))})"
            )
            results["flows"]["harden_replay"] = {"title": title, **sem}
            await page.shot(out / "flow-harden-replay-1440x900.png")
            await page.eval(
                "(() => { const s = document.getElementById('harden-focus'); s.value = 'm2-100N-500'; s.dispatchEvent(new Event('change', {bubbles: true})); })()"
            )
            await page.click(
                "document.querySelector('.controls-col details.advanced:not([hidden]) > summary') && [...document.querySelectorAll('.controls-col details.advanced')].find(d => d.offsetParent !== null).querySelector('summary')"
            )
            await asyncio.sleep(0.3)
            await page.eval(
                "(() => { const g = [...document.querySelectorAll('label.field')].find(l => l.offsetParent && l.textContent.startsWith('generations')).querySelector('input'); const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; set.call(g, '3'); g.dispatchEvent(new Event('input', {bubbles: true})); })()"
            )
            started = time.monotonic()
            await page.click(button("Harden controller", "document.querySelector('.controls-col')"))
            await asyncio.sleep(1.2)
            await page.shot(out / "flow-harden-running-1440x900.png")
            await page.until(
                "[...document.querySelectorAll('.badge.live')].some(b => b.textContent.includes('Executed now · hd-'))",
                120,
            )
            title = await page.until(
                f"{TITLE}.startsWith('Hardening') && !document.querySelector('.stage-overlay') && {TITLE}",
                60,
            )
            await asyncio.sleep(1.5)
            results["flows"]["harden_run"] = {"title": title, "seconds": round(time.monotonic() - started, 2),
                                              "status_pill": await page.eval("document.querySelector('.status-pill')?.textContent"),
                                              "summary": await page.eval("[...document.querySelectorAll('.controls-col table.compare tr')].filter(r => r.offsetParent).map(r => r.textContent)")}  # fmt: skip
            await page.shot(out / "flow-harden-run-1440x900.png")
            await page.eval("document.getElementById('evidence').scrollIntoView()")
            await asyncio.sleep(0.6)
            await page.shot(out / "evidence-harden-1440x900.png")

            # ---------------------------------------------------------- Results + error state
            await page.call("Page.navigate", url=base + "/?tab=results")
            await page.until(
                "!document.querySelector('.stage-overlay') && !!document.querySelector('.transport .play')",
                60,
            )
            await page.eval("document.getElementById('evidence').scrollIntoView()")
            await asyncio.sleep(0.6)
            await page.shot(out / "evidence-results-1440x900.png")
            await page.call(
                "Page.navigate", url=base + "/?tab=harden&hreplay=preset-v1:nope:baseline"
            )
            err = await page.until(
                "document.querySelector('.stage-overlay .error')?.textContent", 60
            )
            results["flows"]["error_state"] = err
            await page.shot(out / "state-error-1440x900.png")

            # ---------------------------------------------------------- legacy deep links (M3-M6 URLs)
            links = {}
            import glob as _glob

            sc = sorted(Path(p).name for p in _glob.glob("runs/web/scenarios/sc-*"))
            ds = sorted(
                Path(p).name
                for p in _glob.glob("runs/web/discovery/ds-*")
                if (Path(p) / "discovery.json").exists()
                and json.loads((Path(p) / "discovery.json").read_text()).get("final")
            )
            ls_ = sorted(
                Path(p).name
                for p in _glob.glob("runs/web/live-stress/ls-*")
                if (Path(p) / "experiment.json").exists()
            )
            cases = [
                ("?tab=hidden&replay=611", "Recorded sweep", "Results"),
                (
                    "?tab=hardening&hreplay=preset-v1:m4-cmaes-s0:baseline",
                    "Hardening",
                    "Harden controller",
                ),
            ]
            if sc:
                cases.append(
                    (
                        f"?tab=designer&scenario_replay={sc[-1]}:0:original",
                        "Scenario run",
                        "Design scenario",
                    )
                )
            if ds:
                cases.append(
                    (
                        f"?tab=discovery&run={ds[-1]}&dreplay=final",
                        "Discovered failure",
                        "Discover failure",
                    )
                )
            if ls_:
                cases.append((f"?tab=live&run={ls_[-1]}", "Sample", "Stress test"))
            for query, prefix, tab in cases:
                await page.call("Page.navigate", url=base + "/" + query)
                try:
                    await page.until(
                        f"{TITLE}.startsWith({json.dumps(prefix)}) && !document.querySelector('.stage-overlay')",
                        60,
                    )
                    active = await page.eval(
                        "document.querySelector('nav.nav [aria-current=page]')?.textContent"
                    )
                    links[query] = active == tab
                except TimeoutError:
                    links[query] = False
            results["flows"]["deep_links"] = links

            # ---------------------------------------------------------- no OpenRouter key
            if nokey:
                await page.call("Page.navigate", url=nokey + "/")
                await page.until(
                    "!document.querySelector('.stage-overlay') && !!document.querySelector('.transport .play')",
                    60,
                )
                await asyncio.sleep(1.0)
                results["flows"]["no_key"] = await page.eval(
                    "({prompt_disabled: document.getElementById('scenario-prompt').disabled, placeholder: document.getElementById('scenario-prompt').placeholder, compile_disabled: "
                    + button("Compile", "document")
                    + " === undefined, run_enabled: !"
                    + button("Run scenario")
                    + "?.disabled, sample: document.querySelector('.stage-head .badge')?.textContent})"
                )
                await page.shot(out / "state-no-key-1440x900.png")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--nokey-base", default=None)
    parser.add_argument("--out", default="runs/m6ui/after")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(run(args.base, args.nokey_base, out))
    (out / "ui-checks.json").write_text(json.dumps(results, indent=1))
    failed = [
        f"{k}.{c}" for k, v in results["layouts"].items() for c, ok in v["checks"].items() if not ok
    ]
    flows = results["flows"]
    failed += [
        f"persist.{m}" for m, ok in flows["mode_switch_keeps_viewer_and_replay"].items() if not ok
    ]
    failed += [f"camera.{k}" for k, v in flows["camera"].items() if not v]
    failed += [f"deep_link {q}" for q, ok in flows["deep_links"].items() if not ok]
    if flows["harden_replay"].get("page_text_promoted"):
        failed.append("harden_replay.promotion_wording")
    print(json.dumps(flows, indent=1)[:6000])
    print(
        "FAILED: " + ", ".join(failed)
        if failed
        else f"all {sum(len(v['checks']) for v in results['layouts'].values())} layout checks and flow checks passed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
