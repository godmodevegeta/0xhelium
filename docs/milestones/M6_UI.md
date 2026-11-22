# M6.1 — Simulation-first UI rebuild

Status: **implemented and checked locally; awaiting UI4 acceptance.** This is a UI-only change.
The backend is unchanged: physics, policy, optimizer, validation, jobs and endpoint contracts.
`git diff m6-cpu-hardening -- src/ tests/` is empty.

- Spec: `.kiro/specs/m6-ui-redesign/`.
- Branch: `m6-ui-redesign`, from `m6-cpu-hardening` (`fef2ad0`, not yet merged).
- Commits:
  - `380d6e4`: spec and UI0 audit.
  - `9542e3b`: rebuild.
  - `f2316eb`: column containment fix and its check.
  - `72f7d95`: caption contrast.

## Reference and provenance

The layout follows the written description of the Nemesis reference: a 1:2 two-column hero, a dark
stage, pill navigation, in-stage HUD, labels and transport, with results below.

- **License check.** The repository has no license file: `LICENSE`, `LICENSE.md`, `LICENSE.txt`
  and `web/LICENSE` all returned 404 on 2026-11-22.
- **Reuse.** No Nemesis source, CSS, assets, brand or copy was used. Every component and style
  here is original.
- **Reference site.** The deployment at nemesis.divagr.com timed out from this machine, so no
  reference screenshot was taken.
- **Dependencies.** No new dependencies; React 18, Three.js 0.187 and Vite 6 as before.

## UI0 audit (before, `runs/m6ui/before/`)

| Size | Problem |
| --- | --- |
| all | Every mode put its whole form in the hero: Design had four stacked cards. The stage head took two lines of badges. Robots had no in-scene labels, so identity relied on colour and a legend. The light canvas made the ghost hard to read. |
| 1280×800 | In Harden, "Harden Controller" was below the fold and a second full-width blue button competed with it. |
| 1024×768 | The control column was about 330 px. "Run single" and "Run suite" stacked, the seed placeholder clipped, example chips wrapped onto two lines, selects truncated, and the transport and outcomes fell below the fold. |
| 390×844 | Navigation wrapped to two rows and the readout row clipped. |

The component inventory and its decisions (keep, restructure, replace) are in `design.md`.

## What changed

- **Shell.** The header is centred and sticky: product name on the left, pill navigation on the
  right. The five modes are **Design scenario · Stress test · Discover failure · Harden
  controller · Results**. The active mode is a filled pill; a running job adds an orange dot.
  - Below 1024 px the nav scrolls horizontally inside its own pill, with a fade at the edge, and
    the active tab scrolls into view.
  - New URL aliases are `design|stress|discover|harden|results`. Every old `?tab=` value still
    works, and the URL is updated when the mode changes.
- **Theme.** Colours, spacing, radii and type sizes are CSS custom properties.
  - Spacing scale: 4, 8, 12, 16, 24, 32, 48 px.
  - Colours: `--bg #0a0a0a`, `--surface #101010`, `--text #ededeb`, `--muted #a3a19c`,
    `--border #232322`.
  - Accents: `--accent #ff5b35` for the attack cue and primary actions, `--ok #d7f25d` for
    survived, `--candidate #8b93f8` for the ghost and candidate. I added `--fell #ff4f6e` so a
    fall is distinct from the orange attack colour.
  - The Three.js scene and SVG charts read the same values from `web/src/theme.ts`.
- **Hero.** A grid of `34fr / 66fr` with a gap of `clamp(24px, 3.5vw, 56px)`, inside a
  1320 px content width. The canvas keeps a 16:11 aspect ratio, capped at
  `max(320px, 100vh − 330px)`. Below 1024 px the stage comes first, at `clamp(280px, 46vh, 440px)`.
- **ReplayStage** replaces `ReplayViewer`. There is still one renderer, scene and camera per
  robot model, and the data contract is unchanged.
  - **Dark stage.** A charcoal floor, a subtle grid that follows the robots, and fog. The robot is
    neutral grey; the ghost is translucent periwinkle and is drawn *before* the robot, so where
    they coincide (before the shove) the real robot shows cleanly.
  - **In-scene text labels.** Each robot carries its name and live state ("upright" /
    "FELL · tilt > 45° at 2.50 s"). The labels follow the projected trunk and stack when the
    robots overlap.
  - **Names by replay kind:**

    | Replay | Opaque robot | Translucent ghost |
    | --- | --- | --- |
    | Stress test / recorded sweep | Shoved robot | Nominal · no shove |
    | Scenario | Scenario (shoved) | Matched baseline, or Original nominal |
    | Hardening | Candidate · frozen policy + reflex | Frozen baseline, or Frozen baseline · no shove |

  - **Head and HUD.** The head shows the title, "Opaque: … · Translucent: …" and a provenance
    badge: Sample · precomputed, re-simulated; re-simulated and matching the record; or a recorded
    server result. HUD chips show the shove (force, direction word and angle, interval) and, when
    they differ from nominal, the floor friction and the payload.
  - **Transport:**
    - a 44 px round play/pause button;
    - a custom track with the orange shove band, a solid red fall marker for the opaque robot, a
      dashed periwinkle marker for the ghost, a reflex-activation band, and a transparent range
      input on top for pointer and keyboard;
    - elapsed / duration, tilt, trunk height and feet;
    - a key, the speed choice (0.1–1×), jump to shove, reset view and fullscreen.
  - **Fullscreen fallback.** When the browser refuses fullscreen, the stage expands to cover the
    viewport instead (Esc closes it). Headless Chrome refuses fullscreen, so the checks exercise
    this path.
  - **Loading and job state.** While a replay loads or any job runs, the last replay stays, a
    "Previous replay · kept until the new result arrives" chip appears, and a compact status pill
    sits inside the stage. Nothing covers the page.
- **Modes.** Each mode shows only its own controls, with one primary button, and puts extra
  settings in an "Advanced" disclosure:

  | Mode | Hero controls | Primary action | Under Advanced |
  | --- | --- | --- | --- |
  | Design | sample chips, scenario summary with one-line validation, then the optional composer and example chips | Run scenario | all parameters, 16-trial suite |
  | Stress | preset summary with the precomputed reference; trial chips after a run | Run 16 trials | seed |
  | Discover | segmented algorithm, budget presets 32/64/128 | Discover failure | custom budget, shrink, seed |
  | Harden | candidate status pill, failure select, compact baseline-vs-candidate table, episode comparison with a ghost choice and a secondary replay button | Harden controller | generations, seed |
  | Results (new; replaces the "Hidden Phase" tab) | the M2 sweep with its 11 failing timings as chips, M3–M6 summaries with replay buttons | Replay a recorded failure | — |

  In Design the composer and example chips sit below the Run button. Design, Stress, Discover and
  Harden all auto-replay their new results.
- **Below the hero.**
  - **Summary strip:** the replay title, the shove, each robot's outcome, the source, and a link
    to the evidence.
  - **Evidence section:** the active mode's records as expandable cards. These hold every M2–M6
    table and chart that existed before, plus provenance and reproduce commands: sweep chart,
    stance table, search plot and record, Random vs CMA-ES comparison, scenario preview and case
    tables, hardening history, training/validation tables, grids, parameters and provenance, and
    previous runs.

## UI4 verification

`scripts/ui_checks.py` drives headless Chrome over CDP with real mouse and keyboard events against
the running server, plus a second server with no `OPENROUTER_API_KEY`. The final run was on the
build of `72f7d95` (`runs/m6ui/ui-checks.log`, `runs/m6ui/after/ui-checks.json`): **290 layout
checks and all flow checks passed.**

**Layout checks** run for every mode at every size (5 × 4):

| Check | Rule |
| --- | --- |
| page overflow | no horizontal overflow |
| canvas | exactly one canvas |
| overlap | no two visible interactive hero elements overlap |
| spacing | adjacent buttons ≥ 8 px apart horizontally and vertically |
| primary action | exactly one primary button, ≥ 40 px high |
| play button | ≥ 44 × 44 px |
| labels | every field labelled |
| clipping | no non-chip button clips its text |
| containment | every control stays inside its column; the two columns do not overlap |
| desktop | canvas, transport and primary button fully above the fold; viewer 58–72% of the hero width |
| mobile | canvas inside the first viewport; ≥ 16 px gutters |

Measured positions (identical in all five modes at each size):

| Size | Canvas (top–bottom, width) | Transport bottom | Primary button bottom |
| --- | --- | --- | --- |
| 1440×900 | 159–704 px, 794 px | 827 | 399–544 (by mode) |
| 1280×800 | 159–629 px, 771 px | 751 | 399–544 |
| 1024×768 | 194–613 px, 610 px | 736 | 446–645 |
| 390×844 | 219–607 px, 356 px | 772 | below the stage (stacked) |

**Flows**, all on the real backend:

| Flow | Result |
| --- | --- |
| Mode switching | All five modes kept one canvas, the same element, and the same replay. |
| Camera | Orbit drag changed the view; reset view changed it again (canvas image hashes). |
| Fullscreen | The expanded fallback opened, because headless Chrome refuses fullscreen. |
| Keyboard | Tab moved through the nav buttons with a 2 px focus outline. |
| Design | "Run scenario" auto-replayed "Scenario icy-heavy-shove vs matched baseline" in 2.3 s, labelled "Scenario (shoved) · upright" and "Matched baseline · upright", with HUD chips for shove, μ 0.30 and 3 kg. |
| Stress | "Run 16 trials" showed the previous-replay chip while running and auto-replayed in 5.3 s. |
| Discover | CMA-ES with budget 32 auto-replayed its discovered failure in 6.7 s. |
| Harden replay | The sample-candidate replay was labelled "Candidate · frozen policy + reflex · upright" and "Frozen baseline · FELL · tilt > 45° at 2.50 s", with a shove band, a ghost fall marker and a reflex band. The badge read "matches candidate record". No "promoted", "now live" or "improved policy" wording appears in the hero. |
| Harden run | A 3-generation focus run auto-replayed in 6.1 s, and the status read "Candidate · pending M7 regression evaluation". |
| Deep links | `?tab=hidden&replay=611`, `?tab=hardening&hreplay=…`, `?tab=designer&scenario_replay=…`, `?tab=discovery&run=…&dreplay=final` and `?tab=live&run=…` all opened the right mode and replay. |
| Error state | An unknown episode showed the 404 message inside the stage. |
| No OpenRouter key | The prompt and compile were disabled with an explanation, Run stayed enabled, and the sample was labelled honestly. |

Other checks (`runs/m6ui/validation.txt`; the backend is untouched):

| Check | Result |
| --- | --- |
| `ruff check`, `ruff format --check` | clean |
| fast suite | 265 passed (26.2 s wall, with a browser and a server running; the Stop hook limit is 30 s) |
| integration | 36 passed |
| `tsc --noEmit`, `vite build` | pass |

### Manual screenshot review

I looked at every image in `runs/m6ui/after/` at full resolution. Things found and fixed during
review:

- **Ghost drawn over the robot.** The ghost covered the robot before the shove, tinting it blue.
  Fixed with the render order.
- **Labels too high.** They floated well above the robots; the lift is now 0.14 m.
- **HUD wrapping.** The provenance chip wrapped into the HUD at 1280 px; it moved to the stage
  head.
- **Cramped transport.** Fixed with a two-row layout.
- **Design CTA below the fold.** "Run scenario" was below the fold at 1280×800 until the composer
  moved under it.
- **Navigation clipped at 1024 px.** The tagline is now hidden below 1200 px.
- **Mobile canvas too small.** At 16:11 the robot was tiny; the stage now uses a viewport-height
  rule.
- **Mobile transport wrap.** Fullscreen wrapped alone onto a row; the key now takes its own row.
- **Friction floor.** It showed a glossy hard-edged plate and did not follow the robots; now matte
  and tracking.
- **Evidence spacing.** Evidence cards had no vertical rhythm.
- **Controls overflowing at 1024 px.** In Harden, the select and table widened the control column
  under the stage. The automated checks missed this, so I added the containment and column-overlap
  checks, then fixed it with `min-width: 0` and select/table constraints.

Key images:

- `design-1440x900.png`, `harden-1280x800.png`, `stress-1024x768.png`, `discover-390x844.png`;
- `flow-design`, `flow-stress`, `flow-discover`, `flow-harden-replay` and `flow-harden-run` (each
  `-1440x900.png`);
- `state-previous-replay`, `state-loading`, `state-error`, `state-no-key`, `state-fullscreen` and
  `state-keyboard-focus` (each `-1440x900.png`);
- `evidence-results-1440x900.png`, `evidence-harden-1440x900.png`.

## How to run

```bash
(cd web && npm ci && npm run build) && uv run robustlab serve        # http://127.0.0.1:8000
uv run python scripts/ui_checks.py --out runs/m6ui/after             # against the running server
# optional no-key fallback check: a second server without the key
env -u OPENROUTER_API_KEY ROBUSTLAB_RUNS_DIR=/tmp/nokey uv run robustlab serve --port 8001
uv run python scripts/ui_checks.py --nokey-base http://localhost:8001
```

## Remaining visual issues and limitations

- **Labels on the robot.** When the robots stand close and the camera is low, the ghost's label
  can sit over the opaque robot's body for a few frames.
- **Harden summary table.** At 1024 px the table wraps its row labels onto two or three lines
  ("validation-preview"). It is readable but tall.
- **Stage head on mobile.** At 390 px the provenance badge wraps to two lines, so the head is
  about 110 px tall.
- **Fullscreen in a real browser.** Real `requestFullscreen` was not exercised; headless Chrome
  refuses it and the expanded fallback ran instead.
- **Contrast** (computed WCAG ratios on `#101010`): text 16.2:1, muted 7.4:1, accent 6.2:1,
  candidate 6.9:1, fall 6.0:1, dark text on the orange button 6.4:1, and the caption grey
  `--faint #84827c` 5.0:1 (4.7:1 on the `#161616` inputs). The first faint grey (`#6f6d68`,
  3.7:1) failed 4.5:1 and was lightened after review. Contrast was computed from the palette, not
  measured in the browser.
- **Fast-suite time.** 26.2 s while a browser and a server were running, close to the Stop hook's
  30 s limit.
- **Previous-replay chip.** It is driven by jobs started in this browser tab; a job started
  elsewhere (CLI, another tab) shows only through the 409 busy response.
