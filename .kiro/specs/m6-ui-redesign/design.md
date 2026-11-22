# Design: M6.1 — Simulation-first UI rebuild

## UI0 audit (before screenshots in `runs/m6ui/before/`)

| Size | Where it crowds |
| --- | --- |
| 1440×900 | The Design panel stacks four cards (action, describe, parameters, task) plus the intro. The stage head carries the title, two badges and a fullscreen button on two lines. Legend chips sit over the light canvas. |
| 1280×800 | As at 1440. In Harden mode "Harden Controller" is below the fold and competes with a second full-width blue button. |
| 1024×768 | The side column is about 330 px: "Run single" and "Run suite" stack, the seed placeholder is clipped ("blank = new"), example chips wrap to two lines, the Harden selects truncate, and the status pill wraps. The transport and outcome lines are below the fold. |
| 390×844 | The viewer is first (good). Nav pills wrap to two rows. The readout row is clipped on the right. Timeline markers were clipped in M6. |

Every mode renders the full parameter form in the hero. Robots have no in-scene labels, so
identity depends on the legend and colour.

## Component inventory

| Component | Decision |
| --- | --- |
| `App.tsx` | **Restructure** → `WorkspaceShell` (header, pill nav, hero grid, summary strip, evidence). Keeps the replay state, `show()`, deep links and refresh callbacks. |
| `Workspace.tsx` (`ModeContext`, `Details`, `Panel`) | **Keep** the portal mechanism. Rename `Panel` → `ModeIntro`. Add `Advanced` (disclosure), `Evidence` (expandable section) and `JobContext`, which a mode uses to report a running job to the stage. |
| `ReplayViewer.tsx` | **Restructure** → `ReplayStage` (persistent renderer, dark scene, in-scene labels, HUD) + `Transport`. Data contract unchanged. |
| `ScenarioDesigner.tsx` | **Restructure** the controls: chips, composer, validation line, Run; parameters and suite under Advanced. The preview, results and history move to evidence. |
| `LiveStress.tsx` | **Restructure**: preset summary, Run 16 trials, trial chips; reference and history to evidence. |
| `Discovery.tsx` | **Restructure**: segmented algorithm, budget, Discover; shrink and seed under Advanced; plot, table and comparison to evidence. |
| `Hardening.tsx` | **Restructure**: failure select, Harden controller, status, compact summary, secondary replay; generations and seed under Advanced. |
| `HiddenPhase.tsx` | **Replace** as a mode with `Results.tsx`. The sweep chart, stance table and failing-timing chips become Results evidence. |
| `styles.css` | **Replace** with a token-based dark theme. |

## Layout tokens

`--content-max: 1320px`, `--gutter: clamp(16px, 3vw, 32px)`, `--hero-gap: clamp(24px, 3.5vw, 56px)`.
Spacing `--s1..--s7 = 4, 8, 12, 16, 24, 32, 48 px`. Hero grid `minmax(0, 34fr) minmax(0, 66fr)`.
Canvas `aspect-ratio: 16/11; height: auto; max-height: max(320px, calc(100vh - 250px))`.

Palette (original variable names): `--bg #0a0a0a`, `--surface #101010`, `--surface-2 #161616`,
`--border #232322`, `--text #ededeb`, `--muted #a3a19c`, `--accent #ff5b35`, `--ok #d7f25d`,
`--candidate #8b93f8`, `--warn #f2b45a`, `--numerical #c58af9`. The Three.js scene uses the same
values from `theme.ts`.

## Replay stage

- **Labels.** Two absolutely positioned labels per frame, at the projected trunk positions plus
  0.28 m. They are updated through refs inside the render loop, with no React re-render. The ghost
  label is lifted by one label height so coincident robots never overlap.
- **Short names.** The opaque robot is "Scenario" or "Shoved" or "Candidate"; the ghost is "Matched
  baseline" or "Original nominal" or "Nominal" or "Frozen baseline". Names derive from the replay
  mode and ghost kind, with the full server label in the legend.
- **Transport.** A custom track holds the push band, a fall marker for each robot, the reflex
  activation band and the playhead, with a transparent `<input type=range>` on top for pointer and
  keyboard. Speeds 0.1/0.25/0.5/1×.
- **Job state.** `JobContext` collects `{mode: label}` for running jobs. When any job runs or a
  replay loads, the stage shows "Previous replay · <job> running" and the replay stays.
- **Camera.** Reset view restores the default three-quarter offset relative to the tracked target.

## Evidence

The summary strip under the hero is built from the current replay: title, attack, outcome chips
for both robots, source, and a link to `#evidence`. Each mode's existing report cards become
`<Evidence title summary>` disclosures (open by default for the first one).
