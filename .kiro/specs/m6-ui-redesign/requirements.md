# Requirements: M6.1 — Simulation-first UI rebuild

UI-only. Physics, policy, optimizer, validation, job semantics and endpoint contracts are
unchanged. The layout follows the written description of the Nemesis reference (two-column hero,
dark stage, pill navigation, transport bar). No Nemesis source, CSS, assets or copy is used: the
repository has no license file (checked: `LICENSE`, `LICENSE.md`, `LICENSE.txt`, `web/LICENSE`
all 404 on 2026-11-22), so the implementation is written from the behavior description only.

## R1 — Information hierarchy
1.1 THE first screen SHALL show, without scrolling at 1440×900 and 1280×800: the active mode's
compact input, its single primary action, the 3D viewer, the identities of both robots, the attack
interval and each robot's outcome (survived, or fell and when).
1.2 ONLY the active mode's controls SHALL render in the hero; detailed parameters SHALL be behind
an "Advanced" disclosure and detailed reports SHALL render below the hero.

## R2 — Shell and navigation
2.1 THE header SHALL show the product name and pill navigation with five modes: Design scenario,
Stress test, Discover failure, Harden controller, Results, with a visible active state.
2.2 Existing deep links (`?tab=designer|hidden|live|discovery|hardening`, `?replay=`, `?run=`,
`?scenario_replay=`, `?hrun=`, `?hreplay=`, `?dreplay=`, `?frame=`) SHALL keep working; new aliases
`design|stress|discover|harden|results` SHALL also work.
2.3 Below 1024 px the navigation SHALL scroll horizontally inside its own container with an
overflow cue, and the page SHALL NOT overflow horizontally.

## R3 — Hero layout
3.1 At ≥ 1024 px the hero SHALL be a grid of about 34% controls / 66% viewer with a 36–56 px gap
where width allows, inside a 1200–1360 px content width.
3.2 THE canvas SHALL keep a 16:11 aspect ratio bounded by the viewport height (minimum 320 px) so
that it and the transport bar fit above the fold at 1280×800 and 1440×900.
3.3 Below 1024 px the viewer SHALL come first, inside the first viewport at 390×844, with ≥ 16 px
gutters.

## R4 — Replay stage
4.1 ONE ReplayViewer instance SHALL stay mounted across all five modes; mesh data SHALL be fetched
once.
4.2 THE stage SHALL be dark with a subtle grid; the opaque robot neutral, the comparison robot a
translucent periwinkle ghost; each SHALL carry an in-scene text label (name and live status) so
identity does not depend on colour.
4.3 HUD chips SHALL show the attack (force, direction, interval), friction and payload when they
differ from nominal, and provenance (sample / precomputed, re-simulated + verification, recorded
live result).
4.4 THE transport SHALL contain a round play/pause button (≥ 44×44 px), a scrubber with the attack
band and fall markers for both robots, elapsed/duration, speed choice, jump to shove, reset view
and fullscreen.
4.5 WHILE a job runs or a replay loads, THE stage SHALL keep the last replay and label it
"previous" with a compact in-stage status; it SHALL NOT cover the page.
4.6 Labels SHALL distinguish matched-condition ghost, original nominal, frozen baseline and
unpromoted candidate; no label SHALL say promoted, improved or safe.

## R5 — Mode controls
5.1 Design: sample chips, prompt composer with compile, one-line validation, primary "Run
scenario"; parameters and suite run under Advanced.
5.2 Stress: the verified 150 N preset, primary "Run 16 trials", compact trial chips.
5.3 Discover: algorithm segmented control, bounded budget, primary "Discover failure"; shrink and
seed under Advanced.
5.4 Harden: failure selector, primary "Harden controller", candidate status, secondary replay.
5.5 Results: precomputed evidence entries and recent runs with replay actions.

## R6 — Visual system
6.1 Colours, spacing (4/8/12/16/24/32/48), radii and type sizes SHALL come from CSS custom
properties; components SHALL NOT hard-code colours except SVG/Three.js values taken from the same
palette constants.
6.2 Primary buttons ≥ 44 px high; peer chips and buttons ≥ 8 px apart; one primary button per
mode; text contrast ≥ 4.5:1 for body text on the dark surfaces.

## R7 — Below the fold
7.1 A compact summary strip SHALL follow the hero (scenario, attack, both outcomes, source, link
to evidence), then the active mode's evidence as expandable sections containing all M2–M6 tables,
charts, provenance and reproduce commands that existed before.

## R8 — Verification
8.1 Automated browser checks SHALL measure at 1440×900, 1280×800, 1024×768 and 390×844:
canvas, transport and primary action above the fold (desktop), no horizontal overflow, no
overlapping interactive elements, ≥ 8 px gaps between adjacent buttons, ≥ 40 px primary height,
one persistent canvas across modes, and SHALL run the design, stress, discovery and hardening
flows against the real backend.
8.2 Python fast and integration suites, `tsc` and the Vite build SHALL pass.
