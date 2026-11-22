# Implementation Plan: M6.1 — Simulation-first UI rebuild

- [x] UI0 Audit: before screenshots at 1440×900, 1280×800, 1024×768, 390×844; crowding notes and
  component inventory (design.md); reference license check
- [x] UI1 Shell and theme: tokens, header, pill navigation, responsive 1:2 hero, summary strip,
  evidence section; all behaviors wired; screenshots
- [x] UI2 Replay stage: dark scene, in-scene labels, HUD chips, transport, previous-replay state,
  reset view, fullscreen; semantics check for baseline / matched / original / candidate
- [x] UI3 Mode controls: Design, Stress, Discover, Harden, Results with one CTA each and Advanced
  disclosure; evidence moved below
- [x] UI4 Verify: extended `scripts/ui_checks.py` (four sizes, overlap and gap checks, flows),
  screenshots reviewed, Python suites, tsc, build; `docs/milestones/M6_UI.md`
