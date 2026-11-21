# Third-Party Notices

Robustness Studio is licensed under the MIT License (see `LICENSE`). This repository also contains
or depends on third-party material listed below. Listing a project here records attribution and
license terms only. It does not imply any affiliation with, or endorsement by, the authors of these
projects.

All vendored files are pinned to one upstream commit and hashed in `assets/assets.lock.json`.
`uv run robustlab verify-assets` checks them.

## Vendored files

### weeble (MIT)

- Repository: <https://github.com/mottopanikeiku/weeble>
- Commit: `60878341072256ccc68f6118ec23a298c55e1bb9`
- Files taken from this commit:
  - `policies/go1-walk-s0.json` (pretrained Go1 walking policy weights and normalizer)
  - `assets/go1/walk.xml`, `assets/go1/walk.provenance.json`, `assets/go1/LICENSE-Playground`,
    `assets/go1/LICENSE-Menagerie` and `assets/go1/assets/*.stl` (redistributed through weeble; their
    original licenses are listed in the next two sections)
  - `tests/reference/weeble_sim.py`: a verbatim copy of weeble `eval/sim.py`, used only as a test
    oracle and never imported by `src/`

**Policy license assumption.** `policies/go1-walk-s0.json` carries no separate license file. We
assume it is covered by the weeble repository's MIT license reproduced below. If the weeble authors
state other terms for their policy files, those terms apply instead.

Full license text, copied from `LICENSE` at the commit above:

```text
MIT License

Copyright (c) 2026 Alp Cetin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MuJoCo Playground (Apache-2.0): `assets/go1/walk.xml`

- Repository: <https://github.com/google-deepmind/mujoco_playground>
- Commit: `d59156e099a8785dd58fbce2e25a6ee1f7b5800c`
- Source: `mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_feetonly_flat_terrain.xml`
- Modifications made by weeble (per `assets/go1/walk.provenance.json`): included files flattened into
  one XML, resource paths relocated, training-time runtime options and gains applied; numeric tokens
  preserved. Robustness Studio does not modify the file.
- License: Apache License 2.0, full text in `assets/go1/LICENSE-Playground`.

### MuJoCo Menagerie (BSD-3-Clause): `assets/go1/assets/*.stl`

- Repository: <https://github.com/google-deepmind/mujoco_menagerie>
- Commit: `1b86ece576591213e2b666ebf59508454200ca97`
- Files: `calf.stl`, `hip.stl`, `thigh.stl`, `thigh_mirror.stl`, `trunk.stl` (Unitree Go1 meshes),
  unmodified.
- License: BSD 3-Clause License, Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD.
  ("Unitree Robotics"); full text in `assets/go1/LICENSE-Menagerie`.

## Installed dependencies (not vendored)

These packages are installed from PyPI by `uv sync` at the versions pinned in `uv.lock`. Their source
is not included in this repository.

| Package | License | Project |
| --- | --- | --- |
| `mujoco` 3.15.0 | Apache-2.0 | <https://github.com/google-deepmind/mujoco> |
| `numpy` | BSD-3-Clause | <https://numpy.org> |

Development-only tools (`pytest`, `ruff`) are installed through the `dev` dependency group and are
not distributed with the project.
