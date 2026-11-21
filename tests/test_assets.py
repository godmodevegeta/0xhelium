"""Asset lock and provenance verification (Requirements 3.3, 3.4)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT
from robustlab.assets import LOCK_RELPATH, PROVENANCE_RELPATH, sha256_file, verify_assets
from robustlab.cli import main

SEED = 20261121


@pytest.fixture(scope="module")
def asset_copy(tmp_path_factory) -> Path:
    """A private copy of every locked file, so tests can mutate it freely."""
    root = tmp_path_factory.mktemp("assets_copy")
    lock = json.loads((REPO_ROOT / LOCK_RELPATH).read_text())
    (root / LOCK_RELPATH).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / LOCK_RELPATH, root / LOCK_RELPATH)
    for entry in lock["files"]:
        dest = root / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / entry["path"], dest)
    return root


def _locked_paths(root: Path) -> list[str]:
    return [entry["path"] for entry in json.loads((root / LOCK_RELPATH).read_text())["files"]]


def test_repository_assets_verify() -> None:
    assert verify_assets(REPO_ROOT) == []


def test_clean_copy_verifies(asset_copy: Path) -> None:
    assert verify_assets(asset_copy) == []


def test_verify_assets_cli_exit_codes(asset_copy: Path) -> None:
    ok = subprocess.run(
        [sys.executable, "-m", "robustlab.cli", "verify-assets", "--root", str(asset_copy)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    victim = asset_copy / "assets/go1/LICENSE-Playground"
    saved = victim.read_bytes()
    try:
        victim.write_bytes(saved + b"\n")
        bad = subprocess.run(
            [sys.executable, "-m", "robustlab.cli", "verify-assets", "--root", str(asset_copy)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    finally:
        victim.write_bytes(saved)
    assert bad.returncode == 1
    assert "assets/go1/LICENSE-Playground" in bad.stdout


def test_missing_lock_is_an_issue(tmp_path: Path) -> None:
    issues = verify_assets(tmp_path)
    assert [issue.path for issue in issues] == [LOCK_RELPATH]


def test_provenance_mismatch_detected_even_if_lock_agrees(asset_copy: Path, tmp_path) -> None:
    """A file re-locked to a new hash still fails against walk.provenance.json."""
    root = tmp_path / "relocked"
    shutil.copytree(asset_copy, root)
    xml = root / "assets/go1/walk.xml"
    xml.write_bytes(xml.read_bytes() + b"<!-- edit -->")
    lock = json.loads((root / LOCK_RELPATH).read_text())
    for entry in lock["files"]:
        if entry["path"] == "assets/go1/walk.xml":
            entry["sha256"] = sha256_file(xml)
    (root / LOCK_RELPATH).write_text(json.dumps(lock))
    issues = verify_assets(root)
    assert [(i.path, i.reason) for i in issues] == [
        ("assets/go1/walk.xml", "SHA-256 differs from provenance")
    ]


# Feature: m0-walking-go1, Property 1: Tampered or missing assets are detected
def test_property_tampered_or_missing_assets_detected(asset_copy: Path) -> None:
    rng = np.random.default_rng(SEED)
    paths = _locked_paths(asset_copy)
    cases = [(path, "delete", None) for path in paths]
    while len(cases) < 120:
        path = paths[int(rng.integers(len(paths)))]
        size = (asset_copy / path).stat().st_size
        cases.append((path, "flip", int(rng.integers(size))))
    # Explicit boundaries: first and last byte of every file.
    for path in paths:
        size = (asset_copy / path).stat().st_size
        cases += [(path, "flip", 0), (path, "flip", size - 1)]

    for case_index, (path, kind, offset) in enumerate(cases):
        target = asset_copy / path
        saved = target.read_bytes()
        try:
            if kind == "delete":
                target.unlink()
            else:
                mutated = bytearray(saved)
                mutated[offset] ^= 1 + int(rng.integers(255))
                target.write_bytes(bytes(mutated))
            issues = verify_assets(asset_copy)
            reported = {issue.path for issue in issues}
            exit_code = main(["verify-assets", "--root", str(asset_copy)])
        finally:
            target.write_bytes(saved)
        context = f"seed={SEED} case={case_index} path={path} kind={kind} offset={offset}"
        assert path in reported, context
        assert exit_code == 1, context
        if path != PROVENANCE_RELPATH:
            # Only the mutated file is reported (the provenance file vouches for the others).
            assert reported == {path}, context
    assert verify_assets(asset_copy) == []
