"""The runtime package never depends on test code or the upstream oracle (Requirement 8)."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
ORACLE = REPO_ROOT / "tests" / "reference" / "weeble_sim.py"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_src_does_not_import_tests() -> None:
    sources = sorted(SRC.rglob("*.py"))
    assert sources, "no Python files found under src/"
    offenders = {}
    for path in sources:
        bad = {
            name
            for name in _imported_modules(path)
            if name.split(".", 1)[0] in {"tests", "reference", "weeble_sim", "conftest"}
        }
        if bad:
            offenders[str(path.relative_to(REPO_ROOT))] = sorted(bad)
    assert offenders == {}


def test_oracle_carries_mit_notice_and_source_commit() -> None:
    header = ORACLE.read_text().split("# ---- upstream file begins ----\n", 1)[0]
    assert "60878341072256ccc68f6118ec23a298c55e1bb9" in header
    assert "Copyright (c) 2026 Alp Cetin" in header
    assert "Permission is hereby granted, free of charge" in header
    assert "copies or substantial portions of the Software." in header
