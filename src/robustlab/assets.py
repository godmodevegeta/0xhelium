"""Verification of pinned upstream files against the asset lock and weeble's provenance record.

Stdlib only, so `robustlab verify-assets` stays fast and has no scientific imports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

LOCK_RELPATH = "assets/assets.lock.json"
PROVENANCE_RELPATH = "assets/go1/walk.provenance.json"
PROVENANCE_PREFIX = "assets/go1/"  # keys in provenance "files" are relative to this directory
LOCK_SCHEMA_VERSION = 1
_CHUNK = 1 << 20


@dataclass(frozen=True)
class AssetIssue:
    """One verification failure. `path` is repository-relative."""

    path: str
    reason: str
    expected: str | None = None
    actual: str | None = None

    def __str__(self) -> str:
        detail = ""
        if self.expected is not None or self.actual is not None:
            detail = f" (expected {self.expected}, actual {self.actual})"
        return f"{self.path}: {self.reason}{detail}"


class AssetError(Exception):
    """Raised when the asset lock itself cannot be read or has an unexpected shape."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(root: Path) -> dict:
    """Read and shape-check `assets/assets.lock.json`. Raises `AssetError`."""
    path = Path(root) / LOCK_RELPATH
    try:
        lock = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AssetError(f"{LOCK_RELPATH}: missing") from exc
    except json.JSONDecodeError as exc:
        raise AssetError(f"{LOCK_RELPATH}: invalid JSON ({exc})") from exc
    if not isinstance(lock, dict) or lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise AssetError(f"{LOCK_RELPATH}: unsupported schema_version")
    files = lock.get("files")
    commit = lock.get("source", {}).get("commit") if isinstance(lock.get("source"), dict) else None
    if not isinstance(commit, str) or len(commit) != 40:
        raise AssetError(f"{LOCK_RELPATH}: source.commit must be a 40-hex SHA")
    if not isinstance(files, list) or not files:
        raise AssetError(f"{LOCK_RELPATH}: files must be a non-empty list")
    for entry in files:
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and isinstance(entry.get("sha256"), str)
            and len(entry["sha256"]) == 64
        ):
            raise AssetError(f"{LOCK_RELPATH}: malformed file entry {entry!r}")
    return lock


def locked_sha256(root: Path, relpath: str) -> str:
    """Return the lock's SHA-256 for one repository-relative path. Raises `AssetError`."""
    for entry in load_lock(root)["files"]:
        if entry["path"] == relpath:
            return entry["sha256"]
    raise AssetError(f"{relpath}: not listed in {LOCK_RELPATH}")


def verify_assets(root: Path) -> list[AssetIssue]:
    """Check every lock entry and every provenance entry. An empty list means all files pass."""
    root = Path(root)
    try:
        lock = load_lock(root)
    except AssetError as exc:
        return [AssetIssue(LOCK_RELPATH, str(exc).split(": ", 1)[-1])]

    issues: list[AssetIssue] = []
    actual: dict[str, str | None] = {}
    locked: dict[str, str] = {}
    for entry in lock["files"]:
        relpath = entry["path"]
        locked[relpath] = entry["sha256"]
        path = root / relpath
        if not path.is_file():
            actual[relpath] = None
            issues.append(AssetIssue(relpath, "missing"))
            continue
        actual[relpath] = sha256_file(path)
        if actual[relpath] != entry["sha256"]:
            issues.append(
                AssetIssue(relpath, "SHA-256 differs from lock", entry["sha256"], actual[relpath])
            )

    # The provenance file is itself locked; only trust its hash list if it matched the lock.
    if PROVENANCE_RELPATH not in locked:
        issues.append(AssetIssue(PROVENANCE_RELPATH, f"not listed in {LOCK_RELPATH}"))
        return issues
    if actual.get(PROVENANCE_RELPATH) != locked[PROVENANCE_RELPATH]:
        return issues
    try:
        provenance_files = json.loads((root / PROVENANCE_RELPATH).read_text())["files"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        issues.append(AssetIssue(PROVENANCE_RELPATH, f"unreadable provenance ({exc})"))
        return issues

    for name, expected in sorted(provenance_files.items()):
        relpath = PROVENANCE_PREFIX + name
        if relpath not in locked:
            issues.append(AssetIssue(relpath, f"listed in provenance but not in {LOCK_RELPATH}"))
            continue
        if actual[relpath] is not None and actual[relpath] != expected:
            issues.append(
                AssetIssue(relpath, "SHA-256 differs from provenance", expected, actual[relpath])
            )
    return issues


def verified_paths(root: Path) -> list[str]:
    """Repository-relative paths listed in the lock, in lock order (for CLI reporting)."""
    return [entry["path"] for entry in load_lock(root)["files"]]
