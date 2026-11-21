"""Fetch the pinned weeble Go1 assets and policy, verify them, and write the asset lock.

Stdlib only. Every file is first staged in a temporary directory; nothing under the output root is
touched until all files are downloaded and their SHA-256 hashes match the hashes listed in the
downloaded ``walk.provenance.json``. The lock (``assets/assets.lock.json``) is written last via a
temporary file and ``os.replace``.

Usage:
    python scripts/fetch_assets.py [--out ROOT] [--from-clone PATH]

``--from-clone`` copies the files from a local git clone checked out at the pinned commit instead of
downloading them. The lock still records the canonical raw URLs, and hashes are verified the same way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO = "mottopanikeiku/weeble"
COMMIT = "60878341072256ccc68f6118ec23a298c55e1bb9"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"
LOCK_PATH = "assets/assets.lock.json"
LOCK_SCHEMA_VERSION = 1

# Paths are identical in weeble and in this repository.
FILES = (
    "assets/go1/walk.xml",
    "assets/go1/assets/calf.stl",
    "assets/go1/assets/hip.stl",
    "assets/go1/assets/thigh.stl",
    "assets/go1/assets/thigh_mirror.stl",
    "assets/go1/assets/trunk.stl",
    "assets/go1/LICENSE-Menagerie",
    "assets/go1/LICENSE-Playground",
    "assets/go1/walk.provenance.json",
    "policies/go1-walk-s0.json",
)
PROVENANCE = "assets/go1/walk.provenance.json"
PROVENANCE_PREFIX = "assets/go1/"  # keys in provenance "files" are relative to this directory

TIMEOUT_S = 60.0
MAX_BYTES = 64 * 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parent.parent


class FetchError(Exception):
    """Raised when any file cannot be obtained or verified; nothing has been installed yet."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
            if response.status != 200:
                raise FetchError(f"{url}: HTTP {response.status}")
            data = response.read(MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FetchError(f"{url}: {exc}") from exc
    if len(data) > MAX_BYTES:
        raise FetchError(f"{url}: larger than {MAX_BYTES} bytes")
    return data


def check_clone(clone: Path) -> None:
    """Require the clone to be at the pinned commit with the pinned files unmodified."""
    try:
        head = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(clone), "status", "--porcelain", "--", *FILES],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FetchError(f"{clone}: not a readable git clone ({exc})") from exc
    if head != COMMIT:
        raise FetchError(f"{clone}: HEAD is {head}, expected {COMMIT}")
    if dirty:
        raise FetchError(f"{clone}: pinned files are modified:\n{dirty}")


def stage(staging: Path, clone: Path | None) -> None:
    """Obtain every pinned file into ``staging`` (same relative paths)."""
    for rel in FILES:
        if clone is None:
            data = download(RAW_BASE + rel)
        else:
            src = clone / rel
            if not src.is_file():
                raise FetchError(f"{src}: missing in clone")
            data = src.read_bytes()
        target = staging / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def verify_against_provenance(staging: Path) -> None:
    """Check staged files against the hashes in the staged ``walk.provenance.json``."""
    try:
        provenance = json.loads((staging / PROVENANCE).read_text(encoding="utf-8"))
        expected = provenance["files"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FetchError(f"{PROVENANCE}: cannot read 'files' section ({exc})") from exc
    if not isinstance(expected, dict):
        raise FetchError(f"{PROVENANCE}: 'files' is not an object")

    errors = []
    for key, want in expected.items():
        rel = PROVENANCE_PREFIX + key
        if rel not in FILES:
            continue  # listed upstream but not pinned by us
        got = sha256_bytes((staging / rel).read_bytes())
        if got != want:
            errors.append(f"{rel}: sha256 {got} != provenance {want}")
    if errors:
        raise FetchError("provenance hash mismatch:\n" + "\n".join(errors))


def build_lock(staging: Path) -> dict:
    files = []
    for rel in FILES:
        data = (staging / rel).read_bytes()
        files.append(
            {
                "path": rel,
                "url": RAW_BASE + rel,
                "sha256": sha256_bytes(data),
                "size": len(data),
            }
        )
    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "source": {"repo": REPO, "commit": COMMIT},
        "files": files,
    }


def atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def install(staging: Path, out_root: Path, lock: dict) -> None:
    for rel in FILES:
        atomic_write_bytes(out_root / rel, (staging / rel).read_bytes())
    lock_text = json.dumps(lock, indent=2) + "\n"
    atomic_write_bytes(out_root / LOCK_PATH, lock_text.encode("utf-8"))


def fetch(out_root: Path, clone: Path | None = None) -> dict:
    """Stage, verify, then install all pinned files under ``out_root``; return the lock dict."""
    if clone is not None:
        check_clone(clone)
    with tempfile.TemporaryDirectory(prefix="robustlab-fetch-") as tmp:
        staging = Path(tmp)
        stage(staging, clone)
        verify_against_provenance(staging)
        lock = build_lock(staging)
        install(staging, out_root, lock)
    return lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT, help="output root (default: repository root)"
    )
    parser.add_argument(
        "--from-clone",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"copy from a local weeble clone at {COMMIT} instead of downloading",
    )
    args = parser.parse_args(argv)

    try:
        lock = fetch(args.out.resolve(), args.from_clone)
    except FetchError as exc:
        print(f"fetch_assets: aborted, nothing installed: {exc}", file=sys.stderr)
        return 1

    print(f"weeble {COMMIT} -> {args.out.resolve()}")
    for entry in lock["files"]:
        print(f"{entry['sha256']}  {entry['size']:>9}  {entry['path']}")
    print(f"wrote {LOCK_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
