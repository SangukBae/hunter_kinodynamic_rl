"""Reconstructable package-source provenance for hierarchical artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
from typing import Dict, Iterable, Optional


_SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".xml", ".launch"}
_SOURCE_FILENAMES = {"CMakeLists.txt", "package.xml", "setup.py", "setup.cfg"}
_IGNORED_PARTS = {".git", "build", "install", "log", "runtime", "__pycache__"}


def _looks_like_source_root(path: pathlib.Path) -> bool:
    return (
        (path / "package.xml").is_file()
        and (path / "hunter_kinodynamic_rl").is_dir()
        and (path / "config").is_dir()
    )


def resolve_package_source_root(explicit: str = "") -> pathlib.Path:
    """Finds the package source checkout, never an installed Python tree."""
    if explicit:
        explicit_root = pathlib.Path(explicit).resolve()
        if not _looks_like_source_root(explicit_root):
            raise RuntimeError(
                f"explicit package_source_root {explicit_root} is not a hunter_kinodynamic_rl source root"
            )
        return explicit_root

    candidates = []
    env_path = os.environ.get("HUNTER_KINODYNAMIC_RL_SOURCE_ROOT")
    if env_path:
        candidates.append(pathlib.Path(env_path))

    cwd = pathlib.Path.cwd()
    for base in (cwd, *cwd.parents):
        candidates.extend((base, base / "src" / "hunter_kinodynamic_rl",
                           base / "ros2_ws" / "src" / "hunter_kinodynamic_rl"))

    for prefix_text in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not prefix_text:
            continue
        prefix = pathlib.Path(prefix_text).resolve()
        # <workspace>/install/<package> and <workspace>/install are both
        # common AMENT_PREFIX_PATH forms.
        for workspace in (prefix.parent.parent, prefix.parent):
            candidates.append(workspace / "src" / "hunter_kinodynamic_rl")

    seen = set()
    valid = []
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if _looks_like_source_root(candidate):
            valid.append(candidate)
    if not valid:
        raise RuntimeError(
            "could not resolve hunter_kinodynamic_rl package source root; pass package_source_root or set "
            "HUNTER_KINODYNAMIC_RL_SOURCE_ROOT instead of recording an unrelated parent Git repository"
        )
    # A nested package Git checkout is authoritative over a non-Git copy.
    valid.sort(key=lambda p: (not (p / ".git").exists(), len(p.parts)))
    return valid[0]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: pathlib.Path, *args: str, binary: bool = False):
    git_dir = root / ".git"
    if not git_dir.is_dir():
        raise RuntimeError(
            f"package source {root} does not contain a standalone .git directory; "
            "cannot record package-local provenance"
        )
    # Explicit --git-dir/--work-tree avoids repository discovery entirely.
    # That both prevents an accidental parent-repository SHA and works for
    # read-only/bind-mounted Docker sources whose host UID differs from the
    # container user (Git's safe.directory is only honored from protected
    # config scopes in the Git version shipped with ROS 2 Humble; `-c` is
    # therefore insufficient and changing the user's global config here
    # would be an unacceptable hidden side effect).
    result = subprocess.run(
        ["git", f"--git-dir={git_dir}", f"--work-tree={root}", *args],
        capture_output=True, text=not binary, timeout=10.0,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode() if binary else result.stderr
        raise RuntimeError(f"git {' '.join(args)} failed for package source {root}: {stderr.strip()}")
    return result.stdout


def _source_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in root.rglob("*"):
        if not path.is_file() or any(part in _IGNORED_PARTS for part in path.relative_to(root).parts):
            continue
        if path.suffix in _SOURCE_SUFFIXES or path.name in _SOURCE_FILENAMES:
            yield path


def _content_manifest(root: pathlib.Path, paths: Iterable[pathlib.Path]) -> Dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(paths)
    }


def collect_package_provenance(package_source_root: str = "", execution_file: Optional[str] = None) -> dict:
    root = resolve_package_source_root(package_source_root)
    git_top = pathlib.Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve()
    if git_top != root:
        raise RuntimeError(
            f"resolved Git top-level {git_top} is not the package source root {root}; refusing parent-repo SHA"
        )
    commit = _git(root, "rev-parse", "HEAD").strip()
    status = _git(root, "status", "--porcelain=v1")
    diff_bytes = _git(root, "diff", "--binary", "HEAD", binary=True)
    untracked_raw = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    untracked_paths = []
    for item in untracked_raw.split("\0"):
        path = root / item
        if not item or not path.is_file() or any(part in _IGNORED_PARTS for part in pathlib.Path(item).parts):
            continue
        if path.suffix in _SOURCE_SUFFIXES or path.name in _SOURCE_FILENAMES:
            untracked_paths.append(path)
    untracked_manifest = _content_manifest(root, untracked_paths)
    source_manifest = _content_manifest(root, _source_files(root))

    source_manifest_json = json.dumps(source_manifest, sort_keys=True, separators=(",", ":"))
    untracked_manifest_json = json.dumps(untracked_manifest, sort_keys=True, separators=(",", ":"))
    result = {
        "package_source_root": str(root),
        "package_git_toplevel": str(git_top),
        "package_git_commit_sha": commit,
        "package_git_dirty": bool(status.strip()),
        "package_git_status_porcelain": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "untracked_source_manifest_sha256": hashlib.sha256(untracked_manifest_json.encode()).hexdigest(),
        "source_content_manifest_sha256": hashlib.sha256(source_manifest_json.encode()).hexdigest(),
        "source_content_file_count": len(source_manifest),
    }
    if execution_file:
        execution_path = pathlib.Path(execution_file).resolve()
        result["execution_module_path"] = str(execution_path)
        result["execution_module_sha256"] = _sha256_file(execution_path)
        relative = pathlib.Path("hunter_kinodynamic_rl/evaluation/run_live_hierarchical_benchmark.py")
        source_module = root / relative
        if source_module.is_file():
            result["source_module_path"] = str(source_module)
            result["source_module_sha256"] = _sha256_file(source_module)
            result["execution_matches_source_module"] = (
                result["execution_module_sha256"] == result["source_module_sha256"]
            )
    return result
