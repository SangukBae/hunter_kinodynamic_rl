"""Package-local, dirty-tree provenance must never resolve the outer repository."""

import subprocess

import pytest

from hunter_kinodynamic_rl.evaluation.provenance import (
    collect_package_provenance,
    resolve_package_source_root,
)


def _run_git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def _make_nested_repo(tmp_path):
    outer = tmp_path / "outer"
    package = outer / "ros2_ws" / "src" / "hunter_kinodynamic_rl"
    module = package / "hunter_kinodynamic_rl" / "evaluation" / "run_live_hierarchical_benchmark.py"
    module.parent.mkdir(parents=True)
    (package / "config").mkdir()
    (package / "package.xml").write_text("<package/>")
    module.write_text("VALUE = 1\n")

    outer.mkdir(exist_ok=True)
    _run_git(outer, "init")
    _run_git(outer, "config", "user.email", "test@example.com")
    _run_git(outer, "config", "user.name", "Test")
    (outer / "outer.txt").write_text("outer\n")
    _run_git(outer, "add", "outer.txt")
    _run_git(outer, "commit", "-m", "outer")

    _run_git(package, "init")
    _run_git(package, "config", "user.email", "test@example.com")
    _run_git(package, "config", "user.name", "Test")
    _run_git(package, "add", ".")
    _run_git(package, "commit", "-m", "package")
    return outer, package, module


def test_collects_nested_package_git_and_dirty_source_hashes(tmp_path):
    outer, package, module = _make_nested_repo(tmp_path)
    (package / "untracked.py").write_text("DIRTY = True\n")

    result = collect_package_provenance(str(package), execution_file=str(module))

    assert result["package_source_root"] == str(package)
    assert result["package_git_toplevel"] == str(package)
    assert result["package_git_commit_sha"] == _run_git(package, "rev-parse", "HEAD")
    assert result["package_git_commit_sha"] != _run_git(outer, "rev-parse", "HEAD")
    assert result["package_git_dirty"] is True
    assert result["untracked_source_manifest_sha256"]
    assert result["source_content_manifest_sha256"]
    assert result["execution_matches_source_module"] is True


def test_explicit_non_package_root_is_not_silently_replaced(tmp_path):
    with pytest.raises(RuntimeError, match="explicit package_source_root"):
        resolve_package_source_root(str(tmp_path))
