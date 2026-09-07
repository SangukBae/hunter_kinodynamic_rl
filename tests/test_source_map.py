"""Authoritative check for docs/IMPLEMENTATION_PLAN.md's "verbatim copy" claims
(code review: that table claimed ``common/geometry.py`` was a byte-identical
copy of ``drl_agent/common/geometry_utils.py`` when it had actually gained
an added function, ``to_world_frame()``, and the table's own hash for two
OTHER rows -- ``pure_pursuit.py``/``tqc.py`` -- had a stale/mistyped suffix
even though the files themselves were still genuinely identical). Hand-typed
hashes in a doc are exactly the kind of claim that silently rots; this test
hashes both sides directly, on every run, so it is the source of truth --
IMPLEMENTATION_PLAN.md's table is a human-readable summary of these assertions.

`drl_agent` is explicitly NOT a package.xml dependency of this package (see
IMPLEMENTATION_PLAN.md's header) -- it is a sibling ROS package in the same
``ros2_ws/src/`` checkout, read as a reference only. This test locates it by
relative PATH (never ``import drl_agent...``, which is unreliable here: a
namespace-package stub with no submodules can shadow the real one on
``sys.path`` -- confirmed live) and skips cleanly if that sibling checkout
isn't present, matching this package's established self-skip convention for
optional environment dependencies (torch/rclpy/drl_agent_interfaces)."""

import hashlib
import importlib.util
import inspect
from pathlib import Path

import pytest

_HKRL_ROOT = Path(__file__).resolve().parents[1]  # .../ros2_ws/src/hunter_kinodynamic_rl
_SRC_ROOT = _HKRL_ROOT.parent  # .../ros2_ws/src
_DRL_AGENT_ROOT = _SRC_ROOT / "drl_agent" / "drl_agent"

if not _DRL_AGENT_ROOT.is_dir():
    pytest.skip("drl_agent sibling package not present in this checkout -- IMPLEMENTATION_PLAN.md "
                "verbatim-copy claims cannot be verified without it", allow_module_level=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# (new file relative to hunter_kinodynamic_rl/, source file relative to
# drl_agent/drl_agent/) -- exactly the "Verbatim copies" table in
# docs/IMPLEMENTATION_PLAN.md, geometry.py deliberately excluded (see below).
VERBATIM_COPY_PAIRS = [
    ("hunter_kinodynamic_rl/common/seed.py", "common/seed_utils.py"),
    ("hunter_kinodynamic_rl/trajectory/pure_pursuit.py", "common/pure_pursuit.py"),
    ("hunter_kinodynamic_rl/env/simulation/gazebo_service_wait.py", "env/simulation/gazebo_service_wait.py"),
]


@pytest.mark.parametrize("new_rel,source_rel", VERBATIM_COPY_PAIRS)
def test_verbatim_copy_pairs_are_still_byte_identical(new_rel, source_rel):
    new_path = _HKRL_ROOT / new_rel
    source_path = _DRL_AGENT_ROOT / source_rel
    assert new_path.is_file(), f"IMPLEMENTATION_PLAN.md references a missing file: {new_path}"
    assert source_path.is_file(), f"IMPLEMENTATION_PLAN.md references a missing drl_agent file: {source_path}"
    assert _sha256(new_path) == _sha256(source_path), (
        f"{new_rel} has drifted from drl_agent's {source_rel} -- IMPLEMENTATION_PLAN.md's 'Verbatim copies' "
        "table claims byte-identical; either re-sync or move this row to the 'Adapted' table with an "
        "honest description of what changed and why."
    )


def test_tqc_copy_has_only_the_documentation_link_adaptation():
    """The consolidated docs moved the referenced compatibility contract.

    Network code must otherwise remain byte-identical to the source TQC.
    """
    ours = (_HKRL_ROOT / "hunter_kinodynamic_rl/rl/networks/tqc.py").read_text()
    source = (_DRL_AGENT_ROOT / "rl/networks/tqc.py").read_text()
    restored_reference = ours.replace(
        "docs/CHECKPOINT_AND_COMPATIBILITY.md",
        "docs/experiments/tqc_scaling_improvement_plan.md",
    )
    assert restored_reference == source


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Every function geometry_utils.py defines -- these must stay byte-identical
# (as IMPLEMENTATION_PLAN.md's "Adapted" row now claims) even though geometry.py as a
# WHOLE FILE is no longer verbatim (it gained to_world_frame()).
_GEOMETRY_SHARED_FUNCTIONS = (
    "wrap_to_pi", "angle_to", "heading_error", "euclidean_distance",
    "goal_distance_and_heading", "to_robot_frame",
)


def test_geometry_py_is_no_longer_claimed_verbatim_but_shared_functions_are_byte_identical():
    """The actual code-review finding: common/geometry.py != geometry_utils.py
    as whole files (to_world_frame() was added), but every function
    geometry_utils.py itself defines must still be untouched -- this is the
    finer-grained claim IMPLEMENTATION_PLAN.md's 'Adapted' table row now makes, and
    this test is what verifies it instead of a hand-typed whole-file hash."""
    ours = _load_module(_HKRL_ROOT / "hunter_kinodynamic_rl" / "common" / "geometry.py", "_hkrl_geometry")
    theirs = _load_module(_DRL_AGENT_ROOT / "common" / "geometry_utils.py", "_drl_agent_geometry_utils")

    whole_file_identical = _sha256(_HKRL_ROOT / "hunter_kinodynamic_rl" / "common" / "geometry.py") == _sha256(
        _DRL_AGENT_ROOT / "common" / "geometry_utils.py"
    )
    assert not whole_file_identical, (
        "geometry.py is now byte-identical to geometry_utils.py again -- if to_world_frame() was "
        "removed or geometry_utils.py gained an equivalent, update IMPLEMENTATION_PLAN.md to move this row "
        "back into the 'Verbatim copies' table (with a fresh hash) instead of leaving it in 'Adapted'."
    )

    for fn_name in _GEOMETRY_SHARED_FUNCTIONS:
        assert hasattr(theirs, fn_name), (
            f"drl_agent's geometry_utils.py no longer defines {fn_name!r} -- "
            "_GEOMETRY_SHARED_FUNCTIONS/IMPLEMENTATION_PLAN.md need updating"
        )
        our_src = inspect.getsource(getattr(ours, fn_name))
        their_src = inspect.getsource(getattr(theirs, fn_name))
        assert our_src == their_src, (
            f"geometry.py's {fn_name}() has diverged from geometry_utils.py's -- IMPLEMENTATION_PLAN.md's "
            "'Adapted' row claims every SHARED function stays byte-identical; only to_world_frame() "
            "is new. If this divergence is intentional, update that row's description."
        )

    assert hasattr(ours, "to_world_frame"), "to_world_frame() is the one documented addition -- it's missing"
    assert not hasattr(theirs, "to_world_frame"), (
        "drl_agent's geometry_utils.py now ALSO defines to_world_frame() -- IMPLEMENTATION_PLAN.md's claim that "
        "this is a hunter_kinodynamic_rl-only addition needs updating (or the two should be re-merged "
        "into a verbatim copy again)."
    )
