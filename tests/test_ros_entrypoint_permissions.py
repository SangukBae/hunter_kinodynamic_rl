"""CMake-installed ``ros2 run`` programs must remain executable."""

import os
import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _installed_program_paths():
    cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text()
    match = re.search(
        r"install\(\s*PROGRAMS(?P<body>.*?)DESTINATION\s+lib/\$\{PROJECT_NAME\}",
        cmake,
        flags=re.DOTALL,
    )
    assert match is not None, "CMakeLists.txt has no install(PROGRAMS ...) block"
    return [
        PACKAGE_ROOT / line.strip()
        for line in match.group("body").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_every_ros2_run_program_exists_has_shebang_and_is_executable():
    programs = _installed_program_paths()
    assert programs
    for path in programs:
        assert path.is_file(), path
        assert path.read_bytes().startswith(b"#!"), path
        assert os.access(path, os.X_OK), f"{path} is installed as PROGRAMS but has no executable bit"
