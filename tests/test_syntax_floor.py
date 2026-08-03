"""Every source file must compile on the oldest Python this package supports.

Written 2026-08-03, after 0.15.0 failed to publish. A backslash inside an f-string expression is
legal from 3.12 (PEP 701) and a SyntaxError before it. The local interpreter is 3.12, so the whole
suite passed here and CI rejected the release on 3.10 and 3.11 twenty minutes later.

The first version of this test used `ast.parse(..., feature_version=(3, 10))` and it was useless:
`feature_version` gates a few grammar features but does not revert f-string tokenization, so it
parsed the broken file happily. Verified by running it against the offending commit — it passed.
A test that cannot fail is worse than no test, because it reads as coverage.

So this shells out to a real older interpreter and compiles the package with it. If none is
installed the test skips loudly rather than pretending: CI still has the matrix, and this exists to
make that feedback arrive in a second rather than after a failed release.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "micromotion"


def minimum_version() -> tuple[int, int]:
    """The floor declared in pyproject, so this cannot drift from the packaging metadata."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'requires-python\s*=\s*"[><=~^ ]*(\d+)\.(\d+)', text)
    assert m, "could not read requires-python from pyproject.toml"
    return int(m.group(1)), int(m.group(2))


def oldest_available() -> tuple[str, str] | None:
    """The oldest installed interpreter at or above the declared floor, if any."""
    major, minor = minimum_version()
    for m in range(minor, minor + 3):
        exe = shutil.which(f"python{major}.{m}")
        if exe:
            return exe, f"{major}.{m}"
    return None


def test_compiles_on_the_oldest_supported_python() -> None:
    major, minor = minimum_version()
    found = oldest_available()
    if found is None:
        pytest.skip(f"no python{major}.{minor}+ interpreter older than the current one is "
                    f"installed; CI's matrix is the only check")
    exe, label = found
    files = sorted(str(p) for p in SRC.rglob("*.py"))
    proc = subprocess.run([exe, "-m", "py_compile", *files],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, (
        f"the package does not compile on Python {label}, which is within the range "
        f"pyproject declares (>= {major}.{minor}). It may still run on the interpreter running "
        f"these tests, which is exactly how this reached a release:\n\n{proc.stderr}")
