"""The package version has one source, and the build reads it from there.

Two static copies of a version number drift the moment a bump touches one
of them. Across these toolboxes that has already happened: ambiscape
shipped three releases reporting a version other than their own, and
musiscape shipped one. micromotion's two copies happened to agree, which
is luck rather than a mechanism.

Anything citing a toolbox by version -- a report, a deposit, a methods
section -- is otherwise citing a number the installed package will not
confirm, which makes the drift a correctness problem rather than a
cosmetic one.

The fix is that ``src/micromotion/__init__.py`` holds the number and
hatchling reads it from there. These tests keep it that way.

The checks read ``pyproject.toml`` with a small section scanner rather
than a TOML parser, because ``tomllib`` is standard only from Python 3.11
and this package supports 3.10. What is needed here is whether a key is
present in a section, which does not warrant a dependency.
"""
import re
from pathlib import Path

import pytest

import micromotion

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Running against an installed wheel rather than a checkout: there is no
# pyproject.toml to inspect and nothing here applies.
pytestmark = pytest.mark.skipif(not PYPROJECT.exists(),
                                reason="no pyproject.toml (installed package)")


def _section(name: str) -> list[str]:
    """Non-comment, non-blank lines of one top-level ``[section]``."""
    out, inside = [], False
    for raw in PYPROJECT.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            inside = line == f"[{name}]"
            continue
        if inside and line and not line.startswith("#"):
            out.append(line)
    return out


def _value(section: str, key: str) -> str | None:
    for line in _section(section):
        m = re.match(rf"{re.escape(key)}\s*=\s*(.+)$", line)
        if m:
            return m.group(1).strip()
    return None


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[.-]?\w+)?", micromotion.__version__), \
        f"__version__ is not a version string: {micromotion.__version__!r}"


def test_pyproject_declares_no_second_version():
    """A static version in pyproject.toml is the second copy that drifts."""
    assert _value("project", "version") is None, (
        "pyproject.toml carries its own version again. It must stay dynamic, "
        "or the two numbers will drift as they did in the sibling toolboxes."
    )
    dynamic = _value("project", "dynamic") or ""
    assert "version" in dynamic, \
        "pyproject.toml should declare version in [project].dynamic"


def test_build_reads_the_module():
    """hatchling resolves the version from the module, not from a literal."""
    path = _value("tool.hatch.version", "path") or ""
    assert "micromotion/__init__.py" in path, (
        f"the build resolves the version from {path!r}; it should read "
        "src/micromotion/__init__.py so there is exactly one place to edit"
    )


def test_hatchling_resolves_the_declared_version():
    """The number the build would package equals the one the module reports.

    hatchling is a build-time dependency and is absent from a plain runtime
    environment, so this cross-check skips rather than fails where it is
    not installed. The three checks above need no imports and carry the
    guard on their own.
    """
    try:
        from hatchling.metadata.core import ProjectMetadata
    except ImportError:
        pytest.skip("hatchling is a build-time dependency and is not "
                    "installed in this environment")

    root = str(PYPROJECT.parent)
    resolved = ProjectMetadata(root, None).version
    assert resolved == micromotion.__version__, (
        f"build would package {resolved!r} while the module reports "
        f"{micromotion.__version__!r}"
    )


def test_citation_file_matches_the_module():
    """CITATION.cff is the third place the version is written.

    It cannot be made dynamic: GitHub's "Cite this repository" and Zenodo
    read the file as it stands, so the number has to be literal there. It
    is therefore the one copy a guard still has to compare, and it is the
    copy that matters most for attribution -- it once drifted to 0.7.0
    while the code reached 0.12.3, five minor versions unnoticed, because
    only the other two were checked. A stale version here does not
    misreport a figure; it misattributes the software.

    Regex rather than a YAML parser: pyyaml is not a dependency and this
    is one scalar on one line.
    """
    citation = PYPROJECT.parent / "CITATION.cff"
    if not citation.exists():
        pytest.skip("no CITATION.cff")
    m = re.search(r'(?m)^version:\s*"?([^"\s]+)"?\s*$',
                  citation.read_text(encoding="utf-8"))
    assert m, "no version field in CITATION.cff"
    assert m.group(1) == micromotion.__version__, (
        f"CITATION.cff says {m.group(1)!r} while the module reports "
        f"{micromotion.__version__!r}"
    )
