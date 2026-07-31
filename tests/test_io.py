"""Reader behaviour that has cost this corpus real errors."""

import numpy as np
import pandas as pd
import pytest

import micromotion as mm

def test_read_phone_asymmetric_trim(tmp_path):
    """trim_start_s / trim_end_s override trim_clap_s per end.

    The symmetric-only version forced a choice between keeping an opening synchronisation clap
    and discarding good data at the close. On StillStanding365 day 221 the clap reaches
    10.29 m/s2 against a body maximum of 0.155, so leaving it in is not an option and losing
    35 s of standstill to remove it was the only alternative.
    """
    fs, dur = 100.0, 100.0
    t = np.arange(0, dur, 1 / fs)
    a = np.full((len(t), 3), 0.01)
    a[int(2 * fs)] = 10.0                      # the clap, at t = 2 s
    df = pd.DataFrame({"time": t, "ax": a[:, 0], "ay": a[:, 1], "az": a[:, 2]})
    p = tmp_path / "clap.tsv"
    df.to_csv(p, sep="\t", index=False)

    full = mm.read_phone(str(p))
    assert np.nanmax(full.data) > 9                      # clap present

    sym = mm.read_phone(str(p), trim_clap_s=35)
    asym = mm.read_phone(str(p), trim_start_s=35, trim_end_s=0)
    assert np.nanmax(sym.data) < 1 and np.nanmax(asym.data) < 1     # clap gone from both
    assert len(asym.data) > len(sym.data)                            # but the tail is kept
    assert asym.meta["trimmed_start_s"] == 35 and asym.meta["trimmed_end_s"] == 0

    with pytest.raises(ValueError, match="leaves nothing"):
        mm.read_phone(str(p), trim_start_s=60, trim_end_s=60)


def test_no_edition_is_y_up_any_more():
    """Standstill2019 was rotated to Z-up at source, so the compensation must be off.

    A reader that still claimed Y for 2019 would hand every caller a horizontal axis as the
    vertical one, silently — the mirror image of the bug the rotation fixed.
    """
    from micromotion.io import Y_UP_COLLECTIONS
    assert Y_UP_COLLECTIONS == ()


def test_version_matches_pyproject():
    """__version__ and pyproject.toml must agree.

    They did not at 0.9.0: the release bumped pyproject and left the hardcoded string in
    __init__.py at 0.8.3, so an installed copy reported the previous version. A figure or a
    report that records "produced with micromotion X" would then record the wrong X.
    """
    import pathlib
    import tomllib
    import micromotion as mm

    root = pathlib.Path(mm.__file__).resolve().parents[2]
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():          # installed without the source tree
        return
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert mm.__version__ == declared, f"{mm.__version__} != {declared}"
