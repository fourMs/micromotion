"""The synthetic recordings the documentation runs on.

These exist so that a reader with no data can run the quickstart, which means the
numbers printed in the quickstart are part of the package's contract: if they move, the
documentation is wrong. The values here are the ones `docs/quickstart.md` shows and the
ones `docs/img/one-measure.png` is drawn from.
"""
import numpy as np
import pytest

import micromotion as mm


def test_standstill_record_gives_the_median_the_quickstart_prints():
    rec = mm.examples.standstill_record()
    r = mm.qom(rec.data, rec.fs, kind=rec.kind, unit=rec.unit)
    assert round(r.median_mm_s, 2) == 2.29


def test_worn_acceleration_agrees_with_the_optical_route():
    """The package's headline claim, as a test rather than only as a figure.

    One motion, two sensor families, one band: the medians must agree closely. They
    are derived from the same trajectory, so this bounds the PIPELINE's disagreement
    and says nothing about two real instruments.
    """
    fs = 100.0
    optical = mm.qom(mm.examples.standstill(fs=fs), fs, kind="position", unit="mm")
    worn = mm.qom(mm.examples.worn_acceleration(fs=fs), fs,
                  kind="acceleration", unit="m/s^2")
    ratio = worn.median_mm_s / optical.median_mm_s
    assert 0.95 < ratio < 1.05, f"routes disagree by {ratio:.3f}x"


def test_halving_the_rate_barely_moves_the_answer():
    fs = 100.0
    xyz = mm.examples.standstill(fs=fs)
    full = mm.qom(xyz, fs, kind="position", unit="mm").median_mm_s
    half = mm.qom(xyz[::2], fs / 2, kind="position", unit="mm").median_mm_s
    assert half == pytest.approx(full, rel=0.05)


def test_deterministic():
    """Two people comparing notes must see the same numbers."""
    assert np.array_equal(mm.examples.standstill(), mm.examples.standstill())


def test_it_looks_like_a_head_marker_and_carries_no_sentinels():
    rec = mm.examples.standstill_record()
    assert rec.data.shape == (36000, 3)
    assert rec.unit == "mm" and rec.kind == "position" and rec.vertical == "Z"
    assert np.isfinite(rec.data).all()
    assert 1600 < rec.data[:, 2].mean() < 1800      # a head stands about 1.7 m up
    # Gaps are NaN in this package, never a sentinel, so a synthetic record must not
    # contain the zero triplet that `validate` exists to catch.
    assert not (rec.data == 0.0).all(axis=1).any()
