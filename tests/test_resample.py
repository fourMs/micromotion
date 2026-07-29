"""Rate measurement and the downsample-only rule."""

import numpy as np
import pytest

import micromotion as mm
from micromotion import resample


def test_measured_rate_survives_millisecond_rounding():
    """The failure that made a 256 Hz recording look like 250 Hz.

    Rounding the timestamps to whole milliseconds turns the intervals into a 3/4/5 ms
    mixture whose median is exactly 4 ms, implying 250 Hz. Counting samples over the span
    recovers the truth.
    """
    fs_true = 256.0
    t = np.arange(0, 9001, 1 / fs_true)
    t_rounded = np.round(t, 3)
    assert 1 / np.median(np.diff(t_rounded)) == pytest.approx(250.0, abs=0.5)
    assert resample.measured_rate(t_rounded) == pytest.approx(fs_true, rel=1e-6)


def test_upsampling_is_refused():
    x = np.random.default_rng(0).normal(size=2000)
    with pytest.raises(ValueError, match="refusing to upsample"):
        resample.to_rate(x, 20.0, 25.0)


def test_downsampling_preserves_in_band_amplitude():
    fs = 200.0
    t = np.arange(0, 120, 1 / fs)
    x = 3.0 * np.sin(2 * np.pi * 2.0 * t)
    y = resample.to_rate(x, fs, 20.0)
    assert len(y) == pytest.approx(len(x) * 20 / fs, rel=0.01)
    assert np.std(y[50:-50]) == pytest.approx(np.std(x), rel=0.02)


def test_qom_is_stable_across_the_downsample_to_common_rate():
    """What makes a harmonised cross-collection table meaningful."""
    fs = 200.0
    t = np.arange(0, 180, 1 / fs)
    rng = np.random.default_rng(1)
    x = np.column_stack(
        [
            2.0 * np.sin(2 * np.pi * 0.8 * t) + 0.3 * rng.normal(size=len(t))
            for _ in range(3)
        ]
    )
    native = mm.qom(x, fs, kind="position")
    y = resample.to_rate(x, fs, mm.COMMON_RATE)
    common = mm.qom(y, mm.COMMON_RATE, kind="position")
    trim_n, trim_c = native.edge_samples, common.edge_samples
    assert np.mean(native.speed[trim_n:-trim_n]) == pytest.approx(
        np.mean(common.speed[trim_c:-trim_c]), rel=0.05
    )


def test_regularize_cleans_duplicate_and_backward_timestamps():
    t = np.array([0.0, 0.1, 0.1, 0.05, 0.2, 0.3])
    x = np.arange(6, dtype=float)
    grid, y = resample.regularize(t, x, fs_out=20.0)
    assert np.all(np.diff(grid) > 0)
    assert np.isfinite(y).all()


def test_regularize_leaves_gaps_as_nan():
    """A 132-second hole must not become 132 seconds of apparent stillness."""
    t = np.concatenate([np.arange(0, 30, 0.02), np.arange(162, 192, 0.02)])
    x = np.ones_like(t)
    grid, y = resample.regularize(t, x, fs_out=20.0, max_gap_s=1.0)
    inside = (grid > 31) & (grid < 160)
    assert np.isnan(y[inside]).all()
    assert np.isfinite(y[grid < 29]).all()


def test_rate_quality_reports_the_gap():
    t = np.concatenate([np.arange(0, 30, 0.02), np.arange(162, 192, 0.02)])
    q = resample.rate_quality(t)
    assert q["max_gap_s"] == pytest.approx(132.0, abs=0.1)
    assert q["coverage"] < 0.5
