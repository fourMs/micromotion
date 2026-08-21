"""The canonical feature vector, against constructed answers."""

import numpy as np
import pytest

from micromotion import FEATURE_NAMES, feature_vector


def sway(fs=100.0, dur=300.0, amp=5.0, f=0.4, seed=0, vertical=0.5):
    """A body swaying mostly along one horizontal axis, in millimetres."""
    t = np.arange(0, dur, 1 / fs)
    rng = np.random.default_rng(seed)
    x = amp * np.sin(2 * np.pi * f * t) + rng.normal(0, 0.2, t.size)
    y = 0.25 * amp * np.sin(2 * np.pi * f * t + 1.1) + rng.normal(0, 0.2, t.size)
    z = vertical * rng.normal(0, 1, t.size)
    return np.column_stack([x, y, z])


def test_returns_every_named_descriptor():
    got = feature_vector(sway(), 100.0, kind="position", unit="mm")
    assert set(got) == set(FEATURE_NAMES)


def test_a_recording_shorter_than_two_minutes_is_refused():
    assert feature_vector(sway(dur=60.0), 100.0, kind="position", unit="mm") is None


def test_geometry_is_nan_for_acceleration_because_it_needs_position():
    got = feature_vector(sway(), 100.0, kind="acceleration", unit="m/s^2")
    for k in ("path", "extent", "area", "anis", "vert"):
        assert np.isnan(got[k]), k
    for k in ("qom", "centroid", "f50", "frozen", "burst"):
        assert np.isfinite(got[k]), k


def test_an_unknown_kind_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        feature_vector(sway(), 100.0, kind="velocity", unit="mm")


def test_a_wrong_shape_raises():
    with pytest.raises(ValueError):
        feature_vector(np.zeros((5000, 2)), 100.0, kind="position", unit="mm")


def test_more_movement_gives_more_qom_and_a_larger_sway_area():
    small = feature_vector(sway(amp=2.0), 100.0, kind="position", unit="mm")
    large = feature_vector(sway(amp=8.0), 100.0, kind="position", unit="mm")
    assert large["qom"] > small["qom"]
    assert large["area"] > small["area"]
    assert large["extent"] > small["extent"]


def test_anisotropy_is_large_for_a_line_and_near_one_for_a_circle():
    fs, t = 100.0, np.arange(0, 300, 0.01)
    rng = np.random.default_rng(1)
    line = np.column_stack([5 * np.sin(2 * np.pi * 0.4 * t), rng.normal(0, 0.05, t.size),
                            rng.normal(0, 0.05, t.size)])
    circle = np.column_stack([5 * np.sin(2 * np.pi * 0.4 * t), 5 * np.cos(2 * np.pi * 0.4 * t),
                              rng.normal(0, 0.05, t.size)])
    assert feature_vector(line, fs, kind="position", unit="mm")["anis"] > 5 * feature_vector(circle, fs, kind="position", unit="mm")["anis"]


def test_jerk_is_nan_when_the_sensor_cannot_carry_wideband():
    """The rate that matters is the device's, not the grid the file is stored on."""
    x = sway(fs=100.0)
    on_grid = feature_vector(x, 100.0, kind="position", unit="mm")                      # grid says 100 Hz
    truth = feature_vector(x, 100.0, kind="position", unit="mm", sensor_fs=15.0)        # sensor actually ran at 15
    assert np.isfinite(on_grid["jerk"])
    assert np.isnan(truth["jerk"])


def test_the_caller_s_array_is_not_modified():
    x = sway()
    x[10, 0] = np.nan                      # a gap, which the function fills internally
    before = x.copy()
    feature_vector(x, 100.0, kind="position", unit="mm")
    assert np.array_equal(x, before, equal_nan=True)


def test_kind_and_unit_are_required():
    """Omitting them used to default to position/mm and silently mis-scale accelerometer data."""
    with pytest.raises(TypeError, match="requires both kind and unit"):
        feature_vector(sway(), 100.0)
    with pytest.raises(TypeError, match="requires both kind and unit"):
        feature_vector(sway(), 100.0, kind="position")
