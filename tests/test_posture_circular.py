"""Postural geometry and circular statistics, against constructed answers."""

import numpy as np
import pytest

from micromotion import circular as ci
from micromotion import posture as po
from micromotion import resample


# ------------------------------------------------------------------ sway geometry

def test_isotropic_cloud_has_no_anisotropy():
    rng = np.random.default_rng(0)
    g = po.sway_geometry(rng.normal(size=(4000, 2)))
    assert g["anisotropy"] < 0.15


def test_a_line_is_maximally_anisotropic_and_its_axis_is_recovered():
    t = np.linspace(-1, 1, 2000)
    for deg in (0.0, 30.0, 120.0):
        a = np.radians(deg)
        xy = np.column_stack([t * np.cos(a), t * np.sin(a)])
        g = po.sway_geometry(xy)
        assert g["anisotropy"] > 0.99
        assert g["axis_deg"] == pytest.approx(deg % 180, abs=1.0)


def test_ellipse_area_scales_with_the_square_of_size():
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(4000, 2))
    assert po.ellipse_area_95(xy * 2) == pytest.approx(4 * po.ellipse_area_95(xy), rel=0.02)


def test_path_length_of_a_known_square():
    xy = np.array([[0, 0], [3, 0], [3, 4], [0, 4], [0, 0]], float)
    assert po.path_length(xy)["path"] == pytest.approx(14.0)


def test_path_length_grows_with_added_noise():
    """Why it is not a quantity of motion: a noisier sensor reports a longer path."""
    rng = np.random.default_rng(0)
    clean = np.column_stack([np.linspace(0, 100, 5000), np.zeros(5000)])
    noisy = clean + rng.normal(0, 0.5, clean.shape)
    assert po.path_length(noisy)["path"] > 2 * po.path_length(clean)["path"]


def test_dispersion_radius_matches_a_uniform_disc():
    rng = np.random.default_rng(0)
    r = np.sqrt(rng.uniform(size=20000))
    th = rng.uniform(0, 2 * np.pi, 20000)
    xy = np.column_stack([r * np.cos(th), r * np.sin(th)])
    assert po.dispersion_radius(xy, 0.95) == pytest.approx(np.sqrt(0.95), abs=0.03)


# --------------------------------------------------------------- circular statistics

def test_circular_mean_wraps_correctly():
    """The failure that motivates the module: these average to 0, not to 180."""
    a = np.radians([350.0, 10.0])
    assert np.degrees(ci.circ_mean(a)["mean"]) == pytest.approx(0.0, abs=1e-6)


def test_resultant_length_spans_zero_to_one():
    rng = np.random.default_rng(0)
    assert ci.circ_mean(rng.uniform(0, 2 * np.pi, 20000))["R"] < 0.05
    assert ci.circ_mean(np.full(500, 1.2))["R"] == pytest.approx(1.0)


def test_rayleigh_accepts_uniform_and_rejects_concentrated():
    rng = np.random.default_rng(0)
    assert ci.rayleigh(rng.uniform(0, 2 * np.pi, 500))["p"] > 0.05
    assert ci.rayleigh(rng.normal(1.0, 0.3, 500))["p"] < 0.001


def test_ordinary_rayleigh_is_blind_to_a_bidirectional_axis():
    """The reason rayleigh_axial exists: front-back sway looks uniform to the plain test."""
    rng = np.random.default_rng(0)
    a = np.concatenate([rng.normal(0.0, 0.2, 400), rng.normal(np.pi, 0.2, 400)])
    assert ci.rayleigh(a)["p"] > 0.05
    assert ci.rayleigh_axial(a)["p"] < 0.001


def test_axial_rayleigh_recovers_the_axis():
    rng = np.random.default_rng(0)
    axis = np.radians(40.0)
    a = np.concatenate([rng.normal(axis, 0.15, 400), rng.normal(axis + np.pi, 0.15, 400)])
    got = np.degrees(ci.rayleigh_axial(a)["mean_axis"]) % 180
    assert min(abs(got - 40), abs(got - 220)) == pytest.approx(0, abs=4)


def test_axial_dispersion_is_small_for_a_tight_axis():
    rng = np.random.default_rng(0)
    tight = np.concatenate([rng.normal(0, 0.1, 500), rng.normal(np.pi, 0.1, 500)])
    assert ci.axial_dispersion(tight) < ci.axial_dispersion(rng.uniform(0, 2 * np.pi, 1000))


def test_circular_correlation_detects_a_shared_angle():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 2 * np.pi, 400)
    b = (a + rng.normal(0, 0.2, 400)) % (2 * np.pi)
    assert ci.circ_corr(a, b)["r"] > 0.8
    assert ci.circ_corr(a, rng.uniform(0, 2 * np.pi, 400))["p"] > 0.05


def test_circular_linear_correlation_detects_a_seasonal_pattern():
    """The shape of the question 'does stillness depend on time of year'."""
    rng = np.random.default_rng(0)
    doy = np.arange(365)
    ang = 2 * np.pi * doy / 365
    y = 5 + 2 * np.cos(ang) + rng.normal(0, 0.3, 365)
    assert ci.circ_corr_linear(ang, y)["p"] < 0.001
    assert ci.circ_corr_linear(ang, rng.normal(size=365))["p"] > 0.05


def test_vtest_is_more_powerful_when_the_direction_is_predicted():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.2, 60)
    assert ci.vtest(a, 0.0)["p"] <= ci.rayleigh(a)["p"]


# -------------------------------------------------------------------- gap handling

def test_short_gaps_are_bridged_and_long_ones_are_not():
    x = np.arange(1000, dtype=float)
    x[100:105] = np.nan          # short
    x[400:800] = np.nan          # long
    out = resample.interpolate_gaps(x, max_gap=50)
    assert np.isfinite(out[100:105]).all()
    assert out[102] == pytest.approx(102.0)
    assert np.isnan(out[400:800]).all()


def test_leading_and_trailing_gaps_are_never_invented():
    x = np.arange(100, dtype=float)
    x[:5] = np.nan
    x[-5:] = np.nan
    out = resample.interpolate_gaps(x)
    assert np.isnan(out[:5]).all()
    assert np.isnan(out[-5:]).all()


def test_gap_report_separates_scattered_from_blocked_missingness():
    fs = 100.0
    scattered = np.ones(10000)
    scattered[::100] = np.nan
    blocked = np.ones(10000)
    blocked[5000:5100] = np.nan
    a, b = resample.gap_report(scattered, fs), resample.gap_report(blocked, fs)
    assert a["missing_frac"] == pytest.approx(b["missing_frac"], rel=0.05)
    assert a["longest_gap_s"] < b["longest_gap_s"]
