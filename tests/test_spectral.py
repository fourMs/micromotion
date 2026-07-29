"""Spectral helpers and the extra dynamics measures."""

import numpy as np
import pytest

from micromotion import dynamics as dy
from micromotion import spectral as sp


def tone(f, fs, dur, amp=1.0, noise=0.0, seed=0):
    t = np.arange(0, dur, 1 / fs)
    x = amp * np.sin(2 * np.pi * f * t)
    if noise:
        x = x + np.random.default_rng(seed).normal(0, noise, len(t))
    return x


def test_spectral_peak_finds_a_tone_and_reports_high_snr():
    r = sp.spectral_peak(tone(1.2, 100.0, 300, noise=0.3), 100.0, (0.7, 2.2))
    assert r["freq"] == pytest.approx(1.2, abs=0.03)
    assert r["snr"] > 10


def test_spectral_peak_reports_low_snr_when_there_is_no_rhythm():
    x = np.random.default_rng(0).normal(size=30000)
    assert sp.spectral_peak(x, 100.0, (0.7, 2.2))["snr"] < 5


def test_band_rms_recovers_a_known_amplitude():
    """A sine of amplitude A has RMS A/sqrt(2)."""
    got = sp.band_rms(tone(2.0, 100.0, 200, amp=3.0), 100.0, (0.3, 10.0))
    assert got == pytest.approx(3.0 / np.sqrt(2), rel=0.03)


def test_band_rms_ignores_out_of_band_energy():
    fs = 100.0
    inside = tone(2.0, fs, 200, amp=3.0)
    outside = tone(30.0, fs, 200, amp=10.0)
    assert sp.band_rms(inside + outside, fs, (0.3, 10.0)) == pytest.approx(
        sp.band_rms(inside, fs, (0.3, 10.0)), rel=0.05
    )


def test_band_power_fractions_sum_to_about_one_and_split_correctly():
    fs = 100.0
    x = tone(0.25, fs, 400, amp=1.0) + tone(1.2, fs, 400, amp=1.0)
    fr = sp.band_power_fraction(x, fs, {"resp": (0.1, 0.5), "card": (0.7, 2.2)})
    assert fr["resp"] == pytest.approx(0.5, abs=0.1)
    assert fr["card"] == pytest.approx(0.5, abs=0.1)


def test_mean_frequency_sits_between_two_tones():
    fs = 100.0
    x = tone(1.0, fs, 300) + tone(3.0, fs, 300)
    assert 1.2 < sp.mean_frequency(x, fs, (0.3, 5.0)) < 2.8


def test_mean_frequency_moves_when_energy_shifts_without_the_peak_moving():
    fs = 100.0
    a = tone(1.0, fs, 300, amp=1.0) + tone(4.0, fs, 300, amp=0.2)
    b = tone(1.0, fs, 300, amp=1.0) + tone(4.0, fs, 300, amp=0.8)
    assert sp.mean_frequency(b, fs, (0.3, 6.0)) > sp.mean_frequency(a, fs, (0.3, 6.0))


def test_detect_breaths_counts_a_known_number_of_cycles():
    fs, dur, f = 25.0, 300.0, 0.25          # 15 breaths per minute
    r = sp.detect_breaths(tone(f, fs, dur, amp=5.0, noise=0.2), fs)
    assert r["n_breaths"] == pytest.approx(dur * f, abs=2)
    assert r["rate_per_min"] == pytest.approx(15.0, abs=1.0)


def test_detect_breaths_returns_peaks_and_troughs_in_alternation():
    r = sp.detect_breaths(tone(0.25, 25.0, 200, amp=5.0), 25.0)
    assert abs(len(r["peaks_s"]) - len(r["troughs_s"])) <= 1


# ------------------------------------------------------------------ extra dynamics

def test_apen_is_low_for_a_sine_and_high_for_noise():
    t = np.arange(3000) / 50.0
    assert dy.apen(np.sin(2 * np.pi * 0.5 * t)) < dy.apen(
        np.random.default_rng(0).normal(size=3000)
    )


def test_apen_and_sampen_agree_in_direction():
    rng = np.random.default_rng(0)
    t = np.arange(3000) / 50.0
    regular, noisy = np.sin(2 * np.pi * 0.5 * t), rng.normal(size=3000)
    assert (dy.apen(regular) < dy.apen(noisy)) == (dy.sampen(regular) < dy.sampen(noisy))


def test_dcca_is_near_one_for_identical_series():
    x = np.cumsum(np.random.default_rng(0).normal(size=4000))
    rho = dy.dcca(x, x)["rho"]
    assert np.nanmedian(rho) == pytest.approx(1.0, abs=0.02)


def test_dcca_is_near_zero_for_independent_walks():
    rng = np.random.default_rng(0)
    a, b = np.cumsum(rng.normal(size=6000)), np.cumsum(rng.normal(size=6000))
    assert abs(np.nanmedian(dy.dcca(a, b)["rho"])) < 0.6


def test_dcca_detects_a_shared_component_that_plain_correlation_misses():
    """Two wandering series with a common slow driver."""
    rng = np.random.default_rng(3)
    common = np.cumsum(rng.normal(size=6000))
    a = common + np.cumsum(rng.normal(size=6000)) * 3
    b = common + np.cumsum(rng.normal(size=6000)) * 3
    rho = dy.dcca(a, b)["rho"]
    assert np.nanmax(rho) > 0.3
