"""Spectral helpers and the extra dynamics measures."""

import numpy as np
from scipy import signal
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


def pink(fs, dur, seed=0):
    """A 1/f series: falling spectrum, no peak anywhere in it."""
    n = int(fs * dur)
    w = np.random.default_rng(seed).normal(0, 1, n)
    F = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / fs)
    F[1:] /= f[1:]
    F[0] = 0
    return np.fft.irfft(F, n=n)


BAND = (0.12, 0.40)     # a respiration band, drawn above the knee of a 1/f spectrum


def test_spectral_peak_refuses_a_slope_instead_of_returning_its_lowest_bin():
    """The failure this function was reporting as a measurement.

    A band drawn above the knee of a 1/f spectrum contains no peak, and a bare argmax returns the
    band's lowest bin. That value is not a rhythm, it is the slope, and it is where four analyses
    in the Oslo corpus were reading breathing rates and sway frequencies from -- one of them on
    662 of 930 values, one of them into a book.
    """
    for seed in range(6):
        r = sp.spectral_peak(pink(10.0, 600, seed), 10.0, BAND)
        assert np.isnan(r["freq"]), f"seed {seed} found a peak in a slope"
        assert r["is_peak"] is False


def test_the_snr_does_not_catch_a_slope_which_is_why_the_excess_exists():
    """Tightening an SNR threshold selects the artefact rather than excluding it.

    The band median assumes a flat band. Over 1/f it is dragged down by the high-frequency end,
    so the lowest bins clear any SNR bar without being peaks.
    """
    old = sp.spectral_peak(pink(10.0, 600), 10.0, BAND, require_peak=False)
    assert not np.isnan(old["freq"])
    assert old["snr"] > 3               # would pass the advice the old docstring gave
    assert old["freq"] < BAND[0] + 0.04  # and it is the bottom of the band


def test_rejecting_the_edge_bin_would_not_have_been_enough():
    """The reason the test is "is this a peak" and not "where is it".

    On a falling spectrum, refusing the first bin moves the maximum to the second.
    """
    x = pink(10.0, 600)
    f, p = signal.welch(signal.detrend(x), 10.0, nperseg=int(10.0 * 60))
    pb = p[(f >= BAND[0]) & (f <= BAND[1])]
    assert int(np.argmax(pb)) <= 1                      # at or beside the floor
    assert int(np.argmax(pb[1:])) <= 1                  # and one bin along, the same shape
    assert np.isnan(sp.spectral_peak(x, 10.0, BAND)["freq"])


def test_require_peak_false_restores_the_old_behaviour():
    x = pink(10.0, 600)
    assert np.isnan(sp.spectral_peak(x, 10.0, BAND)["freq"])
    assert not np.isnan(sp.spectral_peak(x, 10.0, BAND, require_peak=False)["freq"])


def test_a_real_rhythm_riding_on_a_slope_still_survives_the_rule():
    """The rule must not throw away the thing it exists to protect.

    A breath is a modest peak on top of a much larger postural slope, which is exactly the shape
    that makes this hard: the raw argmax is still down at the band floor, and the peak is only
    visible once the slope is divided out.
    """
    fs, dur = 10.0, 600
    t = np.arange(0, dur, 1 / fs)
    x = pink(fs, dur, seed=1) + 0.7 * np.sin(2 * np.pi * 0.25 * t)
    raw = sp.spectral_peak(x, fs, BAND, require_peak=False)
    assert raw["freq"] < BAND[0] + 0.02           # the old answer returns the band floor
    r = sp.spectral_peak(x, fs, BAND)
    assert r["is_peak"] is True
    assert r["freq"] == pytest.approx(0.25, abs=0.02)  # the new one finds the rhythm
    assert r["excess"] > 2


def test_the_rule_is_conservative_and_misses_a_rhythm_below_the_slope():
    """What it costs, stated rather than discovered later.

    Dividing out the slope makes a peak findable, not visible from nothing. Below about half the
    amplitude of the case above, a genuine 0.25 Hz rhythm is present and this returns NaN. That is
    the intended direction to fail in -- a missing value rather than a wrong one -- but it is a
    false negative and a rate computed from these will be missing its weakest cases.
    """
    fs, dur = 10.0, 600
    t = np.arange(0, dur, 1 / fs)
    for amp in (0.2, 0.3, 0.4):
        x = pink(fs, dur, seed=1) + amp * np.sin(2 * np.pi * 0.25 * t)
        assert np.isnan(sp.spectral_peak(x, fs, BAND)["freq"])


def test_respiratory_peak_recovers_a_breath_riding_on_much_larger_slow_drift():
    """The case the periodogram version failed.

    It took the global maximum of the in-band spectrum, which on a red background lands near the
    band floor. Measured on Stillness2025's sixteen belts it returned a median 7.5 breaths per
    minute against the belts' own 16.8, and ranked participants at Spearman -0.32 against their own
    breath timing. It is now measured in the time domain, from detect_breaths.
    """
    fs, dur, breath_hz = 25.6, 300.0, 0.25          # 15 breaths per minute
    t = np.arange(0, dur, 1 / fs)
    rng = np.random.default_rng(0)
    drift = 10.0 * np.cumsum(rng.normal(0, 1, t.size)) / np.sqrt(t.size)
    got = sp.respiratory_peak(np.sin(2 * np.pi * breath_hz * t) + drift, fs)
    assert got * 60 == pytest.approx(breath_hz * 60, abs=1.5)


def test_respiratory_peak_agrees_with_detect_breaths():
    fs = 25.6
    t = np.arange(0, 240, 1 / fs)
    x = np.sin(2 * np.pi * 0.3 * t) + 0.05 * np.sin(2 * np.pi * 1.1 * t)
    assert sp.respiratory_peak(x, fs) * 60 == pytest.approx(
        sp.detect_breaths(x, fs)["rate_per_min"], rel=1e-9)


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


def test_peak_from_spectrum_is_the_same_rule_as_spectral_peak():
    """The primitive and the convenience wrapper must not drift apart."""
    fs, dur = 10.0, 600
    t = np.arange(0, dur, 1 / fs)
    for x in (pink(fs, dur, 3), pink(fs, dur, 1) + 0.7 * np.sin(2 * np.pi * 0.25 * t)):
        nper = int(min(len(x), fs * 60.0))
        f, p = signal.welch(signal.detrend(x), fs, nperseg=nper)
        a = sp.spectral_peak(x, fs, BAND)
        b = sp.peak_from_spectrum(f, p, BAND)
        assert a["is_peak"] == b["is_peak"]
        assert (np.isnan(a["freq"]) and np.isnan(b["freq"])) or a["freq"] == b["freq"]
