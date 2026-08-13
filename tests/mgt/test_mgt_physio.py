"""Tests for musicalgestures._physio (respiration + spectral band fractions).

Synthetic ground truth: a 0.25 Hz breathing sinusoid must give a respiration
rate of ~15 breaths/min; a two-tone signal must split its Welch power between
two named bands in the expected proportion.
"""
import numpy as np
import pytest

from micromotion import respiration_rate, spectral_band_fractions


class TestRespirationRate:
    def test_quarter_hz_breathing_is_15_bpm(self):
        fs = 25.0
        dur = 120.0
        t = np.arange(0, dur, 1 / fs)
        wave = np.sin(2 * np.pi * 0.25 * t)  # 0.25 Hz -> 15 br/min
        out = respiration_rate(wave, fs, band=(0.1, 0.6),
                               window_s=30, step_s=30)
        assert out["median_bpm"] == pytest.approx(15.0, abs=1.0)
        assert np.isfinite(out["rate_bpm"]).all()

    def test_windows_and_times(self):
        fs = 20.0
        dur = 120.0
        t = np.arange(0, dur, 1 / fs)
        wave = np.sin(2 * np.pi * 0.2 * t)  # 12 br/min
        out = respiration_rate(wave, fs, window_s=30, step_s=30)
        assert len(out["rate_bpm"]) == len(out["times_s"])
        assert out["median_bpm"] == pytest.approx(12.0, abs=1.0)

    def test_added_noise_still_recovers_rate(self):
        rng = np.random.default_rng(0)
        fs = 25.0
        t = np.arange(0, 120, 1 / fs)
        wave = np.sin(2 * np.pi * 0.25 * t) + 0.3 * rng.standard_normal(len(t))
        out = respiration_rate(wave, fs)
        assert out["median_bpm"] == pytest.approx(15.0, abs=1.5)


class TestSpectralBandFractions:
    def test_two_tone_split(self):
        # Equal-amplitude tones at 0.3 Hz and 1.2 Hz; each named band should
        # capture close to half the in-band power.
        fs = 50.0
        t = np.arange(0, 200, 1 / fs)
        x = np.sin(2 * np.pi * 0.3 * t) + np.sin(2 * np.pi * 1.2 * t)
        bands = {"low": (0.2, 0.5), "high": (1.0, 1.5)}
        frac = spectral_band_fractions(x, fs, bands, total_band=(0.1, 4.0))
        assert frac["low"] == pytest.approx(0.5, abs=0.1)
        assert frac["high"] == pytest.approx(0.5, abs=0.1)
        assert frac["low"] + frac["high"] == pytest.approx(1.0, abs=0.1)

    def test_dominant_band_gets_most_power(self):
        fs = 50.0
        t = np.arange(0, 200, 1 / fs)
        # a strong 1.2 Hz tone and a weak 0.3 Hz tone
        x = 0.2 * np.sin(2 * np.pi * 0.3 * t) + np.sin(2 * np.pi * 1.2 * t)
        bands = {"low": (0.2, 0.5), "high": (1.0, 1.5)}
        frac = spectral_band_fractions(x, fs, bands, total_band=(0.1, 8.0))
        assert frac["high"] > frac["low"]

    def test_empty_signal(self):
        frac = spectral_band_fractions(np.array([]), 50.0, {"a": (0.1, 0.5)},
                                       total_band=(0.1, 8.0))
        assert np.isnan(frac["a"])

    def test_an_unstated_denominator_warns_and_still_returns_the_old_number(self):
        """A share whose denominator nobody wrote down is the failure this warning is for.

        Four standstill shares -- 38, 43, 45 and 58 per cent -- were quoted against one
        another in this corpus while each rested on a different, unstated denominator. The
        fallback band cannot move, because published numbers were computed with it, so the
        warning is the only thing that changes: the value with the band left unset must equal
        the value with (0.1, 8.0) passed.
        """
        import warnings

        fs = 50.0
        t = np.arange(0, 200, 1 / fs)
        x = 0.2 * np.sin(2 * np.pi * 0.3 * t) + np.sin(2 * np.pi * 1.2 * t)
        bands = {"low": (0.2, 0.5), "high": (1.0, 1.5)}

        with pytest.warns(RuntimeWarning, match="not given a total_band"):
            unstated = spectral_band_fractions(x, fs, bands)
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # naming the band silences it
            stated = spectral_band_fractions(x, fs, bands, total_band=(0.1, 8.0))
        assert unstated == stated

    def test_the_half_open_bin_sum_is_still_what_it_computes(self):
        """The published convention, pinned against the other one in this package.

        ``spectral_band_fractions`` sums bins over [lo, hi); ``band_share`` integrates with the
        trapezoid rule over [lo, hi] unless told otherwise. Both numbers are legitimate and
        they are not the same number.
        """
        from micromotion import band_share

        fs = 50.0
        rng = np.random.default_rng(0)
        t = np.arange(0, 400, 1 / fs)
        x = np.cumsum(rng.standard_normal(len(t))) / 1e3 + 0.02 * np.sin(2 * np.pi * 1.2 * t)
        num, den = (0.7, 2.2), (0.1, 3.0)

        theirs = spectral_band_fractions(x, fs, {"card": num}, total_band=den)["card"]
        same = band_share(x, fs, num_band=num, den_band=den, window_s=20,
                          integrate="sum", interval="half_open")
        other = band_share(x, fs, num_band=num, den_band=den, window_s=20)
        # the detrend differs (linear here, mean removal there), so this is close, not equal
        assert same == pytest.approx(theirs, abs=0.005)
        assert abs(other - theirs) > 0.01
