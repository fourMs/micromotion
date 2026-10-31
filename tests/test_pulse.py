"""The pulse envelope and the rate read from it.

A heartbeat on a chest accelerometer is a train of short pulses whose energy sits above the
body's sway. The acceleration spectrum of a standing body hides their fundamental under a steep
postural slope, and the envelope of the band where the pulses live recovers it. These tests build
that case synthetically, with the truth known by construction, and the real-data figures live in
the docstrings and in the Oslo Standstill reports that measured them.
"""

import numpy as np
import pytest
from scipy import signal

import micromotion as mm


def synth_chest(fs=256.0, dur=120.0, bpm=72.0, seed=0, sway_scale=0.02, pulse_scale=0.01,
                noise_scale=0.0003):
    """Postural sway in three axes, a sensor noise floor, and a pulse train at the heart rate.

    The sway is red noise low-passed at 3 Hz, so that it falls through the cardiac band as a
    standing body's does and leaves 5 to 30 Hz to the pulses and the noise. Each pulse is a short
    damped 15 Hz burst, about 10 mg peak, which is the size of a seismocardiogram on the sternum.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    sway = np.cumsum(rng.standard_normal((n, 3)), axis=0)
    sway = signal.detrend(sway, axis=0)
    sos = signal.butter(4, 3.0, btype="low", fs=fs, output="sos")
    sway = signal.sosfiltfilt(sos, sway, axis=0)
    sway *= sway_scale / sway.std()
    pulse = np.zeros(n)
    beats = np.cumsum(60.0 / bpm + 0.01 * rng.standard_normal(int(dur * bpm / 60) + 2))
    tt = np.arange(int(0.08 * fs)) / fs
    shape = np.exp(-tt / 0.02) * np.sin(2 * np.pi * 15 * tt)
    for b in beats[beats < dur - 0.1]:
        k = int(b * fs)
        pulse[k:k + len(tt)] += shape
    x = sway + pulse_scale * pulse[:, None] * np.array([1.0, 0.6, 0.3])
    return x + noise_scale * rng.standard_normal((n, 3)), fs


def test_envelope_rate_and_length():
    x, fs = synth_chest()
    env = mm.pulse_envelope(x, fs)
    assert env.ndim == 1
    assert abs(len(env) - len(x) * mm.PULSE_RATE / fs) <= 1
    assert env.min() >= 0.0            # a rectified sum is never negative


def test_envelope_accepts_a_single_series():
    x, fs = synth_chest()
    e1 = mm.pulse_envelope(x[:, 0], fs)
    e3 = mm.pulse_envelope(x, fs)
    assert len(e1) == len(e3)


def test_rate_matches_the_synthetic_heart():
    x, fs = synth_chest(bpm=72.0)
    r = mm.pulse_rate(x, fs)
    assert r["is_peak"]
    assert abs(r["bpm"] - 72.0) < 2.0


def test_acceleration_spectrum_misses_what_the_envelope_finds():
    """The reason the function exists: the raw cardiac band holds no peak, the envelope does."""
    x, fs = synth_chest(bpm=72.0)
    mag = np.linalg.norm(x - x.mean(axis=0), axis=1)
    raw = mm.spectral_peak(mag, fs, (1.0, 2.0), window_s=30.0)
    env = mm.pulse_rate(x, fs, cardiac_band=(1.0, 2.0))
    assert env["is_peak"]
    assert not raw["is_peak"] or abs(raw["freq"] * 60 - 72.0) > abs(env["bpm"] - 72.0)


def test_no_heart_gives_nan_not_a_number():
    x, fs = synth_chest(pulse_scale=0.0)
    r = mm.pulse_rate(x, fs)
    assert not r["is_peak"]
    assert np.isnan(r["bpm"])


def test_refuses_a_fragment():
    x, fs = synth_chest(dur=1.0)
    with pytest.raises(ValueError):
        mm.pulse_envelope(x, fs)


def test_edge_bin_from_a_measured_rate_is_still_in_the_band():
    """A grid scaled by a measured rate puts the 2.0 Hz bin at 2.00000002 Hz; a rhythm one bin
    below it must keep its interior peak, as it does on the stated-rate grid."""
    df = 1.0 / 30.0
    f_exact = np.arange(1, 91) * df
    p = 1.0 / f_exact ** 2
    k = int(round(1.9667 / df)) - 1          # the bin just below 2.0 Hz
    p[k] *= 8.0
    exact = mm.peak_from_spectrum(f_exact, p, (1.0, 2.0))
    drifted = mm.peak_from_spectrum(f_exact * (1 + 1e-8), p, (1.0, 2.0))
    assert exact["is_peak"] and drifted["is_peak"]
    assert abs(exact["freq"] - drifted["freq"]) < 1e-6
    # and a bin a whole spacing past the edge is still outside
    assert mm.spectral._in_band(np.array([2.0 + df]), (1.0, 2.0)).sum() == 0
