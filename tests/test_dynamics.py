"""Dynamics measures against processes whose answers are known in advance.

The corpus reported one such validation -- "z = +0.85 on an AR(1) process and z = -41 on a
logistic map" -- but no script implementing it survived. These tests restore it and extend
the same treatment to every other measure in the module, because all of them return a
plausible number when they are wrong.
"""

import numpy as np
import pytest

from micromotion import dynamics as dy


# ------------------------------------------------------------------ reference processes

def white(n=8000, seed=0):
    return np.random.default_rng(seed).normal(size=n)


def brownian(n=8000, seed=0):
    return np.cumsum(white(n, seed))


def ar1(n=8000, phi=0.8, seed=0):
    """Linear, Gaussian and reversible: the null these tests must not reject."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def logistic(n=8000, r=3.99, x0=0.4):
    """Deterministic, nonlinear and strongly irreversible."""
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x


def pink(n=8192, seed=0):
    """1/f noise, by spectral shaping. DFA alpha should come out near 1."""
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    return np.fft.irfft(X / np.sqrt(f), n=n)


# ------------------------------------------------------------------------------ DFA

def test_dfa_white_noise_is_one_half():
    assert dy.dfa(white())["alpha"] == pytest.approx(0.5, abs=0.08)


def test_dfa_brownian_motion_is_three_halves():
    assert dy.dfa(brownian())["alpha"] == pytest.approx(1.5, abs=0.1)


def test_dfa_pink_noise_is_one():
    assert dy.dfa(pink())["alpha"] == pytest.approx(1.0, abs=0.15)


def test_dfa_orders_the_three_correctly():
    a = [dy.dfa(s)["alpha"] for s in (white(), pink(), brownian())]
    assert a[0] < a[1] < a[2]


# ---------------------------------------------------------------------------- MFDFA

def test_mfdfa_h2_matches_dfa_alpha_on_a_monofractal():
    x = white(8192)
    assert dy.mfdfa(x)["h2"] == pytest.approx(dy.dfa(x)["alpha"], abs=0.15)


def test_mfdfa_width_is_small_for_a_monofractal():
    """White noise has one scaling exponent, so the singularity spectrum is narrow."""
    assert dy.mfdfa(white(8192))["width"] < 1.0


def test_mfdfa_width_is_larger_for_a_multifractal_cascade():
    """A binomial cascade is multifractal by construction."""
    n, p = 2**13, 0.3
    x = np.ones(1)
    while len(x) < n:
        x = np.concatenate([x * p, x * (1 - p)])
    assert dy.mfdfa(x)["width"] > dy.mfdfa(white(len(x)))["width"]


# ------------------------------------------------------- time reversal and surrogates

def test_ar1_is_not_distinguishable_from_its_surrogates():
    """The published check: an AR(1) process is reversible, so trev must not be significant."""
    r = dy.surrogate_test(ar1(4000), dy.trev, n=49,
                          rng=np.random.default_rng(3))
    assert abs(r["z"]) < 3.0
    assert r["p"] > 0.05


def test_logistic_map_is_strongly_distinguishable():
    """The other half of the published check: a logistic map is grossly irreversible."""
    r = dy.surrogate_test(logistic(4000), dy.trev, n=49,
                          rng=np.random.default_rng(3))
    assert abs(r["z"]) > 5.0
    assert r["p"] < 0.05


def test_iaaft_preserves_spectrum_and_distribution():
    x = logistic(2048)
    s = dy.iaaft(x, iters=200, rng=np.random.default_rng(0))
    assert np.allclose(np.sort(s), np.sort(x), atol=1e-6)
    px = np.abs(np.fft.rfft(x))
    ps = np.abs(np.fft.rfft(s))
    assert np.corrcoef(px, ps)[0, 1] > 0.99


def test_phase_surrogate_preserves_spectrum_but_not_distribution():
    x = logistic(2048)
    s = dy.phase_surrogate(x, rng=np.random.default_rng(0))
    assert np.corrcoef(np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(s)))[0, 1] > 0.99
    assert not np.allclose(np.sort(s), np.sort(x), atol=1e-3)


def test_circular_shift_preserves_the_series_exactly():
    x = white(1000)
    s = dy.circular_shift_surrogate(x, rng=np.random.default_rng(0))
    assert np.allclose(np.sort(s), np.sort(x))


# ------------------------------------------------------------------------------ SDA

def test_sda_on_brownian_motion_gives_one_half_in_both_regions():
    """Free diffusion has a single slope, so the two fitted exponents must agree."""
    rng = np.random.default_rng(0)
    x, y = np.cumsum(rng.normal(size=6000)), np.cumsum(rng.normal(size=6000))
    r = dy.sda(x, y, fs=25.0, maxlag=20.0)
    assert r["Hs"] == pytest.approx(0.5, abs=0.12)
    assert r["Hl"] == pytest.approx(0.5, abs=0.15)


def test_sda_detects_a_bounded_process_as_antipersistent_at_long_lags():
    """An Ornstein-Uhlenbeck process drifts, then is pulled back: Hl must fall below Hs."""
    rng = np.random.default_rng(1)
    n, theta = 20000, 0.02
    x = np.empty(n)
    x[0] = 0
    for i in range(1, n):
        x[i] = x[i - 1] - theta * x[i - 1] + rng.normal()
    r = dy.sda(x, np.zeros_like(x), fs=25.0, maxlag=40.0)
    assert r["Hl"] < r["Hs"]


# ------------------------------------------------------------------- entropy and RQA

def test_sampen_is_low_for_a_sine_and_high_for_noise():
    t = np.arange(4000) / 50.0
    assert dy.sampen(np.sin(2 * np.pi * 0.5 * t)) < dy.sampen(white(4000))


def test_rqa_determinism_is_near_one_for_a_sine():
    t = np.arange(1500) / 50.0
    assert dy.rqa(np.sin(2 * np.pi * 0.5 * t), dim=3, tau=10)["DET"] > 0.95


def test_rqa_determinism_is_low_for_white_noise():
    assert dy.rqa(white(1500), dim=3, tau=5)["DET"] < 0.3


def test_unit_delay_inflates_determinism():
    """A trap worth a test: tau=1 measures the embedding, not the dynamics."""
    x = white(1500)
    assert dy.rqa(x, dim=3, tau=1)["DET"] > 3 * dy.rqa(x, dim=3, tau=5)["DET"]


def test_rqa_recurrence_rate_matches_the_target():
    """The point of solving the threshold per plot rather than fixing it."""
    r = dy.rqa(white(800), dim=3, tau=1, rr=0.05)
    assert r["RR"] == pytest.approx(0.05, abs=0.01)


def test_rqa_is_amplitude_invariant():
    """Two recordings of different size must give the same determinism."""
    x = white(1000)
    assert dy.rqa(x, dim=3, tau=5)["DET"] == pytest.approx(
        dy.rqa(x * 100, dim=3, tau=5)["DET"], abs=0.02
    )


def test_embedding_delay_of_a_sine_is_near_a_quarter_period():
    fs, f = 50.0, 0.5
    t = np.arange(4000) / fs
    tau = dy.first_ami_minimum(np.sin(2 * np.pi * f * t), maxlag=60)
    assert tau == pytest.approx(fs / f / 4, rel=0.3)


# ------------------------------------------------------------------------------ PLV

def test_plv_is_one_for_identical_signals():
    t = np.arange(4000) / 50.0
    x = np.sin(2 * np.pi * 0.5 * t)
    assert dy.plv(x, x, 50.0)["plv"] == pytest.approx(1.0, abs=1e-6)


def test_plv_recovers_a_known_phase_offset():
    t = np.arange(4000) / 50.0
    a = np.sin(2 * np.pi * 0.5 * t)
    b = np.sin(2 * np.pi * 0.5 * t - np.pi / 3)
    r = dy.plv(a, b, 50.0)
    assert r["plv"] == pytest.approx(1.0, abs=1e-3)
    assert r["preferred_phase"] == pytest.approx(np.pi / 3, abs=0.05)


def test_plv_is_near_zero_for_independent_noise():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=8000), rng.normal(size=8000)
    assert dy.plv(a, b, 50.0, band=(0.3, 10.0))["plv"] < 0.2
