"""Alignment against offsets that are known because they were imposed."""

import numpy as np
import pytest

import micromotion as mm
from micromotion import align


def wandering_rate(n, fs=1.0, seed=0):
    """A heart rate that drifts, as a real one does. Flat rates cannot be aligned."""
    rng = np.random.default_rng(seed)
    return 70 + np.cumsum(rng.normal(0, 0.3, n)) * 0.5


def _delayed_pair(n=600, lag=47):
    """``a`` and ``b`` cut from one series so that b carries every feature LAG later.

    Cut rather than zero-padded on purpose: these signals sit around 70, so padding with
    zeros puts a step of 70 at the join and the correlation finds the step instead of the
    signal. That is a real trap and it cost a wrong sign here once already.
    """
    x = wandering_rate(n)
    return x[lag:], x[: len(x) - lag]


def test_xcorr_recovers_an_imposed_lag():
    """``b`` carries each feature later than ``a``, so the lag must come back POSITIVE.

    This assertion read ``-lag`` until 1.13.0 -- the value the implementation produced
    rather than the one the docstring promised. A test written after the fact locks in
    whichever sign the code had.
    """
    lag = 47
    a, b = _delayed_pair(600, lag)
    r = align.xcorr_lag(a, b, fs=1.0)
    assert r["lag_s"] == pytest.approx(lag, abs=1)
    assert r["confident"]
    assert align.xcorr_lag(b, a, fs=1.0)["lag_s"] == pytest.approx(-lag, abs=1)


def test_xcorr_and_search_lag_agree_on_the_sign():
    """Two alignment functions in one module must not disagree about which way is later."""
    lag = 40
    a, b = _delayed_pair(1200, lag)
    ta = np.arange(len(a), dtype=float)
    tb = np.arange(len(b), dtype=float)
    assert align.xcorr_lag(a, b, fs=1.0)["lag_s"] == pytest.approx(lag, abs=2)
    assert align.search_lag(ta, a, tb, b, max_lag_s=200.0,
                            min_overlap_s=200.0)["lag_s"] == pytest.approx(lag, abs=2)


def test_xcorr_agrees_with_musicalgestures():
    """The sibling package's convention is the reference; it was right when this was not."""
    mg = pytest.importorskip("musicalgestures")
    lag = 25
    a, b = _delayed_pair(800, lag)
    theirs, _ = mg.xcorr_lag(a, b, 1.0, max_lag=200)
    assert align.xcorr_lag(a, b, fs=1.0)["lag_s"] == pytest.approx(theirs, abs=1)


def test_xcorr_rejects_unrelated_noise():
    """Two unrelated series still have a maximum; it must not be trusted."""
    rng = np.random.default_rng(0)
    r = align.xcorr_lag(rng.normal(size=2000), rng.normal(size=2000), fs=1.0)
    assert not r["confident"]


def test_differencing_kills_spurious_random_walk_correlation():
    """Over independent walks, raw best-lag r exceeds 0.5 often; differenced it never does.

    This is why differencing is the default rather than an option.
    """
    raw, diffed = [], []
    for seed in range(60):
        r = np.random.default_rng(seed)
        a, b = np.cumsum(r.normal(size=600)), np.cumsum(r.normal(size=600))
        raw.append(align.xcorr_lag(a, b, fs=1.0, difference=False)["r"])
        diffed.append(align.xcorr_lag(a, b, fs=1.0)["r"])
    assert np.median(raw) > 3 * np.median(diffed)
    assert max(diffed) < 0.5
    assert not any(align.xcorr_lag(np.cumsum(np.random.default_rng(s).normal(size=600)),
                                   np.cumsum(np.random.default_rng(s + 500).normal(size=600)),
                                   fs=1.0)["confident"] for s in range(30))


def test_search_lag_handles_unequal_lengths_and_offsets():
    """Shifting b's TIMESTAMPS back by 126 makes every feature happen 126 EARLIER in b.

    So the lag is negative, and this assertion read ``+offset`` until 1.13.0. It is the
    same inverted sign as the xcorr test above, from the same implementation, and having
    both wrong in the same direction is what made them look consistent.
    """
    x = wandering_rate(900)
    t = np.arange(len(x), dtype=float)
    offset = 126.0
    r = align.search_lag(t, x, t[:600] - offset, x[:600], max_lag_s=300)
    assert r["lag_s"] == pytest.approx(-offset, abs=2)
    assert r["confident"]


def test_search_lag_declines_when_the_signals_do_not_match():
    rng = np.random.default_rng(1)
    t = np.arange(900, dtype=float)
    r = align.search_lag(t, rng.normal(size=900), t, rng.normal(size=900), max_lag_s=100)
    assert not r["confident"]


def test_instantaneous_rate_tracks_a_changing_frequency():
    """A chirp from 60 to 90 bpm must be reported as rising through that range."""
    fs, dur = 50.0, 400.0
    t = np.arange(0, dur, 1 / fs)
    f0, f1 = 1.0, 1.5
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * dur))
    ts, rate = align.instantaneous_rate(np.sin(phase), fs, win_s=30, step_s=5)
    assert len(rate) > 10
    assert rate[0] == pytest.approx(60, abs=8)
    assert rate[-1] == pytest.approx(90, abs=8)


def test_find_transient_locates_claps():
    fs = 100.0
    x = np.random.default_rng(0).normal(0, 0.01, int(fs * 120))
    for at in (3.0, 117.0):
        x[int(at * fs)] += 5.0
    found = align.find_transient(x, fs, search_s=20.0)
    assert len(found) == 2
    assert found[0] == pytest.approx(3.0, abs=0.1)
    assert found[1] == pytest.approx(117.0, abs=0.1)


def test_find_transient_finds_nothing_in_quiet_noise():
    fs = 100.0
    x = np.random.default_rng(0).normal(0, 1.0, int(fs * 60))
    assert len(align.find_transient(x, fs, threshold=15.0)) == 0


def test_apply_lag_shifts_a_timebase():
    t = np.arange(10, dtype=float)
    assert align.apply_lag(t, 2.5)[0] == 2.5


def test_alignment_is_reachable_from_the_top_level():
    assert mm.xcorr_lag is align.xcorr_lag
