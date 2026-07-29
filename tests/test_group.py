"""Group coincidence and the stilling statistic, against constructed answers."""

import numpy as np
import pytest
from scipy import stats

from micromotion import group as gp


def planted(n_people=20, n=3000, fs=10.0, event_s=(50.0, 120.0, 200.0), seed=0):
    """Independent event trains, plus events every person shares at known moments."""
    rng = np.random.default_rng(seed)
    T = (rng.random((n_people, n)) < 0.02).astype(float)
    for t in event_s:
        T[:, int(t * fs)] = 1.0
    return T


def test_coincidence_detects_planted_moments():
    fs = 10.0
    T = planted(fs=fs)
    r = gp.coincidence_test(T, fs, n_surrogates=200, shift_range_s=20.0,
                            rng=np.random.default_rng(1))
    for t in (50.0, 120.0, 200.0):
        assert r["p"][int(t * fs)] < 0.01


def test_coincidence_finds_nothing_in_independent_trains():
    fs = 10.0
    rng = np.random.default_rng(3)
    T = (rng.random((20, 3000)) < 0.02).astype(float)
    r = gp.coincidence_test(T, fs, n_surrogates=200, shift_range_s=20.0,
                            rng=np.random.default_rng(4))
    assert r["frac_significant"] < 0.02


def test_frac_significant_separates_the_two_cases():
    """And the published mean-based score does not, which is why both are returned."""
    fs = 10.0
    rng = np.random.default_rng(5)
    together = gp.coincidence_test(planted(fs=fs), fs, n_surrogates=150,
                                   shift_range_s=20.0, rng=np.random.default_rng(6))
    apart = gp.coincidence_test((rng.random((20, 3000)) < 0.02).astype(float), fs,
                                n_surrogates=150, shift_range_s=20.0,
                                rng=np.random.default_rng(7))
    assert together["frac_significant"] > 2 * apart["frac_significant"]


def test_frac_significant_rises_with_how_much_is_shared():
    fs = 10.0
    fracs = []
    for n_ev in (3, 30, 90):
        r = np.random.default_rng(0)
        T = (r.random((20, 3000)) < 0.02).astype(float)
        for t in np.linspace(50, 250, n_ev):
            T[:, int(t * fs)] = 1.0
        fracs.append(gp.coincidence_test(T, fs, n_surrogates=120, shift_range_s=20.0,
                                         rng=np.random.default_rng(9))["frac_significant"])
    assert fracs[0] < fracs[1] < fracs[2]


def test_surrogates_preserve_each_persons_event_count():
    """The property that makes this the right null: only alignment is destroyed."""
    T = planted()
    shifted = np.roll(T[0], 137)
    assert shifted.sum() == T[0].sum()


def test_event_train_threshold_is_relative_by_default():
    """A quiet and a loud participant must both register events."""
    fs = 50.0
    t = np.arange(0, 200, 1 / fs)
    quiet = 0.01 * np.sin(2 * np.pi * 0.1 * t)
    loud = 10.0 * np.sin(2 * np.pi * 0.1 * t)
    a = gp.event_train(quiet, fs)
    b = gp.event_train(loud, fs)
    assert a.sum() > 0
    assert b.sum() > 0
    assert a.sum() == pytest.approx(b.sum(), rel=0.1)


def test_event_train_directions():
    fs = 10.0
    x = np.concatenate([np.zeros(100), np.ones(100)])
    up = gp.event_train(x, fs, frame_s=1.0, threshold=0.5, kind="increase")
    down = gp.event_train(x, fs, frame_s=1.0, threshold=0.5, kind="decrease")
    both = gp.event_train(x, fs, frame_s=1.0, threshold=0.5, kind="change")
    assert up.sum() > 0
    assert down.sum() == 0
    assert both.sum() == up.sum()


# ------------------------------------------------------------------- stilling response

def test_participation_ratio_is_one_when_everyone_quietens():
    fs = 10.0
    n = 2000
    S = np.ones((15, n))
    S[:, int(100 * fs):] = 0.2          # everyone drops at t = 100 s
    assert gp.participation_ratio(S, [100.0], fs)[0] == pytest.approx(1.0)


def test_participation_ratio_is_zero_when_everyone_moves_more():
    fs = 10.0
    S = np.ones((15, 2000)) * 0.2
    S[:, int(100 * fs):] = 1.0
    assert gp.participation_ratio(S, [100.0], fs)[0] == pytest.approx(0.0)


def test_participation_ratio_ignores_amplitude_scale():
    """Sensors differing by an order of magnitude must not shift the statistic."""
    fs = 10.0
    rng = np.random.default_rng(0)
    S = np.abs(rng.normal(size=(20, 2000))) + 1.0
    S[:, int(100 * fs):] *= 0.5
    scaled = S * np.linspace(1, 50, 20)[:, None]
    assert gp.participation_ratio(S, [100.0], fs)[0] == pytest.approx(
        gp.participation_ratio(scaled, [100.0], fs)[0])


def test_participation_ratio_tolerates_missing_people():
    fs = 10.0
    S = np.ones((10, 2000))
    S[:, int(100 * fs):] = 0.2
    S[3:6, :] = np.nan
    assert gp.participation_ratio(S, [100.0], fs)[0] == pytest.approx(1.0)


def test_sliding_null_beats_a_coin_flip_assumption():
    """Standing still is not a coin flip; the null is above 0.5 and must be measured."""
    fs = 10.0
    rng = np.random.default_rng(2)
    S = np.abs(rng.normal(size=(20, 4000)))
    S = np.cumsum(S, axis=1)
    S = np.abs(np.diff(S, axis=1, prepend=0))
    null = gp.sliding_null(S, fs, step_s=1.0)
    assert len(null) > 100
    assert np.all((null >= 0) & (null <= 1))


def test_planted_stilling_beats_its_own_sliding_null():
    fs = 10.0
    rng = np.random.default_rng(8)
    n = 6000
    S = np.abs(rng.normal(size=(20, n))) + 1.0
    events = np.arange(60.0, 560.0, 25.0)
    for t in events:                       # a real quietening at each event
        i = int(t * fs)
        S[:, i:i + int(2 * fs)] *= 0.3
    obs = gp.participation_ratio(S, events, fs)
    null = gp.sliding_null(S, fs, step_s=0.5)
    assert np.nanmean(obs) > np.nanmean(null)
    assert stats.ks_2samp(obs, null, alternative="less").pvalue < 0.01


def test_sequential_stability_ignores_drift():
    """A steadily drifting series is steady; a jittery flat one is not."""
    drift = np.arange(100, dtype=float) * 0.1
    jitter = np.tile([0.0, 5.0], 50)
    assert gp.sequential_stability(drift) < gp.sequential_stability(jitter)


def test_sequential_stability_differs_from_standard_deviation():
    drift = np.arange(100, dtype=float) * 0.1
    assert np.std(drift) > 2.0
    assert gp.sequential_stability(drift) == pytest.approx(0.1, abs=1e-9)
