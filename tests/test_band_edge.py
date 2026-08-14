"""What every bounded peak finder here returns when there is nothing in the band.

Written 2026-08-14, after a fifth instance of the same failure in the Oslo Standstill corpus: a
remote-photoplethysmography pipeline band-passed a colour signal between 0.7 and 4 Hz, reported the
largest peak inside, and returned a near-constant 1.24 to 1.33 times its own lower edge. Sweeping
that edge from 0.5 to 1.5 Hz dragged the reported heart rate from 40 to 116 beats a minute, and the
estimate never correlated with a worn reference above 0.21 at any setting.

The reason it survived four earlier discoveries of the same bug is in one sentence: the values were
not ON the boundary, they were PROPORTIONAL to it. The corpus's audit looks for measurements piled
up on a search boundary and would have passed that pipeline clean. So would every check this
package had.

This file establishes what each finder in this package does on a band with no peak in it, so that
the answers are pinned rather than assumed, and it tests the sweep that catches the proportional
case. The signals are synthetic and the failure is reproduced rather than described.

Every test below was verified able to fail; the mutation used is named in each docstring.
"""

import warnings

import numpy as np
import pytest

import micromotion as mm
from micromotion import align, dynamics as dy, mocap, physio, spectral as sp

FS = 10.0
DUR = 600.0
CARDIAC = (0.7, 2.2)


def coloured(alpha, seed=0, fs=FS, dur=DUR):
    """A 1/f**alpha series: falling spectrum, no peak anywhere in it."""
    n = int(fs * dur)
    w = np.random.default_rng(seed).normal(0, 1, n)
    F = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / fs)
    F[1:] /= f[1:] ** alpha
    F[0] = 0
    x = np.fft.irfft(F, n=n)
    return x / x.std()


def with_tone(freq, amp=1.5, alpha=1.0, seed=0):
    t = np.arange(0, DUR, 1 / FS)
    return coloured(alpha, seed) + amp * np.sin(2 * np.pi * freq * t)


def caught(fn, *a, **kw):
    """Run something and return (value, the band-floor warnings it raised)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn(*a, **kw)
    return value, [w for w in rec if "search band" in str(w.message)
                   or "bounded search" in str(w.message)]


# --------------------------------------------------------- the audit: what they return

def test_the_bare_finders_return_a_multiple_of_the_edge_and_not_the_edge():
    """The measurement this whole file exists for.

    Four estimators, one signal shape with nothing in the band, and none of them returns NaN. What
    matters is the second column: the answers are 1.05 to 1.41 times the lower edge and almost
    never equal to it, which is why an audit for values sitting ON a boundary found nothing.

    Verified able to fail: asserting `on_edge == 40` for any of the four fails, since the answers
    are not on the edge; raising the ratio bound to 1.9 fails on all four.
    """
    xs = [coloured(1.0, s) for s in range(40)]
    lo = CARDIAC[0]

    def sweepless(fn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = np.array([fn(x) for x in xs], float)
        return float(np.median(v)) / lo, int((v == lo).sum())

    ratio, on_edge = {}, {}
    for name, fn in (
        ("cardiac_peak", lambda x: sp.cardiac_peak(x, FS)),
        ("dominant_frequency", lambda x: mocap.dominant_frequency(x, FS, CARDIAC)),
        ("instantaneous_rate", lambda x: float(np.nanmedian(
            align.instantaneous_rate(x, FS, CARDIAC, win_s=60, step_s=10, per_minute=False)[1]))),
        ("respiration_rate", lambda x: physio.respiration_rate(
            x, FS, band=CARDIAC, window_s=60, step_s=30)["median_bpm"] / 60.0),
    ):
        ratio[name], on_edge[name] = sweepless(fn)

    for name in ratio:
        assert 1.0 <= ratio[name] < 1.9, f"{name} returned {ratio[name]:.3f} x the edge"
        assert on_edge[name] < 20, f"{name} sat on the edge {on_edge[name]}/40 times"

    # the band-pass-then-search-the-same-band shape is the worst of the four, by a wide margin
    assert ratio["respiration_rate"] > ratio["cardiac_peak"] + 0.25
    # and the two that never touch the edge are invisible to an equality check
    assert on_edge["dominant_frequency"] == 0
    assert on_edge["respiration_rate"] == 0


def test_spectral_peak_is_the_one_that_was_already_safe():
    """A clean result is a result. The package's flagship finder passes its own audit.

    Verified able to fail: dropping `require_peak` back to False makes every assertion here fail,
    which is the same thing as saying this is what the parameter buys.
    """
    for alpha in (0.0, 1.0, 2.0):
        for seed in range(6):
            r = sp.spectral_peak(coloured(alpha, seed), FS, CARDIAC)
            assert np.isnan(r["freq"])
            assert r["is_peak"] is False


# --------------------------------------------------------- the audit: that they now say so

@pytest.mark.parametrize("call", [
    lambda x: sp.cardiac_peak(x, FS),
    lambda x: mocap.dominant_frequency(x, FS, CARDIAC),
    lambda x: align.instantaneous_rate(x, FS, CARDIAC, win_s=60, step_s=10),
    lambda x: physio.respiration_rate(x, FS, band=CARDIAC, window_s=60, step_s=30),
])
def test_each_bare_finder_warns_when_the_band_holds_no_peak(call):
    """Verified able to fail: removing the `is_band_floor` call from any one finder fails it."""
    for seed in range(4):
        _, w = caught(call, coloured(1.0, seed))
        assert w, "returned a band-edge multiple without saying so"
        assert "not a peak" in str(w[0].message)


@pytest.mark.parametrize("call", [
    lambda x: sp.cardiac_peak(x, FS),
    lambda x: mocap.dominant_frequency(x, FS, CARDIAC),
    lambda x: align.instantaneous_rate(x, FS, CARDIAC, win_s=60, step_s=10),
    lambda x: physio.respiration_rate(x, FS, band=CARDIAC, window_s=60, step_s=30),
])
def test_no_finder_warns_about_a_rhythm_that_is_really_there(call):
    """A warning that fires on good data is a warning people switch off.

    Verified able to fail: warning unconditionally instead of on `is_band_floor` fails all four.
    """
    for seed in range(4):
        _, w = caught(call, with_tone(1.2, seed=seed))
        assert not w, f"warned about a real 1.2 Hz rhythm: {w[0].message if w else ''}"


def test_the_warning_does_not_change_a_single_returned_value():
    """Published numbers came out of these functions and must still come out of them.

    Verified able to fail: changing `_peak` to return NaN when `is_band_floor` fails this at once.
    """
    for seed in range(6):
        x = coloured(1.0, seed)
        f, p = np.array([]), np.array([])
        from scipy import signal as sig
        f, p = sig.welch(sig.detrend(x), FS, nperseg=int(FS * 60))
        m = (f >= CARDIAC[0]) & (f <= CARDIAC[1])
        old = float(f[m][np.argmax(p[m])])                  # the pre-1.12 arithmetic, inline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert sp.cardiac_peak(x, FS) == old
            assert mocap.dominant_frequency(with_tone(1.2, seed=seed), FS, CARDIAC) > 0


def test_the_peak_test_must_be_given_an_unfiltered_spectrum():
    """Why respiration_rate tests the raw segment and not the one it measured.

    A band-pass over the same band builds a rising skirt inside the passband, and that skirt
    survives dividing out the log-log slope: the rule calls it a peak on most series that have
    none. This is the same reason report 103's own diagnostics -- peak height over band median,
    in-band power share -- failed to flag the pipeline they shipped with.

    Verified able to fail: swapping the two assertions' bounds fails both, which is the same
    statement as the measurement; and raising `min_excess` to 5 takes `n_filtered` to 0.
    """
    from scipy import signal as sig

    b, a = sig.butter(2, [CARDIAC[0] / (FS / 2), CARDIAC[1] / (FS / 2)], btype="band")
    n_raw = n_filtered = 0
    for seed in range(20):
        x = coloured(1.0, seed)
        f, p = sig.welch(sig.detrend(x), FS, nperseg=int(FS * 60))
        n_raw += not sp.is_band_floor(f, p, CARDIAC)
        xf = sig.filtfilt(b, a, x - x.mean())
        ff, pf = sig.welch(xf, FS, nperseg=int(FS * 60))
        n_filtered += not sp.is_band_floor(ff, pf, CARDIAC)
    assert n_raw <= 2, f"the rule called a slope a peak on {n_raw}/20 raw spectra"
    assert n_filtered >= 12, f"the filtered spectrum fooled it only {n_filtered}/20 times"


def test_respiration_rate_at_its_own_defaults_reproduces_the_historical_failure():
    """The regime the check has to work in, and the one where the raw-segment choice earns itself.

    At the function's own defaults -- a 0.1-0.6 Hz band and 30 s windows -- a 1/f series with no
    breathing in it returns 8 to 11 breaths a minute, which is where four analyses in this corpus
    were reading respiration rates from, one of them at a median of 9.0 and one into a book. It
    now warns on every one of them and stays quiet on a real 0.25 Hz breath.

    This is a narrow band on short windows, which is the regime the check is weakest in, so it is
    tested here as well as at the wide cardiac band above.

    Verified able to fail: dropping the `n_floor` count fails the first half and warning
    unconditionally fails the second. Making `_floor_spectrum` band-pass its input to the
    same band does NOT fail this, and that was measured rather than assumed: per window the
    filtered spectrum fools the rule far more often, but one flagged window out of twenty is
    enough for the warning, so the verdict survives. The raw segment is the more sensitive
    choice, not the difference between warning and silence.
    """
    t = np.arange(0, DUR, 1 / FS)
    for seed in range(6):
        rate, w = caught(physio.respiration_rate, coloured(1.0, seed), FS)
        assert 6.0 < rate["median_bpm"] < 13.0, "the historical artefact is not being reproduced"
        assert w, "returned a plausible breathing rate from a slope without saying so"
    for seed in range(3):
        breath = coloured(1.0, seed) + 1.5 * np.sin(2 * np.pi * 0.25 * t)
        rate, w = caught(physio.respiration_rate, breath, FS)
        assert 12.0 < rate["median_bpm"] < 18.0
        assert not w


def test_first_ami_minimum_says_when_it_only_reached_its_own_maxlag():
    """The same failure wearing a lag instead of a frequency.

    Verified able to fail: deleting the warn call fails the first half; warning unconditionally
    fails the second.
    """
    _, w = caught(dy.first_ami_minimum, coloured(2.0, 0), maxlag=100)
    assert w and "no local minimum" in str(w[0].message)

    t = np.arange(0, 200, 0.01)
    _, w = caught(dy.first_ami_minimum, np.sin(2 * np.pi * 1.0 * t), maxlag=100)
    assert not w


# --------------------------------------------------------- the sweep

def test_the_sweep_flags_an_estimate_that_is_its_own_band_edge():
    """The generalisable test: move the boundary and see whether the answer follows.

    Verified able to fail: replacing the estimator with one returning a constant 1.2 Hz makes
    `follows` False and fails this; so does hardcoding `follows = False`.
    """
    for seed in range(6):
        r = mm.band_edge_sweep(coloured(1.0, seed), FS, CARDIAC)
        assert r["follows"] is True, f"seed {seed} was not flagged"
        assert 0.9 < r["factor"] < 1.6
        assert r["rss_edge"] < r["rss_constant"]
        # the ratio is flat, which is what "proportional to the edge" means
        assert np.nanstd(r["ratio"]) < 0.1


def test_the_sweep_stays_quiet_on_a_rhythm_that_is_really_there():
    """It must not cry wolf, or the four finders above will be re-flagged forever.

    Verified able to fail: hardcoding `follows = True` fails this; so does dropping the constant
    fit and thresholding the slope of freq against edge, which is 0 here and would need a
    threshold this test does not supply.
    """
    for seed in range(6):
        r = mm.band_edge_sweep(with_tone(1.2, seed=seed), FS, CARDIAC)
        assert r["follows"] is False, f"seed {seed} was wrongly flagged"
        assert np.allclose(r["freq"], 1.2, atol=0.05)          # the same answer at every edge
        assert r["rss_constant"] < r["rss_edge"]


def test_the_sweep_audits_a_collection_against_a_reference():
    """The shape of the real case: many recordings, one independent measurement each.

    Verified able to fail: giving the follower collection the tone estimator raises `r_max` above
    0.9 and fails the second block.
    """
    ref = np.array([0.9 + 0.03 * s for s in range(20)])
    real = [with_tone(f0, seed=s) for s, f0 in enumerate(ref)]
    r = mm.band_edge_sweep(real, FS, CARDIAC, reference=ref)
    assert r["follows"] is False
    assert r["r_max"] > 0.9

    junk = [coloured(1.0, s) for s in range(20)]
    r = mm.band_edge_sweep(junk, FS, CARDIAC, reference=ref)
    assert r["follows"] is True
    assert r["r_max"] < 0.4, "a 1/f series should carry no information about the reference"


def test_the_sweep_takes_any_estimator_including_one_from_outside_this_package():
    """The case it was written from was a video pipeline, not a function in here.

    Verified able to fail: making the fake pipeline return a constant makes `follows` False.
    """
    def pipeline(item, fs, band):
        """A caricature of the rPPG estimator: 1.28 times whatever edge it is given."""
        return 1.28 * band[0] + 0.001 * item["day"]

    days = [{"day": d} for d in range(1, 5)]
    r = mm.band_edge_sweep(days, 25.0, (0.7, 4.0), estimator=pipeline)
    assert r["follows"] is True
    assert r["factor"] == pytest.approx(1.28, abs=0.01)


def test_the_sweep_refuses_a_band_or_an_edge_range_it_cannot_use():
    """Verified able to fail: dropping either guard turns both raises into silent nonsense."""
    with pytest.raises(ValueError):
        mm.band_edge_sweep(coloured(1.0), FS, (2.2, 0.7))
    with pytest.raises(ValueError):
        mm.band_edge_sweep(coloured(1.0), FS, CARDIAC, edges=[0.5, 0.6])          # too few
    with pytest.raises(ValueError):
        mm.band_edge_sweep(coloured(1.0), FS, CARDIAC, edges=[0.5, 0.9, 2.5])     # above hi
    with pytest.raises(ValueError):
        mm.band_edge_sweep([coloured(1.0, 0), coloured(1.0, 1)], FS, CARDIAC, reference=[1.0])


def test_an_edge_pushed_above_the_rhythm_makes_everything_follow_it():
    """The stated limit of the method, tested rather than left as advice.

    A real 1.2 Hz rhythm swept with edges from 1.4 to 2.0 Hz is out of band at every one of them,
    so the answer follows the edge and the verdict is True. That is the estimator behaving
    correctly, and it is why the swept edges must stay below the frequency being looked for.

    Verified able to fail: sweeping the same signal below 1.2 Hz returns False, which is the
    assertion this one inverts.
    """
    x = with_tone(1.2, seed=0)
    assert mm.band_edge_sweep(x, FS, CARDIAC, edges=[1.4, 1.6, 1.8, 2.0])["follows"] is True
    assert mm.band_edge_sweep(x, FS, CARDIAC, edges=[0.5, 0.7, 0.9, 1.1])["follows"] is False
