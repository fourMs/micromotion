"""Aligning recordings that share no clock.

The recurring problem in this corpus: two instruments recorded the same person at the same
time and neither knows what time it was. A phone counts seconds from when its app started, an
fNIRS headband from when its acquisition began, an audio recorder from when someone pressed
the button. Putting them on one timeline means recovering the offset from the signals
themselves.

Two methods, for two situations. When both signals carry the same physiological rhythm, track
that rhythm in each and cross-correlate the resulting curves: :func:`instantaneous_rate` then
:func:`xcorr_lag`. When the signals are of different kinds, different lengths, or sampled far
apart, search integer offsets directly with :func:`search_lag`, which tolerates all three.

Both report how confident they are, and neither should be used without looking. A
cross-correlation always has a maximum; that the maximum means anything is a separate claim.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as _signal

from .spectral import CARDIAC_BAND, _floor_spectrum, _warn_band_floor, is_band_floor


def instantaneous_rate(x, fs: float, band: tuple[float, float] = CARDIAC_BAND,
                       win_s: float = 20.0, step_s: float = 1.0,
                       per_minute: bool = True):
    """Track a rhythm's frequency over time, by sliding a Welch estimate along it.

    Returns ``(times, rate)``. With ``per_minute`` the rate is in beats or breaths per
    minute; otherwise in Hz.

    This turns a raw signal into something two instruments can be compared on even when they
    measure entirely different quantities. A haemoglobin trace and a chest accelerometer have
    no common units, but both carry a heartbeat, and the way that heart rate wanders over ten
    minutes is a signature specific enough to align them.

    Each window is a bare maximum inside the band, and it stays one, because the values are what
    published alignments were computed from. What is new is that the function counts the windows
    whose band held no peak at all, by :func:`~micromotion.spectral.is_band_floor`, and warns once
    if any did. On synthetic 1/f series with nothing in the cardiac band every window returns the
    band's lowest bin, so the track is a flat line sitting on the edge — and two such flat lines
    from two instruments will cross-correlate with each other perfectly while carrying nothing.
    Where a NaN per window is preferable to a number, loop over
    :func:`~micromotion.spectral.spectral_peak` instead; where a track already exists,
    :func:`~micromotion.spectral.band_edge_sweep` settles whether it is the band edge.
    """
    x = np.asarray(x, float)
    n, s = int(win_s * fs), int(step_s * fs)
    if n <= 0 or s <= 0 or len(x) < n:
        return np.array([]), np.array([])
    times, out = [], []
    n_floor = 0
    nper = int(min(n, fs * 10))
    for i in range(0, len(x) - n, s):
        seg = _signal.detrend(x[i:i + n])
        f, p = _signal.welch(seg, fs, nperseg=nper)
        m = (f >= band[0]) & (f <= band[1])
        if m.any():
            out.append(f[m][np.argmax(p[m])])
            times.append((i + n / 2) / fs)
            n_floor += is_band_floor(*_floor_spectrum(x[i:i + n], fs, band), band)
    if n_floor:
        _warn_band_floor("instantaneous_rate()", float(np.median(out)), band, n_floor, len(out))
    rate = np.array(out) * (60.0 if per_minute else 1.0)
    return np.array(times), rate


def xcorr_lag(a, b, fs: float = 1.0, max_lag_s: float | None = None,
              min_r: float = 0.5, difference: bool = True) -> dict:
    """Offset between two equally sampled signals, by cross-correlation.

    Returns the lag in seconds and samples, the correlation at that lag, and ``confident``,
    which is whether that correlation reaches ``min_r``. A positive lag means ``b`` starts
    later than ``a``.

    That sentence was false until 1.13.0. The function returned the negative of what it
    documented, and the test asserted the value the code produced rather than the value the
    docstring promised, so the two agreed with each other and neither agreed with the
    convention. A sign is exactly the kind of error a test written after the fact preserves.
    ``musicalgestures.xcorr_lag`` documented and returned the opposite -- the correct --
    sign throughout, which is how this was found.

    Both series are differenced first, and that default is load-bearing rather than a
    stylistic choice. Correlating two drifting series and taking the best lag is the classic
    spurious-regression trap: over 200 pairs of independent random walks, the best-lag
    correlation had a median of 0.47, exceeded 0.5 forty per cent of the time, and reached
    0.98 at worst. No threshold can survive that. Differencing the same pairs gave a median
    of 0.11 and never exceeded 0.16. Differencing is shift-equivariant, so the lag itself is
    unaffected — it removes the drift, not the alignment. Pass ``difference=False`` only if
    the inputs are already stationary and you know it.

    The confidence test is a threshold on the correlation and deliberately not a measure of
    how sharply the peak stands out from the rest of the curve. Two such measures were tried
    and both get it backwards: a drifting series correlates highly with itself at every
    nearby lag, so its true peak looks unremarkable against the curve, while white noise
    gives a flat curve whose maximum looks strikingly sharp. A genuine alignment scored 0.98
    and pure noise 0.07, but the noise had the higher peak-to-background ratio of the two.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if difference and len(a) > 1 and len(b) > 1:
        a, b = np.diff(a), np.diff(b)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)

    # correlate(b, a), not correlate(a, b): the lag wanted is b's relative to a, so b is the
    # series being slid. Reversing these two arguments is the whole of the sign error above.
    c = _signal.correlate(b, a, mode="full") / min(len(a), len(b))
    lags = _signal.correlation_lags(len(b), len(a), mode="full")
    if max_lag_s is not None:
        keep = np.abs(lags) <= int(max_lag_s * fs)
        c, lags = c[keep], lags[keep]
    if not len(c):
        return {"lag_s": float("nan"), "r": float("nan"), "confident": False}

    # Among near-tied maxima, take the SMALLEST lag. A periodic envelope correlates almost
    # as well at plus or minus one period as at the true offset, and which of those wins is
    # then floating-point noise: the answer jumps a whole period between two runs that differ
    # in the last bit. Adopted from musicalgestures.xcorr_lag, which had it first.
    tied = np.flatnonzero(c >= c.max() - 1e-9)
    k = int(tied[np.argmin(np.abs(lags[tied]))])
    peak = float(c[k])
    return {
        "lag_samples": int(lags[k]),
        "lag_s": float(lags[k]) / fs,
        "r": peak,
        "confident": bool(peak >= min_r),
    }


def search_lag(t_a, x_a, t_b, x_b, max_lag_s: float = 300.0, step_s: float = 1.0,
               min_overlap_s: float = 120.0, min_r: float = 0.5) -> dict:
    """Offset between two irregular or unequal-length signals, by direct search.

    Both series are put on a common grid, then every integer offset within ``max_lag_s`` is
    scored by Pearson correlation over whatever the two share at that offset. Slower than
    :func:`xcorr_lag` and far more tolerant: the series may differ in length, in sampling,
    and in how much of the session they cover.

    A positive lag means ``x_b`` starts later than ``x_a``, the same convention as
    :func:`xcorr_lag`. This function stated no convention at all until 1.13.0 and returned
    the negative of that one; both were corrected together, since two alignment functions in
    one module disagreeing about a sign is worse than either being wrong alone.

    ``confident`` is True only when the best correlation reaches ``min_r`` and the overlap
    reaches ``min_overlap_s``. Recordings that fail the test are better left without an offset
    than given one nobody can trust: a wrong alignment is harder to detect downstream than a
    missing one.
    """
    ga = np.arange(np.min(t_a), np.max(t_a), step_s)
    gb = np.arange(np.min(t_b), np.max(t_b), step_s)
    a = np.interp(ga, t_a, x_a)
    b = np.interp(gb, t_b, x_b)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)

    # The two grids start at different absolute times, so an offset in array index is not
    # the offset in seconds. Carrying the origins back in is the whole point: without it the
    # function silently reports every alignment as zero, whatever the true offset.
    origin = float(ga[0] - gb[0])

    n_min = int(min_overlap_s / step_s)
    k_max = int(max_lag_s / step_s)
    best = {"lag_s": float("nan"), "r": -np.inf, "n_overlap": 0, "confident": False}
    for k in range(-k_max, k_max + 1):
        if k >= 0:
            aa, bb = a[k:], b[: len(a) - k]
        else:
            aa, bb = a[: len(b) + k], b[-k:]
        n = min(len(aa), len(bb))
        if n < n_min:
            continue
        r = float(np.corrcoef(aa[:n], bb[:n])[0, 1])
        if np.isfinite(r) and r > best["r"]:
            best = {"lag_s": -(origin + k * step_s), "r": r, "n_overlap": n,
                    "confident": False}
    if not np.isfinite(best["r"]):
        return {"lag_s": float("nan"), "r": float("nan"), "n_overlap": 0,
                "confident": False}
    best["confident"] = bool(best["r"] >= min_r and best["n_overlap"] >= n_min)
    return best


def find_transient(x, fs: float, threshold: float = 8.0, search_s: float | None = None,
                   min_separation_s: float = 1.0):
    """Locate impulsive events: a hand clap, a tap on a sensor, a heel strike.

    Returns their times in seconds. ``threshold`` is in robust standard deviations above the
    median of the signal envelope, using the median absolute deviation so that the events
    themselves do not inflate the scale they are measured against.

    Where sessions open and close with a synchronisation clap, the usual handling is to trim a
    fixed window from each end. Detecting it instead gives the offset rather than discarding it,
    which is what turns a clap into an
    alignment rather than a nuisance.
    """
    x = np.asarray(x, float)
    if x.ndim > 1:
        x = np.linalg.norm(x, axis=1)
    env = np.abs(_signal.hilbert(x - np.median(x)))

    if search_s is not None:
        n = int(search_s * fs)
        mask = np.zeros(len(env), bool)
        mask[:n] = True
        mask[-n:] = True
    else:
        mask = np.ones(len(env), bool)

    med = np.median(env)
    mad = np.median(np.abs(env - med)) * 1.4826
    if mad <= 0:
        return np.array([])
    peaks, _ = _signal.find_peaks(
        np.where(mask, env, -np.inf),
        height=med + threshold * mad,
        distance=max(1, int(min_separation_s * fs)),
    )
    return peaks / fs


def apply_lag(t, lag_s: float):
    """Shift a timebase onto another recording's origin."""
    return np.asarray(t, float) + lag_s
