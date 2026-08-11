"""Spectral helpers: finding the physiological peaks in a motion signal.

A body-worn accelerometer on someone standing still picks up two rhythms that are not
movement in the intentional sense. Respiration sits at roughly 0.2-0.4 Hz and the
ballistocardiac impulse, the recoil of the heart ejecting blood, at roughly 0.8-1.8 Hz.
Both are inside or adjacent to the micromotion band, so isolating postural motion means
locating them first.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

CARDIAC_BAND = (0.7, 2.2)
"""Hz. 42-132 bpm, which covers rest through mild exertion."""

RESPIRATORY_BAND = (0.1, 0.5)
"""Hz. 6-30 breaths per minute."""


def _peak(x, fs: float, band: tuple[float, float], window_s: float) -> float:
    x = np.asarray(x, float)
    if len(x) < fs * 10:
        return float("nan")
    nper = int(min(len(x), fs * window_s))
    f, p = signal.welch(signal.detrend(x), fs, nperseg=nper)
    m = (f >= band[0]) & (f <= band[1])
    return float(f[m][np.argmax(p[m])]) if m.any() else float("nan")


def cardiac_peak(x, fs: float, window_s: float = 60.0) -> float:
    """Dominant frequency in the cardiac band, in Hz.

    Pass the acceleration magnitude. Multiply by 60 for beats per minute. Returns NaN if
    the recording is too short for the band to be resolved.
    """
    return _peak(x, fs, CARDIAC_BAND, window_s)


def respiratory_peak(x, fs: float, window_s: float = 120.0) -> float:
    """Dominant respiratory frequency, in Hz. Multiply by 60 for breaths per minute.

    This is measured in the time domain, from :func:`detect_breaths`, and NOT as a peak in a
    periodogram. It used to be the latter and the change corrects a wrong number rather than
    trading one convention for another.

    Why, in short: a periodogram of belt or body motion is red, so the breathing bump sits on a
    much larger downward slope and never becomes the global maximum inside the band. Measured on
    the one dataset in this corpus with a ground truth -- Stillness2025's sixteen thoracic belts at
    25.6 Hz -- the old version returned a median 7.5 breaths per minute where the belts' own breath
    timing gives 16.8 and where a resting adult breathes 12-20. It was not a calibration offset: it
    ranked those sixteen participants at Spearman -0.32 against their own breath timing, so it
    carried no usable information about who was breathing faster.

    Four repairs were measured and all four rejected, which is why the spectral approach was
    abandoned rather than patched. Raising the band floor to 0.20 Hz leaves a 3.4 breaths-per-minute
    gap. Band-passing before the periodogram changes nothing whatever, and cannot, because the
    maximum inside a band is unaffected by filtering inside that same band. The most prominent local
    maximum instead of the global one reaches Spearman +0.22. Dividing out a fitted power law before
    taking the maximum reaches +0.26 and biases the median high, to 21.5.

    :func:`cardiac_peak` still uses the periodogram and is right to: its band sits above the slope
    and the ballistocardiac impulse is a genuinely prominent peak, giving a median 75 bpm with an
    interquartile range of 70-82 on a year of chest-phone data.

    ``window_s`` is accepted for backward compatibility and is unused; breath detection does not
    need a spectral window.
    """
    rate = detect_breaths(x, fs)["rate_per_min"]
    return float(rate / 60.0) if np.isfinite(rate) else float("nan")


def band_power(x, fs: float, band: tuple[float, float], window_s: float = 60.0) -> float:
    """Integrated power between two frequencies."""
    x = np.asarray(x, float)
    nper = int(min(len(x), fs * window_s))
    f, p = signal.welch(signal.detrend(x), fs, nperseg=nper)
    m = (f >= band[0]) & (f <= band[1])
    return float(np.trapezoid(p[m], f[m])) if m.sum() > 1 else float("nan")


def spectral_peak(x, fs: float, band: tuple[float, float],
                  window_s: float = 60.0, require_peak: bool = True,
                  min_excess: float = 2.0) -> dict:
    """Peak frequency in a band, or NaN when the band contains no peak.

    Every spectrum has a maximum somewhere inside any band you choose, and on a falling
    spectrum that maximum is the lowest bin. It is not a rhythm, it is the slope. With
    ``require_peak`` the result is NaN unless the band holds something that stands above its own
    background: an interior local maximum of the spectrum divided by a log-log straight-line fit
    across the band, exceeding ``min_excess``. ``is_peak`` and ``excess`` in the returned dict say
    which happened and by how much.

    WHY THE BASELINE IS A FITTED SLOPE AND NOT THE BAND MEDIAN. Measuring a peak against the
    median of its own band assumes the band is flat. Over a 1/f spectrum it is not: the median is
    dragged down by the high-frequency end, so the low bins clear any threshold without being
    peaks. On plain 1/f noise the band's largest value scores 3.7 against the median and 1.3 to
    1.7 against the fitted slope, where a real rhythm scores 5 and up against either.

    THE SIGNAL-TO-NOISE RATIO DOES NOT CATCH THIS, and an earlier version of this docstring
    advised using it that way. It is wrong in the one case that matters. The lowest bin of a 1/f
    spectrum has both the most power and the highest power-over-band-median of anything inside a
    band drawn above the knee, so raising an SNR threshold SELECTS the artefact rather than
    excluding it. In a year of daily standstill recordings, tightening the threshold from nothing
    to 5 took the share of days whose "respiration rate" sat exactly on the band floor from 21 per
    cent to 32, and moved the median from 10.5 breaths a minute to 9.0. Four analyses in the Oslo
    Standstill corpus reported a band edge as a measurement before this was found, one of them on
    662 of 930 values, and one of those numbers had reached a book.

    REJECTING THE EDGE BIN IS NOT ENOUGH EITHER. On a monotone slope, refusing the first bin moves
    the maximum to the second: the same 662 of 930 became 198 of 268, one bin along. Only asking
    whether the thing is a peak at all separates the two cases.

    IT IS MORE ACCURATE AND NOT ONLY MORE CAUTIOUS. Dividing out the slope finds peaks the raw
    maximum misses: on 1/f noise plus a modest 0.25 Hz tone, the old answer is the band floor and
    this returns 0.25.

    WHAT IT COSTS. It is conservative, and a rhythm weaker than about half that returns NaN even
    though it is really there. That is the right direction to fail in -- a missing value rather
    than a wrong one -- but it is a false negative, so a rate aggregated over many recordings will
    be missing its weakest cases and the count of NaNs is part of the result rather than a
    nuisance. Lower ``min_excess`` to trade the other way, knowingly.

    ``snr`` is still the peak over the band median, unchanged, because callers report it.
    ``require_peak=False`` restores the old behaviour exactly, for a caller who wants the largest
    value in a band and knows that is what they are asking for.
    """
    nan = {"freq": float("nan"), "power": float("nan"), "snr": float("nan"),
           "excess": float("nan"), "is_peak": False}
    x = np.asarray(x, float)
    if len(x) < fs * 10:
        return nan
    nper = int(min(len(x), fs * window_s))
    f, p = signal.welch(signal.detrend(x), fs, nperseg=nper)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        return nan
    fb, pb = f[m], p[m]
    med = float(np.median(pb))

    def result(k, excess, is_peak):
        return {"freq": float(fb[k]), "power": float(pb[k]),
                "snr": float(pb[k] / med) if med > 0 else float("nan"),
                "excess": float(excess), "is_peak": bool(is_peak)}

    if not require_peak:
        return result(int(np.argmax(pb)), float("nan"), False)

    ok = (fb > 0) & (pb > 0)
    if ok.sum() < 5:
        return nan
    lf, lp = np.log(fb[ok]), np.log(pb[ok])
    slope, intercept = np.polyfit(lf, lp, 1)
    ratio = np.full(len(pb), np.nan)
    ratio[ok] = pb[ok] / np.exp(slope * lf + intercept)
    if np.all(np.isnan(ratio)):
        return nan
    k = int(np.nanargmax(ratio))
    interior = 0 < k < len(pb) - 1
    if not (interior and ratio[k] > ratio[k - 1] and ratio[k] > ratio[k + 1]):
        return nan
    if not ratio[k] >= min_excess:
        return nan
    return result(k, ratio[k], True)


def band_rms(x, fs: float, band: tuple[float, float]) -> float:
    """Root-mean-square amplitude within a band, in the input units.

    Rate-independent, unlike an integrated measure, which makes it the right thing to quote
    when comparing instruments that sample at different rates.
    """
    from .filters import bandpass

    return float(np.sqrt(np.mean(bandpass(np.asarray(x, float), fs, *band) ** 2)))


def band_power_fraction(x, fs: float, bands: dict, window_s: float = 60.0) -> dict:
    """Proportion of total power falling in each named band.

    Pass something like ``{"respiratory": (0.1, 0.5), "cardiac": (0.7, 2.2)}``. This is how
    the finding that roughly 38 per cent of the chest-phone signal power sits at the heart
    rate was established, which is in turn why the compensated quantity-of-motion variant
    exists.
    """
    x = np.asarray(x, float)
    nper = int(min(len(x), fs * window_s))
    f, p = signal.welch(signal.detrend(x), fs, nperseg=nper)
    total = float(np.trapezoid(p, f))
    out = {}
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f <= hi)
        out[name] = (float(np.trapezoid(p[m], f[m]) / total)
                     if m.sum() > 1 and total > 0 else float("nan"))
    return out


def mean_frequency(x, fs: float, band: tuple[float, float] = (0.1, 5.0),
                   window_s: float = 60.0) -> float:
    """Power-weighted mean frequency within a band, in Hz.

    A different statistic from the peak: it moves when energy shifts between frequencies even
    if the dominant one does not change, so it is the more sensitive of the two to a gradual
    change in how a person is standing.
    """
    x = np.asarray(x, float)
    nper = int(min(len(x), fs * window_s))
    f, p = signal.welch(signal.detrend(x), fs, nperseg=nper)
    m = (f >= band[0]) & (f <= band[1])
    return float(np.sum(f[m] * p[m]) / np.sum(p[m])) if p[m].sum() > 0 else float("nan")


def detect_breaths(x, fs: float, band: tuple[float, float] = RESPIRATORY_BAND,
                   prominence_sd: float = 0.4, min_period_s: float = 2.0) -> dict:
    """Breath timing from a respiration belt or a band-limited motion signal.

    Returns peak and trough times and the cycle durations between them. Peaks and troughs are
    both returned because inhale and exhale are not symmetric, and a rate alone hides that.
    """
    from .filters import bandpass

    y = bandpass(np.asarray(x, float), fs, *band)
    kw = dict(distance=max(1, int(min_period_s * fs)),
              prominence=prominence_sd * np.std(y))
    peaks, _ = signal.find_peaks(y, **kw)
    troughs, _ = signal.find_peaks(-y, **kw)
    cycles = np.diff(peaks) / fs if len(peaks) > 1 else np.array([])
    return {"peaks_s": peaks / fs, "troughs_s": troughs / fs, "cycle_s": cycles,
            "rate_per_min": float(60.0 / np.median(cycles)) if len(cycles) else float("nan"),
            "n_breaths": len(peaks)}


def detect_breaths_adaptive(x, fs: float, band: tuple[float, float] = RESPIRATORY_BAND,
                            vel_frac: float = 0.55, baseline_hz: float = 0.2) -> dict:
    """Breath detection that rejects chest movement which is not breathing.

    Peak-and-trough detection, which :func:`detect_breaths` does, counts any sufficiently
    prominent bump. On a belt worn by someone standing that includes postural sway, weight
    shifts and swallows, all of which look like small breaths.

    This asks a different question. A breath is a sustained rise in chest expansion whose
    velocity exceeds a threshold set from the signal's own positive-derivative mean, and which
    crosses an adaptive baseline -- a heavily low-passed copy of the signal rather than zero.
    A rise that never crosses that baseline did not start from an exhaled state and is
    discarded. That rejection step is what removes the sway.

    Returns inspiration and expiration onsets, cycle durations and the rate. Expiration onset
    is taken as the end of the rise, which assumes passive expiration.

    After Finn Upham's respiration work, reimplemented and used with permission.

    Do not reach for this by default, on the evidence available here. It was added on the
    expectation that rejecting non-breath rises would beat plain peak detection, and measured
    against it that expectation did not hold. On twelve HpSp respiration-belt recordings the
    two agree: median error against the spectral estimate 1.82 breaths per minute for both.
    On eight chest accelerometers, which is the case the rejection step was
    supposed to help, it is markedly worse -- median error against the same participant's belt
    10.3 breaths per minute against 3.6 for :func:`detect_breaths`, over-counting throughout.

    That may be the parameters rather than the idea; the velocity threshold is derived from a
    belt's amplitude distribution and an accelerometer's is not the same shape. It is kept
    because the approach is sound in its original setting and because someone should be able
    to tune it, not because it is currently the better detector.
    """
    from .filters import bandpass, lowpass

    x = np.asarray(x, float)
    y = bandpass(x, fs, *band)
    d = np.gradient(y, 1.0 / fs)

    pos = d[d > 0]
    if not len(pos):
        return {"inspiration_s": np.array([]), "expiration_s": np.array([]),
                "cycle_s": np.array([]), "rate_per_min": float("nan"), "n_breaths": 0}
    thresh = pos.mean() * vel_frac

    # Crossings of an adaptive baseline, not of zero: the belt drifts.
    base = lowpass(y, fs, min(baseline_hz, fs / 2 * 0.4), order=2)
    above = y > base

    rising = d > thresh
    edges = np.diff(rising.astype(int))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if len(ends) and len(starts) and ends[0] < starts[0]:
        ends = ends[1:]
    n = min(len(starts), len(ends))
    starts, ends = starts[:n], ends[:n]

    insp, expi = [], []
    for s, e in zip(starts, ends):
        # A genuine breath rises through the baseline; a sway bump does not.
        if e > s and above[s:e].any() and not above[s:e].all():
            insp.append(s / fs)
            expi.append(e / fs)
    insp, expi = np.asarray(insp), np.asarray(expi)
    cycles = np.diff(insp) if len(insp) > 1 else np.array([])
    return {
        "inspiration_s": insp, "expiration_s": expi, "cycle_s": cycles,
        "rate_per_min": float(60.0 / np.median(cycles)) if len(cycles) else float("nan"),
        "n_breaths": len(insp),
    }
