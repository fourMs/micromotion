"""Spectral helpers: finding the physiological peaks in a motion signal.

A body-worn accelerometer on someone standing still picks up two rhythms that are not
movement in the intentional sense. Respiration sits at roughly 0.2-0.4 Hz and the
ballistocardiac impulse, the recoil of the heart ejecting blood, at roughly 0.8-1.8 Hz.
Both are inside or adjacent to the micromotion band, so isolating postural motion means
locating them first.
"""

from __future__ import annotations

import warnings

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
    """Integrated power between two frequencies.

    Trapezoid quadrature over a closed interval, ``[lo, hi]``. A ratio of two of these is not
    the same number as a ratio computed by summing bins over ``[lo, hi)``, which is what
    :func:`~micromotion.physio.spectral_band_fractions` does; :func:`band_share` names the
    convention it uses and can express either.
    """
    x = np.asarray(x, float)
    nper = int(min(len(x), fs * window_s))
    f, p = signal.welch(signal.detrend(x), fs, nperseg=nper)
    m = (f >= band[0]) & (f <= band[1])
    return float(np.trapezoid(p[m], f[m])) if m.sum() > 1 else float("nan")


def peak_from_spectrum(f, p, band: tuple[float, float], require_peak: bool = True,
                       min_excess: float = 2.0) -> dict:
    """The peak rule, for a caller that already has a spectrum.

    `spectral_peak` is this with a Welch in front of it. It exists separately because several
    analyses compute one spectrum and read three bands off it, and recomputing the transform per
    band to get at the rule would be both wasteful and an invitation to reimplement it locally.
    A rule that is easier to copy than to import gets copied; that is how the reference markers
    got into one of these analyses twice.

    See `spectral_peak` for what the rule is and why.
    """
    f = np.asarray(f, float)
    p = np.asarray(p, float)
    nan = {"freq": float("nan"), "power": float("nan"), "snr": float("nan"),
           "excess": float("nan"), "is_peak": False}
    m = (f >= band[0]) & (f <= band[1]) & np.isfinite(p)
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
    if not (0 < k < len(pb) - 1 and ratio[k] > ratio[k - 1] and ratio[k] > ratio[k + 1]):
        return nan
    if not ratio[k] >= min_excess:
        return nan
    return result(k, ratio[k], True)


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
    x = np.asarray(x, float)
    if len(x) < fs * 10:
        return {"freq": float("nan"), "power": float("nan"), "snr": float("nan"),
                "excess": float("nan"), "is_peak": False}
    nper = int(min(len(x), fs * window_s))
    f, p = signal.welch(signal.detrend(x), fs, nperseg=nper)
    return peak_from_spectrum(f, p, band, require_peak=require_peak, min_excess=min_excess)


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
    the chest-phone cardiac share that motivated the compensated quantity-of-motion variant
    was established. The number that finding long circulated as -- 38 per cent -- did not
    survive re-measurement under stated bands; the figure that reproduces is 25 per cent of
    0.2-5 Hz acceleration power at 0.8-2.5 Hz, on the raw accelerometer channel. That history
    is why :func:`band_share` exists and makes both bands mandatory; prefer it whenever the
    result is going to be quoted.

    WHICH CONVENTION THIS IS. Trapezoid quadrature over a closed interval, ``[lo, hi]``, with the
    whole spectrum as the denominator rather than a named band -- the same arithmetic as
    :func:`band_power` and as :func:`band_share` at its defaults, and NOT the same arithmetic as
    :func:`~micromotion.physio.spectral_band_fractions`, which sums bins over ``[lo, hi)`` and
    divides by a named band. The two are not interchangeable: on chest-accelerometer standstill
    recordings the quadrature rule alone moves a respiratory share by up to 0.034 absolute on a
    share of about 0.16, and the closure by up to 0.025. Do not compare a fraction from here with
    one from there; :func:`band_share` can express either, and says which it used.
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


def _share_bands_checked(num_band: tuple[float, float],
                         den_band: tuple[float, float]) -> None:
    for name, (lo, hi) in (("num_band", num_band), ("den_band", den_band)):
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo < 0 or hi <= lo:
            raise ValueError(f"{name} {lo}-{hi} Hz is not a band; need 0 <= lo < hi")
    if num_band[0] < den_band[0] or num_band[1] > den_band[1]:
        raise ValueError(
            f"numerator band {num_band[0]}-{num_band[1]} Hz reaches outside the denominator "
            f"band {den_band[0]}-{den_band[1]} Hz. A 'share' of power the denominator does not "
            "contain can exceed 1 and is a ratio, not a share; if a ratio of two disjoint or "
            "overlapping bands is what you mean, compute band_power() twice and divide, and "
            "report it as a ratio."
        )


SHARE_RULES = ("trapezoid", "sum")
"""The quadrature rules a share can be taken under. See :func:`band_share`."""

SHARE_INTERVALS = ("closed", "half_open")
"""The band-edge conventions a share can be taken under. See :func:`band_share`."""


def _share_rule_checked(integrate: str, interval: str) -> None:
    if integrate not in SHARE_RULES:
        raise ValueError(
            f"unknown integration rule {integrate!r}; use 'trapezoid' or 'sum'")
    if interval not in SHARE_INTERVALS:
        raise ValueError(
            f"unknown interval {interval!r}; use 'closed' or 'half_open'")


def _band_mask(f, band: tuple[float, float], interval: str):
    lo, hi = band
    return (f >= lo) & (f <= hi) if interval == "closed" else (f >= lo) & (f < hi)


def _share(f, p, num_band, den_band, integrate: str = "trapezoid",
           interval: str = "closed") -> float:
    mn = _band_mask(f, num_band, interval)
    md = _band_mask(f, den_band, interval)
    if mn.sum() < 2 or md.sum() < 2:
        return float("nan")
    if integrate == "sum":
        num, den = float(p[mn].sum()), float(p[md].sum())
    else:
        num = float(np.trapezoid(p[mn], f[mn]))
        den = float(np.trapezoid(p[md], f[md]))
    if not np.isfinite(den) or den <= 0:
        return float("nan")
    return float(num / den)


def band_share_from_spectrum(f, p, *, num_band: tuple[float, float],
                             den_band: tuple[float, float],
                             integrate: str = "trapezoid",
                             interval: str = "closed") -> float:
    """The share rule, for a caller that already has a spectrum.

    :func:`band_share` is this with a Welch in front of it, and its docstring states the rule and
    the incident behind it. The split mirrors :func:`peak_from_spectrum`: several analyses compute
    one spectrum and read more than one share off it, and a rule that is easier to copy than to
    import gets copied.

    The deliverability check here is against the spectrum itself: a denominator edge above the
    highest frequency the spectrum reaches means the share is taken over a truncated denominator,
    and the warning says so. Everything else -- the mandatory bands, the containment rule, the
    non-finite warning, and both convention parameters -- is the same.

    ``integrate`` is ``"trapezoid"`` or ``"sum"`` and ``interval`` is ``"closed"`` or
    ``"half_open"``, exactly as in :func:`band_share`, which states what the four combinations
    mean and what they cost. ``integrate="sum", interval="half_open"`` is the convention
    :func:`~micromotion.physio.spectral_band_fractions` implements, and on the same spectrum the
    two then agree to floating point; the defaults here are the convention
    :func:`band_power`, :func:`band_power_fraction` and :func:`band_share` use.
    """
    _share_bands_checked(num_band, den_band)
    _share_rule_checked(integrate, interval)
    f = np.asarray(f, float)
    p = np.asarray(p, float)
    n_bad = int((~np.isfinite(p)).sum())
    if n_bad:
        warnings.warn(
            f"band_share_from_spectrum() was given {n_bad} non-finite spectrum value(s); "
            "the integrals will be NaN. A Welch spectrum of a series with any non-finite "
            "sample is entirely NaN -- clean the series before the transform.",
            RuntimeWarning, stacklevel=2)
    top = float(f[np.isfinite(f)].max()) if np.isfinite(f).any() else float("nan")
    if den_band[1] > top:
        warnings.warn(
            f"denominator band reaches {den_band[1]} Hz but the spectrum ends at {top:.4g} Hz. "
            f"The result is a share over {den_band[0]}-{top:.4g} Hz, not "
            f"{den_band[0]}-{den_band[1]} Hz, and is not comparable with a share computed over "
            "the full denominator.",
            RuntimeWarning, stacklevel=2)
    return _share(f, p, num_band, den_band, integrate, interval)


def band_share(x, fs: float, *, num_band: tuple[float, float],
               den_band: tuple[float, float], window_s: float = 60.0,
               integrate: str = "trapezoid", interval: str = "closed") -> float:
    """Fraction of spectral power in one band over another. Both bands are mandatory.

    Returns numerator-band power over denominator-band power, integrated on a Welch spectrum,
    as a number in 0 to 1. The numerator band must lie inside the denominator band, and neither
    has a default.

    THE TWO BANDS ARE NOT THE WHOLE LABEL: THE ARITHMETIC IS PART OF IT. Two conventions for
    turning a Welch spectrum into a band power are in use in this field, and this package used
    both before it said so. ``integrate`` selects the quadrature rule -- ``"trapezoid"``, which
    weights the two edge bins by a half, or ``"sum"``, which adds the bins in the mask (the
    rectangle rule; the bin width cancels in a ratio, so a bin-summed fraction is exactly this).
    ``interval`` selects what the band edges mean -- ``"closed"``, ``[lo, hi]``, or
    ``"half_open"``, ``[lo, hi)``. The defaults, ``"trapezoid"`` and ``"closed"``, are what
    :func:`band_power`, :func:`band_power_fraction` and this function have always computed, and
    what the 25 per cent chest-phone cardiac share was measured under.
    ``integrate="sum", interval="half_open"`` is the other convention in this package,
    implemented by :func:`~micromotion.physio.spectral_band_fractions`, and it is what the
    older cardiac and respiratory composition figures for chest-accelerometer standstill were
    taken under; pass it to reproduce those rather than silently restating them.

    BOTH CHOICES MOVE REAL SHARES BY MORE THAN THEY LOOK LIKE THEY SHOULD. Measured on
    chest-accelerometer standstill recordings with the mask held fixed so that only the
    quadrature rule changed, the respiratory share moved by up to 0.034 absolute on a share of
    about 0.16 -- over a fifth of the value. Closure costs up to 0.025 absolute on the same
    recordings, and it costs that much because analysts choose round band edges: at the
    conventional 60 s window the bin spacing is 1/60 Hz, so 0.40, 0.70, 2.20, 3.0, 5.0 and
    8.0 Hz all land exactly on a bin, and closing the interval adds a whole bin at the
    numerator's upper edge and at the denominator's, which do not cancel. Four analysis scripts
    in one corpus were carried from the bin-sum convention onto the defaults here, so that every
    share in that corpus is taken the same way and each is comparable with the rest. The move
    republished one share from 58 to 59 per cent, another from 16.6 to 15.1, a fold from 3.1 to
    3.2, and 18 of the 24 numbers in one table. That is a re-measurement rather than a refactor,
    and it is why these parameters exist: a convention has to be nameable before a corpus can
    decide to hold one.

    THE TWO AGREE EXACTLY ON A FLAT BAND, which is why they look interchangeable. Where the
    spectrum is flat, the trapezoid's half-weighted end bins remove precisely one bin's worth,
    so ``"trapezoid", "closed"`` equals ``"sum", "half_open"`` to floating point. The
    disagreement is driven by the slope inside the band, so it is largest at the low end of a
    red spectrum -- the respiratory band, on every body-worn sensor here.

    A share is comparable only with a share taken the same way. Record the rule and the closure
    beside the two bands whenever the number is going to be quoted.

    WHY THERE ARE NO DEFAULTS. A share is a ratio of two integrals and it moves when either band
    moves. Four published-looking figures for the share of standstill motion -- 38, 43, 45 and 58
    per cent -- circulated in one project and were quoted against one another as though they
    measured the same thing. Traced to their origins, each came from a hand-rolled fraction with a
    different, sometimes unstated, denominator; the 58 traces only to its own 0.10-3.0 Hz
    denominator, and the 45 is untraceable to any measurement at all. A share whose two bands are
    not stated beside it is not a reportable number. Report the domain (power of what quantity),
    the site (sensor and placement) and both bands, every time: acceleration power and position
    power weight the spectrum by a factor of frequency to the fourth relative to each other, so a
    share of one is not even approximately a share of the other from the same sensor on the same
    body.

    WHAT IT REFUSES AND WHAT IT WARNS ABOUT. A numerator band wider than or outside the
    denominator raises, because the result would not be a share. A denominator edge above what
    ``fs`` can deliver warns, the way :func:`~micromotion.filters.bandpass` warns, because the
    share silently becomes one over a narrower band -- and ``fs`` must be the channel's own rate,
    not the file's row rate or a resampled grid's; measure it with
    :func:`~micromotion.io.channel_rate` on an interleaved log. A non-finite input warns, as the
    filters have since 1.7.0: one NaN makes the whole Welch spectrum NaN, so the share is NaN
    rather than mostly right.

    IF TWO CHANNELS MUST MEET AT ONE RATE FIRST, resample with
    :func:`~micromotion.resample.to_rate`, which is an anti-aliased polyphase FIR resampler and
    refuses to upsample. Plain interpolation onto a slower clock is not a resampler: measured in
    the vest decomposition work, interpolating 256 Hz accelerometer axes to a belt's 25.6 Hz
    folded high-frequency sensor noise into the cardiac band, which inflates exactly the kind of
    share this function computes.

    For a spectrum already in hand, :func:`band_share_from_spectrum` applies the same rule
    without recomputing the transform. For several named bands over one common total,
    :func:`band_power_fraction` remains the convenience -- it computes the default convention
    here, over the whole spectrum rather than a named denominator. For the bin-sum convention
    with a named denominator band, :func:`~micromotion.physio.spectral_band_fractions` is where
    it already lives. This function is the one whose result is meant to be quoted, which is why
    it makes the denominator explicit and lets the convention be named rather than assumed.
    """
    _share_bands_checked(num_band, den_band)
    _share_rule_checked(integrate, interval)
    x = np.asarray(x, float)
    from .filters import NYQUIST_MARGIN

    deliverable = fs / 2.0 * NYQUIST_MARGIN
    if den_band[1] > deliverable:
        warnings.warn(
            f"denominator band reaches {den_band[1]} Hz but {fs} Hz sampling delivers only "
            f"{deliverable:.4g} Hz. The result is a share over a truncated denominator and is "
            "not comparable with one computed at a rate that carries the full band. Note also "
            "that fs must be the channel's own rate, not the file's row rate or a resampled "
            "grid's -- see channel_rate().",
            RuntimeWarning, stacklevel=2)
    n_bad = int((~np.isfinite(x)).sum())
    if n_bad:
        warnings.warn(
            f"band_share() was given {n_bad} non-finite sample(s); a spectrum of a series with "
            "any non-finite sample is entirely NaN, so the share is NaN rather than mostly "
            "right. Interpolate short gaps or split the series before calling.",
            RuntimeWarning, stacklevel=2)
        return float("nan")
    nper = int(min(len(x), fs * window_s))
    f, p = signal.welch(signal.detrend(x), fs, nperseg=nper)
    return _share(f, p, num_band, den_band, integrate, interval)


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
