"""
Physiology signal features for standstill / micromotion studies.

Two pure numpy/scipy surfaces ported from the "still standing" study:

* :func:`respiration_rate` -- windowed breathing rate (breaths per minute)
  from a respiration waveform, via band-pass filtering and a Welch spectral
  peak per window.
* :func:`spectral_band_fractions` -- the fraction of a signal's Welch power
  falling in each of a set of caller-supplied named frequency bands. This is
  the generic "cardiorespiratory QoM" spectral-composition diagnostic with
  the heart-rate/respiration bands supplied by the caller, so the function
  carries no dependency on any particular physiological sensor. It sums bins
  over half-open bands, which is not the convention
  :mod:`~micromotion.spectral` computes shares under; its docstring says what
  the difference costs and how to ask for either.

Source: still standing study (Jensenius) -- Deichman / Equivital physiology
analyses.
"""

import warnings

import numpy as np


def respiration_rate(waveform, fs, *, band=(0.1, 0.6), window_s=30,
                     step_s=30):
    """
    Windowed respiration rate (breaths per minute) from a breathing waveform.

    Each analysis window is band-pass filtered to the respiration band and
    its dominant frequency is taken as the Welch spectral peak inside that
    band; the rate is that frequency times 60. Windows advance by ``step_s``
    seconds. The default band ``(0.1, 0.6)`` Hz corresponds to about
    6-36 breaths/min. Each window must contain at least 15 seconds of valid
    samples for spectral estimation.

    Source: still standing study (Jensenius), Deichman respiration analysis
    (``compute_qom_resp``).

    Args:
        waveform (np.ndarray): 1-D respiration/breathing waveform.
        fs (float): Sampling rate in Hz.
        band (tuple, optional): ``(low, high)`` respiration band in Hz.
            Defaults to ``(0.1, 0.6)``.
        window_s (float, optional): Window length in seconds. Defaults to 30.
        step_s (float, optional): Hop between windows in seconds. Defaults to
            30.

    Returns:
        dict: ``{"rate_bpm", "times_s", "median_bpm"}`` where ``rate_bpm`` is
            the per-window rate (breaths/min, ``nan`` for windows without a
            clear peak), ``times_s`` the window centre times in seconds, and
            ``median_bpm`` the median across valid windows.
    """
    from scipy.signal import butter, filtfilt, welch

    x = np.asarray(waveform, dtype=float)
    x = x[np.isfinite(x)]
    lo, hi = band
    nyq = fs / 2.0
    if len(x) < int(fs * window_s):
        # single short window: still attempt one estimate
        window_s = max(1.0, len(x) / fs)

    b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
    xf = filtfilt(b, a, x - x.mean())

    win = int(fs * window_s)
    step = int(fs * step_s)
    win = max(win, 1)
    step = max(step, 1)

    rates = []
    times = []
    for start in range(0, max(len(xf) - win + 1, 1), step):
        seg = xf[start:start + win]
        if len(seg) < int(fs * 15):  # need at least 15 s for spectral estimation
            rates.append(np.nan)
            times.append((start + win / 2) / fs)
            continue
        nperseg = min(len(seg), int(fs * window_s))
        f, P = welch(seg, fs, nperseg=nperseg)
        mask = (f >= lo) & (f <= hi)
        if mask.any() and P[mask].sum() > 0:
            fpk = f[mask][np.argmax(P[mask])]
            rates.append(float(fpk * 60.0))
        else:
            rates.append(np.nan)
        times.append((start + win / 2) / fs)

    rate_bpm = np.array(rates, dtype=float)
    return dict(rate_bpm=rate_bpm, times_s=np.array(times, dtype=float),
                median_bpm=float(np.nanmedian(rate_bpm))
                if np.isfinite(rate_bpm).any() else np.nan)


DEFAULT_TOTAL_BAND = (0.1, 8.0)
"""Hz. The denominator :func:`spectral_band_fractions` falls back on when none is given.

It is a fallback and not a standard. Published shares in this corpus rest on it, so it cannot
move; a caller who does not name a denominator is warned rather than quietly given this one.
"""


def spectral_band_fractions(signal, fs, bands, *, total_band=None,
                            nperseg_s=20):
    """
    Fraction of a signal's power in each of a set of named frequency bands.

    Estimates the Welch power spectrum and, for each named band in ``bands``,
    returns that band's summed power divided by the summed power in
    ``total_band``. This is the generic spectral-composition diagnostic used
    for the "cardiorespiratory QoM artifact" analysis (e.g. how much of a
    chest-accelerometer QoM signal sits in a cardiac vs a respiration band),
    with the bands supplied by the caller so there is no built-in dependence
    on a heart-rate or respiration sensor.

    WHICH CONVENTION THIS IS, AND WHAT ELSE IS IN THIS PACKAGE. Power is
    bin-summed on the Welch grid over a half-open interval, ``[lo, hi)``, at
    both the numerator and the denominator. That is one of the two spectral-share
    conventions micromotion contains, and the other one is the default
    everywhere else: :func:`~micromotion.spectral.band_power`,
    :func:`~micromotion.spectral.band_power_fraction` and
    :func:`~micromotion.spectral.band_share` integrate with the trapezoid rule
    over a closed ``[lo, hi]``. A share from here and a share from there are
    not comparable, and until 1.11.0 neither docstring said so.

    An earlier version of this docstring claimed the two "yield nearly
    identical results on the uniform frequency spacing of Welch". They do not.
    Measured on chest-accelerometer standstill recordings, holding the mask
    fixed so that only the quadrature rule changed, the respiratory share moved
    by up to 0.034 absolute on a share of about 0.16 -- over a fifth of the
    value. Interval closure costs a further 0.025 absolute on the same
    recordings, because analysts pick round band edges and at the conventional
    60 s window (1/60 Hz bins) 0.40, 0.70, 2.20, 3.0, 5.0 and 8.0 Hz all land
    exactly on a bin, so closing the interval adds a whole bin at both the
    numerator's and the denominator's upper edge and the two do not cancel. The
    two conventions agree only where the band is flat, which the low end of a
    body-worn sensor's spectrum never is.

    Nothing here is deprecated and no default has moved. This is a named
    convention with a stated arithmetic, and the standstill composition figures
    the corpus carried before it settled on the trapezoid-and-closed rule were
    computed by exactly it, so this is where they reproduce. To compute a share
    under the other convention, or to state
    which convention a number was taken under, call
    ``band_share(..., integrate="sum", interval="half_open")``, which reproduces
    this function on the same spectrum, or leave those parameters at their
    defaults to get the trapezoid-and-closed convention. New work that will
    quote a number should prefer :func:`~micromotion.spectral.band_share`: it
    makes the denominator mandatory and the convention explicit.

    ``total_band`` has no silent default. Left unset it falls back to
    :data:`DEFAULT_TOTAL_BAND`, 0.1-8.0 Hz, and warns, because a share is a
    ratio of two integrals and a denominator nobody wrote down is the failure
    that put four incompatible standstill shares -- 38, 43, 45 and 58 per cent
    -- into one project's writing. Passing the band explicitly, including
    passing ``(0.1, 8.0)``, silences the warning and changes no number.

    Source: still standing study (Jensenius), Deichman chest-QoM
    cardiorespiratory spectral-composition analysis (``deichman_full``).

    Args:
        signal (np.ndarray): 1-D input signal.
        fs (float): Sampling rate in Hz.
        bands (dict): Mapping of band name to ``(low, high)`` in Hz, e.g.
            ``{"cardiac": (0.9, 1.3), "resp": (0.12, 0.5)}``.
        total_band (tuple, optional): ``(low, high)`` reference band whose
            power is the denominator. Unset falls back to
            :data:`DEFAULT_TOTAL_BAND`, ``(0.1, 8.0)``, with a warning.
        nperseg_s (float, optional): Welch segment length in seconds.
            Defaults to 20.

    Returns:
        dict: Mapping of each band name to its power fraction in ``[0, 1]``
            (``nan`` if the total band contains no power).
    """
    from scipy.signal import welch

    if total_band is None:
        total_band = DEFAULT_TOTAL_BAND
        warnings.warn(
            "spectral_band_fractions() was not given a total_band, so the fractions are over "
            f"{DEFAULT_TOTAL_BAND[0]}-{DEFAULT_TOTAL_BAND[1]} Hz. A share whose denominator is "
            "not stated beside it is not a reportable number -- four shares of standstill "
            "motion were quoted against one another in this corpus while each rested on a "
            "different, unstated denominator. Pass total_band explicitly; passing "
            f"{DEFAULT_TOTAL_BAND} changes nothing but the warning. Note also that this "
            "function sums bins over a half-open interval, which is not what band_share() "
            "computes by default; see its docstring.",
            RuntimeWarning, stacklevel=2)

    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return {name: np.nan for name in bands}
    nperseg = min(len(x), max(8, int(fs * nperseg_s)))
    f, P = welch(x - x.mean(), fs, nperseg=nperseg)
    tlo, thi = total_band
    total = P[(f >= tlo) & (f < thi)].sum()
    if total <= 0:
        return {name: np.nan for name in bands}
    out = {}
    for name, (lo, hi) in bands.items():
        out[name] = float(P[(f >= lo) & (f < hi)].sum() / total)
    return out


def _centred_diff(y: np.ndarray) -> np.ndarray:
    """First difference evaluated at the sample points rather than between them.

    ``np.diff`` returns values that belong at the midpoints, which shifts every downstream
    threshold by half a sample and drops one point. This interpolates the forward differences
    from the midpoints back onto the original grid, extrapolating at the ends -- the behaviour
    of ``respy.diffed``.
    """
    d = np.diff(y)
    mid = np.arange(len(d)) + 0.5
    return np.interp(np.arange(len(y)), mid, d,
                     left=d[0] if len(d) else 0.0, right=d[-1] if len(d) else 0.0)


def _butter_zero_phase(y: np.ndarray, fs: float, cutoff, btype: str, order: int = 2):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    wn = np.asarray(cutoff, float) / nyq
    wn = float(wn.reshape(-1)[0]) if wn.size == 1 else wn   # scipy wants a scalar for low/high
    return filtfilt(*butter(order, wn, btype=btype), y)


def respiration_onsets(x, fs: float, *, lowpass_hz: float = 1.0,
                       onset_frac: float = 0.55, baseline_hz: float = 0.2) -> dict:
    """Inspiration and expiration onset times from a chest-expansion recording.

    Inspiration onset is defined by chest-expansion *velocity* crossing a threshold rather than
    by a local minimum, which is what makes this robust on quiet standing: a belt on a standing
    body carries sway and weight shifts that produce local minima with no breath behind them.
    Expiration onset is the end of that rise, which assumes passive expiration.

    The threshold is taken from the signal's own distribution -- ``onset_frac`` times the mean
    positive velocity -- and candidate rises are then required to contain an upward crossing of
    a heavily low-passed copy of the signal, so a rise that never returns to an exhaled baseline
    is discarded.

    Zero-crossing technique after Matsuda et al. and Upham (2018). Ported from Finn Upham's
    ``respy`` (MIT, 2023) and reimplemented on numpy; see :func:`respiratory_phases` for why.

    Returns ``inspiration_s``, ``expiration_s``, the normalised signal, and its velocity.
    """
    y = np.asarray(x, float)
    if y.ndim != 1:
        raise ValueError("respiration_onsets expects a 1-D waveform")
    finite = np.isfinite(y)
    if finite.sum() < 3:
        raise ValueError("respiration_onsets needs at least three finite samples")
    if not finite.all():                       # NaNs sneak in; filtfilt would spread them
        y = np.interp(np.arange(len(y)), np.flatnonzero(finite), y[finite])
    y = y - y.mean()

    filt = _butter_zero_phase(y, fs, [lowpass_hz], "lowpass")
    scale = fs * np.median(np.abs(np.diff(filt)))
    norm = filt / scale if scale > 0 else filt
    vel = _centred_diff(norm)

    thresh = vel[vel > 0].mean() * onset_frac if (vel > 0).any() else 0.0

    # candidate rises: contiguous runs where velocity exceeds the threshold
    flat = _butter_zero_phase(norm, fs, [baseline_hz], "lowpass")
    crossings = np.diff(np.sign(norm - flat), prepend=np.nan)   # +2 marks an upward crossing
    up = np.flatnonzero(crossings == 2)

    V = np.where(vel < thresh, 0.0, vel)
    a = np.diff(np.sign(V), prepend=np.nan)
    seg_in = np.flatnonzero(a > 0.5) - 2        # respy backs the onset off two samples
    seg_out = np.flatnonzero(a < -0.5)
    seg_in, seg_out = _trim_segments(seg_in, seg_out)

    # drop rises that never cross the baseline: chest movement that is not a breath
    for lo, hi in zip(seg_in, seg_out):
        if not np.any((up >= lo) & (up <= hi)):
            V[max(lo, 0):hi + 1] = 0.0
    a = np.diff(np.sign(V), prepend=np.nan)
    seg_in, seg_out = _trim_segments(np.flatnonzero(a > 0.5), np.flatnonzero(a < -0.5))

    return {"inspiration_s": seg_in / fs, "expiration_s": seg_out / fs,
            "inspiration_i": seg_in, "expiration_i": seg_out,
            "normalised": norm, "velocity": vel, "n_breaths": len(seg_in)}


def _trim_segments(seg_in: np.ndarray, seg_out: np.ndarray):
    """Drop an incomplete rise at either end, so every onset has a matching offset."""
    if len(seg_out) and len(seg_in) and seg_out[0] < seg_in[0]:
        seg_out = seg_out[1:]
    n = min(len(seg_in), len(seg_out))
    return np.clip(seg_in[:n], 0, None), seg_out[:n]


def respiratory_phases(x, fs: float, *, scale_high: float = 0.7, scale_low: float = 0.3,
                       **kw) -> dict:
    """Decompose a respiration recording into the phases of the breath cycle.

    A breathing *rate* says how often; this says where in each cycle the body is. That matters
    here specifically, because the post-expiration pause is the moment in the cycle when the
    body is most nearly still, so relating breathing to micromotion wants phases rather than a
    rate.

    Returns boolean masks over the input samples:

    ``inspiration``, ``expiration``
        the two half-cycles.
    ``inspiration_high``, ``expiration_high``
        high-flow moments judged against the whole recording -- the ``scale_high`` quantile of
        velocity within each phase.
    ``inspiration_v``, ``expiration_v``
        high-flow moments judged *within each breath*, against that breath's own peak velocity.
        Use these when breath size varies across the recording, which it does during settling.
    ``post_expiration``
        the pause after expiration has slowed to ``scale_low`` of its own peak rate.

    The defaults come from the coordination analysis in Upham (2018).

    Ported from Finn Upham's ``respy`` (MIT, 2023) with permission, and reimplemented on numpy.
    The port is not gratuitous: ``respy.Resp_phases`` assigns through ``df[col].loc[idx]``, which
    under pandas copy-on-write silently does not write, so on pandas 2 and later it returns all
    twelve of its phase columns empty without raising. Verified against ``respy`` 0.1.1 on
    pandas 3.0.3, where every phase column came back 0.0 per cent populated.

    References
    ----------
    Upham, F. (2018). *Detecting the Adaptation of Listeners' Respiration to Heard Music*.
    PhD thesis, New York University.
    """
    o = respiration_onsets(x, fs, **kw)
    n = len(o["normalised"])
    norm, vel = o["normalised"], o["velocity"]
    ins, exp = o["inspiration_i"], o["expiration_i"]

    out = {k: np.zeros(n, bool) for k in
           ("inspiration", "expiration", "inspiration_high", "expiration_high",
            "inspiration_v", "expiration_v", "post_expiration")}

    for lo, hi in zip(ins, exp):                       # inspiration: onset to offset
        out["inspiration"][lo:hi + 1] = True
    for hi, lo in zip(exp[:-1], ins[1:]):              # expiration: offset to the next onset
        out["expiration"][hi:lo + 1] = True

    # sequence-wise: one threshold for the whole recording, per phase
    if out["inspiration"].any():
        t = np.quantile(vel[out["inspiration"]], scale_high)
        out["inspiration_high"] = out["inspiration"] & (vel >= t)
    if out["expiration"].any():
        t = np.quantile(vel[out["expiration"]], 1 - scale_high)
        out["expiration_high"] = out["expiration"] & (vel <= t)

    # breath-wise: each half-cycle against its own peak rate
    for lo, hi in zip(ins, exp):
        seg = vel[lo:hi + 1]
        if len(seg) and seg.max() > 0:
            out["inspiration_v"][lo:hi + 1] = seg > seg.max() * scale_high
    for hi, lo in zip(exp[:-1], ins[1:]):
        seg = vel[hi:lo + 1]
        if not len(seg):
            continue
        if seg.min() < 0:
            out["expiration_v"][hi:lo + 1] = seg < seg.min() * scale_high
            # the pause: after the steepest point, once the rate has fallen below scale_low of it
            k = int(np.argmin(seg))
            slowed = seg.copy()
            slowed[:k] = seg.min()                     # never call the run-up a pause
            out["post_expiration"][hi:lo + 1] = slowed > seg.min() * scale_low

    out["inspiration_onset_s"] = o["inspiration_s"]
    out["expiration_onset_s"] = o["expiration_s"]
    out["normalised"] = norm
    out["velocity"] = vel
    out["n_breaths"] = o["n_breaths"]
    return out
