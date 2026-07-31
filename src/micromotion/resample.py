"""Rate measurement and resampling.

Two rules, both learned the hard way.

Measure the rate, do not read it. Nominal rates in this corpus are wrong by up to 4.4 per
cent, and one record's documented rate was out by a factor of 37.

Downsample, never upsample. Upsampling invents structure between samples, and every method
that reads across scales treats the invention as real. An analysis that
upsampled 20 Hz data to 25 Hz produced multifractal widths up to 6.6 where the plausible
range is around 1. Nothing failed; the numbers were simply wrong.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

COMMON_RATE = 20.0
"""Hz. The rate at which every collection can be compared, including the slowest.

It is the greatest common divisor of the corpus's optical rates -- 20, 100, 120 and 200 Hz --
so every recording reaches it by an integer decimation and none requires upsampling. That is
its one virtue, and it is a real one: it is the only rate at which the natively-20 Hz origin
study can be placed beside the rest at all.

It is a lossy rate, not a free one. The argument that the band stops at 10 Hz so 20 Hz discards
nothing does not survive measurement, because a 10 Hz upper edge cannot be realised at 20 Hz:
Nyquist sits exactly on it, the margin rule pulls the edge inside, and the anti-alias filter is
already rolling off below it. Decimating 34 natively-200 Hz person-recordings and re-measuring
moves quantity of motion by -2.09 per cent at the median, and between -0.30 and -10.57 per cent
across recordings. A per-recording spread that wide is a distortion rather than a bias: it
cannot be measured once and corrected away.

Prefer :data:`HARMONISED_RATE` where every series in the comparison can reach it. Use this one
when the comparison must include a 20 Hz collection, and say in the output that it is the lossy
view.
"""

HARMONISED_RATE = 100.0
"""Hz. The preferred rate for a comparison whose series can all reach it.

Native for several collections, an exact halving from 200 Hz, and a 6-to-5 polyphase step from
120 Hz. Measured against native 200 Hz values it costs +0.02 per cent at the median and stays
within +/-0.85 per cent, against -2.09 and up to -10.57 at 20 Hz. It also leaves the 0.2-5 Hz
band comfortably inside Nyquist rather than sitting on it.
"""


def measured_rate(t) -> float:
    """Sampling rate in Hz, from a timestamp vector.

    Sample count over elapsed span, deliberately not the reciprocal of the median interval.
    Where timestamps are rounded to whole milliseconds the intervals become a mixture of
    adjacent integers and their median is a quantisation artefact: on one dataset that
    route returns exactly 250 Hz for a recording that runs at 256, and on the pre-study
    phones it returned 636 Hz for a stream arriving at 106.
    """
    t = np.asarray(t, float)
    if len(t) < 2:
        raise ValueError("need at least two timestamps")
    span = t[-1] - t[0]
    if span <= 0:
        raise ValueError("timestamps do not increase")
    return (len(t) - 1) / span


def rate_quality(t) -> dict:
    """How regular a timestamp vector actually is.

    Returns the measured rate, the jitter, the largest gap, and counts of duplicated and
    backward timestamps. The balance-board files carry 4.7 per cent duplicates and 83
    backward steps, and one pre-study file is 19 per cent covered because of a single
    132-second gap; neither is visible from the rate alone.
    """
    t = np.asarray(t, float)
    d = np.diff(t)
    fs = measured_rate(t)
    nominal = 1.0 / fs
    return {
        "measured_rate_hz": fs,
        "n_samples": len(t),
        "duration_s": float(t[-1] - t[0]),
        "jitter_cv": float(np.std(d) / np.mean(d)) if np.mean(d) else float("nan"),
        "max_gap_s": float(d.max()) if len(d) else float("nan"),
        "gap_ratio": float(d.max() / nominal) if len(d) and nominal else float("nan"),
        "n_duplicate_t": int((d == 0).sum()),
        "n_backward_t": int((d < 0).sum()),
        "coverage": float(d[d <= 5 * nominal].sum() / (t[-1] - t[0])) if len(d) else 1.0,
    }


def to_rate(x, fs_in: float, fs_out: float = COMMON_RATE):
    """Anti-alias resample to ``fs_out``, refusing to upsample.

    Raises rather than upsampling. If a series genuinely cannot reach the target it does
    not belong in a comparison built at that rate, and interpolating it in would corrupt
    the comparison silently.
    """
    if fs_out > fs_in + 1e-9:
        raise ValueError(
            f"refusing to upsample {fs_in:.4g} Hz to {fs_out:.4g} Hz. "
            "Upsampling invents structure between samples; exclude this series instead."
        )
    x = np.asarray(x, float)
    if abs(fs_in - fs_out) < 1e-9:
        return x

    # Polyphase FIR, not signal.resample. The FFT resampler treats the series as periodic,
    # so on optical position data -- which carries a large offset and does not begin and end
    # at the same value -- it rings across the whole recording. That moved quantity of
    # motion by up to 8 per cent and did so non-monotonically in the rate, which is exactly
    # the kind of silent error a harmonised table cannot carry.
    up, down = Fraction(fs_out / fs_in).limit_denominator(1000).as_integer_ratio()
    if up == 0:
        raise ValueError(f"rate ratio {fs_out}/{fs_in} is too extreme to resample")
    return signal.resample_poly(x, up, down, axis=0, padtype="line")


def regularize(t, x, fs_out: float | None = None, max_gap_s: float | None = None):
    """Put an irregularly sampled signal onto a uniform grid.

    Sorts, drops duplicate and backward timestamps, then interpolates linearly. Samples
    that fall inside a gap longer than ``max_gap_s`` are returned as NaN rather than
    bridged, so that a 132-second hole cannot be mistaken for 132 seconds of stillness.

    Written for the Wii balance board, whose timestamps are unsorted, 4.7 per cent
    duplicated, and arrive at a median 61.8 Hz that varies between files.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    x2 = x[:, None] if x.ndim == 1 else x

    order = np.argsort(t, kind="stable")
    t, x2 = t[order], x2[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, x2 = t[keep], x2[keep]
    if len(t) < 2:
        raise ValueError("fewer than two usable timestamps after cleaning")

    fs_out = fs_out or measured_rate(t)
    grid = np.arange(t[0], t[-1], 1.0 / fs_out)
    out = np.column_stack([np.interp(grid, t, x2[:, i]) for i in range(x2.shape[1])])

    if max_gap_s is not None:
        d = np.diff(t)
        for i in np.flatnonzero(d > max_gap_s):
            out[(grid > t[i]) & (grid < t[i + 1])] = np.nan

    return grid, (out[:, 0] if x.ndim == 1 else out)


def interpolate_gaps(x, max_gap: int = 200):
    """Bridge short runs of NaN, leave long ones alone.

    Filters cannot run across missing samples, so gaps have to be handled before anything
    else. Bridging a dropped frame is reconstruction; bridging a 469-second hole is
    invention, and the difference is only a matter of degree, which is why the threshold is
    explicit and the long gaps stay NaN for the caller to exclude.

    Works column by column. Leading and trailing gaps are never filled, since there is
    nothing on one side to interpolate from.
    """
    a = np.array(x, float)
    one_d = a.ndim == 1
    if one_d:
        a = a[:, None]
    for j in range(a.shape[1]):
        v = a[:, j]
        bad = np.isnan(v)
        if not bad.any() or bad.all():
            continue
        good = np.flatnonzero(~bad)
        idx = np.flatnonzero(bad)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
            if len(run) <= max_gap and run[0] > 0 and run[-1] < len(v) - 1:
                v[run] = np.interp(run, good, v[good])
    return a[:, 0] if one_d else a


def gap_report(x, fs: float) -> dict:
    """Where the missing data is, and how it is distributed.

    A single missing fraction hides the distinction that matters: one per cent scattered
    evenly is a usable recording, and one per cent in a single block in the middle is two
    recordings.
    """
    a = np.asarray(x, float)
    bad = np.isnan(a)
    if a.ndim > 1:
        bad = bad.any(axis=1)
    if not bad.any():
        return {"missing_frac": 0.0, "n_gaps": 0, "longest_gap_s": 0.0}
    idx = np.flatnonzero(bad)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    return {
        "missing_frac": float(bad.mean()),
        "n_gaps": len(runs),
        "longest_gap_s": float(max(len(r) for r in runs) / fs),
        "median_gap_s": float(np.median([len(r) for r in runs]) / fs),
    }
