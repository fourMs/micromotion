"""Rate measurement and resampling.

Two rules, both learned the hard way.

Measure the rate, do not read it. Nominal rates in this corpus are wrong by up to 4.4 per
cent, and one record's documented rate was out by a factor of 37.

Downsample, never upsample. Upsampling invents structure between samples, and every method
that reads across scales treats the invention as real. On 2026-07-28 an analysis that
upsampled 20 Hz data to 25 Hz produced multifractal widths up to 6.6 where the plausible
range is around 1. Nothing failed; the numbers were simply wrong.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

COMMON_RATE = 20.0
"""Hz. The cross-collection analysis rate.

Not a compromise: the micromotion band stops at 10 Hz, so 20 Hz is its Nyquist rate and
everything above it is discarded by the band-pass anyway. It is also the native rate of the
slowest collection, so reaching it never requires upsampling. See
``deposit/SAMPLING_RATES.md``.
"""


def measured_rate(t) -> float:
    """Sampling rate in Hz, from a timestamp vector.

    Sample count over elapsed span, deliberately not the reciprocal of the median interval.
    Where timestamps are rounded to whole milliseconds the intervals become a mixture of
    adjacent integers and their median is a quantisation artefact: on Stillness2025 that
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
