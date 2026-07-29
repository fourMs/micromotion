"""Band-limiting for the micromotion band.

One definition, used everywhere. Across the Still Standing repository 46 scripts defined
their own filter and they did not all agree; the differences were invisible in the output
and moved quantity of motion by up to 10 per cent.

The canonical band is 0.3-10 Hz, a zero-phase Butterworth of order 4 applied as
second-order sections. The lower edge sits below the respiratory rate and above the
postural drift that integration turns into a ramp; the upper edge is where accelerometer
noise begins to dominate the signal of a person standing still.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import signal

BAND = (0.2, 10.0)
"""The micromotion band, in Hz.

The lower edge was 0.3 Hz until 2026-07-29, inherited rather than chosen. Swept across seven
optical datasets, 665 recordings, the between-dataset spread is 3.2 per cent at 0.15 Hz,
2.1 at 0.20, 2.7 at 0.25, 6.2 at 0.30 and 10.1 at 0.40. 0.2 Hz is a clear optimum, and it is
also where the 20 Hz origin dataset stops being an outlier in either direction.

A lower edge this close to DC has to be checked against integration drift, since that is what
the edge is there to control. It survives: on accelerometer data the ratio of mean to median
speed, which rises if drift is leaking in, is 2.07 at 0.2 Hz against 2.00 at 0.3 -- flat. What
does change is the absolute value, by about 73 per cent, because more of the low-frequency
content is retained.

See ``deposit/_analysis/osdb_qom/REPORT.md``.
"""

OPTICAL_LEGACY_BAND = (None, 10.0)
"""A 10 Hz low-pass with no lower edge: the convention behind the published championship
quantity of motion.

It is kept because those numbers are in print, not because it is interchangeable with
:data:`BAND`. Optical position is an absolute measurement, so sub-0.3 Hz postural drift in
it is real movement and there is no reason to discard it. A body-worn accelerometer cannot
offer the same choice: gravity is a DC term, and integrating any residual offset produces a
ramp that swamps the result. So the lower edge is optional for position and mandatory for
acceleration.

The two are not the same measure. On the 2015 championship the band-pass reads 15.5 per
cent below the low-pass. Any table that puts optical and accelerometer collections side by
side must therefore use :data:`BAND` throughout, and say so.
"""

ORDER = 4


NARROW_BAND_RATIO = 40.0
"""Warn when the sampling rate exceeds the upper band edge by more than this.

A band-pass whose edges sit very close to zero in normalised frequency is numerically
fragile, and how fragile depends on how it is realised. Second-order sections, which this
module uses, stay accurate far longer than the transfer-function form -- but not forever, and
a caller designing their own filter should be warned before they are bitten.

The failure is silent and it is not hypothetical. A 0.1-0.5 Hz third-order band-pass at
250 Hz, written in the usual ``butter(3, [lo/ny, hi/ny])`` transfer-function form, has a
largest pole radius of 0.9979 and a measured passband gain of 0.84 at 0.15 Hz where it should
be 0.99. Nothing raises, nothing looks wrong, and every amplitude downstream is 16 per cent
low. The same design as second-order sections gives 0.9875.

The fix when this warns is to decimate first, so the band sits comfortably inside the new
Nyquist range, then filter.
"""


NYQUIST_MARGIN = 0.99
"""How close to Nyquist an upper band edge may sit, as a fraction.

A filter designed right at Nyquist has no transition band left, so the edge is pulled in.
The default is conservative. The Still Standing corpus uses 0.999 throughout, which matters
whenever the band edge is already near Nyquist -- a 10 Hz low-pass on 20 Hz data becomes
9.9 Hz here and 9.99 Hz there, and on real optical data that moved quantity of motion by
7e-5. Pass ``margin`` to match whatever convention the surrounding analysis uses.
"""


def _edges(fs: float, lo: float, hi: float,
           margin: float = NYQUIST_MARGIN) -> tuple[float, float]:
    ny = fs / 2.0
    if lo <= 0 or lo >= ny:
        raise ValueError(f"low edge {lo} Hz is not below Nyquist {ny} Hz")
    hi = min(hi, ny * margin)
    if hi > 0 and fs / hi > NARROW_BAND_RATIO:
        warnings.warn(
            f"band {lo}-{hi} Hz is very low against a {fs} Hz sampling rate "
            f"(ratio {fs / hi:.0f}:1). Second-order sections hold up here, but a "
            "transfer-function filter of the same design would not, and accuracy degrades "
            "as the ratio grows. Consider decimating first.",
            RuntimeWarning, stacklevel=3)
    if hi <= lo:
        raise ValueError(
            f"sampling rate {fs} Hz is too low for a {lo}-{hi} Hz band; "
            "the micromotion band needs at least 20 Hz"
        )
    return lo / ny, hi / ny


def bandpass(x, fs: float, lo: float | None = BAND[0], hi: float = BAND[1],
             order: int = ORDER, margin: float = NYQUIST_MARGIN):
    """Zero-phase band-limiting along the first axis.

    ``lo=None`` gives a pure low-pass, which is the :data:`OPTICAL_LEGACY_BAND` convention.
    """
    if lo is None:
        return lowpass(x, fs, hi, order, margin)
    wl, wh = _edges(fs, lo, hi, margin)
    sos = signal.butter(order, [wl, wh], btype="band", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, float), axis=0)


def lowpass(x, fs: float, fc: float = BAND[1], order: int = ORDER,
            margin: float = NYQUIST_MARGIN):
    """Zero-phase low-pass along the first axis.

    ``margin`` is how close to Nyquist ``fc`` may sit; see :data:`NYQUIST_MARGIN`.
    """
    ny = fs / 2.0
    fc = min(fc, ny * margin)
    sos = signal.butter(order, fc / ny, btype="low", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, float), axis=0)


def highpass(x, fs: float, fc: float = BAND[0], order: int = ORDER):
    """Zero-phase high-pass along the first axis.

    Used where the upper edge is meaningless because the rate is already near the band
    limit, and for gravity removal when no low-pass is wanted.
    """
    ny = fs / 2.0
    if fc <= 0 or fc >= ny:
        raise ValueError(f"cutoff {fc} Hz is not below Nyquist {ny} Hz")
    sos = signal.butter(order, fc / ny, btype="high", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, float), axis=0)


def notch(x, fs: float, f0: float, q: float = 6.0):
    """Zero-phase notch at ``f0`` Hz.

    Used to remove the cardiac peak when isolating postural micromotion. ``f0`` is
    normally found with :func:`micromotion.spectral.cardiac_peak`.
    """
    if not np.isfinite(f0) or f0 <= 0 or f0 >= fs / 2:
        return np.asarray(x, float)
    b, a = signal.iirnotch(f0 / (fs / 2), Q=q)
    return signal.filtfilt(b, a, np.asarray(x, float), axis=0)


def edge_transient_samples(fs: float, lo: float | None = BAND[0], order: int = ORDER) -> int:
    """Samples at each end that ``filtfilt`` contaminates.

    A conservative estimate: the impulse response of the low edge, doubled for the
    forward-backward pass. Callers should either trim this or flag it, as the deposited
    five-second binning does with its ``edge`` column.
    """
    return int(np.ceil(2 * order * fs / (lo if lo else BAND[1])))
