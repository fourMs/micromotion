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

import numpy as np
from scipy import signal

BAND = (0.3, 10.0)
"""The micromotion band, in Hz. See ``deposit/SAMPLING_RATES.md``."""

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


def _edges(fs: float, lo: float, hi: float) -> tuple[float, float]:
    ny = fs / 2.0
    if lo <= 0 or lo >= ny:
        raise ValueError(f"low edge {lo} Hz is not below Nyquist {ny} Hz")
    hi = min(hi, ny * 0.99)
    if hi <= lo:
        raise ValueError(
            f"sampling rate {fs} Hz is too low for a {lo}-{hi} Hz band; "
            "the micromotion band needs at least 20 Hz"
        )
    return lo / ny, hi / ny


def bandpass(x, fs: float, lo: float | None = BAND[0], hi: float = BAND[1], order: int = ORDER):
    """Zero-phase band-limiting along the first axis.

    ``lo=None`` gives a pure low-pass, which is the :data:`OPTICAL_LEGACY_BAND` convention.
    """
    if lo is None:
        return lowpass(x, fs, hi, order)
    wl, wh = _edges(fs, lo, hi)
    sos = signal.butter(order, [wl, wh], btype="band", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, float), axis=0)


def lowpass(x, fs: float, fc: float = BAND[1], order: int = ORDER):
    """Zero-phase low-pass along the first axis."""
    ny = fs / 2.0
    fc = min(fc, ny * 0.99)
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
