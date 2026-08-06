"""Band-limiting for the micromotion band.

One definition, used everywhere. Across the Still Standing repository 46 scripts defined
their own filter and they did not all agree; the differences were invisible in the output
and moved quantity of motion by up to 10 per cent.

The canonical band is 0.2-5 Hz, a zero-phase Butterworth of order 4 applied as
second-order sections. The lower edge sits below the respiratory rate and above the
postural drift that integration turns into a ramp; the upper edge is set by what the
slowest instrument in the corpus can actually deliver.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import signal

BAND = (0.2, 5.0)
"""The micromotion band, in Hz.

**The lower edge** was chosen by a sweep across seven optical datasets and 665 recordings: the
between-dataset spread is 3.2 per cent at 0.15 Hz, 2.1 at 0.20, 2.7 at 0.25, 6.2 at 0.30 and
10.1 at 0.40. 0.2 Hz is a clear optimum, and it is also where the 20 Hz origin dataset stops
being an outlier in either direction.

An edge this close to DC has to be checked against integration drift, since that is what it is
there to control. It survives: on accelerometer data the ratio of mean to median speed, which
rises if drift is leaking in, is 2.07 at 0.2 Hz against 2.00 at 0.3 -- flat.

**The upper edge is set by deliverability, not by taste.** A band above Nyquist is not a
convention but a defect that returns a plausible number, and the ceiling must therefore be one
that every instrument in the corpus can support:

============================  ==================  ==========  ==========
collection                    sampling            Nyquist     5 Hz?
============================  ==================  ==========  ==========
phone, fused linear accel     ~15 Hz              7.5         yes
optical, 20 Hz subset         20 Hz               10          yes
collaborators' audience data  10 Hz               5           at Nyquist
optical and inertial, rest    100-256 Hz          50-128      yes
============================  ==================  ==========  ==========

A 10 Hz ceiling fails the first row on 354 of 355 days and sits exactly on Nyquist for the
second. 5 Hz clears every one of them.

The first row is worth stating precisely, because the obvious reading is wrong. The phone's
*accelerometer* runs at about 50 Hz. What runs at 15 Hz is the **linear acceleration**, which the
logging app derives by fusing the accelerometer with the gyroscope and magnetometer to remove
gravity -- and a fusion cannot output faster than its slowest input, which is the 15 Hz gyroscope.
The deposited files carry that fused channel, so their usable Nyquist really is 7.5 Hz. The limit
is the channel chosen, not the hardware.

**What it costs.** Band-limited *speed* mostly does not notice: measured over 466
person-recordings, 5 Hz against 10 differs by a median of 1.3 per cent, and 95 per cent of
quiet-standing sway power lies below 1 Hz anyway. That is a distribution rather than a bound.
Its ninetieth percentile is 3.2 per cent and its maximum 9.1, largest on optical collections
sampled at 100 to 200 Hz, which genuinely resolve the octave between 5 and 10 Hz. Rankings
survive: on one 199-recording optical collection the change moves the median 0.8 per cent and
leaves the ranking at Spearman 0.996.

**What it costs that matters.** *Jerk* is two derivatives higher and lives in the discarded
region: at 5 Hz it is 37 to 66 per cent of its 10 Hz value, and the ranking shifts too. So jerk
must not be computed at this band on data that could support a wider one -- use
:data:`WIDEBAND` and say so. On the phone collection the wider jerk was never real: the 15 Hz
sensor cannot produce it, and computing it there inflated jerk 18 to 27 per cent with
interpolation.
"""

WIDEBAND = (0.2, 10.0)
"""0.2-10 Hz: the band for measures that need the octave :data:`BAND` gives up.

Jerk and other high-derivative quantities live between 5 and 10 Hz, where band-limited speed
does not. This band is for them, **on collections sampled fast enough to deliver it** -- check
with :func:`effective_band` first, and never assume it from a file's grid, which may be an
upsample of a much slower sensor.

It is deliberately not the default. A quantity computed here is not comparable with one
computed at :data:`BAND`, and it is not computable at all on the slower collections.
"""

OPTICAL_LEGACY_BAND = (None, 10.0)
"""A 10 Hz low-pass with no lower edge: the convention behind the published championship
quantity of motion.

It is kept because those numbers are in print, not because it is interchangeable with
:data:`BAND`. Optical position is an absolute measurement, so sub-0.2 Hz postural drift in
it is real movement and there is no reason to discard it. A body-worn accelerometer cannot
offer the same choice: gravity is a DC term, and integrating any residual offset produces a
ramp that swamps the result. So the lower edge is optional for position and mandatory for
acceleration.

The two are not the same measure. On the 2015 championship the band-pass reads 15.5 per
cent below the low-pass. Any table that puts optical and accelerometer collections side by
side must therefore use :data:`BAND` throughout, and say so.
"""

ORDER = 4


NARROW_BAND_RATIO = 1000.0
"""Warn when the sampling rate exceeds the **lower** band edge by more than this.

The lower edge is what drives the conditioning, which is why the ratio is taken against it.
The worked example below fails at 0.15 Hz, near the bottom of its band, not near the top; a
test keyed on the upper edge measures the wrong thing and moves whenever the ceiling moves.
It did: halving the canonical ceiling from 10 Hz to 5 doubled an upper-edge ratio while the
conditioning was unchanged, and the warning began firing on every high-rate call.

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
    requested_hi = hi
    hi = min(hi, ny * margin)
    if hi < requested_hi:
        # Silently returning a number computed in a narrower band than the caller asked for is
        # how two results get compared that were never in the same band. A 10 Hz recording
        # cannot carry a 10 Hz upper edge -- Nyquist is 5 -- so the band becomes 0.2-4.95 and
        # the result is not comparable with one from a faster recording unless both are
        # deliberately limited to the same edge.
        warnings.warn(
            f"upper edge {requested_hi} Hz exceeds Nyquist for {fs} Hz sampling and was "
            f"clamped to {hi:.4g} Hz. The result is band-limited to {lo}-{hi:.4g} Hz, not "
            f"{lo}-{requested_hi} Hz, and is not comparable with results computed at the "
            f"requested band. Use effective_band() to check before comparing.",
            RuntimeWarning, stacklevel=3)
    if lo and lo > 0 and fs / lo > NARROW_BAND_RATIO:
        warnings.warn(
            f"lower edge {lo} Hz is very low against a {fs} Hz sampling rate "
            f"(ratio {fs / lo:.0f}:1). Second-order sections hold up here, but a "
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


def effective_band(fs: float, lo: float | None = BAND[0], hi: float = BAND[1],
                   margin: float = NYQUIST_MARGIN) -> tuple[float | None, float]:
    """The band that will actually be applied at this sampling rate.

    The requested upper edge is clamped to just below Nyquist, so a rate below twice ``hi``
    silently narrows the band. Ask before comparing two results computed at different rates:
    at 8 Hz the canonical 0.2-5 Hz band becomes 0.2-3.96, which is a different measurement.

    >>> effective_band(100.0)
    (0.2, 5.0)
    >>> effective_band(10.0)
    (0.2, 4.95)
    """
    ny = fs / 2.0
    return lo, float(min(hi, ny * margin))
