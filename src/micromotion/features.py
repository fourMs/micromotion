"""One canonical feature vector per recording.

Every attempt to compare recordings across this corpus -- clustering, identity classification,
condition classification, the dimensionality reductions in :mod:`micromotion.descriptors` -- needs
the same thing first: a fixed set of numbers describing one recording. Each attempt had been
inventing its own, which makes two results incomparable for reasons that have nothing to do with
the question either was asking.

This is that set, and it is the only thing of its kind the package offers. Models belong outside
the package, in an analysis repository where a train/test split and its leakage are visible; what
belongs here is the input they all start from.

Eleven descriptors, in three groups:

* amount and smoothness -- ``qom``, ``jerk``
* frequency and texture -- ``centroid``, ``f50``, ``frozen``, ``burst``
* sway geometry -- ``path``, ``extent``, ``area``, ``anis``, ``vert``

The geometric five need true position and are ``nan`` for accelerometer collections. A chest-worn
accelerometer cannot give a sway ellipse, and doubly integrating one to fake it reports drift as
posture.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .filters import WIDEBAND, bandpass, effective_band
from .qom import derivative, velocity_from_acceleration, velocity_from_position

__all__ = ["FEATURE_NAMES", "feature_vector"]

FEATURE_NAMES = ("qom", "jerk", "centroid", "f50", "frozen", "burst",
                 "path", "extent", "area", "anis", "vert")


def feature_vector(x, fs: float, kind: str | None = None, unit: str | None = None,
                   sensor_fs: float | None = None) -> dict | None:
    """The eleven descriptors for one recording, or ``None`` if it is too short to describe.

    ``kind`` and ``unit`` must be passed and are not guessed, because the collections do not
    record the same quantity: the optical ones store position in mm and the accelerometer ones
    store acceleration in g or m/s^2. Differentiating an acceleration series as though it were
    position shifts every descriptor two derivatives up and still returns finite,
    plausible-looking numbers, which is the failure mode this signature exists to prevent.

    ``sensor_fs`` is the rate the device actually sampled at, which is not always the rate
    the file is stored on: a uniform grid can be an upsample of a much slower sensor, and
    checking the grid would admit a band the data cannot carry. Pass it whenever the two differ.

    Everything derives from a band-limited velocity, so the filter order and the units are
    identical whichever way the recording arrived.
    """
    # Checked before anything else, because omitting these is a programming error rather than a
    # property of the data. Until 0.13.0 they defaulted to position and mm, which contradicted the
    # paragraph above and reinstated exactly the silent failure it describes: an accelerometer
    # series passed without `kind` was differentiated as though it were position and came back
    # finite and plausible. The docstring was right and the signature was wrong.
    if kind is None or unit is None:
        raise TypeError(
            "feature_vector requires both kind and unit; they are not guessed. Use "
            "kind='position', unit='mm' for optical data, or kind='acceleration' with "
            "unit='g' or 'm/s^2' for accelerometer data.")

    # copy, because the gap fill below writes into it and the caller's array is not ours to edit
    x = np.array(x, dtype=float, copy=True)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"expected an (n, 3) array of x/y/z, got {x.shape}")
    # fill short gaps by linear interpolation along each axis, then require the rest to be finite
    for j in range(x.shape[1]):
        col = x[:, j]
        m = np.isfinite(col)
        if m.any() and not m.all():
            x[:, j] = np.interp(np.arange(len(col)), np.flatnonzero(m), col[m])
    if len(x) < fs * 120 or not np.isfinite(x).all():
        return None

    lo, hi = effective_band(fs)
    if kind == "position":
        vel = velocity_from_position
    elif kind == "acceleration":
        vel = velocity_from_acceleration
    else:
        raise ValueError(f"unknown kind {kind!r}; use 'position' or 'acceleration'")
    V = vel(x, fs, unit=unit, lo=lo, hi=hi)
    v = np.linalg.norm(V, axis=1)                                       # speed, mm/s

    # Jerk is computed at WIDEBAND rather than at the canonical band, because it lives in the
    # octave the canonical band gives up: at a 5 Hz ceiling it falls to between a third and two
    # thirds of its 10 Hz value and the ranking shifts. One definition of jerk across the corpus
    # matters more than one band within this vector. `nan` where the rate cannot deliver it,
    # rather than a narrower measure reported under a wider name.
    wlo, whi = effective_band(fs, *WIDEBAND)
    if (sensor_fs or fs) < 2 * WIDEBAND[1] or whi < WIDEBAND[1] * 0.999:
        j = np.array([np.nan])
    else:
        W = vel(x, fs, unit=unit, lo=wlo, hi=whi)
        j = np.linalg.norm(derivative(derivative(W, fs), fs), axis=1)

    f_, P = signal.welch(v - v.mean(), fs=fs, nperseg=int(min(len(v), fs * 30)))
    k = (f_ >= lo) & (f_ <= hi)
    centroid = float((f_[k] * P[k]).sum() / P[k].sum())
    cum = np.cumsum(P[k]) / P[k].sum()
    f50 = float(np.interp(0.5, cum, f_[k]))

    out = dict(
        qom=float(np.median(v)),                        # amount
        jerk=float(np.median(j)),                       # smoothness, at WIDEBAND
        centroid=centroid,                              # frequency, energy-weighted
        f50=f50,                                        # frequency, median
        frozen=float((v < np.median(v) / 2).mean()),    # fraction nearly stopped
        burst=float(np.percentile(v, 99) / np.median(v)),                    # peakiness
        path=np.nan, extent=np.nan, area=np.nan, anis=np.nan, vert=np.nan,
    )
    if kind != "position":
        return out

    c = bandpass(x if unit == "mm" else x * 1000.0, fs, lo, hi)
    c = c - c.mean(0)
    w = np.linalg.eigvalsh(np.cov(c[:, :2].T))
    d = np.linalg.norm(c, axis=1)
    out.update(
        path=float(np.sum(v) / fs),                     # cumulative distance travelled
        extent=float(np.percentile(d, 95)),             # how far it strays
        area=float(np.pi * np.sqrt(max(w[0], 1e-12) * max(w[1], 1e-12))),    # sway ellipse
        anis=float(np.sqrt(max(w[1], 1e-12) / max(w[0], 1e-12))),            # elongation
        vert=float(np.std(c[:, 2])),                    # vertical excursion
    )
    return out
