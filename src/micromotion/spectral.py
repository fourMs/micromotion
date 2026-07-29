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
    """Dominant frequency in the respiratory band, in Hz.

    Needs a long window: at 0.2 Hz, a 60-second segment holds twelve cycles, which is few
    enough that the estimate wanders.
    """
    return _peak(x, fs, RESPIRATORY_BAND, window_s)


def band_power(x, fs: float, band: tuple[float, float], window_s: float = 60.0) -> float:
    """Integrated power between two frequencies."""
    x = np.asarray(x, float)
    nper = int(min(len(x), fs * window_s))
    f, p = signal.welch(signal.detrend(x), fs, nperseg=nper)
    m = (f >= band[0]) & (f <= band[1])
    return float(np.trapezoid(p[m], f[m])) if m.sum() > 1 else float("nan")


def spectral_peak(x, fs: float, band: tuple[float, float],
                  window_s: float = 60.0) -> dict:
    """Peak frequency in a band, with a signal-to-noise ratio against the band's own median.

    The ratio is what makes the peak reportable. Every spectrum has a maximum somewhere
    inside any band you choose; whether it rises above the surrounding noise decides whether
    a rhythm is present at all. Below about 3, treat the peak as absent rather than weak.
    """
    x = np.asarray(x, float)
    if len(x) < fs * 10:
        return {"freq": float("nan"), "power": float("nan"), "snr": float("nan")}
    nper = int(min(len(x), fs * window_s))
    f, p = signal.welch(signal.detrend(x), fs, nperseg=nper)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        return {"freq": float("nan"), "power": float("nan"), "snr": float("nan")}
    k = int(np.argmax(p[m]))
    med = float(np.median(p[m]))
    return {"freq": float(f[m][k]), "power": float(p[m][k]),
            "snr": float(p[m][k] / med) if med > 0 else float("nan")}


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
