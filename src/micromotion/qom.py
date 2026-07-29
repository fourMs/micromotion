"""Quantity of motion.

The signature measure of this research programme: the average speed of a body part, in
millimetres per second, restricted to the micromotion band.

There is one definition and three named variants. The point of naming them is that they
answer different questions and previously differed only by an undocumented line of filter
code. Any figure or paper should be able to say which variant produced its numbers.

Definition
----------
Band-limit each axis to 0.3-10 Hz, bring it to velocity, band-limit again, take the
Euclidean norm across axes, and report the mean in mm/s.

Acceleration is brought to velocity by integration, position by differentiation. The second
band-limiting is not cosmetic. Integrating a signal with any residual offset produces a
ramp that dominates the result, and differentiating amplifies the high-frequency noise that
the first pass was meant to exclude.

Variants
--------
``raw``
    The band as defined. Contains respiration, the ballistocardiac impulse and postural
    sway together. This is what the deposited files report unless they say otherwise.
``compensated``
    Respiration and cardiac activity removed: the lower edge is raised to 0.5 Hz and the
    per-recording cardiac peak is notched out. What is left is postural micromotion.
``tilt_corrected``
    For a single accelerometer only. A body-worn accelerometer cannot distinguish leaning
    from translating: tilting into gravity produces an acceleration with no displacement.
    Where a gyroscope is available the tilt component is estimated and removed. Measured
    directly on the fNIRS session, tilt inflates raw QoM by 1.56x.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import filters
from .spectral import cardiac_peak

G = 9.80665
"""Standard gravity, m/s^2.

Present because accelerometers export in g and the band-limited result must be in SI
before integration. Getting this wrong is not hypothetical: every phone quantity of motion
in this project was 9.80665x too large until 2026-07-28.
"""

MM_PER_M = 1000.0


@dataclass
class QomResult:
    """Quantity of motion, with the series it was reduced from."""

    mean_mm_s: float
    median_mm_s: float
    speed: np.ndarray = field(repr=False)
    fs: float = 0.0
    variant: str = "raw"
    cardiac_hz: float = float("nan")
    n_samples: int = 0
    edge_samples: int = 0

    def binned(self, bin_s: float = 5.0):
        """Average speed in fixed-width bins, as a DataFrame.

        The final bin is usually partial and is flagged rather than dropped, because a
        short bin is not comparable with a full one and silently including it inflated the
        deposited five-second series three- to fourteenfold.
        """
        import pandas as pd

        idx = (np.arange(self.n_samples) / self.fs // bin_s).astype(int)
        counts = np.bincount(idx)
        means = np.bincount(idx, self.speed) / counts
        full = int(round(bin_s * self.fs))
        edge = np.where(counts < full, "partial", "ok").astype(object)
        n_edge = max(1, int(np.ceil(self.edge_samples / (bin_s * self.fs))))
        for i in list(range(n_edge)) + list(range(len(counts) - n_edge, len(counts))):
            if 0 <= i < len(counts) and edge[i] == "ok":
                edge[i] = "filter_transient"
        return pd.DataFrame(
            {
                "time_s": np.arange(len(counts)) * bin_s,
                "qom_mm_s": means,
                "n_samples": counts,
                "edge": edge,
            }
        )


def _to_2d(a) -> np.ndarray:
    a = np.asarray(a, float)
    return a[:, None] if a.ndim == 1 else a


def speed_from_acceleration(
    acc,
    fs: float,
    unit: str = "m/s^2",
    lo: float = filters.BAND[0],
    hi: float = filters.BAND[1],
    notch_hz: float | None = None,
) -> np.ndarray:
    """Band-limited speed, in mm/s, from acceleration.

    ``acc`` is (n_samples, n_axes). ``unit`` is ``"m/s^2"`` or ``"g"``.
    """
    a = _to_2d(acc)
    if unit == "g":
        a = a * G
    elif unit != "m/s^2":
        raise ValueError(f"unknown acceleration unit {unit!r}; use 'm/s^2' or 'g'")
    a = filters.bandpass(a, fs, lo, hi)
    if notch_hz:
        a = filters.notch(a, fs, notch_hz)
    v = np.cumsum(a, axis=0) / fs
    v = filters.bandpass(v, fs, lo, hi)
    return np.linalg.norm(v, axis=1) * MM_PER_M


def derivative(x, fs: float) -> np.ndarray:
    """Fourth-order central difference along the first axis.

    The two-point central difference that ``numpy.gradient`` computes has the frequency
    response sin(w*dt)/dt rather than w, so it increasingly under-reads towards Nyquist.
    That does not matter at 200 Hz, where the top of the micromotion band is a twentieth of
    Nyquist, but it matters at the 20 Hz common rate where the band edge is Nyquist itself:
    on real optical data the two-point rule loses 4.9 per cent of the quantity of motion
    across that resampling, and this rule loses 2.0 per cent.

    A spectral derivative would be exact for a truly band-limited signal and is not used
    here, because these recordings do not begin and end at the same value and the implied
    wraparound step adds broadband energy that differentiation then amplifies. Measured on
    the same file, it inflated the result by 24 per cent.
    """
    x = _to_2d(x)
    if len(x) < 5:
        return np.gradient(x, 1.0 / fs, axis=0)
    v = np.empty_like(x)
    v[2:-2] = (x[:-4] - 8 * x[1:-3] + 8 * x[3:-1] - x[4:]) * (fs / 12.0)
    v[:2], v[-2:] = v[2], v[-3]
    return v


def speed_from_position(
    pos,
    fs: float,
    unit: str = "mm",
    lo: float = filters.BAND[0],
    hi: float = filters.BAND[1],
) -> np.ndarray:
    """Band-limited speed, in mm/s, from position.

    ``pos`` is (n_samples, n_axes), normally the three coordinates of one optical marker.
    ``unit`` is ``"mm"`` or ``"m"``.
    """
    p = _to_2d(pos)
    if unit == "m":
        p = p * MM_PER_M
    elif unit != "mm":
        raise ValueError(f"unknown position unit {unit!r}; use 'mm' or 'm'")
    p = filters.bandpass(p, fs, lo, hi)
    v = derivative(p, fs)
    v = filters.bandpass(v, fs, lo, hi)
    return np.linalg.norm(v, axis=1)


BANDS = {
    "micromotion": filters.BAND,
    "optical_legacy": filters.OPTICAL_LEGACY_BAND,
}
"""The two conventions in use, by name.

``micromotion`` is 0.3-10 Hz and is the only one that can be applied to every sensor, so it
is the one a cross-collection comparison must use. ``optical_legacy`` is the 10 Hz low-pass
behind the published championship figures; it retains sub-0.3 Hz postural drift and reads
about 15 per cent higher.
"""


def qom(
    data,
    fs: float,
    kind: str = "acceleration",
    unit: str | None = None,
    variant: str = "raw",
    band: str = "micromotion",
    gyro=None,
) -> QomResult:
    """Quantity of motion for one recording.

    Parameters
    ----------
    data
        (n_samples, n_axes) acceleration or position.
    fs
        Measured sampling rate. Use the rate measured from the timestamps, not the nominal
        one; see :func:`micromotion.resample.measured_rate`.
    kind
        ``"acceleration"`` or ``"position"``.
    unit
        Defaults to ``"m/s^2"`` for acceleration and ``"mm"`` for position.
    variant
        ``"raw"``, ``"compensated"`` or ``"tilt_corrected"``.
    band
        ``"micromotion"`` or ``"optical_legacy"``. See :data:`BANDS`.
    gyro
        (n_samples, 3) angular velocity in rad/s. Required by ``tilt_corrected``.
    """
    x = _to_2d(data)
    if band not in BANDS:
        raise ValueError(f"unknown band {band!r}; use one of {sorted(BANDS)}")
    lo, hi = BANDS[band]
    hz = float("nan")

    if lo is None and kind == "acceleration":
        raise ValueError(
            "the 'optical_legacy' band has no lower edge, and an accelerometer cannot be "
            "integrated without one: gravity is a DC term and any residual offset becomes "
            "a ramp. Use band='micromotion' for accelerometer data."
        )

    if kind == "acceleration":
        unit = unit or "m/s^2"
        if variant == "compensated":
            mag = np.linalg.norm(x, axis=1)
            hz = cardiac_peak(mag, fs)
            speed = speed_from_acceleration(x, fs, unit, lo=0.5, hi=hi, notch_hz=hz)
        elif variant == "tilt_corrected":
            if gyro is None:
                raise ValueError("variant 'tilt_corrected' needs a gyroscope signal")
            speed = speed_from_acceleration(
                remove_tilt(x, gyro, fs, unit), fs, "m/s^2", lo, hi
            )
        elif variant == "raw":
            speed = speed_from_acceleration(x, fs, unit, lo, hi)
        else:
            raise ValueError(f"unknown variant {variant!r}")
    elif kind == "position":
        if variant != "raw":
            raise ValueError(
                f"variant {variant!r} applies to accelerometers only; optical position "
                "measures displacement directly, so there is no tilt ambiguity and no "
                "integration drift to compensate"
            )
        speed = speed_from_position(x, fs, unit or "mm", lo, hi)
    else:
        raise ValueError(f"unknown kind {kind!r}; use 'acceleration' or 'position'")

    return QomResult(
        mean_mm_s=float(np.mean(speed)),
        median_mm_s=float(np.median(speed)),
        speed=speed,
        fs=fs,
        variant=variant,
        cardiac_hz=hz,
        n_samples=len(speed),
        edge_samples=filters.edge_transient_samples(fs, lo),
    )


def remove_tilt(acc, gyro, fs: float, unit: str = "m/s^2") -> np.ndarray:
    """Subtract the gravity component that rotation moves between axes.

    The sensor's orientation is tracked by integrating the gyroscope, the gravity vector is
    rotated into the sensor frame at each sample, and what remains is translation. The
    integration drifts, so the estimated gravity direction is high-passed back towards the
    measured one; this is a complementary filter, not an attitude estimator, and it is
    adequate only because the body barely moves.
    """
    a = _to_2d(acc) * (G if unit == "g" else 1.0)
    w = _to_2d(gyro)
    if w.shape[0] != a.shape[0]:
        raise ValueError("gyroscope and accelerometer must have the same length")

    g_hat = np.zeros_like(a)
    g_hat[0] = a[0] / (np.linalg.norm(a[0]) + 1e-12)
    dt = 1.0 / fs
    tau = 1.0                       # s; trust the accelerometer beyond this
    alpha = tau / (tau + dt)
    for i in range(1, len(a)):
        # rotate the previous estimate by -omega*dt (the frame turns, the vector does not)
        gyro_step = g_hat[i - 1] - np.cross(w[i], g_hat[i - 1]) * dt
        meas = a[i] / (np.linalg.norm(a[i]) + 1e-12)
        g = alpha * gyro_step + (1 - alpha) * meas
        g_hat[i] = g / (np.linalg.norm(g) + 1e-12)

    g_mag = float(np.median(np.linalg.norm(a, axis=1)))
    return a - g_hat * g_mag


def tilt_fraction(acc, gyro, fs: float, unit: str = "m/s^2") -> dict:
    """How much of a body-worn accelerometer's quantity of motion is tilt rather than travel.

    An accelerometer cannot tell leaning from moving: rotating in the gravity field produces
    an acceleration with no displacement. Where a gyroscope is present the rotation is known,
    so the gravity component it accounts for can be removed and the two compared.

    Measured on the fNIRS session this returns 1.56, meaning the raw figure is a little over
    half again the translational one. Do not read a single session as a population value; the
    point is that the inflation is measurable rather than assumed, and the assumption in
    circulation was 1.3 to 1.5.
    """
    raw = qom(acc, fs, kind="acceleration", unit=unit).mean_mm_s
    corrected = qom(remove_tilt(acc, gyro, fs, unit), fs,
                    kind="acceleration", unit="m/s^2").mean_mm_s
    return {"raw_mm_s": raw, "translation_mm_s": corrected,
            "inflation": raw / corrected if corrected else float("nan"),
            "tilt_fraction": 1 - corrected / raw if raw else float("nan")}
