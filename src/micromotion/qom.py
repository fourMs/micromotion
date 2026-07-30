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
from scipy.integrate import cumulative_trapezoid

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
    integrate: str = "rectangle",
) -> np.ndarray:
    """Band-limited speed, in mm/s, from acceleration.

    ``acc`` is (n_samples, n_axes). ``unit`` is ``"m/s^2"`` or ``"g"``.

    ``integrate`` selects the quadrature rule, and the choice is not cosmetic. Both are in
    use in this corpus: the StillStanding365 and fNIRS pipelines integrate with the trapezoid
    rule, the Stillness2025, Taqasim and 2024 championship pipelines with a rectangle sum.
    They differ by about 0.26 per cent on real phone data, which is small but is a systematic
    bias rather than noise -- the rectangle rule lags the signal by half a sample.

    Neither rule is universally right here, so the default is the one that keeps this
    package's own published numbers self-consistent: ``"rectangle"`` is what the harmonised
    cross-collection table and every figure derived from it were computed with, and it
    reproduces the deposited Taqasim value (93.140 against 93.091 mm/s) where the trapezoid
    rule gives 93.405. Pass ``integrate="trapezoid"`` to reproduce StillStanding365 and
    fNIRS, whose deposited pipelines use it.

    Which rule the project should standardise on is an open question, deliberately not
    settled by this default.
    """
    a = _to_2d(acc)
    if unit == "g":
        a = a * G
    elif unit != "m/s^2":
        raise ValueError(f"unknown acceleration unit {unit!r}; use 'm/s^2' or 'g'")
    a = filters.bandpass(a, fs, lo, hi)
    if notch_hz:
        a = filters.notch(a, fs, notch_hz)
    if integrate == "trapezoid":
        v = cumulative_trapezoid(a, dx=1.0 / fs, initial=0, axis=0)
    elif integrate == "rectangle":
        v = np.cumsum(a, axis=0) / fs
    else:
        raise ValueError(f"unknown rule {integrate!r}; use 'trapezoid' or 'rectangle'")
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
    integrate: str = "rectangle",
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
            speed = speed_from_acceleration(x, fs, unit, lo=0.5, hi=hi, notch_hz=hz,
                                            integrate=integrate)
        elif variant == "tilt_corrected":
            if gyro is None:
                raise ValueError("variant 'tilt_corrected' needs a gyroscope signal")
            speed = speed_from_acceleration(
                remove_tilt(x, gyro, fs, unit), fs, "m/s^2", lo, hi
            )
        elif variant == "raw":
            speed = speed_from_acceleration(x, fs, unit, lo, hi, integrate=integrate)
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


# ---------------------------------------------------------------------------------------
# Absorbed from musicalgestures (MGT) on 2026-07-29, which credited them to the same source
# study as this package. MGT now depends on micromotion rather than carrying its own copies.
#
# These are kept byte-faithful to what MGT published, because their numbers are in use. They
# are NOT the same measure as `qom` above: `band_limited_qom` defaults to a 0.3-15 Hz band,
# differentiates with a two-point difference, and does not band-limit again afterwards. On
# one 200 Hz optical recording it reads 4.0 per cent above `qom` at its own default and 2.6
# per cent above it on a matched band. Prefer `qom` for new work; use these to reproduce
# published MGT results.
# ---------------------------------------------------------------------------------------
def _interp_nans(x):
    """Linearly interpolate non-finite entries of a 1-D array (returns a copy)."""
    x = np.asarray(x, float).copy()
    m = np.isfinite(x)
    if m.sum() > 2 and not m.all():
        x = np.interp(np.arange(len(x)), np.flatnonzero(m), x[m])
    return x


def envelope(x, fs, smooth=1.0, normalize=True):
    """
    Smooth, optionally z-scored envelope of a signal: Savitzky-Golay
    smoothing (order 2, window `smooth` seconds) followed by
    standardisation. Used to compare motion/audio envelopes across sources
    on a common, amplitude-free scale.

    Source: Westney-comparisons study (Jensenius).

    Args:
        x (np.ndarray): Input 1-D signal.
        fs (float): Sampling rate of the signal (Hz).
        smooth (float, optional): Smoothing window in seconds. None or 0 disables
            smoothing. Defaults to 1.0.
        normalize (bool, optional): If True, z-score the result. Defaults to True.

    Returns:
        np.ndarray: The smoothed (and optionally z-scored) envelope, same length
            as the input.
    """
    from scipy.signal import savgol_filter
    x = np.asarray(x, float)
    if smooth:
        w = max(3, int(smooth * fs) | 1)
        if len(x) > w:
            x = savgol_filter(x, w, 2)
    if normalize:
        x = (x - x.mean()) / (x.std() + 1e-9)
    return x


def bin_series(x, fs, bin_s=1.0):
    """
    Mean of consecutive, non-overlapping bins of a signal (e.g. a per-second
    quantity-of-motion envelope from a per-frame speed series). Trailing
    samples that do not fill a whole bin are dropped.

    Source: stillstanding study (Jensenius); also used in the
    Westney-comparisons study as a per-second envelope.

    Args:
        x (np.ndarray): Input 1-D signal.
        fs (float): Sampling rate of the signal (Hz).
        bin_s (float, optional): Bin length in seconds. Defaults to 1.0.

    Returns:
        np.ndarray: One mean value per bin (empty if the signal is shorter than
            two bins).
    """
    x = np.asarray(x, float)
    step = max(1, int(round(fs * bin_s)))
    n = len(x) // step
    if n < 2:
        return np.array([])
    return x[:n * step].reshape(n, step).mean(axis=1)


def band_limited_qom(pos, fs, lo=0.3, hi=15.0, order=4, auto_decimate=True):
    """
    Band-limited quantity of motion from a position trajectory: the position
    is band-pass filtered (zero phase) to `[lo, hi]` Hz and the QoM is the
    per-frame speed, i.e. the Euclidean norm of the first difference times
    the sampling rate (units of the input per second).

    For very low bands relative to the sampling rate (band edge below about
    fs/40), a direct high-order band-pass is numerically fragile; in that
    regime the trajectory is first decimated (zero phase) so the band sits
    comfortably in the new Nyquist range, then filtered with a second-order
    section (SOS) band-pass. This is the "slow sway" regime (e.g. 0.1-0.5 Hz
    postural sway from 100 Hz mocap). Set `auto_decimate=False` to force the
    direct filter.

    Source: stillstanding study and Westney-comparisons study (Jensenius) --
    this unifies the band-limited QoM cores used on mocap markers (mm),
    MediaPipe landmarks (px) and slow postural sway across both studies.

    .. warning::

       The name overstates what this does, and it reads high as a result.
       The *position* is band-limited; the speed derived from it is not.
       Differentiation amplifies the high end, so the velocity carries energy
       above ``hi`` that the stated band excludes, and none of it is removed.

       Measured against :func:`speed_from_position`, which band-limits again
       after differentiating, on a 200 Hz optical recording at a matched
       0.3-10 Hz band: this returns 3.1475 mm/s against 2.9754, i.e. **5.5 per
       cent high**. The decomposition is one-sided -- the differentiation rule
       (first difference here, central difference there) accounts for 0.05 per
       cent, and the missing second band-pass for the remaining 5.5. At the
       respective defaults the gap is larger still, because this function
       defaults to 0.3-15 Hz.

       Prefer :func:`qom` or :func:`speed_from_position` for new work. This is
       kept, unchanged, so that figures computed with ``musicalgestures``
       continue to reproduce -- not because it is the better measure. Whichever
       you use, say which.

    Args:
        pos (np.ndarray): Position trajectory of shape (N,) or (N, D) (e.g. D=2
            image coordinates or D=3 mocap coordinates). Non-finite samples are
            linearly interpolated per dimension.
        fs (float): Sampling rate of the trajectory (Hz).
        lo (float, optional): Lower band edge (Hz). Defaults to 0.3.
        hi (float, optional): Upper band edge (Hz), clipped to 0.9 x Nyquist.
            Defaults to 15.0.
        order (int, optional): Butterworth order of the direct band-pass.
            Defaults to 4.
        auto_decimate (bool, optional): Enable the decimate+SOS low-band regime.
            Defaults to True.

    Returns:
        tuple: `(speed, fs_out)` where `speed` is the per-frame speed series
            (length N-1, or shorter when decimated) and `fs_out` is its
            sampling rate (equal to `fs` unless decimated). `speed` is empty
            (and `fs_out` equals the input `fs`) when the input has fewer
            than `int(fs) + 5` samples, or when it still contains non-finite
            samples after per-dimension interpolation (i.e. a dimension had
            fewer than 3 finite samples to interpolate from). In the
            auto-decimate regime, `speed` is also empty (with `fs_out` the
            decimated rate) when decimation leaves fewer than ~30 samples --
            too few for a stable SOS band-pass.

    Raises:
        ValueError: If the band is invalid, i.e. does not satisfy
            `0 < lo < hi <= 0.45*fs` (after `hi` is clipped to 0.9 x Nyquist).
    """
    from scipy import signal
    pos = np.asarray(pos, float)
    if pos.ndim == 1:
        pos = pos[:, None]
    pos = np.column_stack([_interp_nans(pos[:, i]) for i in range(pos.shape[1])])

    hi_eff = min(hi, 0.9 * fs / 2)
    if not (0 < lo < hi_eff <= 0.45 * fs + 1e-9):
        raise ValueError("band must satisfy 0 < lo < hi <= 0.45*fs")

    if len(pos) < int(fs) + 5 or not np.isfinite(pos).all():
        return np.array([]), fs

    if auto_decimate and fs / hi_eff >= 40:
        q = int(min(13, fs // (20 * hi_eff)))
        pos = signal.decimate(pos, q, axis=0, zero_phase=True)
        fs_out = fs / q
        if len(pos) < 30:
            return np.array([]), fs_out
        sos = signal.butter(2, [lo / (fs_out / 2), hi_eff / (fs_out / 2)],
                            btype="band", output="sos")
        filtered = signal.sosfiltfilt(sos, pos, axis=0)
    else:
        fs_out = fs
        b, a = signal.butter(order, [lo / (fs / 2), hi_eff / (fs / 2)], btype="band")
        filtered = signal.filtfilt(b, a, pos, axis=0)
    speed = np.linalg.norm(np.diff(filtered, axis=0), axis=1) * fs_out
    return speed, fs_out


def accel_to_speed(acc, fs, highpass=0.3, order=2, normalize_gravity=False):
    """
    Integrated speed from a 3-axis accelerometer: each axis is high-pass
    filtered (removing gravity and DC), integrated to velocity, high-pass
    filtered again (killing integration drift), and the speed is the
    Euclidean norm of the velocity (m/s for input in m/s^2).

    Source: stillstanding study (Jensenius) -- the "corpus method" for
    integrated quantity of motion from chest-worn accelerometers.

    Args:
        acc (np.ndarray): Acceleration of shape (N, 3) in m/s^2 (or raw counts
            with `normalize_gravity=True`).
        fs (float): Sampling rate (Hz).
        highpass (float, optional): High-pass cutoff (Hz) used both before and
            after integration. Defaults to 0.3.
        order (int, optional): Butterworth order of the high-pass filters.
            Defaults to 2.
        normalize_gravity (bool, optional): If True, rescale the raw input so
            that the median vector magnitude equals 1 g (9.80665 m/s^2) before
            filtering -- useful for uncalibrated sensors whose resting output
            should be gravity. Defaults to False.

    Returns:
        np.ndarray: Speed series of length N (m/s).
    """
    from scipy.signal import butter, filtfilt
    G = 9.80665
    acc = np.asarray(acc, float)
    if normalize_gravity:
        norm = np.linalg.norm(acc, axis=1)
        acc = acc / np.median(norm) * G
    b, a = butter(order, highpass / (fs / 2), btype="high")
    acc = filtfilt(b, a, acc, axis=0)
    vel = np.cumsum(acc, axis=0) / fs
    vel = filtfilt(b, a, vel, axis=0)
    return np.linalg.norm(vel, axis=1)


def group_qom(points, fs, lo=0.3, hi=15.0, **kwargs):
    """
    Mean band-limited quantity of motion over a group of markers/landmarks,
    plus the group's mean speed envelope: each trajectory is passed through
    `band_limited_qom` and the per-trajectory speeds are averaged.

    Source: stillstanding study and Westney-comparisons study (Jensenius) --
    per-body-part QoM (head, shoulders, arms, wrists) from mocap markers and
    pose landmarks.

    Args:
        points (np.ndarray): Trajectories of shape (N, M, D): N frames, M
            markers/landmarks, D spatial dimensions.
        fs (float): Sampling rate (Hz).
        lo (float, optional): Lower band edge (Hz). Defaults to 0.3.
        hi (float, optional): Upper band edge (Hz). Defaults to 15.0.
        **kwargs: Passed on to `band_limited_qom`.

    Returns:
        tuple: `(qom, speed, fs_out)` where `qom` is the mean speed across
            markers and time (NaN if no marker yields a valid series), `speed`
            is the group's mean per-frame speed series, and `fs_out` its
            sampling rate.
    """
    points = np.asarray(points, float)
    speeds, fs_out = [], fs
    for m in range(points.shape[1]):
        sp, fs_out = band_limited_qom(points[:, m, :], fs, lo=lo, hi=hi, **kwargs)
        if len(sp):
            speeds.append(sp)
    if not speeds:
        return np.nan, np.array([]), fs_out
    L = min(len(s) for s in speeds)
    mean_speed = np.mean([s[:L] for s in speeds], axis=0)
    return float(np.mean([s.mean() for s in speeds])), mean_speed, fs_out


def pose_qom(landmarks, fs, lo=0.3, hi=5.0, **kwargs):
    """
    Band-limited quantity of motion of 2-D pose landmarks (px/s): a thin
    wrapper around `group_qom` with the band used for image-space pose
    trajectories (0.3-5 Hz), where higher bands are dominated by landmark
    jitter rather than motion.

    Source: Westney-comparisons study (Jensenius).

    Args:
        landmarks (np.ndarray): Landmark trajectories of shape (N, L, 2) in
            pixels (a single landmark of shape (N, 2) is also accepted).
        fs (float): Sampling rate (Hz, e.g. video frame rate).
        lo (float, optional): Lower band edge (Hz). Defaults to 0.3.
        hi (float, optional): Upper band edge (Hz). Defaults to 5.0.
        **kwargs: Passed on to `band_limited_qom`.

    Returns:
        tuple: `(qom, speed, fs_out)` as in `group_qom`.
    """
    landmarks = np.asarray(landmarks, float)
    if landmarks.ndim == 2:
        landmarks = landmarks[:, None, :]
    return group_qom(landmarks, fs, lo=lo, hi=hi, **kwargs)


def body_scale(landmarks, upper=(11, 12), lower=(23, 24)):
    """
    Body-size scale (in the landmarks' own units, e.g. pixels) as the median
    torso length: the distance from the midpoint of the `upper` landmarks
    (shoulders) to the midpoint of the `lower` landmarks (hips). The torso
    length is preferred over shoulder width because it stays robust in a
    profile view, where the shoulder width collapses.

    The default indices are MediaPipe Pose landmarks (11/12 shoulders,
    23/24 hips).

    Source: Westney-comparisons study (Jensenius).

    Args:
        landmarks (np.ndarray): Landmark trajectories of shape (N, L, C) with
            C >= 2; only the first two coordinates are used.
        upper (tuple, optional): Indices of the two shoulder landmarks.
            Defaults to (11, 12).
        lower (tuple, optional): Indices of the two hip landmarks.
            Defaults to (23, 24).

    Returns:
        float: Median torso length (NaN if no finite frames).
    """
    landmarks = np.asarray(landmarks, float)
    um = (landmarks[:, upper[0], :2] + landmarks[:, upper[1], :2]) / 2
    lm = (landmarks[:, lower[0], :2] + landmarks[:, lower[1], :2]) / 2
    d = np.linalg.norm(um - lm, axis=1)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else np.nan


def normalized_qom(landmarks, fs, scale=None, lo=0.3, hi=5.0,
                   upper=(11, 12), lower=(23, 24), **kwargs):
    """
    Body-scale-normalised quantity of motion (body-lengths per second):
    the pose QoM divided by the performer's own body scale (median torso
    length, see `body_scale`). Being dimensionless, this is invariant to
    camera framing/zoom and comparable across recordings.

    Source: Westney-comparisons study (Jensenius) -- framing-invariant
    with/without-audience comparison of a pianist's motion.

    Args:
        landmarks (np.ndarray): Landmark trajectories of shape (N, L, 2).
        fs (float): Sampling rate (Hz).
        scale (float, optional): Precomputed body scale. Defaults to None (which
            computes `body_scale(landmarks, upper, lower)`).
        lo (float, optional): Lower band edge (Hz). Defaults to 0.3.
        hi (float, optional): Upper band edge (Hz). Defaults to 5.0.
        upper (tuple, optional): Shoulder landmark indices for `body_scale`. Defaults to (11, 12).
        lower (tuple, optional): Hip landmark indices for `body_scale`. Defaults to (23, 24).
        **kwargs: Passed on to `band_limited_qom`.

    Returns:
        tuple: `(qom, speed, fs_out)` as in `group_qom`, with both `qom` and
            `speed` divided by the body scale. When `scale` is non-finite
            (e.g. `body_scale` found no finite torso-length sample) or not
            strictly positive (degenerate, coincident upper/lower landmarks),
            division would otherwise silently propagate NaN/inf through
            `qom` and `speed`; instead both are explicitly returned as NaN
            (`qom` as a NaN scalar, `speed` as an all-NaN array of the same
            shape) so the invalid-scale case is unambiguous rather than
            merely inferred from the arithmetic.
    """
    landmarks = np.asarray(landmarks, float)
    if scale is None:
        scale = body_scale(landmarks, upper=upper, lower=lower)
    qom, speed, fs_out = pose_qom(landmarks, fs, lo=lo, hi=hi, **kwargs)
    if not np.isfinite(scale) or scale <= 0:
        return float("nan"), np.full_like(speed, np.nan), fs_out
    return qom / scale, speed / scale, fs_out


def grid_qom(frames, grid=(6, 4), region=(0.0, 1.0, 0.0, 1.0), threshold=8.0):
    """
    Spatial grid quantity of motion from a stack of grayscale frames: the
    absolute inter-frame difference is thresholded (small differences set to
    zero to suppress sensor noise) and averaged within each cell of a
    `grid[0]` x `grid[1]` grid laid over `region`, yielding one motion time
    series per cell plus a per-cell mean-motion heatmap.

    Source: Westney-comparisons study (Jensenius) -- audience-region motion
    mapping in a concert hall.

    Args:
        frames (np.ndarray): Grayscale frames of shape (T, H, W).
        grid (tuple, optional): Grid size (columns, rows). Defaults to (6, 4).
        region (tuple, optional): Region of interest as fractions
            (x0, x1, y0, y1) of the frame. Defaults to the full frame.
        threshold (float, optional): Absolute-difference threshold below which
            pixel changes are zeroed (0-255 scale). Defaults to 8.0.

    Returns:
        tuple: `(series, heat)` where `series` has shape (T-1, rows*cols)
            (cells in row-major order) and `heat` has shape (rows, cols) with
            each cell's time-mean motion.

    Raises:
        ValueError: If `frames` is not 3-D (T, H, W), as in
            `_motionanalysis.motiongram_data`.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError("grid_qom expects frames of shape (T, H, W)")
    T, H, W = frames.shape
    gx, gy = grid
    x0, x1, y0, y1 = region
    xs = np.linspace(int(x0 * W), int(x1 * W), gx + 1).astype(int)
    ys = np.linspace(int(y0 * H), int(y1 * H), gy + 1).astype(int)
    d = np.abs(np.diff(frames, axis=0))
    d[d < threshold] = 0.0
    series = np.empty((T - 1, gy * gx), dtype=np.float32)
    for r in range(gy):
        for c in range(gx):
            cell = d[:, ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            series[:, r * gx + c] = cell.mean(axis=(1, 2))
    heat = series.mean(axis=0).reshape(gy, gx)
    return series, heat
