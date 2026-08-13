"""Quantity of motion.

The signature measure of this research programme: the average speed of a body part, in
millimetres per second, restricted to the micromotion band.

There is one definition and three named variants. The point of naming them is that they
answer different questions and previously differed only by an undocumented line of filter
code. Any figure or paper should be able to say which variant produced its numbers.

Definition
----------
Band-limit each axis to 0.2-5 Hz, bring it to velocity, band-limit again, take the
Euclidean norm across axes, and report the MEDIAN in mm/s.

The statistic is part of the definition rather than a presentation choice. The two are not
close: on accelerometer data the mean-to-median speed ratio is about 2, and one corpus record
carries a deposited mean of 12.79 mm/s beside a report quoting 11.12 for the same recordings
at the same band, the whole difference being this choice with neither document saying which it
had made. ``speed_from_position`` and ``speed_from_acceleration`` return the speed SERIES and
take no statistic, so the caller decides. Decide explicitly, and say which one a published
figure used.

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
is then 9.80665x too large.
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


def identify_acceleration_unit(acc, tol: float = 0.25) -> str:
    """Which unit an accelerometer file is in, from its own values: ``"g"``, ``"mg"`` or ``"m/s^2"``.

    A device that is mostly stationary is mostly measuring gravity, so the median vector norm of a
    resting recording sits near 1, 981 or 9.81 depending on the convention. Those are three orders
    of magnitude apart, and no plausible unit lies between them, so the identification is
    unambiguous whenever the recording is dominated by gravity.

    THIS EXISTS BECAUSE NO FILE FORMAT IN THIS FIELD DECLARES ITS UNITS. Four accelerometers
    recorded simultaneously on one body in the Oslo corpus stored their values in three different
    conventions, none of them stated anywhere in the files. The method is Finn Upham's, from the
    analysis those recordings were made for: take the mean total acceleration while the participant
    lies still, and read off which constant it is near.

    A UNIT ERROR IS THE HARDEST KIND TO NOTICE. It scales one recording by 9.8 or 981 and leaves
    every correlation, every rank statistic and every reliability estimate untouched, so nothing
    downstream complains. It shows up as a suspiciously ROUND ratio against a known value -- a real
    disagreement is ragged and a unit error is a constant.

    ``tol`` is the fractional distance from a candidate at which the answer is still accepted.
    Raises if the norm is near none of them, which means either the recording is not
    gravity-dominated or the units are something this does not know about; in both cases guessing
    would be worse than stopping.

    >>> identify_acceleration_unit(np.full((100, 3), [0.0, 0.0, 1.0]))
    'g'
    """
    a = _to_2d(np.asarray(acc, float))
    if a.shape[1] != 3:
        raise ValueError(f"need three axes to take a vector norm, got {a.shape[1]}")
    n = float(np.nanmedian(np.linalg.norm(a, axis=1)))
    if not np.isfinite(n) or n <= 0:
        raise ValueError("the median vector norm is not a positive finite number")
    for unit, expect in (("g", 1.0), ("mg", 1000.0), ("m/s^2", G)):
        if abs(n - expect) / expect <= tol:
            return unit
    raise ValueError(
        f"median vector norm {n:.4g} is not near 1 (g), 981 (mg) or {G:.4g} (m/s^2). "
        "Either this recording is not dominated by gravity -- a stationary stretch is what the "
        "method needs -- or the units are not one of these three.")


def _to_2d(a) -> np.ndarray:
    a = np.asarray(a, float)
    return a[:, None] if a.ndim == 1 else a


def velocity_from_acceleration(
    acc,
    fs: float,
    unit: str = "m/s^2",
    lo: float = filters.BAND[0],
    hi: float = filters.BAND[1],
    notch_hz: float | None = None,
    integrate: str = "rectangle",
) -> np.ndarray:
    """Band-limited velocity, per axis, in mm/s, from acceleration.

    ``acc`` is (n_samples, n_axes). ``unit`` is ``"m/s^2"`` or ``"g"``. Returns an array of
    the same shape; :func:`speed_from_acceleration` is its Euclidean norm.

    Use this when a descriptor needs the velocity *vector* rather than its magnitude --
    jerk, spectral measures per axis, anything directional. Computing it from the speed
    alone is not equivalent, and integrating by hand invites a pipeline that differs from
    the rest of the corpus in the filter order or the quadrature rule.

    ``integrate`` selects the quadrature rule, and the choice is not cosmetic. Both are in
    common use and they differ by about 0.26 per cent on real phone data -- small, but a
    systematic bias rather than noise, since the rectangle rule lags the signal by half a
    sample.

    Neither is universally right, so the default is ``"rectangle"``, which is what this
    package's own reference numbers were computed with. Pass ``integrate="trapezoid"`` to
    reproduce a pipeline that used the trapezoid rule; on one deposited value the two give
    93.140 and 93.405 mm/s against a published 93.091.

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
    return filters.bandpass(v, fs, lo, hi) * MM_PER_M


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

    ``acc`` is (n_samples, n_axes). ``unit`` is ``"m/s^2"`` or ``"g"``. This is the norm of
    :func:`velocity_from_acceleration`; see that function for the integration options.
    """
    v = velocity_from_acceleration(acc, fs, unit, lo, hi, notch_hz, integrate)
    return np.linalg.norm(v, axis=1)


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


def velocity_from_position(
    pos,
    fs: float,
    unit: str = "mm",
    lo: float = filters.BAND[0],
    hi: float = filters.BAND[1],
) -> np.ndarray:
    """Band-limited velocity, per axis, in mm/s, from position.

    ``pos`` is (n_samples, n_axes), normally the three coordinates of one optical marker.
    ``unit`` is ``"mm"`` or ``"m"``. Returns an array of the same shape;
    :func:`speed_from_position` is its Euclidean norm.

    Pair with :func:`velocity_from_acceleration` when a descriptor must be computed the same
    way across optical and accelerometer collections: take the velocity from whichever
    function matches the recorded quantity, and everything downstream is identical.
    """
    p = _to_2d(pos)
    if unit == "m":
        p = p * MM_PER_M
    elif unit != "mm":
        raise ValueError(f"unknown position unit {unit!r}; use 'mm' or 'm'")
    p = filters.bandpass(p, fs, lo, hi)
    return filters.bandpass(derivative(p, fs), fs, lo, hi)


def speed_from_position(
    pos,
    fs: float,
    unit: str = "mm",
    lo: float = filters.BAND[0],
    hi: float = filters.BAND[1],
) -> np.ndarray:
    """Band-limited speed, in mm/s, from position.

    ``pos`` is (n_samples, n_axes), normally the three coordinates of one optical marker.
    ``unit`` is ``"mm"`` or ``"m"``. This is the norm of :func:`velocity_from_position`.
    """
    return np.linalg.norm(velocity_from_position(pos, fs, unit, lo, hi), axis=1)


BANDS = {
    "micromotion": filters.BAND,
    "wideband": filters.WIDEBAND,
    "noresp": filters.NORESP_BAND,
    "optical_legacy": filters.OPTICAL_LEGACY_BAND,
}
"""The four conventions in use, by name.

``micromotion`` is 0.2-5 Hz and is the only one every instrument in the corpus can deliver, so
it is the one a cross-collection comparison must use.

``wideband`` is 0.2-10 Hz, for jerk and other high-derivative measures that need the octave the
canonical band gives up. Only on collections sampled fast enough to reach it; check with
:func:`effective_band`, and do not infer the rate from a file's grid, which may be an upsample.

``noresp`` is 0.45-5 Hz, the canonical band with its lower edge above respiration. On a
chest-worn sensor the 0.15-0.45 Hz stretch is dominated by respiratory chest tilt -- gravity
re-projected by the breathing ribcage, a rotation rather than a translation -- so the choice
between this and ``micromotion`` is a purpose decision: keep the respiratory term or exclude
it. See :data:`~micromotion.filters.NORESP_BAND`.

``optical_legacy`` is the 10 Hz low-pass behind the published championship figures. It retains
sub-0.2 Hz postural drift and reads about 15 per cent higher. It is kept because those numbers
are in print, not because it is interchangeable with the others.
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

    Args:
        data (np.ndarray): (n_samples, n_axes) acceleration or position.
        fs (float): Measured sampling rate. Use the rate measured from the timestamps,
            not the nominal one; see :func:`micromotion.resample.measured_rate`.
        kind (str): ``"acceleration"`` or ``"position"``.
        unit (str, optional): Defaults to ``"m/s^2"`` for acceleration and ``"mm"`` for
            position.
        variant (str, optional): ``"raw"``, ``"compensated"`` or ``"tilt_corrected"``.
            Defaults to ``"raw"``.
        band (str, optional): ``"micromotion"``, ``"wideband"``, ``"noresp"`` or
            ``"optical_legacy"``. See :data:`BANDS`. Defaults to ``"micromotion"``.
        gyro (np.ndarray, optional): (n_samples, 3) angular velocity in rad/s. Required by
            ``tilt_corrected``.
        integrate (str, optional): ``"rectangle"`` or ``"trapezoid"``, the quadrature rule
            used to bring acceleration to velocity. The two differ by a systematic fraction
            of a per cent, so the choice is the caller's and belongs in the record of the
            analysis. Defaults to ``"rectangle"``.

    Returns:
        QomResult: The mean and median speed in mm/s, the full speed series, the rate it was
            computed at, the variant, the notched cardiac frequency where one was found, and
            ``edge_samples``, the length of the filter transient at each end.

    Raises:
        ValueError: If ``band`` or ``variant`` is unknown, if ``optical_legacy`` is asked for
            on acceleration, or if ``tilt_corrected`` is asked for without a gyroscope.
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
# An older family, kept because published results depend on it. Also present in
# musicalgestures, which took it from the same source.
#
# These are NOT the same measure as `qom` above. `band_limited_qom` band-limits the *position*,
# differentiates with a two-point difference, and does not band-limit again afterwards -- so the
# speed carries energy above `hi` that the stated band excludes. On a 200 Hz optical recording it
# reads about 2.6 per cent above `qom` on a matched band, and the residual is almost entirely that
# missing second pass; the differentiation rule accounts for 0.05.
#
# Prefer `qom` or `speed_from_position` for new work.
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


def band_limited_qom(pos, fs, lo=filters.BAND[0], hi=filters.BAND[1], order=4, auto_decimate=True):
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

    .. note::

       Defaults follow ``filters.BAND``, like everything else in the package,
       so a number cannot be quoted from a second band by accident.
       :func:`pose_qom` pins its upper edge at 5 Hz because image-space
       landmark jitter dominates above that. Since the band became 0.2-5 Hz
       the two coincide, so that pin currently changes nothing; it is kept
       because it is a statement about pose data rather than about the band.

       **Use** :func:`speed_from_position` **for new work**, which band-limits
       again after differentiating; this one does not, and reads high as a
       result. See the interop guide for the comparison against
       ``musicalgestures``, which carries the same function.

    .. warning::

       The name overstates what this does, and it reads high as a result.
       The *position* is band-limited; the speed derived from it is not.
       Differentiation amplifies the high end, so the velocity carries energy
       above ``hi`` that the stated band excludes, and none of it is removed.

       Measured against :func:`speed_from_position`, which band-limits again
       after differentiating, on a 200 Hz optical recording at a matched
       band: this returns 3.1475 mm/s against 2.9754, i.e. **5.5 per
       cent high**. The decomposition is one-sided -- the differentiation rule
       (first difference here, central difference there) accounts for 0.05 per
       cent, and the missing second band-pass for the remaining 5.5. At the
       respective defaults the gap is larger still, because this function
       band-limits only the position, not the speed derived from it.

       Prefer :func:`qom` or :func:`speed_from_position` for new work. This is
       kept, unchanged, so that figures computed with ``musicalgestures``
       continue to reproduce -- not because it is the better measure. Whichever
       you use, say which.

    Args:
        pos (np.ndarray): Position trajectory of shape (N,) or (N, D) (e.g. D=2
            image coordinates or D=3 mocap coordinates). Non-finite samples are
            linearly interpolated per dimension.
        fs (float): Sampling rate of the trajectory (Hz).
        lo (float, optional): Lower band edge (Hz). Defaults to ``filters.BAND[0]`` (0.2 Hz).
        hi (float, optional): Upper band edge (Hz), clipped to 0.9 x Nyquist.
            Defaults to ``filters.BAND[1]`` (5.0 Hz).
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

    ``lo=None`` gives a pure low-pass with no lower edge, which is the
    :data:`~micromotion.OPTICAL_LEGACY_BAND` convention and what the pre-2020 optical
    standstill studies used. It retains the slow postural drift the corpus band removes, so a
    value computed that way is not comparable with one computed at ``BAND``.

    Raises:
        ValueError: If the band is invalid, i.e. does not satisfy
            `0 < lo < hi <= 0.45*fs` (after `hi` is clipped to 0.9 x Nyquist).
            With `lo=None` only the upper edge is checked.
    """
    from scipy import signal
    pos = np.asarray(pos, float)
    if pos.ndim == 1:
        pos = pos[:, None]
    pos = np.column_stack([_interp_nans(pos[:, i]) for i in range(pos.shape[1])])

    hi_eff = min(hi, 0.9 * fs / 2)
    # `lo=None` is a pure low-pass with no lower edge. That is the package's own
    # OPTICAL_LEGACY_BAND convention, which `filters.bandpass` has always honoured and this
    # function used to reject with a TypeError -- so `band_limited_qom(x, fs, *OPTICAL_LEGACY_BAND)`
    # failed on a constant the package itself exports. Several analyses in the source corpus
    # deliberately work at that band, because it is what the older optical studies used, and they
    # each had to reimplement the filter to do it.
    if lo is None:
        if not (0 < hi_eff <= 0.45 * fs + 1e-9):
            raise ValueError("band must satisfy 0 < hi <= 0.45*fs")
    elif not (0 < lo < hi_eff <= 0.45 * fs + 1e-9):
        raise ValueError("band must satisfy 0 < lo < hi <= 0.45*fs")

    if len(pos) < int(fs) + 5 or not np.isfinite(pos).all():
        return np.array([]), fs

    if auto_decimate and fs / hi_eff >= 40:
        q = int(min(13, fs // (20 * hi_eff)))
        pos = signal.decimate(pos, q, axis=0, zero_phase=True)
        fs_out = fs / q
        if len(pos) < 30:
            return np.array([]), fs_out
        if lo is None:
            sos = signal.butter(2, hi_eff / (fs_out / 2), btype="low", output="sos")
        else:
            sos = signal.butter(2, [lo / (fs_out / 2), hi_eff / (fs_out / 2)],
                                btype="band", output="sos")
        filtered = signal.sosfiltfilt(sos, pos, axis=0)
    else:
        fs_out = fs
        if lo is None:
            b, a = signal.butter(order, hi_eff / (fs / 2), btype="low")
        else:
            b, a = signal.butter(order, [lo / (fs / 2), hi_eff / (fs / 2)], btype="band")
        filtered = signal.filtfilt(b, a, pos, axis=0)
    speed = np.linalg.norm(np.diff(filtered, axis=0), axis=1) * fs_out
    return speed, fs_out


def accel_to_speed(acc, fs, highpass=filters.BAND[0], order=2, normalize_gravity=False):
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
            after integration. Defaults to ``filters.BAND[0]`` (0.2 Hz).
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


def _presence_at_output_rate(present, n_speed, fs, fs_out):
    """Map a per-frame presence mask onto the speed series `band_limited_qom` returned.

    Two things move between the input and the output. `band_limited_qom` may decimate, which it
    does whenever `fs / hi >= 40` -- so at 200 Hz with the default 5 Hz upper edge it always does,
    and 200 Hz optical data is the common case here. And a speed sample needs two positions, so it
    is present only where both of its endpoints were.

    A decimated output sample is treated as present only if every real input sample behind it was.
    That is the conservative direction: it can discard a sample that was mostly real, and it cannot
    admit one that was interpolated. The padding at the end is marked present so that a partial
    final block is judged on its real samples alone.
    """
    q = int(round(fs / fs_out)) if fs_out and fs_out < fs else 1
    if q > 1:
        n_blocks = int(np.ceil(len(present) / q))
        pad = n_blocks * q - len(present)
        padded = np.concatenate([present, np.ones(pad, bool)]) if pad else present
        frames = padded.reshape(n_blocks, q).all(axis=1)
    else:
        frames = present
    both = frames[1:] & frames[:-1] if len(frames) > 1 else np.zeros(0, bool)
    if len(both) >= n_speed:
        return both[:n_speed]
    return np.concatenate([both, np.ones(n_speed - len(both), bool)])


def group_qom(points, fs, lo=filters.BAND[0], hi=filters.BAND[1], normalize="visible",
               **kwargs):
    """
    Mean band-limited quantity of motion over a group of markers/landmarks,
    plus the group's mean speed envelope: each trajectory is passed through
    `band_limited_qom` and the per-trajectory speeds are averaged.

    Source: stillstanding study and Westney-comparisons study (Jensenius) --
    per-body-part QoM (head, shoulders, arms, wrists) from mocap markers and
    pose landmarks.

    .. warning::

       ``normalize`` decides what the divisor is, and the number this returns
       moves with it. Say which was used.

       ``normalize="visible"``, the default, excludes each marker at the frames
       where it was absent and averages over the rest. On twelve markers with a
       realistic dropout pattern, a median of eight visible, it lands within
       0.8 per cent of the unoccluded truth.

       ``normalize="worn"`` averages over every marker at every frame instead.
       Since ``band_limited_qom`` interpolates gaps, an occluded marker then
       contributes a near-zero speed while still counting in the divisor, and
       the result tracks how much the cameras saw: on the same data it reads
       16 to 17 per cent low and its speed series correlates +0.25 to +0.70
       with the per-frame count of visible markers. It is kept so that a figure
       computed that way keeps reproducing; it is bit-for-bit identical on one
       machine and agrees to about 1 part in 10^7 across platforms, since
       ``filtfilt`` is not bit-reproducible between scipy builds.

    Args:
        points (np.ndarray): Trajectories of shape (N, M, D): N frames, M
            markers/landmarks, D spatial dimensions.
        fs (float): Sampling rate (Hz).
        lo (float, optional): Lower band edge (Hz). Defaults to ``filters.BAND[0]`` (0.2 Hz).
        hi (float, optional): Upper band edge (Hz). Defaults to ``filters.BAND[1]`` (5.0 Hz).
        normalize (str, optional): ``"visible"`` averages over the markers
            present in each frame; ``"worn"`` averages over every marker that
            produced a series, which is the pre-1.0 behaviour. Defaults to
            ``"visible"``.
        **kwargs: Passed on to `band_limited_qom`.

    Returns:
        tuple: `(qom, speed, fs_out)` where `qom` is the mean speed across
            markers and time (NaN if no marker yields a valid series), `speed`
            is the group's mean per-frame speed series, and `fs_out` its
            sampling rate. With ``normalize="visible"`` a frame in which no
            marker was present is NaN in `speed`, since nothing was measured
            there.
    """
    import warnings

    if normalize not in ("visible", "worn"):
        raise ValueError("normalize must be 'visible' or 'worn', not %r" % (normalize,))
    points = np.asarray(points, float)
    present = np.isfinite(points).all(axis=2)
    speeds, masks, fs_out = [], [], fs
    for m in range(points.shape[1]):
        sp, fs_out = band_limited_qom(points[:, m, :], fs, lo=lo, hi=hi, **kwargs)
        if len(sp):
            speeds.append(sp)
            masks.append(_presence_at_output_rate(present[:, m], len(sp), fs, fs_out))
    if not speeds:
        return np.nan, np.array([]), fs_out

    if normalize == "worn":
        # Bit-for-bit the pre-1.0 computation, so a published figure still reproduces.
        L = min(len(s) for s in speeds)
        mean_speed = np.mean([s[:L] for s in speeds], axis=0)
        return float(np.mean([s.mean() for s in speeds])), mean_speed, fs_out

    L = min(len(s) for s in speeds)
    stacked = np.where(np.array([m[:L] for m in masks]),
                       np.array([s[:L] for s in speeds]), np.nan)
    with warnings.catch_warnings():
        # A frame in which no marker was visible is legitimately empty, not an error.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_speed = np.nanmean(stacked, axis=0)
        qom = float(np.nanmean(stacked))
    return qom, mean_speed, fs_out


def pose_qom(landmarks, fs, lo=filters.BAND[0], hi=5.0, **kwargs):
    """
    Band-limited quantity of motion of 2-D pose landmarks (px/s): a thin
    wrapper around `group_qom` with the band used for image-space pose
    trajectories (0.2-5 Hz), where higher bands are dominated by landmark
    jitter rather than motion.

    Source: Westney-comparisons study (Jensenius).

    Args:
        landmarks (np.ndarray): Landmark trajectories of shape (N, L, 2) in
            pixels (a single landmark of shape (N, 2) is also accepted).
        fs (float): Sampling rate (Hz, e.g. video frame rate).
        lo (float, optional): Lower band edge (Hz). Defaults to ``filters.BAND[0]`` (0.2 Hz).
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


def normalized_qom(landmarks, fs, scale=None, lo=filters.BAND[0], hi=5.0,
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
        lo (float, optional): Lower band edge (Hz). Defaults to ``filters.BAND[0]`` (0.2 Hz).
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
