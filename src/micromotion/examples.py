"""Synthetic recordings, so that everything in the documentation runs with no data.

Every example in this package used to begin `mm.read("mocap_data/A0001.tsv")`, which is
a file the reader does not have. A student following the quickstart got as far as line
one. These generators remove that step: they need no download, no licence and no
network, and they are deterministic, so two people comparing notes see the same numbers.

They are SYNTHETIC and that is the point and the limit. The signals are built from
components whose frequencies are known, which is what makes them useful for showing what
a band does and useless for saying what a body does. No figure or claim about real
micromotion should rest on them. Where the test suite needs a known answer it builds its
inputs the same way, for the same reason.

    >>> import micromotion as mm
    >>> rec = mm.examples.standstill_record()
    >>> r = mm.qom(rec.data, rec.fs, kind=rec.kind, unit=rec.unit)
    >>> round(r.median_mm_s, 2)
    2.29
"""

from __future__ import annotations

import numpy as np

from .record import MotionRecord

__all__ = ["standstill", "standstill_record", "worn_acceleration"]


def standstill(fs: float = 100.0, dur_s: float = 360.0, seed: int = 7) -> np.ndarray:
    """A synthetic head marker: postural sway, breathing, heartbeat and one fidget.

    Returns an ``(n_samples, 3)`` array of millimetres, with the marker about 1.7 m off
    the floor as a real head marker would be. Amplitudes are chosen so that the
    band-limited speed lands in the few-mm/s range quiet standing actually produces,
    which makes the output comparable in ORDER OF MAGNITUDE with real recordings and in
    nothing else.

    The components are deliberately separable: drift below the band, sway near 0.3 Hz,
    breathing at 0.25 Hz, heartbeat at 1.2 Hz, and a three-second fidget at t = 210 s.

    Args:
        fs (float): Sampling rate in Hz.
        dur_s (float): Duration in seconds.
        seed (int): Seed for the drift and the measurement noise.

    Returns:
        np.ndarray: (n_samples, 3) position in millimetres.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * dur_s)
    t = np.arange(n) / fs

    drift = np.cumsum(rng.standard_normal((n, 3)), axis=0) / fs
    drift -= drift.mean(axis=0)
    sway = 0.9 * np.column_stack(
        [
            np.sin(2 * np.pi * 0.31 * t),
            0.7 * np.sin(2 * np.pi * 0.27 * t + 1.1),
            0.2 * np.sin(2 * np.pi * 0.29 * t + 2.0),
        ]
    )
    breathing = 0.6 * np.column_stack(
        [0.3 * np.sin(2 * np.pi * 0.25 * t), 0.3 * np.cos(2 * np.pi * 0.25 * t),
         np.sin(2 * np.pi * 0.25 * t)]
    )
    heartbeat = 0.08 * np.column_stack(
        [np.sin(2 * np.pi * 1.2 * t), 0.5 * np.sin(2 * np.pi * 1.2 * t + 0.4),
         np.sin(2 * np.pi * 1.2 * t + 0.9)]
    )
    noise = 0.03 * rng.standard_normal((n, 3))

    xyz = 4.0 * drift + sway + breathing + heartbeat + noise
    xyz[:, 2] += 1700.0

    fidget = (t > 210.0) & (t < 213.0)
    xyz[fidget, 0] += 9.0 * np.hanning(int(fidget.sum()))
    return xyz


def standstill_record(fs: float = 100.0, dur_s: float = 360.0,
                      seed: int = 7) -> MotionRecord:
    """:func:`standstill` wrapped as a :class:`MotionRecord`, as a reader would return it.

    Use this wherever the documentation would otherwise open a file, so that the example
    exercises the same type real data arrives in.
    """
    return MotionRecord(
        data=standstill(fs=fs, dur_s=dur_s, seed=seed),
        fs=fs,
        channels=["head X", "head Y", "head Z"],
        kind="position",
        unit="mm",
        vertical="Z",
        source="micromotion.examples.standstill_record()",
        meta={"synthetic": True},
    )


def worn_acceleration(fs: float = 100.0, dur_s: float = 360.0,
                      seed: int = 7) -> np.ndarray:
    """The acceleration a body-worn sensor would report for the same motion, in m/s².

    Two derivatives of :func:`standstill`, so the two routes describe ONE body and any
    disagreement between them is the pipeline's rather than the sensor's. Gravity is not
    added: ``variant="raw"`` does not remove it, so including it would test the
    high-pass rather than the measure.

    Returns:
        np.ndarray: (n_samples, 3) acceleration in m/s².
    """
    xyz = standstill(fs=fs, dur_s=dur_s, seed=seed)
    accel_mm = np.gradient(np.gradient(xyz, 1 / fs, axis=0), 1 / fs, axis=0)
    return accel_mm / 1000.0
