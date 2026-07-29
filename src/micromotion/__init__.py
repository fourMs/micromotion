"""micromotion: analysis of human micromotion in motion time series.

The measure this package exists for is quantity of motion: the average speed of a body part,
band-limited to 0.3-10 Hz, in millimetres per second. It applies equally to optical markers,
body-worn accelerometers, and force-plate centre of pressure, because the shared abstraction
is the band and not the sensor.

    >>> import micromotion as mm
    >>> rec = mm.read("Standstill2017/mocap_data/A0001.tsv")
    >>> mm.qom(rec.marker("P01"), rec.fs, kind="position").mean_mm_s

The package is organised by what you are asking of a signal. :mod:`~micromotion.qom` asks how
much movement there was, :mod:`~micromotion.posture` where it went,
:mod:`~micromotion.dynamics` how it was structured in time, :mod:`~micromotion.spectral` at
what frequencies, :mod:`~micromotion.circular` in which direction, and
:mod:`~micromotion.align` how to put two recordings on one clock. :mod:`~micromotion.io`
reads the files and :mod:`~micromotion.resample` enforces the rate rules.
"""

from . import align, circular, dynamics, posture
from .align import apply_lag, find_transient, instantaneous_rate, search_lag, xcorr_lag
from .filters import BAND, OPTICAL_LEGACY_BAND, bandpass, highpass, lowpass, notch
from .io import (
    read,
    read_ax3,
    read_balance_board,
    read_equivital,
    read_fnirs,
    read_phone,
    read_qualisys,
    read_sverm,
    sniff,
)
from .posture import ellipse_area_95, path_length, sway_geometry
from .qom import (
    BANDS,
    G,
    QomResult,
    derivative,
    qom,
    remove_tilt,
    speed_from_acceleration,
    speed_from_position,
    tilt_fraction,
)
from .record import MotionRecord
from .resample import (
    COMMON_RATE,
    gap_report,
    interpolate_gaps,
    measured_rate,
    rate_quality,
    regularize,
    to_rate,
)
from .spectral import (
    band_power,
    band_power_fraction,
    band_rms,
    cardiac_peak,
    detect_breaths,
    mean_frequency,
    respiratory_peak,
    spectral_peak,
)

__version__ = "0.2.0"

__all__ = [
    "BAND",
    "BANDS",
    "COMMON_RATE",
    "G",
    "MotionRecord",
    "OPTICAL_LEGACY_BAND",
    "QomResult",
    "align",
    "apply_lag",
    "band_power",
    "band_power_fraction",
    "band_rms",
    "bandpass",
    "cardiac_peak",
    "circular",
    "derivative",
    "detect_breaths",
    "dynamics",
    "ellipse_area_95",
    "find_transient",
    "gap_report",
    "highpass",
    "instantaneous_rate",
    "interpolate_gaps",
    "lowpass",
    "mean_frequency",
    "measured_rate",
    "notch",
    "path_length",
    "posture",
    "qom",
    "rate_quality",
    "read",
    "read_ax3",
    "read_balance_board",
    "read_equivital",
    "read_fnirs",
    "read_phone",
    "read_qualisys",
    "read_sverm",
    "regularize",
    "remove_tilt",
    "respiratory_peak",
    "search_lag",
    "sniff",
    "spectral_peak",
    "speed_from_acceleration",
    "speed_from_position",
    "sway_geometry",
    "tilt_fraction",
    "to_rate",
    "xcorr_lag",
]
