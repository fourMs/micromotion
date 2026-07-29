"""micromotion: analysis of human micromotion in motion time series.

The measure this package exists for is quantity of motion: the average speed of a body
part, band-limited to 0.3-10 Hz, in millimetres per second. It applies equally to optical
markers, body-worn accelerometers, and force-plate centre of pressure, because the shared
abstraction is the band and not the sensor.

    >>> import micromotion as mm
    >>> rec = mm.read("Standstill2017/mocap_data/A0001.tsv")
    >>> rec.qom().mean_mm_s

See ``deposit/SAMPLING_RATES.md`` for the resampling convention and ``TOOLBOX-PLAN.md`` for
why this is a separate package from ``musicalgestures``.
"""

from . import dynamics
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
from .qom import BANDS, G, QomResult, qom, speed_from_acceleration, speed_from_position
from .record import MotionRecord
from .resample import COMMON_RATE, measured_rate, rate_quality, regularize, to_rate
from .spectral import band_power, cardiac_peak, respiratory_peak

__version__ = "0.1.0"

__all__ = [
    "BAND",
    "BANDS",
    "COMMON_RATE",
    "G",
    "MotionRecord",
    "OPTICAL_LEGACY_BAND",
    "QomResult",
    "band_power",
    "bandpass",
    "cardiac_peak",
    "dynamics",
    "highpass",
    "lowpass",
    "measured_rate",
    "notch",
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
    "respiratory_peak",
    "sniff",
    "speed_from_acceleration",
    "speed_from_position",
    "to_rate",
]
