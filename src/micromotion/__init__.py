"""micromotion: analysis of human micromotion in motion time series.

The measure this package exists for is quantity of motion: the average speed of a body part,
band-limited to 0.2-5 Hz, in millimetres per second. It applies equally to optical markers,
body-worn accelerometers, and force-plate centre of pressure, because the shared abstraction
is the band and not the sensor.

    >>> import micromotion as mm
    >>> rec = mm.read("Standstill2017/mocap_data/A0001.tsv")     # doctest: +SKIP
    >>> mm.qom(rec.marker("P01"), rec.fs, kind="position").mean_mm_s  # doctest: +SKIP

The package is organised by what you are asking of a signal. :mod:`~micromotion.qom` asks how
much movement there was, :mod:`~micromotion.posture` where it went,
:mod:`~micromotion.dynamics` how it was structured in time, :mod:`~micromotion.spectral` at
what frequencies, :mod:`~micromotion.circular` in which direction, and
:mod:`~micromotion.align` how to put two recordings on one clock. :mod:`~micromotion.io`
reads the files and :mod:`~micromotion.resample` enforces the rate rules.
"""

from . import align, circular, dynamics, group, posture, validate
from . import balance, mocap, physio
from .balance import dfa
from .balance import (
    axial_rayleigh,
    confidence_ellipse_area,
    convex_hull_area,
    cop_sway_metrics,
    principal_axis_projection,
    sample_entropy,
    spatial_extent,
    spectral_edges,
    stabilogram_diffusion,
    sway_orientation,
    sway_texture,
)
from .group import (
    coincidence_test,
    event_train,
    participation_ratio,
    sequential_stability,
    sliding_null,
)
from .mocap import compare_modality_envelopes, dominant_frequency, read_qtm_tsv
from .physio import (respiration_onsets, respiration_rate, respiratory_phases,
                     spectral_band_fractions)
from .qom import (
    accel_to_speed,
    band_limited_qom,
    bin_series,
    body_scale,
    envelope,
    grid_qom,
    group_qom,
    normalized_qom,
    pose_qom,
)
from .align import apply_lag, find_transient, instantaneous_rate, search_lag, xcorr_lag
from .descriptors import effective_dimensionality, intraclass_correlation
from .equivalence import (equivalence_correlation, interpret, tost_independent,
                          tost_paired)
from .features import FEATURE_NAMES, feature_vector
from .filters import (BAND, OPTICAL_LEGACY_BAND, WIDEBAND, bandpass, effective_band, highpass,
                      lowpass, notch)
from .io import (
    read,
    read_ax3,
    read_balance_board,
    read_equivital,
    read_fnirs,
    channel_rate,
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
    velocity_from_acceleration,
    velocity_from_position,
    tilt_fraction,
)
from .record import MotionRecord
from .validate import (
    Finding,
    edge_motion,
    settling_time,
    duplicate_files,
    longest_finite_span,
    raise_on_error,
    validate_series,
)
from .resample import (
    COMMON_RATE,
    HARMONISED_RATE,
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
    detect_breaths_adaptive,
    mean_frequency,
    respiratory_peak,
    spectral_peak,
)

__version__ = "0.15.2"

__all__ = [
    "sliding_null",
    "sequential_stability",
    "participation_ratio",
    "group",
    "event_train",
    "coincidence_test",
    "accel_to_speed",
    "axial_rayleigh",
    "balance",
    "band_limited_qom",
    "bin_series",
    "body_scale",
    "compare_modality_envelopes",
    "confidence_ellipse_area",
    "convex_hull_area",
    "cop_sway_metrics",
    "dfa",
    "dominant_frequency",
    "envelope",
    "grid_qom",
    "group_qom",
    "mocap",
    "normalized_qom",
    "physio",
    "pose_qom",
    "principal_axis_projection",
    "read_qtm_tsv",
    "respiration_rate",
    "respiration_onsets",
    "respiratory_phases",
    "sample_entropy",
    "spatial_extent",
    "spectral_band_fractions",
    "spectral_edges",
    "stabilogram_diffusion",
    "sway_orientation",
    "sway_texture",
    "BAND",
    "BANDS",
    "effective_dimensionality",
    "intraclass_correlation",
    "equivalence_correlation",
    "feature_vector",
    "FEATURE_NAMES",
    "tost_paired",
    "tost_independent",
    "COMMON_RATE",
    "HARMONISED_RATE",
    "Finding",
    "duplicate_files",
    "edge_motion",
    "settling_time",
    "longest_finite_span",
    "raise_on_error",
    "validate",
    "validate_series",
    "G",
    "MotionRecord",
    "OPTICAL_LEGACY_BAND",
    "WIDEBAND",
    "effective_band",
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
    "detect_breaths_adaptive",
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
    "channel_rate",
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
    "velocity_from_acceleration",
    "velocity_from_position",
    "sway_geometry",
    "tilt_fraction",
    "to_rate",
    "xcorr_lag",
]
