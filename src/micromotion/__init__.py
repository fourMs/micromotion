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

from . import align, circular, dynamics, group, posture, validate, windows
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
from .windows import balance as window_balance
from .windows import equalise as equalise_windows
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
    channel_resolution,
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
    identify_acceleration_unit,
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
    peak_from_spectrum,
    spectral_peak,
)

__version__ = "1.9.0"

__all__ = [
    
    
    "BAND",
    "BANDS",
    "COMMON_RATE",
    "FEATURE_NAMES",
    "Finding",
    "G",
    "HARMONISED_RATE",
    "MotionRecord",
    "OPTICAL_LEGACY_BAND",
    "QomResult",
    "WIDEBAND",
    "accel_to_speed",
    "align",
    "apply_lag",
    "axial_rayleigh",
    "balance",
    "band_limited_qom",
    "band_power",
    "band_power_fraction",
    "band_rms",
    "bandpass",
    "bin_series",
    "body_scale",
    "cardiac_peak",
    "channel_rate",
    "channel_resolution",
    "circular",
    "coincidence_test",
    "compare_modality_envelopes",
    "confidence_ellipse_area",
    "convex_hull_area",
    "cop_sway_metrics",
    "derivative",
    "detect_breaths",
    "detect_breaths_adaptive",
    "dfa",
    "dominant_frequency",
    "duplicate_files",
    "dynamics",
    "edge_motion",
    "effective_band",
    "effective_dimensionality",
    "ellipse_area_95",
    "envelope",
    "equalise_windows",
    "equivalence_correlation",
    "event_train",
    "feature_vector",
    "find_transient",
    "gap_report",
    "grid_qom",
    "group",
    "group_qom",
    "highpass",
    "instantaneous_rate",
    "interpolate_gaps",
    "intraclass_correlation",
    "longest_finite_span",
    "lowpass",
    "mean_frequency",
    "measured_rate",
    "mocap",
    "normalized_qom",
    "notch",
    "participation_ratio",
    "path_length",
    "physio",
    "pose_qom",
    "posture",
    "principal_axis_projection",
    "qom",
    "raise_on_error",
    "rate_quality",
    "read",
    "read_ax3",
    "read_balance_board",
    "read_equivital",
    "read_fnirs",
    "read_phone",
    "read_qtm_tsv",
    "read_qualisys",
    "read_sverm",
    "regularize",
    "remove_tilt",
    "respiration_onsets",
    "respiration_rate",
    "respiratory_peak",
    "respiratory_phases",
    "sample_entropy",
    "search_lag",
    "sequential_stability",
    "settling_time",
    "sliding_null",
    "sniff",
    "spatial_extent",
    "spectral_band_fractions",
    "spectral_edges",
    "peak_from_spectrum",
    "spectral_peak",
    "identify_acceleration_unit",
    "speed_from_acceleration",
    "speed_from_position",
    "stabilogram_diffusion",
    "sway_geometry",
    "sway_orientation",
    "sway_texture",
    "tilt_fraction",
    "to_rate",
    "tost_independent",
    "tost_paired",
    "validate",
    "validate_series",
    "velocity_from_acceleration",
    "velocity_from_position",
    "window_balance",
    "xcorr_lag",
]
