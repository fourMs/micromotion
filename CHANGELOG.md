# Changelog

## 0.2.0

Added the function families an audit against the source study's 159 analysis scripts found
missing.

- `align` — offset recovery between recordings sharing no clock, by cross-correlation or
  direct search, plus instantaneous rate tracking and transient detection. Reproduces the
  fNIRS session's 126.0 s offset, which the deposited sync clap independently confirms.
- `circular` — mean direction, Rayleigh and its axial form, V test, circular and
  circular-linear correlation, axial dispersion.
- `posture` — sway anisotropy and principal axis, 95 per cent confidence ellipse, path
  length, dispersion radius.
- `spectral` — peak with signal-to-noise ratio, band RMS, band power fractions, mean
  frequency, breath detection.
- `dynamics` — approximate entropy, detrended cross-correlation.
- `resample` — short-gap interpolation with a stated ceiling, and gap reporting.
- `qom` — tilt decomposition made public, with `tilt_fraction`.

`xcorr_lag` now differences its inputs by default. Correlating two drifting series and taking
the best lag is the spurious-regression trap: over 200 independent random-walk pairs the
best-lag correlation had a median of 0.47, exceeded 0.5 forty per cent of the time and
reached 0.98 at worst. Differenced, the same pairs never passed 0.16.

89 tests.

## 0.1.0

First release. Filters, quantity of motion, resampling, spectral peaks, dynamics, seven
readers and a common record type, extracted from 159 analysis scripts in which 58 defined
their own band-pass filter and 37 computed quantity of motion.

Restores two validations that had been lost: the time-reversal test returns z = +0.85 on an
AR(1) process and z = -41 on a logistic map. A report stated both; no script implementing
them had survived.
