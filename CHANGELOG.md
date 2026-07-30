# Changelog

## Unreleased

Added `validate`, a module of checks that fail loudly on the errors this corpus makes silently.
Every check exists because the failure it catches happened, went unnoticed, and produced a
believable wrong number: gaps written as three exact zeros and read as a point on the floor, a
gap running off the end of a series emptying it through the band-pass, timestamps that step
backwards, a documented rate out by a factor of 37, a C3D conversion truncating at the 16-bit
ceiling, a sensor stored at a higher rate than it was sampled at, and the same recording
deposited twice under two names. `validate_series` runs the applicable ones and
`raise_on_error` turns them into a build gate.

Run against the deposited optical records it found 14 series carrying zero-triplet gaps, one of
them 41 per cent of the recording, in a collection that had not been repaired.

`edge_motion` and `settling_time` ask whether a recording is standstill throughout. Deposited
standstill data should contain standstill, and much of this corpus does not: measured against
each recording's own settled interior, all 64 Taqasim recordings and all 365 StillStanding365
days move at more than twice their settled level at an edge, and one HpSp recording ends at
1608 mm/s, which is walking. `settling_time` says how much to trim rather than leaving it to a
guess — the fixed twelve seconds StillStanding365 uses should be about thirty-five.

`longest_finite_span` is exposed alongside it: what to measure over when a gap cannot be
bridged, which had been written out by hand in three separate places.

`HARMONISED_RATE` (100 Hz) is added and `COMMON_RATE` (20 Hz) keeps its value but loses its
justification. The claim that the band stops at 10 Hz so 20 Hz discards nothing does not survive
measurement: a 10 Hz upper edge cannot be realised at 20 Hz, and decimating 34 natively-200 Hz
person-recordings moves quantity of motion by -2.09 per cent at the median and between -0.30 and
-10.57 per cent across recordings. That per-recording spread is a distortion rather than a bias,
so it cannot be corrected away. 100 Hz costs +0.02 per cent and stays inside +/-0.85.

The package docstring still described the 0.3 Hz band that 0.6.0 replaced.

## 0.6.0

First release published to PyPI.

**The lower band edge moved from 0.3 Hz to 0.2 Hz, and every number this package produces changes
with it.** 0.3 was inherited rather than chosen. Swept across seven optical datasets and 665
recordings, the between-dataset spread is 3.2 per cent at 0.15 Hz, 2.1 at 0.20, 2.7 at 0.25, 6.2
at 0.30 and 10.1 at 0.40. 0.2 is a clear optimum and is where the 20 Hz origin dataset stops being
an outlier in either direction. Checked against the failure this edge exists to prevent: on
accelerometer data the mean-to-median speed ratio, which rises when integration drift leaks in, is
2.07 at 0.2 Hz against 2.00 at 0.3 — flat, so integration is safe. Absolute values rise about 73
per cent because more low-frequency content is kept.

One real cost, now under test: the filter transient runs 40 s at each end rather than 27, so a
recording under about two minutes has no clean interior at all.

- `bandpass` warns when the sampling rate exceeds the upper band edge by more than 40:1. Porting
  the analysis scripts turned up a filter that is wrong in published work rather than merely
  different: a 0.1–0.5 Hz third-order band-pass at 250 Hz, written the usual way as
  `butter(3, [lo/ny, hi/ny])` with `filtfilt`, has a largest pole radius of 0.9979 and a measured
  passband gain of 0.84 at 0.15 Hz where it should be 0.99. Nothing raises and nothing looks
  wrong; every amplitude downstream is sixteen per cent low. This package uses second-order
  sections and was never affected, but it accepted such a band silently, and a caller designing
  their own filter deserves to be told. The warning names the fix, which is to decimate first.
- `bandpass` and `lowpass` take a `margin` argument. The upper edge was pinned at 0.99 of Nyquist
  with no way to change it, and the Still Standing corpus uses 0.999 throughout, so no script
  written against that convention could be ported.
- `detect_breaths_adaptive`, added with its benchmark and an explicit note not to default to it.
  The expectation was that rejecting chest rises which never cross an adaptive baseline would beat
  plain peak detection. Measured, it does not: on twelve belt recordings the two agree, and on
  eight chest accelerometers — the case the rejection was meant to help — it over-counts badly,
  10.3 breaths/min against 3.6 for the peak-based detector.
- The balance readers are described by content rather than by a `.tsv` extension they no longer
  have.

164 tests.

## 0.5.0

The integration rule used to turn acceleration into speed is now selectable, and it was hiding a
systematic 0.26 per cent. This package integrated with a rectangle sum; the StillStanding365 and
fNIRS pipelines use the trapezoid rule, and on real phone data the two differ consistently because
the rectangle sum lags the signal by half a sample.

Both are available. The default stays rectangle, and not because it is the better rule — it is
not. It is what the harmonised cross-collection table and every figure derived from it were
computed with, and it reproduces the deposited Taqasim value, 93.140 against 93.091 mm/s, where
trapezoid gives 93.405. Changing the default would silently invalidate numbers already published.
Which rule the project standardises on is left open.

## 0.4.0

Added `group`, for the question everything else here could not ask. The rest of the package
describes one recording or relates a pair; this answers whether a room full of people moved at the
same moments.

- `event_train` — continuous signal to point process, thresholded relative to each person's own
  variability, so a quiet participant and a loud one both register events.
- `coincidence_test` — surrogates shift each person by an independent bounded random offset, so
  every individual keeps their own event rate and local structure and only the between-person
  alignment is destroyed.
- `participation_ratio` — the fraction of people whose motion fell after an event. Sign-based, so
  sensors differing by an order of magnitude do not shift it.

## 0.3.0

Absorbed the quantity-of-motion modules from `musicalgestures`, whose unreleased main branch
carried `band_limited_qom`, `accel_to_speed`, `read_qtm_tsv`, `cop_sway_metrics` and
`respiration_rate`, all credited in their docstrings to the same source study as this package.
Rather than ship a competing fifth implementation of the project's central measure, they move here
and MGT depends on micromotion.

MGT's behaviour is preserved exactly rather than silently reconciled. `band_limited_qom` still
defaults to a 0.3–15 Hz band, differentiates with a two-point difference and does not band-limit
again afterwards; on a 200 Hz optical recording it still returns 5.675 mm/s against 5.455 for this
package's `qom()`. They are different measures and are documented as such.

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
