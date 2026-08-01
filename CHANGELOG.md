# Changelog

## 0.12.0 — 2026-07-31

### Added — the two reductions the corpus kept re-implementing

New module `descriptors`, exporting `effective_dimensionality` and `intraclass_correlation`.

- **`effective_dimensionality(x, rank=True, by=...)`** — how many independent axes a descriptor set
  spans: variance shares, components for 80 and 90 per cent, and the participation ratio of the
  eigenvalue spectrum. Rank-transforms by default (these descriptors are heavy-tailed, and on raw
  values one long tail dominates the first component) and standardises within `by` (otherwise a
  between-edition difference in level reads as shared variance). Degenerate columns are dropped and
  counted rather than silently deflating the ratio.
- **`intraclass_correlation(values, groups)`** — the share of a measure's variance that is the group
  rather than the occasion, as a random-intercept mixed model. Returns `boundary`, which flags a
  variance component estimated at exactly zero: that is the optimiser at the edge of the parameter
  space, not a measured zero, and printing `0.000` from such a fit implies a precision that is not
  there. Log-transforms strictly positive input by default and reports that it did.

Both were previously reimplemented per report with the arithmetic drifting between copies. Migrating
two reports to them reproduced their numbers exactly.

**Name collision worth knowing about:** `group.participation_ratio` is a different quantity — the
fraction of a group whose movement decreased after an event. The participation ratio *of an
eigenvalue spectrum* is deliberately called `effective_dimensionality` instead.

### Fixed

- **The narrow-band warning keyed on the wrong edge.** It compared the sampling rate against the
  *upper* band edge, but conditioning is driven by the lower one — the worked example in its own
  docstring fails at 0.15 Hz, near the bottom of its band. Halving the canonical ceiling in 0.11.0
  therefore doubled the ratio while the conditioning was unchanged, and the warning began firing on
  every high-rate call. Now keyed on the lower edge, threshold retuned.

## 0.11.0 — 2026-07-31

### Changed — the canonical band is now 0.2–5 Hz

`BAND` was `(0.2, 10.0)` and is now `(0.2, 5.0)`. **This changes every quantity of motion this
package produces**, by up to 2.3 per cent, and changes jerk by considerably more.

The reason is deliverability. A band above Nyquist is not a convention but a defect that returns a
plausible number, and the 10 Hz ceiling could not be delivered by the whole corpus:

- the 365-day phone collection carries **fused linear acceleration at about 15 Hz**, so Nyquist is
  7.5 Hz and the ceiling was unreachable on **354 of 355 days**. Those files sit on a 100 Hz grid,
  which is a six-fold upsample, and nothing above 7.5 Hz in them was ever real. (Corrected
  2026-08-01: the *accelerometer* runs at about 50 Hz. The 15 Hz figure is the fusion of
  accelerometer, gyroscope and magnetometer that the app performs to remove gravity, gated by the
  15 Hz gyroscope. The conclusion is unchanged — the deposited channel is the fused one — but the
  limit is the channel, not the sensor.);
- a 20 Hz optical subset has Nyquist exactly at 10 Hz, where the band was silently clamped to 9.9;
- collaborators' 10 Hz data could not carry the band at all.

5 Hz clears every one of these. Band-limited speed barely notices — within 2.3 per cent on every
collection, and 95 per cent of quiet-standing sway power is below 1 Hz. On a 199-recording optical
collection the median moves 0.8 per cent with the ranking at Spearman 0.996.

### Added

- **`WIDEBAND` (0.2–10 Hz)** and a `"wideband"` entry in `BANDS`, for jerk and other
  high-derivative measures that need the octave the canonical band gives up. Jerk at 5 Hz is 37–66
  per cent of its 10 Hz value and the ranking shifts, so jerk belongs here — but **only on
  collections sampled fast enough to deliver it**. Check with `effective_band`, and never infer the
  rate from a file's grid.
- A test asserting the ceiling stays at or below half the slowest sensor in the corpus, so a future
  change cannot quietly make the band uncomputable on a collection again.

### Documented

- **The nominal band edge is not where the filter stops.** A zero-phase fourth-order Butterworth
  rolls off well before its corner: at the canonical 5 Hz ceiling a pure tone survives at **0.249**
  of its amplitude, and full fidelity extends only to about 2.5 Hz. Roll-off tables for both bands
  are in `docs/conventions.md`. Two practical consequences: do not read a band edge as a content
  boundary, and do not probe a filter at its own corner — that test fails when the filter is working.

## 0.10.0 — 2026-07-31

### Added

- **`velocity_from_position` and `velocity_from_acceleration`** return the band-limited velocity
  per axis, in mm/s. `speed_from_position` and `speed_from_acceleration` are now defined as their
  Euclidean norms, so there is one integration path rather than two that can drift apart, and a
  test asserts the identity.

  These exist because descriptors that need the velocity *vector* -- jerk, anything directional --
  previously had to re-implement the integration by hand. In this project that produced two
  incompatible definitions of jerk and one analysis that differentiated an accelerometer series as
  though it were position, shifting every descriptor two derivatives up while still returning
  finite, reasonable-looking numbers.

- **`effective_band(fs)`** reports the band that will actually be applied at a given sampling rate,
  and `bandpass` now warns when it clamps the upper edge to Nyquist instead of clamping silently.
  At 10 Hz the canonical 0.2-10 Hz band becomes 0.2-4.95 Hz, which is a different measurement.

### Notes

- **The 0.2 Hz lower edge is not modality-neutral.** Measured across seven collections, moving the
  high-pass corner from 0.3 to 0.2 Hz changes the reported quantity of motion by 7-18 per cent on
  optical markers and by 25-94 per cent on accelerometers. Optical speed comes from
  *differentiating* position, which suppresses low frequencies; accelerometer speed comes from
  *integrating* acceleration, which amplifies them. The effect is largest on chest-worn sensors,
  where 0.2 Hz sits on the respiration fundamental. Rankings are robust (Spearman 0.94 and above
  where only the band differs); absolute levels are not. State the band when quoting a level from
  an accelerometer.

## 0.9.1 — 2026-07-31

Fixes `__version__`, which the 0.9.0 release left at `0.8.3`: `pyproject.toml` was bumped and the
hardcoded string in `__init__.py` was not, so an installed 0.9.0 reported the previous version.
Anything recording "produced with micromotion X" would have recorded the wrong X. A test now reads
`pyproject.toml` and asserts the two agree, so a release cannot repeat it.

## 0.9.0 — 2026-07-31

### Added

- **`read_phone` accepts asymmetric trimming**: `trim_start_s` and `trim_end_s` override
  `trim_clap_s` per end. The symmetric-only version forced a choice between keeping an opening
  synchronisation clap and throwing away good data at the close. On StillStanding365 day 221 the
  clap reaches 10.29 m/s² against a body maximum of 0.155 — **66× the signal** — so it cannot be
  left in, and removing it symmetrically cost 35 s of standstill at the end. Raises if the trim
  leaves nothing.
- **`read_phone` reads the raw Physics Toolbox export**, not only the cleaned TSV, detecting the
  variant from the first two lines. The app's own CSV has a blank first line, a semicolon
  delimiter, a decimal comma, **Unicode minus U+2212** (so negatives silently become NaN, which
  fails asymmetrically — the sign of a value decides whether it survives), `∞` in `Gain`, a
  phantom trailing column, and identifying GPS columns.
- **`read_phone` reports dropouts**: `gaps`, `segments`, `longest_continuous_s`. Physics Toolbox
  stops logging when backgrounded and resumes with no marker, and its timestamps are the app's own
  accumulated awake-time, so a suspended app loses time from the timeline rather than leaving a
  gap. `fs` is now measured over the longest continuous run — the span average returned 3.8 Hz for
  a 120 Hz file with one 108 s hole, which a band-pass filter would have accepted without complaint.
- Documentation of **coordinate frames** in `formats.md`: which systems are Y-up, how to rotate to
  Z-up, why a Y/Z *swap* is a reflection that reverses every sway direction while leaving all
  magnitudes intact, and the QoM-invariance test that distinguishes the two.
- Documentation of the **Physics Toolbox** format and its traps in `formats.md`.
- Cross-links between `micromotion` and `ambiscape` docs, in both directions.

### Changed

- **`Y_UP_COLLECTIONS` is now empty.** It held `Standstill2019`, whose OptiTrack files were
  rotated to Z-up at source on 2026-07-31. Every optical record in the corpus is now Z-up. The
  constant is kept, empty, because the mechanism is still right: a system's frame is a property of
  how it was configured on the day, not of the system — the same OptiTrack produced Y-up files in
  2019 and Z-up files in 2022.

### Note

If you rotate data at source, empty the compensation in the same change. Ten analysis scripts in
this project carried `VERT={"2019": 1}`; each was correct before the rotation and wrong after it.

## 0.8.3

Added `validate.implausible_position`, for the gap that is not a zero.

`zero_triplets` catches a dropped frame written as three exact zeros. It cannot catch a
reconstruction that lands near the laboratory origin without being exactly on it: those samples are
ordinary finite numbers and pass every sentinel test. The new check is physical instead — a marker
on a standing body stays within a band around its own median height, and anything far outside that
band is a tracking artefact rather than a posture. Markers whose median is not a standing height
are skipped.

Across 1018 optical person-recordings it fires twice: one recording places a head marker 139 mm
below the floor for 63 samples, another at 491 mm against a median of 1716.

The reason it is worth a check of its own is how uneven the damage is. Those 63 samples are 0.5 per
cent of the recording and the median quantity of motion barely registers them, while the sway
extent reads 977 mm where the true figure is about 48. Robustness is a property of a statistic, not
of a dataset.

## 0.8.2

`read_qualisys` now tolerates metadata keys it does not recognise instead of treating the first
one as the end of the header.

QTM 2.0.0 exports open with a `FILE_VERSION` line. The parser stopped at the first unknown key, so
every file from that version read as having no header at all and raised "not a Qualisys export" --
on files that are unambiguously Qualisys exports. Re-exporting an 86-session collection from a
current QTM made all 86 unreadable at once, which is how this was found.

Unknown lines whose key is bare uppercase are now recorded and skipped; a column-name row like
`HF X` still ends the header, because it is not a metadata key. QTM will keep adding fields, and
this is the behaviour that survives it.

## 0.8.1

Added `validate.marker_average`, a check for a failure this corpus had been carrying silently.

Averaging several markers into one position -- a "head" from three head markers -- is routine, and
it launders gaps. Pipelines repair gaps at the end, on the series they are about to measure, which
is too late: the mean of two real coordinates and one zero triplet is a perfectly finite point that
no later gap check will ever flag.

The bias is clean and multiplicative, because markers on one rigid segment move together. With `n`
markers of which `k` carry no data, the average moves at `(n - k) / n` of the true amplitude, so one
dead marker of three understates motion by exactly a third. Partial gaps push the other way, each
gap frame dragging the average half a metre toward the origin and back, inflating the result.

Both were live in one 86-session collection. 22 files had head-marker gaps and four had a marker
never tracked at all; correcting it moved 19 files by more than half a per cent, the largest by 79,
and the maximum quantity of motion fell from 38.9 to 15.2 mm/s. The median moved one per cent --
which is why it went unnoticed. A robust statistic is robust to a real defect too, and a steady
headline number is not evidence that the recordings under it are sound.

## 0.8.0

Added respiratory phase decomposition to `physio`: `respiration_onsets` and `respiratory_phases`.
A breathing rate says how often; these say where in the cycle the body is, which is the more
useful question for standstill work because the post-expiration pause is the moment in the breath
when the body is most nearly still. `respiratory_phases` returns the two half-cycles, high-flow
moments judged both across the recording and within each individual breath, and the pause itself.

Inspiration onset is defined by chest-expansion velocity crossing a threshold taken from the
signal's own distribution, not by a local minimum, and a candidate rise is kept only if it crosses
a heavily low-passed baseline. That rejection step is what makes it usable on quiet standing,
where a belt also records sway, weight shifts and swallows -- all of which produce local minima
with no breath behind them.

This is Finn Upham's method, from his `respy` package (MIT, 2023), reimplemented here on numpy
with permission. The reimplementation is not gratuitous: `respy.Resp_phases` assigns through
`df[col].loc[idx]`, which under pandas copy-on-write does not write, so from pandas 2 onward it
returns all twelve of its phase columns empty without raising anything. Measured against `respy`
0.1.1 on pandas 3.0.3, every phase column came back 0.0 per cent populated.

Validated against the one `respy` function that still works. Across six respiration-belt
recordings and 2331 breaths the two agree on inspiration onset to a median of 0.000 s, 100 per
cent within 0.5 s, with this implementation finding 1.4 per cent more breaths at the recording
boundaries. Phase coverage is physiological: inspiration 21-28 per cent of samples, expiration
70-79, post-expiration pause 26-43.

## 0.7.0

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

184 tests.

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
