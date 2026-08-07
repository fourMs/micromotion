# Changelog

## 1.2.0 — 2026-08-07

### Added
- **`validate.marker_noise`**, the third member of the marker-artefact family and the one
  the other two cannot cover. `zero_triplets` catches a dropped frame written as three
  exact zeros; `implausible_position` catches a reconstruction that lands near the
  laboratory origin without being exactly on it. Neither can see a marker whose every
  sample is plausible and whose sample-to-sample noise is several times what the body
  contributes: it stays at head height, it never leaps, and its median-based quantity of
  motion is perfectly ordinary.

  It is destroyed only by measures that SUM. The recording that motivated this has a
  band-limited quantity of motion of 4.95 mm/s — the corpus median, an unremarkable
  standstill — and a raw sample-to-sample path length of 79.18 mm/s, sixteen times higher.
  Plotted as cumulative distance beside 190 other recordings it was the obvious outlier,
  and it is not a person who moved.

  The test is the ratio of raw path speed to band-limited speed, which is how much of the
  measured path lies outside the band a standing body moves in. Over 193 Sverm
  person-recordings that ratio has a median of 1.39 and a 95th percentile of 2.30, then a
  gap to 4.5, 5.8, 10.7 and 16.0, so the default threshold of 5 sits in empty space rather
  than on a shoulder.

  Deliberately NOT added to `validate_series`, which does not always have a sampling rate
  or millimetre positions to work with. Call it explicitly where both are known.

### Fixed
- `__version__` said 1.1.0 while `pyproject.toml` said 1.1.1, so the two disagreed from the
  1.1.1 release onward and `test_version_matches_pyproject` had been failing.

## 1.1.1 — 2026-08-06

### Changed
- Documentation overhaul: beginner-oriented README, corrected API-page
  rendering (docstring style was set to numpy against Google-style
  docstrings), seven wrongly documented signatures fixed, the
  `descriptors` module given an API page, new demo figures from genuine
  tool output.

### Fixed
- CITATION.cff described the band as 0.2–10 Hz; the implemented band is
  0.2–5 Hz.

## 1.1.0 — 2026-08-06

**`channel_resolution`**, the neighbour of `channel_rate`. That one asks whether a channel updates
fast enough to carry your band; this asks whether it resolves finely enough to carry your amplitude.
It returns the quantisation step, the number of distinct values, the span, and `need / step` where
you say what amplitude you are after.

Written after a secondary accelerometer cost an afternoon. Delsys EMG sensors carry a three-axis
accelerometer whose step is 0.0395 m/s², identical on all twelve axes, against a head acceleration
of 0.033 m/s² median: the whole signal sat inside one step. Every correlation came back at about
0.03, which reads exactly like a real null, and nothing in the data announced the problem. See
`docs/rates.md`.

No behaviour changes. Nothing else in the package is affected.

## 1.0.0 — 2026-08-05

**`group_qom` returns a different number, and the old one was confounded.** Read this before
upgrading if you have published a figure from it.

The mean was over every marker at every frame. `band_limited_qom` interpolates gaps before
filtering, so an occluded marker contributed a near-zero speed while still counting in the divisor,
and the result rose and fell with how much the cameras saw rather than with how much the body moved.
Measured on twelve markers with a realistic dropout pattern, a median of eight visible: the old
default read 16 to 17 per cent low, and its speed series correlated +0.25 to +0.70 with the
per-frame count of visible markers. The same signature was found in a real dance recording where a
median of 14 of 32 markers were visible, at +0.44 to +0.62.

`normalize="visible"` is the default from this release. It excludes each marker at the frames where
that marker was absent and averages over the rest. On the same data it lands within 0.8 per cent of
the unoccluded truth and the correlation falls to near zero. It works through decimation, which
matters because the default band decimates any recording at 200 Hz or above, and a presence mask
left at the input rate would mask the wrong frames.

`normalize="worn"` performs the pre-1.0 computation unchanged. It is bit-for-bit identical on one
machine, across twelve random cases at four sampling rates with and without occlusion, and agrees to
about 1 part in 10^7 across platforms, because `filtfilt` is not bit-reproducible between scipy
builds. That is filter arithmetic rather than a difference in method. Use it to reproduce a
published figure, and say which you used.

A guard band around each gap was tried and rejected: widening the mask by 10, 25 or 50 samples
changes the result by under one per cent and does not improve the correlation, so it would be a
parameter that buys nothing.

`pose_qom` and `normalized_qom` pass `normalize` through and change with it.

Four tests were added. `group_qom` had no occlusion test at all, which is why this survived. They
assert both directions: that the new default recovers the unoccluded value and that the old one does
not, so neither an implementation that ignores occlusion nor one that over-corrects passes.

Why 1.0. The number changes, so semantic versioning requires a major release. `validate.marker_average`
has documented the identical bias one level up, for spatial averaging, since long before this, and
records that four recordings read 33.3 per cent low for years. The concept was in the package; it
was not connected to the function that had it.

## 0.15.2 — 2026-08-05

Documentation only. No code changed, and no number this package returns moves.

Two docstrings told the reader that the upper band edge defaults to 15.0 Hz. It has been 5.0 since
the band moved on 2026-07-31, and those strings render straight into the generated API page, so the
documentation published the old band from a package whose code used the new one. `band_limited_qom`
even contradicted itself, its own note saying defaults follow `filters.BAND`. Both now name the
constant the way the `lo` lines already did.

The doctest for `effective_band` asserted `(0.2, 10.0)` at 100 Hz, where the function returns
`(0.2, 5.0)`. It had been wrong since the same date and could not fail, because doctests were not
collected.

They are collected now. `--doctest-modules` on its own did nothing, since `testpaths` scoped the run
to `tests` and the doctests are in `src`; both paths are named. The one example that reads a real
dataset is marked `+SKIP`, being illustration rather than a test.

`tests/test_qom.py` gains a check that a docstring stating a band edge as a number agrees with the
signature. The existing check compared signatures against `filters.BAND` and passed throughout,
because it never read the prose.

And a note in `band_limited_qom` described `pose_qom` as overriding the upper edge. Since the band
became 0.2-5 Hz the two coincide and it overrides nothing. The pin is kept, being a statement about
pose data rather than about the band, and the note now says so.

## 0.15.1 — 2026-08-04

Documentation only. No code changed, and no number this package returns moves.

### Added: these definitions are duplicated inside archival records, and nothing keeps the copies in step

`docs/conventions.md` now records a coupling that existed and was written down nowhere. Six
deposited data records in the project this package was written for carry a hand-written copy of the
band-pass, and some of them of the fourth-order derivative and the gap-bridging rule as well. That
is deliberate — an archival record has to run for someone who has only that folder, so it may not
import this package — and it makes a fork that nothing keeps in step.

The failure is quiet from both sides. Change `BAND`, `ORDER`, `NYQUIST_MARGIN`, the interpolation
limit or the derivative rule here, and those copies go on computing the old definition; each record
checks itself against its own generator, so it stays internally consistent while drifting away from
the corpus it belongs to. A change to any of those constants is therefore a change to seven places.

Two conventions differ legitimately between those records and are now stated rather than assumed:
some pass acceleration in g and some in m/s², and some integrate with the rectangle rule while
others use the trapezoid. The two rules differ systematically by a fraction of a per cent, which is
why `velocity_from_acceleration` takes `integrate` rather than choosing.

### Added to the wiki

Three traps and one recipe, all from the same reconciliation. A check that reads the first line of a
file has not read the file, which is how internal paths shipped inside twenty-seven archival tables
that an audit had passed every time. A lookup that matches nothing looks exactly like a lookup with
nothing to find. A definition copied for good reasons is still a fork. And a recipe for comparing a
copy against this package numerically, including the three things that hide a real difference:
constants that are inert at the rate you test, a unit you assumed rather than found, and a
quadrature rule that should be reported rather than enforced.

### Fixed

`CITATION.cff` carried the release date of 0.14.x against the version number of 0.15.0. It is a
one-line file that has now drifted twice, most recently in 0.12.4, which was a release made to fix
exactly this.

## 0.15.0 — 2026-08-03

One breaking change, and it is a correctness fix: `read_phone` now returns the accelerometer by
default instead of the fused channel.

### Changed, and it will change every number a caller gets from a phone file

`read_phone(path)` reads `gFx/gFy/gFz` — the accelerometer, converted from g to m/s^2 — where it
previously read `ax/ay/az`. Pass `channel="fused"` for the old behaviour. The unit is m/s^2 either
way, so a caller that does not look will get plausible numbers from the wrong sensor, which is
exactly what happened in the project this package was written for.

Why the default is worth breaking. `ax/ay/az` is not the accelerometer: it is a fusion of
accelerometer, gyroscope and magnetometer, and a fusion cannot output faster than its slowest input,
so it advances at the gyroscope's rate — about 15 Hz on the handsets tested. In a 0.2-5 Hz band most
of what it carries is therefore its own noise floor rather than the body, and at standstill
amplitudes the body sits below that floor. The floor differs between handsets. Three consequences,
all found in deposited data and all now corrected:

* Two Samsung phones recording the same stillness differed by a factor of 2.29, which looked like a
  device calibration difference and was not. On the accelerometer the difference is 0.855, pointing
  the other way, and confounded with the period each phone covered.
* A chest sensor read 1.65 mm/s against an accelerometer value of 6.74, making a head sensor look
  4.6 times more active than the chest when the true ratio is 1.12.
* A device comparison in a pilot dataset gave an S21:S23 ratio of 1.52; on the accelerometer, from
  the same recordings and the same band, it is 1.11.

A file carrying only one of the two channels now raises rather than silently reading whichever it
has, and the error names the other channel and how to ask for it.

### Added

`channel_rate(t, x)` — how often a channel actually advances, as opposed to how often the file has
a row for it. A multi-sensor log is an interleaved union of streams: on one Physics Toolbox file the
rows arrive at about 426 Hz while the accelerometer updates at 51 Hz and the fused channel at 15 Hz.
Taking the row rate for the sensor rate is how a 100 Hz resampling grid and a 12.5 Hz decimation both
came to be quoted as sampling rates. Counted as value changes over the elapsed span, which is robust
to repeated values and to dropouts; the obvious alternative, the reciprocal of the median interval
between changes, returns 680 Hz on that same file because its updates arrive in bursts.

`read_phone` now reports `meta["channel"]` and `meta["channel_rates"]`, the latter giving each
sensor's own rate, and `meta["extra"]` carries whichever acceleration channel was not selected.

## 0.14.1 — 2026-08-03

Documentation only; no behaviour change.

`docs/interop.md` said MGT's overlapping functions were unreleased and invisible to anyone reading
PyPI. MGT-python 1.7.0 was published on 2026-08-03, so that is no longer true and the overlap is now
between two published packages.

The version is declared in `pyproject.toml`, `src/micromotion/__init__.py` and `CITATION.cff`, and
the test suite checks all three agree. It caught this release halfway through, when two of the three
had been updated.

## 0.14.0 — 2026-08-02

One breaking change, and it is a correctness fix: `feature_vector` now requires `kind` and `unit`
instead of defaulting to position and millimetres.

### Changed, and it will break callers that relied on the defaults

`feature_vector(x, fs)` now raises `TypeError`. Pass `kind` and `unit` explicitly.

The docstring has said since 0.13.0 that the two "must be passed and are not guessed", and gave the
reason: the collections do not record the same quantity, so differentiating an acceleration series
as though it were position shifts every descriptor two derivatives up and still returns finite,
plausible-looking numbers. The signature did not enforce it. It defaulted to `kind="position"`,
`unit="mm"`, which reinstated exactly the silent failure the paragraph describes — an accelerometer
recording passed without `kind` was quietly treated as optical position data and came back with a
full set of eleven descriptors, none of them right.

Found by testing the documentation against the code rather than reading either on its own. The
docstring was right and the signature was wrong, so the signature moved.

The migration is mechanical: `kind="position", unit="mm"` for optical data, `kind="acceleration"`
with `unit="g"` or `"m/s^2"` for accelerometer data. The one caller in the Oslo Standstill analysis
layer already passed both, so nothing there changes.

## 0.13.0 — 2026-08-02

Two functions returned wrong answers and are fixed; two new modules. The fixes change numbers, so
anything computed with `axial_rayleigh`'s p-value or with `respiratory_peak` before this release
should be recomputed. Nothing else moves.

### Fixed

`axial_rayleigh` could return a NEGATIVE p-value. It computed Zar's small-Z series,
`exp(-Z)(1 + (2Z - Z^2)/(4n))`, which goes negative once `Z^2 - 2Z` exceeds `4n` — that is, whenever
the axis is strongly preferred, which is precisely what the test exists to detect. It had reached
published output: a corpus report printed a negative probability for every one of six championship
editions. The p-value now comes from `circular.rayleigh`, so the package has one Rayleigh
approximation rather than two. `R` and the mean axis are unchanged.

While there: `balance.axial_rayleigh` and `circular.rayleigh_axial` were reported as disagreeing.
They do not. One takes degrees and the other radians, and given their own units both reproduce the
textbook resultant length of the doubled angles to 2e-16. Feeding radians to the degrees function
drives `R` towards 1, which is what that report was seeing. Both docstrings now say which unit they
take.

`respiratory_peak` did not measure breathing. On the one dataset with a ground truth — sixteen
thoracic belts at 25.6 Hz — it returned a median 7.5 breaths per minute where the belts' own breath
timing gives 16.8, and it ranked participants at Spearman −0.32 against that timing, so it carried
no usable information about who was breathing faster. A periodogram of belt or body motion is red,
so the breathing bump sits on a much larger downward slope and never becomes the global maximum.

Four repairs were measured and rejected before the approach was abandoned: raising the band floor to
0.20 Hz (3.4 breaths per minute short), band-passing before the periodogram (no effect, and it
cannot have one — the maximum inside a band is unaffected by filtering inside that same band), the
most prominent local maximum (Spearman +0.22), and dividing out a fitted power law (+0.26, biased
high). It is now measured in the time domain from `detect_breaths` and returns that rate in Hz. The
signature is unchanged and `window_s` is accepted and ignored. `cardiac_peak` keeps the periodogram,
because on its band the periodogram is right.

`sniff` could not identify Sverm files recorded without a head marker, so `read` could not dispatch
on them. It required `_head_` in the header; it now tests the actual signature, a `Time` column
followed by whole `_x/_y/_z` triples. All 75 Sverm files in the corpus now dispatch, and all 88
Qualisys championship files are unaffected.

### Added

`micromotion.equivalence` — `tost_paired`, `tost_independent`, `equivalence_correlation` and
`interpret`. Two one-sided tests, for stating that an effect is absent rather than failing to show
it is present. Bounds are taken in the data's own units rather than as standardised effect sizes,
because a reader can argue about millimetres per second and cannot argue about Cohen's d. Each
returns one of four verdicts rather than two: `effect`, `equivalent`, `trivial` (detectable and
smaller than the bound) and `inconclusive`. That last is the case a non-significant p-value is
usually reported as "no effect".

`micromotion.feature_vector` and `FEATURE_NAMES` — the canonical eleven descriptors for one
recording: amount and smoothness, frequency and texture, and five sway-geometry measures that need
true position and are `nan` for accelerometer collections. Every cross-recording comparison needs a
fixed set of numbers first, and two analyses that reduce a recording differently are incomparable
for reasons that have nothing to do with the question either is asking. Moved in from a corpus
report, verified bit-identical to it across 732 recordings.

## 0.12.4 — 2026-08-01

Documentation and test hygiene. No change to any measure, so results computed with 0.12.3 stand.

### Fixed

`CITATION.cff` declared 0.7.0 while the package had reached 0.12.3 — five minor versions of drift,
unnoticed because `test_version_matches_pyproject` checked `__version__` against `pyproject.toml`
and nothing checked the third file. That is the file GitHub's "Cite this repository" and Zenodo
read, so anyone citing this package was citing a version that had not existed since July. The test
now covers all three.

The two `intraclass_correlation` tests failed rather than skipped when `statsmodels` was absent.
It is an optional extra, so a plain `pip install micromotion` produced two failures on an
installation that was in fact correct. They now skip, with the reason.

### Added

A "Related toolboxes" section in the README pointing to MGT, ambiscape and musiscape, which share
implementations with this package so a measure computed in one agrees with the same measure
computed in another.

## 0.12.3 — 2026-08-01

Contents identical to 0.12.2 plus the `tomllib` fix below. 0.12.2 was tagged before that fix, so
its release gate failed on Python 3.10 and never reached PyPI; 0.12.3 is the version that ships.

### Fixed

`test_version_matches_pyproject` used `tomllib`, which is stdlib only from 3.11 while this package
supports 3.10. It falls back to a regex rather than skipping — the test exists to catch
`__version__` and `pyproject.toml` drifting apart, and the oldest supported Python is exactly where
a release would otherwise slip through unchecked.

## 0.12.2 — 2026-08-01

### Fixed — statsmodels was undeclared, so `intraclass_correlation` failed on a clean install

`descriptors.intraclass_correlation` imports `statsmodels`, which appeared in no dependency list.
It worked wherever statsmodels happened to be installed and raised `ModuleNotFoundError` everywhere
else — including on every CI runner since 0.12.0, which went unnoticed because that release was
never pushed. It is now an optional extra, `micromotion[mixed]`, since it is the only function in
the package that needs it, and the import raises with that instruction rather than a bare
`ModuleNotFoundError`. The `test` extra installs it so CI exercises the function.

### Added

CI, documentation, PyPI, Python-version and licence badges in the README.

## 0.12.1 — 2026-08-01

### Fixed — the package could not consume its own constant

**`band_limited_qom` now accepts `lo=None`.** `OPTICAL_LEGACY_BAND` is `(None, 10.0)` and
`filters.bandpass` has always documented and honoured a `None` lower edge as a pure low-pass, but
`band_limited_qom` validated its band with `0 < lo < hi` and raised a `TypeError` on `None`. So
`band_limited_qom(x, fs, *OPTICAL_LEGACY_BAND)` failed on a constant this package exports.

This is not hypothetical. A survey of the source corpus on 2026-08-01 found five analyses working
at a low-pass with no lower edge, because that is what the pre-2020 optical standstill studies
used and reproducing them requires it. Each had reimplemented the filter privately, which is
exactly the duplication this package exists to remove.

A low-pass-only value retains the slow postural drift the corpus band removes and is **not
comparable** with a value at `BAND`; the docstring now says so at the point of use.

### Documented

- `band_limited_qom`: an over-high `hi` is clipped to 0.9 × Nyquist rather than rejected. That was
  already the behaviour and is now stated, since it makes the upper-edge validation look absent.

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
