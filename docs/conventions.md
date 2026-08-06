# The three bands

There are three quantity-of-motion conventions here and they are not interchangeable. Reporting
a number without saying which one produced it is the single most common way results in this
field stop being comparable. The three are named in `mm.BANDS` and are passed to `qom` by name,
as `band="micromotion"`, `band="wideband"` or `band="optical_legacy"`.

## `micromotion`—0.2 to 5 Hz

A band-pass, and the package default, exported as `mm.BAND`. It is the only convention that can
be applied to every sensor, so it is the one any cross-collection comparison must use.

## `optical_legacy`—10 Hz low-pass, no lower edge

Exported as `mm.OPTICAL_LEGACY_BAND`, which is `(None, 10.0)`. It retains sub-0.2 Hz postural
drift. For optical position that drift is real movement, and this is the convention behind most
published standstill figures. Every band-limiting entry point accepts `lo=None` for it.

**A value computed here is not comparable with one at `BAND`, and the gap is large.** On one
corpus, moving the lower edge from 0.3 Hz to 0.2 Hz alone raised per-marker levels about ten per
cent; removing the lower edge entirely raises them much further, because on an accelerometer
roughly half the velocity power sits below 0.3 Hz. Reporting a legacy-band figure beside a
canonical-band one without saying which is which is the most common way this corpus has produced
an apples-to-oranges comparison. One report quoted a three per cent agreement between two
numbers computed at different bands and called it striking.

## `wideband`—0.2 to 10 Hz

Exported as `mm.WIDEBAND`. It is for jerk and other high-derivative measures that need the
octave the canonical band gives up.

Use it only where `effective_band(fs)` confirms the rate supports it, and never infer the rate
from a file's grid, since a uniform grid may be an upsample of a much slower sensor, and this
corpus contains 364 days where it is. A quantity computed here is not comparable with one
computed at `BAND`, and is not computable at all on the slower collections.

**Check deliverability where the value is produced and again where it is consumed.** A scan that
computes a wideband jerk for every recording regardless of rate, leaving the gate downstream,
will be defeated by the obvious downstream filter: "the value is not null" readmits every
recording the gate was meant to exclude. Carry the measured sensor rate alongside the value and
test against it.

**Deliverability is a property of a channel, not of a collection.** A collection whose deposited
channel cannot carry `WIDEBAND` may have a faster channel on the same instrument that can, as in
the fusion discussion in [Sampling rates](rates.md). "This collection has no jerk" and "this
channel has no jerk" are different statements, and only the second is usually true.

## Why the lower edge is optional for position and mandatory for acceleration

An optical system measures position directly, so slow drift is a genuine displacement and
keeping it is a choice.

An accelerometer cannot offer that choice. Gravity is a DC term, and integrating a signal with
any residual offset produces a ramp that swamps the result. The package therefore raises on
`band="optical_legacy"` for acceleration rather than return a plausible-looking number.

```python
mm.qom(acc, fs, kind="acceleration", band="optical_legacy")
# ValueError: the 'optical_legacy' band has no lower edge, and an accelerometer
# cannot be integrated without one...
```

## Why 0.2 Hz at the bottom

The lower edge was chosen by measurement rather than by convention. Swept across seven optical
datasets and 665 recordings, the between-dataset spread is:

| lower edge | spread |
|---|---|
| 0.15 Hz | 3.2 % |
| **0.20 Hz** | **2.1 %** |
| 0.25 Hz | 2.7 % |
| 0.30 Hz | 6.2 % |
| 0.40 Hz | 10.1 % |

A clear optimum, and also where the 20 Hz origin dataset stops being an outlier in either
direction.

The edge interacts with the statistic, which is easy to miss. At 0.2 Hz the *median* speed
converges to 2.1 per cent and the mean only to 4.6; at 0.3 Hz it is the other way round. Report
the median at this band. A mean here would be looser than what it replaced.

An edge this close to DC has to be checked against integration drift, since that is the failure
it exists to prevent. It survives: on accelerometer data the mean-to-median speed ratio, which
climbs when drift leaks in, is 2.07 at 0.2 Hz against 2.00 at 0.3, which is flat. The filter
transient does lengthen to 40 seconds at each end rather than 27, so a recording shorter than
about two minutes has no clean interior at all. That rules the band out for short trials; it does
not affect standstill protocols, which run six minutes or more.

## Why 5 Hz at the top

**Because that is what the instruments can deliver.** A band above Nyquist is not a convention,
it is a defect that returns a plausible number, and the ceiling has to be one every collection
can support:

| instrument | sampling | Nyquist | 5 Hz? |
|---|---|---|---|
| phone, fused linear acceleration | ~15 Hz | 7.5 | yes |
| optical, 20 Hz subset | 20 Hz | 10 | yes |
| collaborators' audience data | 10 Hz | 5 | at Nyquist |
| optical and inertial, rest | 100–256 Hz | 50–128 | yes |

A 10 Hz ceiling fails the first row on 354 of 355 days and sits exactly on Nyquist for the
second, where it is silently clamped to 9.9.

The phone case is the instructive one, and it has two layers. Those files are stored on a 100 Hz
grid, which is a six-fold upsample, so nothing above 7.5 Hz in them is real. The reason is not
that the accelerometer is slow, since measured across the year it runs at about 50 Hz. The
deposited channel is *linear acceleration*, which the logging app derives by fusing
accelerometer, gyroscope and magnetometer to remove gravity, and a fusion runs no faster than
its slowest input, which here is the 15 Hz gyroscope. So the constraint is the channel, not the
sensor, and a pipeline built on the raw accelerometer channel would have roughly three times the
bandwidth.

**What the ceiling costs.** Measured over 466 person-recordings from seven collections,
computing both bands from the same array so that only the band differs. The typical recording
barely notices—the median difference is 1.3 per cent, and 95 per cent of quiet-standing sway
power lies below 1 Hz anyway—but this is a distribution, not a bound, and it has a tail:

| collection | median | 90th percentile | maximum |
|---|---|---|---|
| Championships (optical, 100–200 Hz) | 2.83 % | 5.66 % | 9.07 % |
| Taqasim | 1.66 % | 2.67 % | 4.16 % |
| Stillness2025 | 1.11 % | 1.70 % | 2.29 % |
| HpSp | 1.10 % | 3.05 % | 5.32 % |
| Sverm | 0.98 % | 2.93 % | 7.03 % |
| StillStanding365 | 0.61 % | 1.79 % | 2.38 % |
| all | 1.28 % | 3.24 % | 9.07 % |

Nineteen per cent of recordings exceed 2.3 per cent and the largest exceeds it fourfold, so the
median is not a bound. The difference is largest exactly where the ceiling matters most: on the
optical championships the *median* is above 2.3 per cent and 50 of 80 sampled recordings exceed
it, because markers sampled at 100 to 200 Hz genuinely resolve the octave between 5 and 10 Hz,
while a chest accelerometer that never delivered that octave loses nothing by giving it up.

Rankings survive this, since it is a small and fairly uniform shrinkage rather than a
reordering. Quoted absolute levels do not.

**What it costs that matters.** Jerk is two derivatives higher and lives in the octave being
given up: at 5 Hz it is 37 to 66 per cent of its 10 Hz value, and the ranking shifts as well.
Jerk is therefore computed at `WIDEBAND`, on collections fast enough to deliver it, and nowhere
else. On the phone collection the wider jerk was never real, and computing it there inflated the
figure by 18 to 27 per cent with interpolation.

## The nominal edge is not where the filter stops

A band written as 0.2–5 Hz does not pass everything below 5 Hz and reject everything above. It
is a zero-phase fourth-order Butterworth, applied forward and backward, and it is rolling off
well before its nominal corner. Measured on pure tones of known amplitude, recovering the
analytic mean speed:

![Fraction of a pure tone's analytic speed recovered at each band, against tone frequency: the canonical band is at 0.25 by 5 Hz while the wideband is still at 0.99](img/band-rolloff.png)

| tone | canonical `BAND` 0.2–5 Hz | `WIDEBAND` 0.2–10 Hz |
|---|---|---|
| 0.30 Hz | 0.951 | 0.939 |
| 0.50 Hz | 1.000 | 0.999 |
| 1.00 Hz | 1.000 | 1.000 |
| 2.00 Hz | 0.999 | 1.000 |
| 2.50 Hz | 0.997 | 0.999 |
| 3.00 Hz | 0.982 | 1.000 |
| 3.50 Hz | 0.924 | 1.000 |
| 4.00 Hz | 0.773 | 1.000 |
| 4.50 Hz | 0.513 | 0.998 |
| 5.00 Hz | **0.249** | 0.993 |
| 6.00 Hz | 0.030 | 0.976 |
| 7.00 Hz | 0.003 | 0.912 |
| 8.00 Hz | 0.000 | 0.758 |

**Read the 5.00 Hz row.** A pure tone exactly at the canonical ceiling survives at a quarter of
its true amplitude. Full fidelity, better than 99 per cent, extends only to about 2.5 Hz, so the
usable passband is roughly half the nominal upper edge. The same holds at the bottom: 0.3 Hz
already reads 5 per cent low, so content near the 0.2 Hz corner is heavily attenuated rather
than passed.

Three consequences worth keeping in mind.

**Do not read a band edge as a content boundary.** Moving the ceiling from 10 Hz to 5 does not
discard the 5–10 Hz octave; it discards, in effect, everything above roughly 3.5 Hz. That is why
the jerk change is as large as it is.

**Do not probe a filter at its own corner.** A validation test that puts its test tone at the
stated edge will fail against the analytic answer, and the failure is the filter working
correctly. Probe inside the passband and characterise the roll-off separately, as here.

**The steepness is a consequence of zero-phase filtering.** `filtfilt` applies the response
twice, which is what buys zero phase distortion. That matters when the output is going to be
differentiated or integrated, and it costs a sharper effective roll-off than the nominal order
suggests. The trade is deliberate, but the effective bandwidth is narrower than the label.

## The edge is not modality-neutral

The sweep above is over optical datasets, and the accelerometer check above is about drift.
Drift is not the only thing that moves. Holding the pipeline fixed and changing only the lower
edge from 0.3 to 0.2 Hz:

| recorded quantity | change in reported speed |
|---|---|
| position, optical marker | +7 to +18 % |
| acceleration, body-worn | +25 to +94 % |

The asymmetry is structural. Speed comes from *differentiating* position and from *integrating*
acceleration, so the operator converting the recorded quantity to velocity is multiplication by
*f* in one case and division by *f* in the other. Measurement noise is roughly white in whatever
the instrument records, which means the same sliver of extra low-frequency bandwidth is
suppressed for a marker and amplified for an accelerometer. It is largest on chest-worn sensors,
where 0.2 Hz sits on the respiration fundamental.

What follows:

- **Rankings survive; levels do not.** Where only the band differs, Spearman correlations stay
  at 0.94 and above. Comparisons of who moved more are robust to this. Comparisons of *how many
  millimetres per second* are not.
- **State the band whenever quoting an absolute level from an accelerometer.** The same
  recording supports two defensible numbers a factor of two apart.
- **Treat cross-modality level comparisons as approximate**, however harmonised the band. A
  shared band does not by itself make an optical and an inertial measure interchangeable.

## What the difference costs

Measured on one championship edition, the band-pass reads 15.5 per cent below the low-pass on
identical data.

The choice is not only about matching what is in print. Across seven independent optical
datasets, covering two tracking systems, four sampling rates, twelve years and 789 recordings,
the convergence is:

| Convention | Spread across datasets |
|---|---|
| `micromotion` band-pass | 3.5 % (median speed) |
| `optical_legacy` low-pass | 5.9 % |

The low-frequency drift the legacy filter keeps is the part that varies between datasets; the
band-limited part is the part that does not. If an invariant is the claim, the band-pass states
it better, and it is also the only convention under which an accelerometer can confirm it.

## A trap at 20 Hz

The legacy low-pass cannot be computed at 20 Hz. Its 10 Hz cutoff is Nyquist there, so the
filter has no transition band left. Compute legacy values at the native rate. This also means
any dataset natively recorded at 20 Hz has always had its legacy figures computed at the edge of
what its rate supports.

## These definitions are duplicated inside archival records

Six deposited data records carry a hand-written copy of the band-pass, and some of them of the
fourth-order derivative and the gap-bridging rule as well. That is deliberate: an archival
record has to run for someone who has only that folder, so it may not import this package, and
the definition is written out in numpy and scipy inside the record instead.

The consequence is a fork that nothing keeps in step. Changing `BAND`, `ORDER`,
`NYQUIST_MARGIN`, the interpolation limit or the derivative rule here does not change those
copies, and nothing fails: each record's own check imports its own generator, so the record
stays internally consistent while drifting away from the corpus it belongs to. It is the same
shape as a test holding its own copy of a constant, and it is invisible from either side alone.

A change to any of those constants is therefore a change to seven places, not one. The deposit
tree carries `deposited_vs_toolbox.py`, which runs both implementations on the same synthetic
signals and reports the worst disagreement; run it after touching `filters`, `qom` or
`resample`.

Two conventions differ legitimately between records, and the check reports which each uses
rather than enforcing one. Some pass acceleration in g and some in m/s², and some integrate with
the rectangle rule while others use the trapezoid. The two quadrature rules differ by a fraction
of a per cent—systematic rather than noise—which is why `velocity_from_acceleration` takes
`integrate` rather than choosing for the caller.
