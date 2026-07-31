# The two bands

There are two quantity-of-motion conventions in circulation and they are **not
interchangeable**. Reporting a number without saying which one produced it is the single
most common way results in this field stop being comparable.

## `micromotion` — 0.2 to 10 Hz

A band-pass. The only convention that can be applied to every sensor, so the one any
cross-collection comparison must use.

## `optical_legacy` — 10 Hz low-pass, no lower edge

Retains sub-0.2 Hz postural drift. For optical position that drift is real movement, and this
is the convention behind most published standstill figures.

## Why the lower edge is optional for position and mandatory for acceleration

An optical system measures position directly, so slow drift is a genuine displacement and
keeping it is a choice.

An accelerometer cannot offer that choice. Gravity is a DC term, and integrating a signal
with any residual offset produces a ramp that swamps the result. Asking for
`band="optical_legacy"` on acceleration therefore raises rather than returning a
plausible-looking number.

```python
mm.qom(acc, fs, kind="acceleration", band="optical_legacy")
# ValueError: the 'optical_legacy' band has no lower edge, and an accelerometer
# cannot be integrated without one...
```


## Why 0.2 Hz, and why a median

The lower edge is not arbitrary.
Swept across seven optical datasets and 665 recordings, the between-dataset spread is:

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
converges to 2.1 per cent and the mean only to 4.6; at 0.3 Hz it was the other way round.
Report the median at this band. A mean here would be looser than what it replaced.

Two things were checked before adopting it. A lower edge closer to DC risks integration drift
on accelerometer data, which is the failure the edge exists to prevent: the mean-to-median
speed ratio, which climbs when drift leaks in, is 2.07 at 0.2 Hz against 2.00 at 0.3, so it is
flat and integration is safe. And the filter transient lengthens to 40 seconds at each end
rather than 27, so a recording shorter than about two minutes has no clean interior at all.
That rules the band out for short trials; it does not affect standstill protocols, which run
six minutes or more.

## The edge is not modality-neutral

The sweep above is over **optical** datasets, and the accelerometer check above is about drift.
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
- **State the band whenever quoting an absolute level from an accelerometer.** The same recording
  supports two defensible numbers a factor of two apart.
- **Treat cross-modality level comparisons as approximate**, however harmonised the band. A shared
  band does not by itself make an optical and an inertial measure interchangeable.

## What the difference costs

Measured on one championship edition, the band-pass reads **15.5 per cent below** the
low-pass on identical data.

The choice is not only about matching what is in print. Across seven independent optical
datasets — two tracking systems, four sampling rates, twelve years, 789 recordings — the
convergence is:

| Convention | Spread across datasets |
|---|---|
| `micromotion` band-pass | 3.5 % (median speed) |
| `optical_legacy` low-pass | 5.9 % |

The low-frequency drift the legacy filter keeps is the part that varies between datasets; the
band-limited part is the part that does not. If an invariant is the claim, the band-pass
states it better — and it is also the only convention under which an accelerometer can
confirm it.

## A trap at 20 Hz

The legacy low-pass **cannot be computed at 20 Hz**. Its 10 Hz cutoff is Nyquist there, so the
filter has no transition band left. Compute legacy values at the native rate. This also means
any dataset natively recorded at 20 Hz has always had its legacy figures computed at the edge
of what its rate supports.
