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

The lower edge was 0.3 Hz until 2026-07-29 — inherited from earlier work rather than chosen.
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
