# The two bands

There are two quantity-of-motion conventions in circulation and they are **not
interchangeable**. Reporting a number without saying which one produced it is the single
most common way results in this field stop being comparable.

## `micromotion` — 0.3 to 10 Hz

A band-pass. The only convention that can be applied to every sensor, so the one any
cross-collection comparison must use.

## `optical_legacy` — 10 Hz low-pass, no lower edge

Retains sub-0.3 Hz postural drift. For optical position that drift is real movement, and this
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
