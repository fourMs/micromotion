# micromotion

Analysis of human micromotion in motion time series: optical marker data, body-worn
accelerometers, respiration belts and force-plate centre of pressure.

The measure the package is built around is **quantity of motion** — the average speed of a body
part, band-limited to 0.2–5 Hz, in millimetres per second. It applies equally to all of those
sensor families because the shared abstraction is the frequency band, not the instrument.

```bash
pip install micromotion
```

## Who this is for

Anyone measuring small involuntary movement in a standing, sitting or otherwise stationary body —
postural sway, physiological tremor, the mechanical trace of breathing and heartbeat, or how a
group of people responds to a shared stimulus. It grew out of research on standstill and music,
but nothing in it is specific to that: the methods are the standard nonlinear time-series and
posturography ones, and the readers cover common laboratory formats.

If you work on gait, gesture or dance, this is probably the wrong tool. The band and most of the
defaults assume the interesting movement is small and the body is not travelling.

## Why a package

Micromotion analysis has an unusually wide gap between "the method" and "a number". The same
recording yields materially different answers depending on the filter band, the differentiation
rule, whether the result is band-limited a second time, whether you take a mean or a median, and
what the true sampling rate turns out to be. None of those choices announces itself in a results
table.

This package makes each of them explicit, gives each a measured default rather than an inherited
one, and refuses a few comparisons that look reasonable and are not.

## What it will not let you do

- Asking for the legacy optical band on accelerometer data **raises**, because gravity is a DC
  term and an accelerometer cannot be integrated without a lower edge.
- Upsampling **raises**, because it invents structure that scale-sensitive methods treat as real.
- Gap sentinels become `NaN` on read — a marker "at the origin", or a centre of pressure at the
  exact middle of a board, are plausible-looking values that are not measurements.
- Partial and filter-contaminated bins are **flagged**, not silently included.

## Where to start

| If you want to | Read |
|---|---|
| compute a quantity of motion | [Getting started](quickstart.md) |
| understand a method, or cite it | [Methods](methods.md) |
| know which filter convention to use | [The two bands](conventions.md) |
| compare across datasets or devices | [Sampling rates](rates.md) |
| load a file | [Reading files](formats.md) |
| check data before trusting it | [Validating data](validation.md) |
| combine with other toolboxes | [Working with other packages](interop.md) |
| look up a function | [API reference](api.md) |

Practical notes, worked recipes and known traps are on the
[wiki](https://github.com/fourMs/micromotion/wiki), which changes independently of a release.

## Citing

If you use this package, cite it — see `CITATION.cff` in the repository. Cite the underlying
methods too; [Methods](methods.md) gives the reference for each.

Licence: GPL-3.0-or-later.
