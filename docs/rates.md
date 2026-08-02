# Sampling rates

Two rules, both learned by getting them wrong.

## Measure the rate, do not read it

```python
mm.measured_rate(timestamps)      # samples over elapsed span
```

We deliberately do not use the reciprocal of the median interval. Where timestamps are rounded to
whole milliseconds the intervals become a mixture of adjacent integers whose median is a
quantisation artefact:

| Truth | Median-interval estimate |
|---|---|
| 256 Hz | 250 Hz |
| 106 Hz | 636 Hz |

Nominal rates in headers and documentation were wrong by up to 4.4 per cent in one corpus,
and by a factor of 37 in one record.

## A file's grid is not its sampling rate, and a channel's rate is not its instrument's

Three distinct numbers get called "the sampling rate", and confusing them is the most expensive
mistake in this document.

**The row rate is not the sensor rate.** One phone logger writes a row whenever *any* sensor
updates and repeats the previous accelerometer value in between: rows arrive at about 170 Hz while
the accelerometer updates at 50 and the fused channel at 15. About 89 per cent of rows repeat the
previous sample. Reading such a file as though each row were a measurement gives a staircase, and a
staircase is a sequence of step edges, which is broadband high-frequency energy that
differentiation amplifies once per derivative.

**The stored grid is not the sensor rate either.** Resample that logger's output to a uniform
100 Hz file and nothing downstream can tell it is a 6.7-fold upsample of a 15 Hz signal. We made
`to_rate` refuse to upsample precisely to prevent this, but it never sees the raw file. Carry the measured
sensor rate as its own column and check deliverability against *that*.

**And a channel's rate is not its instrument's.** This one is the subtlest. That phone's deposited
motion channel is *linear acceleration*, which the logger derives by fusing accelerometer,
gyroscope and magnetometer. Measured per sensor across a year:

| sensor | rate |
|---|---|
| accelerometer | ~50 Hz |
| gyroscope | ~15 Hz |
| magnetometer | 17–50 Hz |
| fused linear acceleration | ~15 Hz |

**A fusion runs at the speed of its slowest input.** The fused channel is pinned to the gyroscope,
so the ceiling is a property of the channel that was recorded, not of the hardware. The same
instrument has a raw channel three times faster sitting underneath it.

Which does not make the raw channel simply better. It reports total g-force including gravity, so a
lean rotates the gravity vector into the band and reads as movement, and on that corpus it inflates
band-limited speed about fourfold. But it inflates *jerk* only about 1.5×, because tilt is a
low-frequency contaminant and jerk is a high-frequency measure. So the right channel depends on the
measure, and a row can legitimately take its speed from one channel and its jerk from another,
provided the table records which.

```python
mm.rate_quality(timestamps)
# jitter, largest gap, duplicate and backward timestamps, coverage
```

## Downsample, never upsample

```python
mm.to_rate(x, fs_in, 20.0)     # raises if this would upsample
```

Upsampling invents structure between samples, and every method that reads across scales treats the
invention as real. That includes multifractal analysis, recurrence quantification, and anything
with an embedding. One analysis that upsampled 20 Hz data to 25 Hz produced multifractal
widths up to 6.6 where the plausible range is about 1. Nothing failed; the numbers were simply
wrong.

If a series cannot reach the target rate it does not belong in a comparison built at that
rate. Excluding it is correct; interpolating it in corrupts the comparison silently.

## Which rate to harmonise at

Two constants, and the choice between them matters more than it looks.

```python
mm.HARMONISED_RATE     # 100.0 — use this where every series can reach it
mm.COMMON_RATE         #  20.0 — use this only when a 20 Hz collection must be included
```

We set `COMMON_RATE` to 20 Hz because that is the greatest common divisor of the optical rates in
the source corpus, which are 20, 100, 120 and 200 Hz, so every recording reaches it by an integer
decimation and none needs upsampling. That is its one virtue, and it is a real one: it is the only rate at
which a natively-20 Hz collection can be placed beside the rest at all.

It is not free, and the argument that it is does not survive measurement. A 10 Hz upper edge
cannot be realised at 20 Hz: Nyquist sits exactly on it, the margin rule pulls the edge inside,
and the anti-alias filter is already rolling off below it. Decimating 34 natively-200 Hz
person-recordings and re-measuring:

| Target | Median change | Range across recordings |
|---|---|---|
| 120 Hz | +0.11 % | −0.83 to +0.60 % |
| **100 Hz** | **+0.02 %** | **−0.84 to +0.47 %** |
| 20 Hz | −2.09 % | −0.30 to **−10.57 %** |

The last column is the important one. A per-recording error running from a third of a per cent
to more than ten is a distortion rather than a bias, so nothing downstream can correct it.
Prefer 100 Hz; where you must use 20, say so and call it the lossy view.

Non-integer ratios are fine, since 120 → 100 Hz is a 6-to-5 polyphase step and costs about a tenth
of a per cent.

## Irregular sampling

```python
grid, y = mm.regularize(t, x, fs_out=20.0, max_gap_s=2.0)
```

Sorts, drops duplicate and backward timestamps, interpolates, and returns NaN inside gaps longer
than `max_gap_s`. We leave those gaps open rather than bridging them, so that a 132-second hole
cannot be mistaken for 132 seconds of stillness.
