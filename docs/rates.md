# Sampling rates

Two rules, both learned by getting them wrong.

## Measure the rate, do not read it

```python
mm.measured_rate(timestamps)      # samples over elapsed span
```

Deliberately not the reciprocal of the median interval. Where timestamps are rounded to whole
milliseconds the intervals become a mixture of adjacent integers whose median is a
quantisation artefact:

| Truth | Median-interval estimate |
|---|---|
| 256 Hz | 250 Hz |
| 106 Hz | 636 Hz |

Nominal rates in headers and documentation were wrong by up to 4.4 per cent in one corpus,
and by a factor of 37 in one record.

Beware also of the difference between a **row rate** and a **sensor rate**. One phone logger
writes a row whenever any sensor updates and repeats the previous accelerometer value in
between: the rows arrive at 100 Hz while the accelerometer updates at 16.7 Hz.

```python
mm.rate_quality(timestamps)
# jitter, largest gap, duplicate and backward timestamps, coverage
```

## Downsample, never upsample

```python
mm.to_rate(x, fs_in, 20.0)     # raises if this would upsample
```

Upsampling invents structure between samples, and every method that reads across scales —
multifractal analysis, recurrence quantification, anything with an embedding — treats the
invention as real. One analysis that upsampled 20 Hz data to 25 Hz produced multifractal
widths up to 6.6 where the plausible range is about 1. Nothing failed; the numbers were simply
wrong.

If a series cannot reach the target rate it does not belong in a comparison built at that
rate. Excluding it is correct; interpolating it in corrupts the comparison silently.

## The common rate

`mm.COMMON_RATE` is 20 Hz. It is the Nyquist rate of the 0.2–10 Hz band, so nothing the band
keeps is lost, and it is low enough that the slowest collections need no upsampling.

It costs about **2 per cent** against the native-rate value on 200 Hz optical data. Report
both if it matters.

## Irregular sampling

```python
grid, y = mm.regularize(t, x, fs_out=20.0, max_gap_s=2.0)
```

Sorts, drops duplicate and backward timestamps, interpolates, and returns NaN inside gaps
longer than `max_gap_s` rather than bridging them — so that a 132-second hole cannot be
mistaken for 132 seconds of stillness.
