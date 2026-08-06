# Getting started

## Install

```bash
pip install micromotion
```

Python 3.10 or newer, with numpy, scipy and pandas. There is no computer-vision or audio stack
to install. One function, `intraclass_correlation`, needs statsmodels; install it with
`pip install "micromotion[mixed]"`.

## Quantity of motion from a motion-capture file

```python
import micromotion as mm

rec = mm.read("mocap_data/A0001.tsv")
head = rec.marker("P01")                       # (n_samples, 3), gaps already NaN
result = mm.qom(head, rec.fs, kind="position", unit=rec.unit)
print(result.median_mm_s, result.mean_mm_s)
```

`QomResult` carries the two summaries, the full `speed` series, the rate it was computed at,
the number of samples, and `edge_samples`, which is how many samples at each end the filter
transient reaches into.

!!! warning "Do not name a local variable `mm`"

    The conventional alias collides with two very natural variable names: a mean, and a value
    in millimetres. Binding either at module level silently replaces the package for the rest
    of the file, and nothing fails until something later reaches for `mm.` and gets a float.
    This has happened twice in the source corpus. One script opened with
    `mm, ss, nn = msd(...)`, and another named a NaN mask `mm`. In both cases the error
    surfaced far from its cause.

`read` identifies the layout from the file's contents rather than its extension, because in
this corpus the extension lies: the balance-board files are named `.tsv` and are in fact
space-delimited and headerless.

Always select a marker by name. Six files in one collection break the documented marker order,
and reading positionally mis-assigns every one of them.

## From an accelerometer

```python
rec = mm.read("accelerometer_data/subject_01.tsv")
mm.qom(rec.data, rec.fs, kind="acceleration", unit=rec.unit).median_mm_s
```

The reader records the unit, so passing `rec.unit` is safer than typing it. Getting this wrong
is not hypothetical: every phone quantity of motion in the source project was 9.80665 times too
large until the error was found.

## Comparing across datasets

```python
y = mm.to_rate(rec.data, rec.fs, mm.HARMONISED_RATE)   # 100 Hz, downsample only
mm.qom(y, mm.HARMONISED_RATE, kind="acceleration", unit=rec.unit)
```

`to_rate` raises rather than upsampling. Use `mm.COMMON_RATE`, which is 20 Hz, only when a
natively-20 Hz collection has to be included, and say that the table is the lossy view. See
[Sampling rates](rates.md).

## Binning, and the edges

```python
bins = result.binned(5.0)
usable = bins[bins.edge == "ok"]
```

`binned` returns a DataFrame with `time_s`, `qom_mm_s`, `n_samples` and `edge`. The final bin
is usually partial, and the first and last carry filter transients; both are flagged rather
than dropped, so that the caller decides. Including the partial bin once inflated a published
series three- to fourteenfold.

## Putting two recordings on one clock

```python
t1, hr1 = mm.instantaneous_rate(haemoglobin, 75.0)          # cardiac band, beats per minute
t2, hr2 = mm.instantaneous_rate(chest_accel_magnitude, 100.0)
mm.search_lag(t1, hr1, t2, hr2, max_lag_s=300)
# {'lag_s': 126.0, 'r': 0.632, 'n_overlap': 174, 'confident': True}
```

Two instruments that share no clock can still be aligned if both carry the same physiological
rhythm. Always check `confident`. See the `xcorr_lag` entry in the [API reference](api.md) for
why a sharp-looking correlation peak is not evidence.
