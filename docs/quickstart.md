# Getting started

## Install

```bash
pip install git+https://github.com/fourMs/micromotion
```

Requires numpy, scipy and pandas. Nothing else — no computer-vision or audio stack.

## Quantity of motion from a motion-capture file

```python
import micromotion as mm

rec = mm.read("Standstill2017/mocap_data/A0001.tsv")
head = rec.marker("P01")                       # (n_samples, 3), gaps already NaN
result = mm.qom(head, rec.fs, kind="position")
print(result.mean_mm_s, result.median_mm_s)
```

!!! warning "Do not name a local variable `mm`"

    The conventional alias collides with two very natural variable names: a **m**ean, and a value
    in **m**illimetres. Binding either at module level silently replaces the package for the rest
    of the file, and nothing fails until something later reaches for `mm.` and gets a float. This
    has happened twice in the source corpus — `mm, ss, nn = msd(...)` in one script and a NaN mask
    called `mm` in another — and in both cases the error surfaced far from its cause.


`read` identifies the layout from the file's contents, not its extension — in this corpus the
extension lies, since the balance-board files are named `.tsv` and are space-delimited and
headerless.

Always select a marker by name. Six files in one collection break the documented marker
order, and reading positionally mis-assigns every one of them.

## From an accelerometer

```python
rec = mm.read("accelerometer_data/subject_01.tsv")
mm.qom(rec.data, rec.fs, kind="acceleration", unit="g").mean_mm_s
```

The reader records the unit, so passing `rec.unit` is safer than typing it. Getting this
wrong is not hypothetical: every phone quantity of motion in the source project was 9.80665
times too large until the error was found.

## Comparing across datasets

```python
y = mm.to_rate(rec.data, rec.fs, mm.COMMON_RATE)      # 20 Hz, downsample only
mm.qom(y, mm.COMMON_RATE, kind="acceleration", unit=rec.unit)
```

`to_rate` raises rather than upsampling. See [Sampling rates](rates.md).

## Binning, and the edges

```python
bins = result.binned(5.0)
usable = bins[bins.edge == "ok"]
```

The final bin is usually partial, and the first and last carry filter transients. Both are
flagged rather than dropped. Including the partial bin once inflated a published series three-
to fourteenfold.

## Putting two recordings on one clock

```python
t1, hr1 = mm.instantaneous_rate(haemoglobin, 75.0)     # cardiac band
t2, hr2 = mm.instantaneous_rate(chest_accel_magnitude, 100.0)
mm.search_lag(t1, hr1, t2, hr2, max_lag_s=300)
# {'lag_s': 126.0, 'r': 0.632, 'confident': True}
```

Two instruments that share no clock can still be aligned if both carry the same
physiological rhythm. Always check `confident` — see [the API notes on `xcorr_lag`](api.md)
for why a sharp-looking correlation peak is not evidence.
