# One feature vector per recording

Every attempt to compare recordings needs the same thing first: a fixed set of numbers describing
one recording. Clustering, identity classification, condition classification, dimensionality
reduction — each of them starts there, and each will happily invent its own set.

That is the problem this solves. Two analyses that reduce a recording differently are incomparable
for reasons that have nothing to do with the question either is asking.

```python
import micromotion as mm

f = mm.feature_vector(xyz, fs, kind="position", unit="mm")
mm.FEATURE_NAMES        # the column order
```

Returns `None` if the recording is shorter than two minutes, rather than a vector computed from too
little data.

## The eleven

| group | descriptors | what they answer |
|---|---|---|
| amount and smoothness | `qom`, `jerk` | how much, and how abruptly |
| frequency and texture | `centroid`, `f50`, `frozen`, `burst` | how fast, and how evenly spread in time |
| sway geometry | `path`, `extent`, `area`, `anis`, `vert` | how far it travels, how large a region, how elongated |

The geometric five need true position and are `nan` for accelerometer collections. A chest-worn
accelerometer cannot give a sway ellipse, and doubly integrating one to fake it reports drift as
posture.

## Two arguments you must pass

`kind` and `unit` are required and are not guessed. Collections do not record the same quantity:
optical systems store position in millimetres, accelerometers store acceleration in g or m/s².
Differentiating an acceleration series as though it were position shifts every descriptor two
derivatives up and still returns finite, plausible-looking numbers. That failure is silent, which is
why the signature refuses to default.

## The rate that matters is the sensor's

```python
mm.feature_vector(xyz, fs=100.0, sensor_fs=15.0)     # jerk comes back nan, correctly
```

A uniform grid can be an upsample of a much slower sensor. A file stored at 100 Hz whose sensor
updated at 15 cannot carry a 10 Hz band, and no filter can see the difference — the grid looks fast
enough. Pass `sensor_fs` whenever the two differ, and `jerk` will be `nan` rather than a number
computed from interpolation.

## Jerk is computed at a different band, deliberately

Ten of the eleven use the canonical band. `jerk` uses `WIDEBAND`, because it lives in the octave the
canonical band gives up: at a 5 Hz ceiling it falls to between a third and two thirds of its value
and the ranking of recordings shifts. One definition of jerk across a corpus is worth more than one
band within a single table. Where the sampling rate cannot deliver `WIDEBAND`, `jerk` is `nan`
rather than a narrower measure reported under a wider name.

## What belongs here and what does not

This is a measurement library. A classifier belongs in an analysis repository, where its train/test
split and its leakage are visible to a reader. What belongs here is the input every such model
starts from, so that two attempts at the same question begin from the same numbers.
