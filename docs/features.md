# One feature vector per recording

Every attempt to compare recordings needs the same thing first, which is a fixed set of numbers
describing one recording. Clustering, identity classification, condition classification and
dimensionality reduction all start there, and each of them will happily invent its own set.

That is what this solves. Two analyses that reduce a recording differently are incomparable for
reasons that have nothing to do with the question either one is asking.

```python
import micromotion as mm

f = mm.feature_vector(xyz, fs, kind="position", unit="mm")
mm.FEATURE_NAMES        # the column order
```

The function returns `None` if the recording is shorter than two minutes, or if it contains any
non-finite sample, rather than a vector computed from too little data.

## The eleven

| group | descriptors | what they answer |
|---|---|---|
| amount and smoothness | `qom`, `jerk` | how much, and how abruptly |
| frequency and texture | `centroid`, `f50`, `frozen`, `burst` | how fast, and how evenly spread in time |
| sway geometry | `path`, `extent`, `area`, `anis`, `vert` | how far it travels, how large a region, how elongated |

The five geometric descriptors need true position and are `nan` for accelerometer collections. A
chest-worn accelerometer cannot give a sway ellipse, and double-integrating one to fake it
reports drift as posture.

## Two arguments that must be passed

`kind` and `unit` are required and are not guessed, because collections do not all record the
same quantity. Optical systems store position in millimetres, while accelerometers store
acceleration in g or m/s². Differentiating an acceleration series as though it were position
shifts every descriptor two derivatives up and still returns finite, plausible-looking numbers.
That failure is silent, which is why the signature refuses to default and raises instead.

## The rate that matters is the sensor's

```python
mm.feature_vector(xyz, fs=100.0, kind="position", unit="mm", sensor_fs=15.0)   # jerk is nan, correctly
```

A uniform grid can be an upsample of a much slower sensor. A file stored at 100 Hz whose sensor
updated at 15 Hz cannot carry a 10 Hz band, and no filter can see the difference, since the grid
looks fast enough. Pass `sensor_fs` wherever the two differ, and `jerk` will be `nan` rather than
a number computed from interpolation.

## Jerk is computed at a different band, deliberately

Ten of the eleven use the canonical band, but `jerk` uses `WIDEBAND`, because it lives in the
octave that the canonical band gives up. At a 5 Hz ceiling it falls to between a third and two
thirds of its value, and the ranking of recordings shifts. One definition of jerk across a
corpus is worth more than one band within a single table. Where the sampling rate cannot deliver
`WIDEBAND`, `jerk` is `nan` rather than a narrower measure reported under a wider name.

## The set is a commitment

Fixing eleven descriptors and their names is the point, since two analyses that reduce a
recording differently cannot be compared. It also means the set is something other people's
results depend on, so adding or removing one is a breaking change in practice.

## What belongs here and what does not

This is a measurement library. A classifier belongs in an analysis repository, where a reader
can see its train/test split and its leakage. What belongs here is the input that every such
model starts from, so that two attempts at the same question begin from the same numbers.
