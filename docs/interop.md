# Working with other packages

## The intended shape

The fourMs packages divide by signal domain, and the dependency arrow points from the heavy
packages to the light ones:

| Package | Owns | Weight |
|---|---|---|
| `musicalgestures` (MGT) | video in, visual features out | ~282 MB — opencv, librosa, numba, scikit-image; ffmpeg binary |
| `ambiscape` | spatial audio in, soundscape features out | numpy, scipy, soundfile, matplotlib |
| `musiscape` | music corpora, fingerprints, similarity | + librosa, ambiscape |
| `micromotion` | motion time series: mocap, IMU, force plate | numpy, scipy, pandas |

`micromotion` deliberately depends on none of them. Someone analysing accelerometer data
should not install a computer-vision stack to do it, and the reverse arrow — MGT depending on
this package — is the one that makes sense.

There is an existing precedent for how to cross the boundary. MGT's `_soundscape.py` consumes
ambiscape's output and adapts it at the seam, and ambiscape never imports MGT.

!!! warning "Unresolved overlap with MGT"
    MGT's **unreleased** `main` branch already contains `band_limited_qom`, `accel_to_speed`,
    `read_qtm_tsv`, `cop_sway_metrics` and `respiration_rate`, credited in their docstrings to
    the same source study as this package. They are not in the released 1.6.9, so they are
    invisible if you look only at PyPI.

    The two implementations do not agree. On the same 200 Hz optical recording:

    | | mean speed |
    |---|---|
    | MGT `band_limited_qom`, its 0.3–15 Hz default | 5.675 mm/s |
    | MGT `band_limited_qom`, forced to 0.3–10 Hz | 5.594 mm/s |
    | `micromotion.qom`, 0.3–10 Hz | 5.455 mm/s |

    4.0 per cent apart at their respective defaults; 2.6 per cent apart on the same band. The
    residual comes from the differentiation and whether the result is band-limited a second
    time. Until this is resolved, **state which package produced any number you report.**

## MGT: video motion into micromotion

MGT's released `motiondata()` returns a *file path*, not an array:

```python
import pandas as pd, musicalgestures as mg

path = mg.MgVideo("dance.mp4").motiondata(data_format="csv")   # -> str
df = pd.read_csv(path)
arr = df[["ComX", "ComY"]].to_numpy()      # (n_samples, 2)
fs = 1000.0 / df.Time.diff().median()      # Time is in milliseconds
```

Three things to know.

Use the **CSV**, not the TSV or TXT. Those write `Qom` with an integer format, so values
normalised to 0–1 truncate to zero.

The `Qom` column is divided by its own maximum, so it is dimensionless and **not comparable
between recordings**. The centroid columns are normalised to frame width and height. If you
want a physical speed, feed the centroid columns to `micromotion.qom` with
`kind="position"` — but the units are frame fractions, not millimetres, so scale them first
or treat the result as relative.

Prefer `MgVideo.fps` over recovering the rate from `Time`, which is rounded to integer
milliseconds and loses precision at rates like 29.97.

## ambiscape: soundscape features alongside motion

ambiscape emits a `dict[str, np.ndarray]` in an `.npz`, on four fixed time bases:

| Vector | Rate | Contents |
|---|---|---|
| `t_hi` | 50 Hz | `env_hi` |
| `t_fast` | 8 Hz | `fast_db`, `fast_dba` |
| `t` | 1 Hz | `rms_w`, `centroid`, `az`, `el`, `diffuse`, octave bands |
| `min_t` | 1/60 Hz | `minspec` |

Times are absolute seconds from session midnight and are **not monotonic** across overlapping
takes. Levels are uncalibrated dBFS, not SPL.

To relate a soundscape to a body, downsample both to a common rate and align:

```python
F = np.load("features.npz")
lag = mm.search_lag(F["t"], F["rms_w"], t_motion, qom_1hz, max_lag_s=300)
```

ambiscape computes no quantity of motion and explicitly defers video analysis to MGT, so
there is no overlap to worry about there.

## musiscape and respy

`musiscape` emits only per-track scalars — it time-averages its per-frame features away
before writing `features.json`. Nothing to interoperate with.

`respy` is not a fourMs package; it is a personal one by a RITMO colleague, single release,
2023. It emits a time-indexed `pandas.DataFrame` and infers its sampling rate from the index
with a `round()`, which mis-reads genuine 12.5 or 62.5 Hz recordings. Its published metadata
declares no dependencies, so `pip install respy` installs nothing and fails on import. Adapt
it at the boundary if you need it:

```python
fs = 1.0 / np.mean(np.diff(df.index))     # do not round
```

`micromotion.detect_breaths` covers the common case without the dependency.
