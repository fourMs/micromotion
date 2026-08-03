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

We deliberately made `micromotion` depend on none of them. Someone analysing accelerometer data
should not have to install a computer-vision stack to do it, and the arrow that makes sense runs
the other way, with MGT depending on this package.

There is an existing precedent for how to cross the boundary. MGT's `_soundscape.py` consumes
ambiscape's output and adapts it at the seam, and ambiscape never imports MGT.

## What to reach for ambiscape for

ambiscape owns the sound side of a cross-modal study, and its outputs are per-second tables that
join straight onto a micromotion series on a shared clock. In this corpus it supplied the whole
soundscape layer: the feature pass, the `mechanical` / `anthropophony` / `geophony` / `tonality`
domain indices, `enf` for mains hum, `speechgate` for a privacy check, and `taxonomy` for Schafer
and Schaeffer figures. Docs: <https://fourms.github.io/ambiscape/>, and its
[interop page](https://fourms.github.io/ambiscape/interop/) is the mirror of this one.

**Two hazards live at that join, both learned the hard way here.**

*Clocks are not shared just because recordings are simultaneous.* Devices started separately drift,
and a phone app that suspends loses time from its timeline rather than leaving a gap, as
[formats](formats.md) describes for Physics Toolbox. Align on a physical event present in both
signals, such
as a tap that registers in the accelerometer and the microphone at once. An acoustic clap will not
do it: it leaves nothing in an accelerometer.

*A cross-modal null is worth little without a positive control.* Before concluding that a body does
not track its surroundings, check that the same statistic detects something it should. Correlating
two environmental channels against each other is the cheap version of that check. If sound and light track each
other and the body tracks neither, the null is about the body rather than about the pipeline.

!!! warning "Unresolved overlap with MGT"
    **The functions originated here.** Their docstrings credit the stillstanding and
    Westney-comparisons studies, and they were copied into MGT rather than the other way
    round, so this is not a case of matching someone else's prior art. Re-checked
    MGT's repository version matches its PyPI release, so they remain unreleased and invisible
    to anyone reading only PyPI.

    MGT contains `band_limited_qom`, `accel_to_speed`, `read_qtm_tsv`, `cop_sway_metrics` and
    `respiration_rate`, credited in their docstrings to the same source study as this package.

    Updated 2026-08-03: these ARE now released. MGT-python 1.7.0 went to PyPI on that date, so the
    note that used to stand here -- that they existed only on an unreleased branch and were
    invisible to anyone reading PyPI -- no longer holds. The overlap is now a published overlap
    between two published packages, which makes the disagreement below more important rather than
    less.

    The two implementations do not agree. On the same 200 Hz optical recording:

    | | mean speed |
    |---|---|
    | MGT `band_limited_qom`, its 0.3–15 Hz default | 5.675 mm/s |
    | MGT `band_limited_qom`, forced to 0.3–10 Hz | 5.594 mm/s |
    | `micromotion.qom`, matched 0.3–10 Hz | 5.455 mm/s |

    (Rows 2 and 3 were run at a matched 0.3–10 Hz to isolate the algorithmic difference from
    the band difference. Every `micromotion` function defaults to 0.2–5 Hz, so the first row
    is the only one comparing defaults.)

    4.0 per cent apart at their respective defaults; 2.6 per cent apart on the same band. The
    residual comes from the differentiation and whether the result is band-limited a second
    time. Until this is resolved, state which package produced any number you report.

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

Use the CSV, not the TSV or TXT. Those write `Qom` with an integer format, so values
normalised to 0–1 truncate to zero.

The `Qom` column is divided by its own maximum, so it is dimensionless and not comparable
between recordings. The centroid columns are normalised to frame width and height. If you
want a physical speed, feed the centroid columns to `micromotion.qom` with
`kind="position"`, but remember that the units are frame fractions rather than millimetres, so
scale them first or treat the result as relative.

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

Times are absolute seconds from session midnight and are not monotonic across overlapping
takes. Levels are uncalibrated dBFS, not SPL.

To relate a soundscape to a body, downsample both to a common rate and align:

```python
F = np.load("features.npz")
lag = mm.search_lag(F["t"], F["rms_w"], t_motion, qom_1hz, max_lag_s=300)
```

ambiscape computes no quantity of motion and explicitly defers video analysis to MGT, so
there is no overlap to worry about there.

## musiscape and respy

`musiscape` emits only per-track scalars, since it time-averages its per-frame features away
before writing `features.json`. There is nothing to interoperate with.

`respy` is not a fourMs package; it is a personal one by a RITMO colleague, single release,
2023. It emits a time-indexed `pandas.DataFrame` and infers its sampling rate from the index
with a `round()`, which mis-reads genuine 12.5 or 62.5 Hz recordings. Its published metadata
declares no dependencies, so `pip install respy` installs nothing and fails on import. Adapt
it at the boundary if you need it:

```python
fs = 1.0 / np.mean(np.diff(df.index))     # do not round
```

`micromotion.detect_breaths` covers the common case without the dependency.
