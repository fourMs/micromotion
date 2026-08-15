# micromotion

[![tests](https://github.com/fourMs/micromotion/actions/workflows/tests.yml/badge.svg)](https://github.com/fourMs/micromotion/actions/workflows/tests.yml)
[![docs](https://github.com/fourMs/micromotion/actions/workflows/docs.yml/badge.svg)](https://fourms.github.io/micromotion/)
[![PyPI](https://img.shields.io/pypi/v/micromotion.svg)](https://pypi.org/project/micromotion/)
[![Python](https://img.shields.io/pypi/pyversions/micromotion.svg)](https://pypi.org/project/micromotion/)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21948988.svg)](https://doi.org/10.5281/zenodo.21948988)

A Python package for measuring human micromotion: the small movement of a body that is
standing, sitting or otherwise trying to stay still. It reads optical marker data, body-worn
accelerometers, respiration belts and force plates, and reduces all of them to one measure.

That measure is **quantity of motion**—the average speed of a body part, band-limited to
0.2–5 Hz, in millimetres per second. It can be computed from every sensor family, because the
shared abstraction is the frequency band rather than the instrument.

![Band-limited speed of a synthetic head marker, with the median, the mean, and the same series in five-second bins](docs/img/qom-standstill.png)

## Install

```bash
pip install micromotion
```

Python 3.10 or newer, with numpy, scipy and pandas. There is no computer-vision or audio stack
to install.

## Quickstart

```python
import micromotion as mm

rec = mm.read("mocap_data/A0001.tsv")      # dispatches on content, not on the extension
head = rec.marker("P01")                   # (n_samples, 3), gaps already NaN
result = mm.qom(head, rec.fs, kind="position", unit=rec.unit)

print(result.median_mm_s, result.mean_mm_s)
```

Report the median, and say that it is the median. The mean and the median can rank the same
recordings differently, so both are returned and neither is chosen for the caller.

Do not name a local variable `mm`. The conventional alias collides with a mean and with a value
in millimetres, and rebinding it replaces the package for the rest of the file.

## Documentation

| | |
|---|---|
| [Reference documentation](https://fourms.github.io/micromotion/) | how to use it, every function, the conventions |
| [Wiki](https://github.com/fourMs/micromotion/wiki) | traps, worked recipes, design decisions |
| [Changelog](CHANGELOG.md) | what changed between releases |

Read [Getting started](https://fourms.github.io/micromotion/quickstart/) first, then
[The three bands](https://fourms.github.io/micromotion/conventions/), which is the one
convention that cannot be skipped. [Reading files](https://fourms.github.io/micromotion/formats/)
covers what each reader handles, which axis is vertical in which system, and the traps that
produce plausible numbers rather than errors.

## What is in it

| Module | Contents |
|---|---|
| `qom` | quantity of motion from position or acceleration, in three named variants |
| `filters` | the band definitions—`BAND`, `WIDEBAND`, `OPTICAL_LEGACY_BAND`—with band-pass, low-pass, high-pass and notch |
| `resample` | rate measurement, downsample-only resampling, irregular-to-regular gridding, gap handling |
| `io` | one reader per file layout, a content sniffer, and the per-channel rate and resolution checks |
| `record` | `MotionRecord`, the common type every reader returns |
| `validate` | checks that fail loudly on silently-wrong data |
| `posture`, `balance` | sway geometry, spatial extent, centre-of-pressure measures |
| `spectral`, `physio` | cardiac and respiratory peaks, band power, breathing rate and breath phase |
| `dynamics` | detrended fluctuation analysis, multifractality, recurrence, entropy, surrogates |
| `group` | whether several people moved at the same moments |
| `align` | offsets between instruments that share no clock |
| `circular` | directional statistics, including the axial tests postural sway needs |
| `features` | `feature_vector`, one fixed set of eleven descriptors per recording |
| `equivalence` | stating that an effect is absent rather than failing to show it is present |
| `descriptors` | how many independent dimensions a descriptor set holds, and whether a measure is a trait |

Readers: Qualisys and Qualisys-style TSV in all three header shapes, Sverm, Axivity AX3,
Physics Toolbox phone logs, Equivital, Wii balance board, and Artinis fNIRS. `read` dispatches
on content rather than on extension, because in this field the extension is frequently wrong.

## Licence and credit

GPL-3.0-or-later. Built at the [fourMs lab](https://github.com/fourMs), RITMO Centre for
Interdisciplinary Studies in Rhythm, Time and Motion, University of Oslo. If you use the
package, please cite it—see [CITATION.cff](CITATION.cff)—and cite the underlying methods too,
since the [Methods](https://fourms.github.io/micromotion/methods/) page gives a reference for
each.

Issues and pull requests are welcome at
[fourMs/micromotion](https://github.com/fourMs/micromotion/issues). A case where a default here
gives a misleading answer is the most useful kind of issue to file.

## Related toolboxes

These come out of the same lab, as separate packages with separate release cycles. They are
built to be used together and share several implementations, so a measure computed in one
agrees with the same measure computed in another.

- [Musical Gestures Toolbox](https://github.com/fourMs/MGT-python) (`musicalgestures`)—video and audio: motiongrams, videograms, and motion analysis from ordinary video files
- [ambiscape](https://github.com/fourMs/ambiscape)—soundscapes: the sonic ambience of a place, across level, spectral, spatial, temporal, ecological and source descriptors
- [musiscape](https://github.com/fourMs/musiscape)—music collections: comparing many tracks and albums held as audio files in folders

## Citing

Cite the CONCEPT DOI, which always resolves to the newest version:

> Jensenius, A. R., Upham, F., Zelechowska, A., Gonzalez-Sanchez, V. E., & Swarbrick, D. (2026). *micromotion: analysis of human micromotion in motion time series* (Version 1.12.1) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.21948988

Where the exact behaviour matters, cite the version you ran instead. This package HAS changed
behaviour at releases — `read_phone` at 0.15.0, `group_qom` at 1.0.0, `to_rate` at 1.2.2 — so which
version produced a number is part of the method. Version 1.12.1 is https://doi.org/10.5281/zenodo.21948989.

`CITATION.cff` in this repository carries the same information in machine-readable form.
