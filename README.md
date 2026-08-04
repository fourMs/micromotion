# micromotion

[![tests](https://github.com/fourMs/micromotion/actions/workflows/tests.yml/badge.svg)](https://github.com/fourMs/micromotion/actions/workflows/tests.yml)
[![docs](https://github.com/fourMs/micromotion/actions/workflows/docs.yml/badge.svg)](https://fourms.github.io/micromotion/)
[![PyPI](https://img.shields.io/pypi/v/micromotion.svg)](https://pypi.org/project/micromotion/)
[![Python](https://img.shields.io/pypi/pyversions/micromotion.svg)](https://pypi.org/project/micromotion/)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

Analysis of human micromotion in motion time series: optical marker data, body-worn
accelerometers, respiration belts and force plates.

The measure the package exists for is **quantity of motion** — the average speed of a body part,
band-limited to 0.2–5 Hz, in millimetres per second. It applies equally to all three sensor
families because the shared abstraction is the frequency band, not the instrument.

```bash
pip install micromotion
```

```python
import micromotion as mm

rec  = mm.read("Standstill2017/mocap_data/A0001.tsv")   # dispatches on content, not extension
head = rec.marker("P01")
mm.qom(head, rec.fs, kind="position").median_mm_s   # the median is the convention; .mean_mm_s is also there
```

## Documentation

| | |
|---|---|
| [Reference documentation](https://fourms.github.io/micromotion/) | how to use it, every function, the conventions |
| [Wiki](https://github.com/fourMs/micromotion/wiki) | why it works this way — traps, recipes, open questions |
| [Changelog](CHANGELOG.md) | what changed, and why |

New to it: [Getting started](https://fourms.github.io/micromotion/quickstart/), then
[The three bands](https://fourms.github.io/micromotion/conventions/), which is the one convention
you cannot skip.

Reading files is its own subject, because the formats in this field lie about themselves:
[Reading files](https://fourms.github.io/micromotion/formats/) covers what each reader handles,
which axis is vertical in which system, and the traps that produce plausible numbers rather than
errors — a Y/Z axis swap that reverses every sway direction while leaving magnitudes intact, a
Unicode minus that turns negative values into NaN, an app whose clock stops when the phone sleeps.

## What is in it

| Module | Contents |
|---|---|
| `qom` | quantity of motion from position or acceleration |
| `filters` | the band definitions, band-pass, low-pass, high-pass, cardiac notch |
| `resample` | rate measurement, downsample-only resampling, irregular-to-regular gridding |
| `validate` | checks that fail loudly on silently-wrong data |
| `posture`, `balance` | sway geometry, centre-of-pressure measures |
| `spectral`, `physio` | cardiac and respiratory peaks, band power, breathing rate |
| `dynamics` | DFA, multifractality, recurrence, surrogates, entropy |
| `group` | whether these people moved at the same moments |
| `align`, `circular` | offsets between clocks; directional statistics |
| `io`, `record` | one reader per corpus layout, a content sniffer, a common record type |

Readers: Qualisys and Qualisys-style TSV in all three header shapes, Sverm, Axivity AX3,
Physics Toolbox phone logs (raw app export or cleaned), Equivital, Wii balance board, Artinis
fNIRS. `read` dispatches on content rather than extension, because in this corpus the extension
is frequently wrong.

## Why it exists

It was built while constructing a single analysis across every dataset in the Oslo Standstill
Database. There were 159 analysis scripts, of which 58 defined their own band-pass filter and 37
computed quantity of motion. The project's central measure existed in dozens of copies that did
not all agree, and the disagreements were invisible.

Every default here was measured rather than assumed, and the reasoning is kept beside it. The
[traps page](https://github.com/fourMs/micromotion/wiki/Traps) lists the mistakes that shaped the
design; each one happened, produced a believable wrong number, and raised nothing at the time.

## Requirements

Python 3.10+, numpy, scipy, pandas.

## Licence

GPL-3.0-or-later. If you use it, please cite it — see [CITATION.cff](CITATION.cff).

## Related toolboxes

These four toolboxes come out of the [fourMs lab](https://github.com/fourMs) at the University of
Oslo. They are separate packages with separate release cycles, but they are built to be used
together and share several implementations, so a measure computed in one agrees with the same
measure computed in another.

- [Musical Gestures Toolbox](https://github.com/fourMs/MGT-python) (`musicalgestures`) — video and audio: motiongrams, videograms, and motion analysis from ordinary video files
- [ambiscape](https://github.com/fourMs/ambiscape) — soundscapes: the sonic ambience of a place, across level, spectral, spatial, temporal, ecological and source descriptors
- [musiscape](https://github.com/fourMs/musiscape) — music collections: comparing many tracks and albums held as audio files in folders
