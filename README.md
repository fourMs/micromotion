# micromotion

Analysis of human micromotion in motion time series: optical marker data, body-worn
accelerometers, respiration belts and force plates.

The measure the package exists for is **quantity of motion** — the average speed of a body part,
band-limited to 0.2–10 Hz, in millimetres per second. It applies equally to all three sensor
families because the shared abstraction is the frequency band, not the instrument.

```bash
pip install micromotion
```

```python
import micromotion as mm

rec  = mm.read("Standstill2017/mocap_data/A0001.tsv")   # dispatches on content, not extension
head = rec.marker("P01")
mm.qom(head, rec.fs, kind="position").mean_mm_s
```

## Documentation

| | |
|---|---|
| [Reference documentation](https://fourms.github.io/micromotion/) | how to use it, every function, the conventions |
| [Wiki](https://github.com/fourMs/micromotion/wiki) | why it works this way — traps, recipes, open questions |
| [Changelog](CHANGELOG.md) | what changed, and why |

New to it: [Getting started](https://fourms.github.io/micromotion/quickstart/), then
[The two bands](https://fourms.github.io/micromotion/conventions/), which is the one convention
you cannot skip.

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
