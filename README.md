# Micromotion

A Python toolbox for analysing human micromotion in various types of motion time series coming from optical marker data, body-worn accelerometers, respiration belts, and force plates.

The package is built around analysing *quantity of motion* (QoM), band-limited to velocities of 0.2–10 Hz, in millimetres per second. It applies equally to data from different sensor families because the shared abstraction is the frequency band, not the instrument.

## Why it exists

The toolbox was built during construction of a unified analysis for all datasets in the Oslo Standstill Database. There were numerous scripts floating around and many different ways of filtering, thresholding and calculating QoM. To avoid further confusion, this toolbox aims to consolidate all of this into a single package. 

## Install

```bash
pip install micromotion
```

## Requirements

Requires numpy, scipy and pandas.

## Use

```python
import micromotion as mm

rec = mm.read("Standstill2017/mocap_data/A0001.tsv")   # dispatches on content
head = rec.marker("P01")
mm.qom(head, rec.fs, kind="position").mean_mm_s
```

`read` identifies the layout from the file itself rather than from its extension because not all TSV or CSV files are correctly formatted.

## The two bands, and why it matters

There are two conventions in use: 

- `micromotion` is defined as being within the range 0.2–10 Hz. It is the only band that can be applied to every sensor, so it is the one any cross-collection comparison must use.

- `optical_legacy` is a 10 Hz low-pass with no lower edge. It keeps sub-0.3 Hz postural drift, which, for optical position, is real movement, and it is the convention by which earlier Championship of Standstill figures were computed. On the 2015 championship, it reads 15.5 per cent above the band-pass. A 10 Hz low-pass filter degenerates as the sampling rate approaches 20 Hz, where its cutoff reaches the Nyquist frequency, and it stops filtering entirely. On the 20 Hz origin dataset, it removed 0.0000 per cent of the signal energy while removing 27.9 per cent from 200 Hz recordings, so the two were never the same measure. It is kept to reproduce older results, not to produce new ones.

The lower edge is optional for position and mandatory for acceleration: gravity is a DC term, and integrating a signal with any residual offset produces a ramp that swamps the result. Asking for `optical_legacy` on accelerometer data raises an error rather than returning a plausible number.

## Toolbox Rules

Here are some important principles to keep in mind: 

**Measure the rate, do not read it.** `measured_rate` counts samples over the elapsed span rather than inverting the median interval. When timestamps are rounded to milliseconds, the median-interval route returns exactly 250 Hz for a recording that runs at 256 Hz, and 636 Hz for one arriving at 106 Hz.

**Downsample, never upsample.** `to_rate` raises rather than upsampling. Upsampling invents structure between samples, and every method that reads across scales treats the invention as real. An analysis that upsampled 20 Hz data to 25 Hz once produced multifractal widths up to 6.6 where the plausible range is about 1. Nothing failed; the numbers were simply wrong.

**Gaps are NaN, never a sentinel.** Readers convert the Qualisys zero triplet, the phone's exact-zero rows, the respiration belt's rails and the balance board's no-load (0.5, 0.5) centre of pressure. Each of those is a plausible-looking value that is not a measurement.

**Say which end of the series you trust.** `QomResult.binned` flags both the partial final bin and the filter transients at each end rather than dropping them silently. Including the partial bin, once inflated, increased the deposited series three- to fourteenfold.

## What is in it

| Module | Contents |
|---|---|
| `filters` | the band definitions, band-pass, low-pass, high-pass, cardiac notch |
| `qom` | quantity of motion from position or acceleration; raw, compensated and tilt-corrected variants |
| `resample` | rate measurement, quality metrics, downsample-only resampling, irregular-to-regular gridding |
| `spectral` | cardiac and respiratory peaks, band power |
| `dynamics` | DFA, multifractal DFA, stabilogram diffusion, recurrence quantification, IAAFT and other surrogates, time reversal, sample entropy, phase locking |
| `io` | one reader per corpus layout, and a content sniffer |
| `record` | the common structure the readers return |

## Validation

Every numerical claim the package makes is covered by a test against a process whose answer is known in advance: white noise, pink noise and Brownian motion for the scaling exponents, a binomial cascade for multifractality, an Ornstein–Uhlenbeck process for the diffusion crossover, and a sinusoid of known amplitude and frequency for quantity of motion itself, which both the position route and the acceleration route must recover.

Two of these restore a check that had been lost. A report stated that the time-reversal test returns z = +0.85 on an AR(1) process and z = −41 on a logistic map, but no script implementing it survived; those are now tests.

Against the corpus, the package reproduces the deposited Taqāsīm quantity of motion to within 0.1 per cent on four of five subjects, and the cardiac peak it finds in the StillStanding365 phone signal matches the wrist heart rate at a median ratio of 0.99.

```bash
python -m pytest micromotion/tests -q
```

## Relationship to other packages

This toolbox builds on and connects with these packages: 

- `musicalgestures` takes video in and produces visual features
- `ambiscape` takes spatial audio and produces soundscape features
- `respy` handles respiration belts.

Each of those either produces a time series or analyses one signal type. This package is the analysis layer several of them feed into. The interface is a pipe, not a merge: they produce time series; micromotion consumes them.

## Licence

GPL-3.0-or-later.
