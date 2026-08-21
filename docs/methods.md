# Methods

What each analysis answers, how this package computes it, and where the method comes from.
Grouped by the question being asked of a signal rather than by module.

For a first reading, the first section is the one that matters. The rest can be treated as
reference.

---

## How much did it move?

### Quantity of motion

The measure the package is built around: the average speed of a body part, band-limited, in
millimetres per second.

For position input the trajectory is band-pass filtered, differentiated, band-pass filtered
again, and the Euclidean norm taken per sample. For acceleration input the signal is
band-passed, integrated to velocity, band-passed again to remove integration drift, and the
norm taken. The second filter is not optional in either case, since differentiation amplifies
the high end and integration accumulates the low end, so a signal band-limited only before the
operation is not band-limited after it.

```python
mm.qom(data, fs, kind="position", unit="mm")        # or kind="acceleration"
mm.speed_from_position(xyz, fs, unit="mm")
mm.speed_from_acceleration(acc, fs, unit="g")
```

`qom` returns a `QomResult` carrying `mean_mm_s`, `median_mm_s`, the full `speed` series, the
rate, the variant, and `edge_samples`, the length of the filter transient at each end.

Reduce the speed series with a median or a mean, and say which. They are not interchangeable:
on one corpus of 365 daily recordings the two ranked the recordings at a correlation of 0.44,
the mean running 1.6 times the median typically and 6.8 times at worst, because the mean follows
occasional large movements and the median the baseline. Reporting one after tuning a parameter
on the other is a mistake this package cannot prevent.

The measure has a long history in music-related movement research, where it is usually computed
from video frame differences; the formulation here follows the motion-capture and accelerometer
line of that work.

- Jensenius, Zelechowska & Gonzalez-Sanchez (2017). The musical influence on people's micromotion
  when standing still in groups. *Sound and Music Computing*.
- Burger & Toiviainen (2013). MoCap Toolbox—a Matlab toolbox for computational analysis of
  movement data. *Sound and Music Computing*.

### Spatial extent

Speed is not the only summary. How large a region the body occupies is a partly independent
quantity, and in one large corpus how much a person moved predicted only about a quarter of how
large a region they covered.

```python
mm.sway_geometry(xy)              # anisotropy, principal axis, ellipse area, path length
mm.ellipse_area_95(xy)            # 95 % confidence ellipse
mm.convex_hull_area(xy)
mm.path_length(xy, fs)            # {'path', 'path_rate'}
mm.posture.dispersion_radius(xy)
mm.spatial_extent(pos, fs)        # windowed extent from a full (n, 3) trajectory
```

The 95 % confidence ellipse and the sway-path measures follow the standard posturographic
definitions.

- Prieto, Myklebust, Hoffmann, Lovett & Myklebust (1996). Measures of postural steadiness:
  differences between healthy young and elderly adults. *IEEE Transactions on Biomedical
  Engineering*.

---

## Where did it go, and in which direction?

### Centre of pressure

```python
mm.cop_sway_metrics(cop_xy, fs=100.0)     # fs is keyword; the second positional slot is t
mm.principal_axis_projection(xy)
mm.sway_orientation(xy)
mm.spectral_edges(x, fs)
```

Force-plate and balance-board measures: path length, area, mean velocity, the medio-lateral and
antero-posterior split, and the frequency below which a given fraction of the sway power lies.
Same reference as above.

### Circular statistics

Directions are not ordinary numbers, since 359° and 1° are two degrees apart rather than 358, so
they need their own mean and their own tests.

```python
mm.circular.circ_mean(angles)             # radians throughout this module
mm.circular.rayleigh(angles)              # is there a preferred direction at all?
mm.circular.rayleigh_axial(angles)        # ...for an axis, where 0 and 180 are the same
mm.circular.axial_dispersion(angles)
mm.circular.circ_corr(a, b)               # circular-circular
mm.circular.circ_corr_linear(angles, x)   # circular-linear
mm.circular.vtest(angles, mu)             # a preferred direction at a *predicted* angle
```

Postural sway is usually axial rather than directional, since a body rocking along a line visits
both ends of it, so `rayleigh_axial`, which doubles the angles before testing, is normally the
right one. Using the ordinary Rayleigh test on axial data will report no preferred direction
where there plainly is an axis.

!!! warning "Two axial tests, two angle units"

    `circular.rayleigh_axial` takes **radians** and `balance.axial_rayleigh` takes **degrees**.
    In their own units they agree with each other and with the textbook to 2e-16. Feeding
    radians to the degrees function does not raise: every angle collapses towards zero and `R`
    runs towards 1, which reads as an extremely concentrated axis.

- Fisher (1993). *Statistical Analysis of Circular Data*. Cambridge University Press.
- Mardia & Jupp (2000). *Directional Statistics*. Wiley.
- Berens (2009). CircStat: a MATLAB toolbox for circular statistics. *Journal of Statistical
  Software*.

---

## Resampling before a short-lag estimator

Bring series to a common rate with `mm.to_rate`, not with a bare `scipy.signal` call.

```python
mm.to_rate(x, fs_in, fs_out)      # anti-aliased, and refuses to upsample
```

Two failure modes, both measured on this corpus rather than supposed.

**Aliasing.** A bare polyphase resample folds near-Nyquist content into the band. On optical
head-marker position it leaves the first difference anti-correlated at -0.70, where `to_rate` leaves
it at +0.95. Whether that matters depends entirely on where the estimator reads:

| estimator | reads | effect of the bare call |
|---|---|---|
| short-term Hurst exponent | the shortest lags | 0.107 against 0.908 — a reversed conclusion |
| recurrence, sample entropy | short lags | at risk, same route |
| multifractal width | a range of scales | 0.002 |
| multifractal width, awkward rate ratio | a range of scales, 25 to 20 Hz | 1.115 against 0.768 |

So the rule is not "never resample". It is that a local resampler feeding a short-lag estimator is
the combination that has cost a conclusion, and an awkward rate ratio can reach the others too.

**Band-limiting first immunises you**, and that is usually the cheapest fix. The fault needs
broadband content near Nyquist to fold down; filtering to the analysis band removes it before the
resampler can. Measured on one corpus analysis that band-limits to 0.2--5 Hz and then decimates to
5 Hz, a bare `resample_poly` and `to_rate` agree to 0.03 per cent and give identical lag-one
autocorrelations. The analysis that lost a conclusion resampled RAW position with the full tracking
noise floor still in it. So the question to ask of any pipeline is not "does it resample?" but "does
it resample something unfiltered, and does anything downstream read the shortest lags?"

**Upsampling.** `to_rate` raises rather than upsampling, because interpolation invents structure
between samples and a scaling exponent reads that invention as real. A corpus mixing 20 and 100 Hz
recordings will hit this; the answer is to analyse at a rate every series reaches, or to exclude the
slow ones, not to interpolate them up.

Where series must be compared across rates, state scale-dependent parameters in SECONDS. `mm.dfa`
takes `fs` and `min_scale_s` for exactly this: its sample-count default is a different physical
scale at every rate.

---

## Which unit is this file in?

```python
mm.identify_acceleration_unit(acc)      # -> "g", "mg" or "m/s^2"
```

No accelerometer format in this field declares its units, and the conventions in use are three
orders of magnitude apart. A device that is mostly stationary is mostly measuring gravity, so the
median vector norm of a resting stretch sits near 1, 981 or 9.81 and nothing plausible lies between
them. Pass a stationary stretch and the answer is unambiguous; pass a recording that is not
gravity-dominated and it raises rather than guesses.

Four accelerometers recorded simultaneously on one body in the Oslo corpus stored their values in
three different conventions, none of them stated in any file. Get one wrong and that recording is
scaled by 9.8 or 981 — while every correlation, rank statistic and reliability estimate stays
exactly the same, so nothing downstream complains. The tell is a suspiciously ROUND ratio against a
known value: a real disagreement is ragged and a unit error is a constant.

---

## One missing sample voids the whole series

Every band-limiting call here is zero-phase: the filter runs forwards and then backwards, so each
input sample influences every output sample. A single NaN therefore does not leave a hole, it
leaves nothing — the returned array is NaN throughout.

That is easy to miss, because nothing raises. A caller takes the median of the result, gets NaN,
drops the recording as unusable, and never learns that one absent sample out of thousands is what
made it unusable. Measured on StillStanding365: 108 of its 365 days carry at least one missing
accelerometer sample, and 242 missing samples across the year account for all of them. The obvious
analysis path discards 30 per cent of the record and says nothing.

`bandpass`, `lowpass` and `highpass` now warn, and the warning reaches anything routed through
them, `speed_from_acceleration` included. What to do about it depends on the gap:

- ISOLATED SAMPLES, which is what dropout usually looks like: interpolate before filtering.
  Linear is enough at these rates, and a handful of samples in thousands cannot move a band-limited
  descriptor.
- LONG GAPS: split the series and analyse the pieces, or drop the recording. Interpolating across
  seconds invents low-frequency content, and the micromotion band is where that lands.

Do not simply drop the non-finite samples. That closes the gap by shortening the series, which
shifts every later sample earlier in time and puts a step where the gap was.

---

## At what frequencies?

```python
mm.spectral_peak(x, fs, band)
mm.peak_from_spectrum(f, p, band)     # the same rule, on a spectrum you already have
mm.band_power(x, fs, band)
mm.band_rms(x, fs, band)
mm.band_power_fraction(x, fs, {"resp": (0.1, 0.5), "card": (0.7, 2.2)})
mm.band_share(x, fs, num_band=(0.7, 2.2), den_band=(0.1, 3.0))
mm.band_share(x, fs, num_band=(0.7, 2.2), den_band=(0.1, 3.0),   # the other convention
              integrate="sum", interval="half_open")
mm.band_share_from_spectrum(f, p, num_band=(0.7, 2.2), den_band=(0.1, 3.0))
mm.mean_frequency(x, fs, band)
mm.cardiac_peak(x, fs)
mm.respiratory_peak(x, fs)
mm.respiration_rate(x, fs)
mm.detect_breaths(x, fs)
mm.is_band_floor(f, p, band)                        # is the largest value in there a peak at all
mm.band_edge_sweep(signals, fs, band)               # does the answer follow the band edge
```

Spectra are estimated by Welch's method, with priors on where to look and a signal-to-noise
criterion, so that a peak is only returned when it stands above the local background.

`peak_from_spectrum` is `spectral_peak` without the transform in front of it, for the common case
of computing one spectrum and reading several bands off it. It exists so that the rule is imported
rather than reimplemented: recomputing a Welch per band to get at the rule is wasteful, and a rule
that is easier to copy than to import gets copied.

`is_band_floor` is that same rule asked as a yes-or-no question about a spectrum, and it is what
`cardiac_peak`, `dominant_frequency`, `instantaneous_rate` and `respiration_rate` call before
warning. Give it a raw spectrum and an averaged one: a band-pass over the same band, or a lightly
averaged Welch, both make it call a slope a peak. See
[a bounded search returns its own boundary](conventions.md#a-bounded-search-returns-its-own-boundary).

`band_edge_sweep` moves the lower edge of a search band and reports whether the answer moves with
it. That is the test for an estimate that is its own band edge without ever landing on it, which is
the case a check for equality with the edge cannot see.

`band_share` is the fraction of power in one band over another, with both bands mandatory and
keyword-only. Four hand-rolled versions of that fraction produced four published-looking shares —
38, 43, 45 and 58 per cent — that were quoted against one another while resting on different,
sometimes unstated, denominators, and one of the four is untraceable to any measurement. Quote a
share with its domain, its site and both bands, or not at all; see
[the four bands](conventions.md) for the rule. `band_share_from_spectrum` is the same split as
`peak_from_spectrum`. If two channels have to meet at one rate before their shares are compared,
resample with `to_rate`, which anti-aliases and refuses to upsample; plain interpolation onto a
slower clock folds sensor noise into the cardiac band.

The bands are not the whole label. `integrate` picks the quadrature rule, `"trapezoid"` or
`"sum"`, and `interval` picks what the edges mean, `"closed"` or `"half_open"`; the defaults are
`"trapezoid"` and `"closed"`, which is what `band_power` and `band_power_fraction` compute, and
`integrate="sum", interval="half_open"` is what `spectral_band_fractions` computes. On
chest-accelerometer standstill data the rule alone moves a respiratory share by up to 0.034
absolute on a share of about 0.16 and the closure by up to 0.025, so the two are not
interchangeable — see [the arithmetic label](conventions.md#and-the-arithmetic-which-is-the-fourth-label).

`cardiac_peak` works this way. `respiratory_peak` does not, and the reason is worth knowing
before reaching for a spectrum on any slow rhythm. A periodogram of belt or body motion is red:
power falls with frequency, so a breathing bump sits on a much larger downward slope and never
becomes the global maximum inside the band. Measured against sixteen thoracic belts, a spectral
version returned a median 7.5 breaths per minute where the belts' own breath timing gives 16.8,
and it ranked participants at Spearman −0.32 against that timing, which carries no usable
information about who was breathing faster. Four repairs were tried and all rejected: raising
the band floor, band-passing first, taking the most prominent local maximum, and dividing out a
fitted power law.

An earlier version of that list said band-passing first "cannot help, since the maximum inside a
band is unaffected by filtering inside that same band". The second half is false. A Butterworth is
not flat inside its own passband: its rising skirt reaches in, and multiplying a falling spectrum
by that skirt moves the maximum up. Over a 0.12-0.40 Hz band on twenty synthetic 1/f series the
unfiltered maximum sits at 1.11 times the lower edge, a fourth-order zero-phase band-pass moves it
to 1.25 and a second-order one to 1.39, and filtered and unfiltered agree on 6 and 4 of the twenty.
Band-passing is still not a repair, but for the opposite reason to the one given: the answer is the
band edge either way, and filtering moves it about a fifth further from the edge, which is why it
is harder to spot.

`respiratory_peak` is therefore measured in the time domain, from `detect_breaths`, and returns
that rate in Hz. Its `window_s` argument is accepted and ignored. `cardiac_peak` keeps the
periodogram, because its band sits above the slope and the ballistocardiac impulse is a
genuinely prominent peak.

A standing body carries both signals mechanically. The heartbeat appears in whole-body motion as
a ballistocardiographic component, which is the recoil of blood ejection, and breathing appears
as a slow postural oscillation. Both are inside the micromotion band, which is why a measure of
"stillness" is partly a measure of physiology.

- Welch (1967). The use of fast Fourier transform for the estimation of power spectra.
  *IEEE Transactions on Audio and Electroacoustics*.
- Inan, Migeotte, Park, Etemadi, Tavakolian, Casanella, Zanetti, Tank, Funtova, Prisk & Di Rienzo
  (2015). Ballistocardiography and seismocardiography: a review of recent advances. *IEEE Journal
  of Biomedical and Health Informatics*.

!!! note "Two peaks that are easy to confuse"

    Respiration sits around 0.2–0.4 Hz and the heartbeat around 1–2 Hz, but the second harmonic
    of respiration can land near the low end of the cardiac search band. Check that a "cardiac"
    peak is not twice the respiratory one before believing it.

### Where in the breath cycle?

```python
mm.respiration_onsets(x, fs)      # inspiration and expiration onset times
mm.respiratory_phases(x, fs)      # boolean masks over the cycle
```

A breathing rate says how often. This says where in each cycle the body is, which is the more
useful question when relating breathing to stillness: the post-expiration pause is the moment in
the cycle when the body is most nearly still, so a measure of micromotion that ignores phase is
averaging across a systematically varying quantity.

Inspiration onset is defined by chest-expansion *velocity* crossing a threshold rather than by a
local minimum. That distinction is what makes it usable on quiet standing, where a belt also
picks up sway, weight shifts and swallows, all of which produce local minima with no breath
behind them. The threshold comes from the signal's own distribution, and a candidate rise is
kept only if it contains an upward crossing of a heavily low-passed copy of the signal, so a
rise that never returned to an exhaled baseline is rejected. Expiration onset is the end of the
rise, which assumes passive expiration.

`respiratory_phases` returns `inspiration` and `expiration`, high-flow moments judged both
against the whole recording (`*_high`) and within each individual breath (`*_v`), and
`post_expiration`. Prefer the per-breath variants where breath size varies across a recording,
which it does while someone is settling.

- Upham (2018). *Detecting the Adaptation of Listeners' Respiration to Heard Music*. PhD thesis,
  New York University. The phase definitions and the default thresholds come from its
  coordination analysis.

!!! note "Provenance, and why this is a port"

    This is Finn Upham's method, from the [`respy`](https://pypi.org/project/respy/) package
    (MIT), reimplemented here on numpy with permission. The reimplementation was necessary:
    `respy.Resp_phases` assigns through `df[col].loc[idx]`, which under pandas copy-on-write
    does not write, so from pandas 2 onward it returns all twelve of its phase columns empty
    without raising anything. Measured against `respy` 0.1.1 on pandas 3.0.3, every phase column
    came back 0.0 per cent populated. The onset detector, which still works there, was used as
    the oracle: across six recordings and 2331 breaths the two agree on inspiration onset to a
    median of 0.000 s, with this port finding 1.4 per cent more breaths at the recording
    boundaries.

---

## How is it structured in time?

These are the nonlinear time-series methods. They ask not how much a signal moved but how its
fluctuations are organised across timescales, which turns out to distinguish standing bodies
more sharply than amplitude does.

### Detrended fluctuation analysis

```python
mm.dfa(x)                  # -> float, the scaling exponent alpha
mm.dynamics.dfa(x)         # -> dict with 'alpha', the scales and the fluctuations
```

Two entry points, because two callers want different things. The top-level `mm.dfa`, which comes
from `balance`, returns the exponent alone; `dynamics.dfa` returns the fitted curve as well, for
checking that the scaling is straight before believing the slope.

The method integrates the series, splits it into windows of many sizes, removes a polynomial
trend inside each, and measures how the residual fluctuation grows with window size. The slope
on log-log axes is the scaling exponent α: 0.5 is white noise, 1.0 pink noise, 1.5 Brownian
motion. Above 0.5 the series is persistent, which means a deviation tends to be followed by more
of the same.

It exists because ordinary autocorrelation and spectral estimates are unreliable on
non-stationary signals, and postural data is non-stationary by nature.

- Peng, Buldyrev, Havlin, Simons, Stanley & Goldberger (1994). Mosaic organization of DNA
  nucleotides. *Physical Review E*.
- Peng, Havlin, Stanley & Goldberger (1995). Quantification of scaling exponents and crossover
  phenomena in nonstationary heartbeat time series. *Chaos*.
- Duarte & Zatsiorsky (2000). On the fractal properties of natural human standing. *Neuroscience
  Letters*.

### Multifractal DFA

```python
mm.dynamics.mfdfa(x)        # -> h(q), spectrum, width
```

The same procedure repeated for many moments *q*, which weights large and small fluctuations
differently. If one exponent describes the whole series the spectrum is narrow; if large and
small fluctuations scale differently it is wide.

- Kantelhardt, Zschiegner, Koscielny-Bunde, Havlin, Bunde & Stanley (2002). Multifractal
  detrended fluctuation analysis of nonstationary time series. *Physica A*.

!!! warning "A wide spectrum is more often a preprocessing fault than a finding"

    Widths above about 2 for postural data should prompt a check of the pipeline. Upsampling
    alone has produced widths up to 6.6 where the plausible range is around 1, since the
    interpolation creates structure between samples and the method reads it as real.

### Stabilogram diffusion analysis

```python
mm.dynamics.sda(x, y, fs=100.0)     # -> Hs, Hl, critical time and displacement
mm.stabilogram_diffusion(xy, fs)    # the same analysis from an (n, 2) array
```

Treats sway as diffusion and plots mean-square displacement against time lag. Two regimes
appear: over short intervals the body drifts away from where it was (open-loop, exponent above
0.5), and over longer ones it is pulled back (closed-loop, exponent below 0.5). The crossover
between them is estimated by fitting two lines and taking their intersection, and is interpreted
as the point at which postural feedback engages.

- Collins & De Luca (1993). Open-loop and closed-loop control of posture: a random-walk analysis
  of center-of-pressure trajectories. *Experimental Brain Research*.

### Entropy

```python
mm.dynamics.sampen(x, m=2, r=None)     # sample entropy
mm.dynamics.apen(x, m=2, r=None)       # approximate entropy
```

How unpredictable the series is: the negative log probability that two segments matching for *m*
samples still match for *m* + 1. Higher means less regular. Sample entropy corrects a bias in
approximate entropy caused by self-matches and should be preferred for new work; approximate
entropy is provided for comparison with older results.

- Pincus (1991). Approximate entropy as a measure of system complexity. *PNAS*.
- Richman & Moorman (2000). Physiological time-series analysis using approximate entropy and
  sample entropy. *American Journal of Physiology—Heart and Circulatory Physiology*.

### Recurrence quantification

```python
mm.dynamics.rqa(x, dim=3, tau=None, rr=0.05)   # determinism, laminarity, entropy, trapping time
mm.dynamics.embed(x, dim, tau)
mm.dynamics.ami(x); mm.dynamics.first_ami_minimum(x)
```

Reconstructs a state space from the series by time-delay embedding, then measures how often the
trajectory revisits earlier states. Diagonal structures in the recurrence plot indicate
deterministic dynamics; vertical ones indicate the system becoming trapped in a state.

The threshold is solved per plot, so that a fixed fraction of pairs count as recurrent. With a
fixed absolute threshold, determinism partly measures how tightly packed the trajectory is, and
two recordings of different amplitude are then not comparable. That is a trap worth knowing
about when reading the older literature.

The embedding delay defaults to the first minimum of the average mutual information, which is
the standard choice for a nonlinear system where the first autocorrelation zero would be wrong.

- Eckmann, Kamphorst & Ruelle (1987). Recurrence plots of dynamical systems. *Europhysics Letters*.
- Webber & Zbilut (1994). Dynamical assessment of physiological systems and states using
  recurrence plot strategies. *Journal of Applied Physiology*.
- Marwan, Romano, Thiel & Kurths (2007). Recurrence plots for the analysis of complex systems.
  *Physics Reports*.
- Fraser & Swinney (1986). Independent coordinates for strange attractors from mutual
  information. *Physical Review A*.
- Takens (1981). Detecting strange attractors in turbulence. *Lecture Notes in Mathematics*.

### Time-reversal asymmetry

```python
mm.dynamics.trev(x, tau=1)
```

A linear Gaussian process looks statistically the same run backwards; most nonlinear ones do
not. The statistic is the skew of the lag-τ increments normalised by their variance, and it is
one of the more sensitive discriminators of nonlinearity. Use it with surrogates, below.

---

## How many things is a descriptor set measuring?

```python
mm.effective_dimensionality(x, rank=True, by=None)
```

A corpus accumulates descriptors one paper at a time, until a table carries amount, extent, sway
area, elongation, two frequency measures, a frozen fraction, burstiness and jerk, and nobody
asks how many of them are the same dimension under different names. If eleven collapse onto
three, most of the multiple-comparison arithmetic across those papers is wrong in the
conservative direction, and a table reporting all eleven is showing a reader one number several
times.

The answer is the eigenvalue spectrum of the descriptor correlation matrix, summarised three
ways: components needed for 80 and 90 per cent of variance, and the participation ratio
$(\sum\lambda)^2 / \sum\lambda^2$, which needs no cutoff and is not an integer.

Three choices matter more than the method:

- **Rank, not value.** These descriptors are heavy-tailed, since burstiness is a 99th percentile
  over a median. On raw values one long-tailed descriptor dominates the first component and the
  result becomes a statement about its outliers.
- **Standardise within group** (`by=` the session, edition or collection). Otherwise a
  between-group difference in level reads as shared variance. Descriptors do not become more
  correlated because two editions were recorded at different rates, but they look it.
- **Drop degenerate columns**, and say how many. A constant column contributes an eigenvalue of
  zero and silently deflates the participation ratio.

On 732 championship recordings and eleven descriptors the answer is a participation ratio of
about 3.6: four components for 80 per cent, five for 90. Eleven descriptors, three or four real
dimensions.

Not to be confused with `group.participation_ratio`, which is a different quantity with a
similar name, the fraction of a group whose movement decreased after an event.

## Is a measure a trait or a state?

```python
mm.intraclass_correlation(values, groups)     # needs the [mixed] extra
```

Dimensionality asks about a set of measures. This asks about one: how much of its variance is
the *person* and how much is the *occasion*. That is the intraclass correlation, fitted as a
random-intercept mixed model, `value ~ 1` with a random intercept per person, where `groups` is
the person and the residual is the session.

A high ICC means a trait: the measure identifies the body. A low one means a state, however
reliably it is measured within a session. On this corpus quantity of motion comes out near 0.7
and jerk near zero, so how much a person moves is characteristic of them, while how abruptly
they move is characteristic of the day. The same conclusion arrives independently from a
classifier asked to name the performer.

Two cautions the function makes explicit:

- **Check `boundary`.** A random-effect variance can be estimated at exactly zero, which is the
  optimiser hitting the edge of the parameter space rather than a measurement of no person
  effect. Printing `0.000` from such a fit implies a precision that is not there. This is common
  where the number of groups is small, and the standard design here, three performers, is small.
- **The log transform is on by default** for strictly positive input, because these are scale
  quantities whose residuals are otherwise skewed. It is reported in the result rather than
  assumed.

An ICC needs several observations per group and several groups. Three people across sixty-nine
sessions is enough to show that 0.7 and 0.0 differ; it is not enough to put a confidence
interval on either.

## Is that structure real?

A scaling exponent or an entropy value means nothing on its own, since the question is always
whether it differs from what a simpler process would give. That is what surrogate testing is
for.

```python
mm.dynamics.iaaft(x)                       # amplitude distribution + power spectrum preserved
mm.dynamics.phase_surrogate(x)             # power spectrum preserved, phases randomised
mm.dynamics.circular_shift_surrogate(x)    # within-series structure preserved, alignment destroyed
mm.dynamics.surrogate_test(x, mm.dynamics.trev, n=99, method=mm.dynamics.iaaft)
```

Choosing the surrogate is choosing the null hypothesis, and the choice is not a detail. A
phase-randomised surrogate tests against a linear Gaussian process with the same spectrum. An
IAAFT surrogate additionally preserves the amplitude distribution, so it will not report
nonlinearity merely because the data are non-Gaussian. A circular shift preserves everything
within a series and destroys only its alignment to something else, which is the right null for a
question about timing between signals.

Use at least 99 surrogates for anything reportable. With 25, the p-value cannot go below 0.038.

- Theiler, Eubank, Longtin, Galdrikian & Farmer (1992). Testing for nonlinearity in time series:
  the method of surrogate data. *Physica D*.
- Schreiber & Schmitz (1996). Improved surrogate data for nonlinearity tests. *Physical Review
  Letters*.
- Schreiber & Schmitz (2000). Surrogate time series. *Physica D*.

---

## Are two signals related?

```python
mm.dynamics.dcca(a, b, scales=None)         # correlation per timescale
mm.dynamics.plv(a, b, fs, band)             # phase locking
mm.xcorr_lag(a, b, fs)                      # offset between two sampled series
mm.search_lag(t_a, x_a, t_b, x_b)           # offset between two irregular series
mm.instantaneous_rate(x, fs)                # -> (times, rate), for aligning on a rhythm
mm.find_transient(x, fs)                    # a clap or a tap, for aligning on an event
mm.apply_lag(t, lag_s)                      # shift a time vector by a measured offset
```

**Detrended cross-correlation** answers the correlation question separately at each timescale.
Two bodies can be uncorrelated second to second and correlated over a minute, and one number
cannot express that. An ordinary correlation between two signals that each wander is dominated
by the wandering.

**Phase locking** measures whether two signals maintain a consistent phase relationship, which
is a different question from whether their amplitudes covary. Band-pass first: phase is only
well-defined within a band.

- Podobnik & Stanley (2008). Detrended cross-correlation analysis: a new method for analyzing two
  nonstationary time series. *Physical Review Letters*.
- Lachaux, Rodriguez, Martinerie & Varela (1999). Measuring phase synchrony in brain signals.
  *Human Brain Mapping*.

!!! warning "Never cross-correlate two drifting signals directly"

    Over 200 pairs of *independent* random walks, the best-lag correlation had a median of 0.47,
    exceeded 0.5 forty per cent of the time, and reached 0.98. `xcorr_lag` differences its
    inputs by default; the same pairs then never exceed 0.16. This is the spurious-regression
    problem and it will manufacture an alignment out of nothing.

---

## Did a group move together?

```python
mm.event_train(x, fs)                                  # continuous signal -> point process
mm.coincidence_test(trains, fs)                        # did events align across people?
mm.participation_ratio(series, event_times, fs)
mm.sliding_null(series, fs)
mm.sequential_stability(x)
```

`participation_ratio` answers a question that comes up wherever a group is exposed to a shared
stimulus: of the people present, what fraction moved less after this moment than before it? It
takes event times from outside, whether those are musical onsets, annotated moments or stimulus
boundaries, so the events can come from an analysis of the stimulus rather than from the
movement itself.

Signs are counted rather than sizes, deliberately. Participants in group recordings wear
different sensors in different places and their quantity of motion can differ by more than an
order of magnitude; any statistic that averages magnitudes is dominated by whoever wore the
noisiest device. A proportion is immune to that, and to missing data.

!!! danger "Do not test a group stilling statistic against 0.5"

    People standing or sitting still are not coin flips. Movement decays after any excursion, so
    at a randomly chosen moment rather more than half of a group is already slowing down.
    Testing against a half will find an effect in any recording whatsoever. Use `sliding_null`,
    which computes the same statistic at every moment of the same recording and gives the
    distribution the observed value should be compared against.

`coincidence_test` returns both `score`, the mean of −log10(p) across samples, and
`frac_significant`. Prefer `frac_significant` where coordination is sparse; the wiki's
[design decisions](https://github.com/fourMs/micromotion/wiki/Open-questions) page has the
measurements behind that advice.

The surrogate for `coincidence_test` shifts each person by an independent bounded random offset,
so every individual keeps their own event rate and local structure and only the between-person
alignment is destroyed. Shuffling samples instead would destroy each person's own rhythm along
with the alignment, and so would test a hypothesis nobody holds.

- Upham, Høffding & Rosas (2024). The stilling response. *Music & Science*.

---

## Leaning versus moving

```python
mm.remove_tilt(acc, gyro, fs)
mm.tilt_fraction(acc, gyro, fs)
```

A body-worn accelerometer cannot by itself separate translation from a change of orientation:
tilting the sensor rotates the gravity vector across its axes and looks exactly like
acceleration. With a gyroscope on the same unit the two can be separated. Measured on one
recording, tilt inflated the quantity of motion by a factor of 1.56, so this is not a small
correction for sensors worn on a body that sways.

Optical marker data does not have this problem, which is one reason optical and body-worn
measures of the same standing are not directly comparable.

---

## Choosing parameters

Most of these methods have parameters that change the answer. Where a default is chosen here,
it is because the alternative was measured and found worse, and the docstring says so. Three
worth knowing:

| Choice | Default here | Why |
|---|---|---|
| Filter band | 0.2–5 Hz | Lower edge swept across seven datasets, spread smallest at 0.2 Hz; upper edge set by the slowest device's Nyquist. `WIDEBAND` (0.2–10 Hz) for jerk where the rate allows. See [the four bands](conventions.md) |
| Embedding delay | first AMI minimum | The autocorrelation-zero rule assumes linearity |
| RQA threshold | fixed recurrence rate | A fixed absolute threshold confounds determinism with amplitude |

One choice is deliberately left to the caller: the integration rule for turning acceleration
into speed. Rectangle and trapezoid differ by about 0.26 per cent systematically, the rectangle
sum lagging by half a sample. Neither is universally right, so pick one and record it.
