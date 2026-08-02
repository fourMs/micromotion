# Stating that an effect is absent

Most interesting results in standstill research are nulls. No environment effect, no seasonal
rhythm, no coupling between people standing together, no association between how loud a room is and
how much a body moves.

A non-significant test does not support any of those claims. It says the data are compatible with no
effect — and equally compatible with an effect too small for this sample to resolve. Written up as
"we found no effect", that is an overclaim, and reviewers say so.

Equivalence testing states the claim you actually want to make. Two one-sided tests invert the usual
logic: the null becomes "the effect is at least this large", and rejecting it lets you say the effect
is smaller than that. The bound is a smallest effect size of interest, and choosing it is a
scientific judgement rather than a statistical one. That is the point. "No effect" is not a testable
claim; "smaller than half a millimetre per second" is.

## Four outcomes, not two

```python
import micromotion as mm

r = mm.tost_paired(silence, music, bound=0.5)      # bound in mm/s, the data's own units
r["verdict"]      # 'effect' | 'equivalent' | 'trivial' | 'inconclusive'
mm.equivalence.interpret(r)                        # one sentence a report can quote
```

| verdict | meaning | what to write |
|---|---|---|
| `effect` | significant, and the interval reaches past the bound | there is an effect worth caring about |
| `equivalent` | not significant, and the interval fits inside the bound | the effect is smaller than the bound |
| `trivial` | significant AND inside the bound | real, and too small to matter |
| `inconclusive` | neither | this sample cannot decide |

`inconclusive` is the one that matters most, because it is the case usually reported as "no effect".
A small study that finds nothing has not shown that nothing is there.

## The functions

```python
mm.tost_paired(a, b, bound)                  # paired difference, bound in the data's units
mm.tost_independent(a, b, bound)             # two groups, Welch standard error
mm.equivalence_correlation(r, n, bound)      # a correlation, from r and n alone
```

`equivalence_correlation` takes `r` and `n` rather than the raw series, so it can be applied to a
correlation that is already published without recomputing it.

## Choosing the bound

Give it in the units the reader thinks in. A bound of 0.5 mm/s says "differences smaller than half a
millimetre per second are not interesting here", which is a claim someone can argue with. A bound of
`d = 0.2` is a claim about a standardised quantity that nobody has an intuition for.

For correlations, 0.2 is a common convention for "small", but the honest move is to say what
magnitude would change your conclusion and use that.

## Worked example

Five nulls from a standstill corpus, at a bound of `r = 0.20`:

| claim | r | n | 90 % CI | verdict |
|---|---|---|---|---|
| body tracks the environment, within-day | +0.010 | 365 | [−0.076, +0.096] | equivalent |
| year-long habituation, one device | −0.050 | 331 | [−0.140, +0.041] | equivalent |
| movement settling vs heart-rate settling | +0.100 | 365 | [+0.014, +0.185] | equivalent |
| coupling between three performers | +0.015 | 69 | [−0.185, +0.214] | inconclusive |
| coupling between sixteen performers | +0.070 | 16 | [−0.368, +0.483] | inconclusive |

The first three can be stated as findings. The last two cannot, and that is the useful part: with
sixteen people the interval runs from −0.37 to +0.48, so the same data is compatible with nothing
and with a substantial coupling. No further analysis will separate them. The sample size is the
binding constraint, not the choice of measure — and that is a more honest thing to report than a
p-value of 0.8.
