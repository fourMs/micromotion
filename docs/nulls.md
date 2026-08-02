# Stating that an effect is absent

Many of the most interesting results in our standstill research are nulls. We have found no effect
of the environment, no seasonal rhythm, no coupling between people standing together, and no
association between how loud a room is and how much a body moves.

The problem is that a non-significant test does not support any of those claims. It only says that
the data are compatible with no effect, and they are equally compatible with an effect that is too
small for this sample to resolve. If we write that up as "we found no effect," we are overclaiming,
and reviewers rightly say so.

Equivalence testing lets us state the claim we actually want to make. Two one-sided tests invert the
usual logic, so that the null becomes "the effect is at least this large." Rejecting it lets us say
that the effect is smaller than that. The bound is a smallest effect size of interest, and choosing
it is a scientific judgement rather than a statistical one. I think that is the point. "No effect"
is not a testable claim, while "smaller than half a millimetre per second" is.

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
| `trivial` | significant and inside the bound | real, and too small to matter |
| `inconclusive` | neither | this sample cannot decide |

I find `inconclusive` the most useful of the four, because it covers the case that is usually
reported as "no effect." A small study that finds nothing has not shown that nothing is there.

## The functions

```python
mm.tost_paired(a, b, bound)                  # paired difference, bound in the data's units
mm.tost_independent(a, b, bound)             # two groups, Welch standard error
mm.equivalence_correlation(r, n, bound)      # a correlation, from r and n alone
```

We wrote `equivalence_correlation` to take `r` and `n` rather than the raw series, so that it can
be applied to a correlation that is already published without recomputing it.

## Choosing the bound

Give the bound in the units the reader thinks in. A bound of 0.5 mm/s says that differences smaller
than half a millimetre per second are not interesting here, which is a claim someone can argue
with. A bound of `d = 0.2` is a claim about a standardised quantity that nobody has an intuition
for.

For correlations, 0.2 is a common convention for "small," but I would rather say what magnitude
would change my conclusion and use that.

## Worked example

Here are five nulls from our standstill corpus, at a bound of `r = 0.20`:

| claim | r | n | 90 % CI | verdict |
|---|---|---|---|---|
| body tracks the environment, within-day | +0.010 | 365 | [−0.076, +0.096] | equivalent |
| year-long habituation, one device | −0.050 | 331 | [−0.140, +0.041] | equivalent |
| movement settling vs heart-rate settling | +0.100 | 365 | [+0.014, +0.185] | equivalent |
| coupling between three performers | +0.015 | 69 | [−0.185, +0.214] | inconclusive |
| coupling between sixteen performers | +0.070 | 16 | [−0.368, +0.483] | inconclusive |

We can state the first three as findings. We cannot state the last two, and I find that the useful
part. With sixteen people the interval runs from −0.37 to +0.48, so the same data are compatible
with nothing and with a substantial coupling. No further analysis will separate them. Here the
sample size is the binding constraint, not the choice of measure, and I would rather report that
than a p-value of 0.8.
