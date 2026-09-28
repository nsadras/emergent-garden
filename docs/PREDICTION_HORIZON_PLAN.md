# Prediction timescale experiment

Status: all three passive comparisons and six new native 600-second starts are
complete. Two-second sensory predictions beat zero in two communities, and
recent-rate extrapolation in all three. Native births improve in two starts,
but own feedback still loses to shuffling in two. See
[the results](PREDICTION_HORIZONS.md). The shorter horizon remains optional.

V23's extra sensory features modestly improve paired forecasts, but still fail
to beat predicting zero in two communities. A faster rate worsens accuracy.
Clipping is asymmetric yet nearly absent in the third community, and most
updates do not hit the weight norm bound. Test forecast timescale before
changing those bounds or adding more inherited layers.

Replay the same three V22 direct-return communities with two passive predictors
at maximum rate .02. Retain the hidden-only and hidden-plus-input representations,
100-second warmup, identical mature forecast cohorts, two-second eligibility,
norm bound 4, and clipped TD updates. Change only their discount horizon from
20 seconds to two seconds. Native bodies, controllers, learning, and world laws
must remain unchanged; full physical states should reproduce the existing
paired diagnostic. Each predictor must be evaluated against the discounted
returns it was actually trained to predict.

The shorter horizon makes the objective more local in time. It also changes
the relationship between discount and eligibility; this is not an isolated
information-content or architecture test. Compare each predictor with zero
and recent-rate extrapolation under its own horizon, and compare the two feature
sets against exactly the same outcomes. Do not rank the absolute MSE of different
return targets as if they measured the same task. Include deaths and the existing
age subgroup without choosing creatures by their later outcomes.

If near-term energy becomes predictably learnable, test the shorter horizon in
native motor control with matched ecological and shuffled-feedback comparisons.
If it remains weak, inspect representation and reward timing before expanding
the controller further. This experiment uses an existing V23 parameter and
adds no native V24 mechanism.
