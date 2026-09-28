# V22 experiment plan: learn to predict energetic returns

Status: implemented and evaluated. All 15 ecological pilots and six forecast
diagnostics are complete. CPU and RTX 5080 exact replay exercises pass through
births and module growth. Neither target establishes a reliable ecological
advantage; prospective prediction is also weak. See the
[complete results and limitations](VALUE_PREDICTION.md). No V22 long ecological
continuations or fixed-genotype assays were started. All V21 follow-ups are complete.

The present motor learner compares recent net energy flow against a running
average. It has no learned estimate of which neural states precede good or bad
outcomes. A small value readout could propagate information about delayed meals
back to the observations and actions preceding them. Test that hypothesis before
expanding inherited layers or increasing mutation further.

The reference is the linear, online actor–critic formulation in
[Degris, Pilarski, and Sutton (2012)](https://people.bordeaux.inria.fr/degris/papers/DegrisACC2012.pdf).
Use ordinary TD learning and eligibility traces, not their natural-gradient
optimizer or variance-scaled policy update. The ecological controller has partial
observations, changing recurrent features, bounded updates, and reproduction;
the reference does not establish convergence or benefits in this world.

## Mechanism under test

Give each body module a disposable linear value readout of its normalized
hidden features and bias. Start weights and traces at zero. At a controller
boundary, both predictions use the current value weights:

```text
centered_return = integrated_net_energy - previous_baseline_rate * elapsed
gamma = exp(-elapsed / prediction_horizon)
delta = centered_return + gamma V(current_features) - V(previous_features)
value_weights += learning_rate * bounded(delta) * previous_value_trace
```

The previous motor eligibility trace receives the same prediction error in
place of the existing centered return. Current features enter the critic trace
for the next transition; the current sampled action similarly enters the next
motor trace. Use separate readiness so a newborn does not invent a transition
from a nonexistent previous observation. Keep lifetime state in checkpoints,
and clear it at birth and module growth. Do not inherit acquired predictions.

The direct-target variant sets `motor_value_centered = 0`: its TD error uses
the integrated net-energy return without subtracting the running mean. The
temporal baseline is still recorded, but does not modify either the critic or
actor's TD signal in that mode. With the value head disabled, the original
centered-return learner remains exact under either setting. This is a focused
comparison because the centered variant's target itself changes with recent
experience. The reference's discounted-return formulation uses direct returns;
its average-reward formulation instead combines centering with discount 1.

Initial settings: a 20-second prediction horizon, two-second eligibility,
maximum critic learning rate .02 modulated by the existing inherited learning
trait, and a readout norm bound of 4 in normalized-return units. Retain the
existing actor learning rate, .15 motor-row norm bound, and exploration magnitude.
These are screening settings, not an established optimum. The isolated chain
test matches its analytical discounted returns and shifts its prediction error
to an earlier cue. That constructed representation is never seeded into the dish.

No food location, future source schedule, hidden nutritional rule, desired turn,
or fitness ranking enters the predictor. It sees only the existing circuit's
features and its own subsequent energetic returns. The first ecological test
should use independent noise and receptor radius 1 to isolate prediction from
the mixed V21 persistence and V20 reach effects. Existing learning-capacity costs
remain; no additional head-specific compute/memory energy charge is proposed in
this first comparison, which must be stated explicitly.

## Evidence required before calling this an improvement

1. Check transition timing, previous-feature credit, elapsed-time discounting,
   bounds, and fresh-state behavior against hand-computed cases.
2. In a constructed chain with known future returns, verify that the predictor
   learns them and transfers prediction errors earlier in the chain. This is a
   mechanism fixture, not an ecological learning result.
3. If the prototype passes, compare matched ecological starts with the value
   head enabled, disabled, and with shuffled return signals. A neutral version
   must preserve the prior simulation state exactly.
4. Follow any promising community outcome with fixed-genotype learning controls,
   recording food intake as well as reproduction and persistence. Retain failures.
