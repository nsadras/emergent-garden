# V21 experiment plan: persistent exploratory motor changes

Status: implemented as V21. All 336 tests, CPU/CUDA replay exercises, and 12
matched 600-second ecological comparisons pass their mechanical/accounting
checks. All six continuations and 24 community transplants are complete.
Ecological and learning effects remain mixed. See
[the result record](PERSISTENT_EXPLORATION.md) for measurements and limitations.

The present motor rule draws independent noise every controller update (10 Hz
in the research presets). Much of that variation can cancel before it changes a
creature's route. The delayed-credit fixture also found that a longer eligibility
trace alone gave little adaptation. Test temporally correlated exploration,
while preserving random inherited controllers, food laws, and the existing
energetic learning signal.

## Policy and credit

Use a first-order autoregressive Gaussian policy in motor-logit space. Let
`f_t` be the normalized hidden features including bias, `b_t` the inherited
motor logits, `P` the acquired motor rows, and `z_previous` the last sampled
motor logits. At the new update, evaluate both means with the **current** `P`:

```text
rho = exp(-elapsed / noise_tau)
mu_t = b_t + P f_t
mu_previous = b_previous + P f_previous
innovation_std = sigma sqrt(1 - rho²)
z_t = mu_t + rho (z_previous - mu_previous) + innovation_std epsilon_t
action_t = sigmoid(z_t)
```

Sigma is the existing inherited exploration magnitude. The first sample has
`rho = 0`, giving its full stationary variance without a warm-up transient.
Disabled exploration must clear the history contribution immediately. A neutral
`noise_tau = 0` must preserve the independent-noise trajectory exactly.

For the acquired linear motor weights, the conditional likelihood score is:

```text
(epsilon_t / innovation_std) outer (f_t - rho f_previous)
```

The previous feature term is essential. Adding correlated noise and retaining
the old independent-noise score would not describe this policy. Both mean
evaluations contribute to the derivative, as in the autoregressive policy
formulation of [Korenkevych et al. (2019)](https://www.ijcai.org/proceedings/2019/0382.pdf).
The existing modulation gate, eligibility decay, learning rate, row bounds,
and energetic-return baseline remain. They make the complete online learning
rule a bounded approximation; this change does not confer a convergence claim.

Store previous features, inherited logits, sampled logits, and readiness per
body module. These are acquired controller state, checkpointed but never
inherited. New children and newly grown modules start empty. The genome and
42-input/seven-output interface remain unchanged. The observer must display
the actual applied motor perturbation, including its history contribution.

## Comparisons

Keep the V19 learning habitat and receptor radius 1 to isolate this experiment.
Start with two-second correlation versus the independent-noise baseline, crossed
with motor learning enabled or disabled. Reuse matched founder genomes and
exploration random draws. Existing recurrent plasticity remains enabled. The
shuffled-return control remains available if a benefit merits follow-up.

Do not increase the noise magnitude or motor-learning cap at the same time.
Population, reproduction, intake, and movement need evaluation separately:
more persistent random movement is not evidence of useful learning. Compare
three fresh starts first, then continue all starts for the chosen treatment.

## Required checks

- Conditional score agrees with finite differences of the actual Gaussian
  log-likelihood, including changing current/previous features.
- With fixed weights, perturbations have the intended stationary variance and
  lag correlation; a changed inherited mean is not accidentally low-pass filtered.
- Neutral correlation reproduces legacy common physical state, and learning
  suppression retains identical exploration behavior.
- No-exploration and zero-sigma modes are finite and remove history effects.
- Checkpoints, births, growth, and passive observation preserve the new state
  correctly on CPU and CUDA.
- Use paired ecological controls; retain failures and partial-run boundaries.
