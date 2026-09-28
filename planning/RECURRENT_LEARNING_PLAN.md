# V25 plan: energetic credit for recurrent connections

Status: native implementation, 447 tests, CPU/CUDA replay, inspector checks, and
the native delayed-cue diagnostic are complete for V25 / package 0.26.0.
All twelve ecological pilots are complete and audited; the predeclared
continuation screen failed. See the [results](RECURRENT_LEARNING.md).
The independent
[score and adaptation checks](RECURRENT_LEARNING_RESEARCH.md) are complete.
The V24 timing transplants are complete. This plan records the next native
experiment before seeing its results; it is not a release announcement.

## Hypothesis and scope

Motor adaptation can change how a fixed recurrent representation drives the
body. The new mechanism will also let actual energetic experience change the
recurrent representation. Small independent perturbations of hidden-neuron
drive supply a local eligibility score. Later food intake, maintenance costs,
and damage supply the learning signal. No desired movement, food direction,
future resource state, or external fitness function enters the controller.

Keep the current sensory interface, inherited graph, body and resource laws,
motor learner, and older inherited plasticity rule. The 32-slot genome remains
5,304 values. Existing neuron/edge/timing mutation still controls inherited
structure. Add acquired recurrent offsets separately from the older plastic
offsets so each mechanism can be measured and disabled independently.

## Per-module state and update order

Each module gains a recurrent offset matrix, a score-trace matrix, an energetic
baseline, and the actual applied hidden perturbation. All start at zero at birth
or when a module develops, and are included in exact checkpoints. None enters
the child's genome. Connection masks exclude dormant or absent synapses.

At a controller boundary:

1. Receive the body's integrated net energy since its previous update, divided
   by the existing area and feedback scale. Reproduction and growth transfers
   remain excluded, matching the current motor feedback convention.
2. Compare the return rate with the baseline from previous experience. Convert
   back to an integrated advantage and clip to `[-1, 1]`, as in the simple motor
   learner. Update the baseline using the actual elapsed time.
3. Decay the acquired recurrent offsets and add `rate * advantage * old_trace`.
   Apply the connection mask and project each row onto its norm bound. This
   credits earlier perturbations before adding the current perturbation.
4. Advance the brain with inherited weights, the previous local-plasticity
   offsets, the newly credited recurrent offsets, and independent Gaussian
   noise before `tanh`. Preserve the inherited per-neuron integration factors.
5. Decay the score trace and add
   `(standard_normal_noise / sigma) outer previous_hidden`, with the recurrent
   mask. Do not insert an extra leak or `tanh` derivative. No score is added
   when exploration is disabled.
6. Continue the existing motor and local-plasticity updates with their current
   meanings. Each receives its own appropriate feedback, so shuffling recurrent
   feedback cannot silently shuffle motor feedback as well.

The continuous, clipped, bounded update is an approximation. The exact
episodic gradient argument in the diagnostic does not establish an unbiased
native update when weights and activity change continually.

## Initial settings

| Parameter | Neutral/default | Experimental preset |
|---|---:|---:|
| `recurrent_noise_sigma` | 0 | .15 |
| `recurrent_learning_rate` | 0 | .001 maximum |
| `recurrent_learning_limit` | .15 | .15 per-row offset norm |
| `recurrent_trace_tau` | 2 s | 2 s |
| `recurrent_baseline_tau` | 10 s | 10 s |
| `recurrent_half_life` | 120 s | 120 s |

Configuration validation accepts zero noise or an enabled amplitude in
`[1e-4, 10]`, a learning rate in `[0, 1]`, and positive finite times/bounds.
A positive learning rate requires positive noise. These limits guard the new
float32 score calculation; the experimental values remain as specified above.

Scale the new rate by the existing inherited recurrent-learning allocation
(`sigmoid` of trait 9). This deliberately couples learning capacity to that
existing trait; it does not introduce a new gene or inherit acquired offsets.
The .001 rate is a conservative initial choice, not transferred calibration
from the fixture's .1 terminal update. Native updates occur ten times per
second and have different feedback magnitudes and temporal structure.

Use separate checkpointed random streams for hidden perturbations and the
recurrent-feedback shuffle. Preserve every existing random stream. The first
screen adds no separate energy charge for the extra matrices; all controls
retain the existing recurrent and motor capacity charges. This simplification
must remain explicit when interpreting any benefit. Neuron and synapse costs
continue to constrain inherited circuit size.

Start from the sparse, mobile-patch `v24-inherited.toml` ecology: heterogeneous
timing without direct timing mutation and a disabled value predictor. The
[completed V24 transplants](NEURAL_TIMING_ASSAYS.md) support retaining this
baseline: original timing beats reassignment for births in all six comparisons,
and motor updates beat disabled updates in all six. Shuffled-feedback effects
remain mixed. This baseline decision was finalized before any V25 native runs.
Preserve the user's working `configs/v16.toml`.

## Controls and required checks

Add `no_recurrent_learning` and `shuffled_recurrent_reward` interventions.
Both retain the same hidden noise, motor learning, recurrent local plasticity,
and capacity costs as the native treatment. The shuffle assigns another body's
return rate using a nonzero cyclic shift; singleton batches remain unchanged.
`no_plasticity` also disables the new acquired updates; `no_exploration`
suppresses hidden perturbations as well as motor exploration. Physical energy
and sensory feedback must remain untouched by feedback interventions.

Default-disabled V25 must preserve V24 trajectories exactly. Test the actual
native transition against the conditional score and the supplied delayed task;
do not rely only on a parallel mathematical implementation. Verify causal update
ordering, masked connections, row bounds, birth and growth resets, erasing
learned state, archived source compatibility, and exact device-local replay on
CPU and the RTX 5080. Preserve older-version tests and neutral behavior.

The inspector must show the effective weights used at that update. The new
offsets are applied before the recurrent step, unlike the older local update
applied afterward. Capture that distinction, show hidden perturbation as its
own contribution, and keep all tabs fitting existing window sizes. Rendering
and observation must leave the complete physical state and random streams
unchanged.

Report acquired offset magnitude, row saturation, hidden-noise magnitude,
and cumulative update magnitude, alongside births, energy/resource accounting,
diet allocation, food uptake, and lineage composition. Apply active-neuron and
module masks in every summary.

## First ecological screen

Before the ecological screen, repeat the supplied delayed-cue task through the
actual `advance` and `controller_step` functions. The prospective native check
uses seeds 11/12/13, 64 separate two-neuron circuits per condition, and 256
episodes. Compare no updates, own-score learning, and shuffled scores with
identical cues and perturbations. Each episode has 24 noisy updates and one
feedback update at 10 Hz. Only the recurrent offsets and learned baseline
persist between episodes. A separate fixed bank of 32 noisy trajectories per
circuit measures error without training.

This implementation check retains the fixture's effective rate .1 and row
bound .5, uses a 48-second baseline, and negligible forgetting. Those values
deliberately differ from the conservative ecological preset. There is no body,
energetic survival, evolution, or hand-tuned movement policy in this diagnostic.
The complete training sequence spans 640 seconds; report intermediate results
and failures as well as its endpoint. Archive the script, native source, input
genomes, final acquired states, and measurements. A positive result verifies
that the native machinery can learn this supplied task; it does not establish
useful adaptation during ordinary creature lifetimes.

Run three matching founder seeds, 1/2/3, for 600 simulated seconds per treatment:

1. Fully disabled new mechanism, for exact V24 compatibility.
2. Hidden noise with recurrent updates disabled.
3. Hidden noise with own-return recurrent learning.
4. Hidden noise with shuffled recurrent feedback.

Archive all twelve runs and retain extinctions. Full founder genomes must match
within each seed; no hand-tuned movement weights or selected positive starts.
Verify accounting, histories, final checkpoints, actual noisy neural samples,
and every neutral physical-state comparison. Inspect a real viewer capture.

Only consider longer evolutionary runs if own-return learning improves births
and fresh-food absorption in at least two of three starts against **each** of
the two noise controls. That is a screening rule, not statistical proof. If it
passes, continue all three selected treatment starts and use fresh, mutation-
disabled transplants for a more direct learning comparison. If it fails, retain
the implementation as optional and investigate the measured failure before
adding more layers or increasing learning rates.

The broader goal remains richer adaptive behavior and ecological diversity.
Neither changed neural activity nor larger populations alone demonstrates
intelligence, and recurrent learning does not by itself repair the loss of
scavenger specialists in the current ecology.

## Diagnostic follow-up after the completed screen

The twelve pilots failed the stated continuation rule. No longer ecological
trials are selected. Before changing noise, update scale, or architecture,
instrument a separate 30-second replay fork of **each** own-return community
at 600 seconds. This is measurement of all three completed treatments, not
another evolutionary replicate or a continuation selected for performance.

Observe actual controller boundaries, including newborn updates. Measure the
inherited, older-local, and reward-acquired recurrent drive; total sensory
drive and its spatial contrast; applied hidden noise; reward, raw advantage,
clipping, score traces, reinforcement proposals, decay, and actual bounded
offset changes. Include only expressed modules, neurons, and connections.
Report pooled update-weighted summaries and separate ages below 30 seconds,
30–120 seconds, and at least 120 seconds. These are correlated samples, not
independent organisms or tests of fitness.

At each observed boundary, recompute the actual transition and compare two
instantaneous alternatives: omit acquired reward offsets, or omit current
hidden noise. Keep inputs, previous hidden state, old local offsets, inherited
genome, newly credited motor offsets, and motor perturbation fixed. Measure
hidden and motor-output differences only. Do not advance either alternative
through the environment or claim that a larger difference demonstrates benefit.

Archive the diagnostic script, native source, source-checkpoint hashes, final
forks, and summaries. Replay each fork again without instrumentation and require
exact equality of its complete state, including random streams and events.
Reconstruct the observed transition and motors at every sampled update. Preserve
interrupted or failed diagnostics; do not silently replace a failed sample.

## Quieter perturbations: prospective follow-up

The passive diagnostic completed all three forks with exact full-state replay.
Single-update motor effects from acquired recurrent offsets are 15–22% of the
current hidden-noise effects. Fewer than .7% of sampled module updates clip their
advantage, losing 1.1–4.4% of absolute advantage mass. These observations motivate
a quieter perturbation screen; they do not establish the cause of the pilot's
mixed results or the full historical influence of acquired weights.

Keep the V25 ecology and maximum learning rate .001 unchanged, and reduce only
`recurrent_noise_sigma` from .15 to .05 in `configs/v25-quiet.toml`. The likelihood
score coefficient scales inversely with sigma, so the multiplier on
`epsilon outer previous_hidden` rises threefold. Actual traces also depend on
the changed activity. Retain the existing norm bound and
measure saturation and actual offset changes. This is not a claim of equal
learning variance or a calibrated optimal amplitude.

Run nine new 600-second trials: seeds 1/2/3 crossed with noise-only, own-return,
and shuffled-return conditions. Reuse the three completed mechanism-off V25
runs as the reference; label this reuse explicitly. Match complete founder
genomes, check every source archive and account, and retain all outcomes.
No selected starts and no changes to inherited weights are allowed.

Only consider longer runs if quiet own-return learning improves births AND
fresh-food absorption in at least two starts against **each** of noise-only,
shuffled feedback, and the reused mechanism-off baseline. This is a prospective
screening rule, not proof of adaptive intelligence. Continue all three learning
starts if that screen passes; otherwise retain the quieter variant as optional
and reassess the measured effects before another change.

The quieter screen subsequently completed and failed; see the
[full results](QUIET_RECURRENT_LEARNING.md). Its own-return treatment improves
both primary outcomes in 0/3 comparisons with noise-only, 1/3 with shuffled,
and 0/3 with the reused baseline. No longer runs were launched.

## Gentler learning at the quieter amplitude: prospective comparison

Keep sigma .05 and reduce the maximum rate from .001 to `.001 / 3`, changing
only that rate. Run six new 600-second trials: seeds 1/2/3 for own-return and
shuffled recurrent feedback. Reuse the completed quiet noise-only and original
mechanism-off references, labeling both. The disabled recurrent learner never
uses the maximum rate for physics or costs; verify exact common-state replay
with the lower rate, excluding only configuration and potential-rate telemetry,
before accepting that reuse.

The motivation is to restore the original .15-noise preset's rate/sigma
coefficient for otherwise identical activity and returns. This does not ensure
equal realized updates or variance. Retain the existing masks, row cap, decay,
trace, baseline, and inherited allocation. Report both acquired magnitudes and
ecological outcomes, including failures, with matching founders and archives.

Only consider longer runs when births AND fresh uptake improve in at least two
starts against each of quiet noise-only, matched gentle shuffled feedback, and
the original mechanism-off baseline. Do not launch selected-winner continuations.
If this candidate fails too, end this rate/noise screen and investigate the
credit-assignment mechanism before another parameter grid. This ends neither
the broader neural work nor the user's autonomous research request.

All six gentler trials subsequently completed and failed the screen: both
primary outcomes improve in 1/3 starts against quiet noise-only, 1/3 against
gentle shuffled feedback, and 2/3 against the reused mechanism-off baseline.
Three complete control replays verify the unused-rate neutrality exactly.
This rate/noise screen is now closed; no longer runs or additional rate points
are selected. See the [results](QUIET_RECURRENT_LEARNING.md#gentler-rate-follow-up).
