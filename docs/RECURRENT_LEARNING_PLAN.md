# V25 plan: energetic credit for recurrent connections

Status: prospective; native implementation has not started. The independent
[score and adaptation checks](RECURRENT_LEARNING_RESEARCH.md) are complete.
The V24 timing transplants are running. This plan records the next native
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
completed V24 transplant results should be reviewed before finalizing that
baseline. Preserve the user's working `configs/v16.toml`.

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
