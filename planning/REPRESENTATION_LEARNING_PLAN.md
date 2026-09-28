# Passive test of learned neural representations

Status: **complete and independently audited; continuation screen failed**.
The plan and observer were committed as `4be302c` before the three-start
experiment. Native derivative checks and eleven kernel/measurement tests pass.
The [completed outcome](#completed-outcome) records the forecast comparison;
the original prospective design below is preserved. Research is paused at the
user's request. No new native controller or ecological version is implied.
Preserve the user's scarce, valuable food and inexpensive exploration settings.

## Question and scope

Can energetic prediction feedback improve a creature's internal representation
within its lifetime? First compare predictors observing identical physical
experience. They cannot steer bodies, change evolution, share weights between
creatures, inherit acquired weights, or consume future information. A positive
forecast result would justify a later controlled native experiment, not a claim
of more intelligent behavior.

Each observer starts with the creature's inherited input weights, recurrent
weights, biases, masks, and neuron response times. Its hidden activity and
acquired weights start at zero. Observe only the core module's actual sensory
inputs. Native local plasticity, motor learning, and internal signals continue
in the physical creature; the shadow representation does not copy their acquired
weights. Its sensory history is consequently imposed, not a self-consistent
alternative controller trajectory.

## Derivatives before learning

Write the masked leaky transition as
`h_t = M * ((1-alpha)*h_previous + alpha*tanh(W*z_t))`, with
`z_t = [inputs_t, h_previous, 1]`. The mask on each weight includes its receiving
node and, for recurrent weights, its source node. Alpha uses the actual inherited
response times. Frozen weights give the exact hidden Jacobian
`J_t = diag(M*(1-alpha)) + diag(M*alpha*(1-tanh(u_t)^2))*W_recurrent`.
An exact online sensitivity obeys `P_t = J_t*P_previous + direct_t`.

The inexpensive candidate retains only the leak term when propagating a
synapse's sensitivity: `E_ij,t = (1-alpha_i)*E_ij,previous +
M_i*alpha_i*(1-tanh(u_i,t)^2)*z_j,t*weight_mask_ij`.
It discards propagation through recurrent connections, including self-edges.
This is a local approximation, not an exact gradient of a recurrent network.
Verify the exact sensitivity against autograd through the actual native
`advance` function, verify the one-step derivative with detached history, and
quantify the local approximation's error. Require masks, inactive nodes, varied
response times, recurrent self-edges, input weights, and biases in the checks.

The prediction features are `[hidden, 1]` divided by their norm, matching the
existing hidden-feature normalization. Differentiate that normalization too.
For fixed readout `w`, the hidden learning signal is
`w_hidden/norm - (w*features).sum()*hidden/norm^2`.
Check this independently with autograd, including zero hidden activity.

## Candidate causal update

At a controller boundary, receive only energy accumulated since the previous
boundary. Advance the shadow circuit with its pre-update weights and record
the current prediction before learning. Re-evaluate the previous features with
the same current readout weights. Use
`delta = received_return + exp(-elapsed/horizon)*current_prediction -
previous_prediction`. The bootstrap target, prior activity, and sensory history
are held fixed: this is TD semi-gradient learning, with the additional local
recurrent approximation described above.

Train the value readout with `rate*clip(delta)*previous_features` (TD(0)). Train
the representation with the previous local sensitivities and the derivative
of the previous prediction using the pre-update readout. Apply acquired masks
and row bounds. Current inputs and the just-created sensitivities must not be
credited for returns already received. No terminal update can influence a dead
creature's descendants. The first observation has no previous transition to
train. Freeze these choices before the matched-experience experiment.

Initial diagnostic settings: two-second prediction horizon, maximum readout
rate .05 scaled by the existing motor-learning allocation, maximum representation
rate .01 scaled by the existing recurrent-learning allocation, readout norm
bound 4, acquired representation row norm bound .3, TD error clipped to `[-1,1]`.
The input/recurrent/bias offsets share each receiving row's bound. No forgetting
or additional exploratory noise is added to these passive predictors. These
are first settings, not calibrated rates transferred from the published tasks.

Compare fixed inherited representation, adaptive representation, and adaptive
representation with another creature's TD error. All three value readouts still
learn from their owner's returns. A fourth linear predictor uses the actual
sensations directly, to test whether representation learning improves on that
simpler source of information. Zero and a causal running-return predictor supply
additional forecast references. Use a separate RNG for feedback shuffling, and
do not modify any world RNG.

## Matched experience and evaluation

After the derivative and causal checks pass, replay the three mechanism-off V25
founder starts, seeds 1/2/3. Train observers from birth, including subsequent
newborns, for 600 simulated seconds. Record predictions near ages 5, 30, and 60
seconds, before their outcomes, and retain their next 20 seconds of discounted
net returns, including deaths. Continue physical replay only as needed to finish
these windows. Report how many creatures reach each age and obtain a forecast;
surviving to an age is selection, not a neutral sample of all newborns.

Engineering details fixed before the full experiment: forecast ages use recorded
birth ticks rather than accumulated float32 age. Returns use the exact received
motor-feedback accumulator and each interval's current body area, including
growth. A passive wrapper captures the partial return immediately before native
death removal; it does not train the dead observer. Its timestamp is the end of
that physics step (native death events label the step's beginning). Shuffle only
ready learners using a nonzero cyclic shift; report singleton updates that cannot
be shuffled. Keep predictions frozen in the record, even while the observers
continue learning. Replay the complete world without observation alongside each
run and also compare the 600-second state/events with the original baseline.

The primary comparison is mean squared forecast error for the 30-second cohort
within each community. Age 5 and age 60 are descriptive secondary cohorts.
Record the finite observed return as the target, without adding a learned tail.
Twenty seconds is ten two-second horizons; finite truncation remains explicit.
Use the same temporal discount and body-area normalization as the received
learning returns, handle off-phase newborn intervals and growth, and preserve
partial terminal intervals. These details require mechanical verification.

Archive the executed scripts, native source, observations/predictions sufficient
to check timing and targets, checkpoints, and acquired states. Require complete
physical and RNG equality against plain replay in all three starts. Predictions
within a community are correlated; the three starts are the comparison units.
Do not count these replays as new independent ecological trials.

Consider native integration only if the adaptive representation lowers primary
error against both fixed and shuffled representations in at least two of three
starts, and improves on zero in those starts. Compare direct sensations and the
running-return reference explicitly even if that screen passes. Inspect actual
update magnitudes, saturation, timing, and younger cohorts before deciding a next
step. A failed screen calls for reassessment, not automatic new rate points.

## Completed outcome

All three physical replays with observers exactly match unobserved full-world/RNG
replay, and their 600-second states/events match the original V25 baselines.
The independent audit reconstructed forecasts and checked interval normalization,
finite states, bounds, inherited masks, and source provenance. These are replays
of existing communities, not new independent ecological trials.

Primary age-30 forecast MSE (lower is better):

| Predictor | Seed 1 (470 forecasts) | Seed 2 (360 forecasts) | Seed 3 (428 forecasts) |
| --- | ---: | ---: | ---: |
| Fixed representation | 0.352907035 | 0.564305597 | 0.510838311 |
| Adaptive representation | 0.352894806 | 0.564296871 | 0.510834353 |
| Shuffled representation feedback | 0.352885042 | 0.564305093 | 0.510829401 |
| Direct sensations | 0.356736006 | 0.561719735 | 0.504582126 |
| Zero | 0.382697337 | 0.595543518 | 0.507206906 |
| Running-return reference | 0.439796055 | 0.549311761 | 0.531085007 |

The adaptive predictor beats fixed, shuffled, and zero together only in seed 2,
failing the required two-of-three screen. Its improvement over fixed features
is below .004% in every start; direct sensations and the running-return reference
also beat it in seed 2. No native integration is justified by this screen.

The primary forecasts include 26/37/31 deaths within their outcome windows.
Age-five and age-sixty cohorts, update magnitudes, clipping, and all comparison
values remain in the [checked record](../docs/results/learned-representations.json).
Raw data are in `runs/learned-representation-pilot`. Forecasts within each
community are correlated, reaching an age selects survivors, and targets are
truncated at 20 seconds. Passive prediction does not establish behavioral or
ecological benefit. No further experiment was launched before the pause.
