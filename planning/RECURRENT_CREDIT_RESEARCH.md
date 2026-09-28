# Reassessing recurrent credit after V25

Status: research notes and candidate diagnostics. No alternative native learner
has been implemented. The gentler V25 screen has completed and also failed its
original decision rule. The rate/noise sweep is closed. These alternatives are
research candidates, not evidence that another algorithm will improve the ecology.

The [passive experiment](REPRESENTATION_LEARNING_PLAN.md) is complete and
independently audited, and fails its predefined continuation screen. The separate
observer and derivative diagnostic leave the physical controller unchanged.
Research is paused at the user's request.

## What the measured failure does and does not say

The [native V25 fixture](RECURRENT_LEARNING.md) learns its supplied delayed-cue
task, but the first two ecological screens do not establish useful recurrent
adaptation. The [passive probes](../docs/results/v25-recurrent-credit.json) show nonzero
acquired effects, mostly unclipped energetic advantages, and larger immediate
effects from current hidden perturbations. A quieter perturbation increases the
score coefficient and worsens reproduction with the same rate. These observations
do not prove a bug in the conditional score, nor identify a single bottleneck.

Two practical constraints remain. The original learning cohorts' median founder
lifetimes are about 104–110 seconds, much shorter than the supplied diagnostic's
640 seconds of stronger training. The preset's quality-reversal period is
240 seconds with jitter .25; the resource sources also move and deplete. This
does not mean the world is stationary within a lifetime, but long diagnostic
episodes and quality reversals cannot stand in for evidence of timely learning.

## Primary literature checked

Murray's RFLO method approximates online recurrent gradients with local traces
and fixed random feedback of output error. It drops nonlocal recurrence terms;
the reported approximation loses performance on sufficiently long tasks. Its
main training setups use supplied output targets. Applying it here would
require an independently justified energetic or policy-feedback signal, not
invented desired steering. The model and Appendix 1 were reviewed in the
[author-hosted paper (2019)](https://ctn.zuckermaninstitute.columbia.edu/sites/default/files/content/Publications/2019/Murray,%20Local%20Online%20Learning%20in%20Recurrent%20Networks%20with%20Random%20Feedback.pdf).

Bellec and colleagues' e-prop combines synaptic eligibility with learning
signals, including reinforcement-learning examples in spiking networks. Its
neuron dynamics and learning protocol differ from our leaky rate neurons and
short independent lives. The article's indexed text and
[author repository](https://github.com/IGITUGraz/eligibility_propagation) were
accessible; the live publisher page was blocked by its access frontend. This
is a motivation for checking derivatives and feedback, not a reviewed native
port. See the [2020 primary article](https://www.nature.com/articles/s41467-020-17236-y).

Tsurumi and colleagues study online learning of recurrent state representations
and a value readout from TD prediction errors, including fixed random feedback
and nonnegative variants. Their cue/reward tasks address prediction and timing;
they do not demonstrate foraging, evolution, or survival in this simulation.
The indexed methods and the
[author's MATLAB implementation](https://github.com/kenjimoritagithub/oVRNN1/blob/main/rnrl1coslra.m)
were also inspected. The code credits the previous neural transition with the
next TD error, using either pre-update readout weights or fixed random feedback.
Its local derivative omits recurrent history; values are recorded before weight
updates. It uses centered logistic units, without our leaky integration,
connection masks, acquired bounds, or short lifetimes. This is a reference for
update order, not a native port. [Primary article (2025)](https://elifesciences.org/articles/104101).

## Candidate diagnostic before another native mechanism

Our inference is that learning a useful representation may be worth separating
from immediately perturbing the organism's entire controller. V22/V23 trained
value readouts on inherited recurrent features or direct sensations; their weak
forecast results do not test whether energetic prediction can train those
features themselves.

A next diagnostic could compare a fixed representation with bounded acquired
input/recurrent offsets trained through a value-prediction signal. Run both as
observers of exactly the same native sensory histories and energy returns.
They must not steer bodies or obtain future labels. Keep a shuffled-feedback
control and unchanged reference trajectories, and assess future returns recorded
after each prediction. Reset each learner at birth and report deaths and young
cohorts. Any terminal bookkeeping must not create useful posthumous learning
or inherited acquired state.

Before that experiment, derive the update for the actual leaky, masked neurons
and inherited response times. Verify fixed-weight derivatives against autograd
on a small supplied task, and distinguish an exact derivative from a local
approximation. Existing local plasticity, acquired motor normalization, internal
signals, and deterministic nonmotor outputs create additional dependencies;
an output-noise policy score cannot silently ignore them and be called an exact
gradient of the whole creature's return.

If prediction feedback is used instead, identify the bootstrap target and what
is held fixed during its update. Record the timing of the neural transition,
prediction, received energy, and learning step so no current prediction is
replaced after its outcome becomes known. Use the learned critic's actual
previous weights or an explicitly stated feedback approximation. Do not assume
positive-only rewards: net energetic returns can be negative here.

Only a demonstrated prediction improvement on matched experience would justify
testing that representation in a live controller. Such an improvement would
still need separate behavioral and ecological controls. Adding layers, sharing
weights across unrelated organisms, or copying acquired weights to offspring
would change other hypotheses and is outside this initial diagnostic.

## Completed derivative diagnostic

The six fixed-weight checks use three founder seeds (11/12/13), each with its
inherited recurrent weights and a separate zero-recurrence control. Five of six
hidden slots are active; absent sensory and recurrent edges, a self-edge,
heterogeneous response times, biases, and normalized readout derivatives are
included. Four sampled points span 1–40 updates (0.1–4 seconds).

Exact online sensitivities agree with autograd through the actual native
`advance` function to a maximum absolute error of `8.89e-16`. Conditional
one-transition derivatives agree to `2.78e-17`. The separate normalization check
includes zero activity. Local leak-only sensitivities are exact in the
zero-recurrence controls; their error with nonzero recurrence is substantial:

| Fixture seed | Relative gradient error at 4 s | Gradient cosine at 4 s |
|---|---:|---:|
| 11 | 28.5% | .963 |
| 12 | 41.7% | .944 |
| 13 | 18.7% | .984 |

These are implementation and approximation measurements on a small supplied
fixture, not evidence of prediction learning, foraging, or ecological benefit.
Online weight changes introduce further approximation. All sampled derivatives
respect the inherited masks.

The [checked record](../docs/results/representation-gradients.json) points to the raw
report and archived executed scripts/native source under
`runs/representation-gradient-diagnostic`. Source and script digests were
verified. The subsequent script formatting does not change these archived
executed files or the reported results.

```bash
uv run python scripts/probe_representation_gradients.py --output runs/new-gradient-check
```

The observer's causal update has six additional passing tests for fresh state,
old-trace credit, pre-update readout feedback, unchanged owner value learning
under shuffled representation feedback, masks/bounds, and disabled adaptation.
Five more measurement tests cover forecast timing, terminal intervals, growth
normalization, and exact passive replay through native births and deaths.
The full three-start prediction experiment is complete. Its
[outcome](REPRESENTATION_LEARNING_PLAN.md#completed-outcome) and
[checked record](../docs/results/learned-representations.json) retain all starts:
only one of three passes the primary comparison, so the continuation screen
fails. No adaptive representation has been added to the native controller.
