# V19: varied founder circuits and controlled learning experiments

The aim is to make more neural strategies available to evolution while testing
whether they help creatures find food and reproduce. The habitat retains V18's
rare, valuable meals, inexpensive propulsion, finite carrying, and faster
processing. The user's working `configs/v16.toml` remains untouched.

## What changes

`configs/v19.toml` draws each founder's active neuron count uniformly from 8–24
and enables each possible recurrent connection with probability 0.5. Inputs and
outputs initially connect to all active neurons. These are different bounded
recurrent graphs, **not extra stacked hidden layers**. The shared body-module
template still has 42 inputs, seven outputs, 32 available neurons, and 5,272
inherited values. Existing structural mutations can subsequently explore 4–32
active neurons.

The new settings are `initial_neuron_spread = 8` around `initial_neurons = 16`,
and `initial_recurrent_density = 0.5`. Neutral defaults (0 and 1) preserve the old
founders. An independent, checkpointed random stream chooses their structures
without moving food, changing bodies, or consuming the other random streams.

Founder recurrent weights use standard deviation `1/sqrt(max(1, N*density))`;
output weights use `1/sqrt(N)`. This controls expected input variance across
widths and densities. Input weights retain their previous scaling and biases
start at zero. No preferred movement is encoded. **Birth does not redraw or
rescale the parent's circuit.** Children inherit its genes and masks with the
configured mutations; activations, traces, and acquired synaptic changes start
empty. Existing construction and maintenance costs charge expressed circuitry.

The presets separate three questions:

| Preset | Founder graphs | Weight mutation | Node / edge mutation | Motor exploration / learning |
|---|---|---|---|---|
| `v19-baseline.toml` | 16 neurons, dense | probability .02, sigma .05 | .05 / .1 | No exploration |
| `v19.toml` | 8–24 neurons, half recurrent edges | .02, .05 | .05 / .1 | No exploration |
| `v19-mutation.toml` | Same varied founders | .04, .1 | .2 / .4 | No exploration |
| `v19-learning.toml` | Same varied founders | .02, .05 | .05 / .1 | Sigma .05–.15; learning rate at most .01 |

Mutation probabilities retain their existing meanings: weight changes are
per inherited neural value; node/edge probabilities govern the structural
mutation attempts for each child. Body-trait and body-module mutation settings
are unchanged.

The learning preset reduces the acquired motor-row norm limit from .5 to .15
and the maximum motor-learning rate from .2 to .01. Normalized features bound
the learned correction to each motor logit by .15. The existing energetic
reinforcement rule keeps a two-second eligibility trace, a ten-second reward
baseline, and a 120-second decay half-life. These are tentative settings, not
an established optimum. Recurrent plasticity remains active in all presets.

The paired learning control uses the **same preset and exploration noise
settings**, with `--ablation no_motor_learning`. It disables acquired motor
updates, retains recurrent plasticity, and still pays for its inherited
learning capacity. This distinguishes the effect of motor updates from merely
adding noise. `no_plasticity` disables both learning mechanisms and is a
different intervention.

## Completed screening

Each treatment has three fresh 600-second runs, seeds 1/2/3. Within each seed,
all four varied-founder treatments have exactly identical inherited genomes.
The baseline differs in initial graph and its weight scaling. Consequently the
baseline comparison measures their combined effect, not graph shape alone.

| Treatment | Living at 600 s (seeds 1 / 2 / 3) | Cumulative births |
|---|---|---|
| Baseline | 82 / 108 / 79 | 411 / 302 / 113 |
| Varied founders | 151 / 4 / 93 | 455 / 26 / 195 |
| Stronger mutation | 143 / 16 / 158 | 262 / 116 / 297 |
| Gentle motor learning | 144 / 78 / 100 | 288 / 265 / 156 |
| Same noise, motor learning disabled | 194 / 88 / 112 | 367 / 264 / 310 |

![Paired pilot population histories](v19-circuits.png)

Varied founders improved two starts and weakened another. Stronger mutation
also had mixed effects. Motor learning did not provide a consistent advantage:
its birth counts were lower in two starts and essentially tied in the third.
These small screens justify continued comparisons, not a claim of improved
intelligence or a general winner.

All three varied-founder starts were continued to 1,800 seconds. Populations
were **251 / 34 / 327**, births **2,413 / 237 / 2,198**, and maximum living
generations **40 / 25 / 41**. Seed 2 recovered from its early four-creature
bottleneck. These are continuations, not three additional independent starts.
The mutation and paired learning continuations, and fresh-environment learning
assays, are running separately; their incomplete results are not included here.

The [audit](results/v19-circuits.json) verifies the checkpoint against recorded
measurements, birth/death events, carrying constraints, and energy/resource
ledgers. With neutral settings, **all recorded physical measurements and the
complete final physical state match V18 exactly** for all three 600-second runs.
Only the versioned configuration and unused founder-structure RNG differ.

## A diagnostic for circling

`scripts/probe_food_response.py` exposes sampled inherited circuits to uniform,
left-rich, and right-rich food inputs at the same mean intensity. It starts
neural state empty, disables plasticity/noise, holds energy at half capacity,
and averages motor outputs over seconds 20–25. The directional readout is half
the right-rich minus left-rich motor imbalance. This separates a response to
food direction from steady turning in a uniform field.

For fresh-food mean .4 and a ±.02 side contrast:

| Selected source | Median absolute uniform turn | Median absolute directional response | Positive directional sign |
|---|---|---|---|
| V18 fast carrying, seed 1, 1,800 s | .0787 | .0010 | 7 / 57 |
| V18 fast carrying, seed 2, 1,800 s | .1308 | .0091 | 64 / 64 |
| V19 varied founders, seed 1, 600 s | .0571 | .0014 | 61 / 64 |
| V19 stronger mutation, seed 1, 600 s | .1470 | .0018 | 45 / 64 |

The second V18 population has a larger, consistently aligned inherited
response, whereas steady turning dominates the small response in several other
samples. This is consistent with uneven evolution of directional sensitivity;
it does not establish its ecological usefulness. The assay omits body geometry,
other observations, and acquired state. Sampled descendants share ancestry and
are not independent evolutionary replicates. Raw responses at three intensities
for fresh food and detritus, founder comparisons, genotype hashes, and source
provenance are retained in [the diagnostic record](results/v19-food-response.json).

## Verification and use

All 310 tests pass, including variable founder graphs, weight scaling,
inheritance, fresh child neural state, unchanged legacy trajectories, observer
purity, and the directional diagnostic. Ruff checks pass. Separate
[CPU](results/v19-cpu-circuits.json) and
[RTX 5080](results/v19-cuda-circuits.json) checks passed exact checkpoint replay
through births, growth, changing rules, and motor learning. They verify
mechanics, not useful learning.

```bash
uv run garden run --config configs/v19.toml --seed 1 --view --device cpu --seconds 0
uv run garden run --resume runs/v19-varied-long-1/latest.pt --view --device cpu --seconds 0

uv run garden calibrate --config configs/v19-learning.toml --seeds 1 2 3 \
  --seconds 600 --device cpu --output runs/my-learning-pilot
uv run garden calibrate --config configs/v19-learning.toml --seeds 1 2 3 \
  --ablation no_motor_learning --seconds 600 --device cpu --output runs/my-noise-control

uv run python scripts/audit_circuits.py --long-treatments varied
uv run python scripts/plot_circuits.py
uv run python scripts/probe_food_response.py runs/v19-varied-long-1 \
  --genomes 64 --output runs/my-food-response
```

## Next questions

The priority is to distinguish exploration, inherited searching, and useful
within-lifetime adaptation. Fresh-environment transplants compare identical
descendant genomes with and without motor updates, with genetic mutation
disabled. Longer runs test whether the early population differences persist.

One possible limitation is delayed credit: food search can take longer than the
current two-second eligibility trace. Another is a weak spatial signal compared
with intensity and inherited turning bias. Neither is established as the cause.
Measure these before adding depth or simply enlarging networks.

Reward-modulated recurrent learning can solve constructed tasks with delayed
rewards in [Miconi (2017)](https://elifesciences.org/articles/20899). Evolution of
plastic controllers across a family of cognitive tasks can also produce
transfer to withheld tasks in [Miconi (2023)](https://proceedings.mlr.press/v202/miconi23a.html).
Those results motivate explicit credit-assignment and held-out tests; they do
not demonstrate that the same capabilities will emerge in this ecology.
