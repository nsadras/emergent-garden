# Continuing evolution experiments

The user authorized continuous implementation, simulations, evaluation, and local
git commits until they request a stop. Dependencies remain managed with uv.
V0–V5 and their evidence are preserved in `EVOLUTION.md`.

## Research sequence

1. **V6 — learnable environmental variation.** Add stable patch identity cues,
   irregular reversals of their nutritional value, and sensed food/damage events.
   Preserve the fixed recurrent brain as the baseline. Check that hidden quality
   never directly enters observations, that energy is accounted for, and that
   ecological populations remain viable. Add direct, controlled adaptation probes.
2. **V7 — inherited plasticity rules.** Separate inherited base weights from
   per-module learned changes; start children with empty plastic state. Bound
   plastic changes, evolve rate/decay/modulation, and compare learning-enabled
   and frozen controls in otherwise matched environments.
3. **V8 — variable neural structure.** Start with bounded neuron/connection
   variation and small structural mutations. Preserve useful inherited behavior
   during enlargement and charge explicit neural construction/maintenance costs.
4. **Combined experiment.** Compare fixed/plastic synapses crossed with
   fixed/evolving architecture across independent evolutionary seeds. Measure
   within-individual changes and held-out reversals alongside population outcomes.
5. **Further ecology and development.** Use the results to prioritize spatial
   niches, refuges, module specialization, internal signaling, gradual growth,
   and organism-mediated environmental changes.

The order can change with evidence. Larger brains, more features, or changed
trajectories alone do not establish learning. Keep failed trials and treatment
controls, inspect recordings, preserve checkpoints, and commit completed versions.

## Relevant research

- [Soltoggio et al. (2008)](https://andrea.soltoggio.net/data/AlifeSoltoggio2008/ALIFExi_pp569-576.pdf):
  evolved neuromodulated plasticity in changing reward tasks.
- [Stanley and Miikkulainen (2002)](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf):
  incremental topology evolution; the full NEAT system also uses speciation and
  population selection that differ from this ecology.
- [Duan et al. (2016)](https://arxiv.org/abs/1611.02779): recurrent activity can
  encode adaptive learning even when deployed synaptic weights remain fixed.

## V6 design record

Two equally strong chemical identity channels mark fixed patches, independent
of their current food quality. The favorable identity starts randomly and flips
after intervals sampled uniformly around 240 seconds with ±25% jitter. Quality
multiplies fresh-food assimilation by 1.0 or 0.25; it does not change particle
energy, scent, or create resources. Unclustered food retains its usual quality.
The quality rule applies at consumption time, representing environmental
digestibility. Detritus retains its existing properties.

Two new body-wide inputs report particle food absorbed and energy lost to bites
since the previous controller update, normalized by body area and a scale of 10
energy units. These are physical observations, not an external fitness score.
Together with two new four-point identity fields, V6 has 32 inputs, 16 recurrent
units per module, and the existing four outputs. Old input meanings remain
stable through explicit named interfaces when transferring genomes.

V6 has 852 neural parameters plus nine trait genes (861 total). Named input
interfaces preserve existing sensory connections during transfers. `no_identity`
and `no_feedback` community controls selectively remove the new observations.

`garden association RUN --output FILE` presents both identity cues with matched
total food feedback, counterbalances which identity is valuable and which was
experienced last, then measures motor responses to mirrored cue positions. It
repeats after reversing the value association. `no_feedback` and `reset_h` are
controls. This measures isolated controller responses; useful navigation still
requires embodied tests.

Mechanical verification: 76 tests pass, including scent invariance under hidden
quality changes, conservative digestion, feedback timing, named-input transfer,
irregular-reversal replay, renderer purity, and matched probe controls. A CUDA
checkpoint check passed through 17 reversals on the RTX 5080.

Initial random-founder pilots (51/52/53) completed 1,800 seconds with populations
8/1/4, births 21/6/28, and maximum living generations 4/0/4. The single survivor
in seed 52 was a founder, so persistence there does not represent sustained
evolution. The three V5-inherited populations (new environments 61/62/63) reached
populations 45/32/51, births 1,009/754/908, and living generations 30/20/19 after
3,600 seconds. These share V5 source ancestry and are separate from random starts.

All-nutritious controls (`low_quality=1`) improved the random starts to populations
11/15/15 and births 49/65/84. This changes mean nutrition as well as the contrast.
A second control with both qualities set to 0.625 ended at populations 23/1/6,
births 57/6/26, and living generations 7/0/3 after 1,800 seconds. It matches the
mean digestibility of equally encountered patch identities but does not force
equal realized intake or occupancy. Equalizing patch quality did not rescue
sustained reproduction in the weak seed 52 at the original dish size.

Reducing dish diameter from 768 to **512 units**, with other settings unchanged,
gave populations 16/9/32, births 82/32/116, and generations 10/6/11 at 1,800 seconds.
Final module counts (1/2/3) were 0/13/3, 0/0/9, and 32/0/0. More frequent encounters
are a plausible explanation, not a measured mechanism. The compact layout is
retained as the V6 preset because all three starts continued reproducing. This
does not establish long-term diversity or learning.

Two 480-second community assays of environment-61 descendants ended at population
56/48 with normal sensing, 58/40 without food/damage feedback, and 60/56 without
identity cues. Corresponding births were 183/152, 186/142, and 183/151. Founder
transplants ended at 46/33 with 110/54 births. Feedback effects were mixed, and
the identity channels did not confer an observed final-population advantage.

The isolated association probe produced near-zero, slightly negative mean
alignment in descendants from random seed 51 (-1.83e-6) and inherited environment
61 (-3.15e-9). Removing feedback or clearing recurrent state before readout made
the paired effects exactly zero. No useful association or reversal learning is
established by these tests. Raw per-genome effects are retained in
`docs/results/v6-association-51.json` and `v6-association-61.json`.

A 30.03-second V6 video decoded to 901 frames; the renderer was inspected with
identity fields displayed. CUDA verification is under `runs/v6-cuda`.

An audit found that previous batches hashed source files when each run started,
which could reflect edits made while a process retained earlier imported code.
Those historical hashes are not exact executable snapshots. New processes now
capture package sources and the uv project/lock files once at import, retain
that hash and git provenance for the batch, and write `source.zip` in every run.
Restart a process after code edits to use and capture the new implementation.

Decision: retain the compact reversal ecology as a fixed-brain baseline and
proceed to bounded synaptic plasticity.

## V7 design record

Each module now maintains a matrix of acquired recurrent synaptic offsets and a
matrix of recent activity correlations. Its effective recurrent connections are
the inherited weights plus these offsets. Offsets and correlation traces start
at zero at birth, remain separate between modules, and are saved for exact
checkpoint resume. They are never written into the inherited genome.

A fifth controller output is a signed modulation gate in [-1, 1]. It multiplies
a two-second low-pass trace of post/pre recurrent activity. Two additional genes
set learning rate (a sigmoid fraction of a configured ceiling) and offset decay
half-life (30–600 seconds). Initial pilots used a rate ceiling of 0.2 per second
and offsets clipped to [-1, 1]; the selected preset uses 0.02 and [-0.1, 0.1].
Time constants use simulated seconds. Every active module pays 0.02 energy per second for the
plasticity machinery, including in the `no_plasticity` control.

There is no external reward optimizer or prescribed action policy. Food and
damage feedback can influence the evolved modulation output through the brain.
This rule makes adaptive plasticity possible; it does not supply the appropriate
learning rule by itself or establish that evolution will discover one.

The interface has 32 inputs, 16 recurrent units, and five outputs: left/right
propulsion, attack, secretion, and modulation. There are 869 neural parameters
and 11 trait genes (880 inherited values). The 512 acquired recurrent values
(offsets plus traces) per module are separate, nonheritable state.

Transferring V6 genomes sets modulation weights and bias to zero (neutral gate),
preserves the four existing outputs, and initializes the learning-rate gene to
-2. Random V7 founders have the usual random connections. `no_plasticity`
suppresses both acquired offsets and traces while retaining the inherited
controller, gate output, and maintenance cost. It is available for fresh runs,
calibration batches, community assays, and isolated association probes.

Mechanical verification: 91 tests pass, including offset limits, signed gating,
real-time decay at two controller rates, per-module state, reset at birth,
neutral genotype transfer, frozen controls, energy accounting, and exact replay.
A deliberately constructed diagnostic circuit demonstrates that the association
probe can detect preference acquisition and reversal through plastic state; it
is used only in tests and never seeds a population. CUDA checkpoint replay passed
through 37 accelerated quality reversals on the RTX 5080, with a 0.00103 energy
ledger residual in the small verification world.

Completed pilot: three random starts (71/72/73) and three transplants from the V6
environment-61 population (81/82/83), each paired with a `no_plasticity` run for
3,600 simulated seconds, plus three random starts with tighter plasticity bounds.
The initial genomes are exactly matched across each seed's treatments.

| Treatment | Final populations | Births | Maximum living generations |
| --- | --- | --- | --- |
| Random, original plasticity | 8 / 12 / 2 | 149 / 49 / 132 | 8 / 8 / 12 |
| Random, frozen | 16 / 0 / 0 | 173 / 26 / 0 | 14 / — / — |
| Random, tighter plasticity | 29 / 2 / 23 | 327 / 23 / 156 | 17 / 1 / 10 |
| Inherited, original plasticity | 55 / 66 / 51 | 1,402 / 1,413 / 1,393 | 32 / 30 / 25 |
| Inherited, frozen | 51 / 50 / 54 | 1,330 / 1,316 / 1,586 | 27 / 27 / 37 |

Frozen random starts 72 and 73 went extinct at 2,228.3 and 663.7 seconds.
The other pilots reached the full duration. Transplants share ancestry and do
not add three independent evolutionary origins. These comparisons show effects
on ecological persistence, not successful within-life learning.

`garden challenge RUN --output DIRECTORY --seconds 180` clones the same living
population with its ages, energy, body shapes, positions, food, fields, and random
states intact. It compares retained state, cleared plastic offsets/traces,
cleared neural activity, and plasticity disabled throughout. Each intervention
faces both unchanged and immediately reversed quality; further reversals wait
until after the readout. Mutation is disabled and reproduction remains active.
The report tracks food intake, offspring, and survival of the original cohort,
including creatures that die. Each branch includes an exact starting checkpoint.
These are paired perturbations of one community, not independent replicates;
erasure can disturb useful or harmful ordinary dynamics without testing learning.

The original plasticity settings permit strong saturation: in random seed 71,
75% of active synaptic offsets reached the limit and 76% of neurons exceeded
0.95 absolute activity at 3,600 seconds. A second pilot uses a learning-rate
ceiling of 0.02 and an offset bound of 0.1. In that seed's tighter trial, only
0.39% of active neurons exceeded 0.95 absolute activity at the endpoint. Body
composition also diverged: eight two-module creatures in the original trial
versus 26 one-module and three two-module creatures with tighter plasticity.
Those are selected endpoint populations, not a same-genome stability experiment.
The tighter setting is retained as `configs/v7.toml`; `configs/v7-wide.toml`
preserves the initial pilot. Ecological effects varied by seed. These settings
were chosen after inspecting the first pilot and require evaluation on new seeds.

All isolated association effects remained tiny. Random seed 71 descendants had
mean association/reversal alignment -2.29e-7/-1.72e-7; inherited environment 81
had +4.67e-9/-6.10e-10. The tighter seed-71 population produced +1.44e-6/+1.65e-6,
but freezing plasticity produced comparable +1.55e-6/+1.69e-6 responses. Clearing
activity removed nearly all of that response. These measurements do not show
useful plastic association learning; raw paired responses are committed.

The adult challenges also had mixed effects. For random seed 71, intact original
creatures acquired 2,867 energy over 180 seconds with unchanged quality and 2,346
after reversal. Clearing synaptic state increased those totals to 4,716 and
4,547. In inherited environment 81, clearing synapses improved intake with
unchanged quality (12,562 → 13,577) but reduced it after reversal
(14,986 → 14,299). These are two community snapshots. Intake includes predation;
fresh-food and favorable/unfavorable intake are reported separately. They show
state dependence, with no consistent adaptive benefit established.

The [paired trajectories](docs/v7-learning.png) are regenerated from compact
evidence by `uv run python scripts/plot_learning.py`. A 30.03-second recording
decoded to 901 frames at 1,024×1,024. The selected-creature inspector was checked
visually. Maximum absolute energy-ledger error across the 15 completed pilots
was 2.02e-7 of cumulative introduced energy. Data, matched-founder checks,
state-saturation measurements, CUDA checks, and raw challenge results are in
`docs/results/v7*.json`; exact sources/checkpoints remain under `runs/`.

Ongoing validation: tighter plasticity versus frozen controls on previously
unused seeds 91/92/93, and continuation of tighter random seed 71 to three
simulated hours. These outcomes will be recorded as they finish. Proceeding to
V8's controlled structural mutations while those independent trials run.

## V8 implementation plan

Use a bounded recurrent graph with up to 32 neuron slots, initially 16 active,
and inherited binary masks for neurons and input/recurrent/output connections.
Inactive units, connections, and their plastic state must not affect behavior.
Mutation may duplicate/delete a neuron or add/remove one edge, with a minimum of
four active units. The controller remains recurrent; this first topology version
does not introduce distinct feed-forward layers or the full NEAT algorithm.

Neuron duplication copies incoming connections and bias, splits every outgoing
connection between the two copies, and handles self-connections consistently.
With matching initial hidden states and plasticity disabled, this should preserve
the circuit's function until later mutations differentiate the copies. Test this
directly over input sequences. Function-preserving network enlargement is also
the principle behind [Net2Net](https://arxiv.org/abs/1511.05641); its published
feed-forward results are not evidence for this recurrent implementation.

Charge explicit energy for active neurons/connections at birth and during life.
Record architecture sizes, mutations, and costs. Preserve older checkpoints and
provide tested, neutral padding from 16-unit V7 genomes into the larger template.
Then cross plastic/frozen connections with evolving/fixed topology, matching
initial genotypes, resource laws, and cost coefficients across treatments.
