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

New-seed validation is complete. Tighter plasticity versus frozen controls on
previously unused seeds 91/92/93 gave populations 32/7/17 versus 23/4/32 and
births 209/64/203 versus 366/66/231 at 3,600 seconds. All six survived. The plastic
treatment had more survivors in two seeds but fewer births in all three;
there is no consistent ecological advantage across these measures. Each pair
had identical founding genomes. Tighter random seed 71 reached three simulated
hours with 44 creatures, 1,327 births, and living generation 46; all survivors
had one module. These seven runs are in `docs/results/v7-validation.json`.
Probing that three-hour population with both 3 and 30 conditioning rounds still
gave tiny associations. At 30 rounds, mean association/reversal alignment was
6.18e-7/1.24e-6 with plasticity versus 6.09e-6/6.09e-6 without it. Clearing neural
activity left -1.49e-7/-2.81e-8. More conditioning changed synapses but did not
produce evidence of useful plastic association learning in this assay.

## V8 design and validation

The implemented preset uses a recurrent graph with 32 neuron slots, initially 16 active,
and inherited binary masks for neurons and input/recurrent/output connections.
Inactive units, connections, and their plastic state must not affect behavior.
Mutation may duplicate/delete a neuron (5% chance per birth attempt) and/or
add/remove one edge (10%), with a minimum of four active units. Each selected
operation chooses addition or removal equally; impossible operations do nothing.
The controller remains recurrent; this first topology version
does not introduce distinct feed-forward layers or the full NEAT algorithm.

Neuron duplication copies incoming connections and bias, splits every outgoing
connection between the two copies, and handles self-connections consistently.
With matching initial hidden states and plasticity disabled, this preserves
the circuit's function until later mutations differentiate the copies; a test
checks 100 successive input patterns, including recurrent self-connections.
Subsequent plastic trajectories need not remain equivalent because duplication
changes the set of independently updated synapses. Function-preserving enlargement is also
the principle behind [Net2Net](https://arxiv.org/abs/1511.05641); its published
feed-forward results are not evidence for this recurrent implementation.

The 4,496 inherited values comprise 2,245 possible neural weights/biases, 11
physical/plasticity traits, and 2,240 binary mask genes. The masks cover 32 nodes,
1,024 input edges, 1,024 recurrent edges, and 160 output edges. A dense 16-unit
founder expresses 848 connections. Dormant weights can mutate but do not affect
activity, actions, or plastic traces. Edge addition starts its weight at zero;
later weight mutations may make it useful. This avoids large immediate changes,
but new structure still pays a cost and has no NEAT-style speciation protection.

Per module, neuron/connection maintenance costs are 0.0005/0.00001 energy per
second, and construction costs are 0.05/0.001 energy on successful reproduction.
The initial 16-unit graph therefore costs 0.01648 energy per second and 1.648
energy to build. These are additional to the existing body and plasticity costs.
Construction is included in reproduction loss and maintenance in its existing
ledger; their separate counters are informational, not additional deductions.
Blocked births incur no construction cost. Founders are externally initialized
with built bodies/brains and the usual starting energy.

Neural architecture sizes and successful structural births are recorded. The
new inspector can show inherited recurrent weights and module 1's acquired
offsets (`B`). V7-to-V8 transfer preserves the old active circuit while padding
into dormant slots; tests also check preservation of plastic trajectories.
Older checkpoints remain loadable. Evaluation and state challenges disable
structural mutation as well as ordinary weight/trait mutation.

Mechanical verification: 107 tests pass; the GPU smoke check passed through 37
quality reversals with a 0.000382 energy-ledger residual and about 11 MB peak
allocated CUDA memory in its deliberately small world. This is a correctness
check, not a full-population memory benchmark.

Completed experiment: environments 111/112/113 each ran all four combinations of
plastic/frozen synapses and evolving/fixed topology for 3,600 seconds, using
matched transplants from V7's three-hour seed-71 population. All treatments share
the same starting graph, resource laws, and cost coefficients. Fixed topology
uses `configs/v8-fixed.toml` (both structural mutation probabilities zero),
while ordinary inherited weights and traits still mutate. A separate random-
founder batch (121/122/123) checked viability without the inherited circuit.
These shared-ancestry transplants are separate from independent random origins.

| Treatment | Final populations | Births | Maximum living generations |
| --- | --- | --- | --- |
| Evolving topology, plastic | 44 / 43 / 39 | 562 / 556 / 585 | 14 / 15 / 22 |
| Evolving topology, frozen | 49 / 41 / 41 | 594 / 506 / 574 | 16 / 18 / 15 |
| Fixed topology, plastic | 46 / 45 / 40 | 612 / 486 / 580 | 26 / 17 / 21 |
| Fixed topology, frozen | 44 / 39 / 37 | 532 / 519 / 563 | 12 / 15 / 17 |
| Random founders, evolving/plastic | 8 / 0 / 44 | 50 / 9 / 1,018 | 8 / — / 34 |

All 12 transplants survived; random seed 122 went extinct at 1,009.8 seconds.
Living brains spanned 14–18 neurons across the six evolving-topology endpoints,
with population means 15.61–16.29. Fixed controls retained exactly 16 neurons and
848 connections throughout. Initial genomes match exactly across each four-way
treatment set, and cached neuron/connection counts match their actual masks.
Ecological effects are mixed; more complicated circuits are not established as
better. The [figure](docs/v8-topology.png), raw records, audits, and probes are
committed. Sources and exact checkpoints remain under `runs/`.

Thirty-round association probes of transplanted environment 111 and random seed
123 descendants remained near zero or slightly negative. Clearing plasticity
did not remove an otherwise useful association. In the adult environment-111
challenge, intact original creatures acquired 15,674/15,634 energy with
unchanged/reversed quality; clearing plastic offsets yielded 15,128/14,854, and
disabling plasticity throughout yielded 16,774/15,995. Survival also varied by
intervention. These are state effects in one community, not proof of learning.

A 30.03-second V8 video decoded to 901 frames at 1,024×1,024; the circuit inspector
was visually checked. Random seed 123 completed three simulated hours with 56
living creatures, 3,341 births, and maximum living generation 87. Its mean active
circuit had 16.30 neurons and its mean fresh-food allocation was 0.916. The
[long-run record](docs/results/v8-long.json) retains the trajectory; this is one
surviving origin, not a general survival guarantee.

## Community assembly experiment

The V7 three-hour seed-71 population consisted of 44 scavenger-allocated bodies
(mean fresh-food allocation 0.085). Its cumulative uptake transfers were 8.8%
fresh food, 56.5% detritus, and 34.7% predation. The V7 environment-81 population
consisted of 55 grazer-allocated bodies (mean allocation 0.885), with uptake
shares 57.2% fresh food, 5.9% detritus, and 36.9% predation. These are shares of
recorded uptake transfers, not independent primary energy sources: energy can
be eaten, recycled, and transferred again. Both groups also eat other organisms.

Detritus retains its quality when fresh-food identities reverse. This gives
scavengers an alternative route to persistence that may reduce pressure to learn
those reversals. The four-way V8 experiment uses that scavenger ancestry, so its
scope is narrower than testing all ecological strategies.

`scripts/assemble_communities.py` tests the two naturally evolved source pools
alone (192 founders) or mixed (96 each) in new environments 151/152/153 for
3,600 seconds. All use V8's shared ecology and plasticity settings, reset acquired
states, and retain normal mutation. The script preserves per-founder source
IDs/generations, source-file hashes, the exact experiment script, and ancestry
through each lineage. Its separate `origins.jsonl` tracks populations, births,
deaths, mean diet, and cumulative uptake, including dead individuals. Tests check
balanced mixtures, pure-source treatments, neutral genome padding, reset states,
and saved ancestry. Different body sizes retain different founder energy.

The purpose is to test coexistence and resource use. These are deliberately
assembled populations, not spontaneous speciation. All 15 trials completed:

| Treatment | Final grazer/scavenger ancestry, environments 151 / 152 / 153 |
| --- | --- |
| Grazer source alone | 47/0; 53/0; 61/0 |
| Scavenger source alone | 0/44; 0/42; 0/43 |
| Mixture, all mechanisms | 0/43; 0/41; 0/44 |
| Mixture, attacks disabled | 35/47; 60/36; 18/52 |
| Mixture, recycling disabled | 58/0; 49/0; 59/0 |

Grazer ancestry disappeared at 2,378, 1,494, and 1,252 seconds in the mixtures.
Removing attacks retained both ancestries for the measured hour; that does not
establish indefinite coexistence. Removing recycling eliminated scavenger
ancestry after 137, 127, and 147 seconds. The recycling intervention preserves
primary assimilation and dissipates the otherwise recycled energy. The attack
intervention disables bites while retaining weapon costs and digestive penalties.
[The plot](docs/v8-assembly.png) is generated from [compact records](docs/results/v8-assembly.json).

These interventions establish dependence on the mechanisms in these particular
communities. They do not identify each direct transfer or prove cooperation.

## V9 — food-flow provenance

V9 adds passive accounting with the same physical laws and 32-input/5-output
controllers as V8. Each food packet carries four energy-credit values: material
produced by grazer-, generalist-, or scavenger-allocated bodies, or externally
introduced material with no attributed producer. Reporting bins are diet >0.65,
0.35–0.65 inclusive, and <0.35. They are not species or labels available to brains.
Producer guild is measured when fresh food is processed; consumer guild is
measured when material is eaten. Ancestry remains separately recorded.

Several bodies consuming one fresh packet still create exactly one detritus
packet, carrying their proportional contributions. Partial feeding preserves that
mixture. The ledger follows absorbed energy, digestive dissipation, expiration,
and remaining stock separately for each producer. Another matrix records actual
prey-to-predator assimilation. Death records label the immediate final step as
maintenance or predation; prior bite damage can still contribute to a later
maintenance death. These measurements cannot infer cooperation or independent
primary-energy sources, because energy can be transferred more than once.

Tests cover shared packets, partial uptake, expiry, unattributed additions,
disabled recycling, simultaneous attacks, and save/resume. A paired V8/V9 test
checks exact CPU equality of physical agents, food, fields, random streams, and
totals through births, deaths, and nutritional reversals. Rendering is also
checked as an observer. CPU and RTX 5080 replay checks pass; the latter reports
an energy residual of 0.000382 and zero detritus-credit residual in its small
eight-second test. This is a correctness check, not a performance benchmark.

All six traced repeats completed. Their final agents, food, fields, random
streams, totals, and complete birth/death histories match the corresponding V8
runs exactly; only the new death-cause annotations differ. The largest
detritus-credit balance residual was 2.91e-11. The [audits](docs/results/v9-audit-mixed.json)
and [no-attack audits](docs/results/v9-audit-no-attacks.json) are reproducible with
`scripts/audit_trophic.py`; exact checkpoints remain under `runs/v9-traced-*`.

The [food-flow figure](docs/v9-trophic.png) shows that 95.8–97.9% of detritus uptake
by scavenger-allocated creatures came from their own guild with attacks active.
Even with attacks disabled and grazers present, that share was 90.7–92.3%.
Scavenger-to-scavenger predation also dominated their meat uptake. Transfers
between guilds exist, but this is mostly recycling within a guild, rather than
an obligate grazer-to-scavenger food chain. The current law allows every body
to process fresh particles instantly and recycle a fixed fraction, regardless
of its own assimilation efficiency. That is a candidate mechanism to revisit
with finite, specialization-dependent handling rates.

The next environmental hypothesis is spatial shelter that locally obstructs
attacks for any organism. It will be tested against an otherwise matched world
without protection. Research motivates testing refuges, not presuming success:
[Li et al. (2017)](https://peerj.com/articles/2993/) found coexistence effects in
their modeled predator–prey systems, while [an empirical study of intraguild
predation](https://pubmed.ncbi.nlm.nih.gov/23004014/) found that habitat complexity
weakened predation without promoting coexistence. Our mixed omnivores differ
from both systems and need their own controls.

## V10 — spatial shelter

The first shelter preset selects four of eight resource patches with its own
random stream. Each has a radius-36 cover region: full cover within 80% of the
radius, then a linear taper to zero. Overlapping regions take the maximum, and
the field is clipped to the dish. It is permeable terrain, not a solid obstacle.
Physical attacks and sensors sample the same bilinear field.

Cover obstructs a bite by `protection * max(attacker_cover, prey_cover)`, with
protection initially 0.95. Thus a sheltered body cannot attack outward at full
strength. The rule uses positions, not feeding guild or ancestry. Food supply,
digestion, movement, armor, attack costs, and birth investment retain their
existing laws. Spatially limited cover is the hypothesis under test, not a
guarantee of coexistence.

Each module gains four shelter readings, bringing its controller to 36 inputs
and five outputs. They are bounded local coverage values, rather than food-like
concentrations. The 32-slot genome now has 4,752 values; a new dense 16-unit
circuit expresses 912 connections. Transfers preserve old sensory names and
put new sensory weights at zero. Existing V8/V9 masks retain added edges dormant;
transfers from earlier dense networks expose the added inputs with zero weights.
The three matched assembly treatments all use the same transferred genomes and
the same resulting brain costs.

The viewer outlines covered patches and provides a shelter overlay with Tab;
[the preview](docs/v10-preview.png) shows environment 161 at 1,200 seconds. Logs
record current shelter occupancy by diet guild and each creature's cumulative
cover exposure, including its final life record. The obstruction counter reports
potential bite demand removed before target-energy and storage caps; it is not
energy created or a measurement of actual energy saved.

The `no_shelter` control removes physical protection while retaining the cue.
The `no_shelter_cue` control removes that sensory channel while retaining
protection. Three new environments, 161/162/163, each run all three treatments
for one simulated hour from the same two source pools used in V8. Both acquired
state and age start at zero; ordinary mutation remains active. The criterion is
persistence and reproduction of both source ancestries relative to the matched
control, followed by longer tests if warranted.

All nine initial trials completed. Endpoints show grazer/scavenger ancestry:

| Treatment | Environment 161 | 162 | 163 |
| --- | --- | --- | --- |
| Cover and sensing | 0/47 | 0/46 | 0/50 |
| Protection disabled | 12/35 | 0/43 | 0/41 |
| Cover sensing disabled | 0/48 | 0/48 | 0/52 |

With protection and sensing, grazer ancestry disappeared at 1,531/1,945/1,392
seconds. Removing protection retained it for the hour in environment 161, but
lost it at 613/2,223 seconds in the others. Removing the cue lost it at
2,303/860/687 seconds. Shelter can change the timing of exclusion, and sensing
can change outcomes, but this preset did not sustain coexistence or establish
a consistent sensory advantage. The [figure](docs/v10-shelter.png),
[records](docs/results/v10.json), and [matched-genome audit](docs/results/v10-audit.json)
include every treatment. Maximum absolute energy residual was 0.0486; detritus
credits balanced exactly at all nine endpoints.

The video at `runs/v10-video/timelapse.mp4` continues protected environment 163
from 600 to 900 simulated seconds. All 901 frames decoded at 1,024×1,024 and
30 FPS. `scripts/verify_video.py` checks the decoded count against the run record.

An exploratory parameter adjustment, [v10-covered.toml](configs/v10-covered.toml),
places cover around all eight patches. Its 161/162/163 batch is running under
`runs/v10-full-shelter`. It changes the fraction of patches covered, not the
protection law; the initial half-covered preset and all its failed outcomes
remain available.

### CUDA replay correction

The first V10 CUDA smoke check found a reproducibility problem: paired copies
of one checkpoint differed in position by 0.0000153 after six simulated seconds.
Random-generator states agreed. This is a numerical execution issue, not an
ecological or shelter-learning effect. Enabling deterministic PyTorch algorithms
removed the discrepancy in the diagnostic repeat.

V10+ CUDA construction and resume now enable deterministic kernels for the
process, disable cuDNN benchmarking, and set a cuBLAS workspace if one was not
already configured. Older versions do not switch these settings off. Runtime
metadata records the mode. This follows [PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/2.11/notes/randomness.html)
and its [deterministic-operations API](https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html).
It does not promise identical results across devices, package versions, or
hardware. CPU trials already use exact replay and are unaffected.

The corrected small GPU check passes with zero tensor tolerance, identical
events/totals, 37 quality reversals, and an energy residual of -0.000464. Peak
allocated memory was about 36 MB in this deliberately small world. A separate
exercise raises founder energy and forces structural mutation attempts to test
replay through births; those settings are mechanical test inputs, not an ecology
trial. Both CPU and CUDA birth exercises passed exact replay with eight births
and eight structurally mutated offspring; the GPU energy residual was 0.00117.
The test suite has 128 passing cases. Both the original discrepancy and the
correction are retained in `docs/results/v10-cuda-*.json`.

## Next feeding-rate experiment

V9's provenance identifies a route around specialization: a body can process
all contacted fresh food in one tick, recycle 35%, and later eat those remains,
even when its own fresh-food assimilation is poor. The next prototype will cap
raw processing per second, scaled by digestive tissue and the square of the
existing allocation to each food type. Separate fresh/detritus budgets avoid
unwanted fresh particles blocking a scavenger from eating nearby detritus.
Their combined capacity cannot exceed the body's processing ceiling. All
existing assimilation, recycling, and energy-loss accounting will still apply.

The initial ceiling will be 60 raw energy units per unit digestive tissue per
second. Tests will cover timestep scaling, simultaneous food competition,
storage limits, source-credit conservation, and a control restoring unlimited
processing. Pilot runs will check particle counts and throughput: partial
processing can create more small detritus particles. This is an experimental
capacity constraint, inspired by the importance of handling time in
[Holling's predation analysis](https://hahana.soest.hawaii.edu/cmoreserver/summercourse/2010/documents/Holling_1959b.pdf),
not a reproduction of that model or a guarantee of a stable food chain.
