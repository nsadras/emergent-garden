# Continuing evolution experiments

Development **resumed at the user's request on September 23, 2026**, after the
V15 pause and subsequent interface/resource work. The current iteration is V20
(package 0.21.0); [foraging experiments](docs/FORAGING.md) focus on rare, valuable
meals and affordable exploration. Current evidence, historical interrupted work,
and viewing commands are in [HANDOFF.md](HANDOFF.md). Dependencies remain managed with uv.
V0–V5 and their evidence are preserved in `EVOLUTION.md`.

The [V19 neural experiments](docs/NEURAL_VARIATION.md) introduce varied founder
graphs and compare architecture, mutation, and gentle motor learning across
three seeds. All 15 pilots, 12 continuations, and 18 paired learning transplants are
complete. Learning improved births in five of six same-genotype descendant pairs,
but not in the three evolutionary continuations. Completed shuffled-return controls favored own returns in four of six
comparisons, leaving general adaptive credit assignment unestablished. Stronger mutation lost one start to extinction; the delayed-credit
diagnostic remained weak despite extending the eligibility trace.
All 310 tests and separate CPU/CUDA replay exercises pass. The user's V16 working
configuration is preserved.

[V20](docs/SENSOR_RADIUS.md) tests larger sensory footprints with matched
genomes and unchanged energy laws. Nine pilots show stronger spatial signals
but mixed reproduction. All three four-radius starts reached 1,800 seconds, but had fewer births than
the radius-1 controls. The shuffled-return control has also completed; the
evidence still supports conditional effects rather than a general learning advantage. All 322 tests pass; CPU/CUDA mechanics and replay
checks pass. The active research loop continues.

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
places cover around all eight patches. The completed 161/162/163 batch is under
`runs/v10-full-shelter`:
endpoints were 0/53, 0/56, and 0/55; grazer ancestry disappeared at
2,614/2,596/1,155 seconds. Broader coverage did not preserve both ancestries.
The [full-coverage records](docs/results/v10-full-shelter.json) and
[12-run audit](docs/results/v10-complete-audit.json) retain this failed parameter
adjustment alongside the original half-covered preset. Founder genomes and
patch layouts match across all four treatments in each environment.

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

## V11 — finite processing capacity

V9's provenance identifies a route around specialization: a body can process
all contacted fresh food in one tick, recycle 35%, and later eat those remains,
even when its own fresh-food assimilation is poor. The V11 prototype caps
raw processing per second, scaled by digestive tissue and the square of the
existing allocation to each food type. Separate fresh/detritus budgets avoid
unwanted fresh particles blocking a scavenger from eating nearby detritus.
Their combined capacity cannot exceed the body's processing ceiling. All
existing assimilation, recycling, and energy-loss accounting still apply.

For tissue `T = modules × (core_radius / base_radius)²` and fresh-food allocation
`d`, capacities are `60 × T × d²` for fresh food and `60 × T × (1-d)²` for
detritus, in raw energy units per second. Membrane area adds no digestive
capacity. Unused capacity in one pathway cannot be borrowed by the other.
Contested particles still split their available energy equally among touching
creatures before capacity and body-storage limits apply. Unprocessed food
remains in place. This is an experimental
capacity constraint, inspired by the importance of handling time in
[Holling's predation analysis](https://hahana.soest.hawaii.edu/cmoreserver/summercourse/2010/documents/Holling_1959b.pdf),
not a reproduction of that model or a guarantee of a stable food chain.

### Fragmentation pilot and correction

Processing on all 30 physics steps per second created a new remnant from each
partially handled fresh particle on every feeding step. The 300-second mixed
pilot reached 35,295 food particles, ran at 2.42x simulated/wall time, and had
an energy residual of +1.636. Repeated float32 additions of tiny meals to much
larger body-energy stores amplified rounding error.

The [V11 preset](configs/v11.toml) instead feeds at 5 Hz, budgeting 1/5 second's
capacity at each feeding step. Physics remains at 30 Hz. Intake is summed in
float64 per creature before one update to each float32 body store or lifetime
counter. This keeps the physical residual visible rather than adding a balancing
term to the ledger. The same mixed pilot then peaked at 7,318 particles, ran
at 7.96x, and had an energy residual of +0.00147. These are observed concurrent
run speeds, not isolated benchmarks. The updated per-tick alternative remains
in [v11-per-tick.toml](configs/v11-per-tick.toml); original prototype runs require
their archived sources to reproduce their earlier arithmetic exactly.

The viewer aggregates overlapping crumbs by screen pixel and colors them by
summed fresh/detritus energy. Small remnants are smaller and dimmer than full
particles. This only changes rendering. The [inspector](docs/v11-preview.png)
shows each creature's two processing capacities; life histories and total
metrics record raw fresh and detritus amounts processed.
The recording at `runs/v11-video/timelapse.mp4` follows the mixed pilot from
300 to 600 seconds at 10x speed. All 901 frames decoded at 1,024×1,024 and
30 FPS; [the verification record](docs/results/v11-video.json) stores its hashes.

`unlimited_handling` removes the capacity limit while preserving 5 Hz feeding
and the revised accumulation arithmetic. It is the main comparison for the
capacity hypothesis. `unlimited_feeding` restores V10's original feeding cadence
and arithmetic; an exact parity test covers reproduction, deaths, and quality
reversals. The treatments have distinct purposes and are not interchangeable.

### Ecology protocol and current evidence

The same evolved V7 grazer and scavenger source pools initialize fresh bodies
and neural states in environments 181/182/183. Mixed communities start 96 of
each ancestry; single-source controls start 192 from that source. All use the
V11 half-covered landscape, ordinary mutation, and a 3,600-second horizon unless
the community becomes extinct. Comparisons include limited/unlimited mixtures,
limited grazers alone, limited/unlimited scavengers alone, and mixtures with
recycling disabled. The recycling intervention preserves fresh-food assimilation
and dissipates the fraction that would otherwise become detritus.

All 18 trials completed. Endpoints show grazer/scavenger source ancestry:

| Treatment | Environment 181 | 182 | 183 |
| --- | --- | --- | --- |
| Limited mixed community | 46/27 | 26/15 | 14/0 |
| Unlimited mixed community | 32/18 | 0/50 | 34/26 |
| Limited grazers alone | 29/0 | 37/0 | 25/0 |
| Limited scavengers alone | 0/0 | 0/0 | 0/0 |
| Unlimited scavengers alone | 0/48 | 0/53 | 0/44 |
| Limited mixture, recycling off | 42/0 | 45/0 | 59/0 |

Both ancestries reproduced and persisted for the hour in two of three mixtures
under each capacity treatment, in different environments. Limited processing
lost scavenger ancestry at 855 seconds in environment 183; unlimited processing
lost grazer ancestry at 1,310 seconds in environment 182. This does not establish
a higher frequency of coexistence. It does change the measured dependency:
limited scavengers alone died at 136.33/130.8/133.1 seconds without births,
whereas unlimited scavengers alone produced 491/560/531 offspring. Removing
recycling eliminated scavenger ancestry from limited mixtures at 113/98/100
seconds, again without births, while grazer ancestry persisted and reproduced.

The grazer guild produced 98.89%/98.68%/98.57% of detritus energy absorbed by
scavengers in the limited mixtures, versus 15.21%/3.60%/7.43% with unlimited
handling. These are full-hour cumulative fractions; the third limited mixture
contains an early scavenger extinction. Guild labels describe diet when energy
was transferred, and differ conceptually from source ancestry. The combined
provenance and intervention evidence supports a grazer-to-scavenger food
dependency in these tested communities. It does not establish cooperation,
spontaneous speciation, or permanent coexistence. See the
[population figure](docs/v11-handling.png), [food-flow figure](docs/v11-trophic.png),
[complete records](docs/results/v11.json), and [18-run audit](docs/results/v11-audit.json).
Maximum absolute energy residual was 0.0272 units. Founder samples and patch
layouts match across paired treatments, and all food-credit checks passed.

The two limited mixtures retaining both ancestries completed three simulated
hours under `runs/v11-limited-long-181` and `-182`. They ended with ancestry
populations 53/18 and 14/11, 3,728 and 1,905 total births, and maximum generations
68 and 51. Energy residuals were -0.00561 and -0.0751 units. All surviving bodies
had one module. These are selected continuations of successful mixtures, not
independent replicates or an unbiased estimate of long-term coexistence frequency.
The [continuation records](docs/results/v11-long.json) retain their histories.

The fresh random-founder pilots were much less viable than the assembled
communities: seed 171 reached 1,800 seconds with one founding creature and only
two births across the run; seed 172 became extinct at 480.3 seconds after two
births. Neither establishes a reproducing population from a new random origin.
[All pilot records](docs/results/v11-pilots.json) include the failed initial
fragmentation runs.

The 139-test suite covers time scaling at 30/60 Hz, the 5 Hz control cadence,
simultaneous competition, storage caps, many tiny meals, food-credit conservation,
birth-state reset, exact old-law parity, and checkpoint and observer behavior.
Small [CPU](docs/results/v11-cpu-births.json) and
[RTX 5080](docs/results/v11-cuda-births.json) exercises passed exact checkpoint
replay through eight births and eight structurally mutated offspring each.
Their energy residuals were +0.000778 and -0.000329; the GPU exercise used about
38 MB of peak allocated tensor memory. These are correctness checks, not full
population performance measurements.

## V12 — energetic feedback for acquired motor readouts

The earlier recurrent plasticity rule changed synapses but did not establish
useful learning in measured populations. V12 adds a more direct credit path:
small exploratory motor changes leave an eligibility trace, and subsequent
energetic outcomes modify the motor readout. The inherited recurrent circuit,
its plasticity rule, body development, ecology, and reproduction remain active.
This is a designed learning mechanism whose rates can evolve, not a claim that
evolution discovered the algorithm.

The update uses the likelihood score of Gaussian motor exploration, following
the stochastic-unit approach in [Williams (1992)](https://doi.org/10.1007/BF00992696).
Short eligibility traces and modulatory feedback also draw on the approach
discussed by [Miconi (2017)](https://elifesciences.org/articles/20899). Our implementation
adapts motor readouts from continuous physical energy flow; it does not reproduce
that paper's recurrent-network tasks or delayed trial rewards. Bounds, forgetting,
and finite traces make this an online approximation, without an optimality guarantee.

### Inherited rules and acquired state

There are still 36 sensory inputs and five controller outputs per module, with
up to 32 recurrent neurons and inherited connectivity. The genome gains two
continuous traits, bringing its length to 4,754 values: one sets the motor-learning
rate and the other sets exploratory noise. Existing V11 transfers preserve every
old neural and body gene; added traits start at -2 and -1 respectively. They
mutate and inherit like the other developmental traits.

Each module adds a transient `2 × (hidden_size + 1)` motor-offset matrix and an
eligibility trace of the same shape. The extra column is a bias feature. A
running energetic baseline is also acquired. Offspring start all these states
at zero; none is copied into their genome or inherited from the parent's life.
Offset connections follow the inherited motor-output masks, with two always
available bias offsets. Inactive body modules remain zeroed.

At each controller update, the preceding interval's physical return is credited
before the next action is chosen. Return is actual food and meat absorbed minus
maintenance, propulsion, secretion and attack costs, and energy lost to bites,
divided by `feedback_scale × body_area`. Reproduction transfers are excluded.
The feedback uses no hidden patch-quality label, target trajectory, or externally
assigned behavioral score. A body shares its return across its active modules.

The energetic baseline tracks a rate, so a newborn's shorter first control
interval does not imply a different expected rate. The integrated difference
between received return and that baseline is clipped to [-1, 1]. It scales the
previous eligibility trace and the inherited learning rate. The fifth neural
output gates new motor eligibility through its nonnegative sigmoid value;
the older recurrent rule continues to use that output's signed transform.

| Parameter | Initial setting |
| --- | --- |
| Maximum motor-learning rate | 0.2, multiplied by sigmoid of trait 11 |
| Exploration standard deviation | 0.05 + 0.45 × sigmoid of trait 12 |
| Transferred initial rate / noise | approximately 0.0238 / 0.171 |
| Eligibility decay time | 2 seconds |
| Energetic baseline time | 10 seconds |
| Motor-offset forgetting half-life | 120 seconds |
| Total acquired motor-logit correction bound | 0.5 per motor, before exploration |
| Added maintenance | 0.02 energy per active module per second |

The [current preset](configs/v12.toml) normalizes `(hidden_state, 1)` to unit
length and constrains each offset row's length to at most 0.5. Their dot product
therefore cannot exceed 0.5 in magnitude, regardless of hidden-layer width.
This bounds each decision's correction, not divergence of whole trajectories.
Exploration uses a separate checkpointed random stream. The original
[per-weight-bound preset](configs/v12-wide.toml) retains `motor_normalized = 0`;
older prototype checkpoints without this field load with that original setting.

`no_motor_learning` removes motor offsets and their eligibility while retaining
exploration, the energetic baseline, and maintenance costs. `no_exploration`
uses zero perturbations while still advancing the exploration random stream;
freshly initialized motor offsets then stay zero. `no_motor_reward` withholds
only this new reinforcement signal. `no_plasticity` disables both acquired
weight mechanisms while retaining exploration and costs. Acquired-state
challenges erase the motor matrices and baseline when erasing plastic state.
The older cue-only association assay explicitly reports that it does not test
the new motor mechanism.

### Mechanism check and pilot correction

`scripts/probe_motor_learning.py` constructs 64 identical circuits with a supplied
binary cue feature and a motor-error penalty. Genomes remain unchanged, each
circuit encounters both cues, and the desired association reverses after 120
assay seconds. Three random streams, 201/202/203, each receive matched exploration
with learning enabled or disabled. These circuits are never seeded into the dish.

With bounded corrections, the late initial mean squared error was approximately
0.184–0.185, versus 0.254–0.255 with learning disabled. It worsened to
0.319–0.321 after reversal, then returned to 0.192–0.193. The original per-weight
rule reached about 0.153–0.156, with a larger reversal error. The
[figure](docs/v12-rule.png), [original rule records](docs/results/v12-motor-rule.json),
and [bounded rule records](docs/results/v12-bounded-rule.json) establish that the
readout can learn this supplied task. They do not demonstrate evolved feature
learning, ecological usefulness, or intelligent navigation.

The first physical mixed-community pilot, environment 191 for 300 seconds,
showed why a behavioral bound was needed. Per-weight motor limits left 13
creatures and 24 births, versus 69 and 96 with motor learning disabled. Many
small weight changes could combine into a large motor correction. Bounding
the total correction left 76 creatures and 94 births in the same pilot.
This is recovery from a harmful rule setting, not consistent evidence of a
learning benefit. The disabled-learning control's physics is exactly unchanged
by normalization, as checked independently. All three pilots and matched-founder
checks are retained in [the pilot records](docs/results/v12-pilots.json) and
[audit](docs/results/v12-pilot-audit.json).

### Completed ecological and lifetime evaluation

Nine 3,600-second community trials use new environments 201/202/203 with bounded
learning, matched exploration without motor learning, or no exploration. They
start from the same evolved V7 source pools, with ordinary mutation active.
Their results can measure the combined consequences of lifetime adaptation and
subsequent genetic selection, and must not be described as a pure learning effect.

`scripts/assay_motor_lifetimes.py` separately samples eight grazer genotypes from
the V11 limited mixture in environment 181. Each is tested in environments
10021/10022/10023 under all three interventions for 240 seconds, with every
mutation probability zero. Outcomes track the founding individual through
survival or its death record: energy acquired/spent, offspring, age, and distance.
Each world starts with one creature; its genetically identical offspring may
share the dish. The assay uses a 128-unit dish, four 12-unit patches, 160 initial
food particles, 12 new particles per second, capacity 16, and 60-second nutritional
reversals. This is a deliberately richer evaluation habitat, not a claim about
performance in the main 512-unit community. Source genomes, package sources,
configuration, experiment code, and every completed paired trial are retained.
Genotypes from one source community are not independent evolutionary replicates.

The nine community trials completed with the following final source-ancestry
populations (grazer/scavenger); ordinary genetic mutation remained active:

| Environment | Learning and exploration | Exploration only | Neither |
| --- | --- | --- | --- |
| 201 | 25/20; 882 births | 51/23; 1,045 births | 62/21; 1,280 births |
| 202 | 35/0; 544 births | 24/0; 474 births | 54/27; 1,139 births |
| 203 | extinct at 740 s; 84 births | 19/11; 531 births | 39/29; 1,034 births |

Both ancestries persisted for the hour in one, two, and three communities,
respectively. Three environments are too few to establish general persistence
probabilities. They do show that the new motor mechanism is not a consistent
improvement for these starting populations. The [community figure](docs/v12-outcomes-communities.png),
[all nine records](docs/results/v12.json), and [matched-founder audit](docs/results/v12-audit.json)
retain the failed trials as well as the survivors.

All 72 fixed-genotype lifetime trials also completed. Averaged across eight
genotypes and three environments, the original individual acquired 298.0 energy
with learning, 418.4 with exploration alone, and 438.5 with neither. Mean offspring
counts were 1.00, 1.50, and 1.83; 15/24, 23/24, and 21/24 individuals survived the
full 240 seconds. Acquired energy includes all intake, including any meat.
Compared with exploration alone, learning reduced mean acquired energy for
seven of eight genotypes after averaging their three environments; one genotype
improved. The paired mean difference was -120.5 energy and -0.50 offspring.
These genotypes share one source community, so environmental repeats are not
independent evolutionary replicates. See the [paired figure](docs/v12-outcomes-lifetimes.png)
and [all lifetime records](docs/results/v12-lifetime-assay.json).

The constructed cue task establishes that the readout can learn a supplied
association. The physical assays establish no useful ecological adaptation at
these settings. Exploration and credit assignment both need further work; the
next body-development experiment will use zero exploratory motor noise so that
this measured harm does not confound its comparison. The motor-learning
mechanism and original presets remain available for later targeted experiments.

The 154-test suite passes, including causal timing of motor credit, inactive-edge
masking, inherited-rule preservation, newborn resets, energetic feedback,
normalization independent of circuit width, control parity, and exact checkpoint
and observer behavior. Both [CPU](docs/results/v12-bounded-cpu-births.json) and
[CUDA](docs/results/v12-bounded-cuda-births.json) exercises passed exact replay
through eight births and structural mutations, with active motor learning and
the readout bound checked. Their energy residuals were -0.000248 and -0.000330.
The GPU exercise used about 38 MB of peak allocated tensor memory. The earlier
per-weight-bound checks are retained separately.

`runs/v12-video/timelapse.mp4` continues the bounded pilot from 300 to 600 seconds.
All 901 frames decoded at 1,024×1,024 and 30 FPS; the
[verification record](docs/results/v12-video.json) retains frame hashes. The
[inspector preview](docs/v12-preview.png) displays inherited recurrent weights,
acquired recurrent changes, and the two acquired motor rows separately.

## V13 — juvenile growth and accessible body plans

The persisting V11 communities remain entirely single-module organisms.
`scripts/audit_development.py` tests whether changing only the encoded module
count is affordable under the existing birth law. Among all 73 and 41 living
parents at the two one-hour mixed-community checkpoints, **none could afford a
two-module offspring even at maximum stored energy**. Median required debits
were 106.32% and 105.57% of the parents' maximum stores. A three-module offspring
cost still more. [The counterfactual audit](docs/results/development-affordability.json)
includes each parent's bound and the proposed child's debit.

This identifies an accessibility barrier, not proof that larger bodies are
intrinsically unfit or impossible to evolve: simultaneous changes to core size,
spacing, or neural construction could alter affordability. V13 tests juvenile
development using the existing bounded one-to-three-module grammar. It does not
yet add differentiated cells or arbitrarily shaped bodies.

### Inheritance, construction, and maturation

The genome remains 4,754 values with 13 continuous developmental traits.
Trait 6 still encodes the final module count. Each founder and newborn initially
expresses one module; its inherited target is stored separately from its current
developmental stage. Growth adds one module at a time until the target is met.
Only mature bodies may reproduce. A one-module plan is mature at birth.

| Parameter | V13 setting |
| --- | --- |
| Neighboring module-plan mutation | 3% per attempted offspring |
| Additional body construction | 20 energy per added area unit |
| Neural construction | Existing per-neuron and per-connection costs, per added module |
| Required reserve after growth | 50 energy per unit of the resulting body area |
| Earliest growth / interval after successful growth | 10 seconds |
| Retry after insufficient energy or blocked space | 1 second |
| Exploratory motor noise | Zero in both compared treatments |

An explicit module mutation moves 1→2, 3→2, or 2→1/3 with equal probability.
It changes only the module-count gene, after ordinary continuous and neural
structural mutation. The gene is placed at the center of the destination
interval, subject to the configured gene bound. Attempted module events,
accepted births carrying those events, and actual parent-to-child body-plan
changes are counted separately. Continuous trait mutation can still cross a
module-count threshold without an explicit event.

Birth pays for the currently expressed newborn, including its initial stored
energy and first circuit. Growth later pays for additional body area and neural
construction without creating stored energy. The reserve is an eligibility
condition, not a further charge. Construction is an explicit dissipation term
in the energy ledger, with a per-creature lifetime counter. The informational
neural-construction counter includes its component of this cost without charging
it twice. Like reproduction investment, growth is excluded from motor reward.

A proposed larger circular membrane must fit inside the dish and avoid every
other membrane. Contending proposals are considered in a random order drawn
from a separate checkpointed stream. Accepted growth immediately constrains
later proposals. Failed proposals charge nothing. Existing modules retain
their own neural states as the linear body layout recenters; newly activated
modules start with zero activity, plastic weights, eligibility, and motor
baseline. They first act at the next scheduled controller update. Every growth
event records its creature, ancestry, module counts, radius, and energy cost.

The `adult_births` control builds the entire encoded child at birth using the
previous financing rule. Founders are initialized identically in both treatments,
so the comparison concerns offspring development, not different initial placement
or endowments. All assembled source founders happen to encode one module.
Ordinary mutation, neural architecture changes, recurrent plasticity, finite
feeding, predation, recycling, and shelter remain active in both treatments.
Motor exploration is set to zero following the V12 outcomes; its maintenance
cost remains matched, and fresh motor offsets stay zero. This setting is tested
for exact physical parity with the earlier `no_exploration` intervention.

### Verification and experiments

The 171-test suite includes staged affordability, energy conservation, wall and
neighbor blocking, competing growth proposals, maturation before reproduction,
fresh offspring state, neighboring mutation, rendering purity, and exact replay
through growth and births. All assay paths that disable mutation also disable
the new body-plan events. The counterfactual affordability audit explicitly
constructs a fully formed child even when reading a V13 checkpoint.

[CPU](docs/results/v13-cpu-growth.json) and [CUDA](docs/results/v13-cuda-growth.json)
mechanical exercises passed exact replay through four births with neural
structural mutation and four/nine growth events, respectively. Energy residuals
were +0.000402 and -0.000239 units; peak CUDA allocation was about 37 MB.
The exercise supplies reproduction energy and accelerates growth and mutations,
so these are correctness checks, not evidence of ecological viability.

All six matched 600-second pilots in environments 211/212/213 completed:

| Environment | Juvenile births: population / births / growth events | Fully formed births: population / births / growth events |
| --- | --- | --- |
| 211 | 58 / 165 / 2 | 42 / 142 / 0 |
| 212 | 44 / 138 / 1 | 44 / 136 / 0 |
| 213 | 42 / 164 / 12 | 42 / 149 / 0 |

Juvenile trials ended with 2/0/7 two-module bodies, versus none in the controls.
Two-module parents produced 3/0/26 offspring. The control attempted 2/1/6
explicit module mutations but accepted none of those offspring. The juvenile
treatment accepted 2/1/6 such births; descendants could then inherit their body
plans without a new module event. The [six-run records](docs/results/v13-pilots.json),
[matched-founder and phenotype audit](docs/results/v13-pilot-audit.json), and
[reconstructed developmental histories](docs/results/v13-pilot-development.json)
retain the evidence. The largest absolute energy residual was 0.0105 units.
The [pilot figure](docs/v13-development-pilots.png) uses ten-second samples to
retain the short-lived two-module body in environment 212; two-minute sampling
would have missed that episode.

The first juvenile pilot provides a concrete example of the newly accessible life cycle:
creature 300, descended from single-module founder 118, was born at 417.63 s
with a two-module plan, grew at 490.63 s, and produced three offspring by 600 s.
Its child 348 also grew. The [adult preview](docs/v13-preview.png) and
[juvenile preview](docs/v13-juvenile.png) show that family at the same checkpoint.
This establishes access and reproduction, not a general advantage or long-term
survival of larger bodies. A recording from 600 to 900 seconds is saved at
`runs/v13-video/timelapse.mp4`; all 901 frames decoded at 1,024×1,024 and 30 FPS
in the [video verification](docs/results/v13-video.json).

All six fresh 3,600-second assembled-community trials completed:

| Environment | Juvenile: population / births / expressed module counts (1, 2, 3) | Fully formed: population / births / expressed module counts (1, 2, 3) |
| --- | --- | --- |
| 221 | 51 / 1,125 / (27, 24, 0) | 60 / 939 / (60, 0, 0) |
| 222 | 44 / 937 / (18, 26, 0) | 40 / 899 / (40, 0, 0) |
| 223 | 38 / 864 / (18, 18, 2) | 49 / 982 / (49, 0, 0) |

Both source ancestries survived in every trial. Juvenile development did not
consistently increase population or births, but it made larger reproducing
bodies accessible in all three environments. Two-module parents produced
559/357/392 offspring. Three-module parents produced six offspring in environment
222, although no three-module body survived there to the endpoint. Three-module
growth occurred 3/3/2 times, and two such bodies remained in environment 223.
The fully formed treatments attempted 32/18/39 explicit module mutations but
accepted no larger offspring. See the [figure](docs/v13-development.png),
[six full records](docs/results/v13.json), [audit](docs/results/v13-audit.json),
and [parenthood/growth reconstruction](docs/results/v13-development.json).
Largest absolute energy residual was 0.0654 units, and all food-credit checks passed.

### Fresh random founders

Six separate random-founder trials, with no transferred foraging circuits,
completed 1,800 seconds in the same three environments:

| Environment | Juvenile offspring: population / births / two-module bodies | Fully formed offspring: population / births / two-module bodies |
| --- | --- | --- |
| 221 | 43 / 138 / 9 | 45 / 157 / 2 |
| 222 | 42 / 392 / 34 | 43 / 216 / 41 |
| 223 | 25 / 104 / 10 | 17 / 72 / 8 |

All six retained reproducing populations. Unlike the assembled sources, these
random founders include genes for larger bodies. Both treatments start those
founders as juveniles, and they can grow before producing their first offspring.
The comparison therefore changes offspring development, not founder development.
It does **not** establish that juvenile offspring are necessary for larger bodies
or that V13 beats the original fully formed V11 initialization. It shows viable
random establishment with V13 founder development, and mixed effects of the
offspring rule. The clearest access result remains the one-module-source experiment.

The [population/body figure](docs/v13-development-native.png),
[six histories](docs/results/v13-native.json), [invariant audit](docs/results/v13-native-audit.json),
and [event reconstruction](docs/results/v13-native-development.json) are retained.
Maximum absolute energy residual was 0.222 units, within one part per million
of injected energy in each run; food-credit conservation checks also passed.
All three juvenile-offspring populations completed three simulated hours under
`runs/v13-native-long-221`, `-222`, and `-223`. Every preselected environment
continued; none was dropped after its 30-minute result.

| Environment | Population / births / living generation | Expressed modules (1, 2, 3) | Births by two-/three-module parents |
| --- | --- | --- | --- |
| 221 | 58 / 956 / 31 | (57, 1, 0) | 101 / 0 |
| 222 | 73 / 2,952 / 30 | (60, 13, 0) | 2,044 / 0 |
| 223 | 28 / 726 / 19 | (14, 14, 0) | 522 / 22 |

The [full trajectories](docs/v13-development-long.png) show substantially
different outcomes: larger bodies became rare in 221, remained common in 222,
and coexisted with single-module bodies in 223. Twenty-four and 22 survivors
in the latter two runs encoded two-module plans; some had not yet grown.
All 22 births by three-module parents in 223 came from one individual, and no
three-module creature remained at any endpoint. These counts demonstrate
physical access and reproduction, not a general selective advantage of size.

Environment 222 also retained 37 fresh-food specialists and 36 scavengers,
with mean diet allocations 0.906 and 0.129. They descend from founders 32 and
101, whose initial allocations were already 0.869 and 0.145. This is ecological
sorting and persistence from random founders, not evidence that a new dietary
split evolved. Fixed-genotype tests of these two dietary pools are described below.

The [three histories](docs/results/v13-native-long.json),
[invariant audit](docs/results/v13-native-long-audit.json), and
[event reconstruction](docs/results/v13-native-long-development.json) include
both saved segments. Resume boundaries must agree on every physical metric,
configuration, seed, controller, and intervention before records are joined.
All growth and birth events reconcile with the final checkpoint. Maximum
absolute energy residual was 1.317 units against more than 4.8 million units
injected in that run (0.274 parts per million); food-credit checks also passed.

The verified recording `runs/v13-native-long-video/timelapse.mp4` follows
environment 222 from 10,800 to 11,100 seconds. All 901 frames decoded at
1,024×1,024 and 30 FPS; the [video audit](docs/results/v13-native-long-video.json)
retains frame hashes. This is an illustrative continuation, outside the
three-hour comparison horizon.

### Dependence of the native dietary groups

Twelve additional 1,800-second trials transplant the final environment-222
dietary pools into new environments 251/252/253. Pools retain all 37 grazer and
36 scavenger genotypes at their observed frequencies, including repeated clones;
the [source record](docs/results/v13-native-foodweb-source.json) preserves IDs,
ancestry, diets, and hashes. Each mixture starts with 96 newborns from each pool,
and each alone treatment starts with 192 from its one pool. All five mutation
pathways are disabled. A matched mixed treatment removes recycling while
preserving fresh-food assimilation and the rest of the physical rules.

| Environment | Mixture: grazer / scavenger survivors | Grazers alone | Scavengers alone: extinction | Mixture without recycling: grazer / scavenger survivors |
| --- | --- | --- | --- | --- |
| 251 | 46 / 51 | 55 | 107.9 s | 56 / 0 |
| 252 | 48 / 32 | 73 | 104.5 s | 66 / 0 |
| 253 | 19 / 24 | 76 | 90.7 s | 58 / 0 |

Scavengers produced 249/218/302 offspring in normal mixtures, versus 0/0/1
when alone. Without recycling they disappeared at 173/150/284 seconds, after
1/0/1 offspring. Between 98.01% and 98.74% of scavenger detritus uptake in normal
mixtures was traced to grazer producers. These controls support dependence on
grazer-produced detritus in this community. They do not demonstrate cooperation,
speciation, indefinite stability, or how often native populations establish a
food web: the source was deliberately selected after observing coexistence.

The [population figure](docs/v13-native-foodweb.png),
[12 records](docs/results/v13-native-foodweb.json), and
[audit](docs/results/v13-native-foodweb-audit.json) are retained. In addition to
matched founders, geometry, topology, energy, and food-credit checks, every
living descendant genome exactly matched its founding lineage's genome.
Maximum absolute energy residual was 0.220 units.

## V14 — local coordination within a developing body

V13 provides a reproducing modular body, but its repeated circuits have no
direct neural connection. V14 adds local body-position readings and private
signals between adjacent modules. Connected local controllers are inspired by
[Sims (1994)](https://www.karlsims.com/papers/siggraph94.pdf), which generated local
circuits alongside repeated body parts and allowed neural connections between
adjacent parts. Our model retains a rigid 2D body and local energy-funded
reproduction; it does not reproduce that paper's articulated physics or
task-ranked selection. The hypothesis is that local coordination may help
shared circuits control a growing body. Usefulness must be measured.

### Interface and timing

Each module now receives 40 inputs and produces seven sigmoid outputs.
The old 36 inputs retain their meanings. Four added readings are the module's
two coordinates in its own body frame, divided by twice the core radius, and
two incoming internal signals. Coordinates and signals are bounded within
[-1, 1]. Coordinates describe expressed geometry, not an absolute world location
or a hidden food-quality label. A one-module body has zero coordinates and no
incoming neighbor signals.

Outputs 0–4 remain left/right propulsion, attack, secretion, and plasticity
modulation. Outputs 5–6 specify two internal emissions through `2 × output − 1`.
Their held values follow a discrete low-pass update with a 0.3-second time
constant at the configured controller rate. Every controller reads the previous
signals before any updated signal is committed. Routing averages adjacent
active modules along the body chain, excludes self-connections, and never
crosses between creatures. Information can traverse at most one module boundary
per controller update.

Maintaining a signal costs `0.01 × (A² + B²)` energy per module per second.
This is included in maintenance and energetic motor feedback, and tracked in
its own informational counter without being charged twice. A newly grown
module's signals start at zero and remain zero until its first control update.
Offspring inherit no signals. Neural-activity erasure and recurrent-memory reset
also clear the internal signal state; erasing only plastic weights leaves it intact.

The inherited template still has up to 32 recurrent nodes, initially 16 active,
with evolving masks and the same 13 developmental traits. Its genome now holds
5,140 values. Transfers preserve old weights, named sensory channels, masks,
and traits, and initialize all new weights and signal biases to zero. The new
input/output paths are enabled for existing active neurons, so an ordinary
weight mutation can use them without waiting for a separate edge mutation.
Their construction and maintenance charges apply immediately: six extra
connections per active neuron. Tests preserve the old circuit's computed
outputs to floating-point tolerance; this is not a claim of unchanged full-world
trajectories across versions.

### Controls and verification

`no_internal` suppresses received signals while retaining emission and its costs.
`no_body_sense` suppresses only the two body coordinates. `no_coordination`
suppresses both kinds of new readings. The resulting four treatments keep the
same genomes, anatomy, available actions, and cost rules; their trajectories
and actual expenditure can subsequently differ. The preset keeps motor
exploration at zero as in V13. The isolated-circuit association assay explicitly
reports that it does not test body coordination.

An additional `self_internal` intervention feeds each module its own held
signals in place of its neighbors' signals. Single-module bodies still receive
zero, as they do normally. This preserves an extra local memory loop while
removing information from other modules, helping distinguish inter-module
coordination from the benefit of merely adding recurrent state. It is available
for subsequent fixed-genotype assays and is not part of the four evolution treatments.

All 186 tests pass. Coverage includes private adjacent routing, exclusion of self and
inactive modules, delayed causal influence on motors, bounded position readings,
separate interventions, energy costs including starvation, newborn/growth resets,
neutral inherited circuit extension, and checkpoint replay. Mechanical
[CPU](docs/results/v14-cpu-coordination.json) and
[CUDA](docs/results/v14-cuda-coordination.json) exercises passed exact replay
through four births and six/nine growth events with active internal signaling.
Energy residuals were -0.000109 and -0.000944 units; peak CUDA allocation was
about 37.5 MB. These mechanical checks do not demonstrate evolved coordination.

Eight 600-second pilot trials test all four interventions in environments
231/232. Half of each community comes from V13 native environment 222 at 1,800
seconds, where all 42 survivors encoded two modules; half comes from the
previously used V7 scavenger source. The fresh-food-biased V13 source had mean
diet allocation 0.685 and three founder ancestries, so it is not a pure dietary
guild or a single independently evolved genotype. All eight pilots completed:

| Environment | Both channels: population / births | Body position only | Internal reception only | Neither |
| --- | --- | --- | --- | --- |
| 231 | 62 / 109 | 64 / 102 | 59 / 104 | 61 / 104 |
| 232 | 64 / 127 | 68 / 129 | 54 / 114 | 64 / 111 |

Both source ancestries persisted in every pilot, with no consistent advantage
for the complete interface. Signal magnitude averaged 0.0022–0.0068 across
all pilot endpoints; nonzero emissions also occurred when reception was disabled.
[All eight histories](docs/results/v14-pilots.json) and the
[matched-founder audit](docs/results/v14-pilot-audit.json) are retained.
Maximum absolute energy residual was 0.0199 units, with all food-credit checks passing.
The [inspector preview](docs/v14-preview.png) shows a body selected for visible
emission, not measured communication benefit.

Twelve 3,600-second trials completed all four treatments in new environments
241/242/243, retaining the same source pools:

| Environment | Both channels: population / births | Body position only | Internal reception only | Neither |
| --- | --- | --- | --- | --- |
| 241 | 50 / 896 | 69 / 751 | 73 / 879 | 60 / 851 |
| 242 | 45 / 762 | 52 / 826 | 59 / 987 | 65 / 971 |
| 243 | 50 / 677 | 56 / 684 | 53 / 687 | 59 / 687 |

Both source ancestries persisted in every trial. The complete interface produced
fewer living bodies than each restricted treatment in every environment, while
birth counts were mixed. Population is not individual fitness, and the evolving
trajectories diverge, but these comparisons establish no consistent community
benefit. Signal magnitudes averaged 0.0141–0.0315 across endpoints, including
the reception-disabled treatments. Nonzero emission is not evidence of useful
communication. See the [ancestry trajectories](docs/v14-coordination-communities.png),
[12 histories](docs/results/v14.json), and [audit](docs/results/v14-audit.json).
All founder, neural topology, development, and food-credit checks passed;
maximum absolute energy residual was 0.136 units.

`scripts/assay_coordination.py` now tests eight distinct two-module grazer
genotypes from each completed full-interface community. Selection is uniform
without replacement among eligible living genotypes, using a fixed selection
seed. Each genotype starts as a fresh juvenile in three new 128-unit habitats
for 240 seconds, paired across all four interventions plus `self_internal`.
Every genetic mutation pathway is disabled. Age, growth, intake, reproduction,
and death are recorded for the original body; clonal offspring may share its
dish. This can test use of the interface without ongoing genetic selection,
but related genotypes and repeated habitats are not independent evolutionary
replicates, and the assay habitat differs from the evolution dish. The script
retains exact source checkpoint hashes, selected genomes, package source,
experiment source, and completed trials if interrupted.

All 360 juvenile-start lifetimes completed and passed the
[paired audit](docs/results/v14-lifetime-audit.json). Within each genotype and
habitat, first-growth times matched exactly across interventions, as required:
the new inputs are all zero until a second module exists.

| Source environment | Bodies that grew, out of 24 per mode | Mean intake: full / no incoming signals / self-signals | Mean offspring: full / no incoming signals / self-signals |
| --- | --- | --- | --- |
| 241 | 3 | 463.91 / 463.86 / 464.40 | 0.167 / 0.167 / 0.167 |
| 242 | 4 | 439.42 / 441.83 / 441.95 | 0.042 / 0.083 / 0.083 |
| 243 | 20 | 1,008.83 / 1,003.21 / 1,005.22 | 1.458 / 1.458 / 1.458 |

Survivors were 22/19/24 for the three sources, identical across all five modes.
Relative to self-signals, neighbor reception changed mean intake by -0.49,
-2.53, and +3.61 units. It did not improve mean offspring count in any source.
Removing body-position readings increased mean intake in all three sources,
although individual genotype responses varied. These results show small, mixed
effects and no consistent reproductive benefit. The
[figure](docs/v14-coordination-lifetimes.png) and raw reports for
[241](docs/results/v14-lifetimes-241.json),
[242](docs/results/v14-lifetimes-242.json), and
[243](docs/results/v14-lifetimes-243.json) retain every treatment and early death.

The low frequency of second-module growth in two sources limits the assay's
exposure to the interface. A supplementary `--start-mature` comparison was started
using the same selected genotypes and habitats, beginning the original body
fully grown, with fresh neural state and the same birth energy per unit of
adult area. Its position reuses the normalized initial disk draw, scaled to
fit the larger body. Offspring still begin as juveniles. This deliberately
changes initial anatomy and energy across the two assay protocols, while
matching them across interventions within each protocol. It tests interface
use when available and cannot measure juvenile developmental success.

This follow-up was interrupted when the user requested the development wrap-up.
Sources 241/242/243 retain 43/46/40 completed individual lifetimes out of 120
planned per source. The incomplete reports are preserved for
[241](docs/results/v14-adult-partial-241.json),
[242](docs/results/v14-adult-partial-242.json), and
[243](docs/results/v14-adult-partial-243.json), with provenance and row checks in
the [stop record](docs/results/iteration-stop.json). No result from this partial
comparison is treated as a completed paired outcome. Selected genomes and
completed trials remain local; the assay currently requires a fresh output
directory and has no automatic resume mode.

`runs/v14-video/timelapse.mp4` records the full-interface pilot in environment
231 from 600 to 900 seconds. All 901 frames decoded at 1,024×1,024 and 30 FPS;
the [verification record](docs/results/v14-video.json) retains frame hashes.

## V15 — inherited local learning rules

The earlier recurrent plasticity mechanism always used the same correlation
rule; evolution changed its rate, decay, and neural modulation. V15 gives
evolution four signed coefficients controlling the rule itself. The family is
inspired by the parameterized heterosynaptic rules of
[Niv et al. (2002)](https://nivlab.princeton.edu/wp-content/uploads/sites/938/2024/02/nivetal2002.pdf).
Their bee-foraging study used a small specialized circuit and a genetic
algorithm selecting nectar intake. Here a shared recurrent body circuit
evolves through local energy-funded reproduction. This is an extension of
our mechanism, not a replication of their learning results.

### Encoding and update

The interface remains 40 inputs, seven outputs, and up to 32 recurrent neurons,
initially 16 active. Four genes follow the existing 13 traits, bringing the
genome to 5,144 values: 2,567 initial weights/biases, 17 trait/rule values, and
2,560 structural mask values. Unlike the sigmoid body allocations, the new
genes are signed. Decode their vector `g` as `g / max(1, sum(abs(g)))`, yielding
coefficients A, B, C, and D with an absolute-sum budget of one.

For a connection from the previous sender activity `x` to the updated receiver
activity `y`, the eligibility drive becomes:

```text
q = A * y * x + B * x + C * y + D
E = (1 - beta) * E + beta * q
P = clamp(decay * P + dt * inherited_rate * neural_modulation * E, -0.1, +0.1)
```

The preset retains the two-second trace time constant, inherited rate up to
0.02, inherited 30–600-second half-life, and signed fifth-output modulation.
The coefficient budget bounds the drive to [-1, 1] for bounded neuron activity;
it expands the rule family without increasing its maximum update scale.
Inactive connections are masked after the trace and offset updates. All-zero
coefficients add no new trace, while existing traces and offsets still decay.
Food and damage enter through existing sensory feedback; the rule receives no
hidden resource-quality label or externally supplied desired action.

Fresh founders and transfers start with A=1, B=C=D=0 under the standard weight
limit, exactly recovering the previous rule. V15 preserves the V14 founder
random draws for existing genes and structures. Ordinary trait mutations
(probability 0.15, Gaussian sigma 0.15, inherited gene bounds ±3) can change the
four rule genes. Children inherit those genes, with zero neural activity,
eligibility traces, acquired weights, and internal signals. Growing modules
also start with fresh state. The acquired offsets never rewrite the genome.

The `fixed_rule` control expresses A=1, B=C=D=0 regardless of the encoded
coefficients. Those ignored genes can still mutate, preserving the mutation
machinery. All other channels and cost rules remain active. The existing
`no_plasticity` control removes acquired offsets and traces while retaining
their metabolic cost. Motor exploration stays disabled following V12's
negative physical learning results. Records distinguish encoded and expressed
rule coefficients, and the inspector shows the expressed A/B/C/D values.

### Verification and initial experiment

All 205 tests pass. New checks cover signed bounded coefficients, previously
unavailable updates at silent connections, structural masks, fixed-rule
intervention, inheritance with fresh state, mutation of rule genes, exact
V14-to-V15 controller transfer, unchanged native initialization before mutation,
V15 checkpoint replay, and rendering without changing the simulation.

Mechanical [CPU](docs/results/v15-cpu-rules.json) and
[CUDA](docs/results/v15-cuda-rules.json) checks exercised varied signed rules,
four births each, and six/nine growth events. Both passed exact checkpoint
replay on their respective devices; energy residuals were -0.001045 and
-0.001187 units. Peak CUDA allocation was about 37.5 MB. These intentionally
funded, accelerated fixtures verify mechanics and do not demonstrate ecological
learning benefits.

Six initial native trials compared evolving rules and `fixed_rule` in
environments 261/262/263, with identical random founders and habitat within
each pair and ordinary evolution active. Five of six became extinct before
the requested 1,800-second horizon:

| Environment | Evolving rule: outcome / births | Fixed rule: outcome / births |
| --- | --- | --- |
| 261 | Extinct at 1,776.0 s / 8 | Extinct at 1,036.2 s / 5 |
| 262 | Extinct at 487.0 s / 2 | Extinct at 487.0 s / 2 |
| 263 | Extinct at 643.4 s / 7 | 2 survivors at 1,800 s / 17 |

These starts did not reliably establish populations under either rule. The
[population and rule histories](docs/v15-rules-native.png),
[six records](docs/results/v15-native.json), and
[audit](docs/results/v15-native-audit.json) retain these failures. Largest
absolute energy residual was 0.0362 units, with all invariant checks passing.
V13's successful starts used different seeds and a different neural interface,
so they do not provide a matched version comparison here.

Nine follow-up 3,600-second trials were planned to assemble the established V13 native
food-web pools used above, with 96 newborns from each pool. Environments
271/272/273 compare evolving rules, `fixed_rule`, and `no_plasticity`, with
identical initial genotypes and habitat within each environment. All ordinary
genetic mutations remain active. This measures the rules' effects on viable
inherited foraging populations; it does not rescue or replace the failed native
starts. Any claim of useful lifetime learning additionally requires
fixed-genotype interventions and behavioral evidence.

Only environment 271 completed all three treatments before the user requested
a stop. Its completed trials are separate from the incomplete three-seed batches:

| Treatment | Grazer-source / scavenger-source survivors | Births |
| --- | --- | --- |
| Evolving rule | 39 / 34 | 1,308 |
| Fixed correlation rule | 23 / 21 | 1,133 |
| Acquired plasticity disabled | 46 / 20 | 1,063 |

Both source ancestries persisted in each treatment. These are outcomes from one
matched environment, not evidence of a general advantage or useful learning.
The [three complete histories](docs/results/v15-assembled-271.json) and
[audit](docs/results/v15-assembled-271-audit.json) preserve the comparison.
Founder genomes, source provenance, and patch geometry matched exactly; topology,
development, energy, and food-credit checks passed. Maximum absolute energy
residual was 0.135 units. The audit's explicit `--completed-trials-only` option
checks finished trials while retaining `source_batch_completed=false`; its
default still rejects incomplete batches.

Environment 272 was interrupted in all three treatments; 273 had not started.
Last logged times were 984/1,184/1,198 seconds in the order above, but terminal
interruption did not write final checkpoints. Each has a valid archived checkpoint
at 600 seconds, verified against the corresponding logged metrics. Later logs are
preserved without being presented as saved final states. Exact paths, hashes,
and unfinished work are in the [stop record](docs/results/iteration-stop.json)
and [handoff](HANDOFF.md).

The [inspector preview](docs/v15-preview.png) shows body 577 from the evolving
rule treatment in environment 271 at 2,400 seconds. It has two modules and
coefficients approximately (0.616, -0.119, 0.108, 0.156). It was selected for
visible rule variation, not measured learning benefit. Its 12 offspring do
not isolate the contribution of that rule from its other inherited traits.

The verified recording `runs/v15-video/timelapse.mp4` follows that environment
from 2,400 to 2,700 seconds. All 901 frames decoded at 1,024×1,024 and 30 FPS;
the [video audit](docs/results/v15-video.json) retains frame hashes. This is an
illustrative checkpoint fork, not an additional independent comparison.

## Pygame observation interface — package 0.16.1

On September 23, 2026, the user requested fading paths and a live neural inspector,
and agreed to retain Pygame. This is a bounded interface iteration; the paused
ecology development loop has not resumed. Controller equations, mutation,
ecology parameters, and checkpoint formats are unchanged.

The viewer now samples centroid trails at 5 Hz of simulated time, retains at most
120 seconds, and offers 10/30/120-second fading windows. Trails preserve organism
identity across births, deaths, and array compaction; pause freezes fading.
Recording samples the same history independently of video frame rate. Histories
start at observation time and are not stored in checkpoints.

Selecting a creature opens exact per-module inputs, recurrent activations, and
outputs. Selecting a hidden neuron reveals its incoming input/recurrent links
and outgoing actuator links. Effective weights respect topology masks and both
learning mechanisms. A controller-boundary observer captures consumed feedback,
previous hidden state, and pre-update recurrent offsets; post-update motor offsets
and actual exploration noise complete the sampled decision. Rendering never
resenses the world or advances a controller. The panel also separates field-mean,
field-contrast, and recurrent drives, shows morphology/learning maps, and supports
module selection, follow-camera, and paused controller-interval stepping.

All 227 tests and Ruff checks pass. Twenty-two new tests cover sample
reconstruction, inspection purity under interventions, scripted controllers,
trail history, stable selection, and viewer controls. An actual V15 CLI run
combined three paused steps, 32x playback, and 61 recording frames over 60 physics
ticks; its complete state matched headless execution exactly. A separate native
CUDA V15 check also matched with shuffled sensing on the RTX 5080.
The [verification record](docs/results/viewer-validation.json) retains the scope.

The [interface preview](docs/viewer.png) follows body 577 from the existing V15
environment-271 checkpoint for 32 additional seconds, to t=2,432 seconds. Its
two modules have different live sensory values and neural states. This is an
illustrative fork for interface validation, not a new ecological experiment.
See the [README controls](README.md) for use and interpretation.

## Energy bars and an easier V15 reproduction preset

On September 23, the user requested a reproduction progress indicator, lower
reproduction thresholds/costs, and an HP-style energy bar. These are bounded
follow-ups; the autonomous development loop remains paused.

Every inspector tab now shows current / maximum stored energy and a separate
energy / birth-threshold bar. The storage bar turns amber below 50% and red below
25%. The birth status identifies unfinished development, cooldown, or a full
population. Its threshold scales with the current parent's area and can increase
on growth. "Ready to try" does not predict offspring mutations or available
placement: the simulation checks those at the real attempt. All values are read
from the current world without advancing physics or random streams.

Only two numeric settings in `configs/v15.toml` changed: the parent-area threshold
220 -> 150 and child-area debit 120 -> 110. Birth energy stays 100 per child area,
and neural construction is still charged in addition to the debit. Body development,
food, controller initialization, and inheritance are unchanged. The original file
was copied verbatim to `configs/v15-research.toml` for historical comparisons.
Existing checkpoints retain their saved settings; they are not migrated.

Six fresh CPU runs compared the presets at the same seeds, with a 600-second
horizon and early stopping at extinction. Founder genomes matched within each
pair. The original preset had 0/22/0 births and final populations 0/20/0 at seeds
1/2/3; the revised preset had 3/19/3 births and populations 3/11/1. First births
under the new settings occurred at 15.43/5.43/15.93 seconds. Seed 2 had fewer total
births despite its earlier first birth. This improves access to reproduction in
these starts, but the small, fragile populations do not establish long-term
viability. Full logs and checkpoints are under `runs/v15-birth-readiness-*`;
the [compact audit](docs/results/v15-birth-readiness.json) records the comparison.

All 233 tests and Ruff checks pass. Six new cases compare the readout with actual
birth gates across V0/V1/V8/V12/V15, distinguish storage capacity from the birth
threshold, and check growth, retry, and population blockers. Existing rendering
purity tests exercise both bars and still match complete unobserved trajectories.
All tabs fit at 640/768/984/1024 pixels, with no overflowing text. The new
[preview](docs/reproduction-readiness.png) shows fresh seed 1, creature 13, at
30.1 seconds; it is a separate UI check, not an additional ecology replicate.

## V16: irregular, drifting sources and local recovery — package 0.17.0

The next bounded user request adds three resource mechanisms and records simple
producers as a possible later step. The autonomous research loop remains paused.
The [design guide](docs/DYNAMIC_RESOURCES.md) specifies the rules and tuning values.

Each source uses an area-preserving ellipse and sinusoidal shear, with independently
sampled orientation and signed bend. Its center moves at 0.5 units/s, with unbiased
heading diffusion and reflections at an inset that keeps its entire shape inside
the world. Existing particles stay put; source stock still includes their remaining
food even after the source moves away. Forecast and identity fields move with the
sources; shelter and the fertility grid remain fixed in world coordinates.

The 64-by-64 production grid holds two energy units per square world unit, or 128
per default cell. Only accepted fresh-food particles spend reserve. Deficits recover
exponentially with a 120-second time constant. Proposals rejected by source caps or
insufficient fertility create no food and carry no budget into a later batch. The
production ledger is audited separately from edible and trophic energy. Drift,
shapes, fertility, counters, and the dedicated resource RNG are checkpointed.

The V16 preset preserves V15's easier reproduction, controller architecture, and
inheritance. Neutral resource settings match V15's trajectory in a test. A separate
archived-source check also matched a saved V15 world's complete continuation over
60 ticks, apart from the newly present inactive configuration defaults.

Fresh 600-second CPU starts at seeds 1/2/3 produced 13/30/12 births, ending with
6/4/5 creatures and living generations 2/6/4. The previous easier V15 preset gave
3/19/3 births and populations 3/11/1 at the same seeds. Founder genomes match.
These are combined-environment startup checks: actual food supply and the source
placement inset differ, so they do not isolate shape, drift, or depletion effects.
They establish neither reliable long-term persistence nor improved food tracking.
See the [audit](docs/results/v16-resources.json) for full scope and measurements.

All 253 tests passed, including 20 resource cases. Validation requires a cell's
capacity to exceed one particle because exponential recovery approaches its
maximum asymptotically. Ruff checks pass. CPU and RTX 5080 mechanics checks each passed
exact replay, including four births and six/ten growth events respectively.
The 120-second seed-1 recording contains 901 verified frames at 1,024 by 1,024,
30 FPS, and 4x playback. The source and fertility screenshots follow that run
to 150.1 seconds and are an illustration, not another ecological replicate.

**Possible later step:** plant-like producers could consume fertility to grow
edible biomass and spread locally, replacing some automatic spawning. Their
growth, offspring, and recycling should retain explicit resource costs. Producer
organisms are documented only and are not implemented in V16.

## Single energy bar with a birth-threshold marker

The inspector now combines stored energy and birth readiness into one bar. Fill
still represents energy / capacity; a gold vertical line marks the body's birth
threshold on that same scale. Current / maximum energy and the threshold value
appear above the bar, while growth, retry, and population-limit status remains
below. The Details legend and current viewer documentation describe the marker.

All 64 existing observation, runtime, field-storage, and resource tests pass,
including complete-trajectory rendering checks. Ruff checks pass. Every tab fits
at window heights 256, 640, 768, 984, and 1024; header font bounds do not overlap.
The [preview](docs/energy-threshold.png) uses the V16 seed-1 UI fork at 150.1 seconds,
creature 72, matching the earlier source preview. This is a display refinement;
simulation mechanics and the paused research loop are unchanged.

## Sortable living-creature leaderboard

The sidebar now switches between the inspector and a live leaderboard using `L`
or the panel buttons. Column headers sort lifetime, generation, cumulative food
energy absorbed, offspring, stored energy, distance traveled, and creature ID.
Clicking the active header reverses the direction; ties resolve by ID. Food is
absorbed energy from fresh food, detritus, and prey, excluding initial energy.
Existing counters also cover activity before a resumed checkpoint. The table
lists living creatures, so deaths remove rows.

Clicking a row selects its permanent ID, centers the camera, and opens the
inspector. A stale click on a creature that has died is ignored. Dish selection
also opens the inspector when the sidebar is visible. The table preserves its
sort and page while inspecting, clamps pages when the population shrinks, and
uses Previous/Next, Page Up/Down, or the wheel over the panel for navigation.
The wheel over the dish still zooms. Layout scales for small windows without a
scrollbar.

All 80 relevant observation, leaderboard, runtime, field-storage, and resource
tests pass. Sixteen new cases cover sorting, checkpoint counters, selection
after population compaction, stale clicks, empty populations, pagination, input
routing, and scaled controls. Existing complete-trajectory tests now draw the
leaderboard too, including shuffled sensing and V16 resource replay. Ruff checks
pass. Manual checks cover all panels at heights 256/640/768/984/1024.

The [preview](docs/leaderboard.png) uses the archived V16 seed-2 native-run
configuration at 80.1 simulated seconds, with 46 living creatures, seven births,
and creature 123 selected. It is a UI check, not an ecology comparison. Its
layout record is at `runs/leaderboard-preview/verification.json`. User edits to
`configs/v16.toml` are separate from this change. The research loop remains paused.

## Research resumed: sparse meals and V17 spatial sensing — package 0.18.0

The user authorized renewed experiments and clarified that the large V16 habitat,
valuable particles, and inexpensive movement were intended to reward food seeking.
Those uncommitted settings remain untouched. A separate baseline snapshot and
seven treatments cover 24 fresh starts with exactly matched inherited founders.
The [full record](docs/FORAGING.md) includes results, comparison confounds,
mechanical checks, source provenance, and reproducible plots.

The baseline went extinct in all three seeds. Raising raw food handling from 60
to 300 while retaining the sparse habitat produced 236/59/17 births by 600 seconds.
All three continuations reached 1,800 seconds, with populations 97/122/15 and
births 809/799/44. Seed 3 passed through a single-creature bottleneck before
recovering. Larger particles take longer to process under the old finite-contact
feeding rule; they do not automatically deliver a larger meal to a passing body.

V17 optionally replaces each four-receptor field group with a mean and three
spatial contrasts, with shared local normalization. No inherited weights or
movement policy are prescribed. Architecture and genotype size remain unchanged.
Three paired compact-habitat starts showed mixed effects: populations 55/7/4
versus 23/19/20 with absolute encoding. Retain this as an experiment, not a proven
improvement. Neutral V17 matched every recorded physical measurement of the V16
compact control. All 281 tests passed and CPU/CUDA replay exercises passed with
births, growth, variable plasticity rules, and resource changes.

Current next steps are matched sensory interventions on evolved fast-feeding
communities and bounded carried food, separating collection from digestion while
retaining finite processing budgets and producer/consumer energy accounting.

## V18: carried food and internal fullness — package 0.19.0

V18 separates collection from digestion with two bounded compartments. Food
retains its expiry, source stock, and producer credits; on death it drops at the
carrier's current position. Carried food produces no external scent. Processing
retains the existing finite budgets and energy storage limits, and recycled
material is deposited during travel. Two fullness inputs raise the interface
to 42 inputs and the genome to 5,272 values; no movement policy is supplied.

Twelve matched starts cross capacity 0/200 with raw processing rate 60/300. At
rate 60, carrying yielded 27/16/8 births and populations 16/6/2 at 600 seconds;
the contact controls all went extinct with 0/1/0 births. At rate 300, carrying
yielded 411/302/113 births and populations 82/108/79, versus 97/37/32 births and
47/1/15 creatures with contact feeding. All four treatments have exactly matched
founder genomes within seeds. These V18 founders differ from V16/V17 because
of the two additional inputs. Detailed accounting and limitations are in
[CARRIED_FOOD.md](docs/CARRIED_FOOD.md).

The rate-60 carrying extensions reached populations 94/16 at 1,800 seconds for
seeds 1/2; seed 3 became extinct at 834.4 seconds. Fast-carrying extensions were
started for all three seeds. A short video fork is recorded at 4× with trails;
the selected creature inspector shows the inventory and both new neural inputs.
CPU and RTX 5080 exercises pass exact replay through collection, digestion,
births, growth, and changing plasticity rules. The three inspector tabs fit
640–1,024-pixel heights. Parsed-config comparison now accepts newly explicit
neutral defaults in resumed histories while rejecting changed world settings.

Sixteen completed transplants of two V16 fast-feeding communities show some
dependence on environmental signals but weak/mixed effects from rotating them.
They do not establish directed food seeking or within-lifetime learning. The
user reinforced interest in greater neural/mutation variation and learning.
Prioritize controlled circuit-diversity, mutation, and conservative motor-learning
experiments in the more viable carrying environment next.

The three fast-carrying continuations subsequently completed 1,800 seconds:
populations 57/254/219, births 1,521/2,307/1,218, and maximum living generations
48/59/23. The audit now includes all twelve pilots and six continuations. All
299 tests and Ruff checks passed at the V18 commit `cd95624`.
