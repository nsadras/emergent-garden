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
A second control with both qualities set to 0.625 is being evaluated; it matches
the mean digestibility of equally encountered patch identities but does not force
equal realized intake or occupancy.

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
proceed to bounded synaptic plasticity. Uniform-nutrition controls continue in
parallel. Current phase: preparing the V7 learning mechanism.
