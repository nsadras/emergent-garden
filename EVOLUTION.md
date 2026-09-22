# Five-version development experiment

V0 baseline: commit `84e0c48`. Python dependencies remain managed by uv.

The objective is a persistent artificial ecology with inherited differences,
organism-mediated resource flows, and experimentally demonstrated uses of
information. Open-ended evolution, learning, intelligence, cooperation, and
species formation are hypotheses, not labels assigned from appearance.

## Adaptive sequence

1. V1: fixed recoverable food patches, detritus, inherited size/power/diet.
2. V2: costly contact predation and defense; inspect stability and feeding roles.
3. V3: predictive patch cues and matched sensory/memory interventions.
4. V4: a bounded developmental genome for repeated body modules and interfaces.
5. V5: environmental signals if supported by the ecology; longer integrated runs.

Each version receives mechanical tests, independent evolutionary seeds, a
recorded inspection, a written result, and a git commit. Negative results can
redirect the sequence. Existing V0 runs and defaults remain available.

## V1 design

Fresh food has a patch identifier. Each fixed patch caps its standing energy;
external supply fills vacancies at a finite rate. Unused supply is not injected.
Eating fresh food converts a fraction to detritus at the feeding location.
Digestion dissipates energy; detritus cannot recursively generate more detritus.
Unassimilated detritus energy and expired resources leave the system.

Three inherited genes, additional to neural weights, control size, propulsion
capacity, and digestion allocation. Maintenance, storage, birth investment, and
motion costs depend on the phenotype. Fresh/detritus enzymes share a budget;
specialists digest their chosen food more efficiently. Both resources have four
local chemical readings. Controllers receive energy and touch as before.

The ecological presets use a 768-unit dish, a 128-square chemical grid, 30 Hz
physics, 10 Hz brains, and 5 Hz fields to make repeated longer experiments
practical. V0's resolution and rates remain unchanged. These are abstract
overdamped circles; smaller timesteps can be checked independently.

All energy enters through founder stores and recorded fresh-food injection.
All transfers, digestive losses, maintenance, movement, birth overhead, expiry,
and living/resource stores participate in the energy ledger. Population caps
prevent births without charging parents; extinct populations are not reseeded.

## Research context

- [Taylor (2015), Requirements for Open-Ended Evolution](https://www.tim-taylor.com/papers/taylor2015requirements.web.html):
  motivates environmental feedback and viable pathways between phenotypes.
- [Sims (1994), Evolving Virtual Creatures](https://www.karlsims.com/papers/siggraph94.pdf):
  motivates developmental body/controller encodings; its task-based selection
  differs from this continuously reproducing ecology.

## Results

Experiments and decisions are appended as each version is evaluated. Raw run
directories are local and ignored by git; compact reports are committed.

### V1 — resource cycle and inherited traits

Implemented and verified with 38 tests, including V0 regressions, conservative
feeding, variable-radius contacts, birth investment, delayed detritus,
checkpoint replay, and population-assay setup. Lint and format checks pass.

The first three 600-second pilots (seeds 11–13) persisted, but detritus stores
were very small and one seed retained only one scavenger-class individual.
An explicit **3-second maturation delay** was introduced. Food discarded during
feeding becomes edible and emits its chemical cue after that delay. This leaves
more material accessible to later visitors. The original experiment is retained.

The revised preset ran for 1,200 simulated seconds per independent seed:

| Seed | Population | Births | Maximum generation | Fresh specialists | Detritus specialists | Generalists |
|---|---:|---:|---:|---:|---:|---:|
| 11 | 146 | 440 | 7 | 96 | 18 | 32 |
| 12 | 141 | 442 | 9 | 32 | 91 | 18 |
| 13 | 214 | 553 | 10 | 202 | 8 | 4 |

These categories use inherited allocation thresholds 0.65/0.35; they are **not
species**, nor proof of exclusive feeding behavior. Detritus stores were
1,267–1,911 energy units. Absolute ledger residuals were below 0.022 units.
Each run reached its full duration without reseeding or capacity saturation.
Finite-duration coexistence is observed; long-term stable coexistence is unknown.

A held-out community assay of the initial pilot's descendants across three new
environment seeds found final populations 160–165 with ordinary sensing,
168–177 with smell removed, and 131–147 without recycling. Founder communities
ended at 112–142. Thus recycling supported population persistence in these
trials, while a useful sensory advantage remains unestablished. Founder and
descendant communities have different inherited bodies and energy endowments;
this is a community comparison, not a clean individual-fitness estimate.

Artifacts: `runs/v1-pilot`, `runs/v1-maturation`, `runs/v1-assay`, and the
30-second MP4 in `runs/v1-mature-video`. The MP4 decoder and image inspection were
checked. [Compact numerical evidence](docs/results/v1.json) includes source
hashes, seeds, final metrics, and sampled trajectories. A further assay of the
revised preset is retained under `runs/v1-mature-assay`.

Decision: retain the maturation delay and proceed to costly contact predation.
Do not claim evolved chemotaxis or memory from these trajectories.

The revised V1 assay (three held-out seeds, 180 seconds) ended with 153/172/169
descendants under normal sensing, 152/170/163 without smell, and 127/139/151
without recycling. Birth counts were higher without smell in all three pairs.
This reinforces the need to distinguish population persistence from sensory use.

### V2 design — costly predation and defense

A third chemical channel indicates local organism density. A neural output
controls a forward-facing bite within three units of the body surface. Weapon
investment reduces particle digestion and adds maintenance; armor reduces damage
but adds maintenance and movement drag. Attack attempts also cost energy.
There are no protected lineages or predefined predator/prey classes.

All attacks in a tick are settled simultaneously. Multiple attackers share a
prey's available energy; storage caps limit assimilation; remaining energy is
dissipated. Predation transfers energy rather than generating food. Reproduction
investment prevents profitable perpetual parent/offspring energy cycles.

The `no_attacks` assay suppresses transfers while retaining the same weapon and
attempt costs. It measures the ecological effect of attacks under fixed traits,
not the optimal ecology of a newly evolved peaceful population.

[Wagner et al.](https://arxiv.org/abs/1310.1369) observed digital predator/prey
coevolution exploring alternative behavioral strategies, while emphasizing that
ever-increasing trait complexity is not inevitable. This motivates the experiment
without predicting that our much smaller system will reproduce their findings.

V2 passes 42 tests. New checks cover simultaneous attacks on shared prey, storage
limits, armor costs, attack suppression, and exact V2 checkpoint replay.

| Seed | Seconds | Population | Births | Max generation | Predation deaths | Meat-eating survivors |
|---|---:|---:|---:|---:|---:|---:|
| 11 | 1200 | 66 | 208 | 6 | 67 | 21 |
| 12 | 1200 | 88 | 339 | 9 | 148 | 40 |
| 13 | 1200 | 80 | 338 | 11 | 69 | 10 |

Meat-eating survivors have acquired >20 total energy and >20% of their intake
from predation during their own lives. This is an observed feeding category,
not an inherited species label. All populations persisted; absolute energy
residuals were below 0.016 units. No parameter change was required for viability.

Matched community assays on seed-12 descendants (two new seeds, 180 seconds)
ended at populations 104/94 with attacks and 112/137 without attacks. Births
were 82/72 versus 80/98. Attacks materially alter the ecology; these observations
do not establish evolved pursuit, escape, or an escalating arms race.

Artifacts: `runs/v2-pilot`, `runs/v2-assay`, `runs/v2-video` (30-second MP4).
[Compact evidence](docs/results/v2.json). Decision: retain predation at its tested
costs, then introduce forecast cues and direct tests of history dependence.

### V3 design — forecast cues and history probes

Patch supply now bursts for 10 seconds of each 60-second cycle, with 15% of the
mean supply rate between bursts. Burst amplitude preserves mean *offered*
supply; actual injected food still respects patch caps. Patch phases are seeded
independently. A distinct chemical cue lasts 2 seconds and ends 3 seconds before
the next burst. A sixth trait controls the recurrent integration time constant
between 0.2 and 5 seconds (with gene bounds narrowing the reachable interval).

The first pilot retained the old 180-second food lifetime. That allows food to
outlast three cycles and can leave an immediate food cue throughout the waiting
period. A second experiment shortens fresh-food lifetime to **18 seconds**, while
retaining the other settings. Both experiments are preserved. Detritus retains
its separate 240-second lifetime and maturation delay.

`garden probe RUN --output FILE.json` supplies paired left/right forecast-cue
histories to the same sampled controller and then identical observations for
1, 3, and 6 seconds. It reports motor history effects and whether their sign
points toward the past cue, with state-reset controls. This is an isolated
controller probe, not a claim of successful navigation. Community assays add
`no_cue` and `memory_reset` treatments. State resets alter ordinary dynamics as
well as stored information; performance changes alone cannot prove useful memory.

[Zou et al.](https://arxiv.org/abs/2102.12638) evolved recurrent controllers in an
explicit spatial/working-memory maze task. Their result motivates testing history
directly, but differs from our ecology and does not imply spontaneous learning.

V3 passes 47 tests, including forecast timing, mean offered resource supply,
selective cue removal, history/reset controls, and full checkpoint replay.

| Seed | Seconds | Population | Births | Max generation | Mean neural time constant |
|---|---:|---:|---:|---:|---:|
| 11 | 1800 | 40 | 158 | 11 | 1.03 s |
| 12 | 1800 | 44 | 181 | 6 | 2.38 s |
| 13 | 1800 | 29 | 149 | 8 | 1.87 s |

These are the revised 18-second fresh-food-lifetime runs. All persisted;
absolute energy residuals were below 0.033 units. The original long-food pilots
also persisted, at populations 92/71/79. The shorter lifetime produces a much
sparser ecology; it is retained as an explicit temporal-pressure experiment.

Seed-11 descendant community assays (three fresh environments, 180 seconds):

| Treatment | Final populations | Births |
|---|---|---|
| Normal | 40 / 38 / 37 | 18 / 9 / 21 |
| Forecast cue removed | 36 / 45 / 38 | 13 / 14 / 27 |
| Recurrent state reset each update | 3 / 3 / 4 | 1 / 2 / 2 |
| All chemical sensing removed | 28 / 23 / 23 | 12 / 6 / 16 |

There is a sensory benefit in this sample and a strong dependence on recurrent
dynamics. **Useful forecast memory is not established**: removing the forecast
cue has mixed effects. The isolated probe detects a mean absolute turn difference
of 0.0274 after a 3-second gap, but only 1/39 sampled descendants turns in the
cue-aligned direction. All reset controls have zero history effect. Intrinsic
dynamics can support circling or other movement patterns without adaptive recall.

Artifacts: `runs/v3-pilot`, `runs/v3-short-food`, `runs/v3-assay`, `runs/v3-video`.
[Compact trajectories](docs/results/v3.json) and independent-run probes
([11](docs/results/v3-probe-11.json), [12](docs/results/v3-probe-12.json),
[13](docs/results/v3-probe-13.json)) retain the numerical evidence. Additional
community assays of evolutionary seeds 12/13 test whether the sensory effect
generalizes; their results will be included in the next checkpoint report.

Decision: preserve the controls and negative forecast result. Proceed to bounded
developmental morphology to expand the ways creatures can sense and act.

The additional V3 assays generalized the chemical-input effect: across all
seven matched environment pairs from three independent evolutionary runs,
normal descendant communities ended with more survivors than communities with
all chemical inputs zeroed. Seed-12 pairs were 51/39 versus 36/30; seed-13 pairs
were 30/32 versus 14/17. Forecast removal still had mixed effects. State resets
left only 1–4 survivors in these additional trials. This supports a role for
chemical input and recurrent dynamics, but does not isolate spatial gradient
following from intensity responses, or useful memory from motor dynamics.

### V4 design — bounded developmental bodies

Three additional genes specify module count (1–3), spacing, and the axis of
placement. A decoder repeats a template containing four chemical sensors, two
propulsion actuators, a digestive surface, and a copy of the inherited recurrent
circuit. Circuit **parameters are shared**, but each module has its own state
and local measurements. Child neural states start empty.

Modules live within a circular exclusion membrane. This deliberately retains
simple collision physics; the visible membrane is the physical collision shape.
Feeding requires contact with a digestive module, rather than any point inside
the membrane. Off-center thrust produces torque. Tissue, membrane, extra neural
circuits, weapon capacity, and active propulsion all have costs; birth investment
scales with the developed body. More modules increase total bite capacity and
its associated costs. Copy-number mutations can fail to produce a child when
the parent cannot afford the body, without charging a failed birth.

This is a small developmental grammar with nine trait genes plus the shared
brain template. It is **not arbitrary topology, articulated mechanics, or an
unbounded genome**. It expands inherited physical organization in a testable way.

The first 18-second-food pilot produced very sparse populations. The next
experiment extends fresh-food lifetime to **45 seconds**: the last food from a
10-second burst expires before the next 60-second cycle, leaving an opportunity
for the forecast cue. Other parameters remain unchanged.

V4 passes 57 tests. Developmental tests cover bounded assembly, tissue costs,
local sensing and separate neural states, off-center torque, digestive contact,
heritable structure with fresh child states, observation purity, checkpoint
replay, and neutral extension of existing neural circuits during version transfer.

| Random-founder seed | Food lifetime | Seconds | Population | Births | Max generation | Modules 1 / 2 / 3 |
|---|---:|---:|---:|---:|---:|---|
| 11 | 18 s | 1800 | 5 | 40 | 5 | 0 / 5 / 0 |
| 12 | 18 s | 1800 | 5 | 20 | 2 | 5 / 0 / 0 |
| 13 | 18 s | 1800 | 63 | 301 | 10 | 63 / 0 / 0 |
| 11 | 45 s | 2400 | 9 | 89 | 11 | 0 / 9 / 0 |
| 12 | 45 s | 2400 | 33 | 148 | 7 | 33 / 0 / 0 |
| 13 | 45 s | 2400 | 60 | 405 | 13 | 60 / 0 / 0 |

These results show viable repeated bodies in one run, but **do not show an
advantage for greater morphological complexity**. Three-module bodies disappeared
in these experiments. Absolute energy residuals remained below 0.025 units.

An explicit `--seed-from RUN` facility transfers living genomes into a fresh
world. Existing neural weights and trait meanings are preserved. New sensory
weights start at zero, new effector biases at -2, and a new module-count gene
starts with one module near a duplication boundary. New bodies receive their
ordinary founder energy and empty neural state; this is not checkpoint resume.
Source run and source IDs are recorded. Hidden sizes must match; backward
version transfers are rejected.

V4 seeded from V3 evolutionary seed 11 reached population 59, 498 births, and
generation 16 at 2,400 seconds in its first new environment (seed 21). All living
bodies had one module. Two further environments are being evaluated. These
experiments share an evolved source population and are not independent origins
of adaptation. A `pooled` assay removes the modules' distinct local readings
while preserving body geometry and costs. A `rotated` sensory control reverses
directional channels while preserving their instantaneous mean intensity.

Artifacts: `runs/v4-pilot`, `runs/v4-recovery`, `runs/v4-seeded`, `runs/v4-body-assay`,
and `runs/v4-video` (30-second recording and zoomed inspection).
[Compact evidence](docs/results/v4.json).

Decision: retain the 45-second preset and explicit genotype transfer, keep the
cost of extra body structure, and test persistent chemical trails in V5. Greater
body complexity remains an evolutionary opportunity, not a rewarded objective.
