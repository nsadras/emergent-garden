# Experiment overview

This overview preserves the detailed project history formerly on the front page.
Research is **paused at the user's request**. For the latest state and completed
experiments, see the [handoff](HANDOFF.md); for setup and controls, see the
[usage guide](../docs/USAGE.md). Earlier run paths below identify local artifacts,
not files bundled with the repository.

A continuous 2D artificial-life world with evolving neural controllers, resource
cycling, predation, inherited bodies, and persistent chemical trails. Creatures
spend energy, reproduce, and mutate. Reproduction uses locally acquired energy;
there is no global parent ranking. V12 additionally uses each creature's net
energy flow for within-lifetime motor reinforcement. V0 and every subsequent
experimental preset remain available.

The current experimental implementation is **V25 / package 0.26.0**. The
September 23 research continuation focused on scarce, valuable food and affordable
exploration.
The [foraging experiments](FORAGING.md) found that faster food processing
supported all three tested populations for 30 minutes; a new optional spatial
sensory encoding had mixed results. [V18](CARRIED_FOOD.md) adds finite food
carrying and two fullness inputs so creatures can digest while moving. All three
fast-carrying starts reached 30 minutes. [V19](NEURAL_VARIATION.md) adds
varied founder circuits and compares stronger mutation and gentle motor
learning. Initial outcomes are mixed. Paired descendant tests show conditional
benefits from motor updates; shuffled-return controls remain mixed.
[V20](SENSOR_RADIUS.md) tests a wider sensory footprint: directional signals
increase, while reproduction remains dependent on the start.
[V21](PERSISTENT_EXPLORATION.md) tests two-second exploratory motor changes
with a matching conditional learning rule. Twelve paired pilots show mixed
ecological effects; persistent noise alone reduces births in all three starts.
The movement diagnostic shows changed routes, without a consistent increase in
net travel. All six longer runs and 24 learning transplants are complete;
matched descendant tests show mixed effects and no reliable learning advantage.
[V22](VALUE_PREDICTION.md) adds learned predictions of later energetic
returns. All 15 pilots and six prospective forecast checks are complete: neither
prediction target establishes a reliable ecological improvement, and forecast
quality remains weak. [V23](SENSORY_VALUE.md) gives that predictor direct
access to existing sensations. On identical experience it modestly improves
forecast error in three communities, without a reliable ecological benefit.
A faster prediction-learning rate increases error. All nine ecological pilots
and the paired prediction diagnostics are complete.
An optional [two-second prediction horizon](PREDICTION_HORIZONS.md) improves
local forecasts but still gives mixed ecological results and weak learning evidence.
[V24](NEURAL_TIMESCALES.md) gives individual recurrent neurons inherited
response times. Matched pilots show mixed ecological effects: varied timing helps
two of three starts, direct timing mutation helps one. A separate circuit probe
confirms changed responses without demonstrating useful memory. Twelve pilots
and six exact neutral comparisons are complete. All nine longer continuations
survived: inherited timing increased births in all three starts, with limited
lineage and dietary-specialist diversity. All thirty
[controlled genotype/feedback trials](NEURAL_TIMING_ASSAYS.md) are complete:
original timing beats reassignment for births in all six comparisons, while
motor-learning evidence remains conditional. A verified 1× video and trail
image are linked from that report. Details shows the selected neuron's response time.
[V25](RECURRENT_LEARNING.md) adds optional energetic reinforcement of
recurrent connections. The native controller learns a supplied delayed-cue task;
twelve matched ecological pilots show mixed results and fail the continuation
screen. It remains optional; inherited timing with the new learner disabled
is retained as the reference.
Nine quieter-noise trials and six subsequent gentler-rate trials also fail
the [learning screen](QUIET_RECURRENT_LEARNING.md). This rate/noise sweep
is complete. A [passive prediction experiment](REPRESENTATION_LEARNING_PLAN.md)
separates learning a useful sensory representation from changing the creature's
movement; its checks pass, but the completed comparison fails its predefined
continuation screen.
The neural interface, inherited genome, and sparse-food ecology stay the same.
The live inspector, leaderboard, and energy
bar remain available. See [HANDOFF.md](HANDOFF.md) for current evidence and runs.

![V5 modular creatures and their chemical trails](../docs/v5-detail.png)

The five-version development record, including failed hypotheses and controlled
comparisons, is in [EVOLUTION.md](EVOLUTION.md). Persistence and sensory effects
have been observed; useful forecast memory, communication, and open-ended
intelligence have not been established.

Continuing experiments on learning and evolving neural architecture are recorded
in [CONTINUATION.md](CONTINUATION.md). The experimental [V6 preset](../configs/v6.toml)
adds patch identities with changing nutritional value and food/damage feedback.
The [V7 preset](../configs/v7.toml) adds heritable plasticity rules: each body module
can alter its recurrent connections during life, and offspring inherit the rule
with fresh synaptic state. These mechanisms are implemented; useful association
learning has not yet emerged in the measured populations.

To inspect V7 locally, resume `runs/v7-inherited-plastic/seed-81/latest.pt` with
`--view`, or start `uv run garden run --config configs/v7.toml --seed 71 --view
--device cpu --seconds 0`. [The inspector](../docs/v7-preview.png) shows synaptic
change and the modulation gate. A verified 30-second recording is at
`runs/v7-video/timelapse.mp4`. The original wider plasticity range remains in
[v7-wide.toml](../configs/v7-wide.toml); every checkpoint retains its own settings.

The experimental [V8 preset](../configs/v8.toml) permits inherited neuron and
connection changes, with 16 initially active recurrent units in a 32-slot
template. Extra neurons and connections incur construction and maintenance
costs. [Its four-way comparisons](../docs/v8-topology.png) found modest structural
variation and mixed ecological effects. The [brain inspector](../docs/v8-preview.png)
shows the expressed circuit and acquired changes. A V8 video is saved locally
at `runs/v8-video/timelapse.mp4`.

The [V9 preset](../configs/v9.toml) traces detritus producers and consumers and
predation transfers, while preserving V8's physics. [Community assembly trials](../docs/v8-assembly.png)
found that two evolved lineages persisted together when attacks were disabled;
with attacks active, grazer ancestry disappeared in all three tested mixtures.
The accounting will guide experiments on sustaining richer food webs.

The [V10 preset](../configs/v10.toml) adds local shelter: covered patches reduce
attacks into and out of cover, and creatures gain four directional shelter
readings. [The overlay](../docs/v10-preview.png) shows this terrain. Controlled
[trials](../docs/v10-shelter.png) found mixed effects on extinction timing and no
sustained coexistence with the initial shelter settings. A verified recording
is at `runs/v10-video/timelapse.mp4`. Covering every patch also failed to preserve
both ancestries in three follow-up environments.

The experimental [V11 preset](../configs/v11.toml) gives creatures finite fresh-food
and detritus processing capacities, determined by their inherited bodies and diet.
This closes the route by which poor fresh-food assimilators could rapidly turn
unlimited fresh food into their own preferred detritus. The
[inspector](../docs/v11-preview.png) shows those capacities. In
[matched food-web tests](../docs/v11-handling.png), the tested scavengers became
dependent on recycled food produced largely by grazers. Both ancestries persisted
for an hour in two of three mixtures under each processing treatment, so an
improvement in coexistence frequency is unestablished. Random-founder pilots
have not established reliable populations.
A verified 30-second recording is at `runs/v11-video/timelapse.mp4`.

The experimental [V12 preset](../configs/v12.toml) adds heritable exploration and
learning rates, with acquired motor readouts that respond to food, costs, and
damage. [The inspector](../docs/v12-preview.png) shows these offsets. The rule can
[learn and reverse a constructed cue–action task](../docs/v12-rule.png), but
[physical lifetime assays](../docs/v12-outcomes-lifetimes.png) found worse average
intake and reproduction with the tested learning settings. The
[community comparisons](../docs/v12-outcomes-communities.png) also found no consistent
benefit. The first, less constrained version
disrupted inherited foraging; it remains in [v12-wide.toml](../configs/v12-wide.toml).
A verified video is at `runs/v12-video/timelapse.mp4`.

The experimental [V13 preset](../configs/v13.toml) expresses inherited body plans
through juvenile growth. Offspring start with one module, pay to grow additional
encoded modules when energy and space permit, and mature before reproducing.
New modules receive fresh neural state. This tests a measured barrier in the
older birth law: single-module parents often could not finance a larger child
at once. Its preset disables exploratory motor noise following the V12 results;
recurrent plasticity and evolving neural architecture remain active.
In three short matched pilots, two-module parents reproduced in two juvenile
treatments, while fully formed births produced no larger children. The
[adult](../docs/v13-preview.png) and [juvenile](../docs/v13-juvenile.png) previews show
one parent and its offspring; a verified recording is at
`runs/v13-video/timelapse.mp4`. [Three one-hour comparisons](../docs/v13-development.png)
retained larger bodies in every juvenile treatment and none in the fully formed
offspring controls. Three-module bodies also reproduced. Population and birth
counts did not consistently improve. [Fresh random populations](../docs/v13-development-native.png)
persisted for 30 minutes under both offspring rules; their founders already
included larger body plans. [All three native populations](../docs/v13-development-long.png)
also persisted for three simulated hours, ending with 58, 73, and 28 creatures.
Larger bodies remained common in two populations and became rare in the third.
One population retained both fresh-food specialists and scavengers descended
from differently allocated founders. [Twelve transplantation controls](../docs/v13-native-foodweb.png)
support scavenger dependence on grazer-produced detritus in that selected
community. Its verified recording is at
`runs/v13-native-long-video/timelapse.mp4`.

The experimental [V14 preset](../configs/v14.toml) adds private signals between
adjacent modules within a body, plus each module's body-relative coordinates.
Its shared circuit has 40 sensory inputs and seven outputs, including two signed
internal emissions. Signals incur energy costs and reach neighboring controllers
on their next update. [The inspector](../docs/v14-preview.png) shows the held signals.
Matched tests remove position readings, signal reception, or both; a self-signal
control distinguishes extra local memory from information sharing. Short pilots
and [twelve hour-long comparisons](../docs/v14-coordination-communities.png)
showed no consistent population or reproduction benefit. A further
[360 fixed-genotype lifetimes](../docs/v14-coordination-lifetimes.png) found small,
mixed effects and no consistent reproductive benefit. Many founders remained
juveniles. A supplementary adult-start assay was interrupted at wrap-up, with
129 of 360 individual lifetimes retained; its comparison remains incomplete.
A verified recording is at `runs/v14-video/timelapse.mp4`.

The experimental [V15 preset](../configs/v15.toml) lets evolution alter the local
plasticity rule itself through four inherited signed coefficients. They control
responses to joint activity, activity on either side of a connection, and a
constant term. Updates remain bounded, offspring start with fresh neural state,
and a fixed-rule control preserves the earlier mechanism. The
[inspector](../docs/v15-preview.png) shows the coefficients. Mechanical and replay
checks pass. [Native starts](../docs/v15-rules-native.png) established poorly under
both rules. One matched environment completed all three comparisons using
established food-web genomes; the remaining environments are unfinished.
A learning benefit has not been established. A verified 30-second recording
is at `runs/v15-video/timelapse.mp4`.

The V15 live preset now makes reproduction easier: `reproduction_threshold`
is **150** (previously 220) per unit of parent body area, and
`reproduction_debit` is **110** (previously 120) per unit of child body area,
plus the child's neural construction cost. Newborn energy remains **100** per
unit of child area. Three fresh CPU starts produced births in 600-second checks,
but two ended with only one or three creatures; sustained establishment is
still unresolved. See the [comparison](../docs/results/v15-birth-readiness.json).
The original settings used for the earlier V15 research are preserved in
[v15-research.toml](../configs/v15-research.toml). Resuming a checkpoint keeps its
saved settings; start a fresh run to use the revised preset:

```bash
uv run garden run --config configs/v15.toml --view --device cpu --seconds 0
```

## Historical viewing examples

These descriptions record the earlier presets; current TOML files are the source
of truth for settings. Referenced checkpoints are local research artifacts.

These presets start 192 juvenile creatures, with a capacity of 1,024. The original
V13–V16 presets use a 512-unit dish; the new sparse-food experiments use 1,024.
Each inherited body plan encodes one to three modules, which grow as food and
space permit. Modules have local sensors, propulsion, and a shared recurrent
template with up to 32 neurons, initially 16 active. Each module has its own
acquired neural state. A circular membrane
defines collision geometry; the inner
modules collect food. Zoom in to inspect the body, actuators, and inherited traits.
Green bodies favor fresh food; amber bodies favor detritus. Red marks show attacks.

For the earlier V13 environment, use `--config configs/v13.toml --seed 222`.
That starting point persisted for three simulated
hours, ending with 73 creatures, two dietary groups, and living generation 30.
Survival is not guaranteed, and there is no automatic reseeding. To watch the
already evolved population, resume `runs/v13-native-long-222/latest.pt`.
V7 seed 71 also has a verified three-hour population at
`runs/v7-bounded-long-71/latest.pt`. The earlier presets remain available;
see [their experiment record](EVOLUTION.md).
