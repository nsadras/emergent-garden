# Emergent Garden

A continuous 2D artificial-life world with evolving neural controllers, resource
cycling, predation, inherited bodies, and persistent chemical trails. Creatures
spend energy, reproduce, and mutate. Reproduction uses locally acquired energy;
there is no global parent ranking. V12 additionally uses each creature's net
energy flow for within-lifetime motor reinforcement. V0 and every subsequent
experimental preset remain available.

The current release is **V16 / package 0.17.0**. Bounded follow-ups add the live
inspector, a sortable creature leaderboard, an energy bar with a birth-threshold
marker, and dynamic food sources.
The autonomous research loop remains paused after the V15 wrap-up. See
[HANDOFF.md](HANDOFF.md) for completed findings, interrupted experiments,
verification, and commands to continue later.

![V5 modular creatures and their chemical trails](docs/v5-detail.png)

The five-version development record, including failed hypotheses and controlled
comparisons, is in [EVOLUTION.md](EVOLUTION.md). Persistence and sensory effects
have been observed; useful forecast memory, communication, and open-ended
intelligence have not been established.

Continuing experiments on learning and evolving neural architecture are recorded
in [CONTINUATION.md](CONTINUATION.md). The experimental [V6 preset](configs/v6.toml)
adds patch identities with changing nutritional value and food/damage feedback.
The [V7 preset](configs/v7.toml) adds heritable plasticity rules: each body module
can alter its recurrent connections during life, and offspring inherit the rule
with fresh synaptic state. These mechanisms are implemented; useful association
learning has not yet emerged in the measured populations.

To inspect V7 locally, resume `runs/v7-inherited-plastic/seed-81/latest.pt` with
`--view`, or start `uv run garden run --config configs/v7.toml --seed 71 --view
--device cpu --seconds 0`. [The inspector](docs/v7-preview.png) shows synaptic
change and the modulation gate. A verified 30-second recording is at
`runs/v7-video/timelapse.mp4`. The original wider plasticity range remains in
[v7-wide.toml](configs/v7-wide.toml); every checkpoint retains its own settings.

The experimental [V8 preset](configs/v8.toml) permits inherited neuron and
connection changes, with 16 initially active recurrent units in a 32-slot
template. Extra neurons and connections incur construction and maintenance
costs. [Its four-way comparisons](docs/v8-topology.png) found modest structural
variation and mixed ecological effects. The [brain inspector](docs/v8-preview.png)
shows the expressed circuit and acquired changes. A V8 video is saved locally
at `runs/v8-video/timelapse.mp4`.

The [V9 preset](configs/v9.toml) traces detritus producers and consumers and
predation transfers, while preserving V8's physics. [Community assembly trials](docs/v8-assembly.png)
found that two evolved lineages persisted together when attacks were disabled;
with attacks active, grazer ancestry disappeared in all three tested mixtures.
The accounting will guide experiments on sustaining richer food webs.

The [V10 preset](configs/v10.toml) adds local shelter: covered patches reduce
attacks into and out of cover, and creatures gain four directional shelter
readings. [The overlay](docs/v10-preview.png) shows this terrain. Controlled
[trials](docs/v10-shelter.png) found mixed effects on extinction timing and no
sustained coexistence with the initial shelter settings. A verified recording
is at `runs/v10-video/timelapse.mp4`. Covering every patch also failed to preserve
both ancestries in three follow-up environments.

The experimental [V11 preset](configs/v11.toml) gives creatures finite fresh-food
and detritus processing capacities, determined by their inherited bodies and diet.
This closes the route by which poor fresh-food assimilators could rapidly turn
unlimited fresh food into their own preferred detritus. The
[inspector](docs/v11-preview.png) shows those capacities. In
[matched food-web tests](docs/v11-handling.png), the tested scavengers became
dependent on recycled food produced largely by grazers. Both ancestries persisted
for an hour in two of three mixtures under each processing treatment, so an
improvement in coexistence frequency is unestablished. Random-founder pilots
have not established reliable populations.
A verified 30-second recording is at `runs/v11-video/timelapse.mp4`.

The experimental [V12 preset](configs/v12.toml) adds heritable exploration and
learning rates, with acquired motor readouts that respond to food, costs, and
damage. [The inspector](docs/v12-preview.png) shows these offsets. The rule can
[learn and reverse a constructed cue–action task](docs/v12-rule.png), but
[physical lifetime assays](docs/v12-outcomes-lifetimes.png) found worse average
intake and reproduction with the tested learning settings. The
[community comparisons](docs/v12-outcomes-communities.png) also found no consistent
benefit. The first, less constrained version
disrupted inherited foraging; it remains in [v12-wide.toml](configs/v12-wide.toml).
A verified video is at `runs/v12-video/timelapse.mp4`.

The experimental [V13 preset](configs/v13.toml) expresses inherited body plans
through juvenile growth. Offspring start with one module, pay to grow additional
encoded modules when energy and space permit, and mature before reproducing.
New modules receive fresh neural state. This tests a measured barrier in the
older birth law: single-module parents often could not finance a larger child
at once. Its preset disables exploratory motor noise following the V12 results;
recurrent plasticity and evolving neural architecture remain active.
In three short matched pilots, two-module parents reproduced in two juvenile
treatments, while fully formed births produced no larger children. The
[adult](docs/v13-preview.png) and [juvenile](docs/v13-juvenile.png) previews show
one parent and its offspring; a verified recording is at
`runs/v13-video/timelapse.mp4`. [Three one-hour comparisons](docs/v13-development.png)
retained larger bodies in every juvenile treatment and none in the fully formed
offspring controls. Three-module bodies also reproduced. Population and birth
counts did not consistently improve. [Fresh random populations](docs/v13-development-native.png)
persisted for 30 minutes under both offspring rules; their founders already
included larger body plans. [All three native populations](docs/v13-development-long.png)
also persisted for three simulated hours, ending with 58, 73, and 28 creatures.
Larger bodies remained common in two populations and became rare in the third.
One population retained both fresh-food specialists and scavengers descended
from differently allocated founders. [Twelve transplantation controls](docs/v13-native-foodweb.png)
support scavenger dependence on grazer-produced detritus in that selected
community. Its verified recording is at
`runs/v13-native-long-video/timelapse.mp4`.

The experimental [V14 preset](configs/v14.toml) adds private signals between
adjacent modules within a body, plus each module's body-relative coordinates.
Its shared circuit has 40 sensory inputs and seven outputs, including two signed
internal emissions. Signals incur energy costs and reach neighboring controllers
on their next update. [The inspector](docs/v14-preview.png) shows the held signals.
Matched tests remove position readings, signal reception, or both; a self-signal
control distinguishes extra local memory from information sharing. Short pilots
and [twelve hour-long comparisons](docs/v14-coordination-communities.png)
showed no consistent population or reproduction benefit. A further
[360 fixed-genotype lifetimes](docs/v14-coordination-lifetimes.png) found small,
mixed effects and no consistent reproductive benefit. Many founders remained
juveniles. A supplementary adult-start assay was interrupted at wrap-up, with
129 of 360 individual lifetimes retained; its comparison remains incomplete.
A verified recording is at `runs/v14-video/timelapse.mp4`.

The experimental [V15 preset](configs/v15.toml) lets evolution alter the local
plasticity rule itself through four inherited signed coefficients. They control
responses to joint activity, activity on either side of a connection, and a
constant term. Updates remain bounded, offspring start with fresh neural state,
and a fixed-rule control preserves the earlier mechanism. The
[inspector](docs/v15-preview.png) shows the coefficients. Mechanical and replay
checks pass. [Native starts](docs/v15-rules-native.png) established poorly under
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
still unresolved. See the [comparison](docs/results/v15-birth-readiness.json).
The original settings used for the earlier V15 research are preserved in
[v15-research.toml](configs/v15-research.toml). Resuming a checkpoint keeps its
saved settings; start a fresh run to use the revised preset:

```bash
uv run garden run --config configs/v15.toml --view --device cpu --seconds 0
```

## Start watching

The [V16 preset](configs/v16.toml) adds area-preserving irregular patches, slowly
drifting sources, and local fertility that depletes when food grows and recovers
over time. Existing food stays put as its source moves. **P** toggles source
outlines; **Tab** includes a fertility view. See [parameters, results, and the
possible later producer step](docs/DYNAMIC_RESOURCES.md). All three 600-second
startup checks produced births, but reliable long-term persistence and improved
food tracking remain unestablished.

Python 3.13 and all Python packages are managed with **uv**:

```bash
uv sync --locked
uv run garden run --config configs/v16.toml --seed 2 --view --device cpu --seconds 0
```

`--seconds 0` runs until you close the window, press Ctrl+C, or the population
becomes extinct. A checkpoint and preview are saved on exit. Every run gets its
own directory under `runs/`; a supplied `--output` directory must not already exist.

V13–V16 start 192 juvenile creatures in a 512-unit dish, with a capacity of 1,024.
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

Omitting `--config` preserves the original V0 defaults (256 fixed-body creatures).

| Control | Action |
| --- | --- |
| Space | Pause/resume |
| `N` while paused | Advance one controller interval using the usual physics ticks |
| `+` / `-` | Increase/decrease simulation speed |
| Mouse wheel over the dish | Zoom, up to 8x |
| Mouse wheel over the leaderboard / Page Up / Page Down | Change leaderboard page |
| Right-drag | Pan |
| Left-click | Select a creature in the dish or leaderboard; click a hidden neuron in the inspector to inspect its links |
| `T` | Cycle fading trails: all creatures, selected creature, off |
| `[` / `]` | Change trail duration: 10, 30, or 120 simulated seconds |
| `G` | Follow the selected creature |
| `M` / module buttons | Choose a body module's controller |
| `F` | Toggle the smell overlay |
| Tab | Cycle resource, organism, forecast, secretion, patch-identity, shelter, and V16 fertility fields |
| `P` | Toggle V16 food-source outlines and center marks |
| `C` | Switch between inherited diet and lineage colors |
| `B` | Hide/show the sidebar |
| `L` / Leaderboard or Inspector button | Switch between the leaderboard and creature inspector; `L` opens the leaderboard if the sidebar is hidden |
| Leaderboard column headers | Sort by that statistic; click again to reverse the order |
| Brain / Body / Details buttons | Switch between the network, morphology/weight maps, and diagnostics |
| `R` | Restore the initial close view and stop following |
| Home | Center and fit the entire circular dish |
| Escape / close window | Save and exit |

The Pygame viewer opens with trails and an inspector sidebar. The window is
resizable and initially fits the desktop. The live camera starts centered and
zoomed in so the circular dish's radius is one viewing-panel width, putting its
boundary beyond the screen. This changes the view, not the habitat's physical
size or food density. Scroll to adjust zoom, use Home for a whole-dish overview,
or `R` to restore the initial view. Recordings and saved previews keep the overview.
The header has a transparent background,
and controls are stacked at the bottom left. The inspector adapts to the window's
height without scrolling. Trails follow body centers, are sampled
at 5 Hz of **simulated time**, and freeze when paused. They also appear in new
recordings. History starts when the viewer/recorder opens; checkpoints do not
contain old trails. A dead creature's trail fades normally, and its last inspected
controller sample remains available until another creature is selected.

The **Leaderboard** lists living creatures and updates as the simulation runs.
Sort by lifetime (age in simulated seconds), generation, food energy absorbed,
children born, current stored energy, distance traveled, or creature ID. **Food**
is cumulative energy absorbed from fresh food, detritus, and prey over that
creature's lifetime; it excludes starting energy and measures energy gained,
rather than the number of particles eaten. **Distance** uses world units. These
counters come from the simulation and include activity before a checkpoint was
loaded. Dead creatures leave the table.

Click a row to select that creature, center the camera on it, and open its
inspector; `G` follows it. Press `L` to return to the table with the same sort and
page. Click headers to switch between highest and lowest first; ties use creature
ID. Previous/Next buttons, Page Up/Down, or the mouse wheel over the table change
pages. The table fits the panel without a scrollbar, including smaller windows.
Space pauses it along with the simulation. See the [leaderboard preview](docs/leaderboard.png).

Selecting a living creature shows one **Energy** bar beside its name on every
inspector tab. It fills according to stored energy / capacity, with amber below
50% and red below 25%. A gold vertical line marks the **birth threshold** on the
same scale; current / maximum energy and the birth threshold are printed above
the bar. Growth, retry, or population-limit status appears below. Values use the
creature's current body size; growth can raise the threshold. Reaching the marker
still requires a mature body, an expired retry timer, population capacity, enough
energy to fund the mutated child, and an unoccupied birth location. **Ready to
try** means the known gates are satisfied; child cost and placement are checked
at the actual attempt. The display reads current state without changing the
simulation. See the [energy and birth threshold preview](docs/energy-threshold.png).

The **Brain** tab shows named sensory inputs, active recurrent neurons, and actual
actuator outputs for one module. Click a hidden neuron to show its incoming input
and recurrent links, plus its outgoing actuator links. Colors indicate sign;
activation brightness uses a fixed unit scale. Inactive neural slots are omitted
from the graph. Links include the learned weights used for that decision; motor
exploration is listed separately. Inputs are copied at the controller update,
including consumed feedback and each module's local senses, rather than sampled
again during rendering. A selection made while paused waits for `N` or resume.

The **Details** tab separates inputs, recurrent state, and bias in a drive breakdown.
The field mean/contrast readout splits each field's four sensors into their shared
mean and directional differences, then reports their weighted drive RMS over
active neurons. It is a sensitivity diagnostic, not a measure of intelligence or
proof of why a behavior evolved. The **Body / learning** tab includes the actual
body turn command, expressed modules, learning-rule coefficients, and inherited,
acquired, and effective recurrent weight maps. Each module has separate acquired
state. Secondary diagnostics and the color legend live in **Details** so the
main graph can stay fully visible; weight maps adapt to the available height.

The inspector reads the latest controller update (usually 10 Hz), so slow the
simulation or use pause/step to examine rapid decisions. Inspection and trail
sampling do not change physics steps, random streams, or checkpoint formats.
See [the viewer preview](docs/viewer.png).

## Headless runs and recordings

```bash
# Run an experiment without a window; record a 100x timelapse.
uv run garden run --config configs/v7.toml --seed 71 --device cpu --seconds 3600 --record --output runs/experiment-1

# Run overnight until stopped; sample video at a higher timelapse speed.
uv run garden run --config configs/v7.toml --seed 71 --device cpu --seconds 0 --record --video-speed 1000 --output runs/overnight-1

# Resume for 600 additional simulated seconds, optionally with a window.
uv run garden run --resume runs/experiment-1/latest.pt --device cpu --seconds 600 --view
```

All durations describe simulated time. Rendering, video sampling, and wall-clock
throughput do not change the physics timestep. Video streams to MP4 rather than
accumulating frames in RAM. Recording slows execution if the encoder cannot keep
up; frames are not silently skipped. Use a larger `--video-speed` for compact
overnight recordings, or omit `--record`.

Ctrl+C and termination requests finish the current simulation tick before saving.
Checkpoints are also written every five wall-clock minutes. Resume uses the
checkpoint's configuration, seed, controller, fields, population, and random
streams. Resume on the same CPU/CUDA device type; different hardware or dependency
versions may produce numerical differences. Forced process kills cannot save.

Run outputs include:

- `config.toml` and `metadata.json`: resolved settings and runtime details.
- `source.zip`: package sources and uv project/lock files captured when the
  process imported the storage module, so ongoing batches retain their provenance.
- `metrics.jsonl` and `events.jsonl`: population, resources, costs, births, deaths,
  genetic diversity, lineages, and throughput.
- `latest.pt`: an atomic full-world checkpoint.
- `checkpoints/`: periodic full-state archives when `archive_sim_seconds > 0`;
  the V5 preset archives every 600 simulated seconds.
- `founders.pt` and `population.pt`: initial and current genomes for evaluation.
- `preview.png`, `summary.json`, and optional `timelapse.mp4`.

`latest.pt` and `population.pt` retain the latest saved state. V0–V4 presets leave
automatic historical archives disabled; set `archive_sim_seconds` to enable them.
Independent resumed runs write to new directories, preserving the source run.

## CPU and GPU

The locked Linux/Windows PyTorch build uses CUDA 12.8 and was tested on an RTX 5080.
CUDA is optional at execution time; the same implementation also runs on CPU.

```bash
uv run garden doctor --device cuda
uv run garden run --device cuda --seconds 600
```

`--device auto` selects CUDA when available. **CPU is currently faster for the V0
population size**, so the examples explicitly use CPU. Initial full-size checks
measured roughly 5–6x simulated speed on CPU and about 0.9x on the 5080. The current
CUDA path is functional but has significant overhead from small, dynamic tensor
operations. Throughput depends on population, contacts, field resolution, and
recording; measure a representative run before allocating an overnight experiment.
The later V5 long runs achieved about 11–13x on CPU while running concurrently.
V5 also passed an RTX 5080 checkpoint-continuation smoke test; this is not a
full-size GPU performance benchmark.

V10+ CUDA runs and resumes enable deterministic PyTorch operations for more
reliable checkpoint replay; this affects the whole process and may cost speed.
The mode is recorded in runtime metadata. Reproduction and ecology comparisons
still require the same software and hardware environment; CPU and CUDA runs
with the same seed are not identical. Earlier GPU checks used a small numerical
tolerance and did not establish exact replay.

CPU execution defaults to one PyTorch thread because these operations are small.
To experiment with another setting, use `uv run garden --threads 2 run ...`.
A live window needs a working desktop/display; headless simulation and recording
do not. `doctor` reports the selected backend and package/runtime versions.

## Configure an experiment

The original starting values are in [configs/v0.toml](configs/v0.toml).

| Preset | Added mechanics |
| --- | --- |
| [V1](configs/v1.toml) | Recoverable patches, delayed detritus, inherited size/power/diet |
| [V2](configs/v2.toml) | Costly predation, defense, organism odor |
| [V3](configs/v3.toml) | Forecast cues, pulsed resources, inherited neural timescales |
| [V4](configs/v4.toml) | A developmental genome that repeats and places sensor/motor modules |
| [V5](configs/v5.toml) | Costly secretion, diffusion, decay, and historical archives |
| [V6](configs/v6.toml) | Hidden quality reversals, patch identities, and experienced food/damage inputs |
| [V7](configs/v7.toml) | Inherited rules for within-lifetime synaptic change |
| [V8](configs/v8.toml) | Evolving neurons and connections with construction and maintenance costs |
| [V9](configs/v9.toml) | Passive food-flow provenance, predation transfers, and immediate death causes |
| [V10](configs/v10.toml) | Spatial shelter, local cover sensing, and deterministic CUDA execution |
| [V11](configs/v11.toml) | Finite local food handling and simultaneous capacity limits |
| [V12](configs/v12.toml) | Bounded within-lifetime motor learning |
| [V13](configs/v13.toml) | Juvenile growth and discrete inherited body plans |
| [V14](configs/v14.toml) | Body-position sensing and internal module signaling |
| [V15](configs/v15.toml) | Evolving local plasticity rules; easier reproduction in the live preset |
| [V16](configs/v16.toml) | Irregular drifting food sources and local fertility depletion/recovery |

```bash
cp configs/v5.toml configs/my-experiment.toml
# Edit the new file, then run it:
uv run garden run --config configs/my-experiment.toml --device cpu --view
```

Partial TOML files inherit unspecified **V0 code defaults**, not a version preset.
`uv run garden config PATH` writes those V0 defaults; copy a later preset to start
an experiment with its full settings. Unknown keys and invalid values are rejected.
Keep physics/controller/field rates at integer ratios. Tune food
supply, metabolism, and food spacing first, and change one group of parameters at
a time. The saved configuration makes each experiment reviewable.

## Calibration and behavioral evaluation

```bash
# Five independent 600-second ecological runs with the default parameters.
uv run garden calibrate --device cpu --output runs/calibration-1

# Diagnostic controllers; these do not seed the evolving population.
uv run garden calibrate --device cpu --controller forager --seeds 1 --seconds 120 --output runs/forager-1
uv run garden calibrate --device cpu --controller rest --seeds 1 --seconds 120 --output runs/rest-1

# V0: compare saved founders and living descendants on fresh environments.
uv run garden evaluate runs/calibration-1/seed-1 --device cpu --output runs/evaluation-1

# A smaller exploratory evaluation before committing to the full batch.
uv run garden evaluate runs/calibration-1/seed-1 --device cpu --genomes 4 --seeds 10001 10002 10003 10004 --output runs/evaluation-small
```

The full evaluation samples up to 32 founder and 32 descendant genomes, tests each
on eight fresh seeds for 120 simulated seconds, and repeats descendant trials with
smell disabled or sampled at shuffled spatial locations. Each trial has its own
dish and fresh neural state. Mutation is disabled during evaluation; ordinary
feeding, energy costs, and reproduction remain active. Reports distinguish the
tested founding creature from its offspring.

Results include individual trials, sampled genomes, summary statistics, and
paired sensory effects. Repeat evaluations across independent evolutionary runs.
The small exploratory batch is a workflow check, not sufficient evidence of
evolved sensing or learning. Population survival and reproduction are implemented;
reliable improvement over founders and a sensory advantage must be established
through these experiments.

## Development

```bash
uv add PACKAGE
uv add --dev PACKAGE
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Commit changes to both `pyproject.toml` and `uv.lock`. The CUDA package index is
configured in the project; additional GPU libraries are not needed for CPU use.
Video encoding uses the FFmpeg executable bundled by `imageio-ffmpeg`.

The tests cover conservative food allocation, energy costs, starvation, collision
handling, mutation, birth failures, recurrent state, field sampling, exact CPU
checkpoint continuation, and independence from rendering/recording.

- [V0 specification and implementation milestones](PLAN.md)
- [V0 validation results](VALIDATION.md)
- [V1–V5 experiments, evidence, and limitations](EVOLUTION.md)
- [Original long-term research specification](spec.md)

## Ecology assays and version transfer

V1+ evaluations retain a whole community so feeding partners and competitors
remain present. `assay` samples founders and descendants into fresh environments,
disables mutation, preserves reproduction, and records each complete trial:

```bash
uv run garden assay runs/experiment-1 --output runs/ecology-assay \
  --seeds 10001 10002 10003 --seconds 240 \
  --modes none disabled rotated no_signal no_emission memory_reset

uv run garden probe runs/experiment-1 --output runs/cue-probe.json

# V6+: counterbalanced cue/outcome associations and reversal responses.
uv run garden association runs/your-v6-run --output runs/association-probe.json

# V7+: intervene on acquired state in cloned living communities.
uv run garden challenge runs/your-v7-run --output runs/state-challenge --seconds 180

# Matched evolution with the plasticity mechanism disabled but its cost retained.
uv run garden calibrate --config configs/v7.toml --ablation no_plasticity \
  --seeds 71 72 73 --seconds 3600 --output runs/frozen-plasticity --device cpu

# Initialize a new world from living genomes in an earlier experiment.
uv run garden run --config configs/v5.toml --seed-from runs/earlier-v4-run \
  --device cpu --seconds 3600 --output runs/inherited-v5
```

`disabled` zeros chemical inputs; `rotated` reverses directional readings while
retaining their mean intensity. `no_signal` removes secretion sensing;
`no_emission` suppresses production while retaining its cost. `memory_reset`
clears recurrent state before each controller update. Other controls include
`no_recycling`, `no_attacks`, `no_cue`, `pooled` module observations, and spatially
`shuffled` readings. V6 adds `no_identity` and `no_feedback`; V7 adds
`no_plasticity`, which removes acquired synaptic offsets and suppresses further
updates while preserving cost. `memory_reset` clears neural activity but retains
plastic offsets and traces. Version-specific controls require the corresponding
version.

The acquired-state challenge keeps the original living community, disables
mutation, and tracks its members through survival, feeding, and reproduction.
It compares retained state, erased synapses, erased activity, and disabled
plasticity with both unchanged and reversed food quality. Its branches share
the starting environment and random states; they are paired interventions, not
independent evolutionary samples. Erasure effects alone do not establish learning.

The cue probe compares different past cues followed by identical current inputs.
It measures intrinsic history dependence, not successful navigation or learning.
State resets also disrupt ordinary movement dynamics, so their effects alone
do not demonstrate useful memory. Community transplants have inherited body
differences and corresponding differences in founder energy endowment.

`--seed-from` preserves inherited circuits and existing traits while neutralizing
new sensory connections. New founders have fresh neural state, ages, and energy;
source IDs and generations are recorded. This is distinct from exact `--resume`.
Transfer supports forward version changes and rejects weight limits that would
alter the source circuit. V8 also supports padding into a larger neuron template
with new units dormant; earlier versions require equal hidden sizes. Python
callers should use
`create_world(config)` from `emergent_garden.world` to select the correct engine.

The committed [version trajectories](docs/evolution-versions.png),
[V5 random-founder trajectories](docs/evolution-native.png), and
[V5 inherited-population trajectories](docs/evolution-long.png) are generated
with `uv run python scripts/plot_results.py`. The script uses raw recorded metrics
when present and falls back to committed compact evidence. SVG versions are also saved.
Raw runs and videos stay under `runs/`; compact evidence is in `docs/results/`.

The [V7 plasticity comparisons](docs/v7-learning.png) use
`uv run python scripts/plot_learning.py`. They include failed starts and the
original plasticity settings, as well as the tighter experimental preset.
The V8 figure uses `uv run python scripts/plot_topology.py`.

To test deliberately assembled communities, run
`uv run python scripts/assemble_communities.py --help`. It compares two evolved
source populations separately or in a balanced mixture, preserving source
ancestry and recording each group's population, diet, and uptake. This is an
ecology experiment with evolved organisms; it does not demonstrate spontaneous
speciation or cooperative food sharing.
