# Development handoff — V24

Autonomous ecology research **resumed at the user's request on September 23,
2026**, after the earlier V15 pause and subsequent interface/resource updates.
The current code is **V24 / package 0.25.0**. Python and dependencies remain
managed with **uv**. Detailed history is in [CONTINUATION.md](CONTINUATION.md).

## V24: inherited neural response times

[V24](docs/NEURAL_TIMESCALES.md) adds one timing gene per hidden slot after the
structural masks, giving 5,304 values for the 32-slot preset. Each neuron's time
constant is the body time constant multiplied by
`exp(log(neural_timing_range) * tanh(gene))`. Range 1 preserves legacy arithmetic;
range 4 is the heterogeneous screen. Founder genes use sigma .5 and a separate
checkpointed RNG. Direct timing mutation uses probability .1 and sigma .15.
Neuron duplication copies timing along with incoming influence. Acquired
neural/learning states still reset at birth; timing has no extra energy charge.

All twelve 600-second pilots are complete, including fully neutral V24 runs.
Homogeneous populations are 144/78/100, births 288/265/156; inherited timing
without direct timing mutation gives populations 164/42/147, births 341/195/290;
timing mutation gives populations 121/29/176, births 208/87/373. All six neutral
comparisons match common physical states, measurements, and events exactly.
Added timing genes/RNG, their derived statistics, total-genome variance, and
full-genome digests are explicitly excluded from those compatibility comparisons.

The paired 576-circuit pulse probe uses unchanged inherited weights and no
learning or movement. Varied timing increases immediate response magnitude in
all three seeds, but does not consistently increase retained activity. The
new timing factors also alter mean response speed, so no benefit can yet be
attributed specifically to heterogeneity or useful temporal memory.

All nine continuations are **complete and audited** through 1,800 seconds.
Homogeneous populations are 235/125/274, births 1,727/994/1,563; inherited timing
gives populations 273/124/289, births 1,785/1,112/2,101; timing mutation gives
populations 291/166/333, births 1,620/1,296/2,675. Only 1–2 founder lineages remain
in each community, and no digestive-allocation scavenger specialists remain.
Detritus and prey still contribute energy. Greater reproduction has not
established a richer food web or useful memory.

Paths are `runs/v24-{homogeneous,inherited,evolving}-long-{1,2,3}`. All three
batch processes exited successfully; do not restart them. The audit command
`uv run python scripts/audit_neural_timing.py --include-long` and
`scripts/plot_neural_timing.py` now include all continuations, source archives,
resource/energy/trophic accounting, and pulse provenance. The next
[controlled assay](docs/NEURAL_TIMING_ASSAY_PLAN.md) compares original timing,
mean-preserved uniform timing, timing reassignment, and motor-feedback controls
on the three inherited-timing descendant communities. All six assay tests pass.
All thirty trials are now running in three sequential batches under
`runs/v24-timing-assay-{1,2,3}`. Their exec sessions are 11962, 57248, and 2905;
logs are `/tmp/emergent-garden-v24-timing-assay-{1,2,3}.log`. Each trial saves
its complete initial state and final checkpoint. Do not restart a batch without
checking its existing summary. The prospective plan records interventions and
interpretation; auditing and summarizing all thirty trials is the next task.

CPU and RTX 5080 checks pass exact device-local replay through births, growth,
and signed plasticity rules. Details shows the selected neuron's response time
without another row, uses the actual per-neuron integration factors, and omits
inactive value predictions. All inspector tabs fit at 640–1,024 pixels. All
414 tests and repository-wide Ruff checks pass, including six transplant checks.
The release guide retains the
mechanical checks, source provenance, and initial experimental results.
Research remains active. Preserve the user's V16 edits and unrelated notes.

## V23: direct sensory value features

[V23](docs/SENSORY_VALUE.md) optionally appends the actual 42 controller inputs
to the critic's hidden activity and bias: 75 acquired weights instead of 33.
`motor_value_inputs = 0` preserves V22 exactly. The inherited circuit and actor
are unchanged; no extra energy is charged in this representation screen.
`configs/v23.toml` enables the feature path, while `v23-baseline.toml` retains
hidden-only prediction. Both use V22's direct returns and .02 maximum critic rate.

All nine 600-second pilots are complete. Hidden-only populations are 147/67/22,
births 348/230/71; sensory populations 84/81/4, births 131/262/16; shuffled sensory
populations 172/7/171, births 480/25/384. There is no reliable ecological advantage.
All three neutral runs match V22's common final state and measurements exactly.
Three native forecast forks include deaths and beat zero in only one community;
the third has just two initially mature creatures.

Paired passive critics trained on identical V22 direct-return trajectories
reduce MSE with sensory features by about 1.2%/4.7%/6.0%. Both representations
still lose to predicting zero in two of three communities. A fivefold faster
critic rate (.1) worsens MSE in all three for both representations, despite
increasing correlation. Full physical states match across rates. All six
passive forks are complete; neither their larger sample counts nor the native
forks are independent ecological replicates. No longer ecological or transplant
follow-up was launched for these mixed results.

All 388 tests, Ruff, and separate CPU/CUDA replay exercises pass. Inspector
Details states the feature source; all tabs fit 640–1,024 pixels. Full evidence
and commands are in the guide; local runs use `runs/v23-*`. Six additional
instrumented replays match the original complete worlds and shadow critics.
Positive error clipping removes 2.56%/7.30%/<.01% of sensory error mass at the .02
rate; negative errors never clip. Norm bounds are uncommon at that rate. This
does not establish clipping as the main limitation, especially in seed 3 where
it is nearly absent. The [two-second horizon follow-up](docs/PREDICTION_HORIZONS.md)
is now complete: three passive comparisons beat zero in two full sensory cohorts,
and all three older subgroups. Six new native pilots produce own-return births
210/13/281 versus shuffled 543/26/249. The shorter target helps two evolutionary
starts but does not establish useful motor adaptation. `configs/v23-short.toml`
is optional; the main preset retains 20 seconds. All 390 tests and Ruff checks
pass. No runs from this follow-up remain unfinished. The
[V24 timing plan](docs/NEURAL_TIMESCALES_PLAN.md) subsequently led to the
implemented timing experiments described above.
Research remains active. Preserve
the user's V16 edits and unrelated notes. V21–V23 have no unfinished scheduled
ecological follow-ups.

## V22: learned energy predictions

[V22](docs/VALUE_PREDICTION.md) adds an optional acquired linear value readout
per body module. Its TD error replaces the running-mean actor's feedback. Hidden
activity plus bias gives 33 critic weights; the inherited 42-input, seven-output,
32-slot, 5,272-gene controller is unchanged. Acquired predictions reset at birth
and module growth. The head costs no additional energy in this first experiment.
`configs/v22.toml` uses centered returns, `v22-raw.toml` direct net returns,
and `v22-baseline.toml` disables prediction. Both use independent exploration.

All 15 matched 600-second pilots are complete. Baseline births are 288/265/156,
centered births 228/167/303, direct-target births 348/230/71. Shuffled controls
produce 246/76/323 and 481/193/300 respectively. Neither predictor establishes a
reliable benefit. All three neutral starts exactly reproduce the common V21
final state and physical measurements. Six 100-second forecast forks are also
complete, including deaths: centered predictions lose to a zero prediction in
all three full cohorts; direct predictions win in one. The older subgroup has
the same centered result and two direct-target wins. Shared communities are not
independent creature-level replicates. No V22 long continuation or fixed-genotype
fitness assay was launched after these weak results.

All 363 tests and Ruff checks pass, plus CPU/CUDA device-local exact replay.
All inspector tabs fit at 640–1,024 pixels, and Details displays the actual value
and TD error. Source records, plots, forecast timing, and limitations are linked
in the guide. Next test: give the predictor direct access to existing sensory
observations, keeping the inherited circuit and actor unchanged, to distinguish
a representation limitation from insufficient learning. This is an unproven
hypothesis. Research remains active; preserve the user's V16 edits and unrelated
notes. All V21 follow-ups are complete.

## V21: persistent motor exploration

[V21](docs/PERSISTENT_EXPLORATION.md) adds two-second correlated motor exploration
with the matching conditional likelihood score. Previous features/logits are
acquired module state, retained by checkpoints and reset at birth/growth. No
genome or interface dimensions change. The observer displays the actual applied
perturbation and persistence time. Use `configs/v21.toml` for persistence and
`configs/v21-baseline.toml` for independent noise; the remaining settings match
V19's gentle learning preset, including receptor radius 1.

All 12 matched pilots are complete. Independent-noise births are 288/265/156
with learning and 367/264/310 without. Persistent-noise births are 128/314/365
with learning and 10/102/160 without. All six independent runs reproduce V19's
complete common physical state and recorded measurements exactly. A separate
192-circuit movement diagnostic confirms the intended temporal correlation,
but finds no consistent net-displacement improvement. All 336 tests, Ruff
checks, and separate CPU/CUDA exact replay exercises pass. The three inspector
tabs fit 640–1,024-pixel heights.

Both persistent-noise continuation groups are complete: learning produced
populations 65/181/295 and births 621/1,604/2,145; without motor updates the
populations were 0/33/189 and births 11/257/1,374. The first noise-only start
became extinct at 1,397.63 seconds. All 24 matched community transplants into
seeds 801/802 are complete. Own-return learning produced more births in three
of six comparisons against disabled motor updates and three of six against
shuffled returns. Fresh-food uptake improved in four and two comparisons,
respectively. Mutation was disabled and starting descendant genomes matched.
Effects remain conditional; reliable adaptive credit assignment is unestablished.
The V21 record now has no unfinished experimental follow-ups.
Keep the research loop active and preserve the user's `configs/v16.toml` edits.

## V20: sensory reach and a learning control

[V20](docs/SENSOR_RADIUS.md) adds optional receptor distances of 1, 2, or 4 body
module radii. Nine matched 600-second starts are complete. Four-radius births
were 266/111/201 versus 455/26/195 at radius 1: mixed effects. All four-radius
starts reached 1,800 seconds with populations 253/13/277 and births
1,527/193/2,039, fewer births than the matching radius-1 run in every seed. Frozen-world samples confirm an
approximately fourfold directional-signal increase, with little change in mean
intensity. Radius 1 matches V19's full final physical state and recorded metrics
in all three 600-second starts. No genomes, body shapes, or energy costs change.

The new `shuffled_motor_reward` control gives each learner another creature's
normalized energetic return rate, with its own checkpointed random stream.
Physical energy and sensory feedback are untouched. Its six V19 follow-ups are complete; own-return learning produced more births
in four comparisons, with no general advantage established. All 322 tests and Ruff checks pass, plus CPU/CUDA mechanical replay
and a separate native CUDA shuffled-return replay. See the V20 guide for exact
settings, caveats, and commands. Research remains authorized and active.

## V19: neural diversity

[V19](docs/NEURAL_VARIATION.md) initializes inherited circuits with 8–24 active
neurons and half the recurrent edges, using independent randomness and fan-in
weight scaling. Neutral settings match V18's complete final physical states and
all recorded physical measurements in three 600-second runs. The 42-input,
seven-output, 32-slot recurrent template and 5,272-value genome are unchanged.
Children inherit circuit masks/weights without founder rescaling and start with
empty learned state.

Fifteen 600-second pilots compare baseline, varied founders, stronger mutation,
gentle motor learning, and identical noise with motor updates disabled. Results
are mixed; learning produced 288/265/156 births versus the control's 367/264/310.
All three varied-founder continuations reached 1,800 seconds: populations
251/34/327 and births 2,413/237/2,198. Stronger mutation ended at populations
162/0/363; seed 2 became extinct at 771.13 seconds. Learning and noise-only
continuations all reached 1,800 seconds; learning had fewer births in all three
starts. The [audit](docs/results/v19-circuits.json) includes 15 pilots and all
12 continuations. The 18 paired learning transplants are complete: births
improved in five of six descendant pairs, food intake in three. These
conditional effects do not establish adaptive credit assignment. Shuffled-return
follow-ups are complete: four of six favor own-return learning for births.

A new synthetic food-response probe separates inherited turning bias from
directional sensitivity. Selected populations differ considerably; its outputs
do not establish whole-body searching or useful learning. Raw paired stimuli,
responses, and provenance are retained. All 310 tests and Ruff checks pass;
separate CPU/CUDA exercises passed exact replay through variable graphs,
growth, births, evolving rules, and motor learning. A 36-trial delayed-credit
fixture found only small cue adaptation, especially with delayed outcomes;
longer eligibility alone did not solve it. An illustrative learning-population
preview is at `docs/v19-learning-brain.png`.

Keep the user's `configs/v16.toml` edits intact. The research loop remains
authorized and active. Continue with measured sensory/credit-assignment
limitations and paired learning comparisons, not assumptions that larger
networks are automatically more capable.

## V18: carried meals

[V18](docs/CARRIED_FOOD.md) adds bounded fresh/detritus compartments, 42 neural
inputs including fullness, and 5,272 inherited values. Food can be collected and
digested while moving; old expiry, processing, and source/provenance rules remain.
Carried meals drop on death and are not inherited. The live inspector shows them.

Twelve matched 600-second starts cross carrying (0/200) and processing (60/300).
With carrying at rate 300, births were 411/302/113 versus 97/37/32 without carrying.
The combined treatment is the next brain-experiment baseline. Carrying at rate
60 helped initial establishment, but one of its three longer runs became extinct
at 834.4 seconds; the other two reached 1,800 seconds with populations 94/16.
All three fast-carrying continuations reached 1,800 seconds with populations
57/254/219, births 1,521/2,307/1,218, and maximum living generations 48/59/23.
They are saved under `runs/v18-fast-carrying-long-*` and included in the audit.

All 299 tests and Ruff checks pass. Separate CPU/CUDA carrying checks passed
exact replay and all ledgers; food
transactions now use float64 in V18. Existing versions retain their old layout
and arithmetic. History auditing permits explicit neutral defaults on resume.
The [viewer previews](docs/v18-body.png) fit the window without scrolling, and
`runs/v18-carrying-video/timelapse.mp4` is a 4× recording with trails.

The user reinforced interest in neural/mutation diversity and within-lifetime
learning. Prioritize controlled founder-architecture, mutation, and motor-learning
comparisons next. Keep neural weights unprescribed and preserve the user's V16
working settings. The completed [sensory assays](docs/FORAGING.md#follow-up-sensory-interventions)
show some environmental dependence but weak/mixed directional dependence.

## V17 foraging record

The user clarified that their V16 edits aim for rare, valuable meals and cheap
exploration. Their working `configs/v16.toml` is preserved; an explicit baseline
snapshot and separate experiment presets are added. See the
[design, results, commands, and limitations](docs/FORAGING.md).

Twenty-four 600-second pilots show an intake bottleneck: larger food particles
still require sustained physical contact for digestion. Increasing handling rate
from 60 to 300 in the user's large habitat produced 236/59/17 births. Continuing
all three starts to 1,800 seconds yielded populations 97/122/15 and cumulative
births 809/799/44. Seed 3 recovered from a one-creature bottleneck. This supports
the treatment's viability in those starts, not reliable general establishment.
Completed sensory interventions found weak/mixed directional dependence.

V17 adds an optional mean/contrast encoding without changing the 40-input,
seven-output recurrent architecture or hand-tuning neural weights. Its three
paired tests were mixed; it is not promoted as a proven improvement. Disabled
encoding preserves legacy trajectories exactly. All 281 tests and Ruff checks
pass; separate CPU/CUDA mechanical exercises passed exact checkpoint replay.
Committed evidence is in [foraging-pilots.json](docs/results/foraging-pilots.json).

The subsequent V18 carried-food implementation is described above. Simple
producers remain a later option.

The remainder records earlier completed work and the V15 interruption; its
historical partial batches have not silently been completed by this restart.

## V16 dynamic resources

`configs/v16.toml` adds equal-area warped ellipses (aspect ratio 2, maximum bend
0.35), source drift at 0.5 units/s with independent wandering headings, and a
fixed 64-by-64 fertility grid. Cells hold 128 energy at the default world size,
spend 20 per accepted fresh-food particle, and recover with a 120-second time
constant. Existing food and shelter do not move. Forecast and identity fields
follow source centers. Neither fertility nor source geometry adds neural inputs.

Press `P` for source outlines and use Tab to reach fertility. The
[design and parameter guide](docs/DYNAMIC_RESOURCES.md) includes previews,
compatibility, and **simple producers as a possible later step**. Launch with
`uv run garden run --config configs/v16.toml --seed 2 --view --device cpu --seconds 0`.
Old presets and resumed checkpoints retain their previous rules.

Three 600-second CPU starts produced 13/30/12 births and final populations 6/4/5
at seeds 1/2/3. First births were at 232.03/5.43/16.23 seconds. This verifies that
the combined environment can support reproduction in these starts, not reliable
persistence, better learning, or a causal explanation of circling. The
[audit](docs/results/v16-resources.json) retains comparison and accounting data.
The first 120 seconds of seed 1 are recorded at
`runs/v16-resources-preview/timelapse.mp4`; all 901 frames decoded correctly.

## September 23 interface and reproduction update

Pygame now provides fading centroid trails, a resizable window with an inspector sidebar,
module selection, live input/hidden/output values, focused connection graphs,
inherited/acquired/effective weight maps, follow-camera, and paused stepping.
The header background is transparent and controls are stacked at the bottom left.
The inspector fits the window without scrolling; the Details tab holds the drive
breakdown and legend, and graph spacing and weight-map sizes adapt to the height.
The live viewer starts centered with a projected dish radius of one panel width,
filling the screen with habitat. `R` restores this view and Home fits the whole
dish. The physical circular world and recording framing are unchanged.
Trails sample simulation time at 5 Hz, retain up to 120 seconds, and also appear
in new recordings. The default visible duration is 30 seconds.

The selected creature has one HP-style energy/capacity bar with a gold vertical
birth-threshold marker. Current / maximum energy and the threshold value appear
above it; growth, cooldown, or population-cap status remains below. These use
actual body size and saved world settings. Reaching the marker is not a guaranteed
birth: child funding and placement are checked after mutation at the real attempt.
The bar fits on every tab without scrolling. See the
[preview](docs/energy-threshold.png).

The sidebar also has a live **Leaderboard** (`L` or its button). Column headers
sort living creatures by lifetime, generation, cumulative food energy absorbed,
offspring, stored energy, distance traveled, or ID. Food includes fresh food,
detritus, and prey; it excludes birth energy. Existing lifetime counters work
immediately on checkpoint resume. Dead creatures leave the list. Pages adapt to
the window, with Previous/Next, Page Up/Down, and wheel navigation over the table.
Clicking a row centers the creature and opens its inspector; `L` returns to the
same sort/page. Selection uses the ID displayed in the row and ignores a click
if that creature has since died. See the [preview](docs/leaderboard.png).

At the user's request, `configs/v15.toml` lowers the parent-area birth threshold
from 220 to 150 and the child-area base debit from 120 to 110; neural construction
is still charged, and newborn energy remains 100 per child area. The unchanged
original preset is `configs/v15-research.toml`; use it for continuing earlier
comparisons. Saved checkpoints retain their original settings when resumed.
Three paired 600-second fresh CPU starts (seeds 1/2/3) produced 3/19/3 births and
ended with 3/11/1 creatures, compared with 0/22/0 births and 0/20/0 creatures under
the original settings. This is a small startup check, not evidence of reliable
long-term persistence. The [record](docs/results/v15-birth-readiness.json)
retains commands, first-birth times, and source hashes.

Controller inputs are captured at the actual update boundary. Recurrent weights
include acquired offsets **before** their update; motor readouts include learned
offsets **after** their update, as used by the controller. This avoids resampling
cleared feedback or consuming the shuffled-sensing random stream. The observer
copies only the selected creature's modules. Neither observer state nor trails
enter checkpoints. A selection made while paused waits for `N` or resume before
showing its first exact sample.

See [controls and interpretation](README.md), [preview](docs/viewer.png), and
[verification record](docs/results/viewer-validation.json). Launch with
`uv run garden run --config configs/v15.toml --view --device cpu --seconds 0`,
or use `--resume` with a saved checkpoint instead of `--config`.

## What the V15 research iteration added

- Four inherited, signed coefficients let evolution vary the local recurrent
  plasticity rule. Updates stay bounded, and offspring inherit the rule with
  fresh neural state. Fixed-rule and disabled-plasticity controls are available.
- The controller remains 40 inputs, seven outputs, and up to 32 recurrent
  neurons per body module, initially 16 active. The genome now has 5,144 values.
- The inspector displays the expressed rule coefficients. V14's supplemental
  lifetime assay can also start the original body fully grown, so its coordination
  interface is available immediately.
- Completed food-web and lifetime evidence, failed random starts, and the exact
  boundary of interrupted work are documented and retained.

## What the evidence supports

The strongest ecological result is the selected V13 native environment-222
community. Its grazer and scavenger descendants persisted for three simulated
hours. Twelve subsequent controlled transplants showed that scavengers persisted
with grazers, but disappeared alone or when recycling was disabled. More than
98% of their detritus uptake in the mixtures came from grazer producers. This
supports food-web dependence in this selected community; it does not establish
speciation, cooperation, or how commonly that community would emerge.
See the [figure](docs/v13-native-foodweb.png) and
[audited records](docs/results/v13-native-foodweb-audit.json).

V14's 12 completed community comparisons and 360 juvenile-start lifetime assays
found no consistent reproductive benefit from the coordination interface.
Limited growth in two source communities restricted exposure to the interface.
The adult-start follow-up is incomplete.

V15's six new random-start pilots established poorly: five became extinct by
1,800 seconds. In the established-community comparison, only environment 271
completed all treatments: final populations were 73 with evolving rules, 44 with
the fixed rule, and 66 with acquired plasticity disabled; births were
1,308 / 1,133 / 1,063. This is one matched environment, so neither a general
advantage nor useful within-lifetime learning has been established.
The [complete histories](docs/results/v15-assembled-271.json) and
[invariant audit](docs/results/v15-assembled-271-audit.json) are retained.

## Verification

All **253 tests pass**, including 20 resource cases. Ruff lint and formatting
checks pass. V16 covers shape area and boundaries, stationary food and
shelter, per-cell funding, recovery, rejection accounting, exact replay, observer
purity, legacy parity, and viewer controls. Separate CPU and RTX 5080 exercises
passed exact replay through births, growth, and resource changes. A saved V15 run
also matched its archived code's 60-tick continuation bit for bit.

The interface's 22 cases cover exact controller samples, modular state, scripted controllers,
trail timing/identity/bounds, selection/resize, and complete trajectory equality.
Six further cases check birth eligibility across versions, separate storage and
birth scales, maturity, growth, cooldown, and capacity. The trajectory tests also
exercise the new bars. All three tabs fit at 640/768/984/1024-pixel heights.
The earlier V15 CLI check combined pause/step, 32x playback, and recording for 60 physics
ticks and matched the complete headless state bit for bit. A separate 30-tick
native CUDA check on the RTX 5080 also matched with randomized sensing enabled.

The earlier V15 mechanical checks
passed exact checkpoint replay separately on CPU and the RTX 5080, including
signed rule variation, births, growth, fresh inherited state, and energy
accounting. This means replay on each device, not identical trajectories across
devices. The [CPU](docs/results/v15-cpu-rules.json) and
[CUDA](docs/results/v15-cuda-rules.json) reports retain the measurements.

The V15 video decoded all **901 frames**, at 1,024×1,024 and 30 FPS
([verification](docs/results/v15-video.json)). It is a checkpoint fork for
illustration, not another independent experiment.

## Interrupted work

The [stop record](docs/results/iteration-stop.json) records paths, hashes,
completed work, and partial progress. Large checkpoints, videos, source archives,
and full logs remain under the git-ignored `runs/` directory. Compact results,
figures, and incomplete lifetime reports are committed.

Each V15 batch planned environments 271, 272, and 273 for 3,600 seconds each:

| Local batch | 271 | 272: last log / saved checkpoint | 273 |
| --- | --- | --- | --- |
| `runs/v15-evolving-mixed` | Complete | 984 s / 600 s | Not started |
| `runs/v15-fixed-mixed` | Complete | 1,184 s / 600 s | Not started |
| `runs/v15-no-plasticity-mixed` | Complete | 1,198 s / 600 s | Not started |

The three `seed-272` directories have no `latest.pt`: terminal interruption
ended the processes without a final checkpoint. Their last valid saved states
are `checkpoints/tick-000000018000.pt`. These were loaded and checked against
the corresponding 600-second metrics. Later logs are preserved but are not
resumable final states. Batch summaries remain `completed=false`.

The adult-start V14 assays retain **43 / 46 / 40** completed individual lifetimes
out of 120 planned for sources 241 / 242 / 243, respectively. Reports are
[241](docs/results/v14-adult-partial-241.json),
[242](docs/results/v14-adult-partial-242.json), and
[243](docs/results/v14-adult-partial-243.json). Their selected genomes match the
juvenile assays, provenance checks pass, and summary rows match the trial logs.
These incomplete comparisons are not reported as final treatment outcomes.
The selected genomes and completed rows remain in
`runs/v14-coordination-adult-{241,242,243}`. An unfinished individual lifetime
was not saved; the script has no automatic resume mode.

## Watch the saved populations

To continue viewing the established V13 food web from its three-hour checkpoint:

```bash
uv run garden run --resume runs/v13-native-long-222/latest.pt \
  --view --device cpu --seconds 0
```

To view V15's evolved-rule population after one simulated hour:

```bash
uv run garden run --resume runs/v15-evolving-mixed/seed-271/latest.pt \
  --view --device cpu --seconds 0
```

Both commands create a new run directory. They continue the saved world until
the window closes, Ctrl+C is pressed, or the population becomes extinct.
For a recording without running a simulation, open
`runs/v13-native-long-video/timelapse.mp4` or `runs/v15-video/timelapse.mp4`.

## Earlier V15 comparisons to resume when requested

To complete the existing comparisons, rerun V15 environments 272 and 273 into new
directories, retaining completed
271 as the first matched environment:

```bash
for mode in none fixed_rule no_plasticity; do
  uv run python scripts/assemble_communities.py \
    --config configs/v15-research.toml \
    --sources runs/v13-native-foodweb-source-222/grazers \
              runs/v13-native-foodweb-source-222/scavengers \
    --condition mixed --seeds 272 273 --seconds 3600 --device cpu \
    --ablation "$mode" --output "runs/v15-remaining-$mode"
done
```

This deliberately reruns the interrupted environments from initialization;
the assembly driver does not restore its ancestry counters from a checkpoint.
Use another fresh output path if one above already exists. This command is a
handoff instruction and has not been started.

For the adult-start comparisons, either implement explicit trial-level resume
using the preserved selected genomes, or rerun the complete assays in fresh
directories. The latter needs no implementation change:

```bash
for source in 241 242 243; do
  uv run python scripts/assay_coordination.py \
    --source "runs/v14-full-mixed/seed-$source" \
    --output "runs/v14-coordination-adult-rerun-$source" --start-mature
done
```

Do not combine duplicated lifetime rows from partial and rerun assays. Once
the comparisons finish, audit every planned treatment and use fixed-genotype
behavioral interventions to distinguish useful adaptation from mere state or
rule variation. Reliable establishment from fresh random founders remains a
separate unresolved problem.
