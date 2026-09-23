# Development handoff — V15

Autonomous ecology development was paused at the user's request on
**September 22, 2026**, and its experiment processes were stopped. Bounded
interface and reproduction-preset updates followed on September 23; the
autonomous research loop remains paused.
The current code is **V15 / package 0.16.1**. Python and dependencies remain
managed with **uv**. The detailed design and
experiment history is in [CONTINUATION.md](CONTINUATION.md).

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

The selected creature now has an HP-style energy/capacity bar with its numeric
values, and a separate birth-energy progress bar with growth, cooldown, or
population-cap status. Both use actual body size and saved world settings. A
full birth bar is not a guaranteed birth: child funding and placement are checked
after mutation at the real attempt. Both bars fit on every tab without scrolling.
See the [preview](docs/reproduction-readiness.png).

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

## What this iteration adds

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

All **233 tests pass**. Ruff lint and formatting checks pass. The interface's
22 new cases cover exact controller samples, modular state, scripted controllers,
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

## Next development session

First complete the existing comparisons before adding another mechanism.
Rerun V15 environments 272 and 273 into new directories, retaining completed
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
