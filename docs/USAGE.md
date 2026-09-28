# Usage guide

[Project overview](../README.md) · [Development guide](DEVELOPMENT.md) · [Research and plans](../planning/README.md)

## Start watching

Run commands from the repository root. Install uv before the first setup; uv
manages Python 3.13 and the project dependencies. The live viewer requires a
desktop/display. Headless runs and recording do not.

The [V16 preset](../configs/v16.toml) adds area-preserving irregular patches, slowly
drifting sources, and local fertility that depletes when food grows and recovers
over time. Existing food stays put as its source moves. **P** toggles source
outlines; **Tab** includes a fertility view. See [parameters, results, and the
possible later producer step](../planning/DYNAMIC_RESOURCES.md). All three 600-second
startup checks produced births, but reliable long-term persistence and improved
food tracking remain unestablished.

```bash
uv sync --locked
uv run garden run --config configs/v16.toml --seed 2 --view --device cpu --seconds 0
```

`--seconds 0` runs until you close the window, press Ctrl+C, or the population
becomes extinct. A checkpoint and preview are saved on exit. Every run gets its
own directory under `runs/`; a supplied `--output` directory must not already exist.

Creatures can grow inherited bodies with multiple sensor/propulsion modules.
Each module runs the same inherited recurrent circuit with its own acquired state.
Population, body, and neural limits depend on the preset. Green bodies favor fresh
food; amber bodies favor detritus. Red marks show attacks.

Omitting `--config` preserves the original V0 defaults (256 fixed-body creatures).

### Viewer controls

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

### Camera and trails

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

### Leaderboard

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
Space pauses it along with the simulation. See the [leaderboard preview](leaderboard.png).

### Energy and birth readiness

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
simulation. See the [energy and birth threshold preview](energy-threshold.png).

### Brain and body inspector

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
See [the viewer preview](viewer.png).

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

`--device auto` selects CUDA when available. **CPU was faster in the initial V0
benchmarks**, so the examples explicitly use CPU. Initial full-size checks
measured roughly 5–6x simulated speed on CPU and about 0.9x on the 5080. The measured
CUDA path was functional but has significant overhead from small, dynamic tensor
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

Presets are experiments, not a ranking: later versions add mechanisms whose
benefits may be mixed. See the [research index](../planning/README.md) and
[current handoff](../planning/HANDOFF.md) for evidence and status. Raw runs and videos named
in historical reports are local artifacts under `runs/`, not bundled downloads.

The original starting values are in [configs/v0.toml](../configs/v0.toml).

| Preset | Added mechanics |
| --- | --- |
| [V1](../configs/v1.toml) | Recoverable patches, delayed detritus, inherited size/power/diet |
| [V2](../configs/v2.toml) | Costly predation, defense, organism odor |
| [V3](../configs/v3.toml) | Forecast cues, pulsed resources, inherited neural timescales |
| [V4](../configs/v4.toml) | A developmental genome that repeats and places sensor/motor modules |
| [V5](../configs/v5.toml) | Costly secretion, diffusion, decay, and historical archives |
| [V6](../configs/v6.toml) | Hidden quality reversals, patch identities, and experienced food/damage inputs |
| [V7](../configs/v7.toml) | Inherited rules for within-lifetime synaptic change |
| [V8](../configs/v8.toml) | Evolving neurons and connections with construction and maintenance costs |
| [V9](../configs/v9.toml) | Passive food-flow provenance, predation transfers, and immediate death causes |
| [V10](../configs/v10.toml) | Spatial shelter, local cover sensing, and deterministic CUDA execution |
| [V11](../configs/v11.toml) | Finite local food handling and simultaneous capacity limits |
| [V12](../configs/v12.toml) | Bounded within-lifetime motor learning |
| [V13](../configs/v13.toml) | Juvenile growth and discrete inherited body plans |
| [V14](../configs/v14.toml) | Body-position sensing and internal module signaling |
| [V15](../configs/v15.toml) | Evolving local plasticity rules; easier reproduction in the live preset |
| [V16](../configs/v16.toml) | Irregular drifting food sources and local fertility depletion/recovery |
| [V17](../configs/v17-contrast.toml) | Experimental mean/contrast sensory encoding; [paired results](../planning/FORAGING.md) |
| [V18](../configs/v18-fast-feeding.toml) | Bounded carried food, digestion while traveling, and fullness inputs; [results](../planning/CARRIED_FOOD.md) |
| [V19](../configs/v19.toml) | Varied founder graphs and circuit/mutation experiments; [results](../planning/NEURAL_VARIATION.md) |
| [V20](../configs/v20.toml) | Configurable sensory footprint; [results](../planning/SENSOR_RADIUS.md) |
| [V21](../configs/v21.toml) | Temporally correlated motor exploration; [results](../planning/PERSISTENT_EXPLORATION.md) |
| [V22](../configs/v22.toml) | Acquired predictions of future energetic returns; [results](../planning/VALUE_PREDICTION.md) |
| [V23](../configs/v23.toml) | Sensory features for the value predictor; [results](../planning/SENSORY_VALUE.md) |
| [V24](../configs/v24-inherited.toml) | Inherited per-neuron response times; [results](../planning/NEURAL_TIMESCALES.md) |
| [V25](../configs/v25.toml) | Optional energetic reinforcement of recurrent connections; [results](../planning/RECURRENT_LEARNING.md) |

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

For command-specific options, use `uv run garden --help` or `uv run garden run --help`.
