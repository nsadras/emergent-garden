# Development guide

[Project overview](../README.md) · [Usage guide](USAGE.md) · [Research and plans](../planning/README.md)

Use **uv** for Python and dependency management. Commands below run from the
repository root. For viewer controls, configuration, checkpoints, and CPU/CUDA
replay constraints, see the [usage guide](USAGE.md).

## Code and documentation map

| Area | Entry points |
| --- | --- |
| CLI, presets, and defaults | [cli.py](../emergent_garden/cli.py), [config.py](../emergent_garden/config.py), [configs/](../configs/) |
| Simulation and version dispatch | [world.py](../emergent_garden/world.py), [ecology.py](../emergent_garden/ecology.py) |
| Recurrent controllers and graph structure | [brain.py](../emergent_garden/brain.py), [topology.py](../emergent_garden/topology.py), [neural_timing.py](../emergent_garden/neural_timing.py) |
| Development and genome transfer | [development.py](../emergent_garden/development.py), [morphology.py](../emergent_garden/morphology.py), [inheritance.py](../emergent_garden/inheritance.py) |
| Acquired learning | [plasticity.py](../emergent_garden/plasticity.py), [learning.py](../emergent_garden/learning.py), [value.py](../emergent_garden/value.py), [recurrent_learning.py](../emergent_garden/recurrent_learning.py) |
| Observation, UI, and saved runs | [observation.py](../emergent_garden/observation.py), [viewer.py](../emergent_garden/viewer.py), [storage.py](../emergent_garden/storage.py) |
| Tests, assays, and audits | [tests/](../tests/), [scripts/](../scripts/) |

Use `create_world(config)` to choose the simulation engine. Physics advances in
fixed simulated-time steps; viewing and recording observe that state. Presets
select versioned mechanics. Inherited genomes and acquired neural state are
separate, and new offspring start with fresh acquired state.

Technical specifications, equations, parameter choices, experimental plans, and
results live in [planning/](../planning/README.md). The original V0 specification
is historical; later reports document subsequent mechanics. Images and compact
audited data remain in `docs/` and `docs/results/`; full runs stay in ignored
`runs/`. Keep commands repository-relative and links relative to each document.
Update the planning index when adding a research document, and link evidence
from its report rather than adding a version-by-version history to the README.

## Environment and checks

```bash
uv sync --locked
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

- [V0 specification and implementation milestones](../planning/PLAN.md)
- [V0 validation results](../planning/VALIDATION.md)
- [V1–V5 experiments, evidence, and limitations](../planning/EVOLUTION.md)
- [Original long-term research specification](../planning/spec.md)

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

The committed [version trajectories](evolution-versions.png),
[V5 random-founder trajectories](evolution-native.png), and
[V5 inherited-population trajectories](evolution-long.png) are generated
with `uv run python scripts/plot_results.py`. The script uses raw recorded metrics
when present and falls back to committed compact evidence. SVG versions are also saved.
Raw runs and videos stay under `runs/`; compact evidence is in `docs/results/`.

The [V7 plasticity comparisons](v7-learning.png) use
`uv run python scripts/plot_learning.py`. They include failed starts and the
original plasticity settings, as well as the tighter experimental preset.
The V8 figure uses `uv run python scripts/plot_topology.py`.

To test deliberately assembled communities, run
`uv run python scripts/assemble_communities.py --help`. It compares two evolved
source populations separately or in a balanced mixture, preserving source
ancestry and recording each group's population, diet, and uptake. This is an
ecology experiment with evolved organisms; it does not demonstrate spontaneous
speciation or cooperative food sharing.
