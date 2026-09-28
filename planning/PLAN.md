# Emergent Garden: V0 implementation plan

This is the preserved V0 specification. Subsequent V1–V5 designs, parameter
changes, experiments, and limitations are recorded in [EVOLUTION.md](EVOLUTION.md).

Status: V0 software implemented; population persistence verified. Reproducible
foraging improvement and a sensory advantage remain experimental criteria.
Recorded: 2026-09-22.

This document defines the first implementation. [spec.md](spec.md) remains the
longer-term research roadmap; where it proposes broader or different initial
requirements, this V0 plan takes precedence. Numerical values are tunable
engineering defaults; the initial validation results are recorded in
[VALIDATION.md](VALIDATION.md).
The detailed mechanics below fill in the remaining implementation choices.

The executable defaults are recorded in [configs/v0.toml](../configs/v0.toml). See
[usage guide](../docs/USAGE.md) for uv setup, running the viewer, recording, resume, and
evaluation commands. Reproduction and population persistence are distinct from
the research criteria of improved foraging and a demonstrated sensory advantage.

## 1. Objective and success criteria

Build a living, watchable 2D dish of creatures that sense nearby food, propel
themselves, consume resources, reproduce, and evolve through inherited variation.
Each creature has a physical body and an abstract persistent neural controller.

V0 succeeds when:

1. A population sustains multiple generations in a shared dish.
2. The run produces a useful 2D recording and can be resumed from a checkpoint.
3. Descendants forage better than their original founders across fresh food
   layouts and matched starting conditions.
4. Disrupting descendants' food-sensing inputs measurably reduces that advantage.

Population survival and interesting-looking motion alone do not establish the
last two criteria. Memory, prediction, lifetime learning, and communication remain
questions for later experiments; they are not assumed consequences of evolution.

## 2. Scope and development environment

### Included in V0

- One shared, bounded, continuous 2D world with local resource competition.
- Fixed circular bodies with two independently controlled propulsion actuators.
- Food particles, a derived smell field, local sensors, and energy accounting.
- Fixed-size recurrent brains with evolving weights and biases.
- Continuous asexual reproduction, starvation, mutation, and lineage tracking.
- A live 2D viewer, zoom, timelapse recording, and accelerated headless runs.
- Configuration files, measurements, checkpoints, and separate evaluation runs.

### Deferred

- Soft bodies, muscle-driven swimming, and morphology evolution.
- Neural topology evolution, developmental genomes, and spiking neurons.
- Synaptic plasticity, evolved learning rules, and explicit predictive learning.
- Predation, damage, sexual reproduction, and environmental manipulation.
- Sound, light, chemical signaling, and communication experiments.
- Physical neural tissue and detailed biological physiology.
- Generational selection as an alternative evolutionary mode.

Future emitters and sensors should fit the same source/field/sensor boundary.
Different fields may have different propagation rules; sound and light need not
behave like the initial smell approximation.

### Hardware and Python tooling

| Item | Initial choice |
| --- | --- |
| Target GPU | NVIDIA RTX 5080, 16 GB VRAM |
| System memory | 16 GB RAM |
| Experiment duration | Overnight evolutionary runs are acceptable |
| Python | 3.13, matching the existing `.python-version` |
| Package and environment manager | **uv** |
| Numerical backend | Python with batched PyTorch tensors; GPU execution is the target |
| Configuration format | TOML, with the resolved configuration saved for every run |

Manage Python dependencies through `uv` and `pyproject.toml`. Generate and retain
`uv.lock` in version control when implementation dependencies are added. Use the
project `.venv`; do not maintain a parallel pip or Conda dependency workflow.

```bash
# Dependency management; replace PACKAGE with the actual package name.
uv add PACKAGE
uv add --dev PACKAGE
uv remove PACKAGE

# Reproduce the environment after the lockfile exists.
uv sync --locked

# Run project code and development tools through the environment.
uv run python main.py run --view --device cpu
uv run pytest
uv run ruff check .
```

`main.py` forwards to the same CLI as `uv run garden`. The uv project selects a
PyTorch/CUDA build supporting the RTX 5080, and GPU execution has been checked.
Record backend and driver versions with runs; compare CPU/GPU throughput before
an overnight experiment.
Follow the official [uv project workflow](https://docs.astral.sh/uv/guides/projects/)
and [uv/PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

## 3. Units, world geometry, and display scale

World units and energy units are abstract. Every duration and rate refers to
**simulated time**, except checkpoint intervals and measured wall-clock throughput.
Display pixels do not define the simulation lattice.

| Parameter | Initial value |
| --- | --- |
| Dish shape | Circle with solid boundary |
| Dish diameter | 1,024 world units |
| Coordinate extent | `[0, 1024]` on each axis; center `(512, 512)` |
| Creature body diameter | 8 units; radius 4 |
| Initial population | 256 |
| Population capacity | 2,048 |
| Full-dish display | Approximately 1,024 x 1,024 pixels |
| Inspection zoom | 4x to 8x |
| Chemical field grid | 512 x 512 cells over the coordinate extent |
| Field cell width | 2 world units |

Positions and headings are continuous. The field grid only approximates sensory
concentrations, and the renderer samples world state independently. At the nominal
display scale a creature is 8 pixels across; at 4x to 8x zoom it is 32 to 64 pixels
across. A 512-pixel overview makes the same creature 4 pixels across without
changing its physics or brain.

The initial bodies occupy about 1.56% of the dish area. The 2,048-creature capacity
is a computational limit, not a population target. A run that repeatedly reaches
it must be identified in the measurements because blocked births alter selection.

## 4. Bodies, propulsion, and collisions

Each creature stores an ID, position, heading, energy, age, genome, recurrent
state, current actuator activations, contact state, and lineage metadata.
Founders start at valid nonoverlapping positions sampled throughout the dish,
with independent random headings. Bodies remain the same size throughout life.

The two actuators represent body-mounted cilia or equivalent propulsion devices.
Their outputs apply forces in the body's forward direction at opposite lateral
offsets. Strong drag gives an overdamped response. There is no gravity or retained
momentum in V0. The controller emits only local actuator activations.

Let `a_left` and `a_right` be actuator activations in `[0, 1]`. Calibrate the
physical response as:

```text
forward_speed = 12 * (a_left + a_right)                # world units / second
angular_speed = (pi / 2) * (a_right - a_left)          # radians / second
world_velocity = forward_speed * [cos(heading), sin(heading)]
```

These are the simulation's force/drag response equations, not neural outputs.
They can be realized with unit maximum force per actuator, lateral offsets of
2 units, translational drag `1/12`, and rotational drag `4/pi`.

| Parameter | Initial value |
| --- | --- |
| Maximum forward speed | 24 units/second, or 3 body diameters/second |
| Maximum turning rate | 90 degrees/second |
| Motion integration | Fixed timestep; use the midpoint heading for translation |
| Creature collisions | Circle-circle overlap resolution |
| Boundary collisions | Keep the entire body inside the dish |
| Collision damage | None in V0 |

Contact constrains motion without adding propulsion or energy. Propulsion still
costs energy when a creature pushes against another body or the wall. Use local
neighbor lookup for collision candidates and avoid systematic processing-order
advantages. Verify that passive collision resolution does not generate persistent
motion. Record unresolved overlap or numerical failures as diagnostics.

## 5. Food particles and patch dynamics

| Parameter | Initial value |
| --- | --- |
| Food particle radius | 1 unit |
| Energy per new particle | 20 |
| Initial particle count | 1,024 |
| Replenishment rate | 20 particles/second for the entire dish |
| Patch distribution | 80% of new particles in patches; 20% scattered |
| Active patch count | 8 |
| Patch radius | 48 units |
| Patch relocation period | 120 seconds per patch |
| Relocation staggering | One patch relocates every 15 seconds |
| Uneaten particle lifetime | 180 seconds from spawning |

Food is stationary and nonblocking. Sample patch centers so patches fit inside
the dish; choose among patches uniformly, then sample a position uniformly by
area within the selected patch. Scattered food is uniform by area across the
dish. Apply the 80/20 mixture to both initial food and replenishment.

Patch relocation changes future spawn locations. Existing particles stay where
they are until eaten or expired. Food supply is independent of creature count.
Use a fractional spawn accumulator so replenishment does not depend on frame rate.

Food contact occurs when body and particle circles overlap. V0 uses automatic
absorption on contact, with no separate feeding action or digestion delay.
Resolve simultaneous claims with equal shares capped by each creature's remaining
storage capacity, including its claims on other particles that step. Retain any
unabsorbed energy in the particle. Remove a particle when depleted or expired.
The total energy awarded must equal the energy removed from particles; no creature
may receive energy beyond its storage capacity.

## 6. Sensory field and sensors

The initial chemical field is a spatial approximation derived from the current
food particles. It is not yet a time-integrated chemical diffusion model.

| Parameter | Initial value |
| --- | --- |
| Field grid | 512 x 512 |
| Gaussian characteristic spread, `sigma` | 24 world units |
| Source cutoff distance | 72 units, or `3 * sigma` |
| Field reconstruction rate | 10 Hz |
| Sensor sampling | Bilinear interpolation at the sensor's world position |
| Smell sensors | Four fixed points on the body perimeter |
| Sensor angles relative to heading | -135, -45, +45, +135 degrees |
| Other inputs | Stored energy and contact |

Define the reference concentration at location `x` as:

```text
q_particle = remaining_particle_energy / 20
kernel(distance) = exp(-distance^2 / (2 * 24^2))       # distance <= 72
kernel(distance) = 0                                # distance > 72
concentration(x) = sum(q_particle * kernel(distance_to_particle))
```

Deposit sources onto the grid and convolve with a kernel scaled to approximate
this definition. Keep concentration scaling consistent if grid resolution changes.
There are no sources outside the dish and no toroidal wrap. Particles are the
source of truth; consumption and expiration affect smell at the next field update.

For the initial implementation, map each concentration to
`concentration / (concentration + 8)`. This fixed compression scale is a calibration
default. Do not normalize against the current world-wide maximum or expose the
field grid to a brain.

The six brain inputs are the four smell readings, `energy / 250`, and a contact
flag. The flag records whether contact occurred since the last controller update.
Sensors initially have no added noise. No absolute position, absolute heading,
nearest-food vector, food identity, or population information is available to the
controller.

## 7. Energy, survival, reproduction, and inheritance

| Parameter | Initial value |
| --- | --- |
| Founder and newborn energy | 100 |
| Maximum stored energy | 250 |
| Baseline metabolic cost | 1 energy/second |
| Additional propulsion cost | `0.5 * (a_left^2 + a_right^2)` energy/second |
| Reproduction eligibility | Energy at least 220 |
| Parent debit per successful birth | 120 |
| Energy transferred to newborn | 100 |
| Energy spent on reproduction | 20 |
| Starvation | Energy reaches zero |
| Maximum age | None in V0; age is recorded |

Integrate costs using elapsed simulation time:

```text
energy -= dt * (1 + 0.5 * (a_left^2 + a_right^2))
energy += food_energy_actually_absorbed
energy -= 120                       # only for a successful birth
```

Reproduction is continuous and automatic for eligible creatures. There is no
global parent ranking, behavioral reward, or generation reset. Resource access,
costs, and actual reproduction determine which genomes spread.

Place offspring adjacent to their parent at a valid nonoverlapping location,
with a random heading. Initial placement search: 16 random directions, at a
center-to-center distance of 8.1 units. If placement fails or population capacity
is reached, postpone the birth without debiting energy; retry after one simulated
second while eligibility holds. Limit each parent to one birth per physics tick
and process competing birth attempts in a seeded randomized order.

A newborn inherits a mutated copy of the parent's genetic weights and biases,
with zeroed recurrent state and fresh contact state. No experienced neural state
is inherited. Assign a unique ID, parent ID, lineage ID, birth time, generation
number, and genome hash. Founders have generation zero and distinct lineage IDs.
Remove dead creatures; carcasses do not become food in V0.

If all creatures die, record extinction and end the run. A new seeded run is an
explicit separate experiment, not a hidden continuation of the extinct population.

### Budget sanity checks

- Without food, a newborn lasts 100 seconds at rest or 50 seconds at full thrust.
- With both actuators at 0.5, speed is 12 units/second and cost is 1.25
  energy/second: 80 seconds of reserve and about 960 units of possible travel.
- At maximum speed, traversing a dish diameter takes about 42.7 seconds.
- A birth at energy 220 leaves the parent at 100 and supplies the newborn with
  100; 20 energy is spent.
- Food supplies 400 energy/second. Dividing by a representative cost of 1.25
  gives 320 creatures before reproduction costs and uncollected or expired food.
  This is a budget bound, not a predicted equilibrium population.

## 8. Neural controller and genome

| Parameter | Initial value |
| --- | --- |
| Input channels | 6 |
| Hidden units | 16, fully recurrent |
| Output channels | 2 |
| Hidden activation | `tanh` |
| Actuator activation | `sigmoid`, producing values in `[0, 1]` |
| Neural time constant | 0.5 seconds for all units |
| Controller update rate | 20 Hz |
| Genome | Directly encoded network weights and biases |
| Genome parameter count | 402 |
| Lifetime weight updates | None in V0 |

```text
alpha = 1 - exp(-controller_dt / 0.5)
h_next = (1 - alpha) * h + alpha * tanh(W_rec @ h + W_in @ inputs + b_hidden)
actuators = sigmoid(W_out @ h_next + b_out)
```

At 20 Hz, `alpha` is approximately 0.0952. Recurrent state persists between
observations throughout life. Hold actuator outputs between controller updates.
Initialize each founder independently: matrix weights are Gaussian with mean
zero and standard deviation `1 / sqrt(number_of_input_connections)` for that
matrix; biases and recurrent state start at zero. The network matrices have
shapes `(16, 6)`, `(16, 16)`, and `(2, 16)`.

At each birth, independently mutate each weight and bias with probability 0.02.
For each selected parameter, add zero-mean Gaussian noise with standard deviation
0.05, then clip to `[-3, 3]`. This gives about 8 changed parameters per offspring
on average; unchanged clones are allowed. Initialize genetic parameters within
the same bounds. Time constants, sensor placement, body parameters, and network
topology do not mutate in V0.

## 9. Scheduling and run modes

| Subsystem | Initial cadence |
| --- | --- |
| Physics, contacts, feeding, metabolism | 60 Hz; `dt = 1/60` second |
| Controller | 20 Hz; every 3 physics ticks |
| Chemical field | 10 Hz; every 6 physics ticks |
| Reproduction and death | Checked each physics tick |
| Live viewer | Target 60 FPS; allow 30 FPS, independently of simulation speed |
| Metrics sampling | Once per simulated second, plus lifecycle events |
| Checkpoints | Every 5 wall-clock minutes and on orderly shutdown |

Use integer physics ticks for scheduling. Initialize fields and controllers before
the first movement step. Within each tick:

1. Expire old food, relocate patch sources when due, and replenish food.
2. Reconstruct the chemical field if scheduled.
3. Sample sensors and update controllers if scheduled.
4. Apply motion and resolve body/wall contacts.
5. Charge metabolism and propulsion, then remove creatures whose energy is zero
   or below. They do not feed or reproduce on that tick.
6. Resolve food absorption for surviving creatures, then eligible births.
7. Advance age/time and emit scheduled measurements and observation snapshots.

Births become active on the following tick, with their first controller update
before their first movement. They incur no costs for a partial birth tick.

At maximum speed, travel is 0.4 units per physics step, or 5% of body diameter.
Validate collision handling and lifecycle bookkeeping at this rate; acceleration
changes wall-clock throughput, not the simulation timestep.

Support live viewing with pause and speed controls, headless accelerated runs,
recording, checkpoint resume, and evaluation of saved genomes. Rendering must not
consume simulation randomness or determine when evolution advances.

## 10. Visualization, recording, and persistence

The 2D view should show the dish boundary, food particles, an optional smell
overlay, lineage colors, heading, sensor positions, and actuator intensity.
Selecting a creature shows its energy, age, generation, offspring count, and
current neural activity. Include pan/zoom and a visible simulation-time counter.

Record actual runs as timelapse video. Use configurable sampling in simulation
time, a configurable playback speed, and a default output frame rate of 30 FPS.
Stream frames to disk; do not hold an overnight recording in RAM. Recording is
optional and can use selected intervals. Log any dropped frames. The exact
renderer and video encoder packages are implementation choices, managed through
uv when they are Python dependencies.

Each run directory should contain:

- The resolved configuration, master seed, code version, and backend metadata.
- Sampled metrics and birth/death/lineage events.
- Initial founder genomes and selected descendant genomes.
- Periodic versioned checkpoints.
- Optional recordings and evaluation reports.

A checkpoint includes the complete live population, genomes, neural states,
actuators, contact accumulators, energy, ages, reproduction retry times, IDs,
lineage metadata, food particles and remaining energy, source locations and
schedules, field buffers, spawn accumulators, tick counters, and all random
generator states. Saving only a seed and successful genomes is insufficient to
resume a shared ecology midway through its history.

Use separate random streams for world generation, initialization, mutation,
reproduction, and evaluation. Record enough runtime information to investigate
GPU reproducibility differences; do not promise bit-identical trajectories across
different hardware or dependency versions. Videos preserve the actual observed
run independently of later replay.

## 11. Configuration, measurements, and calibration

Put numerical parameters in a versioned default TOML configuration rather than
scattering constants through code. Save all resolved values with each run,
including derived values and overrides. Changes to the defaults should include
the observation that motivated them.

Measure:

- Population, births, deaths, blocked births, lifespan, and generation distribution.
- Food remaining, spawned, expired, and absorbed; energy spent on each cost.
- Movement distance, propulsion expenditure, and food acquisition efficiency.
- Lineage diversity, genome variation, and actual reproductive success.
- Wall/contact frequency, overlaps, and nonfinite-state failures.
- Simulated seconds per wall-clock second, organism-steps per second, and memory.

First calibration batch: five seeds, 600 simulated seconds per seed. Confirm that
random founders sometimes acquire food and reproduce. In separate diagnostics,
check that a simple scripted forager can sustain itself using the same sensors,
actuators, resource supply, and costs. Diagnostic controllers do not seed or enter
the evolving population.

Tune food replenishment, metabolism, and food spacing first. Hold body and network
structure fixed while diagnosing the ecology. Check that resting indefinitely
does not dominate useful foraging under the chosen resource dynamics. If even the
scripted forager fails, investigate physics, sensing, or resource accessibility
before changing mutation parameters.

After short-run validation, run overnight experiments and retain independent
seeds. Population capacity, high frame rate, or a visually lively run are not
substitutes for the success criteria.

### Behavioral evaluation

Use separate evaluation environments that cannot affect live reproduction.
An initial evaluation batch uses 32 founder genomes and 32 saved descendant
genomes, each on eight held-out seeds for 120 simulated seconds. Test each founding
creature in its own dish using matched starting conditions and food-generation
rules, with genetic mutation disabled and fresh neural state. Retain the ordinary
energy and reproduction rules and distinguish the founding creature's metrics
from those of its offspring.

Compare energy acquired, propulsion cost, survival, and viable offspring. Evaluate
the same descendants with smell inputs disabled and, separately, sampled at
shuffled spatial locations. Report the variation across environments and
independent evolutionary runs, including failed runs. Avoid drawing a conclusion
from the single best lineage or recording.

Repeatable improvement over founders, together with a loss of advantage under
sensory disruption, is the V0 evidence target. Later tests can isolate the value
of recurrent state, plasticity, or communication.

## 12. Implementation sequence and validation

| Milestone | Deliverable | Completion check |
| --- | --- | --- |
| 1. Project and configuration | uv dependencies/lockfile, default TOML, seeded state, headless runner, GPU smoke check | Resolved configuration is saved; the numerical backend runs on the 5080 |
| 2. World and observation | Fixed bodies, propulsion, contacts, food, smell field, minimal 2D viewer | Scripted motion is stable; sensor responses follow food positions; display scale and zoom work |
| 3. Living population | Recurrent brains, energy, continuous births/deaths, mutation, lineage events | Multiple generations occur; energy and resource bookkeeping remain consistent |
| 4. Long runs | Batched execution, measurements, full checkpoints, resume, recording, headless acceleration | A short run resumes correctly and produces a playable recording with bounded memory use |
| 5. Calibration and evaluation | Diagnostic controllers, founder/descendant comparisons, sensory ablations, multiple seeds | The stated V0 success criteria can be assessed and their results are recorded |

Validate the mechanics that could otherwise produce false evolutionary success:

- Applied actuator responses, boundary handling, passive contacts, and timestep
  sensitivity.
- Field symmetry, interpolation, fixed normalization, and disappearance after
  source removal at the next field update.
- Food-energy conservation under contention and storage caps.
- Birth debits, failed-birth behavior, population capacity, and starvation order.
- Genome inheritance, mutation bounds, persistent recurrent state, and newborn
  state reset.
- Checkpoint/resume consistency within the documented runtime and tolerances.
- Independence of the simulated trajectory from viewer/recording settings for
  the same seed and numerical execution path.

Profile the working loop before optimizing kernels or raising population targets.
Keep hot state in batched tensors and transfer only observation data needed by the
viewer or recorder. Exact throughput, collision solver settings, recorder storage
budgets, and compatible GPU package versions are established during implementation
and calibration rather than assumed from GPU capacity.
