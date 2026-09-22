# Emergent Embodied Evolution Simulator

## Technical Specification

## 1. Purpose

The goal of this project is to build a computationally feasible artificial-life environment in which embodied agents can evolve increasingly sophisticated behavior through natural selection and lifetime learning.

The system should not directly reward or hand-design behaviors such as navigation, memory, prediction, communication, cooperation, or tool use.

Instead, the simulator should provide:

- a continuous 2D world,
- physical bodies,
- local sensors,
- low-level motor effectors,
- persistent recurrent neural controllers,
- energy and homeostatic constraints,
- reproduction with heredity and mutation,
- and an ecology whose structure makes increasingly sophisticated behavior advantageous.

The central research question is:

> Can complex embodied behavior, internal memory, prediction, learning, and eventually social behavior emerge from relatively simple evolutionary and neural primitives?

The system is intentionally more abstract than biology. It does not attempt to reproduce molecular biology, realistic neural tissue, or detailed embryogenesis.

Instead, it aims to preserve the functional ingredients that may matter for embodied intelligence while remaining feasible on a consumer gaming GPU.

---

# 2. Core Abstraction Boundary

The simulator should distinguish between:

1. **physical embodiment**, and
2. **internal neural computation**.

The body exists physically inside the simulated world.

The nervous system initially does not.

Instead, each organism has an abstract persistent neural controller that can be thought of as a "brain in a vat."

This controller:

- receives signals from physically embodied sensors,
- receives signals representing internal physiological state,
- maintains persistent internal neural state,
- may alter its own weights during life,
- and outputs low-level signals to physically embodied motor effectors.

The controller should not directly move the organism through high-level commands.

It should control the body indirectly through muscles or equivalent low-level actuators.

Conceptually:

```text
WORLD
  ↓
PHYSICAL SENSORS
  ↓
ABSTRACT NEURAL CONTROLLER
  ↓
PHYSICAL EFFECTORS
  ↓
BODY PHYSICS
  ↓
WORLD
```

This abstraction is intended to preserve the core embodied intelligence loop while avoiding the computational and evolutionary complexity of explicitly growing physical neurons and axons.

---

# 3. Design Principles

## 3.1 Evolve solutions rather than specifying them

The simulator should expose primitive capabilities but avoid encoding high-level behaviors.

Allowed primitives include:

- sense local chemical concentration,
- sense touch,
- sense body deformation,
- sense internal energy,
- activate an individual muscle,
- emit a generic signal,
- alter synaptic strength,
- push a physical object.

Avoid explicit commands such as:

- move forward,
- move toward food,
- turn toward target,
- flee predator,
- communicate,
- remember,
- cooperate,
- build shelter,
- use tool.

These behaviors should arise through combinations of lower-level capabilities.

---

## 3.2 Abstract biology, preserve embodied computation

The simulator may explicitly provide:

- sensors,
- motor effectors,
- neural units,
- recurrent connections,
- metabolism,
- reproduction.

It should not directly provide:

- useful neural circuits,
- locomotion policies,
- sensorimotor mappings,
- spatial representations,
- memory strategies,
- useful learning rules,
- communication protocols.

The initial abstraction boundary assumes that evolution does not need to rediscover the physical existence of neurons.

Instead, evolution must discover how to organize and use a neural system.

---

## 3.3 Physical embodiment should remain meaningful

Although the neural controller is abstract, its inputs and outputs should be grounded in physical morphology.

The controller should not receive privileged global information such as:

```text
absolute world position
absolute heading
food direction
distance to nearest food
predator distance
global map
```

unless those quantities are physically sensed by an evolved sensor.

Similarly, the controller should not output:

```text
desired velocity
desired world-space direction
move_forward()
turn_left()
```

Instead, outputs should control physical effectors such as:

```text
muscle 1 activation
muscle 2 activation
joint torque
contractile edge activation
fin activation
signal emitter intensity
```

This preserves the requirement that evolution must solve sensorimotor control.

---

## 3.4 Continuous existence

Agents should exist continuously rather than as isolated inference episodes.

An organism maintains persistent:

- physical state,
- neural state,
- physiological state,
- memory,
- synaptic modifications,
- lifetime experience.

The controller should not reset between environmental observations.

---

## 3.5 Local information only

Organisms should never receive global world state.

Sensors expose only information physically available to the body.

This creates evolutionary pressure for:

- memory,
- exploration,
- spatial inference,
- temporal inference,
- prediction,
- internal representations.

---

## 3.6 No explicit behavioral reward

The long-term evolutionary objective is survival and reproduction.

The simulator should avoid direct reward shaping such as:

```text
+1 find food
+5 avoid predator
+10 communicate
```

Instead, useful behavior should affect:

```text
energy
damage
survival
reproductive success
```

---

# 4. Scope

## 4.1 Version 0 objective

The first implementation should answer:

> Can evolution discover neural controllers and body-control strategies that produce coordinated locomotion and resource-seeking behavior?

Version 0 does not require:

- physically embedded neurons,
- neuron spatial development,
- predators,
- communication,
- sexual reproduction,
- tool use,
- complex social behavior,
- realistic embryogenesis,
- realistic neural physiology,
- language.

---

# 5. World Model

## 5.1 Geometry

The environment is a continuous 2D plane.

Suggested initial size:

```text
1024 × 1024 world units
```

Configurable topology:

- bounded walls,
- toroidal wrapping.

Toroidal topology may be preferred during early experiments.

---

## 5.2 Physical regime

The initial environment should behave more like a viscous fluid or microscopic environment than terrestrial rigid-body physics.

Strong damping should dominate motion.

Approximate regime:

```text
velocity ∝ applied force
```

rather than momentum-dominated Newtonian mechanics.

Benefits:

- easier locomotion,
- stable simulation,
- cheaper integration,
- fewer irrelevant mechanical failures.

Gravity is excluded from Version 0.

---

# 6. Environmental Fields

The environment may contain continuous scalar fields represented on GPU grids.

Initial candidates:

```text
food / nutrient field
temperature
generic chemical fields
```

Later versions may add:

```text
light
toxins
pheromones
oxygen
fluid currents
seasonal variables
```

Fields may:

- diffuse,
- decay,
- be generated by sources,
- be consumed,
- be modified by organisms.

---

# 7. Resource Model

Version 0 should support discrete food particles or persistent resource patches.

Each resource may contain:

```text
position
radius
energy content
regeneration behavior
```

Food should be spatially structured rather than uniformly distributed.

Example:

```text
resource clusters form
clusters persist temporarily
clusters eventually disappear
new clusters appear elsewhere
```

This avoids rewarding purely stationary strategies.

---

# 8. Organism Architecture

Each organism consists of three major subsystems:

```text
PHYSICAL BODY
ABSTRACT CONTROLLER
GENOME
```

Conceptually:

```text
                GENOME
                  │
          ┌───────┴────────┐
          ▼                ▼
       BODY            CONTROLLER
          │                │
      sensors ───────────► │
          │                │
          │ ◄────────── motors
          ▼                │
        WORLD ◄──────── BODY PHYSICS
```

---

# 9. Physical Body

## 9.1 Initial body representation

The first body representation should use a deformable graph.

An organism consists of:

```text
body nodes
elastic edges
contractile edges
sensor attachments
motor effectors
```

---

## 9.2 Body nodes

Each node contains:

```text
position
velocity
radius
effective mass
collision state
sensor attachments
```

Recommended Version 0 range:

```text
8–32 body nodes
```

Later:

```text
32–256+
```

---

## 9.3 Structural edges

Edges connect nodes.

Each edge may contain:

```text
node A
node B
rest length
stiffness
damping
maximum extension
```

---

## 9.4 Contractile edges

Some edges act as muscles.

A motor signal changes the preferred rest length or generated force.

For example:

```text
rest_length =
    baseline_length
    × f(motor_activation)
```

The controller therefore cannot directly command body movement.

It must discover coordinated muscle activation patterns.

---

# 10. Sensors

Sensors are physically attached to the body.

Their number and placement may eventually evolve.

Initial sensor types include:

## 10.1 Resource sensors

Measures local nutrient concentration.

Possible outputs:

```text
local concentration
local gradient component
```

Prefer arrangements where direction must be inferred from multiple spatially separated sensors.

---

## 10.2 Touch sensors

Detect contact with:

- walls,
- objects,
- other organisms,
- resources.

---

## 10.3 Proprioceptive sensors

Provide local body state such as:

```text
edge extension
joint angle
relative node position
muscle stretch
local velocity
```

This allows the controller to infer how its own body responds to motor commands.

---

## 10.4 Interoceptive sensors

Expose physiological variables such as:

```text
energy
damage
fatigue
temperature
```

These provide an internal body state.

---

# 11. Controller Architecture

## 11.1 Controller abstraction

The neural controller initially has no physical position.

It receives a vector or structured set of sensor channels and produces motor activations.

The controller should nevertheless be:

- stateful,
- recurrent,
- evolvable,
- potentially plastic.

---

## 11.2 Version 0 controller

Recommended initial architecture:

> small recurrent neural network using leaky continuous units or simple recurrent MLP units.

Example:

```text
h[t+1] = f(
    sensor[t],
    interoception[t],
    h[t]
)

motor[t] = g(h[t])
```

The hidden state persists throughout the organism's life.

---

# 12. MLP vs Recurrent Controller

A purely feedforward controller:

```text
action[t] = MLP(sensor[t])
```

should be supported as a baseline but not used as the primary architecture.

It represents a reflex system and cannot naturally maintain temporal state.

The preferred controller is recurrent:

```text
h[t+1] = F(h[t], input[t])

output[t] = G(h[t])
```

This allows:

- working memory,
- oscillation,
- temporal integration,
- sequence-sensitive behavior,
- internal context.

---

# 13. Neural Unit Models

The controller architecture should be modular.

Supported or planned unit types:

### A. Continuous recurrent units

Cheap and recommended for Version 0.

Example:

```text
h[t+1] =
    (1 - α) h[t]
    + α tanh(W h[t] + U x[t] + b)
```

---

### B. Leaky integrate-and-fire neurons

Useful when investigating spiking dynamics.

Conceptually:

```text
V[t+1] =
    leak × V[t]
    + weighted input
```

If:

```text
V > threshold
```

the neuron emits a spike and resets.

---

### C. Continuous-time units

Potentially useful for evolving neurons with different intrinsic timescales.

---

## 13.1 Initial recommendation

Use:

> continuous recurrent leaky units first.

Reasons:

- simpler,
- faster,
- easier to debug,
- differentiable if needed,
- still capable of persistent internal dynamics.

SNNs should remain a first-class optional controller type.

---

# 14. Neural Topology Evolution

The controller should not necessarily remain fixed.

The genome may encode:

```text
number of neural units
connection topology
connection weights
unit time constants
biases
activation parameters
plasticity parameters
```

Mutations may:

```text
add neural unit
delete neural unit
add connection
remove connection
modify weight
change time constant
change plasticity parameter
```

Thus neural architecture can evolve even though neurons have no physical spatial position.

---

# 15. Body-Brain Coevolution

Body morphology and neural architecture should be able to evolve together.

An important design principle is:

> morphology defines the controller's sensory and motor interface.

If evolution adds a sensor:

```text
new body sensor
→ new controller input channel
```

If evolution adds a muscle:

```text
new body motor
→ new controller output channel
```

The neural controller must therefore adapt to changes in body morphology.

This preserves genuine body-brain coevolution without requiring physical neural tissue.

---

# 16. Modular Genome

The genome should conceptually contain three modules:

```text
GENOME

├── Body genome
│     morphology
│     sensors
│     muscles
│     body development
│
├── Brain genome
│     neural topology
│     dynamics
│     initial connectivity
│
└── Learning genome
      plasticity
      neuromodulation
      weight decay
      memory timescales
```

These modules may mutate semi-independently.

This structure allows experimental ablations.

---

# 17. Experimental Ablations

The system should support experiments such as:

## Experiment A

```text
fixed body
evolving brain
```

Tests neural evolution alone.

---

## Experiment B

```text
evolving body
fixed controller architecture
```

Tests morphological adaptation.

---

## Experiment C

```text
evolving body
evolving brain
```

Tests morphology-controller coevolution.

---

## Experiment D

```text
evolving body
evolving brain
evolving learning rules
```

Tests whether lifetime learning becomes an evolutionary advantage.

---

# 18. Lifetime Learning

Lifetime learning should eventually be available to all controller types.

Weights may consist of:

```text
genetic initial weight
+
lifetime plastic component
```

For example:

```text
w_current =
    w_genetic
    + Δw_lifetime
```

Lifetime modifications should not automatically become genetically inherited.

This distinguishes:

```text
learning during life
```

from:

```text
evolution across generations
```

---

# 19. Evolvable Plasticity

Each connection may contain:

```text
weight
learning rate
decay
plasticity coefficients
modulatory sensitivity
```

A general local update rule may use:

```text
presynaptic activity
postsynaptic activity
local neural state
global modulatory signals
prediction error
```

The genome may control the coefficients governing this rule.

Evolution therefore searches not only over network solutions, but also over:

> how individual organisms learn.

---

# 20. Multiple Timescales

The neural controller should support internal variables operating at different timescales.

Examples:

```text
fast neural activity
medium recurrent state
slow neural state
lifetime synaptic plasticity
generational evolution
```

Different units may have evolvable leak constants.

Conceptually:

```text
τ_fast
τ_medium
τ_slow
```

This may allow evolution to discover:

- reflex-like responses,
- locomotor oscillators,
- working memory,
- slowly changing motivational states.

---

# 21. Neuromodulation

A later extension should introduce a small number of generic internal modulatory channels.

For example:

```text
M1
M2
M3
M4
```

These channels are not assigned semantic meaning.

They may influence:

```text
neuron gain
plasticity
motor activity
sensory sensitivity
memory persistence
```

Evolution determines their function.

Potential emergent roles may resemble:

```text
hunger
arousal
reward
surprise
fear-like state
```

without those concepts being explicitly programmed.

---

# 22. Internal Physiology

Each organism should initially maintain:

```text
energy
age
damage
```

Later:

```text
temperature
fatigue
hydration
toxin level
reproductive readiness
```

These variables should be accessible to the controller only through interoceptive channels.

---

# 23. Energy Economy

Energy is the primary ecological currency.

Energy costs may include:

```text
baseline metabolism
muscle activation
growth
neural activity
reproduction
```

Food increases energy.

Death occurs if:

```text
energy <= 0
```

or damage exceeds a threshold.

This allows behavioral effectiveness to influence reproductive success without task-specific rewards.

---

# 24. Predictive Learning

A later controller extension should allow organisms to predict future sensory input.

Given:

```text
current neural state
current sensation
current motor action
```

predict:

```text
next sensory state
next interoceptive state
```

Prediction error may serve as:

- a self-supervised learning signal,
- a modulatory signal,
- a measure of surprise.

This could encourage organisms to build internal world models.

---

# 25. Reproduction

Two modes should eventually be supported.

## 25.1 Generational evolutionary mode

Used first for debugging and scientific analysis.

Workflow:

```text
initialize population
↓
simulate lifetime
↓
evaluate reproductive fitness
↓
select parents
↓
mutate / recombine
↓
next generation
```

Suggested initial population:

```text
1,000–10,000 organisms
```

---

## 25.2 Continuous ecological mode

Long-term preferred mode.

Organisms reproduce asynchronously inside the simulation.

Possible rule:

```text
if energy > reproduction threshold:
    create offspring
    transfer energy to offspring
```

Offspring inherit a mutated genome.

Fitness then becomes implicit:

```text
number of viable descendants
```

rather than externally calculated.

---

# 26. Genome Encoding

The genome should avoid directly encoding every body coordinate and every network parameter whenever possible.

Instead, use compact developmental or generative encodings.

The body and controller may initially use separate developmental encodings.

---

# 27. Body Development Encoding

The body genome may encode graph-generation rules such as:

```text
create node
duplicate segment
branch
connect nodes
mark edge structural
mark edge contractile
attach sensor
attach effector
```

The adult body is generated before simulation begins.

Later versions may allow development during life.

---

# 28. Controller Development Encoding

The brain genome may encode rules for generating the controller.

Possible approaches:

### Direct sparse graph encoding

Genome explicitly encodes:

```text
unit list
connection list
unit parameters
```

Best for early implementation.

### Indirect developmental encoding

Genome encodes rules such as:

```text
duplicate neural module
connect sensor class to module
connect module A to module B
repeat motif
```

Useful once architecture complexity increases.

---

# 29. Morphology-Dependent Interfaces

The controller should not assume a fixed number of sensors or motors in the long term.

The implementation should support dynamic or padded interfaces.

For example:

```text
MAX_SENSORS = N
MAX_MOTORS = M
```

Unused channels are masked.

This allows the GPU representation to remain regular while morphology evolves.

---

# 30. Mutation Operators

Supported body mutations may include:

```text
add node
remove node
add structural edge
remove edge
add muscle
remove muscle
move sensor
add sensor
remove sensor
```

Brain mutations may include:

```text
add neural unit
remove neural unit
add synapse
remove synapse
modify weight
modify bias
modify time constant
modify plasticity
```

Learning-genome mutations may include:

```text
change learning rate
change decay
change modulatory sensitivity
change plasticity coefficients
```

---

# 31. Environment Variability

Each organism should experience a slightly different world.

Examples:

```text
different food locations
different obstacle layout
different starting position
different resource timing
different nearby organisms
```

This discourages memorized coordinate-based solutions.

The environment should remain:

> variable but statistically learnable.

---

# 32. Temporal Structure

The world should contain predictable dynamics over time.

Examples:

```text
periodic food blooms
moving resource fields
resource regeneration
day/night cycles
temperature cycles
currents
```

This creates selection pressure for:

```text
memory
prediction
temporal inference
internal state
```

---

# 33. Environmental Manipulation

Later versions should contain persistent physical objects.

Objects may have:

```text
position
shape
mass
friction
```

Organisms interact only through physical forces.

There is no:

```text
use_tool()
build()
hide()
```

Possible emergent behavior may include:

```text
resource caching
blocking competitors
creating barriers
moving shelter materials
trapping prey
```

---

# 34. Communication

Communication should arise from generic signaling capabilities.

Possible signal channels:

```text
chemical emission
scalar acoustic-like pulse
light/color signal
contact signal
```

These signals have no predefined meaning.

Evolution determines whether they become useful.

---

# 35. Predation and Ecological Arms Races

Predators should be introduced only after robust resource-seeking behavior exists.

Predator-prey systems may generate pressure for:

```text
prediction
pursuit
avoidance
camouflage
group behavior
signaling
memory
```

Coevolution should provide an automatically escalating curriculum.

---

# 36. Controller Interface Philosophy

The controller interface is one of the most important design choices in the system.

A good interface exposes:

```text
local sensory consequences
internal physiological consequences
low-level actuator controls
```

A poor interface exposes:

```text
semantic world state
global navigation information
high-level actions
```

Preferred:

```text
left sensor = 0.72
right sensor = 0.48

muscle 1 = 0.81
muscle 2 = 0.15
```

Avoid:

```text
food is 32° left

turn_left()
```

---

# 37. Future Physical Nervous-System Mode

A later experimental mode may replace the abstract controller with a physically embedded nervous system.

Possible extensions:

```text
neuron spatial positions
axon growth
local connection formation
chemical guidance
ganglia
distributed neural tissue
```

This mode would ask a different research question:

> Can physical nervous-system organization itself emerge?

It should not be required for early versions.

---

# 38. Planned Abstraction Progression

Recommended development path:

```text
V0
fixed-size recurrent controller
evolving weights
+
physical body and low-level motors

        ↓

V1
evolving neural topology
+
evolvable morphology

        ↓

V2
body-brain coevolution
+
lifetime plasticity

        ↓

V3
evolvable learning rules
+
neuromodulation
+
predictive learning

        ↓

V4
physically embedded nervous-system development
```

The project should aim to reach V2 quickly before investing heavily in V4.

---

# 39. Simulation Frequencies

Subsystems may run at different rates.

Example:

```text
physics:
30–60 Hz

neural controller:
30–100 Hz

environment fields:
5–20 Hz

metabolism:
5–10 Hz

reproduction:
event-driven
```

This reduces unnecessary computation.

---

# 40. GPU Architecture

The simulator should be GPU-native.

Avoid per-organism Python objects in hot paths.

Prefer padded tensors.

Example:

```text
body_positions[
    organisms,
    max_nodes,
    2
]

body_edges[
    organisms,
    max_edges,
    properties
]

sensor_values[
    organisms,
    max_sensors
]

neural_state[
    organisms,
    max_neurons
]

synapses[
    organisms,
    max_neurons,
    max_connections
]

motor_outputs[
    organisms,
    max_motors
]

genomes[
    organisms,
    genome_parameters
]
```

Masks indicate active elements.

---

# 41. Technology

Initial prototype:

```text
Python
JAX or PyTorch
GPU batching
```

Possible later optimization:

```text
Triton
custom CUDA
Taichi
custom JAX kernels
```

Implementation speed should be prioritized before low-level optimization.

---

# 42. Performance Target

A plausible initial target on a modern gaming GPU:

```text
1,000–10,000 organisms

8–32 body nodes

16–64 neural units

4–16 neural connections per unit

30–100 neural updates / second
```

Primary performance metric:

```text
organism-steps / wall-clock second
```

Rendering should not constrain simulation speed.

---

# 43. Rendering

Real-time visualization should support:

```text
body shape
sensors
muscle activation
resource locations
environmental fields
organism lineage colors
neural activity summaries
```

Headless mode should support accelerated evolutionary runs.

---

# 44. Instrumentation

Track:

```text
population size
lifespan
energy acquisition
offspring count
body size
sensor count
muscle count
neural unit count
connection count
movement distance
movement efficiency
neural activity
genetic diversity
lineage diversity
```

---

# 45. Neural Analysis Tools

Because the abstract brain is easier to inspect than a physically simulated neural system, the simulator should expose strong interpretability tooling.

Useful analyses:

```text
neuron activation traces
recurrent-state trajectories
connection graphs
weight changes over life
motor correlations
sensor-neuron correlations
ablation of individual units
ablation of connections
```

This may reveal emergent motifs such as:

```text
oscillators
integrators
memory units
left-right comparators
context states
```

---

# 46. Behavioral Evaluation

Controlled tests should be separate from evolutionary fitness.

Possible evaluations:

## Locomotion test

Measure movement without resource incentives.

## Chemotaxis test

Place resources in varying locations.

## Memory test

Present a cue, remove it, then test delayed behavior.

## Temporal prediction test

Expose an organism to periodic or moving resources.

## Generalization test

Evaluate on unseen maps.

## Lifetime learning test

Compare behavior early versus late in life.

## Morphological robustness test

Perturb body state or environment.

These tests should normally not affect reproduction.

---

# 47. Replay System

Store sufficient information to reproduce interesting organisms:

```text
random seed
genome
world configuration
lineage
initial conditions
```

A selected organism should be rerunnable independently.

---

# 48. Lineage Tracking

Each organism should record:

```text
unique ID
parent ID
birth time
generation
genome hash
lineage ID
```

Potential later analyses:

```text
phylogenetic trees
innovation events
behavioral transitions
species divergence
```

---

# 49. Initial Research Milestones

## Milestone 0

Random bodies simulate stably.

Success:

```text
soft-body physics remains numerically stable
```

---

## Milestone 1

Controllers produce coordinated locomotion.

Success:

```text
evolved organisms move farther or more efficiently than random baselines
```

---

## Milestone 2

Resource-seeking behavior emerges.

Success:

```text
organisms reliably acquire distributed resources
```

---

## Milestone 3

Body and controller coevolve.

Success:

```text
different morphologies produce different effective control strategies
```

---

## Milestone 4

Persistent recurrent state provides an advantage.

Success:

```text
recurrent controllers outperform stateless controllers
```

---

## Milestone 5

Lifetime plasticity improves behavior.

Success:

```text
organisms become measurably better during the same lifetime
```

---

## Milestone 6

Temporal prediction becomes useful.

Success:

```text
organisms exploit temporally structured environmental patterns
```

---

## Milestone 7

Ecological interactions emerge.

Success:

```text
competition, pursuit, avoidance, or specialization arise
```

---

## Milestone 8

Communication emerges.

Success:

```text
one organism's emitted signal systematically changes another organism's behavior in a fitness-relevant way
```

---

# 50. Explicit Non-Goals for Early Versions

Do not initially attempt:

```text
molecular biology
realistic metabolism
realistic embryology
physical axon growth
Hodgkin-Huxley neurons
photorealistic rendering
language
human-level cognition
large transformers
complex gravity-driven locomotion
real biological evolutionary timescales
```

---

# 51. Key Open Design Questions

## A. Body representation

Candidates:

1. node-and-spring soft body,
2. connected circular cells,
3. articulated segments.

Current preference:

> node-and-spring soft body.

---

## B. Controller model

Candidates:

1. feedforward MLP,
2. recurrent MLP,
3. leaky recurrent network,
4. spiking neural network,
5. continuous-time recurrent network.

Current preference:

> small leaky recurrent neural controller.

---

## C. Controller evolution

Open question:

Should V0 use:

```text
fixed architecture + evolved weights
```

or immediately support:

```text
evolving topology + evolving weights
```

Recommended progression:

```text
fixed topology
→ sparse topology mutation
→ unit addition/removal
```

---

## D. Morphological evolution

Open question:

Should V0 begin with:

```text
fixed body
```

or:

```text
simple evolving morphology
```

Recommended approach:

Start with a fixed or lightly parameterized morphology to validate the controller and physics, then introduce full morphology evolution.

---

## E. Lifetime learning

Recommended progression:

```text
no plasticity baseline
→ fixed plasticity rule
→ evolvable plasticity parameters
→ evolvable learning rule
→ prediction-modulated plasticity
```

---

## F. Reproduction

Recommended progression:

```text
generational evolution
→ continuous ecological reproduction
```

---

# 52. Long-Term Vision

The desired long-term system is a persistent artificial ecology in which:

```text
genomes produce bodies
genomes produce initial brains
bodies generate sensations
brains generate actions
actions change the world
experience changes brains
world structure determines survival
survival determines reproduction
reproduction changes future genomes
```

The complete loop is:

```text
GENOME
   ↓
BODY + INITIAL CONTROLLER
   ↓
SENSATION
   ↓
PERSISTENT INTERNAL STATE
   ↓
ACTION
   ↓
BODY PHYSICS
   ↓
ENVIRONMENTAL CONSEQUENCES
   ↓
LEARNING
   ↓
SURVIVAL / REPRODUCTION
   ↓
GENOME
```

The system is successful if increasingly complex behavior emerges without being explicitly specified.

The ultimate objective is not to recreate biological anatomy.

It is to create a computational world rich enough that evolution can discover its own solutions for:

- perception,
- locomotion,
- memory,
- temporal inference,
- prediction,
- adaptation,
- learning,
- communication,
- cooperation,
- competition,
- and potentially more general forms of embodied intelligence.
