# Research and planning

[Project overview](../README.md) · [Usage guide](../docs/USAGE.md) · [Development guide](../docs/DEVELOPMENT.md)

Specifications, research notes, experiment plans, and results are collected here.
The current implementation is **V25 / package 0.26.0**. Research is **paused at the
user's request**; all launched simulations and audits have finished. Older notes
record the plans and decisions at their time, and do not authorize resuming work.

Start with the [current handoff](HANDOFF.md) for evidence and outstanding work,
or the [experiment overview](EXPERIMENT_OVERVIEW.md) for a narrative tour. Added
mechanics do not necessarily improve survival or learning. Inherited V24 timing
with V25 recurrent reinforcement disabled remains the reference configuration.

## Foundations and development history

| Document | Contents |
| --- | --- |
| [Original specification](spec.md) | Long-term vision and early design ideas |
| [V0 implementation plan](PLAN.md) | Initial mechanics, parameters, milestones, and success criteria |
| [V0 validation](VALIDATION.md) | Original checks, persistence results, and limitations |
| [V1–V5 evolution record](EVOLUTION.md) | Ecological and morphological development, experiments, and evidence |
| [Continuing experiments](CONTINUATION.md) | Later neural, ecological, and interface development history |

## Resources, bodies, and exploration

| Experiment | Design and results | Planning |
| --- | --- | --- |
| V16 dynamic food | [Irregular patches, drifting sources, local depletion/recovery](DYNAMIC_RESOURCES.md) | [Simple producers as a possible later step](DYNAMIC_RESOURCES.md#possible-later-step-simple-producers) |
| V17 foraging | [Scarce valuable meals, handling rates, and sensory encoding](FORAGING.md) | — |
| V18 carried food | [Digestion while moving](CARRIED_FOOD.md) | — |
| V19 neural variation | [Founder graphs, mutation, and motor learning](NEURAL_VARIATION.md) | — |
| V20 sensing | [Sensory footprint experiments](SENSOR_RADIUS.md) | — |
| V21 exploration | [Temporally persistent motor exploration](PERSISTENT_EXPLORATION.md) | [Plan](COHERENT_EXPLORATION_PLAN.md) |

## Prediction, memory, and recurrent learning

| Experiment | Design and results | Planning and research |
| --- | --- | --- |
| V22 energetic prediction | [Value prediction](VALUE_PREDICTION.md) | [Plan](VALUE_PREDICTION_PLAN.md) |
| V23 sensory prediction | [Sensory value features](SENSORY_VALUE.md) | [Plan](SENSORY_VALUE_PLAN.md) |
| Shorter prediction horizon | [Results](PREDICTION_HORIZONS.md) | [Plan](PREDICTION_HORIZON_PLAN.md) |
| V24 neuron timing | [Inherited response times](NEURAL_TIMESCALES.md) | [Plan](NEURAL_TIMESCALES_PLAN.md) |
| V24 controlled timing assays | [Genotype and feedback comparisons](NEURAL_TIMING_ASSAYS.md) | [Plan](NEURAL_TIMING_ASSAY_PLAN.md) |
| V25 recurrent reinforcement | [Implementation and first screen](RECURRENT_LEARNING.md) | [Literature and diagnostic](RECURRENT_LEARNING_RESEARCH.md), [integration plan](RECURRENT_LEARNING_PLAN.md) |
| V25 quieter/gentler learning | [Completed rate/noise sweep](QUIET_RECURRENT_LEARNING.md) | [Credit reassessment](RECURRENT_CREDIT_RESEARCH.md) |
| Passive representation learning | [Completed comparison](REPRESENTATION_LEARNING_PLAN.md#completed-outcome) | [Prospective plan](REPRESENTATION_LEARNING_PLAN.md), [literature and derivatives](RECURRENT_CREDIT_RESEARCH.md) |

## Evidence and reproduction

Figures and screenshots remain under [docs/](../docs/); compact audited records
remain under [docs/results/](../docs/results/). The reports link their supporting
files. Raw checkpoints, full logs, and videos under `runs/` are local artifacts
excluded from Git and may be absent in a fresh checkout.

Use the [preset catalog](../docs/USAGE.md#configure-an-experiment) to choose a
configuration and the [development guide](../docs/DEVELOPMENT.md) for assays,
controls, plotting, and genome transfer. Commands in this archive run from the
repository root. Historical presets and success criteria remain documented even
when later experiments supersede them.
