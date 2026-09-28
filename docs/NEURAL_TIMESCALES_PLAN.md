# V24 plan: inherited response times within a recurrent circuit

Status: implemented as V24 / package 0.25.0. All twelve 600-second pilots,
including the three fully neutral compatibility runs, and the paired circuit
pulse probe are complete. The branch addresses the user's interest in
greater neural and mutational variation by varying recurrent dynamics within
one brain, before adding additional hidden layers.

## Hypothesis and scope

Before V24, every neuron used the organism's inherited `memory_tau`, developed as
`.2 + 4.8 * sigmoid(trait)`. Different organisms can have different response
times, but their individual neurons share one time constant. A mixture of fast
and slow neurons may represent immediate sensory changes and longer temporal
context in the same circuit.

[Perez-Nieves et al. (2021)](https://www.nature.com/articles/s41467-021-26022-3)
report benefits from heterogeneous neural time constants in trained spiking
networks, especially on temporally structured tasks. Their networks, training,
and tasks differ from this evolved, leaky-rate ecology. The paper motivates an
experiment; it does not establish a benefit here. No spike model or supervised
target is proposed.

## Initial mechanism

Append one inherited timing gene per hidden slot, after the existing structural
masks. The 32-slot preset would contain 5,304 values, versus 5,272 previously.
Existing weights, body traits, input/output meanings, and mask offsets retain
their positions. Express a neuron's response time as:

```text
tau_i = body_memory_tau * exp(log(timing_range) * tanh(timing_gene_i))
alpha_i = 1 - exp(-controller_dt / tau_i)
h_i = (1 - alpha_i) h_i + alpha_i tanh(drive_i)
```

An initial `timing_range` of 4 permits factors between one quarter and four
times the existing body time constant. Range 1 is the neutral default and must
take the exact legacy update path. Zero genes also express the body time
constant. All inherited values remain bounded by the existing gene limit.

Initial heterogeneous genes: independent Gaussian draws with standard deviation
.5, from a separate checkpointed random stream. Initial screening mutation:
per-gene probability .1 and Gaussian sigma .15 per birth. These are provisional
screening parameters, not an optimum. Changes in timing remain inherited;
neural activity, synaptic plasticity, and motor/value learning state still reset
at birth. No within-lifetime optimization of time constants is proposed initially.

Neuron duplication must copy its timing gene as well as incoming weights and
bias before splitting outgoing influence. Otherwise the existing claim of
function-preserving growth would no longer hold. Deleted/dormant slots are not
expressed. Genome transfers from older versions initialize the new genes to
zero; V24 transfers preserve timing genes for existing slots.

This changes dynamics, not neuron or synapse counts. Keep their existing energy
costs. The first screen adds no separate gene-storage or timing-computation
charge; disclose that simplification.

## Comparison and verification

Use the viable V22 disabled-value-head reference (`v22-baseline.toml`), which
retains V19's gentle motor learning, independent exploration, varied founder
circuits, and carried meals. This avoids combining the new representation with
an unproven critic treatment. The user's V16 working configuration stays intact.

Compare three matched 600-second starts per group:

1. Homogeneous expression, timing range 1.
2. Heterogeneous expression, range 4, without direct timing-gene mutation.
3. Heterogeneous expression with timing-gene mutation.

Use identical initial genes in all groups, including timing genes, and matched
founder/food randomness. The homogeneous group carries unexpressed timing genes.
The second group still permits selection and structural duplication/deletion;
it is not a fixed-population or mutation-free control. Preserve the historical
V22 trajectories under a separate fully neutral V24 configuration, comparing
all common state and measurements while accounting for the additional genes.

Before ecological runs, verify analytical step/decay responses, finite bounded
integration, neutral compatibility, node duplication, mutation isolation,
inheritance and version transfers, exact resume, and inactive-slot behavior.
Update the observer to reconstruct the actual per-neuron integration factors;
show the selected neuron's time constant without adding a scroll bar. Report
the distribution across active neurons and variation within organisms, with
clear weighting. Check CPU and CUDA separately and retain source archives.

A synthetic temporal-response probe should distinguish added response diversity
from claims about useful memory. Follow ecological improvements with matched
learning/feedback controls before calling them adaptive intelligence. Broader
spatially structured or reflection-equivariant circuits remain later options;
they are not part of this first timing experiment.

## Continuation decision after the initial screen

The no-direct-timing-mutation treatment improves births in two of three starts;
timing mutation improves only one, but all nine treatment populations remain
alive at 600 seconds. Extend **all three seeds of all three treatments** to
1,800 seconds, by resuming each exact checkpoint for 1,200 additional seconds.
The extra fully neutral compatibility runs need no separate continuation.

Report persistence, births, lineage counts, trophic composition, neural size,
and time-constant distributions for every continuation, including extinction.
Treat them as extensions of the same three evolutionary starts, not additional
replicates. Do not select only the successful seed-3 mutation population.
No new physics or neural setting changes during these continuations.

Different timing factors also shift each brain's mean response time. These
comparisons therefore do not isolate the benefit of within-brain heterogeneity
from a change in average response speed. Any later claim of useful adaptation
needs matched genotype and feedback interventions; no such claim follows from
more births or longer neural responses alone. The continuation outcomes will
determine the next controlled assay.
