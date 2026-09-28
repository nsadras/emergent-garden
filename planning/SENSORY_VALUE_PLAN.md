# V23 plan: direct sensory features for value prediction

Status: implementation, nine pilots, three native forecast forks, and both
three-community passive comparisons are complete. The
[results](SENSORY_VALUE.md) retain all treatments. Sensory access modestly improves
paired forecasts, but neither ecological success nor useful motor adaptation is
reliable. Faster prediction learning increases error and is not promoted.

V22's learned values do not reliably forecast later returns or improve births.
One possible limitation is the representation: its critic sees only filtered
recurrent activity and a bias. Gut contents and local fields may be useful before
random inherited recurrent circuits preserve them. Test direct access to those
already available observations before expanding the inherited brain.

Add an optional `motor_value_inputs = 1` feature path to the acquired value
head. Concatenate current hidden activity, the **actual 42 controller inputs**,
and bias, then normalize the entire vector to unit length. This gives 75 value
weights instead of 33. Inputs must include the same sensing interventions as the
actor; no unobserved resource, future schedule, or desired steering is supplied.
The actor, genome, observation interface, exploration, mutation, and energy laws
stay at the V22 direct-return settings. New modules and children start with zero
value weights; learned values remain lifetime state.

Zero is the neutral default and must exactly preserve V22. It also keeps the old
state shape. A disabled critic must preserve the old running-mean policy even
when sensory features are configured. As in V22, this representation screen
charges no extra critic-specific energy cost. Whole-vector normalization changes
the relative feature scale as well as access to information; results cannot
attribute any effect solely to the additional information.

Before ecological screening, verify correct sensory routing and normalization,
no instantaneous actor change from an untrained head, bounded predictions,
fresh inherited state, erasure behavior, and exact checkpoint replay. Check
disabled/neutral parity and the live observer, including CPU/CUDA mechanics.

Run three matched 600-second starts for each treatment:

1. Hidden features only, direct net returns: neutral V23/V22 reference.
2. Hidden plus actual sensory inputs, direct net returns.
3. The same expanded features, with shuffled learning returns.

Use seeds 1/2/3, matching founder genomes and the sparse V19 learning habitat.
Record population, births, food uptake, critic bounds, and all energy/resource
accounts. Reproduce the existing V22 direct-return trajectories under neutral
V23 settings. Compare predictions with observed subsequent discounted returns,
including deaths, if viable cohorts remain. Community differences alone do not
establish that the expanded readout learns a better predictor.

A useful diagnostic follow-up is to train both readouts passively on exactly
the same creatures' trajectories, so improved prediction can be distinguished
from changed populations or behavior. Longer ecological or fixed-genotype
fitness runs should follow evidence of useful prediction, not merely larger
circuits. Keep negative outcomes and source/checkpoint provenance.

## Follow-up after the first paired diagnostic

All nine pilots completed with mixed effects. Both fresh predictors were then
trained passively for 100 seconds in each of the three V22 direct-return
communities, and forecast the following 100 seconds for the same mature cohort.
Sensory features reduced forecast MSE by about 1.2%, 4.7%, and 6.0% relative to
hidden-only features. Both predictors still lost to predicting zero in two of
the three communities. This motivates one focused learning-timescale comparison:
repeat the identical passive experiment with a maximum critic rate of .1 instead
of .02. Keep the two representations, source checkpoints, warmup, horizon,
eligibility, and bounds fixed. These shadow updates cannot change actions.
Record both rates and compare complete physical states as an additional check.
