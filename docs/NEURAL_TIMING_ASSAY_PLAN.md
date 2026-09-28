# Controlled V24 timing and feedback assays

Status: all thirty trials complete and audited. See the
[results](NEURAL_TIMING_ASSAYS.md). The design and checks below were recorded
before launch.
All nine V24 continuations reached 1,800 seconds.
Inherited timing without direct timing mutation increased births in all three
starts: 1,785/1,112/2,101 versus homogeneous 1,727/994/1,563. This warrants a
controlled follow-up, although the communities retain only 1/2/1 founder
lineages and have no digestive-allocation scavenger specialists at the endpoint.

## Sources and comparison

Use all three `runs/v24-inherited-long-{1,2,3}` populations at 1,800 seconds.
These are selected treatment communities, not a representative sample of all
possible ecologies. Sample 192 living descendant genotypes with replacement
into each of two new environments, seeds **901 and 902**. Use the same sampled
genotypes and initial environment within each five-treatment comparison.
Run each transplant for **360 simulated seconds**, or until extinction.
Thirty community trials are planned; do not select only positive outcomes.
The three batch processes wrote `runs/v24-timing-assay-{1,2,3}`;
each ran its ten treatments sequentially and exited successfully.

1. **Native:** original timing genes and normal learning.
2. **Mean tau:** all active neurons within a genotype share a time constant
   equal to that genotype's original arithmetic mean active time constant.
3. **Permuted:** reassign the original timing genes among active neuron slots.
4. **No motor learning:** native genes and exploration, without acquired motor updates.
5. **Shuffled return:** native genes and learning, with return rates shuffled
   among controllers updating in the same batch.

The first three treatments share normal learning. Both learning controls retain
recurrent plasticity and learning-capacity costs. All six mutation probabilities
are zero, including timing mutation. Births, deaths, competition, body growth,
and selection continue. New organisms inherit the intervened genotype, with
fresh activity and learned states. No learned state is transplanted.
Assay lineage labels identify newly sampled founders, including separate copies
of the same genotype; they are not the historical source lineage labels.

## Timing interventions

Only active timing genes change. Existing weights, traits, masks, and dormant
timing genes retain their original values. For the mean-tau control, express the
mean factor through the inverse of the original timing development map, with
float-rounding protection at the inherited bound. Verify the original and
intervened mean time constants agree within float32 tolerance.

This controls the **arithmetic mean of intrinsic time constants**, not the
mean integration factor or the recurrent network's relaxation modes. Removing
timing variation necessarily changes the temporal computation; a performance
loss does not by itself demonstrate information storage or useful memory.

The permutation preserves the full timing multiset within each brain, including
its mean and variance. Its random stream is keyed to the original complete
genotype and a fixed external permutation seed, **907**. Identical genotypes
therefore get identical interventions across clones and environments. The
intervention does not consume any native world random stream. Some permutations
may leave identical timing values in place; report actual changed counts.

## Checks and interpretation

Before launch, verify gene/mean preservation, duplicate-genotype consistency,
pool-order independence, fresh state, frozen inheritance, common initial
physical state, complete output records, and an explicit incomplete stop record.
Archive the assay script and native package sources. Preserve source checkpoint,
pool, selected genotype, and script hashes.
Save every initial physical state as well as its final checkpoint, so the audit
can verify the common starting state and fresh acquired variables directly.

Audit all thirty completed trials, sample provenance, all six frozen mutation
settings, source archives, final checkpoints, and energy/resource/trophic
accounting. Primary readouts are cumulative births, final population, and
fresh-food absorption, with other food sources and lineage composition retained.
Report all six within-source/environment contrasts per intervention; do not
treat sampled creatures or cloned descendants as independent experimental seeds.

Native versus mean-tau tests the effect of the detailed timing distribution on
these genomes. Native versus permuted tests the importance of assigning timing
to particular neurons. Native versus learning controls tests the contribution
of motor adaptation, with the established caveat that shuffled feedback retains
population-wide signals and cannot mix singleton batches. Even favorable results
would be conditional community evidence, not a general intelligence claim.

```bash
uv run python scripts/assay_neural_timing.py \
  --source runs/v24-inherited-long-1 --output runs/v24-timing-assay-1 \
  --seeds 901 902 --seconds 360 --device cpu
```

Repeat for source seeds 2 and 3. Each process runs its ten trials sequentially
and saves checkpoints on normal completion or a handled stop. It refuses to
overwrite an existing output. A stopped assay must be audited before resuming
or rerunning any missing treatment; no automatic batch resume is implemented.
