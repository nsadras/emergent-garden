# V16: moving sources and renewable fertility

Start the new preset with:

```bash
uv run garden run --config configs/v16.toml --seed 2 --view --device cpu --seconds 0
```

V16 adds irregular source shapes, slow source drift, and local depletion/recovery.
It retains V15's controllers, genomes, food particles, quality reversals, and
reproduction settings. The controller still has 40 inputs and seven outputs per
module; fertility is a viewer diagnostic, not an additional sense.

## What moves and what stays put

Each circular spawn region becomes an ellipse with a sinusoidal sideways bend.
The transformation preserves area, so changing shape alone does not change the
nominal food-producing area. Every source gets an independently sampled orientation
and signed bend. Shapes remain fixed relative to their centers in this version.

Source centers drift at a fixed speed while their travel directions wander
independently. Direction changes have no preferred rotation. A conservative
boundary inset keeps each complete source inside the circular habitat; sources
reflect at that inset. Updates use the existing field cadence, 5 Hz in this preset.

Only the location of **new food** moves. Existing food and detritus remain at
their deposited coordinates, and particles retain their source's identity for
nutritional reversals. A moving source therefore leaves an edible trail until
particles are eaten or expire. Forecast and identity fields follow source centers.
Shelter stays at its initial location, and its field and rendered circles agree.

## Local production reserves

A separate 64-by-64 grid is fixed to world coordinates. At the 512-unit world
diameter, each cell is 8 units wide. Each valid cell starts with 128 units of
food-production reserve, enough for six complete 20-energy particles with eight
units left over. Producing a particle consumes 20 reserve units from its cell.
This applies to clustered and background fresh food; detritus uses the existing
recycling rules and does not spend fertility.

When a cell cannot fund a whole particle, that spawn proposal is rejected. The
budget is not moved elsewhere or accumulated for a later burst. The old per-source
stock limit is checked first, and rejected proposals never consume fertility.
Competing proposals for one cell are funded in proposal order without overdrawing.

Reserve recovers exponentially toward capacity: 120 simulated seconds restores
about 63% of whatever was missing. Recovery freezes when the simulation is paused.
The cell's capacity must exceed one particle's energy, because exponential recovery
approaches full capacity asymptotically. Grid resolution and reserve per area are
separate settings from the smell grid.

The existing food-energy ledger records energy actually entering edible particles.
A separate production ledger checks initial fertility + recovery - production
spending - remaining fertility. Recovery represents external environmental input;
fertility does not become extra organism energy without a food particle.

Grazing removes existing food and opens room under source stock limits. Regrowth
also depends on local reserve, giving heavily used source areas a recovery period.
Fertility is spent on production, not directly on an organism's contact with soil.

## Parameters and controls

| Setting | V16 value | Meaning |
| --- | --- | --- |
| `patch_aspect_ratio` | 2.0 | Ellipse major/minor ratio before bending |
| `patch_irregularity` | 0.35 | Maximum signed bend, relative to the original radius |
| `patch_drift_speed` | 0.5 | Source travel speed in world units/second |
| `patch_drift_turn_time` | 60 s | About one radian RMS of heading change over this time, away from walls |
| `fertility_grid_size` | 64 | Cells along each side of the production grid |
| `fertility_capacity_per_area` | 2.0 | Maximum reserve per square world unit |
| `fertility_recovery_time` | 120 s | Recovery time constant |

The eight sources retain their 40-unit reference radius, 60-second production
cycles, and 2,400-energy stock caps. Food lifetime stays at 45 seconds. The proposed
food rate remains 24 particles/second on average, but stock and fertility limits
can reduce actual supply. Birth threshold/debit remain 150/110, with newborn
energy 100 and the usual body-size scaling and neural construction charges.

- **P:** toggle green source outlines and center marks. Outlines show where new
  food can appear; older food can remain outside them as they move.
- **Tab:** cycle to the fertility view. Bright cells have more reserve; dark
  cells have less. This view is independent of the actual-food smell overlay.
- **F:** toggle the current field overlay. Cyan circles continue to mark shelter.

See the [source preview](v16-sources.png) and [fertility preview](v16-fertility.png).
A 4x recording of the first 120 simulated seconds of seed 1 is available locally
at `runs/v16-resources-preview/timelapse.mp4`. Its 901 frames were decoded and
checked. Seed 1's first birth in the pilot was later, at about 232 seconds; seed 2
first reproduced at about 5.4 seconds.

## Compatibility and initial evidence

V0–V15 presets and saved checkpoint settings are unchanged. Use a fresh V16 run
to enable these rules. The resource grid, source geometry, headings, counters,
update clock, and random stream are all checkpointed. Rendering consumes no random
draws. Separate CPU and RTX 5080 checks passed exact continuation through births,
growth, depletion, and recovery; this does not imply cross-device equivalence.

To disable individual mechanisms in a copy of V16, use drift speed 0, fertility
capacity per area 0, or aspect ratio 1 plus irregularity 0. Disabling all three
matches V15's simulation trajectory in a regression test. The original V15 research
preset and the later V15 reproduction preset remain separate files.

Three fresh CPU starts ran for 600 seconds:

| Seed | Births | Final population | First birth |
| --- | --- | --- | --- |
| 1 | 13 | 6 | 232.03 s |
| 2 | 30 | 4 | 5.43 s |
| 3 | 12 | 5 | 16.23 s |

All three reproduced and survived the horizon, but their populations are fragile.
These are checks of the combined environment, not isolated tests of each mechanism
or evidence that circling was caused by patch geometry. Source placement now uses
the larger shape inset, and rejected proposals change actual food supply.
Improved tracking, learning, and long-term persistence have not been established.
The [audit](results/v16-resources.json) records source hashes, proposal accounting,
the previous V15 comparison, checkpoint checks, and verification scope.

## Possible later step: simple producers

Add plant-like entities that use local fertility to grow edible biomass, spread
nearby, and can be grazed. Their growth and recovery could progressively replace
automatic food spawning. Keep biomass creation, offspring costs, and recycling
explicitly budgeted so producers do not introduce a free-energy reproduction loop.
This is a possible future direction only; producer organisms are not implemented
in V16.
