# V0 initial validation

For the five subsequent versions and their current validation, see
[EVOLUTION.md](EVOLUTION.md). This document preserves the original V0 results.

Recorded 2026-09-22. These are implementation checks and initial experiments,
not a claim of robust evolved chemotaxis or lifetime learning.

## Software checks

- 29 tests cover local neighbor lookup, conservative feeding under competition,
  storage limits, propulsion, collisions, starvation order, birth failures,
  mutation, recurrent state, field sampling, and invalid configurations.
- A CPU checkpoint saved between controller/field updates produces an identical
  continuation, including random streams, births, resources, and recurrent state.
- A SIGTERM test compares the resulting checkpoint against an uninterrupted run
  and verifies shutdown at a complete physics tick.
- Rendering, recording, zoom, selection, and pause controls leave simulation
  state and random streams unchanged.
- A recorded MP4 was decoded successfully: 1,024 x 1,024, 30 FPS, 31 frames.
- CUDA execution and checkpoint continuation were checked on an NVIDIA GeForce
  RTX 5080 with PyTorch 2.11.0+cu128. CUDA continuation matched within 1e-5
  tolerance in the small checkpoint check.
- Ruff lint and formatting checks pass. Dependencies are recorded in `uv.lock`.

## Population persistence

Configuration: [configs/v0.toml](configs/v0.toml), unmodified ecological parameters.
Five independent CPU runs, each lasting 600 simulated seconds:

| Seed | Final population | Births | Deaths | Maximum living generation | Simulated/wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 298 | 747 | 705 | 9 | 6.29x |
| 2 | 290 | 775 | 741 | 9 | 6.19x |
| 3 | 305 | 762 | 713 | 11 | 6.14x |
| 4 | 296 | 725 | 685 | 10 | 6.21x |
| 5 | 294 | 746 | 708 | 8 | 6.23x |

Final global energy-balance residuals were below 0.017 energy units in absolute
value. Float32 arithmetic introduces small residuals; no systematic resource
creation was observed in these runs.

Separate 120-second diagnostics using seed 1:

| Controller | Final population | Births | Deaths | Food energy absorbed by population |
| --- | ---: | ---: | ---: | ---: |
| Scripted local-sensor forager | 307 | 126 | 75 | 66,600 |
| Stationary | 18 | 0 | 238 | 1,680 |

The ecology permits feeding and reproduction, and stationary behavior does not
sustain the initial population in this diagnostic. These controllers were not
introduced into any evolving population.

Local artifacts are under `runs/calibration-v0/`, `runs/forager-check/`, and
`runs/rest-check/`. Generated run directories are excluded from version control.

## Exploratory behavioral comparison

Sample: four random founder genomes and four random living descendant genomes
from calibration seed 1 at 600 seconds. Each was evaluated for 120 seconds in
four fresh environments (seeds 10001–10004). Each row below contains 16 trials.
Mutation was disabled; the ordinary energy and reproduction rules remained active.
Measurements refer to the tested founding creature, separately from its offspring.

| Treatment | Mean food energy collected | Mean offspring count |
| --- | ---: | ---: |
| Founders, normal sensing | 221.25 | 0.81 |
| Descendants, normal sensing | 440.00 | 2.19 |
| Descendants, smell disabled | 521.80 | 2.69 |
| Descendants, smell sampled at shuffled positions | 315.00 | 1.63 |

Descendants outperformed founders in this small sample, but **a reliable benefit
from food sensing has not been established**. Disabling smell improved the mean
food collection, while spatial shuffling reduced it. Normal sensing beat disabled
sensing in 8/16 paired trials and shuffled sensing in 11/16. Outcomes vary widely
between environments and genomes, and all descendants came from one evolutionary
run. This is insufficient to establish the full research success criterion.

The next experiments should use longer evolutionary runs, the full 32-genome /
eight-environment evaluation, and independent evolutionary seeds. Preserve these
negative and mixed results when comparing later changes. Do not treat the video,
population stability, or higher descendant reproduction as proof of chemotaxis.

Individual trials, sampled genomes, and summaries are available locally in
`runs/evaluation-v0-seed1/`. A separate short workflow check is in
`runs/evaluation-smoke/`.

## Performance and current limitations

- CPU is presently the faster backend for this small V0 population. A 10-second
  full-size run measured about 5.1x on CPU with recording and 0.9x on the 5080
  without recording. The longer CPU calibration runs averaged about 6.2x.
- CUDA is functional, but small tensor operations, variable population/food
  arrays, and synchronization limit throughput. It is not yet an optimized GPU
  implementation for thousands of creatures. Use `--device cpu` for the current
  baseline and profile larger populations before choosing a backend.
- The chemical field approximates current particle positions. It does not model
  physical diffusion over time or persistent chemical trails.
- Checkpoints retain the latest population. Archive selected checkpoint files
  explicitly when preserving multiple evolutionary milestones.
- Exact replay across different hardware, device types, or library versions is
  not promised. Recordings preserve what actually happened.
- Emergent memory, plasticity, communication, and morphology evolution remain
  outside V0.
