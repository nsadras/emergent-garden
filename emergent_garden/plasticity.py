"""Inherited, bounded coefficients for a local recurrent plasticity rule."""

RULE_NAMES = ("correlation", "presynaptic", "postsynaptic", "constant")


def rule_coefficients(config, genomes, *, evolved=True):
    """Decode A/B/C/D with an absolute-sum budget of one.

    Raw genes are signed, unlike sigmoid developmental allocations. The budget
    keeps |A post*pre + B pre + C post + D| <= 1 for bounded neural activity,
    so a more expressive rule does not silently increase the maximum step size.
    Zero coefficients mean no new trace; existing offsets still decay.
    """
    if config.ecology_version < 15 or not evolved:
        values = genomes.new_zeros((len(genomes), 4))
        values[:, 0] = 1
        return values
    start = config.brain_parameter_count + 13
    values = genomes[:, start : start + 4]
    return values / values.abs().sum(1, keepdim=True).clamp_min(1)
