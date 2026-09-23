"""Local field encoding, independent of behavior or a controller's weights."""

import torch


def directional_basis(samples):
    """Four receptors -> mean, right-left, front-back, and diagonal contrasts.

    Receptor order is -135, -45, +45, +135 degrees relative to heading.
    This invertible change of basis does not choose a movement direction.
    """
    back_left, front_left, front_right, back_right = samples.unbind(-1)
    return (
        torch.stack(
            (
                back_left + front_left + front_right + back_right,
                -back_left - front_left + front_right + back_right,
                -back_left + front_left + front_right - back_right,
                back_left - front_left + front_right - back_right,
            ),
            -1,
        )
        / 4
    )


def receptor_values(features):
    """Undo the spatial basis; the pooled normalization stays in place."""
    mean, side, front, diagonal = features.unbind(-1)
    return torch.stack(
        (
            mean - side - front + diagonal,
            mean - side + front - diagonal,
            mean + side + front + diagonal,
            mean + side - front - diagonal,
        ),
        -1,
    )


def encode_field(config, samples, scale=None):
    """Bound local contrasts using one shared denominator per receptor group.

    `scale=None` is for already bounded fields such as shelter. Old presets
    preserve the original independent receptor compression exactly. With
    contrast sensing, the mean remains in [0, 1] and each contrast in [-1, 1].
    The positive denominator also suppresses contrast in nearly empty fields.
    """
    if not config.sensory_contrast:
        return samples if scale is None else samples / (samples + scale)
    if scale is not None:
        samples = samples / (samples.mean(-1, keepdim=True) + scale)
    return directional_basis(samples)


def probe_features(config, samples):
    """Express an isolated probe's receptor pattern in the configured basis.

    Probes specify bounded stimuli, not raw environmental concentrations.
    This preserves their mean and spatial information without a second
    compression. They remain circuit assays, separate from ecological trials.
    """
    return directional_basis(samples) if config.sensory_contrast else samples
