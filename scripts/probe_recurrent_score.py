"""Check a candidate recurrent learning score before adding it to the ecology.

This is a two-neuron mathematical fixture, not a native controller or a food-
seeking organism. Compare the Gaussian likelihood score with exact conditional
autograd and with pathwise gradients of an episodic, differentiable objective.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def transition_check():
    weights = torch.tensor([[0.4, -0.2], [0.1, 0.5]], dtype=torch.float64, requires_grad=True)
    previous = torch.tensor([0.25, -0.4], dtype=torch.float64)
    inputs = torch.tensor([0.6, 0.2], dtype=torch.float64)
    alpha = -torch.expm1(-0.1 / torch.tensor([0.7, 2.0], dtype=torch.float64))
    sigma = 0.15
    epsilon = torch.tensor([0.6, -1.1], dtype=torch.float64)
    latent = weights @ previous + inputs + sigma * epsilon
    observed = ((1 - alpha) * previous + alpha * latent.tanh()).detach()
    # Hold the observed transition fixed when differentiating its probability.
    transformed = (observed - (1 - alpha) * previous) / alpha
    reconstructed = transformed.atanh()
    log_probability = (
        -0.5 * ((reconstructed - weights @ previous - inputs) / sigma).square()
        - math.log(sigma * math.sqrt(2 * math.pi))
        - alpha.log()
        - (1 - transformed.square()).log()
    ).sum()
    gradient = torch.autograd.grad(log_probability, weights)[0]
    score = (epsilon / sigma)[:, None] * previous[None, :]
    torch.testing.assert_close(gradient, score, rtol=1e-12, atol=1e-12)
    return dict(
        score=score.tolist(),
        autograd=gradient.tolist(),
        maximum_absolute_error=(score - gradient).abs().max().item(),
    )


def episode(weights, cue, noise, alpha, sigma):
    h = torch.zeros(len(cue), 2, dtype=torch.float64)
    eligibility = torch.zeros(len(cue), 2, 2, dtype=torch.float64)
    for step, epsilon in enumerate(noise):
        inputs = cue[:, None] * torch.tensor([0.8, 0.2], dtype=torch.float64) if step < 6 else 0
        # Conditional scores treat the already realized recurrent input as fixed.
        eligibility += (epsilon / sigma)[:, :, None] * h.detach()[:, None, :]
        h = (1 - alpha) * h + alpha * (h @ weights.T + inputs + sigma * epsilon).tanh()
    output = h @ torch.tensor([0.7, -0.4], dtype=torch.float64)
    reward = -0.5 * (output - 0.12 * cue).square()
    return reward, eligibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=8488)
    args = parser.parse_args()
    if args.batches < 2 or args.batch_size < 2:
        parser.error("Use at least two independent batches and two trajectories per batch")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).read_bytes()
    (args.output / Path(__file__).name).write_bytes(script)
    rng = torch.Generator().manual_seed(args.seed)
    alpha = -torch.expm1(-0.1 / torch.tensor([0.7, 2.0], dtype=torch.float64))
    sigma, steps = 0.15, 24
    estimates, gradients, rows = [], [], []
    for batch in range(args.batches):
        weights = torch.tensor([[0.4, -0.2], [0.1, 0.5]], dtype=torch.float64, requires_grad=True)
        cue = torch.where(torch.rand(args.batch_size, generator=rng) < 0.5, -1.0, 1.0).double()
        noise = torch.randn(steps, args.batch_size, 2, generator=rng, dtype=torch.float64)
        reward, eligibility = episode(weights, cue, noise, alpha, sigma)
        # This baseline is independent of the current episode's sampled perturbations.
        # It is a fixture-only variance reduction, not a proposed ecological oracle.
        with torch.no_grad():
            baseline, _ = episode(weights, cue, torch.zeros_like(noise), alpha, sigma)
        estimate = ((reward.detach() - baseline)[:, None, None] * eligibility).mean(0)
        gradient = torch.autograd.grad(reward.mean(), weights)[0]
        estimates.append(estimate)
        gradients.append(gradient)
        rows.append(
            dict(
                batch=batch,
                mean_reward=reward.mean().item(),
                score_estimate=estimate.tolist(),
                pathwise_gradient=gradient.tolist(),
            )
        )
    estimates, gradients = torch.stack(estimates), torch.stack(gradients)
    differences = estimates - gradients
    se = differences.std(0) / math.sqrt(args.batches)
    mean_difference = differences.mean(0)
    z = mean_difference.abs() / se
    cosine = torch.nn.functional.cosine_similarity(
        estimates.mean(0).flatten(), gradients.mean(0).flatten(), dim=0
    )
    report = dict(
        completed=True,
        script_sha256=hashlib.sha256(script).hexdigest(),
        torch=str(torch.__version__),
        dtype="float64",
        seed=args.seed,
        batches=args.batches,
        trajectories_per_batch=args.batch_size,
        sigma=sigma,
        controller_dt=0.1,
        tau=[0.7, 2.0],
        steps=steps,
        transition=transition_check(),
        mean_score_estimate=estimates.mean(0).tolist(),
        mean_pathwise_gradient=gradients.mean(0).tolist(),
        paired_difference=mean_difference.tolist(),
        paired_standard_error=se.tolist(),
        maximum_absolute_standardized_difference=z.max().item(),
        mean_gradient_cosine=cosine.item(),
        relative_gradient_error=(mean_difference.norm() / gradients.mean(0).norm()).item(),
        batches_detail=rows,
        interpretation="A two-neuron leaky-tanh recurrent fixture, with fixed weights for "
        "each 2.4-second trajectory, Gaussian preactivation perturbations, and a supplied "
        "delayed cue target. The likelihood score epsilon/sigma outer previous activity "
        "contains no extra leak or tanh derivative. Twenty independent batch estimates "
        "compare it with autograd through the complete noisy trajectory; standard errors "
        "are across paired batch differences. The perturbation-free return is a diagnostic "
        "baseline independent of current noise, not information available to an organism. "
        "This verifies a candidate gradient, not learning performance. It does not verify "
        "finite eligibility decay, clipped feedback, changing online weights, energetic "
        "constraints, or ecological adaptation. No native controller was changed.",
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "mean_score_estimate",
                    "mean_pathwise_gradient",
                    "paired_standard_error",
                    "maximum_absolute_standardized_difference",
                    "mean_gradient_cosine",
                    "relative_gradient_error",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
