import math

import torch

from emergent_garden.exploration import history_state, sample_motors


def test_conditional_score_matches_finite_differences_through_both_mean_evaluations():
    dtype = torch.float64
    features = torch.tensor([[0.2, -0.4, 0.3, 0.8]], dtype=dtype)
    old_features = torch.tensor([[-0.1, 0.5, 0.2, 0.7]], dtype=dtype)
    base = torch.tensor([[0.4, -0.3]], dtype=dtype)
    plastic = torch.tensor([[[0.1, 0.02, -0.03, 0.04], [-0.05, 0.06, 0.07, 0.08]]], dtype=dtype)
    state = history_state(3, features)
    state["motor_previous_features"] = old_features
    state["motor_previous_base"][:] = torch.tensor([0.15, -0.2], dtype=dtype)
    state["motor_previous_logits"][:] = torch.tensor([0.3, -0.05], dtype=dtype)
    state["motor_history_ready"][:] = True
    noise = torch.tensor([[0.3, -0.7]], dtype=dtype)
    sigma, elapsed, tau = torch.tensor([0.1], dtype=dtype), torch.tensor([0.1], dtype=dtype), 2.0
    sampled, score, credit_features, _ = sample_motors(
        features, base, plastic, state, noise, sigma, elapsed, tau
    )
    analytic = score[..., None] * credit_features[:, None]
    variance = sigma.square() * (-2 * elapsed / tau).expm1().neg()

    def log_density(weights):
        mean, _, _, _ = sample_motors(
            features, base, weights, state, torch.zeros_like(noise), sigma, elapsed, tau
        )
        return (-(sampled - mean).square() / (2 * variance[:, None])).sum()

    finite = torch.zeros_like(plastic)
    epsilon = 1e-6
    for motor in range(2):
        for feature in range(4):
            plus, minus = plastic.clone(), plastic.clone()
            plus[0, motor, feature] += epsilon
            minus[0, motor, feature] -= epsilon
            finite[0, motor, feature] = (log_density(plus) - log_density(minus)) / (2 * epsilon)
    torch.testing.assert_close(analytic, finite, rtol=1e-8, atol=1e-8)
    naive = (noise / sigma[:, None])[..., None] * features[:, None]
    assert (analytic - naive).abs().max() > 1


def test_stationary_variance_and_lag_correlation_survive_a_changed_inherited_mean():
    count = 8192
    rng = torch.Generator().manual_seed(9173)
    features = torch.ones(count, 1)
    base = torch.zeros(count, 2)
    plastic = torch.zeros(count, 2, 1)
    sigma, elapsed, tau = torch.full((count,), 0.1), torch.full((count,), 0.1), 2.0
    state = history_state(0, features)
    first, _, _, state = sample_motors(
        features, base, plastic, state, torch.randn(count, 2, generator=rng), sigma, elapsed, tau
    )
    shifted = base + torch.tensor([0.4, -0.3])
    second, _, _, _ = sample_motors(
        features, shifted, plastic, state, torch.randn(count, 2, generator=rng), sigma, elapsed, tau
    )
    a, b = first.flatten(), (second - shifted).flatten()
    assert abs(a.mean().item()) < 0.003
    assert abs(b.mean().item()) < 0.003
    assert abs(a.var().item() / 0.01 - 1) < 0.04
    assert abs(b.var().item() / 0.01 - 1) < 0.04
    correlation = torch.corrcoef(torch.stack((a, b)))[0, 1].item()
    assert abs(correlation - math.exp(-0.1 / tau)) < 0.006


def test_neutral_mode_matches_independent_sampling_and_ignores_history_exactly():
    features = torch.tensor([[0.2, -0.4, 0.8]])
    base = torch.tensor([[0.3, -0.2]])
    plastic = torch.tensor([[[0.1, 0.2, -0.1], [0.2, -0.3, 0.4]]])
    state = history_state(2, features)
    state["motor_previous_logits"].fill_(99)
    state["motor_history_ready"].fill_(True)
    noise, sigma, elapsed = torch.tensor([[0.2, -0.4]]), torch.tensor([0.1]), torch.tensor([0.1])
    sampled, score, credit, history = sample_motors(
        features, base, plastic, state, noise, sigma, elapsed, 0
    )
    expected = base + ((plastic @ features[..., None]).squeeze(-1) + sigma[:, None] * noise)
    torch.testing.assert_close(sampled, expected, rtol=0, atol=0)
    torch.testing.assert_close(score, noise / sigma[:, None], rtol=0, atol=0)
    torch.testing.assert_close(credit, features, rtol=0, atol=0)
    assert history["motor_history_ready"].all()


def test_disabled_exploration_or_zero_sigma_removes_history_immediately():
    features = torch.tensor([[0.5, 0.5]])
    base, plastic = torch.ones(1, 2), torch.zeros(1, 2, 2)
    state = history_state(1, features)
    state["motor_previous_logits"].fill_(99)
    state["motor_history_ready"].fill_(True)
    for enabled, sigma in ((False, 0.1), (True, 0.0)):
        logits, score, _, history = sample_motors(
            features,
            base,
            plastic,
            state,
            torch.ones(1, 2),
            torch.tensor([sigma]),
            torch.tensor([0.1]),
            2.0,
            enabled,
        )
        torch.testing.assert_close(logits, base, rtol=0, atol=0)
        assert score.count_nonzero() == 0
        assert not history["motor_history_ready"].any()
        assert history["motor_applied_noise"].count_nonzero() == 0
