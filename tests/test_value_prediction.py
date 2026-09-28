import math

import torch

from emergent_garden.value import advance_value, value_state


def test_motor_value_target_can_use_raw_returns_without_changing_the_temporal_baseline(config):
    from dataclasses import replace

    from emergent_garden.learning import motor_policy, motor_state

    c = replace(config, ecology_version=22, motor_value_rate=0.02, motor_normalized=1)
    genomes = torch.zeros(1, c.parameter_count)
    state = motor_state(c, genomes)
    state["motor_value_ready"].fill_(True)
    state["motor_baseline"].fill_(0.2)
    # Keep the critic weights zero to isolate the reward component of the error.
    results = []
    for centered in (1, 0):
        result, _ = motor_policy(
            replace(c, motor_value_centered=centered),
            genomes,
            torch.zeros(1, c.hidden_size),
            torch.zeros(1, c.output_size),
            state,
            torch.zeros(1, 2),
            torch.tensor([0.1]),
            torch.tensor([0.5]),
        )
        results.append(result)
    torch.testing.assert_close(results[0]["motor_value_error"], torch.tensor([0.0]))
    torch.testing.assert_close(results[1]["motor_value_error"], torch.tensor([0.1]))
    torch.testing.assert_close(
        results[0]["motor_baseline"], results[1]["motor_baseline"], rtol=0, atol=0
    )


def test_transition_credits_previous_features_with_preupdate_predictions():
    features = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    state = value_state(1, features)
    state["motor_value_weights"][:] = torch.tensor([0.2, 0.4], dtype=torch.float64)
    state["motor_value_previous"][:, 0] = 1
    state["motor_value_trace"][:, 0] = 1
    state["motor_value_ready"][:] = True
    elapsed, reward, rate = (torch.tensor([v], dtype=torch.float64) for v in (0.25, 0.1, 0.05))
    changed, error = advance_value(state, features, reward, elapsed, rate, 2.0, 1.0, 4.0)
    expected = 0.1 + math.exp(-0.25 / 2) * 0.4 - 0.2
    torch.testing.assert_close(error, torch.tensor([expected], dtype=torch.float64))
    torch.testing.assert_close(
        changed["motor_value_weights"],
        torch.tensor([[0.2 + 0.05 * expected, 0.4]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        changed["motor_value_trace"],
        torch.tensor([[math.exp(-0.25), 1.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(state["motor_value_weights"], torch.tensor([[0.2, 0.4]]).double())


def test_fresh_state_does_not_credit_an_unobserved_transition_and_terminal_clears_history():
    features = torch.tensor([[0.6, 0.8]])
    elapsed, rate = torch.tensor([0.1]), torch.tensor([0.2])
    state, error = advance_value(
        value_state(1, features), features, torch.tensor([10.0]), elapsed, rate, 20, 2, 4
    )
    assert error.item() == 0
    assert state["motor_value_weights"].count_nonzero() == 0
    assert state["motor_value_ready"].all()
    state["motor_value_weights"][:] = features
    state, error = advance_value(
        state,
        features,
        torch.tensor([0.25]),
        elapsed,
        rate,
        20,
        2,
        4,
        terminal=torch.tensor([True]),
    )
    torch.testing.assert_close(error, torch.tensor([-0.75]))
    assert state["motor_value_weights"].count_nonzero() > 0
    for key in ("motor_value_previous", "motor_value_trace", "motor_value_ready"):
        assert state[key].count_nonzero() == 0


def test_elapsed_time_discounting_and_readout_bounds():
    features = torch.ones(2, 1)
    state = value_state(0, features)
    state["motor_value_weights"].fill_(0.5)
    state["motor_value_previous"].fill_(1)
    state["motor_value_trace"].fill_(1)
    state["motor_value_ready"].fill_(True)
    elapsed = torch.tensor([0.1, 1.0])
    new, error = advance_value(state, features, torch.zeros(2), elapsed, torch.ones(2), 2, 1, 4)
    torch.testing.assert_close(error, (torch.exp(-elapsed / 2) - 1) * 0.5)
    assert error[1] < error[0]
    new, _ = advance_value(new, features, torch.full((2,), 1e6), elapsed, torch.ones(2), 2, 1, 0.1)
    assert new["motor_value_weights"].norm(dim=-1).max() <= 0.1 + 1e-7
    assert all(torch.isfinite(value).all() for value in new.values())


def test_known_delayed_chain_learns_returns_and_moves_error_before_the_meal():
    # Four observable states, one second apart. The last transition pays 0.2.
    # This supplies a representation, never an inherited controller or steering.
    features = torch.eye(4, dtype=torch.float64)
    state = value_state(3, features[:1])
    elapsed, rate = torch.ones(1, dtype=torch.float64), torch.tensor([0.1], dtype=torch.float64)
    reward = torch.zeros(1, dtype=torch.float64)
    terminal = torch.tensor([True])
    for _ in range(160):
        state, _ = advance_value(state, features[:1], reward, elapsed, rate, 20, 2, 4)
        for index in (1, 2, 3):
            state, _ = advance_value(
                state, features[index : index + 1], reward, elapsed, rate, 20, 2, 4
            )
        state, _ = advance_value(
            state, features[:1] * 0, reward + 0.2, elapsed, rate, 20, 2, 4, terminal=terminal
        )
    expected = 0.2 * torch.exp(-torch.arange(3, -1, -1, dtype=torch.float64) / 20)
    torch.testing.assert_close(state["motor_value_weights"][0], expected, rtol=0, atol=2e-5)
    # Enter the learned chain from an observation with zero predicted value.
    state, _ = advance_value(state, features[:1] * 0, reward, elapsed, rate, 20, 2, 4)
    state, error = advance_value(state, features[:1], reward, elapsed, rate, 20, 2, 4)
    assert state["motor_value_prediction"].item() > 0.17
    assert error.item() > 0.16
    for index in (1, 2, 3):
        state, _ = advance_value(
            state, features[index : index + 1], reward, elapsed, rate, 20, 2, 4
        )
    _, error = advance_value(
        state, features[:1] * 0, reward + 0.2, elapsed, rate, 20, 2, 4, terminal=terminal
    )
    assert abs(error.item()) < 2e-5
