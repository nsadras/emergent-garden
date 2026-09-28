import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture
def learner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("representation_learning")


def circuit():
    # Two sensory inputs, two recurrent sources, and a bias. One edge is absent.
    weights = torch.tensor(
        [[[0.2, -0.1, 0.4, 0.3, 0.1], [0.1, 0.5, 0.2, -0.1, -0.2]]], dtype=torch.float64
    )
    mask = torch.ones_like(weights, dtype=torch.bool)
    mask[0, 1, 0] = False
    return dict(
        weights=weights,
        mask=mask,
        nodes=torch.ones(1, 2, dtype=torch.bool),
        alpha=torch.tensor([[0.1, 0.3]], dtype=torch.float64),
    )


def test_first_observation_cannot_train_without_previous_experience(learner):
    c = circuit()
    state = learner.representation_state(c)
    observed = learner.prepare(
        c,
        state,
        torch.ones(1, 2, dtype=torch.float64),
        torch.tensor([1e6]),
        torch.tensor([0.03]),
        2,
    )
    new = learner.credit(
        c, state, observed, torch.ones(1), torch.ones(1), representation_error=torch.tensor([1e6])
    )
    assert new["ready"].all()
    assert new["eligibility"].count_nonzero()
    assert not new["offsets"].count_nonzero()
    assert not new["readout"].count_nonzero()
    assert not observed["prediction"].count_nonzero()
    assert not observed["error"].count_nonzero()


def experienced(learner):
    c = circuit()
    s = learner.representation_state(c)
    s["hidden"][:] = torch.tensor([[0.2, -0.3]])
    s["readout"][:] = torch.tensor([[0.4, -0.6, 0.1]])
    s["eligibility"][:] = c["mask"] * 0.3
    s["ready"].fill_(True)
    observed = learner.prepare(
        c,
        s,
        torch.tensor([[0.5, 0.2]], dtype=torch.float64),
        torch.tensor([0.2]),
        torch.tensor([0.1]),
        2,
    )
    return c, s, observed


def test_representation_credit_uses_old_trace_and_preupdate_readout(learner):
    c, state, observed = experienced(learner)
    rate, feedback = torch.tensor([0.01]), torch.tensor([0.3])
    original = {k: v.clone() for k, v in state.items()}
    inherited = c["weights"].clone()
    result = learner.credit(c, state, observed, rate, rate, representation_error=feedback)
    # Independently differentiate the previous normalized value, not the next one.
    h = state["hidden"].clone().requires_grad_()
    joined = torch.cat((h, torch.ones_like(h[:, :1])), -1)
    value = ((joined / joined.norm(dim=-1, keepdim=True)) * state["readout"]).sum()
    derivative = torch.autograd.grad(value, h)[0]
    expected = (rate * feedback)[:, None, None] * derivative[..., None] * state["eligibility"]
    torch.testing.assert_close(result["offsets"], expected)
    changed = observed | {"eligibility": observed["eligibility"] * 10000}
    other = learner.credit(c, state, changed, rate * 20, rate, representation_error=feedback)
    assert torch.equal(result["offsets"], other["offsets"])
    assert not torch.equal(result["readout"], other["readout"])
    assert torch.equal(inherited, c["weights"])
    assert all(torch.equal(original[k], v) for k, v in state.items())


def test_shuffled_representation_feedback_preserves_owner_readout_and_prediction(learner):
    c, state, observed = experienced(learner)
    prediction = observed["prediction"].clone()
    rate = torch.tensor([0.01])
    positive = learner.credit(
        c, state, observed, rate, rate, representation_error=torch.tensor([0.2])
    )
    negative = learner.credit(
        c, state, observed, rate, rate, representation_error=torch.tensor([-0.2])
    )
    assert torch.equal(positive["readout"], negative["readout"])
    assert torch.equal(positive["offsets"], -negative["offsets"])
    assert torch.equal(positive["hidden"], negative["hidden"])
    assert torch.equal(prediction, observed["prediction"])


def test_masks_and_bounds_hold_under_extreme_credit_and_newborns_reset(learner):
    c, state, observed = experienced(learner)
    observed["error"][:] = 1e9
    state["eligibility"][:] = 1e9
    large = torch.tensor([1e9])
    result = learner.credit(c, state, observed, large, large)
    assert result["offsets"].norm(dim=-1).max() <= 0.3 + 1e-12
    assert result["readout"].norm(dim=-1).max() <= 4 + 1e-12
    assert not result["offsets"][~c["mask"]].count_nonzero()
    assert all(torch.isfinite(v).all() for v in result.values())
    fresh = learner.representation_state(c)
    assert all(not v.count_nonzero() for v in fresh.values())


def test_zero_representation_rate_keeps_the_inherited_circuit_fixed(learner):
    c, state, observed = experienced(learner)
    result = learner.credit(c, state, observed, torch.tensor([0.01]), torch.zeros(1))
    assert not result["offsets"].count_nonzero()
    assert not torch.equal(result["readout"], state["readout"])


def test_bootstrap_and_previous_value_use_the_same_preupdate_readout(learner):
    c, state, observed = experienced(learner)
    _, previous, _ = learner.value_features(state["hidden"], state["readout"])
    _, current, _ = learner.value_features(observed["hidden"], state["readout"])
    expected = 0.2 + torch.exp(torch.tensor(-0.1 / 2)) * current - previous
    torch.testing.assert_close(observed["error"], expected)
