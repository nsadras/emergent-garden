import math
from dataclasses import replace

import pytest
import torch

from emergent_garden.development import grow, mutate_module_count
from emergent_garden.ecology import EcologyWorld
from emergent_garden.morphology import module_count
from emergent_garden.storage import load_checkpoint, save_checkpoint


def juvenile(config, target=2, ablation="none", **changes):
    c = replace(
        config,
        **{
            "ecology_version": 13,
            "initial_population": 1,
            "growth_delay": 0.1,
            "growth_retry": 0.1,
            "mutation_probability": 0.0,
            "trait_mutation_probability": 0.0,
            "node_mutation_probability": 0.0,
            "edge_mutation_probability": 0.0,
            "module_mutation_probability": 0.0,
            "exploration_min": 0.0,
            "exploration_max": 0.0,
            **changes,
        },
    )
    w = EcologyWorld(c, ablation=ablation)
    a = w.agents
    a["genome"][:, c.brain_parameter_count : c.brain_parameter_count + c.trait_count] = 0
    allocation = (target - 0.5) / 3
    a["genome"][:, c.brain_parameter_count + 6] = math.log(allocation / (1 - allocation))
    w.develop(a)
    a["pos"][:] = c.diameter / 2
    a["energy"] = c.birth_energy * a["area"]
    w.initial_energy = a["energy"].double().sum().item()
    return w


def fund(w):
    previous = w.agents["energy"].double().sum().item()
    w.agents["energy"] = w.config.max_energy * w.agents["area"]
    w.initial_energy += w.agents["energy"].double().sum().item() - previous


@pytest.mark.parametrize("target", [1, 2, 3])
def test_inherited_plan_starts_with_one_expressed_module(config, target):
    w = juvenile(config, target)
    assert w.agents["modules"].item() == 1
    assert w.agents["target_modules"].item() == target
    assert w.metrics()["juvenile_population"] == (target > 1)
    assert w.agents["radius"].item() == w.agents["core_radius"].item()


def test_growth_waits_for_time_and_energy_then_pays_and_preserves_existing_state(config):
    w = juvenile(config, 3)
    a, c = w.agents, w.config
    genome = a["genome"].clone()
    fund(w)
    grow(w)
    assert w.totals["growth_attempts"] == 0
    w.tick = int(a["growth_tick"][0])
    previous = a["energy"].item()
    a["energy"].fill_(1)
    w.initial_energy -= previous - 1
    grow(w)
    assert w.totals["growth_unaffordable"] == 1
    assert w.totals["development_cost"] == 0
    fund(w)
    w.tick = int(a["growth_tick"][0])
    old_area, old_brain = a["area"].clone(), a["brain_construction"].clone()
    a["module_h"][:, 0].fill_(0.2)
    a["module_plastic"][:, 0].fill_(0.04)
    a["module_motor_plastic"][:, 0].fill_(0.03)
    a["module_actions"][:, 0].fill_(0.4)
    before = a["energy"].item()
    grow(w)
    assert a["modules"].item() == 2
    cost = c.growth_cost * (a["area"] - old_area) + a["brain_construction"] - old_brain
    assert before - a["energy"].item() == pytest.approx(cost.item(), abs=2e-5)
    assert w.totals["development_cost"] == pytest.approx(cost.item(), abs=2e-5)
    assert a["development_spent"].item() == pytest.approx(cost.item(), abs=2e-5)
    assert a["motor_reward"].item() == 0
    assert a["module_h"][0, 0].eq(0.2).all()
    assert a["module_plastic"][0, 0].eq(0.04).all()
    assert a["module_motor_plastic"][0, 0].eq(0.03).all()
    for key in (
        "module_h",
        "module_plastic",
        "module_trace",
        "module_motor_plastic",
        "module_motor_trace",
        "module_motor_baseline",
        "module_actions",
    ):
        assert a[key][:, 1:].count_nonzero() == 0
    assert a["actions"].eq(0.2).all()
    torch.testing.assert_close(a["genome"], genome, rtol=0, atol=0)
    assert abs(w.metrics()["energy_balance_error"]) < 1e-4
    grow(w)
    assert a["modules"].item() == 2
    fund(w)
    w.tick = int(a["growth_tick"][0])
    grow(w)
    assert a["modules"].item() == 3
    assert w.metrics()["adult_module_histogram"] == [0, 0, 1]
    assert abs(w.metrics()["energy_balance_error"]) < 1e-4


def test_wall_blocks_growth_without_charge_and_allows_a_later_retry(config):
    w = juvenile(config)
    fund(w)
    a = w.agents
    a["pos"][0, 0] = a["radius"][0] + 0.1
    w.tick = int(a["growth_tick"][0])
    before = a["energy"].clone()
    grow(w)
    assert w.totals["growth_blocked"] == 1
    assert w.totals["development_cost"] == 0
    torch.testing.assert_close(a["energy"], before, rtol=0, atol=0)
    a["pos"].fill_(64)
    w.tick = int(a["growth_tick"][0])
    grow(w)
    assert a["modules"].item() == 2


def test_competing_growth_proposals_cannot_create_overlap(config):
    w = juvenile(config, initial_population=2)
    fund(w)
    a = w.agents
    a["pos"][:] = torch.tensor([[57.5, 64], [70.5, 64]])
    w.tick = int(a["growth_tick"][0])
    grow(w)
    assert sorted(a["modules"].tolist()) == [1, 2]
    assert w.totals["growths"] == w.totals["growth_blocked"] == 1
    assert (a["pos"][0] - a["pos"][1]).norm() >= a["radius"].sum()
    assert abs(w.metrics()["energy_balance_error"]) < 1e-4


def test_only_mature_bodies_reproduce_and_children_reset_body_and_acquired_state(config):
    w = juvenile(config)
    fund(w)
    w.reproduce()
    assert w.totals["births"] == 0
    w.tick = int(w.agents["growth_tick"][0])
    grow(w)
    fund(w)
    a = w.agents
    a["module_h"][:, :2].fill_(0.2)
    a["module_plastic"][:, :2].fill_(0.04)
    a["module_motor_plastic"][:, :2].fill_(0.03)
    w.reproduce()
    a = w.agents
    assert w.population == 2
    assert a["modules"].tolist() == [2, 1]
    assert a["target_modules"].tolist() == [2, 2]
    torch.testing.assert_close(a["genome"][0], a["genome"][1], rtol=0, atol=0)
    for key in ("module_h", "module_plastic", "module_motor_plastic", "development_spent"):
        assert a[key][1].count_nonzero() == 0
        assert a[key][0].count_nonzero() > 0
    assert a["growth_tick"][1] == w.tick + math.ceil(w.config.growth_delay * w.config.physics_hz)
    assert w.totals.get("structural_births", 0) == 0
    assert w.totals["planned_module_births_2"] == 1
    assert abs(w.metrics()["energy_balance_error"]) < 1e-4


def test_juvenile_birth_can_cross_full_child_financing_barrier(config):
    young = juvenile(config, 1, module_mutation_probability=1.0)
    adult = juvenile(config, 1, ablation="adult_births", module_mutation_probability=1.0)
    for key in young.agents:
        torch.testing.assert_close(young.agents[key], adult.agents[key], rtol=0, atol=0)
    for w in (young, adult):
        fund(w)
        w.reproduce()
    assert young.population == 2 and adult.population == 1
    assert young.agents["modules"].tolist() == [1, 1]
    assert young.agents["target_modules"].tolist() == [1, 2]
    assert young.totals["module_event_births"] == 1
    assert young.totals["structural_births"] == 1
    assert adult.totals["blocked_births"] == 1
    assert adult.totals["module_mutation_events"] == 1
    assert adult.totals["reproduction"] == 0
    assert abs(young.metrics()["energy_balance_error"]) < 1e-4


def test_full_child_control_expresses_entire_inherited_plan(config):
    w = juvenile(config, 2, ablation="adult_births")
    fund(w)
    w.tick = int(w.agents["growth_tick"][0])
    grow(w)
    fund(w)
    w.reproduce()
    assert w.agents["modules"].tolist() == [2, 2]
    assert w.agents["target_modules"].tolist() == [2, 2]
    assert abs(w.metrics()["energy_balance_error"]) < 1e-4


@pytest.mark.parametrize("target", [1, 2, 3])
def test_discrete_body_mutation_changes_only_the_count_gene_and_uses_neighbors(config, target):
    w = juvenile(config, target, module_mutation_probability=1.0)
    original = w.agents["genome"][0].clone()
    generator = torch.Generator().manual_seed(211)
    proposals = set()
    for _ in range(20):
        changed, event = mutate_module_count(w.config, original, generator)
        assert (changed != original).nonzero().flatten().tolist() == [
            w.config.brain_parameter_count + 6
        ]
        actual = int(module_count(changed[w.config.brain_parameter_count + 6].sigmoid()))
        assert abs(actual - target) == 1
        assert event == dict(module_event=True, module_event_from=target, module_event_to=actual)
        proposals.add(actual)
    assert proposals == ({1, 3} if target == 2 else {2})
    torch.testing.assert_close(w.agents["genome"][0], original, rtol=0, atol=0)


def test_replay_through_growth_birth_and_discrete_body_mutation(config, tmp_path):
    w = juvenile(config, 2, module_mutation_probability=1.0, growth_reserve=10.0)
    fund(w)
    w.step(3)
    save_checkpoint(w, tmp_path / "juvenile.pt")
    other = load_checkpoint(tmp_path / "juvenile.pt")
    for world in (w, other):
        world.step(10)
        assert world.totals["growths"] > 0
        fund(world)
        world.step(10)
    assert w.totals["births"] > 0
    assert w.totals["module_event_births"] > 0
    assert w.totals == other.totals
    assert w.events == other.events
    for key in w.agents:
        torch.testing.assert_close(w.agents[key], other.agents[key], rtol=0, atol=0)
    for key in w.rng:
        assert torch.equal(w.rng[key].get_state(), other.rng[key].get_state())
    assert abs(w.metrics()["energy_balance_error"]) < 1e-3


def test_module_mutation_requires_representable_counts(config):
    with pytest.raises(ValueError, match="encode all counts"):
        replace(config, ecology_version=13, weight_limit=0.5).validate()
    replace(
        config, ecology_version=13, weight_limit=0.5, module_mutation_probability=0.0
    ).validate()


def test_rendering_a_juvenile_does_not_advance_development(config):
    from emergent_garden.viewer import Renderer

    w = juvenile(config, 3)
    before = w.state_dict()
    renderer = Renderer(640)
    renderer.selected = 0
    renderer.show_brain = True
    renderer.draw(w)
    assert w.tick == before["tick"]
    for key in w.agents:
        torch.testing.assert_close(w.agents[key], before["agents"][key], rtol=0, atol=0)
    for key in w.rng:
        assert torch.equal(w.rng[key].get_state(), before["rng"][key])
