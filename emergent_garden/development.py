"""Energy- and space-constrained expression of an inherited modular body plan."""

import math

import torch

from .morphology import MAX_MODULES, module_count


def mutate_module_count(config, genome, generator):
    """Attempt one neighboring body-plan change without modifying other genes."""
    c = config
    if torch.rand((), generator=generator, device=genome.device) >= c.module_mutation_probability:
        return genome, dict(module_event=False)
    index = c.brain_parameter_count + 6
    current = int(module_count(genome[index].sigmoid()))
    if current in (1, MAX_MODULES):
        target = 2
    else:
        target = 1 if torch.rand((), generator=generator, device=genome.device) < 0.5 else 3
    allocation = (target - 0.5) / MAX_MODULES
    changed = genome.clone()
    changed[index] = max(
        -c.weight_limit, min(c.weight_limit, math.log(allocation / (1 - allocation)))
    )
    assert int(module_count(changed[index].sigmoid())) == target
    return changed, dict(module_event=True, module_event_from=current, module_event_to=target)


def grow(world):
    """Add at most one module per eligible body, paying only accepted proposals.

    Each step enlarges the circular membrane at its existing center. Growth
    waits when it would intersect another membrane or the dish wall. A separate
    random stream resolves competing proposals; accepted geometry immediately
    constrains later proposals. Existing module states retain their identities.
    """
    a, c = world.agents, world.config
    eligible = (
        ((a["modules"] < a["target_modules"]) & (a["growth_tick"] <= world.tick))
        .nonzero()
        .flatten()
    )
    if not len(eligible):
        return
    proposal = world.empty_agents(len(eligible))
    proposal["genome"] = a["genome"][eligible].clone()
    proposal["development_stage"] = a["modules"][eligible] + 1
    world.develop(proposal)
    neural_cost = proposal["brain_construction"] - a["brain_construction"][eligible]
    cost = c.growth_cost * (proposal["area"] - a["area"][eligible]) + neural_cost
    affordable = a["energy"][eligible] >= cost + c.growth_reserve * proposal["area"]
    world.totals["growth_attempts"] += len(eligible)
    world.totals["growth_unaffordable"] += int((~affordable).sum())
    a["growth_tick"][eligible] = world.tick + math.ceil(c.growth_retry * c.physics_hz)
    candidates = affordable.nonzero().flatten()
    order = torch.randperm(len(candidates), generator=world.rng["development"], device=world.device)
    for k in candidates[order].tolist():
        i = int(eligible[k])
        radius = proposal["radius"][k]
        within = (a["pos"][i] - c.diameter / 2).norm() <= c.diameter / 2 - radius
        overlap = (a["pos"] - a["pos"][i]).norm(dim=1) < a["radius"] + radius
        overlap[i] = False
        if not within or overlap.any():
            world.totals["growth_blocked"] += 1
            continue
        before = int(a["modules"][i])
        for key in (
            "development_stage",
            "modules",
            "module_mask",
            "module_offset",
            "radius",
            "area",
            "core_radius",
            "body_axis",
            "brain_construction",
            "brain_maintenance",
        ):
            a[key][i] = proposal[key][k]
        # Unexpressed module states are zero from birth and remain masked during
        # control updates. Recompute shared readouts for the enlarged body;
        # the new module first acts at the next scheduled controller update.
        a["actions"][i] = a["module_actions"][i].sum(0) / a["modules"][i]
        a["motors"][i] = a["actions"][i, :2]
        a["h"][i] = a["module_h"][i].sum(0) / a["modules"][i]
        a["energy"][i] -= cost[k]
        a["development_spent"][i] += cost[k]
        a["growth_tick"][i] = world.tick + math.ceil(c.growth_delay * c.physics_hz)
        world.totals["growths"] += 1
        world.totals["development_cost"] += float(cost[k])
        # Informational component, already included in development_cost.
        world.totals["neural_construction"] += float(neural_cost[k])
        world.events.append(
            dict(
                event="growth",
                time=(world.tick + 1) * c.dt,
                id=int(a["id"][i]),
                lineage=int(a["lineage"][i]),
                from_modules=before,
                modules=int(a["modules"][i]),
                target_modules=int(a["target_modules"][i]),
                energy_cost=float(cost[k]),
                radius=float(radius),
            )
        )
