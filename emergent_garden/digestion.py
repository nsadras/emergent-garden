"""V18 food collection: carried packets retain their energy, expiry, and provenance.

Digestion still uses EcologyWorld.feed's finite processing and assimilation laws.
Only collection is immediate. Ownership uses permanent creature IDs, so deaths
and population compaction cannot transfer a meal to an unrelated creature.
"""

import torch

GUT_INPUTS = ("gut_fresh", "gut_detritus")


def enabled(world):
    return (
        world.config.ecology_version >= 18
        and world.config.gut_capacity > 0
        and world.ablation != "no_gut"
    )


def owned_food(world):
    """Return live owner rows and their carried packet rows."""
    packets = (world.food_owner >= 0).nonzero().flatten()
    ids = world.agents["id"]
    if not len(packets):
        return packets, packets
    if not len(ids):
        raise ValueError("Carried food has no living owner")
    order = ids.argsort()
    rows = torch.searchsorted(ids[order], world.food_owner[packets]).clamp_max(len(ids) - 1)
    rows = order[rows]
    if not torch.equal(ids[rows], world.food_owner[packets]):
        raise ValueError("Carried food refers to a missing creature")
    return rows, packets


def capacity(world):
    a, c = world.agents, world.config
    tissue = a["modules"] * (a["core_radius"] / c.body_radius).square()
    allocation = torch.stack((a["diet"], 1 - a["diet"]), 1)
    return c.gut_capacity * tissue[:, None] * allocation


def loads(world):
    amounts = torch.zeros((world.population, 2), dtype=torch.float64, device=world.device)
    owners, packets = owned_food(world)
    amounts.flatten().index_add_(
        0, 2 * owners + world.food_kind[packets], world.food_energy[packets]
    )
    return amounts


def fullness(world):
    if not enabled(world):
        return torch.zeros((world.population, 2), device=world.device)
    return (loads(world) / capacity(world).clamp_min(1e-20)).clamp(0, 1).float()


def follow(world):
    owners, packets = owned_food(world)
    world.food_pos[packets] = world.agents["pos"][owners]


def release_dead(world):
    owners, packets = owned_food(world)
    dead = world.agents["energy"][owners] <= 0
    world.food_pos[packets[dead]] = world.agents["pos"][owners[dead]]
    world.food_owner[packets[dead]] = -1


def compact(world):
    """Merge equivalent carried packets to avoid one new crumb per bite forever.

    Keeping expiry and readiness in the key preserves individual food timers.
    Summing producer credits is a transfer, never new detritus production.
    """
    owners, packets = owned_food(world)
    if not len(packets):
        return
    columns = ("food_owner", "food_kind", "food_patch", "food_expiry", "food_ready")
    keys = torch.stack([getattr(world, key)[packets] for key in columns], 1)
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    if len(unique) == len(keys):
        return
    energy = torch.zeros(len(unique), dtype=torch.float64, device=world.device)
    credits = torch.zeros((len(unique), 4), dtype=torch.float64, device=world.device)
    energy.index_add_(0, inverse, world.food_energy[packets])
    credits.index_add_(0, inverse, world.food_credit[packets])
    # Every member of a group has the same owner position.
    positions = torch.zeros((len(unique), 2), device=world.device)
    positions.index_add_(0, inverse, world.agents["pos"][owners])
    positions /= torch.bincount(inverse, minlength=len(unique))[:, None]
    world.filter_food(world.food_owner < 0)
    for k, name in enumerate(columns):
        setattr(world, name, torch.cat((getattr(world, name), unique[:, k])))
    world.food_energy = torch.cat((world.food_energy, energy))
    world.food_credit = torch.cat((world.food_credit, credits))
    world.food_pos = torch.cat((world.food_pos, positions))


def capture(world, owners, packets):
    """Share contacted free food simultaneously, bounded by each pathway's room."""
    free = world.food_owner[packets] < 0
    owners, packets = owners[free], packets[free]
    if not len(packets):
        return
    room = (capacity(world).double() - loads(world)).clamp_min(0)
    slot = 2 * owners + world.food_kind[packets]
    # A full pathway cannot reserve food against another creature with room.
    keep = room.flatten()[slot] > 0
    owners, packets, slot = owners[keep], packets[keep], slot[keep]
    if not len(packets):
        return
    claims = torch.bincount(packets, minlength=len(world.food_energy)).clamp_min(1)
    shares = world.food_energy[packets] / claims[packets]
    demand = torch.zeros_like(room).flatten().index_add_(0, slot, shares)
    shares *= (room.flatten() / demand.clamp_min(1e-300)).clamp_max(1)[slot]
    moved = torch.zeros_like(world.food_energy).index_add_(0, packets, shares)
    # All transactions are float64 in V18, including a partially collected meal.
    proportions = world.food_credit / world.food_energy.clamp_min(1e-300)[:, None]
    new_credits = proportions[packets] * shares[:, None]
    world.food_energy = (world.food_energy - moved).clamp_min(0)
    world.food_credit = proportions * world.food_energy[:, None]
    for key in ("food_kind", "food_patch", "food_expiry", "food_ready"):
        data = getattr(world, key)
        setattr(world, key, torch.cat((data, data[packets])))
    world.food_owner = torch.cat((world.food_owner, world.agents["id"][owners]))
    world.food_pos = torch.cat((world.food_pos, world.agents["pos"][owners]))
    world.food_energy = torch.cat((world.food_energy, shares))
    world.food_credit = torch.cat((world.food_credit, new_credits))
    world.totals["food_collected"] += shares.sum().item()
    world.filter_food(world.food_energy > 0)
    compact(world)
