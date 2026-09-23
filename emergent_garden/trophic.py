"""Passive energy provenance by expressed feeding allocation, not ancestry.

Detritus packets carry mixtures of producer credits without changing packet
count, geometry, energy, scent, or any controller input. A guild is a reporting
bin for a continuous trait; organisms have no access to these labels.
"""

import torch

GUILDS = ("grazer", "generalist", "scavenger")
PRODUCERS = (*GUILDS, "unattributed")


def guilds(diet):
    result = torch.ones_like(diet, dtype=torch.long)
    result[diet > 0.65] = 0
    result[diet < 0.35] = 2
    return result


class TrophicLedger:
    shapes = dict(
        fresh_uptake=(3,),
        detritus_introduced=(4,),
        detritus_uptake=(4, 3),
        detritus_dissipated=(4,),
        detritus_expired=(4,),
        predation_uptake=(3, 3),
        predation_kills=(3,),
    )

    def __init__(self, device):
        for key, shape in self.shapes.items():
            setattr(self, key, torch.zeros(shape, dtype=torch.float64, device=device))

    def feeding(self, old_energy, new_energy, kinds, credits, claims, consumers, received):
        """Record actual uptake and update credits to the remaining packet energy."""
        proportions = credits / old_energy.double().clamp_min(1e-300)[:, None]
        fresh = kinds[claims] == 0
        self.fresh_uptake.index_add_(0, consumers[fresh], received[fresh].double())
        parts = proportions[claims[~fresh]] * received[~fresh, None].double()
        self.detritus_uptake.index_add_(1, consumers[~fresh], parts.T)
        absorbed = torch.zeros_like(old_energy, dtype=torch.float64)
        absorbed.index_add_(0, claims, received.double())
        detritus = kinds == 1
        # Use actual packet depletion, including float32 subtraction roundoff,
        # so producer-credit conservation is independent of accumulation order.
        loss = old_energy.double() - new_energy.double() - absorbed
        self.detritus_dissipated += (proportions[detritus] * loss[detritus, None]).sum(0)
        return proportions * new_energy.double()[:, None]

    def recycled_credits(self, packet_count, claims, consumers, amounts, keep, energy):
        credits = torch.zeros((packet_count * 4,), dtype=torch.float64, device=amounts.device)
        credits.index_add_(0, claims * 4 + consumers, amounts.double())
        credits = credits.reshape(packet_count, 4)[keep]
        credits *= energy.double()[:, None] / credits.sum(1).clamp_min(1e-300)[:, None]
        return credits

    def predation(self, prey, predators, received, killed):
        self.predation_uptake.flatten().index_add_(0, prey * 3 + predators, received.double())
        self.predation_kills += torch.bincount(killed, minlength=3)

    def metrics(self, credits, kinds):
        result = {f"trophic_{key}": getattr(self, key).tolist() for key in self.shapes}
        stock = credits[kinds == 1].sum(0)
        error = self.detritus_introduced - self.detritus_uptake.sum(1)
        error -= self.detritus_dissipated + self.detritus_expired + stock
        result.update(
            trophic_guilds=list(GUILDS),
            trophic_producers=list(PRODUCERS),
            trophic_detritus_stock=stock.tolist(),
            trophic_detritus_balance_error=error.tolist(),
        )
        return result

    def state_dict(self):
        return {key: getattr(self, key).clone() for key in self.shapes}

    @classmethod
    def from_state(cls, state, device):
        ledger = cls(device)
        for key in cls.shapes:
            setattr(ledger, key, state[key].to(device).clone())
        return ledger
