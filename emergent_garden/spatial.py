"""Local candidate lookup; memory scales with local density, not all world pairs."""

import math

import torch


def neighbors(query, points, cell_size, extent):
    """Return query/point indices in the same or eight adjacent grid cells."""
    device = query.device
    empty = torch.empty(0, device=device, dtype=torch.long)
    if len(query) == 0 or len(points) == 0:
        return empty, empty
    width = math.ceil(extent / cell_size) + 1
    cells = (points / cell_size).floor().long().clamp(0, width - 1)
    keys, order = (cells[:, 0] + width * cells[:, 1]).sort(stable=True)
    offsets = torch.tensor([[x, y] for x in (-1, 0, 1) for y in (-1, 0, 1)], device=device)
    qc = (query / cell_size).floor().long()[:, None, :] + offsets
    valid = ((qc >= 0) & (qc < width)).all(-1).flatten()
    qkeys = (qc[..., 0] + width * qc[..., 1]).flatten()[valid]
    qi = torch.arange(len(query), device=device).repeat_interleave(9)[valid]
    lo = torch.searchsorted(keys, qkeys)
    hi = torch.searchsorted(keys, qkeys, right=True)
    count = hi - lo
    rows = torch.arange(len(count), device=device).repeat_interleave(count)
    start = (count.cumsum(0) - count).repeat_interleave(count)
    locations = lo[rows] + torch.arange(len(rows), device=device) - start
    return qi[rows], order[locations]
