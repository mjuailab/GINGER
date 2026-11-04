"""Coordinate-only graph construction and complete incoming neighborhoods."""

import numpy as np
import torch
from scipy.spatial import Delaunay


def build_graph(coordinates):
    coordinates = np.asarray(coordinates, np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) < 4
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("Delaunay requires at least four finite 2D coordinates")
    triangles = Delaunay(coordinates, qhull_options="QJ").simplices
    pairs = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    nodes = np.arange(len(coordinates), dtype=np.int64)
    edges = np.concatenate([pairs, pairs[:, ::-1], np.column_stack([nodes, nodes])])
    return np.ascontiguousarray(edges[np.lexsort((edges[:, 0], edges[:, 1]))].T)


def incoming_index(edge_index, n):
    """Index every incoming Delaunay edge, without padded or duplicated edges."""
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2
    assert edge_index.dtype == torch.long and bool(
        ((edge_index >= 0) & (edge_index < n)).all()
    )
    (source, target) = edge_index
    order = torch.argsort(target, stable=True)
    counts = torch.bincount(target, minlength=n)
    assert bool((counts > 0).all()), "Every cell needs its self-loop"
    starts = counts.cumsum(0) - counts
    slots = torch.arange(
        len(target), device=edge_index.device
    ) - starts.repeat_interleave(counts)
    incoming = torch.full(
        (n, int(counts.max())), -1, dtype=torch.long, device=edge_index.device
    )
    incoming[target[order], slots] = source[order]
    assert int((incoming >= 0).sum()) == edge_index.shape[1]
    return incoming


def incoming_neighborhood(incoming, cells):
    """Include ALL A -> B edges for root B, even when A is outside the batch."""
    cells = cells.to(incoming.device)
    candidates = incoming[cells]
    valid = candidates >= 0
    sources = candidates[valid]
    (nodes, inverse) = torch.unique(torch.cat([cells, sources]), return_inverse=True)
    roots = inverse[: len(cells)]
    target = roots[:, None].expand_as(candidates)[valid]
    edge = torch.stack([inverse[len(cells) :], target])
    return (nodes, edge, roots)
