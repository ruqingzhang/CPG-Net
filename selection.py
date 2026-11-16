"""Deterministic spatial sampling and task-weighted marginal-coverage selection."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

SelectionMode = Literal["coverage", "fps", "topk", "all"]


def token_coordinates(grid_size: int, *, device: torch.device) -> torch.Tensor:
    """Return row-major token centers in normalized (x, y) image coordinates."""
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    axis = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) / grid_size
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def _validate(scores: torch.Tensor, tokens: torch.Tensor, budget: int) -> int:
    if scores.ndim != 2 or tokens.ndim != 3:
        raise ValueError("scores must be [B,N] and tokens must be [B,N,C]")
    if scores.shape[:2] != tokens.shape[:2]:
        raise ValueError("scores and tokens must refer to the same token grid")
    n = scores.shape[1]
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError("tokens must form a square patch grid")
    if not 1 <= budget <= n:
        raise ValueError(f"budget must be in [1,{n}]")
    if not torch.isfinite(scores).all() or not torch.isfinite(tokens).all():
        raise ValueError("scores and tokens must contain only finite values")
    return side


@lru_cache(maxsize=32)
def _cached_fps(side: int, budget: int) -> tuple[int, ...]:
    """Cache a deterministic FPS order without transferring indices from a GPU."""
    axis = (np.arange(side, dtype=np.float64) + 0.5) / side
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    coordinates = np.stack((xx, yy), axis=-1).reshape(-1, 2)
    squared_distance = ((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(axis=-1)
    first = int(np.argmin(((coordinates - 0.5) ** 2).sum(axis=-1)))
    chosen = [first]
    minimum_distance = squared_distance[:, first].copy()
    taken = np.zeros(side * side, dtype=bool)
    taken[first] = True
    for _ in range(1, budget):
        candidate_distances = np.where(taken, -1.0, minimum_distance)
        next_index = int(np.argmax(candidate_distances))
        chosen.append(next_index)
        taken[next_index] = True
        minimum_distance = np.minimum(minimum_distance, squared_distance[:, next_index])
    return tuple(chosen)


def _fps_indices(coordinates: torch.Tensor, budget: int) -> torch.Tensor:
    side = math.isqrt(coordinates.shape[0])
    return torch.tensor(_cached_fps(side, budget), dtype=torch.long, device=coordinates.device)


def coverage_kernel(
    tokens: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    feature_temperature: float = 0.2,
    spatial_sigma: float = 0.125,
    coverage_radius: float = 0.25,
) -> torch.Tensor:
    """Compute the truncated feature-spatial kernel, with rows as demands."""
    if min(feature_temperature, spatial_sigma, coverage_radius) <= 0:
        raise ValueError("coverage kernel parameters must be positive")
    z = F.normalize(tokens.float(), dim=-1, eps=1e-8)
    cosine = torch.matmul(z, z.transpose(1, 2)).clamp(-1.0, 1.0)
    d2 = (coordinates[:, None, :] - coordinates[None, :, :]).square().sum(-1)
    feature = torch.exp(-(1.0 - cosine).clamp_min(0.0) / feature_temperature)
    spatial = torch.exp(-d2 / (2.0 * spatial_sigma**2))
    return feature * spatial[None, :, :] * (d2 <= coverage_radius**2)[None, :, :]


def coverage_value(
    indices: torch.Tensor,
    scores: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Evaluate sum_i q_i max_{j in S}(q_j * kappa_ij) for each sample."""
    if indices.ndim != 2:
        raise ValueError("indices must be [B,K]")
    supplied = kernel * scores.float().unsqueeze(1)
    chosen = torch.gather(
        supplied, dim=2, index=indices[:, None, :].expand(-1, scores.shape[1], -1)
    )
    return (scores.float() * chosen.amax(dim=-1)).sum(dim=1)


@torch.no_grad()
def select_nodes(
    scores: torch.Tensor,
    tokens: torch.Tensor,
    *,
    budget: int = 64,
    context_fraction: float = 0.125,
    mode: SelectionMode = "coverage",
    feature_temperature: float = 0.2,
    spatial_sigma: float = 0.125,
    coverage_radius: float = 0.25,
    fallback_tolerance: float = 1e-12,
) -> torch.Tensor:
    """Select fixed-budget token indices without backpropagating through selection.

    The first round uses deterministic farthest-point sampling for context nodes.
    Remaining nodes maximize marginal task-weighted coverage. Degenerate gains
    fall back to farthest-point sampling, and token indices resolve score ties.
    """
    side = _validate(scores, tokens, budget)
    if not 0.0 <= context_fraction <= 1.0:
        raise ValueError("context_fraction must lie in [0,1]")
    if mode not in ("coverage", "fps", "topk", "all"):
        raise ValueError(f"Unsupported selection mode: {mode}")
    scores = scores.detach().float().clamp(0.0, 1.0)
    tokens = tokens.detach().float()
    coords = token_coordinates(side, device=scores.device)
    if mode == "all":
        if budget != scores.shape[1]:
            raise ValueError("All-node interaction requires budget equal to token count")
        return torch.arange(budget, device=scores.device)[None, :].expand(scores.shape[0], -1).clone()
    if mode == "topk":
        return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :budget]

    if mode == "fps":
        sequence = _fps_indices(coords, budget)
        return sequence[None, :].expand(scores.shape[0], -1).clone()

    initial_count = min(budget, int(math.floor(context_fraction * budget + 0.5)))
    initial = _fps_indices(coords, initial_count) if initial_count else None
    if initial_count == budget:
        return initial[None, :].expand(scores.shape[0], -1).clone()

    kernel = coverage_kernel(
        tokens,
        coords,
        feature_temperature=feature_temperature,
        spatial_sigma=spatial_sigma,
        coverage_radius=coverage_radius,
    )
    # supplied[b,i,j] denotes coverage at i provided by candidate j.
    supplied = kernel * scores[:, None, :]
    squared_distance = (coords[:, None, :] - coords[None, :, :]).square().sum(-1)
    batch_size, token_count = scores.shape
    taken = torch.zeros((batch_size, token_count), dtype=torch.bool, device=scores.device)
    if initial_count:
        starting = initial[None, :].expand(batch_size, -1)
        taken.scatter_(1, starting, True)
        initial_supply = supplied.gather(
            dim=2, index=starting[:, None, :].expand(-1, token_count, -1)
        )
        current_coverage = initial_supply.amax(dim=-1)
        minimum_distance = squared_distance[:, initial].amin(dim=1)[None].expand(
            batch_size, -1
        ).clone()
        chosen = [starting[:, idx] for idx in range(initial_count)]
    else:
        current_coverage = torch.zeros_like(scores)
        minimum_distance = -(coords - 0.5).square().sum(dim=-1)[None].expand(
            batch_size, -1
        ).clone()
        chosen = []

    for _ in range(initial_count, budget):
        improvements = (supplied - current_coverage[:, :, None]).clamp_min(0.0)
        marginal_gains = (scores[:, :, None] * improvements).sum(dim=1)
        gains = marginal_gains.masked_fill(taken, -1.0)
        best_gains, best_indices = gains.max(dim=1)
        fallback_indices = minimum_distance.masked_fill(taken, -1.0).argmax(dim=1)
        next_indices = torch.where(best_gains <= fallback_tolerance, fallback_indices, best_indices)
        chosen.append(next_indices)
        taken.scatter_(1, next_indices[:, None], True)
        next_coverage = supplied.gather(
            dim=2, index=next_indices[:, None, None].expand(-1, token_count, 1)
        ).squeeze(-1)
        current_coverage = torch.maximum(current_coverage, next_coverage)
        minimum_distance = torch.minimum(minimum_distance, squared_distance[next_indices])
    return torch.stack(chosen, dim=1)
