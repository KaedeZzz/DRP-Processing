"""Spectral TV decomposition utilities for separating text and paper features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class SpectralTVResult:
    times: np.ndarray
    phi: list[np.ndarray]
    bands: list[np.ndarray]
    u_history: list[np.ndarray]


def _tv_divergence(u: np.ndarray, eps: float) -> np.ndarray:
    """Compute divergence of normalized gradient with Neumann boundaries."""
    grad_x = np.zeros_like(u)
    grad_y = np.zeros_like(u)
    grad_x[:, :-1] = u[:, 1:] - u[:, :-1]
    grad_y[:-1, :] = u[1:, :] - u[:-1, :]

    norm = np.sqrt(grad_x**2 + grad_y**2 + eps**2)
    px = grad_x / norm
    py = grad_y / norm

    div = np.zeros_like(u)
    div[:, :-1] += px[:, :-1]
    div[:, 1:] -= px[:, :-1]
    div[:-1, :] += py[:-1, :]
    div[1:, :] -= py[:-1, :]
    return div


def spectral_tv_decomposition(
    image: np.ndarray,
    *,
    num_steps: int = 160,
    dt: float = 0.2,
    eps: float = 1e-6,
    band_edges: Sequence[float] | None = None,
    verbose: bool = False,
    progress_every: int | None = None,
) -> SpectralTVResult:
    """Run spectral TV flow and return spectral bands.

    Args:
        image: 2D grayscale image (float32/float64 recommended).
        num_steps: Number of TV flow steps.
        dt: Time step for TV flow (smaller is more stable).
        eps: Small stabilizer for gradient norm.
        band_edges: Optional time edges for band integration.
        verbose: Print progress information during the flow.
        progress_every: Optional manual stride for step logging when verbose.

    Returns:
        SpectralTVResult with time samples, spectral responses, band images,
        and the full TV-flow history.
    """
    u = image.astype(np.float32, copy=True)
    u_history = [u]
    times = np.arange(0, num_steps + 1, dtype=np.float32) * dt
    log_every = progress_every or max(1, num_steps // 10)
    if verbose:
        band_info = band_edges if band_edges is not None else "(auto)"
        print(f"[spectral_tv] start flow: num_steps={num_steps}, dt={dt}, eps={eps}, bands={band_info}")

    for step in range(num_steps):
        div = _tv_divergence(u, eps)
        u = u + dt * div
        u_history.append(u)
        if verbose and ((step + 1) % log_every == 0 or step == num_steps - 1):
            print(f"[spectral_tv] step {step + 1}/{num_steps} complete")

    phi: list[np.ndarray] = []
    for k in range(1, num_steps):
        t = times[k]
        second_derivative = (u_history[k - 1] - 2 * u_history[k] + u_history[k + 1]) / (
            dt**2
        )
        phi.append(t * second_derivative)

    if band_edges is None:
        band_edges = (0.0, dt * 2, dt * 6, dt * 20, dt * num_steps)

    if verbose:
        print(f"[spectral_tv] integrating {len(band_edges) - 1} bands with edges {band_edges}")
    bands = _integrate_bands(phi, times[1:-1], band_edges)

    if verbose:
        print("[spectral_tv] decomposition complete")
    return SpectralTVResult(times=times, phi=phi, bands=bands, u_history=u_history)


def _integrate_bands(
    phi: Sequence[np.ndarray],
    times: np.ndarray,
    band_edges: Sequence[float],
) -> list[np.ndarray]:
    bands: list[np.ndarray] = []
    for start, end in zip(band_edges[:-1], band_edges[1:]):
        mask = (times >= start) & (times < end)
        if not np.any(mask):
            bands.append(np.zeros_like(phi[0]))
            continue
        selected = [phi[idx] for idx in np.where(mask)[0]]
        bands.append(np.sum(selected, axis=0) * (times[1] - times[0]))
    return bands


def split_text_background(
    image: np.ndarray,
    *,
    text_max_time: float = 4.0,
    num_steps: int = 160,
    dt: float = 0.1,
    eps: float = 1e-7,
    verbose: bool = False,
    progress_every: int | None = None,
    band_edges: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, SpectralTVResult]:
    """Split an image into text (small scales) and background (large scales)."""
    result = spectral_tv_decomposition(
        image,
        num_steps=num_steps,
        dt=dt,
        eps=eps,
        verbose=verbose,
        progress_every=progress_every,
        band_edges=band_edges,
    )
    times = result.times[1:-1]
    text_mask = times <= text_max_time
    if verbose:
        print(f"[spectral_tv] integrating text/background split at t<={text_max_time}")
    k = int(round(text_max_time / dt))
    background = result.u_history[k]
    text_layer = image - background
    # text_layer = np.sum(
    #     [result.phi[idx] for idx in np.where(text_mask)[0]], axis=0
    # ) * dt
    background = image - text_layer
    return text_layer, background, result
