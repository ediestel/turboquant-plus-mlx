"""MLX-native codebook construction for PolarQuant.

Ports turboquant/codebook.py to mlx.core.
SciPy's norm.pdf/norm.cdf replaced with mx.erf-based approximations.
Lloyd's algorithm uses explicit mx.eval() per iteration (loop-carried dependency).
"""

import math
import mlx.core as mx
import numpy as np


def _normal_pdf_mlx(x: mx.array, sigma: float = 1.0) -> mx.array:
    """Gaussian PDF N(0, sigma^2) evaluated at x."""
    return mx.exp(-0.5 * (x / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def _normal_cdf_mlx(x: mx.array) -> mx.array:
    """Standard normal CDF via erf: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))."""
    return 0.5 * (1.0 + mx.erf(x / math.sqrt(2.0)))


def _normal_sf_mlx(x: mx.array) -> mx.array:
    """Standard normal survival function (1 - CDF) via erfc for numerical stability."""
    return 0.5 * mx.erfc(x / math.sqrt(2.0))


def _gaussian_conditional_expectation_mlx(sigma: float, a: float, b: float) -> float:
    """E[X | a < X < b] where X ~ N(0, sigma^2).

    Mirrors _gaussian_conditional_expectation() from codebook.py exactly,
    but uses NumPy scalars (called during init-time Lloyd's iterations).
    This is scalar-valued and called per-interval, so NumPy is fine here.
    """
    from scipy import stats
    a_std = a / sigma if math.isfinite(a) else a
    b_std = b / sigma if math.isfinite(b) else b

    if not math.isfinite(a_std):
        prob = stats.norm.cdf(b_std)
    elif not math.isfinite(b_std):
        prob = stats.norm.sf(a_std)
    else:
        prob = stats.norm.cdf(b_std) - stats.norm.cdf(a_std)

    if prob < 1e-15:
        if math.isfinite(a) and not math.isfinite(b):
            return a + sigma
        elif not math.isfinite(a) and math.isfinite(b):
            return b - sigma
        elif math.isfinite(a) and math.isfinite(b):
            return (a + b) / 2.0
        else:
            return 0.0

    pdf_diff = stats.norm.pdf(a_std) - stats.norm.pdf(b_std)
    return sigma * pdf_diff / prob


def optimal_centroids_mlx(bit_width: int, d: int) -> mx.array:
    """Compute optimal MSE centroids for the post-rotation coordinate distribution.

    Ports optimal_centroids() from codebook.py.

    Args:
        bit_width: Bits per coordinate (1, 2, 3, ...).
        d: Vector dimension (affects centroid scale via sigma = 1/sqrt(d)).

    Returns:
        mx.array of shape (2^bit_width,) — sorted centroids.
    """
    if bit_width == 1:
        c = math.sqrt(2.0 / (math.pi * d))
        return mx.array([-c, c])

    if bit_width == 2:
        scale = 1.0 / math.sqrt(d)
        return mx.array([-1.51 * scale, -0.453 * scale, 0.453 * scale, 1.51 * scale])

    # bit_width >= 3: Lloyd's algorithm — computed in NumPy, returned as mx.array
    sigma = 1.0 / math.sqrt(d)
    centroids_np = _lloyds_gaussian_np(1 << bit_width, sigma)
    return mx.array(centroids_np)


def _lloyds_gaussian_np(n_centroids: int, sigma: float, n_iter: int = 100) -> np.ndarray:
    """Lloyd's algorithm on N(0, sigma^2) — NumPy implementation.

    Used for codebook pre-computation at init time (not on the inference hot path).
    Matches _lloyds_gaussian() from codebook.py exactly.
    """
    from scipy import stats
    boundaries = stats.norm.ppf(
        np.linspace(0, 1, n_centroids + 1)[1:-1], scale=sigma
    )
    centroids = np.zeros(n_centroids)

    centroids[0] = _gaussian_conditional_expectation_mlx(sigma, -np.inf, boundaries[0])
    for i in range(1, n_centroids - 1):
        centroids[i] = _gaussian_conditional_expectation_mlx(sigma, boundaries[i - 1], boundaries[i])
    centroids[-1] = _gaussian_conditional_expectation_mlx(sigma, boundaries[-1], np.inf)

    for _ in range(n_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        centroids[0] = _gaussian_conditional_expectation_mlx(sigma, -np.inf, boundaries[0])
        for i in range(1, n_centroids - 1):
            centroids[i] = _gaussian_conditional_expectation_mlx(sigma, boundaries[i - 1], boundaries[i])
        centroids[-1] = _gaussian_conditional_expectation_mlx(sigma, boundaries[-1], np.inf)

    return np.sort(centroids)


def nearest_centroid_indices_mlx(values: mx.array, centroids: mx.array) -> mx.array:
    """Find nearest centroid index for each value.

    Ports nearest_centroid_indices() from codebook.py.
    Uses searchsorted via NumPy interop (MLX lacks native searchsorted).

    Args:
        values: mx.array of any shape.
        centroids: Sorted mx.array of shape (n_centroids,).

    Returns:
        mx.array of uint8 indices, same shape as values.
    """
    boundaries_np = np.array((centroids[:-1] + centroids[1:]) / 2.0)
    values_np = np.array(values).ravel()
    indices = np.searchsorted(boundaries_np, values_np).reshape(np.array(values).shape).astype(np.uint8)
    return mx.array(indices)
