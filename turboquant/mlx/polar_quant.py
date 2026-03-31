"""MLX-native PolarQuant: rotation + optimal scalar quantization.

Ports turboquant/polar_quant.py to mlx.core.
Rotation matrix and centroids are pre-computed at init time.
All quantize/dequantize operations run as mx.array graphs.
"""

import mlx.core as mx
import numpy as np

from turboquant.mlx.rotation import random_rotation_dense_mlx
from turboquant.mlx.codebook import optimal_centroids_mlx, nearest_centroid_indices_mlx


class PolarQuantMLX:
    """MSE-optimized vector quantizer via random rotation + scalar quantization.

    MLX port of PolarQuant. Rotation and centroids are mx.arrays initialized once.

    Usage:
        pq = PolarQuantMLX(d=128, bit_width=2, seed=42)
        indices, norms = pq.quantize(x)   # x: mx.array (d,) or (batch, d)
        x_hat = pq.dequantize(indices, norms)
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42, norm_correction: bool = True):
        self.d = d
        self.bit_width = bit_width
        self.n_centroids = 1 << bit_width
        self.norm_correction = norm_correction

        self.rotation = random_rotation_dense_mlx(d, seed)
        self.centroids = optimal_centroids_mlx(bit_width, d)
        mx.eval(self.rotation, self.centroids)

    def quantize(self, x: mx.array) -> tuple:
        """Quantize a vector or batch of vectors.

        Args:
            x: mx.array shape (d,) or (batch, d).

        Returns:
            (indices, norms) — uint8 indices, float32 norms.
        """
        single = x.ndim == 1
        if single:
            x = x[None, :]

        # Extract norms and normalize (paper page 5)
        norms = mx.linalg.norm(x, axis=1)          # (batch,)
        safe_norms = mx.where(norms > 0, norms, mx.ones_like(norms))
        x_normalized = x / safe_norms[:, None]

        # Rotate: (batch, d) @ rotation.T → (batch, d)
        y = x_normalized @ self.rotation.T

        # Nearest centroid per coordinate
        indices = nearest_centroid_indices_mlx(y, self.centroids)  # (batch, d) uint8

        mx.eval(indices, norms)

        if single:
            return indices[0], norms[0]
        return indices, norms

    def dequantize(self, indices: mx.array, norms: mx.array) -> mx.array:
        """Dequantize indices back to vectors.

        Args:
            indices: uint8 mx.array shape (d,) or (batch, d).
            norms: float32 mx.array scalar or (batch,).

        Returns:
            Reconstructed mx.array, same shape as original input.
        """
        single = indices.ndim == 1
        if single:
            indices = indices[None, :]
            norms = norms[None]

        # Look up centroids — indices are numpy uint8, centroids is mx.array
        indices_np = np.array(indices)
        centroids_np = np.array(self.centroids)
        y_hat_np = centroids_np[indices_np]         # (batch, d)
        y_hat = mx.array(y_hat_np)

        if self.norm_correction:
            y_hat_norms = mx.linalg.norm(y_hat, axis=1, keepdims=True)
            y_hat_norms = mx.where(y_hat_norms > 1e-10, y_hat_norms, mx.ones_like(y_hat_norms))
            y_hat = y_hat / y_hat_norms

        # Inverse rotate: (batch, d) @ rotation → (batch, d)
        x_hat_unit = y_hat @ self.rotation

        # Rescale by original norms
        x_hat = x_hat_unit * norms[:, None]

        mx.eval(x_hat)

        return x_hat[0] if single else x_hat

    def quantize_and_residual(self, x: mx.array) -> tuple:
        """Quantize and return (indices, norms, residual).

        Used by TurboQuantMLX's second stage (QJL on residual).
        """
        indices, norms = self.quantize(x)
        x_hat = self.dequantize(indices, norms)
        residual = x - x_hat
        mx.eval(residual)
        return indices, norms, residual
