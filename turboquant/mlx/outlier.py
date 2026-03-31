"""MLX-native outlier channel strategy for non-integer bit precision.

Ports turboquant/outlier.py to mlx.core.
Channel splitting and index operations are purely index-based — direct MLX port.
"""

import mlx.core as mx
import numpy as np
from dataclasses import dataclass

from turboquant.mlx.polar_quant import PolarQuantMLX
from turboquant.mlx.qjl import QJLMLX


@dataclass
class OutlierCompressedVectorMLX:
    """Container for outlier-strategy compressed vector — MLX variant."""
    outlier_indices: mx.array
    outlier_norms: mx.array
    normal_indices: mx.array
    normal_norms: mx.array
    qjl_signs: mx.array
    residual_norms: mx.array
    effective_bits: float


def _compute_channel_split(d: int, target_bits: float) -> tuple:
    """Compute channel split for fractional bit rates.

    Mirrors _compute_channel_split() from outlier.py exactly.
    """
    low_bits = int(np.floor(target_bits))
    high_bits = low_bits + 1
    frac = target_bits - low_bits
    n_outlier = int(round(d * frac))
    n_normal = d - n_outlier
    return n_outlier, high_bits, n_normal, low_bits


class OutlierTurboQuantMLX:
    """TurboQuantMLX with outlier channel strategy for non-integer bit rates.

    MLX port of OutlierTurboQuant. Splits channels into outlier (higher bits)
    and normal (lower bits) to achieve fractional average bit rates.

    Usage:
        oq = OutlierTurboQuantMLX(d=128, target_bits=2.5, seed=42)
        compressed = oq.quantize(x)
        x_hat = oq.dequantize(compressed)
    """

    def __init__(self, d: int, target_bits: float, seed: int = 42):
        self.d = d
        self.target_bits = target_bits

        n_outlier, high_bits, n_normal, low_bits = _compute_channel_split(d, target_bits)
        self.n_outlier = n_outlier
        self.n_normal = n_normal
        self.high_bits = high_bits
        self.low_bits = low_bits

        self.effective_bits = (n_outlier * high_bits + n_normal * low_bits) / d

        # Fixed channel indices — first n_outlier channels are outlier
        self.outlier_idx = np.arange(n_outlier)
        self.normal_idx = np.arange(n_outlier, d)

        self.pq_outlier = PolarQuantMLX(n_outlier, bit_width=high_bits - 1, seed=seed) if n_outlier > 0 else None
        self.pq_normal = PolarQuantMLX(n_normal, bit_width=low_bits - 1, seed=seed + 500) if n_normal > 0 else None
        self.qjl = QJLMLX(d, seed=seed + 1000)

    def quantize(self, x: mx.array) -> OutlierCompressedVectorMLX:
        """Quantize with outlier channel split."""
        single = x.ndim == 1
        if single:
            x = x[None, :]

        batch = x.shape[0]

        # Split channels via numpy indexing on mx.array
        x_np = np.array(x)
        x_outlier = mx.array(x_np[:, self.outlier_idx])   # (batch, n_outlier)
        x_normal = mx.array(x_np[:, self.normal_idx])     # (batch, n_normal)

        if self.pq_outlier is not None:
            inp = x_outlier if batch > 1 else x_outlier[0]
            out_idx, out_norms, out_residual = self.pq_outlier.quantize_and_residual(inp)
        else:
            out_idx = mx.array(np.array([], dtype=np.uint8))
            out_norms = mx.array(np.array([], dtype=np.float32))
            out_residual = mx.zeros((batch, 0) if batch > 1 else (0,))

        if self.pq_normal is not None:
            inp = x_normal if batch > 1 else x_normal[0]
            norm_idx, norm_norms, norm_residual = self.pq_normal.quantize_and_residual(inp)
        else:
            norm_idx = mx.array(np.array([], dtype=np.uint8))
            norm_norms = mx.array(np.array([], dtype=np.float32))
            norm_residual = mx.zeros((batch, 0) if batch > 1 else (0,))

        # Reconstruct full residual
        full_residual_np = np.zeros((batch, self.d), dtype=np.float32)
        if self.n_outlier > 0:
            res_np = np.array(out_residual)
            if res_np.ndim == 1:
                res_np = res_np[None, :]
            full_residual_np[:, self.outlier_idx] = res_np
        if self.n_normal > 0:
            res_np = np.array(norm_residual)
            if res_np.ndim == 1:
                res_np = res_np[None, :]
            full_residual_np[:, self.normal_idx] = res_np

        full_residual = mx.array(full_residual_np)
        inp_residual = full_residual if batch > 1 else full_residual[0]
        qjl_signs, residual_norms = self.qjl.quantize(inp_residual)

        if single:
            return OutlierCompressedVectorMLX(
                outlier_indices=out_idx,
                outlier_norms=out_norms,
                normal_indices=norm_idx,
                normal_norms=norm_norms,
                qjl_signs=qjl_signs,
                residual_norms=residual_norms,
                effective_bits=self.effective_bits,
            )

        return OutlierCompressedVectorMLX(
            outlier_indices=out_idx,
            outlier_norms=out_norms,
            normal_indices=norm_idx,
            normal_norms=norm_norms,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            effective_bits=self.effective_bits,
        )

    def dequantize(self, compressed: OutlierCompressedVectorMLX) -> mx.array:
        """Dequantize outlier-strategy compressed vector."""
        single = compressed.qjl_signs.ndim == 1

        if self.pq_outlier is not None:
            x_outlier = self.pq_outlier.dequantize(compressed.outlier_indices, compressed.outlier_norms)
        else:
            x_outlier = None

        if self.pq_normal is not None:
            x_normal = self.pq_normal.dequantize(compressed.normal_indices, compressed.normal_norms)
        else:
            x_normal = None

        x_qjl = self.qjl.dequantize(compressed.qjl_signs, compressed.residual_norms)

        if single:
            x_hat_np = np.zeros(self.d, dtype=np.float32)
            if self.n_outlier > 0 and x_outlier is not None:
                x_hat_np[self.outlier_idx] = np.array(x_outlier)
            if self.n_normal > 0 and x_normal is not None:
                x_hat_np[self.normal_idx] = np.array(x_normal)
            x_hat = mx.array(x_hat_np) + x_qjl
        else:
            batch = np.array(compressed.qjl_signs).shape[0]
            x_hat_np = np.zeros((batch, self.d), dtype=np.float32)
            if self.n_outlier > 0 and x_outlier is not None:
                x_hat_np[:, self.outlier_idx] = np.array(x_outlier)
            if self.n_normal > 0 and x_normal is not None:
                x_hat_np[:, self.normal_idx] = np.array(x_normal)
            x_hat = mx.array(x_hat_np) + x_qjl

        mx.eval(x_hat)
        return x_hat

    def compression_ratio(self, original_bits: int = 16) -> float:
        """Compression ratio vs original precision. Mirrors OutlierTurboQuant.compression_ratio."""
        per_vector_bits = self.d * self.effective_bits + 32 + 64
        original = self.d * original_bits
        return original / per_vector_bits
