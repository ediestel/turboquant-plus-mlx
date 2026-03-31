"""MLX-native TurboQuant: PolarQuantMLX (b-1 bits) + QJLMLX (1 bit).

Ports turboquant/turboquant.py to mlx.core.
"""

import mlx.core as mx
import numpy as np
from dataclasses import dataclass

from turboquant.mlx.polar_quant import PolarQuantMLX
from turboquant.mlx.qjl import QJLMLX


@dataclass
class CompressedVectorMLX:
    """Container for a TurboQuantMLX-compressed vector.

    Fields mirror CompressedVector from turboquant.py exactly.
    Arrays are mx.arrays (lazy — call mx.eval before inspecting values).
    """
    mse_indices: mx.array    # uint8, shape (d,) or (batch, d)
    vector_norms: mx.array   # float32, scalar or (batch,)
    qjl_signs: mx.array      # float32 {+1,-1}, shape (d,) or (batch, d)
    residual_norms: mx.array # float32, scalar or (batch,)
    bit_width: int


class TurboQuantMLX:
    """Full TurboQuant quantizer in MLX: PolarQuantMLX (b-1 bits) + QJLMLX (1 bit).

    MLX port of TurboQuant. Initializes once; quantize/dequantize run on GPU.

    Usage:
        tq = TurboQuantMLX(d=128, bit_width=3, seed=42)
        compressed = tq.quantize(x)   # x: mx.array
        x_hat = tq.dequantize(compressed)
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42, norm_correction: bool = True):
        if bit_width < 2:
            raise ValueError("TurboQuantMLX requires bit_width >= 2.")

        self.d = d
        self.bit_width = bit_width

        self.polar_quant = PolarQuantMLX(d, bit_width=bit_width - 1, seed=seed, norm_correction=norm_correction)
        self.qjl = QJLMLX(d, seed=seed + 1000)

    def quantize(self, x: mx.array) -> CompressedVectorMLX:
        """Quantize a vector or batch.

        Args:
            x: mx.array shape (d,) or (batch, d).

        Returns:
            CompressedVectorMLX.
        """
        mse_indices, vector_norms, residual = self.polar_quant.quantize_and_residual(x)
        qjl_signs, residual_norms = self.qjl.quantize(residual)

        return CompressedVectorMLX(
            mse_indices=mse_indices,
            vector_norms=vector_norms,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            bit_width=self.bit_width,
        )

    def dequantize(self, compressed: CompressedVectorMLX) -> mx.array:
        """Dequantize back to approximate vector.

        Args:
            compressed: CompressedVectorMLX from quantize().

        Returns:
            Reconstructed mx.array, same shape as original.
        """
        x_mse = self.polar_quant.dequantize(compressed.mse_indices, compressed.vector_norms)
        x_qjl = self.qjl.dequantize(compressed.qjl_signs, compressed.residual_norms)
        result = x_mse + x_qjl
        mx.eval(result)
        return result

    def compressed_size_bits(self, n_vectors: int) -> int:
        """Total storage bits for n_vectors. Mirrors TurboQuant.compressed_size_bits."""
        per_vector = self.d * self.bit_width
        norms = 64  # 2 × float32
        return n_vectors * (per_vector + norms)

    def compression_ratio(self, original_bits_per_value: int = 16) -> float:
        """Compression ratio vs original precision."""
        original_per_vector = self.d * original_bits_per_value
        compressed_per_vector = self.d * self.bit_width + 64
        return original_per_vector / compressed_per_vector


class TurboQuantMSEMLX:
    """MSE-only TurboQuant (Algorithm 1) in MLX — no QJL stage.

    MLX port of TurboQuantMSE. Use for V cache compression.
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42, norm_correction: bool = True):
        self.d = d
        self.bit_width = bit_width
        self.polar_quant = PolarQuantMLX(d, bit_width=bit_width, seed=seed, norm_correction=norm_correction)

    def quantize(self, x: mx.array) -> tuple:
        """Returns (indices, norms)."""
        return self.polar_quant.quantize(x)

    def dequantize(self, indices: mx.array, norms: mx.array) -> mx.array:
        return self.polar_quant.dequantize(indices, norms)
