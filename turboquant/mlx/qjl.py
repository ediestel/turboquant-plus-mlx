"""MLX-native QJL: Quantized Johnson-Lindenstrauss 1-bit quantizer.

Ports turboquant/qjl.py to mlx.core.
Sign quantization via mx.sign, projection via matrix multiply.
"""

import math
import mlx.core as mx
import numpy as np

QJL_CONST = math.sqrt(math.pi / 2)


class QJLMLX:
    """Quantized Johnson-Lindenstrauss 1-bit quantizer — MLX native.

    MLX port of QJL. Projection matrix S is an mx.array initialized once.

    Usage:
        qjl = QJLMLX(d=128, seed=42)
        signs, norm = qjl.quantize(residual)
        r_hat = qjl.dequantize(signs, norm)
    """

    def __init__(self, d: int, seed: int = 123):
        """
        Args:
            d: Vector dimension.
            seed: Random seed for projection matrix.
        """
        self.d = d
        # Use NumPy seeding to match QJL.__init__ exactly (same PRNG, same S matrix)
        # This ensures parity tests pass and the same seed produces the same S
        import numpy as np
        rng = np.random.default_rng(seed)
        S_np = rng.standard_normal((d, d)).astype(np.float32)
        self.S = mx.array(S_np)
        mx.eval(self.S)

    def quantize(self, r: mx.array) -> tuple:
        """Quantize residual vector(s) to sign bits.

        Args:
            r: mx.array shape (d,) or (batch, d).

        Returns:
            (signs, norms) — int8 signs {+1,-1}, float32 norms.
        """
        single = r.ndim == 1
        if single:
            r = r[None, :]

        # Norms before projection
        norms = mx.linalg.norm(r, axis=1)          # (batch,)

        # Project: (batch, d) @ S.T → (batch, d)
        projected = r @ self.S.T

        # Sign quantization — mx.sign returns {-1, 0, +1}
        signs = mx.sign(projected)
        # Map zeros to +1 (matches qjl.py line 62)
        signs = mx.where(signs == 0, mx.ones_like(signs), signs)

        mx.eval(signs, norms)

        if single:
            return signs[0], norms[0]
        return signs, norms

    def dequantize(self, signs: mx.array, norms: mx.array) -> mx.array:
        """Dequantize sign bits back to approximate residual.

        Formula: x̃_qjl = sqrt(π/2) / d · ||r|| · S^T @ signs

        Args:
            signs: mx.array shape (d,) or (batch, d).
            norms: Residual norms, scalar or (batch,).

        Returns:
            Approximate residual, same shape as original.
        """
        single = signs.ndim == 1
        if single:
            signs = signs[None, :]
            norms = norms[None]

        signs_f = signs.astype(mx.float32)

        # (batch, d) @ S → (batch, d)  [S^T @ signs per-vector = signs @ S]
        reconstructed = signs_f @ self.S

        # Scale: sqrt(π/2) / m * norm  (m = d, matches qjl.py line 95)
        m = self.S.shape[0]
        scale = (QJL_CONST / m) * norms            # (batch,)
        reconstructed = reconstructed * scale[:, None]

        mx.eval(reconstructed)

        return reconstructed[0] if single else reconstructed
