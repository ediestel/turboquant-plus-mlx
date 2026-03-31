"""MLX-native rotation matrix generation.

Ports turboquant/rotation.py to mlx.core:
- random_rotation_dense: mx.linalg.qr-based Haar rotation
- fast_walsh_hadamard_mlx: butterfly WHT in MLX
- random_rotation_fast_mlx: structured D@H@D rotation components
"""

import math
import mlx.core as mx


def random_rotation_dense_mlx(d: int, seed: int) -> mx.array:
    """Haar-distributed random rotation via QR decomposition.

    Ports random_rotation_dense() from rotation.py.

    Args:
        d: Matrix dimension.
        seed: Integer seed for reproducibility.

    Returns:
        Orthogonal mx.array of shape (d, d) with det = +1.
    """
    # QR is not yet GPU-accelerated in MLX — run on CPU stream, same as numpy interop
    # This is init-time only (called once per compressor), not on the inference hot path
    import numpy as np
    rng = np.random.default_rng(seed)
    G_np = rng.standard_normal((d, d))
    Q_np, R_np = np.linalg.qr(G_np)

    # Fix signs to ensure Haar distribution (mirrors rotation.py lines 28-29)
    signs = np.sign(np.diag(R_np))
    signs[signs == 0] = 1.0
    Q_np = Q_np * signs[np.newaxis, :]

    # Ensure det = +1 (mirrors rotation.py lines 33-35)
    sign, _ = np.linalg.slogdet(Q_np)
    if sign < 0:
        Q_np[:, 0] = -Q_np[:, 0]

    Q = mx.array(Q_np.astype(np.float32))

    return Q


def fast_walsh_hadamard_mlx(x: mx.array) -> mx.array:
    """O(n log n) Walsh-Hadamard Transform via butterfly operations in MLX.

    Ports fast_walsh_hadamard_transform() from rotation.py.
    Input last dimension must be a power of 2.

    Args:
        x: mx.array of shape (..., n).

    Returns:
        Transformed array, same shape, normalized by 1/sqrt(n).
    """
    n = x.shape[-1]
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Last dimension must be a positive power of 2, got {n}")

    h = 1
    while h < n:
        x_reshaped = x.reshape(*x.shape[:-1], n // (h * 2), 2, h)
        a = x_reshaped[..., 0, :]
        b = x_reshaped[..., 1, :]
        x_reshaped = mx.concatenate(
            [mx.expand_dims(a + b, axis=-2), mx.expand_dims(a - b, axis=-2)],
            axis=-2,
        )
        x = x_reshaped.reshape(*x.shape[:-1], n)
        h *= 2

    return x * (1.0 / math.sqrt(n))


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def random_rotation_fast_mlx(d: int, seed: int) -> tuple:
    """Fast structured rotation components: D @ H @ D'.

    Ports random_rotation_fast() from rotation.py.

    Args:
        d: Original dimension.
        seed: Integer seed.

    Returns:
        (signs1, signs2, padded_d) as mx.arrays and int.
    """
    padded_d = _next_power_of_2(d)
    key1 = mx.random.key(seed)
    key2 = mx.random.key(seed + 1)
    # Random ±1 signs: sample from {0,1} then map to {-1,+1}
    bits1 = mx.random.randint(0, 2, shape=(padded_d,), key=key1)
    bits2 = mx.random.randint(0, 2, shape=(padded_d,), key=key2)
    signs1 = bits1.astype(mx.float32) * 2 - 1
    signs2 = bits2.astype(mx.float32) * 2 - 1
    return signs1, signs2, padded_d


def apply_fast_rotation_mlx(x: mx.array, signs1: mx.array, signs2: mx.array, padded_d: int) -> mx.array:
    """Apply structured random rotation to a vector or batch.

    Ports apply_fast_rotation / apply_fast_rotation_batch from rotation.py.

    Args:
        x: mx.array shape (d,) or (batch, d).
        signs1, signs2: From random_rotation_fast_mlx.
        padded_d: Power-of-2 padded dimension.

    Returns:
        Rotated mx.array, same shape as input.
    """
    single = x.ndim == 1
    if single:
        x = x[None, :]

    batch, d = x.shape

    # Pad to padded_d
    if d < padded_d:
        pad = mx.zeros((batch, padded_d - d))
        padded = mx.concatenate([x, pad], axis=1)
    else:
        padded = x

    # D1 @ x
    padded = padded * signs1[None, :]
    # H @ D1 @ x
    padded = fast_walsh_hadamard_mlx(padded)
    # D2 @ H @ D1 @ x
    padded = padded * signs2[None, :]

    result = padded[:, :d]
    return result[0] if single else result


def apply_fast_rotation_transpose_mlx(y: mx.array, signs1: mx.array, signs2: mx.array, padded_d: int) -> mx.array:
    """Apply transpose of the structured rotation.

    Ports apply_fast_rotation_transpose from rotation.py.
    Since D and H are symmetric, transpose is D1 @ H @ D2.
    """
    single = y.ndim == 1
    if single:
        y = y[None, :]

    batch, d = y.shape

    if d < padded_d:
        pad = mx.zeros((batch, padded_d - d))
        padded = mx.concatenate([y, pad], axis=1)
    else:
        padded = y

    padded = padded * signs2[None, :]
    padded = fast_walsh_hadamard_mlx(padded)
    padded = padded * signs1[None, :]

    result = padded[:, :d]
    return result[0] if single else result
