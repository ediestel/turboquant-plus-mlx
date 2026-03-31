"""MLX bit-packing utilities.

Ports turboquant/utils.py to mlx.core.
Sub-byte packing uses NumPy interop (MLX lacks native packbits/unpackbits).
"""

import mlx.core as mx
import numpy as np


def pack_bits_mlx(signs: mx.array) -> np.ndarray:
    """Pack {+1, -1} sign array into uint8 bitfield.

    Ports pack_bits() from utils.py. Uses NumPy interop for packbits.
    Returns np.ndarray (not mx.array) since bit-packed data is typically
    stored/transferred, not computed on.

    Args:
        signs: mx.array {+1,-1} of shape (d,) or (batch, d).

    Returns:
        uint8 np.ndarray of shape (ceil(d/8),) or (batch, ceil(d/8)).
    """
    mx.eval(signs)
    signs_np = np.array(signs).astype(np.int8)

    bits = (signs_np > 0).astype(np.uint8)

    if bits.ndim == 1:
        padded_len = (len(bits) + 7) // 8 * 8
        padded = np.zeros(padded_len, dtype=np.uint8)
        padded[:len(bits)] = bits
        return np.packbits(padded)
    else:
        batch, d = bits.shape
        padded_len = (d + 7) // 8 * 8
        padded = np.zeros((batch, padded_len), dtype=np.uint8)
        padded[:, :d] = bits
        return np.packbits(padded, axis=1)


def unpack_bits_mlx(packed: np.ndarray, d: int) -> mx.array:
    """Unpack uint8 bitfield back to {+1, -1} signs as mx.array.

    Ports unpack_bits() from utils.py.

    Args:
        packed: uint8 np.ndarray from pack_bits_mlx.
        d: Original dimension (to truncate padding).

    Returns:
        mx.array int8 of shape (d,) or (batch, d) with values {+1,-1}.
    """
    if packed.ndim == 1:
        bits = np.unpackbits(packed)[:d]
        signs_np = (bits.astype(np.int8) * 2 - 1)
    else:
        bits = np.unpackbits(packed, axis=1)[:, :d]
        signs_np = (bits.astype(np.int8) * 2 - 1)

    return mx.array(signs_np)


def pack_indices_mlx(indices: mx.array, bit_width: int) -> np.ndarray:
    """Pack b-bit indices into compact byte array.

    Ports pack_indices() from utils.py. Uses NumPy interop for packbits.
    Returns np.ndarray since packed data is stored, not computed on.

    Args:
        indices: mx.array integer indices, shape (d,) or (batch, d).
        bit_width: Bits per index (1–8).

    Returns:
        Packed uint8 np.ndarray.
    """
    if bit_width <= 0 or bit_width > 8:
        raise ValueError(f"bit_width must be 1-8, got {bit_width}")

    mx.eval(indices)
    indices_np = np.array(indices)

    if bit_width <= 4:
        flat = indices_np.ravel().astype(np.uint8)
        bits = np.zeros(len(flat) * bit_width, dtype=np.uint8)
        for b in range(bit_width):
            bits[b::bit_width] = (flat >> (bit_width - 1 - b)) & 1
        return np.packbits(bits).reshape(-1)
    else:
        return indices_np.astype(np.uint8)


def memory_footprint_bytes(n_vectors: int, d: int, bit_width: int) -> dict:
    """Calculate memory footprint of compressed KV cache.

    Mirrors memory_footprint_bytes() from utils.py exactly.
    """
    mse_bits = bit_width - 1
    qjl_bits = 1

    mse_bytes = int(np.ceil(n_vectors * d * mse_bits / 8))
    qjl_bytes = int(np.ceil(n_vectors * d * qjl_bits / 8))
    norm_bytes = n_vectors * 8  # 2 × float32
    total = mse_bytes + qjl_bytes + norm_bytes
    original = n_vectors * d * 2  # fp16

    return {
        "mse_indices_bytes": mse_bytes,
        "qjl_signs_bytes": qjl_bytes,
        "norms_bytes": norm_bytes,
        "total_bytes": total,
        "original_fp16_bytes": original,
        "compression_ratio": original / total if total > 0 else float("inf"),
    }
