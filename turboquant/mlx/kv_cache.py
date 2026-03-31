"""MLX-native KV cache compressor.

Ports turboquant/kv_cache.py to mlx.core.
Operates on mx.array KV tensors with batched compress/decompress per layer/head.
"""

import mlx.core as mx
import numpy as np
from dataclasses import dataclass, field

from turboquant.mlx.turboquant import TurboQuantMLX, TurboQuantMSEMLX, CompressedVectorMLX


@dataclass
class CompressedKVCacheMLX:
    """Container for a compressed KV cache — MLX variant."""
    k_compressed: list = field(default_factory=list)   # list[list[CompressedVectorMLX]]
    v_indices: list = field(default_factory=list)       # list[list[mx.array]]
    v_norms: list = field(default_factory=list)         # list[list[mx.array]]

    num_layers: int = 0
    num_heads: int = 0
    seq_len: int = 0
    head_dim: int = 0
    k_bit_width: int = 0
    v_bit_width: int = 0


class KVCacheCompressorMLX:
    """Compress and decompress transformer KV cache tensors using MLX.

    MLX port of KVCacheCompressor. Inputs/outputs are mx.arrays.

    - K cache: TurboQuantMLX (inner product preservation for Q @ K^T)
    - V cache: TurboQuantMSEMLX (MSE preservation for attn_weights @ V)

    Usage:
        compressor = KVCacheCompressorMLX(head_dim=128, k_bits=3, v_bits=3)
        compressed = compressor.compress(k_cache, v_cache)
        k_hat, v_hat = compressor.decompress(compressed)
    """

    def __init__(
        self,
        head_dim: int,
        k_bits: int = 3,
        v_bits: int = 3,
        seed: int = 42,
        norm_correction: bool = True,
    ):
        self.head_dim = head_dim
        self.k_bits = k_bits
        self.v_bits = v_bits

        self.k_quantizer = TurboQuantMLX(
            head_dim, bit_width=k_bits, seed=seed, norm_correction=norm_correction,
        )
        self.v_quantizer = TurboQuantMSEMLX(
            head_dim, bit_width=v_bits, seed=seed + 500, norm_correction=norm_correction,
        )

    def compress(self, k_cache: mx.array, v_cache: mx.array) -> CompressedKVCacheMLX:
        """Compress full KV cache tensors.

        Args:
            k_cache: mx.array shape (num_layers, num_heads, seq_len, head_dim).
            v_cache: mx.array same shape.

        Returns:
            CompressedKVCacheMLX.
        """
        num_layers, num_heads, seq_len, head_dim = k_cache.shape
        assert head_dim == self.head_dim
        assert v_cache.shape == k_cache.shape

        result = CompressedKVCacheMLX(
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            k_bit_width=self.k_bits,
            v_bit_width=self.v_bits,
        )

        for layer in range(num_layers):
            k_layer = []
            v_layer_idx = []
            v_layer_norms = []
            for head in range(num_heads):
                # K: batch-quantize all seq positions for this layer/head
                k_vecs = k_cache[layer, head]      # (seq_len, head_dim)
                k_compressed = self.k_quantizer.quantize(k_vecs)
                k_layer.append(k_compressed)

                # V: MSE quantize
                v_vecs = v_cache[layer, head]      # (seq_len, head_dim)
                v_indices, v_norms = self.v_quantizer.quantize(v_vecs)
                v_layer_idx.append(v_indices)
                v_layer_norms.append(v_norms)

            result.k_compressed.append(k_layer)
            result.v_indices.append(v_layer_idx)
            result.v_norms.append(v_layer_norms)

        return result

    def decompress(self, compressed: CompressedKVCacheMLX) -> tuple:
        """Decompress back to full KV cache tensors.

        Returns:
            (k_cache, v_cache) both mx.array shape (num_layers, num_heads, seq_len, head_dim).
        """
        shape = (compressed.num_layers, compressed.num_heads,
                 compressed.seq_len, compressed.head_dim)

        k_layers = []
        v_layers = []

        for layer in range(compressed.num_layers):
            k_heads = []
            v_heads = []
            for head in range(compressed.num_heads):
                k_vec = self.k_quantizer.dequantize(compressed.k_compressed[layer][head])
                v_vec = self.v_quantizer.dequantize(
                    compressed.v_indices[layer][head],
                    compressed.v_norms[layer][head],
                )
                k_heads.append(k_vec)
                v_heads.append(v_vec)

            k_layers.append(mx.stack(k_heads, axis=0))   # (num_heads, seq_len, head_dim)
            v_layers.append(mx.stack(v_heads, axis=0))

        k_cache = mx.stack(k_layers, axis=0)             # (num_layers, num_heads, seq_len, head_dim)
        v_cache = mx.stack(v_layers, axis=0)

        mx.eval(k_cache, v_cache)
        return k_cache, v_cache

    def compress_single(self, keys: mx.array, values: mx.array, layer_idx: int) -> CompressedVectorMLX:
        """Compress a single (keys, values) pair for streaming decode.

        Args:
            keys: mx.array shape (num_heads, head_dim) or (head_dim,).
            values: mx.array same shape.
            layer_idx: Unused — kept for API symmetry with mlx_lm integration.

        Returns:
            CompressedVectorMLX for keys (V stored separately as tuple).
        """
        k_compressed = self.k_quantizer.quantize(keys)
        v_indices, v_norms = self.v_quantizer.quantize(values)
        # Return a simple container; caller handles V storage
        return k_compressed, v_indices, v_norms

    def decompress_single(self, compressed_tuple: tuple, layer_idx: int) -> tuple:
        """Decompress a single (keys, values) pair."""
        k_compressed, v_indices, v_norms = compressed_tuple
        k_hat = self.k_quantizer.dequantize(k_compressed)
        v_hat = self.v_quantizer.dequantize(v_indices, v_norms)
        mx.eval(k_hat, v_hat)
        return k_hat, v_hat

    def memory_stats(self, seq_len: int, num_layers: int, num_heads: int) -> dict:
        """Compute memory usage statistics. Mirrors KVCacheCompressor.memory_stats."""
        n_vectors = num_layers * num_heads * seq_len
        original_bytes = n_vectors * self.head_dim * 2  # fp16

        k_bits_total = n_vectors * (self.head_dim * self.k_bits + 64)
        v_bits_total = n_vectors * (self.head_dim * self.v_bits + 32)

        compressed_bytes = (k_bits_total + v_bits_total) / 8

        return {
            "original_mb": original_bytes / 1024 / 1024,
            "compressed_mb": compressed_bytes / 1024 / 1024,
            "compression_ratio": original_bytes / compressed_bytes,
            "k_bits_per_value": self.k_bits,
            "v_bits_per_value": self.v_bits,
        }
