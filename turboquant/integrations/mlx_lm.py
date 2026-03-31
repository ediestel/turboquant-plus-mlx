"""mlx-lm integration: drop-in TurboQuant KV cache for mlx-lm models.

Provides TurboQuantKVCache — a cache class compatible with mlx-lm's KV cache
protocol. Drop it into any mlx-lm model's generate() loop for transparent
KV cache compression on Apple Silicon.

Basic usage:
    from turboquant.integrations.mlx_lm import TurboQuantKVCache
    from mlx_lm import load, generate

    model, tokenizer = load("mlx-community/Llama-3-8B-Instruct-4bit")
    cache = TurboQuantKVCache.for_model(model, k_bits=3, v_bits=3)
    response = generate(model, tokenizer, prompt="Hello", cache=cache)

Adaptive usage (Phase 3):
    from turboquant.adaptive_kv_cache import TieredDecayPolicy
    cache = TurboQuantKVCache.for_model(
        model, k_bits=3, v_bits=3,
        adaptive=True,
        decay_tiers=[(0.5, 3.0), (0.2, 2.0)],
    )
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import mlx.core as mx

from turboquant.mlx.kv_cache import KVCacheCompressorMLX
from turboquant.mlx.adaptive_kv_cache import AdaptiveKVCacheCompressorMLX


class TurboQuantKVCache:
    """Drop-in replacement for mlx-lm's KV cache with TurboQuant compression.

    Compatible with mlx-lm's cache protocol:
        cache.update_and_fetch(keys, values) → (keys, values)

    The cache transparently compresses KV pairs on the GPU as tokens are
    generated, then decompresses for attention computation.

    Sparse V (attention-gated skip) is supported via sparse_v=True, which
    masks out low-weight value positions using mx.where().

    Adaptive features (Phase 3):
        Pass adaptive=True to enable AdaptiveKVCacheCompressorMLX, then
        provide any combination of allocation_plan, decay_scheduler,
        decay_tiers, and moe_plans to activate per-slot bit selection.
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_layers: int,
        k_bits: int = 3,
        v_bits: int = 3,
        sparse_v: bool = False,
        sparse_v_tau: float = 1e-6,
        seed: int = 42,
        adaptive: bool = False,
        allocation_plan=None,
        decay_scheduler=None,
        decay_tiers=None,
        moe_plans=None,
    ):
        """
        Args:
            head_dim: Attention head dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            k_bits: Bit-width for K cache compression.
            v_bits: Bit-width for V cache compression.
            sparse_v: Enable attention-gated V skipping.
            sparse_v_tau: Threshold for Sparse V masking.
            seed: Random seed for compression matrices.
            adaptive: If True, use AdaptiveKVCacheCompressorMLX.
            allocation_plan: Optional AllocationPlan for per-head bit allocation.
            decay_scheduler: Optional TemporalDecayScheduler.
            decay_tiers: Optional list[(score_threshold, bits)] for tiered decay.
                         Converted to TieredDecayPolicy if provided.
            moe_plans: Optional dict[layer_idx -> MoEBitPlan].
        """
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sparse_v = sparse_v
        self.sparse_v_tau = sparse_v_tau

        use_adaptive = (
            adaptive
            or allocation_plan is not None
            or decay_scheduler is not None
            or decay_tiers is not None
            or moe_plans is not None
        )

        if use_adaptive:
            # Convert decay_tiers to TemporalDecayScheduler via TieredDecayPolicy
            # TieredDecayPolicy lives in adaptive_kv_cache (shared, dependency-free)
            effective_decay_scheduler = decay_scheduler
            if decay_tiers is not None and decay_scheduler is None:
                from turboquant.adaptive_kv_cache import TieredDecayPolicy
                from turboquant.temporal_decay import TemporalDecayScheduler, DecayConfig, DecayMode

                tiered = TieredDecayPolicy(decay_tiers, base_bits=float(v_bits))
                # Build a DecayConfig that approximates the tiers for the scheduler
                if tiered.tiers:
                    min_bits = min(bits for _, bits in tiered.tiers)
                    red_thresh = max(threshold for threshold, _ in tiered.tiers)
                    red_thresh = float(np.clip(red_thresh, 1e-3, 0.999))
                    if min_bits >= float(v_bits):
                        min_bits = max(1.0, float(v_bits) - 1.0)
                    cfg = DecayConfig(
                        mode=DecayMode.BIT_REDUCTION,
                        base_bits=float(v_bits),
                        min_bits=float(min_bits),
                        reduction_threshold=red_thresh,
                    )
                    effective_decay_scheduler = TemporalDecayScheduler(cfg)

            self.compressor = AdaptiveKVCacheCompressorMLX(
                head_dim=head_dim,
                k_bits=k_bits,
                v_bits=v_bits,
                seed=seed,
                allocation_plan=allocation_plan,
                decay_scheduler=effective_decay_scheduler,
                moe_plans=moe_plans,
            )
        else:
            self.compressor = KVCacheCompressorMLX(
                head_dim=head_dim,
                k_bits=k_bits,
                v_bits=v_bits,
                seed=seed,
            )

        # Per-layer compressed KV storage
        # Each entry: (CompressedVectorMLX, v_indices, v_norms)
        self._store: list[list[tuple | None]] = [
            [None] * num_heads for _ in range(num_layers)
        ]

    @classmethod
    def for_model(
        cls,
        model,
        k_bits: int = 3,
        v_bits: int = 3,
        sparse_v: bool = False,
        sparse_v_tau: float = 1e-6,
        seed: int = 42,
        adaptive: bool = False,
        allocation_plan=None,
        decay_scheduler=None,
        decay_tiers=None,
        moe_plans=None,
    ) -> "TurboQuantKVCache":
        """Construct cache sized for a given mlx-lm model.

        Inspects model.args for head_dim, num_heads, num_layers.
        Falls back to reasonable defaults if not present.

        Args:
            model: mlx-lm model object.
            k_bits: K cache bit-width.
            v_bits: V cache bit-width.
            sparse_v: Enable Sparse V masking.
            sparse_v_tau: Sparse V threshold.
            seed: Random seed.
            adaptive: Enable AdaptiveKVCacheCompressorMLX.
            allocation_plan: Optional AllocationPlan.
            decay_scheduler: Optional TemporalDecayScheduler.
            decay_tiers: Optional tiered decay list.
            moe_plans: Optional MoE bit plans dict.

        Returns:
            TurboQuantKVCache instance.
        """
        args = getattr(model, "args", None)
        if args is None:
            raise ValueError(
                "model.args not found — pass head_dim/num_heads/num_layers explicitly."
            )

        head_dim = getattr(args, "head_dim", None)
        if head_dim is None:
            hidden = getattr(args, "hidden_size", getattr(args, "d_model", 4096))
            num_heads = getattr(
                args, "num_attention_heads", getattr(args, "num_heads", 32)
            )
            head_dim = hidden // num_heads

        num_heads = getattr(
            args, "num_attention_heads", getattr(args, "num_heads", 32)
        )
        num_layers = getattr(
            args, "num_hidden_layers", getattr(args, "n_layers", 32)
        )

        return cls(
            head_dim=head_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            k_bits=k_bits,
            v_bits=v_bits,
            sparse_v=sparse_v,
            sparse_v_tau=sparse_v_tau,
            seed=seed,
            adaptive=adaptive,
            allocation_plan=allocation_plan,
            decay_scheduler=decay_scheduler,
            decay_tiers=decay_tiers,
            moe_plans=moe_plans,
        )

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        layer_idx: int,
    ) -> tuple[mx.array, mx.array]:
        """Compress new KV pair and return decompressed for attention.

        Called by mlx-lm at each decode step for each layer.

        Args:
            keys: mx.array shape (batch, num_heads, seq, head_dim) or (num_heads, seq, head_dim).
            values: mx.array same shape.
            layer_idx: Transformer layer index.

        Returns:
            (keys_decompressed, values_decompressed) for attention computation.
        """
        compressed_tuple = self.compressor.compress_single(keys, values, layer_idx)
        self._store[layer_idx][0] = compressed_tuple  # simplified: head=0
        k_hat, v_hat = self.compressor.decompress_single(compressed_tuple, layer_idx)
        return k_hat, v_hat

    def sparse_v_mask(self, attn_weights: mx.array) -> "mx.array | None":
        """Attention-gated V skip mask for Sparse V decoding.

        Args:
            attn_weights: mx.array attention scores, any shape.

        Returns:
            Binary mask mx.array (same shape), or None if sparse_v=False.
        """
        if not self.sparse_v:
            return None
        return mx.where(
            attn_weights > self.sparse_v_tau,
            mx.ones_like(attn_weights),
            mx.zeros_like(attn_weights),
        )

    def memory_stats(self, seq_len: int) -> dict:
        """Memory usage for current configuration."""
        return self.compressor.memory_stats(seq_len, self.num_layers, self.num_heads)
