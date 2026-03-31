"""Parity tests: MLX backend produces numerically equivalent results to NumPy.

All existing tests continue to use the NumPy backend unchanged.
These tests verify MLX matches NumPy within floating-point tolerance.

Run only on Apple Silicon (requires mlx):
    pytest tests/test_mlx/ -v
"""

import math
import pytest
import numpy as np

mlx = pytest.importorskip("mlx.core", reason="MLX not available — skipping parity tests")
import mlx.core as mx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np_to_mx(x: np.ndarray) -> mx.array:
    return mx.array(x.astype(np.float32))


def _mx_to_np(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.array(x)


# ---------------------------------------------------------------------------
# rotation parity
# ---------------------------------------------------------------------------

class TestRotationParity:
    def test_dense_rotation_orthogonality(self):
        """MLX dense rotation is orthogonal: Q^T Q ≈ I."""
        from turboquant.mlx.rotation import random_rotation_dense_mlx
        Q = random_rotation_dense_mlx(64, seed=42)
        mx.eval(Q)
        Q_np = np.array(Q)
        eye = Q_np.T @ Q_np
        np.testing.assert_allclose(eye, np.eye(64), atol=1e-5)

    def test_dense_rotation_det_positive(self):
        """MLX dense rotation has det = +1."""
        from turboquant.mlx.rotation import random_rotation_dense_mlx
        Q = random_rotation_dense_mlx(32, seed=7)
        mx.eval(Q)
        Q_np = np.array(Q)
        sign, _ = np.linalg.slogdet(Q_np)
        assert sign > 0

    def test_hadamard_normalization(self):
        """MLX WHT output has correct norm: ||WHT(e_0)||_2 = 1."""
        from turboquant.mlx.rotation import fast_walsh_hadamard_mlx
        e0 = mx.zeros((16,))
        # MLX doesn't support item assignment; build via concatenate
        e0_np = np.zeros(16, dtype=np.float32)
        e0_np[0] = 1.0
        e0 = mx.array(e0_np)
        out = fast_walsh_hadamard_mlx(e0)
        mx.eval(out)
        out_np = np.array(out)
        np.testing.assert_allclose(np.linalg.norm(out_np), 1.0, atol=1e-6)

    def test_hadamard_matches_numpy(self):
        """MLX WHT matches NumPy WHT output within 1e-5."""
        from turboquant.mlx.rotation import fast_walsh_hadamard_mlx
        from turboquant.rotation import fast_walsh_hadamard_transform

        rng = np.random.default_rng(0)
        x_np = rng.standard_normal(128).astype(np.float32)

        np_out = fast_walsh_hadamard_transform(x_np.astype(np.float64)).astype(np.float32)
        mlx_out = _mx_to_np(fast_walsh_hadamard_mlx(mx.array(x_np)))

        np.testing.assert_allclose(mlx_out, np_out, atol=1e-5)

    def test_fast_rotation_roundtrip(self):
        """Apply fast rotation then transpose recovers original vector."""
        from turboquant.mlx.rotation import (
            random_rotation_fast_mlx,
            apply_fast_rotation_mlx,
            apply_fast_rotation_transpose_mlx,
        )
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal(64).astype(np.float32)
        x = mx.array(x_np)

        signs1, signs2, padded_d = random_rotation_fast_mlx(64, seed=3)
        rotated = apply_fast_rotation_mlx(x, signs1, signs2, padded_d)
        recovered = apply_fast_rotation_transpose_mlx(rotated, signs1, signs2, padded_d)

        np.testing.assert_allclose(_mx_to_np(recovered), x_np, atol=1e-5)


# ---------------------------------------------------------------------------
# codebook parity
# ---------------------------------------------------------------------------

class TestCodebookParity:
    @pytest.mark.parametrize("bits,d", [(1, 128), (2, 128), (3, 64), (4, 64)])
    def test_centroids_match_numpy(self, bits, d):
        """MLX centroids match NumPy centroids within 1e-5."""
        from turboquant.mlx.codebook import optimal_centroids_mlx
        from turboquant.codebook import optimal_centroids

        np_centroids = optimal_centroids(bits, d)
        mlx_centroids = _mx_to_np(optimal_centroids_mlx(bits, d))

        np.testing.assert_allclose(mlx_centroids, np_centroids, atol=1e-5,
                                   err_msg=f"Centroids mismatch for bits={bits}, d={d}")

    def test_nearest_centroid_matches_numpy(self):
        """MLX nearest centroid indices match NumPy searchsorted."""
        from turboquant.mlx.codebook import optimal_centroids_mlx, nearest_centroid_indices_mlx
        from turboquant.codebook import optimal_centroids, nearest_centroid_indices

        d, bits = 128, 3
        rng = np.random.default_rng(5)
        values_np = rng.standard_normal(d).astype(np.float32) / math.sqrt(d)

        np_centroids = optimal_centroids(bits, d)
        np_indices = nearest_centroid_indices(values_np, np_centroids)

        mlx_centroids = optimal_centroids_mlx(bits, d)
        mlx_indices_arr = nearest_centroid_indices_mlx(mx.array(values_np), mlx_centroids)
        mx.eval(mlx_indices_arr)
        mlx_indices = np.array(mlx_indices_arr)

        np.testing.assert_array_equal(mlx_indices, np_indices)


# ---------------------------------------------------------------------------
# PolarQuant parity
# ---------------------------------------------------------------------------

class TestPolarQuantParity:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_quantize_dequantize_mse(self, bits):
        """MLX PolarQuant MSE is within tolerance of NumPy PolarQuant MSE."""
        from turboquant.mlx.polar_quant import PolarQuantMLX
        from turboquant.polar_quant import PolarQuant

        d = 128
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((16, d)).astype(np.float32)

        # NumPy
        pq_np = PolarQuant(d, bit_width=bits, seed=42)
        idx_np, norms_np = pq_np.quantize(x_np)
        x_hat_np = pq_np.dequantize(idx_np, norms_np)
        mse_np = float(np.mean((x_np - x_hat_np) ** 2))

        # MLX
        pq_mlx = PolarQuantMLX(d, bit_width=bits, seed=42)
        x_mx = _np_to_mx(x_np)
        idx_mlx, norms_mlx = pq_mlx.quantize(x_mx)
        x_hat_mlx = _mx_to_np(pq_mlx.dequantize(idx_mlx, norms_mlx))
        mse_mlx = float(np.mean((x_np - x_hat_mlx) ** 2))

        # MSE should be within 5% relative difference
        assert abs(mse_mlx - mse_np) / (mse_np + 1e-10) < 0.05, (
            f"MSE mismatch at bits={bits}: numpy={mse_np:.6f}, mlx={mse_mlx:.6f}"
        )

    def test_indices_match_numpy(self):
        """MLX PolarQuant produces same indices as NumPy for same inputs."""
        from turboquant.mlx.polar_quant import PolarQuantMLX
        from turboquant.polar_quant import PolarQuant

        d, bits = 64, 3
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((8, d)).astype(np.float32)

        pq_np = PolarQuant(d, bit_width=bits, seed=42)
        idx_np, _ = pq_np.quantize(x_np)

        pq_mlx = PolarQuantMLX(d, bit_width=bits, seed=42)
        idx_mlx, _ = pq_mlx.quantize(_np_to_mx(x_np))
        mx.eval(idx_mlx)

        np.testing.assert_array_equal(np.array(idx_mlx), idx_np)


# ---------------------------------------------------------------------------
# QJL parity
# ---------------------------------------------------------------------------

class TestQJLParity:
    def test_signs_match_numpy(self):
        """MLX QJL produces identical signs and norms as NumPy for the same input.

        The QJL inner product preservation property is a statistical guarantee
        (E[<x̃,ỹ>] ≈ <x,y> when using the sign-based estimator directly).
        Parity is tested by verifying the compressed representation is identical
        bit-for-bit between backends, since both now use the same NumPy-seeded S.
        """
        from turboquant.mlx.qjl import QJLMLX
        from turboquant.qjl import QJL

        d = 128
        rng = np.random.default_rng(99)
        x_np = rng.standard_normal(d).astype(np.float32)

        qjl_np = QJL(d, seed=42)
        s_np, n_np = qjl_np.quantize(x_np)

        qjl_mlx = QJLMLX(d, seed=42)
        s_mlx, n_mlx = qjl_mlx.quantize(mx.array(x_np))
        mx.eval(s_mlx, n_mlx)

        # Signs must be identical (same S matrix, same input)
        np.testing.assert_array_equal(np.array(s_mlx).astype(np.int8), s_np)
        # Norms must match (||x||_2 is deterministic)
        np.testing.assert_allclose(float(n_mlx), float(n_np), rtol=1e-5)

    def test_qjl_matches_numpy_reconstruction(self):
        """MLX QJL dequantize approximation has similar norm to NumPy."""
        from turboquant.mlx.qjl import QJLMLX
        from turboquant.qjl import QJL

        d = 128
        rng = np.random.default_rng(7)
        r_np = rng.standard_normal(d).astype(np.float32)

        qjl_np = QJL(d, seed=42)
        s_np, n_np = qjl_np.quantize(r_np)
        r_hat_np = qjl_np.dequantize(s_np, n_np)

        qjl_mlx = QJLMLX(d, seed=42)
        s_mlx, n_mlx = qjl_mlx.quantize(mx.array(r_np))
        r_hat_mlx = _mx_to_np(qjl_mlx.dequantize(s_mlx, n_mlx))

        # Reconstruction norms should match within 1%
        norm_np = float(np.linalg.norm(r_hat_np))
        norm_mlx = float(np.linalg.norm(r_hat_mlx))
        np.testing.assert_allclose(norm_mlx, norm_np, rtol=0.01)


# ---------------------------------------------------------------------------
# TurboQuant parity
# ---------------------------------------------------------------------------

class TestTurboQuantParity:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_roundtrip_mse(self, bits):
        """MLX TurboQuant roundtrip MSE is within tolerance of NumPy."""
        from turboquant.mlx.turboquant import TurboQuantMLX
        from turboquant.turboquant import TurboQuant

        d = 128
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((8, d)).astype(np.float32)

        tq_np = TurboQuant(d, bit_width=bits, seed=42)
        c_np = tq_np.quantize(x_np)
        x_hat_np = tq_np.dequantize(c_np)
        mse_np = float(np.mean((x_np - x_hat_np) ** 2))

        tq_mlx = TurboQuantMLX(d, bit_width=bits, seed=42)
        c_mlx = tq_mlx.quantize(_np_to_mx(x_np))
        x_hat_mlx = _mx_to_np(tq_mlx.dequantize(c_mlx))
        mse_mlx = float(np.mean((x_np - x_hat_mlx) ** 2))

        assert abs(mse_mlx - mse_np) / (mse_np + 1e-10) < 0.05, (
            f"TurboQuant MSE mismatch at bits={bits}: numpy={mse_np:.6f}, mlx={mse_mlx:.6f}"
        )

    def test_compression_ratio(self):
        """MLX and NumPy TurboQuant report same compression ratio."""
        from turboquant.mlx.turboquant import TurboQuantMLX
        from turboquant.turboquant import TurboQuant

        tq_np = TurboQuant(128, bit_width=3, seed=42)
        tq_mlx = TurboQuantMLX(128, bit_width=3, seed=42)
        assert tq_np.compression_ratio() == tq_mlx.compression_ratio()


# ---------------------------------------------------------------------------
# KV cache parity
# ---------------------------------------------------------------------------

class TestKVCacheParity:
    def test_compress_decompress_mse(self):
        """MLX KV cache roundtrip MSE within 5% of NumPy."""
        from turboquant.mlx.kv_cache import KVCacheCompressorMLX
        from turboquant.kv_cache import KVCacheCompressor

        num_layers, num_heads, seq_len, head_dim = 2, 4, 32, 64
        rng = np.random.default_rng(0)
        k_np = rng.standard_normal((num_layers, num_heads, seq_len, head_dim)).astype(np.float32)
        v_np = rng.standard_normal((num_layers, num_heads, seq_len, head_dim)).astype(np.float32)

        # NumPy
        c_np = KVCacheCompressor(head_dim, k_bits=3, v_bits=3, seed=42)
        compressed_np = c_np.compress(k_np, v_np)
        k_hat_np, v_hat_np = c_np.decompress(compressed_np)
        mse_k_np = float(np.mean((k_np - k_hat_np) ** 2))
        mse_v_np = float(np.mean((v_np - v_hat_np) ** 2))

        # MLX
        c_mlx = KVCacheCompressorMLX(head_dim, k_bits=3, v_bits=3, seed=42)
        compressed_mlx = c_mlx.compress(mx.array(k_np), mx.array(v_np))
        k_hat_mlx, v_hat_mlx = c_mlx.decompress(compressed_mlx)
        mse_k_mlx = float(np.mean((k_np - _mx_to_np(k_hat_mlx)) ** 2))
        mse_v_mlx = float(np.mean((v_np - _mx_to_np(v_hat_mlx)) ** 2))

        assert abs(mse_k_mlx - mse_k_np) / (mse_k_np + 1e-10) < 0.05
        assert abs(mse_v_mlx - mse_v_np) / (mse_v_np + 1e-10) < 0.05

    def test_memory_stats_match(self):
        """MLX and NumPy KVCacheCompressor report identical memory stats."""
        from turboquant.mlx.kv_cache import KVCacheCompressorMLX
        from turboquant.kv_cache import KVCacheCompressor

        c_np = KVCacheCompressor(128, k_bits=3, v_bits=3)
        c_mlx = KVCacheCompressorMLX(128, k_bits=3, v_bits=3)

        stats_np = c_np.memory_stats(seq_len=512, num_layers=32, num_heads=32)
        stats_mlx = c_mlx.memory_stats(seq_len=512, num_layers=32, num_heads=32)

        for key in stats_np:
            np.testing.assert_allclose(stats_mlx[key], stats_np[key], rtol=1e-6,
                                       err_msg=f"memory_stats[{key!r}] mismatch")


# ---------------------------------------------------------------------------
# Outlier parity
# ---------------------------------------------------------------------------

class TestOutlierParity:
    @pytest.mark.parametrize("target_bits", [2.5, 3.5])
    def test_roundtrip_mse(self, target_bits):
        """MLX OutlierTurboQuant MSE within 5% of NumPy."""
        from turboquant.mlx.outlier import OutlierTurboQuantMLX
        from turboquant.outlier import OutlierTurboQuant

        d = 128
        rng = np.random.default_rng(11)
        x_np = rng.standard_normal((8, d)).astype(np.float32)

        oq_np = OutlierTurboQuant(d, target_bits=target_bits, seed=42)
        c_np = oq_np.quantize(x_np)
        x_hat_np = oq_np.dequantize(c_np)
        mse_np = float(np.mean((x_np - x_hat_np) ** 2))

        oq_mlx = OutlierTurboQuantMLX(d, target_bits=target_bits, seed=42)
        c_mlx = oq_mlx.quantize(_np_to_mx(x_np))
        x_hat_mlx = _mx_to_np(oq_mlx.dequantize(c_mlx))
        mse_mlx = float(np.mean((x_np - x_hat_mlx) ** 2))

        assert abs(mse_mlx - mse_np) / (mse_np + 1e-10) < 0.05, (
            f"Outlier MSE mismatch at {target_bits}b: numpy={mse_np:.6f}, mlx={mse_mlx:.6f}"
        )

    def test_compression_ratio_match(self):
        """MLX and NumPy OutlierTurboQuant report same compression ratio."""
        from turboquant.mlx.outlier import OutlierTurboQuantMLX
        from turboquant.outlier import OutlierTurboQuant

        oq_np = OutlierTurboQuant(128, target_bits=2.5, seed=42)
        oq_mlx = OutlierTurboQuantMLX(128, target_bits=2.5, seed=42)
        np.testing.assert_allclose(oq_mlx.compression_ratio(), oq_np.compression_ratio(), rtol=1e-6)


# ---------------------------------------------------------------------------
# Utils parity
# ---------------------------------------------------------------------------

class TestUtilsParity:
    def test_pack_unpack_bits_roundtrip(self):
        """MLX pack_bits → unpack_bits recovers original signs."""
        from turboquant.mlx.utils import pack_bits_mlx, unpack_bits_mlx

        rng = np.random.default_rng(0)
        signs_np = rng.choice([-1, 1], size=128).astype(np.int8)
        signs_mx = mx.array(signs_np)

        packed = pack_bits_mlx(signs_mx)
        recovered = unpack_bits_mlx(packed, 128)
        mx.eval(recovered)

        np.testing.assert_array_equal(np.array(recovered), signs_np)

    def test_pack_bits_matches_numpy(self):
        """MLX pack_bits output matches NumPy pack_bits output."""
        from turboquant.mlx.utils import pack_bits_mlx
        from turboquant.utils import pack_bits

        rng = np.random.default_rng(3)
        signs_np = rng.choice([-1, 1], size=64).astype(np.int8)

        np_packed = pack_bits(signs_np)
        mlx_packed = pack_bits_mlx(mx.array(signs_np))

        np.testing.assert_array_equal(mlx_packed, np_packed)
