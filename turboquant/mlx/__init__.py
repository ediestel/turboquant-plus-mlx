"""MLX-native TurboQuant+ implementations for Apple Silicon."""

from turboquant.mlx.rotation import (
    random_rotation_dense_mlx,
    fast_walsh_hadamard_mlx,
    apply_fast_rotation_mlx,
    apply_fast_rotation_transpose_mlx,
    random_rotation_fast_mlx,
)
from turboquant.mlx.codebook import optimal_centroids_mlx, nearest_centroid_indices_mlx
from turboquant.mlx.polar_quant import PolarQuantMLX
from turboquant.mlx.qjl import QJLMLX
from turboquant.mlx.turboquant import TurboQuantMLX, TurboQuantMSEMLX, CompressedVectorMLX
from turboquant.mlx.kv_cache import KVCacheCompressorMLX, CompressedKVCacheMLX
from turboquant.mlx.outlier import OutlierTurboQuantMLX
from turboquant.mlx.utils import pack_bits_mlx, unpack_bits_mlx, pack_indices_mlx

__all__ = [
    "random_rotation_dense_mlx",
    "fast_walsh_hadamard_mlx",
    "apply_fast_rotation_mlx",
    "apply_fast_rotation_transpose_mlx",
    "random_rotation_fast_mlx",
    "optimal_centroids_mlx",
    "nearest_centroid_indices_mlx",
    "PolarQuantMLX",
    "QJLMLX",
    "TurboQuantMLX",
    "TurboQuantMSEMLX",
    "CompressedVectorMLX",
    "KVCacheCompressorMLX",
    "CompressedKVCacheMLX",
    "OutlierTurboQuantMLX",
    "pack_bits_mlx",
    "unpack_bits_mlx",
    "pack_indices_mlx",
]
