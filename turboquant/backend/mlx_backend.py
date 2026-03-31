"""MLX backend — maps Backend protocol to mlx.core operations."""

import numpy as np

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


class MlxBackend:
    """MLX implementation of the Backend protocol for Apple Silicon."""

    name = "mlx"

    def __init__(self):
        if not _MLX_AVAILABLE:
            raise ImportError("mlx is not installed. Install with: pip install mlx")

    def array(self, data, dtype=None) -> "mx.array":
        mlx_dtype = self._map_dtype(dtype)
        if isinstance(data, np.ndarray):
            arr = mx.array(data)
        else:
            arr = mx.array(data)
        if mlx_dtype is not None:
            arr = arr.astype(mlx_dtype)
        return arr

    def zeros(self, shape, dtype=None) -> "mx.array":
        mlx_dtype = self._map_dtype(dtype) or mx.float32
        return mx.zeros(shape, dtype=mlx_dtype)

    def ones(self, shape, dtype=None) -> "mx.array":
        mlx_dtype = self._map_dtype(dtype) or mx.float32
        return mx.ones(shape, dtype=mlx_dtype)

    def randn(self, shape, seed: int = 0) -> "mx.array":
        key = mx.random.key(seed)
        return mx.random.normal(shape=shape, key=key)

    def sign(self, x) -> "mx.array":
        return mx.sign(x)

    def abs(self, x) -> "mx.array":
        return mx.abs(x)

    def norm(self, x, axis=None) -> "mx.array":
        return mx.linalg.norm(x, axis=axis)

    def searchsorted(self, a, v) -> "mx.array":
        # MLX doesn't have searchsorted natively; use numpy interop
        a_np = np.array(a)
        v_np = np.array(v)
        result = np.searchsorted(a_np, v_np.ravel()).reshape(v_np.shape)
        return mx.array(result)

    def qr(self, x) -> tuple:
        return mx.linalg.qr(x)

    def matmul(self, a, b) -> "mx.array":
        return a @ b

    def where(self, condition, x, y) -> "mx.array":
        return mx.where(condition, x, y)

    def to_numpy(self, x) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        mx.eval(x)
        return np.array(x)

    def from_numpy(self, x: np.ndarray) -> "mx.array":
        return mx.array(x)

    def eval(self, *args) -> None:
        mx.eval(*args)

    def fast_hadamard(self, x) -> "mx.array":
        """Butterfly Walsh-Hadamard transform in MLX. x shape: (..., n)."""
        import math
        n = x.shape[-1]
        if n < 1 or (n & (n - 1)) != 0:
            raise ValueError(f"Last dimension must be a power of 2, got {n}")
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

    def _map_dtype(self, dtype):
        if dtype is None:
            return None
        if isinstance(dtype, type) and issubclass(dtype, np.floating):
            return mx.float32
        if dtype == np.float32 or dtype == "float32":
            return mx.float32
        if dtype == np.float16 or dtype == "float16":
            return mx.float16
        if dtype == np.int8 or dtype == "int8":
            return mx.int8
        if dtype == np.uint8 or dtype == "uint8":
            return mx.uint8
        return None
