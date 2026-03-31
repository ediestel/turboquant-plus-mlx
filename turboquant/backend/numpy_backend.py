"""NumPy/SciPy backend — wraps existing TurboQuant operations into the Backend protocol."""

import numpy as np


class NumpyBackend:
    """Thin adapter over NumPy/SciPy satisfying the Backend protocol."""

    name = "numpy"

    def array(self, data, dtype=None) -> np.ndarray:
        return np.array(data, dtype=dtype)

    def zeros(self, shape, dtype=None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    def randn(self, shape, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape).astype(np.float32)

    def sign(self, x) -> np.ndarray:
        return np.sign(x)

    def abs(self, x) -> np.ndarray:
        return np.abs(x)

    def norm(self, x, axis=None) -> np.ndarray:
        return np.linalg.norm(x, axis=axis)

    def searchsorted(self, a, v) -> np.ndarray:
        return np.searchsorted(a, v)

    def qr(self, x) -> tuple:
        return np.linalg.qr(x)

    def matmul(self, a, b) -> np.ndarray:
        return a @ b

    def where(self, condition, x, y) -> np.ndarray:
        return np.where(condition, x, y)

    def to_numpy(self, x) -> np.ndarray:
        return np.asarray(x)

    def from_numpy(self, x) -> np.ndarray:
        return np.asarray(x)

    def eval(self, *args) -> None:
        pass  # NumPy is eager — no-op

    def fast_hadamard(self, x) -> np.ndarray:
        """Vectorized Walsh-Hadamard transform. x shape: (..., n) where n is power of 2."""
        from turboquant.rotation import fast_walsh_hadamard_transform
        if x.ndim == 1:
            return fast_walsh_hadamard_transform(x)
        # Batch: apply row-wise using the vectorized butterfly
        n = x.shape[-1]
        out = x.copy().astype(np.float64 if x.dtype.kind != 'f' else x.dtype)
        h = 1
        while h < n:
            reshaped = out.reshape(*out.shape[:-1], n // (h * 2), 2, h)
            a = reshaped[..., 0, :].copy()
            b = reshaped[..., 1, :].copy()
            reshaped[..., 0, :] = a + b
            reshaped[..., 1, :] = a - b
            out = reshaped.reshape(*out.shape[:-1], n)
            h *= 2
        return out / np.sqrt(n)
