"""Backend registry with auto-detection and factory functions."""

from turboquant.backend._protocol import Backend

_default_backend: "Backend | None" = None


def get_backend(name: "str | None" = None) -> Backend:
    """Get a backend by name, or auto-detect if name is None.

    Auto-detection: MLX if available on Apple Silicon, else NumPy.
    """
    if name is None:
        try:
            import mlx.core  # noqa: F401
            name = "mlx"
        except ImportError:
            name = "numpy"

    if name == "mlx":
        from turboquant.backend.mlx_backend import MlxBackend
        return MlxBackend()

    from turboquant.backend.numpy_backend import NumpyBackend
    return NumpyBackend()


def set_default_backend(backend: "Backend | str") -> None:
    """Set the global default backend."""
    global _default_backend
    if isinstance(backend, str):
        backend = get_backend(backend)
    _default_backend = backend


def default_backend() -> Backend:
    """Return the global default backend, auto-detecting if not set."""
    global _default_backend
    if _default_backend is None:
        _default_backend = get_backend()
    return _default_backend
