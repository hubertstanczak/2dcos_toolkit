import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

def ensure_dir(path: Path | str | None) -> None:
    """Creates directory if it doesn't exist."""
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def sort_lambda_and_matrix(
    lambda_axis: np.ndarray,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligns spectral data to increasing wavelength order.

    """
    x = np.asarray(lambda_axis, dtype=float)
    z = np.asarray(matrix)

    if x.ndim != 1:
        raise ValueError(f"Lambda axis must be 1D, got {x.ndim}D")

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]

    if z.ndim == 1:
        if z.size != x.size:
            raise ValueError(f"Shape mismatch: matrix size {z.size} != lambda size {x.size}")
        return x_sorted, z[sort_idx]

    if z.shape[-1] != x.size:
         raise ValueError(f"Shape mismatch: matrix last dim {z.shape[-1]} != lambda size {x.size}")

    z_sorted = np.take(z, sort_idx, axis=-1)

    return x_sorted, z_sorted


def make_base_name(path: Path | str, *, max_len: int = 80) -> str:
    """
    Sanitizes a filename to be safe for file systems and ZIP archives.
    Replaces special characters and spaces with underscores.
    """
    p = Path(path)
    name = p.stem if p.name else str(path)
    name = "_".join(name.split())

    for ch in '/\\<>:"|?*':
        name = name.replace(ch, "_")

    while "__" in name:
        name = name.replace("__", "_")

    name = name.strip("._-")
    
    if not name:
        name = "dataset"

    return name[:max_len]