from pathlib import Path

import numpy as np


def ensure_dir(path: Path | str | None) -> None:
    """Ensure that a directory exists.

    Parameters
    ----------
    path:
        Directory path to create. If None or empty, the function is a no-op.

    Notes
    -----
    The operation is idempotent: if the directory already exists, nothing happens.
    """
    if not path:
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def sort_lambda_and_matrix(
    lambda_axis: np.ndarray,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort wavelength axis (lambda) in ascending order and reorder associated data.

    Parameters
    ----------
    lambda_axis:
        1D array of wavelength values (nm).
    matrix:
        Array whose wavelength-dependent dimension matches ``lambda_axis``.
        Supported shapes:
        - (n_spectra, n_lambda): reordered along the last axis
        - (n_lambda, n_lambda): reordered along both axes (rows and columns)
        - (..., n_lambda): reordered along the last axis

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sorted (lambda_axis, matrix).

    Raises
    ------
    ValueError
        If ``lambda_axis`` is not 1D or if ``matrix`` is incompatible with it.
    """
    x = np.asarray(lambda_axis, dtype=float)
    z = np.asarray(matrix)

    if x.ndim != 1:
        raise ValueError(f"lambda_axis must be 1D, got ndim={x.ndim}")

    n = x.size
    order = np.argsort(x)
    x_sorted = x[order]

    if z.ndim == 1:
        if z.size != n:
            raise ValueError(
                "Incompatible shapes: matrix is 1D but its length does not match lambda_axis.\n"
                f"matrix.size={z.size}, len(lambda_axis)={n}"
            )
        return x_sorted, z[order]

    if z.ndim >= 2 and z.shape[-1] == n:
        # Generic case: wavelength axis corresponds to the last dimension.
        z_sorted = np.take(z, order, axis=-1)

        # Special case: square (n_lambda, n_lambda) map -> reorder both axes.
        if z_sorted.ndim == 2 and z_sorted.shape[0] == n and z_sorted.shape[1] == n:
            z_sorted = np.take(z_sorted, order, axis=0)

        return x_sorted, z_sorted

    raise ValueError(
        "Incompatible shapes: the last dimension of matrix must match lambda_axis.\n"
        f"matrix.shape={z.shape}, len(lambda_axis)={n}"
    )

def make_base_name(path: Path | str, *, max_len: int = 80) -> str:
    """Create a filesystem-safe base name from a path.

    Parameters
    ----------
    path:
        Input path (file or arbitrary string).
    max_len:
        Maximum length of the returned name (after sanitization).

    Returns
    -------
    str
        Sanitized name suitable for filenames/folders.

    Notes
    -----
    - Uses the input filename stem (without extension) when a real path is provided.
    - Replaces whitespace with underscores and removes characters problematic on Windows.
    - Never returns an empty string; falls back to "dataset".
    """
    p = Path(path)
    name = p.stem if p.name else str(path)

    # normalize whitespace
    name = "_".join(name.strip().split())

    # replace path separators
    name = name.replace("/", "_").replace("\\", "_")

    # remove characters illegal on Windows filesystems
    for ch in '<>:"|?*':
        name = name.replace(ch, "_")

    # collapse repeated underscores
    while "__" in name:
        name = name.replace("__", "_")

    name = name.strip("._-")  # avoid ugly edge characters
    if not name:
        name = "dataset"

    if max_len > 0:
        name = name[:max_len]

    return name
