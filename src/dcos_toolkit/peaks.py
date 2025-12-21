import numpy as np
from scipy.ndimage import maximum_filter


# Exclude a thin border to mitigate edge artifacts (e.g., wavelength range limits).
# This is intentionally conservative: one pixel on each side.
_DEFAULT_EDGE_EXCLUSION = 1


import logging
logger = logging.getLogger(__name__)

def _validate_grid(z: np.ndarray, x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize inputs for peak picking on a 2D grid.

    Parameters
    ----------
    z:
        2D matrix of values (e.g., synchronous/asynchronous map).
        Shape: (len(y_vals), len(x_vals)).
    x_vals:
        X-axis values (e.g., wavelengths for horizontal axis).
    y_vals:
        Y-axis values (e.g., wavelengths for vertical axis).

    Returns
    -------
    (z, x_vals, y_vals):
        Arrays converted to NumPy with float axes.

    Raises
    ------
    ValueError
        If shapes are inconsistent or if ``z`` is not 2D.
    """
    z_arr = np.asarray(z)
    if z_arr.ndim != 2:
        raise ValueError(f"z must be a 2D array, got ndim={z_arr.ndim}")

    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    if z_arr.shape[1] != x_arr.size:
        raise ValueError(f"Shape mismatch: z.shape[1]={z_arr.shape[1]} but len(x_vals)={x_arr.size}")
    if z_arr.shape[0] != y_arr.size:
        raise ValueError(f"Shape mismatch: z.shape[0]={z_arr.shape[0]} but len(y_vals)={y_arr.size}")

    return z_arr, x_arr, y_arr


def _is_far_enough(row: int, col: int, selected: list[tuple[int, int]], *, min_sep: int) -> bool:
    """Check Chebyshev distance separation on grid indices.

    A candidate (row, col) is accepted if for every already selected peak (r, c):
    max(|row-r|, |col-c|) >= min_sep.
    """
    for r, c in selected:
        if max(abs(row - r), abs(col - c)) < min_sep:
            return False
    return True


def _apply_edge_exclusion(mask: np.ndarray, edge: int) -> np.ndarray:
    """Zero-out a border region in a boolean mask."""
    if edge <= 0:
        return mask
    m = mask.copy()
    m[:edge, :] = False
    m[-edge:, :] = False
    m[:, :edge] = False
    m[:, -edge:] = False
    return m


def _find_peaks(
    z: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    n_peaks: int,
    *,
    min_sep: int,
    upper_triangle_only: bool = False,
    edge_exclusion: int = _DEFAULT_EDGE_EXCLUSION,
) -> list[tuple[float, float, float]]:
    """Find up to ``n_peaks`` strongest peaks (by |z|) using local maxima + greedy separation.

    Key detail: if ``edge_exclusion`` is enabled, the border is removed *before*
    local-maximum detection so that excluded edge pixels do not suppress maxima
    in the interior of large blobs.
    """
    if n_peaks <= 0:
        return []

    z_arr, x_arr, y_arr = _validate_grid(z, x_vals, y_vals)

    min_sep = max(1, int(min_sep))
    size = 2 * min_sep + 1

    z_abs = np.abs(z_arr).astype(float, copy=True)

    # 1) Exclude border BEFORE maximum_filter to avoid edge pixels dominating the neighborhood.
    if edge_exclusion > 0:
        e = int(edge_exclusion)
        z_abs[:e, :] = 0.0
        z_abs[-e:, :] = 0.0
        z_abs[:, :e] = 0.0
        z_abs[:, -e:] = 0.0

    # 2) If needed, restrict to upper triangle for candidate detection (optional).
    if upper_triangle_only:
        z_abs = np.triu(z_abs)

    local_max = (maximum_filter(z_abs, size=size, mode="nearest") == z_abs)
    mask = local_max & (z_abs > 0)

    rows, cols = np.where(mask)
    if rows.size == 0:
        return []

    abs_vals = z_abs[rows, cols]
    order = np.argsort(abs_vals)[::-1]

    axes_compatible_for_swap = (x_arr.size == y_arr.size) and np.allclose(x_arr, y_arr)

    selected_idx: list[tuple[int, int]] = []
    peaks: list[tuple[float, float, float]] = []

    for k in order:
        r0 = int(rows[k])
        c0 = int(cols[k])

        # Map to upper triangle for presentation without losing peaks found elsewhere.
        r, c = r0, c0
        if upper_triangle_only:
            if axes_compatible_for_swap and r > c:
                r, c = c, r
            elif r > c:
                continue

        if _is_far_enough(r, c, selected_idx, min_sep=min_sep):
            selected_idx.append((r, c))
            # IMPORTANT: value from original position (r0, c0), not from swapped (r, c)
            peaks.append((float(x_arr[c]), float(y_arr[r]), float(z_arr[r0, c0])))

            if len(peaks) >= n_peaks:
                break

    return peaks



def find_diagonal_peaks(
    z: np.ndarray,
    axis_vals: np.ndarray,
    n_peaks: int,
    *,
    min_sep: int = 2,
) -> list[tuple[float, float, float]]:
    """Return strongest peaks on the diagonal (x = y).

    Parameters
    ----------
    z:
        2D matrix.
    axis_vals:
        Axis values used for both x and y (e.g., wavelengths).
    n_peaks:
        Number of peaks to return (at most).
    min_sep:
        Minimum separation along diagonal indices (grid units).

    Returns
    -------
    list[tuple[float, float, float]]
        Peaks as (x_value, y_value, z_value) where x_value == y_value.
    """
    z_arr = np.asarray(z, dtype=float)
    axis = np.asarray(axis_vals, dtype=float)

    if n_peaks <= 0 or z_arr.size == 0:
        return []

    diag = np.diag(z_arr)
    if diag.size == 0:
        return []

    if axis.size < diag.size:
        raise ValueError(f"axis_vals is too short for diagonal: len(axis_vals)={axis.size}, diag size={diag.size}")

    order = np.argsort(np.abs(diag))[::-1]
    min_sep = max(1, int(min_sep))

    selected_i: list[int] = []
    peaks: list[tuple[float, float, float]] = []

    for i in order:
        ii = int(i)
        if all(abs(ii - s) >= min_sep for s in selected_i):
            selected_i.append(ii)
            peaks.append((float(axis[ii]), float(axis[ii]), float(diag[ii])))
            if len(peaks) >= n_peaks:
                break

    return peaks


def _cross_peak_matrix(z: np.ndarray, *, polarity: str) -> np.ndarray:
    """Prepare a matrix for cross-peak picking for a given polarity.

    - 'max': keeps positive values, sets others to 0
    - 'min': keeps negative values by storing their magnitudes (negated), sets others to 0

    The diagonal is zeroed out to avoid selecting diagonal peaks as cross peaks.
    """
    z_arr = np.asarray(z, dtype=float)

    if polarity == "max":
        z_pol = np.where(z_arr > 0.0, z_arr, 0.0)
    elif polarity == "min":
        z_pol = np.where(z_arr < 0.0, -z_arr, 0.0)  # negate minima to rank by magnitude
    else:
        raise ValueError("polarity must be 'max' or 'min'.")

    n_diag = min(z_pol.shape)
    idx = np.arange(n_diag)
    z_pol[idx, idx] = 0.0
    return z_pol


def _find_cross_peaks(
    z: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    n_peaks: int,
    *,
    polarity: str,
) -> list[tuple[float, float, float]]:
    """Find strongest cross peaks for the selected polarity.

    Notes
    -----
    - Uses a fixed minimum separation (min_sep=6) in grid index units.
    - Searches only in the upper triangle to avoid mirrored duplicates.
    - Excludes a 1-pixel border to mitigate edge artifacts at axis limits.
    """
    if n_peaks <= 0:
        return []

    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    z_pol = _cross_peak_matrix(z, polarity=polarity)

    return _find_peaks(
        z_pol,
        x_arr,
        y_arr,
        n_peaks=n_peaks,
        min_sep=6,
        upper_triangle_only=True,
        edge_exclusion=_DEFAULT_EDGE_EXCLUSION,
    )


def find_cross_peaks_max(
    z: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    n_peaks: int,
) -> list[tuple[float, float, float]]:
    """Return strongest positive cross peaks (local maxima)."""
    return _find_cross_peaks(z, x_vals, y_vals, n_peaks, polarity="max")


def find_cross_peaks_min(
    z: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    n_peaks: int,
) -> list[tuple[float, float, float]]:
    """Return strongest negative cross peaks (local minima)."""
    return _find_cross_peaks(z, x_vals, y_vals, n_peaks, polarity="min")
