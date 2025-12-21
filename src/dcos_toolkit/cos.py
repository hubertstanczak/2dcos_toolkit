from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .models import SessionState
from .utils import ensure_dir

import logging
logger = logging.getLogger(__name__)


# Floating-point precision used in 2D-COS computations.
# Use float64 for maximum precision (slower, larger files).
FLOAT_DTYPE = np.float32

# Conversion used only when computing 2D-COS directly from CD values:
# cd_mdeg (millidegrees) -> degrees
MDEG_TO_DEG = FLOAT_DTYPE(1.0 / 1000.0)



def _build_dynamic_spectra(spectra: np.ndarray, *, ref_mode: str) -> np.ndarray:
    """Build dynamic spectra ΔS = S - S_ref.

    Parameters
    ----------
    spectra:
        2D array with shape (n_spectra, n_lambda).
    ref_mode:
        One of: 'mean', 'first', 'last', 'none'.

    Returns
    -------
    np.ndarray
        Dynamic spectra with the same shape as the input.
    """
    s = np.asarray(spectra, dtype=FLOAT_DTYPE)
    if s.ndim != 2 or s.size == 0:
        return s.copy()

    mode = (ref_mode or "mean").lower()
    if mode == "none":
        return s.copy()

    if mode == "first":
        ref = s[0:1, :]
    elif mode == "last":
        ref = s[-1:, :]
    else:
        ref = s.mean(axis=0, keepdims=True)

    return s - ref


def _reference_label(ref_mode: str) -> str:
    mode = (ref_mode or "mean").lower()
    if mode == "first":
        return "first spectrum"
    if mode == "last":
        return "last spectrum"
    if mode == "none":
        return "no reference (raw spectra)"
    return "mean spectrum"


@lru_cache(maxsize=32)
def _noda_matrix(n: int) -> np.ndarray:
    """Return Hilbert–Noda matrix N of shape (n, n), cached by n.

    Definition (for i != j):
        N_ij = 1 / (pi * (i - j))
    and N_ii = 0.
    """
    if n <= 0:
        return np.zeros((0, 0), dtype=FLOAT_DTYPE)

    i = np.arange(n, dtype=FLOAT_DTYPE)
    j = np.arange(n, dtype=FLOAT_DTYPE)
    diff = np.subtract.outer(i, j)

    with np.errstate(divide="ignore", invalid="ignore"):
        N = 1.0 / (np.pi * diff)
        N[~np.isfinite(N)] = 0.0

    return N.astype(FLOAT_DTYPE, copy=False)


def _save_square_csv(path: Path, axis: np.ndarray, matrix: np.ndarray) -> None:
    """Save a square (n_lambda x n_lambda) matrix to CSV with axis labels."""
    df = pd.DataFrame(np.asarray(matrix, dtype=float), index=axis, columns=axis)
    df.to_csv(path)


def compute_2dcos(
    session: SessionState,
    *,
    use_mre_for_2dcos: bool,
    reference_type: str,
) -> None:
    """Compute synchronous (Φ) and asynchronous (Ψ) 2D-COS maps for all datasets.

    For each dataset, this function computes:
      - synchronous map:  Φ = (ΔS)^T · (ΔS) / (N - 1)
      - asynchronous map: Ψ = (ΔS)^T · N · (ΔS) / (N - 1)
        where N is the Hilbert–Noda matrix of size (N, N) and N is the number of spectra.

    Parameters
    ----------
    session:
        Current session containing parsed datasets.
    use_mre_for_2dcos:
        If True, uses ds.mre (must be computed earlier) as input matrix.
        If False, uses ds.cd_mdeg converted to degrees.
    reference_type:
        Reference used to compute dynamic spectra ΔS. Allowed values:
        'mean', 'first', 'last', 'none'.

    Notes
    -----
    - The function writes '{dataset}_sync.csv' and '{dataset}_async.csv' to session.output_dir.
    - If any dataset fails, the function prints per-dataset errors and raises RuntimeError at the end.
    """
    if not session.datasets:
        raise RuntimeError("2D-COS: no parsed CD datasets found. Run input/parsing first.")

    ref_mode = (reference_type or "mean").strip().lower()
    allowed = {"mean", "first", "last", "none"}
    if ref_mode not in allowed:
        raise ValueError(f"2D-COS: invalid reference_type={reference_type!r}. Allowed: {sorted(allowed)}.")

    if use_mre_for_2dcos:
        missing = [ds.name for ds in session.datasets if ds.mre is None]
        if missing:
            missing_str = ", ".join(missing)
            raise RuntimeError(
                "2D-COS: use_mre_for_2dcos=True but MRE is missing for:\n"
                f"  {missing_str}\n"
                "Run compute_mre(...) first."
            )

    ensure_dir(session.output_dir)

    errors: list[tuple[str, str]] = []

    for ds in session.datasets:
        try:
            if use_mre_for_2dcos:
                spectra = np.asarray(ds.mre, dtype=FLOAT_DTYPE)  # guaranteed not None by pre-check
                data_label = "MRE"
            else:
                spectra = np.asarray(ds.cd_mdeg, dtype=FLOAT_DTYPE) * MDEG_TO_DEG
                data_label = "CD (deg)"

            if spectra.ndim != 2 or spectra.size == 0:
                raise ValueError("empty or invalid spectra matrix")

            n_spectra, n_lambda = spectra.shape

            logger.debug(
                f"2D-COS: dataset = {ds.name} "
                f"(N={n_spectra}, L={n_lambda}, data={data_label}, ref={_reference_label(ref_mode)})"
            )

            if n_spectra < 3:
                logger.warning(f" {ds} skipped: not enough spectra, need >= 3)")
                ds.sync = None
                ds.async_ = None
                continue

            if np.isnan(spectra).any():
                logger.info(f"NaN values detected in the {ds}; results may be unreliable")

            dyn = _build_dynamic_spectra(spectra, ref_mode=ref_mode)

            sync = dyn.T @ dyn
            sync /= FLOAT_DTYPE(n_spectra - 1)
            ds.sync = sync

            sync_path = session.output_dir / f"{ds.name}_sync.csv"
            _save_square_csv(sync_path, ds.lambda_axis, sync)
            logger.info(f" Saved: {sync_path}")

            N = _noda_matrix(n_spectra)
            async_map = dyn.T @ N @ dyn
            async_map /= FLOAT_DTYPE(n_spectra - 1)
            ds.async_ = async_map

            async_path = session.output_dir / f"{ds.name}_async.csv"
            _save_square_csv(async_path, ds.lambda_axis, async_map)
            logger.info(f" Saved: {async_path}")

        except Exception as exc:
            ds.sync = None
            ds.async_ = None
            msg = str(exc) or exc.__class__.__name__
            logger.info(f"2D-COS: failed for dataset '{ds.name}': {msg}")
            errors.append((ds.name, msg))

    if errors:
        names = ", ".join(name for name, _ in errors)
        raise RuntimeError(
            "2D-COS finished with errors for some datasets.\n"
            f"Failed: {names}"
        )
