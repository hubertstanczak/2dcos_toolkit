from pathlib import Path
import numpy as np
import pandas as pd

from .models import SessionState
from .utils import ensure_dir

import logging
logger = logging.getLogger(__name__)


def _build_dynamic_spectra(spectra: np.ndarray, *, ref_mode: str) -> np.ndarray:
    ''' Builds dynamic spectra matrix by subtracting reference spectrum '''
    
    s = np.asarray(spectra, dtype=float)
    mode = (ref_mode or "mean").strip().lower()

    if mode == "first":
        ref = s[0:1, :]
    elif mode == "last":
        ref = s[-1:, :]
    elif mode == "none":
        return s
    else:
        ref = s.mean(axis=0, keepdims=True)

    return s - ref


def _noda_matrix(n: int) -> np.ndarray:
    N = np.zeros((n, n))

    for i in range(n):
        for k in range(n):
            if i == k:
                N[i, k] = 0
            else:
                N[i, k] = 1 / (np.pi * (k - i))

    return N



def compute_2dcos(session: SessionState, *, use_mre_for_2dcos: bool, reference_type: str) -> None:
    if not session.datasets:
        raise RuntimeError("No parsed CD datasets found. Please load data first.")

    if use_mre_for_2dcos and any(ds.mre is None for ds in session.datasets):
        raise RuntimeError(
            "2DCOS mode is set to use MRE, but MRE has not been calculated yet.\n"
            "Please run the MRE calculation step first, or switch 2DCOS mode to use raw data."
        )

    ref_mode = (reference_type or "mean").strip().lower()

    out_dir = Path(session.output_dir)
    ensure_dir(out_dir)

    ok = []
    skipped = []
    errors = []

    for ds in session.datasets:
        try:
            # Get spectra matrix
            spectra = ds.mre if use_mre_for_2dcos else ds.cd_mdeg

            if spectra.ndim != 2 or spectra.size == 0:
                raise ValueError("Empty or invalid spectra matrix.")

            n_spectra, n_lambda = spectra.shape

            lam = np.asarray(ds.lambda_axis, dtype=float).ravel()
            if lam.size != n_lambda:
                raise ValueError(
                    f"{ds.name}: lambda_axis length {lam.size} does not match spectra columns {n_lambda}"
                )

            if n_spectra < 3:
                ds.sync = None
                ds.async_ = None
                skipped.append(ds.name)
                logger.info(f"{ds.name} skipped (not enough spectra; need >= 3).")
                continue

            if np.isnan(spectra).any():
                logger.info(f"NaN values detected in {ds.name}; results may be unreliable")

            dyn = _build_dynamic_spectra(spectra, ref_mode=ref_mode)

            # synchronous
            sync = (dyn.T @ dyn) / float(n_spectra - 1)
            ds.sync = sync

            sync_path = out_dir / f"{ds.name}_sync.csv"
            pd.DataFrame(sync, index=lam, columns=lam).to_csv(sync_path)

            # asynchronous
            N = _noda_matrix(n_spectra)
            async_map = (dyn.T @ N @ dyn) / float(n_spectra - 1)
            ds.async_ = async_map

            async_path = out_dir / f"{ds.name}_async.csv"
            pd.DataFrame(async_map, index=lam, columns=lam).to_csv(async_path)

            ok.append(ds.name)

        except Exception:
            ds.sync = None
            ds.async_ = None
            errors.append(ds.name)

            
    if not ok:
        raise RuntimeError(f"No 2DCOS results generated.")

    if ok:
        logger.info(f"2DCOS successfully calculated for {len(ok)} dataset(s):")
        for name in ok:
            logger.info(f"- {name}")

    if skipped:
        logger.info(f"2DCOS skipped for {len(skipped)} dataset(s):")
        for name in skipped:
            logger.info(f"- {name}")

    if errors:
        logger.info(f"2DCOS failed for {len(errors)} dataset(s):")
        for name in errors:
            logger.info(f"- {name}")
