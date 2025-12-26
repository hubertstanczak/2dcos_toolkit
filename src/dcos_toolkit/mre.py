import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .models import SessionState
from .utils import ensure_dir

import logging
logger = logging.getLogger(__name__)


def _mre_factor(
    *,
    residues_number: float,
    concentration_mg_ml: float,
    path_length_mm: float,
    molar_mass_g_mol: float,
) -> float:
    """Compute the scaling factor converting ellipticity θ (deg) to MRE.

    Parameters
    ----------
    residues_number:
        Number of residues (n).
    concentration_mg_ml:
        Concentration in mg/mL (c).
    path_length_mm:
        Path length in mm (l).
    molar_mass_g_mol:
        Molar mass in g/mol (M).

    Returns
    -------
    float
        Multiplicative factor such that: MRE = factor * θ(deg)

    Notes
    -----
    The conversion used here matches the conventional form:
    factor = M / (10 * c * l(cm) * n)
    where l(cm) is path length converted from mm to cm.
    """
    path_length_cm = float(path_length_mm) / 10.0  # mm -> cm
    return float(molar_mass_g_mol) / (
        10.0 * float(concentration_mg_ml) * path_length_cm * float(residues_number)
    )


def _save_mre_table(
    session: SessionState,
    dataset_name: str,
    mre: np.ndarray,
    lambda_axis: np.ndarray,
    perturbation_axis: np.ndarray,
) -> Path:
    """Save the MRE matrix as a CSV table.

    The table is saved to ``session.output_dir`` as ``<dataset>_MRE.csv``.

    Rows correspond to perturbation values (typically temperature in °C),
    columns correspond to wavelengths (nm).

    Returns
    -------
    Path
        Path to the created CSV file.
    """
    df_mre = pd.DataFrame(
        np.asarray(mre, dtype=float),
        index=np.asarray(perturbation_axis, dtype=float),
        columns=np.asarray(lambda_axis, dtype=float),
    )
    df_mre.index.name = "Temperature_C"
    df_mre.columns.name = "Wavelength_nm"

    out_csv = Path(session.output_dir) / f"{dataset_name}_MRE.csv"
    df_mre.to_csv(out_csv)
    return out_csv


def _plot_spectra_by_temperature(
    *,
    lambda_axis: np.ndarray,
    perturbation_axis: np.ndarray,
    spectra_matrix: np.ndarray,
    ylabel: str,
    title: str,
    out_png: Path,
    show: bool = True,
) -> None:
    """Plot spectra colored by perturbation (typically temperature) and save to PNG.

    Parameters
    ----------
    lambda_axis:
        Wavelength axis (nm).
    perturbation_axis:
        Perturbation values (typically temperature in °C).
    spectra_matrix:
        Spectra matrix shaped (N_perturbation, N_lambda).
    ylabel:
        Y-axis label.
    title:
        Plot title.
    out_png:
        Output file path (PNG).
    show:
        Whether to display the plot (useful in notebooks). If False, the figure
        is saved and closed without displaying.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    lam = np.asarray(lambda_axis, dtype=float)
    pert = np.asarray(perturbation_axis, dtype=float)
    mat = np.asarray(spectra_matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    norm = mcolors.Normalize(vmin=float(pert.min()), vmax=float(pert.max()))
    cmap = mpl.colormaps["plasma"]

    for t, row in zip(pert, mat):
        ax.plot(lam, row, color=cmap(norm(float(t))), alpha=0.8)

    ax.margins(x=0)
    ax.yaxis.get_offset_text().set_visible(False)

    ax.set_xlabel("λ (nm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Temperature (°C)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


@dataclass(frozen=True)
class MREParams:
    """Validated parameters required for MRE computation."""
    residues_number: float
    concentration_mg_ml: float
    path_length_mm: float
    molar_mass_g_mol: float


def _coerce_and_validate_mre_params(
    *,
    residues_number: Any,
    concentration_mg_ml: Any,
    path_length_mm: Any,
    molar_mass_g_mol: Any,
) -> MREParams:
    """Coerce user inputs to floats and validate them.

    Raises
    ------
    ValueError
        If any parameter is non-numeric or not strictly positive.
    """

    def to_float(val: Any, name: str) -> float:
        try:
            return float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"MRE parameter '{name}' must be a number. Got: {val!r}") from exc

    n = to_float(residues_number, "residues_number")
    c = to_float(concentration_mg_ml, "concentration_mg_ml")
    l = to_float(path_length_mm, "path_length_mm")
    m = to_float(molar_mass_g_mol, "molar_mass_g_mol")

    if n <= 0:
        raise ValueError("MRE parameter 'residues_number' must be greater than 0.")
    if c <= 0:
        raise ValueError("MRE parameter 'concentration_mg_ml' must greater than 0.")
    if l <= 0:
        raise ValueError("MRE parameter 'path_length_mm' must greater than 0.")
    if m <= 0:
        raise ValueError("MRE parameter 'molar_mass_g_mol' must greater than 0.")

    return MREParams(
        residues_number=n,
        concentration_mg_ml=c,
        path_length_mm=l,
        molar_mass_g_mol=m,
    )


def compute_mre_tables(session: SessionState, *, params: MREParams) -> float:
    """Compute MRE for all datasets and save MRE tables to CSV.

    This function:
    - computes the MRE scaling factor,
    - converts CD from mdeg to deg and multiplies by the factor,
    - stores the result in ``ds.mre``,
    - writes ``<dataset>_MRE.csv`` into ``session.output_dir``.

    Returns
    -------
    float
        The MRE scaling factor used for conversion.

    Raises
    ------
    RuntimeError
        If there are no datasets in the session or if any dataset fails.
    """
    if not session.datasets:
        raise RuntimeError("No parsed CD datasets available. Load data first.")

    ensure_dir(session.output_dir)

    factor = _mre_factor(
        residues_number=params.residues_number,
        concentration_mg_ml=params.concentration_mg_ml,
        path_length_mm=params.path_length_mm,
        molar_mass_g_mol=params.molar_mass_g_mol,
    )
    logger.info(f"Using MRE factor: {factor:.6g}")

    ok: list[str] = []
    failures: list[tuple[str, str]] = []

    for ds in session.datasets:
        try:
            theta_deg = np.asarray(ds.cd_mdeg, dtype=float) / 1000.0  # mdeg -> deg
            ds.mre = factor * theta_deg

            _save_mre_table(
                session,
                dataset_name=ds.name,
                mre=ds.mre,
                lambda_axis=ds.lambda_axis,
                perturbation_axis=ds.perturbation_axis,
            )

            ok.append(ds.name)

        except Exception as exc:
            ds.mre = None
            msg = str(exc) or exc.__class__.__name__
            failures.append((ds.name, msg))

    if ok:
        logger.info(f"MRE successfully calculated for {len(ok)} file(s):")
        for name in ok:
            logger.info(f"- {name}")

    if failures:
        logger.warning(f"MRE couldn't be calculated for {len(failures)} file(s):")
        for name, _ in failures:
            logger.warning(f"- {name}")


def generate_mre_plots(
    session: SessionState,
    *,
    use_mre_for_plot: bool = False,
    show: bool = True,
) -> None:
    """Generate and save plots for each dataset.

    Parameters
    ----------
    use_mre_for_plot:
        If True, plots MRE spectra (requires ``ds.mre``). If False, plots raw CD (mdeg).
    show:
        Whether to display plots in addition to saving them.
        Set to False for batch runs.
    """
    if not session.datasets:
        raise RuntimeError("MRE plots: no datasets in session.")

    ensure_dir(session.output_dir)

    failures: list[tuple[str, str]] = []

    for ds in session.datasets:
        try:
            out_dir = Path(session.output_dir)

            if use_mre_for_plot:
                if ds.mre is None:
                    raise RuntimeError("MRE not computed for this dataset (run compute_mre_tables first).")

                matrix = ds.mre
                ylabel = "[θ] (deg·cm²·dmol⁻¹)"
                title = f"MRE spectra: {ds.name}"
                out_png = out_dir / f"{ds.name}_MRE_plot.png"
            else:
                matrix = ds.cd_mdeg
                ylabel = "CD (mdeg)"
                title = f"CD spectra: {ds.name}"
                out_png = out_dir / f"{ds.name}_CD_plot.png"

            _plot_spectra_by_temperature(
                lambda_axis=ds.lambda_axis,
                perturbation_axis=ds.perturbation_axis,
                spectra_matrix=matrix,
                ylabel=ylabel,
                title=title,
                out_png=out_png,
                show=show,
            )

        except Exception as exc:
            msg = str(exc) or exc.__class__.__name__
            logger.error(f"{ds.name}: plot failed -> {msg}")
            failures.append((ds.name, msg))

    if failures:
        details = "; ".join([f"{name} ({msg})" for name, msg in failures])
        raise RuntimeError(f"MRE plots: failed for: {details}")


def compute_mre(
    session: SessionState,
    *,
    residues_number: float,
    concentration_mg_ml: float,
    path_length_mm: float,
    molar_mass_g_mol: float,
    generate_plot: bool = True,
    use_mre_for_plot: bool = False,
    show: bool = True,
) -> float:
    """Notebook-friendly wrapper: validate inputs, compute MRE tables, optionally plot.

    Parameters
    ----------
    session:
        Active session containing parsed datasets.
    residues_number, concentration_mg_ml, path_length_mm, molar_mass_g_mol:
        Parameters required for MRE scaling factor.
    generate_plot:
        If True, also generate plots for each dataset.
    use_mre_for_plot:
        If True and ``generate_plot`` is enabled, plot MRE instead of raw CD.
    show:
        Whether to display plots (only relevant when ``generate_plot`` is True).

    Returns
    -------
    float
        The MRE scaling factor used for conversion.
    """
    params = _coerce_and_validate_mre_params(
        residues_number=residues_number,
        concentration_mg_ml=concentration_mg_ml,
        path_length_mm=path_length_mm,
        molar_mass_g_mol=molar_mass_g_mol,
    )

    factor = compute_mre_tables(session, params=params)

    if generate_plot:
        generate_mre_plots(session, use_mre_for_plot=use_mre_for_plot, show=show)

    return factor
