import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

from .models import SessionState
from .utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MREParams:
    '''
    Container for MRE params.
    '''
    residues_number: float      
    concentration: float  
    path_length: float       
    molar_mass: float     


def _validate_mre_params(
    residues_number,
    concentration,
    path_length,
    molar_mass,
) -> MREParams:
 
    try:
        n = float(residues_number)
        c = float(concentration)
        l = float(path_length)
        m = float(molar_mass)

    except (TypeError, NameError):
        raise ValueError(f"MRE parameter must be a number.")

    for params, value in [("residues_number", n), ("concentration", c), ("path_length", l), ("molar_mass", m)]:
        if value <= 0:
            raise ValueError(f"MRE parameter {params} must be greater than 0.")

    return MREParams(n, c, l, m)


def _mre_factor(params: MREParams) -> float:
    return params.molar_mass / (
        10.0 
        * params.concentration 
        * params.path_length
        * params.residues_number
    )


def _save_mre_table(session, dataset_name, mre, lambda_axis, perturbation_axis):

    mre_arr = np.asarray(mre)
    df_mre = pd.DataFrame(
        mre_arr.T, 
        index=lambda_axis,          
        columns=perturbation_axis   
    )
    
    out_csv = Path(session.output_dir) / f"{dataset_name}_MRE.csv"
    df_mre.to_csv(out_csv)
    return out_csv


def _plot_spectra(
    lambda_axis,
    perturbation_axis,
    spectra_matrix,
    ylabel,
    dataset_name,
    out_png,
    show=True,
):
   
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(0, color='black', linestyle=':', linewidth=0.8, alpha=0.6, zorder=1)

    norm = mcolors.Normalize(vmin=perturbation_axis.min(), vmax=perturbation_axis.max())
    cmap = mpl.colormaps["plasma"]

    for t, row in zip(perturbation_axis, spectra_matrix):
        ax.plot(lambda_axis, row, color=cmap(norm(t)), alpha=0.8, zorder=10)

    ax.margins(x=0)

    ax.set_xlabel(r"$\lambda$ (nm)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Temperature (°C)", labelpad=15, fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    fig.text(
        0.5, 0.02, 
        f"Sample: {dataset_name}", 
        ha='center', 
        fontsize=9, 
        color='gray'
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    
    fig.savefig(out_png, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


def compute_mre_tables(session: SessionState, params: MREParams) -> float:
 
    if not session.datasets:
        raise RuntimeError("No parsed CD datasets available. Load data first.")

    ensure_dir(session.output_dir)

    factor = _mre_factor(params)
    logger.info(f"Using MRE factor: {factor:.6g}")

    ok = []
    failures = []

    for ds in session.datasets:
        try:
            data_matrix = np.asarray(ds.cd_mdeg, dtype=float)
            ds.mre = data_matrix * factor

            _save_mre_table(
                session,
                ds.name,
                ds.mre,
                ds.lambda_axis,
                ds.perturbation_axis,
            )
            ok.append(ds.name)

        except Exception as exc:
            ds.mre = None

            failures.append(ds.name)

    if ok:
        logger.info(f"MRE successfully calculated for {len(ok)} dataset(s):")
        for name in ok:
            logger.info(f"- {name}")

    if failures:
        logger.warning(f"MRE couldn't be calculated for {len(failures)} dataset(s):")
        for name, msg in failures:
            logger.warning(f"- {name} (Error: {msg})")
    
    return factor


def generate_mre_plots(
    session: SessionState,
    use_mre_for_plot=False,
    show=True,
):
   
    if not session.datasets:
        raise RuntimeError("MRE plots: no datasets in session.")

    ensure_dir(session.output_dir)
    out_dir = Path(session.output_dir)
    failures = []

    for ds in session.datasets:
        try:
            if use_mre_for_plot:
                matrix = ds.mre
                ylabel = r"$[\theta]$ (deg $\cdot$ cm$^2 \cdot$ dmol$^{-1}$)"
                out_png = out_dir / f"{ds.name}_MRE_plot.png"
            else:
                matrix = ds.cd_mdeg
                ylabel = r"CD (mdeg)"
                out_png = out_dir / f"{ds.name}_CD_plot.png"

            _plot_spectra(
                lambda_axis=ds.lambda_axis,
                perturbation_axis=ds.perturbation_axis,
                spectra_matrix=matrix,
                ylabel=ylabel,
                dataset_name=ds.name,
                out_png=out_png,
                show=show,
            )

        except Exception as exc:
            msg = str(exc) 
            failures.append((ds.name, msg))

    if failures:
        details = "; ".join([f"{name} ({msg})" for name, msg in failures])
        raise RuntimeError(f"Plotting failed for: {details}")


def compute_mre(
    session: SessionState,
    residues_number,
    concentration,
    path_length,
    molar_mass,
) -> float:
   
    params = _validate_mre_params(
        residues_number,
        concentration,
        path_length,
        molar_mass,
    )

    factor = compute_mre_tables(session, params=params)

    return factor