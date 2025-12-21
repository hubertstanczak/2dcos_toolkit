from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class CDDataset:
    """Container for a single circular dichroism (CD) dataset.

    This dataclass stores the raw input data and optional computed results used by the
    subsequent analysis steps (MRE conversion and 2D correlation spectroscopy).

    Parameters
    ----------
    name:
        Dataset identifier (typically derived from the input filename).
    lambda_axis:
        1D array of wavelengths in nanometers (nm), length = n_lambda.
    perturbation_axis:
        1D array of perturbation values (e.g., temperature), length = n_spectra.
    cd_mdeg:
        2D array of CD values in millidegrees (mdeg), shape = (n_spectra, n_lambda).

    Attributes
    ----------
    mre:
        Optional 2D array of molar residue ellipticity values derived from ``cd_mdeg``.
        Expected shape = (n_spectra, n_lambda).
    sync:
        Optional synchronous 2D-COS map computed from the dataset.
        Expected shape = (n_lambda, n_lambda).
    async_:
        Optional asynchronous 2D-COS map computed from the dataset.
        Expected shape = (n_lambda, n_lambda).
"""
    name: str
    lambda_axis: np.ndarray
    perturbation_axis: np.ndarray
    cd_mdeg: np.ndarray

    # computed results (filled later by the pipeline)
    mre: np.ndarray | None = None
    sync: np.ndarray | None = None
    async_: np.ndarray | None = None


@dataclass
class SessionState:
    """State container for a single analysis run.

    The session aggregates input paths, discovered files, and parsed datasets.
    It is populated and mutated by the pipeline functions (loading/parsing, MRE, 2D-COS,
    visualization, export).

    Parameters
    ----------
    job_name:
        Short run identifier used for naming outputs (e.g., folders, filenames, figure titles).
    input_dir:
        Directory used as the default input root (e.g., where raw data files are placed).
    output_dir:
        Directory where generated results (figures, tables, archives) are stored.

    Attributes
    ----------
    cd_files:
        List of discovered input file paths (as strings) accepted for parsing.
    datasets:
        List of parsed :class:`CDDataset` objects created from ``cd_files``.
    """
    job_name: str = "run"
    input_dir: Path = Path("data")
    output_dir: Path = Path("out")
    cd_files: list[str] = field(default_factory=list)
    datasets: list[CDDataset] = field(default_factory=list)
