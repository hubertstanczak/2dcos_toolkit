from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class CDDataset:
    """
    Container for a single Circular Dichroism (CD) dataset.
    Holds raw data (cd_mdeg) and computed results (mre, sync, async_).
    """
    name: str
    lambda_axis: np.ndarray        # 1D array (nm)
    perturbation_axis: np.ndarray  # 1D array (Temperature)
    cd_mdeg: np.ndarray            # 2D array (Spectra x Lambda)

    # Optional: Original file path for export logic
    source_path: str | None = None

    # Computed results 
    mre: np.ndarray | None = None    # Molar Ellipticity
    sync: np.ndarray | None = None   # Synchronous 2D Map
    async_: np.ndarray | None = None # Asynchronous 2D Map


@dataclass
class SessionState:
    """
    State container for the analysis session.
    Keeps track of paths, configuration, and loaded datasets.
    """
    job_name: str = "run"
    input_dir: Path = Path("data")
    output_dir: Path = Path("out")
    
    # List of raw file paths found
    cd_files: list[str] = field(default_factory=list)
    
    # List of successfully parsed dataset objects
    datasets: list[CDDataset] = field(default_factory=list)