import pytest
import numpy as np
from pathlib import Path

from dcos_toolkit.models import CDDataset

@pytest.fixture
def data_dir():
    return Path(__file__).parent / "data"

@pytest.fixture
def dim_file(data_dir):
    return data_dir / "Dim.csv"

@pytest.fixture
def tri_file(data_dir):
    return data_dir / "Tri_2.csv"

@pytest.fixture
def sample_dataset():
    import numpy as np
    return CDDataset(
        name="synthetic",
        lambda_axis=np.array([200, 201, 202]),
        perturbation_axis=np.array([20, 40, 60]),
        cd_mdeg=np.array([[10, 11, 12], [20, 22, 24], [30, 33, 36]])
    )