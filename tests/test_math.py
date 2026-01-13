import pytest
import numpy as np
from dcos_toolkit.mre import compute_mre
from dcos_toolkit.cos import _noda_matrix, compute_2dcos
from dcos_toolkit.peaks import find_diagonal_peaks
from dcos_toolkit.models import SessionState

def test_mre_calculation(sample_dataset, tmp_path):
    # Verify MRE calculation
    session = SessionState(output_dir=tmp_path)
    session.datasets = [sample_dataset]
    
    # Expected: 1000 / (10 * 2.0 * 0.1 * 10) = 50.0
    factor = compute_mre(session, 10, 2.0, 0.1, 1000.0)
    assert factor == 50.0
    assert sample_dataset.mre[0, 0] == 500.0

def test_2dcos(sample_dataset, tmp_path):
    # Verify 2DCOS computation
    session = SessionState(output_dir=tmp_path)
    sample_dataset.mre = sample_dataset.cd_mdeg.copy()
    session.datasets = [sample_dataset]
    compute_2dcos(session, use_mre_for_2dcos="MRE [θ]", reference_type="mean")
    assert sample_dataset.sync is not None
    assert sample_dataset.async_ is not None

def test_peak_detection():
    # Verify peak detection on a simple diagonal matrix
    matrix = np.zeros((4, 4))
    np.fill_diagonal(matrix, [1, 10, 1, 1])
    axis = np.array([100, 101, 102, 103])

    peaks = find_diagonal_peaks(matrix, axis=axis, n_peaks=1)
    assert peaks[0] == (101, 101, 10.0)  