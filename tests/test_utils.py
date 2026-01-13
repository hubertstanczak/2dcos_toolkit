import pytest
import numpy as np
from dcos_toolkit.utils import make_base_name, sort_lambda_and_matrix

def test_make_base_name():
    # Verify filename sanitization
    assert make_base_name("Sample File / Name*") == "Name"
    assert make_base_name("Data  (v1)...csv") == "Data_(v1)"
    assert make_base_name("") == "dataset"

def test_sort_lambda_and_matrix():
    # Verify sorting of lambda and corresponding matrix columns
    lam = np.array([500, 300, 400])

    # Matrix where each column corresponds to the lambda above
    matrix = np.array([
        [10, 30, 20], # Spectrum 1
        [15, 35, 25]  # Spectrum 2
    ])
    
    sorted_lam, sorted_mat = sort_lambda_and_matrix(lam, matrix)
    
    # Check if lambda is now [300, 400, 500]
    np.testing.assert_array_equal(sorted_lam, [300, 400, 500])
    
    # Check if matrix columns followed the lambda sorting
    expected_mat = np.array([
        [30, 20, 10],
        [35, 25, 15]
    ])
    np.testing.assert_array_equal(sorted_mat, expected_mat)

def test_sort_lambda_mismatch_error():
    # Verify that a ValueError is raised when dimensions do not match
    lam = np.array([1, 2, 3])
    matrix = np.array([[10, 20]]) 
    
    with pytest.raises(ValueError):
        sort_lambda_and_matrix(lam, matrix)