import pytest
from dcos_toolkit.io import parse_cd_file

def test_parse_dim_file(dim_file):
    # Verify parser
    ds = parse_cd_file(dim_file)
    assert ds.name == "Dim"
    assert len(ds.lambda_axis) > 0
    assert ds.cd_mdeg.shape[0] > 0  


def test_parse_tri_file(tri_file):
    # Verify parser
    assert tri_file.exists(), f"Plik {tri_file} nie istnieje w tests/data!"
    ds = parse_cd_file(tri_file)
    assert ds.name == "Tri_2"
    assert ds.cd_mdeg.shape[0] >= 3