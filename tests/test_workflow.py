import shutil
from dcos_toolkit.api import init_session, load_input_data_and_parse, compute_mre, compute_2dcos, package_results

def test_full_pipeline(tri_file, tmp_path):
    # Test full pipeline
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    shutil.copy(tri_file, input_dir / "Tri_2.csv")
    
    session = init_session(job_name="test_run", output_dir=str(tmp_path / "out"))
    session.input_dir = input_dir
    
    load_input_data_and_parse(session, paths=[input_dir])
    compute_mre(session, 10, 1.0, 0.1, 1000)
    compute_2dcos(session, use_mre_for_2dcos="MRE [θ]", reference_type="mean")
    
    zip_path = package_results(session, include_mre_plot=False, include_2dcos_plot=False)
    assert zip_path.exists()