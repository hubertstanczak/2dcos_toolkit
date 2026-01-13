import logging
import zipfile
from pathlib import Path

from .models import SessionState
from .utils import ensure_dir, make_base_name

logger = logging.getLogger(__name__)

def package_results(
    session: SessionState,
    *,
    include_input_file: bool = True,
    include_mre: bool = True,
    include_mre_plot: bool = True,
    include_2dcos: bool = True,
    include_2dcos_plot: bool = True,
) -> Path:
   
    ensure_dir(session.output_dir)
    out_dir = Path(session.output_dir).resolve()

    if not session.datasets:
        raise RuntimeError("No parsed CD datasets found. Please load data first.")

    datasets = sorted(session.datasets, key=lambda d: d.name)
    
    missing_msgs = []

    def any_file_exists(suffix):
        for ds in datasets:
            path = out_dir / f"{ds.name}{suffix}"
            if path.exists() and path.is_file():
                return True
        return False

    if include_input_file:
        has_any_input = False
        for ds in datasets:
            if ds.source_path and Path(ds.source_path).exists():
                has_any_input = True
                break
        if not has_any_input:
            raise RuntimeError(
                "Export aborted: no input CD files found.\n"
                "Upload files first.\n"
            )

    if include_mre and not any_file_exists("_MRE.csv"):
        missing_msgs.append("- export_mre_tables is checked, but no MRE table files were found.")

    if include_mre_plot:
        has_cd = any_file_exists("_CD_plot.png")
        has_mre = any_file_exists("_MRE_plot.png")
        if not (has_cd or has_mre):
             missing_msgs.append("- export_1d_plots is checked, but no MRE/CD plot files were found.")

    if include_2dcos:
        has_sync = any_file_exists("_sync.csv")
        has_async = any_file_exists("_async.csv")
        if not (has_sync or has_async):
             missing_msgs.append("- export_2dcos_matrices is checked, but no 2DCOS matrices files were found.")

    if include_2dcos_plot and not any_file_exists("_2DCOS_combined.png"):
        missing_msgs.append("- export_2dcos_maps is checked, but no 2DCOS plot files were found.")
    if missing_msgs:
        details = "\n".join(missing_msgs)
        raise RuntimeError(
            "Export cannot be created yet.\n"
            "Some selected result types are missing:\n"
            f"{details}\n"
            "\nRun the missing step(s) or disable the corresponding option(s) and try again."
        )

    files_to_pack = []

    for ds in datasets:
        zip_folder = make_base_name(ds.name)

        if include_input_file:
            if hasattr(ds, 'source_path') and ds.source_path:
                src_path = Path(ds.source_path).resolve()
                if src_path.exists():
                    arc_name = f"{zip_folder}/{src_path.name}"
                    files_to_pack.append((src_path, arc_name))
                else:
                    logger.warning(f"Missing input file for '{ds.name}': Not found at {src_path}")

        
        def try_add_file(file_path, description):
            if file_path.exists() and file_path.is_file():
                arc_name = f"{zip_folder}/{file_path.name}"
                files_to_pack.append((file_path, arc_name))
            else:
                logger.warning(f"Missing {description} for '{ds.name}'")

        if include_mre:
            try_add_file(out_dir / f"{ds.name}_MRE.csv", "MRE table")

        if include_mre_plot:
            if ds.mre is not None:
                try_add_file(out_dir / f"{ds.name}_MRE_plot.png", "MRE plot")
            else:
                try_add_file(out_dir / f"{ds.name}_CD_plot.png", "CD plot")

        if include_2dcos:
            try_add_file(out_dir / f"{ds.name}_sync.csv", "Sync table")
            try_add_file(out_dir / f"{ds.name}_async.csv", "Async table")

        if include_2dcos_plot:
            try_add_file(out_dir / f"{ds.name}_2DCOS_combined.png", "2D-COS plot")

    if not files_to_pack:
        raise RuntimeError("No files to download. Please select files first.")

    job_label = (session.job_name or "analysis").strip()
    zip_filename = f"{job_label}_2DCOS_results.zip"
    zip_path = out_dir / zip_filename

    files_to_pack = [
        (src, arc) for src, arc in files_to_pack 
        if src != zip_path
    ]

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for src_path, arc_name in files_to_pack:
                zf.write(src_path, arcname=arc_name)
                
        logger.info(f"Packed {len(files_to_pack)} file(s) into:\n{zip_path.name}")
        return zip_path

    except Exception as e:
        logger.error(f"Error creating ZIP archive: {e}")
        raise RuntimeError(f"Failed to create archive: {e}")