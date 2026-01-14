import logging
from pathlib import Path

from .io import collect_cd_files_from_paths, parse_cd_file
from .models import SessionState
from .utils import ensure_dir, make_base_name

logger = logging.getLogger(__name__)

def load_input_data_and_parse(
    session: SessionState,
    *,
    paths: list[str | Path] | None = None,
) -> SessionState:
    """
    Main pipeline entry point.
    Scans for CSV files, deduplicates them, parses content, and updates the session.
    """
    session.output_dir = Path(session.output_dir)
    session.input_dir = Path(session.input_dir)

    ensure_dir(session.output_dir)
    ensure_dir(session.input_dir)

    if paths:
        root_paths = [Path(str(p).strip()) for p in paths if p and str(p).strip()]
    else:
        root_paths = [session.input_dir]

    logger.debug("Scanning input roots:")
    for rp in root_paths:
        logger.debug(f" - {rp}")

    cd_files = collect_cd_files_from_paths(
        [str(p) for p in root_paths],
        input_dir=str(session.input_dir),
    )

    cd_files = _dedupe_files_by_name(cd_files)

    if not cd_files:
        has_any_files = False
        if session.input_dir.exists():
            for _ in session.input_dir.iterdir():
                has_any_files = True
                break
        
        if has_any_files:
            raise RuntimeError(
                "Input files were detected, but none are in a supported format.\n"
                "Please verify the file extensions and ensure that the files contain valid data."
            )
        else:
            raise RuntimeError("No input files found. Please upload .csv or .zip files.")

    logger.info(f"Found {len(cd_files)} supported file(s):")
    for f in cd_files:
        logger.info(f"- {Path(f).name}")


    parsed_datasets = []
    failed_files = []

    for f_path_str in cd_files:
        f_path = Path(f_path_str)
        try:
            ds = parse_cd_file(f_path)
            
            ds.name = make_base_name(f_path)
            ds.source_path = str(f_path.resolve())

            parsed_datasets.append(ds)

        except Exception as e:
            failed_files.append((f_path.name, str(e)))

    if parsed_datasets:
        logger.info("")
        logger.info(f"Successfully parsed {len(parsed_datasets)} dataset(s):")
        for ds in parsed_datasets:
            logger.info(f"- {ds.name}")

    if failed_files:
        logger.info("")
        logger.info(f"Failed to parse {len(failed_files)} file(s):")
        for fname, error_msg in failed_files:
            logger.info(f"- {fname}")
            logger.info(f"  Error: {error_msg}")

        
        logger.info("\nCheck file structure: expected CSV with numeric data.")

    session.cd_files = cd_files
    session.datasets = parsed_datasets

    if not parsed_datasets:
        raise RuntimeError(
            "Input files were found, but none could be parsed successfully.\n"
            "Check file structure: expected CSV with numeric data."
        )
    
    return session


def _dedupe_files_by_name(file_paths: list[str]) -> list[str]:
    """
    Deduplicates files based on their base name.
    If multiple files resolve to the same name (e.g. duplicates), 
    keeps the one with the most recent modification time.
    """
    best_candidates = {}
    ordered_names = []

    for f in file_paths:
        path = Path(f)
        name = make_base_name(path)
        
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue 

        if name not in best_candidates:
            best_candidates[name] = (f, mtime)
            ordered_names.append(name)
        else:
            _, current_mtime = best_candidates[name]
            if mtime > current_mtime:
                best_candidates[name] = (f, mtime)

    return [best_candidates[name][0] for name in ordered_names]