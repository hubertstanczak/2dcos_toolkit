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

    # Collect CSV files 
    cd_files = collect_cd_files_from_paths(
        [str(p) for p in root_paths],
        input_dir=str(session.input_dir),
    )

    # Deduplicate files 
    cd_files = _dedupe_files_by_name(cd_files)


    logger.info(f"\nFound {len(cd_files)} file(s). Parsing...")

    parsed_datasets = []
    failed_files = []

    for f_path_str in cd_files:
        f_path = Path(f_path_str)
        try:
            ds = parse_cd_file(f_path)
            
            ds.name = make_base_name(f_path)
            ds.source_path = str(f_path.resolve())

            parsed_datasets.append(ds)
            logger.info(f"- Parsed: {f_path.name}")

        except Exception as e:
            failed_files.append(f"{f_path.name} ({e})")

    if failed_files:
        logger.info(f"\nFailed to parse {len(failed_files)} file(s):")
        logger.info("Check file format: expected CSV with numeric data.")

    # Update Session
    session.cd_files = cd_files
    session.datasets = parsed_datasets

    if not parsed_datasets:
        raise RuntimeError(
            "Input files were found, but none could be parsed successfully.\n"
        )
    
    logger.info(f"\nSuccessfully loaded {len(parsed_datasets)} dataset(s).")
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