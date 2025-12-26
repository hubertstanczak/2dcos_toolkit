from typing import Sequence
from pathlib import Path

from .io import collect_cd_files_from_paths, parse_cd_file
from .models import SessionState
from .utils import ensure_dir, make_base_name

import logging
logger = logging.getLogger(__name__)

PathLike = str | Path

def _dedupe_by_dataset_name(cd_files: list[str]) -> list[str]:
    """De-duplicate discovered files by canonical dataset name.

    The canonical name is derived the same way as in the parsing loop
    (make_base_name from the filename). If multiple paths map to the same
    dataset name (common after repeated ZIP uploads in Colab), keep the newest
    file (by mtime; tie-breaker: larger size).

    Parameters
    ----------
    cd_files:
        List of discovered file paths (strings).

    Returns
    -------
    list[str]
        File paths reduced to unique dataset names (stable order).
    """
    chosen: dict[str, tuple[str, float, int]] = {}  
    order: list[str] = []

    for f in cd_files:
        name = make_base_name(Path(f), max_len=80)
        st = Path(f).stat()
        mtime = float(st.st_mtime)
        size = int(st.st_size)

        if name not in chosen:
            chosen[name] = (f, mtime, size)
            order.append(name)
            continue

        _, old_mtime, old_size = chosen[name]
        if (mtime, size) > (old_mtime, old_size):
            chosen[name] = (f, mtime, size)

    return [chosen[name][0] for name in order]

def load_input_data_and_parse(
    session: SessionState,
    *,
    paths: Sequence[PathLike] | None = None,
) -> SessionState:
    """Load input CD files and parse them into datasets stored in the session.

    The function scans the given roots (files/directories/zip archives) and parses
    supported inputs into dataset objects.

    Parameters
    ----------
    session:
        Active session object. The function ensures ``session.output_dir`` and
        ``session.input_dir`` exist and writes parsed datasets to ``session.datasets``.
    paths:
        Optional sequence of file/directory/zip paths to scan.
        If None, the function scans ``session.input_dir``.

    Returns
    -------
    SessionState
        The same session instance (mutated in-place), returned for convenience.

    Side Effects
    ------------
    - Creates ``session.output_dir`` and ``session.input_dir`` if missing.
    - Populates:
      - ``session.cd_files``: list of discovered supported files
      - ``session.datasets``: list of parsed datasets

    Raises
    ------
    ValueError
        If ``paths`` contains invalid entries (None/empty).
    FileNotFoundError
        If no supported input files are found.
    RuntimeError
        If files are found but none can be parsed successfully.
    """
    session.output_dir = Path(session.output_dir)
    session.input_dir = Path(session.input_dir)

    ensure_dir(session.output_dir)
    ensure_dir(session.input_dir)

    if paths is None:
        root_paths: list[Path] = [Path(session.input_dir)]
    else:
        root_paths = []
        for p in paths:
            if p is None:
                raise ValueError("paths contains None. Provide valid file/directory paths.")
            p_str = str(p).strip()
            if not p_str:
                raise ValueError("paths contains an empty path. Provide valid file/directory paths.")
            root_paths.append(Path(p_str))

    logger.debug("\nInput data: scanning")
    logger.debug("Roots:")
    for rp in root_paths:
        logger.debug(f" - {rp}")

    cd_files = collect_cd_files_from_paths(
        [str(p) for p in root_paths],
        input_dir=str(session.input_dir),
    )

    cd_files = _dedupe_by_dataset_name(cd_files)

    if not cd_files:
        scanned = ", ".join(str(p) for p in root_paths)
        raise FileNotFoundError(
            "No supported input files found.\n"
            f"Scanned: [{scanned}]\n"
            f"Put files into: {session.input_dir}\n"
            "Supported: .csv, .xls, .xlsx (including .zip archives containing those)."
        )

    logger.info(f"\nFound {len(cd_files)} supported file(s).")

    parsed = []
    parsed_ok: list[str] = []
    failed_parse: list[str] = []

    for f in cd_files:
        try:
            ds = parse_cd_file(f)

            ds.name = make_base_name(Path(f), max_len=80)

            try:
                ds.source_path = str(f)
            except Exception:
                pass

            parsed.append(ds)
            parsed_ok.append(Path(f).name)

        except Exception:
            failed_parse.append(Path(f).name)

    if parsed_ok:
        logger.info("Successfully parsed file(s):")
        for name in parsed_ok:
            logger.info(f"- {name}")

    if failed_parse:
        logger.info("Failed to parse files(s):")
        for name in failed_parse:
            logger.info(f"- {name}")

    session.cd_files = cd_files
    session.datasets = parsed

    if not parsed:
        raise RuntimeError(
            "Input files were found, but none could be parsed.\n"
            "See 'Skipped:' messages above for exact reasons.\n"
            "Common causes: wrong delimiter/decimal, non-numeric cells, corrupted file, unsupported layout."
        )

    return session
