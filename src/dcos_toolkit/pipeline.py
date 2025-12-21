from typing import Sequence
from pathlib import Path

from .io import collect_cd_files_from_paths, parse_cd_file
from .models import SessionState
from .utils import ensure_dir, make_base_name

PathLike = str | Path


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

    print("\nInput data: scanning")
    print("Roots:")
    for rp in root_paths:
        print(f" - {rp}")

    cd_files = collect_cd_files_from_paths(
        [str(p) for p in root_paths],
        input_dir=str(session.input_dir),
    )

    if not cd_files:
        scanned = ", ".join(str(p) for p in root_paths)
        raise FileNotFoundError(
            "No supported input files found.\n"
            f"Scanned: [{scanned}]\n"
            f"Put files into: {session.input_dir}\n"
            "Supported: .csv, .xls, .xlsx (including .zip archives containing those)."
        )

    print(f"\nFound {len(cd_files)} supported file(s):")
    for f in cd_files:
        print(f" - {Path(f).name}")

    parsed = []
    skipped: list[tuple[str, str]] = []

    print("\nParsing files...")
    for f in cd_files:
        try:
            ds = parse_cd_file(f)

            # Canonical dataset name derived from input filename (deterministic)
            ds.name = make_base_name(Path(f), max_len=80)

            # Optional: preserve source path (debug/export)
            try:
                ds.source_path = str(f)
            except Exception:
                pass

            parsed.append(ds)
            n_spec, n_lam = ds.cd_mdeg.shape
            print(f"Parsed: {Path(f).name} -> dataset='{ds.name}' (spectra={n_spec}, lambda={n_lam})")

        except Exception as exc:
            reason = str(exc) or exc.__class__.__name__
            skipped.append((f, reason))
            print(f"Skipped: {Path(f).name} -> {reason}")

    session.cd_files = cd_files
    session.datasets = parsed

    print("\nInput summary:")
    print(f" found  : {len(cd_files)}")
    print(f" parsed : {len(parsed)}")
    print(f" skipped: {len(skipped)}")

    if not parsed:
        raise RuntimeError(
            "Input files were found, but none could be parsed.\n"
            "See 'Skipped:' messages above for exact reasons.\n"
            "Common causes: wrong delimiter/decimal, non-numeric cells, corrupted file, unsupported layout."
        )

    if skipped:
        print("\nSkipped details:")
        for f, reason in skipped:
            print(f" - {Path(f).name}: {reason}")

    print("\nInput parsing complete.")
    return session


__all__ = ["load_input_data_and_parse"]
