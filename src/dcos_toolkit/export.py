import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import logging
logger = logging.getLogger(__name__)

from .models import SessionState
from .utils import ensure_dir, make_base_name


def list_input_cd_files(session: SessionState) -> list[Path]:
    """Discover supported input CD files under ``session.input_dir``.

    This helper is intentionally recursive because ZIP extraction and user workflows
    may create nested subfolders inside the input directory.

    Parameters
    ----------
    session:
        Active analysis session. The function uses ``session.input_dir`` as the root.

    Returns
    -------
    list[Path]
        A deterministic, de-duplicated list of discovered files with extensions:
        ``.csv``, ``.xls``, ``.xlsx``. The ordering is stable across runs.

    Notes
    -----
    - If ``session.input_dir`` does not exist or is not a directory, returns an empty list.
    - De-duplication is performed on resolved absolute paths.
    """
    input_dir: Path = session.input_dir
    if not input_dir or not input_dir.is_dir():
        return []

    exts = {".csv", ".xlsx", ".xls"}
    files: list[Path] = []

    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)

    # stable ordering + de-dup
    unique = sorted({f.resolve() for f in files}, key=lambda x: (x.name.lower(), str(x)))
    return unique


# ----------------------------
# Selection & grouping
# ----------------------------

def _unique_sorted(paths: Iterable[Path]) -> list[Path]:
    """Return unique existing files as a stable, sorted list.

    Parameters
    ----------
    paths:
        Iterable of candidate paths.

    Returns
    -------
    list[Path]
        Unique file paths (only entries that are actual files), sorted
        deterministically by (filename, full path).

    Notes
    -----
    This helper is used to keep packaging output deterministic and to avoid
    duplicates introduced by multiple discovery passes.
    """
    unique = {p for p in paths if p is not None and p.is_file()}
    return sorted(unique, key=lambda p: (p.name.lower(), str(p)))


def _starts_with_any_base(filename: str, base_names: Sequence[str]) -> str | None:
    """Return the matched base name if ``filename`` starts with ``<base>_`` (case-insensitive).

    Parameters
    ----------
    filename:
        File name to match (not a path).
    base_names:
        Candidate base names derived from input files.

    Returns
    -------
    str | None
        The matching base name if found, otherwise None.

    Notes
    -----
    This is used to map output artifacts to dataset subfolders inside the ZIP
    using the naming convention: ``<dataset_base>_<artifact_suffix>``.
    """
    lower = filename.lower()
    for base in base_names:
        prefix = (base + "_").lower()
        if lower.startswith(prefix):
            return base
    return None


def _is_mre_table(p: Path) -> bool:
    """Return True if path looks like an MRE CSV table (case-insensitive)."""
    return p.name.lower().endswith("_mre.csv")


def _is_mre_plot(p: Path) -> bool:
    """Return True if path looks like a plot produced in the MRE step (case-insensitive).

    Recognized suffixes:
    - ``*_CD_plot.png``
    - ``*_MRE_plot.png``
    """
    name = p.name.lower()
    return name.endswith("_cd_plot.png") or name.endswith("_mre_plot.png")


def _is_2dcos_matrix(p: Path) -> bool:
    """Return True if path looks like a saved 2D-COS matrix CSV (case-insensitive)."""
    name = p.name.lower()
    return name.endswith("_sync.csv") or name.endswith("_async.csv")


def _is_2dcos_plot(p: Path) -> bool:
    """Return True if path looks like a combined 2D-COS plot PNG (case-insensitive).

    The plotting module produces: ``<dataset>_2DCOS_combined.png``.
    """
    return p.name.lower().endswith("_2dcos_combined.png")


def collect_selected_files(
    session: SessionState,
    *,
    input_files: Sequence[Path],
    include_input_file: bool,
    include_mre: bool,
    include_mre_plot: bool,
    include_2dcos: bool,
    include_2dcos_plot: bool,
) -> list[Path]:
    """Collect files to package from ``output_dir`` and (optionally) raw input files.

    Parameters
    ----------
    session:
        Active analysis session. Files are collected from ``session.output_dir``.
    input_files:
        Discovered input files (usually from :func:`list_input_cd_files`).
    include_input_file:
        If True, input files are included in the ZIP (placed under per-dataset folders).
    include_mre:
        If True, include MRE tables (``*_MRE.csv`` / case-insensitive match).
    include_mre_plot:
        If True, include MRE/CD plots (``*_MRE_plot.png`` or ``*_CD_plot.png``).
    include_2dcos:
        If True, include 2D-COS matrices (``*_sync.csv`` and ``*_async.csv``).
    include_2dcos_plot:
        If True, include the combined 2D-COS figure (``*_2DCOS_combined.png``).

    Returns
    -------
    list[Path]
        Deterministic, de-duplicated list of selected files.

    """
    output_dir = session.output_dir
    ensure_dir(output_dir)

    selected: list[Path] = []

    if include_input_file:
        selected.extend([p for p in input_files if p.is_file()])

    # Outputs are expected in output_dir top-level (deterministic scan).
    for p in output_dir.iterdir():
        if not p.is_file():
            continue

        name_lower = p.name.lower()

        if name_lower.endswith("_2dcos_results.zip"):
            continue

        if include_mre and _is_mre_table(p):
            selected.append(p)
            continue

        if include_mre_plot and _is_mre_plot(p):
            selected.append(p)
            continue

        if include_2dcos and _is_2dcos_matrix(p):
            selected.append(p)
            continue

        if include_2dcos_plot and _is_2dcos_plot(p):
            selected.append(p)
            continue

    return _unique_sorted(selected)


def assign_zip_folder(
    file_path: Path,
    *,
    input_file_to_base: dict[Path, str],
    input_resolved_to_base: dict[Path, str],
    base_names: Sequence[str],
) -> str:
    """Assign a deterministic folder name (subdirectory) for a file inside the ZIP.

    Parameters
    ----------
    file_path:
        File being packaged.
    input_file_to_base:
        Mapping of original input file paths to their dataset base names.
        (Kept for completeness / debugging; the resolved mapping is used for robustness.)
    input_resolved_to_base:
        Mapping of resolved input file paths to dataset base names.
        This is the primary lookup to ensure consistent matching across relative paths.
    base_names:
        Ordered list of dataset base names derived from input files.

    Returns
    -------
    str
        ZIP subfolder name. One of:
        - a sanitized dataset base name (via :func:`make_base_name`), or
        - ``"_misc"`` if the file cannot be mapped deterministically.

    Rules
    -----
    1) If file is one of the input files => folder = its base name
    2) Else if filename starts with ``'<base>_'`` => folder = that base
    3) Else => ``'_misc'``

    Notes
    -----
    The folder name is passed through :func:`make_base_name` to avoid problematic
    characters in ZIP paths and to keep the structure platform-friendly.
    """
    abs_path = file_path.resolve()

    # (1) Input file
    base = input_resolved_to_base.get(abs_path)
    if base is not None:
        return make_base_name(base)

    # (2) Output file by prefix '<base>_'
    base = _starts_with_any_base(file_path.name, base_names)
    if base is not None:
        return make_base_name(base)

    return "_misc"


def package_results(
    session: SessionState,
    *,
    include_input_file: bool = True,
    include_mre: bool = True,
    include_mre_plot: bool = True,
    include_2dcos: bool = True,
    include_2dcos_plot: bool = True,
) -> Path:
    """Package selected session artifacts into a ZIP archive.

    The ZIP is created inside ``session.output_dir`` and uses per-input subfolders
    to keep results for each dataset grouped together.

    Parameters
    ----------
    session:
        Active analysis session. Uses:
        - ``session.input_dir`` to discover input files (recursive),
        - ``session.output_dir`` to collect output artifacts (top-level).
    include_input_file:
        If True, include raw input files in the ZIP.
    include_mre:
        If True, include MRE tables (``*_MRE.csv`` / case-insensitive match).
    include_mre_plot:
        If True, include CD/MRE plots (``*_CD_plot.png`` / ``*_MRE_plot.png``).
    include_2dcos:
        If True, include 2D-COS matrices (``*_sync.csv`` and ``*_async.csv``).
    include_2dcos_plot:
        If True, include combined 2D-COS plots (``*_2DCOS_combined.png``).

    Returns
    -------
    Path
        Path to the created ZIP archive.

    Raises
    ------
    RuntimeError
        If input files are missing, if requested artifact categories are missing,
        or if no files are selected for packaging.

    Side Effects
    ------------
    - Ensures ``session.output_dir`` exists.
    - Writes a ZIP file to ``session.output_dir``.

    Behavior
    --------
    This function performs a preflight check: if any requested artifact category
    is absent in ``session.output_dir``, it prints actionable instructions and
    raises a ``RuntimeError`` without creating a ZIP.
    """
    ensure_dir(session.output_dir)

    # 1) Inputs used to build per-file folders deterministically
    input_files = list_input_cd_files(session)
    if include_input_file and not input_files:
        raise RuntimeError(
            "Export aborted: no input CD files found.\n"
            "Upload files first.\n"
        )

    input_file_to_base: dict[Path, str] = {p: make_base_name(p) for p in input_files}
    input_resolved_to_base: dict[Path, str] = {p.resolve(): base for p, base in input_file_to_base.items()}

    base_names: list[str] = []
    seen: set[str] = set()
    for b in input_file_to_base.values():
        if b not in seen:
            base_names.append(b)
            seen.add(b)

    # 2) Preflight: check what exists BEFORE creating ZIP
    outputs = [p for p in session.output_dir.iterdir() if p.is_file()]

    def count(pred) -> int:
        return sum(1 for p in outputs if pred(p))

    missing_msgs: list[str] = []

    if include_mre and count(_is_mre_table) == 0:
        missing_msgs.append(
            "- include_mre is True, but no MRE files are found."
        )
    if include_mre_plot and count(_is_mre_plot) == 0:
        missing_msgs.append(
            "- include_mre_plot is True, but no MRE/CD plot files are found."
        )
    if include_2dcos and count(_is_2dcos_matrix) == 0:
        missing_msgs.append(
            "- include_2dcos is True, but no 2DCOS matrix files are found."
        )
    if include_2dcos_plot and count(_is_2dcos_plot) == 0:
        missing_msgs.append(
            "- include_2dcos_plot is True, but no 2DCOS plot files are found."
        )

    if missing_msgs:
        details = "\n".join(missing_msgs)

        raise RuntimeError(
            "Export cannot be created yet.\n"
            "Some selected result types are missing:\n"
            f"{details}\n"
            "\nRun the missing step(s) or disable the corresponding include option(s) and try again."
        )

    # 3) Collect files
    selected_files = collect_selected_files(
        session,
        input_files=input_files,
        include_input_file=include_input_file,
        include_mre=include_mre,
        include_mre_plot=include_mre_plot,
        include_2dcos=include_2dcos,
        include_2dcos_plot=include_2dcos_plot,
    )

    if not selected_files:
        raise RuntimeError(
            "Nothing selected for packaging.\n"
        )

    # 4) Create ZIP
    job_label = (session.job_name or "my_analysis").strip() or "my_analysis"
    zip_path = session.output_dir / f"{job_label}_2DCOS_results.zip"

    written = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in selected_files:
            if path.resolve() == zip_path.resolve():
                continue

            folder = assign_zip_folder(
                path,
                input_file_to_base=input_file_to_base,
                input_resolved_to_base=input_resolved_to_base,
                base_names=base_names,
            )
            arcname = Path(folder) / path.name
            zf.write(path, arcname=str(arcname))
            written += 1

    logger.info(f"Packed {written} file(s) into:\n{zip_path.name}")
    return zip_path
