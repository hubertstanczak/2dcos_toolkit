from pathlib import Path
import logging
import sys

from .models import SessionState
from .utils import ensure_dir, make_base_name
from .cos import compute_2dcos
from .export import package_results
from .pipeline import load_input_data_and_parse
from .mre import compute_mre
from .plotting import visualize_session

import logging
logger = logging.getLogger(__name__)

_INPUT_SUBDIR = "input_cd"
_DEFAULT_JOB_NAME = "my_analysis"
_DEFAULT_OUTPUT_DIR = "results"




def setup_logging(level: str = "INFO", *, style: str = "colab") -> None:
    import logging, sys

    lvl = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(message)s" if style == "colab" else "%(levelname)s:%(name)s:%(message)s"

    logging.basicConfig(
        level=lvl,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    logging.getLogger("dcos_toolkit").setLevel(lvl)



def _sanitize_job_name(job_name: str, *, max_len: int = 60) -> str:
    """Return a deterministic, filesystem/ZIP-safe job name.

    Parameters
    ----------
    job_name:
        User-provided label used in exported filenames/ZIP archives.
    max_len:
        Maximum length of the sanitized name. If exceeded, the name is shortened
        and suffixed with a short hash.

    Notes
    -----
    Sanitization is intentionally conservative:
    whitespace is collapsed to underscores and path separators are removed to
    avoid accidental nested paths in exported artifacts.
    """
    raw = (job_name or "").strip()
    safe = make_base_name(raw, max_len=max_len)

    if safe == "_unnamed":
        safe = _DEFAULT_JOB_NAME

    if raw and safe == "dataset" and raw.lower() != "dataset":
        safe = _DEFAULT_JOB_NAME
    
    if safe and set(safe) <= {"_", "-"}:
        safe = _DEFAULT_JOB_NAME

    if raw and safe != raw:
        logger.info(f"job_name sanitized: {raw!r} -> {safe!r}")
    return safe


def _validate_output_dir(output_dir: str) -> Path:
    """Validate and normalize the output directory path.

    This is a lightweight guard against obvious path traversal attempts.
    It is not intended as a security boundary in hostile environments.

    Raises
    ------
    ValueError
        If the path contains '..' components.
    """
    raw = (output_dir or "").strip() or _DEFAULT_OUTPUT_DIR
    path = Path(raw)

    if ".." in path.parts:
        raise ValueError("output_dir must not contain '..'")

    return path


def init_session(job_name: str = _DEFAULT_JOB_NAME, output_dir: str = _DEFAULT_OUTPUT_DIR) -> SessionState:
    """Initialize a new analysis session and create output directories.

    Parameters
    ----------
    job_name:
        Human-readable label used to name exported artifacts (e.g., ZIP bundle).
        The value is sanitized to a safe base name.
    output_dir:
        Directory in which all results and intermediate files are stored.
        A subdirectory named ``input_cd`` is created for raw input files.

    Returns
    -------
    SessionState
        Session object carrying configuration and results between steps.

    Side Effects
    ------------
    Creates (if missing) the following directories:
    - ``output_dir`` (root for this session)
    - ``output_dir/input_cd`` (place input spectra files here)

    Raises
    ------
    ValueError
        If ``output_dir`` contains '..' path components.
    """
    safe_job = _sanitize_job_name(job_name)
    out_path = _validate_output_dir(output_dir)

    session = SessionState(job_name=safe_job, output_dir=out_path)
    session.input_dir = session.output_dir / _INPUT_SUBDIR

    ensure_dir(session.output_dir)
    ensure_dir(session.input_dir)

    return session


__all__ = [
    "SessionState",
    "init_session",
    "load_input_data_and_parse",
    "compute_mre",
    "compute_2dcos",
    "visualize_session",
    "package_results",
]

load_input = load_input_data_and_parse
