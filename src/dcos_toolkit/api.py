import logging
import sys
from pathlib import Path

from .models import SessionState
from .utils import ensure_dir, make_base_name
from .cos import compute_2dcos
from .export import package_results
from .pipeline import load_input_data_and_parse
from .mre import compute_mre, generate_mre_plots 
from .plotting import visualize_session

logger = logging.getLogger(__name__)

_INPUT_SUBDIR = "input_cd"
_DEFAULT_JOB_NAME = "my_analysis"
_DEFAULT_OUTPUT_DIR = "results"


def setup_logging(level: str = "INFO", *, style: str = "colab") -> None:
    """"
    Configures logging style for Google Colab.
    """
    lvl = getattr(logging, level.upper(), logging.INFO)

    if style == "colab":
        fmt = "%(message)s"
    else:
        fmt = "%(levelname)s:%(name)s:%(message)s"

    logging.basicConfig(
        level=lvl,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    logging.getLogger("dcos_toolkit").setLevel(lvl)


def init_session(job_name: str = _DEFAULT_JOB_NAME, output_dir: str = _DEFAULT_OUTPUT_DIR) -> SessionState:
    """
    Initializes a new analysis session and prepares directories.
    """
    
    # Clean name
    safe_job = (job_name or "").strip()
    if not safe_job:
        safe_job = _DEFAULT_JOB_NAME
        logger.info(f"Empty job_name provided, using default: {_DEFAULT_JOB_NAME}")
    else:
        cleaned = make_base_name(safe_job, max_len=60)

        if cleaned == "dataset":
            cleaned = _DEFAULT_JOB_NAME

        if cleaned != safe_job:
            logger.info(f"job_name adjusted to '{cleaned}'")
        
        safe_job = cleaned

    out_path_str = (output_dir or "").strip() or _DEFAULT_OUTPUT_DIR
    out_path = Path(out_path_str)
    
    if ".." in out_path.parts:
        raise ValueError("Output directory must not contain '..' ")

    # Create Session
    session = SessionState(job_name=safe_job, output_dir=out_path)
    session.input_dir = session.output_dir / _INPUT_SUBDIR
    ensure_dir(session.output_dir)
    ensure_dir(session.input_dir)

    return session

load_input = load_input_data_and_parse

__all__ = [
    "SessionState",
    "init_session",
    "load_input",
    "load_input_data_and_parse",
    "compute_mre",
    "generate_mre_plots",
    "compute_2dcos",
    "visualize_session",
    "package_results",
    "setup_logging",
]