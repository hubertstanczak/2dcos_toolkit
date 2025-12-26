from pathlib import Path
import shutil
import zipfile
from io import StringIO

import numpy as np
import pandas as pd

from .models import CDDataset
from .utils import ensure_dir, sort_lambda_and_matrix


import logging
logger = logging.getLogger(__name__)

__all__ = ["collect_cd_files_from_paths", "parse_cd_file", "parse_cd_files"]


def collect_cd_files_from_paths(root_paths: list[str], *, input_dir: str) -> list[str]:
    """Collect supported CD input files from the provided roots.

    Parameters
    ----------
    root_paths:
        Roots to scan (files or directories). ZIP archives are supported.
    input_dir:
        Workspace directory used for extracted ZIP contents.

    Returns
    -------
    list[str]
        De-duplicated list of supported file paths (as strings).
    """
    workspace = Path(input_dir)
    ensure_dir(workspace)

    roots: list[Path] = []
    for raw in root_paths:
        s = str(raw).strip()
        if not s:
            raise ValueError("root_paths contains an empty path.")
        roots.append(Path(s))

    collected: list[Path] = []
    extracted_dirs: list[Path] = []

    for root in roots:
        for p in _iter_files(root):
            kind = _detect_file_kind(p)
            if kind == "zip":
                out_dir = workspace / f"_zip_{p.stem}"
                ensure_dir(out_dir)
                if _safe_extract_zip(p, out_dir):
                    extracted_dirs.append(out_dir)
                continue

            if kind != "unsupported":
                collected.append(p)

    for out_dir in extracted_dirs:
        for p in out_dir.rglob("*"):
            if p.is_file() and _detect_file_kind(p) != "unsupported":
                collected.append(p)

    # Dedupe (preserve order)
    out: list[str] = []
    seen: set[str] = set()
    for p in collected:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p.absolute())
        if key not in seen:
            seen.add(key)
            out.append(str(p))

    return out


def parse_cd_file(path: str | Path) -> CDDataset:
    """Parse a single supported CD file into a CDDataset.

    Raises ValueError if the file is unsupported or not convertible to a numeric matrix.
    """
    p = Path(path)
    df = _read_table(p)
    if df is None:
        raise ValueError(f"Unsupported or unreadable input: {p}")

    ds = _table_to_dataset(df, name=p.stem)
    if ds is None:
        raise ValueError(f"Input read but not convertible to CD matrix: {p}")

    return ds


def parse_cd_files(paths: list[str]) -> list[CDDataset]:
    """Parse multiple files (silent-skip on failures).

    Prefer pipeline.load_input_data_and_parse(...) for user-facing reporting.
    """
    datasets: list[CDDataset] = []
    for p in paths:
        try:
            datasets.append(parse_cd_file(p))
        except Exception:
            continue
    return datasets


def _iter_files(root: Path):
    """Yield files under root (file itself or recursive directory walk)."""
    if root.is_file():
        yield root
        return
    if root.is_dir():
        yield from (p for p in root.rglob("*") if p.is_file())


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> bool:
    """Safely extract ZIP into target_dir (prevents zip-slip)."""
    try:
        target_root = target_dir.resolve()
        extracted_any = False

        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue

                dest = (target_dir / info.filename).resolve()

                # zip-slip guard: skip entries escaping target_dir
                if target_root != dest and target_root not in dest.parents:
                    continue

                ensure_dir(dest.parent)
                with zf.open(info, "r") as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_any = True

        return extracted_any
    except zipfile.BadZipFile:
        return False


def _detect_file_kind(path: Path) -> str:
    """Return one of: 'zip' | 'excel' | 'jasco' | 'csv' | 'unsupported'."""
    ext = path.suffix.lower()

    if ext == ".zip":
        return "zip"
    if ext in {".xls", ".xlsx"}:
        return "excel"
    if ext not in {".csv", ".txt"}:
        return "unsupported"

    # CSV: check whether it is actually a JASCO ASCII export (XYDATA marker).
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(160):
                line = f.readline()
                if not line:
                    break
                if line.strip().upper() == "XYDATA":
                    return "jasco"
    except Exception:
        pass

    return "csv"


def _read_table(path: Path) -> pd.DataFrame | None:
    """Read supported input into a raw DataFrame."""
    kind = _detect_file_kind(path)

    if kind == "excel":
        return pd.read_excel(path)

    if kind == "csv":
        return _read_csv_matrix(path)

    if kind == "jasco":
        return _read_jasco_matrix(path)

    return None


def _read_csv_matrix(path: Path) -> pd.DataFrame:
    """Read CSV using a simple delimiter/decimal heuristic.

    Heuristic:
    - many ';' or startswith ';' => sep=';' and decimal=','
    - many tabs => sep='\\t'
    - otherwise => sep=',' and decimal='.'
    """
    first = ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(80):
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if s:
                    first = s
                    break
    except Exception:
        first = ""

    if first.startswith(";") or first.count(";") >= 3:
        return pd.read_csv(path, sep=";", decimal=",", engine="python", on_bad_lines="skip")

    if first.count("\t") >= 2:
        return pd.read_csv(path, sep="\t", decimal=".", engine="python", on_bad_lines="skip")

    return pd.read_csv(path, sep=",", decimal=".", engine="python", on_bad_lines="skip")


def _read_jasco_matrix(path: Path) -> pd.DataFrame:
    """Parse JASCO ASCII export into a DataFrame.

    Primary mode:
    - locate 'XYDATA' and then 'Channel 1'
    - read header with perturbation values (semicolon-separated)
    - read numeric matrix rows (wavelength;v1;v2;...)

    Fallback mode:
    - parse XY pairs after 'XYDATA' (two columns)
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    def find_line(marker: str, start: int) -> int:
        m = marker.strip().upper()
        for i in range(start, len(lines)):
            if lines[i].strip().upper() == m:
                return i
        raise ValueError(f"Marker '{marker}' not found in JASCO export.")

    i_xy = find_line("XYDATA", 0)

    # Try matrix format (Channel 1 + header + numeric rows)
    try:
        i_ch = find_line("Channel 1", i_xy + 1)

        # next non-empty line: header
        header_idx = None
        for i in range(i_ch + 1, len(lines)):
            if lines[i].strip():
                header_idx = i
                break
        if header_idx is None:
            raise ValueError("Missing Channel 1 header line.")

        header_parts = [p.strip() for p in lines[header_idx].split(";")]
        headers = [h for h in header_parts[1:] if h]  # skip wavelength column label/empty
        if not headers:
            raise ValueError("Could not parse Channel 1 headers.")

        # collect consecutive numeric-looking rows
        data_lines: list[str] = []
        for raw in lines[header_idx + 1 :]:
            s = raw.strip()
            if not s:
                break
            u = s.upper()
            if u.startswith("CHANNEL") or u.startswith("END") or u.startswith("###"):
                break
            if not (s[0].isdigit() or s[0] in "+-."):
                break
            data_lines.append(raw)

        if not data_lines:
            raise ValueError("No numeric matrix rows found (fallback to XY pairs).")

        def read_matrix(decimal: str) -> pd.DataFrame | None:
            df = pd.read_csv(
                StringIO("\n".join(data_lines)),
                sep=";",
                decimal=decimal,
                header=None,
                names=["Wavelength"] + headers,
                engine="python",
                on_bad_lines="skip",
            )
            if df.shape[1] < 2:
                return None
            try:
                pd.to_numeric(df["Wavelength"], errors="raise")
            except Exception:
                return None
            return df

        df = read_matrix(decimal=",")
        if df is None:
            df = read_matrix(decimal=".")
        if df is None:
            raise ValueError("Could not parse numeric matrix rows.")
        return df   

    except Exception:
        # Fallback: XY pairs after XYDATA
        xs: list[float] = []
        ys: list[float] = []

        for raw in lines[i_xy + 1 :]:
            s = raw.strip()
            if not s:
                break
            u = s.upper()
            if u.startswith("CHANNEL") or u.startswith("END"):
                break

            parts = [p for p in s.replace(",", ".").replace(";", " ").split() if p]
            if len(parts) < 2:
                continue
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except Exception:
                continue

        if not xs:
            raise ValueError("Could not parse JASCO export (neither matrix nor XY pairs).")

        return pd.DataFrame({"Wavelength": xs, "0": ys})



def _table_to_dataset(df: pd.DataFrame, *, name: str) -> CDDataset | None:
    """Convert a raw DataFrame into a CDDataset.

    Expected layout:
    - first column: wavelength (nm)
    - remaining columns: CD values for successive perturbation points (one column per spectrum)

    Output convention:
    - cd_mdeg has shape (n_spectra, n_lambda)
    """
    if df is None or df.shape[1] < 2:
        return None

    df = df.dropna(axis=1, how="all")

    wl = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    if wl.isna().all():
        return None

    mat = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

    valid = ~wl.isna()
    wl = wl[valid]
    mat = mat.loc[valid]

    if mat.shape[1] == 0 or mat.isna().all().all():
        return None

    lambda_axis = wl.to_numpy(dtype=float)
    cd_mdeg = mat.to_numpy(dtype=float).T  # (n_spectra, n_lambda)

    try:
        perturb = np.asarray([float(str(h).strip().replace(",", ".")) for h in mat.columns], dtype=float)
        if perturb.size != cd_mdeg.shape[0]:
            raise ValueError
    except Exception:
        perturb = np.arange(cd_mdeg.shape[0], dtype=float)

    lambda_axis, cd_mdeg = sort_lambda_and_matrix(lambda_axis, cd_mdeg)

    return CDDataset(
        name=name,
        lambda_axis=lambda_axis,
        perturbation_axis=perturb,
        cd_mdeg=cd_mdeg,
    )
