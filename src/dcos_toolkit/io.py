import logging
import shutil
import zipfile
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .models import CDDataset
from .utils import ensure_dir, sort_lambda_and_matrix

logger = logging.getLogger(__name__)


def collect_cd_files_from_paths(root_paths: list[str], *, input_dir: str) -> list[str]:
    workspace = Path(input_dir)
    ensure_dir(workspace)

    paths_to_check = []
    for p in root_paths:
        if p and p.strip():
            paths_to_check.append(Path(p.strip()))

    collected_files = []
    
    for path in paths_to_check:
        if path.is_file():
            if path.suffix.lower() == ".zip":
                target_dir = workspace / f"_zip_{path.stem}"
                
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                ensure_dir(target_dir)
                
                if _extract_zip(path, target_dir):
                    collected_files.extend(target_dir.rglob("*.csv"))
            
            elif path.suffix.lower() == ".csv":
                collected_files.append(path)

        elif path.is_dir():
            collected_files.extend(path.rglob("*.csv"))

    unique_paths = []
    seen_paths = set()

    for p in collected_files:
        abs_path = str(p.resolve())
      
        if abs_path not in seen_paths:
            seen_paths.add(abs_path)
            unique_paths.append(abs_path)

    return unique_paths


def parse_cd_file(path: str | Path) -> CDDataset:

    file_path = Path(path)
    
    df_raw = _read_file_data(file_path)
    
    if df_raw is None:
        raise ValueError(f"Failed to read table from file: {file_path.name}")

    dataset = _parse_to_dataset(df_raw, name=file_path.stem)
    
    if dataset is None:
        raise ValueError(f"File does not contain valid numeric data: {file_path.name}")

    return dataset


def parse_cd_files(paths: list[str]) -> list[CDDataset]:
    valid_datasets = []
    for p in paths:
        try:
            ds = parse_cd_file(p)
            valid_datasets.append(ds)
        except Exception as e:
            logger.warning(f"Skipped file '{Path(p).name}': {e}")
            continue
    return valid_datasets


def _extract_zip(zip_path: Path, output_folder: Path) -> bool:
    try:
        output_abs = output_folder.resolve()
        did_extract = False

        with zipfile.ZipFile(zip_path, "r") as zf:
            for file_info in zf.infolist():
                if file_info.is_dir():
                    continue

                target_file = (output_folder / file_info.filename).resolve()
                if output_abs not in target_file.parents and output_abs != target_file:
                    continue

                ensure_dir(target_file.parent)
                
                with zf.open(file_info, "r") as source:
                    with open(target_file, "wb") as target:
                        shutil.copyfileobj(source, target)
                
                did_extract = True

        return did_extract
    except zipfile.BadZipFile:
        return False


def _detect_jasco(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(100):
                line = f.readline()
                if not line:
                    break
                if "XYDATA" in line.upper():
                    return True
    except Exception:
        pass
    return False


def _read_file_data(path: Path) -> pd.DataFrame | None:
    if _detect_jasco(path):
        try:
            return _parse_jasco(path)
        except Exception as e:
            return _parse_csv(path)
    
    return _parse_csv(path)


def _parse_csv(path: Path) -> pd.DataFrame:
    first_line = ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(50):
                line = f.readline()
                if line.strip():
                    first_line = line.strip()
                    break
    except Exception:
        pass

    sep = ","
    decimal = "."

    if ";" in first_line:
        sep = ";"
        decimal = ","
    elif "\t" in first_line:
        sep = "\t"
        decimal = "."

    return pd.read_csv(
        path, 
        sep=sep, 
        decimal=decimal, 
        engine="python"
    )


def _parse_jasco(path: Path) -> pd.DataFrame:
    '''
    Extracts Channel 1 data from JASCO export file
    '''
 
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    start_idx = -1
    for i, line in enumerate(lines):
        if "XYDATA" in line.upper():
            for j in range(i + 1, len(lines)):
                if "Channel 1" in lines[j]:
                    start_idx = j
                    break
            break
    
    if start_idx == -1:
        raise ValueError("Channel 1 section not found")

    header_idx = -1
    for i in range(start_idx + 1, len(lines)):
        if lines[i].strip():
            header_idx = i
            break
            
    if header_idx == -1:
        raise ValueError("Missing data header")

    header_line = lines[header_idx]
    
    if ";" in header_line:
        sep = ";"
    elif "\t" in header_line:
        sep = "\t"
    else:
        sep = ","

    raw_headers = header_line.split(sep)
    column_names = []
    
    for h in raw_headers:
        clean_h = h.strip()
        if clean_h and clean_h.lower() not in ["wavelength", "nanometers", "nm"]:
            column_names.append(clean_h)

    data_rows = []
    expected_cols = len(column_names) + 1  

    for i in range(header_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            break
            
        first_char = line[0]
        if not (first_char.isdigit() or first_char in "-+."):
            break
        
        parts = line.replace(",", ".").split(sep)
        
        row_values = []
        for p in parts:
            p_clean = p.strip()
            if p_clean:
                try:
                    row_values.append(float(p_clean))
                except ValueError:
                    row_values.append(np.nan)
        
        if row_values:
            if len(row_values) != expected_cols:
                raise ValueError(f"Inconsistent column count at line {i+1}: expected {expected_cols}, found {len(row_values)}")
            data_rows.append(row_values)

    if not data_rows:
        raise ValueError("No data found below header")
    
    full_columns = ["Wavelength"] + column_names
    
    df = pd.DataFrame(data_rows, columns=full_columns)

    return df


def _parse_to_dataset(df: pd.DataFrame, *, name: str) -> CDDataset | None:
    if df is None or df.shape[1] < 2:
        return None

    df = df.dropna(axis=1, how="all")

    wavelengths = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    
    if wavelengths.isna().all():
        return None

    data_matrix = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

    valid_mask = ~wavelengths.isna()
    wavelengths = wavelengths[valid_mask]
    data_matrix = data_matrix.loc[valid_mask]

    if data_matrix.shape[1] == 0:
        return None

    lambda_arr = wavelengths.to_numpy(dtype=float)
    

    cd_values = data_matrix.to_numpy(dtype=float).T

    try:
        temp_values = []
        for col in data_matrix.columns:
            found = re.findall(r"[-+]?\d*\.\d+|\d+", str(col).replace(",", "."))

            if found:
                temp_values.append(float(found[0]))
            else:
                raise ValueError 
        
        perturbation_arr = np.array(temp_values, dtype=float)
        
    except ValueError:
        perturbation_arr = np.arange(cd_values.shape[0], dtype=float)

    if cd_values.shape[0] != perturbation_arr.size:
        raise ValueError(f"Dimension mismatch: {cd_values.shape[0]} spectra vs {perturbation_arr.size} perturbation values")
    
    if cd_values.shape[1] != lambda_arr.size:
        raise ValueError(f"Dimension mismatch: {cd_values.shape[1]} data points vs {lambda_arr.size} wavelengths")

    final_lambda, final_matrix = sort_lambda_and_matrix(lambda_arr, cd_values)

    return CDDataset(
        name=name,
        lambda_axis=final_lambda,
        perturbation_axis=perturbation_arr,
        cd_mdeg=final_matrix,
    )