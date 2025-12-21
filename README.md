# 2dcos_toolkit — 2DCOS toolkit for temperature-dependent CD spectra

Notebook-first toolkit to:
- load temperature-dependent **circular dichroism (CD)** spectra (including **JASCO “XYDATA” exports**),
- convert CD → **MRE** (molar residue ellipticity),
- compute **2D correlation spectroscopy (2DCOS)** maps (**synchronous** and **asynchronous**),
- generate figures and export a ZIP bundle with results.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hubertstanczak/2dcos_toolkit/blob/main/notebooks/2dcos_toolkit_colab.ipynb)

---

## Colab workflow (recommended)

### Upload inputs (popup file picker)
1. Open the Colab notebook from the badge above.
2. Use the upload cell in the notebook (it opens a popup file picker).
3. Select one or more files (`.csv`, `.xls`, `.xlsx`, or `.zip`) and upload them.
4. The notebook places uploaded files into the session input folder.

### Run the pipeline
Run notebook cells **top to bottom**. The pipeline order is:
1) parse inputs  
2) compute MRE  
3) compute 2D-COS  
4) generate plots + peak annotations  
5) export ZIP

### Download results (popup download)
At the end, the notebook produces a single ZIP (e.g. `my_run_2DCOS_results.zip`) and triggers a download.   You can also download individual PNG/CSV files from the Colab file browser, but the ZIP is the intended “one-click” deliverable.

---

## What files can be loaded?

- `.csv`
- `.xls`, `.xlsx`
- `.zip` archive containing any mixture of the above

---

## Input formats (how your files must look)

### 1) Standard table format (CSV / Excel) — REQUIRED layout

Your input file must contain a **numeric table**:

- **Column 1**: wavelength in **nm**
- **Columns 2..N**: CD values in **mdeg**, one column per spectrum (typically different temperatures)

Column headers for columns 2..N are used as the perturbation axis:
- if headers are numeric (e.g. `5`, `10`, `15` or `5.0`) they are used as temperature/perturbation values
- if headers are not numeric, the toolkit falls back to `0..N-1` (still works, but you lose true temperature labels)

Minimal CSV example:
```text
wavelength_nm,5,10,15,20
190,-12.3,-11.8,-10.2,-9.5
191,-12.1,-11.6,-10.1,-9.4
192,-11.9,-11.3,-9.9,-9.2
```

CSV example (semicolon + decimal):
```text
wavelength_nm;5;10;15
190;-12,3;-11,8;-10,2
191;-12,1;-11,6;-10,1
```

Excel (`.xls/.xlsx`) must follow the same structure:
- first column: wavelength (nm)
- remaining columns: CD values (mdeg)
- headers: temperatures/perturbation values if numeric

Practical rule: keep cells numeric (avoid units inside cells like `-12 mdeg`).

---

### 2) JASCO exports (ASCII with `XYDATA`)

The toolkit supports ASCII exports from **JASCO spectrometers**. If an uploaded `.csv` contains the `XYDATA` marker, it is treated as a JASCO export.

In JASCO files the CD signal is read from **`Channel 1`** (this is where CD data are stored in the export). The parser locates the `XYDATA` block, finds `Channel 1`, and then reads the numeric data that follow.

Supported (recommended) format: **matrix-style export**, where one file contains multiple spectra (e.g., multiple temperatures):
- header row contains the perturbation values (typically temperatures),
- each numeric row contains: `wavelength ; v1 ; v2 ; v3 ; ...` (values interpreted as mdeg).

If the export contains only **XY pairs** (one spectrum: wavelength + value), it can still be parsed as a single-spectrum dataset, but **2D-COS requires at least 3 spectra**, so 2D-COS will be skipped for such inputs.

---

### 3) ZIP archives

You can upload a `.zip` containing multiple input files (CSV/Excel/JASCO-CSV). The notebook extracts it into the session input folder and processes all supported files.

---

## What the pipeline produces (outputs)

For each input dataset (typically one input file), the toolkit can create:

MRE (optional):
- `<dataset>_MRE.csv`
- `<dataset>_CD_plot.png` and/or `<dataset>_MRE_plot.png`

2D-COS:
- `<dataset>_sync.csv`
- `<dataset>_async.csv`

Plots (optional):
- `<dataset>_2DCOS_combined.png`

Export:
- `<job_name>_2DCOS_results.zip`

The exported ZIP is organized into **one folder per dataset**, so all artifacts for each input are grouped together.

---

## Parameters (what they mean)

The notebook exposes parameters as simple cells. Exact defaults are visible in the notebook and in docstrings in `src/dcos_toolkit/api.py`.

MRE (only if you enable MRE):
- `residues_number` — number of residues
- `concentration_mg_ml` — concentration (mg/mL)
- `path_length_mm` — cuvette path length (mm)
- `molar_mass_g_mol` — molar mass (g/mol)
- `generate_plot` — save spectra plots
- `use_mre_for_plot` — plot MRE instead of raw CD
- `show` — show plots in the notebook

2D-COS:
- `use_mre_for_2dcos`
  - `True`: compute 2D-COS using MRE (requires running MRE first)
  - `False`: compute 2D-COS from CD (internally converts mdeg → deg)
- `reference_type` — reference spectrum for dynamic spectra ΔS:
  - `"mean"` (recommended default)
  - `"first"`, `"last"`, `"none"`

Important: **2D-COS requires at least 3 spectra** in a dataset.

Plotting / peak annotations (optional):
- `colormap` — Matplotlib colormap name (e.g., `"jet"`)
- `lambda_min`, `lambda_max` — crop wavelength range for maps (set both to `0` to disable cropping)
- `mark_peaks_sync`, `mark_peaks_async` — enable peak markers
- `n_sync_diag_peaks`, `n_sync_cross_max_peaks`, `n_sync_cross_min_peaks` — counts for synchronous peak markers
- `n_async_cross_max_peaks`, `n_async_cross_min_peaks` — counts for asynchronous peak markers
- `mark_mirror_peaks` — mirror peak coordinates across diagonal (visual option)

---

## Common issues

- “2D-COS skipped”: you need **≥ 3 spectra** per dataset (single-spectrum XY exports are not enough).
- “CSV parsed incorrectly”: delimiter/decimal mismatch or non-numeric cells. Export a clean numeric table.
- “Export ZIP fails”: export checks if requested artifacts exist; run plotting/MRE steps before exporting those artifacts.

---

## Local usage (optional)

Colab is the recommended path. Local usage is for offline runs.

Requirements:
- Python 3.10+

Install (repo root):
```bash
python -m pip install -U pip
python -m pip install -e .
```

Run the local notebook:
- `notebooks/2dcos_toolkit_local.ipynb`

Typical local folders:
- inputs: `data/input_cd/`
- outputs: `data/results/`

---

## License

MIT (see `LICENSE`).
