# 2DCOS Analysis Toolkit for CD Spectra

A notebook-first toolkit designed to process temperature-dependent **Circular Dichroism (CD)** spectra. It handles the entire pipeline: parsing raw data, converting to **MRE** (Mean Residue Ellipticity), computing **2D Correlation Spectroscopy (2DCOS)** maps (synchronous & asynchronous), and exporting figures.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hubertstanczak/2dcos_toolkit/blob/main/notebooks/2dcos_toolkit_colab.ipynb)

The repository includes sample datasets (`Dim.csv` and `Tri_2.csv`) used for automated testing. Files are located in the tests/data/ directory.

---

## Colab Workflow (Recommended)
The easiest way to use this toolkit is via Google Colab using the provided notebook.

### 1. Setup & Upload
* Open the notebook using the badge above.
* Run the **Setup** cell to install the toolkit environment.
* Run the **Data Upload** cell. You can upload:
    * `.csv` files (Formatted tables).
    * `JASCO` exports (containing `XYDATA` marker); Channel 1 signal is extracted.
    * `.zip` archives containing the above.

### 2. Processing Pipeline
Execute the cells sequentially:
1.  **MRE Calculation:** Converts raw signal [mdeg] to MRE $[\theta]$ (optional). 

2.  **1D Spectra visualization:** Plots the spectra (either raw or MRE).

3.  **2DCOS Calculation:** Computes Synchronous ($\Phi$) and Asynchronous ($\Psi$) correlation matrices.

4.  **2DCOS maps visualization:** Generates high-resolution contour maps with optional **automated peak annotation**.

### 3. Export Results
The final cell packages all selected files (MRE tables, PNG figures, 2DCOS matrices) into a single ZIP archive (`<job_name>_2DCOS_results.zip`) for download.

---

##  Input Data Formats

The toolkit supports **CSV** and **JASCO** files. 

*Note: Excel (`.xls`, `.xlsx`) files are not directly supported; please save them as CSV first.*

### 1. Standard Table (CSV)
The file must be a numeric table with a specific layout:

* **Row 1 (Header):** Perturbation values in increasing order.
* **Column 1:** Wavelength values (nm).
* **Body:** CD Signal values (mdeg).

The parser automatically detects ,, ;, or \t delimiters.

### 2. JASCO Exports
The toolkit supports raw ASCII exports from JASCO spectrometers.
* It looks for the `XYDATA` block.
* It specifically extracts data from **Channel 1**.
* It expects a matrix format (Wavelength vs. Temperature columns).

---

## Local Installation & Usage

If you prefer running locally instead of Colab:

### Requirements
* Python 3.10+

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/hubertstanczak/2dcos_toolkit.git](https://github.com/hubertstanczak/2dcos_toolkit.git)
    cd 2dcos_toolkit
    ```
2.  Install in editable mode:
    ```bash
    pip install -e 
    ```

### Running test
This project uses pytest for unit testing. To verify the installation and core mathematical logic:

```bash
pytest
```

---

## Analysis Parameters

### MRE Formula
The Mean Residue Ellipticity is calculated using the following parameters:


* **$M$**: Molar mass (g/mol).
* **$\theta_{obs}$**: Observed ellipticity (mdeg).
* **$c$**: Concentration (mg/mL).
* **$l$**: Path length (**cm**).
* **$n$**: Number of residues.

### 2DCOS Reference
The Dynamic Spectrum is calculated by subtracting a reference spectrum :
* **mean**: Average of all spectra (recommended default).
* **first**: The spectrum at the lowest perturbation.
* **last**: The spectrum at the highest perturbation.
* **none**: No subtraction.



### Peak Annotation
The visualization module includes an automated peak picker with the following features:
* **Peak Counts**: You can specify the maximum number of peaks (0–5) to label for Synchronous Diagonal, Synchronous Cross-peaks, and Asynchronous Cross-peaks. Peaks are searched in the upper half of the map, above the diagonal.

* **Mirroring**: Option to mark peaks symmetrically across the diagonal for visual clarity. 

---

## 📄 License

MIT License