# AuditFlux Research Dashboard

A lightweight Streamlit presentation layer for exploring selected AuditFlux experiment outputs. It reads the bundled CSV files and **does not rerun experiments**.

## Run on Windows

1. Extract the `AuditFlux_Dashboard.zip` folder.
2. Open Command Prompt or PowerShell in the extracted folder.
3. (Recommended) Create and activate a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\activate
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. Launch:

```powershell
streamlit run app.py
```

Streamlit will open the dashboard in your browser, usually at `http://localhost:8501`.

## Pages

- Overview: research workflow and headline results.
- Fairness audit: baseline group metrics and threshold sweeps.
- Instability analysis: sample-size and threshold sensitivity.
- Representation study: Program 20/21 results.
- Robustness validation: Program 21B Huber and influence diagnostics.
- Data explorer: inspect and download bundled result tables.

## Data and reproducibility notes

- CSV files in `data/` are curated copies of selected experiment outputs.
- The dashboard intentionally does not include the full 850 MB experiment archive.
- All visualisations are based on the selected CSV outputs. Refer to the original scripts, manifests and complete output archive for full experimental provenance.
- The 74/108, 23/27, 80/81 and 0/90 headline counts are based on the project results reviewed for this dashboard. The detailed tables remain available in the relevant pages.
- No causal inference is claimed from the representation-space associations.

## Adding more results

Place the relevant CSV in `data/` and add its filename to the `FILES` dictionary in `app.py` before exposing it in the Data Explorer.
