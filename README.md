# Kenya Soil DSS (prototype)

Workflow for mapping soil properties from point data and (later) serving them
through an AI agent that supports agricultural planning in Kenya.

This prototype uses legacy GloSIS-harmonized data for **Kakamega and Trans Nzoia**
(pH and SOC, 0-30 cm) to build and test the workflow. The legacy data is a
placeholder: the workflow is designed so that new soil testing data can be
plugged in and the system scaled to more counties and properties.

## Workflow

| Step | File | Runs in | Output |
|---|---|---|---|
| 1. Point table | `scripts/01_build_point_table.py` | Colab / Python | `data/points_ph_soc_kakamega_transnzoia.csv`, point map |
| 2. Covariates | `gee/02_covariates_gee.js` | GEE Code Editor | 30-band covariate GeoTIFF (100 m, UTM 36N) + band list CSV, exported to Drive |
| 3. Model + prediction | `scripts/03_model_predict.py` | Colab / Python | QRF prediction rasters (q05, q50, q95, PI width, DI, AOA), CV metrics, quick-look maps |

Planned: 4. indicators (threshold classes), 5. agent tools,
6. smolagents agent, 7. config-driven scaling to new data and counties.

## Data (not in this repo)

Data and outputs are excluded by `.gitignore`. Place files as follows:

```
data/raw/          Kakamega_KCEP_AGRA_GloSIS_harmonized.xlsx
                   TransNzoia_KCEP_GloSIS_harmonized.xlsx
data/covariates/   cov_kak_tnz_100m*.tif          (GEE export)
                   cov_kak_tnz_100m_bands.csv     (GEE export)
```

## How to run

**Step 2 (GEE):** paste `gee/02_covariates_gee.js` into the Code Editor, run it,
then start both exports in the Tasks tab and download them from Drive
(`soil_dss_covariates` folder) to `data/covariates/`.

**Steps 1 and 3 (Colab):** put this repo folder in Google Drive, open
`notebooks/colab_run.ipynb` in Colab, set `REPO` in the first cell and run all cells.

Locally, from the repo root:

```bash
pip install -r requirements.txt
python scripts/01_build_point_table.py
python scripts/03_model_predict.py
```

## Methods (short)

- Topsoil values (0-30 cm) per site; sites with several specimens are averaged.
- Covariates: terrain (NASADEM, MERIT Hydro), climate (CHIRPS, WorldClim,
  TerraClimate, MODIS LST), Sentinel-2 all-year and dry-season composites.
- Model: Quantile Regression Forest; spatial block cross-validation (10 km).
- Outputs include prediction intervals and an Area of Applicability
  (Meyer & Pebesma 2021) so downstream tools can flag weak predictions.

## Status

Prototype under the FAO SoilFER project. Private - do not share data or outputs
without clearance.
