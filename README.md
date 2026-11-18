# China_Lake_TN_Analysis

Code supporting **Spatiotemporal Variability of Total Nitrogen in Lakes and Reservoirs of China**.

## Research objectives

This study examines total nitrogen (TN) variability in Chinese lakes and reservoirs during 2021–2024. It characterizes spatiotemporal patterns, evaluates environmental and anthropogenic drivers, and explores nonlinear responses and interactions within a lake–basin framework.

## Analysis code

| Script | Function |
| --- | --- |
| scripts/01_temporal/stl.py | Daily TN aggregation and robust STL decomposition |
| scripts/02_clustering/kmeans.py | TN feature construction and K-means clustering for 2021–2024; default comparison: 2021–2022 versus 2023–2024 |
| scripts/03_drivers/xgboost_shap.py | XGBoost regression and seasonal SHAP summaries |
| scripts/04_thresholds/gam.py | Generalized additive models and candidate turning points |
| scripts/05_interactions/pdp.py | Gradient boosting regression and partial-dependence analysis |

The scripts cover the supplied analysis steps. Preparation of monitoring records, station–lake matching, basin delineation and integration of environmental inputs must be completed separately.

## Installation

Python 3.11 is recommended. requirements.txt lists direct package versions used for component execution; it is not a complete environment lockfile.

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

## Data and execution

See [input fields](data/README.md) and [data availability](DATA_AVAILABILITY.md). Place authorized inputs in data/private or configure external paths:

~~~powershell
$env:TN_DAILY_INPUT = "C:/authorized-data/lake_TN_daily_2021_2024.csv"
$env:TN_CLUSTER_INPUT_DIR = "C:/authorized-data"
$env:TN_MERGED_INPUT = "C:/authorized-data/merged_lake_data_complete.csv"
python -X utf8 scripts/01_temporal/stl.py
python -X utf8 scripts/02_clustering/kmeans.py
python -X utf8 scripts/03_drivers/xgboost_shap.py
python -X utf8 scripts/04_thresholds/gam.py
python -X utf8 scripts/05_interactions/pdp.py
~~~

Run from the repository root. Outputs are saved to analysis-specific directories under outputs; TN_OUTPUT_DIR overrides the directory for an individual run. STL also writes a decomposition table beside its input CSV. Use a writable authorized working copy. Some scripts execute on import or prompt for a target column; run them as separate programs.

## Scope of reproducibility

The supplied XGBoost script fits an overall model with a train/test split and computes seasonal summaries; it does not implement separate seasonal models with five-fold cross-validation and 100 tuning iterations. PDP uses GradientBoostingRegressor. These implementations should be distinguished from any broader methodology described in the manuscript. Candidate turning points require scientific interpretation and uncertainty assessment. Full reproduction requires the authorized study inputs.

## License

Code is released under the [MIT License](LICENSE). This license does not grant rights to third-party monitoring or environmental data.
