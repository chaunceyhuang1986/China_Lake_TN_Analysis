# Local data schemas

No real observations or derived study tables are included. Create data/private locally if needed; it is excluded from publication.

| Analysis | Input | Required or configured fields |
| --- | --- | --- |
| STL | lake_TN_daily_2021_2024.csv | year, month, day and a TN-named column; filter June 2021–December 2024 |
| Clustering | processed_water_quality.csv | Hylak_id, Year, Month, TN (renamed internally) |
| XGBoost–SHAP | merged_lake_data_complete.csv | TN, Month; available predictors from the list below |
| GAM | merged_lake_data_complete.csv | Hylak_id; TN or a configured alias; six configured predictors |
| PDP | merged_lake_data_complete.csv | TN or a detectable target alias; key predictors defined in script |

XGBoost predictor candidates: mean_pre, max_pre, min_pre, mean_tem, max_tem, min_tem, Lake surface area, HFI, PLAND-Grassland, PLAND-Cropland, PLAND-Water, PLAND-Forest, PLAND-Impervious. Month is used for seasonal grouping, not as a predictor.

GAM predictors: PLAND-Cropland, PLAND-Grassland, PLAND-Water, PLAND-Impervious, Lake surface area, HFI. Lake area is configured as m2 and converted for analysis. Check script-specific transformations and field aliases.

Clustering retains 2021–2024; default periods are 2021–2022 and 2023–2024. A lake requires at least six records and records in both periods.

Confirm TN units, climate aggregation/units, land-cover percentage scale, area units and spatial matching against the actual prepared inputs. The supplied scripts do not implement all upstream data preparation.
