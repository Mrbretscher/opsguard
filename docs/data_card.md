# Data Card: AI4I 2020 Predictive Maintenance Dataset

## Source and License

- Dataset: AI4I 2020 Predictive Maintenance Dataset
- Repository: UCI Machine Learning Repository
- URL: <https://archive.ics.uci.edu/dataset/601/ai4i>
- DOI: `10.24432/C5HS5C`
- License: Creative Commons Attribution 4.0 International, CC BY 4.0
- File: `ai4i2020.csv`

The dataset is synthetic and intended to reflect predictive-maintenance data encountered in industrial settings. Redistribution must preserve appropriate attribution under CC BY 4.0.

## Acquisition

Fetch the dataset locally:

```powershell
.\scripts\fetch_data.ps1
```

The raw CSV is saved to `data/raw/ai4i2020.csv`. This path is ignored by Git so the full dataset is not committed.

## Schema

Expected columns:

- `UDI`
- `Product ID`
- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `Machine failure`
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

## Modeling Use

The primary target is `Machine failure`.

Included features:

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

Excluded from features:

- `UDI` / `UID`
- `Product ID`
- `Machine failure`
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

## Validation Checks

The project validates:

- Required columns are present.
- Required values are not missing.
- Target and failure-mode columns are binary.
- The primary target contains both classes.
- Numeric features are numeric.
- Feature matrices exclude identifier and target/leakage columns.

## Known Limitations

- The dataset is synthetic, not collected from a live plant.
- The minority failure class makes accuracy misleading.
- Failure-mode labels are useful for analysis but are excluded from the binary failure baseline to avoid target leakage.
- Row order is not treated as a reliable chronological split for Milestone 1.
