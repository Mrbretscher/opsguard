# Local Data

Raw data files generated locally by OpsGuard belong under `data/raw/`.

The AI4I dataset used by this project is:

- Dataset: AI4I 2020 Predictive Maintenance Dataset
- Repository: UCI Machine Learning Repository
- UCI dataset ID: `601`
- URL: <https://archive.ics.uci.edu/dataset/601/ai4i>
- DOI: `10.24432/C5HS5C`
- License: Creative Commons Attribution 4.0 International, CC BY 4.0
- Nature: synthetic predictive-maintenance data

Fetch and validate it locally with:

```powershell
python scripts/fetch_ai4i.py
```

This writes `data/raw/ai4i2020.csv`. The downloaded dataset is intentionally
ignored by Git and should not be committed.

The UCI API may return normalized column names and keep units in metadata. During
acquisition, OpsGuard normalizes those names to the original CSV-style canonical
schema, including `UDI`, `Air temperature [K]`, `Process temperature [K]`,
`Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`.
