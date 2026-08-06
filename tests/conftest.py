import pandas as pd
import pytest


@pytest.fixture()
def valid_ai4i_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UDI": [1, 2, 3, 4, 5, 6],
            "Product ID": ["L47181", "M14860", "H29424", "L47184", "M14864", "H29428"],
            "Type": ["L", "M", "H", "L", "M", "H"],
            "Air temperature [K]": [298.1, 298.2, 299.3, 300.1, 301.0, 302.4],
            "Process temperature [K]": [308.6, 308.7, 309.1, 310.2, 311.1, 312.0],
            "Rotational speed [rpm]": [1551, 1408, 1498, 1600, 1300, 1200],
            "Torque [Nm]": [42.8, 46.3, 49.4, 40.1, 55.2, 60.0],
            "Tool wear [min]": [0, 3, 5, 10, 15, 20],
            "Machine failure": [0, 0, 0, 1, 0, 1],
            "TWF": [0, 0, 0, 0, 0, 1],
            "HDF": [0, 0, 0, 1, 0, 0],
            "PWF": [0, 0, 0, 0, 0, 0],
            "OSF": [0, 0, 0, 0, 0, 0],
            "RNF": [0, 0, 0, 0, 0, 0],
        }
    )


@pytest.fixture()
def baseline_ai4i_frame() -> pd.DataFrame:
    rows = 40
    target = [1 if index in {5, 13, 27, 34} else 0 for index in range(rows)]
    return pd.DataFrame(
        {
            "UDI": list(range(1, rows + 1)),
            "Product ID": [f"L{47181 + index}" for index in range(rows)],
            "Type": [["L", "M", "H"][index % 3] for index in range(rows)],
            "Air temperature [K]": [298.0 + (index % 8) * 0.3 for index in range(rows)],
            "Process temperature [K]": [
                308.0 + (index % 7) * 0.4 for index in range(rows)
            ],
            "Rotational speed [rpm]": [
                1200 + (index * 37) % 500 for index in range(rows)
            ],
            "Torque [Nm]": [35.0 + (index * 1.7) % 35 for index in range(rows)],
            "Tool wear [min]": [(index * 6) % 240 for index in range(rows)],
            "Machine failure": target,
            "TWF": [1 if index == 5 else 0 for index in range(rows)],
            "HDF": [1 if index == 13 else 0 for index in range(rows)],
            "PWF": [1 if index == 27 else 0 for index in range(rows)],
            "OSF": [1 if index == 34 else 0 for index in range(rows)],
            "RNF": [0 for _ in range(rows)],
        }
    )
