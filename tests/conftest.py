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
