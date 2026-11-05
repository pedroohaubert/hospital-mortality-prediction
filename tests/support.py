import numpy as np
import pandas as pd

from desfecho.schema import AGE_COL, BINARY_COLS, CAT_COLS, ID_COL, TARGET_COL


def synthetic_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(0, 96, n)
    columns = {
        ID_COL: [f"A{i:06d}" for i in range(n)],
        AGE_COL: age,
        CAT_COLS[0]: rng.integers(1, 8, n),
        CAT_COLS[1]: rng.integers(1, 5, n),
    }
    binaries = {}
    for col in BINARY_COLS:
        binaries[col] = rng.integers(0, 2, n)
        columns[col] = binaries[col]
    logit = -1.5 + 0.04 * age + 0.9 * binaries["comorbidade5"] + 0.4 * binaries["dadoclin6"]
    prob = 1 / (1 + np.exp(-logit))
    columns[TARGET_COL] = (rng.random(n) < prob).astype(int)
    return pd.DataFrame(columns)
