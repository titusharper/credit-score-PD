import numpy as np
import pandas as pd
from .agg import sanitize_columns

def ensure_numeric_float32(df: pd.DataFrame, drop_cols=("SK_ID_CURR",)) -> pd.DataFrame:
    out = df.drop(columns=list(drop_cols), errors="ignore").copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
        else:
            out[c] = out[c].astype("category").cat.codes.astype("float32")

    out.columns = sanitize_columns(out.columns)
    return out
