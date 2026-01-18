import re
import pandas as pd
import numpy as np

def sanitize_columns(cols) -> list[str]:
    out = []
    for c in cols:
        c = str(c)
        c = re.sub(r"[^0-9a-zA-Z_]+", "_", c)
        c = re.sub(r"_{2,}", "_", c).strip("_")
        out.append(c)
    return out

def aggregate_table(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != key]
    num_cols = [c for c in df.columns if c != key and c not in cat_cols]
    agg_num = df[[key] + num_cols].groupby(key).agg(["count", "mean", "min", "max", "sum"])
    agg_num.columns = [f"{prefix}__{c}__{stat}" for c, stat in agg_num.columns]

    if len(cat_cols) > 0:
        ohe = pd.get_dummies(df[[key] + cat_cols], columns=cat_cols, dummy_na=True)
        agg_cat = ohe.groupby(key).mean()
        agg_cat.columns = [f"{prefix}__{c}__mean" for c in agg_cat.columns]
        out = agg_num.join(agg_cat, how="left")
    else:
        out = agg_num

    return out.reset_index()

def recent_agg(df: pd.DataFrame, key: str, time_col: str, prefix: str, window: int) -> pd.DataFrame:
    """
    window=6 means last 6 months-like filter: time_col >= -6
    """
    d = df.copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d = d[d[time_col] >= -window]
    return aggregate_table(d, key=key, prefix=f"{prefix}_last{window}")

def prep_agg_for_join(df: pd.DataFrame, key: str = "SK_ID_CURR") -> pd.DataFrame:
    out = df.copy()
    if key in out.columns:
        out = out.set_index(key)

    num_cols = out.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        out[num_cols] = out[num_cols].astype("float32")

    out = out.replace([np.inf, -np.inf], np.nan)
    return out
