import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def group_report(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for g, d in df.groupby(group_col, dropna=False):
        if d["y"].nunique() < 2 or len(d) < 500:
            continue
        rows.append({
            group_col: g,
            "n": int(len(d)),
            "target_rate": float(d["y"].mean()),
            "mean_pred": float(d["p"].mean()),
            "auc": float(roc_auc_score(d["y"], d["p"])),
            "pr_auc": float(average_precision_score(d["y"], d["p"])),
            "brier": float(brier_score_loss(d["y"], d["p"])),
        })
    if not rows:
        return pd.DataFrame(columns=[group_col, "n", "target_rate", "mean_pred", "auc", "pr_auc", "brier"])
    return pd.DataFrame(rows).sort_values("auc", ascending=False)

def build_gender_age_frames(app_train: pd.DataFrame, y, oof_pred):
    df = app_train[["CODE_GENDER", "DAYS_BIRTH"]].copy()
    df["y"] = np.asarray(y).astype(int)
    df["p"] = np.asarray(oof_pred)
    df["AGE"] = (-df["DAYS_BIRTH"] / 365.25).astype(float)
    df["AGE_BIN"] = pd.cut(df["AGE"], bins=[18, 25, 35, 45, 55, 65, 120], right=False)

    return df
