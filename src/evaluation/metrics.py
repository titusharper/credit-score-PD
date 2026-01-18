import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

def ece_score(y_true, p, n_bins=15) -> float:
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        conf = p[m].mean()
        ece += (m.sum() / len(p)) * abs(acc - conf)

    return float(ece)

def compute_metrics(y_true, oof_pred, test_pred=None, model_name="v3") -> dict:
    y_true = np.asarray(y_true).astype(int)
    oof_pred = np.asarray(oof_pred)
    out = {
        "model": model_name,
        "oof_auc": float(roc_auc_score(y_true, oof_pred)),
        "oof_pr_auc": float(average_precision_score(y_true, oof_pred)),
        "oof_brier": float(brier_score_loss(y_true, oof_pred)),
        "oof_logloss": float(log_loss(y_true, np.clip(oof_pred, 1e-6, 1 - 1e-6))),
        "oof_ece_15bins": float(ece_score(y_true, oof_pred, n_bins=15)),
        "oof_mean_pred": float(np.mean(oof_pred)),
        "target_rate": float(np.mean(y_true)),
    }

    if test_pred is not None:
        test_pred = np.asarray(test_pred)
        out.update({
            "test_mean_pred": float(np.mean(test_pred)),
            "test_min": float(np.min(test_pred)),
            "test_max": float(np.max(test_pred)),
        })

    return out

def ks_statistic(y_true, p) -> float:
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(p)}).sort_values("p")
    cum_bad = np.cumsum(df["y"]) / (df["y"].sum() + 1e-9)
    cum_good = np.cumsum(1 - df["y"]) / ((1 - df["y"]).sum() + 1e-9)
    return float(np.max(np.abs(cum_bad - cum_good)))

def gini_from_auc(auc: float) -> float:
    return float(2 * auc - 1)

