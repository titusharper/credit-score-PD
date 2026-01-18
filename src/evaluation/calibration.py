import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from .metrics import ece_score

def calibration_table(y_true, p, n_bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    out = df.groupby("bin").agg(
        n=("y", "size"),
        avg_pred=("p", "mean"),
        event_rate=("y", "mean"),
        p_min=("p", "min"),
        p_max=("p", "max"),
    ).reset_index(drop=True)
    return out

def save_reliability_plot(calib_df: pd.DataFrame, out_path):
    x = calib_df["avg_pred"].values
    y = calib_df["event_rate"].values
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.plot(x, y, marker="o")
    plt.xlabel("Average predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Reliability Diagram")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def compare_and_fit_calibrator(y_true, oof_pred):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(oof_pred)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y_true)
    oof_iso = iso.predict(p)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(p.reshape(-1, 1), y_true)
    oof_platt = lr.predict_proba(p.reshape(-1, 1))[:, 1]

    def summarize(name, pred):
        return {
            "name": name,
            "auc": float(roc_auc_score(y_true, pred)),
            "brier": float(brier_score_loss(y_true, pred)),
            "logloss": float(log_loss(y_true, np.clip(pred, 1e-6, 1 - 1e-6))),
            "ece_15": float(ece_score(y_true, pred, 15)),
            "mean_pred": float(np.mean(pred)),
        }

    s_iso = summarize("isotonic", oof_iso)
    s_pl = summarize("platt", oof_platt)
    best = "isotonic" if s_iso["brier"] <= s_pl["brier"] else "platt"
  
    return best, iso, lr, s_iso, s_pl

def apply_calibrator(best_name: str, iso: IsotonicRegression, lr: LogisticRegression, pred):
    pred = np.asarray(pred)
    if best_name == "isotonic":
        return iso.predict(pred)
    return lr.predict_proba(pred.reshape(-1, 1))[:, 1]
