import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def risk_capture_table(y_true, oof_pred, tops=(1, 2, 5, 10, 15, 20, 30, 40, 50)) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(oof_pred)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    total_bad = df["y"].sum()
    total_n = len(df)
    rows = []
    for k in tops:
        m = int(np.ceil(total_n * (k / 100)))
        top = df.iloc[:m]
        bad_captured = top["y"].sum()
        capture_rate = bad_captured / total_bad if total_bad > 0 else np.nan
        precision_at_k = top["y"].mean()
        rows.append([k, m, int(bad_captured), float(capture_rate), float(precision_at_k)])

    return pd.DataFrame(rows, columns=["top_%", "n_customers", "bad_captured", "bad_capture_rate", "precision_in_top"])

def save_risk_capture_plot(cap_table: pd.DataFrame, out_path, title="Bad capture rate vs Top-% scored"):
    plt.figure()
    plt.plot(cap_table["top_%"], cap_table["bad_capture_rate"], marker="o")
    plt.title(title)
    plt.xlabel("Top-% highest risk")
    plt.ylabel("Captured share of defaults")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
