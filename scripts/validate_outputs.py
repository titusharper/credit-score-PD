from credit_risk_pd.config import get_paths
import pandas as pd

def main():
    paths = get_paths()
    files = [
        "submission_best_v3.csv",
        "metrics_best_v3.json",
        "calibration_curve_v3.png",
        "calibration_table_v3.csv",
        "fairness_gender_v3.csv",
        "risk_capture_v3.csv",
        "report_summary_v3.md",
    ]

    missing = [f for f in files if not (paths.outputs / f).exists()]
    print("Missing:", missing)
    sub = pd.read_csv(paths.outputs / "submission_best_v3.csv")
    print("Submission columns:", sub.columns.tolist())
    print("Submission rows:", len(sub))
    print(sub["TARGET"].describe())

if __name__ == "__main__":
    main()
