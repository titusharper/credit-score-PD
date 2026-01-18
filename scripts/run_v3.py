import json
from pathlib import Path
import numpy as np
import pandas as pd
from credit_risk_pd.config import get_paths
from credit_risk_pd.data.io import load_processed_tables
from credit_risk_pd.features.v1 import prepare_base_application, build_v1_features
from credit_risk_pd.features.v3 import build_v3_features
from credit_risk_pd.features.numeric import ensure_numeric_float32
from credit_risk_pd.modeling.lgbm import lgbm_cv_train
from credit_risk_pd.evaluation.metrics import compute_metrics, ks_statistic, gini_from_auc
from credit_risk_pd.evaluation.calibration import calibration_table, save_reliability_plot, compare_and_fit_calibrator, apply_calibrator
from credit_risk_pd.evaluation.fairness import build_gender_age_frames, group_report
from credit_risk_pd.evaluation.risk import risk_capture_table, save_risk_capture_plot
from credit_risk_pd.evaluation.plots import save_histogram, plot_feature_importance_top30
from credit_risk_pd.evaluation.report import build_report_markdown

def main():
    paths = get_paths()

    dfs = load_processed_tables(paths.data_processed)
    X_train, X_test, y = prepare_base_application(dfs)

    # V1 base (needed by V3)
    X_train_v1, X_test_v1 = build_v1_features(
        X_train=X_train,
        X_test=X_test,
        bureau=dfs["bureau"],
        prev=dfs["prev"],
    )

    # V3 features
    X_train_v3, X_test_v3 = build_v3_features(
        X_train_v1=X_train_v1,
        X_test_v1=X_test_v1,
        dfs_processed=dfs,
        raw_dir=paths.data_raw,
    )

    # Numeric
    Xtr3 = ensure_numeric_float32(X_train_v3, drop_cols=("SK_ID_CURR",))
    Xte3 = ensure_numeric_float32(X_test_v3, drop_cols=("SK_ID_CURR",))
    Xte3 = Xte3.reindex(columns=Xtr3.columns)

    # Train
    oof, test_pred, fold_scores = lgbm_cv_train(Xtr3, y, X_test=Xte3, seed=42, n_splits=5)

    # IDs and submissions
    id_test = dfs["app_test"]["SK_ID_CURR"].copy()
    sub_path = paths.outputs / "submission_best_v3.csv"
    pd.DataFrame({"SK_ID_CURR": id_test.astype(int), "TARGET": test_pred}).to_csv(sub_path, index=False)

    # Metrics
    metrics = compute_metrics(y_true=y.values, oof_pred=oof, test_pred=test_pred, model_name="v3")
    ks = ks_statistic(y.values, oof)
    gini = gini_from_auc(metrics["oof_auc"])

    metrics_path = paths.outputs / "metrics_best_v3.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({**metrics, "ks": ks, "gini": gini, "fold_auc": fold_scores}, f, indent=2)

    # Calibration table + plot
    calib_df = calibration_table(y, oof, n_bins=10)
    calib_csv = paths.outputs / "calibration_table_v3.csv"
    calib_df.to_csv(calib_csv, index=False)

    calib_plot = paths.outputs / "calibration_curve_v3.png"
    save_reliability_plot(calib_df, calib_plot)

    # Optional: choose calibrator and create calibrated submission
    best_cal, iso, lr, s_iso, s_pl = compare_and_fit_calibrator(y, oof)
    test_cal = apply_calibrator(best_cal, iso, lr, test_pred)
    sub_cal_path = paths.outputs / f"submission_v3_calibrated_{best_cal}.csv"
    pd.DataFrame({"SK_ID_CURR": id_test.astype(int), "TARGET": test_cal}).to_csv(sub_cal_path, index=False)

    # Distributions
    save_histogram(oof, paths.outputs / "dist_oof_v3.png", "OOF score distribution (v3)")
    save_histogram(test_pred, paths.outputs / "dist_test_v3.png", "Test score distribution (v3)")

    # Fairness reports
    df_fair = build_gender_age_frames(dfs["app_train"], y, oof)
    gender_rep = group_report(df_fair[df_fair["CODE_GENDER"].isin(["F", "M"])], "CODE_GENDER")
    age_rep = group_report(df_fair, "AGE_BIN")
    gender_path = paths.outputs / "fairness_gender_v3.csv"
    age_path = paths.outputs / "fairness_age_v3.csv"
    gender_rep.to_csv(gender_path, index=False)
    age_rep.to_csv(age_path, index=False)

    # Risk capture
    cap_df = risk_capture_table(y, oof)
    cap_csv = paths.outputs / "risk_capture_v3.csv"
    cap_df.to_csv(cap_csv, index=False)
    cap_plot = paths.outputs / "risk_capture_v3.png"
    save_risk_capture_plot(cap_df, cap_plot, title="Bad capture rate vs Top-% scored (OOF, v3)")

    # Report
    artifacts = [
        sub_path.name,
        metrics_path.name,
        calib_csv.name,
        calib_plot.name,
        f"submission_v3_calibrated_{best_cal}.csv",
        "dist_oof_v3.png",
        "dist_test_v3.png",
        gender_path.name,
        age_path.name,
        cap_csv.name,
        cap_plot.name,
    ]
    report_md = build_report_markdown("v3", metrics, ks, gini, artifacts)
    report_path = paths.outputs / "report_summary_v3.md"
    report_path.write_text(report_md, encoding="utf-8")
    print("Done. Outputs written to:", paths.outputs)

if __name__ == "__main__":
    main()
