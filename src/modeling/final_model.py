import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping

def train_final_model(X, y, seed=42, test_size=0.15):
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    spw = n_neg / max(n_pos, 1)

    model = LGBMClassifier(
        n_estimators=12000,
        learning_rate=0.02,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=-1,
        metric="auc",
        force_col_wise=True,
        max_bin=255,
        verbose=-1,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[early_stopping(300, first_metric_only=True)],
    )

    pred_va = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, pred_va)
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return model, auc, imp
