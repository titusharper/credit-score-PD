import gc
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping

def lgbm_cv_train(X, y, X_test=None, seed=42, n_splits=5):
    oof = np.zeros(len(y), dtype=np.float32)
    test_pred = None if X_test is None else np.zeros(X_test.shape[0], dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = LGBMClassifier(
            n_estimators=12000,
            learning_rate=0.03,
            num_leaves=64,
            min_data_in_leaf=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            max_bin=255,
            objective="binary",
            metric="auc",
            n_jobs=-1,
            random_state=seed,
            force_col_wise=True,
            verbose=-1,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[early_stopping(200, first_metric_only=True)],
        )

        pred_va = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1].astype(np.float32)
        oof[va_idx] = pred_va
        fold_auc = roc_auc_score(y_va, pred_va)
        fold_scores.append(fold_auc)

        if X_test is not None:
            test_pred += model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1].astype(np.float32) / n_splits

        del model, X_tr, X_va, y_tr, y_va, pred_va
        gc.collect()

    return oof, test_pred, fold_scores
