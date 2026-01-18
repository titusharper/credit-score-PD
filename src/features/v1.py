import pandas as pd
import numpy as np
from .agg import aggregate_table, prep_agg_for_join, sanitize_columns

def prepare_base_application(dfs: dict[str, pd.DataFrame]):
    app_train = dfs["app_train"].copy()
    app_test = dfs["app_test"].copy()

    y = app_train["TARGET"].astype(int)

    X_train = app_train.drop(columns=["TARGET"])
    X_test = app_test.copy()
  
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()

    return X_train, X_test, y

def build_v1_features(X_train: pd.DataFrame, X_test: pd.DataFrame, bureau: pd.DataFrame, prev: pd.DataFrame):
    bureau_agg = aggregate_table(bureau, key="SK_ID_CURR", prefix="bureau")
    prev_agg = aggregate_table(prev, key="SK_ID_CURR", prefix="prev")

    X_train_v1 = (
        X_train.set_index("SK_ID_CURR")
        .join(prep_agg_for_join(bureau_agg), how="left")
        .join(prep_agg_for_join(prev_agg), how="left")
        .reset_index()
    )

    X_test_v1 = (
        X_test.set_index("SK_ID_CURR")
        .join(prep_agg_for_join(bureau_agg), how="left")
        .join(prep_agg_for_join(prev_agg), how="left")
        .reset_index()
    )

    return X_train_v1, X_test_v1

def encode_train_test_ohe(train_df: pd.DataFrame, test_df: pd.DataFrame, id_col: str = "SK_ID_CURR"):
    all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    all_enc = pd.get_dummies(all_data, dummy_na=True)

    X_train_enc = all_enc.iloc[: len(train_df)].copy()
    X_test_enc = all_enc.iloc[len(train_df) :].copy()

    X_train_enc.columns = sanitize_columns(X_train_enc.columns)
    X_test_enc.columns = sanitize_columns(X_test_enc.columns)

    # Fix duplicate column names
    dup = pd.Index(X_train_enc.columns).duplicated()
    if dup.sum() > 0:
        seen = {}
        new_cols = []
        for c in X_train_enc.columns:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
        X_train_enc.columns = new_cols
        X_test_enc.columns = new_cols

    ids = X_test_enc[id_col].copy()
    Xtr = X_train_enc.drop(columns=[id_col]).astype("float32")
    Xte = X_test_enc.drop(columns=[id_col]).astype("float32")

    return Xtr, Xte, ids
