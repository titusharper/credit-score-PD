import pandas as pd
from pathlib import Path
from .agg import aggregate_table, prep_agg_for_join
from .v1 import encode_train_test_ohe

def build_v2_features(X_train, X_test, dfs_processed: dict[str, pd.DataFrame], raw_dir: Path):
    inst_df = pd.read_csv(raw_dir / "installments_payments.csv")
    pos_df = pd.read_csv(raw_dir / "POS_CASH_balance.csv")
    cc_df = pd.read_csv(raw_dir / "credit_card_balance.csv")
    bb_df = pd.read_csv(raw_dir / "bureau_balance.csv")

    inst_agg = aggregate_table(inst_df, key="SK_ID_CURR", prefix="inst")
    pos_agg = aggregate_table(pos_df, key="SK_ID_CURR", prefix="pos")
    cc_agg = aggregate_table(cc_df, key="SK_ID_CURR", prefix="cc")

    bb_agg = aggregate_table(bb_df, key="SK_ID_BUREAU", prefix="bb")
    bureau_enriched = dfs_processed["bureau"].merge(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau2_agg = aggregate_table(bureau_enriched, key="SK_ID_CURR", prefix="bureau2")

    prev_agg = aggregate_table(dfs_processed["prev"], key="SK_ID_CURR", prefix="prev")

    X_train_v2 = (
        X_train.set_index("SK_ID_CURR")
        .join(prep_agg_for_join(bureau2_agg), how="left")
        .join(prep_agg_for_join(prev_agg), how="left")
        .join(prep_agg_for_join(inst_agg), how="left")
        .join(prep_agg_for_join(pos_agg), how="left")
        .join(prep_agg_for_join(cc_agg), how="left")
        .reset_index()
    )

    X_test_v2 = (
        X_test.set_index("SK_ID_CURR")
        .join(prep_agg_for_join(bureau2_agg), how="left")
        .join(prep_agg_for_join(prev_agg), how="left")
        .join(prep_agg_for_join(inst_agg), how="left")
        .join(prep_agg_for_join(pos_agg), how="left")
        .join(prep_agg_for_join(cc_agg), how="left")
        .reset_index()
    )

    return X_train_v2, X_test_v2

def encode_v2_ohe(X_train_v2, X_test_v2):
    return encode_train_test_ohe(X_train_v2, X_test_v2, id_col="SK_ID_CURR")
