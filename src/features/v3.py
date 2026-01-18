import gc
import pandas as pd
from pathlib import Path
from ..data.io import read_csv_mem
from .agg import aggregate_table, recent_agg, prep_agg_for_join

def build_v3_features(X_train_v1: pd.DataFrame, X_test_v1: pd.DataFrame, dfs_processed: dict[str, pd.DataFrame], raw_dir: Path):
    # Usecols lists
    pos_use = [
        "SK_ID_PREV","SK_ID_CURR","MONTHS_BALANCE","CNT_INSTALMENT","CNT_INSTALMENT_FUTURE",
        "NAME_CONTRACT_STATUS","SK_DPD","SK_DPD_DEF"
    ]
    inst_use = [
        "SK_ID_PREV","SK_ID_CURR","NUM_INSTALMENT_NUMBER","DAYS_INSTALMENT","DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT","AMT_PAYMENT"
    ]
    cc_use = [
        "SK_ID_PREV","SK_ID_CURR","MONTHS_BALANCE","AMT_BALANCE","AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_ATM_CURRENT","AMT_DRAWINGS_CURRENT","AMT_DRAWINGS_POS_CURRENT","AMT_DRAWINGS_OTHER_CURRENT",
        "CNT_DRAWINGS_ATM_CURRENT","CNT_DRAWINGS_CURRENT","CNT_DRAWINGS_POS_CURRENT","CNT_DRAWINGS_OTHER_CURRENT"
    ]
    bb_use = ["SK_ID_BUREAU","MONTHS_BALANCE","STATUS"]

    # 1) POS
    pos_df = read_csv_mem(raw_dir / "POS_CASH_balance.csv", usecols=pos_use)
    pos_df["MONTHS_BALANCE"] = pd.to_numeric(pos_df["MONTHS_BALANCE"], errors="coerce", downcast="integer")

    pos_6 = recent_agg(pos_df, "SK_ID_CURR", "MONTHS_BALANCE", "pos", 6)
    pos_12 = recent_agg(pos_df, "SK_ID_CURR", "MONTHS_BALANCE", "pos", 12)
    gc.collect()
    del pos_df

    # 2) Installments (derived small df)
    inst_df = read_csv_mem(raw_dir / "installments_payments.csv", usecols=inst_use)

    payment_delay = (inst_df["DAYS_ENTRY_PAYMENT"] - inst_df["DAYS_INSTALMENT"]).astype("float32")
    payment_ratio = (inst_df["AMT_PAYMENT"] / (inst_df["AMT_INSTALMENT"] + 1e-6)).astype("float32")
    late_flag = (payment_delay > 0).astype("int8")
    underpay_flag = (payment_ratio < 0.9).astype("int8")

    inst2 = pd.DataFrame({
        "SK_ID_CURR": inst_df["SK_ID_CURR"].astype("int32"),
        "PAYMENT_DELAY": payment_delay,
        "PAYMENT_RATIO": payment_ratio,
        "LATE_FLAG": late_flag,
        "UNDERPAY_FLAG": underpay_flag
    })

    inst2_agg = aggregate_table(inst2, key="SK_ID_CURR", prefix="inst2")
    gc.collect()
    del inst_df, inst2, payment_delay, payment_ratio, late_flag, underpay_flag

    # 3) Credit card + utilization + recent
    cc_df = read_csv_mem(raw_dir / "credit_card_balance.csv", usecols=cc_use)
    cc_df["MONTHS_BALANCE"] = pd.to_numeric(cc_df["MONTHS_BALANCE"], errors="coerce", downcast="integer")

    if "AMT_CREDIT_LIMIT_ACTUAL" in cc_df.columns and "AMT_BALANCE" in cc_df.columns:
        cc_df["UTILIZATION"] = (cc_df["AMT_BALANCE"] / (cc_df["AMT_CREDIT_LIMIT_ACTUAL"] + 1e-6)).astype("float32")

    cc_6 = recent_agg(cc_df, "SK_ID_CURR", "MONTHS_BALANCE", "cc", 6)
    cc_12 = recent_agg(cc_df, "SK_ID_CURR", "MONTHS_BALANCE", "cc", 12)
    gc.collect()
    del cc_df

    # 4) Bureau balance delinquency flag -> bureau merge -> agg by SK_ID_CURR
    bb_df = read_csv_mem(raw_dir / "bureau_balance.csv", usecols=bb_use)
    bb_df["BB_DELINQ_FLAG"] = bb_df["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("int8")

    bb2_agg = aggregate_table(bb_df, key="SK_ID_BUREAU", prefix="bb2")
    gc.collect()
    del bb_df

    bureau_cleaned = dfs_processed["bureau"].copy()
    bureau_enriched = bureau_cleaned.merge(bb2_agg, on="SK_ID_BUREAU", how="left")
    bureau3_agg = aggregate_table(bureau_enriched, key="SK_ID_CURR", prefix="bureau3")
    gc.collect()
    del bb2_agg, bureau_enriched

    # Join onto V1 base 
    X_train_base = X_train_v1.set_index("SK_ID_CURR")
    X_test_base = X_test_v1.set_index("SK_ID_CURR")

    for agg in [bureau3_agg, inst2_agg, pos_6, pos_12, cc_6, cc_12]:
        X_train_base = X_train_base.join(prep_agg_for_join(agg), how="left")
        X_test_base = X_test_base.join(prep_agg_for_join(agg), how="left")
        gc.collect()

    X_train_v3 = X_train_base.reset_index()
    X_test_v3 = X_test_base.reset_index()

    return X_train_v3, X_test_v3

