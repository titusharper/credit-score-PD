from pathlib import Path
import pandas as pd

def load_processed_tables(data_processed_dir: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "app_train": data_processed_dir / "application_train_cleaned.csv",
        "app_test":  data_processed_dir / "application_test_cleaned.csv",
        "bureau":    data_processed_dir / "bureau_cleaned.csv",
        "prev":      data_processed_dir / "previous_application_cleaned.csv",
    }
    dfs = {k: pd.read_csv(v) for k, v in paths.items()}
    return dfs

def read_csv_mem(path: Path, usecols=None, category_threshold: int = 200) -> pd.DataFrame:
    """
    CSV reader: downcast numerics, low-cardinality object -> category.
    """
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")

    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")  # float32

    for c in df.select_dtypes(include=["object"]).columns:
        nun = df[c].nunique(dropna=False)
        if nun <= category_threshold:
            df[c] = df[c].astype("category")

    return df

