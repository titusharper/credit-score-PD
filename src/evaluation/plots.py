import matplotlib.pyplot as plt

def save_histogram(values, out_path, title):
    plt.figure()
    plt.hist(values, bins=50)
    plt.title(title)
    plt.xlabel("Predicted PD")
    plt.ylabel("Count")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_feature_importance_top30(imp_df, out_path, title):
    top = imp_df.head(30).iloc[::-1]
    plt.figure(figsize=(8, 10))
    plt.barh(top["feature"], top["importance"])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
