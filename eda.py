"""Quick exploratory plots -> reports/figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIG_DIR, TARGET
from .features import build_features
from .make_target import build_dataset


def main():
    df = build_dataset()
    X = build_features(df)
    X[TARGET] = df[TARGET]

    ax = df[TARGET].value_counts().sort_index().plot(kind="bar", color=["#4C9F70", "#D1495B"])
    ax.set_xticklabels(["Low risk (0)", "High risk (1)"], rotation=0)
    ax.set_title("Target distribution"); plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_target_distribution.png", dpi=150); plt.close()

    plt.figure(figsize=(11, 9))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Feature correlation heatmap"); plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_correlation_heatmap.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["daily_screen_time_hours", "sleep_hours", "app_opens_per_day"]):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax)
    plt.tight_layout(); plt.savefig(FIG_DIR / "eda_key_features_vs_target.png", dpi=150); plt.close()
    print("EDA figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
