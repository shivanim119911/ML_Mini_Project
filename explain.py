"""SHAP explainability for the trained tree models.

Usage:  python -m src.explain            (explains XGBoost)
        python -m src.explain --model rf (explains Random Forest)
"""
import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from .config import FIG_DIR, MODELS_DIR
from .features import load_split


def get_explanation(model, X):
    """Return a shap.Explanation for the POSITIVE (high-risk) class."""
    explanation = shap.TreeExplainer(model)(X)
    if explanation.values.ndim == 3:            # RandomForest: (n, features, classes)
        explanation = explanation[:, :, 1]
    return explanation


def main(which: str = "xgb"):
    fname, label = ("xgboost.joblib", "xgboost") if which == "xgb" else ("random_forest.joblib", "rf")
    model = joblib.load(MODELS_DIR / fname)
    _, X_te, _, _ = load_split()
    exp = get_explanation(model, X_te)

    # 1. global importance (beeswarm)
    shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.title(f"SHAP summary - {label}"); plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_summary_{label}.png", dpi=150, bbox_inches="tight"); plt.close()

    # 2. mean |SHAP| bar
    shap.plots.bar(exp, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_bar_{label}.png", dpi=150, bbox_inches="tight"); plt.close()

    # 3. dependence plot of the most important feature
    top = X_te.columns[np.abs(exp.values).mean(0).argmax()]
    shap.plots.scatter(exp[:, top], color=exp, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_dependence_{label}.png", dpi=150, bbox_inches="tight"); plt.close()

    # 4. local explanations: most-at-risk and least-at-risk test person
    totals = exp.values.sum(axis=1)
    for tag, idx in (("highest_risk", int(totals.argmax())), ("lowest_risk", int(totals.argmin()))):
        shap.plots.waterfall(exp[idx], max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"shap_waterfall_{tag}_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()
    print(f"Most influential feature: {top}\nSHAP figures saved to {FIG_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgb", "rf"], default="xgb")
    main(ap.parse_args().model)
