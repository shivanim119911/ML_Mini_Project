"""Flowcharts / diagrams for the report (matplotlib only)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parent / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, PURPLE, GREY = "#dce6f7", "#fde3c8", "#dcefe0", "#eadcf5", "#eeeeee"


def box(ax, x, y, w, h, text, color, fs=8, lw=0.8):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="#444444", lw=lw))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, wrap=True)


def diamond(ax, x, y, w, h, text, color=ORANGE, fs=7.5):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                         closed=True, fc=color, ec="#444444", lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, text=None, fs=7):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.9))
    if text:
        ax.text((p[0] + q[0]) / 2 + 0.1, (p[1] + q[1]) / 2 + 0.08, text, fontsize=fs)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(-0.35, h); ax.axis("off")
    return fig, ax


def fig_dataflow():
    fig, ax = canvas(7, 8.6)
    cx = 3.5
    items = [
        (8.1, "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv\n(7,500 records, 16 columns)", BLUE),
        (7.0, "EDA: missing values, duplicates,\nclass balance, distributions", ORANGE),
        (5.9, "Target creation: proxy label msk_risk\n(ergonomics risk score + noise)", ORANGE),
        (4.8, "Feature engineering: encoding, 3 derived features\n(18 model features)", ORANGE),
        (3.7, "Stratified train / test split  80% / 20%", BLUE),
    ]
    for y, t, c in items:
        box(ax, cx, y, 5.4, 0.78, t, c)
    for (y1, _, _), (y2, _, _) in zip(items, items[1:]):
        arrow(ax, (cx, y1 - 0.39), (cx, y2 + 0.39))
    box(ax, 1.7, 2.45, 3.0, 0.8, "Random Forest\n5-fold CV + RandomizedSearch", GREEN)
    box(ax, 5.3, 2.45, 3.0, 0.8, "XGBoost\n5-fold CV + RandomizedSearch", GREEN)
    arrow(ax, (cx, 3.31), (1.7, 2.85)); arrow(ax, (cx, 3.31), (5.3, 2.85))
    box(ax, cx, 1.2, 5.4, 0.8, "Select best model by test ROC-AUC\nThreshold analysis | SHAP explanations", GREEN)
    arrow(ax, (1.7, 2.05), (2.4, 1.6)); arrow(ax, (5.3, 2.05), (4.6, 1.6))
    box(ax, cx, 0.2, 5.4, 0.55, "Risk probability for a new user (HIGH / MODERATE / LOW)", PURPLE)
    arrow(ax, (cx, 0.8), (cx, 0.48))
    fig.savefig(OUT / "fig_dataflow.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_training_flow():
    fig, ax = canvas(7, 8.8)
    cx = 3.0
    box(ax, cx, 8.5, 3.2, 0.5, "START", GREY)
    box(ax, cx, 7.7, 4.4, 0.6, "Load CSV, create target msk_risk", BLUE)
    box(ax, cx, 6.8, 4.4, 0.6, "Build features (encode + derive)", BLUE)
    box(ax, cx, 5.9, 4.4, 0.6, "Stratified 80/20 train-test split", BLUE)
    box(ax, cx, 5.0, 4.4, 0.6, "FOR each model in {Random Forest, XGBoost}", ORANGE)
    box(ax, cx, 4.1, 4.4, 0.6, "RandomizedSearchCV (5-fold, ROC-AUC)", GREEN)
    box(ax, cx, 3.2, 4.4, 0.6, "Refit best parameters on full training set", GREEN)
    diamond(ax, cx, 2.2, 3.6, 1.1, "More models\nleft?")
    box(ax, cx, 0.9, 4.4, 0.6, "Evaluate on test set; pick best; SHAP", PURPLE)
    box(ax, cx, 0.2, 3.2, 0.45, "END", GREY)
    ys = [8.25, 7.4, 6.5, 5.6, 4.7, 3.8, 2.9, 1.65, 1.2]
    pts = [8.5, 7.7, 6.8, 5.9, 5.0, 4.1, 3.2]
    for a, b in zip([8.25, 7.4, 6.5, 5.6, 4.7, 3.8], [7.99, 7.1, 6.2, 5.3, 4.4, 3.5]):
        arrow(ax, (cx, a), (cx, b))
    arrow(ax, (cx, 2.9), (cx, 2.75))
    arrow(ax, (cx, 1.65), (cx, 1.2), "No")
    arrow(ax, (cx, 0.6), (cx, 0.43))
    ax.plot([cx + 1.8, 5.6, 5.6], [2.2, 2.2, 5.0], color="#333333", lw=0.9)
    arrow(ax, (5.6, 5.0), (cx + 2.2, 5.0), "Yes")
    fig.savefig(OUT / "fig_training_flow.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_architecture():
    fig, ax = canvas(10, 4.6)
    cols = [(1.2, "INPUT\nUsage data:\nscreen time, gaming,\nsocial media, sleep,\nnotifications, stress,\naddiction level", BLUE),
            (3.6, "PREPROCESSING\nEncoding of categories\nDerived features\nStratified split", ORANGE),
            (6.0, "MODELS\nRandom Forest\nXGBoost\n(tuned with CV)", GREEN),
            (8.5, "OUTPUT\nRisk probability\nRisk band\nSHAP explanation", PURPLE)]
    for x, t, c in cols:
        box(ax, x, 2.4, 2.0, 2.6, t, c, fs=8)
    for a, b in [(2.2, 2.6), (4.6, 5.0), (7.0, 7.5)]:
        arrow(ax, (a, 2.4), (b, 2.4))
    box(ax, 4.85, 0.55, 6.6, 0.7, "Explainability layer: SHAP TreeExplainer (global + local) and permutation importance", GREY, fs=8)
    arrow(ax, (6.0, 1.1), (6.0, 1.1))
    ax.annotate("", xy=(6.0, 1.0), xytext=(6.0, 1.1))
    fig.savefig(OUT / "fig_architecture.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_shap_flow():
    fig, ax = canvas(9, 2.6)
    labels = [("Trained tree\nmodel", GREEN), ("SHAP\nTreeExplainer", ORANGE),
              ("SHAP value\nper feature\nper person", BLUE), ("Global plots\n(summary, bar)", PURPLE),
              ("Local plots\n(waterfall)", PURPLE)]
    xs = [1.0, 2.9, 4.8, 7.0, 7.0]
    ys = [1.3, 1.3, 1.3, 1.95, 0.65]
    for (t, c), x, y in zip(labels, xs, ys):
        box(ax, x, y, 1.6 if x < 6 else 2.0, 0.85, t, c, fs=8)
    arrow(ax, (1.8, 1.3), (2.1, 1.3)); arrow(ax, (3.7, 1.3), (4.0, 1.3))
    arrow(ax, (5.6, 1.4), (6.0, 1.85)); arrow(ax, (5.6, 1.2), (6.0, 0.75))
    fig.savefig(OUT / "fig_shap_flow.png", dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_dataflow(); fig_training_flow(); fig_architecture(); fig_shap_flow()
    print("diagrams saved to", OUT)
