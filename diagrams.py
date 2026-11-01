"""Flowcharts for the report, drawn with matplotlib (no Graphviz needed)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COL = {"blue": "#dbe7fb", "orange": "#fde3c8", "green": "#d9efdd", "purple": "#eadcf4", "grey": "#eeeeee"}


def _box(ax, cx, cy, w, h, text, color):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=COL[color], ec="#333333", lw=1))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.6, family="serif", linespacing=1.25)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.1, shrinkA=0, shrinkB=0))


def data_flow(path):
    fig, ax = plt.subplots(figsize=(7.6, 8.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 14.2); ax.axis("off")
    W, H = 5.4, 0.95
    seq = [(13.4, "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv\n(7,500 records, 16 columns)", "blue"),
           (12.05, "EDA and cleaning: missing values, duplicates, class balance,\nblank addiction_level = 'None', drop ID columns", "orange"),
           (10.7, "Proxy target creation: ergonomic risk score -> probability\n-> seeded sampling -> msk_risk (0 = low, 1 = high)", "orange"),
           (9.35, "Feature engineering: encode categories, add 3 engineered\nfeatures -> 18 model features", "orange"),
           (8.0, "Stratified train / test split (80% / 20%)\n6,000 training rows | 1,500 test rows", "blue")]
    for y, t, c in seq:
        _box(ax, 5, y, W + 1.2, H, t, c)
    for a, b in zip(seq, seq[1:]):
        _arrow(ax, 5, a[0] - H / 2, 5, b[0] + H / 2)
    _box(ax, 2.7, 6.45, 4.2, 0.95, "Model A: Random Forest\n(class_weight = balanced)", "orange")
    _box(ax, 7.3, 6.45, 4.2, 0.95, "Model B: XGBoost\n(scale_pos_weight)", "orange")
    _arrow(ax, 5, 8 - H / 2, 2.7, 6.45 + .48); _arrow(ax, 5, 8 - H / 2, 7.3, 6.45 + .48)
    _box(ax, 5, 4.9, 6.6, 0.95, "Train + tune with 5-fold stratified CV\n(RandomizedSearchCV, scoring = ROC-AUC)", "green")
    _arrow(ax, 2.7, 6.45 - .48, 5, 4.9 + .48); _arrow(ax, 7.3, 6.45 - .48, 5, 4.9 + .48)
    _box(ax, 2.7, 3.2, 4.2, 1.05, "Test evaluation: accuracy, precision,\nrecall, F1, ROC-AUC, PR-AUC", "purple")
    _box(ax, 7.3, 3.2, 4.2, 1.05, "SHAP explainability: summary,\nbar, dependence, waterfall plots", "purple")
    _arrow(ax, 5, 4.9 - .48, 2.7, 3.2 + .53); _arrow(ax, 5, 4.9 - .48, 7.3, 3.2 + .53)
    _box(ax, 5, 1.4, 6.6, 0.95, "Best model saved (joblib) -> predict.py\nreturns risk probability and risk band", "green")
    _arrow(ax, 2.7, 3.2 - .53, 5, 1.4 + .48); _arrow(ax, 7.3, 3.2 - .53, 5, 1.4 + .48)
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


def proxy_flow(path):
    fig, ax = plt.subplots(figsize=(7.6, 5.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    _box(ax, 5, 7.2, 8.4, 1.05, "Usage features: daily & weekend screen time, gaming, social media,\napp opens, notifications, sleep, age, stress level", "blue")
    _box(ax, 5, 5.5, 8.4, 1.05, "Standardise (z-scores) and combine with ergonomic weights\nlong use, gaming, frequent checks and stress raise risk; sleep lowers it", "orange")
    _box(ax, 5, 3.8, 8.4, 1.05, "risk_score -> probability = sigmoid(2.5 x (score - 60th percentile))", "orange")
    _box(ax, 5, 2.1, 8.4, 1.05, "Random draw with fixed seed (42): label = 1 if draw < probability\n(adds realistic noise, so the label is not a deterministic formula)", "orange")
    _box(ax, 5, 0.6, 5.2, 0.8, "Target column msk_risk (about 43% high risk)", "green")
    for y1, y2 in ((6.67, 6.03), (4.97, 4.33), (3.27, 2.63), (1.57, 1.0)):
        _arrow(ax, 5, y1, 5, y2)
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    from src.config import REPORTS_DIR
    d = REPORTS_DIR / "report_figures"; d.mkdir(exist_ok=True, parents=True)
    data_flow(d / "fig_dataflow.png"); proxy_flow(d / "fig_proxy_flow.png")
