"""
Runs every experiment needed for the mini-project report and stores the results
in report/outputs/ (results.json + figures/*.png).

    python -m report.run_experiments

* If `xgboost` is installed the boosted model is XGBoost, otherwise a clearly
  labelled scikit-learn gradient-boosting stand-in is used.
* If `shap` is installed the SHAP figures are created, otherwise only
  permutation importance is produced.
`results.json` records which case applied, and build_report.py adapts its text.
"""
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.config import CV_FOLDS, SEED, TARGET
from src.features import build_features, load_split
from src.make_target import compute_risk_score
from src.make_target import build_dataset

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
C0, C1, C2 = "#4C9F70", "#D1495B", "#2E6F95"

try:
    from xgboost import XGBClassifier
    BOOST, XGB_REAL = "XGBoost", True
except ImportError:
    XGBClassifier, BOOST, XGB_REAL = None, "Gradient Boosting (scikit-learn stand-in)", False
try:
    import shap
    SHAP_REAL = True
except ImportError:
    shap, SHAP_REAL = None, False

N_ITER = 8
R = {"boost_name": BOOST, "xgboost_real": XGB_REAL, "shap_real": SHAP_REAL}


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ data + EDA
df = build_dataset(save=False)
X_tr, X_te, y_tr, y_te = load_split()
Xall = build_features(df)

R["n_rows"], R["n_raw_cols"] = int(df.shape[0]), int(df.shape[1])
R["n_features"] = int(Xall.shape[1])
R["feature_names"] = list(Xall.columns)
R["n_train"], R["n_test"] = int(len(X_tr)), int(len(X_te))
R["pos_total"], R["pos_rate"] = int(df[TARGET].sum()), float(df[TARGET].mean())
R["pos_train"], R["pos_test"] = int(y_tr.sum()), int(y_te.sum())
R["missing_addiction"] = int(df["addiction_level"].isna().sum())
R["gender_counts"] = df["gender"].value_counts().to_dict()
R["stress_counts"] = df["stress_level"].value_counts().to_dict()
R["addiction_counts"] = df["addiction_level"].fillna("None").value_counts().to_dict()
R["academic_counts"] = df["academic_work_impact"].value_counts().to_dict()
num = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours",
       "sleep_hours", "notifications_per_day", "app_opens_per_day", "weekend_screen_time"]
R["describe"] = {c: {k: float(v) for k, v in df[c].describe().loc[["mean", "std", "min", "max"]].items()}
                 for c in num}
R["means_by_class"] = {c: [float(df.loc[df[TARGET] == 0, c].mean()), float(df.loc[df[TARGET] == 1, c].mean())]
                       for c in num}
R["corr_with_target"] = {c: float(np.corrcoef(Xall[c], df[TARGET])[0, 1]) for c in Xall.columns}
over = (df.social_media_hours + df.gaming_hours + df.work_study_hours) > df.daily_screen_time_hours
R["activity_exceeds_total_pct"] = float(over.mean() * 100)

# risk rate tables
def rate_rows(label, groups):
    rows = []
    for g, mask in groups:
        rows.append([label, g, int(mask.sum()), float(df.loc[mask, TARGET].mean() * 100)])
    return rows

rt = []
d = df["daily_screen_time_hours"]
rt += rate_rows("Daily screen time", [("Up to 4 h", d <= 4), ("4-7 h", (d > 4) & (d <= 7)),
                                      ("7-10 h", (d > 7) & (d <= 10)), ("Above 10 h", d > 10)])
s = df["sleep_hours"]
rt += rate_rows("Sleep", [("Below 5 h", s < 5), ("5-6.5 h", (s >= 5) & (s < 6.5)),
                          ("6.5-8 h", (s >= 6.5) & (s < 8)), ("8 h or more", s >= 8)])
g_ = df["gaming_hours"]
rt += rate_rows("Gaming", [("Up to 1 h", g_ <= 1), ("1-2 h", (g_ > 1) & (g_ <= 2)),
                           ("2-3 h", (g_ > 2) & (g_ <= 3)), ("Above 3 h", g_ > 3)])
rt += rate_rows("Stress level", [(k, df.stress_level == k) for k in ["Low", "Medium", "High"]])
ad = df.addiction_level.fillna("None")
rt += rate_rows("Addiction level", [(k, ad == k) for k in ["None", "Mild", "Moderate", "Severe"]])
rt += rate_rows("Gender", [(k, df.gender == k) for k in ["Male", "Female", "Other"]])
R["rate_table"] = rt

# EDA figures
plt.figure(figsize=(5, 4))
sns.countplot(x=TARGET, data=df, palette=[C0, C1])
plt.xticks([0, 1], ["Low risk", "High risk"]); plt.title("Target class distribution")
save("fig_target.png")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, c in enumerate(["daily_screen_time_hours", "sleep_hours", "app_opens_per_day"]):
    sns.histplot(data=df, x=c, hue=TARGET, kde=True, ax=axes[0, i], stat="density",
                 common_norm=False, palette=[C0, C1])
    axes[0, i].set_title(f"Distribution of {c}")
    sns.boxplot(data=df, x=TARGET, y=c, ax=axes[1, i], palette=[C0, C1])
    axes[1, i].set_title(f"{c} by risk class")
save("fig_distributions.png")

fig, axes = plt.subplots(2, 3, figsize=(16, 9)); axes = axes.ravel()
cats = [("stress_level", ["Low", "Medium", "High"]), ("gender", ["Male", "Female", "Other"]),
        ("academic_work_impact", ["No", "Yes"])]
tmp = df.assign(addiction_level=df.addiction_level.fillna("None"))
cats.insert(1, ("addiction_level", ["None", "Mild", "Moderate", "Severe"]))
for ax, (c, order) in zip(axes, cats):
    rate = tmp.groupby(c)[TARGET].mean().reindex(order) * 100
    sns.barplot(x=rate.index.astype(str), y=rate.values, ax=ax, palette="viridis")
    ax.set_title(f"High-risk rate (%) by {c}"); ax.set_ylabel("High-risk rate (%)")
for k, c in enumerate(["gaming_hours", "sleep_hours"]):
    ax = axes[4 + k]
    bins = pd.qcut(df[c], 5)
    rate = df.groupby(bins, observed=True)[TARGET].mean() * 100
    sns.barplot(x=[f"{iv.left:.1f}-{iv.right:.1f}" for iv in rate.index], y=rate.values, ax=ax, palette="viridis")
    ax.set_title(f"High-risk rate (%) by {c} (quintiles)"); ax.tick_params(axis="x", rotation=20)
save("fig_categorical.png")

plt.figure(figsize=(11, 9))
cm_ = Xall.assign(**{TARGET: df[TARGET]}).corr()
sns.heatmap(cm_, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation heatmap of model features and target")
save("fig_corr.png")

# ------------------------------------------------------------------ modelling
cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
pos_w = float((y_tr == 0).sum() / (y_tr == 1).sum())
R["scale_pos_weight"] = pos_w

rf_space = {"n_estimators": randint(150, 400), "max_depth": randint(4, 20),
            "min_samples_leaf": randint(1, 15), "max_features": ["sqrt", "log2", 0.5]}
if XGB_REAL:
    boost_default = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                  scale_pos_weight=pos_w, random_state=SEED, n_jobs=-1)
    boost_space = {"n_estimators": randint(150, 600), "max_depth": randint(2, 8),
                   "learning_rate": uniform(0.01, 0.2), "subsample": uniform(0.6, 0.4),
                   "colsample_bytree": uniform(0.6, 0.4), "min_child_weight": randint(1, 10)}
else:
    boost_default = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
    boost_space = {"max_iter": randint(100, 500), "max_depth": randint(2, 8),
                   "learning_rate": uniform(0.01, 0.2), "min_samples_leaf": randint(10, 60),
                   "l2_regularization": uniform(0, 5), "max_leaf_nodes": randint(8, 40)}
rf_default = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)

models, cv_info, tuning_s = {}, {}, {}
for name, est, space in [("Random Forest", rf_default, rf_space), (BOOST, boost_default, boost_space)]:
    t = time.time()
    rs = RandomizedSearchCV(est, space, n_iter=N_ITER, scoring="roc_auc", cv=cv,
                            random_state=SEED, n_jobs=-1)
    rs.fit(X_tr, y_tr)
    models[name] = rs.best_estimator_
    cv_info[name] = {"cv_auc": float(rs.best_score_), "params": {k: (float(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else v)) for k, v in rs.best_params_.items()}}
    tuning_s[name] = time.time() - t
    print(name, cv_info[name], f"{tuning_s[name]:.0f}s", flush=True)
R["cv"], R["tuning_seconds"] = cv_info, tuning_s
R["n_iter"], R["cv_folds"] = N_ITER, CV_FOLDS
R["search_space"] = {"rf": {k: (str(v) if isinstance(v, list) else None) for k, v in rf_space.items()}}

# untuned default baselines
default_models = {"Random Forest": rf_default, BOOST: boost_default}
from sklearn.base import clone
default_fit = {k: clone(v).fit(X_tr, y_tr) for k, v in default_models.items()}


def metric_row(model, thr=0.5):
    p = model.predict_proba(X_te)[:, 1]
    pred = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, p), "pr_auc": average_precision_score(y_te, p)}, p, pred


res, probas, preds = {}, {}, {}
for n, m in models.items():
    res[n], probas[n], preds[n] = metric_row(m)
R["test_metrics"] = {k: {a: float(b) for a, b in v.items()} for k, v in res.items()}
R["default_metrics"] = {n: {a: float(b) for a, b in metric_row(m)[0].items()} for n, m in default_fit.items()}
best = max(res, key=lambda k: (res[k]["roc_auc"], res[k]["f1"]))
R["best_model"] = best

# oracle ceiling: AUC of the true generating probability (label noise ceiling)
score = compute_risk_score(df)
prob_true = 1 / (1 + np.exp(-2.5 * (score - score.quantile(0.60))))
R["oracle_auc"] = float(roc_auc_score(y_te, prob_true.loc[X_te.index]))
R["oracle_f1"] = float(f1_score(y_te, (prob_true.loc[X_te.index] >= 0.5).astype(int)))

# classification report + confusion matrices for the best model
rep = classification_report(y_te, preds[best], target_names=["Low risk", "High risk"], output_dict=True)
R["class_report"] = {k: ({a: float(b) for a, b in v.items()} if isinstance(v, dict) else float(v))
                     for k, v in rep.items()}
cmx = confusion_matrix(y_te, preds[best])
R["confusion"] = cmx.tolist()
R["confusion_all"] = {n: confusion_matrix(y_te, preds[n]).tolist() for n in models}

# threshold analysis (best model)
thr_rows = []
for thr in [0.30, 0.40, 0.50, 0.60, 0.70]:
    r, _, _ = metric_row(models[best], thr)
    thr_rows.append([thr, r["precision"], r["recall"], r["f1"], r["accuracy"]])
R["threshold_table"] = thr_rows

# ---- figures: model results
plt.figure(figsize=(6, 5))
for n in models:
    fpr, tpr, _ = roc_curve(y_te, probas[n])
    plt.plot(fpr, tpr, lw=2, label=f"{n} (AUC={res[n]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=.4); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves (test set)"); plt.legend(loc="lower right", fontsize=8)
save("fig_roc.png")

plt.figure(figsize=(6, 5))
for n in models:
    pr, rc, _ = precision_recall_curve(y_te, probas[n])
    plt.plot(rc, pr, lw=2, label=f"{n} (AP={res[n]['pr_auc']:.3f})")
plt.axhline(y_te.mean(), color="k", ls="--", alpha=.4, label="Baseline")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-recall curves (test set)")
plt.legend(fontsize=8)
save("fig_pr.png")

tbl = pd.DataFrame(res).T[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
ax = tbl.plot(kind="bar", figsize=(8, 4.5), ylim=(0, 1), colormap="viridis")
plt.xticks(rotation=0, fontsize=8); plt.title("Test-set metrics of the tuned models"); plt.legend(fontsize=7, ncol=3)
save("fig_comparison.png")

dt = pd.DataFrame({"Default": {n: R["default_metrics"][n]["roc_auc"] for n in models},
                   "Tuned": {n: res[n]["roc_auc"] for n in models}})
dt.plot(kind="bar", figsize=(6, 4), ylim=(0.5, 1), color=[C2, C0]); plt.xticks(rotation=0, fontsize=8)
plt.ylabel("Test ROC-AUC"); plt.title("Default vs tuned hyper-parameters")
save("fig_default_vs_tuned.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, n in zip(axes, models):
    ConfusionMatrixDisplay.from_predictions(y_te, preds[n], ax=ax, cmap="Blues",
                                            display_labels=["Low", "High"], colorbar=False)
    ax.set_title(n, fontsize=9)
save("fig_confusion.png")

plt.figure(figsize=(6, 4))
arr = np.array(thr_rows)
for j, lab in [(1, "Precision"), (2, "Recall"), (3, "F1")]:
    plt.plot(arr[:, 0], arr[:, j], marker="o", label=lab)
plt.xlabel("Decision threshold"); plt.ylabel("Score"); plt.title(f"Threshold trade-off ({best})")
plt.legend(); save("fig_threshold.png")

# ---- importance: built-in, permutation, SHAP
imp = {}
for n, m in models.items():
    if hasattr(m, "feature_importances_"):
        imp[n] = pd.Series(m.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
perm = {}
for n, m in models.items():
    pi = permutation_importance(m, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=-1)
    perm[n] = pd.Series(pi.importances_mean, index=X_te.columns).sort_values(ascending=False)
R["perm_importance"] = {n: {k: float(v) for k, v in s.head(12).items()} for n, s in perm.items()}
R["builtin_importance"] = {n: {k: float(v) for k, v in s.head(12).items()} for n, s in imp.items()}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, n in zip(axes, models):
    perm[n].head(10)[::-1].plot(kind="barh", ax=ax, color="teal")
    ax.set_title(f"{n}\nPermutation importance (drop in ROC-AUC)", fontsize=9)
save("fig_perm.png")
if imp:
    fig, axes = plt.subplots(1, len(imp), figsize=(6.5 * len(imp), 5), squeeze=False)
    for ax, (n, s) in zip(axes[0], imp.items()):
        s.head(10)[::-1].plot(kind="barh", ax=ax, color="teal")
        ax.set_title(f"{n}\nBuilt-in feature importance", fontsize=9)
    save("fig_builtin.png")

if SHAP_REAL:
    def expl(model, X):
        e = shap.TreeExplainer(model)(X)
        return e[:, :, 1] if e.values.ndim == 3 else e
    ex = expl(models[best], X_te)
    shap.plots.beeswarm(ex, max_display=15, show=False); save("fig_shap_summary.png")
    shap.plots.bar(ex, max_display=15, show=False); save("fig_shap_bar.png")
    top = X_te.columns[np.abs(ex.values).mean(0).argmax()]
    shap.plots.scatter(ex[:, top], color=ex, show=False); save("fig_shap_dependence.png")
    tot = ex.values.sum(1)
    shap.plots.waterfall(ex[int(tot.argmax())], max_display=12, show=False); save("fig_shap_waterfall_high.png")
    shap.plots.waterfall(ex[int(tot.argmin())], max_display=12, show=False); save("fig_shap_waterfall_low.png")
    ms = pd.Series(np.abs(ex.values).mean(0), index=X_te.columns).sort_values(ascending=False)
    R["shap_mean_abs"] = {k: float(v) for k, v in ms.head(12).items()}

# ---- designed relationship sanity check (spearman of top features vs predicted prob)
R["pred_corr"] = {c: float(pd.Series(X_te[c].values).corr(pd.Series(probas[best]), method="spearman"))
                  for c in ["daily_screen_time_hours", "sleep_hours", "gaming_hours", "stress_level",
                            "app_opens_per_day", "weekend_screen_time"]}

# ---- inference demonstrations
profiles = [
    ("High-risk heavy user", dict(age=24, gender="Male", daily_screen_time_hours=11.0, social_media_hours=4.5,
        gaming_hours=3.5, work_study_hours=3.0, sleep_hours=4.8, notifications_per_day=210,
        app_opens_per_day=160, weekend_screen_time=13.5, stress_level="High",
        academic_work_impact="Yes", addiction_level="Severe")),
    ("Low-risk light user", dict(age=28, gender="Female", daily_screen_time_hours=3.0, social_media_hours=1.0,
        gaming_hours=0.3, work_study_hours=2.0, sleep_hours=8.2, notifications_per_day=45,
        app_opens_per_day=30, weekend_screen_time=3.8, stress_level="Low",
        academic_work_impact="No", addiction_level="None")),
    ("Moderate-use student", dict(age=21, gender="Other", daily_screen_time_hours=7.5, social_media_hours=3.3,
        gaming_hours=2.0, work_study_hours=3.2, sleep_hours=6.7, notifications_per_day=134,
        app_opens_per_day=98, weekend_screen_time=9.2, stress_level="Medium",
        academic_work_impact="No", addiction_level="Moderate")),
]
R["examples"] = []
for title, row in profiles:
    p = float(models[best].predict_proba(build_features(pd.DataFrame([row])))[0, 1])
    R["examples"].append({"title": title, "inputs": row, "prob": p})

json.dump(R, open(OUT / "results.json", "w"), indent=1, default=float)
print("DONE best =", best, flush=True)
