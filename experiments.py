"""
Computes every number and figure used in the mini-project report.

    python -m report.experiments        (takes a few minutes)

* If `xgboost` and `shap` are installed they are used (the real project setup).
* If not, the script falls back - and SAYS SO in report_data.json and in the
  PDF - to scikit-learn's HistGradientBoostingClassifier (same algorithm family
  as XGBoost) and to a Monte-Carlo permutation estimator of Shapley values.
"""
import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from src.config import CV_FOLDS, REPORTS_DIR, SEARCH_ITER, SEED, TARGET, TEST_SIZE
from src.features import NUMERIC, build_features
from src.make_target import build_dataset

warnings.filterwarnings("ignore")
try:
    from xgboost import XGBClassifier
    USING_XGB = True
except ImportError:                       # pragma: no cover
    USING_XGB = False
try:
    import shap
    USING_SHAP = True
except ImportError:                       # pragma: no cover
    USING_SHAP = False

BOOST = "XGBoost" if USING_XGB else "Gradient Boosting (sklearn)"
FIG = REPORTS_DIR / "report_figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
C0, C1 = "#4C9F70", "#D1495B"


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------- EDA
def eda(df, X):
    out = {}
    n = len(df)
    out["n_rows"], out["n_cols"] = n, df.shape[1] - 1          # without target
    out["n_high"] = int(df[TARGET].sum())
    out["share_high"] = float(df[TARGET].mean())
    out["missing_addiction"] = int(df["addiction_level"].isna().sum())
    out["duplicates"] = int(df.duplicated().sum())
    out["cat_counts"] = {c: df[c].fillna("None").value_counts().to_dict()
                         for c in ["gender", "stress_level", "academic_work_impact", "addiction_level"]}
    out["num_range"] = {c: [float(df[c].min()), float(df[c].max()), float(df[c].mean())] for c in NUMERIC}
    out["group_means"] = {c: [float(df.loc[df[TARGET] == 0, c].mean()), float(df.loc[df[TARGET] == 1, c].mean())]
                          for c in NUMERIC}

    rows = []
    def add(factor, series):
        g = df.groupby(series, observed=True)[TARGET].agg(["size", "mean"])
        for k, r in g.iterrows():
            rows.append([factor, str(k), int(r["size"]), float(r["mean"] * 100)])
    add("Daily screen time (h)", pd.cut(df["daily_screen_time_hours"], [0, 4, 6, 8, 10, 12.01],
                                         labels=["4 or below", "4-6", "6-8", "8-10", "Above 10"]))
    add("Sleep (h)", pd.cut(df["sleep_hours"], [0, 5, 6, 7, 8, 9.01],
                             labels=["5 or below", "5-6", "6-7", "7-8", "Above 8"]))
    add("Stress level", df["stress_level"].astype(pd.CategoricalDtype(["Low", "Medium", "High"], ordered=True)))
    add("Addiction level", df["addiction_level"].fillna("None").astype(
        pd.CategoricalDtype(["None", "Mild", "Moderate", "Severe"], ordered=True)))
    add("Gender", df["gender"])
    add("Academic/work impact", df["academic_work_impact"])
    out["rate_table"] = rows

    # --- figures
    ax = df[TARGET].value_counts().sort_index().plot(kind="bar", color=[C0, C1], figsize=(5.2, 4))
    ax.set_xticklabels(["Low risk", "High risk"], rotation=0); ax.set_ylabel("count")
    ax.set_xlabel(TARGET); save("fig_target.png")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for i, c in enumerate(["daily_screen_time_hours", "sleep_hours", "app_opens_per_day"]):
        sns.histplot(data=df, x=c, hue=TARGET, kde=True, stat="density", common_norm=False,
                     ax=axes[0, i], palette=[C0, C1])
        sns.boxplot(data=df, x=TARGET, y=c, ax=axes[1, i], palette=[C0, C1])
    save("fig_numeric.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8)); axes = axes.ravel()
    for i, c in enumerate(["gender", "stress_level", "addiction_level", "academic_work_impact"]):
        s = df[c].fillna("None")
        rate = df.groupby(s)[TARGET].mean() * 100
        if c == "stress_level": rate = rate.reindex(["Low", "Medium", "High"])
        if c == "addiction_level": rate = rate.reindex(["None", "Mild", "Moderate", "Severe"])
        sns.barplot(x=rate.index.astype(str), y=rate.values, ax=axes[i], palette="viridis")
        axes[i].set_ylabel("High-risk rate (%)"); axes[i].set_xlabel(c)
    b = pd.cut(df["daily_screen_time_hours"], [0, 4, 6, 8, 10, 12.01],
               labels=["<=4", "4-6", "6-8", "8-10", ">10"])
    r = df.groupby(b, observed=True)[TARGET].mean() * 100
    sns.barplot(x=r.index.astype(str), y=r.values, ax=axes[4], palette="viridis")
    axes[4].set_xlabel("daily screen time (h)"); axes[4].set_ylabel("High-risk rate (%)")
    b = pd.cut(df["sleep_hours"], [0, 5, 6, 7, 8, 9.01], labels=["<=5", "5-6", "6-7", "7-8", ">8"])
    r = df.groupby(b, observed=True)[TARGET].mean() * 100
    sns.barplot(x=r.index.astype(str), y=r.values, ax=axes[5], palette="viridis")
    axes[5].set_xlabel("sleep (h)"); axes[5].set_ylabel("High-risk rate (%)")
    save("fig_categorical.png")

    cols = NUMERIC + ["stress_level", "addiction_level"]
    cm = X[cols].copy(); cm[TARGET] = df[TARGET]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
    save("fig_corr.png")
    out["corr_target"] = cm.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3).to_dict()
    return out


# ------------------------------------------------------------- modelling
def make_models(pos_weight):
    rf = (RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
          {"n_estimators": randint(200, 600), "max_depth": randint(4, 20),
           "min_samples_leaf": randint(1, 15), "max_features": ["sqrt", "log2", 0.5]})
    if USING_XGB:
        bo = (XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                            scale_pos_weight=pos_weight, random_state=SEED, n_jobs=-1),
              {"n_estimators": randint(150, 600), "max_depth": randint(2, 8),
               "learning_rate": uniform(0.01, 0.2), "subsample": uniform(0.6, 0.4),
               "colsample_bytree": uniform(0.6, 0.4), "min_child_weight": randint(1, 10)})
    else:
        bo = (HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
              {"max_iter": randint(150, 600), "max_depth": randint(2, 8),
               "learning_rate": uniform(0.01, 0.2), "l2_regularization": uniform(0, 5),
               "min_samples_leaf": randint(10, 80), "max_features": uniform(0.6, 0.4)})
    return {"Random Forest": rf, BOOST: bo}


def metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# ------------------------------------------------------------- Shapley
def mc_shapley(model, Xs, Xbg, n_perm=40, seed=0):
    """Monte-Carlo permutation estimate of interventional Shapley values
    (probability units). Used only when the `shap` package is unavailable."""
    rng = np.random.default_rng(seed)
    A, B, cols = Xs.values.astype(float), Xbg.values.astype(float), list(Xs.columns)
    n, f = A.shape
    vals = np.zeros((n, f))
    for i in range(n):
        batch = np.empty((n_perm, f + 1, f))
        perms = [rng.permutation(f) for _ in range(n_perm)]
        for p, perm in enumerate(perms):
            z = B[rng.integers(len(B))]
            M = np.tile(z, (f + 1, 1))
            for k in range(f):
                M[k + 1:, perm[k]] = A[i, perm[k]]
            batch[p] = M
        pr = model.predict_proba(pd.DataFrame(batch.reshape(-1, f), columns=cols))[:, 1].reshape(n_perm, f + 1)
        for p, perm in enumerate(perms):
            vals[i, perm] += np.diff(pr[p])
    base = float(model.predict_proba(Xbg)[:, 1].mean())
    return vals / n_perm, base


def shap_values(model, Xs, Xbg):
    if USING_SHAP:
        sv = shap.TreeExplainer(model).shap_values(Xs)
        sv = sv[1] if isinstance(sv, list) else sv
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        base = float(np.ravel(shap.TreeExplainer(model).expected_value)[-1])
        return sv, base, "SHAP value (log-odds)"
    sv, base = mc_shapley(model, Xs, Xbg)
    return sv, base, "Shapley value (change in probability)"


def plot_beeswarm(sv, Xs, xlabel, fname, title):
    order = np.argsort(-np.abs(sv).mean(0))[:12][::-1]
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(8, 6))
    for row, j in enumerate(order):
        v = Xs.iloc[:, j].rank(pct=True).values
        ax.scatter(sv[:, j], row + rng.normal(0, 0.09, len(v)), c=v, cmap="coolwarm", s=9, alpha=.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(Xs.columns[order])
    ax.axvline(0, color="grey", lw=.8); ax.set_xlabel(xlabel); ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap="coolwarm"); sm.set_array([0, 1])
    cb = plt.colorbar(sm, ax=ax, pad=.02); cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"])
    cb.set_label("Feature value"); save(fname)


def plot_bar(imp, fname, title, xlabel, top=12, color="#1f6f8b"):
    s = imp.sort_values().tail(top)
    s.plot(kind="barh", figsize=(7, 5), color=color); plt.xlabel(xlabel); plt.title(title); save(fname)


def plot_waterfall(contrib, base, fname, title, top=8):
    s = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    head, rest = s.iloc[:top], s.iloc[top:].sum()
    items = list(head.items()) + ([(f"{len(s) - top} other features", rest)] if len(s) > top else [])
    items = items[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    cur = base
    for i, (name, v) in enumerate(items):
        ax.barh(i, v, left=cur, color=C1 if v > 0 else "#2a6f97")
        ax.text(cur + v + (0.003 if v > 0 else -0.003), i, f"{v:+.3f}", va="center",
                ha="left" if v > 0 else "right", fontsize=8)
        cur += v
    ax.set_yticks(range(len(items))); ax.set_yticklabels([n for n, _ in items])
    ax.axvline(base, color="grey", ls="--", lw=.8)
    ax.set_xlabel("Model output (starts at the average, ends at this person's score)")
    ax.set_title(title); save(fname)


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    df = build_dataset(save=True)
    X, y = build_features(df), df[TARGET]
    R = {"using_xgboost": USING_XGB, "using_shap": USING_SHAP, "boost_name": BOOST, "log": []}
    R["eda"] = eda(df, X)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    R["split"] = {"train": list(Xtr.shape), "test": list(Xte.shape),
                  "train_rate": float(ytr.mean()), "test_rate": float(yte.mean()),
                  "train_pos": int(ytr.sum()), "test_pos": int(yte.sum()),
                  "pos_weight": float((ytr == 0).sum() / (ytr == 1).sum())}
    R["features"] = list(X.columns)

    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    fitted, probas, res, tuning = {}, {}, {}, {}
    for name, (est, space) in make_models(R["split"]["pos_weight"]).items():
        s = RandomizedSearchCV(est, space, n_iter=SEARCH_ITER, scoring="roc_auc", cv=cv,
                               random_state=SEED, n_jobs=1).fit(Xtr, ytr)
        fitted[name] = s.best_estimator_
        tuning[name] = {"cv_auc": float(s.best_score_),
                        "params": {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else
                                       (int(v) if isinstance(v, (int, np.integer)) else v))
                                   for k, v in s.best_params_.items()}}
        probas[name] = s.best_estimator_.predict_proba(Xte)[:, 1]
        res[name] = metrics(yte, probas[name])
        R["log"].append(f"{name:28s} | CV ROC-AUC = {s.best_score_:.4f}")
        R["log"].append(f"  best params: {tuning[name]['params']}")
        print(R["log"][-2], flush=True)
    R["tuning"], R["results"] = tuning, res
    names = list(fitted)
    best = max(names, key=lambda n: res[n]["roc_auc"])
    R["best"] = best

    # threshold sensitivity for the best model
    R["thresholds"] = {str(t): metrics(yte, probas[best], t) for t in (0.3, 0.4, 0.5, 0.6, 0.7)}
    pred = (probas[best] >= .5).astype(int)
    rep = {}
    for lab, nm in ((0, "Low risk"), (1, "High risk")):
        tp = ((pred == lab) & (yte == lab)).sum()
        p = tp / max((pred == lab).sum(), 1); r = tp / (yte == lab).sum()
        rep[nm] = [float(p), float(r), float(2 * p * r / (p + r + 1e-12)), int((yte == lab).sum())]
    R["class_report"] = rep

    # --- evaluation figures
    plt.figure(figsize=(6, 5))
    for n in names:
        fpr, tpr, _ = roc_curve(yte, probas[n]); plt.plot(fpr, tpr, label=f"{n} (AUC={res[n]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.legend(loc="lower right"); save("fig_roc.png")
    plt.figure(figsize=(6, 5))
    for n in names:
        p, r, _ = precision_recall_curve(yte, probas[n]); plt.plot(r, p, label=f"{n} (AP={res[n]['pr_auc']:.3f})")
    plt.axhline(yte.mean(), color="k", ls="--", alpha=.4, label="Baseline")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); save("fig_pr.png")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, n in zip(axes, names):
        cmx = np.array([[res[n]["tn"], res[n]["fp"]], [res[n]["fn"], res[n]["tp"]]])
        sns.heatmap(cmx, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Low", "High"], yticklabels=["Low", "High"])
        ax.set_title(n); ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    save("fig_cm.png")
    pd.DataFrame(res).T[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].plot(
        kind="bar", figsize=(8, 4.5), ylim=(0, 1)); plt.xticks(rotation=0); plt.legend(loc="lower right", ncol=3)
    save("fig_compare.png")

    # --- importance
    rf = fitted["Random Forest"]; bo = fitted[BOOST]
    imp_rf = pd.Series(rf.feature_importances_, index=X.columns)
    if USING_XGB:
        imp_bo, bo_lab = pd.Series(bo.feature_importances_, index=X.columns), "XGBoost built-in importance (gain)"
    else:
        pi = permutation_importance(bo, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
        imp_bo, bo_lab = pd.Series(pi.importances_mean, index=X.columns), "Permutation importance (drop in ROC-AUC)"
    plot_bar(imp_rf, "fig_imp_rf.png", "Random Forest", "Impurity-based importance")
    plot_bar(imp_bo, "fig_imp_boost.png", BOOST, bo_lab, color="#c9732b")
    R["imp_rf"] = imp_rf.sort_values(ascending=False).head(8).round(4).to_dict()
    R["imp_boost"] = imp_bo.sort_values(ascending=False).head(8).round(4).to_dict()
    R["imp_boost_label"] = bo_lab

    # --- SHAP / Shapley on the boosting model
    rs = np.random.default_rng(SEED)
    Xs = Xte.iloc[rs.choice(len(Xte), 300, replace=False)]
    Xbg = Xtr.iloc[rs.choice(len(Xtr), 100, replace=False)]
    sv, base, xl = shap_values(bo, Xs, Xbg)
    tag = "SHAP" if USING_SHAP else "Shapley (Monte-Carlo estimate)"
    plot_beeswarm(sv, Xs, xl, "fig_shap_summary.png", f"{tag} summary - {BOOST}")
    mean_abs = pd.Series(np.abs(sv).mean(0), index=X.columns)
    plot_bar(mean_abs, "fig_shap_bar.png", f"Mean |{tag}| - {BOOST}", f"mean |{xl.split(' (')[0]}|", color="#7a3e9d")
    top = list(mean_abs.sort_values(ascending=False).index)
    R["shap_top"] = mean_abs.sort_values(ascending=False).head(8).round(4).to_dict()
    j, k = X.columns.get_loc(top[0]), X.columns.get_loc(top[1])
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(Xs.iloc[:, j], sv[:, j], c=Xs.iloc[:, k], cmap="coolwarm", s=12)
    plt.colorbar(sc, label=top[1]); plt.xlabel(top[0]); plt.ylabel(xl)
    plt.title(f"Dependence plot: {top[0]}"); save("fig_shap_dependence.png")
    tot = sv.sum(1)
    for tagname, idx in (("high", int(tot.argmax())), ("low", int(tot.argmin()))):
        c = pd.Series(sv[idx], index=X.columns)
        plot_waterfall(c, base, f"fig_shap_waterfall_{tagname}.png",
                       f"Explanation of the {'highest' if tagname == 'high' else 'lowest'}-risk test person",
                       top=8)
    R["shap_label"] = xl; R["shap_base"] = base

    # --- example predictions
    ex = {
        "Example 1: Heavy smartphone user": dict(age=22, gender="Male", daily_screen_time_hours=10.5,
            social_media_hours=5.0, gaming_hours=3.5, work_study_hours=3.0, sleep_hours=5.0,
            notifications_per_day=220, app_opens_per_day=160, weekend_screen_time=13.0,
            stress_level="High", academic_work_impact="Yes", addiction_level="Severe"),
        "Example 2: Light smartphone user": dict(age=30, gender="Female", daily_screen_time_hours=3.0,
            social_media_hours=1.0, gaming_hours=0.5, work_study_hours=4.5, sleep_hours=8.0,
            notifications_per_day=50, app_opens_per_day=40, weekend_screen_time=4.0,
            stress_level="Low", academic_work_impact="No", addiction_level="None"),
        "Example 3: Average smartphone user": dict(age=25, gender="Male", daily_screen_time_hours=7.5,
            social_media_hours=3.3, gaming_hours=2.0, work_study_hours=3.2, sleep_hours=6.7,
            notifications_per_day=134, app_opens_per_day=98, weekend_screen_time=9.2,
            stress_level="Medium", academic_work_impact="No", addiction_level="Moderate"),
    }
    R["examples"] = []
    for title, d in ex.items():
        row = build_features(pd.DataFrame([d]))
        p = float(bo.predict_proba(row)[0, 1]); prf = float(rf.predict_proba(row)[0, 1])
        s1, _, _ = shap_values(bo, row, Xbg) if USING_SHAP else (mc_shapley(bo, row, Xbg, n_perm=200)[0], 0, 0)
        c = pd.Series(np.ravel(s1), index=X.columns)
        topc = c.reindex(c.abs().sort_values(ascending=False).index).head(3)
        R["examples"].append({"title": title, "input": d, "p_boost": p, "p_rf": prf,
                              "label": "HIGH RISK" if p >= .5 else "LOW RISK",
                              "drivers": [[k_, float(v_)] for k_, v_ in topc.items()]})

    R["runtime_sec"] = round(time.time() - t0)
    (REPORTS_DIR / "report_data.json").write_text(json.dumps(R, indent=1, default=float))
    print("done in", R["runtime_sec"], "s; best =", best)


if __name__ == "__main__":
    main()
