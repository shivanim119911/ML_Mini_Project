"""Train + tune Random Forest and XGBoost, evaluate, save the best model."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from .config import CV_FOLDS, FIG_DIR, MODELS_DIR, REPORTS_DIR, SEARCH_ITER, SEED
from .features import load_split


def tune(name, estimator, space, X_tr, y_tr):
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(estimator, space, n_iter=SEARCH_ITER, scoring="roc_auc",
                                cv=cv, random_state=SEED, n_jobs=-1, verbose=0)
    search.fit(X_tr, y_tr)
    print(f"[{name}] best CV ROC-AUC = {search.best_score_:.4f}  params = {search.best_params_}")
    return search


def metrics(model, X_te, y_te):
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
    }, proba, pred


def main():
    X_tr, X_te, y_tr, y_te = load_split()
    print(f"Train {X_tr.shape}  Test {X_te.shape}")
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

    rf_search = tune(
        "RandomForest",
        RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        {"n_estimators": randint(200, 600), "max_depth": randint(4, 20),
         "min_samples_leaf": randint(1, 15), "max_features": ["sqrt", "log2", 0.5]},
        X_tr, y_tr)
    xgb_search = tune(
        "XGBoost",
        XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                      scale_pos_weight=pos_weight, random_state=SEED, n_jobs=-1),
        {"n_estimators": randint(150, 600), "max_depth": randint(2, 8),
         "learning_rate": uniform(0.01, 0.2), "subsample": uniform(0.6, 0.4),
         "colsample_bytree": uniform(0.6, 0.4), "min_child_weight": randint(1, 10)},
        X_tr, y_tr)

    models = {"Random Forest": rf_search.best_estimator_, "XGBoost": xgb_search.best_estimator_}
    results, probas, preds = {}, {}, {}
    for name, m in models.items():
        results[name], probas[name], preds[name] = metrics(m, X_te, y_te)

    table = pd.DataFrame(results).T.round(4)
    print("\n=== Test-set performance ===\n", table)

    best_name = table["roc_auc"].idxmax()
    print(f"\nBest model by ROC-AUC: {best_name}")

    # --- save ---
    joblib.dump(models["Random Forest"], MODELS_DIR / "random_forest.joblib")
    joblib.dump(models["XGBoost"], MODELS_DIR / "xgboost.joblib")
    joblib.dump(models[best_name], MODELS_DIR / "best_model.joblib")
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(list(X_tr.columns)))
    REPORTS_DIR.mkdir(exist_ok=True)
    table.to_csv(REPORTS_DIR / "metrics.csv")
    (REPORTS_DIR / "best_params.json").write_text(json.dumps(
        {"random_forest": rf_search.best_params_, "xgboost": xgb_search.best_params_,
         "best_model": best_name}, indent=2, default=float))

    # --- plots ---
    plt.figure(figsize=(6, 5))
    for name in models:
        fpr, tpr, _ = roc_curve(y_te, probas[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves (test set)"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curves.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, models):
        ConfusionMatrixDisplay.from_predictions(y_te, preds[name], ax=ax, cmap="Blues",
                                                display_labels=["Low", "High"], colorbar=False)
        ax.set_title(name)
    plt.tight_layout(); plt.savefig(FIG_DIR / "confusion_matrices.png", dpi=150); plt.close()

    table.plot(kind="bar", figsize=(8, 4.5), ylim=(0, 1)); plt.xticks(rotation=0)
    plt.title("Model comparison"); plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison.png", dpi=150); plt.close()

    for name, m in models.items():
        imp = pd.Series(m.feature_importances_, index=X_tr.columns).sort_values().tail(12)
        imp.plot(kind="barh", figsize=(7, 5), title=f"{name} - built-in feature importance")
        plt.tight_layout()
        fname = name.lower().replace(" ", "_")
        plt.savefig(FIG_DIR / f"feature_importance_{fname}.png", dpi=150)
        plt.close()
    print("Saved models, metrics and figures.")


if __name__ == "__main__":
    main()
