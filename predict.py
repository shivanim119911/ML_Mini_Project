"""Predict musculoskeletal-risk for one person from the command line.

Example:
  python -m src.predict --age 22 --daily_screen_time_hours 9.5 --sleep_hours 5 \
      --gaming_hours 3 --stress_level High
"""
import argparse

import joblib
import pandas as pd

from .config import MODELS_DIR
from .features import build_features

DEFAULTS = dict(age=25, gender="Male", daily_screen_time_hours=7.5, social_media_hours=3.3,
                gaming_hours=2.0, work_study_hours=3.2, sleep_hours=6.7,
                notifications_per_day=134, app_opens_per_day=98, weekend_screen_time=9.2,
                stress_level="Medium", academic_work_impact="No", addiction_level="Moderate")


def band(p):
    return "LOW" if p < 0.33 else "MODERATE" if p < 0.66 else "HIGH"


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = vars(ap.parse_args())

    row = pd.DataFrame([args])
    X = build_features(row)
    model = joblib.load(MODELS_DIR / "xgboost.joblib")
    p = float(model.predict_proba(X)[0, 1])
    print(f"\nPredicted probability of high neck/MSK risk: {p:.1%}  ->  {band(p)}")

    try:                                    # optional: top drivers via SHAP
        import shap
        sv = shap.TreeExplainer(model)(X).values[0]
        top = pd.Series(sv, index=X.columns).sort_values(key=abs, ascending=False).head(5)
        print("\nTop factors (SHAP, + raises risk / - lowers risk):")
        for f, v in top.items():
            print(f"  {f:28s} {v:+.3f}")
    except Exception as e:                  # pragma: no cover
        print("(SHAP explanation skipped:", e, ")")
    print("\nNote: educational model trained on a proxy label - not a medical diagnosis.")


if __name__ == "__main__":
    main()
