"""
Creates the target variable `msk_risk`.

IMPORTANT
---------
The provided dataset contains smartphone-usage and addiction information but
NO neck-pain / musculoskeletal column. To still build the pipeline, we create a
*proxy* label from an ergonomics-inspired risk score:

  * long daily + weekend screen time   -> prolonged "text-neck" posture
  * gaming / social media hours        -> static, head-down, repetitive use
  * frequent app opens & notifications -> repeated neck flexion / thumb use
  * short sleep                        -> poor muscle recovery
  * high stress                        -> increased muscle tension
  * age                                -> small age effect

The score is turned into a probability and the label is SAMPLED from it
(seeded), so the label is noisy rather than a deterministic formula.

This is SYNTHETIC. Results show that the ML pipeline works, not that the
relationships hold in real people. Replace with a real survey label by setting
`REAL_TARGET_COLUMN` in config.py.
"""
import numpy as np
import pandas as pd

from .config import PROCESSED_CSV, RAW_CSV, REAL_TARGET_COLUMN, SEED, TARGET

STRESS_WEIGHT = {"Low": 0.0, "Medium": 0.35, "High": 0.70}


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    return (
        0.35 * _z(df["daily_screen_time_hours"])
        + 0.15 * _z(df["weekend_screen_time"])
        + 0.15 * _z(df["gaming_hours"])
        + 0.10 * _z(df["social_media_hours"])
        + 0.10 * _z(df["app_opens_per_day"])
        + 0.05 * _z(df["notifications_per_day"])
        - 0.20 * _z(df["sleep_hours"])
        + 0.05 * _z(df["age"])
        + df["stress_level"].map(STRESS_WEIGHT)
    )


def risk_probability(df: pd.DataFrame) -> pd.Series:
    """Probability of high risk used to sample the proxy label."""
    score = compute_risk_score(df)
    centre = score.quantile(0.60)                   # ~40 % high-risk prevalence
    return 1.0 / (1.0 + np.exp(-2.5 * (score - centre)))


def build_dataset(save: bool = True) -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)

    if REAL_TARGET_COLUMN and REAL_TARGET_COLUMN in df.columns:
        df = df.rename(columns={REAL_TARGET_COLUMN: TARGET})
        print(f"Using real target column '{REAL_TARGET_COLUMN}'.")
    else:
        rng = np.random.default_rng(SEED)
        prob = risk_probability(df)
        df[TARGET] = (rng.random(len(df)) < prob).astype(int)
        print("Created SYNTHETIC proxy target 'msk_risk' (see make_target.py).")

    print(f"High-risk share: {df[TARGET].mean():.1%}")
    if save:
        df.to_csv(PROCESSED_CSV, index=False)
    return df


if __name__ == "__main__":
    build_dataset()
