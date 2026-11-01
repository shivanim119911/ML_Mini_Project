"""Feature engineering + train/test split (shared by train, explain, predict)."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SEED, TARGET, TEST_SIZE
from .make_target import build_dataset

NUMERIC = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]
STRESS_MAP = {"Low": 0, "Medium": 1, "High": 2}
ADDICTION_MAP = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
GENDERS = ["Male", "Female", "Other"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe -> numeric model matrix. IDs and `addicted_label`
    (a duplicate of addiction_level) are dropped on purpose."""
    X = df[NUMERIC].copy()
    X["stress_level"] = df["stress_level"].map(STRESS_MAP)
    X["academic_work_impact"] = (df["academic_work_impact"] == "Yes").astype(int)
    # blank addiction_level in the CSV means "not addicted"
    X["addiction_level"] = df["addiction_level"].fillna("None").map(ADDICTION_MAP)
    for g in GENDERS:
        X[f"gender_{g}"] = (df["gender"] == g).astype(int)
    # engineered features
    X["non_work_screen_hours"] = df["social_media_hours"] + df["gaming_hours"]
    X["weekend_screen_increase"] = df["weekend_screen_time"] - df["daily_screen_time_hours"]
    X["checks_per_screen_hour"] = df["app_opens_per_day"] / df["daily_screen_time_hours"].clip(lower=0.5)
    return X


def load_split():
    df = build_dataset(save=True)
    X, y = build_features(df), df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
