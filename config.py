"""Central configuration for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_CSV = ROOT / "data" / "raw" / "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "dataset_with_target.csv"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_ITER = 15          # randomized-search iterations per model

TARGET = "msk_risk"       # 1 = high risk of neck pain / musculoskeletal strain
# If your CSV already has a real, survey-based label (e.g. "neck_pain"),
# put its column name here and the synthetic proxy target is skipped.
REAL_TARGET_COLUMN = None

for d in (MODELS_DIR, FIG_DIR, PROCESSED_CSV.parent):
    d.mkdir(parents=True, exist_ok=True)
