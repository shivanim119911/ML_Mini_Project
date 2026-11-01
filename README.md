# Predicting Neck Pain & Musculoskeletal (MSK) Risk from Smartphone Usage

A mini machine-learning project that predicts whether a person is at **high risk of
neck pain / musculoskeletal strain** from their smartphone-usage habits, using
**Random Forest**, **XGBoost** and **SHAP** explainability.

> ## Important - read this first
> The provided dataset (`Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`, 7,500 rows)
> has **no neck-pain / musculoskeletal column**. The target `msk_risk` is therefore a
> **synthetic proxy label** created in `src/make_target.py` from an ergonomics-inspired
> risk score (screen time, gaming, app opens, short sleep, stress, ...) plus random noise.
>
> * The models learn *that* proxy relationship - the metrics demonstrate a working ML
>   pipeline, **not** real medical findings.
> * With real survey data (e.g. a neck-pain questionnaire), set `REAL_TARGET_COLUMN`
>   in `src/config.py` and the whole pipeline works unchanged.
> * This is an educational project and **not a medical diagnosis tool**.

## Project structure
```
smartphone-msk-risk/
├── data/raw/                 original CSV
├── data/processed/           CSV with the generated target (created at run time)
├── src/
│   ├── config.py             paths, seed, settings
│   ├── make_target.py        proxy target creation (documented assumptions)
│   ├── features.py           feature engineering + train/test split
│   ├── eda.py                exploratory plots
│   ├── train.py              Random Forest + XGBoost tuning, evaluation, plots
│   ├── explain.py            SHAP global + local explanations
│   └── predict.py            CLI prediction for one person
├── models/                   saved models (created at run time)
├── reports/                  metrics + figures (created at run time)
└── requirements.txt
```

## How to run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.eda        # 1. exploratory plots
python -m src.train      # 2. train/tune RF + XGBoost, metrics, ROC, confusion matrices
python -m src.explain    # 3. SHAP plots (summary, bar, dependence, waterfall)
python -m src.predict --age 22 --daily_screen_time_hours 9.5 --sleep_hours 5 --stress_level High
```
Outputs land in `reports/` (`metrics.csv`, `best_params.json`, `figures/*.png`) and `models/`.

## Method
1. **Target** - proxy `msk_risk` (1 = high risk, about 43 % of rows), sampled from a noisy
   probability so it is not a deterministic formula.
2. **Features** - 9 numeric usage columns, ordinal stress and addiction level, gender
   one-hot, plus engineered `non_work_screen_hours`, `weekend_screen_increase`,
   `checks_per_screen_hour`. IDs and `addicted_label` (duplicate of `addiction_level`) are dropped.
3. **Models** - Random Forest and XGBoost, tuned with 5-fold stratified
   `RandomizedSearchCV` (ROC-AUC) on an 80 % training split; evaluated once on the 20 % hold-out.
4. **Explainability** - SHAP `TreeExplainer` for global feature importance and
   per-person explanations.

## Dataset notes
* `addiction_level` is blank for non-addicted users (treated as "None").
* Per-activity hours (social + gaming + work/study) sometimes exceed `daily_screen_time_hours`,
  so the data appears synthetic; this is a further reason to treat results as a demo.

## Ideas for extension
Collect a real neck-pain survey, add posture / device-angle features, calibrate
probabilities, add a Streamlit app.

## Project report
`report/Mini_Project_Report.pdf` is the ML Lab mini-project report. To regenerate it with your own
(real XGBoost + SHAP) results run: `python -m report.experiments && python -m report.build_report`.
