"""
Builds the ML Lab mini-project report (PDF) in the AISSMS IOIT format.

    python -m report.experiments      # computes all numbers + figures
    python -m report.build_report     # writes report/Mini_Project_Report.pdf
"""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.platypus import NextPageTemplate, PageBreak, Paragraph, Spacer, Table, TableStyle

from report.diagrams import data_flow, proxy_flow
from report.layout import *            # noqa: F401,F403
from src.config import REPORTS_DIR, ROOT, TARGET
from src.make_target import build_dataset, risk_probability

OUT = ROOT / "report" / "Mini_Project_Report.pdf"
FIGS = REPORTS_DIR / "report_figures"
D = json.loads((REPORTS_DIR / "report_data.json").read_text())
E, SP, RES, TUN = D["eda"], D["split"], D["results"], D["tuning"]
B, BEST = D["boost_name"], D["best"]
XGB, SHAPLIB = D["using_xgboost"], D["using_shap"]
SHAPN = "SHAP" if SHAPLIB else "Shapley-value"
MODELS = ["Random Forest", B]
OTHER = [m for m in MODELS if m != BEST][0]
f4 = lambda x: f"{x:.4f}"
pc = lambda x: f"{x * 100:.1f}%"
nm = lambda n: f"{n:,}"

# extra facts computed from the data itself
raw = build_dataset(save=False)
from sklearn.metrics import roc_auc_score
ORACLE = roc_auc_score(raw[TARGET], risk_probability(raw))
SUBSUM = float(((raw.social_media_hours + raw.gaming_hours + raw.work_study_hours) > raw.daily_screen_time_hours).mean())
data_flow(FIGS / "fig_dataflow.png"); proxy_flow(FIGS / "fig_proxy_flow.png")
F = lambda n: FIGS / n

TOP_SHAP = list(D["shap_top"])
TOP_RF = list(D["imp_rf"])
N_TEST, N_TRAIN = SP["test"][0], SP["train"][0]
NF = SP["train"][1]
CIT = "[{}]".format
ITEMS = {}


def note_box():
    if XGB and SHAPLIB:
        return []
    txt = ("<b>Implementation note.</b> In the environment in which this copy of the report was generated, the "
           "<i>xgboost</i> and <i>shap</i> packages could not be installed. The figures reported under "
           "&ldquo;Gradient Boosting (sklearn)&rdquo; therefore come from scikit-learn&rsquo;s "
           "HistGradientBoostingClassifier, a histogram-based gradient-boosted-tree model from the same family "
           "as XGBoost, and the Shapley values were obtained with a Monte-Carlo permutation estimator of the "
           "quantity that SHAP computes. The project code (<i>src/train.py</i>, <i>src/explain.py</i>) uses real "
           "XGBoost and SHAP; running <i>python -m report.experiments</i> followed by "
           "<i>python -m report.build_report</i> on a machine with both packages installed regenerates every "
           "number, table and figure of this report with true XGBoost and SHAP results.")
    t = Table([[Paragraph(txt, ParagraphStyle("nb", parent=S["body"], fontSize=9.8, leading=11.8))]],
              colWidths=[CW])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.black),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF6DC")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
    return [t, Spacer(1, 8)]


def story(toc):
    s = []
    A = s.append
    X = s.extend

    # ---------------------------------------------------------------- cover
    cov = lambda t, sz=12, bold=False, before=0, after=0, lead=None: Paragraph(
        t, ParagraphStyle("c", parent=S["cov"], fontName=BOLD if bold else FONT, fontSize=sz,
                          leading=lead or sz * 1.35, spaceBefore=before, spaceAfter=after))
    A(Spacer(1, 8))
    A(cov("A", after=14)); A(cov("ML LAB MINI PROJECT REPORT", after=14)); A(cov("ON", after=44))
    A(cov("PREDICTING NECK PAIN AND<br/>MUSCULOSKELETAL PROBLEMS<br/>FROM SMARTPHONE USAGE", 21, True, 0, 34, 31))
    A(cov("BY", after=26))
    for n in ("Shivam Pravin Varpe &lt;24510132&gt;", "Viraj Hanumant Zalke &lt;24510138&gt;",
              "Tejas Mohan Suryawanshi &lt;24510126&gt;"):
        A(cov(n, 11, True, 0, 9))
    A(cov("UNDER THE GUIDANCE OF", before=36, after=12)); A(cov("Dr. Mrunal K. Pathak", 12, True, 0, 40))
    A(cov("DEPARTMENT OF INFORMATION TECHNOLOGY", 12.5, after=12))
    A(cov("ALL INDIA SHRI SHIVAJI MEMORIAL SOCIETY&rsquo;S", 8.5, after=4))
    A(cov("INSTITUTE OF INFORMATION TECHNOLOGY", 12.5, after=4)); A(cov("PUNE&ndash; 411001", 8.5, after=26))
    A(cov("SAVITRIBAI PHULE PUNE UNIVERSITY, PUNE", 10.5, after=4)); A(cov("A.Y.: 2026-27", 11, True))
    A(NextPageTemplate("normal")); A(PageBreak())

    # ---------------------------------------------------------- certificate
    A(Spacer(1, 4))
    A(Paragraph("CERTIFICATE", ParagraphStyle("ct", parent=S["cov"], fontName=BOLD, fontSize=20, leading=26, spaceAfter=14)))
    A(Paragraph("This is to certify <b>Machine Learning Lab [ITPCC507]</b> Mini Project Report entitled "
                "<b>&ldquo;Predicting Neck Pain and Musculoskeletal Problems from Smartphone Usage&rdquo;</b> being submitted by",
                ParagraphStyle("cb", parent=S["body"], leading=20, spaceAfter=10)))
    for n in ("Shivam Pravin Varpe (24510132)", "Viraj Hanumant Zalke (24510138)", "Tejas Mohan Suryawanshi (24510126)"):
        A(Paragraph(n, ParagraphStyle("cn", parent=S["cov"], fontSize=12, leading=20, spaceAfter=6)))
    A(Paragraph("Is a record of bonafide work carried out by the group under the supervision and guidance of "
                "Dr. Mrunal Pathak in partial fulfillment of the requirement for <b>Third Year BTech (Div-B)</b> from the "
                "Department of Information Technology, AISSMS Institute of Information Technology, Pune in the academic "
                "year 2026-2027.", ParagraphStyle("cb2", parent=S["body"], leading=20, spaceBefore=8, spaceAfter=14)))
    A(Paragraph("Date :", ParagraphStyle("cd", parent=S["left"], leading=20, spaceAfter=10)))
    A(Paragraph("Place : Pune", ParagraphStyle("cp", parent=S["left"], leading=20)))
    A(Spacer(1, 235))
    sig = Table([[Paragraph("Name of the Mentor", S["cellc"]), Paragraph("Head of the Department", S["cellc"])],
                 [Paragraph("Dr. Mrunal Pathak", S["cellc"]), Paragraph("Dr. Harshada Magar", S["cellc"])]],
                colWidths=[CW / 2] * 2)
    sig.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 6), ("FONTSIZE", (0, 0), (-1, -1), 12)]))
    for r in sig._cellvalues:
        for c in r:
            c.style = ParagraphStyle("sg", parent=S["cellc"], fontSize=12, leading=18)
    A(sig); A(PageBreak())

    # ------------------------------------------------------------------ TOC
    A(Paragraph("Table of Contents", ParagraphStyle("tt", parent=S["cov"], fontName=BOLD, fontSize=18, leading=24, spaceAfter=10)))
    for i, (key, label) in enumerate([("s1", "Title"), ("s2", "Aim"), ("s3", "Theory"), ("s4", "Introduction"),
                                      ("s5", "Flowchart &amp; Algorithms"), ("s6", "Methodology"),
                                      ("s7", "Result and Discussion"), ("s8", "Conclusion and Future Scope"),
                                      ("s9", "Input and Output"), ("s10", "References")], 1):
        A(TocLine(f"{i}.  {label}".replace("&amp;", "&"), toc.get(key, 0)))
    A(PageBreak())

    # ------------------------------------------------------- acknowledgement
    A(Paragraph("Acknowledgement", ParagraphStyle("ak", parent=S["cov"], fontName=BOLD, fontSize=18, leading=24, spaceAfter=10)))
    A(P("We convey our warm gratitude and sincere thanks towards Dr. Mrunal K. Pathak Ma&rsquo;am for her valuable "
        "guidance, constant encouragement, and valuable suggestions, without which the present project would not "
        "have come to its shape. We are equally grateful to all present faculty members of the Information "
        "Technology department for helping in choosing the right tech stack as well as clearing doubts from time to "
        "time in accomplishing our project work. We express our gratitude to the authors, editors, and publishers of "
        "the books and papers which helped us in completing preparatory research. We would also like to thank "
        "everyone who gave us valuable inputs which led to the completion and improvement of the project."))
    A(Spacer(1, 12))
    for i, n in enumerate(("Shivam Pravin Varpe", "Viraj Hanumant Zalke", "Tejas Mohan Suryawanshi"), 1):
        A(Paragraph(f"{i}. {n}", ParagraphStyle("an", parent=S["body"], fontName=BOLD, alignment=TA_RIGHT, spaceAfter=2)))
    A(PageBreak())

    # -------------------------------------------------- 1 Title, 2 Aim
    A(H1("1. Title", "s1"))
    A(P("Predicting Neck Pain and Musculoskeletal Problems from Smartphone Usage."))
    A(H1("2. Aim", "s2"))
    A(P("The aim of this project is to build and compare machine learning classifiers (Random Forest and XGBoost) "
        "that predict whether a person is at high risk of neck pain and musculoskeletal (MSK) problems from "
        "smartphone-usage habits, and to explain the predictions with SHAP. The project is an educational "
        "demonstration and is not intended for clinical or diagnostic decisions."))
    A(H2("2.1 Objectives")); A(P("The main objectives of this project are to:"))
    X(numbered([
        f"Explore and preprocess the smartphone-usage dataset of {nm(E['n_rows'])} records using a leakage-free "
        "workflow with stratified 80% / 20% train-test partitioning.",
        f"Construct a documented, ergonomics-inspired proxy target (<i>msk_risk</i>, {pc(E['share_high'])} high risk) "
        f"because the dataset has no recorded neck-pain outcome, and engineer {NF - 15} additional features.",
        "Train, tune and evaluate Random Forest and XGBoost using 5-fold stratified cross-validation optimizing ROC-AUC.",
        "Benchmark the models on held-out test data using accuracy, precision, recall, F1, ROC-AUC and PR-AUC, "
        "explain the best model with SHAP, and articulate dataset and modeling limitations."]))
    A(H2("2.2 Scope")); A(P("The scope of this project encompasses the following areas:"))
    X(bullets([
        f"Binary classification of high (1) versus low (0) risk of neck pain / MSK strain from {NF} features derived "
        "from age, gender, daily and weekend screen time, social-media, gaming and work/study hours, sleep, "
        "notifications, app opens, stress level, academic/work impact and addiction level.",
        "Two algorithms (Random Forest, XGBoost) trained with built-in class weighting, each tuned by randomized search.",
        "Global and local interpretation through SHAP summary, bar, dependence and waterfall plots, together with the "
        "built-in feature importance of the two models.",
        "A reproducible Python project (scripts for EDA, training, SHAP and prediction) that can be re-run on a real "
        "neck-pain survey by naming its label column in <i>config.py</i>.",
        "The system does not diagnose individuals and must not be used for medical decisions."]))

    # --------------------------------------------------------------- 3 Theory
    A(PageBreak()); A(H1("3. Theory", "s3"))
    A(P("Neck pain and other musculoskeletal problems are among the most common complaints linked to prolonged "
        "smartphone use. Holding a phone below eye level makes the head tilt forward, and the load on the cervical "
        f"spine rises sharply with the angle of flexion {CIT(2)}. Reviews of mobile touch-screen use report "
        f"associations between long or repetitive use and neck, shoulder and thumb symptoms {CIT(3)}, a pattern "
        f"popularly called &ldquo;text neck&rdquo; {CIT(1)}. Poor sleep and high stress are widely discussed "
        "as factors that reduce recovery and increase muscle tension. Machine learning can learn how such "
        "usage and lifestyle factors combine to raise risk."))
    A(H2("3.1 The Proxy Label"))
    A(P("Supervised learning needs a target. The supplied dataset records smartphone behaviour and addiction level "
        "but no neck-pain or MSK outcome. We therefore generate a <b>proxy target</b>, <i>msk_risk</i>: an "
        "ergonomics-inspired risk score is built from screen time, gaming, social-media use, app opens, "
        "notifications, sleep, age and stress, converted to a probability with a logistic function, and the label "
        "is then <i>sampled</i> from that probability with a fixed random seed. Sampling makes the label noisy, so "
        "it is not a deterministic formula. Because the label is synthetic, the models show that the pipeline works "
        "and that the learned relationships agree with the design; they are not evidence about real patients. "
        "Replacing the proxy with survey data is the most important future improvement."))
    A(H2("3.2 Classifiers"))
    A(P("Random Forest is an ensemble of decision trees, each trained on a bootstrap sample of the rows and a random "
        f"subset of the features at every split; the votes are averaged {CIT(4)}. It is robust, needs little "
        "preprocessing and gives a built-in importance score. XGBoost is a gradient-boosting framework that builds "
        "trees one after another, each tree correcting the errors of the earlier ones. It minimises a regularised "
        "objective, Obj = &Sigma; loss(y, &#375;) + &Sigma; (&gamma;T + &frac12;&lambda;||w||&sup2;), where T is the number of "
        f"leaves and w the leaf weights, and uses second-order gradients and column/row subsampling {CIT(5)}. "
        "Both models are trained inside the same scikit-learn workflow " + CIT(8) + "."))
    A(H2("3.3 Cross-Validation and Tuning"))
    A(P("In 5-fold stratified cross-validation the training data is split into five folds that keep the same "
        "high-risk share; each fold is used once for validation. RandomizedSearchCV samples random "
        f"hyper-parameter combinations {CIT(12)} and keeps the one with the best average ROC-AUC. The test set is "
        "never used during tuning, which prevents data leakage."))
    A(H2("3.4 Explainability with SHAP"))
    A(P("A model that only returns a probability is hard to trust. SHAP (SHapley Additive exPlanations) assigns every "
        f"feature a contribution to each individual prediction {CIT(6)}. It is based on Shapley values from game "
        f"theory {CIT(13)}: the prediction f(x) is split as f(x) = &phi;<sub>0</sub> + &Sigma; &phi;<sub>i</sub>, where "
        "&phi;<sub>0</sub> is the average prediction and &phi;<sub>i</sub> is the push (positive or negative) given by "
        f"feature i. TreeSHAP computes these values exactly and quickly for tree ensembles {CIT(7)}. Averaging "
        "|&phi;<sub>i</sub>| over many people gives a global importance, while a single person&rsquo;s values explain "
        "that one prediction."))
    A(H2("3.5 Evaluation Metrics"))
    A(P("Metrics used (TP, TN, FP, FN = true/false positives and negatives):"))
    A(P("Accuracy = (TP + TN) / (TP + TN + FP + FN); Precision = TP / (TP + FP); Recall = TP / (TP + FN); "
        "F1 score = 2 &times; Precision &times; Recall / (Precision + Recall); ROC-AUC = area under the curve of "
        f"true-positive rate versus false-positive rate over all thresholds; PR-AUC (average precision) = area under the precision&ndash;recall curve {CIT(9)}."))

    # ------------------------------------------------------- 4 Introduction
    A(PageBreak()); A(H1("4. Introduction", "s4"))
    A(P("Smartphones are used for many hours a day for messaging, social media, gaming, study and work. "
        "Long sessions with the head bent forward, combined with short sleep and stress, are commonly associated "
        "with neck pain and other musculoskeletal complaints. If risky usage patterns can be recognised early, simple "
        "advice such as breaks, posture correction and better sleep can be given before pain becomes chronic, which is "
        "why data-driven risk prediction is an interesting area of study."))
    A(P(f"This project demonstrates a complete, small-scale machine learning workflow for predicting neck-pain / MSK "
        f"risk from smartphone usage. The dataset &ldquo;Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv&rdquo; "
        f"contains {nm(E['n_rows'])} records and {E['n_cols']} columns, including age, gender, daily and weekend screen "
        "time, social-media, gaming and work/study hours, sleep hours, notifications and app opens per day, stress "
        "level, academic/work impact and addiction level. Because it has no neck-pain column, a documented proxy "
        f"target is created, giving {nm(E['n_high'])} high-risk records ({pc(E['share_high'])}) and a mildly "
        "imbalanced binary classification task.", "left"))
    A(P(f"Two classifiers, Random Forest and XGBoost, are trained and compared. Hyper-parameters are tuned with "
        f"5-fold stratified cross-validation, and both models are evaluated on a held-out test set of "
        f"{nm(N_TEST)} records. The best model is then interpreted with SHAP, which shows which usage and lifestyle "
        "factors push an individual&rsquo;s predicted risk up or down."))
    A(P("The whole workflow is implemented as a Python project (<i>make_target.py</i>, <i>features.py</i>, "
        "<i>train.py</i>, <i>explain.py</i>, <i>predict.py</i>) using pandas, scikit-learn, XGBoost, SHAP, "
        "matplotlib and seaborn, and is prepared for publication on GitHub."))
    A(P("An important point must be stated clearly: the model is a student demonstration trained on a synthetic "
        "proxy label. Its scores describe how well the algorithms recover the designed relationship, not how well they "
        "would detect neck pain in real people, and the application must not be used for clinical or diagnostic "
        "decisions."))

    # ------------------------------------------------ 5 Flowchart & Algorithms
    A(PageBreak()); A(H1("5. Flowchart &amp; Algorithms", "s5"))
    A(H2("5.1 Data Flow Architecture"))
    X(fig(F("fig_dataflow.png"), 300, "Fig 5.1: Data Flow Diagram"))
    A(P(f"The diagram above traces the flow of data through our machine learning pipeline. After loading and profiling "
        f"the {nm(E['n_rows'])} records, the proxy target is created and the {NF} model features are built. The data is "
        f"split (stratified 80% / 20%) into {nm(N_TRAIN)} training rows and {nm(N_TEST)} held-out test rows. Both models "
        "handle the remaining class imbalance with built-in weights. Cross-validation guides the hyper-parameter "
        "search, the final performance is measured once on the test set, and SHAP explains the predictions of the "
        "best model, which is saved for use by the prediction script."))
    X(fig(F("fig_proxy_flow.png"), 285, "Fig 5.2: Creation of the Proxy Target (msk_risk)"))
    A(P("Fig 5.2 shows how the proxy label is produced. Heavier and more repetitive smartphone use, higher stress and "
        "shorter sleep increase the risk score; the score is turned into a probability and the label is drawn from it "
        "with a fixed seed so that the experiment is reproducible."))
    A(H2("5.2 Overall Workflow of the Project"))
    A(P("The steps below explain in simple words how the pipeline is implemented in the project for both models; "
        "the formal pseudocode is given in Section 5.6."))
    X(numbered([
        f"<b>Load and clean the data.</b> Read the CSV file, check for missing values and duplicates. Only "
        f"<i>addiction_level</i> is blank ({nm(E['missing_addiction'])} rows) and a blank means &ldquo;not addicted&rdquo;, so it "
        "is filled with &ldquo;None&rdquo;. The ID columns are dropped.",
        "<b>Create the proxy target.</b> Compute the ergonomic risk score, convert it to a probability and sample "
        "the label <i>msk_risk</i> with a fixed seed.",
        f"<b>Build features.</b> Encode stress, addiction level, academic impact and gender, and add three engineered "
        f"features, giving {NF} numeric features. <i>addicted_label</i> is dropped because it duplicates addiction level.",
        f"<b>Split the data once.</b> Keep 80% ({nm(N_TRAIN)} rows) for training and 20% ({nm(N_TEST)} rows) for testing, "
        "with the same high-risk share in both parts. The test set is not used again until the very end.",
        "<b>Tune the settings.</b> Try 15 random hyper-parameter combinations per model with 5-fold stratified "
        "cross-validation and keep the combination with the best ROC-AUC.",
        "<b>Test once.</b> Predict probabilities for the test records, apply the 0.50 threshold and calculate "
        "accuracy, precision, recall, F1, ROC-AUC and PR-AUC.",
        "<b>Pick the best model.</b> Compare the two models by test ROC-AUC and study the best one with its confusion "
        "matrix, ROC and PR curves.",
        "<b>Explain with SHAP.</b> Compute SHAP values on test records and draw summary, bar, dependence and waterfall "
        "plots; save the model and use it to score new users."]))
    rf_p, bo_p = TUN["Random Forest"]["params"], TUN[B]["params"]
    A(H2("5.3 Random Forest in Our Project"))
    A(P("<b>Idea:</b> Random Forest builds many decision trees. Each tree sees a random sample of the rows and only a "
        "random subset of the features at every split, and the final probability is the average of all the trees. "
        "This reduces the over-fitting of a single tree."))
    A(P("<b>How it is used here:</b>"))
    X(numbered([
        "All features are numeric (ordinal and 0/1 encodings), so no scaling is needed because trees do not depend on "
        "feature scale.",
        "Class imbalance is handled with class_weight = &ldquo;balanced&rdquo;.",
        "Try 15 random combinations of the number of trees (200 to 600), maximum depth (4 to 19), minimum samples per "
        "leaf (1 to 14) and features per split (sqrt, log2 or 50%).",
        f"Score each combination by 5-fold ROC-AUC and keep the best one: " +
        ", ".join(f"{k} = {v}" for k, v in rf_p.items()) + f" (CV ROC-AUC {f4(TUN['Random Forest']['cv_auc'])}).",
        "Refit on the full training data and evaluate once on the test set."]))
    A(H2("5.4 XGBoost in Our Project"))
    A(P("<b>Idea:</b> XGBoost is a boosting method. It builds small trees one after another, and every new tree focuses "
        "on the mistakes made by the earlier trees. A regularisation term and shrinkage (learning rate) keep the "
        "model from over-fitting, which makes it fast and accurate on tabular data."))
    A(P("<b>How it is used here:</b>"))
    X(numbered([
        "Use the same numeric feature matrix (no scaling is needed).",
        "Class imbalance is handled with scale_pos_weight = (number of low-risk) / (number of high-risk) in the "
        f"training data ({SP['pos_weight']:.2f}).",
        "Try 15 random combinations of the number of trees (150 to 600), maximum depth (2 to 7), learning rate "
        "(0.01 to 0.21), row and column sampling (0.6 to 1.0) and minimum child weight (1 to 9).",
        f"Score each combination by 5-fold ROC-AUC. The best {B} configuration was: " +
        ", ".join(f"{k} = {v}" for k, v in bo_p.items()) + f" (CV ROC-AUC {f4(TUN[B]['cv_auc'])}).",
        "Refit on the full training data and evaluate once on the test set."]))
    A(H2("5.5 SHAP in Our Project"))
    A(P("<b>Idea:</b> SHAP shares the credit for a prediction fairly among the features. For one person, a feature with "
        "a positive SHAP value pushed the predicted risk above the average and a negative value pushed it below."))
    A(P("<b>How it is used here:</b>"))
    X(numbered([
        "Take the best tree model and a sample of 300 test records.",
        "Use the TreeExplainer of the shap library to obtain one SHAP value per feature per record.",
        "Draw the summary (beeswarm) plot for the global picture and the bar plot of mean |SHAP| for ranking.",
        "Draw a dependence plot for the most important feature and waterfall plots for the highest-risk and "
        "lowest-risk test persons to explain single predictions."]))
    A(PageBreak())

    # ----------------------------------------------------------- 5.6 Pseudocode
    A(H2("5.6 Pseudocode of the Project"))
    A(P("The pseudocode below shows how the project is implemented, how each algorithm is used and how the main "
        "decisions were taken. Algorithm 1 is the complete program, Algorithm 2 gives the decision rules behind it, "
        "and Algorithms 3 to 9 expand its main steps. Words in bold are control keywords; text after &ldquo;//&rdquo; "
        "is a comment."))
    c = lambda t: f"<i>// {t}</i>"
    kw = lambda t: f"<b>{t}</b>"
    A(Paragraph(DASH, S["dash"]))
    X(pseudo([
        (0, f"{kw('INPUT')}: smartphone-usage CSV ({nm(E['n_rows'])} records, {E['n_cols']} columns)"),
        (0, f"{kw('OUTPUT')}: best model, test-set metrics, SHAP explanations, risk predictions"),
        (1, f"1. df &larr; {kw('LOAD')} the CSV file"),
        (1, f"2. {kw('CHECK')} missing values and duplicate rows {c('only addiction_level is blank = not addicted')}"),
        (1, f"3. df.msk_risk &larr; CREATE_PROXY_TARGET(df) {c('Algorithm 3')}"),
        (1, f"4. X &larr; BUILD_FEATURES(df) ; y &larr; df.msk_risk {c('Algorithm 4')}"),
        (1, f"5. X_train, X_test, y_train, y_test &larr; STRATIFIED_SPLIT(X, y, 80% / 20%) {c(f'{N_TRAIN} / {N_TEST} rows')}"),
        (1, "6. results &larr; empty list"),
        (1, f"7. {kw('FOR EACH')} model {kw('IN')} {{Random Forest, XGBoost}}"),
        (3, f"a) best_model &larr; TUNE(model, X_train, y_train) {c('Algorithm 5')}"),
        (3, "b) proba &larr; best_model.predict_proba(X_test) ; pred &larr; (proba &ge; 0.50)"),
        (3, f"c) {kw('STORE')} Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC of pred in results"),
        (1, f"8. best &larr; SELECT_BEST(results) {c('Algorithm 8')}"),
        (1, f"9. phi &larr; EXPLAIN_WITH_SHAP(best, X_test) {c('Algorithm 7')}"),
        (2, f"{kw('SHOW')} confusion matrix, ROC and PR curves, importance and SHAP plots of best"),
        (1, f"10. {kw('SAVE')} best ; {kw('RETURN')} best"),
    ], "Algorithm 1: Overall workflow of the neck pain / MSK risk prediction project",
        f"This algorithm is the complete project as one controlled process. The {nm(E['n_rows'])} records are checked, the "
        "proxy target and the features are created, and the data is split 80/20 so the test set stays untouched until "
        "the end. The program then tunes and tests Random Forest and XGBoost, selects the model with the higher test "
        f"ROC-AUC (here <b>{BEST}</b>, ROC-AUC {f4(RES[BEST]['roc_auc'])}), and explains it with SHAP."))
    X(pseudo([
        (0, f"Q1. {kw('IF')} the high-risk share is between 30% and 70% {kw('THEN')} accuracy is still meaningful, but also report"),
        (2, f"Precision, Recall, F1, ROC-AUC {c('share = ' + pc(E['share_high']))}"),
        (0, "Q2. How to handle the remaining imbalance?"),
        (2, "Random Forest: class_weight = 'balanced' ; XGBoost: scale_pos_weight = n_low / n_high"),
        (0, "Q3. Which algorithms?"),
        (2, "Random Forest (many trees, averaged), XGBoost (boosting, learns from earlier mistakes)"),
        (0, "Q4. How to avoid data leakage?"),
        (2, "split the data first ; tune only with cross-validation on training data ; drop IDs and addicted_label"),
        (0, "Q5. How to tune?"),
        (2, "RandomizedSearchCV, 15 candidates, 5 stratified folds, scoring = ROC-AUC"),
        (0, f"Q6. Which model is final?"),
        (2, f"the highest test ROC-AUC {c(f'{BEST}')}"),
        (0, f"Q7. How to explain predictions?"),
        (2, "SHAP TreeExplainer on the best tree model: global (summary, bar) and local (waterfall) views"),
        (0, f"Q8. How is a prediction used? {kw('IF')} probability &ge; 0.50 {kw('THEN')} HIGH RISK {kw('ELSE')} LOW RISK"),
        (2, c("a screening-level risk ranking, not a diagnosis")),
    ], "Algorithm 2: How the main decisions of the project were taken",
        "This algorithm lists the rules behind each choice in the project. Because the proxy label is only mildly "
        "imbalanced, accuracy is reported but never used alone. Both classifiers get built-in class weighting, "
        "hyper-parameters are chosen by cross-validated ROC-AUC on the training data only, and the final model is the "
        "one with the higher test ROC-AUC. Predictions are explained with SHAP and always treated as a risk ranking."))
    X(pseudo([
        (0, f"{kw('INPUT')}: df"),
        (1, f"1. {kw('FOR EACH')} column c {kw('IN')} {{daily screen time, weekend screen time, gaming, social media, app opens,"),
        (3, f"notifications, sleep, age}}: z_c &larr; (c &minus; mean(c)) / std(c)"),
        (1, "2. score &larr; 0.35&middot;z_daily + 0.15&middot;z_weekend + 0.15&middot;z_gaming + 0.10&middot;z_social + 0.10&middot;z_opens"),
        (3, f"+ 0.05&middot;z_notif &minus; 0.20&middot;z_sleep + 0.05&middot;z_age + w_stress {c('w_stress: Low 0, Medium 0.35, High 0.70')}"),
        (1, "3. centre &larr; 60th percentile of score"),
        (1, f"4. {kw('FOR EACH')} person i: p_i &larr; 1 / (1 + e^(&minus;2.5 &times; (score_i &minus; centre)))"),
        (1, f"5. {kw('SET')} random seed = 42"),
        (1, f"6. {kw('FOR EACH')} person i: msk_risk_i &larr; 1 {kw('IF')} random(0, 1) &lt; p_i {kw('ELSE')} 0"),
        (1, f"7. {kw('RETURN')} msk_risk"),
    ], "Algorithm 3: CREATE_PROXY_TARGET(df)",
        "This algorithm builds the label that the dataset lacks. Each usage column is standardised and combined with "
        "ergonomic weights (more and heavier use, more stress and shorter sleep mean more risk). The score becomes a "
        "probability and the final 0/1 label is drawn at random from that probability with a fixed seed, so the "
        f"label contains realistic noise and the experiment can be repeated exactly. Even the true probability only "
        f"reaches a ROC-AUC of {ORACLE:.3f} against the sampled label, which is the ceiling for any model."))
    X(pseudo([
        (0, f"{kw('INPUT')}: df"),
        (1, "1. X &larr; the 9 numeric columns (age, screen time, social, gaming, work/study, sleep, notifications, app opens, weekend)"),
        (1, "2. X.stress_level &larr; {Low: 0, Medium: 1, High: 2} ; X.academic_work_impact &larr; 1 IF 'Yes' ELSE 0"),
        (1, "3. X.addiction_level &larr; {blank: 0, Mild: 1, Moderate: 2, Severe: 3}"),
        (1, "4. gender &rarr; three 0/1 columns (Male, Female, Other)"),
        (1, "5. ADD non_work_screen_hours &larr; social + gaming"),
        (3, "weekend_screen_increase &larr; weekend &minus; daily"),
        (3, "checks_per_screen_hour &larr; app_opens / max(daily, 0.5)"),
        (1, f"6. {kw('DROP')} transaction_id, user_id, addicted_label"),
        (1, f"7. {kw('RETURN')} X {c(f'{NF} columns')}"),
    ], "Algorithm 4: BUILD_FEATURES(df)",
        f"This algorithm turns the raw table into the numeric matrix used by both models. Ordered categories are "
        "mapped to numbers, gender is one-hot encoded, and three engineered features describe leisure screen time, the "
        "weekend increase and how often the phone is checked per hour of use. IDs and the duplicate addiction label "
        "are removed so that they cannot leak information."))
    X(pseudo([
        (0, f"{kw('INPUT')}: model, parameter space, X_train, y_train"),
        (1, "1. n_iter &larr; 15 ; best_score &larr; &minus;1"),
        (1, f"2. {kw('REPEAT')} n_iter times"),
        (3, "a) params &larr; random combination chosen from the parameter space"),
        (3, "b) scores &larr; empty list"),
        (3, f"c) {kw('FOR EACH')} fold {kw('IN')} 5 stratified folds of the training data"),
        (5, f"{kw('FIT')} model(params) on the other 4 folds"),
        (5, f"p &larr; predicted high-risk probabilities for this fold ; {kw('APPEND')} ROC-AUC(true labels, p) to scores"),
        (3, f"d) {kw('IF')} mean(scores) &gt; best_score {kw('THEN')} best_score &larr; mean(scores) ; best_params &larr; params"),
        (1, "3. REFIT model(best_params) on the whole training set"),
        (1, f"4. {kw('RETURN')} the fitted model"),
    ], "Algorithm 5: TUNE(model, X_train, y_train) &ndash; randomized search with 5-fold cross-validation",
        "This algorithm finds the best settings for each model without touching the test set. It tries 15 random "
        "combinations of settings; each is trained on 4 folds and scored on the fifth using ROC-AUC. The combination "
        "with the highest average ROC-AUC is kept and refitted on the whole training set, giving settings that are not "
        "tuned to one lucky split."))
    X(pseudo([
        (0, "Random Forest:"),
        (2, f"1. {kw('FOR')} t = 1 {kw('TO')} n_estimators"),
        (4, "sample &larr; bootstrap sample of the training rows"),
        (4, "grow tree t on sample, choosing every split from a random subset of features"),
        (2, "2. p &larr; average of the high-risk probabilities of all trees"),
        (0, "XGBoost:"),
        (2, "1. F &larr; starting log-odds of the training data"),
        (2, f"2. {kw('FOR')} m = 1 {kw('TO')} n_estimators"),
        (4, f"g, h &larr; first and second derivative of the log-loss for every sample at F {c('the mistakes so far')}"),
        (4, "tree m &larr; tree grown to maximise gain, penalised by &gamma; and &lambda; (max_depth, subsample)"),
        (4, "F &larr; F + learning_rate &times; tree m"),
        (2, "3. p &larr; 1 / (1 + e^(&minus;F))"),
    ], "Algorithm 6: Core idea of the two classifiers",
        "This algorithm shows in simple steps how each classifier turns a person&rsquo;s usage data into a risk "
        "probability. Random Forest builds many trees on random samples of rows and features and averages their "
        "answers. XGBoost starts with a simple guess and adds small trees, each one correcting the mistakes of the "
        "earlier ones with a regularised, gradient-based objective."))
    X(pseudo([
        (0, f"{kw('INPUT')}: tree model f, explain set X_s (300 test rows), background set X_b"),
        (1, "1. base &larr; average of f over X_b"),
        (1, f"2. {kw('FOR EACH')} person x {kw('IN')} X_s ; {kw('FOR EACH')} feature i"),
        (3, "phi_i(x) &larr; average over feature orders of [ f(features up to and including i taken from x, rest from background)"),
        (5, "&minus; f(same, without feature i) ]   " + c("TreeSHAP gives this exactly for trees")),
        (1, f"3. {kw('CHECK')} base + &Sigma; phi_i(x) = f(x) {c('additivity')}"),
        (1, "4. global_importance_i &larr; mean over x of |phi_i(x)|"),
        (1, "5. DRAW summary (beeswarm), bar, dependence and waterfall plots"),
        (1, f"6. {kw('RETURN')} phi"),
    ], "Algorithm 7: EXPLAIN_WITH_SHAP(model, X_test)",
        "This algorithm explains the model. For every person and every feature it measures how much that feature "
        "moved the prediction away from the average, by comparing the prediction with and without the feature across "
        "many feature orders. The values add up to the prediction, so they can be shown as a waterfall for one person, or "
        "averaged over many people to rank the features."))
    X(pseudo([
        (1, f"1. candidates &larr; the rows of results for Random Forest and XGBoost"),
        (1, f"2. {kw('SORT')} candidates by ROC-AUC (highest first) ; {kw('IF')} ROC-AUC is equal, {kw('SORT')} by F1"),
        (1, f"3. best &larr; first row of candidates {c(BEST)}"),
        (1, f"4. {kw('RETURN')} best"),
    ], "Algorithm 8: SELECT_BEST(results)",
        f"This algorithm picks the final model in a fair and fixed way. It ranks the two models by test ROC-AUC and "
        f"uses F1 to break ties. The winner is <b>{BEST}</b> (ROC-AUC {f4(RES[BEST]['roc_auc'])}); the gap to "
        f"{OTHER} ({f4(RES[OTHER]['roc_auc'])}) is small, so the choice is a ranking and not strong proof that one "
        "algorithm is better."))
    X(pseudo([
        (0, f"{kw('INPUT')}: 13 values of one person: age, gender, daily/social/gaming/work hours, sleep, notifications,"),
        (2, "app opens, weekend screen time, stress level, academic impact, addiction level"),
        (1, "1. row &larr; BUILD_FEATURES(values) " + c("same feature order as in training")),
        (1, "2. risk &larr; best_model.predict_proba(row) " + c("no scaling needed")),
        (1, f"3. {kw('IF')} risk &ge; 0.50 {kw('THEN PRINT')} 'HIGH RISK' {kw('ELSE PRINT')} 'LOW RISK'"),
        (1, "4. top_factors &larr; the 5 largest |SHAP values| of row"),
        (1, f"5. {kw('PRINT')} risk, risk band and top_factors"),
    ], "Algorithm 9: Predicting neck pain / MSK risk for a new person",
        "This algorithm shows how the finished model is used in practice. The usage values of a new person go into "
        "the saved model and return a risk probability, a risk band and the five factors that contributed most. The "
        "result is a screening-level risk ranking and not a medical diagnosis."))

    # ----------------------------------------------------------- 6 Methodology
    A(PageBreak()); A(H1("6. Methodology", "s6"))
    A(H2("6.1 Software Requirements"))
    A(P("Python 3 &ndash; We have chosen Python, a general-purpose programming language with an extensive scientific "
        "computing and machine learning ecosystem, for the following reasons:"))
    X(bullets([
        "Python code is highly modular, compact, and readable, simplifying reproducible experimentation.",
        "It provides a mature open-source stack for tabular data transformation, hyper-parameter optimization, and "
        "publication-quality statistical visualizations.",
        "scikit-learn, XGBoost and SHAP provide industry-standard, rigorously tested implementations of tree "
        "ensembles, boosting and model explanation."]))
    X(tbl([["Package", "Version", "Purpose"],
           ["Pandas", "&ge; 2.0", "Loading, cleaning, and aggregating tabular CSV records"],
           ["Numpy", "&ge; 1.24", "High-performance vector and matrix numerical operations"],
           ["scikit-learn", "&ge; 1.3", "Random Forest, stratified split, cross-validation, randomized search, metrics"],
           ["XGBoost", "&ge; 2.0", "Gradient-boosted decision tree classifier"],
           ["SHAP", "&ge; 0.44", "TreeExplainer and SHAP values for global and local explanations"],
           ["Matplotlib", "&ge; 3.7", "Base plotting framework for ROC, PR curves, confusion matrices and SHAP plots"],
           ["Seaborn", "&ge; 0.12", "Statistical visualizations (distributions, heatmap, bar charts)"],
           ["Joblib", "&ge; 1.3", "Saving and loading the trained models"]],
          [95, 70, 308], "Table 6.1: Software Requirements and Runtime Packages"))
    A(P("This table denotes the software environment of the project. It lists each Python package, its minimum version "
        "and what it is used for (data handling, modelling, explanation and plotting), so that the results can be "
        "reproduced."))
    A(H2("6.2 Hardware Requirements"))
    A(P("The project workflow is computationally light and executes comfortably on standard commercial laptop hardware:"))
    X(bullets(["Processor: 64-bit multi-core CPU (Intel&reg; Core&trade; i3/i5/i7 or AMD Ryzen&trade; equivalent)",
               "Installed memory (RAM): 4 GB minimum (8 GB recommended for faster cross-validation runs)",
               "Operating System: 64-bit Windows, Linux, or macOS",
               "Runtime Interface: Command line or any Python IDE; Git for publishing the project to GitHub"]))
    A(H2("6.3 Dataset"))
    nr = E["num_range"]
    rng = lambda k: f"{nr[k][0]:g} to {nr[k][1]:g}; mean {nr[k][2]:.1f}"
    cc = E["cat_counts"]
    cnt = lambda k: ", ".join(f"{a} ({nm(b)})" for a, b in cc[k].items())
    A(P(f"Source: Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv, provided with the project. It contains "
        f"{nm(E['n_rows'])} records and {E['n_cols']} columns with no duplicate rows. The only blank values are in "
        f"<i>addiction_level</i> ({nm(E['missing_addiction'])} rows), which correspond to users who are not addicted. "
        "The file contains no neck-pain or MSK column, so the target <i>msk_risk</i> is generated as described in "
        "Section 3.1.", "left"))
    X(tbl([["Column", "Type", "Description"],
           ["transaction_id, user_id", "ID", "Unique identifiers (dropped, not used as features)"],
           ["age", "Numeric", f"Age in years ({rng('age')})"],
           ["gender", "Categorical", f"{cnt('gender')}"],
           ["daily_screen_time_hours", "Numeric", f"Average daily screen time ({rng('daily_screen_time_hours')})"],
           ["social_media_hours", "Numeric", f"Daily social-media hours ({rng('social_media_hours')})"],
           ["gaming_hours", "Numeric", f"Daily gaming hours ({rng('gaming_hours')})"],
           ["work_study_hours", "Numeric", f"Daily work/study hours on the phone ({rng('work_study_hours')})"],
           ["sleep_hours", "Numeric", f"Hours of sleep per night ({rng('sleep_hours')})"],
           ["notifications_per_day", "Numeric", f"Notifications received per day ({rng('notifications_per_day')})"],
           ["app_opens_per_day", "Numeric", f"App opens per day ({rng('app_opens_per_day')})"],
           ["weekend_screen_time", "Numeric", f"Screen time on weekends ({rng('weekend_screen_time')})"],
           ["stress_level", "Ordinal", f"{cnt('stress_level')}"],
           ["academic_work_impact", "Binary", f"Phone use affects study/work: {cnt('academic_work_impact')}"],
           ["addiction_level", "Ordinal", f"{cnt('addiction_level')}; blank = None"],
           ["addicted_label", "Binary", "1 if addicted; duplicates addiction_level (dropped)"],
           ["msk_risk", "Binary (Target)", f"Generated proxy label: 1 = high risk ({nm(E['n_high'])}, {pc(E['share_high'])}), "
                                          f"0 = low risk ({nm(E['n_rows'] - E['n_high'])})"]],
          [118, 70, 285], "Table 6.2: Dataset Attribute Dictionary and Preprocessing Schema"))
    A(P("This table denotes the structure of the dataset. It lists the original columns with their data type and "
        "values, and the generated target <i>msk_risk</i>."))
    A(H2("6.4 Training and Testing Data: Implementation Steps"))
    A(P("Rather than executing generic machine learning procedures, our project implemented a specialised 8-stage pipeline "
        "for smartphone-usage data, a generated proxy target, mixed categorical and numeric features, and tree-based "
        "models with explanations, as detailed below:"))
    steps = [
        ("Project Data Ingestion and Integrity Validation",
         f"The dataset was ingested via pandas, confirming {nm(E['n_rows'])} records and {E['n_cols']} columns. Schema "
         f"validation found {E['duplicates']} duplicate rows and blanks only in addiction_level, which were filled with "
         "&ldquo;None&rdquo;. Identifier columns and the duplicate column addicted_label were excluded from the features."),
        ("Proxy Target Creation",
         f"Because no neck-pain outcome exists, <i>msk_risk</i> was created with the weighted ergonomic score of "
         f"Algorithm 3 and seeded sampling. This gave {nm(E['n_high'])} high-risk ({pc(E['share_high'])}) and "
         f"{nm(E['n_rows'] - E['n_high'])} low-risk records. The true generating probability reaches a ROC-AUC of "
         f"{ORACLE:.3f} against the sampled label, which acts as a ceiling for any model."),
        ("Feature Engineering",
         f"Stress level (0&ndash;2) and addiction level (0&ndash;3) were ordinal-encoded, academic impact was mapped to 0/1 "
         f"and gender was one-hot encoded. Three engineered features were added: non_work_screen_hours, "
         f"weekend_screen_increase and checks_per_screen_hour, giving {NF} features. Trees need no scaling."),
        ("Stratified Train / Test Partitioning",
         f"The {nm(E['n_rows'])} records were partitioned into {nm(N_TRAIN)} training samples (80%) and {nm(N_TEST)} "
         f"held-out test samples (20%) using train_test_split with random_state=42 and stratify=y, so the high-risk "
         f"share is {pc(SP['train_rate'])} in training ({nm(SP['train_pos'])} cases) and {pc(SP['test_rate'])} in testing "
         f"({nm(SP['test_pos'])} cases)."),
        ("Class-Imbalance Handling",
         f"The label is only mildly imbalanced, so no oversampling was needed. Random Forest uses "
         f"class_weight=&lsquo;balanced&rsquo; and XGBoost uses scale_pos_weight = {SP['pos_weight']:.2f} "
         "(low-risk count divided by high-risk count)."),
        ("Hyper-parameter Optimization via Stratified Cross-Validation",
         "Hyper-parameters were tuned using RandomizedSearchCV across 5 stratified folds (StratifiedKFold, n_splits=5, "
         "shuffle=True, random_state=42), 15 candidates per model, scored on ROC-AUC. The search spaces are listed in "
         "Table 6.3."),
        ("Held-Out Benchmarking",
         "Both tuned models were evaluated once on the unseen test records using accuracy, precision, recall, F1, "
         "ROC-AUC and PR-AUC at the default 0.50 threshold; a threshold sensitivity analysis was also computed for "
         "the best model. The best model was identified by ROC-AUC."),
        ("Model Explainability with SHAP",
         "SHAP values were computed for 300 test records of the best tree model. The summary, bar, dependence and "
         "waterfall plots give global and per-person explanations, which are checked against the exploratory analysis "
         "to validate plausibility."),
    ]
    for i, (h, t) in enumerate(steps, 1):
        A(P(f"<b>{i}. {h}:</b> {t}"))
    bo_space = ("n_estimators 150&ndash;600; max_depth 2&ndash;7; learning_rate 0.01&ndash;0.21; subsample 0.6&ndash;1.0; "
                "colsample_bytree 0.6&ndash;1.0; min_child_weight 1&ndash;9" if XGB else
                "max_iter 150&ndash;600; max_depth 2&ndash;7; learning_rate 0.01&ndash;0.21; l2_regularization 0&ndash;5; "
                "min_samples_leaf 10&ndash;79; max_features 0.6&ndash;1.0")
    X(tbl([["Model", "Hyper-parameter search space"],
           ["Random Forest", "n_estimators 200&ndash;600; max_depth 4&ndash;19; min_samples_leaf 1&ndash;14; max_features {sqrt, log2, 0.5}; class_weight = balanced"],
           [("XGBoost" if XGB else "XGBoost (this copy: Gradient Boosting, sklearn)"), bo_space + ("; scale_pos_weight = n_low / n_high" if XGB else "; class_weight = balanced")]],
          [105, 368], "Table 6.3: Hyper-parameter Search Spaces for Tuned Classifiers"))
    A(P("This table denotes the hyper-parameter values tried for each classifier during tuning (tree count, depth, leaf "
        "size and feature sampling for Random Forest; boosting rounds, depth, learning rate, regularisation and sampling "
        "for XGBoost). The best combination from these ranges is reported in Table 7.2."))

    # ------------------------------------------------------------ 7 Results
    A(PageBreak()); A(H1("7. Result and Discussion", "s7"))
    X(note_box())
    A(H2("7.1 Dataset Summary and Exploratory Analysis"))
    gm = E["group_means"]
    A(P(f"The dataset contains {nm(E['n_rows'])} records with no duplicates. After target creation, {nm(E['n_high'])} records "
        f"({pc(E['share_high'])}) are high risk and {nm(E['n_rows'] - E['n_high'])} ({pc(1 - E['share_high'])}) are low risk, "
        "so the classes are only mildly imbalanced (about 1.3 low-risk users for every high-risk user). Accuracy "
        "is therefore meaningful, but precision, recall, F1, ROC-AUC and PR-AUC are reported as well."))
    X(fig(F("fig_target.png"), 215, "Fig 7.1: Distribution of the target (msk_risk)"))
    A(P(f"This bar chart denotes the number of records in each class: {nm(E['n_rows'] - E['n_high'])} low risk and "
        f"{nm(E['n_high'])} high risk ({pc(E['share_high'])}). High-risk users average "
        f"{gm['daily_screen_time_hours'][1]:.1f} hours of daily screen time against {gm['daily_screen_time_hours'][0]:.1f} "
        f"hours for low-risk users, sleep {gm['sleep_hours'][1]:.1f} versus {gm['sleep_hours'][0]:.1f} hours, and open apps "
        f"{gm['app_opens_per_day'][1]:.0f} versus {gm['app_opens_per_day'][0]:.0f} times a day. These differences follow "
        "from how the proxy label was designed."))
    X(fig(F("fig_numeric.png"), 468, "Fig 7.2: Distribution of daily screen time, sleep and app opens by risk class"))
    A(P("(a)&ndash;(c) Histograms with density curves and (d)&ndash;(f) boxplots for daily screen time, sleep hours and "
        "app opens per day (0 = low risk, 1 = high risk). High-risk users sit at higher screen time and app opens and "
        "at lower sleep, but the two groups overlap considerably because the label contains random noise, so no single "
        "feature separates the classes."))
    rt = E["rate_table"]
    rows = [["Factor", "Group", "Records", "High-risk rate"]]
    last = None
    for fa, g, n, r in rt:
        rows.append([fa if fa != last else "", g, nm(n), f"{r:.2f}%"]); last = fa
    X(tbl(rows, [150, 120, 90, 113], "Table 7.1: High-risk rate by factor", center_cols=(2, 3)))
    lk = lambda f, g: [r for r in rt if r[0] == f and r[1] == g][0][3]
    A(P(f"This table denotes the high-risk rate inside each subgroup next to the number of records in it. The rate rises "
        f"steeply with screen time ({lk('Daily screen time (h)', '4 or below'):.1f}% for 4 hours or below to "
        f"{lk('Daily screen time (h)', 'Above 10'):.1f}% above 10 hours) and with stress ({lk('Stress level', 'Low'):.1f}% for "
        f"low stress to {lk('Stress level', 'High'):.1f}% for high stress), and it falls with more sleep "
        f"({lk('Sleep (h)', '5 or below'):.1f}% for 5 hours or below to {lk('Sleep (h)', 'Above 8'):.1f}% above 8 hours). "
        f"Gender shows almost no difference ({lk('Gender', 'Male'):.1f}% male, {lk('Gender', 'Female'):.1f}% female)."))
    X(fig(F("fig_categorical.png"), 468, "Fig 7.3: High-risk rate (%) by categorical and binned factor"))
    A(P("(a) Gender, (b) stress level, (c) addiction level, (d) academic/work impact, (e) daily screen time and (f) sleep. "
        "Stress, addiction level and screen time show clear step-wise increases in risk, sleep shows a decrease, while "
        "gender and academic impact are weak indicators. Addiction level is correlated with heavy use, so part of its "
        "effect mirrors screen time; these are associations in the data, not direct causes."))
    ct = E["corr_target"]; ctk = list(ct)[:3]
    X(fig(F("fig_corr.png"), 340, "Fig 7.4: Correlation heatmap of numeric features and the target"))
    A(P(f"The strongest linear correlations with the target are {ctk[0]} ({ct[ctk[0]]:+.3f}), {ctk[1]} ({ct[ctk[1]]:+.3f}) and "
        f"{ctk[2]} ({ct[ctk[2]]:+.3f}). Screen-time variables are correlated with each other, which is not a problem "
        "for tree-based models."))

    A(H2("7.2 Hyper-parameter Tuning"))
    A(P("Table 7.2 shows the best cross-validated ROC-AUC found for each model on the training data and the "
        "hyper-parameters that produced it."))
    fmtp = lambda p: ", ".join(f"{k} = {v}" for k, v in p.items())
    X(tbl([["Model", "CV ROC-AUC", "Best hyper-parameters"]] +
          [[m, f4(TUN[m]["cv_auc"]), fmtp(TUN[m]["params"])] for m in MODELS],
          [105, 70, 298], "Table 7.2: Best cross-validated ROC-AUC and hyper-parameters", center_cols=(1,)))
    A(P(f"Description: This table denotes the best 5-fold cross-validated ROC-AUC of each model on the training data. "
        f"{max(MODELS, key=lambda m: TUN[m]['cv_auc'])} scored slightly higher "
        f"({f4(max(TUN[m]['cv_auc'] for m in MODELS))} versus {f4(min(TUN[m]['cv_auc'] for m in MODELS))}); "
        "the two models are close."))

    A(H2("7.3 Model Comparison"))
    A(P(f"Table 7.3 and Fig 7.5 show the held-out test performance of both models at the default threshold of 0.50 "
        f"({nm(N_TEST)} test records, {nm(SP['test_pos'])} high-risk), sorted by ROC-AUC."))
    order = sorted(MODELS, key=lambda m: -RES[m]["roc_auc"])
    X(tbl([["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]] +
          [[m] + [f4(RES[m][k]) for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")] for m in order],
          [120, 58, 58, 55, 50, 62, 58], f"Table 7.3: Evaluation on the {nm(N_TEST)}-record test set at the 0.50 threshold",
          center_cols=(1, 2, 3, 4, 5, 6)))
    d_auc = RES[BEST]["roc_auc"] - RES[OTHER]["roc_auc"]
    gap = ORACLE - RES[BEST]["roc_auc"]
    A(P(f"<b>{BEST}</b> is the best model: it has the higher ROC-AUC ({f4(RES[BEST]['roc_auc'])} versus "
        f"{f4(RES[OTHER]['roc_auc'])} for {OTHER}) and an F1 of {f4(RES[BEST]['f1'])}. The ROC-AUC difference of "
        f"{abs(d_auc):.4f} is small and, with a single split of {nm(N_TEST)} records, within the noise, so the ranking should not "
        f"be over-interpreted. Accuracy is {pc(RES[BEST]['accuracy'])} with precision {pc(RES[BEST]['precision'])} and recall "
        f"{pc(RES[BEST]['recall'])}. For reference, the true generating probability of the proxy label reaches "
        f"ROC-AUC {ORACLE:.3f}; the best model is {abs(gap):.3f} {'below' if gap >= 0 else 'above'} this ceiling, which "
        "means both models recover almost all of the learnable signal and further tuning cannot help much."))
    X(fig(F("fig_compare.png"), 330, "Fig 7.5: Test-set metrics of the two models"))
    A(P("The grouped bar chart compares accuracy, precision, recall, F1, ROC-AUC and PR-AUC. The two models perform "
        "very similarly on every metric."))

    A(H2("7.4 ROC and Precision&ndash;Recall Curves"))
    X(fig_row([F("fig_roc.png"), F("fig_pr.png")], 225,
              ["Fig 7.6(a): ROC curves", "Fig 7.6(b): Precision&ndash;Recall (PR) curves"],
              "Fig 7.6: ROC and precision&ndash;recall curves of the two models"))
    A(P(f"The ROC-AUC of about {RES[BEST]['roc_auc']:.2f} means that, for a random high-risk user and a random low-risk "
        f"user, the model ranks the high-risk user higher about {RES[BEST]['roc_auc'] * 100:.0f}% of the time. The PR-AUC of "
        f"{RES[BEST]['pr_auc']:.2f} is about {RES[BEST]['pr_auc'] / SP['test_rate']:.1f} times the baseline of "
        f"{SP['test_rate']:.3f} (the high-risk share), which shows clear but not perfect predictive power."))

    A(H2("7.5 Effect of the Decision Threshold"))
    th = D["thresholds"]
    X(tbl([["Threshold", "Accuracy", "Precision", "Recall", "F1", "False positives", "False negatives"]] +
          [[t] + [f4(th[t][k]) for k in ("accuracy", "precision", "recall", "f1")] + [th[t]["fp"], th[t]["fn"]] for t in th],
          [58, 62, 62, 58, 50, 90, 93], f"Table 7.4: Threshold sensitivity of the best model ({BEST}, test set)",
          center_cols=(0, 1, 2, 3, 4, 5, 6)))
    A(P(f"Description: This table denotes how the metrics change when the probability threshold for &ldquo;high risk&rdquo; "
        f"is moved. Lowering the threshold from 0.7 to 0.3 raises recall from {th['0.7']['recall']:.2f} to "
        f"{th['0.3']['recall']:.2f} while precision moves from {th['0.7']['precision']:.2f} to "
        f"{th['0.3']['precision']:.2f}; the number of false negatives (missed high-risk users) falls from "
        f"{th['0.7']['fn']} to {th['0.3']['fn']} but false positives rise from {th['0.7']['fp']} to {th['0.3']['fp']}. "
        "A health-awareness screening use would favour a lower threshold so that few people are missed, while a use "
        "that must limit unnecessary follow-up would favour a higher one. The default 0.50 is used for the main results."))

    A(H2(f"7.6 Best Model: {BEST}"))
    X(fig(F("fig_cm.png"), 330, "Fig 7.7: Confusion matrices of the two models on the test set"))
    r = RES[BEST]
    A(P(f"These confusion matrices denote how the models classified the {nm(N_TEST)} test records, with true labels in rows and "
        f"predicted labels in columns. For {BEST}: {r['tn']} true negatives, {r['fp']} false positives, {r['fn']} false "
        f"negatives and {r['tp']} true positives. The classification report is given in Table 7.5."))
    cr = D["class_report"]
    X(tbl([["Class", "Precision", "Recall", "F1-score", "Support"],
           ["Low risk"] + [f"{cr['Low risk'][i]:.2f}" for i in range(3)] + [nm(cr["Low risk"][3])],
           ["High risk"] + [f"{cr['High risk'][i]:.2f}" for i in range(3)] + [nm(cr["High risk"][3])],
           ["Accuracy", "", "", f"{r['accuracy']:.2f}", nm(N_TEST)]],
          [130, 85, 85, 85, 88], f"Table 7.5: Classification report of the best model (threshold 0.50)", center_cols=(1, 2, 3, 4)))
    A(P(f"This table denotes the class-wise precision, recall and F1 of the best model, with the number of test records "
        f"(support) per class. Both classes are detected with similar quality (F1 {cr['Low risk'][2]:.2f} for low risk and "
        f"{cr['High risk'][2]:.2f} for high risk)."))

    A(H2("7.7 Built-in Feature Importance"))
    X(fig_row([F("fig_imp_rf.png"), F("fig_imp_boost.png")], 225,
              ["Fig 7.8(a): Random Forest importance", f"Fig 7.8(b): {B} importance"],
              "Fig 7.8: Feature importance of the two models"))
    A(P(f"These bar charts denote which features each model relies on. Random Forest ranks {', '.join(TOP_RF[:3])} highest, "
        f"while the boosted model ({D['imp_boost_label']}) ranks {', '.join(list(D['imp_boost'])[:3])} highest. The "
        "ranking agrees with the exploratory analysis in Table 7.1, but built-in importance only shows what a model "
        "relies on; it does not show the direction of the effect, which is why SHAP is used next."))

    A(H2("7.8 Explainability with SHAP"))
    X(fig(F("fig_shap_summary.png"), 320, f"Fig 7.9: {SHAPN} summary (beeswarm) plot of {B}"))
    A(P(f"Each dot is one test person; the colour shows the feature value (red = high, blue = low) and the horizontal "
        f"position shows how much that feature pushed the prediction up (right) or down (left). {TOP_SHAP[0]}, "
        f"{TOP_SHAP[1]} and {TOP_SHAP[2]} have the widest spread. High values of screen-time and usage variables are "
        "red on the right side (more risk), while high sleep hours are red on the left (less risk). This agrees with "
        "the ergonomic logic used to build the label, which shows that the model has learned the intended relationships."))
    X(fig_row([F("fig_shap_bar.png"), F("fig_shap_dependence.png")], 225,
              [f"Fig 7.10(a): Mean |{SHAPN} value| per feature", f"Fig 7.10(b): Dependence plot of {TOP_SHAP[0]}"],
              "Fig 7.10: Global importance and dependence plot"))
    A(P(f"(a) The bar plot ranks features by mean absolute {SHAPN} value. (b) The dependence plot of the most important "
        f"feature, {TOP_SHAP[0]}, shows how its value changes the prediction; the colour shows a second feature, "
        f"{TOP_SHAP[1]}, so interactions can be seen."))
    X(tbl([["Rank", "Feature", f"Mean |{SHAPN} value|"]] +
          [[i, k, f"{v:.4f}"] for i, (k, v) in enumerate(D["shap_top"].items(), 1)],
          [60, 250, 163], f"Table 7.6: Top features by mean |{SHAPN} value| ({B})", center_cols=(0, 2)))
    X(fig_row([F("fig_shap_waterfall_high.png"), F("fig_shap_waterfall_low.png")], 232,
              ["Fig 7.11(a): Highest-risk test person", "Fig 7.11(b): Lowest-risk test person"],
              "Fig 7.11: Waterfall explanations of two individual predictions"))
    A(P("The waterfall plots start at the average model output and add the contribution of each feature to reach the "
        "person&rsquo;s own score. (a) For the highest-risk person several usage features and stress push the score up; "
        "(b) for the lowest-risk person low usage and long sleep push it down. This is the local explanation that a "
        "single feature-importance bar chart cannot give."))
    exrows = [["Example", "Key inputs", ("P(XGB)" if XGB else "P(GB)"), "P(RF)", "Risk", "Top 3 drivers"]]
    for e in D["examples"]:
        d = e["input"]
        exrows.append([e["title"].split(": ")[1],
                       f"screen {d['daily_screen_time_hours']} h, sleep {d['sleep_hours']} h, stress {d['stress_level']}, "
                       f"addiction {d['addiction_level']}", f"{e['p_boost']:.3f}", f"{e['p_rf']:.3f}", e["label"],
                       "; ".join(f"{k} ({v:+.2f})" for k, v in e["drivers"])])
    X(tbl(exrows, [80, 130, 45, 42, 55, 121], "Table 7.7: Predictions for three example users", center_cols=(2, 3, 4)))

    A(H2("7.9 Discussion and Limitations"))
    X(bullets([
        "<b>The target is a synthetic proxy, not measured neck pain.</b> The dataset has no MSK outcome, so the label was "
        f"generated from a risk score. Test metrics therefore measure how well the models recover that designed "
        f"relationship (the generating formula itself reaches ROC-AUC {ORACLE:.3f}); they say nothing about how well the "
        "models would detect real neck pain. Validation on survey or clinical data is essential before any real use.",
        f"The dataset itself appears synthetic: for {pc(SUBSUM)} of users the social-media, gaming and work/study hours add "
        "up to more than the daily screen time, which is not physically consistent.",
        f"Only one train&ndash;test split of {nm(N_TEST)} records is used, and the two models differ by less than 0.01 in "
        "ROC-AUC, so the choice of the best model is not statistically reliable; repeated cross-validation would give "
        "more stable estimates.",
        "Precision and recall are moderate even in this ideal setting because the label contains random noise; a "
        "real-world label would be noisier still.",
        "Important physical factors such as posture, neck angle, phone-holding style, device size, exercise, "
        "occupation and previous injuries are not in the data.",
        "SHAP explains what the model relies on, not what causes neck pain; the associations should not be read as "
        "causal.",
        "Addiction level and screen time are correlated, so their SHAP contributions partly overlap."]))
    A(PageBreak())

    # ------------------------------------------------------ 8 Conclusion
    A(H1("8. Conclusion and Future Scope", "s8"))
    A(H2("8.1 Conclusion"))
    A(P(f"This project designed and evaluated an end-to-end, leakage-free machine learning workflow for predicting neck-pain "
        f"and musculoskeletal risk from smartphone usage on a dataset of {nm(E['n_rows'])} records. Because the dataset "
        f"has no recorded outcome, a documented ergonomics-inspired proxy target ({pc(E['share_high'])} high risk) was "
        "created. By integrating exploratory data analysis, feature engineering and rigorous 5-fold stratified "
        "cross-validation, two tree-based classifiers, Random Forest and XGBoost, were benchmarked. Experimental "
        f"evaluation on an untouched {nm(N_TEST)}-record test set showed that {BEST} attained the higher performance "
        f"(ROC-AUC = {f4(RES[BEST]['roc_auc'])}, PR-AUC = {f4(RES[BEST]['pr_auc'])}, F1 = {f4(RES[BEST]['f1'])}, accuracy = "
        f"{f4(RES[BEST]['accuracy'])}), very close to the ceiling of {ORACLE:.3f} set by the label noise, while the two models "
        f"differed by only {abs(d_auc):.4f} ROC-AUC. SHAP analysis showed that {TOP_SHAP[0]}, {TOP_SHAP[1]} and "
        f"{TOP_SHAP[2]} are the most influential factors, in line with the ergonomic assumptions behind the label. The "
        "system demonstrates the full workflow and its explanations, but since it is trained on a proxy label its "
        "predictions must be treated as demonstration risk rankings rather than diagnostic verdicts."))
    A(H2("8.2 Project Outcomes"))
    A(P("The outcomes achieved in this work directly satisfy and validate each of the project objectives defined in Section 2.1:"))
    A(P(f"<b>Outcome (i) [Satisfies Objective i]:</b> Successfully explored, cleaned and preprocessed {nm(E['n_rows'])} "
        "smartphone-usage records into a leakage-free workflow with a stratified 80%/20% partition and no test data used "
        "during tuning."))
    A(P(f"<b>Outcome (ii) [Satisfies Objective ii]:</b> Created and documented a reproducible proxy target and {NF} "
        "features, and made its limitation explicit, including the ceiling ROC-AUC of "
        f"{ORACLE:.3f} imposed by label noise."))
    A(P(f"<b>Outcome (iii) [Satisfies Objective iii]:</b> Successfully implemented 5-fold stratified cross-validation and "
        f"randomized search for Random Forest and XGBoost, reaching cross-validated ROC-AUC values of "
        f"{f4(TUN['Random Forest']['cv_auc'])} and {f4(TUN[B]['cv_auc'])}."))
    A(P(f"<b>Outcome (iv) [Satisfies Objective iv]:</b> Benchmarked both models on {nm(N_TEST)} unseen test records, "
        f"establishing {BEST} as the top performer (ROC-AUC = {f4(RES[BEST]['roc_auc'])}), explaining it globally and "
        "locally with SHAP, and clearly articulating dataset constraints."))
    A(H2("8.3 Future Scope"))
    A(P("Future enhancements include collecting a real neck-pain / MSK questionnaire (for example the Neck Disability "
        "Index) together with objective usage logs so that the proxy label can be replaced by measured outcomes, and "
        "adding posture and device-angle features from phone sensors. Probability calibration (Platt scaling or "
        "isotonic regression) would give dependable risk percentages, repeated cross-validation would give more stable "
        "estimates, and external validation on another population is required. Finally, the model and its SHAP "
        "explanations can be packaged into a web or mobile app that gives users personalised advice on breaks, posture and "
        "sleep."))
    A(PageBreak())

    # --------------------------------------------------- 9 Input and Output
    A(H1("9. Input and Output", "s9"))
    A(H2("9.1 Input / Output Specifications"))
    A(P("<b>Input:</b> The CSV file Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv (placed in data/raw/) and, for "
        "prediction, the usage values of a new person given as command-line arguments."))
    A(P("<b>Output:</b> Dataset summary, EDA plots, tuning log, evaluation tables and plots, confusion matrices, feature "
        "importance, SHAP plots, saved models, and the predicted risk probability, risk band and top contributing factors "
        "for the new person."))
    A(P("To run the project, install the requirements and run the scripts from the project folder:"))
    A(code("pip install -r requirements.txt\n"
           "python -m src.eda        # exploratory plots\n"
           "python -m src.train      # train/tune Random Forest and XGBoost, metrics, ROC, confusion matrices\n"
           "python -m src.explain    # SHAP plots\n"
           "python -m src.predict --age 22 --daily_screen_time_hours 9.5 --sleep_hours 5 --stress_level High"))
    A(Spacer(1, 6))
    A(H2("9.2 Complete Implementation Code"))
    A(P("The complete, commented code of the project (Python package <i>src</i>) is presented below, organized by module:"))
    mods = [("Configuration", "config.py"), ("Proxy Target Creation", "make_target.py"),
            ("Feature Engineering and Data Split", "features.py"), ("Exploratory Data Analysis", "eda.py"),
            ("Model Training, Tuning and Evaluation", "train.py"), ("SHAP Explainability", "explain.py"),
            ("Predict Risk for a New Person", "predict.py")]
    for i, (title, fn) in enumerate(mods, 1):
        A(Paragraph(f"<b>{i}. {title} ({fn})</b>", ParagraphStyle("mh", parent=S["left"], spaceBefore=6, spaceAfter=3)))
        A(code((ROOT / "src" / fn).read_text()))
    A(H2("9.3 Experimental Outputs and Execution Logs"))
    A(P("The execution traces and statistical outputs generated by running the project are detailed below:"))
    A(Paragraph("<b>Train / Test Split Log:</b>", S["left"]))
    A(code(f"Train: {tuple(SP['train'])} | High-risk rate: {SP['train_rate'] * 100:.2f} %\n"
           f"Test : {tuple(SP['test'])} | High-risk rate: {SP['test_rate'] * 100:.2f} %"))
    A(Spacer(1, 6)); A(Paragraph("<b>Hyper-parameter Tuning Execution Log:</b>", S["left"]))
    import textwrap
    A(code("\n".join(textwrap.fill(l, 118, subsequent_indent="    ") for l in D["log"])))
    A(Spacer(1, 6)); A(Paragraph("<b>Best Model Selection &amp; Test Set Classification Report:</b>", S["left"]))
    A(code(f"Best model : {BEST}\nThreshold  : 0.50\n"
           f"Accuracy   : {r['accuracy']:.4f}\nPrecision  : {r['precision']:.4f}\nRecall     : {r['recall']:.4f}\n"
           f"F1         : {r['f1']:.4f}\nROC-AUC    : {r['roc_auc']:.4f}\nPR-AUC     : {r['pr_auc']:.4f}\n\n"
           "Classification report of the best model:\n              precision  recall  f1-score  support\n"
           f"   Low risk      {cr['Low risk'][0]:.2f}     {cr['Low risk'][1]:.2f}     {cr['Low risk'][2]:.2f}     {cr['Low risk'][3]}\n"
           f"  High risk      {cr['High risk'][0]:.2f}     {cr['High risk'][1]:.2f}     {cr['High risk'][2]:.2f}     {cr['High risk'][3]}\n"
           f"   accuracy                         {r['accuracy']:.2f}     {N_TEST}"))
    A(Spacer(1, 8))
    X(fig(F("fig_cm.png"), 330, "Fig 9.1: Confusion matrices of Random Forest and the boosted model on the test set"))
    A(Paragraph("<i>Description: These confusion matrices denote the notebook-style output of the test set for both models. "
                "They are the same results as Fig 7.7 and confirm the exact counts of correct and incorrect predictions.</i>",
                S["body"]))
    A(Paragraph("<b>Prediction for New Person Cohort (Inference Demonstrations):</b>", S["left"]))
    A(P(f"Three distinct user profiles were passed to the best model ({BEST}) to demonstrate risk stratification:"))
    for e in D["examples"]:
        d = e["input"]
        A(Paragraph(f"<b>{e['title']}</b>", ParagraphStyle("eh", parent=S["left"], spaceBefore=4, spaceAfter=2, keepWithNext=1)))
        A(code(textwrap.fill(", ".join(f"{k}: {v}" for k, v in d.items()), 120)))
        A(code(f"Predicted high-risk probability: {e['p_boost']:.3f} (threshold 0.50)\nPrediction: {e['label']}\n"
               "Top drivers: " + "; ".join(f"{k} ({v:+.3f})" for k, v in e["drivers"])))
    A(P("The first example (heavy use, short sleep, high stress) is classified high risk, while the second (light use, long "
        "sleep, low stress) is low risk. These probabilities reflect patterns learned from a synthetic proxy label and must "
        "not be used as clinical diagnostic determinations."))
    A(PageBreak())

    # ---------------------------------------------------------- 10 References
    A(H1("10. References", "s10"))
    refs = [
        "Neupane, S., U. I. Ali, and A. Mathew. 2017. Text Neck Syndrome &ndash; Systematic Review. Imperial Journal of Interdisciplinary Research 3 (7): 141&ndash;148.",
        "Hansraj, K. K. 2014. Assessment of Stresses in the Cervical Spine Caused by Posture and Position of the Head. Surgical Technology International 25: 277&ndash;279.",
        "Toh, S. H., P. Coenen, E. K. Howie, and L. M. Straker. 2017. The Associations of Mobile Touch Screen Device Use with Musculoskeletal Symptoms and Exposures: A Systematic Review. PLoS ONE 12 (8): e0181220.",
        "Breiman, L. 2001. Random Forests. Machine Learning 45 (1): 5&ndash;32.",
        "Chen, T., and C. Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining: 785&ndash;794.",
        "Lundberg, S. M., and S.-I. Lee. 2017. A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30.",
        "Lundberg, S. M., G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee. 2020. From Local Explanations to Global Understanding with Explainable AI for Trees. Nature Machine Intelligence 2 (1): 56&ndash;67.",
        "Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, et al. 2011. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12: 2825&ndash;2830.",
        "Saito, T., and M. Rehmsmeier. 2015. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLoS ONE 10 (3): e0118432.",
        "McKinney, W. 2010. Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference: 56&ndash;61.",
        "Hunter, J. D. 2007. Matplotlib: A 2D Graphics Environment. Computing in Science &amp; Engineering 9 (3): 90&ndash;95.",
        "Bergstra, J., and Y. Bengio. 2012. Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research 13: 281&ndash;305.",
        "Shapley, L. S. 1953. A Value for n-Person Games. In Contributions to the Theory of Games, Volume II, edited by H. W. Kuhn and A. W. Tucker: 307&ndash;317. Princeton University Press.",
    ]
    for i, t in enumerate(refs, 1):
        A(Paragraph(t, ParagraphStyle("rf", parent=S["body"], leftIndent=36, firstLineIndent=-36, spaceAfter=6),
                    bulletText=None) if False else
          Paragraph(f"[{i}] {t}", ParagraphStyle("rf", parent=S["body"], leftIndent=34, firstLineIndent=-34, spaceAfter=6)))
    return s


def build():
    toc = {}
    for _ in range(3):                                  # 3 passes -> TOC page numbers are exact
        doc = ReportDoc(str(OUT), toc, title="Predicting Neck Pain and Musculoskeletal Problems from Smartphone Usage",
                        author="Shivam Pravin Varpe, Viraj Hanumant Zalke, Tejas Mohan Suryawanshi")
        doc.build(story(toc))
        new = dict(doc.found)
        if new == toc:
            break
        toc = new
    print("TOC pages:", toc, "| total pages:", doc.page)


if __name__ == "__main__":
    build()
