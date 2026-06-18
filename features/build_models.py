"""
features/build_models.py

Distress-prediction models on a small clustered panel.
138 rows · 10 firms · 30 positives · 50% of positives in 2 firms.

With a dataset this small and concentrated, the validation design matters
more than the model architecture.  The comments below explain every
methodological choice at interview depth.

Three estimators
----------------
1. Persistence baseline
   Predict distressed_next = distressed_this (carry the current state forward).
   This is the minimum bar.  It earns good aggregate recall because distress
   is sticky (62% of distressed firms stay distressed — from the contingency
   table in the diagnostics).  But it CANNOT catch healthy→distressed
   transitions by construction: it always predicts 0 for healthy firms.

2. Logistic regression (L2, standardized features)
   - current_ratio is median-imputed WITHIN each training fold so no test-set
     information leaks into the imputed value.  A current_ratio_missing
     indicator is added so the model can distinguish "current_ratio = median"
     (real value) from "current_ratio = imputed" (unknown).
   - All features are standardized (mean=0, std=1) per fold so the fitted
     coefficients are directly comparable across features.

3. XGBoost (shallow trees, conservative regularization)
   - NaN values passed through natively: XGBoost learns an optimal split
     direction for missing values at each node, so no imputation is needed.
   - max_depth=3, n_estimators=100, learning_rate=0.1 — deliberately
     conservative to avoid overfitting 138 rows with 7 degrees of freedom
     (6 features + intercept).

Validation: leave-one-firm-out (LOFO) cross-validation
-------------------------------------------------------
10 folds, one per firm.  Train on 9 firms, predict the held-out firm.
Pool out-of-fold (OOF) predictions across all 10 folds for global metrics.

Why not random k-fold?
  Random k-fold would put rows from the same firm in both train and test.
  AMC, for example, has extreme leverage every year — a model could learn
  "high leverage firm → distressed" as a firm-level intercept, not a
  generalizable signal.  LOFO asks the harder question: "given firms I
  have seen, can I predict distress for a firm I have never seen?"

Why not time-series split (train on earlier years, test on later)?
  With 10 firms and ~15 years each, a chronological split would test on
  2019–2024 while training only on pre-2019 data — missing the COVID
  shock that drives several distress events.  LOFO is still imperfect
  but it explicitly tests out-of-firm generalization, which is more
  relevant for a real application (predicting distress of a new firm
  given known firms).

Evaluation metrics
------------------
- ROC-AUC: threshold-independent ranking quality.
- PR-AUC (average precision): the right metric for class imbalance
  (~22% positive).  A naive random classifier has PR-AUC ≈ base rate = 0.22.
  Any model above that is adding signal.
- Precision / Recall / F1 at the F1-optimal threshold (selected on OOF data).
- Transition recall: recall on the 10 rows where this=0, next=1.
  The persistence baseline scores 0/10 here.  This is the genuine
  early-warning test — does the model catch onset before it happens?
"""

import sqlite3
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve,
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed -- XGBoost skipped.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] shap not installed -- SHAP skipped (pip install shap).")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "db" / "distress.db"

FEATURES = [
    "leverage",           # liabilities / assets
    "equity_ratio",       # equity / assets
    "retained_to_assets", # retained_earnings / assets
    "roa",                # net_income / assets
    "margin",             # net_income / revenue
    "current_ratio",      # assets_current / liabilities_current (13% missing)
]
TARGET = "distressed_next"

# ============================================================
# Data loading — compute ratio features from raw financials
# ============================================================

def load_model_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load annual_labeled, derive ratio features, return 138-row modelling table.

    Ratios are computed here rather than stored in the DB.  Keeping raw
    accounting values in the DB and engineering features in Python means:
      - the DB stays auditable (raw USD, not derived)
      - feature definitions are version-controlled in this file
      - we can change features without rebuilding the DB
    """
    df = pd.read_sql("SELECT * FROM annual_labeled", conn)

    # Ratio features — pandas propagates NaN when either operand is NaN
    df["leverage"]           = df["liabilities"]       / df["assets"]
    df["equity_ratio"]       = df["equity"]            / df["assets"]
    df["retained_to_assets"] = df["retained_earnings"] / df["assets"]
    df["roa"]                = df["net_income"]        / df["assets"]
    df["margin"]             = df["net_income"]        / df["revenue"]
    df["current_ratio"]      = df["assets_current"]    / df["liabilities_current"]

    # Modelling table: rows where we have a t+1 label (target not NaN)
    model = (
        df[df[TARGET].notna()]
        .copy()
        .reset_index(drop=True)   # integer 0..137 aligns with numpy arrays below
    )
    model[TARGET] = model[TARGET].astype(int)
    return model


# ============================================================
# Helper: F1-optimal threshold on OOF predictions
# ============================================================

def f1_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Return the decision threshold that maximises F1 on these predictions.

    Selecting the threshold on the same OOF pool used for final evaluation
    is slightly optimistic (it's not a true hold-out), but with 138 rows
    there is no separate validation set.  The threshold is used only for
    binary reporting (precision/recall/F1); the threshold-independent
    metrics (ROC-AUC, PR-AUC) are unaffected.
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    return float(thresholds[np.argmax(f1)])


# ============================================================
# PART A — Persistence baseline
# ============================================================

def run_baseline(model: pd.DataFrame) -> dict:
    """
    Persistence baseline: predict distressed_next = distressed_this.

    Operates on the 132 rows where distressed_this is not NaN (6 rows have
    no ICR components and no lagged net_income, so no label at time t).

    Confusion matrix deconstructed:
      TN: currently healthy, stays healthy    — baseline gets right
      FN: currently healthy, becomes distressed — baseline ALWAYS misses
      FP: currently distressed, recovers      — baseline over-predicts distress
      TP: currently distressed, stays distressed — baseline gets right
    """
    base_mask = model["distressed"].notna().values
    y_true    = model[TARGET].values[base_mask]
    y_pred    = (model["distressed"].values[base_mask] == 1).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    return {
        "n_rows": int(base_mask.sum()),
        "acc":    accuracy_score(y_true, y_pred),
        "prec":   precision_score(y_true, y_pred, zero_division=0),
        "rec":    recall_score(y_true, y_pred, zero_division=0),
        "f1":     f1_score(y_true, y_pred, zero_division=0),
        "cm":     cm,
    }


# ============================================================
# PART B — Leave-one-firm-out cross-validation
# ============================================================

def run_lofo_cv(model: pd.DataFrame) -> dict:
    """
    LOFO CV: train on 9 firms, predict the 10th.  Repeat for all 10 firms.

    Returns pooled OOF probability arrays (length 138) for LR and XGB.
    Each element i is the predicted probability for row i from the fold
    in which that row was in the test set.

    Pipeline details:
    - LR: SimpleImputer(median) -> StandardScaler -> LogisticRegression.
      Imputation uses only training-fold data; test-fold NaNs are filled with
      the training-fold median.  This is the correct way to impute — using
      the global median would leak the test firm's financial context.
    - XGB: no imputation; XGBoost handles NaN natively by learning the best
      split direction for missing values at each node.  This is equivalent
      to a learned imputation that can vary by feature and tree.
    """
    y      = model[TARGET].values
    groups = model["cik"].values
    n      = len(model)

    # LR feature matrix: 6 ratios + 1 missing indicator
    # The missing flag goes in before the pipeline; the pipeline imputes
    # and scales everything including the flag (scaling a binary is harmless).
    X_lr = model[FEATURES].copy()
    X_lr["current_ratio_missing"] = X_lr["current_ratio"].isna().astype(float)

    # XGB feature matrix: 6 ratios only (NaN native)
    X_xgb = model[FEATURES].copy()

    lr_oof  = np.full(n, np.nan)
    xgb_oof = np.full(n, np.nan) if HAS_XGB else None

    logo       = LeaveOneGroupOut()
    fold_meta  = []

    for train_idx, test_idx in logo.split(X_lr, y, groups):
        test_firm = model.iloc[test_idx[0]]["ticker"]
        n_pos     = int(y[test_idx].sum())

        # --- Logistic Regression ---
        lr_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ])
        # Pass .values to avoid sklearn feature-name warnings in LOFO
        lr_pipe.fit(X_lr.iloc[train_idx].values, y[train_idx])
        lr_oof[test_idx] = lr_pipe.predict_proba(X_lr.iloc[test_idx].values)[:, 1]

        # --- XGBoost ---
        if HAS_XGB:
            xgb_clf = xgb.XGBClassifier(
                max_depth=3,         # shallow: avoids memorising 138 rows
                n_estimators=100,    # modest: low variance for small data
                learning_rate=0.1,   # standard; lower = more regularization
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
                verbosity=0,
            )
            xgb_clf.fit(X_xgb.iloc[train_idx].values, y[train_idx])
            xgb_oof[test_idx] = xgb_clf.predict_proba(
                X_xgb.iloc[test_idx].values
            )[:, 1]

        fold_meta.append({"firm": test_firm, "n_rows": len(test_idx), "n_pos": n_pos})

    return {
        "y":        y,
        "X_lr":     X_lr,
        "X_xgb":    X_xgb,
        "lr_oof":   lr_oof,
        "xgb_oof":  xgb_oof,
        "folds":    fold_meta,
    }


# ============================================================
# PART C — Evaluation
# ============================================================

def print_evaluation(model: pd.DataFrame, baseline: dict, cv: dict) -> None:
    y          = cv["y"]
    lr_oof     = cv["lr_oof"]
    xgb_oof    = cv["xgb_oof"]
    n_pos      = int(y.sum())
    n_all      = len(y)
    base_rate  = n_pos / n_all

    # Transition rows: currently healthy, distressed next year
    # NaN distressed treated as False (not a transition), which is correct.
    trans_mask = ((model["distressed"] == 0) & (model[TARGET] == 1)).values
    n_trans    = int(trans_mask.sum())

    print("=" * 68)
    print("PART C -- Evaluation (pooled out-of-fold predictions)")
    print("=" * 68)

    # ---- AUC metrics -------------------------------------------------------
    lr_roc  = roc_auc_score(y, lr_oof)
    lr_pr   = average_precision_score(y, lr_oof)

    print(f"\n  Positive base rate: {n_pos}/{n_all} = {base_rate:.1%}")
    print(f"  PR-AUC of a random classifier (always predict base rate): {base_rate:.3f}")
    print()
    print(f"  {'Model':<24}  {'ROC-AUC':>8}  {'PR-AUC':>8}  Note")
    print(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*35}")
    print(f"  {'Logistic Regression':<24}  {lr_roc:>8.3f}  {lr_pr:>8.3f}")
    if HAS_XGB:
        xgb_roc = roc_auc_score(y, xgb_oof)
        xgb_pr  = average_precision_score(y, xgb_oof)
        print(f"  {'XGBoost':<24}  {xgb_roc:>8.3f}  {xgb_pr:>8.3f}")
    print(f"  {'Random baseline':<24}  {'   0.500':>8}  {base_rate:>8.3f}  lower bound")
    print()

    # ---- Precision / Recall / F1 at F1-optimal threshold -------------------
    lr_thresh = f1_optimal_threshold(y, lr_oof)
    lr_pred   = (lr_oof >= lr_thresh).astype(int)

    if HAS_XGB:
        xgb_thresh = f1_optimal_threshold(y, xgb_oof)
        xgb_pred   = (xgb_oof >= xgb_thresh).astype(int)

    print(f"  Precision / Recall / F1  (at F1-optimal threshold on OOF data)")
    print()
    print(f"  {'Model':<24}  {'Thresh':>7}  {'Acc':>6}  {'Prec':>6}  {'Recall':>7}  {'F1':>6}  Note")
    print(f"  {'-'*24}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*20}")

    print(f"  {'Persistence baseline':<24}  {'   0/1':>7}  "
          f"{baseline['acc']:>6.3f}  {baseline['prec']:>6.3f}  "
          f"{baseline['rec']:>7.3f}  {baseline['f1']:>6.3f}  "
          f"on {baseline['n_rows']} rows")
    print(f"  {'Logistic Regression':<24}  {lr_thresh:>7.3f}  "
          f"{accuracy_score(y, lr_pred):>6.3f}  "
          f"{precision_score(y, lr_pred, zero_division=0):>6.3f}  "
          f"{recall_score(y, lr_pred, zero_division=0):>7.3f}  "
          f"{f1_score(y, lr_pred, zero_division=0):>6.3f}")
    if HAS_XGB:
        print(f"  {'XGBoost':<24}  {xgb_thresh:>7.3f}  "
              f"{accuracy_score(y, xgb_pred):>6.3f}  "
              f"{precision_score(y, xgb_pred, zero_division=0):>6.3f}  "
              f"{recall_score(y, xgb_pred, zero_division=0):>7.3f}  "
              f"{f1_score(y, xgb_pred, zero_division=0):>6.3f}")
    print()

    # ---- Confusion matrices ------------------------------------------------
    cm_lr = confusion_matrix(y, lr_pred)
    print(f"  LR confusion matrix (actual rows, predicted cols):")
    print(f"              pred=0   pred=1")
    print(f"    actual=0   {cm_lr[0,0]:>5}    {cm_lr[0,1]:>5}")
    print(f"    actual=1   {cm_lr[1,0]:>5}    {cm_lr[1,1]:>5}")
    if HAS_XGB:
        cm_xgb = confusion_matrix(y, xgb_pred)
        print(f"  XGB confusion matrix:")
        print(f"              pred=0   pred=1")
        print(f"    actual=0   {cm_xgb[0,0]:>5}    {cm_xgb[0,1]:>5}")
        print(f"    actual=1   {cm_xgb[1,0]:>5}    {cm_xgb[1,1]:>5}")
    print()

    # ---- Transition recall — THE KEY TABLE ---------------------------------
    #
    # The persistence baseline cannot catch any healthy->distressed transition.
    # This is the one metric where a model built on financial ratios should
    # demonstrably add value — if the firm's ratios are deteriorating (rising
    # leverage, falling ROA), the model should assign elevated probability
    # even while the firm is still technically healthy.
    print(f"  Recall on {n_trans} TRANSITION rows (this=0 -> next=1)")
    print(f"  Baseline earns 0/{n_trans} here by construction.")
    print()
    print(f"  {'Model':<24}  {'Correct':>8}  {'Out of':>7}  {'Recall':>8}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*7}  {'-'*8}")
    print(f"  {'Persistence baseline':<24}  {0:>8}  {n_trans:>7}  {0.000:>8.3f}")

    lr_t_correct = int(lr_pred[trans_mask].sum())
    lr_t_recall  = lr_t_correct / n_trans
    print(f"  {'Logistic Regression':<24}  {lr_t_correct:>8}  {n_trans:>7}  {lr_t_recall:>8.3f}")

    if HAS_XGB:
        xgb_t_correct = int(xgb_pred[trans_mask].sum())
        xgb_t_recall  = xgb_t_correct / n_trans
        print(f"  {'XGBoost':<24}  {xgb_t_correct:>8}  {n_trans:>7}  {xgb_t_recall:>8.3f}")

    print()

    # Honest commentary based on results
    best_t_recall = lr_t_recall if not HAS_XGB else max(lr_t_recall, xgb_t_recall)
    if best_t_recall == 0:
        print(f"  ** Neither model improves on the baseline for transitions.")
        print(f"     With only {n_trans} transition rows across 10 firms in LOFO,")
        print(f"     the early-warning task is likely under-powered.")
        print(f"     The model's value is in probability ranking (AUC), not hard predictions.")
    elif best_t_recall < 0.3:
        print(f"  ** Transition recall is low ({best_t_recall:.1%}).")
        print(f"     The model adds some early-warning signal but is unreliable.")
        print(f"     Frame value primarily through ROC/PR-AUC.")
    else:
        print(f"  Model catches {best_t_recall:.1%} of transitions out-of-fold.")
        print(f"  This is genuine early-warning signal the persistence baseline lacks.")
    print()

    # ---- Per-firm breakdown ------------------------------------------------
    print(f"  Per-firm out-of-fold results")
    print()

    hdr = (f"  {'Firm':<8}  {'Rows':>5}  {'Pos':>4}  "
           f"{'LR_TP':>6}  {'LR_FP':>6}  {'LR_FN':>6}  {'LR_TN':>6}  {'LR_Rec':>7}")
    if HAS_XGB:
        hdr += f"  {'XG_TP':>6}  {'XG_FP':>6}  {'XG_FN':>6}  {'XG_TN':>6}  {'XG_Rec':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for firm in sorted(model["ticker"].unique()):
        fm = (model["ticker"] == firm).values
        yt  = y[fm]
        lp  = lr_pred[fm]

        ltp = int(((lp == 1) & (yt == 1)).sum())
        lfp = int(((lp == 1) & (yt == 0)).sum())
        lfn = int(((lp == 0) & (yt == 1)).sum())
        ltn = int(((lp == 0) & (yt == 0)).sum())
        lr_ = f"{ltp/yt.sum():.3f}" if yt.sum() > 0 else "  N/A"

        row = (f"  {firm:<8}  {fm.sum():>5}  {int(yt.sum()):>4}  "
               f"{ltp:>6}  {lfp:>6}  {lfn:>6}  {ltn:>6}  {lr_:>7}")

        if HAS_XGB:
            xp  = xgb_pred[fm]
            xtp = int(((xp == 1) & (yt == 1)).sum())
            xfp = int(((xp == 1) & (yt == 0)).sum())
            xfn = int(((xp == 0) & (yt == 1)).sum())
            xtn = int(((xp == 0) & (yt == 0)).sum())
            xr_ = f"{xtp/yt.sum():.3f}" if yt.sum() > 0 else "  N/A"
            row += f"  {xtp:>6}  {xfp:>6}  {xfn:>6}  {xtn:>6}  {xr_:>7}"

        print(row)

    print()


# ============================================================
# PART D — Interpretation (full-data models)
# ============================================================

def run_interpretation(cv: dict) -> None:
    """
    Fit models on ALL 138 rows for coefficient/SHAP analysis.

    Why full-data and not one of the LOFO folds?
    - Any single fold trains on ~124 rows and tests on ~14; coefficients
      from a 124-row fit are noisier than from a 138-row fit.
    - We already have honest performance estimates from OOF in Part C.
      Interpretation is about DIRECTION and MAGNITUDE, not performance.
    - This is the standard practice: cross-validate for performance claims,
      fit on all available data for model understanding.
    """
    y     = cv["y"]
    X_lr  = cv["X_lr"]
    X_xgb = cv["X_xgb"]

    print("=" * 68)
    print("PART D -- Interpretation (fit on all 138 rows; see Part C for OOF perf)")
    print("=" * 68)
    print()

    # --- Logistic Regression: standardized coefficients ---------------------
    lr_full = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])
    lr_full.fit(X_lr.values, y)

    lr_feature_names = FEATURES + ["current_ratio_missing"]
    coefs = lr_full.named_steps["clf"].coef_[0]

    # Expected economic sign for each feature:
    # leverage+          : more debt -> higher P(distress)
    # equity_ratio-      : more equity cushion -> lower P(distress)
    # retained_to_assets-: accumulated earnings -> lower P(distress)
    # roa-               : profitable -> lower P(distress)
    # margin-            : profitable -> lower P(distress)
    # current_ratio-     : more liquidity -> lower P(distress)
    expected = {
        "leverage":             "+",
        "equity_ratio":         "-",
        "retained_to_assets":   "-",
        "roa":                  "-",
        "margin":               "-",
        "current_ratio":        "-",
        "current_ratio_missing": "?",
    }

    coef_rows = sorted(
        zip(lr_feature_names, coefs),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    print(f"  Logistic Regression -- standardized coefficients")
    print(f"  (positive -> increases P(distressed_next=1), ranked by |coef|)")
    print()
    print(f"  {'Feature':<25}  {'Coef':>8}  {'Expected':>9}  {'Match?':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*9}  {'-'*8}")
    for feat, c in coef_rows:
        actual  = "+" if c > 0 else "-"
        exp     = expected.get(feat, "?")
        match   = "yes" if actual == exp else ("?" if exp == "?" else "** NO **")
        print(f"  {feat:<25}  {c:>+8.3f}  {exp:>9}  {match:>8}")
    print()

    if not HAS_XGB:
        print("  [XGBoost interpretation skipped -- xgboost not installed]")
        return

    # --- XGBoost: gain-based importance + SHAP ------------------------------
    # Fit with DataFrame so column names are embedded in the booster
    xgb_full = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    xgb_full.fit(X_xgb, y)   # DataFrame preserves feature names

    importance = xgb_full.get_booster().get_score(importance_type="gain")
    imp_total  = sum(importance.values()) or 1.0

    # Features the booster never split on get gain = 0
    all_feats  = FEATURES
    imp_sorted = sorted(
        [(f, importance.get(f, 0.0)) for f in all_feats],
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"  XGBoost -- feature importance (gain)")
    print(f"  (average information gain per split weighted by # of times used)")
    print()
    print(f"  {'Feature':<25}  {'Gain':>10}  {'% of total':>11}  Note")
    print(f"  {'-'*25}  {'-'*10}  {'-'*11}  {'-'*20}")
    for feat, gain in imp_sorted:
        note = "(never used)" if gain == 0 else ""
        print(f"  {feat:<25}  {gain:>10.1f}  {gain/imp_total:>11.1%}  {note}")
    print()

    if not HAS_SHAP:
        print("  [SHAP skipped -- pip install shap for directional analysis]")
        print()
        return

    # --- SHAP values --------------------------------------------------------
    #
    # TreeExplainer uses the exact tree structure to compute Shapley values —
    # no sampling, no approximation.  Each SHAP value answers:
    #   "How much did this feature's value push the model's prediction
    #    above/below the average prediction for this observation?"
    #
    # mean_pos = average SHAP for distressed firm-years (y=1)
    # mean_neg = average SHAP for healthy firm-years (y=0)
    # If mean_pos > mean_neg for feature X, then X tends to push predictions
    # higher (toward distress) for the actually-distressed firms — X is a
    # distress signal with the right direction.
    explainer  = shap.TreeExplainer(xgb_full)
    shap_vals  = explainer.shap_values(X_xgb.values)   # shape (138, 6)

    mean_abs   = np.abs(shap_vals).mean(axis=0)
    pos_mask   = y == 1
    mean_pos   = shap_vals[pos_mask].mean(axis=0)
    mean_neg   = shap_vals[~pos_mask].mean(axis=0)

    shap_rows = sorted(
        zip(FEATURES, mean_abs, mean_pos, mean_neg),
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"  XGBoost -- SHAP summary")
    print(f"  (mean_pos = avg SHAP for actually-distressed;")
    print(f"   mean_neg = avg SHAP for healthy)")
    print()
    print(f"  {'Feature':<25}  {'Mean|SHAP|':>10}  {'mean_pos':>10}  {'mean_neg':>10}  Direction")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*20}")
    for feat, ma, mp, mn in shap_rows:
        direction = "higher -> distress" if mp > mn else "lower  -> distress"
        print(f"  {feat:<25}  {ma:>10.4f}  {mp:>10.4f}  {mn:>10.4f}  {direction}")
    print()

    # --- LR vs XGB sign agreement -------------------------------------------
    #
    # A sign disagreement is a discussion point, not necessarily a bug.
    # LR is a linear model: it estimates a single global coefficient.
    # XGBoost is non-linear: SHAP values can show non-monotone relationships.
    # If they disagree on a feature, it may be that the feature has a
    # non-linear relationship with distress that LR cannot capture.
    lr_coef_map = dict(zip(lr_feature_names, coefs))

    print(f"  LR vs XGBoost -- sign agreement")
    print(f"  (disagreement = feature has non-linear or confounded relationship)")
    print()
    print(f"  {'Feature':<25}  {'LR coef':>9}  {'LR sign':>8}  {'XGB sign':>9}  {'Agree?':>8}")
    print(f"  {'-'*25}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*8}")
    for feat, ma, mp, mn in shap_rows:
        lr_c    = lr_coef_map.get(feat, 0.0)
        lr_sign = "+" if lr_c > 0 else "-"
        xg_sign = "+" if mp > mn else "-"
        agree   = "yes" if lr_sign == xg_sign else "DISAGREE"
        print(f"  {feat:<25}  {lr_c:>+9.3f}  {lr_sign:>8}  {xg_sign:>9}  {agree:>8}")
    print()


# ============================================================
# FINISH 5.1 — Matched-budget comparison (top-K)
# ============================================================

def print_budget_comparison(
        model: pd.DataFrame,
        y: np.ndarray,
        lr_oof: np.ndarray,
        xgb_oof: np.ndarray,
        budgets: tuple = (15, 25),
) -> None:
    """
    Rank all 138 firm-years by predicted probability and report TP, transition
    catches, and precision at fixed alert budgets K=15 and K=25.

    Why this replaces the per-threshold comparison in Part C:
    Part C evaluated each model at its own F1-optimal threshold (LR=0.309,
    XGB=0.047).  Those thresholds are not comparable — a lower threshold
    trivially improves recall at the cost of flooding the analyst with alerts.
    XGB's 100% transition recall in Part C came almost entirely from setting
    the alarm so low it fires for ~40% of all rows.

    Budget-matched comparison forces both models to flag the same K names.
    This is also the real operating mode: an analyst has a fixed capacity
    (say, top-15 or top-25 names per quarter) and wants to know how many
    of those flags are correct distress events.

    Precision@K = TP / K: "of my K alerts, how many are real?"
    Transition recall@K = transitions caught / 10: "how many onsets did I catch?"
    """
    trans_mask = ((model["distressed"] == 0) & (model[TARGET] == 1)).values
    n_trans    = int(trans_mask.sum())
    n_pos      = int(y.sum())
    n          = len(y)

    print("=" * 72)
    print("FINISH 5.1 -- Matched-budget comparison (top-K predicted-riskiest)")
    print("=" * 72)
    print()
    print(f"  {'Model':<24}  {'Budget':>7}  {'TP':>4}  {'FP':>4}  "
          f"{'Prec@K':>8}  {'Trans caught':>13}  {'Trans recall':>13}")
    print(f"  {'-'*24}  {'-'*7}  {'-'*4}  {'-'*4}  "
          f"{'-'*8}  {'-'*13}  {'-'*13}")

    for K in budgets:
        rand_tp = round(K * n_pos / n)
        rand_tr = round(K * n_trans / n)

        for label, probs in [("Logistic Regression", lr_oof), ("XGBoost", xgb_oof)]:
            top_k        = np.argsort(probs)[::-1][:K]
            tp           = int(y[top_k].sum())
            fp           = K - tp
            prec         = tp / K
            trans_caught = int(trans_mask[top_k].sum())
            t_recall     = trans_caught / n_trans
            print(f"  {label:<24}  {K:>7}  {tp:>4}  {fp:>4}  "
                  f"{prec:>8.3f}  {trans_caught:>6}/{n_trans:<6}  {t_recall:>13.3f}")

        print(f"  {'Random baseline':<24}  {K:>7}  {rand_tp:>4}  {K-rand_tp:>4}  "
              f"{n_pos/n:>8.3f}  {rand_tr:>6}/{n_trans:<6}  {n_trans/n:>13.3f}")
        print()

    # Annotated top-25 ranked list for LR
    print(f"  LR top-25 ranked firm-years:")
    print(f"  {'#':>3}  {'Ticker':<8}  {'FY':>4}  {'Prob':>6}  "
          f"{'dist_t':>7}  {'dist_t+1':>9}  {'Flag':>11}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*4}  {'-'*6}  "
          f"{'-'*7}  {'-'*9}  {'-'*11}")
    for rank, idx in enumerate(np.argsort(lr_oof)[::-1][:25], 1):
        row    = model.iloc[idx]
        d_t    = int(row["distressed"]) if pd.notna(row["distressed"]) else "?"
        d_tp1  = int(y[idx])
        flag   = "** TRANS **" if trans_mask[idx] else ""
        print(f"  {rank:>3}  {row['ticker']:<8}  {int(row['fy']):>4}  "
              f"{lr_oof[idx]:>6.3f}  {str(d_t):>7}  {d_tp1:>9}  {flag}")
    print()


# ============================================================
# FINISH 5.2 — LR without equity_ratio (collinearity fix)
# ============================================================

def refit_lr_no_equity_ratio(model: pd.DataFrame, original_pr_auc: float) -> np.ndarray:
    """
    Drop equity_ratio from LR and rerun LOFO.  Report PR-AUC delta and
    confirm all remaining coefficient signs are economically sensible.

    Why equity_ratio is redundant in a linear model:
      By the balance-sheet identity: assets = liabilities + equity, so
      liabilities/assets + equity/assets = 1 (exactly, when both are
      fully tagged in EDGAR; approximately otherwise due to NCI, goodwill, etc.).
      In other words, leverage = 1 - equity_ratio, give or take.

      In a linear model, including two features that are nearly perfectly
      correlated creates near-multicollinearity.  The matrix X'X approaches
      singularity; individual coefficients become high-variance and can flip
      sign while the fitted probabilities remain stable (the two compensate).
      This is why equity_ratio showed the "wrong" sign (+0.191) in Part D
      even though economically it should be negative.

      Dropping equity_ratio removes the redundancy.  leverage is the more
      direct economic measure (debt burden), so it stays.

      XGBoost is unaffected.  Tree splits are local and binary; two
      collinear features don't destabilise the fit because each node picks
      the better split at that step independently.
    """
    FEATURES_REDUCED = [f for f in FEATURES if f != "equity_ratio"]

    y      = model[TARGET].values
    groups = model["cik"].values

    X = model[FEATURES_REDUCED].copy()
    X["current_ratio_missing"] = X["current_ratio"].isna().astype(float)

    logo   = LeaveOneGroupOut()
    lr_oof = np.full(len(y), np.nan)

    for train_idx, test_idx in logo.split(X, y, groups):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ])
        pipe.fit(X.iloc[train_idx].values, y[train_idx])
        lr_oof[test_idx] = pipe.predict_proba(X.iloc[test_idx].values)[:, 1]

    pr_auc_reduced = average_precision_score(y, lr_oof)

    # Full-data fit for coefficient display
    full_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])
    full_pipe.fit(X.values, y)
    coefs      = full_pipe.named_steps["clf"].coef_[0]
    feat_names = FEATURES_REDUCED + ["current_ratio_missing"]

    expected = {
        "leverage":              "+",   # more debt -> more distress
        "retained_to_assets":    "-",   # accumulated profit -> less distress
        "roa":                   "-",   # profitable -> less distress
        "margin":                "-",   # profitable -> less distress
        "current_ratio":         "-",   # more liquidity -> less distress
        "current_ratio_missing": "?",
    }
    coef_rows = sorted(zip(feat_names, coefs), key=lambda x: abs(x[1]), reverse=True)

    print("=" * 68)
    print("FINISH 5.2 -- LR without equity_ratio  (collinearity fix)")
    print("=" * 68)
    print()
    print("  equity_ratio + leverage ~= 1 by balance-sheet identity.")
    print("  Including both in LR causes near-multicollinearity; coefficients")
    print("  become unstable and equity_ratio's sign flipped in Part D.")
    print("  XGBoost keeps all 6 features (trees unaffected by collinearity).")
    print()
    print(f"  {'LR version':<40}  {'PR-AUC':>8}")
    print(f"  {'-'*40}  {'-'*8}")
    print(f"  {'With equity_ratio (original, Part C)':<40}  {original_pr_auc:>8.3f}")
    print(f"  {'Without equity_ratio':<40}  {pr_auc_reduced:>8.3f}")
    change    = pr_auc_reduced - original_pr_auc
    direction = "improvement" if change > 0 else "decline"
    print(f"  Change: {change:>+.3f}  ({direction})")
    print()
    print(f"  Standardized coefficients (equity_ratio removed):")
    print()
    print(f"  {'Feature':<25}  {'Coef':>8}  {'Expected':>9}  {'Match?':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*9}  {'-'*8}")
    for feat, c in coef_rows:
        actual = "+" if c > 0 else "-"
        exp    = expected.get(feat, "?")
        match  = "yes" if actual == exp else ("?" if exp == "?" else "** NO **")
        print(f"  {feat:<25}  {c:>+8.3f}  {exp:>9}  {match:>8}")
    print()

    return lr_oof


# ============================================================
# Save OOF predictions to DB for the Streamlit dashboard
# ============================================================

def save_oof_predictions(model: pd.DataFrame, cv: dict) -> None:
    """
    Persist out-of-fold probability estimates so the Streamlit app (app.py)
    can display them without refitting the model at runtime.

    One row per firm-year in the modelling table (138 rows).
    Rows in annual_labeled where distressed_next is NaN are NOT in the
    modelling table and therefore have no OOF prediction.
    """
    oof_df = model[["cik", "ticker", "fy"]].copy()
    oof_df["lr_prob"]  = cv["lr_oof"]
    oof_df["xgb_prob"] = cv["xgb_oof"] if cv["xgb_oof"] is not None else float("nan")

    conn = sqlite3.connect(DB_PATH)
    oof_df.to_sql("oof_predictions", conn, if_exists="replace", index=False)
    conn.close()
    print(f"  Saved {len(oof_df)} OOF rows to 'oof_predictions' in {DB_PATH}")
    print()


# ============================================================
# Main
# ============================================================

def main() -> None:
    conn  = sqlite3.connect(DB_PATH)
    model = load_model_data(conn)
    conn.close()

    y       = model[TARGET].values
    n_pos   = int(y.sum())
    n_all   = len(y)
    print(f"Modelling table: {n_all} rows, {n_pos} positives ({n_pos/n_all:.1%} base rate)")
    print()

    # ---- PART A: Baseline --------------------------------------------------
    print("=" * 68)
    print("PART A -- Persistence baseline  (predict next = this)")
    print("=" * 68)
    print()
    print("  Rule: distressed_next_pred = distressed_this")
    print("  Operates on 132 rows (6 rows have distressed_this = NaN)")
    print("  Cannot catch healthy->distressed transitions by construction.")
    print()

    baseline = run_baseline(model)
    cm = baseline["cm"]

    print(f"  {'Accuracy':<12}  {baseline['acc']:.3f}")
    print(f"  {'Precision':<12}  {baseline['prec']:.3f}")
    print(f"  {'Recall':<12}  {baseline['rec']:.3f}")
    print(f"  {'F1':<12}  {baseline['f1']:.3f}")
    print()
    print(f"  Confusion matrix:")
    print(f"              pred=0   pred=1")
    print(f"    actual=0   {cm[0,0]:>5}    {cm[0,1]:>5}")
    print(f"    actual=1   {cm[1,0]:>5}    {cm[1,1]:>5}")
    print()
    n_trans = int(((model["distressed"] == 0) & (model[TARGET] == 1)).sum())
    print(f"  Recall on {n_trans} transition rows (this=0 -> next=1):  0/{n_trans} = 0.000")
    print(f"  This is the bar every model must beat to claim early-warning value.")
    print()

    # ---- PART B: LOFO CV ---------------------------------------------------
    print("=" * 68)
    print("PART B -- Leave-one-firm-out CV  (training all 10 folds)")
    print("=" * 68)
    print()

    cv = run_lofo_cv(model)

    print("  Fold summary:")
    for fold in cv["folds"]:
        print(f"    held-out={fold['firm']:<8}  rows={fold['n_rows']:>3}  positives={fold['n_pos']:>2}")
    print()

    # ---- PART C: Evaluation ------------------------------------------------
    print_evaluation(model, baseline, cv)

    # ---- PART D: Interpretation --------------------------------------------
    run_interpretation(cv)

    # ---- FINISH STEP 5 -----------------------------------------------------
    lr_orig_pr_auc = average_precision_score(y, cv["lr_oof"])
    print_budget_comparison(model, y, cv["lr_oof"], cv["xgb_oof"])
    refit_lr_no_equity_ratio(model, lr_orig_pr_auc)
    save_oof_predictions(model, cv)


if __name__ == "__main__":
    main()
