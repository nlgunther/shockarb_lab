"""
marketfit.model — Learned logistic regression for ShockArb-fit prediction.

NOT YET IMPLEMENTED — stub only. The rule engine (rules.py) is the active
primary path. This module is a placeholder for the ML layer.

Design (to implement once ≥30 labeled days exist)
--------------------------------------------------
  - sklearn.linear_model.LogisticRegression on the features.py feature spec
  - Persisted to data/marketfit/fit_model.joblib
  - Sidecar JSON: coefficients, feature names, n_rows, date_range, CV_AUC
  - is_usable() guard: file loads + n_rows>=30 + CV_AUC>=0.55 + no disqualifying NaNs
  - Output: p_favourable → GOOD (≥0.6) / CAUTION (0.4–0.6) / POOR (<0.4)
  - When usable: report shows learned verdict as primary, rule verdict as cross-check
"""

from __future__ import annotations


MIN_TRAIN_ROWS = 30
MIN_CV_AUC     = 0.55
P_GOOD         = 0.60
P_POOR         = 0.40


def is_usable(model_path: str) -> bool:
    """Return True only if the learned model meets all usability criteria."""
    return False   # always False until implemented


def predict(feats: dict, model_path: str) -> dict:
    """
    Return learned verdict dict with p_favourable and verdict label.

    NOT YET IMPLEMENTED.
    """
    raise NotImplementedError(
        "model.predict is not yet implemented. "
        "Use rules.evaluate() for the current production path."
    )


def train(training_parquet: str, out_model_path: str) -> dict:
    """
    Fit and save the logistic regression model.

    NOT YET IMPLEMENTED.
    """
    raise NotImplementedError(
        "model.train is not yet implemented. "
        f"Requires ≥{MIN_TRAIN_ROWS} labeled days in training_parquet."
    )
