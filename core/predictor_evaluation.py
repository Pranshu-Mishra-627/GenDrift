"""Evaluation helpers for the GenDrift CI predictor."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, baseline_pred=None) -> dict:
    """Return regression metrics and optionally compare against persistence."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must have the same non-zero length.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("y_true and y_pred must contain only finite values.")

    result = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n_test_samples": int(len(y_true)),
    }
    result["r2"] = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None

    if baseline_pred is not None:
        baseline_pred = np.asarray(baseline_pred, dtype=float).reshape(-1)
        if len(baseline_pred) != len(y_true):
            raise ValueError("baseline_pred must have the same length as y_true.")
        if not np.isfinite(baseline_pred).all():
            raise ValueError("baseline_pred must contain only finite values.")
        baseline_mae = float(mean_absolute_error(y_true, baseline_pred))
        result["persistence_mae"] = baseline_mae
        result["persistence_rmse"] = float(np.sqrt(mean_squared_error(y_true, baseline_pred)))
        result["beats_persistence_mae"] = bool(result["mae"] < baseline_mae)
        result["mae_vs_persistence_ratio"] = (
            float(result["mae"] / baseline_mae) if baseline_mae > 0 else None
        )
        result["mae_improvement_percent"] = (
            float(100.0 * (baseline_mae - result["mae"]) / baseline_mae)
            if baseline_mae > 0 else None
        )
    return result
