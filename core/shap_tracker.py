"""SHAP feature-reliance tracking across generations."""
import numpy as np
import pandas as pd


def calculate_shap_importance(model, X, max_samples=200, random_state=42):
    if len(X) == 0:
        raise ValueError("X must contain at least one row.")
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1.")
    if not hasattr(X, "columns"):
        raise ValueError("X must be a pandas DataFrame with named columns.")

    names = list(X.columns)
    sample = X if len(X) <= max_samples else X.sample(
        n=max_samples, random_state=random_state
    )

    import shap
    values = shap.TreeExplainer(model).shap_values(sample)
    if hasattr(values, "values"):
        values = values.values

    if isinstance(values, list):
        scores = np.mean(
            [np.abs(np.asarray(v)).mean(axis=0) for v in values], axis=0
        )
    else:
        values = np.abs(np.asarray(values))
        if values.ndim == 2 and values.shape[1] == len(names):
            scores = values.mean(axis=0)
        elif values.ndim == 3 and values.shape[1] == len(names):
            scores = values.mean(axis=(0, 2))  # rows, features, classes
        elif values.ndim == 3 and values.shape[2] == len(names):
            scores = values.mean(axis=tuple(i for i in range(3) if i != 2))
        else:
            raise ValueError(f"Unsupported SHAP output shape: {values.shape}")

    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) != len(names):
        raise ValueError("SHAP output feature count does not match X.")
    total = scores.sum()
    normalized = scores / total if total > 0 else np.zeros_like(scores)

    return pd.DataFrame({
        "feature": names,
        "mean_abs_shap": scores,
        "normalized_importance": normalized,
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def track_shap_importance(model, X, generation, max_samples=200, random_state=42):
    result = calculate_shap_importance(model, X, max_samples, random_state)
    result.insert(0, "generation", int(generation))
    return result


def summarize_shap_drift(history):
    required = {"generation", "feature", "normalized_importance"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    baseline = history.loc[history.generation == 0,
                           ["feature", "normalized_importance"]]
    baseline = baseline.drop_duplicates("feature").rename(
        columns={"normalized_importance": "baseline_importance"}
    )
    if baseline.empty:
        raise ValueError("SHAP history must contain generation 0.")
    result = history.merge(baseline, on="feature", how="left")
    result["importance_change"] = (
        result["normalized_importance"] - result["baseline_importance"]
    )
    result["absolute_importance_change"] = result["importance_change"].abs()
    return result.sort_values(
        ["generation", "absolute_importance_change"],
        ascending=[True, False], ignore_index=True
    )
