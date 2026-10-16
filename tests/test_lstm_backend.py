import numpy as np

from core.sequence_builder import build_sequence_dataset, normalize_histories, FEATURE_COLUMNS
from core.predictor_evaluation import regression_metrics


def make_rows(offset=0, count=10):
    rows = []
    for generation in range(count):
        ci = min(0.9, 0.02 * generation + offset)
        rows.append({
            "generation": generation,
            "accuracy": 0.9 - 0.001 * generation,
            "precision": 0.88 - 0.001 * generation,
            "recall": 0.9 - 0.001 * generation,
            "f1": 0.89 - 0.002 * generation,
            "diversity": 3.5 - 0.01 * generation,
            "class_balance": None if generation == 0 else 0.01 * generation,
            "wasserstein": None if generation == 0 else 0.2,
            "performance_index": 0.001 * generation,
            "distribution_index": 0.03 + 0.001 * generation,
            "collapse_index": ci,
        })
    return rows


def test_sequence_builder_keeps_runs_separate():
    histories = {"alpha": make_rows(0.0), "beta": make_rows(0.01)}
    data = build_sequence_dataset(histories, sequence_length=3, horizon=1)
    assert data["X"].shape == (14, 3, len(FEATURE_COLUMNS))
    assert len(set(data["run_ids"])) == 2
    assert data["y"].shape == (14,)
    assert np.isfinite(data["X"]).all()


def test_single_history_input_supported():
    data = build_sequence_dataset(make_rows(), sequence_length=3, horizon=1)
    assert data["X"].shape[0] == 7
    assert set(data["run_ids"]) == {"run_0"}


def test_regression_metrics_and_baseline():
    metrics = regression_metrics([0.1, 0.2], [0.15, 0.18], baseline_pred=[0.1, 0.2])
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["persistence_mae"] >= 0
    assert isinstance(metrics["beats_persistence_mae"], bool)


def test_regression_metrics_reports_relative_baseline_performance():
    metrics = regression_metrics(
        [0.10, 0.20, 0.30],
        [0.11, 0.19, 0.31],
        baseline_pred=[0.12, 0.22, 0.28],
    )
    assert metrics["mae"] < metrics["persistence_mae"]
    assert metrics["beats_persistence_mae"] is True
    assert metrics["mae_improvement_percent"] > 0
    assert metrics["mae_vs_persistence_ratio"] < 1


def test_regression_metrics_rejects_non_finite_predictions():
    import pytest

    with pytest.raises(ValueError, match="finite"):
        regression_metrics([0.1, 0.2], [0.1, float("nan")])
