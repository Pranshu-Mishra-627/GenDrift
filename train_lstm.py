"""Evaluate the residual LSTM with leave-one-run-out CV, then fit final artifacts.

Run from the GenDrift project root:
    python train_lstm.py

Expected inputs in the project root: results_s7.json, results_s42.json,
results_s99.json, results_s123.json, results_s2026.json.
The new candidate is saved separately so the existing model is preserved.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.collapse_predictor import (
    evaluate_lstm_leave_one_run_out,
    fit_lstm_predictor_on_all_runs,
    save_lstm_predictor,
)

SEEDS = [7, 42, 99, 123, 2026]
SEQUENCE_LENGTH = 3
HORIZON = 1
MAX_EPOCHS = 100
BATCH_SIZE = 8
PATIENCE = 10
RANDOM_STATE = 42
OUTPUT_DIR = Path("results/lstm_collapse_predictor_v2")


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def load_histories():
    histories = []
    for seed in SEEDS:
        path = Path(f"results\\bank_marketing_controlled_multiseed\\seed_{seed}\\results_s{seed}.json")
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {path.resolve()}. Put the five results_s<seed>.json "
                "files in the GenDrift project root or update train_lstm.py paths."
            )
        with path.open("r", encoding="utf-8") as file:
            rows = json.load(file)
        if not isinstance(rows, list) or len(rows) < SEQUENCE_LENGTH + HORIZON:
            raise ValueError(f"{path} does not contain enough generation rows.")
        histories.append({"run_id": f"seed_{seed}", "results": rows})
        print(f"Loaded {path}: {len(rows)} generations")
    return histories


def main():
    histories = load_histories()

    print("\nRunning leave-one-run-out evaluation...")
    cv_metrics = evaluate_lstm_leave_one_run_out(
        histories=histories,
        sequence_length=SEQUENCE_LENGTH,
        horizon=HORIZON,
        random_state=RANDOM_STATE,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
    )

    print("\nCross-validation summary:")
    print(json.dumps(make_json_safe({
        key: value for key, value in cv_metrics.items() if key != "folds"
    }), indent=2))
    print("\nPer-run folds:")
    print(json.dumps(make_json_safe(cv_metrics["folds"]), indent=2))

    final_epochs = max(5, int(cv_metrics["median_best_epoch"]))
    print(f"\nFitting final model on all five runs for {final_epochs} epochs...")
    predictor = fit_lstm_predictor_on_all_runs(
        histories=histories,
        sequence_length=SEQUENCE_LENGTH,
        horizon=HORIZON,
        epochs=final_epochs,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
        metrics=cv_metrics,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = save_lstm_predictor(predictor, OUTPUT_DIR)
    evaluation_path = OUTPUT_DIR / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(make_json_safe(cv_metrics), indent=2), encoding="utf-8"
    )
    report = {
        "evaluation": str(evaluation_path),
        "artifacts": saved,
        "final_training_epochs": final_epochs,
        "note": (
            "Leave-one-run-out evaluation is still preliminary: the five trajectories "
            "share the same underlying dataset split and are not independent datasets. "
            "Compare aggregate MAE with persistence before claiming predictive benefit."
        ),
    }
    print("\nCompleted. Original model artifacts were not overwritten.")
    print(json.dumps(make_json_safe(report), indent=2))


if __name__ == "__main__":
    main()
