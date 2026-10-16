"""Evaluate the v2 residual LSTM on an expanded set of experiment runs.

Run from the GenDrift project root after running expand_predictor_runs.py:
    python train_lstm_v3.py

This script reuses the existing v2 model implementation and saves new artifacts
under results/lstm_collapse_predictor_v3_expanded. It does not overwrite v2.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core.collapse_predictor import (
    evaluate_lstm_leave_one_run_out,
    fit_lstm_predictor_on_all_runs,
    save_lstm_predictor,
)

RESULTS_ROOT = Path("results/bank_marketing_controlled_multiseed")
OUTPUT_DIR = Path("results/lstm_collapse_predictor_v3_expanded")
SEQUENCE_LENGTH = 3
HORIZON = 1
MAX_EPOCHS = 100
BATCH_SIZE = 8
PATIENCE = 10
RANDOM_STATE = 42


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _seed_from_path(path: Path) -> str | None:
    candidates = [path.stem, path.parent.name]
    for candidate in candidates:
        match = re.search(r"(?:results_s|results_seed_|seed_)(\d+)$", candidate)
        if match:
            return match.group(1)
    return None


def _load_json_rows(path: Path) -> list[dict] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        payload = payload["results"]
    if isinstance(payload, list) and len(payload) >= SEQUENCE_LENGTH + HORIZON:
        if all(isinstance(row, dict) for row in payload):
            return payload
    return None


def discover_histories() -> list[dict]:
    """Discover run JSONs while avoiding duplicate copies of the same seed."""
    candidates: list[Path] = []
    if RESULTS_ROOT.exists():
        candidates.extend(RESULTS_ROOT.rglob("results_s*.json"))
        candidates.extend(RESULTS_ROOT.rglob("results.json"))
    candidates.extend(Path(".").glob("results_s*.json"))
    candidates.extend(Path(".").glob("results_seed_*.json"))

    # Prefer explicitly seed-named files over generic results.json files.
    candidates = sorted(set(candidates), key=lambda p: (p.name == "results.json", str(p)))
    by_seed: dict[str, dict] = {}
    for path in candidates:
        seed = _seed_from_path(path)
        if seed is None:
            continue
        rows = _load_json_rows(path)
        if rows is None:
            print(f"Ignoring malformed/short result file: {path}")
            continue
        run_id = f"seed_{seed}"
        if run_id not in by_seed:
            by_seed[run_id] = {"run_id": run_id, "results": rows, "source_file": str(path)}

    histories = [
        {"run_id": run["run_id"], "results": run["results"]}
        for _, run in sorted(by_seed.items(), key=lambda item: int(item[0].split("_")[-1]))
    ]
    print(f"Discovered {len(histories)} unique experiment runs.")
    for _, run in sorted(by_seed.items(), key=lambda item: int(item[0].split("_")[-1])):
        print(f"  {run['run_id']}: {len(run['results'])} generations ({run['source_file']})")
    if len(histories) < 4:
        raise RuntimeError(
            f"Found only {len(histories)} usable runs. At least 4 are required. "
            "Check results/bank_marketing_controlled_multiseed/seed_*/."
        )
    return histories


def main() -> None:
    histories = discover_histories()
    print("\nRunning leave-one-run-out evaluation. This may take a few minutes...")
    cv_metrics = evaluate_lstm_leave_one_run_out(
        histories=histories,
        sequence_length=SEQUENCE_LENGTH,
        horizon=HORIZON,
        random_state=RANDOM_STATE,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
    )

    folds = cv_metrics.get("folds", [])
    fold_wins = sum(bool(fold.get("beats_persistence_mae")) for fold in folds)
    fold_maes = [float(fold["mae"]) for fold in folds if "mae" in fold]
    fold_baselines = [float(fold["persistence_mae"]) for fold in folds if "persistence_mae" in fold]
    cv_metrics["folds_beating_persistence"] = fold_wins
    cv_metrics["folds_total"] = len(folds)
    cv_metrics["fold_mae_mean"] = sum(fold_maes) / len(fold_maes) if fold_maes else None
    cv_metrics["fold_mae_min"] = min(fold_maes) if fold_maes else None
    cv_metrics["fold_mae_max"] = max(fold_maes) if fold_maes else None
    cv_metrics["fold_persistence_mae_mean"] = (
        sum(fold_baselines) / len(fold_baselines) if fold_baselines else None
    )
    cv_metrics["data_scope_note"] = (
        "All trajectories use the same underlying fixed train/test split. This CV measures "
        "generalization across generation seeds, not across independent datasets."
    )

    print("\nCross-validation summary:")
    print(json.dumps(json_safe({k: v for k, v in cv_metrics.items() if k != "folds"}), indent=2))
    print("\nPer-run fold results:")
    print(json.dumps(json_safe(folds), indent=2))

    final_epochs = max(5, int(cv_metrics["median_best_epoch"]))
    print(f"\nFitting final model on all {len(histories)} runs for {final_epochs} epochs...")
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
    evaluation_path.write_text(json.dumps(json_safe(cv_metrics), indent=2), encoding="utf-8")
    run_manifest = {
        "runs": [
            {"run_id": item["run_id"], "generations": len(item["results"])}
            for item in histories
        ],
        "n_runs": len(histories),
        "sequence_length": SEQUENCE_LENGTH,
        "horizon": HORIZON,
        "final_training_epochs": final_epochs,
        "evaluation_file": str(evaluation_path),
        "artifacts": saved,
        "v2_preserved": True,
        "status": "experimental",
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(json_safe(run_manifest), indent=2), encoding="utf-8"
    )
    print("\nCompleted. v2 artifacts were not overwritten.")
    print(json.dumps(json_safe(run_manifest), indent=2))


if __name__ == "__main__":
    main()
