"""Run and persist one offline GenDrift experiment."""
import json
from pathlib import Path
import pandas as pd
from core.data_handler import load_dataset, split_dataset
from core.trainer import run_experiment


def run_offline_experiment(
    dataset_path, target_column, generations=10, real_ratio=0.7,
    output_dir="results", dataset_name=None, optimize=False, n_trials=20,
    track_shap=False, max_shap_samples=200, random_state=42,
):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    name = dataset_name or dataset_path.stem
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_dataset(str(dataset_path))
    train_df, test_df = split_dataset(df, target_column, random_state=random_state)
    results, synthetic_generations = run_experiment(
        train_df, test_df, target_column,
        generations=generations, real_ratio=real_ratio,
        random_state=random_state, track_shap=track_shap,
        max_shap_samples=max_shap_samples, optimize=optimize,
        n_trials=n_trials,
        return_synthetic=True,
    )
    synthetic_dir = out / "synthetic_data"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    synthetic_files = {}
    for generation, synthetic_df in synthetic_generations.items():
        synthetic_path = synthetic_dir / f"{name}_generation_{generation}_synthetic.csv"
        synthetic_df.to_csv(synthetic_path, index=False)
        synthetic_files[str(generation)] = str(synthetic_path)

    csv_path = out / f"{name}_generation_metrics.csv"
    json_path = out / f"{name}_generation_metrics.json"
    csv_df = pd.DataFrame(results)
    if "shap_importance" in csv_df.columns:
        csv_df["shap_importance"] = csv_df["shap_importance"].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else x
        )
    if "optimization_best_params" in csv_df.columns:
        csv_df["optimization_best_params"] = csv_df["optimization_best_params"].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else x
        )
    csv_df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    shap_csv = None
    if track_shap:
        from core.shap_tracker import summarize_shap_drift
        rows = []
        for item in results:
            for feature_row in item.get("shap_importance", []):
                rows.append({"generation": item["generation"], **feature_row})
        if rows:
            shap_df = pd.DataFrame(rows)
            shap_csv = out / f"{name}_shap_history.csv"
            shap_df.to_csv(shap_csv, index=False)
            summarize_shap_drift(shap_df).to_csv(out / f"{name}_shap_drift.csv", index=False)

    optimizations = [
        {"generation": item["generation"],
         "best_score": item.get("optimization_best_score"),
         "best_params": item.get("optimization_best_params")}
        for item in results if "optimization_best_score" in item
    ] if optimize else None
    return {
        "dataset": name,
        "results": results,
        "metrics_csv": str(csv_path),
        "metrics_json": str(json_path),
        "shap_csv": str(shap_csv) if shap_csv else None,
        "optimization": optimizations,
        "synthetic_files": synthetic_files,
    }


def _json_default(value):
    try:
        return value.item()
    except AttributeError:
        return str(value)
