
import json
from pathlib import Path

import pandas as pd

from core.data_handler import load_dataset, split_dataset
from core.trainer import run_experiment


DATA_PATH = "data/bank_marketing/bank_marketing_no_duration.csv"
TARGET = "y"

SPLIT_SEED = 42
EXPERIMENT_SEEDS = [42, 123, 2026, 7, 99]

GENERATIONS = 10
REAL_RATIO = 0.7

OUTPUT_DIR = Path("results/bank_marketing_controlled_multiseed")


def main():
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Creating fixed train/test split using seed {SPLIT_SEED}...")

    train_df, test_df = split_dataset(
        df,
        TARGET,
        test_size=0.2,
        random_state=SPLIT_SEED,
    )

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Experiment seeds: {EXPERIMENT_SEEDS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in EXPERIMENT_SEEDS:
        seed_dir = OUTPUT_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 50}")
        print(f"Running experiment with seed {seed}")
        print(f"{'=' * 50}")

        results = run_experiment(
            train_df=train_df,
            test_df=test_df,
            target_column=TARGET,
            generations=GENERATIONS,
            real_ratio=REAL_RATIO,
            random_state=seed,
            optimize=False,
            track_shap=False,
        )

        # Support either a DataFrame or a list of result dictionaries.
        if isinstance(results, pd.DataFrame):
            serializable_results = results.to_dict(orient="records")
            results_df = results
        else:
            serializable_results = results
            results_df = pd.DataFrame(results)

        json_path = seed_dir / "results.json"
        csv_path = seed_dir / "results.csv"

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(serializable_results, file, indent=2, default=str)

        results_df.to_csv(csv_path, index=False)

        print(f"Saved: {json_path}")
        print(f"Saved: {csv_path}")
        print(f"Result rows: {len(results_df)}")

    print("\nAll controlled experiments completed.")
    print(f"Results folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
