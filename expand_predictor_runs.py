"""Generate additional GenDrift trajectories for LSTM predictor evaluation.

Run from the GenDrift project root:
    python expand_predictor_runs.py

The dataset split and experiment settings match the existing controlled
multi-seed experiment. Existing seed results are skipped by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.data_handler import load_dataset, split_dataset
from core.trainer import run_experiment

DATA_PATH = Path("data/bank_marketing/bank_marketing_no_duration.csv")
TARGET = "y"
SPLIT_SEED = 42
DEFAULT_SEEDS = [314, 2718, 8080, 17, 555, 9001, 2027, 31415, 8675309, 1001]
DEFAULT_GENERATIONS = 10
REAL_RATIO = 0.7
OUTPUT_ROOT = Path("results/bank_marketing_controlled_multiseed")


def find_existing_result(seed: int) -> Path | None:
    """Support both the current results_s<seed>.json and older results.json names."""
    seed_dir = OUTPUT_ROOT / f"seed_{seed}"
    candidates = [seed_dir / f"results_s{seed}.json", seed_dir / "results.json"]
    candidates.extend([Path(f"results_s{seed}.json"), Path(f"results_seed_{seed}.json")])
    for candidate in candidates:
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(payload, list) and payload:
                    return candidate
            except (OSError, json.JSONDecodeError):
                continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Additional experiment seeds to run (default: 10 predefined seeds).",
    )
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--overwrite", action="store_true", help="Rerun seeds even if results exist.")
    args = parser.parse_args()

    if args.generations < 4:
        parser.error("--generations must be at least 4 for useful sequence training.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates.")
    if any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be non-negative integers.")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}\n"
            "Run this script from the GenDrift project root and verify the dataset path."
        )

    print(f"Loading dataset: {DATA_PATH}")
    df = load_dataset(str(DATA_PATH))
    print(f"Dataset shape: {df.shape}")
    print(f"Creating the same fixed train/test split (seed={SPLIT_SEED})")
    train_df, test_df = split_dataset(df, TARGET, test_size=0.2, random_state=SPLIT_SEED)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    completed, skipped, failed = [], [], []
    for index, seed in enumerate(args.seeds, start=1):
        existing = find_existing_result(seed)
        if existing and not args.overwrite:
            print(f"[{index}/{len(args.seeds)}] SKIP seed {seed}: existing results at {existing}")
            skipped.append(seed)
            continue

        seed_dir = OUTPUT_ROOT / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        json_path = seed_dir / f"results_s{seed}.json"
        csv_path = seed_dir / f"results_s{seed}.csv"
        print(f"\n[{index}/{len(args.seeds)}] Running seed {seed} for {args.generations} generations")
        try:
            results = run_experiment(
                train_df=train_df,
                test_df=test_df,
                target_column=TARGET,
                generations=args.generations,
                real_ratio=REAL_RATIO,
                random_state=seed,
                optimize=False,
                track_shap=False,
            )
            json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
            try:
                import pandas as pd
                pd.DataFrame(results).to_csv(csv_path, index=False)
            except Exception as exc:  # JSON remains the canonical output.
                print(f"CSV export skipped ({exc}); JSON was saved.")
            completed.append(seed)
            print(f"Saved {json_path}")
        except Exception as exc:
            failed.append({"seed": seed, "error": repr(exc)})
            print(f"FAILED seed {seed}: {exc}")

    summary = {
        "dataset": str(DATA_PATH),
        "target": TARGET,
        "split_seed": SPLIT_SEED,
        "generations": args.generations,
        "real_ratio": REAL_RATIO,
        "completed_seeds": completed,
        "skipped_seeds": skipped,
        "failed": failed,
        "note": (
            "Trajectories share one fixed train/test split. They help measure variability "
            "across generation seeds but are not independent datasets."
        ),
    }
    summary_path = OUTPUT_ROOT / "predictor_run_expansion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nExpansion summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
