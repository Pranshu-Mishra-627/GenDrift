"""Check that optimization is invoked once for G0 and each later generation."""
from unittest.mock import patch
from sklearn.datasets import load_iris
from core.data_handler import split_dataset
import core.trainer as trainer


def main():
    iris = load_iris(as_frame=True)
    df = iris.data.copy()
    df["target"] = iris.target
    train_df, test_df = split_dataset(df, "target", random_state=42)
    calls = []

    def fake_optimize(data, target_column, optimize, n_trials, random_state, fallback_params):
        calls.append({"rows": len(data), "seed": random_state, "enabled": optimize})
        if not optimize:
            return fallback_params, None
        params = {"n_estimators": 10, "max_depth": None, "random_state": random_state}
        return params, {"best_score": 0.5, "best_params": params}

    def fake_synthesize(data, num_rows, random_state=42):
        return data.sample(n=num_rows, replace=True, random_state=random_state).reset_index(drop=True)

    with patch.object(trainer, "_optimize_params", side_effect=fake_optimize), \
         patch.object(trainer, "generate_synthetic_data", side_effect=fake_synthesize):
        results = trainer.run_experiment(
            train_df, test_df, "target", generations=3,
            random_state=42, optimize=True, n_trials=1,
        )

    assert len(results) == 3
    assert len(calls) == 3, f"Expected G0, G1 and G2 optimization calls, got {len(calls)}"
    assert [r["generation"] for r in results] == [0, 1, 2]
    assert all("optimization_best_score" in r for r in results)
    print("Per-generation optimization tests passed.")


if __name__ == "__main__":
    main()
