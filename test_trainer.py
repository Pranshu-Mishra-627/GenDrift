from core.data_handler import load_dataset, split_dataset
from core.trainer import run_experiment


df = load_dataset("data/iris.csv")

train_df, test_df = split_dataset(
    df,
    target_column="target",
)

results = run_experiment(
    train_df,
    test_df,
    target_column="target",
    generations=10,
    real_ratio=0.7,
)

print("\nGeneration Results:")

for result in results:
    print(
        f"G{result['generation']} | "
        f"F1: {result['f1']:.3f} | "
        f"PI: {result['performance_index']:.3f} | "
        f"DI: {result['distribution_index']:.3f} | "
        f"CI: {result['collapse_index']:.3f}"
    )

assert len(results) == 10

g0 = results[0]

assert g0["generation"] == 0
assert g0["performance_index"] == 0.0
assert g0["distribution_index"] == 0.0
assert g0["collapse_index"] == 0.0

for result in results[1:]:
    pi = result["performance_index"]
    di = result["distribution_index"]
    ci = result["collapse_index"]

    assert 0.0 <= pi <= 1.0
    assert 0.0 <= di <= 1.0
    assert 0.0 <= ci <= 1.0

    expected_ci = 0.5 * pi + 0.5 * di
    assert abs(ci - expected_ci) < 1e-9

    assert result["class_balance"] is not None
    assert result["wasserstein"] is not None

    assert 0.0 <= result["class_balance"] <= 1.0
    assert result["wasserstein"] >= 0.0

print("\nTrainer architecture tests passed.")
