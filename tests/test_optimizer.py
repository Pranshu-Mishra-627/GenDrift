from sklearn.datasets import load_iris
from core.optimizer import optimize_random_forest


def main():
    data = load_iris(as_frame=True)
    df = data.data.copy()
    df["target"] = data.target
    result = optimize_random_forest(df, "target", n_trials=1, random_state=42)
    assert "best_params" in result and "best_score" in result
    assert 0 <= result["best_score"] <= 1

    mixed = df.iloc[:].copy()
    mixed["petal_type"] = ["wide" if x > 1.5 else "narrow" for x in mixed["petal width (cm)"]]
    mixed.loc[mixed.index[:3], "petal_type"] = None
    mixed_result = optimize_random_forest(mixed, "target", n_trials=1, random_state=42)
    assert 0 <= mixed_result["best_score"] <= 1
    print("Optimizer tests passed for numeric and mixed-type data.")
    print("Best validation F1:", result["best_score"])


if __name__ == "__main__":
    main()
