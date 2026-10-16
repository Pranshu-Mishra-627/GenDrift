import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from core.shap_tracker import track_shap_importance, summarize_shap_drift


def main():
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)
    g0 = track_shap_importance(model, X, 0, max_samples=30)
    g1 = track_shap_importance(model, X, 1, max_samples=30)
    assert len(g0) == X.shape[1]
    assert abs(g0.normalized_importance.sum() - 1.0) < 1e-8
    drift = summarize_shap_drift(pd.concat([g0, g1], ignore_index=True))
    assert len(drift) == 2 * X.shape[1]
    print("SHAP tracker tests passed.")


if __name__ == "__main__":
    main()
