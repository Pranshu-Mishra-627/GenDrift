from sklearn.ensemble import RandomForestClassifier

from core.collapse_index import (
    calculate_collapse_index, calculate_distribution_index,
    calculate_performance_index,
)
from core.data_handler import create_hybrid_dataset
from core.evaluator import evaluate_model
from core.preprocessing import TabularPreprocessor
from core.quality_analyzer import (
    calculate_class_balance, calculate_diversity,
    calculate_wasserstein_distance, create_diversity_scaler,
)
from core.synthesizer import generate_synthetic_data

CI_METRIC_VERSION = "ci-v1.1-standardized-wasserstein"


def train_model(train_df, target_column, model_params=None, random_state=42):
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]
    preprocessor = TabularPreprocessor().fit(X)
    X_encoded = preprocessor.transform(X)
    params = {"n_estimators": 100, "random_state": random_state}
    if model_params:
        params.update(model_params)
    model = RandomForestClassifier(**params)
    model.fit(X_encoded, y)
    return model, preprocessor


def make_predictions(model, test_df, target_column, preprocessor):
    X = test_df.drop(columns=[target_column])
    return model.predict(preprocessor.transform(X))


def _shap_rows(model, X, preprocessor, generation, enabled, max_samples, random_state):
    if not enabled:
        return None
    from core.shap_tracker import track_shap_importance
    encoded_X = preprocessor.transform(X)
    df = track_shap_importance(
        model, encoded_X, generation, max_samples=max_samples,
        random_state=random_state,
    )
    return df.drop(columns=["generation"]).to_dict(orient="records")


def _optimize_params(train_df, target_column, optimize, n_trials, random_state, fallback_params):
    if not optimize:
        return fallback_params, None
    from core.optimizer import optimize_random_forest
    optimized = optimize_random_forest(
        train_df, target_column, n_trials=n_trials, random_state=random_state
    )
    return optimized["best_params"], {
        "best_score": optimized["best_score"],
        "best_params": optimized["best_params"],
    }


def run_generation(
    train_df, test_df, target_column, baseline_metrics, real_reference_df,
    real_diversity, diversity_scaler, numerical_columns, real_ratio=0.7,
    model_params=None, generation=1, track_shap=False,
    max_shap_samples=200, random_state=42, optimize=False, n_trials=20,
):
    generation_seed = random_state + generation
    synthetic_rows = len(train_df) - int(len(train_df) * real_ratio)
    synthetic_df = generate_synthetic_data(
        train_df, synthetic_rows, random_state=generation_seed
    )
    synthetic_diversity = calculate_diversity(
        synthetic_df, diversity_scaler, numerical_columns,
        random_state=generation_seed,
    )
    class_balance = calculate_class_balance(
        real_reference_df, synthetic_df, target_column
    )
    wasserstein = calculate_wasserstein_distance(
        real_reference_df, synthetic_df, numerical_columns,
        scaler=diversity_scaler,
    )
    di = calculate_distribution_index(
        wasserstein, class_balance, real_diversity, synthetic_diversity
    )
    hybrid_df = create_hybrid_dataset(
        train_df, synthetic_df, real_ratio=real_ratio,
        random_state=generation_seed,
    )
    generation_params, optimization = _optimize_params(
        hybrid_df, target_column, optimize, n_trials, generation_seed, model_params
    )
    model, preprocessor = train_model(
        hybrid_df, target_column, generation_params, random_state=generation_seed
    )
    metrics = evaluate_model(
        test_df[target_column],
        make_predictions(model, test_df, target_column, preprocessor),
    )
    pi = calculate_performance_index(baseline_metrics, metrics)
    result = {
        "metric_version": CI_METRIC_VERSION,
        **metrics, "diversity": synthetic_diversity,
        "class_balance": class_balance, "wasserstein": wasserstein,
        "performance_index": pi, "distribution_index": di,
        "collapse_index": calculate_collapse_index(pi, di),
    }
    if optimization is not None:
        result["optimization_best_score"] = optimization["best_score"]
        result["optimization_best_params"] = optimization["best_params"]
    shap_rows = _shap_rows(
        model, hybrid_df.drop(columns=[target_column]), preprocessor, generation,
        track_shap, max_shap_samples, generation_seed,
    )
    if shap_rows is not None:
        result["shap_importance"] = shap_rows
    return hybrid_df, result, synthetic_df


def run_experiment(
    train_df, test_df, target_column, generations=10, real_ratio=0.7,
    random_state=42, model_params=None, track_shap=False,
    max_shap_samples=200, optimize=False, n_trials=20,
    return_synthetic=False,
):
    if generations < 1:
        raise ValueError("generations must be at least 1.")
    if not 0 < real_ratio < 1:
        raise ValueError("real_ratio must be greater than 0 and less than 1.")
    if target_column not in train_df.columns or target_column not in test_df.columns:
        raise ValueError("Target column must exist in both dataframes.")

    reference = train_df.copy()
    scaler, numerical_columns = create_diversity_scaler(
        reference, target_column=target_column
    )
    real_diversity = calculate_diversity(
        reference, scaler, numerical_columns, random_state=random_state
    )

    baseline_params, baseline_optimization = _optimize_params(
        train_df, target_column, optimize, n_trials, random_state, model_params
    )
    baseline_model, baseline_preprocessor = train_model(
        train_df, target_column, baseline_params, random_state=random_state
    )
    baseline_metrics = evaluate_model(
        test_df[target_column],
        make_predictions(baseline_model, test_df, target_column, baseline_preprocessor),
    )
    baseline = {
        "generation": 0, "metric_version": CI_METRIC_VERSION,
        **baseline_metrics, "diversity": real_diversity,
        "class_balance": None, "wasserstein": None,
        "performance_index": 0.0, "distribution_index": 0.0,
        "collapse_index": 0.0,
    }
    if baseline_optimization is not None:
        baseline["optimization_best_score"] = baseline_optimization["best_score"]
        baseline["optimization_best_params"] = baseline_optimization["best_params"]
    shap_rows = _shap_rows(
        baseline_model, train_df.drop(columns=[target_column]), baseline_preprocessor,
        0, track_shap, max_shap_samples, random_state,
    )
    if shap_rows is not None:
        baseline["shap_importance"] = shap_rows

    results = [baseline]
    synthetic_generations = {}
    current = train_df.copy()
    for generation in range(1, generations):
        hybrid, metrics, synthetic_df = run_generation(
            current, test_df, target_column, baseline_metrics, reference,
            real_diversity, scaler, numerical_columns, real_ratio,
            model_params=model_params, generation=generation,
            track_shap=track_shap, max_shap_samples=max_shap_samples,
            random_state=random_state, optimize=optimize, n_trials=n_trials,
        )

        results.append({"generation": generation, **metrics})
        synthetic_generations[generation] = synthetic_df
        current = hybrid

    if return_synthetic:
        return results, synthetic_generations

    return results
