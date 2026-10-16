"""Optuna tuning using an internal validation split of training data only."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from core.preprocessing import TabularPreprocessor


def optimize_random_forest(train_df, target_column, n_trials=20, random_state=42):
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")
    if len(train_df) < 10:
        raise ValueError("At least 10 rows are required.")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    import optuna

    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    preprocessor = TabularPreprocessor().fit(X_fit)
    X_fit_encoded = preprocessor.transform(X_fit)
    X_val_encoded = preprocessor.transform(X_val)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": random_state,
            "n_jobs": -1,
        }
        model = RandomForestClassifier(**params).fit(X_fit_encoded, y_fit)
        pred = model.predict(X_val_encoded)
        return float(f1_score(y_val, pred, average="weighted", zero_division=0))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)
    params = dict(study.best_params)
    params.update({"random_state": random_state, "n_jobs": -1})
    return {"best_params": params, "best_score": float(study.best_value), "study": study}
