"""Compact, residual LSTM for forecasting GenDrift Collapse Index (CI).

The model predicts the *change* from the last observed CI rather than the
absolute CI. Evaluation is performed by held-out experiment run, and the
persistence baseline is reported alongside every result. This remains a
research prototype until it demonstrates reliable gains on more independent
trajectories.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.predictor_evaluation import regression_metrics
from core.sequence_builder import (
    FEATURE_COLUMNS,
    build_sequence_dataset,
    normalize_histories,
    rows_to_feature_matrix,
)


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for the LSTM predictor. Install it with "
            "`python -m pip install tensorflow` inside the GenDrift virtual environment."
        ) from exc
    return tf


def build_lstm_model(sequence_length: int, n_features: int, learning_rate: float = 0.001):
    """Build a small regularized network whose output is a standardized CI delta."""
    tf = _tensorflow()
    regularizer = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, n_features)),
            tf.keras.layers.LSTM(8, kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=regularizer),
            # Linear output: the model predicts a standardized change in CI.
            tf.keras.layers.Dense(1, activation="linear"),
        ],
        name="gendrift_ci_residual_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae_delta_scaled")],
    )
    return model


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf = _tensorflow()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    return tf


def _split_x(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n_features = x.shape[-1]
    return scaler.transform(x.reshape(-1, n_features)).reshape(x.shape)


def _fit_residual_model(
    train_histories: dict[str, list[dict]],
    validation_history: dict[str, list[dict]] | None,
    *,
    sequence_length: int,
    horizon: int,
    random_state: int,
    epochs: int,
    batch_size: int,
    patience: int,
):
    """Fit on training runs only; optional validation data comes from a whole run."""
    train_data = build_sequence_dataset(train_histories, sequence_length, horizon)
    if len(train_data["y"]) < 3:
        raise ValueError("At least three training windows are required to fit the LSTM.")

    validation_data = None
    if validation_history:
        validation_data = build_sequence_dataset(validation_history, sequence_length, horizon)
        if len(validation_data["y"]) == 0:
            validation_data = None

    n_features = train_data["X"].shape[-1]
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_data["X"].reshape(-1, n_features))
    x_train = _split_x(train_data["X"], feature_scaler)

    # Residual learning: target is future CI minus the last observed CI.
    delta_train = train_data["y"] - train_data["last_observed_ci"]
    target_scaler = StandardScaler()
    delta_train_scaled = target_scaler.fit_transform(delta_train.reshape(-1, 1)).reshape(-1)

    x_validation = y_validation_scaled = None
    if validation_data is not None:
        x_validation = _split_x(validation_data["X"], feature_scaler)
        delta_validation = validation_data["y"] - validation_data["last_observed_ci"]
        y_validation_scaled = target_scaler.transform(delta_validation.reshape(-1, 1)).reshape(-1)

    tf = _seed_everything(random_state)
    model = build_lstm_model(sequence_length, n_features)
    callbacks = []
    fit_kwargs = {
        "x": x_train,
        "y": delta_train_scaled,
        "epochs": epochs,
        "batch_size": min(batch_size, len(x_train)),
        "verbose": 0,
        "shuffle": True,
    }
    if x_validation is not None:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            )
        ]
        fit_kwargs["validation_data"] = (x_validation, y_validation_scaled)
        fit_kwargs["callbacks"] = callbacks

    history = model.fit(**fit_kwargs)
    best_epoch = int(np.argmin(history.history.get("val_loss", history.history["loss"])) + 1)
    return {
        "model": model,
        "scaler": feature_scaler,
        "target_scaler": target_scaler,
        "history": history,
        "best_epoch": best_epoch,
        "train_data": train_data,
        "validation_data": validation_data,
    }


def _predict_absolute_ci(model, feature_scaler, target_scaler, data):
    x_scaled = _split_x(data["X"], feature_scaler)
    scaled_delta = model.predict(x_scaled, verbose=0).reshape(-1, 1)
    delta = target_scaler.inverse_transform(scaled_delta).reshape(-1)
    return np.clip(data["last_observed_ci"] + delta, 0.0, 1.0)


def train_lstm_predictor(
    histories: Any,
    sequence_length: int = 3,
    horizon: int = 1,
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 100,
    batch_size: int = 8,
    patience: int = 10,
) -> dict:
    """Train a residual LSTM and evaluate on held-out run(s).

    Splitting is by run ID. When enough training runs exist, a complete run is
    reserved for early stopping rather than splitting overlapping windows.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive.")
    if sequence_length < 1 or horizon < 1:
        raise ValueError("sequence_length and horizon must be positive.")

    normalized = normalize_histories(histories)
    usable = {
        run_id: rows for run_id, rows in normalized.items()
        if len(rows) >= sequence_length + horizon
    }
    if len(usable) < 3:
        raise ValueError(
            f"At least 3 usable independent runs are required; found {len(usable)}."
        )

    run_ids = list(usable)
    train_ids, test_ids = train_test_split(
        run_ids, test_size=test_size, random_state=random_state, shuffle=True
    )
    test_histories = {key: usable[key] for key in test_ids}
    fit_ids = list(train_ids)
    validation_history = None
    if len(fit_ids) >= 4:
        validation_id = fit_ids.pop(-1)
        validation_history = {validation_id: usable[validation_id]}
    fit_histories = {key: usable[key] for key in fit_ids}

    fitted = _fit_residual_model(
        fit_histories,
        validation_history,
        sequence_length=sequence_length,
        horizon=horizon,
        random_state=random_state,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )
    test_data = build_sequence_dataset(test_histories, sequence_length, horizon)
    predictions = _predict_absolute_ci(
        fitted["model"], fitted["scaler"], fitted["target_scaler"], test_data
    )
    metrics = regression_metrics(
        test_data["y"], predictions, baseline_pred=test_data["last_observed_ci"]
    )
    metrics.update({
        "sequence_length": int(sequence_length),
        "horizon": int(horizon),
        "train_runs": list(fit_ids),
        "validation_runs": list(validation_history or {}),
        "test_runs": list(test_ids),
        "n_train_windows": int(len(fitted["train_data"]["y"])),
        "n_validation_windows": int(len(fitted["validation_data"]["y"])) if fitted["validation_data"] else 0,
        "n_test_windows": int(len(test_data["y"])),
        "epochs_trained": int(len(fitted["history"].history.get("loss", []))),
        "best_epoch": int(fitted["best_epoch"]),
        "prediction_mode": "residual_delta_from_last_ci",
        "status": "experimental",
    })
    return {
        "model": fitted["model"],
        "scaler": fitted["scaler"],
        "target_scaler": fitted["target_scaler"],
        "metrics": metrics,
        "feature_columns": list(FEATURE_COLUMNS),
        "sequence_length": int(sequence_length),
        "horizon": int(horizon),
        "training_history": {key: [float(x) for x in values] for key, values in fitted["history"].history.items()},
        "prediction_mode": "residual_delta_from_last_ci",
    }


def evaluate_lstm_leave_one_run_out(
    histories: Any,
    sequence_length: int = 3,
    horizon: int = 1,
    random_state: int = 42,
    epochs: int = 100,
    batch_size: int = 8,
    patience: int = 10,
) -> dict:
    """Evaluate all runs as held-out test data once each (run-level CV).

    Each fold reserves one run for testing and one different run for validation;
    the remaining runs are used for fitting. Metrics are aggregated over all
    out-of-fold predictions. With very few runs this remains preliminary.
    """
    normalized = normalize_histories(histories)
    usable = {
        run_id: rows for run_id, rows in normalized.items()
        if len(rows) >= sequence_length + horizon
    }
    run_ids = sorted(usable)
    if len(run_ids) < 4:
        raise ValueError("Leave-one-run-out evaluation needs at least 4 usable runs.")

    all_y, all_pred, all_persistence = [], [], []
    fold_results = []
    best_epochs = []
    for fold_index, test_id in enumerate(run_ids):
        remaining = [run_id for run_id in run_ids if run_id != test_id]
        validation_id = remaining[fold_index % len(remaining)]
        train_ids = [run_id for run_id in remaining if run_id != validation_id]
        fitted = _fit_residual_model(
            {key: usable[key] for key in train_ids},
            {validation_id: usable[validation_id]},
            sequence_length=sequence_length,
            horizon=horizon,
            random_state=random_state + fold_index,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )
        test_data = build_sequence_dataset({test_id: usable[test_id]}, sequence_length, horizon)
        predictions = _predict_absolute_ci(
            fitted["model"], fitted["scaler"], fitted["target_scaler"], test_data
        )
        fold_metrics = regression_metrics(
            test_data["y"], predictions, baseline_pred=test_data["last_observed_ci"]
        )
        fold_metrics.update({
            "test_run": test_id,
            "validation_run": validation_id,
            "train_runs": train_ids,
            "best_epoch": fitted["best_epoch"],
        })
        fold_results.append(fold_metrics)
        best_epochs.append(fitted["best_epoch"])
        all_y.extend(test_data["y"].tolist())
        all_pred.extend(predictions.tolist())
        all_persistence.extend(test_data["last_observed_ci"].tolist())

    aggregate = regression_metrics(all_y, all_pred, baseline_pred=all_persistence)
    aggregate.update({
        "evaluation": "leave_one_run_out",
        "n_runs": len(run_ids),
        "folds": fold_results,
        "median_best_epoch": int(np.median(best_epochs)),
        "sequence_length": int(sequence_length),
        "horizon": int(horizon),
        "status": "experimental",
    })
    return aggregate


def fit_lstm_predictor_on_all_runs(
    histories: Any,
    sequence_length: int = 3,
    horizon: int = 1,
    epochs: int = 30,
    batch_size: int = 8,
    random_state: int = 42,
    metrics: dict | None = None,
) -> dict:
    """Fit a deployable predictor on every supplied run after CV is complete."""
    normalized = normalize_histories(histories)
    usable = {
        run_id: rows for run_id, rows in normalized.items()
        if len(rows) >= sequence_length + horizon
    }
    if len(usable) < 3:
        raise ValueError("At least 3 usable independent runs are required.")
    data = build_sequence_dataset(usable, sequence_length, horizon)
    n_features = data["X"].shape[-1]
    scaler = StandardScaler().fit(data["X"].reshape(-1, n_features))
    x_scaled = _split_x(data["X"], scaler)
    deltas = data["y"] - data["last_observed_ci"]
    target_scaler = StandardScaler().fit(deltas.reshape(-1, 1))
    y_scaled = target_scaler.transform(deltas.reshape(-1, 1)).reshape(-1)
    tf = _seed_everything(random_state)
    model = build_lstm_model(sequence_length, n_features)
    history = model.fit(
        x_scaled, y_scaled,
        epochs=max(1, int(epochs)),
        batch_size=min(batch_size, len(x_scaled)),
        verbose=0,
        shuffle=True,
    )
    return {
        "model": model,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "metrics": metrics or {"status": "not_evaluated"},
        "feature_columns": list(FEATURE_COLUMNS),
        "sequence_length": int(sequence_length),
        "horizon": int(horizon),
        "training_history": {key: [float(x) for x in values] for key, values in history.history.items()},
        "prediction_mode": "residual_delta_from_last_ci",
    }


def predict_future_ci(
    generation_rows: list[dict],
    model,
    scaler: StandardScaler,
    sequence_length: int = 3,
    horizon: int = 1,
    target_scaler: StandardScaler | None = None,
) -> dict:
    """Predict future CI. Supports legacy absolute-output models if no target scaler is given."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    rows = sorted(
        [dict(row) for row in generation_rows if "generation" in row],
        key=lambda row: int(row["generation"]),
    )
    if len(rows) < sequence_length:
        raise ValueError(f"At least {sequence_length} generation rows are required.")
    matrix = rows_to_feature_matrix(rows[-sequence_length:])
    scaled = scaler.transform(matrix).reshape(1, sequence_length, len(FEATURE_COLUMNS))
    raw_output = float(model.predict(scaled, verbose=0).reshape(-1)[0])
    last_ci = float(np.clip(float(rows[-1].get("collapse_index", 0.0) or 0.0), 0.0, 1.0))
    if target_scaler is not None:
        delta = float(target_scaler.inverse_transform(np.array([[raw_output]])).reshape(-1)[0])
        prediction = float(np.clip(last_ci + delta, 0.0, 1.0))
        mode = "residual_delta_from_last_ci"
    else:
        prediction = float(np.clip(raw_output, 0.0, 1.0))
        mode = "legacy_absolute_ci"
    last_generation = int(rows[-1]["generation"])
    return {
        "predicted_collapse_index": prediction,
        "last_observed_generation": last_generation,
        "predicted_generation": last_generation + horizon,
        "horizon": int(horizon),
        "prediction_mode": mode,
        "status": "experimental",
    }


def save_lstm_predictor(predictor: dict, output_dir: str | Path) -> dict:
    """Persist the Keras model, feature scaler, target scaler, and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "collapse_lstm.keras"
    scaler_path = output_dir / "feature_scaler.joblib"
    target_scaler_path = output_dir / "target_scaler.joblib"
    metadata_path = output_dir / "predictor_metadata.json"

    predictor["model"].save(model_path)
    joblib.dump(predictor["scaler"], scaler_path)
    if predictor.get("target_scaler") is not None:
        joblib.dump(predictor["target_scaler"], target_scaler_path)
    metadata = {
        "feature_columns": predictor["feature_columns"],
        "sequence_length": predictor["sequence_length"],
        "horizon": predictor["horizon"],
        "metrics": predictor["metrics"],
        "training_history": predictor.get("training_history", {}),
        "prediction_mode": predictor.get("prediction_mode", "legacy_absolute_ci"),
        "model_file": model_path.name,
        "scaler_file": scaler_path.name,
        "target_scaler_file": target_scaler_path.name if predictor.get("target_scaler") is not None else None,
        "status": "experimental",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "target_scaler_path": str(target_scaler_path) if predictor.get("target_scaler") is not None else None,
        "metadata_path": str(metadata_path),
    }


def load_lstm_predictor(model_dir: str | Path) -> dict:
    """Load a saved predictor artifact, including legacy artifacts."""
    model_dir = Path(model_dir)
    tf = _tensorflow()
    metadata = json.loads((model_dir / "predictor_metadata.json").read_text(encoding="utf-8"))
    target_scaler_file = metadata.get("target_scaler_file")
    return {
        "model": tf.keras.models.load_model(model_dir / metadata["model_file"]),
        "scaler": joblib.load(model_dir / metadata["scaler_file"]),
        "target_scaler": joblib.load(model_dir / target_scaler_file) if target_scaler_file else None,
        "metrics": metadata.get("metrics", {}),
        "feature_columns": metadata.get("feature_columns", list(FEATURE_COLUMNS)),
        "sequence_length": int(metadata.get("sequence_length", 3)),
        "horizon": int(metadata.get("horizon", 1)),
        "training_history": metadata.get("training_history", {}),
        "prediction_mode": metadata.get("prediction_mode", "legacy_absolute_ci"),
    }
