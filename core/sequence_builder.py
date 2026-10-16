"""Build leakage-aware sequences from GenDrift generation histories."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# Keep these aligned with the fields emitted by core.trainer.run_experiment.
BASE_FEATURES = (
    "generation",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "diversity",
    "class_balance",
    "wasserstein",
    "performance_index",
    "distribution_index",
    "collapse_index",
)
MISSING_INDICATOR_FEATURES = ("class_balance_missing", "wasserstein_missing")
FEATURE_COLUMNS = BASE_FEATURES + MISSING_INDICATOR_FEATURES
TARGET_COLUMN = "collapse_index"


def normalize_histories(histories: Any) -> dict[str, list[dict]]:
    """Normalize common JSON/result layouts into {run_id: generation_rows}.

    Supported inputs:
      * {run_id: [generation rows]}
      * [{"run_id": "...", "results": [...]}, ...]
      * a single generation-row list (named ``run_0``)
      * an offline-pipeline payload containing ``results``
    """
    if isinstance(histories, Mapping):
        if "results" in histories and isinstance(histories["results"], list):
            return {"run_0": histories["results"]}
        normalized = {}
        for key, value in histories.items():
            if isinstance(value, Mapping) and isinstance(value.get("results"), list):
                normalized[str(key)] = value["results"]
            elif isinstance(value, list):
                normalized[str(key)] = value
        if normalized:
            return normalized

    if isinstance(histories, Sequence) and not isinstance(histories, (str, bytes)):
        if not histories:
            return {}
        if all(isinstance(item, Mapping) and "generation" in item for item in histories):
            return {"run_0": list(histories)}
        normalized = {}
        for index, item in enumerate(histories):
            if isinstance(item, Mapping) and isinstance(item.get("results"), list):
                normalized[str(item.get("run_id", item.get("name", f"run_{index}")))] = item["results"]
        if normalized:
            return normalized

    raise ValueError(
        "Could not identify generation histories. Provide a list of generation rows, "
        "a {run_id: rows} mapping, or a list of {'run_id': ..., 'results': [...]} objects."
    )


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def rows_to_feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Convert result rows into a numeric matrix with explicit null indicators.

    G0 has no Wasserstein or class-balance measurement; these values are set to
    zero for numeric stability and accompanied by missing-indicator features.
    """
    matrix = []
    for row in rows:
        values = []
        for name in BASE_FEATURES:
            value = _as_float(row.get(name))
            values.append(value)
        class_balance_missing = not np.isfinite(values[BASE_FEATURES.index("class_balance")])
        wasserstein_missing = not np.isfinite(values[BASE_FEATURES.index("wasserstein")])
        values = [0.0 if not np.isfinite(value) else value for value in values]
        values.extend([float(class_balance_missing), float(wasserstein_missing)])
        matrix.append(values)
    if not matrix:
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=float)
    return np.asarray(matrix, dtype=float)


def build_sequence_dataset(
    histories: Any,
    sequence_length: int = 3,
    horizon: int = 1,
) -> dict[str, Any]:
    """Create samples from each run without ever joining separate runs.

    Each input is ``sequence_length`` consecutive generation rows. The target
    is the Collapse Index at ``horizon`` generations after the input window.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    normalized = normalize_histories(histories)
    sequences, targets, run_ids, target_generations, last_observed_ci = [], [], [], [], []

    for run_id, raw_rows in normalized.items():
        rows = sorted(
            [dict(row) for row in raw_rows if isinstance(row, Mapping) and "generation" in row],
            key=lambda row: int(row["generation"]),
        )
        if len(rows) < sequence_length + horizon:
            continue
        features = rows_to_feature_matrix(rows)
        generations = [int(row["generation"]) for row in rows]
        ci_values = [_as_float(row.get(TARGET_COLUMN)) for row in rows]

        for start in range(0, len(rows) - sequence_length - horizon + 1):
            end = start + sequence_length
            target_idx = end + horizon - 1
            target = ci_values[target_idx]
            if not np.isfinite(target):
                continue
            sequences.append(features[start:end])
            targets.append(float(np.clip(target, 0.0, 1.0)))
            run_ids.append(str(run_id))
            target_generations.append(generations[target_idx])
            last_observed_ci.append(float(np.clip(ci_values[end - 1], 0.0, 1.0))
                                    if np.isfinite(ci_values[end - 1]) else 0.0)

    if not sequences:
        return {
            "X": np.empty((0, sequence_length, len(FEATURE_COLUMNS)), dtype=float),
            "y": np.empty((0,), dtype=float),
            "run_ids": np.asarray([], dtype=str),
            "target_generations": np.asarray([], dtype=int),
            "last_observed_ci": np.empty((0,), dtype=float),
            "feature_columns": list(FEATURE_COLUMNS),
            "sequence_length": sequence_length,
            "horizon": horizon,
        }

    return {
        "X": np.asarray(sequences, dtype=float),
        "y": np.asarray(targets, dtype=float),
        "run_ids": np.asarray(run_ids, dtype=str),
        "target_generations": np.asarray(target_generations, dtype=int),
        "last_observed_ci": np.asarray(last_observed_ci, dtype=float),
        "feature_columns": list(FEATURE_COLUMNS),
        "sequence_length": sequence_length,
        "horizon": horizon,
    }
