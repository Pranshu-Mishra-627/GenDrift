from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(y_true, predictions):
    """Calculate the downstream classifier's performance metrics."""
    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
