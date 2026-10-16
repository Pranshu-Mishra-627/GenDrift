import numpy as np


def normalize_performance(baseline, current):
    """
    Convert performance decrease relative to G0 into [0, 1].

    Improvements are clipped to zero because the index measures degradation,
    not improvement.
    """
    if baseline < 0 or current < 0:
        raise ValueError("Performance metrics cannot be negative.")

    if baseline == 0:
        return 0.0

    degradation = (baseline - current) / baseline

    return float(np.clip(degradation, 0.0, 1.0))


def calculate_performance_index(
    baseline_metrics,
    current_metrics,
):
    """Average normalized degradation of Accuracy, Precision, Recall and F1."""
    accuracy_degradation = normalize_performance(
        baseline_metrics["accuracy"],
        current_metrics["accuracy"],
    )

    precision_degradation = normalize_performance(
        baseline_metrics["precision"],
        current_metrics["precision"],
    )

    recall_degradation = normalize_performance(
        baseline_metrics["recall"],
        current_metrics["recall"],
    )

    f1_degradation = normalize_performance(
        baseline_metrics["f1"],
        current_metrics["f1"],
    )

    performance_index = (
        accuracy_degradation
        + precision_degradation
        + recall_degradation
        + f1_degradation
    ) / 4

    return float(performance_index)


def normalize_wasserstein(wasserstein_value):
    """
    Bound a non-negative Wasserstein distance to [0, 1] using x / (1 + x).
    """
    if wasserstein_value < 0:
        raise ValueError("Wasserstein distance cannot be negative.")

    return float(
        wasserstein_value / (1.0 + wasserstein_value)
    )


def normalize_diversity(real_diversity, synthetic_diversity):
    """
    Measure diversity loss relative to the original real-data baseline.

    Synthetic diversity above the baseline is not treated as negative
    degradation. The lower bound is zero.
    """
    if real_diversity < 0 or synthetic_diversity < 0:
        raise ValueError("Diversity values cannot be negative.")

    if real_diversity == 0:
        return 0.0

    degradation = (
        real_diversity - synthetic_diversity
    ) / real_diversity

    return float(
        np.clip(degradation, 0.0, 1.0)
    )


def normalize_class_balance(class_balance_difference):
    """Keep class-distribution deviation in its natural [0, 1] range."""
    if class_balance_difference < 0:
        raise ValueError("Class distribution difference cannot be negative.")

    return float(
        np.clip(
            class_balance_difference,
            0.0,
            1.0,
        )
    )


def calculate_distribution_index(
    wasserstein,
    class_balance_difference,
    real_diversity,
    synthetic_diversity,
):
    """Average the three normalized distribution degradation components."""
    wasserstein_degradation = normalize_wasserstein(
        wasserstein
    )

    class_balance_degradation = normalize_class_balance(
        class_balance_difference
    )

    diversity_degradation = normalize_diversity(
        real_diversity,
        synthetic_diversity,
    )

    distribution_index = (
        wasserstein_degradation
        + class_balance_degradation
        + diversity_degradation
    ) / 3

    return float(distribution_index)


def calculate_collapse_index(
    performance_index,
    distribution_index,
):
    """Combine PI and DI with equal top-level weights."""
    collapse_index = (
        0.5 * performance_index
        + 0.5 * distribution_index
    )

    return float(
        np.clip(
            collapse_index,
            0.0,
            1.0,
        )
    )
