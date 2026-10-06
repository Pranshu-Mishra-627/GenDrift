import numpy as np


# ============================================================
# PERFORMANCE DEGRADATION
# ============================================================

def normalize_performance(baseline, current):
    """
    Calculate performance degradation relative to G0.

    Formula:
        degradation = (baseline - current) / baseline

    The result is clipped to [0, 1].

    0 = no degradation or improvement
    1 = complete degradation
    """

    if baseline == 0:
        return 0.0

    degradation = (baseline - current) / baseline

    return float(np.clip(degradation, 0.0, 1.0))


def calculate_performance_index(
    baseline_metrics,
    current_metrics
):
    """
    Calculate the Performance Index (PI).

    Accuracy, precision, recall and F1 are each
    normalized relative to G0 and then averaged.

    PI is therefore in the range [0, 1].

    0 = no performance degradation
    1 = maximum performance degradation
    """

    accuracy_degradation = normalize_performance(
        baseline_metrics["accuracy"],
        current_metrics["accuracy"]
    )

    precision_degradation = normalize_performance(
        baseline_metrics["precision"],
        current_metrics["precision"]
    )

    recall_degradation = normalize_performance(
        baseline_metrics["recall"],
        current_metrics["recall"]
    )

    f1_degradation = normalize_performance(
        baseline_metrics["f1"],
        current_metrics["f1"]
    )

    performance_index = (
        accuracy_degradation
        + precision_degradation
        + recall_degradation
        + f1_degradation
    ) / 4

    return float(performance_index)


# ============================================================
# DISTRIBUTION DEGRADATION
# ============================================================

def normalize_wasserstein(wasserstein_value):
    """
    Convert Wasserstein distance into a bounded [0, 1)
    degradation score.

    Formula:
        W_norm = W / (1 + W)

    The transformation is monotonic, so larger
    Wasserstein distances always produce larger
    normalized degradation values.
    """

    if wasserstein_value < 0:
        raise ValueError(
            "Wasserstein distance cannot be negative."
        )

    return float(
        wasserstein_value / (1.0 + wasserstein_value)
    )


def normalize_diversity(
    real_diversity,
    synthetic_diversity
):
    """
    Calculate diversity degradation relative to
    the original real training data.

    Formula:
        degradation =
            (real_diversity - synthetic_diversity)
            / real_diversity

    Only a reduction in diversity is treated as
    degradation. The result is clipped to [0, 1].

    0 = synthetic data is equally or more diverse
        than the real reference
    1 = maximum diversity degradation
    """

    if real_diversity == 0:
        return 0.0

    degradation = (
        real_diversity - synthetic_diversity
    ) / real_diversity

    return float(
        np.clip(degradation, 0.0, 1.0)
    )


def normalize_class_balance(class_balance_difference):
    """
    Total Variation Distance is already bounded
    between 0 and 1.

    Therefore it is used directly.
    """

    return float(
        np.clip(class_balance_difference, 0.0, 1.0)
    )


def calculate_distribution_index(
    wasserstein,
    class_balance_difference,
    real_diversity,
    synthetic_diversity
):
    """
    Calculate the Distribution Index (DI).

    The three distribution degradation metrics are
    normalized separately and then averaged.

    DI is therefore in the range [0, 1].

    0 = no distribution degradation
    1 = maximum distribution degradation
    """

    wasserstein_degradation = normalize_wasserstein(
        wasserstein
    )

    class_balance_degradation = normalize_class_balance(
        class_balance_difference
    )

    diversity_degradation = normalize_diversity(
        real_diversity,
        synthetic_diversity
    )

    distribution_index = (
        wasserstein_degradation
        + class_balance_degradation
        + diversity_degradation
    ) / 3

    return float(distribution_index)


# ============================================================
# COLLAPSE INDEX
# ============================================================

def calculate_collapse_index(
    performance_index,
    distribution_index
):
    """
    Calculate the final Collapse Index.

    Formula:
        CI = 0.5 * PI + 0.5 * DI

    The Performance and Distribution groups receive
    equal weight.

    CI is in the range [0, 1].

    0 = no measured degradation
    1 = maximum measured degradation
    """

    collapse_index = (
        0.5 * performance_index
        + 0.5 * distribution_index
    )

    return float(
        np.clip(collapse_index, 0.0, 1.0)
    )