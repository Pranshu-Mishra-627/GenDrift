from core.collapse_index import (
    normalize_performance,
    normalize_wasserstein,
    normalize_diversity,
    normalize_class_balance,
    calculate_performance_index,
    calculate_distribution_index,
    calculate_collapse_index
)


# ============================================================
# PERFORMANCE NORMALIZATION
# ============================================================

assert normalize_performance(
    1.0,
    1.0
) == 0.0

assert abs(
    normalize_performance(1.0, 0.8) - 0.2
) < 1e-9

# Improvement should not produce negative degradation
assert normalize_performance(
    0.8,
    0.9
) == 0.0


# ============================================================
# WASSERSTEIN NORMALIZATION
# ============================================================

assert normalize_wasserstein(
    0.0
) == 0.0

assert 0 < normalize_wasserstein(1.0) < 1

assert (
    normalize_wasserstein(2.0)
    > normalize_wasserstein(1.0)
)


# ============================================================
# DIVERSITY NORMALIZATION
# ============================================================

assert normalize_diversity(
    10.0,
    10.0
) == 0.0

assert abs(
    normalize_diversity(10.0, 8.0) - 0.2
) < 1e-9

# More diverse than real data is not treated as degradation
assert normalize_diversity(
    10.0,
    12.0
) == 0.0


# ============================================================
# CLASS BALANCE
# ============================================================

assert normalize_class_balance(
    0.0
) == 0.0

assert normalize_class_balance(
    0.5
) == 0.5

assert normalize_class_balance(
    1.0
) == 1.0


# ============================================================
# PERFORMANCE INDEX
# ============================================================

baseline_metrics = {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
}

current_metrics = {
    "accuracy": 0.8,
    "precision": 0.9,
    "recall": 0.7,
    "f1": 0.8
}

pi = calculate_performance_index(
    baseline_metrics,
    current_metrics
)

expected_pi = (
    0.2
    + 0.1
    + 0.3
    + 0.2
) / 4

assert abs(
    pi - expected_pi
) < 1e-9


# ============================================================
# DISTRIBUTION INDEX
# ============================================================

di = calculate_distribution_index(
    wasserstein=0.0,
    class_balance_difference=0.0,
    real_diversity=10.0,
    synthetic_diversity=10.0
)

assert di == 0.0


# ============================================================
# COLLAPSE INDEX
# ============================================================

ci = calculate_collapse_index(
    performance_index=0.2,
    distribution_index=0.4
)

assert abs(
    ci - 0.3
) < 1e-9


# ============================================================
# RANGE CHECKS
# ============================================================

assert 0 <= pi <= 1
assert 0 <= di <= 1
assert 0 <= ci <= 1


print("Collapse Index tests passed.")