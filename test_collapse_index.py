from core.collapse_index import (
    calculate_collapse_index,
    calculate_distribution_index,
    calculate_performance_index,
    normalize_class_balance,
    normalize_diversity,
    normalize_performance,
    normalize_wasserstein,
)


# Performance normalization
assert normalize_performance(1.0, 1.0) == 0.0
assert abs(normalize_performance(1.0, 0.8) - 0.2) < 1e-9
assert normalize_performance(0.8, 0.9) == 0.0
assert normalize_performance(0.0, 0.0) == 0.0

# Wasserstein normalization
assert normalize_wasserstein(0.0) == 0.0
assert 0.0 < normalize_wasserstein(1.0) < 1.0
assert normalize_wasserstein(2.0) > normalize_wasserstein(1.0)

# Diversity normalization
assert normalize_diversity(10.0, 10.0) == 0.0
assert normalize_diversity(10.0, 8.0) == 0.2
assert normalize_diversity(10.0, 12.0) == 0.0

# Class balance is already bounded
assert normalize_class_balance(0.0) == 0.0
assert normalize_class_balance(0.5) == 0.5
assert normalize_class_balance(1.0) == 1.0

# Composite indices
di = calculate_distribution_index(
    wasserstein=1.0,
    class_balance_difference=0.2,
    real_diversity=10.0,
    synthetic_diversity=8.0,
)

assert 0.0 <= di <= 1.0

pi = calculate_performance_index(
    baseline_metrics={
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    },
    current_metrics={
        "accuracy": 0.8,
        "precision": 0.9,
        "recall": 0.7,
        "f1": 0.8,
    },
)

assert abs(pi - 0.2) < 1e-9

ci = calculate_collapse_index(pi, di)

assert 0.0 <= ci <= 1.0
assert abs(ci - (0.5 * pi + 0.5 * di)) < 1e-12

print("Collapse Index tests passed.")
