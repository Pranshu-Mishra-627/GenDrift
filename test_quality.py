from core.data_handler import load_dataset, split_dataset
from core.quality_analyzer import (
    create_diversity_scaler,
    calculate_diversity,
    calculate_class_balance,
    calculate_wasserstein_distance
)


# ============================================================
# LOAD DATA
# ============================================================

df = load_dataset("data/iris.csv")

train_df, test_df = split_dataset(
    df,
    target_column="target"
)


# ============================================================
# DIVERSITY
# ============================================================

scaler, numerical_columns = create_diversity_scaler(
    train_df
)

real_diversity = calculate_diversity(
    train_df,
    scaler,
    numerical_columns
)

print("Real diversity:", real_diversity)


# ============================================================
# SYNTHETIC DATA
# ============================================================

from core.synthesizer import generate_synthetic_data

synthetic_df = generate_synthetic_data(
    train_df,
    num_rows=36
)


synthetic_diversity = calculate_diversity(
    synthetic_df,
    scaler,
    numerical_columns
)

print("Synthetic diversity:", synthetic_diversity)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_balance = calculate_class_balance(
    train_df,
    synthetic_df,
    "target"
)

print("Class distribution difference:", class_balance)


# ============================================================
# WASSERSTEIN
# ============================================================

wasserstein = calculate_wasserstein_distance(
    train_df,
    synthetic_df
)

print("Wasserstein distance:", wasserstein)


# ============================================================
# BASIC VALIDATION
# ============================================================

assert real_diversity >= 0
assert synthetic_diversity >= 0

assert 0 <= class_balance <= 1

assert wasserstein >= 0

print("\nQuality analyzer tests passed.")