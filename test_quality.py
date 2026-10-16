from core.data_handler import load_dataset, split_dataset
from core.quality_analyzer import (
    calculate_class_balance,
    calculate_diversity,
    calculate_wasserstein_distance,
    create_diversity_scaler,
)
from core.synthesizer import generate_synthetic_data


df = load_dataset("data/iris.csv")

train_df, test_df = split_dataset(
    df,
    target_column="target",
)

scaler, numerical_columns = create_diversity_scaler(
    train_df,
    target_column="target",
)

assert "target" not in numerical_columns

real_diversity = calculate_diversity(
    train_df,
    scaler,
    numerical_columns,
)

synthetic_df = generate_synthetic_data(
    train_df,
    num_rows=36,
)

synthetic_diversity = calculate_diversity(
    synthetic_df,
    scaler,
    numerical_columns,
)

class_balance = calculate_class_balance(
    train_df,
    synthetic_df,
    "target",
)

wasserstein = calculate_wasserstein_distance(
    train_df,
    synthetic_df,
    numerical_columns,
    scaler=scaler,
)

print("Numerical columns:", numerical_columns)
print("Real diversity:", real_diversity)
print("Synthetic diversity:", synthetic_diversity)
print("Class distribution difference:", class_balance)
print("Wasserstein distance:", wasserstein)

assert real_diversity >= 0.0
assert synthetic_diversity >= 0.0
assert 0.0 <= class_balance <= 1.0
assert wasserstein >= 0.0

print("\nQuality analyzer tests passed.")
