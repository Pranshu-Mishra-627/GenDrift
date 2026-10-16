
from core.data_handler import load_dataset, split_dataset
from core.synthesizer import generate_synthetic_data

DATA_PATH = "data/bank_marketing/bank_marketing_no_duration.csv"

df = load_dataset(DATA_PATH)
train_df, test_df = split_dataset(
    df,
    "y",
    test_size=0.2,
    random_state=42,
)

print("Generating synthetic data with seed 8...")
synthetic_a = generate_synthetic_data(
    train_df, 500, random_state=8
)

print("Generating synthetic data with seed 43...")
synthetic_b = generate_synthetic_data(
    train_df, 500, random_state=43
)

print("\n--- Comparison ---")
print("Same shape:", synthetic_a.shape == synthetic_b.shape)
print("Exactly identical:", synthetic_a.equals(synthetic_b))
print(
    "Identical values per column:",
    (synthetic_a == synthetic_b).all().to_dict(),
)
print("\nFirst 5 rows from seed 8:")
print(synthetic_a.head())

print("\nFirst 5 rows from seed 43:")
print(synthetic_b.head())


print("\n--- Reproducibility check ---")
synthetic_a_repeat = generate_synthetic_data(
    train_df, 500, random_state=8
)

print(
    "Same seed produces identical data:",
    synthetic_a.equals(synthetic_a_repeat),
)
