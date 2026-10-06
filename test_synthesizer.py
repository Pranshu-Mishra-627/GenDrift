from core.data_handler import load_dataset, split_dataset, create_hybrid_dataset
from core.synthesizer import generate_synthetic_data


df = load_dataset("data/iris.csv")

train_df, test_df = split_dataset(
    df,
    target_column="target"
)

synthetic_df = generate_synthetic_data(
    train_df,
    num_rows=36
)

hybrid_df = create_hybrid_dataset(
    train_df,
    synthetic_df,
    real_ratio=0.7
)

print("Real training rows:", len(train_df))
print("Synthetic rows:", len(synthetic_df))
print("Hybrid rows:", len(hybrid_df))