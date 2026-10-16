from core.data_handler import (
    create_hybrid_dataset,
    load_dataset,
    split_dataset,
)
from core.synthesizer import generate_synthetic_data


df = load_dataset("data/iris.csv")

train_df, test_df = split_dataset(
    df,
    target_column="target",
)

synthetic_df = generate_synthetic_data(
    train_df,
    num_rows=36,
    random_state=42,
)
repeat_df = generate_synthetic_data(
    train_df,
    num_rows=36,
    random_state=42,
)

hybrid_df = create_hybrid_dataset(
    train_df,
    synthetic_df,
    real_ratio=0.7,
)

print("Real training rows:", len(train_df))
print("Synthetic rows:", len(synthetic_df))
print("Hybrid rows:", len(hybrid_df))

assert len(synthetic_df) == 36
import pandas as pd
pd.testing.assert_frame_equal(synthetic_df.reset_index(drop=True), repeat_df.reset_index(drop=True))
assert len(hybrid_df) == len(train_df)

assert list(hybrid_df.columns) == list(train_df.columns)

print("Synthesizer and hybrid dataset tests passed.")
