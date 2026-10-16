from core.data_handler import load_dataset, split_dataset


df = load_dataset("data/iris.csv")

print("Original dataset:")
print(df.shape)
print(df.head())

train_df, test_df = split_dataset(
    df,
    target_column="target",
)

print("\nTraining pool:")
print(train_df.shape)

print("\nFrozen test set:")
print(test_df.shape)

print("\nTraining class distribution:")
print(train_df["target"].value_counts())

print("\nTest class distribution:")
print(test_df["target"].value_counts())

assert len(train_df) + len(test_df) == len(df)
assert "target" in train_df.columns
assert "target" in test_df.columns
assert len(train_df) == 120
assert len(test_df) == 30

print("\nData handler tests passed.")
