import pandas as pd
from sklearn.model_selection import train_test_split



def load_dataset(file_path):
    """Load a tabular dataset from a CSV file with delimiter detection."""
    return pd.read_csv(file_path, sep=None, engine="python")


def split_dataset(df, target_column, test_size=0.2, random_state=42):
    """
    Split a dataset into a training pool and a frozen test set.

    The test set is created once and is intended to remain unchanged
    throughout the recursive GenDrift experiment.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df[target_column] = y_train

    test_df = X_test.copy()
    test_df[target_column] = y_test

    return train_df, test_df


def create_hybrid_dataset(real_df, synthetic_df, real_ratio=0.7, random_state=42):
    """
    Create a fixed-size hybrid dataset.

    The current training dataframe is sampled at `real_ratio`, while the
    remainder comes from the current synthetic generation. In generations
    after G1, `real_df` may itself contain synthetic-origin rows because it
    is the previous generation's hybrid dataset.
    """
    if not 0 <= real_ratio <= 1:
        raise ValueError("real_ratio must be between 0 and 1.")

    if len(real_df) == 0:
        raise ValueError("real_df cannot be empty.")

    total_rows = len(real_df)
    real_rows = int(total_rows * real_ratio)
    synthetic_rows = total_rows - real_rows

    if real_rows > len(real_df):
        raise ValueError("Not enough rows in real_df.")

    if synthetic_rows > len(synthetic_df):
        raise ValueError(
            "synthetic_df does not contain enough rows for the requested hybrid."
        )

    real_sample = real_df.sample(
        n=real_rows,
        random_state=random_state,
    )

    synthetic_sample = synthetic_df.sample(
        n=synthetic_rows,
        random_state=random_state,
    )

    hybrid_df = pd.concat(
        [real_sample, synthetic_sample],
        ignore_index=True,
    )

    return hybrid_df
