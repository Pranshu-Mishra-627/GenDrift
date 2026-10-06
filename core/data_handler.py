import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(file_path):
    return pd.read_csv(file_path)


def split_dataset(df, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into a training pool and a frozen test set.

    The test set will remain unchanged throughout the experiment.
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    train_df = X_train.copy()
    train_df[target_column] = y_train

    test_df = X_test.copy()
    test_df[target_column] = y_test

    return train_df, test_df

def create_hybrid_dataset(real_df, synthetic_df, real_ratio=0.7):
    total_rows = len(real_df)

    real_rows = int(total_rows * real_ratio)
    synthetic_rows = total_rows - real_rows

    real_sample = real_df.sample(
        n=real_rows,
        random_state=42
    )

    synthetic_sample = synthetic_df.sample(
        n=synthetic_rows,
        random_state=42
    )

    hybrid_df = pd.concat(
        [real_sample, synthetic_sample],
        ignore_index=True
    )

    return hybrid_df