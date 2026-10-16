import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def create_diversity_scaler(real_df, target_column, numerical_columns=None):
    """Fit the numerical scaler ONCE on the original real training pool."""
    if target_column not in real_df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")
    if numerical_columns is None:
        numerical_columns = real_df.select_dtypes(include=np.number).columns.tolist()
    numerical_columns = [c for c in numerical_columns if c != target_column]
    if len(numerical_columns) < 2:
        raise ValueError("Diversity calculation requires at least two numerical features excluding the target column.")
    missing = [c for c in numerical_columns if c not in real_df.columns]
    if missing:
        raise ValueError(f"Missing numerical columns: {missing}")
    scaler = StandardScaler()
    scaler.fit(real_df[numerical_columns])
    return scaler, numerical_columns


def calculate_diversity(df, scaler, numerical_columns, max_samples=1000, random_state=42):
    """Mean pairwise Euclidean distance after fixed-reference scaling.

    For larger datasets, estimate diversity from a reproducible sample to
    avoid allocating a full quadratic distance matrix.
    """
    if len(numerical_columns) < 2:
        raise ValueError("Diversity calculation requires at least two numerical features.")
    missing = [c for c in numerical_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing numerical columns: {missing}")
    if len(df) < 2:
        raise ValueError("Diversity calculation requires at least two rows.")
    sample = df
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2.")
    if len(df) > max_samples:
        sample = df.sample(n=max_samples, random_state=random_state)
    scaled_data = scaler.transform(sample[numerical_columns])
    distances = pairwise_distances(scaled_data, metric="euclidean")
    upper_triangle = distances[np.triu_indices(len(scaled_data), k=1)]
    if len(upper_triangle) == 0:
        raise ValueError("Diversity calculation requires at least two rows.")
    return float(np.mean(upper_triangle))


def calculate_class_balance(real_df, synthetic_df, target_column):
    """Total variation distance between target-class distributions."""
    if target_column not in real_df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in real_df.")
    if target_column not in synthetic_df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in synthetic_df.")
    real_distribution = real_df[target_column].value_counts(normalize=True)
    synthetic_distribution = synthetic_df[target_column].value_counts(normalize=True)
    all_classes = sorted(set(real_distribution.index) | set(synthetic_distribution.index), key=str)
    real_distribution = real_distribution.reindex(all_classes, fill_value=0)
    synthetic_distribution = synthetic_distribution.reindex(all_classes, fill_value=0)
    return float(abs(real_distribution - synthetic_distribution).sum() / 2)


def calculate_wasserstein_distance(real_df, synthetic_df, numerical_columns, scaler=None):
    """Mean per-feature Wasserstein distance.

    If ``scaler`` is supplied, both datasets are transformed using the scaler
    fitted on the original real training pool before distances are calculated.
    This makes the averaged distance less dependent on raw feature units.
    """
    if len(numerical_columns) == 0:
        raise ValueError("Wasserstein distance requires at least one numerical feature.")
    for column in numerical_columns:
        if column not in real_df.columns:
            raise ValueError(f"Missing column '{column}' in real_df.")
        if column not in synthetic_df.columns:
            raise ValueError(f"Missing column '{column}' in synthetic_df.")
    if scaler is None:
        real_values = real_df[numerical_columns]
        synthetic_values = synthetic_df[numerical_columns]
    else:
        real_values = scaler.transform(real_df[numerical_columns])
        synthetic_values = scaler.transform(synthetic_df[numerical_columns])
    real_array = np.asarray(real_values, dtype=float)
    synthetic_array = np.asarray(synthetic_values, dtype=float)
    distances = [
        wasserstein_distance(real_array[:, i], synthetic_array[:, i])
        for i, _ in enumerate(numerical_columns)
    ]
    return float(np.mean(distances))
