import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.stats import wasserstein_distance


# ============================================================
# DIVERSITY
# ============================================================

def create_diversity_scaler(real_df, numerical_columns=None):
    """
    Fit a StandardScaler using the original real training data.

    The fitted scaler should be created once at the beginning
    of an experiment and reused for every generation.
    """

    if numerical_columns is None:
        numerical_columns = real_df.select_dtypes(
            include=np.number
        ).columns.tolist()

    if len(numerical_columns) < 2:
        raise ValueError(
            "Diversity calculation requires at least two numerical features."
        )

    scaler = StandardScaler()

    scaler.fit(
        real_df[numerical_columns]
    )

    return scaler, numerical_columns


def calculate_diversity(
    df,
    scaler,
    numerical_columns
):
    """
    Calculate dataset diversity using the fixed scaler
    fitted on the original real training data.

    Diversity is measured as the mean pairwise Euclidean
    distance between rows.
    """

    if len(numerical_columns) < 2:
        raise ValueError(
            "Diversity calculation requires at least two numerical features."
        )

    numerical_data = df[numerical_columns]

    scaled_data = scaler.transform(
        numerical_data
    )

    distances = pairwise_distances(
        scaled_data,
        metric="euclidean"
    )

    n = len(scaled_data)

    upper_triangle = distances[
        np.triu_indices(n, k=1)
    ]

    if len(upper_triangle) == 0:
        raise ValueError(
            "Diversity calculation requires at least two rows."
        )

    diversity = np.mean(
        upper_triangle
    )

    return float(diversity)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def calculate_class_balance(
    real_df,
    synthetic_df,
    target_column
):
    """
    Measure the difference between real and synthetic
    class distributions using Total Variation Distance.

    Returns a value in [0, 1].

    0 = identical class distributions
    1 = maximally different class distributions
    """

    real_distribution = (
        real_df[target_column]
        .value_counts(normalize=True)
    )

    synthetic_distribution = (
        synthetic_df[target_column]
        .value_counts(normalize=True)
    )

    all_classes = (
        set(real_distribution.index)
        | set(synthetic_distribution.index)
    )

    real_distribution = real_distribution.reindex(
        all_classes,
        fill_value=0
    )

    synthetic_distribution = synthetic_distribution.reindex(
        all_classes,
        fill_value=0
    )

    difference = (
        abs(
            real_distribution
            - synthetic_distribution
        ).sum()
        / 2
    )

    return float(difference)


# ============================================================
# WASSERSTEIN DISTANCE
# ============================================================

def calculate_wasserstein_distance(
    real_df,
    synthetic_df
):
    """
    Calculate the average Wasserstein distance between
    corresponding numerical features of the original
    real training data and synthetic data.

    The real dataset passed to this function should always
    be the original real training pool.
    """

    numerical_columns = real_df.select_dtypes(
        include=np.number
    ).columns

    distances = []

    for column in numerical_columns:

        real_values = (
            real_df[column]
            .dropna()
        )

        synthetic_values = (
            synthetic_df[column]
            .dropna()
        )

        distance = wasserstein_distance(
            real_values,
            synthetic_values
        )

        distances.append(
            distance
        )

    if not distances:
        raise ValueError(
            "Wasserstein distance requires at least one numerical feature."
        )

    return float(
        np.mean(distances)
    )