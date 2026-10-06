from sklearn.ensemble import RandomForestClassifier

from core.synthesizer import generate_synthetic_data
from core.data_handler import create_hybrid_dataset
from core.evaluator import evaluate_model

from core.quality_analyzer import (
    create_diversity_scaler,
    calculate_diversity,
    calculate_class_balance,
    calculate_wasserstein_distance
)

from core.collapse_index import (
    calculate_performance_index,
    calculate_distribution_index,
    calculate_collapse_index
)


# ============================================================
# CLASSIFIER
# ============================================================

def train_model(train_df, target_column):

    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model


def make_predictions(model, test_df, target_column):

    X_test = test_df.drop(columns=[target_column])

    predictions = model.predict(X_test)

    return predictions


# ============================================================
# ONE GENERATION
# ============================================================

def run_generation(
    train_df,
    test_df,
    target_column,
    baseline_metrics,
    real_reference_df,
    real_diversity,
    diversity_scaler,
    numerical_columns,
    real_ratio=0.7
):
    """
    Run one generation of the recursive experiment.

    Important:
    - train_df is the previous generation's hybrid dataset.
    - real_reference_df is ALWAYS the original real training pool.
    - The frozen test set is used only for model evaluation.
    """

    # --------------------------------------------------
    # 1. Generate synthetic data
    # --------------------------------------------------

    total_rows = len(train_df)

    synthetic_rows = (
        total_rows
        - int(total_rows * real_ratio)
    )

    synthetic_df = generate_synthetic_data(
        train_df,
        synthetic_rows
    )

    # --------------------------------------------------
    # 2. Calculate raw distribution metrics
    #
    # IMPORTANT:
    # All distribution metrics compare the synthetic
    # generation against the ORIGINAL real training pool.
    # --------------------------------------------------

    synthetic_diversity = calculate_diversity(
        synthetic_df,
        diversity_scaler,
        numerical_columns
    )

    class_balance = calculate_class_balance(
        real_reference_df,
        synthetic_df,
        target_column
    )

    wasserstein = calculate_wasserstein_distance(
        real_reference_df,
        synthetic_df
    )

    # --------------------------------------------------
    # 3. Calculate distribution index
    # --------------------------------------------------

    distribution_index = calculate_distribution_index(
        wasserstein=wasserstein,
        class_balance_difference=class_balance,
        real_diversity=real_diversity,
        synthetic_diversity=synthetic_diversity
    )

    # --------------------------------------------------
    # 4. Create real + synthetic hybrid dataset
    # --------------------------------------------------

    hybrid_df = create_hybrid_dataset(
        train_df,
        synthetic_df,
        real_ratio
    )

    # --------------------------------------------------
    # 5. Train classifier
    # --------------------------------------------------

    model = train_model(
        hybrid_df,
        target_column
    )

    # --------------------------------------------------
    # 6. Evaluate classifier on frozen test set
    # --------------------------------------------------

    predictions = make_predictions(
        model,
        test_df,
        target_column
    )

    y_true = test_df[target_column]

    metrics = evaluate_model(
        y_true,
        predictions
    )

    # --------------------------------------------------
    # 7. Calculate Performance Index
    # --------------------------------------------------

    performance_index = calculate_performance_index(
        baseline_metrics=baseline_metrics,
        current_metrics=metrics
    )

    # --------------------------------------------------
    # 8. Calculate final Collapse Index
    # --------------------------------------------------

    collapse_index = calculate_collapse_index(
        performance_index=performance_index,
        distribution_index=distribution_index
    )

    # --------------------------------------------------
    # 9. Calculate individual performance degradations
    #
    # These are returned for transparency and analysis.
    # --------------------------------------------------

    def performance_degradation(
        baseline,
        current
    ):
        if baseline == 0:
            return 0.0

        degradation = (
            baseline - current
        ) / baseline

        return max(
            0.0,
            min(1.0, degradation)
        )

    accuracy_degradation = performance_degradation(
        baseline_metrics["accuracy"],
        metrics["accuracy"]
    )

    precision_degradation = performance_degradation(
        baseline_metrics["precision"],
        metrics["precision"]
    )

    recall_degradation = performance_degradation(
        baseline_metrics["recall"],
        metrics["recall"]
    )

    f1_degradation = performance_degradation(
        baseline_metrics["f1"],
        metrics["f1"]
    )

    # --------------------------------------------------
    # 10. Return generation results
    # --------------------------------------------------

    return hybrid_df, {
        **metrics,

        # Raw distribution metrics
        "diversity": synthetic_diversity,
        "class_balance": class_balance,
        "wasserstein": wasserstein,

        # Individual performance degradation
        "accuracy_degradation": accuracy_degradation,
        "precision_degradation": precision_degradation,
        "recall_degradation": recall_degradation,
        "f1_degradation": f1_degradation,

        # Group indices
        "performance_index": performance_index,
        "distribution_index": distribution_index,

        # Final CI
        "collapse_index": collapse_index
    }


# ============================================================
# COMPLETE EXPERIMENT
# ============================================================

def run_experiment(
    train_df,
    test_df,
    target_column,
    generations=10,
    real_ratio=0.7
):
    """
    Run the complete recursive experiment.

    G0:
        Real-only classifier baseline.

    G1 onwards:
        SDV generates synthetic data from the previous
        generation's hybrid dataset.

    Distribution metrics are ALWAYS measured against
    the original real training pool.

    The test set remains frozen throughout the experiment.
    """

    results = []

    # ==================================================
    # CREATE FIXED REAL REFERENCE
    # ==================================================

    real_reference_df = train_df.copy()

    # --------------------------------------------------
    # Fit diversity scaler ONCE using original real data
    # --------------------------------------------------

    diversity_scaler, numerical_columns = (
        create_diversity_scaler(
            real_reference_df
        )
    )

    # --------------------------------------------------
    # Calculate diversity of original real data
    # --------------------------------------------------

    real_diversity = calculate_diversity(
        real_reference_df,
        diversity_scaler,
        numerical_columns
    )

    # ==================================================
    # G0 - REAL DATA BASELINE
    # ==================================================

    model = train_model(
        train_df,
        target_column
    )

    predictions = make_predictions(
        model,
        test_df,
        target_column
    )

    y_true = test_df[target_column]

    metrics = evaluate_model(
        y_true,
        predictions
    )

    # --------------------------------------------------
    # Store G0 model-performance baseline
    # --------------------------------------------------

    baseline_metrics = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"]
    }

    # --------------------------------------------------
    # G0 has no synthetic data.
    #
    # Therefore distribution degradation metrics
    # are not calculated for G0.
    #
    # CI_0 = 0 by definition.
    # --------------------------------------------------

    results.append({
        "generation": 0,

        **metrics,

        # Real reference diversity
        "diversity": real_diversity,

        # Not applicable for G0
        "class_balance": None,
        "wasserstein": None,

        # No degradation at baseline
        "accuracy_degradation": 0.0,
        "precision_degradation": 0.0,
        "recall_degradation": 0.0,
        "f1_degradation": 0.0,

        "performance_index": 0.0,
        "distribution_index": 0.0,

        "collapse_index": 0.0
    })

    # ==================================================
    # G1 onwards
    # ==================================================

    current_train_df = train_df

    for generation in range(1, generations):

        hybrid_df, generation_metrics = run_generation(
            train_df=current_train_df,
            test_df=test_df,
            target_column=target_column,
            baseline_metrics=baseline_metrics,
            real_reference_df=real_reference_df,
            real_diversity=real_diversity,
            diversity_scaler=diversity_scaler,
            numerical_columns=numerical_columns,
            real_ratio=real_ratio
        )

        results.append({
            "generation": generation,
            **generation_metrics
        })

        # --------------------------------------------------
        # Recursive step:
        #
        # The current hybrid becomes the training data
        # used by SDV for the next generation.
        # --------------------------------------------------

        current_train_df = hybrid_df

    return results