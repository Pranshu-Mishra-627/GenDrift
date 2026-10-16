
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from core.data_handler import load_dataset
from core.offline_pipeline import run_offline_experiment

APP_TITLE = "GenDrift | Synthetic Data Collapse Monitor"
DEFAULT_DATA = Path("data/bank_marketing/bank_marketing_no_duration.csv")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Warm editorial palette */
    .stApp {
        background-color: #F7F3EC;
        color: #332D27;
    }

    [data-testid="stSidebar"] {
        background-color: #EDE4D7;
        border-right: 1px solid #D8C8B5;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1450px;
    }

    /* Terracotta hero panel */
    .hero {
        padding: 2rem;
        border-radius: 14px;
        background: linear-gradient(115deg, #704536, #9B6048, #B78360);
        color: #FFF8F0;
        margin-bottom: 1.5rem;
        border: 1px solid #8B5B46;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.3rem;
        font-weight: 650;
        letter-spacing: -0.7px;
        color: #FFF8F0;
    }

    .hero p {
        color: #F4E4D5;
        margin-top: 0.65rem;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Warm, understated metric cards */
    div[data-testid="stMetric"] {
        background-color: #FFFCF7;
        border: 1px solid #E2D5C6;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: none;
    }

    div[data-testid="stMetricLabel"] {
        color: #766658;
    }

    div[data-testid="stMetricValue"] {
        color: #49382D;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background-color: #98543C;
        border: 1px solid #98543C;
        color: #FFF9F3;
        border-radius: 8px;
        font-weight: 600;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #78412F;
        border-color: #78412F;
        color: #FFFFFF;
    }

    .stButton > button {
        border-radius: 8px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #6E5A4B;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #98543C;
    }

    /* Expanders and tables */
    [data-testid="stExpander"] {
        background-color: #FFFCF7;
        border: 1px solid #E2D5C6;
        border-radius: 10px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E2D5C6;
        border-radius: 8px;
    }

    /* Subtle dividers */
    hr {
        border-color: #DED1C2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "experiment_payload" not in st.session_state:
    st.session_state.experiment_payload = None

st.markdown(
    """
    <div class="hero">
        <h1>🧬 GenDrift</h1>
        <p>
        Explore synthetic-data degradation, run actual experiments,
        understand the metrics, and forecast potential collapse.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR: DATASET AND EXPERIMENT CONFIGURATION
# ---------------------------------------------------------

dataset_df = None
dataset_label = None
uploaded = None
target = None
run_clicked = False

with st.sidebar:
    st.header("1. Dataset")

    uploaded = st.file_uploader(
        "Upload your dataset",
        type=["csv", "txt"],
        help="Comma-, semicolon-, and tab-separated files are supported.",
    )

    use_default = st.checkbox(
        "Use Bank Marketing example",
        value=DEFAULT_DATA.exists(),
        disabled=not DEFAULT_DATA.exists(),
    )

    if uploaded is not None:
        try:
            uploaded.seek(0)
            dataset_df = pd.read_csv(uploaded, sep=None, engine="python")
            dataset_label = Path(uploaded.name).stem
        except Exception as exc:
            st.error(f"Could not read the uploaded dataset: {exc}")

    elif use_default and DEFAULT_DATA.exists():
        try:
            dataset_df = load_dataset(str(DEFAULT_DATA))
            dataset_label = "bank_marketing_no_duration"
        except Exception as exc:
            st.error(f"Could not load the example dataset: {exc}")

    if dataset_df is not None and len(dataset_df.columns) > 0:
        st.success(
            f"{len(dataset_df):,} rows × {len(dataset_df.columns):,} columns"
        )

        columns = list(dataset_df.columns)
        default_target = columns.index("y") if "y" in columns else len(columns) - 1

        target = st.selectbox(
            "Target column",
            options=columns,
            index=default_target,
            help="The label the classifier will predict.",
        )

        st.header("2. Experiment settings")

        generations = st.slider(
            "Generations, including G0",
            min_value=2,
            max_value=20,
            value=10,
        )

        real_ratio = st.slider(
            "Real-data share",
            min_value=0.50,
            max_value=0.90,
            value=0.70,
            step=0.05,
            help="The remaining share is sampled from the synthetic generation.",
        )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
        )

        optimize = st.checkbox(
            "Enable Optuna tuning",
            value=False,
            help="Can significantly increase runtime.",
        )

        n_trials = st.slider(
            "Optuna trials",
            min_value=5,
            max_value=30,
            value=10,
            disabled=not optimize,
        )

        track_shap = st.checkbox("Track SHAP feature importance", value=False)

        max_shap_samples = st.slider(
            "Maximum SHAP samples",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            disabled=not track_shap,
        )

        run_clicked = st.button(
            "Run the GenDrift experiment",
            type="primary",
            use_container_width=True,
        )
    else:
        st.info("Upload a dataset to configure an experiment.")

# ---------------------------------------------------------
# DATASET PREFLIGHT
# ---------------------------------------------------------

st.subheader("Dataset preflight")

if dataset_df is None:
    st.info("Upload a CSV in the sidebar to get started.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(dataset_df):,}")
    c2.metric("Columns", f"{len(dataset_df.columns):,}")
    c3.metric("Missing cells", f"{int(dataset_df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(dataset_df.duplicated().sum()):,}")

    if len(dataset_df.columns) < 2:
        st.error("You need at least one feature column and one target column.")

    if dataset_df.columns.duplicated().any():
        st.error("Duplicate column names detected. Rename them before running.")

    if dataset_df.isna().any().any():
        st.warning(
            "Missing values detected. Missing target labels are not allowed. "
            "Review missing feature values before interpreting the results."
        )

    if target is not None and dataset_df[target].isna().any():
        st.error(f"Target column '{target}' contains missing labels.")

    if target is not None and dataset_df[target].nunique(dropna=True) < 2:
        st.error("The target must contain at least two distinct classes.")

    if len(dataset_df) < 30:
        st.warning(
            "Fewer than 30 rows. Evaluation results may be unstable."
        )

    with st.expander("Preview dataset and target distribution"):
        st.dataframe(
            dataset_df.head(25),
            use_container_width=True,
            hide_index=True,
        )

        if target is not None:
            st.write("Target class counts")
            st.dataframe(
                dataset_df[target]
                .value_counts(dropna=False)
                .rename_axis(target)
                .reset_index(name="count"),
                use_container_width=True,
                hide_index=True,
            )

# ---------------------------------------------------------
# RUN ACTUAL GENDRIFT PIPELINE
# ---------------------------------------------------------

if run_clicked and dataset_df is not None and target is not None:
    error = None

    if len(dataset_df.columns) < 2:
        error = "The dataset needs at least two columns."
    elif dataset_df.columns.duplicated().any():
        error = "Rename duplicate column names before running."
    elif dataset_df[target].isna().any():
        error = "Remove rows with missing target labels before running."
    elif dataset_df[target].nunique(dropna=True) < 2:
        error = "The target needs at least two distinct classes."
    elif len(dataset_df) < 30:
        error = "Use at least 30 rows for this prototype."

    if error:
        st.error(error)
    else:
        temp_path = None

        try:
            with st.spinner(
                "Running GenDrift: generating synthetic data, "
                "training classifiers, and calculating generation metrics..."
            ):
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    delete=False,
                    suffix=".csv",
                ) as temp_file:
                    temp_path = Path(temp_file.name)

                    if uploaded is not None:
                        temp_file.write(uploaded.getvalue())
                    else:
                        dataset_df.to_csv(
                            temp_file,
                            index=False,
                            encoding="utf-8",
                        )

                run_name = f"{dataset_label}_seed_{int(seed)}"

                output_dir = (
                    Path("results") / "streamlit_runs" / run_name
                )

                payload = run_offline_experiment(
                    dataset_path=str(temp_path),
                    target_column=target,
                    generations=int(generations),
                    real_ratio=float(real_ratio),
                    output_dir=str(output_dir),
                    dataset_name=run_name,
                    optimize=bool(optimize),
                    n_trials=int(n_trials),
                    track_shap=bool(track_shap),
                    max_shap_samples=int(max_shap_samples),
                    random_state=int(seed),
                )

                st.session_state.experiment_payload = {
                    "payload": payload,
                    "dataset": dataset_label,
                    "target": target,
                    "seed": int(seed),
                    "generations": int(generations),
                    "real_ratio": float(real_ratio),
                }

                st.success(
                    "Experiment completed. The metrics below were "
                    "calculated from this experiment."
                )

        except Exception as exc:
            st.error("The experiment failed.")
            st.exception(exc)

        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

# ---------------------------------------------------------
# EDUCATIONAL METRIC GUIDE
# ---------------------------------------------------------

with st.expander("Metric guide: what do these numbers mean?"):
    st.markdown("""
    These metrics describe the behaviour observed in an experiment.
    None of them alone proves that synthetic data is universally safe,
    realistic, or suitable for every downstream task.

    | Metric | Meaning | Interpretation | Limitation |
    |---|---|---|---|
    | **Accuracy** | Fraction of predictions that are correct. | Higher is generally better. | Can hide poor performance on rare classes. |
    | **Precision** | Fraction of positive predictions that are correct. | Higher means fewer false positives for the selected class. | Depends on which class is positive. |
    | **Recall** | Fraction of actual positives detected. | Higher means fewer false negatives. | Often trades off against precision. |
    | **F1** | Harmonic mean of precision and recall. | Summarizes both metrics. | Does not capture every application-specific cost. |
    | **Diversity** | Project-defined measure of variation in generated features. | Compare across generations and against the baseline. | Depends on the implementation and feature representation. |
    | **Class balance** | Difference in target-class proportions. | Lower generally means closer class proportions. | Matching proportions does not preserve all relationships. |
    | **Wasserstein distance** | Distribution distance calculated by the project. | Lower generally indicates closer measured distributions. | Marginal distances may miss multivariate dependencies. |
    | **Performance Index (PI)** | Composite classifier-performance degradation relative to G0. | Higher means greater degradation under the implemented formula. | Depends on the chosen metrics and formula. |
    | **Distribution Index (DI)** | Composite distribution-degradation measure. | Higher means greater degradation under the implemented formula. | Depends on component definitions and normalization. |
    | **Collapse Index (CI)** | `CI = 0.5 × PI + 0.5 × DI` in the current project. | Higher indicates more degradation under this index. | Weights and status thresholds need validation. |
    """)

    st.warning(
        "CI thresholds are provisional research choices, not validated "
        "scientific or safety cutoffs. Always inspect the component metrics."
    )

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

saved = st.session_state.experiment_payload

if saved is None:
    st.subheader("Your experiment workspace")

    c1, c2, c3 = st.columns(3)
    c1.metric("Method", "Recursive generation")
    c2.metric("Core metric", "Collapse Index")
    c3.metric("Evaluation", "Held-out test set")

    st.markdown("""
    **Workflow**

    1. Upload a tabular dataset.
    2. Choose the target column.
    3. Review the dataset and configure an experiment.
    4. Run the actual GenDrift pipeline.
    5. Inspect measured metrics and download the results.

    Results will appear after the experiment completes. The app does not
    invent run-specific metrics while waiting for a run.
    """)

else:
    payload = saved["payload"]
    results = payload["results"]

    result_df = (
        pd.DataFrame(results)
        .sort_values("generation")
        .reset_index(drop=True)
    )

    st.subheader("📊 Experiment results")

    st.caption(
        f"Dataset: {saved['dataset']} | "
        f"Target: {saved['target']} | "
        f"Seed: {saved['seed']} | "
        f"Real-data share: {saved['real_ratio']:.0%}"
    )

    if result_df.empty:
        st.warning("The pipeline returned no generation results.")
    else:
        g0 = result_df.iloc[0]
        last = result_df.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)

        if "f1" in result_df.columns:
            f1_change = (float(last["f1"]) - float(g0["f1"])) * 100

            c1.metric("Baseline F1 · G0", f"{float(g0['f1']):.4f}")
            c2.metric(
                f"Final F1 · G{int(last['generation'])}",
                f"{float(last['f1']):.4f}",
                f"{f1_change:+.2f} percentage points",
            )

        if "collapse_index" in result_df.columns:
            peak_idx = result_df["collapse_index"].idxmax()
            peak_ci = float(result_df.loc[peak_idx, "collapse_index"])
            peak_gen = int(result_df.loc[peak_idx, "generation"])

            c3.metric(
                "Final Collapse Index",
                f"{float(last['collapse_index']):.4f}",
            )
            c4.metric(
                "Peak Collapse Index",
                f"{peak_ci:.4f}",
                f"at G{peak_gen}",
            )

        st.warning(
            "CI status bands are provisional. Interpret CI alongside its "
            "component metrics and the experiment setup."
        )

        st.markdown("### Generation trajectories")

        available_metrics = [
            metric for metric in [
                "accuracy", "precision", "recall", "f1",
                "performance_index", "distribution_index",
                "collapse_index", "diversity",
                "class_balance", "wasserstein",
            ]
            if metric in result_df.columns
        ]

        defaults = [
            metric for metric in ["f1", "collapse_index"]
            if metric in available_metrics
        ]

        selected_metrics = st.multiselect(
            "Metrics to plot",
            options=available_metrics,
            default=defaults,
        )

        if selected_metrics:
            st.line_chart(
                result_df.set_index("generation")[selected_metrics],
                use_container_width=True,
            )

        overview_tab, details_tab, predictor_tab, shap_tab, synthetic_tab, download_tab = st.tabs(
            [
                "Interpretation",
                "Full results",
                "LSTM forecast",
                "SHAP",
                "Synthetic Data",
                "Downloads",
            ]
        )

        with overview_tab:
            if "f1" in result_df.columns:
                st.write(
                    f"F1 changed from {float(g0['f1']):.4f} at G0 "
                    f"to {float(last['f1']):.4f} at "
                    f"G{int(last['generation'])}."
                )

            if "collapse_index" in result_df.columns:
                st.write(
                    f"The final observed Collapse Index is "
                    f"{float(last['collapse_index']):.4f}. "
                    "This is a composite measure. Inspect PI, DI, and "
                    "their components before drawing conclusions."
                )

            component_cols = [
                col for col in [
                    "generation", "performance_index",
                    "distribution_index", "diversity",
                    "class_balance", "wasserstein",
                ]
                if col in result_df.columns
            ]

            if component_cols:
                st.dataframe(
                    result_df[component_cols],
                    use_container_width=True,
                    hide_index=True,
                )

        with details_tab:
            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "G0 is the real-data baseline. Later generations are "
                "evaluated on the held-out test set for this experiment."
            )

        with predictor_tab:
            st.markdown(
                "**Forecasts are predictions, not observations.** "
                "The LSTM is experimental."
            )

            predictor_dirs = [
                Path("results/lstm_collapse_predictor_v3_expanded"),
                Path("results/lstm_collapse_predictor_v2"),
                Path("results/lstm_collapse_predictor"),
            ]

            predictor_dir = next(
                (
                    directory for directory in predictor_dirs
                    if (directory / "predictor_metadata.json").exists()
                    and (directory / "collapse_lstm.keras").exists()
                ),
                None,
            )

            if predictor_dir is None:
                st.info(
                    "No compatible saved LSTM model was found yet. "
                    "This panel will become available after predictor "
                    "training has saved its artifacts."
                )
            elif len(result_df) < 3:
                st.info("At least three observed generations are needed.")
            else:
                try:
                    from core.collapse_predictor import (
                        load_lstm_predictor,
                        predict_future_ci,
                    )

                    predictor = load_lstm_predictor(predictor_dir)

                    prediction_args = {
                        "generation_rows": result_df.to_dict(orient="records"),
                        "model": predictor["model"],
                        "scaler": predictor["scaler"],
                        "sequence_length": int(
                            predictor.get("sequence_length", 3)
                        ),
                        "horizon": int(predictor.get("horizon", 1)),
                    }

                    # Newer predictor versions may require a target scaler.
                    try:
                        forecast = predict_future_ci(
                            **prediction_args,
                            target_scaler=predictor.get("target_scaler"),
                        )
                    except TypeError as exc:
                        if "target_scaler" not in str(exc):
                            raise
                        forecast = predict_future_ci(**prediction_args)

                    st.metric(
                        f"Predicted CI · G{forecast['predicted_generation']}",
                        f"{forecast['predicted_collapse_index']:.4f}",
                    )

                    st.caption(
                        f"Last observed generation: "
                        f"G{forecast['last_observed_generation']} | "
                        "Status: experimental"
                    )

                    st.json({
                        "forecast": forecast,
                        "evaluation_metrics": predictor.get("metrics", {}),
                    })

                except Exception as exc:
                    st.warning(f"Could not load or use the LSTM: {exc}")

        with shap_tab:
            shap_rows = []

            for item in results:
                for row in item.get("shap_importance", []) or []:
                    shap_rows.append({
                        "generation": item["generation"],
                        **row,
                    })

            if shap_rows:
                shap_df = pd.DataFrame(shap_rows)

                gen_choice = st.selectbox(
                    "Generation",
                    sorted(shap_df["generation"].unique()),
                )

                selected_shap = shap_df[
                    shap_df["generation"] == gen_choice
                ].sort_values("mean_abs_shap", ascending=False)

                st.bar_chart(
                    selected_shap.set_index("feature")[["mean_abs_shap"]].head(20),
                    use_container_width=True,
                )

                st.dataframe(
                    selected_shap,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "SHAP explains feature contributions for the fitted "
                    "model; it does not establish causality."
                )
            else:
                st.info(
                    "SHAP tracking was not enabled for this run. "
                    "Enable it in the sidebar and rerun to collect results."
                )
        with synthetic_tab:
            st.subheader("Generated Synthetic Data")
            st.caption(
                "These are raw rows generated by the synthesizer for each "
                "generation, not the hybrid classifier training dataset."
            )

            synthetic_files = payload.get("synthetic_files", {})

            if not synthetic_files:
                st.info(
                    "No synthetic datasets were saved for this run. "
                    "Run a new experiment after enabling synthetic-data export."
                )
            else:
                generation_options = sorted(
                    synthetic_files.keys(), key=int
                )
                selected_generation = st.selectbox(
                    "Choose generation",
                    generation_options,
                    format_func=lambda g: f"Generation {g}",
                    key="synthetic_generation_select",
                )

                synthetic_path = Path(
                    synthetic_files[selected_generation]
                )

                if synthetic_path.exists():
                    synthetic_df = pd.read_csv(synthetic_path)

                    st.write(
                        f"Rows: {len(synthetic_df):,} | "
                        f"Columns: {len(synthetic_df.columns):,}"
                    )

                    st.dataframe(
                        synthetic_df.head(25),
                        use_container_width=True,
                    )

                    st.download_button(
                        "Download synthetic dataset (CSV)",
                        data=synthetic_path.read_bytes(),
                        file_name=synthetic_path.name,
                        mime="text/csv",
                        key=f"download_synthetic_{selected_generation}",
                    )
                else:
                    st.error(
                        "The saved synthetic dataset file could not be found."
                    )
        with download_tab:
            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            json_bytes = json.dumps(
                results,
                indent=2,
                default=str,
            ).encode("utf-8")

            st.download_button(
                "Download metrics CSV",
                data=csv_bytes,
                file_name=f"{saved['dataset']}_metrics.csv",
                mime="text/csv",
            )

            st.download_button(
                "Download metrics JSON",
                data=json_bytes,
                file_name=f"{saved['dataset']}_metrics.json",
                mime="application/json",
            )

            if payload.get("shap_csv"):
                shap_path = Path(payload["shap_csv"])

                if shap_path.exists():
                    st.download_button(
                        "Download SHAP history CSV",
                        data=shap_path.read_bytes(),
                        file_name=shap_path.name,
                        mime="text/csv",
                    )

            st.caption(
                f"Metrics saved at {payload['metrics_csv']} "
                f"and {payload['metrics_json']}."
            )

st.markdown("---")
st.caption(
    "GenDrift research prototype. Results are empirical measurements. "
    "CI thresholds and LSTM forecasts remain experimental until validated."
)
