# GenDrift: Synthetic Data Lab

**An interactive platform for studying synthetic tabular data reliability across generations.**

GenDrift is an academic AI project that generates synthetic tabular data and studies how its statistical characteristics and downstream machine-learning utility change when synthetic data is used repeatedly. It brings together generation, model evaluation, data-quality measurements, feature-importance tracking, and an experimental forecast of the project's Collapse Index in one Streamlit interface.

> **Important research note:** GenDrift is an experimental analysis tool, not a guarantee that synthetic data is safe, private, or interchangeable with real data. Its Collapse Index (CI) is a project-defined indicator, not a calibrated probability of model collapse. The LSTM forecast is experimental.

## Contents

- [What the project does](#what-the-project-does)
- [How the workflow works](#how-the-workflow-works)
- [Technology stack](#technology-stack)
- [Requirements](#requirements)
- [Run locally](#run-locally)
- [Using the web interface](#using-the-web-interface)
- [Reproducing the documented experiment](#reproducing-the-documented-experiment)
- [Outputs and generated files](#outputs-and-generated-files)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Privacy and deployment](#privacy-and-deployment)
- [Team](#team)

## What the project does

GenDrift is designed to make it easier to inspect synthetic tabular data over multiple generations rather than evaluating only one generated file.

The application supports these main tasks:

1. **Dataset inspection:** load tabular data, select a target column, and inspect basic dataset properties.
2. **Synthetic data generation:** generate tabular data with the SDV GaussianCopula synthesizer.
3. **Downstream model evaluation:** train a classifier on a real/synthetic hybrid training dataset and evaluate it against a held-out real test set.
4. **Quality and drift analysis:** examine diversity, target class balance, and distributional differences such as Wasserstein distance.
5. **Generation indicators:** calculate the project's Predictive Index (PI), Drift Index (DI), and Collapse Index (CI).
6. **Feature-importance tracking:** use SHAP-related analysis to inspect feature importance and changes across generations.
7. **Hyperparameter optimization:** use Optuna to search classifier settings. The search budget is configurable and can be small for demonstration runs.
8. **Experimental forecasting:** use an LSTM model to forecast a future CI value from previous generation observations.
9. **Export:** inspect and download generation metrics and raw synthetic datasets where enabled by the current application version.

## How the workflow works

At a high level, the experiment follows this sequence:

1. Load a tabular dataset and choose the prediction target.
2. Split the real data into training and held-out test subsets.
3. Evaluate a baseline, **G0**, using real training data.
4. Fit the synthetic-data generator and create a synthetic generation.
5. Construct the configured hybrid training dataset using real and synthetic rows.
6. Train/evaluate the downstream classifier against the same held-out real test set.
7. Calculate and save predictive metrics, data-quality measures, and PI/DI/CI indicators.
8. Repeat for the requested number of generations and inspect the resulting trends.
9. Optionally run feature-importance tracking, hyperparameter optimization, and the experimental LSTM forecast.

Keeping the test split fixed makes comparisons within a run more consistent. It does not, by itself, establish that results generalize to other datasets or random seeds.

## Technology stack

- **Python** for the application and experiment pipeline
- **Streamlit** for the interactive web interface
- **Pandas and NumPy** for tabular data handling
- **SDV GaussianCopula** for synthetic tabular data generation
- **Scikit-learn** for preprocessing, classification, splitting, and evaluation
- **SciPy** for statistical distance calculations
- **SHAP** for model explainability / feature-importance analysis, where enabled
- **Optuna** for hyperparameter optimization, where enabled
- **TensorFlow/Keras** for the experimental LSTM predictor, where enabled

The exact dependency list should be taken from the repository's `requirements.txt`. Optional modules may require additional packages depending on the version of the project being run.

## Requirements

- Python 3.10 or 3.11 is recommended for the project version used during development.
- Git, if cloning the repository.
- A terminal such as PowerShell, Command Prompt, or the VS Code terminal.
- Internet access for the initial package installation.
- Enough memory for the selected dataset and enabled analysis modules. LSTM/TensorFlow dependencies can make installation larger and may behave differently across operating systems.

## Run locally

### 1. Get the code

Clone the repository after replacing the placeholder URL with the actual GitHub repository URL:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Alternatively, download the repository ZIP from GitHub and extract it.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If installation fails, check that you are using a supported Python version and review the package's error message. Do not randomly install packages one by one without updating `requirements.txt`, because that makes the environment difficult to reproduce.

### 4. Start the application

From the repository root, run:

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open that URL in a browser.

If the command fails because Streamlit is not installed, confirm that the virtual environment is active and that `streamlit` is listed in `requirements.txt`.

## Using the web interface

The names and exact layout of tabs may vary slightly between project versions.

1. Open the application in the browser.
2. Load the provided example dataset, if available, or upload a CSV file.
3. Select the target column for the prediction task.
4. Review the dataset preflight information, including row/column counts, missing values, duplicates, and target distribution.
5. Configure the experiment: generation count, random seed, real-data share, and optional analysis settings.
6. Start the experiment and wait for it to finish.
7. Review the generation metrics and trend plots. Compare predictive performance with the quality/drift indicators rather than relying on accuracy alone.
8. Inspect the SHAP and optimization outputs if enabled.
9. Review the CI forecast as an experimental result, not a guaranteed prediction.
10. Open the downloads/outputs area to retrieve the metrics and synthetic CSV files made available by the current run.

**For a demonstration:** use a small dataset or the already prepared Bank Marketing run. Avoid launching a lengthy experiment during a recorded presentation.

## Reproducing the documented experiment

The project report documents the following Bank Marketing experiment:

| Setting | Value |
|---|---|
| Dataset | UCI Bank Marketing, `bank-additional-full` variant |
| Rows / columns | 41,188 / 20 |
| Target | `y` |
| Random seed | 42 |
| Generations | G0 through G5 (six stages including baseline) |
| Real-data share | 0.70 |
| Optuna | Enabled, five trials |
| SHAP | Enabled, maximum 50 samples |

The documented target counts were 36,548 `no` and 4,640 `yes`. In the recorded run, synthetic generations G1–G5 each contained 9,885 rows. These are the results of that particular configuration, not a promise that every run or version will produce identical outputs.

To reproduce it, use the same dataset variant, target, seed, generation settings, real-data share, and optional analysis settings. Dataset preprocessing, package versions, and implementation changes can still affect the result. The report's saved CSV/JSON artifacts can be used to compare against the documented run if they are distributed separately.

**Dataset source:** UCI Machine Learning Repository, Bank Marketing  
https://archive.ics.uci.edu/dataset/222/bank+marketing  
Dataset DOI: https://doi.org/10.24432/C5K306

Check the UCI dataset terms and citation guidance before redistributing the dataset itself.

## Outputs and generated files

Depending on the enabled features and the current version, the application may generate:

- Per-generation metric tables in CSV and/or JSON format
- SHAP history and feature-drift summaries
- Raw synthetic CSV files for generations after G0
- Plots and optimization summaries
- LSTM evaluation metadata and forecasts

Experiment outputs should be treated as generated artifacts, not source code. They are normally written under a results/output directory. Large generated files, trained models, local caches, and personal datasets should not be committed to Git unless there is a clear reason to distribute them.

The exact output folder name may include the dataset name, seed, or experiment settings.

## Testing

The repository includes test scripts for core components such as data loading/splitting, preprocessing, synthesis, quality analysis, optimization, SHAP tracking, and the generation pipeline.

Where tests are written as standalone scripts, run them directly from the repository root, for example:

```powershell
python test_generation.py
python test_quality.py
python test_synthesizer.py
python test_trainer.py
```

Tests under the `tests/` directory can be run using their module path if the package structure supports it, for example:

```powershell
python -m tests.test_data_handler
python -m tests.test_preprocessing
python -m tests.test_optimizer
python -m tests.test_shap_tracker
python -m tests.test_offline_pipeline
```

Some test scripts may launch experiments, use mocks, or require optional dependencies. If a test fails, read the first traceback and identify whether the failure is an import/environment issue, an expected-value mismatch, or a runtime error. Do not report a test as passing unless it was actually run successfully in the current environment.

## Project structure

The core repository is expected to contain files similar to these (the exact set may differ by project version):

```text
GenDrift/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── core/
│   ├── data_handler.py
│   ├── preprocessing.py
│   ├── synthesizer.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── quality_analyzer.py
│   ├── collapse_index.py
│   ├── shap_tracker.py
│   ├── optimizer.py
│   └── offline_pipeline.py
├── tests/
│   ├── test_data_handler.py
│   ├── test_preprocessing.py
│   ├── test_optimizer.py
│   ├── test_shap_tracker.py
│   └── test_offline_pipeline.py
└── data/
    └── iris.csv   # small example dataset, if included
```

## Limitations

- The documented result covers one dataset and one seed. It is a case study, not a universal conclusion.
- CI is a project-defined composite indicator. Its values and any thresholds should not be interpreted as calibrated probabilities.
- The reported offline LSTM evaluation had a negative R² (approximately −0.42). Although its MAE was about 9.3% better than a persistence baseline in that evaluation, the forecasting component remains experimental.
- Optuna used five trials in the documented run, which is a limited search budget.
- SHAP used a maximum of 50 samples in the documented run, so the explainability analysis is bounded by that configuration.
- The workflow does **not** implement a formal differential-privacy guarantee. Synthetic data should not be described as private merely because it is synthetic.
- Results can vary with the dataset, preprocessing, random seed, dependency versions, and model settings.


## Team

- **Pranshu Mishra**
- **Rishabh Batra**

## Acknowledgements and references

- SDV documentation: https://docs.sdv.dev/sdv
- SDMetrics documentation: https://docs.sdv.dev/sdmetrics
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Streamlit documentation: https://docs.streamlit.io/
- Optuna documentation: https://optuna.readthedocs.io/
- SHAP documentation: https://shap.readthedocs.io/
- UCI Bank Marketing dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing
