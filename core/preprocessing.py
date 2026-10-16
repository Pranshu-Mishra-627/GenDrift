"""Leakage-aware preprocessing for mixed-type tabular classification."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class TabularPreprocessor:
    """Encode each input column to numeric values while preserving column names.

    Numeric imputation values and categorical vocabularies are learned only
    from the dataframe passed to ``fit``. Unseen categories at transform time
    receive -1. Each input feature remains one output feature, which keeps the
    SHAP feature names interpretable.
    """

    def __init__(self):
        self.columns = None
        self.numeric_columns = None
        self.categorical_columns = None
        self.numeric_fill_values = {}
        self.categorical_fill_values = {}
        self.encoder = None

    def fit(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        self.columns = list(X.columns)
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = [c for c in self.columns if c not in self.numeric_columns]

        for column in self.numeric_columns:
            values = pd.to_numeric(X[column], errors="coerce")
            median = values.replace([np.inf, -np.inf], np.nan).median()
            self.numeric_fill_values[column] = float(median) if pd.notna(median) else 0.0

        if self.categorical_columns:
            cat = X[self.categorical_columns].copy()
            for column in self.categorical_columns:
                non_missing = cat[column].dropna()
                fill = non_missing.mode(dropna=True)
                self.categorical_fill_values[column] = str(fill.iloc[0]) if len(fill) else "__MISSING__"
                cat[column] = cat[column].where(cat[column].notna(), self.categorical_fill_values[column]).astype(str)
            self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.encoder.fit(cat)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        output = pd.DataFrame(index=X.index)
        for column in self.numeric_columns:
            values = pd.to_numeric(X[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            output[column] = values.fillna(self.numeric_fill_values[column]).astype(float)
        if self.categorical_columns:
            cat = X[self.categorical_columns].copy()
            for column in self.categorical_columns:
                cat[column] = cat[column].where(
                    cat[column].notna(), self.categorical_fill_values[column]
                ).astype(str)
            encoded = self.encoder.transform(cat)
            for index, column in enumerate(self.categorical_columns):
                output[column] = encoded[:, index].astype(float)
        return output[self.columns]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
