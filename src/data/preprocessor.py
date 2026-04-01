"""
Data loading and preprocessing for the mushroom dataset.

primary_data.csv  — 173 real species (initial labeled set)
  Numeric ranges stored as "[10, 20]" → midpoint.
  Categorical multi-values stored as "[x, f]" → first value.
  Extra columns (family, name) are ignored.

secondary_data.csv — 61,069 simulated instances (unlabeled pool)
  Single values throughout; semicolon-delimited.

Categorical features are ordinal-encoded and the encoder is persisted
so that inference time uses the same mapping.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# ── feature schema ─────────────────────────────────────────────────────────
NUMERIC_FEATURES = ["cap-diameter", "stem-height", "stem-width"]
CATEGORICAL_FEATURES = [
    "cap-shape", "cap-surface", "cap-color",
    "does-bruise-or-bleed",
    "gill-attachment", "gill-spacing", "gill-color",
    "stem-root", "stem-surface", "stem-color",
    "veil-type", "veil-color",
    "has-ring", "ring-type",
    "spore-print-color",
    "habitat", "season",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "class"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dt", "MushroomDataset")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def _parse_numeric(value) -> float:
    """
    Parse a numeric feature value from either dataset format:
      - "[10, 20]"  → midpoint 15.0  (primary_data range)
      - "15.26"     → 15.26          (secondary_data single value)
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    # bracket range: [lo, hi]
    m = re.match(r"^\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_categorical(value) -> str:
    """
    Parse a categorical feature value:
      - "[x, f]"  → "x"  (take first value from primary_data list)
      - "x"       → "x"  (secondary_data single value)
    """
    if pd.isna(value):
        return "unknown"
    s = str(value).strip()
    m = re.match(r"^\[\s*(\S+?)(?:\s*,.*?)?\s*\]$", s)
    if m:
        return m.group(1).strip()
    return s if s else "unknown"


def _load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, sep=";", engine="python")
    return df


def load_secondary(labeled: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) from secondary_data — full labeled set."""
    df = _load_csv("secondary_data.csv")
    df = _align_columns(df)
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].map({"e": 0, "p": 1}).astype(int)
    return X, y


def load_primary() -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) from primary_data — real species labeled set."""
    df = _load_csv("primary_data.csv")
    df = _align_columns(df)
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].map({"e": 0, "p": 1}).astype(int)
    return X, y


def _align_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and keep only the expected columns."""
    df.columns = [c.strip().lower().replace("_", "-") for c in df.columns]
    for col in ALL_FEATURES + [TARGET]:
        if col not in df.columns:
            df[col] = np.nan
    return df


# ── Preprocessor class ─────────────────────────────────────────────────────

class MushroomPreprocessor:
    """
    Fit / transform the mushroom feature matrix.

    Usage
    -----
    prep = MushroomPreprocessor()
    X_train = prep.fit_transform(X_raw)
    X_new   = prep.transform(X_new_raw)
    prep.save()                          # persists to models/
    prep = MushroomPreprocessor.load()   # restore
    """

    def __init__(self):
        self.ord_enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self.numeric_medians: dict[str, float] = {}
        self._fitted = False

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        # 1. parse numeric cols (handles both "[lo, hi]" and plain floats)
        for col in NUMERIC_FEATURES:
            X[col] = X[col].apply(_parse_numeric)
            self.numeric_medians[col] = float(np.nanmedian(X[col]))
            X[col] = X[col].fillna(self.numeric_medians[col])

        # 2. parse categoricals (handles "[x, f]" lists and plain values)
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].apply(_parse_categorical)

        # 3. fit + transform categorical encoder
        cat_matrix = self.ord_enc.fit_transform(X[CATEGORICAL_FEATURES])

        self._fitted = True
        return np.hstack([X[NUMERIC_FEATURES].values, cat_matrix])

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_transform first or load a saved preprocessor.")
        X = X.copy()
        for col in NUMERIC_FEATURES:
            X[col] = X[col].apply(_parse_numeric)
            X[col] = X[col].fillna(self.numeric_medians[col])
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].apply(_parse_categorical)
        cat_matrix = self.ord_enc.transform(X[CATEGORICAL_FEATURES])
        return np.hstack([X[NUMERIC_FEATURES].values, cat_matrix])

    def feature_dict_to_array(self, feature_dict: dict) -> np.ndarray:
        """
        Convert a dict of {feature_name: value} (as returned by Gemini)
        into a 2-D numpy array ready for the classifier.
        """
        row = {col: feature_dict.get(col, np.nan) for col in ALL_FEATURES}
        df = pd.DataFrame([row])
        return self.transform(df)

    def save(self, path: str | None = None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = path or os.path.join(MODEL_DIR, "preprocessor.joblib")
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | None = None) -> "MushroomPreprocessor":
        path = path or os.path.join(MODEL_DIR, "preprocessor.joblib")
        return joblib.load(path)
