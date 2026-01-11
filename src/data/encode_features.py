"""
Feature preparation helpers: impute, log1p, frequency encode, scale, and assemble X/y.
"""

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from .constants import CAT_COLS, LOG_COLS, MIN_FREQ, NUMERIC_COLS, TARGET_COL
except ImportError:
    # Allow running as a script: add repo/src to sys.path for absolute import.
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from data.constants import CAT_COLS, LOG_COLS, MIN_FREQ, NUMERIC_COLS, TARGET_COL


def fit_frequency_encoders(train_df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, pd.Series]:
    """Fit frequency encoders (relative frequencies) per categorical column."""
    encoders = {}
    for col in cols:
        freq = train_df[col].value_counts(normalize=True)
        encoders[col] = freq
    return encoders


def apply_frequency_encoders(df: pd.DataFrame, encoders: Dict[str, pd.Series]) -> pd.DataFrame:
    """Map categories to their train frequencies; unseen → 0."""
    out = df.copy()
    for col, freq in encoders.items():
        out[col] = df[col].map(freq).fillna(0.0)
    return out


def fit_numeric_imputers(train_df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, float]:
    """Compute median imputers for numeric columns."""
    return {c: train_df[c].median() for c in cols}


def apply_numeric_imputers(df: pd.DataFrame, imputers: Dict[str, float]) -> pd.DataFrame:
    """Fill missing numerics with train medians."""
    out = df.copy()
    for c, val in imputers.items():
        out[c] = out[c].fillna(val)
    return out


def fill_unknown_categories(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Replace missing categorical entries with 'Unknown'."""
    out = df.copy()
    for c in cols:
        out[c] = out[c].fillna("Unknown")
    return out


def fit_category_maps(df: pd.DataFrame, cols: Iterable[str], min_freq: int) -> Dict[str, set]:
    """Collect categories to keep (freq >= min_freq) per column."""
    maps = {}
    for col in cols:
        counts = df[col].value_counts(dropna=True)
        keep = counts[counts >= min_freq].index
        maps[col] = set(keep)
    return maps


def apply_rare_grouping(df: pd.DataFrame, maps: Dict[str, set]) -> pd.DataFrame:
    """Group rare categories into 'Other'; keep missing/unknown tokens as-is if frequent enough."""
    out = df.copy()
    for col, keepers in maps.items():
        out[col] = out[col].where(out[col].isna() | out[col].isin(keepers), "Other")
    return out


def apply_log1p(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Apply log1p to selected numeric columns (clip negatives to zero)."""
    out = df.copy()
    for c in cols:
        out[c] = np.log1p(out[c].clip(lower=0))
    return out


def fit_scaler(train_df: pd.DataFrame, cols: Iterable[str]) -> StandardScaler:
    """Fit standard scaler on numeric columns."""
    scaler = StandardScaler()
    scaler.fit(train_df[list(cols)])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, cols: Iterable[str]) -> np.ndarray:
    return scaler.transform(df[list(cols)])


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Build X/y matrices for train/val/test with consistent preprocessing.

    Returns:
      X_train, X_val, X_test (np.ndarray)
      y_train, y_val, y_test (np.ndarray)
      artifacts: dict with encoders, category_maps, and scaler
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Fill missing categoricals with a dedicated token, then group rare categories.
    train_df_cat = fill_unknown_categories(train_df, CAT_COLS)
    val_df_cat = fill_unknown_categories(val_df, CAT_COLS)
    test_df_cat = fill_unknown_categories(test_df, CAT_COLS)

    category_maps = fit_category_maps(train_df_cat, CAT_COLS, MIN_FREQ)
    train_df_cat = apply_rare_grouping(train_df_cat, category_maps)
    val_df_cat = apply_rare_grouping(val_df_cat, category_maps)
    test_df_cat = apply_rare_grouping(test_df_cat, category_maps)

    train_num = train_df
    val_num = val_df
    test_num = test_df

    # log1p numerics
    train_num = apply_log1p(train_num, LOG_COLS)
    val_num = apply_log1p(val_num, LOG_COLS)
    test_num = apply_log1p(test_num, LOG_COLS)

    # Frequency encoders
    encoders = fit_frequency_encoders(train_df_cat, CAT_COLS)
    train_enc = apply_frequency_encoders(train_df_cat, encoders)
    val_enc = apply_frequency_encoders(val_df_cat, encoders)
    test_enc = apply_frequency_encoders(test_df_cat, encoders)

    # Scale numerics
    scaler = fit_scaler(train_num, NUMERIC_COLS)
    train_num_scaled = apply_scaler(train_num, scaler, NUMERIC_COLS)
    val_num_scaled = apply_scaler(val_num, scaler, NUMERIC_COLS)
    test_num_scaled = apply_scaler(test_num, scaler, NUMERIC_COLS)

    # Assemble features
    def assemble(num_scaled, enc_df):
        return np.hstack([num_scaled, enc_df[CAT_COLS].to_numpy()])

    X_train = assemble(train_num_scaled, train_enc)
    X_val = assemble(val_num_scaled, val_enc)
    X_test = assemble(test_num_scaled, test_enc)

    y_train = np.log1p(train_df[TARGET_COL].values).astype(np.float32)
    y_val = np.log1p(val_df[TARGET_COL].values).astype(np.float32)
    y_test = np.log1p(test_df[TARGET_COL].values).astype(np.float32)

    artifacts = {
        "encoders": encoders,
        "category_maps": category_maps,
        "scaler": scaler,
    }
    return (
        X_train.astype(np.float32),
        X_val.astype(np.float32),
        X_test.astype(np.float32),
        y_train,
        y_val,
        y_test,
        artifacts,
    )