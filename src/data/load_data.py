"""
Data loading, filtering, and deterministic train/val/test splitting.
"""

from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .constants import CAT_COLS, NUMERIC_COLS, TARGET_COL
except ImportError:
    # Allow running as a script: add repo/src to sys.path for absolute import.
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from data.constants import CAT_COLS, NUMERIC_COLS, TARGET_COL

# Filtering thresholds
MIN_FREQ = 5

# Resolve paths relative to repo root (two levels above this file).
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "movies.csv"
DEFAULT_OUT_DIR = BASE_DIR / "data" / "processed" / "all_features"


def drop_missing_required(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Drop rows that are missing any of the required columns."""
    return df.dropna(subset=list(cols)).reset_index(drop=True)


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV or raise if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")
    return pd.read_csv(path)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic quality filters: drop rows missing required numeric/target columns."""
    out = drop_missing_required(df, NUMERIC_COLS + [TARGET_COL])
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
    """Group rare categories into 'Other'; leave missing as-is for later handling."""
    out = df.copy()
    for col, keepers in maps.items():
        out[col] = out[col].where(out[col].isin(keepers) | out[col].isna(), "Other")
    return out


def filter_and_group(df: pd.DataFrame) -> pd.DataFrame:
    """Run filters then rare-category grouping."""
    filtered = apply_filters(df)
    cat_maps = fit_category_maps(filtered, CAT_COLS, MIN_FREQ)
    grouped = apply_rare_grouping(filtered, cat_maps)
    return grouped


def train_val_test_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test with fixed seed and ratios."""
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
    test_ratio = 1.0 - train_ratio - val_ratio
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=seed, shuffle=True)
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_size), random_state=seed, shuffle=True)
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path = DEFAULT_OUT_DIR) -> None:
    """Write train/val/test CSVs to the output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)


def build_and_save_baseline_splits(
    raw_path: Path = RAW_DATA_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end: load raw, filter/group, split, and save deterministic CSVs."""
    df = load_raw(raw_path)
    filtered = filter_and_group(df)
    train_df, val_df, test_df = train_val_test_split(filtered, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}, total: {len(filtered)}")
    save_splits(train_df, val_df, test_df, out_dir=out_dir)
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Convenience entry point to regenerate baseline splits.
    build_and_save_baseline_splits()