"""
Shared column name constants for feature engineering and data loading.
"""

NUMERIC_COLS = ["budget", "votes", "runtime", "year"]
CAT_COLS = ["rating", "genre", "director", "writer", "star", "country", "company"]
LOG_COLS = ["budget", "votes"]
TARGET_COL = "gross"
MIN_FREQ = 5

__all__ = ["NUMERIC_COLS", "CAT_COLS", "LOG_COLS", "TARGET_COL", "MIN_FREQ"]
