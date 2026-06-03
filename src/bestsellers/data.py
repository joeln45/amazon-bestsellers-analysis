"""Load, clean and prepare the Amazon bestsellers dataset.

The raw Kaggle file ships with human-friendly column names ("User Rating"),
mixed naming conventions, and a few data-quality quirks (the same book listed
twice in one year at different prices). Everything needed to turn it into an
analysis-ready DataFrame lives here, behind a single :func:`load_clean_data`
call so notebooks stay focused on the story rather than the plumbing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Repo root resolved relative to this file (src/bestsellers/data.py -> repo),
# so the default path works whether you're in a notebook, a test, or a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = _REPO_ROOT / "data" / "bestsellers.csv"

# Raw column name -> tidy snake_case name used everywhere in the project.
COLUMN_RENAMES = {
    "Name": "title",
    "Author": "author",
    "User Rating": "user_rating",
    "Reviews": "reviews",
    "Price": "price",
    "Year": "year",
    "Genre": "genre",
}

TEXT_COLUMNS = ["title", "author", "genre"]


def load_raw(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the raw bestsellers CSV exactly as it ships from Kaggle."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return an analysis-ready copy of the raw bestsellers data.

    Steps
    -----
    1. Rename columns to tidy ``snake_case``.
    2. Strip stray whitespace from the text fields.
    3. Drop duplicate ``(title, year)`` rows. These are the same listing
       captured twice in one year (identical rating and review count, only the
       price differs), so we keep the first occurrence to make each book-year
       unique.

    Note: 12 books carry a price of £0. These are kept on purpose — free
    Kindle promotions are a genuine part of the bestseller landscape — and are
    examined explicitly in the price analysis rather than silently dropped.
    """
    df = df.rename(columns=COLUMN_RENAMES).copy()

    for col in TEXT_COLUMNS:
        df[col] = df[col].str.strip()

    df = df.drop_duplicates(subset=["title", "year"], keep="first")

    return df.reset_index(drop=True)


def load_clean_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and return the cleaned, analysis-ready DataFrame."""
    return clean(load_raw(path))
