"""Load, clean and prepare the Amazon bestsellers dataset.

The raw Kaggle file ships with human-friendly column names ("User Rating"),
mixed naming conventions, and a few data-quality quirks (the same book listed
twice in one year at different prices). Everything needed to turn it into an
analysis-ready DataFrame lives here, behind a single :func:`load_clean_data`
call so notebooks stay focused on the story rather than the plumbing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
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

# Price brackets in GBP. The first bucket isolates the £0 promotional books so
# they never get lumped in with genuinely cheap paid titles.
PRICE_BINS = [-0.01, 0.0, 10.0, 20.0, np.inf]
PRICE_LABELS = ["Free", "Budget (≤£10)", "Mid (£10–20)", "Premium (>£20)"]

# Rating tiers. Ratings cluster tightly between 3.3 and 4.9, so the cut points
# are chosen to split that narrow band into meaningful quality groups.
RATING_BINS = [0.0, 4.5, 4.8, 5.01]
RATING_LABELS = ["Good (<4.5)", "Great (4.5–4.7)", "Elite (4.8+)"]


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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns that power the staying-power and pricing analyses.

    Adds
    ----
    times_charted        : how many years a title appears in the top 50.
    is_repeat_bestseller : True when a title charts in two or more years.
    is_free              : True for the £0 promotional titles.
    price_band           : Free / Budget / Mid / Premium (ordered category).
    rating_tier          : Good / Great / Elite quality group (ordered category).
    author_titles        : number of distinct titles an author has charted.
    author_appearances   : total chart entries (book-years) for an author.
    """
    df = df.copy()

    # Staying power: rows are unique per (title, year) after cleaning, so a
    # simple per-title count equals the number of distinct years charted.
    df["times_charted"] = df.groupby("title")["year"].transform("size")
    df["is_repeat_bestseller"] = df["times_charted"] >= 2

    # Pricing.
    df["is_free"] = df["price"] == 0
    df["price_band"] = pd.cut(
        df["price"], bins=PRICE_BINS, labels=PRICE_LABELS, ordered=True
    )

    # Quality grouping. right=False so an exact 4.5 lands in "Great", 4.8 in "Elite".
    df["rating_tier"] = pd.cut(
        df["user_rating"],
        bins=RATING_BINS,
        labels=RATING_LABELS,
        right=False,
        ordered=True,
    )

    # Author reach.
    df["author_titles"] = df.groupby("author")["title"].transform("nunique")
    df["author_appearances"] = df.groupby("author")["title"].transform("size")

    return df


def load_clean_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and return the cleaned, analysis-ready DataFrame.

    This is *cleaning only* — call :func:`engineer_features` on the result to
    add the derived analysis columns.
    """
    return clean(load_raw(path))
