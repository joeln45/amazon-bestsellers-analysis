"""Exploratory analysis helpers for the bestsellers dataset.

Each function takes a DataFrame and returns numbers or a small DataFrame.
There is no plotting or printing here, which keeps the functions easy to test
and lets the notebook stay readable: one question per cell. The matching plot
helpers live in bestsellers.viz.

Functions that read engineered columns (times_charted, is_repeat_bestseller,
author_*) say so in their docstring; run engineer_features first for those.
"""
from __future__ import annotations

import pandas as pd

# The numeric columns we correlate. year is numeric too but behaves like a
# label, so it is left out of the correlations.
NUMERIC_COLUMNS = ["user_rating", "reviews", "price"]


def dataset_overview(df: pd.DataFrame) -> pd.Series:
    """Quick summary of the dataset's shape and coverage.

    Works on cleaned data, before any engineered columns exist.
    """
    return pd.Series(
        {
            "rows": len(df),
            "unique_titles": df["title"].nunique(),
            "unique_authors": df["author"].nunique(),
            "year_min": int(df["year"].min()),
            "year_max": int(df["year"].max()),
            "free_books": int((df["price"] == 0).sum()),
        }
    )


def correlation(df: pd.DataFrame, x: str, y: str, method: str = "pearson") -> float:
    """Correlation between two numeric columns.

    correlation(df, "user_rating", "reviews") is the key one: it lands near
    zero, which is the hook for the whole analysis.
    """
    return float(df[x].corr(df[y], method=method))


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Correlation matrix across the numeric columns, for a heatmap."""
    return df[NUMERIC_COLUMNS].corr(method=method)


def repeat_vs_one_hit(df: pd.DataFrame) -> pd.DataFrame:
    """Compare one-hit titles against repeat bestsellers.

    Needs the is_repeat_bestseller column. Returns one row per group with the
    title count, total chart entries, and mean rating / reviews / price.

    Stats are computed over chart entries (rows), so a title that charts three
    years counts three times. That is the "how the chart actually looked" view
    rather than a per-book average.
    """
    grouped = df.groupby("is_repeat_bestseller").agg(
        titles=("title", "nunique"),
        chart_entries=("title", "size"),
        mean_rating=("user_rating", "mean"),
        mean_reviews=("reviews", "mean"),
        mean_price=("price", "mean"),
    )
    # Nicer labels than False / True.
    grouped.index = grouped.index.map({False: "One-hit", True: "Repeat"})
    grouped.index.name = "group"
    return grouped


def staying_power_summary(df: pd.DataFrame) -> pd.DataFrame:
    """How many years titles stay on the chart.

    Needs the times_charted column. Returns a table indexed by times_charted
    (1, 2, 3, ...) with the number of distinct titles at each level and its
    share of all titles.
    """
    per_title = df.drop_duplicates("title")
    counts = per_title["times_charted"].value_counts().sort_index()
    out = counts.rename("titles").to_frame()
    out["share"] = out["titles"] / out["titles"].sum()
    out.index.name = "times_charted"
    return out


def top_authors(df: pd.DataFrame, by: str = "appearances", n: int = 10) -> pd.DataFrame:
    """Rank authors by reach.

    Needs the author_titles / author_appearances columns. by="appearances"
    ranks on total chart entries (rewards endurance); by="titles" ranks on
    distinct titles (rewards volume).
    """
    column = {"appearances": "author_appearances", "titles": "author_titles"}[by]
    per_author = (
        df.drop_duplicates("author")
        .set_index("author")[["author_titles", "author_appearances"]]
        .sort_values(column, ascending=False)
    )
    return per_author.head(n)
