"""Exploratory analysis helpers for the bestsellers dataset.

Every function here takes a DataFrame and returns plain numbers or small
DataFrames -- no plotting, no printing. That keeps them unit-testable and lets
the notebook read like a narrative: each cell asks one question and gets one
answer. The matching plot helpers live next door in :mod:`bestsellers.viz`.

Functions that read engineered columns (``times_charted``,
``is_repeat_bestseller``, ``author_*``) say so in their docstrings; call
:func:`bestsellers.data.engineer_features` first for those.
"""
from __future__ import annotations

import pandas as pd

# The four genuinely numeric columns. ``year`` is numeric but behaves like a
# label here, so correlations against it are rarely meaningful -- kept in the
# list only so the overview can report the span.
NUMERIC_COLUMNS = ["user_rating", "reviews", "price"]


def dataset_overview(df: pd.DataFrame) -> pd.Series:
    """One-glance summary of the dataset's shape and coverage.

    Works on cleaned data alone (no engineered columns needed), so it can open
    the notebook before any features exist.
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
    """Correlation between two numeric columns -- the project's central tool.

    The opening hook leans on ``correlation(df, "user_rating", "reviews")``
    landing near zero: a book's star rating barely moves its review count, so
    quality and popularity are all but unrelated. The follow-up,
    ``correlation(df, "price", "reviews")``, is mildly negative -- cheaper books
    pull slightly more reviews.
    """
    return float(df[x].corr(df[y], method=method))


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Correlation matrix across the numeric columns, for a heatmap overview."""
    return df[NUMERIC_COLUMNS].corr(method=method)


def repeat_vs_one_hit(df: pd.DataFrame) -> pd.DataFrame:
    """Contrast one-hit titles against repeat bestsellers.

    Needs the engineered ``is_repeat_bestseller`` column. Returns one row per
    group with the title count, total chart entries, and mean rating / reviews
    / price, so the staying-power reveal is a single readable table.

    Stats are computed over chart entries (rows), so a title that charts three
    years contributes three rows -- repeat titles are therefore weighted by how
    often they appear, which is the intended "how the chart actually looked"
    view rather than a per-book average.
    """
    grouped = df.groupby("is_repeat_bestseller").agg(
        titles=("title", "nunique"),
        chart_entries=("title", "size"),
        mean_rating=("user_rating", "mean"),
        mean_reviews=("reviews", "mean"),
        mean_price=("price", "mean"),
    )
    # Friendlier labels than False / True for the notebook and README.
    grouped.index = grouped.index.map({False: "One-hit", True: "Repeat"})
    grouped.index.name = "group"
    return grouped


def staying_power_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of how many years titles stay on the chart.

    Needs the engineered ``times_charted`` column. Returns a table indexed by
    ``times_charted`` (1, 2, 3, ...) giving the number of distinct titles at
    each level and that level's share of all titles. The headline -- most
    titles chart once, a small core endures -- falls straight out of it.
    """
    per_title = df.drop_duplicates("title")
    counts = per_title["times_charted"].value_counts().sort_index()
    out = counts.rename("titles").to_frame()
    out["share"] = out["titles"] / out["titles"].sum()
    out.index.name = "times_charted"
    return out


def top_authors(df: pd.DataFrame, by: str = "appearances", n: int = 10) -> pd.DataFrame:
    """Rank authors by reach, two ways.

    Needs the engineered ``author_titles`` / ``author_appearances`` columns.

    ``by="appearances"`` ranks on total chart entries (book-years) and rewards
    *endurance*; ``by="titles"`` ranks on distinct titles and rewards *volume*.
    The contrast surfaces the two routes to chart dominance: Jeff Kinney's 12
    different titles (volume) versus Suzanne Collins charting 5 titles across
    11 years (endurance).
    """
    column = {"appearances": "author_appearances", "titles": "author_titles"}[by]
    per_author = (
        df.drop_duplicates("author")
        .set_index("author")[["author_titles", "author_appearances"]]
        .sort_values(column, ascending=False)
    )
    return per_author.head(n)
