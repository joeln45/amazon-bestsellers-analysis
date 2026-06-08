"""Styled plotting helpers for the bestsellers analysis.

Two conventions keep the charts consistent:

1. set_theme() is called once at the top of the notebook, so every chart shares
   the same palette, fonts and grid.
2. Each helper takes ax=None and returns the Axes it drew on. Passing an Axes
   lets the notebook build multi-panel figures; returning one lets a test check
   the result without a window opening.

The helpers get their numbers from bestsellers.analysis, so the maths lives in
one place.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes

from . import analysis

# One palette for the whole project. "deep" on a whitegrid background keeps the
# focus on the data.
PALETTE = "deep"
THEME_STYLE = "whitegrid"

# Accent colours pulled from the palette so they always match it.
ACCENT = sns.color_palette(PALETTE)[0]     # muted blue
HIGHLIGHT = sns.color_palette(PALETTE)[1]  # orange, for the highlighted group


def set_theme() -> None:
    """Apply the project-wide seaborn theme. Call once before plotting."""
    sns.set_theme(style=THEME_STYLE, palette=PALETTE, context="notebook")


def _get_ax(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    """Return ax if given, otherwise make a fresh figure and Axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def scatter_rating_vs_reviews(df, ax: Axes | None = None) -> Axes:
    """Star rating against review count: the opening hook.

    The flat cloud is the point. The Pearson r goes in the title so the visual
    and the number agree.
    """
    ax = _get_ax(ax, (7, 5))
    sns.scatterplot(
        data=df, x="user_rating", y="reviews", alpha=0.5, color=ACCENT, ax=ax
    )
    r = analysis.correlation(df, "user_rating", "reviews")
    ax.set_title(f"Rating barely predicts popularity (r = {r:.3f})")
    ax.set_xlabel("User rating (stars)")
    ax.set_ylabel("Reviews")
    return ax


def bar_repeat_vs_one_hit(df, metric: str = "mean_reviews", ax: Axes | None = None) -> Axes:
    """One chosen metric for one-hit titles vs repeat bestsellers.

    Defaults to mean_reviews, where repeat titles roughly double the one-hit
    figure while their rating stays level. Pass another column from
    analysis.repeat_vs_one_hit to compare a different metric.
    """
    summary = analysis.repeat_vs_one_hit(df)
    ax = _get_ax(ax, (6, 5))
    ax.bar(summary.index, summary[metric], color=[ACCENT, HIGHLIGHT])
    ax.set_title(f"{metric.replace('_', ' ').title()}: one-hit vs repeat")
    ax.set_xlabel("")
    ax.set_ylabel(metric.replace("_", " ").title())
    return ax


def bar_staying_power(df, ax: Axes | None = None) -> Axes:
    """Number of titles that chart for 1 year, 2 years, up to all 10."""
    summary = analysis.staying_power_summary(df)
    ax = _get_ax(ax, (8, 5))
    ax.bar(summary.index, summary["titles"], color=ACCENT)
    ax.set_title("Staying power: most titles chart only once")
    ax.set_xlabel("Years charted")
    ax.set_ylabel("Number of titles")
    ax.set_xticks(summary.index)
    return ax


def barh_top_authors(df, by: str = "appearances", n: int = 10, ax: Axes | None = None) -> Axes:
    """Horizontal bar chart of the top n authors by reach.

    by="appearances" rewards endurance, by="titles" rewards volume. Horizontal
    bars keep the long author names readable.
    """
    column = {"appearances": "author_appearances", "titles": "author_titles"}[by]
    authors = analysis.top_authors(df, by=by, n=n).sort_values(column)
    ax = _get_ax(ax, (8, 6))
    ax.barh(authors.index, authors[column], color=ACCENT)
    label = "chart appearances" if by == "appearances" else "distinct titles"
    ax.set_title(f"Top {n} authors by {label}")
    ax.set_xlabel(label.title())
    ax.set_ylabel("")
    return ax


def heatmap_correlations(df, ax: Axes | None = None) -> Axes:
    """Correlation heatmap across rating, reviews and price.

    Shows that every off-diagonal value sits near zero.
    """
    corr = analysis.correlation_matrix(df)
    ax = _get_ax(ax, (5, 4))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax
    )
    ax.set_title("Numeric features barely correlate")
    return ax
