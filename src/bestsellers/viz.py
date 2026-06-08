"""Styled plotting helpers for the bestsellers data story.

Two conventions hold the visuals together:

1. :func:`set_theme` is called once at the top of the notebook. Every chart
   then inherits the same palette, fonts and grid, so the whole story reads as
   one designed piece rather than a pile of mismatched plots.
2. Each plot helper accepts ``ax=None`` and *returns the Axes it drew on*.
   Passing an Axes lets the notebook compose multi-panel figures; returning one
   lets a test assert on the result (bar count, title) without a window ever
   opening.

The helpers lean on the analysis functions in :mod:`bestsellers.analysis` for
their numbers, so the maths lives in exactly one place.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes

from . import analysis

# One palette for the whole project. "deep" is seaborn's calm, professional
# default; the whitegrid backdrop keeps focus on the data, not the chrome.
PALETTE = "deep"
THEME_STYLE = "whitegrid"

# A single accent colour for the "headline" charts, pulled from the palette so
# it always matches. Index 0 of "deep" is the muted blue.
ACCENT = sns.color_palette(PALETTE)[0]
# A warm contrast colour (index 1, the orange) for highlighting one group.
HIGHLIGHT = sns.color_palette(PALETTE)[1]


def set_theme() -> None:
    """Apply the project-wide seaborn theme. Call once before plotting."""
    sns.set_theme(style=THEME_STYLE, palette=PALETTE, context="notebook")


def _get_ax(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    """Return ``ax`` if given, otherwise make a fresh figure and Axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def scatter_rating_vs_reviews(df, ax: Axes | None = None) -> Axes:
    """The opening hook: star rating against review count.

    The near-flat cloud is the point -- knowing a book's rating tells you almost
    nothing about how many reviews (i.e. how much popularity) it gathered. The
    Pearson r is printed in the title so the visual and the number agree.
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
    """The reveal: one chosen metric for one-hit titles vs repeat bestsellers.

    Defaults to ``mean_reviews`` -- repeat titles roughly double it while their
    ``mean_rating`` stays level, which is the staying-power argument in one bar
    pair. Pass another column from :func:`analysis.repeat_vs_one_hit` to compare
    a different metric.
    """
    summary = analysis.repeat_vs_one_hit(df)
    ax = _get_ax(ax, (6, 5))
    colors = [ACCENT, HIGHLIGHT]
    ax.bar(summary.index, summary[metric], color=colors)
    ax.set_title(f"{metric.replace('_', ' ').title()}: one-hit vs repeat")
    ax.set_xlabel("")
    ax.set_ylabel(metric.replace("_", " ").title())
    return ax


def bar_staying_power(df, ax: Axes | None = None) -> Axes:
    """How many titles chart for 1 year, 2 years, ... up to all 10.

    The steep drop-off after a single appearance is the long tail of one-hit
    wonders; the thin bars on the right are the enduring core.
    """
    summary = analysis.staying_power_summary(df)
    ax = _get_ax(ax, (8, 5))
    ax.bar(summary.index, summary["titles"], color=ACCENT)
    ax.set_title("Staying power: most titles chart only once")
    ax.set_xlabel("Years charted")
    ax.set_ylabel("Number of titles")
    ax.set_xticks(summary.index)
    return ax


def barh_top_authors(df, by: str = "appearances", n: int = 10, ax: Axes | None = None) -> Axes:
    """Horizontal bar chart of the top ``n`` authors by reach.

    ``by="appearances"`` rewards endurance, ``by="titles"`` rewards volume --
    the two ways an author dominates the chart. Horizontal bars keep the (often
    long) author names readable.
    """
    column = {"appearances": "author_appearances", "titles": "author_titles"}[by]
    authors = analysis.top_authors(df, by=by, n=n)
    # Plot smallest-at-bottom so the leader sits at the top of the chart.
    authors = authors.sort_values(column)
    ax = _get_ax(ax, (8, 6))
    ax.barh(authors.index, authors[column], color=ACCENT)
    label = "chart appearances" if by == "appearances" else "distinct titles"
    ax.set_title(f"Top {n} authors by {label}")
    ax.set_xlabel(label.title())
    ax.set_ylabel("")
    return ax


def heatmap_correlations(df, ax: Axes | None = None) -> Axes:
    """Correlation heatmap across rating, reviews and price.

    A compact way to show that every off-diagonal value sits near zero: none of
    the three numeric features move together with any strength.
    """
    corr = analysis.correlation_matrix(df)
    ax = _get_ax(ax, (5, 4))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax
    )
    ax.set_title("Numeric features barely correlate")
    return ax
