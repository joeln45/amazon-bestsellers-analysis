"""Regenerate the figures embedded in the README.

Run from the repo root::

    python scripts/export_figures.py

Every image is built from the same :mod:`bestsellers.viz` helpers the notebook
uses, so the README, the notebook and the code can never drift apart. Re-run
this whenever a chart changes and commit the refreshed PNGs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to file, no window
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bestsellers import viz  # noqa: E402
from bestsellers.data import engineer_features, load_clean_data  # noqa: E402

IMAGES_DIR = REPO_ROOT / "images"
DPI = 120

# name -> function that draws the chart and returns its Axes. The three charts
# that carry the story arc: the hook, the reveal, and the staying-power tail.
FIGURES = {
    "hook_rating_vs_reviews": lambda df: viz.scatter_rating_vs_reviews(df),
    "staying_power": lambda df: viz.bar_staying_power(df),
    "repeat_vs_one_hit": lambda df: viz.bar_repeat_vs_one_hit(df, metric="mean_reviews"),
}


def main() -> None:
    IMAGES_DIR.mkdir(exist_ok=True)
    df = engineer_features(load_clean_data())
    viz.set_theme()

    for name, draw in FIGURES.items():
        ax = draw(df)
        out_path = IMAGES_DIR / f"{name}.png"
        ax.figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
