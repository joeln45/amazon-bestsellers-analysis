# Amazon Bestsellers: Does Quality Sell, or Does Staying Power Win?

A data-story exploring Amazon's Top 50 bestselling books each year from 2009–2019.

> **The question:** A book's star rating barely predicts how popular it becomes — so what *actually* makes a bestseller last? This project follows the data from a counter-intuitive hook (quality ≠ popularity) to the real driver of lasting success: **staying power across years.**

## Dataset

[Amazon Top 50 Bestselling Books 2009–2019](https://www.kaggle.com/datasets/sootersaalu/amazon-top-50-bestselling-books-2009-2019) (Kaggle) — 550 rows, 7 columns: title, author, user rating, review count, price, year, and genre.

## Project structure

```
amazon-bestsellers-analysis/
├── data/bestsellers.csv          # raw dataset
├── src/bestsellers/              # reusable analysis toolkit
│   ├── data.py                   # load, clean, feature engineering
│   ├── analysis.py               # EDA helpers
│   └── viz.py                    # styled plotting helpers
├── tests/                        # pytest unit tests for the data layer
└── notebooks/
    └── bestsellers_analysis.ipynb   # the data story
```

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter notebook notebooks/bestsellers_analysis.ipynb
```

Run the tests:

```bash
pytest
```

## Key findings

_Coming soon — populated once the analysis is complete._

## About

A portfolio reworking of a university data-science assignment (CSCU9M3, University of Stirling), rebuilt with a reusable code module, unit tests, and a tighter narrative.
