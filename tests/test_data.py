"""Unit tests for the bestsellers data layer.

The logic tests use tiny, hand-built DataFrames so the expected results are
obvious at a glance. Two integration tests then confirm the real shipped CSV
still flows through the pipeline as expected.
"""
import pandas as pd
import pytest

from bestsellers.data import (
    PRICE_LABELS,
    RATING_LABELS,
    clean,
    engineer_features,
    load_clean_data,
    load_raw,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def raw_sample():
    """Raw-shaped frame with a duplicate book-year and untrimmed whitespace."""
    return pd.DataFrame(
        {
            "Name": ["Book A", "Book A", " Book B "],
            "Author": ["Ann", "Ann", " Bob "],
            "User Rating": [4.6, 4.6, 4.2],
            "Reviews": [100, 100, 50],
            "Price": [10, 12, 0],  # rows 0 & 1: same title/year, differ on price
            "Year": [2010, 2010, 2011],
            "Genre": ["Fiction", "Fiction", " Non Fiction "],
        }
    )


@pytest.fixture
def clean_sample():
    """Cleaned-shape frame chosen so every engineered feature is checkable."""
    return pd.DataFrame(
        {
            "title": ["A", "A", "B", "C"],
            "author": ["Ann", "Ann", "Ann", "Bob"],
            "user_rating": [4.4, 4.5, 4.8, 4.9],  # Good / Great / Elite / Elite
            "reviews": [10, 20, 30, 40],
            "price": [0, 8, 15, 25],  # Free / Budget / Mid / Premium
            "year": [2010, 2011, 2010, 2010],
            "genre": ["Fiction", "Fiction", "Non Fiction", "Fiction"],
        }
    )


# --------------------------------------------------------------------------
# clean()
# --------------------------------------------------------------------------
def test_clean_renames_to_snake_case(raw_sample):
    out = clean(raw_sample)
    assert list(out.columns) == [
        "title", "author", "user_rating", "reviews", "price", "year", "genre",
    ]


def test_clean_drops_duplicate_book_years(raw_sample):
    out = clean(raw_sample)
    # The two "Book A" / 2010 rows collapse to one; "Book B" survives.
    assert len(out) == 2
    assert out.duplicated(subset=["title", "year"]).sum() == 0


def test_clean_keeps_first_of_a_duplicate(raw_sample):
    out = clean(raw_sample)
    book_a = out[out["title"] == "Book A"].iloc[0]
    assert book_a["price"] == 10  # the first occurrence, not the £12 one


def test_clean_strips_whitespace(raw_sample):
    out = clean(raw_sample)
    assert "Book B" in out["title"].values
    assert "Bob" in out["author"].values
    assert "Non Fiction" in out["genre"].values


# --------------------------------------------------------------------------
# engineer_features()
# --------------------------------------------------------------------------
def test_times_charted_counts_distinct_years(clean_sample):
    out = engineer_features(clean_sample)
    times = dict(zip(out["title"], out["times_charted"]))
    assert times["A"] == 2  # charts in 2010 and 2011
    assert times["B"] == 1
    assert times["C"] == 1


def test_is_repeat_bestseller_threshold(clean_sample):
    out = engineer_features(clean_sample)
    repeats = dict(zip(out["title"], out["is_repeat_bestseller"]))
    assert repeats["A"]       # charts twice -> repeat
    assert not repeats["B"]   # charts once -> not a repeat


def test_is_free_flag(clean_sample):
    out = engineer_features(clean_sample)
    assert out.loc[out["price"] == 0, "is_free"].all()
    assert not out.loc[out["price"] > 0, "is_free"].any()


def test_price_bands_cover_all_brackets(clean_sample):
    out = engineer_features(clean_sample)
    bands = list(out["price_band"].astype(str))
    assert bands == PRICE_LABELS  # [Free, Budget, Mid, Premium] in row order


def test_rating_tier_boundaries(clean_sample):
    out = engineer_features(clean_sample)
    tiers = list(out["rating_tier"].astype(str))
    # 4.4 -> Good, 4.5 -> Great (right=False), 4.8 -> Elite, 4.9 -> Elite
    assert tiers == [RATING_LABELS[0], RATING_LABELS[1], RATING_LABELS[2], RATING_LABELS[2]]


def test_author_stats(clean_sample):
    out = engineer_features(clean_sample)
    ann = out[out["author"] == "Ann"].iloc[0]
    bob = out[out["author"] == "Bob"].iloc[0]
    assert ann["author_titles"] == 2       # distinct titles A and B
    assert ann["author_appearances"] == 3  # three chart rows
    assert bob["author_titles"] == 1
    assert bob["author_appearances"] == 1


# --------------------------------------------------------------------------
# Integration against the real dataset
# --------------------------------------------------------------------------
def test_real_dataset_cleans_to_unique_book_years():
    df = load_clean_data()
    assert df.shape == (547, 7)
    assert df.duplicated(subset=["title", "year"]).sum() == 0


def test_real_features_have_no_missing_categories():
    df = engineer_features(load_clean_data())
    assert df.shape == (547, 14)
    assert df[["price_band", "rating_tier"]].isna().sum().sum() == 0
