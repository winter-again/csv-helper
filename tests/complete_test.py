import polars as pl
from polars.testing import assert_frame_equal

from csv_helper import impute


def test_complete_exists() -> None:
    df = pl.DataFrame(
        {
            "country": ["France", "France", "UK", "UK", "Spain"],
            "year": [2020, 2021, 2019, 2020, 2022],
            "value": [1, 2, 3, 4, 5],
        }
    )
    df = df.pipe(impute.complete, "country", "year").sort("country", "year")
    result = pl.DataFrame(
        {
            "country": [
                country for country in ["France", "UK", "Spain"] for _ in range(4)
            ],
            "year": [y for _ in range(3) for y in range(2019, 2023)],
            "value": [None, 1, 2, None, 3, 4, None, None, None, None, None, 5],
        }
    ).sort("country", "year")

    assert_frame_equal(df, result)

    lf = pl.LazyFrame(
        {
            "country": ["France", "France", "UK", "UK", "Spain"],
            "year": [2020, 2021, 2019, 2020, 2022],
            "value": [1, 2, 3, 4, 5],
        }
    )
    lf = lf.pipe(impute.complete, "country", "year").sort("country", "year")
    result = pl.LazyFrame(
        {
            "country": [
                country for country in ["France", "UK", "Spain"] for _ in range(4)
            ],
            "year": [y for _ in range(3) for y in range(2019, 2023)],
            "value": [None, 1, 2, None, 3, 4, None, None, None, None, None, 5],
        }
    ).sort("country", "year")

    assert_frame_equal(lf, result)


def test_complete_not_exists() -> None:
    # TODO: add lazy test
    df = pl.DataFrame(
        {
            "country": ["France", "France", "UK", "UK", "Spain"],
            "year": [2020, 2021, 2019, 2020, 2022],
            "value": [1, 2, 3, 4, 5],
        }
    )
    df = df.pipe(
        impute.complete,
        pl.Series("country", ["France", "UK", "Spain", "China"]),
        "year",
    ).sort("country", "year")
    result = pl.DataFrame(
        {
            "country": [
                country
                for country in ["China", "France", "UK", "Spain"]
                for _ in range(4)
            ],
            "year": [y for _ in range(4) for y in range(2019, 2023)],
            "value": [
                None,
                None,
                None,
                None,
                None,
                1,
                2,
                None,
                3,
                4,
                None,
                None,
                None,
                None,
                None,
                5,
            ],
        }
    ).sort("country", "year")

    assert_frame_equal(df, result)
