import polars as pl

from csv_helper import impute

df_inp = pl.DataFrame(
    {
        "id": ["A", "A", "A", "B", "B", "C", "C", "A", "A", "A", "D", "D"],
        "count": [
            "10",
            "15",
            "<=5",
            "<=5",
            "12",
            "50",
            "<=5",
            "10",
            "15",
            "<=5",
            "<=5",
            "<=5",
        ],
        "count_2": [
            "15",
            "10",
            "<=5",
            "12",
            "<=5",
            "<=5",
            "10",
            "50",
            "<=5",
            "<=5",
            "15",
            "<=5",
        ],
    }
)

# TODO: test values are <= 5
# TODO: test with seed?


def test_impute_columns_single() -> None:
    df = df_inp.pipe(impute.columns, ["count"], "<=5", (1, 5))

    assert df.select((pl.col("count").cast(pl.String) == "<=5").any()).item() is False


def test_impute_columns_multi() -> None:
    df = df_inp.pipe(impute.columns, ["count", "count_2"], "<=5", (1, 5))

    assert (
        df.select((pl.col("count").cast(pl.String) == "<=5").any()).item() is False
        and df.select((pl.col("count_2").cast(pl.String) == "<=5").any()).item()
        is False
    )
