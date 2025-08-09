import textwrap
from io import StringIO

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from csv_helper import complete


@pytest.fixture
def df_inp() -> pl.DataFrame:
    data = """\
    country,year,value
    France,2020,1
    France,2021,2
    UK,2019,3
    UK,2020,4
    Spain,2022,5
    """
    df = pl.read_csv(
        StringIO(textwrap.dedent(data)),
        schema={
            "country": pl.String,
            "year": pl.Int64,
            "value": pl.Int64,
        },
    ).sort("country", "year")

    return df


@pytest.fixture
def lf_inp(df_inp: pl.DataFrame) -> pl.LazyFrame:
    return df_inp.lazy()


@pytest.fixture
def df_out() -> pl.DataFrame:
    data = """\
    country,year,value
    France,2019,
    France,2020,1
    France,2021,2
    France,2022,
    UK,2019,3
    UK,2020,4
    UK,2021,
    UK,2022,
    Spain,2019,
    Spain,2020,
    Spain,2021,
    Spain,2022,5
    """
    df = pl.read_csv(
        StringIO(textwrap.dedent(data)),
        schema={
            "country": pl.String,
            "year": pl.Int64,
            "value": pl.Int64,
        },
    ).sort("country", "year")

    return df


def test_complete_existing(df_inp: pl.DataFrame, df_out: pl.DataFrame) -> None:
    df = df_inp.pipe(complete.complete, "country", "year").sort("country", "year")

    assert_frame_equal(df, df_out)


def test_complete_existing_series(df_inp: pl.DataFrame, df_out: pl.DataFrame) -> None:
    country = pl.Series("country", ["France", "UK", "Spain"])
    year = pl.Series("year", [year for year in range(2019, 2023)])
    df = df_inp.pipe(complete.complete, country, year).sort("country", "year")

    assert_frame_equal(df, df_out)


def test_complete_exception(df_inp: pl.DataFrame) -> None:
    with pytest.raises(TypeError):
        df_inp.pipe(complete.complete, 0, 1).sort("country", "year")  # pyright: ignore[reportArgumentType, reportUnusedCallResult]


def test_complete_existing_lazy(lf_inp: pl.LazyFrame, df_out: pl.DataFrame) -> None:
    lf = lf_inp.pipe(complete.complete, "country", "year").sort("country", "year")

    assert_frame_equal(lf.collect(), df_out)


def test_complete_existing_lazy_series(
    lf_inp: pl.LazyFrame, df_out: pl.DataFrame
) -> None:
    country = pl.Series("country", ["France", "UK", "Spain"])
    year = pl.Series("year", [year for year in range(2019, 2023)])
    lf = lf_inp.pipe(complete.complete, country, year).sort("country", "year")

    assert_frame_equal(lf.collect(), df_out)


@pytest.fixture
def df_out_non_exist() -> pl.DataFrame:
    data = """\
    country,year,value
    China,2019,
    China,2020,
    China,2021,
    China,2022,
    France,2019,
    France,2020,1
    France,2021,2
    France,2022,
    UK,2019,3
    UK,2020,4
    UK,2021,
    UK,2022,
    Spain,2019,
    Spain,2020,
    Spain,2021,
    Spain,2022,5
    """
    df = pl.read_csv(
        StringIO(textwrap.dedent(data)),
        schema={
            "country": pl.String,
            "year": pl.Int64,
            "value": pl.Int64,
        },
    ).sort("country", "year")

    return df


def test_complete_non_existing(
    df_inp: pl.DataFrame, df_out_non_exist: pl.DataFrame
) -> None:
    df = df_inp.pipe(
        complete.complete,
        pl.Series("country", ["France", "UK", "Spain", "China"]),
        "year",
    ).sort("country", "year")

    assert_frame_equal(df, df_out_non_exist)


def test_complete_non_existing_lazy(
    lf_inp: pl.LazyFrame, df_out_non_exist: pl.DataFrame
) -> None:
    lf = lf_inp.pipe(
        complete.complete,
        pl.Series("country", ["France", "UK", "Spain", "China"]),
        "year",
    ).sort("country", "year")

    assert_frame_equal(lf.collect(), df_out_non_exist)
