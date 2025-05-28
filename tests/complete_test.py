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
    )

    return df


@pytest.fixture
def lf_inp() -> pl.LazyFrame:
    data = """\
    country,year,value
    France,2020,1
    France,2021,2
    UK,2019,3
    UK,2020,4
    Spain,2022,5
    """
    lf = pl.scan_csv(
        StringIO(textwrap.dedent(data)),
        schema={
            "country": pl.String,
            "year": pl.Int64,
            "value": pl.Int64,
        },
    )

    return lf


def test_complete_existing(df_inp: pl.DataFrame) -> None:
    df_out = df_inp.pipe(complete.complete, "country", "year").sort("country", "year")

    data_res = """\
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
    result = pl.read_csv(
        StringIO(textwrap.dedent(data_res)),
        schema={
            "country": pl.String,
            "year": pl.Int64,
            "value": pl.Int64,
        },
    ).sort("country", "year")

    assert_frame_equal(df_out, result)


def test_complete_existing_lazy(lf_inp: pl.LazyFrame) -> None:
    lf_out = lf_inp.pipe(complete.complete, "country", "year").sort("country", "year")

    data_res = """\
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
    result = pl.read_csv(
        StringIO(textwrap.dedent(data_res)),
        schema={
            "country": pl.String,
            "year": pl.Int64,
            "value": pl.Int64,
        },
    ).sort("country", "year")

    assert_frame_equal(lf_out.collect(), result)


def test_complete_non_existing(df_inp: pl.DataFrame) -> None:
    df_out = df_inp.pipe(
        complete.complete,
        pl.Series("country", ["France", "UK", "Spain", "China"]),
        "year",
    ).sort("country", "year")

    data_res = """\
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
    result = pl.read_csv(
        StringIO(textwrap.dedent(data_res)),
        schema={
            "country": pl.String,
            "year": pl.Int64,
            "value": pl.Int64,
        },
    ).sort("country", "year")

    assert_frame_equal(df_out, result)


def test_complete_non_existing_lazy(lf_inp: pl.LazyFrame) -> None:
    lf_out = lf_inp.pipe(
        complete.complete,
        pl.Series("country", ["France", "UK", "Spain", "China"]),
        "year",
    ).sort("country", "year")

    data_res = """\
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
    result = pl.read_csv(
        StringIO(textwrap.dedent(data_res)),
        schema={
            "country": pl.String,
            "year": pl.Int64,
            "value": pl.Int64,
        },
    ).sort("country", "year")

    assert_frame_equal(lf_out.collect(), result)
