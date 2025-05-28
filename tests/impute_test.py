import textwrap
from io import StringIO

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_frame_not_equal

from csv_helper import impute


@pytest.fixture
def df_inp() -> pl.DataFrame:
    data = """\
    id,numerator,denominator,imp_num,imp_denom
    A,10,15,false,false
    A,<=5,<=5,true,true
    A,12,23,false,false
    B,<=5,<=5,true,true
    A,22,24,false,false
    B,<=5,13,true,false
    B,<=5,<=5,true,true
    A,10,15,false,false
    C,<=5,<=5,false,true
    C,<=5,<=5,true,true
    A,<=5,<=5,true,true
    A,22,15,false,false
    B,<=5,13,true,false
    A,<=5,<=5,false,true
    C,100,128,false,false
    C,<=5,<=5,true,true
    D,<=5,<=5,true,true
    A,22,23,false,false
    B,<=5,18,true,false
    H,8,17,false,false
    A,10,16,false,false
    A,<=5,<=5,true,true
    H,<=5,<=5,true,true
    A,22,88,false,false
    B,<=5,23,true,false
    C,<=5,<=5,true,true
    A,<=5,<=5,false,true
    C,100,1300,false,false
    C,<=5,<=5,true,true
    D,<=5,<=5,true,true
    """
    df = pl.read_csv(
        StringIO(textwrap.dedent(data)),
        schema={
            "id": pl.String,
            "numerator": pl.String,
            "denominator": pl.String,
            "imp_num": pl.Boolean,
            "imp_denom": pl.Boolean,
        },
    )

    return df


@pytest.fixture
def lf_inp() -> pl.LazyFrame:
    data = """\
    id,numerator,denominator,imp_num,imp_denom
    A,10,15,false,false
    A,<=5,<=5,true,true
    A,12,23,false,false
    B,<=5,<=5,true,true
    A,22,24,false,false
    B,<=5,13,true,false
    B,<=5,<=5,true,true
    A,10,15,false,false
    C,<=5,<=5,false,true
    C,<=5,<=5,true,true
    A,<=5,<=5,true,true
    A,22,15,false,false
    B,<=5,13,true,false
    A,<=5,<=5,false,true
    C,100,128,false,false
    C,<=5,<=5,true,true
    D,<=5,<=5,true,true
    A,22,23,false,false
    B,<=5,18,true,false
    H,8,17,false,false
    A,10,16,false,false
    A,<=5,<=5,true,true
    H,<=5,<=5,true,true
    A,22,88,false,false
    B,<=5,23,true,false
    C,<=5,<=5,true,true
    A,<=5,<=5,false,true
    C,100,1300,false,false
    C,<=5,<=5,true,true
    D,<=5,<=5,true,true
    """
    lf = pl.scan_csv(
        StringIO(textwrap.dedent(data)),
        schema={
            "id": pl.String,
            "numerator": pl.String,
            "denominator": pl.String,
            "imp_num": pl.Boolean,
            "imp_denom": pl.Boolean,
        },
    )

    return lf


def test_impute_columns_single(df_inp: pl.DataFrame) -> None:
    df_out = df_inp.pipe(impute.columns, ["numerator"], "<=5", (1, 5))

    assert df_inp.shape == df_out.shape
    assert (
        df_out.select((pl.col("numerator").cast(pl.String) == "<=5").any()).item()
        is False
    )
    assert (
        df_out.filter(pl.col("imp_num")).select((pl.col("numerator") <= 5).all()).item()
        is True
    )


def test_impute_columns_single_lazy(lf_inp: pl.LazyFrame) -> None:
    lf_out = lf_inp.pipe(impute.columns, ["numerator"], "<=5", (1, 5))

    assert lf_inp.collect().shape == lf_out.collect().shape
    assert (
        lf_out.select((pl.col("numerator").cast(pl.String) == "<=5").any())
        .collect()
        .item()
        is False
    )
    assert (
        lf_out.filter(pl.col("imp_num"))
        .select((pl.col("numerator") <= 5).all())
        .collect()
        .item()
        is True
    )


def test_impute_columns_multi(df_inp: pl.DataFrame) -> None:
    df_out = df_inp.pipe(impute.columns, ["numerator", "denominator"], "<=5", (1, 5))

    assert df_inp.shape == df_out.shape
    assert (
        df_out.select((pl.col("numerator").cast(pl.String) == "<=5").any()).item()
        is False
    )
    assert (
        df_out.select((pl.col("denominator").cast(pl.String) == "<=5").any()).item()
        is False
    )

    assert (
        df_out.filter(pl.col("imp_num")).select((pl.col("numerator") <= 5).all()).item()
        is True
    )
    assert (
        df_out.filter(pl.col("imp_denom"))
        .select((pl.col("denominator") <= 5).all())
        .item()
        is True
    )


def test_impute_columns_multi_lazy(lf_inp: pl.LazyFrame) -> None:
    lf_out = lf_inp.pipe(impute.columns, ["numerator", "denominator"], "<=5", (1, 5))

    assert lf_inp.collect().shape == lf_out.collect().shape
    assert (
        lf_out.select((pl.col("numerator").cast(pl.String) == "<=5").any())
        .collect()
        .item()
        is False
    )
    assert (
        lf_out.select((pl.col("denominator").cast(pl.String) == "<=5").any())
        .collect()
        .item()
        is False
    )

    assert (
        lf_out.filter(pl.col("imp_num"))
        .select((pl.col("numerator") <= 5).all())
        .collect()
        .item()
        is True
    )
    assert (
        lf_out.filter(pl.col("imp_denom"))
        .select((pl.col("denominator") <= 5).all())
        .collect()
        .item()
        is True
    )


def test_impute_columns_seed(df_inp: pl.DataFrame) -> None:
    df_1 = df_inp.pipe(impute.columns, ["numerator"], "<=5", (1, 5))
    df_2 = df_inp.pipe(impute.columns, ["numerator"], "<=5", (1, 5))

    assert df_1.shape == df_2.shape
    assert_frame_not_equal(df_1, df_2)

    df_1 = df_inp.pipe(impute.columns, ["numerator"], "<=5", (1, 5), seed=18)
    df_2 = df_inp.pipe(impute.columns, ["numerator"], "<=5", (1, 5), seed=18)

    assert df_1.shape == df_2.shape
    assert_frame_equal(df_1, df_2)


def test_impute_columns_seed_lazy(lf_inp: pl.LazyFrame) -> None:
    lf_1 = lf_inp.pipe(impute.columns, ["numerator"], "<=5", (1, 5))
    lf_2 = lf_inp.pipe(impute.columns, ["numerator"], "<=5", (1, 5))

    assert lf_1.collect().shape == lf_2.collect().shape
    assert_frame_not_equal(lf_1, lf_2)

    lf_1 = lf_inp.pipe(impute.columns, ["numerator"], "<=5", (1, 5), seed=18)
    lf_2 = lf_inp.pipe(impute.columns, ["numerator"], "<=5", (1, 5), seed=18)

    assert lf_1.collect().shape == lf_2.collect().shape
    assert_frame_equal(lf_1, lf_2)


def test_impute_pair(df_inp: pl.DataFrame) -> None:
    df_out = df_inp.pipe(impute.column_pair, "numerator", "denominator", "<=5", (1, 5))

    assert df_inp.shape == df_out.shape
    assert (
        df_out.select((pl.col("numerator").cast(pl.String) == "<=5").any()).item()
        is False
        and df_out.select((pl.col("denominator").cast(pl.String) == "<=5").any()).item()
        is False
    )
    assert (
        df_out.filter(pl.col("imp_num")).select((pl.col("numerator") <= 5).all()).item()
        is True
    )
    assert (
        df_out.filter(pl.col("imp_denom"))
        .select((pl.col("denominator") <= 5).all())
        .item()
        is True
    )

    assert (
        df_out.filter(pl.col("imp_denom"))
        .select((pl.col("numerator") <= pl.col("denominator")).all())
        .item()
        is True
    )


def test_impute_pair_lazy(lf_inp: pl.LazyFrame) -> None:
    lf_out = lf_inp.pipe(impute.column_pair, "numerator", "denominator", "<=5", (1, 5))

    assert lf_inp.collect().shape == lf_out.collect().shape
    assert (
        lf_out.select((pl.col("numerator").cast(pl.String) == "<=5").any())
        .collect()
        .item()
        is False
        and lf_out.select((pl.col("denominator").cast(pl.String) == "<=5").any())
        .collect()
        .item()
        is False
    )
    assert (
        lf_out.filter(pl.col("imp_num"))
        .select((pl.col("numerator") <= 5).all())
        .collect()
        .item()
        is True
    )
    assert (
        lf_out.filter(pl.col("imp_denom"))
        .select((pl.col("denominator") <= 5).all())
        .collect()
        .item()
        is True
    )

    assert (
        lf_out.filter(pl.col("imp_denom"))
        .select((pl.col("numerator") <= pl.col("denominator")).all())
        .collect()
        .item()
        is True
    )
