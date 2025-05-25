from typing import NamedTuple

import numpy as np
import polars as pl
from polars.datatypes.classes import FloatType, IntegerType


def check[T: (pl.DataFrame, pl.LazyFrame)](
    df: T, columns: list[str], fill_flag: str
) -> T:
    """
    Summarize counts and proportion of instances of `fill_flag` in each of
    the given columns.
    """
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} doesn't exist")

        if not _fill_flag_exists(df, col, fill_flag):
            raise ValueError(
                f"Column {col} doesn't contain any instances of '{fill_flag}'"
            )

    if len(columns) > 1:
        return (
            df.select(columns)
            .unpivot(variable_name="column", value_name="value")
            .group_by("column")
            .agg(
                count=pl.col("value").filter(pl.col("value") == fill_flag).count(),
                prop=pl.col("value").filter(pl.col("value") == fill_flag).count()
                / pl.count(),
            )
            .sort("column")
        )

    fill_col = columns[0]

    return (
        df.select(fill_col)
        .unpivot(variable_name="column", value_name="value")
        .group_by("column")
        .agg(
            count=pl.col("value").filter(pl.col("value") == fill_flag).count(),
            prop=pl.col("value").filter(pl.col("value") == fill_flag).count()
            / pl.count(),
        )
        .sort("column")
    )


def columns[T: (pl.DataFrame, pl.LazyFrame)](
    df: T,
    columns: list[str],
    fill_flag: str,
    fill_range: tuple[int, int],
    dtype: type[IntegerType] | type[FloatType] = pl.Int64,
    seed: int | None = None,
) -> T:
    """
    Independently fill instances of `fill_flag` (a string)
    in the given columns with random integers in the given range
    (bounds inclusive).

    If `dtype` is specified, will attempt to cast the filled columns
    to that Polars type. Only supports Polars integer and float types.
    """
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} doesn't exist")

        if not _fill_flag_exists(df, col, fill_flag):
            raise ValueError(
                f"Column {col} doesn't contain any instances of '{fill_flag}'"
            )

    fill_range_int = _parse_fill_range(fill_range)

    n_cols = len(columns)
    if n_cols > 1:
        rng = np.random.default_rng(seed)
        n_rows = df.lazy().select(pl.len()).collect().item()
        # must gen enough numbers for all columns up-front, otherwise they get reused
        shape = (n_cols, n_rows)
        fill_nums = rng.integers(
            fill_range_int.lb,
            fill_range_int.ub,
            size=shape,
            endpoint=True,  # include ub in sample
        )

        for col, num in zip(columns, fill_nums):
            df = df.with_columns(
                pl.when(pl.col(col) == fill_flag)
                .then(pl.lit(num))
                .otherwise(pl.col(col))
                .alias(col)
                .cast(dtype)
            )
    else:
        column = columns[0]
        # NOTE: this implementation and numpy implementation for filling values are roughly the same speed
        # with this Polars-only impl barely faster
        df = df.with_columns(
            pl.when(pl.col(column) == fill_flag)
            .then(
                pl.int_range(fill_range_int.lb, fill_range_int.ub + 1).sample(
                    pl.len(), with_replacement=True, seed=seed
                )
            )
            .otherwise(pl.col(column))
            .alias(column)
            .cast(dtype)
        )

    return df


def _fill_flag_exists[T: (pl.DataFrame, pl.LazyFrame)](
    df: T, column: str, fill_flag: str
) -> bool:
    return df.lazy().select((pl.col(column) == fill_flag).any()).collect().item()


class _FillRange(NamedTuple):
    lb: int
    ub: int


def _parse_fill_range(fill_range: tuple[int, int]) -> _FillRange:
    if len(fill_range) != 2:
        raise ValueError("Must only pass 2 values")

    fill_range_int = _FillRange(*fill_range)
    if fill_range_int.lb > fill_range_int.ub:
        raise ValueError("Lower bound can't be greater than the upper bound")

    return fill_range_int


def column_pair[T: (pl.DataFrame, pl.LazyFrame)](
    df: T,
    numerator: str,
    denominator: str,
    fill_flag: str,
    fill_range: tuple[int, int],
    dtype: type[IntegerType] | type[FloatType] = pl.Int64,
    seed: int | None = None,
) -> T:
    """
    Fill instances of the `fill_flag` in both the `numerator` column
    and the `denominator` column such that numerator <= denominator.

    If `dtype` is specified, will attempt to cast the final result
    to that Polars type. Only supports Polars integer and float types.

    Note: `seed` is only used for (1) imputing the denominator and (2) the
    numerator case where the denominator is greater than the `fill_range`
    upper bound. This is because we cannot guarantee desired reproducible
    behavior for the numerator when denominator is less than or equal to the
    `fill_range` upper bound since such imputation happens per-row.
    """
    if numerator not in df.columns:
        raise ValueError(f"Column {numerator} doesn't exist")

    if denominator not in df.columns:
        raise ValueError(f"Column {numerator} doesn't exist")

    if not _fill_flag_exists(df, numerator, fill_flag):
        raise ValueError(
            f"Column {numerator} doesn't contain any instances of '{fill_flag}'"
        )

    if not _fill_flag_exists(df, denominator, fill_flag):
        raise ValueError(
            f"Column {denominator} doesn't contain any instances of '{fill_flag}'"
        )

    fill_range_int = _parse_fill_range(fill_range)

    df = df.with_columns(
        pl.when(pl.col(denominator) == fill_flag)
        .then(
            pl.int_range(fill_range_int.lb, fill_range_int.ub + 1).sample(
                pl.len(),
                with_replacement=True,
                seed=seed,
            )
        )
        .otherwise(pl.col(denominator))
        .alias(denominator)
        .cast(dtype)
    ).with_columns(
        # NOTE: sometimes oddly high mem consumption b/c of pl.int_ranges(),
        # but not sure how to improve
        pl.when(
            (pl.col(numerator) == fill_flag)
            & (pl.col(denominator) <= fill_range_int.ub)
        )
        .then(
            pl.int_ranges(fill_range_int.lb, pl.col(denominator) + 1)
            .list.sample(1)
            .explode()
        )
        .when(
            (pl.col(numerator) == fill_flag) & (pl.col(denominator) > fill_range_int.ub)
        )
        .then(
            pl.int_range(fill_range_int.lb, fill_range_int.ub + 1).sample(
                pl.len(),
                with_replacement=True,
                seed=seed,
            )
        )
        .otherwise(pl.col(numerator))
        .alias(numerator)
        .cast(dtype)
    )

    return df
