from typing import NamedTuple

import numpy as np
import polars as pl
from polars._typing import PolarsDataType


# TODO: make this a check func that returns bool instead?
# and separate this into another func like summarize()?
def check(df: pl.DataFrame, fill_cols: list[str], fill_flag: str) -> pl.DataFrame:
    """
    Return dataframe with counts and proportion of instances of `fill_flag` in each of
    the given `fill_cols`
    """
    for col in fill_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} doesn't exist")

        if not fill_flag_exists(df, col, fill_flag):
            raise ValueError(
                f"Column {col} doesn't contain any instances of '{fill_flag}'"
            )

    if len(fill_cols) > 1:
        return (
            df.select(fill_cols)
            .unpivot(variable_name="column", value_name="value")
            .group_by("column")
            .agg(
                count=pl.col("value").filter(pl.col("value") == fill_flag).count(),
                prop=pl.col("value").filter(pl.col("value") == fill_flag).count()
                / pl.count(),
            )
            .sort("column")
        )

    fill_col = fill_cols[0]
    return (
        df.select(fill_col)
        .unpivot(variable_name="column", value_name="value")
        .group_by("column")
        .agg(
            count=pl.col("value").filter(pl.col("value") == fill_flag).count(),
            prop=pl.col("value").filter(pl.col("value") == fill_flag).count()
            / pl.count(),
        )
    )


# TODO: instead of separate lazy func, let this take df or lf
# or have bool arg that determines whether .lazy() conversion happens?
# TODO: add asserts for shape?
# TODO: use TFrame here?
def impute_columns(
    df: pl.DataFrame,
    fill_cols: list[str],
    fill_flag: str,
    fill_range: tuple[int, int],
    col_type: PolarsDataType = pl.Int64,
    seed: int | None = None,
) -> pl.DataFrame:
    """
    Fill instances of `fill_flag` (a string) in the given column
    with random integers in the given range (inclusive).

    If `col_type` is specified, will attempt to cast the final result
    of `fill_cols` to that Polars type.
    """
    for col in fill_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} doesn't exist")

        if not fill_flag_exists(df, col, fill_flag):
            raise ValueError(
                f"Column {col} doesn't contain any instances of '{fill_flag}'"
            )

    fill_range_int = parse_fill_range(fill_range)

    if len(fill_cols) > 1:
        rng = np.random.default_rng(seed)
        n = (len(fill_cols), df.height)
        # must gen enough numbers for whole column up-front, otherwise reused
        fill_nums = rng.integers(
            fill_range_int.lb,
            fill_range_int.ub,
            size=n,
            endpoint=True,
        )

        for col, num in zip(fill_cols, fill_nums):
            df = df.with_columns(
                pl.when(pl.col(col) == fill_flag)
                .then(pl.lit(num))
                .otherwise(pl.col(col))
                .alias(col)
                .cast(col_type)
            )
    else:
        fill_col = fill_cols[0]
        # NOTE: this implementation and numpy implementation for filling values are roughly the same speed
        # with this Polars-only impl barely faster
        df = df.with_columns(
            pl.when(pl.col(fill_col) == fill_flag)
            .then(
                pl.int_range(fill_range_int.lb, fill_range_int.ub + 1).sample(
                    pl.len(), with_replacement=True, seed=seed
                )
            )
            .otherwise(pl.col(fill_col))
            .alias(fill_col)
            .cast(col_type)
        )

    return df


def _impute_columns_lazy(
    lf: pl.DataFrame | pl.LazyFrame,
    fill_column: str,
    fill_flag: str,
    fill_range: tuple[int, int],
    seed: int | None = None,
) -> pl.DataFrame:
    """
    Experimental with Lazy
    """
    lf = lf.lazy()

    if fill_column not in lf.collect_schema().names():
        raise ValueError(f"Column {fill_column} doesn't exist")

    if not _fill_flag_exists_lazy(lf, fill_column, fill_flag):
        raise ValueError(
            f"Column {fill_column} doesn't contain any instances of '{fill_flag}'"
        )

    fill_range_int = parse_fill_range(fill_range)

    df = lf.with_columns(
        pl.when(pl.col(fill_column) == fill_flag)
        .then(
            pl.int_range(fill_range_int.lb, fill_range_int.ub + 1).sample(
                pl.len(), with_replacement=True, seed=seed
            )
        )
        .otherwise(pl.col(fill_column))
        .alias(fill_column)
    ).collect()

    return df


def fill_flag_exists(df: pl.DataFrame, fill_col: str, fill_flag: str) -> bool:
    return df.select((pl.col(fill_col) == fill_flag).any()).item()


def _fill_flag_exists_lazy(
    df: pl.DataFrame | pl.LazyFrame, fill_col: str, fill_flag: str
) -> bool:
    lf = df.lazy()
    return lf.select((pl.col(fill_col) == fill_flag).any()).collect().item()


class FillRange(NamedTuple):
    lb: int
    ub: int


def parse_fill_range(fill_range: tuple[int, int]) -> FillRange:
    if len(fill_range) != 2:
        raise ValueError("Must only pass 2 values")

    fill_range_int = FillRange(*fill_range)
    if fill_range_int.lb > fill_range_int.ub:
        raise ValueError("Lower bound can't be greater than the upper bound")

    return fill_range_int


def complete[T: (pl.DataFrame, pl.LazyFrame)](df: T, *columns: str | pl.Series) -> T:
    """
    Generate rows for implicit missing values based on column combinations,
    thus making them explicit missing values. Generated values marked as null.

    If columns are referenced with strings, then only existing values in those
    columns are used for completion. If Series are specified instead, then
    those Series can specify the full set of possible values, provided that
    the Series is named after an existing column.
    """
    cols = []
    for col in columns:
        if isinstance(col, str):
            cols.append(pl.col(col).unique().implode())
        elif isinstance(col, pl.Series):
            cols.append(col.unique().implode())
        else:
            raise TypeError(
                f"The columns argument(s) must be either string or polars Series. Got {type(col)} instead."
            )

    unique_combos = df.select(cols)
    col_names = unique_combos.collect_schema().names()
    for col in col_names:
        unique_combos = unique_combos.explode(col)

    return unique_combos.join(
        df, on=col_names, how="left", coalesce=True, validate="1:1"
    )


def impute_column_pair(
    df: pl.DataFrame,
    numerator: str,
    denominator: str,
    fill_flag: str,
    fill_range: tuple[int, int],
    col_type: PolarsDataType = pl.Int64,
    seed: int | None = None,
) -> pl.DataFrame:
    """
    Fill instances of the fill_flag in both the numerator column
    and the denominator column such that numerator <= denominator.

    If col_type is specified, will attempt to cast the final result
    of fill_cols to that type. Currently, the only options are
    Polars numeric types.
    """
    # TODO: should this also handle denom being in another file or dataframe (like the CLI
    # command?)

    if numerator not in df.columns:
        raise ValueError(f"Column {numerator} doesn't exist")

    if denominator not in df.columns:
        raise ValueError(f"Column {numerator} doesn't exist")

    if not fill_flag_exists(df, numerator, fill_flag):
        raise ValueError(
            f"Column {numerator} doesn't contain any instances of '{fill_flag}'"
        )

    if not fill_flag_exists(df, denominator, fill_flag):
        raise ValueError(
            f"Column {denominator} doesn't contain any instances of '{fill_flag}'"
        )

    fill_range_int = parse_fill_range(fill_range)

    # TODO: I think repeated use of the same seed is undesirable
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
        .cast(col_type)
    )

    df = df.with_columns(
        # TODO: use list b/c no arr.sample() what about struct perf?
        pl.when(
            (pl.col(numerator) == fill_flag)
            & (pl.col(denominator) <= fill_range_int.ub)
        )
        .then(
            # TODO: look into high mem consumption for this pl.when()
            # TODO: use of seed?
            pl.int_ranges(fill_range_int.lb, pl.col(denominator) + 1)
            .list.sample(1)
            .explode()
        )
        .when(
            (pl.col(numerator) == fill_flag) & (pl.col(denominator) > fill_range_int.ub)
        )
        .then(
            pl.int_range(fill_range_int.lb, fill_range_int.ub + 1).sample(
                # TODO: use of seed?
                pl.len(),
                with_replacement=True,
                seed=seed,
            )
        )
        .otherwise(pl.col(numerator))
        .alias(numerator)
        .cast(pl.Int64)
    )

    return df
