from typing import NamedTuple

import numpy as np
import polars as pl
from polars._typing import PolarsDataType


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


def complete_total_rows(df: pl.DataFrame, columns: list[pl.Series]) -> pl.DataFrame:
    """
    Generate missing rows based on unique combinations of the
    given list of series. The missing values will be nulls.
    """
    lfs = [pl.LazyFrame(col.unique()) for col in columns]
    combos = lfs[0]
    for lf in lfs[1:]:
        combos = combos.join(lf, how="cross")

    df_combos = combos.collect()

    col_names = [col.name for col in columns]
    df = df_combos.join(
        df,
        on=col_names,
        how="left",
        validate="1:1",
    )

    return df


def complete_present_rows(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    Generate missing rows based on the unique combinations
    of the given columns' values. The missing values will be nulls.
    """
    df_expand = df.select(pl.col(columns).unique().implode())
    for col in columns:
        df_expand = df_expand.explode(col)

    df = df_expand.join(df, on=columns, how="left", coalesce=True)

    return df


def _complete_rows_lazy(
    lf: pl.DataFrame | pl.LazyFrame, columns: list[str]
) -> pl.DataFrame:
    lf = lf.lazy()
    lf_expand = lf.select(pl.col(columns).unique().implode())
    for col in columns:
        lf_expand = lf_expand.explode(col)

    df = lf_expand.join(lf, on=columns, how="left", coalesce=True).collect()

    return df


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
    # TODO: this should also handle denom being in another file like the CLI
    # command?

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
            pl.int_ranges(fill_range_int.lb, pl.col(denominator) + 1)
            # TODO: use of seed?
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
