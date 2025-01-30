from typing import NamedTuple, TypeAlias

import numpy as np
import polars as pl

PolarsNumericType: TypeAlias = (
    pl.Decimal
    | pl.Float32
    | pl.Float64
    | pl.Int8
    | pl.Int16
    | pl.Int32
    | pl.Int64
    | pl.Int128
    | pl.UInt8
    | pl.UInt16
    | pl.UInt32
    | pl.UInt64
)


def check(df: pl.DataFrame, fill_cols: list[str], fill_flag: str) -> pl.DataFrame:
    """
    Return dataframe with counts and proportion of instances of fill_flag in each of
    the given fill_cols
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
    else:
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
    col_type: PolarsNumericType | None = None,
    seed: int | None = None,
) -> pl.DataFrame:
    """
    Fill instances of the fill flag (a string) in the given column
    with random integers in the given range (inclusive).

    If col_type is specified, will attempt to cast the final result
    of fill_cols to that type. Currently, the only options are
    Polars numeric types.
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
        # must gen all nums up front
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
            )

            if col_type is not None:
                df = df.with_columns(pl.col(col).cast(col_type))
    else:
        fill_col = fill_cols[0]
        # NOTE: this implementation and numpy implementation for filling values are roughly the same speed
        # with this native impl barely faster
        df = df.with_columns(
            pl.when(pl.col(fill_col) == fill_flag)
            .then(
                pl.int_range(fill_range_int.lb, fill_range_int.ub + 1).sample(
                    pl.len(),
                    with_replacement=True,
                    seed=seed,
                )
            )
            .otherwise(pl.col(fill_col))
            .alias(fill_col),
        )

        if col_type is not None:
            df = df.with_columns(pl.col(fill_col).cast(col_type))

    return df


def impute_columns_lazy(
    lf: pl.DataFrame | pl.LazyFrame,
    fill_column: str,
    fill_flag: str,
    fill_range: tuple[int, int],
    seed: int | None = None,
) -> pl.DataFrame:
    lf = lf.lazy()

    if fill_column not in lf.collect_schema().names():
        raise ValueError(f"Column {fill_column} doesn't exist")

    if not fill_flag_exists(lf, fill_column, fill_flag):
        raise ValueError(
            f"Column {fill_column} doesn't contain any instances of '{fill_flag}'"
        )

    fill_range_int = parse_fill_range(fill_range)

    df = lf.with_columns(
        pl.when(pl.col(fill_column) == fill_flag)
        .then(
            pl.int_range(fill_range_int.lb, fill_range_int.ub + 1).sample(
                pl.len(), with_replacement=True
            )
        )
        .otherwise(pl.col(fill_column))
        .alias(fill_column)
    ).collect()

    return df


def fill_flag_exists(
    df: pl.DataFrame | pl.LazyFrame, fill_col: str, fill_flag: str
) -> bool:
    # TODO: could just do lf = df.lazy() then don't need isinstance()
    if isinstance(df, pl.DataFrame):
        return df.select((pl.col(fill_col) == fill_flag).any()).item()
    else:
        # TODO: is there another way to check that doesn't materialize lf?
        return df.select((pl.col(fill_col) == fill_flag).any()).collect().item()


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


# TODO: difficult to benchmark without reasonably sized data
def complete_rows(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    Generate implicit missing rows based on the unique combinations
    of the given columns' values. The missing values will be nulls.
    """
    df_expand = df.select(pl.col(columns).unique().implode())
    for col in columns:
        df_expand = df_expand.explode(col)

    df = df_expand.join(df, on=columns, how="left", coalesce=True)
    return df


def complete_rows_lazy(
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
    seed: int | None = None,
) -> pl.DataFrame:
    return df


if __name__ == "__main__":
    import timeit

    print("Benchmarking implementations...")

    repeat = 2
    number = 100_000

    # NOTE: impute_flag somehow slower? For gen_rows, if instead of passing in a LazyFrame I
    # take a DataFrame and convert it to LazyFrame inside the func, the times are more comparable
    # Maybe I'm doing benchmark wrong?

    setup = """
import polars as pl
from __main__ import impute_columns
df = pl.read_csv('../../tests/data/test_impute_data.csv')
    """
    t_eager = timeit.repeat(
        "impute_columns(df, ['cases'], '<=5', (1, 5))",
        setup=setup,
        repeat=repeat,
        number=number,
    )

    setup = """
import polars as pl
from __main__ import impute_columns_lazy
df = pl.scan_csv('../../tests/data/test_impute_data.csv')
    """
    t_lazy = timeit.repeat(
        "impute_columns_lazy(df, ['cases'], '<=5', (1, 5))",
        setup=setup,
        repeat=repeat,
        number=number,
    )

    print(f"Min. time of impute.impute_columns(): {min(t_eager)}")
    print(f"Min. time of impute.impute_columns_lazy(): {min(t_lazy)}")

#     setup = """
# import polars as pl
# from __main__ import complete_rows
# df = pl.DataFrame(
#     {
#         "orig": ["France", "France", "UK", "UK", "Spain"],
#         "dest": ["Japan", "Vietnam", "Japan", "China", "China"],
#         "year": [2020, 2021, 2019, 2020, 2022],
#         "value": [1, 2, 3, 4, 5],
#     }
# )
#     """
#     t_eager = timeit.repeat(
#         "complete_rows(df, ['orig', 'dest', 'year'])",
#         setup=setup,
#         repeat=repeat,
#         number=number,
#     )
#
#     setup = """
# import polars as pl
# from __main__ import complete_rows_lazy
# df = pl.LazyFrame(
#     {
#         "orig": ["France", "France", "UK", "UK", "Spain"],
#         "dest": ["Japan", "Vietnam", "Japan", "China", "China"],
#         "year": [2020, 2021, 2019, 2020, 2022],
#         "value": [1, 2, 3, 4, 5],
#     }
# )
#     """
#     t_lazy = timeit.repeat(
#         "complete_rows_lazy(df, ['orig', 'dest', 'year'])",
#         setup=setup,
#         repeat=repeat,
#         number=number,
#     )
#
#     print(f"Min. time of impute.complete_rows(): {min(t_eager)}")
#     print(f"Min. time of impute.complete_rows_lazy(): {min(t_lazy)}")
