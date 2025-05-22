import polars as pl


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

    df = unique_combos.join(
        df,
        on=col_names,
        how="left",
        coalesce=True,
        validate="1:1",
    )

    return df
