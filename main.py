from pathlib import Path

import numpy as np
import polars as pl
import typer
from typing_extensions import Annotated

app = typer.Typer()


# TODO: commands
# - impute:
# - input csv file
# - req output csv file -> # should check if file exists and confirm overwrite?
# - req int seed
# - req column name to impute
# - req num range e.g., 1-5, 0-5
# - req a string/symbol to look for in the named column for sub/impute
# - verbose flag for whether summary stats should be printed

# abort vs. typer.Exit(code=1)

# - impute-pair should be a separate command b/c it has more delicate logic


@app.command()
def preview(input: Annotated[str, typer.Argument()]) -> None:
    print(f"File: {input}")
    # read all as str
    df = pl.read_csv(input, infer_schema_length=0)
    print(df.head())


@app.command()
def impute(
    input: Annotated[str, typer.Argument()],
    output: Annotated[str, typer.Argument()],
    fill_col: Annotated[str, typer.Option("--fill-col", "-c")],
    fill_flag: Annotated[str, typer.Option("--fill-flag", "-f")],
    fill_range: Annotated[str, typer.Option("--fill-range", "-r")],
    seed: Annotated[int, typer.Option("--seed", "-s")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """
    Impute the column FILL_COL in a CSV file INPUT. Save the result to OUTPUT
    """
    input_file = Path(input)
    if not input_file.is_file():
        print(f"File {input_file} does not exist")
        # raise typer.Exit(code=1)
        raise typer.Abort()

    output_file = Path(output)
    if output_file.is_file():
        overwrite_file = typer.confirm(
            f"{output_file} already exists. Do you want to overwrite it?"
        )
        if not overwrite_file:
            print("Won't overwrite")
            raise typer.Abort()

    # should use Pathlib for paths instead of strings
    df = pl.read_csv(input_file, infer_schema_length=0)
    if fill_col not in df.columns:
        print(f"Column {fill_col} cannot be found in {input_file}")
        raise typer.Abort()

    fill_range_parsed = tuple(fill_range.split("-", maxsplit=1))
    if not fill_range_parsed[0].isdigit() or not fill_range_parsed[1].isdigit():
        print(f"Invalid range given for --fill-range: {fill_range_parsed}")
        # raise typer.Exit(code=1)
        raise typer.Abort()
    fill_range_parsed = tuple(int(x) for x in fill_range_parsed)

    rng = np.random.default_rng(seed)
    # imp_size only used for reporting
    imp_size = len(df.filter(pl.col(fill_col) == fill_flag))
    print(df.head())
    df = df.with_columns(
        pl.when(pl.col(fill_col) == fill_flag)
        .then(
            pl.lit(
                rng.integers(
                    fill_range_parsed[0], fill_range_parsed[1] + 1, size=df.height
                )
            ).cast(pl.String)
        )
        .otherwise(pl.col(fill_col))
        .alias(fill_col)
    )
    print(df.head())
    print(f"Imp size: {imp_size}")


if __name__ == "__main__":
    app()
