import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import typer
from numpy.random import Generator
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table
from typing_extensions import Annotated


class FillRange(NamedTuple):
    lb: int
    ub: int


class FillCols(NamedTuple):
    numerator: str
    denominator: str


app = typer.Typer()
console = Console(stderr=True)


@app.command()
def preview(
    input: Annotated[str, typer.Argument()],
    n_rows: Annotated[int, typer.Option("--num-rows", "-n")] = 10,
) -> None:
    """
    Preview a given CSV file.
    """
    input_file = Path(input)
    if not input_file.is_file():
        console.print(f"File {input_file} does not exist")
        raise typer.Abort()

    print(f"File: {input_file}")
    df = pl.read_csv(input_file, infer_schema_length=0)
    print(df.head(n_rows))


@app.command()
def check(
    input: Annotated[str, typer.Argument(help="The CSV file to check")],
    fill_col: Annotated[
        str, typer.Option("--fill-col", "-c", help="Name of the column to check")
    ],
    fill_flag: Annotated[
        str,
        typer.Option("--fill-flag", "-f", help="Flag (string) to look for in FILL_COL"),
    ],
) -> None:
    """
    Check the column FILL_COL in INPUT for occurrences of FILL_FLAG.
    """
    # TODO: better control over what columns are shown; want to make sure the requested col is showing
    input_file = Path(input)
    if not input_file.is_file():
        console.print(f"File {input_file} does not exist")
        raise typer.Abort()

    df = pl.read_csv(input_file, infer_schema_length=0)
    if fill_col not in df.columns:
        console.print(f"Column {fill_col} cannot be found in {input_file}")
        raise typer.Abort()

    imp_size = len(df.filter(pl.col(fill_col) == fill_flag))
    print(
        f"Found [blue]{imp_size:_}[/blue] occurrences of '{fill_flag}' in '{fill_col}' -> [blue]{(imp_size / df.height):0.2f}[/blue] of rows (n = {df.height:_})"
    )
    print(df.filter(pl.col(fill_col) == fill_flag).head())


@app.command()
def impute(
    input: Annotated[str, typer.Argument(help="The CSV file to impute")],
    output: Annotated[str, typer.Argument(help="Path to save the output CSV")],
    fill_col: Annotated[
        str, typer.Option("--fill-col", "-c", help="Name of the column to impute")
    ],
    fill_flag: Annotated[
        str,
        typer.Option("--fill-flag", "-f", help="Flag (string) to look for in FILL_COL"),
    ],
    fill_range: Annotated[
        str,
        typer.Option(
            "--fill-range",
            "-r",
            help="Closed integer interval in which to sample for random integer imputation. Specify as comma-separated values. For example: '1,5' corresponds to the range [1, 5]",
        ),
    ],
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Random seed for reproducibility")
    ] = 123,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Whether to show additional imputation summary information",
        ),
    ] = False,
) -> None:
    """
    Impute the column FILL_COL in a CSV file INPUT. Will look for the
    symbol FILL_FLAG in FILL_COL and substitute with a random
    integer in the closed range FILL_RANGE. Save the resulting data to OUTPUT.
    """
    input_file = Path(input)
    if not input_file.is_file():
        console.print(f"File {input_file} does not exist")
        raise typer.Abort()

    df = pl.read_csv(input_file, infer_schema_length=0)

    if fill_col not in df.columns:
        console.print(f"Column {fill_col} cannot be found in {input_file}")
        raise typer.Abort()

    fill_range_parsed = tuple(x.strip() for x in fill_range.split(",", maxsplit=1))
    if not fill_range_parsed[0].isdigit() or not fill_range_parsed[1].isdigit():
        # isdigit() returns False for negative ints
        console.print(
            f"Invalid range given for --fill-range: [{fill_range_parsed[0]}, {fill_range_parsed[1]}]"
        )
        raise typer.Abort()
    else:
        fill_range_int = FillRange(int(fill_range_parsed[0]), int(fill_range_parsed[1]))
        if fill_range_int.lb > fill_range_int.ub:
            console.print(
                f"Lower bound of --fill-range is larger than upper bound: [{fill_range_int.lb}, {fill_range_int.ub}]"
            )
            raise typer.Abort()

    output_file = Path(output)
    if output_file.is_file():
        overwrite_file = Confirm.ask(
            f"[blue bold]{output_file}[/blue bold] already exists. Do you want to overwrite it?"
        )
        if not overwrite_file:
            print("Won't overwrite")
            raise typer.Abort()

    imp_size = len(df.filter(pl.col(fill_col) == fill_flag))
    if imp_size == 0:
        print(f"Cannot find any instances of {fill_flag} in {fill_col}")
        raise typer.Abort()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Imputing...", total=None)

        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        # TODO: can I actually do this without having to compute random integers for every row?
        df = df.with_columns(
            pl.when(pl.col(fill_col) == fill_flag)
            .then(
                pl.lit(
                    # must specify size to be height of df despite not filling every row
                    # thus, we get "new" rand int per row
                    rng.integers(
                        fill_range_int.lb, fill_range_int.ub + 1, size=df.height
                    )
                )
            )
            .otherwise(pl.col(fill_col))
            .alias(fill_col)
            .cast(pl.Int64)  # TODO: float or int
        )
        t1 = time.perf_counter()
        df.write_csv(output_file)

    print("[green]Finished imputing[/green]...")

    if verbose:
        table = Table(title="Imputation statistics", show_header=False)
        table.add_row("[blue]Count of imputed values[/blue]", f"{imp_size:_}")
        table.add_row(
            "[blue]Proportion of imputed values[/blue]",
            f"{(imp_size / df.height):0.2f} (n = {df.height:_})",
            end_section=True,
        )
        table.add_row("[blue]Seed[/blue]", f"{seed}")
        table.add_row("[blue]Time taken[/blue]", f"~{(t1 - t0):0.3f} s")
        console.print(table)

        # print(
        #     f"Imputed [blue]{imp_size:_}[/blue] values -> [blue]{(imp_size / df.height):0.2f}[/blue] of rows (n = {df.height:_}) affected"
        # )
        # print(f"Seed: {seed}")
        # print(f"Took ~{(t1 - t0):0.3f} s\n")
        # TODO: better control over what columns are shown; want to make sure the requested col is showing
        print(df.head())


@app.command("impute-pair")
def impute_pair(
    input: Annotated[str, typer.Argument(help="The CSV file to impute")],
    output: Annotated[str, typer.Argument(help="Path to save the output CSV")],
    fill_cols: Annotated[
        str,
        typer.Option(
            "--fill-cols",
            "-c",
            help="Pair of columns (numerator and denominator) to be imputed. Specify as comma-separated values. For example, 'count_col,denom_col' specifies 'count_col' as the numerator and 'denom_col' as the denominator.",
        ),
    ],
    fill_flag: Annotated[
        str,
        typer.Option(
            "--fill-flag", "-f", help="Flag (string) to look for in FILL_COLS"
        ),
    ],
    fill_range: Annotated[
        str,
        typer.Option(
            "--fill-range",
            "-r",
            help="Closed integer interval in which to sample for random integer imputation. Specify as comma-separated values. For example: '1,5' corresponds to the range [1, 5].",
        ),
    ],
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Random seed for reproducibility")
    ] = 123,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Whether to show additional imputation summary information",
        ),
    ] = False,
):
    """
    Impute a pair of columns FILL_COLS in a CSV file INPUT. Will look for the
    symbol FILL_FLAG in FILL_COLS and substitute with a random
    integer in the closed range FILL_RANGE. Importantly, the pair of columns
    are comprised of a numerator column and a denominator column such that
    the imputed values of the numerator column must not exceed the imputed
    values of the denominator column. Save the resulting data to OUTPUT.
    """
    input_file = Path(input)
    if not input_file.is_file():
        console.print(f"File {input_file} does not exist")
        raise typer.Abort()

    df = pl.read_csv(input_file, infer_schema_length=0)

    fill_cols_parsed = tuple(x.strip() for x in fill_cols.split(",", maxsplit=1))
    fill_cols_parsed = FillCols(fill_cols_parsed[0], fill_cols_parsed[1])
    if fill_cols_parsed.numerator not in df.columns:
        console.print(f"{fill_cols_parsed.numerator} cannot be found in {input_file}")
        raise typer.Abort()

    if fill_cols_parsed.denominator not in df.columns:
        console.print(f"{fill_cols_parsed.denominator} cannot be found in {input_file}")
        raise typer.Abort()

    fill_range_parsed = tuple(x.strip() for x in fill_range.split(",", maxsplit=1))
    if not fill_range_parsed[0].isdigit() or not fill_range_parsed[1].isdigit():
        # isdigit() returns False for negative ints
        console.print(
            f"Invalid range given for --fill-range: [{fill_range_parsed[0]}, {fill_range_parsed[1]}]"
        )
        raise typer.Abort()
    else:
        # fill_range_int = tuple(int(x) for x in fill_range_parsed)
        fill_range_int = FillRange(int(fill_range_parsed[0]), int(fill_range_parsed[1]))
        if fill_range_int.lb > fill_range_int.ub:
            console.print(
                f"Lower bound of --fill-range is larger than upper bound: [{fill_range_int.lb}, {fill_range_int.ub}]"
            )
            raise typer.Abort()

    output_file = Path(output)
    if output_file.is_file():
        overwrite_file = Confirm.ask(
            f"[blue bold]{output_file}[/blue bold] already exists. Do you want to overwrite it?"
        )
        if not overwrite_file:
            print("Won't overwrite")
            raise typer.Abort()

    imp_sizes = (
        len(df.filter(pl.col(fill_cols_parsed.numerator) == fill_flag)),
        len(df.filter(pl.col(fill_cols_parsed.denominator) == fill_flag)),
    )
    if imp_sizes[0] == 0 and imp_sizes[1] == 0:
        print(
            f"Cannot find any instances of {fill_flag} in either {fill_cols_parsed.numerator} or {fill_cols_parsed.denominator}"
        )
        raise typer.Abort()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Imputing...", total=None)

        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        df = df.with_columns(
            pl.when(pl.col(fill_cols_parsed.denominator) == fill_flag)
            .then(
                pl.lit(
                    rng.integers(
                        fill_range_int.lb, fill_range_int.ub + 1, size=df.height
                    )
                )
            )
            .otherwise(pl.col(fill_cols_parsed.denominator))
            .alias(fill_cols_parsed.denominator)
            .cast(pl.Float64)  # TODO: is this ok; float or int?
        ).with_columns(
            pl.when(
                (pl.col(fill_cols_parsed.numerator) == fill_flag)
                & (pl.col(fill_cols_parsed.denominator) <= fill_range_int.ub)
            )
            # TODO: is this correct use of map_elements()? I believe this is now running Python
            # so will be slow
            .then(
                pl.col(fill_cols_parsed.denominator).map_elements(
                    lambda x: fill_parallel(
                        x,
                        fill_range_int,
                        rng,
                    ),
                    return_dtype=pl.Int64,
                )
            )
            .when(
                (pl.col(fill_cols_parsed.numerator) == fill_flag)
                & (pl.col(fill_cols_parsed.denominator) > fill_range_int.ub)
            )
            .then(
                pl.lit(
                    rng.integers(
                        fill_range_int.lb, fill_range_int.ub + 1, size=df.height
                    )
                )
            )
            .otherwise(pl.col(fill_cols_parsed.numerator))
            .alias(fill_cols_parsed[0])
            .cast(pl.Float64)  # TODO: is this ok; float or int?
        )
        t1 = time.perf_counter()
        df.write_csv(output_file)

    print("[green]Finished imputing[/green]...")

    # TODO: consider rich table here
    # TODO: if prop imputed really small or 0, should prob have a diff message
    # TODO: this "0.4f" is arbitrary
    if verbose:
        table = Table(title="Imputation statistics", show_header=False)
        table.add_row(
            f"[blue]Count of imputed values in[/blue] '{fill_cols_parsed.numerator}'",
            f"{imp_sizes[0]:_}",
        )
        table.add_row(
            f"[blue]Proportion of imputed values in[/blue] '{fill_cols_parsed.numerator}",
            f"{(imp_sizes[0] / df.height):0.2f} (n = {df.height:_})",
            end_section=True,
        )
        table.add_row(
            f"[blue]Count of imputed values in[/blue] '{fill_cols_parsed.denominator}'",
            f"{imp_sizes[1]:_}",
        )
        table.add_row(
            f"[blue]Proportion of imputed values in[/blue] '{fill_cols_parsed.denominator}'",
            f"{(imp_sizes[1] / df.height):0.2f} (n = {df.height:_})",
            end_section=True,
        )
        table.add_row("[blue]Seed[/blue]", f"{seed}")
        table.add_row("[blue]Time taken[/blue]", f"~{(t1 - t0):0.3f} s")
        console.print(table)

        # print(
        #     f"Imputed [blue]{imp_sizes[0]:_}[/blue] values in {fill_cols_parsed.numerator} -> [blue]{(imp_sizes[0] / df.height):0.2f}[/blue] of rows (n = {df.height:_}) affected"
        # )
        # print(
        #     f"Imputed [blue]{imp_sizes[1]:_}[/blue] values in {fill_cols_parsed.denominator} -> [blue]{(imp_sizes[1] / df.height):0.4f}[/blue] of rows (n = {df.height:_}) affected"
        # )
        # print(f"Seed: {seed}")
        # print(f"Took ~{(t1 - t0):0.3f} s\n")

        print(
            df.filter(pl.col(fill_cols_parsed.denominator) <= 5)
            .select(fill_cols_parsed.numerator, fill_cols_parsed.denominator)
            .head()
        )


def fill_parallel(denominator, fill_range_int: FillRange, rng: Generator) -> int:
    """
    Return a random integer from a range that is capped
    at the 'denominator' value
    """
    # TODO: what's the type of denominator?
    if denominator <= fill_range_int.ub:
        val = rng.integers(fill_range_int.lb, denominator + 1)
    else:
        val = rng.integers(fill_range_int.lb, fill_range_int.ub + 1)
    return val


if __name__ == "__main__":
    app()
