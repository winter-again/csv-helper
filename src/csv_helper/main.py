import time
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Optional

import click
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


# NOTE: see https://github.com/tiangolo/typer/issues/151#issuecomment-1975322806
# for workaround for working with enums like this such that Typer understands the args properly
# without having to translate strings or ints to the values we really want
class ColType(Enum):
    INT64 = pl.Int64
    FLOAT64 = pl.Float64


app = typer.Typer()
err_console = Console(stderr=True)


def fill_cols_exist(df: pl.DataFrame, fill_cols: list[str]) -> bool:
    for col in fill_cols:
        if col not in df.columns:
            return False
    return True


def parse_validate_fill_range(fill_range: str) -> Optional[FillRange]:
    fill_range_parsed = tuple(x.strip() for x in fill_range.split(",", maxsplit=1))
    if not fill_range_parsed[0].isdigit() or not fill_range_parsed[1].isdigit():
        return None
    fill_range_int = FillRange(int(fill_range_parsed[0]), int(fill_range_parsed[1]))
    if fill_range_int.lb > fill_range_int.ub:
        return None
    return fill_range_int


def fill_flag_exists(df: pl.DataFrame, fill_col: str, fill_flag: str) -> bool:
    imp_size = len(df.filter(pl.col(fill_col) == fill_flag))
    if imp_size == 0:
        return False
    return True


@app.command()
def preview(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="The CSV file to preview",
        ),
    ],
    n_rows: Annotated[
        int, typer.Option("--nrows", "-n", min=1, help="Number of rows to preview")
    ] = 10,
) -> None:
    """
    Preview a given CSV file.
    """
    print(f"File: {input}")
    df = pl.read_csv(input, infer_schema_length=0)
    print(df.head(n_rows))


@app.command()
def check(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="The CSV file to check",
        ),
    ],
    fill_col: Annotated[
        str, typer.Option("--col", "-c", help="Name of the column to check")
    ],
    fill_flag: Annotated[
        str,
        typer.Option("--flag", "-f", help="Flag (string) to look for in COL"),
    ],
) -> None:
    """
    Check a column in a CSV file for occurrences of some string flag.
    """
    df = pl.read_csv(input, infer_schema_length=0)
    if not fill_cols_exist(df, [fill_col]):
        err_console.print(f"Column {fill_col} cannot be found in {input}")
        raise typer.Abort()

    imp_size = len(df.filter(pl.col(fill_col) == fill_flag))
    print(
        f"Found [blue]{imp_size:_}[/blue] occurrences of '{fill_flag}' in '{fill_col}' -> [blue]{(imp_size / df.height):0.2f}[/blue] of rows (n = {df.height:_})"
    )
    print(df.filter(pl.col(fill_col) == fill_flag).head())


@app.command()
def impute(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="The CSV file to impute",
        ),
    ],
    output: Annotated[
        Path,
        # NOTE: if exists = False, other checks still run if the Path happens to (file/dir) exist
        typer.Argument(
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=False,
            help="Path to save the output CSV",
        ),
    ],
    fill_col: Annotated[
        str, typer.Option("--col", "-c", help="Name of the column to impute")
    ],
    fill_flag: Annotated[
        str,
        typer.Option(
            "--flag",
            "-f",
            help="Flag (string) to look for and replace in the target column",
        ),
    ],
    fill_range: Annotated[
        str,
        typer.Option(
            "--range",
            "-r",
            help="Closed, integer interval from which to sample random integer for imputation. Specify as comma-separated values. For example: '1,5' corresponds to the range [1, 5]",
        ),
    ],
    col_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Intended data type of the target column. Can be a Polars Int64 or Float64.",
            click_type=click.Choice(ColType._member_names_, case_sensitive=False),
        ),
    ] = ColType.INT64.name,
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Random seed for reproducibility")
    ] = 123,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-F",
            help="Force imputing the data even if the specified output file already exists",
        ),
    ] = False,
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
    Impute a target column in a CSV file. Will look for the specified filler flag in the target column
    and replace it with a random integer from the specified range. Save the result to a new CSV file.
    """
    if output.is_file() and not force:
        overwrite_file = Confirm.ask(
            f"The file [blue bold]{output}[/blue bold] already exists. Do you want to overwrite it?"
        )
        if not overwrite_file:
            print("Won't overwrite")
            raise typer.Abort()

    create_dir = False
    if not output.parent.is_dir():
        create_dir = Confirm.ask(
            f"The specified output's parent directory [blue bold]{output.parent}[/blue bold] doesn't exist. Do you want to create it along with any missing parents?"
        )
        if not create_dir:
            print("Won't create directories")
            raise typer.Abort()

    df = pl.read_csv(input, infer_schema_length=0)

    if not fill_cols_exist(df, [fill_col]):
        err_console.print(f"Column {fill_col} cannot be found in {input}")
        raise typer.Abort()

    fill_range_int = parse_validate_fill_range(fill_range)
    if fill_range_int is None:
        err_console.print(f"Invalid range given for --range: {fill_range}")
        raise typer.Abort()

    if not fill_flag_exists(df, fill_col, fill_flag):
        err_console.print(f"Cannot find any instances of '{fill_flag}' in {fill_col}")
        raise typer.Abort()

    if verbose:
        imp_size = len(df.filter(pl.col(fill_col) == fill_flag))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Imputing...", total=None)

        rng = np.random.default_rng(seed)
        cast_type = ColType[col_type]

        t0 = time.perf_counter()
        df = df.with_columns(
            pl.when(pl.col(fill_col) == fill_flag)
            .then(
                pl.lit(
                    # NOTE: must specify size to be height of df despite not filling every row
                    # thus, we get "new" rand int per row
                    rng.integers(
                        fill_range_int.lb, fill_range_int.ub + 1, size=df.height
                    )
                )
            )
            .otherwise(pl.col(fill_col))
            .alias(fill_col)
            .cast(cast_type.value)
        )
        t1 = time.perf_counter()

        if create_dir:
            output.parent.mkdir(parents=True)

        df.write_csv(output)

    print("[green]Finished imputing[/green]...")

    if verbose:
        table = Table(title="Imputation statistics", show_header=False)
        table.add_row("[blue]Count of imputed values[/blue]", f"{imp_size:_}")
        table.add_row(
            "[blue]Proportion of imputed values[/blue]",
            f"{(imp_size / df.height):0.2f} (n = {df.height:_})",
        )
        table.add_row("[blue]Seed[/blue]", f"{seed}")
        table.add_row("[blue]Time taken[/blue]", f"~{(t1 - t0):0.3f} s")
        print(table)

        print(df.filter(pl.col(fill_col) <= fill_range_int.ub).head())


@app.command("impute-dir")
def impute_dir(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Directory of CSV files to impute",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            exists=False,
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="Directory to save output CSV files",
        ),
    ],
    fill_col: Annotated[
        str, typer.Option("--col", "-c", help="Name of the column to impute")
    ],
    fill_flag: Annotated[
        str,
        typer.Option(
            "--flag",
            "-f",
            help="Flag (string) to look for and replace in the target column",
        ),
    ],
    fill_range: Annotated[
        str,
        typer.Option(
            "--range",
            "-r",
            help="Closed, integer interval from which to sample random integer for imputation. Specify as comma-separated values. For example: '1,5' corresponds to the range [1, 5]",
        ),
    ],
    col_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Intended data type of the target column. Can be a Polars Int64 or Float64.",
            click_type=click.Choice(ColType._member_names_, case_sensitive=False),
        ),
    ] = ColType.INT64.name,
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Random seed for reproducibility")
    ] = 123,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-F",
            help="Force imputing the data if INPUT is identical to OUTPUT",
        ),
    ] = False,
    suffix: Annotated[
        str,
        typer.Option(
            "--suffix", "-x", help="Optional suffix to append to each imputed CSV file"
        ),
    ] = "",
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
    Impute a target column for a directory of uniform CSV files. Will look for a specific filler flag in the target column
    and replace with a random integer from the the specified range. Save the result in the given output directory.
    """
    files = list(input_dir.glob("*.csv"))
    if len(files) == 0:
        err_console.print(
            f"The specified input directory [blue bold]{input_dir}[/blue bold] is either empty or doesn't contain any CSV files."
        )
        raise typer.Abort()

    create_dir = False
    if not output_dir.is_dir():
        create_dir = Confirm.ask(
            f"The specified output directory [blue bold]{output_dir}[/blue bold] doesn't exist. Do you want to create it along with any missing parents?"
        )
        if not create_dir:
            print("Won't create directories")
            raise typer.Abort()

    for file in files:
        if suffix != "":
            output_file = output_dir / f"{file.stem}_{suffix}{file.suffix}"
        else:
            output_file = output_dir / file.name

        if output_file.is_file() and not force:
            overwrite_file = Confirm.ask(
                f"The intended output file [blue bold]{output_file}[/blue bold] already exists. Should it be overwritten?"
            )
            if not overwrite_file:
                print("Won't overwrite")
                raise typer.Abort()

        df = pl.read_csv(file, infer_schema_length=0)

        if not fill_cols_exist(df, [fill_col]):
            err_console.print(f"Column {fill_col} cannot be found in {file}")
            raise typer.Abort()

        fill_range_int = parse_validate_fill_range(fill_range)
        if fill_range_int is None:
            err_console.print(f"Invalid range given for --range: {fill_range}")
            raise typer.Abort()

        if verbose:
            imp_size = len(df.filter(pl.col(fill_col) == fill_flag))

        if not fill_flag_exists(df, fill_col, fill_flag):
            err_console.print(
                f"Cannot find any instances of '{fill_flag}' in {fill_col}"
            )
            raise typer.Abort()

        rng = np.random.default_rng(seed)
        cast_type = ColType[col_type]

        t0 = time.perf_counter()
        df = df.with_columns(
            pl.when(pl.col(fill_col) == fill_flag)
            .then(
                pl.lit(
                    # NOTE: must specify size to be height of df despite not filling every row
                    # thus, we get "new" rand int per row
                    rng.integers(
                        fill_range_int.lb, fill_range_int.ub + 1, size=df.height
                    )
                )
            )
            .otherwise(pl.col(fill_col))
            .alias(fill_col)
            .cast(cast_type.value)
        )
        t1 = time.perf_counter()

        if create_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        df.write_csv(output_file)

        print(f"\nFinished imputing [blue]{file}[/blue]...")

        if verbose:
            table = Table(title="Imputation statistics", show_header=False)
            table.add_row("[blue]Count of imputed values[/blue]", f"{imp_size:_}")
            table.add_row(
                "[blue]Proportion of imputed values[/blue]",
                f"{(imp_size / df.height):0.2f} (n = {df.height:_})",
            )
            table.add_row("[blue]Seed[/blue]", f"{seed}")
            table.add_row("[blue]Time taken[/blue]", f"~{(t1 - t0):0.3f} s")
            print(table)

            print(df.filter(pl.col(fill_col) <= fill_range_int.ub).head())


def fill_parallel(denominator: int, fill_range_int: FillRange, rng: Generator) -> int:
    """
    Return a random integer from a range that is capped
    at the 'denominator' value
    """
    val = rng.integers(fill_range_int.lb, denominator + 1)
    return val


@app.command("impute-pair")
def impute_pair(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="The CSV file to impute",
        ),
    ],
    output: Annotated[
        Path,
        typer.Argument(
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=False,
            help="Path to save the output CSV",
        ),
    ],
    fill_cols: Annotated[
        str,
        typer.Option(
            "--cols",
            "-c",
            help="Pair of columns (numerator and denominator) to be imputed. Specify as comma-separated values. For example, 'count_col,denom_col' specifies 'count_col' as the numerator and 'denom_col' as the denominator.",
        ),
    ],
    fill_flag: Annotated[
        str,
        typer.Option("--flag", "-f", help="Flag (string) to look for in COLS"),
    ],
    fill_range: Annotated[
        str,
        typer.Option(
            "--range",
            "-r",
            help="Closed integer interval in which to sample for random integer imputation. Specify as comma-separated values. For example: '1,5' corresponds to the range [1, 5].",
        ),
    ],
    col_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Intended data type of COLS. Can be a Polars Int64 or Float64.",
            click_type=click.Choice(ColType._member_names_, case_sensitive=False),
        ),
    ] = ColType.INT64.name,
    sep_file: Annotated[
        Optional[Path],
        typer.Option(
            "--sep",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to some separate CSV file in which to look for denominator data. Currently only supports a separate file that has the exact same structure as the input file except for the numerator column being swapped for the denominator column.",
        ),
    ] = None,
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Random seed for reproducibility")
    ] = 123,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-F",
            help="Force imputing the data if INPUT is identical to OUTPUT",
        ),
    ] = False,
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
    Impute a pair of columns COLS in a CSV file INPUT. Will look for the
    symbol FLAG in COLS and substitute with a random integer in the closed
    range RANGE. Importantly, the pair of columns is comprised of a numerator
    column and a denominator column such that the imputed values of the
    numerator column must not exceed the imputed values of the denominator
    column. Save the resulting data to OUTPUT.

    If --sep is provided, then specified file will be used to find the denominator column given in COLS.
    """
    if output.is_file():
        overwrite_file = Confirm.ask(
            f"[blue bold]{output}[/blue bold] already exists. Do you want to overwrite it?"
        )
        if not overwrite_file:
            print("Won't overwrite")
            raise typer.Abort()

    if input == output and not force:
        err_console.print(
            "Cannot specify output to be identical to input. Use the --force/-F option to force this behavior"
        )
        raise typer.Abort()

    create_dir = False
    if not output.parent.is_dir():
        create_dir = Confirm.ask(
            f"The specified output's parent directory [blue bold]{output.parent}[/blue bold] doesn't exist. Do you want to create it along with any missing parents?"
        )
        if not create_dir:
            print("Won't create directories")
            raise typer.Abort()

    df = pl.read_csv(input, infer_schema_length=0)

    if sep_file is not None:
        df_sep = pl.read_csv(sep_file, infer_schema_length=0)

    if input == output and not force:
        err_console.print(
            "Cannot specify output to be identical to input. Use the --force/-F option to force this behavior"
        )
        raise typer.Abort()

    fill_cols_parsed = tuple(x.strip() for x in fill_cols.split(",", maxsplit=1))
    if sep_file is not None:
        # TODO: why doesn't this work?
        # if not fill_cols_exist(df_sep, list(fill_cols_parsed[1])):
        if fill_cols_parsed[1] not in df_sep.columns:
            err_console.print(
                "Separate data file doesn't contain the given denominator column"
            )
            raise typer.Abort()
    else:
        if not fill_cols_exist(df, list(fill_cols_parsed)):
            err_console.print("Invalid columns specified for --cols")
            raise typer.Abort()

    fill_cols_parsed = FillCols(fill_cols_parsed[0], fill_cols_parsed[1])

    fill_range_int = parse_validate_fill_range(fill_range)
    if fill_range_int is None:
        err_console.print(f"Invalid range given for --range: {fill_range}")
        raise typer.Abort()

    if sep_file is not None:
        imp_sizes = (
            len(df.filter(pl.col(fill_cols_parsed.numerator) == fill_flag)),
            len(df_sep.filter(pl.col(fill_cols_parsed.denominator) == fill_flag)),
        )
    else:
        imp_sizes = (
            len(df.filter(pl.col(fill_cols_parsed.numerator) == fill_flag)),
            len(df.filter(pl.col(fill_cols_parsed.denominator) == fill_flag)),
        )

    if imp_sizes[0] == 0 and imp_sizes[1] == 0:
        err_console.print(
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
        cast_type = ColType[col_type]

        if sep_file is not None:
            # TODO: is this correct? currently primitive capabilities
            join_cols = [
                col
                for col in df.columns
                if col in df_sep.columns and col != fill_cols_parsed.numerator
            ]
            df = df.join(df_sep, on=join_cols, how="inner", coalesce=True)
            print(df)

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
            .cast(cast_type.value)
        ).with_columns(
            pl.when(
                (pl.col(fill_cols_parsed.numerator) == fill_flag)
                & (pl.col(fill_cols_parsed.denominator) <= fill_range_int.ub)
            )
            # map_elements() will run Python so it's slow
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
            .cast(cast_type.value)
        )
        t1 = time.perf_counter()

        print(df)

        if create_dir:
            output.parent.mkdir(parents=True)

        df.write_csv(output)

    print("[green]Finished imputing[/green]...")

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
        print(table)

        print(
            df.filter(
                (pl.col(fill_cols_parsed.numerator) <= fill_range_int.ub)
                | (pl.col(fill_cols_parsed.denominator) <= fill_range_int.ub)
            ).head()
        )


if __name__ == "__main__":
    app()
