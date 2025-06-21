import time
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple

import click
import polars as pl
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from typing_extensions import Annotated

from . import impute

app = typer.Typer(no_args_is_help=True, help="A CLI for working with CSV data")
impute_app = typer.Typer(no_args_is_help=True, help="Impute CSV data")
app.add_typer(impute_app, name="impute")

console = Console()
err_console = Console(stderr=True)


def version_callback(value: bool):
    """Print CLI version"""
    if value:
        print(f"csv-helper version {version(__package__)}")  # pyright: ignore[reportArgumentType]
        raise typer.Exit()


@app.callback()
def callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        is_eager=True,
        help="Print the version and exit.",
        callback=version_callback,
    ),
) -> None:
    pass


@app.command()
def show(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Target CSV file",
        ),
    ],
    n_rows: Annotated[
        int, typer.Option("--nrows", "-n", min=1, help="Number of rows to show")
    ] = 10,
) -> None:
    """
    Show preview of a given CSV file.
    """
    df = pl.read_csv(input, infer_schema_length=0)

    if n_rows > df.height:
        print(df)
    else:
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
    columns: Annotated[
        list[str],
        typer.Option(
            "--col",
            "-c",
            help="Name of a column to check. Specify this for each column you want checked.",
        ),
    ],
    fill_flag: Annotated[
        str,
        typer.Option("--flag", "-f", help="Flag (string) to look for in COL"),
    ],
) -> None:
    """
    Summarize counts and proportion of instances of `fill_flag` in each of
    the given columns.
    """
    df = pl.read_csv(input, infer_schema_length=0)

    try:
        out = impute.check(df, columns, fill_flag)
    except ValueError as e:
        if f"doesn't contain any instances of '{fill_flag}'" in str(e):
            print(e)
    else:
        print(out)


class FillRange(NamedTuple):
    lb: int
    ub: int


# NOTE: see https://github.com/fastapi/typer/issues/182#issuecomment-1708245110
# and https://github.com/fastapi/typer/issues/151#issuecomment-1975322806
# for this workaround for working with enums such that Typer understands the args properly
# without having to map strings or ints to the values we really want
class ColType(Enum):
    FLOAT32 = pl.Float32
    FLOAT64 = pl.Float64
    INT8 = pl.Int8
    INT16 = pl.Int16
    INT32 = pl.Int32
    INT64 = pl.Int64
    INT128 = pl.Int128
    UINT8 = pl.UInt8
    UINT16 = pl.UInt16
    UINT32 = pl.UInt32
    UINT64 = pl.UInt64


def validate_inp_out(input: Path, output: Path, force: bool) -> None:
    if output.is_file() and not force:
        overwrite_file = Confirm.ask(
            f"[blue bold]{output}[/blue bold] already exists. Do you want to overwrite it?"
        )
        if not overwrite_file:
            err_console.print("Won't overwrite")
            raise typer.Abort()

    if input == output and not force:
        err_console.print(
            "Cannot specify output to be identical to input. Use the --force/-F option to force this behavior"
        )
        raise typer.Abort()


def check_create_dir(output: Path) -> bool:
    if not output.parent.is_dir():
        create_dir = Confirm.ask(
            f"The specified output's parent directory [blue bold]{output.parent}[/blue bold] doesn't exist. Do you want to create it along with any missing parents?"
        )
        if not create_dir:
            return False

        return True

    return False


def all_cols_exist(df: pl.DataFrame, fill_cols: list[str]) -> bool:
    for col in fill_cols:
        if col not in df.columns:
            return False

    return True


def fill_flag_exists(df: pl.DataFrame, fill_col: str, fill_flag: str) -> bool:
    if df.select((pl.col(fill_col) == fill_flag).any()).item():
        return True

    return False


def parse_fill_range(fill_range: str) -> FillRange:
    fill_range_parsed = tuple(x.strip() for x in fill_range.split(","))
    if len(fill_range_parsed) != 2:
        raise typer.BadParameter(f"Invalid fill range: {fill_range}")

    if not fill_range_parsed[0].isdigit() or not fill_range_parsed[1].isdigit():
        raise typer.BadParameter(f"Invalid fill range: {fill_range}")

    fill_range_int = FillRange(int(fill_range_parsed[0]), int(fill_range_parsed[1]))
    if fill_range_int.lb > fill_range_int.ub:
        err_console.print(f"Invalid fill range given: {fill_range}")
        raise typer.Abort()

    return fill_range_int


@impute_app.command("file")
def impute_file(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Target CSV file",
        ),
    ],
    columns: Annotated[
        list[str],
        typer.Option(
            "--col",
            "-c",
            help="Name of column to impute. Specify this for each colum you wanted imputed.",
        ),
    ],
    fill_flag: Annotated[
        str,
        typer.Option(
            "--flag",
            "-f",
            help="Flag/marker to find and replace in the target column(s)",
        ),
    ],
    fill_range: Annotated[
        FillRange,
        typer.Option(
            "--range",
            "-r",
            metavar="TEXT",
            help='Closed, integer interval from which to sample random integer for imputation. Specify as comma-separated values. For example: "1,5" corresponds to the range [1, 5]',
            parser=parse_fill_range,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            # NOTE: if exists=False, file/directory doesn't need to exist;
            # if doesn't exist, other checks skipped
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=False,
            help="Path to save the imputed CSV file. If not specified, defaults to printing result to stdout.",
        ),
    ] = None,
    col_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Intended data type of the target column. Can be a Polars int or float type.",
            click_type=click.Choice(ColType._member_names_, case_sensitive=False),
        ),
    ] = ColType.INT64.name,
    seed: Annotated[
        int | None, typer.Option("--seed", "-s", help="Random seed for reproducibility")
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Whether to show additional imputation summary information",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-F",
            help="""
            Allow overwriting data even if (1) the specified output file already exists or
            (2) the path to the input file is identical to the path of the output file. Both
            checks will be ignored.
            """,
        ),
    ] = False,
) -> None:
    """
    Impute target column(s) in a CSV file. Will look for the specified flag and replace
    it with a random integer from the specified range. Optionally, save the result to a new CSV file.
    """
    create_dir = False
    if output is not None:
        validate_inp_out(input, output, force)
        create_dir = check_create_dir(output)

        if not output.parent.is_dir() and not create_dir:
            err_console.print("Won't create directories")
            raise typer.Abort()

    df = pl.read_csv(input, infer_schema_length=0)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Imputing...", total=None)

        t0 = time.perf_counter()
        df = impute.columns(
            df, columns, fill_flag, fill_range, ColType[col_type].value, seed
        )
        t1 = time.perf_counter()

        if output is not None:
            if create_dir:
                output.parent.mkdir(parents=True)

            df.write_csv(output, separator=",")

    if verbose:
        console.print(f"[bold]Time taken[/bold]: {(t1 - t0):0.3f}s", highlight=False)

    if output is None:
        print(df.head(10))


class FillCols(NamedTuple):
    numerator: str
    denominator: str


def parse_fill_cols(fill_cols: str) -> FillCols:
    fill_cols_parsed = tuple(x.strip() for x in fill_cols.split(","))
    if len(fill_cols_parsed) != 2:
        raise typer.BadParameter(f"Invalid fill cols: {fill_cols}")

    return FillCols(fill_cols_parsed[0], fill_cols_parsed[1])


def parse_sep_cols(sep_cols: str) -> list[str]:
    return [col.strip() for col in sep_cols.split(",")]


@impute_app.command("pair")
def impute_pair(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Target CSV file",
        ),
    ],
    numerator: Annotated[
        str, typer.Option("--numerator", "-n", help="Numerator in the pair imputation")
    ],
    denominator: Annotated[
        str,
        typer.Option("--denominator", "-d", help="Denominator in the pair imputation"),
    ],
    fill_flag: Annotated[
        str,
        typer.Option(
            "--flag", "-f", help="Flag/marker to find and replace in the target columns"
        ),
    ],
    fill_range: Annotated[
        FillRange,
        typer.Option(
            "--range",
            "-r",
            metavar="TEXT",
            help='Closed, integer interval from which to sample random integer for imputation. Specify as comma-separated values. For example: "1,5" corresponds to the range [1, 5]',
            parser=parse_fill_range,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=False,
            help="Path to save the imputed CSV file. If not specified, defaults to printing result to stdout.",
        ),
    ] = None,
    col_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Intended data type of target columns. Can be a Polars int or float type.",
            click_type=click.Choice(ColType._member_names_, case_sensitive=False),
        ),
    ] = ColType.INT64.name,
    seed: Annotated[
        int | None, typer.Option("--seed", "-s", help="Random seed for reproducibility")
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Whether to show additional imputation summary information",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-F",
            help="""
            Allow overwriting data even if (1) the specified output file already exists or
            (2) the path to the input file is identical to the path of the output file. Both
            checks will be ignored.
            """,
        ),
    ] = False,
    sep_denom: Annotated[
        Path | None,
        typer.Option(
            "--sep-denom",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="""
            Path to a separate CSV file in which to look for denominator column. Will perform an
            inner join between the input file and this file containing the denominator.
            """,
        ),
    ] = None,
    sep_cols: Annotated[
        list[str] | None,
        typer.Option(
            "--sep-col",
            help="Name of column on which to join the numerator and denominator data. Specify for each column to be used.",
        ),
    ] = None,
    sep_out: Annotated[
        Path | None,
        typer.Option(
            "--sep-out",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=False,
            help="Path to save imputed version of the separate denominator file",
        ),
    ] = None,
):
    # TODO: review
    """
    Impute a pair of columns in a CSV file. Will look for the
    flag in both of the specified columns and substitute with a random
    integer in the closed range. Note, the pair of columns
    is comprised of a numerator column and a denominator column such that
    the imputed values of the numerator column must not exceed the imputed
    values of the denominator column.

    Separate denominator data:

    If --sep-denom is provided, then that file will be used instead to source
    the denominator column. You must then also use --sep-cols to specify the
    columns to use for (inner) joining the numerator and denominator data. Note
    that a 1:1 relationship in the join is enforced. Optionally, use --sep-out
    to specify where to save the imputed version of the denominator data from
    --sep-denom.
    """
    create_dir = False
    if output is not None:
        validate_inp_out(input, output, force)
        create_dir = check_create_dir(output)

    create_sep_dir = False
    if sep_out is not None:
        create_sep_dir = check_create_dir(sep_out)

    if sep_denom is None and (sep_cols is not None or sep_out is not None):
        err_console.print("Must specify --sep-denom to use --sep-cols or --sep-out")
        raise typer.Abort()

    if sep_denom is not None and sep_cols is None:
        err_console.print("Must specify --sep-cols if using --sep-denom")
        raise typer.Abort()

    df = pl.read_csv(input, infer_schema_length=0)

    if sep_denom is None:
        if numerator not in df.columns or denominator not in df.columns:
            err_console.print("Invalid numerator or denominator column specified")
            raise typer.Abort()
    else:
        # NOTE: extract since it gives nested list; maybe some type coercion going on
        # sep_cols = sep_cols[0]
        df_denom = pl.read_csv(sep_denom, infer_schema_length=0)

        if numerator not in df.columns or denominator not in df_denom.columns:
            err_console.print("Invalid numerator or denominator column specified")
            raise typer.Abort()

        if sep_cols is not None and (
            not all_cols_exist(df, sep_cols) or not all_cols_exist(df_denom, sep_cols)
        ):
            err_console.print(
                "Some of the --sep-col columns are missing from the numerator or denominator data"
            )
            raise typer.Abort()

        if sep_out is not None:
            # TODO: needed?
            if not fill_flag_exists(df_denom, denominator, fill_flag):
                print(
                    f"""
                    The denominator file {sep_denom} doesn't contain any instancees of {fill_flag}
                    in {denominator}. Rerun the command without specifying --sep-out.
                    """
                )
                raise typer.Abort()

            if sep_out.is_file() and not force:
                overwrite_out = Confirm.ask(
                    f"[blue bold]{sep_out}[/blue bold] already exists. Do you want to overwrite it?"
                )
                if not overwrite_out:
                    print("Won't overwrite")
                    raise typer.Abort()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Imputing...", total=None)

        if sep_denom is None:
            t0 = time.perf_counter()
            df = impute.column_pair(
                df,
                numerator,
                denominator,
                fill_flag,
                fill_range,
                ColType[col_type].value,
                seed,
            )
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            try:
                df = df.join(
                    df_denom, on=sep_cols, how="inner", coalesce=True, validate="1:1"
                )
            except pl.exceptions.ComputeError:
                err_console.print(
                    "The join with --sep-denom failed because there is not a 1:1 relationship between the join columns specified via --sep-col."
                )
                raise typer.Abort()

            df = impute.column_pair(
                df,
                numerator,
                denominator,
                fill_flag,
                fill_range,
                ColType[col_type].value,
                seed,
            )
            t1 = time.perf_counter()

        if output is not None:
            if create_dir:
                output.parent.mkdir(parents=True)

            if sep_denom is not None:
                df.select(pl.col("*").exclude(denominator)).write_csv(
                    output, separator=","
                )

                if sep_out is not None:
                    if create_sep_dir:
                        sep_out.parent.mkdir(parents=True)

                    df.select(pl.col("*").exclude(numerator)).write_csv(
                        sep_out, separator=","
                    )
            else:
                df.write_csv(output, separator=",")

    if verbose:
        console.print(f"[bold]Time taken[/bold]: {(t1 - t0):0.3f}s", highlight=False)

    if output is None:
        print(
            df.filter(
                (pl.col(numerator) <= fill_range.ub)
                | (pl.col(denominator) <= fill_range.ub)
            ).head(10)
        )
