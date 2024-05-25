import time
from pathlib import Path

import numpy as np
import polars as pl
import typer
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from typing_extensions import Annotated

app = typer.Typer()
err_console = Console(stderr=True)


# TODO:
# abort vs. typer.Exit(code=1); I'm guessing latter is useful when the CLI
# is composed alongside other commands
# impute-pair should be a separate command b/c it has more delicate logic


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
        err_console.print(f"File {input_file} does not exist")
        # raise typer.Exit(code=1)
        raise typer.Abort()

    print(f"File: {input_file}")
    df = pl.read_csv(input_file, infer_schema_length=0)
    print(df.head(n_rows))


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
        err_console.print(f"File {input_file} does not exist")
        # raise typer.Exit(code=1)
        raise typer.Abort()

    output_file = Path(output)
    if output_file.is_file():
        overwrite_file = Confirm.ask(
            f"[blue bold]{output_file}[/blue bold] already exists. Do you want to overwrite it?"
        )
        if not overwrite_file:
            err_console.print("Won't overwrite")
            raise typer.Abort()

    # some try/catch handling in case problems reading the file?
    df = pl.read_csv(input_file, infer_schema_length=0)
    # TODO: consider validation for whether the col is an int col?
    if fill_col not in df.columns:
        err_console.print(f"Column {fill_col} cannot be found in {input_file}")
        raise typer.Abort()

    fill_range_parsed = tuple(fill_range.split("-", maxsplit=1))
    if not fill_range_parsed[0].isdigit() or not fill_range_parsed[1].isdigit():
        err_console.print(f"Invalid range given for --fill-range: {fill_range_parsed}")
        # raise typer.Exit(code=1)
        raise typer.Abort()
    fill_range_parsed = tuple(int(x) for x in fill_range_parsed)

    rng = np.random.default_rng(seed)
    # imp_size only used for reporting
    imp_size = len(df.filter(pl.col(fill_col) == fill_flag))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Imputing...", total=None)
        df = df.with_columns(
            pl.when(pl.col(fill_col) == fill_flag)
            .then(
                pl.lit(
                    # must specify size to be height of df despite not filling every row
                    # thus, we get "new" rand int per row
                    rng.integers(
                        fill_range_parsed[0], fill_range_parsed[1] + 1, size=df.height
                    )
                ).cast(pl.String)
            )
            .otherwise(pl.col(fill_col))
            .alias(fill_col)
        )
        df.write_csv(output_file)

    print("[green]Finished imputing[/green]...")
    if verbose:
        print(
            f"Imputed [blue]{imp_size:_}[/blue] values -> [blue]{(imp_size / df.height):0.2f}[/blue] of rows (n = {df.height:_}) affected"
        )
        print(df.head())


if __name__ == "__main__":
    app()
