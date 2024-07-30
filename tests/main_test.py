import shutil
from pathlib import Path, PureWindowsPath
from textwrap import dedent

import polars as pl
import pytest
from csv_helper.main import app
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def test_data(tmp_path) -> Path:
    """
    Fixture that moves test CSV data to new dir for testing and
    returns the file's path
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy("./tests/data/test_impute_data.csv", data_dir)
    return data_dir / "test_impute_data.csv"


def test_preview(tmp_path, test_data):
    result = runner.invoke(app, ["preview", str(test_data), "-n", "15"])
    assert result.exit_code == 0

    # NOTE: second is Windows file path
    assert (
        f"File: {test_data}\nshape: (15, 4)" in result.stdout.strip("\n")
        # or "File: tests\\data\\test_impute_data.csv\nshape: (15, 4)" in result.stdout
        or f"File: {PureWindowsPath(test_data)}\nshape: (15, 4)"
        in result.stdout.strip("\n")
    )

    out = dedent(
        """\
        shape: (15, 4)
        ┌────────┬──────┬───────┬───────────┐
        │ county ┆ year ┆ cases ┆ all_cause │
        │ ---    ┆ ---  ┆ ---   ┆ ---       │
        │ str    ┆ str  ┆ str   ┆ str       │
        ╞════════╪══════╪═══════╪═══════════╡
        │ 01001  ┆ 2016 ┆ 8     ┆ 20        │
        │ 01002  ┆ 2018 ┆ 10    ┆ 15        │
        │ 01003  ┆ 2018 ┆ <=5   ┆ <=5       │
        │ 01005  ┆ 2019 ┆ 7     ┆ 8         │
        │ 01001  ┆ 2019 ┆ 6     ┆ 10        │
        │ …      ┆ …    ┆ …     ┆ …         │
        │ 01003  ┆ 2020 ┆ <=5   ┆ 20        │
        │ 01295  ┆ 2021 ┆ 10    ┆ 13        │
        │ 36081  ┆ 2022 ┆ 11    ┆ 15        │
        │ 01003  ┆ 2020 ┆ <=5   ┆ <=5       │
        │ 01295  ┆ 2021 ┆ 10    ┆ 12        │
        └────────┴──────┴───────┴───────────┘
        """
    )
    assert out in result.stdout


def test_preview_not_file():
    result = runner.invoke(app, ["preview", "./tests/data/test_dir", "-n", "15"])
    assert result.exit_code == 2


def test_check():
    result = runner.invoke(
        app, ["check", "./tests/data/test_impute_data.csv", "-c", "cases", "-f", "<=5"]
    )
    assert result.exit_code == 0

    out = dedent(
        """\
        Found 33 occurrences of '<=5' in 'cases' -> 0.38 of rows (n = 86)
        shape: (5, 4)
        ┌────────┬──────┬───────┬───────────┐
        │ county ┆ year ┆ cases ┆ all_cause │
        │ ---    ┆ ---  ┆ ---   ┆ ---       │
        │ str    ┆ str  ┆ str   ┆ str       │
        ╞════════╪══════╪═══════╪═══════════╡
        │ 01003  ┆ 2018 ┆ <=5   ┆ <=5       │
        │ 01001  ┆ 2020 ┆ <=5   ┆ <=5       │
        │ 01003  ┆ 2020 ┆ <=5   ┆ <=5       │
        │ 01001  ┆ 2020 ┆ <=5   ┆ 100       │
        │ 01003  ┆ 2020 ┆ <=5   ┆ 20        │
        └────────┴──────┴───────┴───────────┘
        """
    )
    assert result.stdout == out


def test_check_not_file():
    result = runner.invoke(
        app, ["check", "./tests/data/test_dir", "-c", "cases", "-f", "<=5"]
    )
    assert result.exit_code == 2


def test_impute(tmp_path):
    inp_file = Path("./tests/data/test_impute_data.csv")
    tmp_dir = Path(tmp_path)
    out_file = tmp_dir / "test_impute_output.csv"
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute",
            str(inp_file),
            str(out_file),
            "-c",
            "cases",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-s",
            "8",
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file() is True

    df_in = (
        pl.read_csv(inp_file, infer_schema_length=0)
        .with_row_index(name="id")
        .select(["id", "cases"])
    )
    df_out = (
        pl.read_csv(out_file, infer_schema_length=0)
        .with_row_index(name="id")
        .select(["id", "cases"])
    )
    df = df_in.join(df_out, on="id", how="inner", suffix="_imputed").filter(
        pl.col("cases") == f"<={fill_range[1]}"
    )
    assert (
        df.select("cases_imputed").to_series().str.contains(f"<={fill_range[1]}").all()
        is False
    )
    assert (
        df.select("cases_imputed")
        .cast(pl.Int64)
        .filter(
            (pl.col("cases_imputed") < fill_range[0])
            | (pl.col("cases_imputed") > fill_range[1])
        )
        .height
        == 0
    )


def test_impute_output_exists(tmp_path):
    inp_file = Path("./tests/data/test_impute_data.csv")
    tmp_dir = Path(tmp_path)
    out_file = tmp_dir / "output_that_exists.csv"
    fill_range = (1, 5)

    assert out_file.is_file() is False
    out_file.touch()
    assert out_file.is_file() is True

    result = runner.invoke(
        app,
        [
            "impute",
            str(inp_file),
            str(out_file),
            "-c",
            "cases",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-s",
            "8",
        ],
    )
    assert result.exit_code == 1


def test_impute_overwrite(tmp_path):
    inp_file = Path("./tests/data/test_impute_data.csv")
    tmp_dir = Path(tmp_path)
    out_file = tmp_dir / "output_that_exists.csv"
    fill_range = (1, 5)

    assert out_file.is_file() is False
    out_file.touch()
    assert out_file.is_file() is True

    result = runner.invoke(
        app,
        [
            "impute",
            str(inp_file),
            str(out_file),
            "-c",
            "cases",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-s",
            "8",
        ],
        input="y\n",
    )
    assert result.exit_code == 0


def test_impute_dir(tmp_path):
    inp_dir = Path("./tests/data/test_dir")
    out_dir = Path(tmp_path) / "test_impute_output_dir"
    out_dir.mkdir()
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute-dir",
            str(inp_dir),
            str(out_dir),
            "-c",
            "cases",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-s",
            "8",
        ],
    )
    assert result.exit_code == 0
    assert out_dir.is_dir()
    for i in range(5):
        f = out_dir / f"test_impute_data_{i}.csv"
        assert f.is_file() is True

    for input, output in zip(inp_dir.iterdir(), out_dir.iterdir()):
        df_in = (
            pl.read_csv(input, infer_schema_length=0)
            .with_row_index(name="id")
            .select(["id", "cases", "all_cause"])
        )
        df_out = (
            pl.read_csv(output, infer_schema_length=0)
            .with_row_index(name="id")
            .select(["id", "cases", "all_cause"])
        )
        df = df_in.join(df_out, on="id", how="inner", suffix="_imputed").filter(
            (pl.col("cases") == f"<={fill_range[1]}")
            | (pl.col("all_cause") == f"<={fill_range[1]}")
        )
        assert (
            df.select("cases_imputed")
            .to_series()
            .str.contains(f"<={fill_range[1]}")
            .all()
            is False
        )
        assert (
            df.select("cases_imputed")
            .cast(pl.Int64)
            .filter(
                (pl.col("cases_imputed") < fill_range[0])
                | (pl.col("cases_imputed") > fill_range[1])
            )
            .height
            == 0
        )


def test_impute_dir_force(tmp_path):
    inp_dir = Path("./tests/data/test_dir")
    out_dir = Path(tmp_path) / "test_impute_output_dir"
    out_dir.mkdir()
    fill_range = (1, 5)

    inp_files = inp_dir.glob("*.csv")
    for file in inp_files:
        out_file = out_dir / file.name
        out_file.touch()

    result = runner.invoke(
        app,
        [
            "impute-dir",
            str(inp_dir),
            str(out_dir),
            "-c",
            "cases",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-s",
            "8",
            "--force",
        ],
    )
    assert result.exit_code == 0
    assert out_dir.is_dir()
    for i in range(5):
        f = out_dir / f"test_impute_data_{i}.csv"
        assert f.is_file() is True

    for input, output in zip(inp_dir.iterdir(), out_dir.iterdir()):
        df_in = (
            pl.read_csv(input, infer_schema_length=0)
            .with_row_index(name="id")
            .select(["id", "cases", "all_cause"])
        )
        df_out = (
            pl.read_csv(output, infer_schema_length=0)
            .with_row_index(name="id")
            .select(["id", "cases", "all_cause"])
        )
        df = df_in.join(df_out, on="id", how="inner", suffix="_imputed").filter(
            (pl.col("cases") == f"<={fill_range[1]}")
            | (pl.col("all_cause") == f"<={fill_range[1]}")
        )
        assert (
            df.select("cases_imputed")
            .to_series()
            .str.contains(f"<={fill_range[1]}")
            .all()
            is False
        )
        assert (
            df.select("cases_imputed")
            .cast(pl.Int64)
            .filter(
                (pl.col("cases_imputed") < fill_range[0])
                | (pl.col("cases_imputed") > fill_range[1])
            )
            .height
            == 0
        )


def test_impute_dir_suffix(tmp_path):
    inp_dir = Path("./tests/data/test_dir")
    out_dir = Path(tmp_path) / "test_impute_output_dir"
    out_dir.mkdir()
    fill_range = (1, 5)
    suffix = "imputed"

    result = runner.invoke(
        app,
        [
            "impute-dir",
            str(inp_dir),
            str(out_dir),
            "-c",
            "cases",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-s",
            "8",
            "-x",
            suffix,
        ],
    )
    assert result.exit_code == 0
    assert out_dir.is_dir() is True
    for i in range(5):
        f = out_dir / f"test_impute_data_{i}_{suffix}.csv"
        assert f.is_file() is True

    for input, output in zip(inp_dir.iterdir(), out_dir.iterdir()):
        df_in = (
            pl.read_csv(input, infer_schema_length=0)
            .with_row_index(name="id")
            .select(["id", "cases", "all_cause"])
        )
        df_out = (
            pl.read_csv(output, infer_schema_length=0)
            .with_row_index(name="id")
            .select(["id", "cases", "all_cause"])
        )
        df = df_in.join(df_out, on="id", how="inner", suffix="_imputed").filter(
            (pl.col("cases") == f"<={fill_range[1]}")
            | (pl.col("all_cause") == f"<={fill_range[1]}")
        )
        assert (
            df.select("cases_imputed")
            .to_series()
            .str.contains(f"<={fill_range[1]}")
            .all()
            is False
        )
        assert (
            df.select("cases_imputed")
            .cast(pl.Int64)
            .filter(
                (pl.col("cases_imputed") < fill_range[0])
                | (pl.col("cases_imputed") > fill_range[1])
            )
            .height
            == 0
        )


def test_impute_pair(tmp_path):
    inp_file = Path("./tests/data/test_impute_data.csv")
    tmp_dir = Path(tmp_path) / "csv-helper"
    tmp_dir.mkdir()
    out_file = tmp_dir / "test_impute_pair_output.csv"
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute-pair",
            str(inp_file),
            str(out_file),
            "-c",
            "cases,all_cause",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-s",
            "8",
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file() is True

    df_in = (
        pl.read_csv(inp_file, infer_schema_length=0)
        .with_row_index(name="id")
        .select(["id", "cases", "all_cause"])
    )
    df_out = (
        pl.read_csv(out_file, infer_schema_length=0)
        .with_row_index(name="id")
        .select(["id", "cases", "all_cause"])
    )
    df = df_in.join(df_out, on="id", how="inner", suffix="_imputed").filter(
        (pl.col("cases") == f"<={fill_range[1]}")
        | (pl.col("all_cause") == f"<={fill_range[1]}")
    )
    assert (
        df.select("cases_imputed").to_series().str.contains(f"<={fill_range[1]}").all()
        is False
    )
    assert (
        df.select("all_cause_imputed")
        .to_series()
        .str.contains(f"<={fill_range[1]}")
        .all()
        is False
    )
    assert (
        df.select(pl.col("cases"), pl.col("cases_imputed").cast(pl.Int64))
        .filter(
            (
                (pl.col("cases") == f"<={fill_range[1]}")
                & (pl.col("cases_imputed") < fill_range[0])
            )
            | (
                (pl.col("cases") == f"<={fill_range[1]}")
                & (pl.col("cases_imputed") > fill_range[1])
            )
        )
        .height
        == 0
    )
    assert (
        df.select(["cases_imputed", "all_cause_imputed"])
        .cast(pl.Int64)
        .filter(pl.col("cases_imputed") > pl.col("all_cause_imputed"))
    ).height == 0
