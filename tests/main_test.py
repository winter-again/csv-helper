import shutil
from pathlib import Path, PureWindowsPath
from sys import platform
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


@pytest.fixture
def test_data_dir(tmp_path) -> Path:
    """
    Fixture that moves test dir of CSV data to new dir for
    testing and returns the dir's path
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copytree("./tests/data/test_dir", data_dir / "test_dir")
    return data_dir / "test_dir"


@pytest.fixture
def test_data_sep(tmp_path) -> Path:
    """
    Fixture that moves test dir of pair CSV data to new dir for
    testing and returns the dir's path
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copytree("./tests/data/test_pair_sep", data_dir / "test_pair_sep")
    return data_dir / "test_pair_sep"


# NOTE: can also access funcs in csv_helper.main directly:
# from csv_helper.main import preview
# preview("./tests/data/test_impute_data.csv", 10)


def test_preview(test_data):
    result = runner.invoke(app, ["preview", str(test_data), "-n", "15"])
    assert result.exit_code == 0

    if platform == "linux" or platform == "darwin":
        msg = f"File: {test_data}"
    elif platform == "win32":
        msg = f"File: {PureWindowsPath(test_data)}"

    # NOTE: stripping newlines and then slicing; for some reason on macos and windows
    # the stdout has newlines inserted
    assert result.stdout.replace("\n", "")[: len(msg)] == msg

    out = dedent(
        """\
        shape: (15, 4)
        ┌────────┬───────────┬───────┬───────────┐
        │ county ┆ year_week ┆ cases ┆ all_cause │
        │ ---    ┆ ---       ┆ ---   ┆ ---       │
        │ str    ┆ str       ┆ str   ┆ str       │
        ╞════════╪═══════════╪═══════╪═══════════╡
        │ 55107  ┆ 2020-05   ┆ <=5   ┆ 334       │
        │ 28101  ┆ 2021-20   ┆ <=5   ┆ <=5       │
        │ 26099  ┆ 2023-34   ┆ 11    ┆ 31416     │
        │ 35043  ┆ 2022-24   ┆ <=5   ┆ 5862      │
        │ 28077  ┆ 2022-41   ┆ 8     ┆ 703       │
        │ …      ┆ …         ┆ …     ┆ …         │
        │ 26093  ┆ 2020-42   ┆ <=5   ┆ 7606      │
        │ 17197  ┆ 2017-20   ┆ <=5   ┆ 25940     │
        │ 47167  ┆ 2021-19   ┆ <=5   ┆ <=5       │
        │ 27091  ┆ 2019-12   ┆ 8     ┆ 469       │
        │ 51085  ┆ 2018-09   ┆ 26    ┆ 2348      │
        └────────┴───────────┴───────┴───────────┘
        """
    )
    assert out in result.stdout


def test_preview_not_file(tmp_path):
    dir = tmp_path / "data"
    result = runner.invoke(app, ["preview", str(dir), "-n", "15"])
    assert result.exit_code == 2


def test_check(test_data):
    result = runner.invoke(app, ["check", str(test_data), "-c", "cases", "-f", "<=5"])
    assert result.exit_code == 0

    out = dedent(
        """\
        Found 308 occurrences of '<=5' in 'cases' -> 0.62 of rows (n = 500)
        shape: (5, 4)
        ┌────────┬───────────┬───────┬───────────┐
        │ county ┆ year_week ┆ cases ┆ all_cause │
        │ ---    ┆ ---       ┆ ---   ┆ ---       │
        │ str    ┆ str       ┆ str   ┆ str       │
        ╞════════╪═══════════╪═══════╪═══════════╡
        │ 55107  ┆ 2020-05   ┆ <=5   ┆ 334       │
        │ 28101  ┆ 2021-20   ┆ <=5   ┆ <=5       │
        │ 35043  ┆ 2022-24   ┆ <=5   ┆ 5862      │
        │ 28043  ┆ 2023-09   ┆ <=5   ┆ 811       │
        │ 26093  ┆ 2020-42   ┆ <=5   ┆ 7606      │
        └────────┴───────────┴───────┴───────────┘
        """
    )
    assert result.stdout == out


def test_check_not_file(tmp_path):
    dir = tmp_path / "data"
    result = runner.invoke(app, ["check", str(dir), "-c", "cases", "-f", "<=5"])
    assert result.exit_code == 2


def test_impute_file(tmp_path, test_data):
    out_file = tmp_path / "test_impute_output.csv"
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute",
            "file",
            str(test_data),
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
        pl.read_csv(test_data, infer_schema_length=0)
        # .with_row_index(name="id")
        # .select("id", "cases")
    )
    df_out = (
        pl.read_csv(out_file, infer_schema_length=0)
        # .with_row_index(name="id")
        # .select("id", "cases")
    )
    assert df_in.shape == df_out.shape

    df = df_in.join(
        df_out, on=["county", "year_week"], how="inner", suffix="_imputed"
    ).filter(pl.col("cases") == f"<={fill_range[1]}")
    assert (
        df.select((pl.col("cases_imputed") == f"<={fill_range[1]}").any()).item()
        is False
    )
    assert (
        df.select("cases_imputed")
        .cast(pl.Int64)
        .select(pl.col("cases_imputed").is_between(fill_range[0], fill_range[1]).all())
        .item()
        is True
    )


def test_impute_file_output_exists(tmp_path, test_data):
    out_file = tmp_path / "output_that_exists.csv"
    fill_range = (1, 5)

    assert out_file.is_file() is False
    out_file.touch()
    assert out_file.is_file() is True

    result = runner.invoke(
        app,
        [
            "impute",
            "file",
            str(test_data),
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


def test_impute_file_overwrite(tmp_path, test_data):
    out_file = tmp_path / "output_that_exists.csv"
    fill_range = (1, 5)

    assert out_file.is_file() is False
    out_file.touch()
    assert out_file.is_file() is True

    result = runner.invoke(
        app,
        [
            "impute",
            "file",
            str(test_data),
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


def test_impute_dir(tmp_path, test_data_dir):
    out_dir = Path(tmp_path) / "test_impute_dir_output"
    out_dir.mkdir()
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute",
            "dir",
            str(test_data_dir),
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

    for input, output in zip(test_data_dir.iterdir(), out_dir.iterdir()):
        df_in = (
            pl.read_csv(input, infer_schema_length=0)
            # .with_row_index(name="id")
            # .select("id", "cases", "all_cause")
        )
        df_out = (
            pl.read_csv(output, infer_schema_length=0)
            # .with_row_index(name="id")
            # .select("id", "cases", "all_cause")
        )
        assert df_in.shape == df_out.shape

        df = df_in.join(
            df_out, on=["county", "year_week"], how="inner", suffix="_imputed"
        ).filter(
            (pl.col("cases") == f"<={fill_range[1]}")
            | (pl.col("all_cause") == f"<={fill_range[1]}")
        )
        assert (
            df.select((pl.col("cases_imputed") == f"<={fill_range[1]}").any()).item()
            is False
        )
        assert (
            df.select("cases_imputed")
            .cast(pl.Int64)
            .select(
                pl.col("cases_imputed").is_between(fill_range[0], fill_range[1]).all()
            )
            .item()
            is True
        )


def test_impute_dir_force(tmp_path, test_data_dir):
    out_dir = Path(tmp_path) / "test_impute_dir_output"
    out_dir.mkdir()
    fill_range = (1, 5)

    inp_files = test_data_dir.glob("*.csv")
    for file in inp_files:
        out_file = out_dir / file.name
        out_file.touch()

    result = runner.invoke(
        app,
        [
            "impute",
            "dir",
            str(test_data_dir),
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

    for input, output in zip(test_data_dir.iterdir(), out_dir.iterdir()):
        df_in = (
            pl.read_csv(input, infer_schema_length=0)
            # .with_row_index(name="id")
            # .select("id", "cases", "all_cause")
        )
        df_out = (
            pl.read_csv(output, infer_schema_length=0)
            # .with_row_index(name="id")
            # .select("id", "cases", "all_cause")
        )
        assert df_in.shape == df_out.shape

        df = df_in.join(
            df_out, on=["county", "year_week"], how="inner", suffix="_imputed"
        ).filter(
            (pl.col("cases") == f"<={fill_range[1]}")
            | (pl.col("all_cause") == f"<={fill_range[1]}")
        )
        assert (
            df.select((pl.col("cases_imputed") == f"<={fill_range[1]}").any()).item()
            is False
        )
        assert (
            df.select("cases_imputed")
            .cast(pl.Int64)
            .select(
                pl.col("cases_imputed").is_between(fill_range[0], fill_range[1]).all()
            )
            .item()
            is True
        )


def test_impute_dir_suffix(tmp_path, test_data_dir):
    out_dir = Path(tmp_path) / "test_impute_dir_output"
    out_dir.mkdir()
    fill_range = (1, 5)
    suffix = "imputed"

    result = runner.invoke(
        app,
        [
            "impute",
            "dir",
            str(test_data_dir),
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

    for input, output in zip(test_data_dir.iterdir(), out_dir.iterdir()):
        df_in = (
            pl.read_csv(input, infer_schema_length=0)
            # .with_row_index(name="id")
            # .select("id", "cases", "all_cause")
        )
        df_out = (
            pl.read_csv(output, infer_schema_length=0)
            # .with_row_index(name="id")
            # .select("id", "cases", "all_cause")
        )
        assert df_in.shape == df_out.shape

        df = df_in.join(
            df_out, on=["county", "year_week"], how="inner", suffix="_imputed"
        ).filter(
            (pl.col("cases") == f"<={fill_range[1]}")
            | (pl.col("all_cause") == f"<={fill_range[1]}")
        )
        assert (
            df.select((pl.col("cases_imputed") == f"<={fill_range[1]}").any()).item()
            is False
        )
        assert (
            df.select("cases_imputed")
            .cast(pl.Int64)
            .select(
                pl.col("cases_imputed").is_between(fill_range[0], fill_range[1]).all()
            )
            .item()
            is True
        )


def test_impute_pair(tmp_path, test_data):
    out_file = tmp_path / "test_impute_pair_output.csv"
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute",
            "pair",
            str(test_data),
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
        pl.read_csv(test_data, infer_schema_length=0)
        # .with_row_index(name="id")
        # .select("id", "cases", "all_cause")
    )
    df_out = (
        pl.read_csv(out_file, infer_schema_length=0)
        # .with_row_index(name="id")
        # .select("id", "cases", "all_cause")
    )
    assert df_in.shape == df_out.shape

    df = df_in.join(
        df_out, on=["county", "year_week"], how="inner", suffix="_imputed"
    ).filter(
        (pl.col("cases") == f"<={fill_range[1]}")
        | (pl.col("all_cause") == f"<={fill_range[1]}")
    )
    assert (
        df.select((pl.col("cases_imputed") == f"<={fill_range[1]}").any()).item()
        is False
    )
    assert (
        df.select((pl.col("all_cause_imputed") == f"<={fill_range[1]}").any()).item()
        is False
    )
    assert (
        df.select(pl.col("cases"), pl.col("cases_imputed").cast(pl.Int64))
        .filter(pl.col("cases") == f"<={fill_range[1]}")
        .select(pl.col("cases_imputed").is_between(fill_range[0], fill_range[1]).all())
        .item()
        is True
    )
    assert (
        df.select("all_cause", pl.col("all_cause_imputed").cast(pl.Int64))
        .filter(pl.col("all_cause") == f"<={fill_range[1]}")
        .select(
            pl.col("all_cause_imputed").is_between(fill_range[0], fill_range[1]).all()
        )
        .item()
        is True
    )
    assert (
        df.select("cases_imputed", "all_cause_imputed")
        .cast(pl.Int64)
        .select((pl.col("cases_imputed") > pl.col("all_cause_imputed")).any())
    ).item() is False


def test_impute_pair_sep(tmp_path, test_data_sep):
    num_file = test_data_sep / "test_impute_numerator_only_data.csv"
    out_file = tmp_path / "numerator_output.csv"
    denom_file = test_data_sep / "test_impute_denom_only_data.csv"
    sep_cols = "county,year_week"
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute",
            "pair",
            str(num_file),
            str(out_file),
            "-c",
            "cases,all_cause",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-s",
            "8",
            "--sep-denom",
            str(denom_file),
            "--sep-cols",
            sep_cols,
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file() is True

    df_num = pl.read_csv(num_file, infer_schema_length=0)
    df_out = pl.read_csv(out_file, infer_schema_length=0)
    assert df_num.shape == df_out.shape

    # NOTE: can't test if all imputed cases <= all-cause since we don't save imputed all-cause in this case
    df = df_num.join(
        df_out, on=["county", "year_week"], how="inner", suffix="_imputed"
    ).filter(pl.col("cases") == f"<={fill_range[1]}")

    assert (
        df.select((pl.col("cases_imputed") == f"<={fill_range[1]}").any()).item()
        is False
    )
    assert (
        df.select(pl.col("cases_imputed").cast(pl.Int64))
        .select(pl.col("cases_imputed").is_between(fill_range[0], fill_range[1]).all())
        .item()
        is True
    )


def test_impute_pair_join_fails(tmp_path, test_data_sep):
    num_file = test_data_sep / "test_impute_numerator_only_data.csv"
    out_file = tmp_path / "numerator_output.csv"
    denom_file = test_data_sep / "test_impute_denom_only_join_fails.csv"
    sep_cols = "county,year_week"
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute",
            "pair",
            str(num_file),
            str(out_file),
            "-c",
            "cases,all_cause",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-s",
            "8",
            "--sep-denom",
            str(denom_file),
            "--sep-cols",
            sep_cols,
        ],
    )
    assert result.exit_code == 1


def test_impute_pair_sep_output(tmp_path, test_data_sep):
    num_file = test_data_sep / "test_impute_numerator_only_data.csv"
    out_file = tmp_path / "test_impute_sep_files_numerator_output.csv"
    denom_file = test_data_sep / "test_impute_denom_only_data.csv"
    sep_cols = "county,year_week"
    sep_out = tmp_path / "test_impute_sep_files_denom_output.csv"
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute",
            "pair",
            str(num_file),
            str(out_file),
            "-c",
            "cases,all_cause",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-s",
            "8",
            "--sep-denom",
            str(denom_file),
            "--sep-cols",
            sep_cols,
            "--sep-out",
            str(sep_out),
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file() is True

    df_num = pl.read_csv(num_file, infer_schema_length=0)
    df_denom = pl.read_csv(denom_file, infer_schema_length=0)
    assert df_num.shape == df_denom.shape

    df_out = pl.read_csv(out_file, infer_schema_length=0)
    assert df_num.shape == df_out.shape

    df_sep_out = pl.read_csv(sep_out, infer_schema_length=0)
    assert df_denom.shape == df_sep_out.shape

    df = (
        df_num.join(df_denom, on=["county", "year_week"], how="inner", coalesce=True)
        .join(df_out, on=["county", "year_week"], how="inner", suffix="_imputed")
        .join(df_sep_out, on=["county", "year_week"], how="inner", suffix="_imputed")
        .filter(
            (pl.col("cases") == f"<={fill_range[1]}")
            | (pl.col("all_cause") == f"<={fill_range[1]}")
        )
    )
    print(df)
    assert (
        df.select((pl.col("cases_imputed") == f"<={fill_range[1]}").any()).item()
        is False
    )
    assert (
        df.select((pl.col("all_cause_imputed") == f"<={fill_range[1]}").any()).item()
        is False
    )
    assert (
        df.select("cases", pl.col("cases_imputed").cast(pl.Int64))
        .select(pl.col("cases_imputed").is_between(fill_range[0], fill_range[1]).all())
        .item()
        is True
    )
    assert (
        df.select("all_cause", pl.col("all_cause_imputed").cast(pl.Int64))
        .filter(pl.col("all_cause") == f"<={fill_range[1]}")
        .select(
            pl.col("all_cause_imputed").is_between(fill_range[0], fill_range[1]).all()
        )
        .item()
        is True
    )
    assert (
        df.select("cases_imputed", "all_cause_imputed")
        .cast(pl.Int64)
        .select((pl.col("cases_imputed") > pl.col("all_cause_imputed")).any())
    ).item() is False
