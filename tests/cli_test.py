import shutil
import textwrap
from importlib.metadata import version
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from csv_helper.cli import app

runner = CliRunner()


@pytest.fixture
def test_data(tmp_path: Path) -> Path:
    """
    Fixture that moves test CSV data to new dir for testing and
    returns the file's path
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy("./tests/data/test_impute_data.csv", data_dir)

    return data_dir / "test_impute_data.csv"


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """
    Fixture that moves test dir of CSV data to new dir for
    testing and returns the dir's path
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copytree("./tests/data/test_dir", data_dir / "test_dir")

    return data_dir / "test_dir"


@pytest.fixture
def test_data_sep(tmp_path: Path) -> Path:
    """
    Fixture that moves test dir of pair CSV data to new dir for
    testing and returns the dir's path
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copytree("./tests/data/test_pair_sep", data_dir / "test_pair_sep")

    return data_dir / "test_pair_sep"


def test_print_version():
    result = runner.invoke(app, ["--version"])
    ver = version("csv_helper")

    assert result.stdout.replace("\n", "") == f"csv-helper version {ver}"


def test_show(test_data):
    result = runner.invoke(app, ["show", str(test_data), "-n", "15"])
    assert result.exit_code == 0

    out = textwrap.dedent(
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

    assert result.stdout == out


def test_show_not_file(tmp_path):
    dir = tmp_path / "data"
    result = runner.invoke(app, ["show", str(dir), "-n", "15"])
    assert result.exit_code == 2


def test_check(test_data):
    result = runner.invoke(app, ["check", str(test_data), "-c", "cases", "-f", "<=5"])
    assert result.exit_code == 0

    out = textwrap.dedent(
        """\
        shape: (1, 3)
        ┌────────┬───────┬───────┐
        │ column ┆ count ┆ prop  │
        │ ---    ┆ ---   ┆ ---   │
        │ str    ┆ u32   ┆ f64   │
        ╞════════╪═══════╪═══════╡
        │ cases  ┆ 308   ┆ 0.616 │
        └────────┴───────┴───────┘
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
            "-o",
            str(out_file),
            "-c",
            "cases",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file() is True

    df_in = pl.read_csv(test_data, infer_schema_length=0)
    df_out = pl.read_csv(out_file, infer_schema_length=0)
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


def test_impute_file_repro(tmp_path, test_data):
    out_file_1 = tmp_path / "test_impute_output_1.csv"
    out_file_2 = tmp_path / "test_impute_output_2.csv"
    fill_range = (1, 5)

    result_1 = runner.invoke(
        app,
        [
            "impute",
            "file",
            str(test_data),
            "-o",
            str(out_file_1),
            "-c",
            "cases",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-s",
            "88",
        ],
    )
    assert result_1.exit_code == 0
    assert out_file_1.is_file() is True

    result_2 = runner.invoke(
        app,
        [
            "impute",
            "file",
            str(test_data),
            "-o",
            str(out_file_2),
            "-c",
            "cases",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-s",
            "88",
        ],
    )
    assert result_2.exit_code == 0
    assert out_file_2.is_file() is True

    df_1 = pl.read_csv(out_file_1, infer_schema_length=0)
    assert df_1.select((pl.col("cases") == f"<={fill_range[1]}").any()).item() is False

    df_2 = pl.read_csv(out_file_2, infer_schema_length=0)
    assert df_2.select((pl.col("cases") == f"<={fill_range[1]}").any()).item() is False

    assert_frame_equal(df_1, df_2)


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
            "-o",
            str(out_file),
            "-c",
            "cases",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
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
            "-o",
            str(out_file),
            "-c",
            "cases",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
        ],
        input="y\n",
    )
    assert result.exit_code == 0


def test_impute_pair(tmp_path, test_data):
    out_file = tmp_path / "test_impute_pair_output.csv"
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute",
            "pair",
            str(test_data),
            "-n",
            "cases",
            "-d",
            "all_cause",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-o",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file() is True

    df_in = pl.read_csv(test_data, infer_schema_length=0)
    df_out = pl.read_csv(out_file, infer_schema_length=0)
    assert df_in.shape == df_out.shape

    df = df_in.join(
        df_out,
        on=["county", "year_week"],
        how="inner",
        suffix="_imputed",
        validate="1:1",
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
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute",
            "pair",
            str(num_file),
            "-n",
            "cases",
            "-d",
            "all_cause",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-o",
            str(out_file),
            "--sep-denom",
            str(denom_file),
            "--sep-col",
            "county",
            "--sep-col",
            "year_week",
        ],
    )
    assert result.exit_code == 0
    assert out_file.is_file() is True

    df_num = pl.read_csv(num_file, infer_schema_length=0)
    df_out = pl.read_csv(out_file, infer_schema_length=0)

    assert df_num.shape == df_out.shape
    assert (
        df_out.select((pl.col("cases") == f"<={fill_range[1]}").any()).item() is False
    )
    assert df_num.null_count().equals(df_out.null_count()) is True

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
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute",
            "pair",
            str(num_file),
            "-n",
            "cases",
            "-d",
            "all_cause",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-o",
            str(out_file),
            "--sep-denom",
            str(denom_file),
            "--sep-col",
            "county",
            "--sep-col",
            "year_week",
        ],
    )
    assert result.exit_code == 1


def test_impute_pair_sep_output(tmp_path, test_data_sep):
    num_file = test_data_sep / "test_impute_numerator_only_data.csv"
    out_file = tmp_path / "test_impute_sep_files_numerator_output.csv"
    denom_file = test_data_sep / "test_impute_denom_only_data.csv"
    sep_out = tmp_path / "test_impute_sep_files_denom_output.csv"
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute",
            "pair",
            str(num_file),
            "-n",
            "cases",
            "-d",
            "all_cause",
            "-f",
            f"<={fill_range[1]}",
            "-r",
            f"{fill_range[0]},{fill_range[1]}",
            "-o",
            str(out_file),
            "--sep-denom",
            str(denom_file),
            "--sep-col",
            "county",
            "--sep-col",
            "year_week",
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
    assert (
        df_out.select((pl.col("cases") == f"<={fill_range[1]}").any()).item() is False
    )
    assert df_num.null_count().equals(df_out.null_count()) is True

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
