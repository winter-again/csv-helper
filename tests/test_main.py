from pathlib import Path

import polars as pl
from csv_helper.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_preview_cmd():
    result = runner.invoke(
        app, ["preview", "./tests/data/test_impute_data.csv", "-n", "15"]
    )

    assert Path("./tests/data/test_impute_data.csv").is_file()
    assert result.exit_code == 0
    assert "File: tests/data/test_impute_data.csv" in result.stdout


def test_impute_result(tmp_path):
    tmp_dir = tmp_path / "csv-helper"
    tmp_dir.mkdir()
    tmp_file = tmp_dir / "test_impute_output.csv"
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute",
            "./tests/data/test_impute_data.csv",
            str(tmp_file),
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
    assert tmp_file.is_file()

    df_in = (
        pl.read_csv("./tests/data/test_impute_data.csv", infer_schema_length=0)
        .with_row_index(name="id")
        .select(["id", "cases"])
    )
    df_out = (
        pl.read_csv(tmp_file, infer_schema_length=0)
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


def test_impute_pair_result(tmp_path):
    tmp_dir = tmp_path / "csv-helper"
    tmp_dir.mkdir()
    tmp_file = tmp_dir / "test_impute_pair_output.csv"
    fill_range = (1, 5)

    result = runner.invoke(
        app,
        [
            "impute-pair",
            "./tests/data/test_impute_data.csv",
            str(tmp_file),
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
    assert tmp_file.is_file()

    df_in = (
        pl.read_csv("./tests/data/test_impute_data.csv", infer_schema_length=0)
        .with_row_index(name="id")
        .select(["id", "cases", "all_cause"])
    )
    df_out = (
        pl.read_csv(tmp_file, infer_schema_length=0)
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
