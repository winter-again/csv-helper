from csv_helper.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_app():
    result = runner.invoke(
        app, ["preview", "./data/county_week_covid_v2.csv", "-n", "15"]
    )
    assert result.exit_code == 0
