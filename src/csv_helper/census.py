from io import StringIO
from pathlib import Path

import polars as pl
import requests

OUT = Path(__file__).parents[2]


def main() -> int:
    df = get_census_popn("state", ".")
    with pl.Config(tbl_cols=-1):
        print(df)

    return 0


# TODO: should consider the county FIPS changes over time, boundary changes, etc.
# TODO: irregular data availability and naming schemes mean it's impossible to generalize this?
# TODO: sep function or just state?
def get_census_popn(geo: str, out: str | Path | None = None) -> pl.DataFrame:
    """
    Request and return state or county population data from Census Bureau's
    FTP site. Restricted to 2020-2023 dataset, which has data from
    2020 to 2023. Also takes optional path to a directory in which
    to save the raw data.
    """
    # TODO: this county file actually has state data too so could just it for both?
    # but then we're throwing away most of it so inefficient
    arg_mapper = {
        "state": ("2020-2023", "state", "NST-EST2023-ALLDATA"),
        "county": ("2020-2023", "counties", "co-est2023-alldata"),
    }

    time, geo, file_name = arg_mapper[geo]
    url = f"https://www2.census.gov/programs-surveys/popest/datasets/{time}/{geo}/totals/{file_name}.csv"

    try:
        req = requests.get(url)
        req.raise_for_status()
    except requests.exceptions.HTTPError:
        print("HTTP error while requesting")
        raise
    except requests.exceptions.RequestException:
        print("Some fatal request error")
        raise

    if out is not None:
        out = Path(out)
        if not out.is_dir():
            raise ValueError(f"Directory at {out} doesn't exist")

        with open(out / f"{file_name}.csv", "w") as f:
            f.write(req.text)

    with StringIO(req.text) as f:
        lf = pl.scan_csv(f, schema_overrides={"STATE": pl.String, "COUNTY": pl.String})

    invalid_states = ["60", "66", "69", "72", "74", "78"]
    df = (
        lf.select(
            "STATE",
            "COUNTY",
            "STNAME",
            "CTYNAME",
            "POPESTIMATE2020",
            "POPESTIMATE2021",
            "POPESTIMATE2022",
            "POPESTIMATE2023",
        )
        .filter(
            pl.col("COUNTY") != "000",
            ~pl.col("STATE").is_in(invalid_states),
        )
        .with_columns(county_fips=pl.col("STATE") + pl.col("COUNTY"))
        .drop("STATE", "COUNTY")
        .rename(
            {
                "STNAME": "state_name",
                "CTYNAME": "county_name",
                "POPESTIMATE2020": "popn_2020",
                "POPESTIMATE2021": "popn_2021",
                "POPESTIMATE2022": "popn_2022",
                "POPESTIMATE2023": "popn_2023",
            }
        )
        .select(
            "state_name",
            "county_name",
            "county_fips",
            "popn_2020",
            "popn_2021",
            "popn_2022",
            "popn_2023",
        )
        .collect()
    )

    assert df.select(pl.col("county_fips").n_unique()).item() == df.height, (
        "Expected to have one row per county FIPS"
    )

    return df


if __name__ == "__main__":
    raise SystemExit(main())
