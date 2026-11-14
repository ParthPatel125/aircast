"""Build a ground-truth PM2.5 series for Chicago from EPA regulatory monitors.

Downloads the EPA AQS annual "daily summary" files for PM2.5 (parameter 88101),
keeps Cook County (Chicago) and Milwaukee County monitors, and averages all
24-hour monitor values per day into one series per city. These regulatory
monitors are the ground truth AirCast predicts. Usage: python src/fetch_epa.py
Writes: data/epa_chicago.csv, data/epa_milwaukee.csv
"""
import io
import zipfile

import pandas as pd
import requests

YEARS = [2022, 2023, 2024, 2025]
URL = "https://aqs.epa.gov/aqsweb/airdata/daily_88101_{year}.zip"
COUNTIES = {"chicago": (17, 31), "milwaukee": (55, 79)}  # (state, county) FIPS

frames = {name: [] for name in COUNTIES}
for year in YEARS:
    print(f"Downloading EPA daily PM2.5 for {year} (~10 MB)...")
    r = requests.get(URL.format(year=year), timeout=300)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), low_memory=False)
    for name, (state, county) in COUNTIES.items():
        sel = df[(df["State Code"] == state) & (df["County Code"] == county)
                 & (df["Sample Duration"].str.contains("24"))]
        daily = sel.groupby("Date Local")["Arithmetic Mean"].mean().rename("epa_pm25")
        frames[name].append(daily)
        print(f"  {name}: {len(daily)} days from {sel['Site Num'].nunique()} monitors")

for name in COUNTIES:
    out = pd.concat(frames[name]).sort_index()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out.to_csv(f"data/epa_{name}.csv")
    print(f"Wrote data/epa_{name}.csv: {len(out)} days")
