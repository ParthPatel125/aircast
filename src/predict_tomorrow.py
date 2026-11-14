"""AirCast live demo: forecast tomorrow's PM2.5 for Chicago, right now.

Trains the model on the full historical dataset (data/daily.csv), fetches
today's conditions from Open-Meteo, and prints tomorrow's forecast with the
alert level. Run: python src/predict_tomorrow.py
"""
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

CITY = "Chicago"
LAT, LON = 41.8781, -87.6298
TIMEZONE = "America/Chicago"
WATCH_TRIGGER = 11.5   # tuned on the 2024 validation year (see AirCast.ipynb)
WARNING_TRIGGER = 15.0  # WHO 2021 daily guideline


def build_features(d):
    """Same features as the notebook: recent PM2.5, today's weather, season."""
    f = pd.DataFrame(index=d.index)
    f["pm25_today"] = d["pm25"]
    f["pm25_lag1"] = d["pm25"].shift(1)
    f["pm25_lag2"] = d["pm25"].shift(2)
    f["pm25_3day_mean"] = d["pm25"].rolling(3).mean()
    for col in ["temperature_2m_mean", "temperature_2m_max", "precipitation_sum",
                "wind_speed_10m_max", "relative_humidity_2m_mean",
                "shortwave_radiation_sum", "surface_pressure_mean"]:
        f[col] = d[col]
    wd = np.deg2rad(d["wind_direction_10m_dominant"])
    f["wind_dir_sin"], f["wind_dir_cos"] = np.sin(wd), np.cos(wd)
    doy = d.index.dayofyear / 365.25 * 2 * np.pi
    f["season_sin"], f["season_cos"] = np.sin(doy), np.cos(doy)
    return f


def fetch_recent():
    """Last 10 days of PM2.5 and weather, including today so far."""
    air = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
        "latitude": LAT, "longitude": LON, "timezone": TIMEZONE,
        "past_days": 10, "forecast_days": 1, "hourly": "pm2_5"}, timeout=60).json()
    a = pd.DataFrame(air["hourly"])
    a["time"] = pd.to_datetime(a["time"])
    a = a.dropna(subset=["pm2_5"])
    a["date"] = a["time"].dt.date
    pm = a.groupby("date")["pm2_5"].agg(["mean", "count"])
    hours_today = int(pm["count"].iloc[-1])
    pm = pm["mean"].rename("pm25")
    pm.index = pd.to_datetime(pm.index)

    wx = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": LAT, "longitude": LON, "timezone": TIMEZONE,
        "past_days": 10, "forecast_days": 1,
        "daily": ",".join([
            "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "wind_speed_10m_max",
            "wind_direction_10m_dominant", "shortwave_radiation_sum",
            "relative_humidity_2m_mean", "surface_pressure_mean"])}, timeout=60).json()
    w = pd.DataFrame(wx["daily"])
    w["time"] = pd.to_datetime(w["time"])
    return w.set_index("time").join(pm, how="inner").dropna(subset=["pm25"]), hours_today


def main():
    print(f"AirCast — next-day PM2.5 forecast for {CITY}\n" + "-" * 48)
    hist = pd.read_csv("data/daily.csv", parse_dates=["date"], index_col="date")
    X = build_features(hist)
    y = hist["pm25"].shift(-1)
    train = X.join(y.rename("y")).dropna()
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train.drop(columns="y"), train["y"])
    print(f"Model trained on {len(train)} days "
          f"({train.index.min().date()} to {train.index.max().date()})")

    recent, hours_today = fetch_recent()
    today_row = build_features(recent).iloc[[-1]]
    today = today_row.index[0].date()
    if today < date.today():
        print(f"Note: most recent complete data is for {today}.")
    print(f"Today ({today}) so far: {today_row['pm25_today'].iloc[0]:.1f} µg/m³ "
          f"({hours_today} hours of data)")

    pred = float(model.predict(today_row)[0])
    tomorrow = today + timedelta(days=1)
    print(f"\nForecast for tomorrow ({tomorrow}): {pred:.1f} µg/m³")
    if pred > WARNING_TRIGGER:
        print("Alert level: WARNING — likely above the WHO daily guideline. "
              "Move outdoor practice indoors.")
    elif pred > WATCH_TRIGGER:
        print("Alert level: WATCH — may approach the WHO daily guideline. "
              "Have an indoor backup plan.")
    else:
        print("Alert level: ALL CLEAR — outdoor activity looks fine.")


if __name__ == "__main__":
    main()
