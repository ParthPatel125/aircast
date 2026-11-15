"""AirCast live demo: forecast tomorrow's PM2.5 for Chicago, right now.

Trains on historical CAMS estimates, fetches current Open-Meteo inputs, and
prints a CAMS-based next-day estimate with the alert level.
Run: python src/predict_tomorrow.py
"""
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
WARNING_TRIGGER = 15.0  # WHO 2021 daily guideline
# 80% prediction interval, from 2024 validation residuals of this script's
# no-key CAMS pipeline (residual quantiles, as in notebook section 9)
INTERVAL_LO, INTERVAL_HI = -5.3, 3.5
# WATCH fires when the top of the interval crosses the guideline (notebook s10)
WATCH_TRIGGER = WARNING_TRIGGER - INTERVAL_HI
UPWIND = {"minneapolis": (44.98, -93.27), "green_bay": (44.51, -88.01),
          "madison": (43.07, -89.40), "thunder_bay": (48.38, -89.25)}


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


def add_upwind(f, up):
    """Upwind cities' PM2.5 today: the smoke early-warning signal."""
    f = f.copy()
    for c in UPWIND:
        f[f"up_{c}"] = up[c]
        f[f"up_{c}_chg"] = up[c] - up[c].shift(1)
    f["up_max"] = up[list(UPWIND)].max(axis=1)
    return f


def fetch_live_json(url, params, section):
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get(section), dict):
        raise ValueError(f"Open-Meteo response is missing {section} data.")
    return payload[section]


def fetch_recent_pm25(lat, lon):
    """Last 10 days of daily-mean PM2.5 at one location, including today."""
    hourly = fetch_live_json("https://air-quality-api.open-meteo.com/v1/air-quality", {
        "latitude": lat, "longitude": lon, "timezone": TIMEZONE,
        "past_days": 10, "forecast_days": 1, "hourly": "pm2_5"}, "hourly")
    a = pd.DataFrame(hourly)
    a["time"] = pd.to_datetime(a["time"])
    a = a.dropna(subset=["pm2_5"])
    a["date"] = a["time"].dt.date
    daily = a.groupby("date")["pm2_5"].agg(["mean", "count"])
    daily.index = pd.to_datetime(daily.index)
    return daily


def fetch_recent(today):
    """Last 10 days of PM2.5 (Chicago + upwind) and weather, including today."""
    chi = fetch_recent_pm25(LAT, LON)
    if today not in chi.index:
        raise ValueError(f"No Chicago PM2.5 estimates for {today.date()}.")
    hours_today = int(chi.at[today, "count"])
    pm = chi["mean"].rename("pm25")
    up = pd.concat({name: fetch_recent_pm25(lat, lon)["mean"]
                    for name, (lat, lon) in UPWIND.items()}, axis=1)

    daily_weather = fetch_live_json("https://api.open-meteo.com/v1/forecast", {
        "latitude": LAT, "longitude": LON, "timezone": TIMEZONE,
        "past_days": 10, "forecast_days": 1,
        "daily": ",".join([
            "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "wind_speed_10m_max",
            "wind_direction_10m_dominant", "shortwave_radiation_sum",
            "relative_humidity_2m_mean", "surface_pressure_mean"])}, "daily")
    w = pd.DataFrame(daily_weather)
    w["time"] = pd.to_datetime(w["time"])
    dates = pd.date_range(end=today, periods=3, name="date")
    recent = w.set_index("time").reindex(dates).join(pm)
    return recent, up.reindex(dates), hours_today


def main():
    today = pd.Timestamp(datetime.now(ZoneInfo(TIMEZONE)).date())
    print(f"AirCast: next-day PM2.5 forecast for {CITY}\n" + "-" * 48)
    hist = pd.read_csv(DATA_DIR / "daily.csv", parse_dates=["date"], index_col="date")
    hist_up = pd.read_csv(DATA_DIR / "upwind.csv", parse_dates=["date"], index_col="date")
    X = add_upwind(build_features(hist), hist_up)
    y = hist["pm25"].shift(-1)
    train = X.join(y.rename("y")).dropna()
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train.drop(columns="y"), train["y"])
    print(f"Model trained on {len(train)} days "
          f"({train.index.min().date()} to {train.index.max().date()})")

    recent, recent_up, hours_today = fetch_recent(today)
    today_row = add_upwind(build_features(recent), recent_up).loc[[today]]
    missing = today_row.columns[~np.isfinite(today_row.to_numpy()).all(axis=0)]
    if len(missing):
        raise ValueError(f"Missing inputs for {today.date()}: {', '.join(missing)}.")
    print(f"Today's CAMS estimate ({today.date()}): {today_row['pm25_today'].iloc[0]:.1f} µg/m³ "
          f"({hours_today} hourly estimates)")
    print("Upwind CAMS estimates: " + ", ".join(
        f"{c.replace('_', ' ').title()} {today_row[f'up_{c}'].iloc[0]:.0f}" for c in UPWIND)
        + " µg/m³")

    raw_pred = float(model.predict(today_row)[0])
    pred = max(0.0, raw_pred)
    lower = max(0.0, raw_pred + INTERVAL_LO)
    upper = max(0.0, raw_pred + INTERVAL_HI)
    tomorrow = today.date() + timedelta(days=1)
    print(f"\nCAMS-based forecast for tomorrow ({tomorrow}): {pred:.1f} µg/m³ "
          f"(estimated 80% interval {lower:.0f}–{upper:.0f})")
    if pred > WARNING_TRIGGER:
        print("Alert level: WARNING. Likely above the WHO daily guideline. "
              "Move outdoor practice indoors.")
    elif pred > WATCH_TRIGGER:
        print("Alert level: WATCH. Tomorrow may approach the WHO daily guideline. "
              "Have an indoor backup plan.")
    else:
        print("Alert level: BELOW GUIDELINE in this forecast.")


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"AirCast forecast unavailable: {exc}") from None
