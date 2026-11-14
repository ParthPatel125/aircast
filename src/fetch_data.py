"""Download historical PM2.5 and weather data for one city from Open-Meteo.

Open-Meteo is free and needs no API key. The air-quality values come from the
CAMS atmosphere model (reanalysis/estimates), not physical monitors - the
write-up mentions this, as the hackathon brief asks.

Usage: python src/fetch_data.py
Writes: data/daily.csv (one row per day: PM2.5 + weather)
"""
import time

import pandas as pd
import requests

# ---- Config: change these three lines to forecast a different city ----
CITY = "Chicago"
LAT, LON = 41.8781, -87.6298
TIMEZONE = "America/Chicago"

START_DATE = "2016-01-01"   # air-quality archive reaches back to ~2013
END_DATE = "2025-11-12"     # last full day before the hackathon started

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_json(url, params, retries=3):
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()


def fetch_pm25():
    """Hourly PM2.5, averaged to one value per day."""
    js = fetch_json(AIR_URL, {
        "latitude": LAT, "longitude": LON, "timezone": TIMEZONE,
        "start_date": START_DATE, "end_date": END_DATE,
        "hourly": "pm2_5",
    })
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    # A day only counts if at least 18 of its 24 hourly values exist
    daily = df.groupby("date")["pm2_5"].agg(["mean", "count"])
    daily = daily[daily["count"] >= 18]["mean"].rename("pm25")
    daily.index = pd.to_datetime(daily.index)
    return daily


def fetch_weather():
    """Daily weather that physically drives PM2.5: wind, rain, temperature."""
    js = fetch_json(WEATHER_URL, {
        "latitude": LAT, "longitude": LON, "timezone": TIMEZONE,
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": ",".join([
            "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "wind_speed_10m_max",
            "wind_direction_10m_dominant", "shortwave_radiation_sum",
            "relative_humidity_2m_mean", "surface_pressure_mean",
        ]),
    })
    df = pd.DataFrame(js["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def main():
    print(f"Fetching PM2.5 for {CITY} ({START_DATE} to {END_DATE})...")
    pm25 = fetch_pm25()
    print(f"  {len(pm25)} days of PM2.5, range {pm25.index.min().date()} to {pm25.index.max().date()}")
    print("Fetching weather...")
    weather = fetch_weather()
    print(f"  {len(weather)} days of weather")
    daily = weather.join(pm25, how="inner").dropna(subset=["pm25"])
    daily.index.name = "date"
    daily.to_csv("data/daily.csv")
    print(f"Wrote data/daily.csv: {len(daily)} rows, columns: {list(daily.columns)}")


if __name__ == "__main__":
    main()
