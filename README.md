# AirCast — next-day air quality forecast for Chicago

**Track:** Climate
**Team:** Parth Patel, Anay Goyal — Emerson High School

## The problem

A school athletics director in Chicago has to decide *today* whether tomorrow's outdoor practice moves indoors. PM2.5 — the pollutant that matters most for kids exercising outside — exceeds the WHO daily guideline about **1 day in 9** in Chicago, and during wildfire-smoke events the city has had some of the worst air in the world. Air-quality apps say what the air is like *now*; the director needs *tomorrow*.

## What we built

AirCast predicts **tomorrow's PM2.5 as Chicago's EPA regulatory monitors will measure it** — the number behind official advisories — from today's monitor readings, weather, and "smoke signals": the CAMS atmospheric-model estimate for Chicago plus PM2.5 in four cities upwind on the Canadian-smoke path (Minneapolis, Green Bay, Madison, Thunder Bay). Each forecast carries an 80% uncertainty interval and feeds a two-level alert: **WATCH** (interval touches the guideline → have an indoor backup) and **WARNING** (forecast exceeds it → move indoors). `src/predict_tomorrow.py` demos it live in seconds, no API key.

## What we added

Starting from the suggested next-day air-quality project:

1. **We validated the convenient dataset against ground truth — and switched targets.** The brief's no-key fallback (CAMS model estimates) correlates only **r = 0.67** with Chicago's real EPA monitors and understated the worst day almost 3× (50 vs 134 µg/m³). So we predict the monitors, and demote CAMS to an input feature.
2. **We found our model's biggest failure and attacked it.** The worst misses were all sudden smoke arrivals, invisible to local data. Adding the upwind smoke signals cut sudden-onset-day error from 21.8 (persistence) and 22.8 (local-only) to **17.9 µg/m³**.
3. **Honest uncertainty, used for decisions.** Split-conformal 80% intervals (realized coverage on 2025: 77% — we report the shortfall and why). The WATCH trigger is *derived* from the interval, after our first design (tune for 85% recall) failed honestly: it needed 234 alerts/year at 6% precision, so we show that and replaced it.
4. **A brutal real-event backtest and a transfer test** (below).

## How it works

Data: 1,197 days (Aug 2022 – Nov 2025), no gaps — EPA AQS daily files (Cook County monitors averaged; same for Milwaukee), Open-Meteo CAMS air quality and historical weather (free, no key). 25 features: monitor PM2.5 lags, today's weather, season, and the smoke signals. Ridge regression won 5-fold *time-series* cross-validation inside the training period (CV MAE 3.35 vs persistence 3.53) against two gradient-boosted-tree variants. Train: Aug 2022 – Dec 2024. Test: all of 2025 — 315 later days, never shuffled, touched once.

## Results

Persistence baseline ("tomorrow = today"): **3.46 µg/m³ MAE** on the 2025 test year. AirCast: **3.11 µg/m³ — 10% better**, growing to 20% at 2 days ahead and 25% at 3. On 2025's 34 bad days (>15 µg/m³): **WATCH caught 88%** (precision 48%); **WARNING caught 71% at 77% precision**; persistence managed 68%/68%.

**Where it fails — the part we'd say to a judge's face.** We replayed June 2023 (Chicago's worst air on record, 134 µg/m³) with a model trained only on pre-June data. Nobody — not our model, not persistence — saw the first smoke day coming (9 µg/m³ → 32 overnight). The model then warns through the event but predicts 17 for the day that hit 134: right decision, catastrophically wrong magnitude. Honest conclusion: AirCast is for the ordinary 1-in-9 bad day; **extreme smoke events need satellite plume data none of our inputs contain**, and during them people should follow live advisories, not any day-ahead forecast. Also: Chicago-trained weights applied to Milwaukee *lose* to Milwaukee's own persistence (3.65 vs 3.60); the identical pipeline retrained on Milwaukee wins (3.51). Method transfers, weights are local. This is a student prototype, not an official air-quality advisory.

## What we'd do next

Add NOAA HMS satellite smoke-plume extent as a feature (the measured gap); use EPA's live AirNow feed in the demo (the daily files publish with delay, so the no-key live demo runs on CAMS inputs); recalibrate intervals on a rolling window so coverage holds in smoky years.

## Run it yourself

```bash
pip install -r requirements.txt
python src/fetch_data.py && python src/fetch_epa.py   # rebuild data/ (also committed)
jupyter notebook AirCast.ipynb                        # full analysis, outputs saved
python src/predict_tomorrow.py                        # live forecast for tomorrow
```

## Sources

- **Data:** [EPA Air Quality System daily summary files](https://aqs.epa.gov/aqsweb/airdata/download_files.html) (PM2.5, parameter 88101); [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) ([Copernicus CAMS](https://atmosphere.copernicus.eu/) model) and [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), [CC BY 4.0](https://open-meteo.com/en/license).
- **Thresholds:** [WHO 2021 Global Air Quality Guidelines](https://www.who.int/publications/i/item/9789240034228) (15 µg/m³ daily); [US EPA 24-hour standard](https://www.epa.gov/pm-pollution) (35 µg/m³).
- **Libraries:** [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), [matplotlib](https://matplotlib.org/), [requests](https://requests.readthedocs.io/), [Jupyter](https://jupyter.org/)/[jupytext](https://jupytext.readthedocs.io/).
- **AI tools:** [Claude Code](https://claude.com/claude-code) (Anthropic) was used as a coding assistant throughout — scaffolding the pipeline, writing and debugging code, and drafting this write-up, with the team directing the project and reviewing the work.
