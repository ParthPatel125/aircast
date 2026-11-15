<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
  <img alt="AirCast: tomorrow's PM2.5 for Chicago, forecast today" src="assets/banner.svg" width="100%">
</picture>

<p align="center">
  <img src="https://img.shields.io/badge/track-Climate-2F6BFF?style=flat-square" alt="Climate track">
  <img src="https://img.shields.io/badge/test_MAE-3.11_vs_3.46_baseline-111318?style=flat-square" alt="Test MAE 3.11 vs 3.46 baseline">
  <img src="https://img.shields.io/badge/bad_days_caught-88%25-E5484D?style=flat-square" alt="88% of bad days caught">
  <img src="https://img.shields.io/badge/live_demo-no_API_key-2F6BFF?style=flat-square" alt="Live demo, no API key">
</p>

<p align="center">
  <img src="assets/hook.gif" alt="June 27, 2023: Chicago's air went from 9 to 134 micrograms overnight" width="840">
</p>

**Track:** Climate
**Team:** Parth Patel and Anay Goyal, Emerson High School

## The problem

A school athletics director in Chicago has to decide today whether tomorrow's outdoor practice moves indoors. PM2.5, the pollutant that matters most for kids exercising outside, exceeds the WHO daily guideline about **1 day in 9** in Chicago, and during wildfire-smoke events the city has had some of the worst air in the world. Air-quality apps tell you what the air is like right now. The director needs tomorrow.

## What we built

AirCast predicts **tomorrow's PM2.5 as Chicago's EPA regulatory monitors will measure it**, which is the number behind official advisories. It works from today's monitor readings, the weather, and what we call smoke signals: the CAMS atmospheric-model estimate for Chicago plus current PM2.5 in four cities upwind on the Canadian-smoke path (Minneapolis, Green Bay, Madison, Thunder Bay). Every forecast comes with an 80% uncertainty interval and a two-level alert: **WATCH** means the interval touches the guideline, so plan an indoor backup; **WARNING** means the forecast itself is over it, so move indoors. `src/predict_tomorrow.py` runs a real forecast in a few seconds, no API key needed.

## What we added

Starting from the suggested next-day air-quality project:

1. Before trusting the brief's no-key data source (CAMS model estimates), we checked it against Chicago's real EPA monitors. It only correlates at **r = 0.67** and understated the worst day in our data almost 3× (50 vs 134 µg/m³). That check changed the whole project: we predict the monitors instead, and CAMS became just another input.
2. Our worst misses were all sudden smoke arrivals, which local data simply cannot see. Adding the upwind smoke signals cut error on those days from 21.8 (persistence) and 22.8 (local-only) to **17.9 µg/m³**.
3. The forecast ships with split-conformal 80% intervals (realized coverage on 2025 was 77%, and we say why it fell short). Our first alert design, tuned for 85% recall, turned out to need 234 alerts a year at 6% precision. We kept that failure in the notebook and derived the WATCH trigger from the interval instead.
4. A replay of the June 2023 smoke event with a frozen model, and a Milwaukee transfer test (both below).

## How it works

Data: 1,197 days (Aug 2022 to Nov 2025), no gaps: EPA AQS daily files (Cook County monitors averaged; same for Milwaukee), Open-Meteo CAMS air quality and historical weather (free, no key). 25 features: monitor PM2.5 lags, today's weather, season, and the smoke signals. Ridge regression won 5-fold *time-series* cross-validation inside the training period (CV MAE 3.35 vs persistence 3.53) against two gradient-boosted-tree variants. Train: Aug 2022 to Dec 2024. Test: all of 2025, 315 later days, never shuffled, touched once.

## Results

Persistence baseline ("tomorrow = today"): **3.46 µg/m³ MAE** on the 2025 test year. AirCast: **3.11 µg/m³, 10% better**, growing to 20% at 2 days ahead and 25% at 3. On 2025's 34 bad days (>15 µg/m³): **WATCH caught 88%** (precision 48%); **WARNING caught 71% at 77% precision**; persistence managed 68%/68%.

**Where it fails.** This is the part we'd tell a judge first. We replayed June 2023 (Chicago's worst air on record, 134 µg/m³) with a model trained only on pre-June data. Nobody saw the first smoke day coming, not our model and not persistence (9 to 32 µg/m³ overnight). The model then warns through the rest of the event, but it predicts 17 for the day that hit 134: right decision, wrong magnitude. So AirCast is for the ordinary 1-in-9 bad day; **extreme smoke events need satellite plume data that none of our inputs contain**, and during one you should follow live advisories, not any day-ahead forecast. One more limit: Chicago-trained weights applied to Milwaukee actually lose to Milwaukee's own persistence (3.65 vs 3.60), while the identical pipeline retrained on Milwaukee wins (3.51). The method transfers; the weights are local. And this is a student prototype, not an official air-quality advisory.

## What we'd do next

Add NOAA's satellite smoke-plume maps as a feature, since that is exactly the gap we measured. Switch the live demo to EPA's AirNow feed (the official daily files publish on a delay, which is why the no-key demo runs on CAMS inputs). And recalibrate the intervals on a rolling window so coverage holds up in smoky years.

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
- **AI tools:** [Claude Code](https://claude.com/claude-code) (Anthropic) was used as a coding assistant throughout: scaffolding the pipeline, writing and debugging code, and drafting this write-up, with the team directing the project and reviewing the work.
