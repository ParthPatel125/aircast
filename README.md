# AirCast — next-day air quality forecast for Chicago

**Track:** Climate
**Team:** Parth Patel, Anay Goyal — Emerson High School

## The problem

Fine-particle pollution (PM2.5) is the air pollutant that matters most for health, and children exercising outdoors are among the most exposed. In Chicago, roughly **1 day in 5** exceeds the WHO daily guideline — and during wildfire-smoke events (June 2023, May 2025) the city briefly had some of the worst air in the world. A school athletics director has to decide *today* whether tomorrow's outdoor practice should move indoors, but air-quality apps only tell you about *right now*.

## What we built

AirCast predicts **tomorrow's average PM2.5 for Chicago** from today's air quality and weather — including PM2.5 measured today in four cities *upwind* on the Canadian-wildfire-smoke path — then turns the number into the decision our user actually needs, using two alert levels like real weather services:

- **WATCH** — "have an indoor backup plan" (sensitive trigger, tuned to catch ~85%+ of bad days)
- **WARNING** — "move practice indoors" (predicted PM2.5 above the WHO daily guideline of 15 µg/m³)

Every forecast comes with an uncertainty range, not just a number. `src/predict_tomorrow.py` runs it live: it fetches current conditions in Chicago and the upwind cities and prints tomorrow's forecast, range, and alert level in a few seconds.

## What we added

We started from the suggested "next-day air quality forecast" project and added:

1. **We found our model's biggest failure and fixed it.** Error analysis showed every large miss was the *first day* of a wildfire-smoke episode — smoke arrives overnight, invisible to local data. So we added an upwind smoke signal: today's PM2.5 in Minneapolis, Green Bay, Madison and Thunder Bay (air that will be over Chicago tomorrow). Error on sudden-onset days fell 30%, and in the June 2023 backtest the upwind model fires the warning on the exact day the local-only model missed.
2. **A real-event backtest.** We retrained the model using only data available before June 2023 and replayed the Canadian-wildfire smoke crisis day by day — an independent check on data the feature idea was never tuned on.
3. **A tuned two-level alert system.** The brief's threshold question ("balance missed warnings against false alarms") became a design decision: the WATCH trigger is chosen on a held-out 2024 validation year, frozen, then scored on 2025. A missed bad day costs far more than a false watch.
4. **Honest uncertainty.** Split-conformal 80% prediction intervals — and we report that real coverage on 2025 was 72%, because 2025 was smokier than the calibration year. We explain why rather than hide it.
5. **A generalization test.** Chicago's trained model applied as-is to Milwaukee *loses* to Milwaukee's own persistence baseline (honest negative result) — but the same pipeline retrained on Milwaukee wins. The method transfers; the weights are local.
6. **Honest model selection, horizon and driver analysis.** Models compete via 5-fold time-series cross-validation inside the training period; permutation importance shows what moves Chicago's air; the model's edge over persistence grows from 1 to 3 days ahead.

## How it works

**Data:** 1,197 days (Aug 2022 – Nov 2025) from [Open-Meteo](https://open-meteo.com/) (free, no API key): hourly PM2.5 from the CAMS atmospheric model averaged to daily means, for Chicago plus the four upwind cities and Milwaukee, joined with daily weather from Open-Meteo's historical archive. **Note (as the brief asks us to say):** these PM2.5 values are model estimates/reanalysis, not physical monitor readings.

**Model:** 24 features — Chicago PM2.5 today + 2 lags + 3-day mean; today's weather (wind direction encoded as sin/cos); season as sin/cos of day-of-year; and for each upwind city its PM2.5 today and one-day change, plus the four-city maximum. Ridge regression on standardized features won time-series cross-validation (CV MAE 3.73 µg/m³) against two gradient-boosted tree variants and the persistence baseline (CV MAE 4.03). Train: Aug 2022 – Dec 2024 (879 days). Test: all of 2025 (315 days), strictly later in time, never shuffled, touched once.

## Results

Our baseline (persistence: "tomorrow = today") scored **4.09 µg/m³ MAE** on the held-out 2025 test year. Our final model scored **3.51 µg/m³ — 14% better** — and on the 14 sudden-onset days (air suddenly turning bad, the days that matter most) the upwind signal cuts error from 13.37 (persistence) and 12.21 (local-only) to **8.59 µg/m³**.

Alert performance on the 65 bad days (>15 µg/m³) in the 2025 test year:

| Alert | Recall (bad days caught) | Precision (alerts correct) |
|---|---|---|
| AirCast WATCH (trigger 11) | **92%** | 39% |
| AirCast WARNING (trigger 15) | 58% | **72%** |
| Persistence (today was bad) | 60% | 60% |

**The June 2023 backtest** (model trained only on pre-June-2023 data): on June 26, the day before Chicago's air became among the worst in the world, the local-only model predicted 13.7 µg/m³ — no warning. The upwind model predicted 23.6 — **WARNING issued**, because Minneapolis and Madison were already smoky that day. Persistence said "all clear" too. Actual June 27 value: 49.7.

**Where it still fails:** the model gets the *decision* right on smoke days but underestimates their *magnitude* (23.6 predicted vs 49.7 actual), and it stays pessimistic for a day after events end. **Other limitations:** CAMS reanalysis values can differ from EPA monitor readings (a deployed version would retrain on AirNow data); interval coverage degrades under distribution shift (72% vs 80% promised); weights are Chicago-specific. This is a student prototype, not an official air-quality advisory.

## What we'd do next

1. **See smoke sooner and size it better:** add NOAA HMS satellite smoke-plume data — the upwind cities fixed detection, satellites should fix magnitude.
2. **Validate against real monitors:** retrain and re-score on EPA AirNow station data for Chicago to check the CAMS-based results hold on physical measurements.
3. **Rolling recalibration** of the prediction intervals so coverage holds in smoky years.

## Run it yourself

```bash
pip install -r requirements.txt
python src/fetch_data.py        # ~2 min: downloads all data/ CSVs (also committed)
jupyter notebook AirCast.ipynb  # full analysis, outputs saved
python src/predict_tomorrow.py  # live forecast for tomorrow, right now
```

## Sources

- **Data:** [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) (PM2.5, from the [Copernicus CAMS](https://atmosphere.copernicus.eu/) model) and [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) — free, no API key, [CC BY 4.0](https://open-meteo.com/en/license).
- **Thresholds:** [WHO 2021 Global Air Quality Guidelines](https://www.who.int/publications/i/item/9789240034228) (15 µg/m³ daily PM2.5); [US EPA 24-hour PM2.5 standard](https://www.epa.gov/pm-pollution) (35 µg/m³).
- **Libraries:** [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), [matplotlib](https://matplotlib.org/), [requests](https://requests.readthedocs.io/), [Jupyter](https://jupyter.org/)/[jupytext](https://jupytext.readthedocs.io/).
- **AI tools:** [Claude Code](https://claude.com/claude-code) (Anthropic) was used as a coding assistant throughout — scaffolding the pipeline, writing and debugging code, and drafting this write-up, with the team directing the project and reviewing the work. The hackathon brief itself guided the baseline design.
