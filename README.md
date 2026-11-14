# AirCast — next-day air quality forecast for Chicago

**Track:** Climate
**Team:** Parth Patel, Anay <!-- TODO: Anay's full name, exactly as it should appear on certificates -->

## The problem

Fine-particle pollution (PM2.5) is the air pollutant that matters most for health, and children exercising outdoors are among the most exposed. In Chicago, roughly **1 day in 5** exceeds the WHO daily guideline — and during wildfire-smoke events (June 2023, May 2025) the city briefly had some of the worst air in the world. A school athletics director has to decide *today* whether tomorrow's outdoor practice should move indoors, but air-quality apps only tell you about *right now*.

## What we built

AirCast predicts **tomorrow's average PM2.5 for Chicago** from today's air quality and weather, then turns the number into the decision our user actually needs, using two alert levels like real weather services:

- **WATCH** — "have an indoor backup plan" (sensitive trigger, tuned to catch ~85%+ of bad days)
- **WARNING** — "move practice indoors" (predicted PM2.5 above the WHO daily guideline of 15 µg/m³)

Input: the last few days of PM2.5 plus today's weather (wind, rain, temperature, humidity, pressure, season). Output: tomorrow's forecast and an alert level. `src/predict_tomorrow.py` runs it live for today — it fetches current conditions and prints tomorrow's forecast in a few seconds.

## What we added

We started from the suggested "next-day air quality forecast" project and added:

1. **A tuned two-level alert system.** The brief's threshold question ("balance missed warnings against false alarms") became a design decision: we chose the WATCH trigger on a held-out 2024 validation year (lowest trigger catching ≥85% of bad days), froze it, and only then scored 2025. A missed bad day costs far more than a false watch.
2. **A real-event backtest.** We retrained the model using only data available before June 2023 and replayed the Canadian-wildfire smoke crisis day by day to see whether it would have warned Chicago (answer below — honestly mixed).
3. **Honest model selection.** Models compete via 5-fold time-series cross-validation inside the training period; the test year is touched exactly once.
4. **Horizon analysis.** How far ahead does the forecast stay useful? (1, 2 and 3 days.)
5. **Driver analysis.** Permutation importance showing what actually moves next-day PM2.5 in Chicago.

## How it works

**Data:** 1,197 days (Aug 2022 – Nov 2025) for Chicago from [Open-Meteo](https://open-meteo.com/) (free, no API key): hourly PM2.5 from the CAMS atmospheric model averaged to daily means (days with <18 hourly values dropped — none were), joined with daily weather from Open-Meteo's historical archive. **Note (as the brief asks us to say):** these PM2.5 values are model estimates/reanalysis, not physical monitor readings.

**Model:** 15 features (PM2.5 today + 2 lags + 3-day mean; today's weather with wind direction encoded as sin/cos; season as sin/cos of day-of-year). Ridge regression on standardized features won time-series cross-validation (CV MAE 3.73 µg/m³) against two gradient-boosted tree variants and the persistence baseline (CV MAE 4.03). Train: Aug 2022 – Dec 2024 (879 days). Test: all of 2025 (315 days), strictly later in time, never shuffled.

## Results

Our baseline (persistence: "tomorrow = today") scored **4.09 µg/m³ MAE** on the held-out 2025 test year. Our model scored **3.70 µg/m³ MAE** — **10% better**, and the gap grows with horizon (17% better at 2 days ahead, 25% at 3 days).

Alert performance on the 65 bad days (>15 µg/m³) in the 2025 test year:

| Alert | Recall (bad days caught) | Precision (alerts correct) |
|---|---|---|
| AirCast WATCH (trigger 11.5) | **89%** | 35% |
| AirCast WARNING (trigger 15) | 62% | **67%** |
| Persistence (today was bad) | 60% | 60% |

**Where it fails, and why:** the first day of every major smoke event. In the June 2023 backtest, on June 26 the model predicted 13.7 µg/m³ for June 27 — the actual value was 49.7, because wildfire smoke arrived overnight and *nothing in today's local measurements can see it coming*. Persistence missed it too. From the second day on, the model warned correctly through the event. The 2025 test year shows the same pattern: the 7 largest errors are all underestimates on smoke-onset days (May 28–31, June 4–9, July 30, Aug 2). This is a fundamental limit of local-only features, and fixing it is our top "what's next."

**Other limitations:** CAMS reanalysis values can differ from EPA monitor readings (a deployed version would retrain on AirNow data); results are for Chicago only, though the pipeline runs for any city by changing three lines in `src/fetch_data.py`. This is a student prototype, not an official air-quality advisory.

## What we'd do next

1. **See smoke coming:** add upwind signals — PM2.5 from cities to the north/west and satellite smoke-plume data (NOAA HMS) — which is exactly the information our model provably lacks on event-onset days.
2. **Validate against real monitors:** retrain and re-score on EPA AirNow station data for Chicago to check the CAMS-based results hold on physical measurements.

## Run it yourself

```bash
pip install -r requirements.txt
python src/fetch_data.py        # ~1 min: downloads data/daily.csv (also committed)
jupyter notebook AirCast.ipynb  # full analysis, outputs saved
python src/predict_tomorrow.py  # live forecast for tomorrow, right now
```

## Sources

- **Data:** [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) (PM2.5, from the [Copernicus CAMS](https://atmosphere.copernicus.eu/) model) and [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) — free, no API key, [CC BY 4.0](https://open-meteo.com/en/license).
- **Thresholds:** [WHO 2021 Global Air Quality Guidelines](https://www.who.int/publications/i/item/9789240034228) (15 µg/m³ daily PM2.5); [US EPA 24-hour PM2.5 standard](https://www.epa.gov/pm-pollution) (35 µg/m³).
- **Libraries:** [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), [matplotlib](https://matplotlib.org/), [requests](https://requests.readthedocs.io/), [Jupyter](https://jupyter.org/)/[jupytext](https://jupytext.readthedocs.io/).
- **AI tools:** [Claude Code](https://claude.com/claude-code) (Anthropic) was used as a coding assistant throughout — scaffolding the pipeline, writing and debugging code, and drafting this write-up, with the team directing the project and reviewing the work. The hackathon brief itself guided the baseline design.
