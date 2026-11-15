<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
  <img alt="AirCast: tomorrow's PM2.5 for Chicago, forecast today" src="assets/banner.svg" width="100%">
</picture>

<p align="center">
  <img src="https://img.shields.io/badge/track-Climate-2F6BFF?style=flat-square" alt="Climate track">
  <img src="https://img.shields.io/badge/test_MAE-3.11_vs_3.46_baseline-111318?style=flat-square" alt="Test MAE 3.11 vs 3.46 baseline">
  <img src="https://img.shields.io/badge/bad_days_caught-88%25-E5484D?style=flat-square" alt="WATCH threshold caught 88% of days above 15 µg/m³ in the 2025 evaluation">
  <img src="https://img.shields.io/badge/live_demo-no_API_key-2F6BFF?style=flat-square" alt="Live demo, no API key">
</p>

<p align="center">
  <img src="assets/hook.gif" alt="June 2023 Chicago smoke event: daily PM2.5 rose from about 9 to 32 to 134 µg/m³ over three days" width="840">
</p>

**Track:** Climate
**Team:** Parth Patel and Anay Goyal, Emerson High School

## The decision AirCast supports

A Chicago school athletics director needs to decide today whether to reserve indoor space for tomorrow's practice. In the study period, daily fine-particle pollution (PM2.5) exceeded the WHO guideline of **15 µg/m³ on 126 of 1,197 days (11%)**.

## How the forecast works

1. **Measure the outcome.** The notebook predicts the next day's average PM2.5 across EPA monitors in Cook County. We checked the Chicago CAMS estimates against those measurements. The daily series had a correlation of **r = 0.67** and differed by **4.2 µg/m³ on average in absolute terms**. On the peak day in our dataset, CAMS estimated **50 µg/m³** and monitors recorded **134 µg/m³**.
2. **Build the inputs.** The model uses 25 features covering recent monitor readings, daily weather, seasonality, and CAMS estimates from Chicago, Minneapolis, Green Bay, Madison, and Thunder Bay. The upwind estimates give the Ridge model a signal for pollution changes outside Chicago.
3. **Compare forecasts chronologically.** Training targets precede January 1, 2025. Five-fold time-series cross-validation on the training period gave the local Ridge model an MAE of **3.33 µg/m³**, versus **3.53** for persistence. The later evaluation covers **315 next-day forecasts** issued January 1 through November 11, 2025. The persistence baseline uses today's monitor reading as tomorrow's forecast.

The [research notebook](AirCast.ipynb) reports accuracy against EPA monitor readings. The [live demo](src/predict_tomorrow.py) runs without an API key and forecasts CAMS model estimates using current Open-Meteo inputs.

## Forecast accuracy

MAE is the average absolute difference between the forecast and the observed value, in µg/m³. Lower values are better.

| Next-day forecast | MAE across 315 forecasts | MAE on 8 sudden pollution increases |
| --- | ---: | ---: |
| Persistence | 3.46 | 21.84 |
| Local inputs only | 3.23 | 22.76 |
| AirCast with smoke signals | **3.11** | **17.86** |

AirCast's MAE was **10% lower** than persistence overall and **18% lower** on the eight days when observed PM2.5 rose by more than 10 µg/m³ from the previous day. The smoke features were designed after reviewing the local model's 2025 misses, so these results for the enhanced model are exploratory. The [notebook](AirCast.ipynb) also reports the two- and three-day forecasts.

## How to read the alerts

**WATCH** means the top of the estimated prediction interval exceeds 15 µg/m³. It gives the director a reason to reserve an indoor backup. **WARNING** means the point forecast exceeds 15 µg/m³. It signals that the director should review official advisories before deciding about practice. An estimated 80% interval built from 2024 forecast errors covered **241 of 315 observations (77%)** in the 2025 evaluation.

| Alert rule | Days above 15 caught (out of 34) | Missed days | False alerts | Precision |
| --- | ---: | ---: | ---: | ---: |
| WATCH threshold, including WARNING days | **30 (88%)** | 4 | 33 | 48% |
| WARNING threshold | 24 (71%) | 10 | **7** | **77%** |
| Persistence | 23 (68%) | 11 | 11 | 68% |

## Where it falls short

- **Wildfire smoke.** In the June 2023 replay, both AirCast and persistence missed the first jump from **9 to 32 µg/m³**. AirCast warned for the following day, but forecast **17 µg/m³** when monitors recorded **134 µg/m³**. The warning flagged the day even though the forecast badly understated the peak. See the [event replay](figures/04_case_study.png).
- **Local model weights.** In Milwaukee, the Chicago-trained model had an MAE of **3.65 µg/m³** versus **3.60** for Milwaukee persistence. Retraining the same pipeline on Milwaukee data lowered MAE to **3.51**.
- **Forecast timing.** The notebook uses full-day inputs. A forecast issued before the end of the school day would need to be tested with inputs available at that hour.

## What we'd test next

Freeze the model and alert rules before evaluating a fresh period. Then test whether satellite smoke-plume data improves forecasts for the first day of a smoke event and whether rolling calibration improves interval coverage.

## Run it yourself

```bash
python -m pip install -r requirements.txt
jupyter notebook AirCast.ipynb                       # analysis using committed CSVs
python src/predict_tomorrow.py                       # live CAMS forecast (internet required)
python src/fetch_data.py && python src/fetch_epa.py    # optional: refresh data from public APIs
```

## Sources

- **Data:** [EPA Air Quality System daily summary files](https://aqs.epa.gov/aqsweb/airdata/download_files.html) (PM2.5, parameter 88101); [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) ([Copernicus CAMS](https://atmosphere.copernicus.eu/) model) and [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), [CC BY 4.0](https://open-meteo.com/en/license).
- **Thresholds:** [WHO 2021 Global Air Quality Guidelines](https://www.who.int/publications/i/item/9789240034228) (15 µg/m³ daily); [US EPA 24-hour standard](https://www.epa.gov/pm-pollution) (35 µg/m³).
- **Libraries:** [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), [matplotlib](https://matplotlib.org/), [requests](https://requests.readthedocs.io/), [Jupyter](https://jupyter.org/)/[jupytext](https://jupytext.readthedocs.io/).
- **AI tools:** [Claude Code](https://claude.com/claude-code) (Anthropic) was used as a coding assistant throughout: scaffolding the pipeline, writing and debugging code, and drafting this write-up, with the team directing the project and reviewing the work. ChatGPT assisted with follow-up code review, validation, and README editing.
