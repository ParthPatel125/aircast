# %% [markdown]
# # AirCast: next-day air quality forecast for Chicago
#
# **Track:** AI + Climate · **Team:** Parth Patel, Anay
#
# **Who is this for?** A school athletics director (or a parent) in Chicago deciding
# *today* whether tomorrow's outdoor practice should be moved indoors. Fine-particle
# pollution (PM2.5) is the pollutant that matters most for that call: it penetrates
# deep into the lungs, and children exercising outdoors are among the most exposed.
#
# **What it does:** predicts tomorrow's average PM2.5 for Chicago from today's air
# quality and weather, then turns the prediction into a concrete warning:
# *"limit outdoor activity tomorrow"* when the forecast crosses the WHO daily
# guideline (15 µg/m³).
#
# **The rules we follow (from the brief):**
# - One location (Chicago), one decision, one decider.
# - Train on earlier dates, test on later dates — never shuffled.
# - Every model is compared against the **persistence baseline**: "tomorrow will be
#   the same as today." Beating it is the whole game in day-ahead forecasting.

# %% [markdown]
# ## 1. Data
#
# Both sources are from [Open-Meteo](https://open-meteo.com/) (free, no API key),
# fetched by `src/fetch_data.py` and saved to `data/daily.csv`:
#
# - **PM2.5**: hourly values from the CAMS atmospheric model, averaged to daily
#   means. These are *model estimates (reanalysis)*, not physical monitor readings —
#   the brief asks us to say this clearly, and we discuss what it means in the
#   limitations section.
# - **Weather**: daily temperature, precipitation, wind, humidity, radiation and
#   pressure from Open-Meteo's historical weather archive.
#
# Coverage: **2022-08-04 to 2025-11-12** (1,197 days, no gaps). The fetch script
# also downloads PM2.5 for four cities *upwind* of Chicago and for Milwaukee —
# those enter the story in sections 10 and 12.

# %%
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# numpy's Accelerate BLAS on macOS raises spurious "divide by zero in matmul"
# RuntimeWarnings during Ridge regression. The outputs are correct - we assert
# every prediction is finite right after computing them (section 4).
warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

plt.rcParams.update({"figure.figsize": (11, 4), "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv("data/daily.csv", parse_dates=["date"], index_col="date")
WHO_DAILY = 15.0   # WHO 2021 daily guideline, ug/m3
EPA_DAILY = 35.0   # US EPA 24-hour standard, ug/m3
df["pm25"].describe().round(1)

# %% [markdown]
# ### What three years of Chicago air looks like

# %%
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(df.index, df["pm25"], lw=0.8, color="#2563eb")
ax.axhline(WHO_DAILY, color="#d97706", ls="--", lw=1, label=f"WHO daily guideline ({WHO_DAILY:.0f})")
ax.axhline(EPA_DAILY, color="#dc2626", ls="--", lw=1, label=f"US EPA 24h standard ({EPA_DAILY:.0f})")
ax.annotate("Canadian wildfire smoke\nJune 27–29, 2023", xy=(pd.Timestamp("2023-06-28"), 57),
            xytext=(pd.Timestamp("2023-10-15"), 50), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_ylabel("PM2.5 (µg/m³, daily mean)")
ax.set_title("Daily PM2.5 in Chicago, Aug 2022 – Nov 2025")
ax.legend(frameon=False)
fig.autofmt_xdate()
fig.tight_layout(); fig.savefig("figures/01_history.png", bbox_inches="tight")
plt.show()

n = len(df)
print(f"Days over WHO guideline (15): {(df.pm25 > WHO_DAILY).sum()} of {n} ({(df.pm25 > WHO_DAILY).mean():.0%})")
print(f"Days over EPA standard (35):  {(df.pm25 > EPA_DAILY).sum()} of {n}")

# %% [markdown]
# Roughly **1 day in 5** exceeds the WHO guideline, so this is a real, recurring
# decision — not a once-a-year event. The extreme days are wildfire smoke.

# %% [markdown]
# ## 2. Features
#
# To predict tomorrow, we only use information available **today**:
#
# - today's PM2.5, yesterday's, the day before, and a 3-day average (recent trend)
# - today's weather: wind (disperses particles), rain (washes them out), temperature,
#   humidity, solar radiation, surface pressure (stagnant high-pressure days trap smog)
# - the season, encoded as sine/cosine of day-of-year so December 31 sits next to January 1
#
# The target is **tomorrow's daily-mean PM2.5** (`pm25` shifted back one day).

# %%
def build_features(d):
    f = pd.DataFrame(index=d.index)
    f["pm25_today"] = d["pm25"]
    f["pm25_lag1"] = d["pm25"].shift(1)
    f["pm25_lag2"] = d["pm25"].shift(2)
    f["pm25_3day_mean"] = d["pm25"].rolling(3).mean()
    for col in ["temperature_2m_mean", "temperature_2m_max", "precipitation_sum",
                "wind_speed_10m_max", "relative_humidity_2m_mean",
                "shortwave_radiation_sum", "surface_pressure_mean"]:
        f[col] = d[col]
    # wind direction is circular (359 deg is next to 0 deg), so encode as sin/cos
    wd = np.deg2rad(d["wind_direction_10m_dominant"])
    f["wind_dir_sin"], f["wind_dir_cos"] = np.sin(wd), np.cos(wd)
    doy = d.index.dayofyear / 365.25 * 2 * np.pi
    f["season_sin"], f["season_cos"] = np.sin(doy), np.cos(doy)
    return f

features = build_features(df)
target = df["pm25"].shift(-1).rename("pm25_tomorrow")
data = features.join(target).dropna()
print(f"{data.shape[0]} usable days, {features.shape[1]} features")
data.head(3).round(2)

# %% [markdown]
# ## 3. Train/test split — by time, never shuffled
#
# A random split would let the model "see the future" (days adjacent to test days),
# which inflates scores dishonestly. We train on **Aug 2022 – Dec 2024** and test on
# **all of 2025** — 316 days the model has never seen, including a 2025 wildfire-smoke
# episode.

# %%
SPLIT = "2025-01-01"
train, test = data[data.index < SPLIT], data[data.index >= SPLIT]
X_train, y_train = train.drop(columns="pm25_tomorrow"), train["pm25_tomorrow"]
X_test, y_test = test.drop(columns="pm25_tomorrow"), test["pm25_tomorrow"]
print(f"Train: {len(train)} days ({train.index.min().date()} to {train.index.max().date()})")
print(f"Test:  {len(test)} days ({test.index.min().date()} to {test.index.max().date()})")

# %% [markdown]
# ## 4. Baseline and models
#
# - **Persistence baseline:** tomorrow = today. Simple, and genuinely hard to beat.
# - **Ridge regression** (linear regression with regularization, on scaled features).
# - **Gradient-boosted trees** (`HistGradientBoostingRegressor`), in a default and a
#   small/conservative version — with only ~900 training days, smaller trees can
#   overfit less.
#
# To choose between models **without peeking at the test year**, we use 5-fold
# time-series cross-validation inside the training period: each fold trains on
# earlier days and validates on the days right after them.

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

candidates = {
    "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Gradient-boosted trees (default)": HistGradientBoostingRegressor(random_state=0),
    "Gradient-boosted trees (small)": HistGradientBoostingRegressor(
        max_leaf_nodes=7, learning_rate=0.05, max_iter=300, random_state=0),
}
tscv = TimeSeriesSplit(n_splits=5)
cv_mae = {}
for name, model in candidates.items():
    scores = cross_val_score(model, X_train, y_train, cv=tscv,
                             scoring="neg_mean_absolute_error")
    cv_mae[name] = -scores.mean()
# persistence needs no training, so just score it on the same validation days
persist_cv = np.mean([mean_absolute_error(y_train.iloc[va], X_train["pm25_today"].iloc[va])
                      for _, va in tscv.split(X_train)])
cv_table = pd.Series({**cv_mae, "Persistence baseline": persist_cv},
                     name="CV MAE on training period (µg/m³)").sort_values().round(2)
cv_table

# %% [markdown]
# The winner of cross-validation becomes **AirCast's model**. Only now do we touch
# the 2025 test year — every model gets exactly one shot at it.

# %%
best_name = cv_table.drop("Persistence baseline").idxmin()
preds = {"Persistence (tomorrow = today)": X_test["pm25_today"]}
fitted = {}
for name, model in candidates.items():
    fitted[name] = model.fit(X_train, y_train)
    preds[name] = pd.Series(model.predict(X_test), index=X_test.index)

for name, p in preds.items():
    assert np.isfinite(p).all(), f"non-finite predictions from {name}"

results = pd.DataFrame({
    name: {"MAE (µg/m³)": mean_absolute_error(y_test, p),
           "RMSE (µg/m³)": root_mean_squared_error(y_test, p)}
    for name, p in preds.items()
}).T.round(2)
print(f"Model chosen by cross-validation: {best_name}")
results

# %% [markdown]
# ### Test year: forecast vs. reality

# %%
best = preds[best_name]
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(y_test.index, y_test, lw=1, color="#111827", label="Actual (next day)")
ax.plot(best.index, best, lw=1, color="#2563eb", alpha=0.85, label=f"Forecast: {best_name}")
ax.axhline(WHO_DAILY, color="#d97706", ls="--", lw=1, label="WHO guideline")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("Held-out test period (2025): next-day forecast vs. actual")
ax.legend(frameon=False)
fig.autofmt_xdate()
fig.tight_layout(); fig.savefig("figures/02_test_forecast.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. From forecast to decision: the outdoor-activity warning
#
# The athletics director doesn't need a number — they need a yes/no by the end of
# the school day. We issue a **warning** when predicted PM2.5 for tomorrow crosses
# the WHO guideline (15 µg/m³), and score the warnings themselves:
#
# - **Recall** — of the days that really were bad, how many did we warn about?
#   (A missed warning means kids exercise outside in bad air.)
# - **Precision** — of our warnings, how many were right?
#   (Too many false alarms and people stop listening.)
#
# There is a real trade-off here, and we resolve it the way weather services do —
# with **two alert levels**:
#
# - **WARNING** (act: move practice indoors): predicted PM2.5 above the WHO
#   guideline itself. Tuned for precision — when it fires, it's usually right.
# - **WATCH** (be ready: have an indoor backup plan): a more sensitive trigger that
#   we tune on the **2024 validation year** — the lowest trigger that would have
#   caught at least 85% of 2024's bad days. It fires more often and is sometimes
#   wrong, but a watch costs little; a missed bad day (a child with asthma at
#   outdoor practice in smoke) costs a lot.
#
# The trigger is frozen on 2024 data *before* we score the 2025 test year.

# %%
from sklearn.metrics import precision_score, recall_score

def warning_scores(y_true, y_pred, trigger, threshold=WHO_DAILY):
    actual, warned = y_true > threshold, y_pred > trigger
    return {"Warnings issued": int(warned.sum()),
            "Bad days (actual)": int(actual.sum()),
            "Recall (bad days caught)": recall_score(actual, warned),
            "Precision (warnings correct)": precision_score(actual, warned)}

# Tune the trigger on 2024 only, with a model that never saw 2024
val_split = "2024-01-01"
v_train = data[data.index < val_split]
v_val = data[(data.index >= val_split) & (data.index < SPLIT)]
val_model = candidates[best_name].fit(v_train.drop(columns="pm25_tomorrow"), v_train["pm25_tomorrow"])
v_pred = pd.Series(val_model.predict(v_val.drop(columns="pm25_tomorrow")), index=v_val.index)
v_actual_bad = v_val["pm25_tomorrow"] > WHO_DAILY
trigger = WHO_DAILY
for t in np.arange(15.0, 8.0, -0.5):
    if recall_score(v_actual_bad, v_pred > t) >= 0.85:
        trigger = t
        break
print(f"WATCH trigger chosen on 2024 validation: predicted PM2.5 > {trigger} µg/m³")

# Refit on the full training period and score both alert levels on the 2025 test year
model = candidates[best_name].fit(X_train, y_train)
best = pd.Series(model.predict(X_test), index=X_test.index)
warning_table = pd.DataFrame({
    f"AirCast WATCH (trigger {trigger:g})": warning_scores(y_test, best, trigger),
    f"AirCast WARNING (trigger {WHO_DAILY:g})": warning_scores(y_test, best, WHO_DAILY),
    "Persistence (warn if today was bad)": warning_scores(y_test, X_test["pm25_today"], WHO_DAILY),
}).T
warning_table.round(2)

# %% [markdown]
# Reading this honestly: the **WARNING** level is more precise than persistence
# (fewer false alarms for the same kind of signal), while the **WATCH** level
# catches far more of the genuinely bad days than either — at the cost of telling
# the director to have a backup plan on days that turn out fine. That cost
# asymmetry is exactly why real advisories use two levels.

# %% [markdown]
# ## 6. Case study: would AirCast have warned before the June 2023 smoke crisis?
#
# On June 27–28, 2023, wildfire smoke gave Chicago some of the worst air in the
# world. This is exactly the day our user needs the tool to work — so we test it
# honestly: we retrain the model using **only data available before June 2023**
# and replay the event day by day.

# %%
from sklearn.base import clone

cs_train = data[data.index < "2023-06-01"]
cs_test = data[("2023-06-01" <= data.index) & (data.index <= "2023-07-15")]
cs_model = clone(candidates[best_name]).fit(
    cs_train.drop(columns="pm25_tomorrow"), cs_train["pm25_tomorrow"])
cs_pred = pd.Series(cs_model.predict(cs_test.drop(columns="pm25_tomorrow")), index=cs_test.index)
cs_persist = cs_test["pm25_today"]
cs_actual = cs_test["pm25_tomorrow"]

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(cs_actual.index, cs_actual, "o-", ms=3, lw=1.2, color="#111827", label="Actual (next day)")
ax.plot(cs_pred.index, cs_pred, "o-", ms=3, lw=1.2, color="#2563eb", label="AirCast forecast")
ax.plot(cs_persist.index, cs_persist, "o-", ms=3, lw=1, color="#9ca3af", label="Persistence")
ax.axhline(WHO_DAILY, color="#d97706", ls="--", lw=1)
ax.axhline(EPA_DAILY, color="#dc2626", ls="--", lw=1)
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("June–July 2023 smoke event, replayed with a model trained only on pre-June data")
ax.legend(frameon=False)
fig.autofmt_xdate()
fig.tight_layout(); fig.savefig("figures/03_case_study.png", bbox_inches="tight")
plt.show()

event = pd.DataFrame({"actual_tomorrow": cs_actual, "aircast": cs_pred.round(1),
                      "persistence": cs_persist.round(1)}).loc["2023-06-24":"2023-07-01"]
event["aircast_warns"] = event["aircast"] > WHO_DAILY
event["persistence_warns"] = event["persistence"] > WHO_DAILY
event

# %% [markdown]
# ## 7. How far ahead does the forecast stay useful?
#
# We retrain the same model to predict 1, 2 and 3 days ahead and compare each
# horizon against persistence at that horizon.

# %%
horizon_rows = {}
for h in [1, 2, 3]:
    tgt = df["pm25"].shift(-h)
    d_h = features.join(tgt.rename("y")).dropna()
    tr, te = d_h[d_h.index < SPLIT], d_h[d_h.index >= SPLIT]
    m = clone(candidates[best_name]).fit(tr.drop(columns="y"), tr["y"])
    horizon_rows[f"+{h} day"] = {
        "Model MAE": mean_absolute_error(te["y"], m.predict(te.drop(columns="y"))),
        "Persistence MAE": mean_absolute_error(te["y"], te["pm25_today"]),
    }
horizons = pd.DataFrame(horizon_rows).T.round(2)
horizons["Improvement"] = (1 - horizons["Model MAE"] / horizons["Persistence MAE"]).map("{:.0%}".format)

ax = horizons[["Model MAE", "Persistence MAE"]].plot.bar(
    rot=0, color=["#2563eb", "#9ca3af"], figsize=(6.5, 3.6))
ax.set_ylabel("MAE on 2025 test year (µg/m³)")
ax.set_title("Forecast error by horizon")
ax.legend(frameon=False)
plt.tight_layout(); plt.savefig("figures/04_horizons.png", bbox_inches="tight")
plt.show()
horizons

# %% [markdown]
# ## 8. What drives tomorrow's air quality in Chicago?

# %%
from sklearn.inspection import permutation_importance

imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0,
                             scoring="neg_mean_absolute_error")
imp_series = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()

fig, ax = plt.subplots(figsize=(7, 4.5))
imp_series.plot.barh(ax=ax, color="#2563eb")
ax.set_xlabel("Increase in MAE when feature is shuffled (µg/m³)")
ax.set_title("Permutation importance on the 2025 test year")
fig.tight_layout(); fig.savefig("figures/05_importance.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Where the model fails

# %%
errors = (best - y_test).rename("error")
worst = errors.abs().nlargest(8).index
fail = pd.DataFrame({"actual": y_test[worst], "forecast": best[worst].round(1),
                     "error": errors[worst].round(1)}).sort_values("error")
fail

# %% [markdown]
# Every one of the worst misses is an **underestimate on the first day of a smoke
# episode**. That makes physical sense: wildfire smoke arrives from hundreds of
# kilometers away, and nothing in *Chicago's own* measurements today can see it
# coming. So instead of stopping at "that's our limitation," we tried to fix it.
#
# ## 10. Fixing our biggest failure: an upwind smoke signal
#
# Canadian wildfire smoke reaches Chicago from the north and northwest. Air that
# will be over Chicago tomorrow is over **Minneapolis, Green Bay, Madison and
# Thunder Bay today** — and their PM2.5 today is public data we can legitimately
# use at prediction time (no leakage: it's all measurable before the forecast is
# made). We add 9 features: each upwind city's PM2.5 today, its one-day change,
# and the maximum across the four cities.
#
# **Honesty note:** we designed this feature *after* studying the model's failures
# on the 2025 test year, which means the test year is no longer perfectly
# untouched for this comparison. That's why the June 2023 event backtest below
# matters most: that model is trained only on data from before June 2023, so the
# event is a fully independent check of the idea.

# %%
up = pd.read_csv("data/upwind.csv", parse_dates=["date"], index_col="date")
UPWIND_CITIES = ["minneapolis", "green_bay", "madison", "thunder_bay"]

def add_upwind(f):
    f = f.copy()
    for c in UPWIND_CITIES:
        f[f"up_{c}"] = up[c]
        f[f"up_{c}_chg"] = up[c] - up[c].shift(1)
    f["up_max"] = up[UPWIND_CITIES].max(axis=1)
    return f

data_up = add_upwind(features).join(target).dropna()
tr_up, te_up = data_up[data_up.index < SPLIT], data_up[data_up.index >= SPLIT]
model_up = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
    tr_up.drop(columns="pm25_tomorrow"), tr_up["pm25_tomorrow"])
pred_up = pd.Series(model_up.predict(te_up.drop(columns="pm25_tomorrow")), index=te_up.index)

# "Onset days": test days where air quality suddenly deteriorated
# (tomorrow's actual PM2.5 at least 10 ug/m3 worse than today's)
onset = y_test.index[(y_test - X_test["pm25_today"]) > 10]
compare = pd.DataFrame({
    "Persistence": X_test["pm25_today"], "Local only": best, "Local + upwind": pred_up})
upwind_results = pd.DataFrame({
    name: {"Test MAE (µg/m³)": mean_absolute_error(y_test, p),
           f"MAE on {len(onset)} sudden-onset days": mean_absolute_error(y_test[onset], p[onset])}
    for name, p in compare.items()
}).T.round(2)
upwind_results

# %% [markdown]
# ### The independent check: June 2023, replayed with upwind eyes

# %%
cs_tr_up = data_up[data_up.index < "2023-06-01"]
cs_te_up = data_up[("2023-06-01" <= data_up.index) & (data_up.index <= "2023-07-15")]
cs_model_up = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
    cs_tr_up.drop(columns="pm25_tomorrow"), cs_tr_up["pm25_tomorrow"])
cs_pred_up = pd.Series(cs_model_up.predict(cs_te_up.drop(columns="pm25_tomorrow")),
                       index=cs_te_up.index)

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(cs_actual.index, cs_actual, "o-", ms=3, lw=1.2, color="#111827", label="Actual (next day)")
ax.plot(cs_pred_up.index, cs_pred_up, "o-", ms=3, lw=1.2, color="#16a34a", label="AirCast + upwind signal")
ax.plot(cs_pred.index, cs_pred, "o-", ms=3, lw=1, color="#2563eb", alpha=0.6, label="AirCast local-only")
ax.axhline(WHO_DAILY, color="#d97706", ls="--", lw=1)
ax.axhline(EPA_DAILY, color="#dc2626", ls="--", lw=1)
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("June 2023 smoke event: the upwind signal fires the warning a day earlier")
ax.legend(frameon=False)
fig.autofmt_xdate()
fig.tight_layout(); fig.savefig("figures/06_upwind_case_study.png", bbox_inches="tight")
plt.show()

key_day = pd.Timestamp("2023-06-26")
print(f"Forecast made June 26 for June 27 (actual {cs_actual[key_day]:.1f} µg/m³):")
print(f"  local-only model:  {cs_pred[key_day]:.1f}  -> {'WARNS' if cs_pred[key_day] > WHO_DAILY else 'no warning'}")
print(f"  + upwind signal:   {cs_pred_up[key_day]:.1f}  -> {'WARNS' if cs_pred_up[key_day] > WHO_DAILY else 'no warning'}")

# %% [markdown]
# The upwind model still underestimates the peak's *size* — but it crosses the
# warning line on June 26, so the athletics director gets the right instruction
# (**move practice indoors**) on the day the local-only model and persistence both
# said "all clear." For a warning system, being right about the *decision* matters
# more than being right about the exact number.
#
# ### Final alert system, re-tuned with the upwind model
#
# Same procedure as section 5: WATCH trigger chosen on the 2024 validation year
# (lowest trigger catching ≥85% of 2024's bad days), frozen, then scored on 2025.

# %%
vu_train = data_up[data_up.index < val_split]
vu_val = data_up[(data_up.index >= val_split) & (data_up.index < SPLIT)]
vu_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
    vu_train.drop(columns="pm25_tomorrow"), vu_train["pm25_tomorrow"])
vu_pred = pd.Series(vu_model.predict(vu_val.drop(columns="pm25_tomorrow")), index=vu_val.index)
vu_bad = vu_val["pm25_tomorrow"] > WHO_DAILY
trigger_up = WHO_DAILY
for t in np.arange(15.0, 8.0, -0.5):
    if recall_score(vu_bad, vu_pred > t) >= 0.85:
        trigger_up = t
        break
print(f"Final WATCH trigger (tuned on 2024): predicted PM2.5 > {trigger_up} µg/m³")

final_alerts = pd.DataFrame({
    f"AirCast WATCH (trigger {trigger_up:g})": warning_scores(y_test, pred_up, trigger_up),
    f"AirCast WARNING (trigger {WHO_DAILY:g})": warning_scores(y_test, pred_up, WHO_DAILY),
    "Persistence (warn if today was bad)": warning_scores(y_test, X_test["pm25_today"], WHO_DAILY),
}).T
final_alerts.round(2)

# %% [markdown]
# ## 11. How sure is each forecast? Prediction intervals
#
# A single number hides uncertainty the decision-maker needs. We use **split
# conformal prediction** — simple and statistically honest: take the model's
# errors on the 2024 validation year, find the spread that contains 80% of them,
# and quote that same spread around every 2025 forecast. If the method is honest,
# about 80% of 2025's actual values should land inside the interval — a claim we
# can check.

# %%
val_resid = vu_val["pm25_tomorrow"] - vu_pred
lo_q, hi_q = val_resid.quantile(0.10), val_resid.quantile(0.90)
covered = ((y_test >= pred_up + lo_q) & (y_test <= pred_up + hi_q)).mean()
print(f"80% interval from 2024 residuals: forecast {lo_q:+.1f} to {hi_q:+.1f} µg/m³")
print(f"Coverage on 2025 test year: {covered:.0%} of actual values fell inside "
      f"(target: 80%)")

# %% [markdown]
# Coverage came in at **72%, short of the promised 80%** — and the reason is
# instructive: 2025 had more wildfire-smoke days than 2024, the year the interval
# was calibrated on, and smoke days produce the largest errors. Uncertainty
# estimates calibrated in a calm year understate risk in a smoky one. A deployed
# system should recalibrate on a rolling window; we report the miss rather than
# hide it.

# %% [markdown]
# ## 12. Does it generalize? Chicago's model, tested on Milwaukee
#
# We take the final model — trained **only on Chicago** — and apply it, with no
# retraining at all, to Milwaukee (150 km north, never seen in training).

# %%
mke = pd.read_csv("data/milwaukee.csv", parse_dates=["date"], index_col="date")
mke_data = add_upwind(build_features(mke)).join(
    mke["pm25"].shift(-1).rename("pm25_tomorrow")).dropna()
mke_test = mke_data[mke_data.index >= SPLIT]
mke_pred = pd.Series(model_up.predict(mke_test.drop(columns="pm25_tomorrow")),
                     index=mke_test.index)
mke_y = mke_test["pm25_tomorrow"]
mke_train = mke_data[mke_data.index < SPLIT]
mke_retrained = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
    mke_train.drop(columns="pm25_tomorrow"), mke_train["pm25_tomorrow"])
mke_pred_rt = pd.Series(mke_retrained.predict(mke_test.drop(columns="pm25_tomorrow")),
                        index=mke_test.index)
transfer = pd.DataFrame({
    "Chicago-trained model, applied as-is": {
        "MAE on Milwaukee 2025 (µg/m³)": mean_absolute_error(mke_y, mke_pred)},
    "Same pipeline, retrained on Milwaukee": {
        "MAE on Milwaukee 2025 (µg/m³)": mean_absolute_error(mke_y, mke_pred_rt)},
    "Milwaukee persistence baseline": {
        "MAE on Milwaukee 2025 (µg/m³)": mean_absolute_error(mke_y, mke_test["pm25_today"])},
}).T.round(2)
transfer

# %% [markdown]
# An honest negative result: the Chicago-trained weights do **not** beat
# Milwaukee's own persistence baseline — each city's pollution has its own local
# character. But the *pipeline* transfers: retrained on Milwaukee's data, the
# same code beats Milwaukee persistence. The method generalizes; the weights are
# local, which is exactly how a multi-city deployment should work anyway.

# %% [markdown]
# ## 13. Conclusions, honestly

# %%
p_mae = results.loc["Persistence (tomorrow = today)", "MAE (µg/m³)"]
u_mae = upwind_results.loc["Local + upwind", "Test MAE (µg/m³)"]
watch, warn = final_alerts.iloc[0], final_alerts.iloc[1]
print(f"Final model: Ridge regression, local + upwind features")
print(f"Next-day MAE on 2025 test year: {u_mae} µg/m³ vs persistence {p_mae} µg/m³ "
      f"({1 - u_mae / p_mae:.0%} better)")
print(f"WATCH level:   caught {watch['Recall (bad days caught)']:.0%} of bad days "
      f"({watch['Precision (warnings correct)']:.0%} of watches correct)")
print(f"WARNING level: caught {warn['Recall (bad days caught)']:.0%} of bad days "
      f"({warn['Precision (warnings correct)']:.0%} of warnings correct)")
print(f"June 2023 backtest: upwind signal fires the June 27 warning that the "
      f"local-only model missed")

# %% [markdown]
#
# **Where it still fails, and why:**
# - **Event magnitude.** Even with upwind eyes, the model underestimates how *bad*
#   peak smoke days get — it now gets the decision right (warn) but not the number.
#   Satellite smoke-plume data (NOAA HMS) is the next signal to add.
# - **Clean-up days.** After an event the model stays pessimistic for a day while
#   the upwind cities are still polluted.
#
# **Limitations we want judges to know about:**
# - PM2.5 values are CAMS *model estimates*, not monitor readings (the brief's
#   suggested fallback). The forecasting task is identical, but absolute levels can
#   differ from EPA monitors; a deployed version would retrain on AirNow data.
# - The upwind-feature comparison was motivated by failures we saw on the test
#   year, so we lean on the independent June 2023 backtest as its honest check.
# - Built for Chicago; the Milwaukee transfer test is encouraging, but we make no
#   claims beyond the cities we measured.
# - This is a student prototype, not an official air-quality advisory.
