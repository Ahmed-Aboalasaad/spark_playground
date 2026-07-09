# Features

Functional spec for the 4 pages. See ARCHITECTURE.md for which layer implements what.

---

## 1. Home

**Project overview**: description, objectives, dataset description, tech stack (static content).

**Dataset loading**
- Select directory of Parquet files → load into Spark.
- Reload action.
- Show progress, load time, status/errors.

**Dataset info (post-load)**: record count, column count, schema + dtypes, # Parquet files, dataset size, date range.

**Dataset statistics**: missing values/column, numeric summary stats, categorical distributions, distinct-value counts for selected columns.

**Spark session info**: version, app name, master config, default partition count, cache status.

**Dataset selector (sidebar, global)**
- Two dataset states exist app-wide: **Raw** (post-load, no processing) and **Cleaned** (post-pipeline).
- Selector switches active dataset; all pages default to operating on whichever is selected.
- Active dataset always visibly indicated.

**Data Preprocessing sub-page**: data-quality overview + cleaning ops for the *currently selected* dataset.
- Missing values (% and count) per column.
- Duplicate record count.
- Invalid/outlier distribution for selected features.
- Before/after summary stats.
- # and % records removed by cleaning.
- List of cleaning operations applied.
- Each viz includes: chart, summary metrics, records processed, execution time.
- When Cleaned is selected: highlight diffs introduced by cleaning.

---

## 2. Analysis

Every analysis returns: **visualization + summary metrics + execution time + records processed**.

| Category | Queries |
|---|---|
| Demand | trips by hour / weekday / month / over time |
| Geographic | top pickup locations, top dropoff locations, pickup vs dropoff hotspots |
| Trip | distance distribution, duration distribution, avg duration, avg distance, avg speed |
| Revenue | total revenue, revenue by hour/weekday/month, avg fare, avg tip |
| Passenger | count distribution, avg fare by passenger count, avg distance by passenger count |
| Airport | trips from airport, trips to airport, airport revenue, traffic trends |

---

## 3. Modeling

**Model selection**: Linear Regression, Decision Tree Regressor, Random Forest Regressor, GBT Regressor (extensible list).

**Hyperparameters**: only params relevant to selected model are shown/configurable.

**Training**: trigger, progress, training time, cancel (if supported). Always trains on **Cleaned** dataset.

**Evaluation (post-training)**: RMSE, MAE, R², prediction samples, feature importance (when available).

**Model management**: save trained model, load saved model, re-evaluate a loaded model.

---

## 4. Spark Insights

Available per processing task: execution time, partition count, cache status, storage level, logical plan, physical plan.

---

## Cross-cutting requirements

**UX**: collapsible sidebar nav, progress indicators on long ops, success/warning/error notifications, responsive charts, consistent layout, clearly labeled metrics.

**Performance reporting** (mandatory for): dataset load, dataset cleaning, every analysis query, model train, model eval, model save/load. Always displayed alongside results.

**Extensibility**: new analyses, new MLlib models, new datasets, new UI sections — all addable without editing existing modules.
