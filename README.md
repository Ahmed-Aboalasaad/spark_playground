# NYC Taxi Analytics with Apache Spark

Interactive analytics and machine-learning application over the NYC Yellow Taxi
dataset, built to showcase Apache Spark as a unified platform for distributed
processing, analytics, and MLlib modeling. The UI is Streamlit; visualization
is Plotly.

> **Skeleton status.** This is a working application skeleton. Every pipeline
> module currently returns **placeholder data** — correctly shaped, but with all
> numeric values set to zero and empty charts — so the entire app is navigable
> and the wiring is testable end-to-end. Real Spark implementations drop into the
> pipeline modules without touching the UI or services. Execution timings are
> **real** even in placeholder mode.

## Running

```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

Then, from the Home page, click **Load dataset**. With no parquet files present
the app still runs fully on placeholder data. To use real data, drop monthly
files into `data/` (see `data/README.md`) or set `NYC_TAXI_DATA_DIR`.

## Offline checks

The pure-Python layers (config, timing, mock data, zone lookup, registries)
have no Spark/Streamlit dependency and can be verified directly:

```bash
python smoke_test.py
```

## Architecture

Layered, with a strict separation between presentation and Spark processing:

```
Streamlit pages  (app/)            UI only — no Spark calls
      │
Application services  (services/)  orchestration + timing, no transforms
      │
Spark pipeline  (pipeline/)        all DataFrame logic lives here
      │
Shared SparkSession  (spark/)      one session, reused everywhere
      │
Parquet files  (data/)  +  zone lookup (reference/)
```

### Layout

| Path | Role |
|------|------|
| `config.py` | Paths, Spark defaults, schema, domain constants, `PLACEHOLDER_MODE` |
| `spark/session.py` | Single cached `SparkSession` factory + session info |
| `pipeline/loader.py` | Load raw dataset from parquet |
| `pipeline/cleaning.py` | Cleaning + data-quality reporting |
| `pipeline/zones.py` | Zone lookup, airport detection, ID→name resolution |
| `pipeline/analysis.py` | Registry of 25 analyses across 6 families |
| `pipeline/features.py` | Feature engineering / train-test split |
| `pipeline/ml.py` | Registry of 4 MLlib regressors + hyperparameters |
| `pipeline/evaluation.py` | RMSE / MAE / R² + feature importance |
| `pipeline/mock.py` | Shared zeroed-but-shaped placeholder generators |
| `services/` | `timing.py`, `state.py` (AppState), `services.py` (workflows) |
| `app/` | `Home.py` entrypoint + `pages/` (Preprocessing, Analysis, Modeling, Spark Insights) |

## How the placeholder contract works

`config.PLACEHOLDER_MODE = True` makes every pipeline module return mock data.
Mock frames have the right columns and row counts (24 hours, 7 weekdays, 12
months, 3 airports, …) with zeroed values; the UI shows a loud banner and a
per-result badge. To implement a real analysis or model, replace the relevant
producer/function in the pipeline module — the service layer, registries, and
pages need no changes. Flip `PLACEHOLDER_MODE` off once the pipeline is real.

## Extending

- **New analysis:** register one `Analysis` entry in `pipeline/analysis.py`. It
  appears in the Analysis page automatically.
- **New model:** register one `ModelSpec` in `pipeline/ml.py`. Its
  hyperparameter controls render automatically on the Modeling page.

## Notes for the team

- The regression target is `fare_amount` (`config.ML_TARGET_COLUMN`). Adding a
  second target later is a one-line change plus a selector.
- **Airport detection quirk:** in the zone lookup, JFK and LaGuardia use
  `service_zone == "Airports"` but **Newark uses `service_zone == "EWR"`**.
  Airport detection matches both (`config.AIRPORT_SERVICE_ZONES`) so Newark
  isn't silently dropped. Whoever writes the real airport analysis should keep
  this in mind.
- Location IDs 264 ("Unknown") and 265 ("Outside of NYC") are sentinels, not
  geography — see `config.UNKNOWN_LOCATION_IDS`.
- `spark.sql.shuffle.partitions` is set to 8 for local work (Spark's default of
  200 is far too many for a few months of data). Override in `config.py`.
```
