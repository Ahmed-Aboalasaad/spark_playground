# NYC Taxi Analytics with Apache Spark

Interactive analytics and machine-learning application over the NYC Yellow Taxi
dataset, built to showcase Apache Spark as a unified platform for distributed
processing, analytics, and MLlib modeling. The UI is Streamlit; visualization
is Plotly.

> **Implementation status.** Dataset loading, **data cleaning**, **data-quality
> reporting**, **feature engineering**, and all **25 analyses** run real Spark
> computations (ported from `notebooks/`). The **Data Preprocessing** and
> **Analysis** pages are fully live. **Modeling** (MLlib train/evaluate) is still
> placeholder-backed — `config.PLACEHOLDER_MODE` now gates only that stage.
> Execution timings are real throughout.

## Running

```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

Then, from the Home page, click **Load dataset**. With no parquet files present
the app still runs fully on placeholder data. To use real data, drop monthly
files into `data/` (see `data/README.md`) or set `NYC_TAXI_DATA_DIR`.

## Running with Docker

No local Python/Java/Spark setup required — the image bundles Java 17 and
every Python dependency.

```bash
docker compose up --build
```

Then open http://localhost:8501. Downloaded parquet files and trained models
persist in named volumes (`taxi-data`, `taxi-models`) across restarts.

Without Compose:

```bash
docker build -t nyc-taxi-spark .
docker run -p 8501:8501 -v taxi-data:/app/data -v taxi-models:/app/models nyc-taxi-spark
```

Nothing in the image needs internet access to *start*: the dataset is fetched
on demand from the Home page's control panel once the app is running. Useful
overrides (pass as `-e KEY=VALUE` or under `environment:` in compose):

| Variable | Default | Purpose |
|---|---|---|
| `NYC_TAXI_DRIVER_MEMORY` | `4g` | Spark driver heap; lower on a memory-constrained host |
| `NYC_TAXI_DATA_DIR` | `/app/data` | Where parquet files are read/written |
| `NYC_TAXI_MODELS_DIR` | `/app/models` | Where trained models are saved |

### GPU acceleration (optional)

GPU-accelerated XGBoost (`device="cuda"`) works inside the container as-is —
no separate CUDA base image needed, since the NVIDIA driver, `nvidia-smi`, and
CUDA libraries are injected by the container runtime at `docker run` time. It
isn't required: the Modeling page falls back to CPU automatically when no GPU
is visible.

Requirements: an NVIDIA GPU + up-to-date driver. On Docker Desktop
(Windows/Mac, WSL2 backend) that's the only requirement — GPU passthrough is
built in. On native Linux Docker you also need the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
# or, without Compose:
docker run --gpus all -p 8501:8501 -v taxi-data:/app/data -v taxi-models:/app/models nyc-taxi-spark
```

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
