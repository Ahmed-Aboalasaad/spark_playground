# Architecture

Layered architecture: UI never computes, Spark never renders. One shared SparkSession.

```
Streamlit UI  →  Application Services  →  Spark Pipeline  →  SparkSession  →  Parquet Files
```

## Layer Contracts

| Layer | Owns | Must NOT do |
|---|---|---|
| **Presentation** (Streamlit) | Navigation, input collection, rendering charts/metrics/errors, status reporting | Any Spark computation |
| **Application Service** | Orchestrate workflows, validate input, invoke pipeline, collect exec metrics, shape results for UI | Business-specific data transforms |
| **Spark Pipeline** | All DataFrame ops: load, clean, analyze, feature-engineer, train, evaluate | Rendering / UI concerns |
| **Data** | NYC Taxi Parquet files | — |

Rule of thumb for agents: if code imports `pyspark`, it belongs in the Spark Pipeline layer, not in Streamlit page code.

## Pipeline Modules

| Module | Responsibilities | Output |
|---|---|---|
| Dataset Loader | Build DataFrame from Parquet, validate availability/schema, report load stats | Raw DataFrame |
| Data Cleaning | Remove invalid records, handle missing values, convert dtypes, derive columns | Clean DataFrame |
| Analysis Engine | Run analytical queries, compute metrics, produce viz-ready aggregates | Aggregated DataFrames + metrics |
| Feature Engineering | Select features, encode categoricals, assemble vectors, split train/test | Train/test datasets |
| Machine Learning | Train/save/load MLlib models, generate predictions | Trained models + predictions |
| Evaluation | Compute RMSE/MAE/R², comparison reports | Metrics |

Each module is independent and addable/replaceable without touching the others (extensibility requirement).

## Shared SparkSession
Single session for the app lifetime; owns dataset reads, SQL execution, DataFrame/cache management, ML pipelines. All modules reuse it — never create a second session.

## Data Flow

```
Load Dataset → Raw DataFrame
      ├─→ Home Statistics
      ├─→ Data Quality
      └─→ Analysis (Raw)

Raw DataFrame → Data Preprocessing → Clean DataFrame
      ├─→ Home Statistics
      ├─→ Data Quality
      └─→ Analysis (Cleaned)

Clean DataFrame → Feature Engineering → Machine Learning → Model Evaluation
```

Rules:
- Home / Data Preprocessing / Analysis operate on **whichever dataset is currently selected** (Raw or Cleaned).
- Machine Learning **always** uses the Cleaned dataset — no exceptions.

## State (held in Application Service layer)
Spark session · raw dataset · cleaned dataset · cached datasets · current dataset selection · cleaning summary · trained models · analysis results · execution metrics.

Reuse state to avoid recomputation — don't reload/reclean on every page render.

## Error Handling
Each layer surfaces errors to the layer above; UI shows actionable messages and never crashes on a caught error.

## Extension Pattern
Add new analyses / models / datasets / UI sections as **new modules**. Do not modify existing modules to add functionality.

## Design Goals
Maintainability · readability · reusability · modularity · scalability · separation of concerns · ease of extension.
