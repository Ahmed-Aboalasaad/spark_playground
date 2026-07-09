# NYC Taxi Analytics with Apache Spark

## What This Is
Interactive Streamlit app demonstrating Apache Spark for distributed analytics and ML on the NYC Yellow Taxi Trip dataset (Parquet). Purpose is to **showcase Spark capabilities**, not to be a production data platform.

## Stack
| Component | Technology |
|---|---|
| Engine | Apache Spark |
| Language | Python |
| ML | Spark MLlib |
| UI | Streamlit |
| Viz | Plotly / Matplotlib |
| Data format | Parquet |

## Core Requirements (non-negotiable)
- ALL data processing goes through Spark — no pandas for core transforms.
- Dataset loads from one or more Parquet files.
- UI layer never touches Spark directly (see ARCHITECTURE.md).
- Every long-running operation reports execution time.
- New analyses/models must be addable without modifying existing modules (see Extensibility).

## Scope (4 pages)
1. **Home** — load dataset, show schema/stats, Spark session info.
2. **Analysis** — run analytical queries, show viz + metrics + timing.
3. **Modeling** — select/configure/train/evaluate Spark MLlib regression models.
4. **Spark Insights** — execution time, partitions, cache status, storage level, query plans.

Full feature detail: see FEATURES.md.
Full layer/module detail: see ARCHITECTURE.md.

## Out of Scope
Streaming, cluster deployment, orchestration, REST APIs, auth, multi-user, deep learning, prod monitoring, CI/CD, cloud deployment.

## Done When
- Dataset loads; stats explorable.
- Analytical queries run and visualize.
- Multiple MLlib models train and evaluate; metrics comparable.
- Spark execution info inspectable.
- Every task shows execution time.

## Design Principles
Spark-first · modular pipeline · processing/presentation separated · reproducible workflows · extensible by addition, not modification.
