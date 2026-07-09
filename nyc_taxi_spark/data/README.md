# Data directory

Place the monthly NYC Yellow Taxi parquet files here, named
`yellow_tripdata_YYYY-MM.parquet` (the loader discovers them via the
`yellow_tripdata_*.parquet` glob).

Files are gitignored. Fetch them with your team's `FetchingNYCTaxiData`
notebook, or point `NYC_TAXI_DATA_DIR` at wherever they already live:

    export NYC_TAXI_DATA_DIR=/path/to/parquet/files

The TLC zone lookup ships with the repo at `reference/taxi_zone_lookup.csv`.
