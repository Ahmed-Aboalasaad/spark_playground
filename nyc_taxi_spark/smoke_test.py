"""Offline smoke test for the pure-Python layers.

Exercises everything that does not require Spark or Streamlit: config,
timing helpers, mock generators, the zone lookup, and the analysis/model
registries. Run with ``python smoke_test.py`` from the repo root.
"""
from __future__ import annotations

import sys


def check(name: str, cond: bool) -> None:
    status = "ok  " if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        sys.exit(1)


def main() -> None:
    # -- config ----------------------------------------------------------- #
    import config
    check("config: 19 expected columns", len(config.EXPECTED_COLUMNS) == 19)
    check("config: shuffle partitions sized for local multi-month loads",
          config.SPARK_CONFIG.shuffle_partitions == 64)
    conf = config.SPARK_CONFIG.as_spark_conf()
    check("config: spark conf flattened",
          conf["spark.sql.shuffle.partitions"] == "64"
          and conf["spark.sql.execution.arrow.pyspark.enabled"] == "true"
          and conf["spark.driver.memory"] == config.SPARK_CONFIG.driver_memory
          and conf["spark.sql.adaptive.enabled"] == "true")
    check("config: ML target is fare_amount", config.ML_TARGET_COLUMN == "fare_amount")

    # -- timing ----------------------------------------------------------- #
    from services.timing import Result, timed, timer
    with timer() as t:
        sum(range(1000))
    check("timing: context manager records elapsed", t.elapsed >= 0.0)

    @timed
    def work():
        return 42
    r = work()
    check("timing: decorator returns Result", isinstance(r, Result) and r.value == 42)
    check("timing: elapsed_str formats", r.elapsed_str.endswith(" s"))

    # -- mock generators -------------------------------------------------- #
    from pipeline import mock
    check("mock: hours frame has 24 rows", len(mock.hours_frame()) == 24)
    check("mock: weekdays frame has 7 rows", len(mock.weekdays_frame()) == 7)
    check("mock: months frame has 12 rows", len(mock.months_frame()) == 12)
    check("mock: all values zero", mock.hours_frame()["trips"].sum() == 0)
    check("mock: top_n uses placeholder labels",
          mock.PLACEHOLDER_LABEL in mock.top_n_frame(5)["zone"].iloc[0])

    # -- zone lookup ------------------------------------------------------ #
    from pipeline import zones
    lookup = zones.load_zone_lookup()
    check("zones: lookup loaded (265 rows)", len(lookup) == 265)
    airports = zones.airport_location_ids()
    check("zones: exactly 3 airports flagged", len(airports) == 3)
    check("zones: JFK=132, LGA=138, EWR=1 detected",
          set(airports) == {1, 132, 138})
    check("zones: zone_name resolves JFK",
          "JFK" in zones.zone_name(132))
    check("zones: 264/265 flagged unknown",
          zones.is_unknown(264) and zones.is_unknown(265))

    # -- analysis registry ------------------------------------------------ #
    from pipeline import analysis
    fams = analysis.all_families()
    check("analysis: 6 families", len(fams) == 6)
    total = sum(len(analysis.analyses_in(f)) for f in fams)
    check("analysis: 25 analyses registered", total == 25)
    # Producers now run real Spark aggregations (exercised in the Spark-backed
    # test, not here). Offline, we validate each entry's shape: a callable
    # producer, a chart type, a non-empty description, and x/y wired for every
    # non-metric chart.
    for fam in fams:
        for a in analysis.analyses_in(fam):
            assert callable(a.producer), f"{a.key} producer not callable"
            assert a.description, f"{a.key} missing description"
            if a.chart is not analysis.ChartType.METRIC:
                assert a.x and a.y, f"{a.key} non-metric chart missing x/y"
    check("analysis: every entry well-formed (producer, chart, x/y, description)", True)

    # -- model registry --------------------------------------------------- #
    from pipeline import ml
    models = ml.list_models()
    check("ml: 4 models registered", len(models) == 4)
    check("ml: keys are linear/dtree/rforest/gbt",
          {m.key for m in models} == {"linear", "dtree", "rforest", "gbt"})
    out = ml.train_model(models[0], {"regParam": 0.0}, None)
    check("ml: placeholder train returns no model", out["model"] is None)

    # -- evaluation ------------------------------------------------------- #
    from pipeline import evaluation
    rep = evaluation.evaluate(None, None, models[1])  # dtree supports importance
    check("eval: zeroed RMSE/MAE/R2",
          rep["metrics"] == {"RMSE": 0, "MAE": 0, "R2": 0})
    check("eval: feature importance present for tree model",
          "feature_importance" in rep)
    check("eval: importance values zeroed",
          rep["feature_importance"]["importance"].sum() == 0)

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
