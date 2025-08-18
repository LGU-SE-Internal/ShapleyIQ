#### Traces

Traces file contains a time series of spans.

|     Column     |   Type   | Description                                              |
| :------------: | :------: | :------------------------------------------------------- |
|      time      | datetime | start time of a span in UTC                              |
|    trace_id    |  string  | unique identifier of a trace (a trace groups many spans) |
|    span_id     |  string  | unique identifier of a span                              |
| parent_span_id |  string  | identifier of the parent span (for trace hierarchy)      |
|  service_name  |  string  | name of the service that generated the span              |
|   span_name    |  string  | name of the operation represented by the span            |
|    duration    |  uint64  | duration of a span in nanoseconds                        |
|     attr.*     |    *     | other attributes of a span                               |

#### Metrics

Metrics file contains a time series of metric values.

|    Column    |   Type   | Description                                         |
| :----------: | :------: | :-------------------------------------------------- |
|     time     | datetime | UTC timestamp of a metric value                     |
|    metric    |  string  | name of the metric value                            |
|    value     | float64  | value of the metric value                           |
| service_name |  string  | name of the service that generated the metric value |
|    attr.*    |    *     | other attributes of a metric value                  |

#### Logs

Logs file contains a time series of log events.

|    Column    |   Type   | Description                                      |
| :----------: | :------: | :----------------------------------------------- |
|     time     | datetime | UTC timestamp of a log event                     |
|   trace_id   |  string  | unique identifier of a trace                     |
|   span_id    |  string  | unique identifier of a span                      |
| service_name |  string  | name of the service that generated the log event |
|    level     |  string  | log level (e.g., INFO, ERROR)                    |
|   message    |  string  | log message                                      |
|    attr.*    |    *     | other attributes of a log event                  |

这是我们的新数据结构

我们新数据load的方法可以这样
```
import datetime
import json
import math
import time
from functools import wraps
from pathlib import Path

import polars as pl
from rcabench_platform.v2.logging import logger


def timeit():
    """Simple timing decorator"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.info(f"{func.__name__} took {end - start:.2f} seconds")
            return result

        return wrapper

    return decorator


def load_json(path: Path) -> dict:
    """Load JSON file"""
    with open(path, "r") as f:
        return json.load(f)


def tt_add_op_name(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add operation name for Train Ticket traces"""
    # This is a simplified version - you might need to adjust based on actual data
    return lf.with_columns(pl.col("span_name").alias("operation_name"))


def replace_enum_values(column: str, enum_values: list, start: int = 0) -> pl.Expr:
    """Replace enum string values with integers"""
    mapping_expr = pl.col(column)
    for i, value in enumerate(enum_values):
        mapping_expr = mapping_expr.str.replace(value, str(start + i))
    return mapping_expr.cast(pl.Int32)


def load_inject_time(input_folder: Path) -> datetime.datetime:
    env = load_json(path=input_folder / "env.json")

    # tz = dateutil.tz.gettz(env["TIMEZONE"])

    normal_start = int(env["NORMAL_START"])
    normal_end = int(env["NORMAL_END"])

    abnormal_start = int(env["ABNORMAL_START"])
    abnormal_end = int(env["ABNORMAL_END"])

    assert normal_start < normal_end <= abnormal_start < abnormal_end

    if normal_end < abnormal_start:
        inject_time = int(math.ceil(normal_end + abnormal_start) / 2)
    else:
        inject_time = abnormal_start

    inject_time = datetime.datetime.fromtimestamp(inject_time, tz=datetime.timezone.utc)
    logger.debug(f"inject_time=`{inject_time}`")

    return inject_time


def merge_two_time_ranges(normal: pl.LazyFrame, anomal: pl.LazyFrame) -> pl.LazyFrame:
    assert "anomal" not in normal.collect_schema().names()
    assert "anomal" not in anomal.collect_schema().names()
    normal = normal.with_columns(anomal=pl.lit(0, dtype=pl.UInt8))
    anomal = anomal.with_columns(anomal=pl.lit(1, dtype=pl.UInt8))
    merged = pl.concat([normal, anomal])
    return merged


@timeit()
def load_metrics(input_folder: Path) -> pl.LazyFrame:
    normal_metrics = pl.scan_parquet(input_folder / "normal_metrics.parquet")
    anomal_metrics = pl.scan_parquet(input_folder / "abnormal_metrics.parquet")
    lf = merge_two_time_ranges(normal_metrics, anomal_metrics)

    return lf


def is_special_constant_metric(metric: str) -> bool:
    return metric in (
        "k8s.container.cpu_request",
        "k8s.container.memory_request",
        "k8s.container.cpu_limit",
        "k8s.container.memory_limit",
    )


@timeit()
def load_metrics_histogram(input_folder: Path) -> pl.LazyFrame:
    normal_histogram = pl.scan_parquet(
        input_folder / "normal_metrics_histogram.parquet"
    )
    anomal_histogram = pl.scan_parquet(
        input_folder / "abnormal_metrics_histogram.parquet"
    )
    lf = merge_two_time_ranges(normal_histogram, anomal_histogram)

    lf = lf.with_columns(
        pl.when(pl.col("metric") == "jvm.gc.duration")
        .then(
            pl.concat_str("metric", "attr.jvm.gc.name", separator=":").alias("metric")
        )
        .otherwise(pl.col("metric"))
    )

    return lf


def ui_span_name_parser(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parse UI dashboard span names by replacing with child span names

    Args:
        df: DataFrame with trace spans

    Returns:
        DataFrame with processed span names
    """
    # Create a mapping from parent span ID to child span name
    child_mapping = df.select(["parent_span_id", "span_name"]).rename(
        {"parent_span_id": "span_id", "span_name": "child_span_name"}
    )

    # Join with original dataframe
    merged_df = df.join(child_mapping, on="span_id", how="left")

    # Replace span names for ts-ui-dashboard service with child span names
    processed_df = merged_df.with_columns(
        pl.when(pl.col("service_name") == "ts-ui-dashboard")
        .then(pl.col("child_span_name"))
        .otherwise(pl.col("span_name"))
        .alias("span_name")
    ).drop("child_span_name")

    return processed_df


@timeit()
def load_traces(input_folder: Path) -> pl.LazyFrame:
    normal_traces = pl.scan_parquet(input_folder / "normal_traces.parquet")
    anomal_traces = pl.scan_parquet(input_folder / "abnormal_traces.parquet")
    lf = merge_two_time_ranges(normal_traces, anomal_traces)

    lf = tt_add_op_name(lf)

    status_code_values = ["Unset", "Ok", "Error"]
    lf = lf.with_columns(
        replace_enum_values("attr.status_code", status_code_values, start=0),
    )

    lf = lf.with_columns(
        pl.col("duration").cast(pl.Float64),
        pl.col("attr.http.response.status_code").cast(pl.Float64),
        pl.col("attr.http.request.content_length").cast(pl.Float64),
        pl.col("attr.http.response.content_length").cast(pl.Float64),
    )

    # Apply UI span name parsing
    # Note: We need to collect to DataFrame for the UI parsing, then convert back
    df = lf.collect()
    df = ui_span_name_parser(df)
    lf = df.lazy()

    return lf


@timeit()
def load_logs(input_folder: Path) -> pl.LazyFrame:
    normal_logs = pl.scan_parquet(input_folder / "normal_logs.parquet")
    anomal_logs = pl.scan_parquet(input_folder / "abnormal_logs.parquet")
    lf = merge_two_time_ranges(normal_logs, anomal_logs)

    level_values = ["", "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "SEVERE"]
    lf = lf.with_columns(pl.col("level").str.replace("WARNING", "WARN", literal=True))
    lf = lf.with_columns(
        replace_enum_values("level", level_values, start=0).alias("level_number")
    )

    return lf ，我们现在需要完全适配新的数据结构，并且把输入参数改成typer app，并且我们需要一个类似这样的接口from functools import partial

import pandas as pd
from rcabench_platform.v2.algorithms.spec import (
    Algorithm,
    AlgorithmAnswer,
    AlgorithmArgs,
)

# type:ignore
from rcabench_platform.vendor.RCAEval.time_series import preprocess
from sklearn.preprocessing import RobustScaler

from ._common import SimpleMetricsAdapter


def baro(
    data: pd.DataFrame,
    inject_time: int | None = None,
    dataset: str | None = None,
    anomalies: list[int] | None = None,
    dk_select_useful: bool = False,
):
    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        anomal_df = data.tail(len(data) - anomalies[0])

    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=dk_select_useful
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=dk_select_useful
    )

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    ranks = []

    for col in normal_df.columns:
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()

        scaler = RobustScaler().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }


class Baro(Algorithm):
    def needs_cpu_count(self) -> int | None:
        return 4

    def __call__(self, args: AlgorithmArgs) -> list[AlgorithmAnswer]:
        adapter = SimpleMetricsAdapter(partial(baro, dk_select_useful=True))
        return adapter(args)
```
这是我们的新数据格式以及其load示例，你需要给原本算法的每一个算法（包括baseline）都创建一个algorithm class的接口