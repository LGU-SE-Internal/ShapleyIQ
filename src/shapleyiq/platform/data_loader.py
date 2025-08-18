"""
New Platform Data Loader
适配新的rcabench_platform.v2数据格式
"""

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
    return lf.with_columns(pl.col("span_name").alias("operation_name"))


def replace_enum_values(column: str, enum_values: list, start: int = 0) -> pl.Expr:
    """Replace enum string values with integers"""
    mapping_expr = pl.col(column)
    for i, value in enumerate(enum_values):
        mapping_expr = mapping_expr.str.replace(value, str(start + i))
    return mapping_expr.cast(pl.Int32)


def load_inject_time(input_folder: Path) -> datetime.datetime:
    """Load injection time from env.json"""
    env = load_json(path=input_folder / "env.json")

    normal_start = int(env["NORMAL_START"])
    normal_end = int(env["NORMAL_END"])
    abnormal_start = int(env["ABNORMAL_START"])
    abnormal_end = int(env["ABNORMAL_END"])

    assert normal_start < normal_end <= abnormal_start < abnormal_end

    if normal_end < abnormal_start:
        inject_time = int(math.ceil((normal_end + abnormal_start) / 2))
    else:
        inject_time = abnormal_start

    inject_time = datetime.datetime.fromtimestamp(inject_time, tz=datetime.timezone.utc)
    # logger.debug(f"inject_time=`{inject_time}`")

    return inject_time


def merge_two_time_ranges(normal: pl.LazyFrame, anomal: pl.LazyFrame) -> pl.LazyFrame:
    """Merge normal and anomalous time ranges with anomal flag"""
    assert "anomal" not in normal.collect_schema().names()
    assert "anomal" not in anomal.collect_schema().names()
    normal = normal.with_columns(anomal=pl.lit(0, dtype=pl.UInt8))
    anomal = anomal.with_columns(anomal=pl.lit(1, dtype=pl.UInt8))
    merged = pl.concat([normal, anomal])
    return merged



def ui_span_name_parser(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parse UI dashboard span names by replacing with child span names
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
    """Load traces data"""
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
    df = lf.collect()
    df = ui_span_name_parser(df)
    lf = df.lazy()

    return lf



class PlatformDataLoader:
    """
    platform data loader for ShapleyIQ algorithms
    """

    def __init__(self, input_folder: Path):
        self.input_folder = Path(input_folder)
        self.inject_time = load_inject_time(self.input_folder)

    def load_all_data(self):
        """Load all available data types"""
        data = {}

        # Load traces
        if (self.input_folder / "normal_traces.parquet").exists():
            data["traces"] = load_traces(self.input_folder)

        data["inject_time"] = self.inject_time
        return data
