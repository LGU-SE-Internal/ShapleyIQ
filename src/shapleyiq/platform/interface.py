"""
Platform Interface Specifications
定义新平台的算法接口规范
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl


@dataclass
class AlgorithmArgs:
    """Algorithm input arguments"""

    input_folder: Path
    traces: Optional[pl.LazyFrame] = None
    metrics: Optional[pl.LazyFrame] = None
    metrics_histogram: Optional[pl.LazyFrame] = None
    logs: Optional[pl.LazyFrame] = None
    inject_time: Optional[Any] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class AlgorithmAnswer:
    """Algorithm output answer"""

    # 保持向后兼容的字段
    ranks: Optional[List[str]] = None  # Ranked list of root cause candidates
    scores: Optional[Dict[str, float]] = None  # Optional scores for each candidate
    node_names: Optional[List[str]] = None  # List of all node names
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata

    # 新增的service级别字段
    level: Optional[str] = None  # "service" or "operation"
    name: Optional[str] = None  # service name or operation name
    rank: Optional[int] = None  # rank in the results
    service_ranking: Optional[List[str]] = None  # Service-level ranking list


class Algorithm(ABC):
    """Abstract base class for all RCA algorithms"""

    @abstractmethod
    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        """Execute the algorithm and return results"""
        pass

    def needs_cpu_count(self) -> Optional[int]:
        """Return the number of CPUs needed, if any"""
        return None

    def get_algorithm_name(self) -> str:
        """Return the algorithm name"""
        return self.__class__.__name__


class BaseAdapter:
    """Base adapter for converting between old and new interfaces"""

    def __init__(self, algorithm_func):
        self.algorithm_func = algorithm_func

    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        """Convert new interface to old and back"""
        raise NotImplementedError("Subclasses must implement this method")


class TracesAdapter(BaseAdapter):
    """Adapter for trace-based algorithms like ShapleyRCA"""

    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        if args.traces is None:
            raise ValueError("Traces data is required for this algorithm")

        # Convert Polars LazyFrame to format expected by algorithm
        traces_df = args.traces.collect().to_pandas()

        # Group by trace_id to create trace structures
        traces = self._convert_to_trace_format(traces_df)

        # Call the original algorithm
        result = self.algorithm_func(traces, **(args.parameters or {}))

        # Convert result to new format
        if isinstance(result, dict):
            if "ranks" in result:
                ranks = result["ranks"]
            else:
                # Assume result is a score dictionary
                sorted_items = sorted(result.items(), key=lambda x: x[1], reverse=True)
                ranks = [item[0] for item in sorted_items]
                scores = dict(sorted_items)

                return [
                    AlgorithmAnswer(
                        ranks=ranks, scores=scores, node_names=list(result.keys())
                    )
                ]

        return [AlgorithmAnswer(ranks=ranks or [])]

    def _convert_to_trace_format(self, traces_df: pd.DataFrame) -> List[List[Dict]]:
        """Convert DataFrame to trace format expected by algorithms"""
        traces = []

        for trace_id in traces_df["trace_id"].unique():
            trace_spans = traces_df[traces_df["trace_id"] == trace_id]

            spans = []
            for _, span in trace_spans.iterrows():
                span_dict = {
                    "spanID": span.get("span_id", ""),
                    "traceID": span.get("trace_id", ""),
                    "operationName": span.get("span_name", ""),
                    "startTime": span.get("time", 0),
                    "duration": span.get("duration", 0),
                    "parentSpanID": span.get("parent_span_id", ""),
                    "process": {"serviceName": span.get("service_name", "")},
                    "tags": {},
                    "logs": [],
                }
                spans.append(span_dict)

            if spans:
                traces.append(spans)

        return traces


class MetricsAdapter(BaseAdapter):
    """Adapter for metrics-based algorithms"""

    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        if args.metrics is None:
            raise ValueError("Metrics data is required for this algorithm")

        # Convert metrics to pandas DataFrame
        metrics_df = args.metrics.collect().to_pandas()

        # Prepare data for algorithm
        inject_time = args.inject_time
        if hasattr(inject_time, "timestamp"):
            inject_time = inject_time.timestamp()

        # Call the original algorithm
        result = self.algorithm_func(
            data=metrics_df, inject_time=inject_time, **(args.parameters or {})
        )

        # Convert result to new format
        if isinstance(result, dict):
            ranks = result.get("ranks", [])
            node_names = result.get("node_names", [])

            return [AlgorithmAnswer(ranks=ranks, node_names=node_names)]

        return [AlgorithmAnswer(ranks=result if isinstance(result, list) else [])]


class SimpleMetricsAdapter(BaseAdapter):
    """Simple adapter for metrics-based algorithms like Baro"""

    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        if args.metrics is None:
            raise ValueError("Metrics data is required for this algorithm")

        # Convert to pandas DataFrame
        metrics_df = args.metrics.collect().to_pandas()

        # Prepare inject_time
        inject_time = args.inject_time
        if hasattr(inject_time, "timestamp"):
            inject_time = inject_time.timestamp()

        # Call algorithm function
        result = self.algorithm_func(
            data=metrics_df, inject_time=inject_time, **(args.parameters or {})
        )

        # Convert result
        if isinstance(result, dict):
            return [
                AlgorithmAnswer(
                    ranks=result.get("ranks", []),
                    node_names=result.get("node_names", []),
                )
            ]

        return [AlgorithmAnswer(ranks=result if isinstance(result, list) else [])]
