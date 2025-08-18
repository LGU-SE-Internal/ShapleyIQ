"""
Data preprocessing modules for RCA algorithms.

This module handles the conversion of raw trace and metric data into
the standardized RCAData format used by all algorithms.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from rcabench_platform.v2.logging import logger

from ..data_structures import Edge, RCAData, ServiceNode, TraceData


class TracePreprocessor:
    """
    Preprocesses trace data from various sources into standardized format.
    """

    def __init__(self):
        self.supported_formats = ["jaeger", "zipkin", "dbaas", "generic"]

    def process_traces(
        self, traces_data: List[Dict[str, Any]], format_type: str = "generic"
    ) -> List[TraceData]:
        """
        Process traces from list of dictionaries.

        Args:
            traces_data: List of trace dictionaries
            format_type: Format of trace data

        Returns:
            List of processed TraceData objects
        """
        processed_traces = []

        for trace_item in traces_data:
            if "trace_id" in trace_item and "spans" in trace_item:
                trace_data = TraceData(
                    trace_id=trace_item["trace_id"],
                    spans=trace_item["spans"],
                    timestamp=trace_item.get(
                        "timestamp", self._extract_trace_timestamp(trace_item["spans"])
                    ),
                )
                processed_traces.append(trace_data)

        return processed_traces

    def process_traces_from_file(
        self, file_path: str, format_type: str = "jaeger"
    ) -> List[TraceData]:
        """
        Load and process traces from file.

        Args:
            file_path: Path to trace data file
            format_type: Format of trace data ('jaeger', 'zipkin', 'dbaas')

        Returns:
            List of processed TraceData objects
        """
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if format_type == "jaeger":
                return self._process_jaeger_traces(raw_data)
            elif format_type == "zipkin":
                return self._process_zipkin_traces(raw_data)
            elif format_type == "dbaas":
                return self._process_dbaas_traces(raw_data)
            elif format_type == "generic":
                return self.process_traces(raw_data)
            else:
                return self._process_jaeger_traces(raw_data)  # Default fallback

        except Exception as e:
            logger.error(f"Failed to process traces from {file_path}: {e}")
            return []

    def _process_jaeger_traces(self, raw_data: Any) -> List[TraceData]:
        """Process Jaeger format traces."""
        traces = []

        # Handle different Jaeger JSON structures
        if isinstance(raw_data, dict) and "data" in raw_data:
            trace_list = raw_data["data"]
        elif isinstance(raw_data, list):
            trace_list = raw_data
        else:
            trace_list = [raw_data]

        for trace_item in trace_list:
            if "spans" in trace_item:
                trace_data = TraceData(
                    trace_id=trace_item.get("traceID", ""),
                    spans=trace_item["spans"],
                    timestamp=self._extract_trace_timestamp(trace_item["spans"]),
                )
                traces.append(trace_data)

        return traces

    def _process_zipkin_traces(self, raw_data: Any) -> List[TraceData]:
        """Process Zipkin format traces."""
        traces = []

        if isinstance(raw_data, list):
            # Group spans by trace ID
            trace_spans = {}
            for span in raw_data:
                trace_id = span.get("traceId", "")
                if trace_id not in trace_spans:
                    trace_spans[trace_id] = []
                trace_spans[trace_id].append(span)

            for trace_id, spans in trace_spans.items():
                trace_data = TraceData(
                    trace_id=trace_id,
                    spans=spans,
                    timestamp=self._extract_trace_timestamp(spans),
                )
                traces.append(trace_data)

        return traces

    def _process_dbaas_traces(self, raw_data: Any) -> List[TraceData]:
        """Process DbaAS format traces."""
        traces = []

        # DbaAS specific processing logic
        if isinstance(raw_data, dict):
            for trace_id, trace_info in raw_data.items():
                if isinstance(trace_info, dict) and "spans" in trace_info:
                    trace_data = TraceData(
                        trace_id=trace_id,
                        spans=trace_info["spans"],
                        timestamp=trace_info.get("timestamp"),
                    )
                    traces.append(trace_data)

        return traces

    def _extract_trace_timestamp(self, spans: List[Dict]) -> Optional[int]:
        """Extract timestamp from trace spans."""
        if not spans:
            return None

        # Find root span or earliest timestamp
        timestamps = []
        for span in spans:
            if "startTime" in span:
                timestamps.append(span["startTime"])
            elif "timestamp" in span:
                timestamps.append(span["timestamp"])

        return min(timestamps) if timestamps else None

    def extract_call_graph(
        self, traces: List[TraceData]
    ) -> Tuple[List[Edge], List[ServiceNode]]:
        """
        Extract call graph structure from traces.

        Args:
            traces: List of processed trace data

        Returns:
            Tuple of (edges, nodes) representing the call graph
        """
        edges = []
        nodes = {}

        for trace in traces:
            trace_edges, trace_nodes = self._extract_single_trace_graph(trace)

            # Merge edges (avoid duplicates)
            for edge in trace_edges:
                if edge not in edges:
                    edges.append(edge)

            # Merge nodes (avoid duplicates)
            for node in trace_nodes:
                if node.node_id not in nodes:
                    nodes[node.node_id] = node

        return edges, list(nodes.values())

    def _extract_single_trace_graph(
        self, trace: TraceData
    ) -> Tuple[List[Edge], List[ServiceNode]]:
        """Extract graph from a single trace."""
        edges = []
        nodes = {}

        # Build span relationships
        span_dict = {
            span.get("spanId", span.get("span_id", span.get("id", ""))): span
            for span in trace.spans
        }

        for span in trace.spans:
            service_name = (
                span.get("process", {}).get("serviceName")
                or span.get("serviceName")
                or span.get("service_name")
                or "unknown"
            )
            operation_name = (
                span.get("operationName")
                or span.get("name")
                or span.get("operation_name")
                or "unknown"
            )

            # Create node ID (service:operation format)
            node_id = f"{service_name}:{operation_name}"

            # Extract server IP
            server_ip = self._extract_server_ip(span)

            # Create node
            if node_id not in nodes:
                node = ServiceNode(
                    node_id=node_id, service_name=service_name, server_ip=server_ip
                )
                nodes[node_id] = node

            # Create edge from parent to current span
            parent_span_id = span.get("parentSpanId") or span.get("parent_span_id")
            if parent_span_id and parent_span_id in span_dict:
                parent_span = span_dict[parent_span_id]
                parent_service = (
                    parent_span.get("process", {}).get("serviceName")
                    or parent_span.get("serviceName")
                    or parent_span.get("service_name")
                    or "unknown"
                )
                parent_operation = (
                    parent_span.get("operationName")
                    or parent_span.get("name")
                    or parent_span.get("operation_name")
                    or "unknown"
                )
                parent_node_id = f"{parent_service}:{parent_operation}"

                # Create parent node if not exists
                if parent_node_id not in nodes:
                    parent_server_ip = self._extract_server_ip(parent_span)
                    parent_node = ServiceNode(
                        node_id=parent_node_id,
                        service_name=parent_service,
                        server_ip=parent_server_ip,
                    )
                    nodes[parent_node_id] = parent_node

                # Create edge
                edge = Edge(source_id=parent_node_id, target_id=node_id)
                if edge not in edges:
                    edges.append(edge)

        return edges, list(nodes.values())

    def _extract_server_ip(self, span: Dict) -> str:
        """Extract server IP from span data."""
        # Try multiple possible locations for IP
        ip_sources = [
            lambda s: s.get("process", {}).get("tags", {}).get("ip"),
            lambda s: s.get("tags", {}).get("ip"),
            lambda s: s.get("localEndpoint", {}).get("ipv4"),
            lambda s: s.get("remoteEndpoint", {}).get("ipv4"),
            lambda s: next(
                (
                    tag.get("value")
                    for tag in s.get("tags", [])
                    if tag.get("key") == "ip"
                ),
                None,
            ),
        ]

        for extract_ip in ip_sources:
            try:
                ip = extract_ip(span)
                if ip:
                    return str(ip)
            except Exception:
                continue

        return "127.0.0.1"  # Default fallback


class MetricPreprocessor:
    """
    Preprocesses metric data for RCA algorithms.
    """

    def __init__(self):
        self.default_metrics = ["MaxDuration", "Duration", "QPS"]

    def process_metrics_from_traces(
        self, traces: List[TraceData], nodes: List[ServiceNode]
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Extract metrics from trace data.

        Args:
            traces: List of trace data
            nodes: List of service nodes

        Returns:
            Dictionary mapping node_id to metric_name to values
        """
        metrics_dict = {}

        # Initialize metrics structure
        for node in nodes:
            metrics_dict[node.node_id] = {metric: [] for metric in self.default_metrics}

        # Extract metrics from traces
        for trace in traces:
            node_metrics = self._extract_trace_metrics(trace)

            # Aggregate metrics
            for node_id, metric_data in node_metrics.items():
                if node_id in metrics_dict:
                    for metric_name, value in metric_data.items():
                        if metric_name in metrics_dict[node_id]:
                            metrics_dict[node_id][metric_name].append(value)

        return metrics_dict

    def _extract_trace_metrics(self, trace: TraceData) -> Dict[str, Dict[str, float]]:
        """Extract metrics from a single trace."""
        node_metrics = {}

        for span in trace.spans:
            service_name = span.get("process", {}).get(
                "serviceName", span.get("serviceName", "unknown")
            )
            operation_name = span.get("operationName", span.get("name", "unknown"))
            node_id = f"{service_name}:{operation_name}"

            if node_id not in node_metrics:
                node_metrics[node_id] = {}

            # Extract duration
            duration = self._extract_duration(span)
            if duration is not None:
                node_metrics[node_id]["Duration"] = duration
                node_metrics[node_id]["MaxDuration"] = (
                    duration  # For single trace, max = duration
                )

            # Extract other metrics as available
            # QPS would need to be calculated at a higher level

        return node_metrics

    def _extract_duration(self, span: Dict) -> Optional[float]:
        """Extract duration from span."""
        duration_sources = [
            lambda s: s.get("duration"),
            lambda s: s.get("endTime", 0) - s.get("startTime", 0),
            lambda s: s.get("finishTime", 0) - s.get("startTime", 0),
        ]

        for extract_duration in duration_sources:
            try:
                duration = extract_duration(span)
                if duration is not None and duration > 0:
                    return float(duration)
            except Exception:
                continue

        return None

    def load_statistical_data(self, file_path: str) -> Tuple[Dict, Dict]:
        """
        Load pre-computed statistical data and thresholds.

        Args:
            file_path: Path to statistical data file

        Returns:
            Tuple of (statistical_data, thresholds)
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            statistical_data = data.get("statistical_data", {})
            thresholds = data.get("thresholds", {})

            return statistical_data, thresholds

        except Exception as e:
            logger.error(f"Failed to load statistical data from {file_path}: {e}")
            return {}, {}


class RCADataBuilder:
    """
    Builds complete RCAData objects from preprocessed components.
    """

    def __init__(self):
        self.trace_processor = TracePreprocessor()
        self.metric_processor = MetricPreprocessor()

    def build_from_files(
        self,
        trace_file: str,
        root_causes: List[str],
        trace_format: str = "jaeger",
        statistical_data_file: Optional[str] = None,
        ip_mapping: Optional[Dict[str, str]] = None,
    ) -> RCAData:
        """
        Build RCAData from input files.

        Args:
            trace_file: Path to trace data file
            root_causes: List of known root cause node IDs
            trace_format: Format of trace data
            statistical_data_file: Optional path to statistical data
            ip_mapping: Optional IP address mapping

        Returns:
            Complete RCAData object
        """
        # Process traces
        traces = self.trace_processor.process_traces_from_file(trace_file, trace_format)
        if not traces:
            logger.warning("No traces processed successfully")
            return RCAData(root_causes=root_causes, ip_mapping=ip_mapping)

        # Extract call graph
        edges, nodes = self.trace_processor.extract_call_graph(traces)

        # Process metrics
        ts_data_dict = self.metric_processor.process_metrics_from_traces(traces, nodes)

        # Load statistical data if available
        metrics_statistical_data = {}
        metrics_threshold = {}
        if statistical_data_file and os.path.exists(statistical_data_file):
            metrics_statistical_data, metrics_threshold = (
                self.metric_processor.load_statistical_data(statistical_data_file)
            )

        # Build trace data dictionary (for backward compatibility)
        trace_data_dict = {}
        for trace in traces:
            for span in trace.spans:
                service_name = span.get("process", {}).get(
                    "serviceName", span.get("serviceName", "unknown")
                )
                operation_name = span.get("operationName", span.get("name", "unknown"))
                node_id = f"{service_name}:{operation_name}"

                if node_id not in trace_data_dict:
                    trace_data_dict[node_id] = {"Duration": [], "serverIp": []}

                duration = self.metric_processor._extract_duration(span)
                if duration is not None:
                    trace_data_dict[node_id]["Duration"].append(duration)

                server_ip = self.trace_processor._extract_server_ip(span)
                if server_ip not in trace_data_dict[node_id]["serverIp"]:
                    trace_data_dict[node_id]["serverIp"].append(server_ip)

        # Determine request timestamp and root ID
        request_timestamp = None
        root_id = None
        if traces:
            request_timestamp = traces[0].timestamp
            # Try to find root span
            for trace in traces:
                for span in trace.spans:
                    if "parentSpanId" not in span or span["parentSpanId"] is None:
                        service_name = span.get("process", {}).get(
                            "serviceName", span.get("serviceName", "unknown")
                        )
                        operation_name = span.get(
                            "operationName", span.get("name", "unknown")
                        )
                        root_id = f"{service_name}:{operation_name}"
                        break
                if root_id:
                    break

        # Create RCAData object
        rca_data = RCAData(
            edges=edges,
            nodes=nodes,
            node_ids=[node.node_id for node in nodes],
            root_causes=root_causes,
            root_id=root_id,
            traces=traces,
            trace_data_dict=trace_data_dict,
            request_timestamp=request_timestamp,
            ts_data_dict=ts_data_dict,
            metrics_statistical_data=metrics_statistical_data,
            metrics_threshold=metrics_threshold,
            ip_mapping=ip_mapping,
        )

        # Validate the built data
        if not rca_data.validate():
            logger.warning("Built RCAData failed validation")

        return rca_data

    def build_from_raw_data(
        self,
        edges: List[tuple],
        nodes_data: List[Dict],
        traces_data: List[Dict],
        root_causes: List[str],
        **kwargs,
    ) -> RCAData:
        """
        Build RCAData from raw data structures.

        Args:
            edges: List of (source_id, target_id) tuples
            nodes_data: List of node dictionaries
            traces_data: List of trace dictionaries
            root_causes: List of root cause node IDs
            **kwargs: Additional parameters

        Returns:
            RCAData object
        """
        # Convert edges
        edge_objects = [Edge(source_id=src, target_id=tgt) for src, tgt in edges]

        # Convert nodes
        node_objects = []
        for node_data in nodes_data:
            node = ServiceNode(
                node_id=node_data["node_id"],
                service_name=node_data.get("service_name", "unknown"),
                server_ip=node_data.get("server_ip", "127.0.0.1"),
            )
            node_objects.append(node)

        # Convert traces
        trace_objects = []
        for trace_data in traces_data:
            trace = TraceData(
                trace_id=trace_data["trace_id"],
                spans=trace_data["spans"],
                timestamp=trace_data.get("timestamp"),
            )
            trace_objects.append(trace)

        return RCAData(
            edges=edge_objects,
            nodes=node_objects,
            traces=trace_objects,
            root_causes=root_causes,
            **kwargs,
        )
