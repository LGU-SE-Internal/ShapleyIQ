"""
ShapleyValueRCA algorithm implementation.

This module implements the main ShapleyIQ algorithm for root cause analysis
using Shapley value influence quantification.
"""

from typing import Any, Dict, List, Optional, Tuple

from rcabench_platform.v2.logging import logger

from ..data_structures import RCAData, TraceData
from ..utils import contribution_to_probability
from .base import BaseRCAAlgorithm


class ShapleyValueRCA(BaseRCAAlgorithm):
    """
    ShapleyIQ algorithm for root cause analysis using Shapley value influence quantification.

    This algorithm analyzes distributed traces to quantify the influence of each service
    operation on performance anomalies using cooperative game theory principles.
    """

    def __init__(self, using_cache: bool = False, sync_overlap_threshold: float = 0.05):
        """
        Initialize ShapleyValueRCA algorithm.

        Args:
            using_cache: Whether to use cached normal duration data
            sync_overlap_threshold: Threshold for merging synchronous timelines
        """
        super().__init__("ShapleyValueRCA")
        self.using_cache = using_cache
        self.sync_overlap_threshold = sync_overlap_threshold
        self.normal_duration_dict = {}
        self.timeline_dict = {}
        self.metrics_statistical_data = {}

    def analyze(self, data: RCAData, **kwargs):
        """
        Perform Shapley value based root cause analysis.

        Args:
            data: Processed RCA data
            **kwargs: Additional parameters
                - strategy: 'avg_by_contribution' or 'avg_by_prob'
                - with_anomaly_nodes: Whether to track anomaly nodes
                - sort_result: Whether to sort results

        Returns:
            Dictionary mapping node IDs to Shapley values, or tuple if with_anomaly_nodes=True
        """
        # 从RCAData中获取基线统计数据（类似MicroRank的处理方式）
        self.metrics_statistical_data = data.metrics_statistical_data or {}
        
        strategy = kwargs.get("strategy", "avg_by_contribution")
        with_anomaly_nodes = kwargs.get("with_anomaly_nodes", False)
        sort_result = kwargs.get("sort_result", True)

        if data.traces:
            return self._analyze_traces(
                data.traces, strategy, with_anomaly_nodes, sort_result
            )
        elif data.spans:
            # Backward compatibility - convert spans to trace format
            trace_data = TraceData(trace_id="legacy", spans=data.spans)
            return self._analyze_single_trace(trace_data)
        else:
            logger.warning("No trace data available for analysis")
            return {}

    def _analyze_traces(
        self,
        traces: List[TraceData],
        strategy: str = "avg_by_contribution",
        with_anomaly_nodes: bool = False,
        sort_result: bool = False,
    ) -> Dict[str, float]:
        """
        Analyze multiple traces and aggregate results.

        Args:
            traces: List of trace data
            strategy: Aggregation strategy
            with_anomaly_nodes: Track anomaly nodes
            sort_result: Sort final results

        Returns:
            Aggregated Shapley values
        """
        impacted_nodes = {}
        aggregated_contribution = {}

        for trace in traces:
            if with_anomaly_nodes and isinstance(trace, dict) and "trace" in trace:
                real_trace = trace["trace"]
                contribution_dict = self._analyze_single_trace(real_trace)

                # Track impacted nodes
                for root_cause_node in contribution_dict:
                    if root_cause_node not in impacted_nodes:
                        impacted_nodes[root_cause_node] = set()
                    impacted_nodes[root_cause_node].update(trace["anomaly_nodes"])

                trace_data = real_trace
            else:
                trace_data = trace
                contribution_dict = self._analyze_single_trace(trace_data)

            # Convert to probabilities if needed
            if strategy == "avg_by_prob":
                contribution_dict = contribution_to_probability(contribution_dict)

            # Aggregate contributions
            for key, value in contribution_dict.items():
                if key not in aggregated_contribution:
                    aggregated_contribution[key] = 0
                aggregated_contribution[key] += value

        # Average the contributions
        k = len(traces)
        if k > 0:
            for key in aggregated_contribution:
                aggregated_contribution[key] /= k

        # Sort results if requested
        if sort_result:
            aggregated_contribution = dict(
                sorted(
                    aggregated_contribution.items(), key=lambda x: x[1], reverse=True
                )
            )

        if with_anomaly_nodes:
            return aggregated_contribution, impacted_nodes
        return aggregated_contribution

    def _analyze_single_trace(self, trace: TraceData) -> Dict[str, float]:
        """
        Analyze a single trace to compute Shapley values.

        Args:
            trace: Single trace data

        Returns:
            Dictionary mapping node IDs to Shapley values
        """
        if isinstance(trace, TraceData):
            spans = trace.spans
        else:
            # Handle legacy format
            spans = trace

        if not spans:
            return {}

        try:
            # Step 1: Convert trace to timelines and calling tree
            timeline_dict = self._trace_to_timelines(spans)
            calling_tree = self._trace_to_calling_tree(spans)

            # logger.debug(
            #     f"Timeline dict: {sorted(timeline_dict.items(), key=lambda x: x[1])}"
            # )
            # logger.debug(f"Calling tree: {calling_tree}")

            # Step 2: Split timelines of callers
            all_timeline_segments = []
            for caller, callees in calling_tree.items():
                if caller not in timeline_dict:
                    continue

                callee_timelines = []
                for callee in callees:
                    if callee in timeline_dict:
                        callee_timelines.append(timeline_dict[callee])

                timeline_segments = self._split_timelines(
                    caller, timeline_dict[caller], callee_timelines
                )
                all_timeline_segments.extend(timeline_segments)

            # Add callees that are not callers
            for key, timeline in timeline_dict.items():
                if key not in calling_tree:
                    all_timeline_segments.append(timeline)

            # logger.debug(
            #     f"Timeline segments after splitting: {sorted(all_timeline_segments, key=lambda x: x[3], reverse=True)}"
            # )

            # Step 3: Merge synchronous timelines
            merged_timelines = self._merge_timelines(all_timeline_segments)
            # logger.debug(f"Merged timelines: {merged_timelines}")

            # Step 4: Calculate Shapley values
            contribution_dict = self._shapley_value_for_timelines(merged_timelines)
            # logger.debug(f"Contribution dict: {contribution_dict}")

            # Step 5: Distribute contributions to original nodes
            adjusted_contribution_dict = self._distribute_contribution_to_nodes(
                contribution_dict
            )

            # Step 6: Filter negative contributions
            for key in adjusted_contribution_dict:
                if adjusted_contribution_dict[key] < 0:
                    adjusted_contribution_dict[key] = 0

            return adjusted_contribution_dict

        except Exception as e:
            logger.error(f"Failed to analyze trace: {e}")
            return {}

    def _trace_to_timelines(self, spans: List[Dict]) -> Dict[str, Tuple]:
        """
        Convert trace spans to timeline representation.

        Args:
            spans: List of span dictionaries

        Returns:
            Dictionary mapping node IDs to timeline tuples (start, end, node_id, diff)
        """
        timeline_dict = {}

        # Rename spans to handle multiple occurrences
        renamed_spans = self._rename_spans_by_count(spans)

        # Build span tree structure
        root_ids, spans_dict = self._build_span_tree(renamed_spans)

        for span_id, span_data in spans_dict.items():
            if "serviceName" not in span_data:
                continue

            service_name = span_data.get("serviceName", "unknown")
            operation_name = span_data.get(
                "operationName", span_data.get("rpc", "unknown")
            )
            node_id = f"{service_name}:{operation_name}"

            # Extract timing information
            duration = int(span_data.get("Duration", span_data.get("duration", 0)))
            start_time = int(span_data.get("TimeStamp", span_data.get("startTime", 0)))
            end_time = start_time + duration

            # Get normal duration (baseline)
            normal_duration = self._get_normal_duration(node_id.split(",")[0])
            if normal_duration is None:
                normal_duration = 0

            # Calculate duration difference from normal
            diff_duration = duration - normal_duration

            timeline_dict[node_id] = (start_time, end_time, node_id, diff_duration)

        self.timeline_dict = timeline_dict
        return timeline_dict

    def _trace_to_calling_tree(self, spans: List[Dict]) -> Dict[str, List[str]]:
        """
        Build calling tree from trace spans.

        Args:
            spans: List of span dictionaries

        Returns:
            Dictionary mapping caller IDs to lists of callee IDs
        """
        calling_tree = {}

        # Rename spans and build tree
        renamed_spans = self._rename_spans_by_count(spans)
        root_ids, spans_dict = self._build_span_tree(renamed_spans)

        for span_id, span_data in spans_dict.items():
            if "serviceName" not in span_data:
                continue

            if "children" in span_data:
                service_name = span_data.get("serviceName", "unknown")
                operation_name = span_data.get(
                    "operationName", span_data.get("rpc", "unknown")
                )
                source_id = f"{service_name}:{operation_name}"

                if source_id not in calling_tree:
                    calling_tree[source_id] = []

                for child_id in span_data["children"]:
                    if child_id in spans_dict:
                        child_data = spans_dict[child_id]
                        child_service = child_data.get("serviceName", "unknown")
                        child_operation = child_data.get(
                            "operationName", child_data.get("rpc", "unknown")
                        )
                        target_id = f"{child_service}:{child_operation}"
                        calling_tree[source_id].append(target_id)

        return calling_tree

    def _rename_spans_by_count(self, spans: List[Dict]) -> List[Dict]:
        """
        Rename spans to handle multiple occurrences of the same operation.

        Args:
            spans: Original spans

        Returns:
            Renamed spans with occurrence counts
        """
        operation_count = {}
        new_spans = [span.copy() for span in spans]

        for span in new_spans:
            service_name = span.get("serviceName", "unknown")
            operation_name = span.get("operationName", span.get("rpc", "unknown"))
            node_id = f"{service_name}:{operation_name}"

            if node_id not in operation_count:
                operation_count[node_id] = 0
            operation_count[node_id] += 1

            # Update operation name with count
            if "operationName" in span:
                span["operationName"] = f"{operation_name},{operation_count[node_id]}"
            elif "rpc" in span:
                span["rpc"] = f"{operation_name},{operation_count[node_id]}"

        return new_spans

    def _build_span_tree(self, spans: List[Dict]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Build span tree structure from spans.

        Args:
            spans: List of span dictionaries

        Returns:
            Tuple of (root_ids, spans_dict)
        """
        spans_dict = {}

        # First pass: build basic span dictionary
        for span in spans:
            span_id = span.get("spanId", span.get("id", ""))
            spans_dict[span_id] = span.copy()
            spans_dict[span_id]["children"] = []

        # Second pass: build parent-child relationships
        root_ids = []
        for span in spans:
            span_id = span.get("spanId", span.get("id", ""))
            parent_id = span.get("parentSpanId")

            if parent_id is None or parent_id == "" or parent_id == "0":
                root_ids.append(span_id)
            else:
                if parent_id in spans_dict:
                    spans_dict[parent_id]["children"].append(span_id)

        return root_ids, spans_dict

    def _get_normal_duration(self, node_id: str) -> Optional[float]:
        """
        Get normal duration for a node (baseline for comparison).
        
        使用metrics_statistical_data中的基线统计数据（类似MicroRank的处理方式）

        Args:
            node_id: Node identifier

        Returns:
            Normal duration or None if not available
        """
        if not self.using_cache:
            return None

        if node_id in self.normal_duration_dict:
            return self.normal_duration_dict[node_id]

        # 从metrics_statistical_data获取基线统计数据（从正常traces计算得出）
        if hasattr(self, 'metrics_statistical_data') and node_id in self.metrics_statistical_data:
            stats = self.metrics_statistical_data[node_id].get("Duration", [])
            if len(stats) >= 1:  # [mean, std, count]
                mean_duration = float(stats[0])
                self.normal_duration_dict[node_id] = mean_duration
                return mean_duration

        return None

    def _split_timelines(
        self, caller_id: str, caller_timeline: Tuple, callee_timelines: List[Tuple]
    ) -> List[Tuple]:
        """
        Split caller timeline based on callee timelines.

        Args:
            caller_id: Caller node ID
            caller_timeline: Caller's timeline tuple
            callee_timelines: List of callee timeline tuples

        Returns:
            List of split timeline segments
        """
        if not callee_timelines:
            return [caller_timeline]

        cnt = 1
        orig_duration = caller_timeline[1] - caller_timeline[0]

        # Sort callee timelines by start time
        sorted_timelines = sorted(callee_timelines, key=lambda x: x[0])
        timeline_segments = []

        current_end_time = caller_timeline[0]

        for timeline in sorted_timelines:
            start_time, end_time = timeline[0], timeline[1]

            if start_time > current_end_time:
                # Create segment for gap between callees
                splitted_node_id = f"{caller_id},{cnt}"
                segment = [current_end_time, start_time, splitted_node_id]
                timeline_segments.append(segment)
                cnt += 1

            if end_time > current_end_time:
                current_end_time = end_time

        # Add final segment if needed
        if caller_timeline[1] > current_end_time:
            splitted_node_id = f"{caller_id},{cnt}"
            segment = [current_end_time, caller_timeline[1], splitted_node_id]
            timeline_segments.append(segment)

        # Calculate and distribute caller's duration difference
        total_time = sum(seg[1] - seg[0] for seg in timeline_segments)

        # Calculate remaining duration increment of the caller (proportional)
        caller_diff = caller_timeline[3] * (total_time / orig_duration)
        caller_diff = min(orig_duration, caller_diff)

        for segment in timeline_segments:
            if total_time != 0:  # Corner case protection
                segment_duration = segment[1] - segment[0]
                diff = caller_diff * (segment_duration / total_time)
                diff = min(segment_duration, diff)
                segment.append(diff)
            else:
                segment.append(0.0)

        # Store complete 4-element tuples in timeline_dict
        for segment in timeline_segments:
            node_id = segment[2]
            self.timeline_dict[node_id] = tuple(segment)

        return [tuple(seg) for seg in timeline_segments]

    def _merge_timelines(self, timelines: List[Tuple]) -> List[Tuple]:
        """
        Merge synchronous timelines into larger timeline units.

        Args:
            timelines: List of timeline tuples

        Returns:
            List of merged timeline tuples
        """

        def are_sync_timelines(timeline1: Tuple, timeline2: Tuple) -> bool:
            """Check if two timelines are synchronous."""
            gap = timeline2[0] - timeline1[1]
            threshold = (timeline1[1] - timeline1[0]) * self.sync_overlap_threshold
            return gap < threshold

        sorted_timelines = sorted(timelines, key=lambda x: x[0])
        merged_timelines = []
        remaining_timelines = sorted_timelines.copy()

        while remaining_timelines:
            # Start with first remaining timeline
            current_idx = 0
            merged_ids = []
            start_time = remaining_timelines[current_idx][0]
            end_time = remaining_timelines[current_idx][1]
            total_diff = remaining_timelines[current_idx][3]
            merged_ids.append(remaining_timelines[current_idx][2])

            indices_to_remove = [current_idx]

            # Find all timelines that can be merged
            while True:
                found_sync = False
                max_end_time = end_time
                max_end_idx = current_idx

                for j in range(current_idx + 1, len(remaining_timelines)):
                    if are_sync_timelines(
                        remaining_timelines[current_idx], remaining_timelines[j]
                    ):
                        if remaining_timelines[j][1] > max_end_time:
                            max_end_time = remaining_timelines[j][1]
                            max_end_idx = j

                        if j not in indices_to_remove:
                            indices_to_remove.append(j)
                            merged_ids.append(remaining_timelines[j][2])
                            total_diff += remaining_timelines[j][3]
                            found_sync = True

                if not found_sync or max_end_idx == current_idx:
                    break

                current_idx = max_end_idx
                end_time = max_end_time

            # Create merged timeline
            merged_timeline = (start_time, end_time, "~~".join(merged_ids), total_diff)
            merged_timelines.append(merged_timeline)

            # Remove processed timelines
            for idx in sorted(set(indices_to_remove), reverse=True):
                del remaining_timelines[idx]

        return merged_timelines

    def _shapley_value_for_timelines(self, timelines: List[Tuple]) -> Dict[str, float]:
        """
        Calculate Shapley values for merged timelines.

        Args:
            timelines: List of merged timeline tuples

        Returns:
            Dictionary mapping timeline IDs to Shapley values
        """
        time_points = []

        # Create time points for Shapley calculation
        for timeline in timelines:
            if timeline[3] < 0:  # Skip negative duration differences
                continue

            start_time, end_time, node_id, diff = timeline
            old_time = end_time - diff

            time_points.extend(
                [
                    {"time": start_time, "node_id": node_id, "type": "start_time"},
                    {"time": end_time, "node_id": node_id, "type": "end_time"},
                    {"time": old_time, "node_id": node_id, "type": "old_time"},
                ]
            )

        # Sort time points in reverse chronological order
        sorted_time_points = sorted(time_points, key=lambda x: x["time"], reverse=True)

        contribution_dict = {}
        processing_node_ids = []
        last_checkout_time = None

        for time_point in sorted_time_points:
            if time_point["type"] in ["end_time", "old_time"]:
                # Distribute contribution among active nodes
                if processing_node_ids and last_checkout_time is not None:
                    time_diff = last_checkout_time - time_point["time"]
                    contribution_per_node = time_diff / len(processing_node_ids)

                    for node_id in processing_node_ids:
                        if node_id not in contribution_dict:
                            contribution_dict[node_id] = 0
                        contribution_dict[node_id] += contribution_per_node

                last_checkout_time = time_point["time"]

            # Update processing nodes list
            node_id = time_point["node_id"]
            if time_point["type"] == "end_time":
                if node_id not in processing_node_ids:
                    processing_node_ids.append(node_id)
            elif time_point["type"] == "old_time":
                if node_id in processing_node_ids:
                    processing_node_ids.remove(node_id)

        return contribution_dict

    def _distribute_contribution_to_nodes(
        self, contribution_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Distribute contributions from merged timelines back to original nodes.

        Args:
            contribution_dict: Contributions for merged timelines

        Returns:
            Contributions distributed to original node IDs
        """
        adjusted_contribution_dict = {}

        for merged_key, contribution in contribution_dict.items():
            operation_names = merged_key.split("~~")

            # Calculate total difference for normalization
            total_diff = 0
            for operation_name in operation_names:
                if operation_name in self.timeline_dict:
                    total_diff += self.timeline_dict[operation_name][3]

            # Distribute contribution proportionally
            for operation_name in operation_names:
                if operation_name not in adjusted_contribution_dict:
                    adjusted_contribution_dict[operation_name] = 0

                if total_diff > 0 and operation_name in self.timeline_dict:
                    fraction = self.timeline_dict[operation_name][3] / total_diff
                    adjusted_contribution_dict[operation_name] += (
                        contribution * fraction
                    )

        # Aggregate by original node ID (remove count suffixes)
        final_dict = {}
        for old_key, value in adjusted_contribution_dict.items():
            new_key = old_key.split(",")[0]  # Remove count suffix
            if new_key not in final_dict:
                final_dict[new_key] = 0
            final_dict[new_key] += value

        return final_dict
