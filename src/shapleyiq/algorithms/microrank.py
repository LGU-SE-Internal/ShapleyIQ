"""
MicroRank algorithm implementation.

This module implements the MicroRank baseline algorithm for root cause analysis.
"""

from typing import Dict, List, Set, Tuple

import numpy as np
from rcabench_platform.v2.logging import logger

from ..data_structures import RCAData
from .base import BaseRCAAlgorithm


class MicroRank(BaseRCAAlgorithm):
    """
    MicroRank algorithm for microservice root cause analysis.

    This algorithm uses spectrum-based fault localization combined with PageRank
    to identify root causes by analyzing normal and abnormal trace patterns.
    """

    def __init__(self, n_sigma: int = 3):
        """
        Initialize MicroRank algorithm.

        Args:
            n_sigma: Number of standard deviations for anomaly threshold
        """
        super().__init__("MicroRank")
        self.n_sigma = n_sigma
        self.rt_threshold = None

    def analyze(self, data: RCAData, **kwargs) -> Dict[str, float]:
        """
        Perform MicroRank root cause analysis.

        Args:
            data: Processed RCA data
            **kwargs: Additional parameters
                - phi: Parameter for preference vector (default: 0.5)
                - omega: Parameter for transition matrix (default: 0.01)
                - d: Damping factor for PageRank (default: 0.04)

        Returns:
            Dictionary mapping node IDs to root cause scores
        """
        phi = kwargs.get("phi", 0.5)
        omega = kwargs.get("omega", 0.01)
        d = kwargs.get("d", 0.04)

        # Prepare data
        self._prepare_data(data)

        # Calculate RT threshold for normal/abnormal classification
        self._calculate_rt_threshold()

        # Classify traces and build graph structures
        result = self._classify_trace()
        if not result:
            logger.warning("Failed to classify traces")
            return {}

        (
            edges_list,
            trace_operation_dict,
            trace_count,
            operation_trace_cover_dict,
            operation_vector,
            trace_vector,
        ) = result

        # Check if we have both normal and abnormal traces
        if not trace_vector[0] or not trace_vector[1]:
            logger.warning("Missing normal or abnormal traces")
            return {}

        # Calculate PageRank scores for normal and abnormal traces
        preference_vector = []
        init_vector = []
        transition_matrix = []
        pagerank_score = []

        for i in range(2):  # 0: normal, 1: abnormal
            preference_vector.append(
                self._get_preference_vector(
                    operation_vector[i],
                    trace_operation_dict[i],
                    trace_count[i],
                    phi,
                    i == 1,
                )
            )
            init_vector.append(
                self._get_init_vector(operation_vector[i], trace_vector[i])
            )
            transition_matrix.append(
                self._get_transition_matrix(
                    edges_list[i],
                    trace_operation_dict[i],
                    operation_vector[i],
                    trace_vector[i],
                    omega,
                )
            )
            pagerank_score.append(
                self._get_pagerank_score(
                    transition_matrix[i],
                    init_vector[i],
                    preference_vector[i],
                    operation_vector[i],
                    d,
                    0.001,
                )
            )

        # Calculate spectrum-based ranking
        spectrum_score = self._weighted_spectrum_ranker(
            operation_trace_cover_dict[0],
            operation_trace_cover_dict[1],
            pagerank_score[0],
            pagerank_score[1],
            trace_count[0],
            trace_count[1],
        )

        return spectrum_score

    def _prepare_data(self, data: RCAData) -> None:
        """Prepare data structures for analysis."""
        self.root_id = data.root_id
        self.request_timestamp = data.request_timestamp
        self.trace_dict = data.trace_dict or {}
        self.metrics_statistical_data = data.metrics_statistical_data or {}

        # 使用已经分类的正常和异常traces
        self.normal_trace_dict = data.normal_trace_dict or {}
        self.abnormal_trace_dict = data.abnormal_trace_dict or {}

        # Convert traces to expected format if needed
        if not self.trace_dict and data.traces:
            for trace in data.traces:
                self.trace_dict[trace.trace_id] = trace.spans

    def _calculate_rt_threshold(self) -> None:
        """Calculate response time threshold for normal/abnormal classification."""
        if self.root_id and self.root_id in self.metrics_statistical_data:
            stats = self.metrics_statistical_data[self.root_id].get(
                "Duration", [0, 1, 0]
            )
            mean_rt = float(stats[0]) if stats[0] else 1000  # Default 1 second
            std_rt = float(stats[1]) if stats[1] else 100  # Default 100ms
            self.rt_threshold = mean_rt + self.n_sigma * std_rt
        else:
            self.rt_threshold = 1000  # Default 1 second threshold

        # logger.debug(f"RT threshold set to: {self.rt_threshold}")

    def _classify_trace(
        self,
    ) -> Tuple[
        List[List[List[str]]],
        List[Dict],
        List[Dict],
        List[Dict],
        List[List[str]],
        List[Set],
    ]:
        """
        Classify traces into normal and abnormal categories.
        正常阶段的数据都是正常的，异常阶段的数据需要根据RT阈值进一步分类。

        Returns:
            Tuple containing classified trace data structures
        """
        edges_list = [set(), set()]  # [normal, abnormal]
        trace_operation_dict = [{}, {}]
        trace_count = [{}, {}]
        operation_trace_cover_dict = [{}, {}]
        operation_vector = [set(), set()]
        trace_vector = [set(), set()]

        edge_trace_list = [[], []]  # For deduplication

        # 处理正常阶段的traces - 这些都直接归类为正常(index=0)
        for trace_id, spans in self.normal_trace_dict.items():
            if not spans:
                continue

            index = 0  # normal traces
            edges = self._extract_edges_from_spans(spans)
            operation_set = self._extract_operations_from_spans(spans)

            edges_list[index].update(edges)
            operation_vector[index].update(operation_set)

            # Handle trace deduplication based on edge patterns
            edge_list = list(edges)
            if edge_list not in edge_trace_list[0]:
                edge_trace_list[0].append(edge_list)
                edge_trace_list[1].append(trace_id)
                trace_operation_dict[index][trace_id] = list(operation_set)
                trace_count[index][trace_id] = 1
                trace_vector[index].add(trace_id)
            else:
                # Find existing trace with same edge pattern
                existing_trace_id = edge_trace_list[1][
                    edge_trace_list[0].index(edge_list)
                ]
                # 只有当existing_trace_id在当前index的trace_count中时才增加计数
                if existing_trace_id in trace_count[index]:
                    trace_count[index][existing_trace_id] += 1
                else:
                    # 如果不在当前index中，添加为新的trace
                    trace_operation_dict[index][trace_id] = list(operation_set)
                    trace_count[index][trace_id] = 1
                    trace_vector[index].add(trace_id)

            # Update operation coverage
            for operation in operation_set:
                if operation in operation_trace_cover_dict[index]:
                    operation_trace_cover_dict[index][operation] += 1
                else:
                    operation_trace_cover_dict[index][operation] = 1

        # 处理异常阶段的traces - 需要根据RT阈值进一步分类
        for trace_id, spans in self.abnormal_trace_dict.items():
            if not spans:
                continue

            # 计算root duration来判断这个trace是否真的异常
            root_duration = self._calculate_root_duration(spans)

            # 根据RT阈值分类: 超过阈值的是异常(index=1)，否则是正常(index=0)
            if self.rt_threshold is not None:
                index = 1 if root_duration > self.rt_threshold else 0
            else:
                # 如果没有阈值，默认认为异常阶段的trace都是异常的
                index = 1

            edges = self._extract_edges_from_spans(spans)
            operation_set = self._extract_operations_from_spans(spans)

            edges_list[index].update(edges)
            operation_vector[index].update(operation_set)

            # Handle trace deduplication based on edge patterns
            edge_list = list(edges)
            if edge_list not in edge_trace_list[0]:
                edge_trace_list[0].append(edge_list)
                edge_trace_list[1].append(trace_id)
                trace_operation_dict[index][trace_id] = list(operation_set)
                trace_count[index][trace_id] = 1
                trace_vector[index].add(trace_id)
            else:
                # Find existing trace with same edge pattern
                existing_trace_id = edge_trace_list[1][
                    edge_trace_list[0].index(edge_list)
                ]
                # 只有当existing_trace_id在当前index的trace_count中时才增加计数
                if existing_trace_id in trace_count[index]:
                    trace_count[index][existing_trace_id] += 1
                else:
                    # 如果不在当前index中，添加为新的trace
                    trace_operation_dict[index][trace_id] = list(operation_set)
                    trace_count[index][trace_id] = 1
                    trace_vector[index].add(trace_id)

            # Update operation coverage
            for operation in operation_set:
                if operation in operation_trace_cover_dict[index]:
                    operation_trace_cover_dict[index][operation] += 1
                else:
                    operation_trace_cover_dict[index][operation] = 1

        # Convert to expected format
        edges_list = [[[edge[0], edge[1]] for edge in edges] for edges in edges_list]
        operation_vector = [list(vector) for vector in operation_vector]
        trace_vector = [trace_vector[0], trace_vector[1]]

        return (
            edges_list,
            trace_operation_dict,
            trace_count,
            operation_trace_cover_dict,
            operation_vector,
            trace_vector,
        )

    def _calculate_root_duration(self, spans: List[Dict]) -> float:
        """Calculate root span duration from trace spans."""
        # Find root span (no parent)
        root_spans = [span for span in spans if not span.get("parentSpanId")]

        if root_spans:
            root_span = root_spans[0]
            return float(root_span.get("duration", root_span.get("Duration", 0)))

        # Fallback: return maximum duration
        durations = [
            float(span.get("duration", span.get("Duration", 0))) for span in spans
        ]
        return max(durations) if durations else 0

    def _extract_edges_from_spans(self, spans: List[Dict]) -> Set[Tuple[str, str]]:
        """Extract edges (parent-child relationships) from spans."""
        edges = set()
        span_dict = {span.get("spanId", span.get("id", "")): span for span in spans}

        for span in spans:
            parent_id = span.get("parentSpanId")
            if parent_id and parent_id in span_dict:
                parent_span = span_dict[parent_id]

                # Create operation identifiers
                parent_op = self._get_operation_id(parent_span)
                child_op = self._get_operation_id(span)

                edges.add((parent_op, child_op))

        return edges

    def _extract_operations_from_spans(self, spans: List[Dict]) -> Set[str]:
        """Extract operation identifiers from spans."""
        operations = set()

        for span in spans:
            operation_id = self._get_operation_id(span)
            operations.add(operation_id)

        return operations

    def _get_operation_id(self, span: Dict) -> str:
        """Get standardized operation identifier from span."""
        service_name = span.get("process", {}).get(
            "serviceName", span.get("serviceName", "unknown")
        )
        operation_name = span.get("operationName", span.get("name", "unknown"))
        return f"{service_name}:{operation_name}"

    def _get_preference_vector(
        self,
        operation_vector: List[str],
        trace_operation_dict: Dict,
        trace_count: Dict,
        phi: float = 0.5,
        anomalous: bool = True,
    ) -> np.ndarray:
        """Calculate preference vector for PageRank."""
        preference_vector = []

        for _ in operation_vector:
            preference_vector.append([0])

        if not trace_count or not trace_operation_dict:
            return np.array(preference_vector)

        sum_k = sum([1 / count for count in trace_count.values()] or [1])
        sum_n = sum(
            [
                (1 / len(operations) if len(operations) > 0 else 0)
                for operations in trace_operation_dict.values()
            ]
        )

        for trace, operations in trace_operation_dict.items():
            if len(operations) == 0 or sum_n == 0:
                theta = 0
            else:
                if anomalous:
                    theta = (
                        phi * 1 / len(operations) / sum_n
                        + (1 - phi) * 1 / trace_count[trace] / sum_k
                    )
                else:
                    theta = 1 / len(operations) / sum_n

            preference_vector.append([theta])

        return np.array(preference_vector)

    def _get_init_vector(
        self, operation_vector: List[str], trace_vector: Set
    ) -> np.ndarray:
        """Create initial vector for PageRank."""
        total_size = len(operation_vector) + len(trace_vector)
        if total_size == 0:
            return np.array([[1]])

        init_vector = []
        for _ in range(total_size):
            init_vector.append([1 / total_size])

        return np.array(init_vector)

    def _get_transition_matrix(
        self,
        edges: List[List[str]],
        trace_operation_dict: Dict,
        operation_vector: List[str],
        trace_vector: Set,
        omega: float = 1,
    ) -> np.ndarray:
        """Build transition matrix for PageRank."""
        num_ops = len(operation_vector)
        num_traces = len(trace_vector)
        total_size = num_ops + num_traces

        if total_size == 0:
            return np.array([[1]])

        transition_matrix = np.zeros((total_size, total_size))

        # Add operation-to-operation transitions based on edges
        op_to_idx = {op: i for i, op in enumerate(operation_vector)}

        for edge in edges:
            if len(edge) >= 2 and edge[0] in op_to_idx and edge[1] in op_to_idx:
                src_idx = op_to_idx[edge[0]]
                dst_idx = op_to_idx[edge[1]]
                transition_matrix[src_idx][dst_idx] = omega

        # Add trace-to-operation transitions
        trace_list = list(trace_vector)
        for i, trace in enumerate(trace_list):
            trace_idx = num_ops + i
            operations = trace_operation_dict.get(trace, [])

            for op in operations:
                if op in op_to_idx:
                    op_idx = op_to_idx[op]
                    transition_matrix[trace_idx][op_idx] = (
                        1.0 / len(operations) if operations else 0
                    )

        # Column normalize
        for col in range(total_size):
            col_sum = np.sum(transition_matrix[:, col])
            if col_sum > 0:
                transition_matrix[:, col] /= col_sum

        return transition_matrix

    def _get_pagerank_score(
        self,
        transition_matrix: np.ndarray,
        init_vector: np.ndarray,
        preference_vector: np.ndarray,
        operation_vector: List[str],
        d: float,
        epsilon: float,
    ) -> Dict[str, float]:
        """Calculate PageRank scores."""
        if transition_matrix.size == 0 or init_vector.size == 0:
            return {}

        current_vector = init_vector.copy()

        # Power iteration
        for _ in range(100):  # Max iterations
            new_vector = (1 - d) * np.dot(
                transition_matrix, current_vector
            ) + d * preference_vector

            # Check convergence
            if np.linalg.norm(new_vector - current_vector) < epsilon:
                break

            current_vector = new_vector

        # Extract operation scores
        pagerank_scores = {}
        for i, operation in enumerate(operation_vector):
            if i < len(current_vector):
                pagerank_scores[operation] = float(current_vector[i][0])

        return pagerank_scores

    def _weighted_spectrum_ranker(
        self,
        normal_trace_cover_dict: Dict,
        abnormal_trace_cover_dict: Dict,
        normal_pagerank_score: Dict,
        abnormal_pagerank_score: Dict,
        normal_trace_count: Dict,
        abnormal_trace_count: Dict,
    ) -> Dict[str, float]:
        """Calculate weighted spectrum-based ranking."""
        spectrum_scores = {}

        all_operations = set(normal_trace_cover_dict.keys()) | set(
            abnormal_trace_cover_dict.keys()
        )

        total_abnormal_traces = (
            sum(abnormal_trace_count.values()) if abnormal_trace_count else 1
        )

        for operation in all_operations:
            # Spectrum-based metrics
            o_ef = abnormal_trace_cover_dict.get(
                operation, 0
            )  # Failed executions with operation
            o_ep = normal_trace_cover_dict.get(
                operation, 0
            )  # Passed executions with operation
            o_nf = total_abnormal_traces - o_ef  # Failed executions without operation

            # Ochiai coefficient
            denominator = np.sqrt((o_ef + o_nf) * (o_ef + o_ep))
            ochiai_score = o_ef / denominator if denominator > 0 else 0

            # PageRank weights
            normal_weight = normal_pagerank_score.get(operation, 0)
            abnormal_weight = abnormal_pagerank_score.get(operation, 0)

            # Combined score
            pagerank_factor = abnormal_weight - normal_weight
            spectrum_scores[operation] = ochiai_score * (1 + pagerank_factor)

        # Sort by score
        sorted_scores = dict(
            sorted(spectrum_scores.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_scores
