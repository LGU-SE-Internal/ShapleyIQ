"""
MicroHECL algorithm implementation.

This module implements the MicroHECL baseline algorithm for root cause analysis.
"""

from typing import Dict, List, Optional, Set

import numpy as np
from rcabench_platform.v2.logging import logger

from ..data_structures import Edge, RCAData
from ..utils import pearson_correlation
from .base import BaseRCAAlgorithm


class MicroHECL(BaseRCAAlgorithm):
    """
    MicroHECL algorithm for microservice root cause analysis.

    This algorithm performs anomaly propagation analysis across the service call graph
    to identify potential root causes based on correlation patterns.
    """

    def __init__(self, time_window: int = 15):
        """
        Initialize MicroHECL algorithm.

        Args:
            time_window: Time window in minutes for analysis
        """
        super().__init__("MicroHECL")
        self.time_window = time_window
        self.metric_list = ["MaxDuration", "QPS", "EC"]  # 支持完整的metric列表
        self.adjacency_node_table = {}

    def analyze(self, data: RCAData, **kwargs) -> Dict[str, float]:
        """
        Perform MicroHECL root cause analysis.

        Args:
            data: Processed RCA data
            **kwargs: Additional parameters
                - initial_anomalous_node: Starting node for propagation analysis
                - detect_metrics: List of metrics to analyze ['RT', 'EC', 'QPS']

        Returns:
            Dictionary mapping node IDs to root cause scores
        """
        initial_anomalous_node = kwargs.get("initial_anomalous_node", data.root_id)
        detect_metrics = kwargs.get("detect_metrics", ["RT"])

        if not initial_anomalous_node:
            # Find an anomalous node from the data
            initial_anomalous_node = self._find_initial_anomalous_node(data)
            if not initial_anomalous_node:
                logger.warning("No initial anomalous node found")
                return {}

        # Build adjacency table
        self.adjacency_node_table = self._build_adjacency_table(data.edges)

        # Prepare data structures
        self._prepare_data(data)

        # Run anomaly propagation analysis
        candidate_list = []
        for metric in detect_metrics:
            if metric in ["RT", "EC", "QPS"]:
                root_cause_candidates = self._anomaly_propagation_analysis(
                    initial_anomalous_node, metric
                )
                candidate_list.extend(root_cause_candidates)

        # Remove duplicates and the initial node
        candidate_list = list(set(candidate_list))
        if initial_anomalous_node in candidate_list and len(candidate_list) > 1:
            candidate_list.remove(initial_anomalous_node)

        # Rank candidates
        ranked_results = self._candidate_root_cause_ranking(
            candidate_list, initial_anomalous_node
        )

        return ranked_results

    def _find_initial_anomalous_node(self, data: RCAData) -> Optional[str]:
        """Find an initial anomalous node from the data."""
        # Simple heuristic: find node with highest metric value
        max_value = 0
        max_node = None

        for node_id in data.node_ids:
            if node_id in data.ts_data_dict:
                metrics = data.ts_data_dict[node_id]
                for metric_name, values in metrics.items():
                    if values and max(values) > max_value:
                        max_value = max(values)
                        max_node = node_id

        return max_node

    def _build_adjacency_table(
        self, edges: List[Edge]
    ) -> Dict[str, Dict[str, Set[str]]]:
        """
        Build adjacency table from edges.

        Args:
            edges: List of edges

        Returns:
            Adjacency table mapping node IDs to in/out neighbors
        """
        adjacency_table = {}

        for edge in edges:
            # Initialize nodes if not present
            for node_id in [edge.source_id, edge.target_id]:
                if node_id not in adjacency_table:
                    adjacency_table[node_id] = {"in_nodes": set(), "out_nodes": set()}

            # Add edge relationships (avoid self-loops)
            if edge.source_id != edge.target_id:
                adjacency_table[edge.source_id]["out_nodes"].add(edge.target_id)
                adjacency_table[edge.target_id]["in_nodes"].add(edge.source_id)

        return adjacency_table

    def _prepare_data(self, data: RCAData) -> None:
        """
        Prepare data structures needed for analysis.

        Args:
            data: RCA data
        """
        self.nodes_id = data.node_ids
        self.trace_data_dict = data.trace_data_dict
        self.request_timestamp = data.request_timestamp
        self.ts_data_dict = data.ts_data_dict or {}
        self.metrics_statistical_data = data.metrics_statistical_data or {}
        self.metrics_threshold = data.metrics_threshold or {}

        # Ensure all nodes have metric data
        for node_id in self.nodes_id:
            if node_id not in self.ts_data_dict:
                self.ts_data_dict[node_id] = {metric: [] for metric in self.metric_list}
                # 确保有MaxDuration数据
                if "MaxDuration" not in self.ts_data_dict[node_id]:
                    self.ts_data_dict[node_id]["MaxDuration"] = []
            if node_id not in self.metrics_statistical_data:
                self.metrics_statistical_data[node_id] = {
                    metric: [0, 1, 0] for metric in self.metric_list
                }
                # 确保有MaxDuration的统计数据
                if "MaxDuration" not in self.metrics_statistical_data[node_id]:
                    self.metrics_statistical_data[node_id]["MaxDuration"] = [0, 1, 0]
            if node_id not in self.metrics_threshold:
                self.metrics_threshold[node_id] = {
                    metric: [0, float("inf")] for metric in self.metric_list
                }
                # 确保有MaxDuration的阈值
                if "MaxDuration" not in self.metrics_threshold[node_id]:
                    self.metrics_threshold[node_id]["MaxDuration"] = [0, float("inf")]

    def _anomaly_detection(
        self, node_id: str, metric_type: str, use_trace_data: bool = True
    ) -> bool:
        """
        Detect anomalies for a specific node and metric.

        Args:
            node_id: Node identifier
            metric_type: Type of metric ('RT', 'EC', 'QPS')
            use_trace_data: Whether to use trace data or time series data

        Returns:
            True if anomaly detected, False otherwise
        """
        metric_type_key_dict = {
            "RT": "MaxDuration",  # 修正：使用MaxDuration而不是Duration
            "EC": "EC", 
            "QPS": "QPS"
        }

        metric_key = metric_type_key_dict.get(metric_type, "MaxDuration")

        # Get time series data
        ts_data = self.ts_data_dict.get(node_id, {}).get(metric_key, [])
        if not ts_data:
            return False

        # Prepare data for analysis
        normal_data = (
            ts_data[: -self.time_window] if len(ts_data) > self.time_window else []
        )

        if use_trace_data and node_id in self.trace_data_dict:
            # Use trace data for detection
            trace_durations = self.trace_data_dict[node_id].get("Duration", [])
            data_to_detect = trace_durations
        else:
            # Use recent time series data
            data_to_detect = (
                ts_data[-self.time_window :]
                if len(ts_data) >= self.time_window
                else ts_data
            )

        if not data_to_detect:
            return False

        # Get statistical baseline
        statistics = self.metrics_statistical_data.get(node_id, {}).get(
            metric_key, [None, None, None]
        )
        ymean = (
            float(statistics[0])
            if statistics[0] is not None
            else (np.mean(normal_data) if normal_data else 0)
        )
        ystd = (
            float(statistics[1])
            if statistics[1] is not None
            else (np.std(normal_data) if normal_data else 1)
        )

        # Calculate thresholds
        threshold1 = ymean - 5 * ystd
        threshold2 = ymean + 5 * ystd

        # Use configured thresholds if available
        upper_threshold = self.metrics_threshold.get(node_id, {}).get(
            metric_key, [None, None]
        )
        if upper_threshold[1] is not None and upper_threshold[1] < threshold2:
            threshold2 = upper_threshold[1]

        # Check for anomalies
        for value in data_to_detect:
            if value < threshold1 or value > threshold2:
                # logger.debug(f"Anomaly detected for {node_id} {metric_type}: {value}")
                return True

        # logger.debug(f"No anomaly for {node_id} {metric_type}")
        return False

    def _get_next_correlated_anomalous_nodes(
        self, current_node: str, metric_type: str
    ) -> List[str]:
        """
        Find next correlated anomalous nodes in the propagation path.

        Args:
            current_node: Current node in propagation
            metric_type: Type of metric being analyzed

        Returns:
            List of correlated anomalous node IDs
        """
        metric_type_key_dict = {
            "RT": "MaxDuration",  # 使用MaxDuration而不是Duration
            "EC": "EC",
            "QPS": "QPS"
        }

        correlated_nodes = []

        if current_node not in self.adjacency_node_table:
            return correlated_nodes

        # 根据metric类型决定传播方向
        if metric_type in ['RT', 'EC']:
            # RT和EC向下游传播
            next_nodes = self.adjacency_node_table[current_node]["out_nodes"]
        elif metric_type in ['QPS']:
            # QPS向上游传播
            next_nodes = self.adjacency_node_table[current_node]["in_nodes"]
        else:
            return correlated_nodes

        metric_key = metric_type_key_dict.get(metric_type, "MaxDuration")

        for candidate_node in next_nodes:
            # 检查候选节点是否异常
            if self._anomaly_detection(candidate_node, metric_type):
                # 计算相关性
                current_node_ts_data = self.ts_data_dict.get(current_node, {}).get(metric_key, [])
                candidate_node_ts_data = self.ts_data_dict.get(candidate_node, {}).get(metric_key, [])
                
                if current_node_ts_data and candidate_node_ts_data:
                    current_data = current_node_ts_data[-self.time_window:] if len(current_node_ts_data) >= self.time_window else current_node_ts_data
                    candidate_data = candidate_node_ts_data[-self.time_window:] if len(candidate_node_ts_data) >= self.time_window else candidate_node_ts_data
                    
                    correlation = pearson_correlation(current_data, candidate_data, default_value=0.01)
                    
                    # 使用原始的相关性阈值：只要 > 0 就认为相关
                    if correlation > 0:
                        correlated_nodes.append(candidate_node)

        return correlated_nodes

    def _calculate_correlation(self, node1: str, node2: str, metric_type: str) -> float:
        """
        Calculate correlation between two nodes for a specific metric.

        Args:
            node1: First node ID
            node2: Second node ID
            metric_type: Type of metric

        Returns:
            Correlation coefficient
        """
        metric_type_key_dict = {"RT": "Duration", "EC": "EC", "QPS": "QPS"}

        metric_key = metric_type_key_dict.get(metric_type, "Duration")

        # Get time series data for both nodes
        data1 = self.ts_data_dict.get(node1, {}).get(metric_key, [])
        data2 = self.ts_data_dict.get(node2, {}).get(metric_key, [])

        if not data1 or not data2:
            return 0.01  # Default low correlation

        # Use recent data for correlation calculation
        recent_data1 = (
            data1[-self.time_window :] if len(data1) >= self.time_window else data1
        )
        recent_data2 = (
            data2[-self.time_window :] if len(data2) >= self.time_window else data2
        )

        return pearson_correlation(recent_data1, recent_data2, default_value=0.01)

    def _anomaly_propagation_analysis(
        self, initial_node: str, metric_type: str
    ) -> List[str]:
        """
        Perform anomaly propagation analysis starting from initial node.

        Args:
            initial_node: Starting node for propagation
            metric_type: Type of metric to analyze

        Returns:
            List of root cause candidate nodes
        """
        root_cause_candidates = []
        visited_nodes = set()
        current_nodes = [initial_node]

        while current_nodes:
            next_nodes = []

            for current_node in current_nodes:
                if current_node in visited_nodes:
                    continue

                visited_nodes.add(current_node)

                # Get correlated anomalous nodes
                correlated_nodes = self._get_next_correlated_anomalous_nodes(
                    current_node, metric_type
                )

                if not correlated_nodes:
                    # No more correlated nodes found, this could be a root cause
                    root_cause_candidates.append(current_node)
                else:
                    # Continue propagation
                    next_nodes.extend(correlated_nodes)

            current_nodes = [node for node in next_nodes if node not in visited_nodes]

        return root_cause_candidates

    def _candidate_root_cause_ranking(
        self, candidate_list: List[str], entry_node: str
    ) -> Dict[str, float]:
        """
        Rank root cause candidates based on correlation with entry node.

        Args:
            candidate_list: List of candidate node IDs
            entry_node: Entry point node

        Returns:
            Dictionary mapping node IDs to ranking scores
        """
        if not candidate_list:
            return {}

        scores = []
        for candidate in candidate_list:
            if candidate is None:  # Handle None candidates
                continue
                
            # 使用MaxDuration计算与entry_node的相关性
            candidate_ts_data = self.ts_data_dict.get(candidate, {}).get("MaxDuration", [])
            entry_node_ts_data = self.ts_data_dict.get(entry_node, {}).get("MaxDuration", [])
            
            if candidate_ts_data and entry_node_ts_data:
                candidate_data = candidate_ts_data[-self.time_window:] if len(candidate_ts_data) >= self.time_window else candidate_ts_data
                entry_data = entry_node_ts_data[-self.time_window:] if len(entry_node_ts_data) >= self.time_window else entry_node_ts_data
                
                correlation = pearson_correlation(candidate_data, entry_data, default_value=0.01)
                scores.append(correlation)
            else:
                scores.append(0.01)

        # 构建排序后的结果字典
        ranked_scores = {}
        for i in np.argsort(scores)[::-1]:  # 按相关性降序排列
            if i < len(candidate_list) and candidate_list[i] is not None:
                ranked_scores[candidate_list[i]] = scores[i]

        return ranked_scores
