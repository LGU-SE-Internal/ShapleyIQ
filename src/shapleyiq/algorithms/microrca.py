"""
MicroRCA algorithm implementation.

This module implements the MicroRCA baseline algorithm for root cause analysis.
"""

import random
from typing import Dict, List, Set

import numpy as np
from rcabench_platform.v2.logging import logger

from ..data_structures import RCAData
from ..utils import pearson_correlation
from .base import BaseRCAAlgorithm


class MicroRCA(BaseRCAAlgorithm):
    """
    MicroRCA algorithm for microservice root cause analysis.

    This algorithm uses Personalized PageRank on an anomalous subgraph
    to identify root causes based on random walk analysis.
    """

    def __init__(self, time_window: int = 15):
        """
        Initialize MicroRCA algorithm.

        Args:
            time_window: Time window in minutes for analysis
        """
        super().__init__("MicroRCA")
        self.time_window = time_window
        self.metric_list = ["MaxDuration"]
        self.machine_metrics = [
            "node_cpu_utilization",
            "memory_utilization",
            "root_partition_utilization",
            "node_sockstat_TCP_tw",
        ]

    def analyze(self, data: RCAData, **kwargs) -> Dict[str, float]:
        """
        Perform MicroRCA root cause analysis.

        Args:
            data: Processed RCA data
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping node IDs to root cause scores
        """
        # Prepare data structures
        self._prepare_data(data)

        # Find anomalous nodes
        self.anomalous_nodes_list = self._get_anomalous_nodes()

        if not self.anomalous_nodes_list:
            logger.warning("No anomalous nodes found")
            return {}

        # Extract anomalous subgraph
        self.anomalous_graph = self._anomalous_graph_extraction()

        if not self.anomalous_graph:
            logger.warning("No anomalous graph found")
            return {}

        # Calculate edge weights
        self.edge_weight = self._edge_weighting(alpha=0.5)

        # Perform personalized PageRank
        sorted_score = self._localizing_root_cause(steps_threshold=10000)

        return sorted_score

    def _prepare_data(self, data: RCAData) -> None:
        """Prepare data structures for analysis."""
        self.edges = data.edges
        self.nodes_id = data.node_ids
        self.trace_data_dict = data.trace_data_dict
        self.request_timestamp = data.request_timestamp
        self.ts_data_dict = data.ts_data_dict or {}
        self.metrics_statistical_data = data.metrics_statistical_data or {}
        self.metrics_threshold = data.metrics_threshold or {}

        # Build operation-IP mapping
        self.operation_ip_dict = {}
        for node in data.nodes:
            # 只有当server_ip不为空且不为"unknown"时才添加
            if node.server_ip and node.server_ip not in ["", "unknown"]:
                self.operation_ip_dict[node.node_id] = node.server_ip

        self.ip_list = list(set(self.operation_ip_dict.values()))
        self.ip_ts_data_dict = data.ip_ts_data_dict or {}

        # Ensure all nodes have required data structures
        for node_id in self.nodes_id:
            if node_id not in self.ts_data_dict:
                self.ts_data_dict[node_id] = {metric: [] for metric in self.metric_list}
            if node_id not in self.metrics_statistical_data:
                self.metrics_statistical_data[node_id] = {
                    metric: [0, 1, 0] for metric in self.metric_list
                }
            if node_id not in self.metrics_threshold:
                self.metrics_threshold[node_id] = {
                    metric: [0, float("inf")] for metric in self.metric_list
                }

    def _anomaly_detection(
        self, node_id: str, metric_type: str, use_trace_data: bool = True
    ) -> bool:
        """
        Detect anomalies for a specific node and metric.

        Args:
            node_id: Node identifier
            metric_type: Type of metric ('RT', 'EC', 'QPS')
            use_trace_data: Whether to use trace data

        Returns:
            True if anomaly detected
        """
        metric_type_key_dict = {"RT": "Duration", "EC": "EC", "QPS": "QPS"}

        metric_key = metric_type_key_dict.get(metric_type, "Duration")

        # Get time series data
        ts_data = self.ts_data_dict.get(node_id, {}).get(metric_key, [])
        if not ts_data:
            return False

        normal_data = (
            ts_data[: -self.time_window] if len(ts_data) > self.time_window else []
        )

        if use_trace_data and node_id in self.trace_data_dict:
            data_to_detect = self.trace_data_dict[node_id].get("Duration", [])
        else:
            data_to_detect = (
                ts_data[-self.time_window :]
                if len(ts_data) >= self.time_window
                else ts_data
            )

        if not data_to_detect:
            return False

        # Get statistical baseline
        statistics = self.metrics_statistical_data.get(node_id, {}).get(
            metric_key, [None, None]
        )
        ymean = (
            float(statistics[0])
            if statistics[0]
            else (np.mean(normal_data) if normal_data else 0)
        )
        ystd = (
            float(statistics[1])
            if statistics[1]
            else (np.std(normal_data) if normal_data else 1)
        )

        # Calculate thresholds
        threshold1 = ymean - 5 * ystd
        threshold2 = ymean + 5 * ystd

        # Use configured upper threshold if available
        upper_threshold = self.metrics_threshold.get(node_id, {}).get(
            metric_key, [None, None]
        )
        if upper_threshold[1] is not None and upper_threshold[1] < threshold2:
            threshold2 = upper_threshold[1]

        # Check for anomalies
        for value in data_to_detect:
            if value < threshold1 or value > threshold2:
                return True

        return False

    def _get_anomalous_nodes(self) -> List[str]:
        """
        Get list of anomalous nodes.

        Returns:
            List of anomalous node IDs
        """
        anomalous_nodes = []

        for node_id in self.nodes_id:
            if self._anomaly_detection(node_id, "RT"):
                anomalous_nodes.append(node_id)

        return anomalous_nodes

    def _anomalous_graph_extraction(self) -> Dict[str, Set[str]]:
        """
        Extract subgraph containing only anomalous nodes and their connections.

        Returns:
            Adjacency representation of anomalous subgraph
        """
        adjacency_table = {}

        # Add edges between anomalous nodes
        for edge in self.edges:
            if (
                edge.source_id in self.anomalous_nodes_list
                or edge.target_id in self.anomalous_nodes_list
            ):
                if edge.source_id not in adjacency_table:
                    adjacency_table[edge.source_id] = set()

                if edge.source_id != edge.target_id:
                    adjacency_table[edge.source_id].add(edge.target_id)

        # Add machine connections for anomalous operations
        for operation, host_ip in self.operation_ip_dict.items():
            if operation in self.anomalous_nodes_list:
                if operation not in adjacency_table:
                    adjacency_table[operation] = set()
                adjacency_table[operation].add(host_ip)

        return adjacency_table

    def _edge_weighting(self, alpha: float = 0.5) -> Dict[str, Dict[str, float]]:
        """
        Calculate weights for edges in the anomalous graph.

        Args:
            alpha: Weight parameter for anomalous connections

        Returns:
            Dictionary mapping source nodes to target nodes to weights
        """
        weighted_adjacency_table = {}

        for source_id, target_id_set in self.anomalous_graph.items():
            weighted_adjacency_table[source_id] = {}

            for target_id in target_id_set:
                if target_id in self.anomalous_nodes_list:
                    # Both nodes are anomalous
                    weighted_adjacency_table[source_id][target_id] = alpha
                elif target_id in self.ip_ts_data_dict:
                    # Target is a machine with metric data
                    if self.ip_ts_data_dict[target_id]:
                        source_ts_data = self.ts_data_dict.get(source_id, {}).get(
                            "MaxDuration", []
                        )

                        if source_ts_data:
                            correlation_list = []

                            for machine_metric in self.machine_metrics:
                                ip_ts_data = self.ip_ts_data_dict[target_id].get(
                                    machine_metric, []
                                )

                                if ip_ts_data:
                                    # Align time series data
                                    target_ts_data = [
                                        ip_ts_data[i]
                                        for i in range(-1, -len(source_ts_data) * 2, -2)
                                    ]
                                    correlation = pearson_correlation(
                                        source_ts_data[-self.time_window :],
                                        target_ts_data[-self.time_window :],
                                        default_value=0.01,
                                    )
                                    correlation_list.append(correlation)

                            weighted_adjacency_table[source_id][target_id] = (
                                alpha * max(correlation_list) if correlation_list else 0
                            )
                        else:
                            weighted_adjacency_table[source_id][target_id] = 0
                    else:
                        weighted_adjacency_table[source_id][target_id] = 0
                elif target_id in self.ts_data_dict:
                    # Target is a service with metric data
                    source_ts_data = self.ts_data_dict.get(source_id, {}).get(
                        "MaxDuration", []
                    )
                    target_ts_data = self.ts_data_dict.get(target_id, {}).get(
                        "MaxDuration", []
                    )

                    if source_ts_data and target_ts_data:
                        correlation = pearson_correlation(
                            source_ts_data[-self.time_window :],
                            target_ts_data[-self.time_window :],
                            default_value=0.01,
                        )
                        weighted_adjacency_table[source_id][target_id] = correlation
                    else:
                        weighted_adjacency_table[source_id][target_id] = 0
                else:
                    logger.debug(f"No time series data for {target_id}")
                    weighted_adjacency_table[source_id][target_id] = 0

        return weighted_adjacency_table

    def _localizing_root_cause(self, steps_threshold: int = 10000) -> Dict[str, float]:
        """
        Perform Personalized PageRank to localize root causes.

        Args:
            steps_threshold: Number of random walk steps

        Returns:
            Dictionary mapping node IDs to root cause scores
        """
        if not self.anomalous_graph:
            return {}

        # Random walk with restart
        location = random.choice(list(self.anomalous_graph.keys()))
        steps_num = 0
        steps = []

        while steps_num < steps_threshold:
            rand = random.random()

            if location in self.anomalous_graph and rand > 0.15:
                # Continue walk
                next_nodes = list(self.anomalous_graph[location])
                if next_nodes:
                    next_nodes_weight = [
                        self.edge_weight[location].get(node, 0) for node in next_nodes
                    ]

                    # Avoid division by zero
                    total_weight = sum(next_nodes_weight)
                    if total_weight > 0:
                        probabilities = [w / total_weight for w in next_nodes_weight]
                        location = np.random.choice(next_nodes, p=probabilities)
                    else:
                        location = random.choice(next_nodes)

                    steps.append(location)
                    steps_num += 1
                else:
                    # No outgoing edges, restart
                    location = random.choice(list(self.anomalous_graph.keys()))
                    steps.append(location)
                    steps_num += 1
            else:
                # Restart
                location = random.choice(list(self.anomalous_graph.keys()))
                steps.append(location)
                steps_num += 1

        # Count frequencies after burn-in period
        frequency_dict = {}
        for source_id in self.anomalous_graph:
            frequency_dict[source_id] = 0
            for target_id_list in self.anomalous_graph.values():
                for target_id in target_id_list:
                    if target_id not in frequency_dict:
                        frequency_dict[target_id] = 0

        length_burned = int(steps_threshold / 2)
        for node in steps[length_burned:]:
            if node in frequency_dict:
                frequency_dict[node] += 1

        # Convert to scores
        score = {}
        remaining_steps = steps_threshold - length_burned
        if remaining_steps > 0:
            for node, frequency in frequency_dict.items():
                score[node] = frequency / remaining_steps

        # Sort by score
        sorted_score = dict(sorted(score.items(), key=lambda x: x[1], reverse=True))

        return sorted_score
