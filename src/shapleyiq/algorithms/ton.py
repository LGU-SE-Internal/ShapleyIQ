"""
TON (Trace-based Opinion Network) algorithm implementation.

This module implements the TON baseline algorithm for root cause analysis.
"""

from typing import Dict, List

import numpy as np
from rcabench_platform.v2.logging import logger

from ..data_structures import RCAData
from ..utils import pearson_correlation
from .base import BaseRCAAlgorithm


class TON(BaseRCAAlgorithm):
    """
    TON (Trace-based Opinion Network) algorithm for microservice root cause analysis.

    This algorithm builds an opinion network based on service correlations
    and propagates opinions to identify potential root causes.
    """

    def __init__(self, time_window: int = 15):
        """
        Initialize TON algorithm.

        Args:
            time_window: Time window in minutes for analysis
        """
        super().__init__("TON")
        self.time_window = time_window
        self.metric_list = ["MaxDuration", "QPS", "EC"]  # 支持完整的metric列表
        self.machine_metrics = [
            "k8s.pod.cpu_limit_utilization",
            "k8s.pod.memory_limit_utilization",
        ]

    def analyze(self, data: RCAData, **kwargs) -> Dict[str, float]:
        """
        Perform TON root cause analysis.

        Args:
            data: Processed RCA data
            **kwargs: Additional parameters
                - operation_only: Whether to return only operation-level results
                - initial_anomalous_node: Starting node for analysis (optional)
                - anomalous_services: List of anomalous services (optional)

        Returns:
            Dictionary mapping node IDs to root cause scores
        """
        operation_only = kwargs.get("operation_only", True)
        initial_anomalous_node = kwargs.get("initial_anomalous_node")
        anomalous_services = kwargs.get("anomalous_services", [])

        # Prepare data structures
        self._prepare_data(data, initial_anomalous_node, anomalous_services)

        # Find anomalous nodes
        anomalous_nodes = self._get_anomalous_nodes()

        if not anomalous_nodes:
            logger.warning("No anomalous nodes found")
            return {}

        # Build opinion network
        opinion_network = self._build_opinion_network()

        # Initialize opinions based on anomalous nodes
        initial_opinions = self._initialize_opinions(anomalous_nodes)

        # Propagate opinions through the network
        final_opinions = self._propagate_opinions(opinion_network, initial_opinions)

        # Filter results if needed
        if operation_only:
            operation_opinions = {}
            for node_id, opinion in final_opinions.items():
                if node_id in self.nodes_id:
                    operation_opinions[node_id] = opinion
            return operation_opinions

        return final_opinions

    def _prepare_data(
        self, data: RCAData, initial_anomalous_node=None, anomalous_services=None
    ) -> None:
        """Prepare data structures for analysis."""
        self.edges = data.edges
        self.nodes_id = data.node_ids
        self.initial_anomalous_node = (
            initial_anomalous_node or data.root_id
        )  # 使用initial_anomalous_node而不是root_id
        self.anomalous_services = anomalous_services or []  # 添加异常服务列表
        self.trace_data_dict = data.trace_data_dict
        self.request_timestamp = data.request_timestamp
        self.ts_data_dict = data.ts_data_dict or {}
        self.metrics_statistical_data = data.metrics_statistical_data or {}
        self.metrics_threshold = data.metrics_threshold or {}

        # Build operation-IP mapping (使用service_name作为IP映射)
        self.operation_ip_dict = {}
        for node in data.nodes:
            # 使用service_name作为"IP"，因为我们的数据没有真实IP
            service_name = (
                node.node_id.split(":")[0] if ":" in node.node_id else node.node_id
            )
            self.operation_ip_dict[node.node_id] = service_name

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
        self, node_id: str, metric_type: str = "RT", use_trace_data: bool = True
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
        # 修正：使用正确的metric映射，与数据适配器和MicroHECL保持一致
        metric_type_key_dict = {"RT": "MaxDuration", "EC": "EC", "QPS": "QPS"}

        metric_key = metric_type_key_dict.get(metric_type, "MaxDuration")

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

        如果有anomalous_services（从conclusion检测的异常服务），
        则查找包含这些服务的所有节点；否则使用原有的异常检测逻辑。

        Returns:
            List of anomalous node IDs
        """
        anomalous_nodes = []

        # 如果有detected anomalous services，优先使用
        if hasattr(self, "anomalous_services") and self.anomalous_services:
            for service in self.anomalous_services:
                # 找到包含该服务的所有节点
                for node_id in self.nodes_id:
                    if service in node_id:
                        anomalous_nodes.append(node_id)

            if anomalous_nodes:
                logger.info(
                    f"TON using detected anomalous services: {self.anomalous_services}"
                )
                logger.info(
                    f"Found {len(anomalous_nodes)} anomalous nodes from detected services"
                )
                return anomalous_nodes

        # 如果有单个initial_anomalous_node，优先使用
        if hasattr(self, "initial_anomalous_node") and self.initial_anomalous_node:
            # 提取服务名称
            if ":" in self.initial_anomalous_node:
                service_name = self.initial_anomalous_node.split(":")[0]
            else:
                service_name = self.initial_anomalous_node

            # 找到包含该服务的所有节点
            for node_id in self.nodes_id:
                if service_name in node_id:
                    anomalous_nodes.append(node_id)

            if anomalous_nodes:
                logger.info(f"TON using initial anomalous node service: {service_name}")
                logger.info(
                    f"Found {len(anomalous_nodes)} anomalous nodes from initial service"
                )
                return anomalous_nodes

        # 回退到原有的异常检测逻辑
        logger.info(
            "TON: No detected anomalous services, using RT-based anomaly detection"
        )
        for node_id in self.nodes_id:
            if self._anomaly_detection(node_id, "RT"):
                anomalous_nodes.append(node_id)

        return anomalous_nodes

    def _build_opinion_network(self) -> Dict[str, Dict[str, float]]:
        """
        Build opinion network based on service correlations.

        Returns:
            Dictionary representing the opinion network with weights
        """
        opinion_network = {}

        # Initialize network structure
        all_nodes = set(self.nodes_id)
        for ip in self.operation_ip_dict.values():
            all_nodes.add(ip)

        for node in all_nodes:
            opinion_network[node] = {}

        # Add edges with correlation weights
        for edge in self.edges:
            if edge.source_id in all_nodes and edge.target_id in all_nodes:
                correlation = self._calculate_correlation(
                    edge.source_id, edge.target_id
                )
                opinion_network[edge.source_id][edge.target_id] = correlation

        # Add operation-to-machine connections
        for operation, host_ip in self.operation_ip_dict.items():
            if operation in all_nodes and host_ip in all_nodes:
                correlation = self._calculate_correlation(operation, host_ip)
                opinion_network[operation][host_ip] = correlation

        return opinion_network

    def _calculate_correlation(self, node1: str, node2: str) -> float:
        """
        Calculate correlation between two nodes.

        For machine nodes (service names acting as IPs), use machine metrics.
        For service nodes, use application metrics.

        Args:
            node1: First node ID
            node2: Second node ID

        Returns:
            Correlation coefficient
        """
        # Check if either node is a machine node (service name acting as IP)
        is_node1_machine = node1 in self.ip_list
        is_node2_machine = node2 in self.ip_list

        # Get reference data (usually root service MaxDuration)
        if hasattr(self, "initial_anomalous_node") and self.initial_anomalous_node:
            reference_data = self.ts_data_dict.get(self.initial_anomalous_node, {}).get(
                "MaxDuration", []
            )
        else:
            # Use first available service data as reference
            for service_id in self.nodes_id:
                reference_data = self.ts_data_dict.get(service_id, {}).get(
                    "MaxDuration", []
                )
                if reference_data:
                    break
            else:
                return 0.1

        # Calculate correlation based on node types
        if is_node1_machine:
            # node1 is a machine, calculate max correlation across all machine metrics
            max_correlation = 0.1  # default
            machine_data = self.ip_ts_data_dict.get(node1, {})

            for metric in self.machine_metrics:
                if metric in machine_data:
                    metric_data = machine_data[metric]
                    if metric_data:
                        recent_metric = (
                            metric_data[-self.time_window :]
                            if len(metric_data) >= self.time_window
                            else metric_data
                        )
                        recent_ref = (
                            reference_data[-self.time_window :]
                            if len(reference_data) >= self.time_window
                            else reference_data
                        )

                        correlation = pearson_correlation(
                            recent_metric, recent_ref, default_value=0.1
                        )
                        max_correlation = max(max_correlation, correlation)

            return max_correlation

        elif is_node2_machine:
            # node2 is a machine, similar logic
            max_correlation = 0.1
            machine_data = self.ip_ts_data_dict.get(node2, {})

            for metric in self.machine_metrics:
                if metric in machine_data:
                    metric_data = machine_data[metric]
                    if metric_data:
                        recent_metric = (
                            metric_data[-self.time_window :]
                            if len(metric_data) >= self.time_window
                            else metric_data
                        )
                        recent_ref = (
                            reference_data[-self.time_window :]
                            if len(reference_data) >= self.time_window
                            else reference_data
                        )

                        correlation = pearson_correlation(
                            recent_metric, recent_ref, default_value=0.1
                        )
                        max_correlation = max(max_correlation, correlation)

            return max_correlation

        else:
            # Both are service nodes, use application metrics
            data1 = self.ts_data_dict.get(node1, {}).get("MaxDuration", [])
            data2 = self.ts_data_dict.get(node2, {}).get("MaxDuration", [])

            if not data1 or not data2:
                return 0.1

            recent_data1 = (
                data1[-self.time_window :] if len(data1) >= self.time_window else data1
            )
            recent_data2 = (
                data2[-self.time_window :] if len(data2) >= self.time_window else data2
            )

            return pearson_correlation(recent_data1, recent_data2, default_value=0.1)

    def _initialize_opinions(self, anomalous_nodes: List[str]) -> Dict[str, float]:
        """
        Initialize opinions based on anomalous nodes.

        Args:
            anomalous_nodes: List of anomalous node IDs

        Returns:
            Dictionary mapping node IDs to initial opinion values
        """
        initial_opinions = {}

        # Initialize all nodes
        all_nodes = set(self.nodes_id)
        for ip in self.operation_ip_dict.values():
            all_nodes.add(ip)

        for node in all_nodes:
            if node in anomalous_nodes:
                initial_opinions[node] = 1.0  # High suspicion for anomalous nodes
            else:
                initial_opinions[node] = 0.1  # Low baseline suspicion

        return initial_opinions

    def _propagate_opinions(
        self,
        opinion_network: Dict[str, Dict[str, float]],
        initial_opinions: Dict[str, float],
        max_iterations: int = 100,
        convergence_threshold: float = 1e-4,
    ) -> Dict[str, float]:
        """
        Propagate opinions through the network.

        Args:
            opinion_network: Network structure with weights
            initial_opinions: Initial opinion values
            max_iterations: Maximum number of iterations
            convergence_threshold: Convergence threshold

        Returns:
            Final opinion values after propagation
        """
        current_opinions = initial_opinions.copy()

        for iteration in range(max_iterations):
            new_opinions = {}

            for node in current_opinions:
                # Calculate new opinion based on neighbors
                neighbor_influences = []
                total_weight = 0

                # Collect influences from neighbors
                for neighbor, weight in opinion_network.get(node, {}).items():
                    if neighbor in current_opinions:
                        neighbor_influences.append(current_opinions[neighbor] * weight)
                        total_weight += weight

                # Calculate weighted average with self-influence
                self_weight = 0.3  # Weight for maintaining current opinion
                if total_weight > 0:
                    neighbor_influence = sum(neighbor_influences) / total_weight
                    new_opinion = (
                        self_weight * current_opinions[node]
                        + (1 - self_weight) * neighbor_influence
                    )
                else:
                    new_opinion = current_opinions[node] * self_weight

                new_opinions[node] = max(0, min(1, new_opinion))  # Clamp to [0, 1]

            # Check convergence
            max_change = max(
                abs(new_opinions[node] - current_opinions[node])
                for node in current_opinions
            )

            current_opinions = new_opinions

            if max_change < convergence_threshold:
                # logger.debug(f"TON converged after {iteration + 1} iterations")
                break

        # Sort by opinion value
        sorted_opinions = dict(
            sorted(current_opinions.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_opinions
