"""
Data structures for RCA algorithms.

This module defines the core data structures used by all RCA algorithms,
including trace data, edge representations, and service nodes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from rcabench_platform.v2.logging import logger


@dataclass
class Edge:
    """Represents a directed edge between two services in a call graph."""

    source_id: str
    target_id: str
    label: str = "default"
    properties: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.source_id, self.target_id, self.label))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (
            self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.label == other.label
        )


@dataclass
class ServiceNode:
    """Represents a service node in the microservice architecture."""

    node_id: str
    service_name: str
    server_ip: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if not isinstance(other, ServiceNode):
            return False
        return self.node_id == other.node_id


@dataclass
class TraceData:
    """Represents trace data for a single request."""

    trace_id: str
    spans: List[Dict[str, Any]]
    root_span_id: Optional[str] = None
    timestamp: Optional[int] = None

    def __post_init__(self):
        if self.root_span_id is None and self.spans:
            # Try to find root span (span with no parent)
            for span in self.spans:
                if "parentSpanId" not in span or span["parentSpanId"] is None:
                    self.root_span_id = span.get("spanId")
                    break


@dataclass
class RCAData:
    """
    Core data structure containing all information needed for RCA algorithms.

    This structure separates the data preprocessing from algorithm execution,
    allowing for clean separation of concerns.
    """

    # Graph structure
    edges: List[Edge] = field(default_factory=list)
    nodes: List[ServiceNode] = field(default_factory=list)
    node_ids: List[str] = field(default_factory=list)

    # Root cause information
    root_causes: List[str] = field(default_factory=list)
    root_id: Optional[str] = None
    root_id_list: List[str] = field(default_factory=list)

    # Trace data
    traces: Optional[List[TraceData]] = None
    spans: Optional[List[Dict[str, Any]]] = None  # For backward compatibility
    trace_data_dict: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    request_timestamp: Optional[int] = None

    # Time series metrics
    ts_data_dict: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    metrics_statistical_data: Dict[str, Dict[str, List[float]]] = field(
        default_factory=dict
    )
    metrics_threshold: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    # Machine metrics
    operation_ip_dict: Dict[str, str] = field(default_factory=dict)
    ip_ts_data_dict: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    # For MicroRank algorithm
    normal_trace_dict: Dict[str, Any] = field(default_factory=dict)
    abnormal_trace_dict: Dict[str, Any] = field(default_factory=dict)
    trace_dict: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metric_list: List[str] = field(default_factory=lambda: ["MaxDuration", "Duration"])
    input_path: Optional[str] = None
    ip_mapping: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Initialize derived fields after creation."""
        if not self.node_ids and self.nodes:
            self.node_ids = [node.node_id for node in self.nodes]

        if not self.operation_ip_dict and self.nodes:
            self.operation_ip_dict = {
                node.node_id: node.server_ip for node in self.nodes
            }

    @property
    def is_empty(self) -> bool:
        """Check if the data structure contains meaningful data."""
        return not self.edges and not self.nodes and not self.traces and not self.spans

    def add_edge(self, source_id: str, target_id: str, label: str = "default") -> None:
        """Add an edge to the graph."""
        edge = Edge(source_id=source_id, target_id=target_id, label=label)
        if edge not in self.edges:
            self.edges.append(edge)

    def add_node(self, node_id: str, service_name: str, server_ip: str) -> None:
        """Add a service node."""
        node = ServiceNode(
            node_id=node_id, service_name=service_name, server_ip=server_ip
        )
        if node not in self.nodes:
            self.nodes.append(node)
            if node_id not in self.node_ids:
                self.node_ids.append(node_id)
            self.operation_ip_dict[node_id] = server_ip

    def get_adjacency_dict(self) -> Dict[str, Set[str]]:
        """Get adjacency representation of the graph."""
        adjacency = {}
        for edge in self.edges:
            if edge.source_id not in adjacency:
                adjacency[edge.source_id] = set()
            adjacency[edge.source_id].add(edge.target_id)
        return adjacency

    def get_node_by_id(self, node_id: str) -> Optional[ServiceNode]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def validate(self) -> bool:
        """Validate the data structure for consistency."""
        try:
            # Check that all edge endpoints exist in nodes
            node_id_set = set(self.node_ids)
            for edge in self.edges:
                if edge.source_id not in node_id_set:
                    logger.warning(f"Edge source {edge.source_id} not found in nodes")
                    return False
                if edge.target_id not in node_id_set:
                    logger.warning(f"Edge target {edge.target_id} not found in nodes")
                    return False

            # Check trace data consistency
            if self.trace_data_dict:
                for node_id in self.trace_data_dict:
                    if node_id not in node_id_set:
                        logger.warning(f"Trace data for unknown node {node_id}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
