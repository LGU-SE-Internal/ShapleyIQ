"""
Test configuration for ShapleyIQ package.
"""

import os
import sys

import pytest

# Add src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def sample_trace_data():
    """Fixture providing sample trace data for testing."""
    from shapleyiq.data_structures import TraceData

    return TraceData(
        trace_id="test-trace-001",
        spans=[
            {
                "span_id": "span-1",
                "service_name": "frontend",
                "operation_name": "GET /api/users",
                "start_time": 1000000,
                "duration": 50000,
                "parent_span_id": None,
            },
            {
                "span_id": "span-2",
                "service_name": "user-service",
                "operation_name": "get_user",
                "start_time": 1005000,
                "duration": 30000,
                "parent_span_id": "span-1",
            },
            {
                "span_id": "span-3",
                "service_name": "database",
                "operation_name": "query_users",
                "start_time": 1010000,
                "duration": 20000,
                "parent_span_id": "span-2",
            },
        ],
        start_time=1000000,
        end_time=1050000,
        duration=50000,
        has_anomaly=True,
    )


@pytest.fixture
def sample_service_nodes():
    """Fixture providing sample service nodes for testing."""
    from shapleyiq.data_structures import ServiceNode

    return [
        ServiceNode(
            service_name="frontend",
            avg_response_time=50.0,
            error_rate=0.01,
            cpu_usage=0.6,
            memory_usage=0.7,
        ),
        ServiceNode(
            service_name="user-service",
            avg_response_time=30.0,
            error_rate=0.02,
            cpu_usage=0.8,
            memory_usage=0.6,
        ),
        ServiceNode(
            service_name="database",
            avg_response_time=20.0,
            error_rate=0.005,
            cpu_usage=0.9,
            memory_usage=0.8,
        ),
    ]


@pytest.fixture
def sample_edges():
    """Fixture providing sample edges for testing."""
    from shapleyiq.data_structures import Edge

    return [
        Edge(
            source="frontend",
            target="user-service",
            weight=1.0,
            call_count=100,
            avg_latency=5.0,
        ),
        Edge(
            source="user-service",
            target="database",
            weight=1.0,
            call_count=80,
            avg_latency=15.0,
        ),
    ]


@pytest.fixture
def sample_rca_data(sample_trace_data, sample_service_nodes, sample_edges):
    """Fixture providing complete RCA data for testing."""
    from shapleyiq.data_structures import RCAData

    return RCAData(
        traces=[sample_trace_data],
        service_nodes=sample_service_nodes,
        edges=sample_edges,
        anomaly_window_start=1000000,
        anomaly_window_end=1100000,
    )
