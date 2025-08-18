"""
Tests for ShapleyIQ data structures.
"""


def test_trace_data_creation(sample_trace_data):
    """Test TraceData creation and validation."""
    assert sample_trace_data.trace_id == "test-trace-001"
    assert len(sample_trace_data.spans) == 3
    assert sample_trace_data.has_anomaly is True
    assert sample_trace_data.duration == 50000


def test_service_node_creation(sample_service_nodes):
    """Test ServiceNode creation and properties."""
    frontend = sample_service_nodes[0]
    assert frontend.service_name == "frontend"
    assert frontend.avg_response_time == 50.0
    assert frontend.error_rate == 0.01


def test_edge_creation(sample_edges):
    """Test Edge creation and properties."""
    edge = sample_edges[0]
    assert edge.source == "frontend"
    assert edge.target == "user-service"
    assert edge.call_count == 100


def test_rca_data_validation(sample_rca_data):
    """Test RCAData validation and adjacency graph."""
    # Test basic properties
    assert len(sample_rca_data.traces) == 1
    assert len(sample_rca_data.service_nodes) == 3
    assert len(sample_rca_data.edges) == 2

    # Test adjacency graph construction
    adj_graph = sample_rca_data.get_adjacency_graph()
    assert "frontend" in adj_graph
    assert "user-service" in adj_graph["frontend"]

    # Test service metrics
    metrics = sample_rca_data.get_service_metrics("frontend")
    assert metrics["avg_response_time"] == 50.0
    assert metrics["error_rate"] == 0.01


def test_rca_data_service_names(sample_rca_data):
    """Test service name extraction."""
    service_names = sample_rca_data.get_service_names()
    expected_services = {"frontend", "user-service", "database"}
    assert set(service_names) == expected_services


def test_rca_data_anomaly_traces(sample_rca_data):
    """Test anomaly trace filtering."""
    anomaly_traces = sample_rca_data.get_anomaly_traces()
    assert len(anomaly_traces) == 1
    assert anomaly_traces[0].has_anomaly is True
