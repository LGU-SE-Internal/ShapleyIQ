"""
Tests for ShapleyIQ preprocessing modules.
"""

import json
import os
import tempfile


def test_trace_preprocessor():
    """Test TracePreprocessor functionality."""
    from shapleyiq.preprocessing import TracePreprocessor

    # Create sample trace data
    sample_traces = [
        {
            "trace_id": "trace-001",
            "spans": [
                {
                    "span_id": "span-1",
                    "service_name": "frontend",
                    "operation_name": "GET /api",
                    "start_time": 1000000,
                    "duration": 50000,
                    "parent_span_id": None,
                }
            ],
        }
    ]

    preprocessor = TracePreprocessor()

    # Test trace processing
    processed = preprocessor.process_traces(sample_traces)
    assert len(processed) == 1
    assert processed[0].trace_id == "trace-001"
    assert len(processed[0].spans) == 1


def test_trace_preprocessor_from_file():
    """Test TracePreprocessor with file input."""
    from shapleyiq.preprocessing import TracePreprocessor

    # Create temporary file with trace data
    sample_data = [
        {
            "trace_id": "trace-001",
            "spans": [
                {
                    "span_id": "span-1",
                    "service_name": "service-a",
                    "operation_name": "operation-1",
                    "start_time": 1000000,
                    "duration": 10000,
                    "parent_span_id": None,
                }
            ],
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_data, f)
        temp_file = f.name

    try:
        preprocessor = TracePreprocessor()
        traces = preprocessor.process_traces_from_file(temp_file, format_type="generic")

        assert len(traces) == 1
        assert traces[0].trace_id == "trace-001"
    finally:
        os.unlink(temp_file)


def test_rca_data_builder():
    """Test RCADataBuilder functionality."""
    from shapleyiq.preprocessing import RCADataBuilder

    # Create sample data files
    trace_data = [
        {
            "trace_id": "trace-001",
            "spans": [
                {
                    "span_id": "span-1",
                    "service_name": "frontend",
                    "operation_name": "GET /api",
                    "start_time": 1000000,
                    "duration": 50000,
                    "parent_span_id": None,
                },
                {
                    "span_id": "span-2",
                    "service_name": "backend",
                    "operation_name": "process",
                    "start_time": 1005000,
                    "duration": 30000,
                    "parent_span_id": "span-1",
                },
            ],
        }
    ]

    metric_data = {
        "frontend": {"avg_response_time": 50.0, "error_rate": 0.01, "cpu_usage": 0.6},
        "backend": {"avg_response_time": 30.0, "error_rate": 0.02, "cpu_usage": 0.8},
    }

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(trace_data, f)
        trace_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(metric_data, f)
        metric_file = f.name

    try:
        builder = RCADataBuilder()
        rca_data = builder.build_from_files(
            trace_file=trace_file,
            root_causes=["frontend", "backend"],
            trace_format="generic",
        )

        assert rca_data is not None
        assert len(rca_data.traces or []) == 1
        assert len(rca_data.nodes) == 2
        assert len(rca_data.edges) >= 1  # Should have at least one edge from call graph

        # Check service names
        service_names = [node.service_name for node in rca_data.nodes]
        assert "frontend" in service_names
        assert "backend" in service_names

    finally:
        os.unlink(trace_file)
        os.unlink(metric_file)


def test_call_graph_extraction():
    """Test call graph extraction from traces."""
    from shapleyiq.preprocessing import TracePreprocessor

    sample_traces = [
        {
            "trace_id": "trace-001",
            "spans": [
                {
                    "spanId": "span-1",
                    "service_name": "frontend",
                    "operation_name": "GET /api",
                    "start_time": 1000000,
                    "duration": 50000,
                    "parentSpanId": None,
                },
                {
                    "spanId": "span-2",
                    "service_name": "backend",
                    "operation_name": "process",
                    "start_time": 1005000,
                    "duration": 30000,
                    "parentSpanId": "span-1",
                },
                {
                    "spanId": "span-3",
                    "service_name": "database",
                    "operation_name": "query",
                    "start_time": 1010000,
                    "duration": 20000,
                    "parentSpanId": "span-2",
                },
            ],
        }
    ]

    preprocessor = TracePreprocessor()
    processed_traces = preprocessor.process_traces(sample_traces)
    edges, nodes = preprocessor.extract_call_graph(processed_traces)

    assert len(edges) >= 2  # Should have at least 2 edges

    # Check for expected edges (using service:operation format)
    edge_pairs = [(edge.source_id, edge.target_id) for edge in edges]
    assert ("frontend:GET /api", "backend:process") in edge_pairs
    assert ("backend:process", "database:query") in edge_pairs
