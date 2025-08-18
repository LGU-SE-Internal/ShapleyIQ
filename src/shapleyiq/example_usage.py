"""
Example usage of the ShapleyIQ package.

This module demonstrates how to use the refactored ShapleyIQ algorithms
for root cause analysis in microservices.
"""

import json
from typing import Dict, List

from rcabench_platform.v2.logging import logger

from shapleyiq import (
    TON,
    Edge,
    MicroHECL,
    MicroRank,
    MicroRCA,
    RCAData,
    RCADataBuilder,
    ServiceNode,
    ShapleyValueRCA,
    TraceData,
)


def create_sample_data() -> RCAData:
    """
    Create sample RCA data for demonstration.

    Returns:
        Sample RCAData object
    """
    # Create sample service nodes
    nodes = [
        ServiceNode(
            node_id="frontend:getUser",
            service_name="frontend",
            server_ip="192.168.1.10",
        ),
        ServiceNode(
            node_id="auth:validateToken", service_name="auth", server_ip="192.168.1.11"
        ),
        ServiceNode(
            node_id="userdb:queryUser", service_name="userdb", server_ip="192.168.1.12"
        ),
        ServiceNode(
            node_id="cache:getUser", service_name="cache", server_ip="192.168.1.13"
        ),
    ]

    # Create sample edges (call relationships)
    edges = [
        Edge(source_id="frontend:getUser", target_id="auth:validateToken"),
        Edge(source_id="frontend:getUser", target_id="cache:getUser"),
        Edge(source_id="auth:validateToken", target_id="userdb:queryUser"),
        Edge(source_id="cache:getUser", target_id="userdb:queryUser"),
    ]

    # Create sample trace data
    sample_spans = [
        {
            "spanId": "span1",
            "operationName": "getUser",
            "serviceName": "frontend",
            "startTime": 1000000,
            "duration": 150000,
            "process": {"serviceName": "frontend"},
        },
        {
            "spanId": "span2",
            "parentSpanId": "span1",
            "operationName": "validateToken",
            "serviceName": "auth",
            "startTime": 1010000,
            "duration": 50000,
            "process": {"serviceName": "auth"},
        },
        {
            "spanId": "span3",
            "parentSpanId": "span2",
            "operationName": "queryUser",
            "serviceName": "userdb",
            "startTime": 1020000,
            "duration": 80000,
            "process": {"serviceName": "userdb"},
        },
    ]

    traces = [TraceData(trace_id="trace1", spans=sample_spans, timestamp=1000000)]

    # Create sample time series data
    ts_data_dict = {
        "frontend:getUser": {
            "MaxDuration": [100, 120, 150, 180, 200],
            "Duration": [100, 120, 150, 180, 200],
        },
        "auth:validateToken": {
            "MaxDuration": [30, 40, 50, 60, 70],
            "Duration": [30, 40, 50, 60, 70],
        },
        "userdb:queryUser": {
            "MaxDuration": [60, 70, 80, 90, 100],
            "Duration": [60, 70, 80, 90, 100],
        },
        "cache:getUser": {
            "MaxDuration": [10, 15, 20, 25, 30],
            "Duration": [10, 15, 20, 25, 30],
        },
    }

    # Create sample trace data dict for compatibility
    trace_data_dict = {
        "frontend:getUser": {"Duration": [150], "serverIp": ["192.168.1.10"]},
        "auth:validateToken": {"Duration": [50], "serverIp": ["192.168.1.11"]},
        "userdb:queryUser": {"Duration": [80], "serverIp": ["192.168.1.12"]},
        "cache:getUser": {"Duration": [20], "serverIp": ["192.168.1.13"]},
    }

    # Create RCAData object
    rca_data = RCAData(
        edges=edges,
        nodes=nodes,
        node_ids=[node.node_id for node in nodes],
        root_causes=["userdb:queryUser"],  # Assume database is the root cause
        root_id="frontend:getUser",
        traces=traces,
        trace_data_dict=trace_data_dict,
        request_timestamp=1000000,
        ts_data_dict=ts_data_dict,
        metrics_statistical_data={},
        metrics_threshold={},
    )

    return rca_data


def run_shapley_analysis(data: RCAData) -> Dict[str, float]:
    """
    Run ShapleyValueRCA analysis.

    Args:
        data: RCA data

    Returns:
        Analysis results
    """
    logger.info("Running ShapleyValueRCA analysis...")

    # Initialize algorithm
    shapley_rca = ShapleyValueRCA(using_cache=False, sync_overlap_threshold=0.05)

    # Run analysis
    results = shapley_rca.run(data, strategy="avg_by_contribution", sort_result=True)

    logger.info(f"ShapleyValueRCA results: {results}")
    return results


def run_baseline_algorithms(data: RCAData) -> Dict[str, Dict[str, float]]:
    """
    Run baseline algorithms for comparison.

    Args:
        data: RCA data

    Returns:
        Dictionary mapping algorithm names to results
    """
    results = {}

    # MicroHECL
    logger.info("Running MicroHECL analysis...")
    microhecl = MicroHECL(time_window=15)
    try:
        microhecl_results = microhecl.run(
            data, initial_anomalous_node="frontend:getUser", detect_metrics=["RT"]
        )
        results["MicroHECL"] = microhecl_results
        logger.info(f"MicroHECL results: {microhecl_results}")
    except Exception as e:
        logger.warning(f"MicroHECL failed: {e}")
        results["MicroHECL"] = {}

    # MicroRCA
    logger.info("Running MicroRCA analysis...")
    microrca = MicroRCA(time_window=15)
    try:
        microrca_results = microrca.run(data)
        results["MicroRCA"] = microrca_results
        logger.info(f"MicroRCA results: {microrca_results}")
    except Exception as e:
        logger.warning(f"MicroRCA failed: {e}")
        results["MicroRCA"] = {}

    # TON
    logger.info("Running TON analysis...")
    ton = TON(time_window=15)
    try:
        ton_results = ton.run(data, operation_only=True)
        results["TON"] = ton_results
        logger.info(f"TON results: {ton_results}")
    except Exception as e:
        logger.warning(f"TON failed: {e}")
        results["TON"] = {}

    # MicroRank (requires special data format)
    logger.info("Running MicroRank analysis...")
    microrank = MicroRank(n_sigma=3)
    try:
        # MicroRank needs trace_dict in special format
        enhanced_data = data
        if not enhanced_data.trace_dict and enhanced_data.traces:
            enhanced_data.trace_dict = {
                trace.trace_id: trace.spans for trace in enhanced_data.traces
            }

        microrank_results = microrank.run(enhanced_data, phi=0.5, omega=0.01, d=0.04)
        results["MicroRank"] = microrank_results
        logger.info(f"MicroRank results: {microrank_results}")
    except Exception as e:
        logger.warning(f"MicroRank failed: {e}")
        results["MicroRank"] = {}

    return results


def evaluate_results(
    shapley_results: Dict[str, float],
    baseline_results: Dict[str, Dict[str, float]],
    true_root_causes: List[str],
) -> None:
    """
    Evaluate and compare algorithm results.

    Args:
        shapley_results: ShapleyValueRCA results
        baseline_results: Baseline algorithm results
        true_root_causes: Known root causes for evaluation
    """
    logger.info("=== Algorithm Evaluation ===")

    all_results = {"ShapleyValueRCA": shapley_results}
    all_results.update(baseline_results)

    for algorithm_name, results in all_results.items():
        if not results:
            logger.info(f"{algorithm_name}: No results")
            continue

        # Get top-3 predictions
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_results[:3]

        # Calculate precision@k
        precision_at_1 = 1 if top_3 and top_3[0][0] in true_root_causes else 0
        precision_at_3 = sum(1 for node, _ in top_3 if node in true_root_causes) / min(
            3, len(top_3)
        )

        logger.info(f"{algorithm_name}:")
        logger.info(f"  Top-3 results: {top_3}")
        logger.info(f"  Precision@1: {precision_at_1:.2f}")
        logger.info(f"  Precision@3: {precision_at_3:.2f}")


def demonstrate_preprocessing() -> RCAData:
    """
    Demonstrate data preprocessing capabilities.

    Returns:
        Preprocessed RCAData
    """
    logger.info("=== Demonstrating Data Preprocessing ===")

    # Create sample trace file content
    sample_jaeger_data = {
        "data": [
            {
                "traceID": "trace123",
                "spans": [
                    {
                        "spanId": "span1",
                        "operationName": "getUser",
                        "startTime": 1000000,
                        "duration": 150000,
                        "process": {"serviceName": "frontend"},
                        "tags": [{"key": "ip", "value": "192.168.1.10"}],
                    },
                    {
                        "spanId": "span2",
                        "parentSpanId": "span1",
                        "operationName": "validateToken",
                        "startTime": 1010000,
                        "duration": 50000,
                        "process": {"serviceName": "auth"},
                        "tags": [{"key": "ip", "value": "192.168.1.11"}],
                    },
                ],
            }
        ]
    }

    # Save sample data to file
    with open("sample_traces.json", "w") as f:
        json.dump(sample_jaeger_data, f)

    # Use RCADataBuilder to build data from file
    builder = RCADataBuilder()
    rca_data = builder.build_from_files(
        trace_file="sample_traces.json",
        root_causes=["auth:validateToken"],
        trace_format="jaeger",
    )

    logger.info(
        f"Preprocessed data: {len(rca_data.edges)} edges, {len(rca_data.nodes)} nodes"
    )
    logger.info(f"Validation result: {rca_data.validate()}")

    return rca_data


def main():
    """
    Main demonstration function.
    """
    logger.info("=== ShapleyIQ Package Demonstration ===")

    # Method 1: Use sample data
    logger.info("\n1. Using sample data:")
    sample_data = create_sample_data()

    if sample_data.validate():
        logger.info("Sample data validation: PASSED")

        # Run ShapleyValueRCA
        shapley_results = run_shapley_analysis(sample_data)

        # Run baseline algorithms
        baseline_results = run_baseline_algorithms(sample_data)

        # Evaluate results
        evaluate_results(shapley_results, baseline_results, ["userdb:queryUser"])
    else:
        logger.error("Sample data validation: FAILED")

    # Method 2: Demonstrate preprocessing
    logger.info("\n2. Using preprocessing:")
    try:
        preprocessed_data = demonstrate_preprocessing()

        if preprocessed_data.validate():
            logger.info("Preprocessed data validation: PASSED")

            # Run analysis on preprocessed data
            shapley_results = run_shapley_analysis(preprocessed_data)
            logger.info(f"Preprocessed data analysis results: {shapley_results}")
        else:
            logger.error("Preprocessed data validation: FAILED")
    except Exception as e:
        logger.error(f"Preprocessing demonstration failed: {e}")

    # Clean up
    try:
        import os

        if os.path.exists("sample_traces.json"):
            os.remove("sample_traces.json")
    except Exception:
        pass

    logger.info("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
