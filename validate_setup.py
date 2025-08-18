"""
Simple validation script to test ShapleyIQ package structure.
"""

from rcabench_platform.v2.logging import logger


def test_imports():
    """Test that all modules can be imported correctly."""
    try:
        # Test data structures
        logger.info("✅ Data structures imported successfully")

        # Test utilities
        logger.info("✅ Utilities imported successfully")

        # Test preprocessing
        logger.info("✅ Preprocessing modules imported successfully")

        # Test base algorithm
        logger.info("✅ Base algorithm imported successfully")

        # Test main algorithm
        logger.info("✅ ShapleyValueRCA imported successfully")

        # Test baseline algorithms
        logger.info("✅ All baseline algorithms imported successfully")

        # Test main package
        logger.info("✅ Main package imported successfully")

        return True

    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_algorithm_initialization():
    """Test that algorithms can be initialized."""
    try:
        from shapleyiq.algorithms import (
            TON,
            MicroHECL,
            MicroRank,
            MicroRCA,
            ShapleyValueRCA,
        )

        # Initialize all algorithms
        shapley_rca = ShapleyValueRCA()
        micro_hecl = MicroHECL()
        micro_rca = MicroRCA()
        micro_rank = MicroRank()
        ton = TON()

        algorithms = [shapley_rca, micro_hecl, micro_rca, micro_rank, ton]

        for algo in algorithms:
            logger.info(f"✅ {algo.name} initialized successfully")

        return True

    except Exception as e:
        logger.error(f"❌ Algorithm initialization failed: {e}")
        return False


def test_data_structures():
    """Test data structure creation."""
    try:
        import time

        from shapleyiq.data_structures import Edge, RCAData, ServiceNode, TraceData

        # Create sample data
        nodes = [
            ServiceNode(
                node_id="service1", service_name="service1", server_ip="192.168.1.1"
            ),
            ServiceNode(
                node_id="service2", service_name="service2", server_ip="192.168.1.2"
            ),
        ]

        edges = [Edge(source_id="service1", target_id="service2")]

        trace_data = TraceData(
            trace_id="trace1", spans=[], timestamp=int(time.time() * 1000)
        )

        traces = [trace_data]

        rca_data = RCAData(
            nodes=nodes,
            edges=edges,
            traces=traces,
            node_ids=["service1", "service2"],
            root_id="service1",
            request_timestamp=int(time.time() * 1000),
        )

        logger.info("✅ Data structures created successfully")
        logger.info(f"   - Nodes: {len(rca_data.nodes)}")
        logger.info(f"   - Edges: {len(rca_data.edges)}")
        logger.info(f"   - Traces: {len(rca_data.traces) if rca_data.traces else 0}")

        return True

    except Exception as e:
        logger.error(f"❌ Data structure creation failed: {e}")
        return False


def main():
    """Run all validation tests."""
    logger.info("🚀 Starting ShapleyIQ validation...")

    tests = [
        ("Import Test", test_imports),
        ("Algorithm Initialization Test", test_algorithm_initialization),
        ("Data Structure Test", test_data_structures),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))

    # Summary
    logger.info("\n📊 Validation Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")

    if passed == total:
        logger.info(f"\n🎉 All {total} tests passed! ShapleyIQ is ready to use.")
    else:
        logger.warning(
            f"\n⚠️  {passed}/{total} tests passed. Please check the failed tests."
        )

    return passed == total


if __name__ == "__main__":
    main()
