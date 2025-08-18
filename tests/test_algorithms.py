"""
Tests for ShapleyIQ algorithms.
"""


def test_shapley_value_rca_basic(sample_rca_data):
    """Test basic ShapleyValueRCA functionality."""
    from shapleyiq.algorithms import ShapleyValueRCA

    algorithm = ShapleyValueRCA()
    results = algorithm.analyze(sample_rca_data)

    # Check that results are returned
    assert results is not None
    assert hasattr(results, "get_top_root_causes")

    # Check that we get some root causes
    top_causes = results.get_top_root_causes(top_k=3)
    assert len(top_causes) > 0

    # Check that scores are reasonable
    for service, score in top_causes:
        assert isinstance(service, str)
        assert isinstance(score, (int, float))
        assert score >= 0


def test_microhecl_basic(sample_rca_data):
    """Test basic MicroHECL functionality."""
    from shapleyiq.algorithms import MicroHECL

    algorithm = MicroHECL()
    results = algorithm.analyze(sample_rca_data)

    assert results is not None
    top_causes = results.get_top_root_causes(top_k=2)
    assert len(top_causes) > 0


def test_microrcan_basic(sample_rca_data):
    """Test basic MicroRCA functionality."""
    from shapleyiq.algorithms import MicroRCA

    algorithm = MicroRCA()
    results = algorithm.analyze(sample_rca_data)

    assert results is not None
    top_causes = results.get_top_root_causes(top_k=2)
    assert len(top_causes) > 0


def test_microrank_basic(sample_rca_data):
    """Test basic MicroRank functionality."""
    from shapleyiq.algorithms import MicroRank

    algorithm = MicroRank()
    results = algorithm.analyze(sample_rca_data)

    assert results is not None
    top_causes = results.get_top_root_causes(top_k=2)
    assert len(top_causes) > 0


def test_ton_basic(sample_rca_data):
    """Test basic TON functionality."""
    from shapleyiq.algorithms import TON

    algorithm = TON()
    results = algorithm.analyze(sample_rca_data)

    assert results is not None
    top_causes = results.get_top_root_causes(top_k=2)
    assert len(top_causes) > 0


def test_algorithm_comparison(sample_rca_data):
    """Test that all algorithms can be compared."""
    from shapleyiq.algorithms import (
        TON,
        MicroHECL,
        MicroRank,
        MicroRCA,
        ShapleyValueRCA,
    )

    algorithms = {
        "ShapleyIQ": ShapleyValueRCA(),
        "MicroHECL": MicroHECL(),
        "MicroRCA": MicroRCA(),
        "MicroRank": MicroRank(),
        "TON": TON(),
    }

    results = {}
    for name, algorithm in algorithms.items():
        result = algorithm.analyze(sample_rca_data)
        results[name] = result.get_top_root_causes(top_k=1)

        # Verify each algorithm returns valid results
        assert len(results[name]) > 0
        service, score = results[name][0]
        assert isinstance(service, str)
        assert isinstance(score, (int, float))
        assert score >= 0
