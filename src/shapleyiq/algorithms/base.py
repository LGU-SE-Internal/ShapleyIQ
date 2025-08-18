"""
Base class for RCA algorithms.

This module defines the common interface that all RCA algorithms should implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from rcabench_platform.v2.logging import logger

from ..data_structures import RCAData


class BaseRCAAlgorithm(ABC):
    """
    Abstract base class for all RCA algorithms.

    This class defines the common interface and provides some shared functionality
    for all root cause analysis algorithms.
    """

    def __init__(self, name: str = "BaseRCA"):
        """
        Initialize the algorithm.

        Args:
            name: Name of the algorithm
        """
        self.name = name
        self.logger = logger

    @abstractmethod
    def analyze(self, data: RCAData, **kwargs) -> Dict[str, float]:
        """
        Perform root cause analysis on the given data.

        Args:
            data: Processed RCA data
            **kwargs: Algorithm-specific parameters

        Returns:
            Dictionary mapping node IDs to root cause scores
        """
        pass

    def validate_data(self, data: RCAData) -> bool:
        """
        Validate that the input data is suitable for this algorithm.

        Args:
            data: RCA data to validate

        Returns:
            True if data is valid, False otherwise
        """
        if data.is_empty:
            self.logger.warning(f"{self.name}: Input data is empty")
            return False

        if not data.edges:
            self.logger.warning(f"{self.name}: No edges found in data")
            return False

        if not data.nodes:
            self.logger.warning(f"{self.name}: No nodes found in data")
            return False

        return data.validate()

    def preprocess_data(self, data: RCAData) -> RCAData:
        """
        Preprocess data before analysis (can be overridden by subclasses).

        Args:
            data: Input RCA data

        Returns:
            Preprocessed RCA data
        """
        # Default implementation returns data as-is
        return data

    def postprocess_results(
        self, results: Dict[str, float], data: RCAData
    ) -> Dict[str, float]:
        """
        Postprocess results after analysis (can be overridden by subclasses).

        Args:
            results: Raw algorithm results
            data: Original RCA data

        Returns:
            Postprocessed results
        """
        # Default implementation filters out invalid nodes and sorts results
        valid_results = {}
        valid_node_ids = set(data.node_ids)

        for node_id, score in results.items():
            if node_id in valid_node_ids:
                valid_results[node_id] = score
            else:
                self.logger.debug(f"{self.name}: Filtering out invalid node {node_id}")

        # Sort by score (descending)
        sorted_results = dict(
            sorted(valid_results.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_results

    def run(self, data: RCAData, **kwargs) -> Dict[str, float]:
        """
        Main entry point to run the algorithm.

        Args:
            data: RCA data
            **kwargs: Algorithm-specific parameters

        Returns:
            Dictionary mapping node IDs to root cause scores
        """
        self.logger.info(f"Running {self.name} algorithm")

        # Validate input data
        if not self.validate_data(data):
            self.logger.error(f"{self.name}: Data validation failed")
            return {}

        try:
            # Preprocess data
            processed_data = self.preprocess_data(data)

            # Run algorithm
            results = self.analyze(processed_data, **kwargs)

            # Postprocess results
            final_results = self.postprocess_results(results, data)

            self.logger.info(
                f"{self.name}: Analysis completed, found {len(final_results)} results"
            )
            return final_results

        except Exception as e:
            self.logger.error(f"{self.name}: Algorithm execution failed: {e}")
            return {}

    def get_top_k_results(self, results: Dict[str, float], k: int = 5) -> List[tuple]:
        """
        Get top k results from the analysis.

        Args:
            results: Algorithm results
            k: Number of top results to return

        Returns:
            List of (node_id, score) tuples
        """
        sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]
