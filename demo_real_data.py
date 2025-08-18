#!/usr/bin/env python3
"""
ShapleyIQ Real Data Demo

This script demonstrates how to use the ShapleyIQ package with real trace data
from the TrainTicket microservice system.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

from rcabench_platform.v2.logging import logger

from shapleyiq import TON, MicroHECL, MicroRank, MicroRCA, ShapleyValueRCA
from shapleyiq.data_structures import RCAData
from shapleyiq.preprocessing import RCADataBuilder


class ShapleyIQDemo:
    """Demo class for analyzing real microservice trace data."""

    def __init__(self, data_dir: str = "ShapleyIQ/rca4tracing/fault_injection/data"):
        """
        Initialize demo with data directory.

        Args:
            data_dir: Path to the fault injection data directory
        """
        self.data_dir = Path(data_dir)
        self.trace_dir = self.data_dir  # 数据文件直接在data目录下
        self.builder = RCADataBuilder()

        # Initialize algorithms
        self.algorithms = {
            "ShapleyValueRCA": ShapleyValueRCA(),
            "MicroHECL": MicroHECL(),
            "MicroRCA": MicroRCA(),
            "MicroRank": MicroRank(),
            "TON": TON(),
        }

        logger.info(f"Initialized ShapleyIQ Demo with data directory: {self.data_dir}")

    def list_available_datasets(self) -> List[str]:
        """List all available trace datasets."""
        if not self.trace_dir.exists():
            logger.warning(f"Trace directory not found: {self.trace_dir}")
            return []

        # Find all JSON files with fault injection patterns
        datasets = []
        for file_path in self.trace_dir.glob("*.json"):
            if any(
                service in file_path.name
                for service in [
                    "ts-basic-service",
                    "ts-order-service",
                    "ts-travel-service",
                    "ts-user-service",
                    "ts-food-service",
                    "ts-station-service",
                ]
            ):
                datasets.append(file_path.name)

        logger.info(f"Found {len(datasets)} datasets")
        return sorted(datasets)

    def parse_dataset_info(self, dataset_name: str) -> Dict[str, str]:
        """
        Parse information from dataset filename.

        Args:
            dataset_name: Name of the dataset file

        Returns:
            Dictionary with parsed information
        """
        # Example: ts-basic-service100_users5_spawn_rate5.json
        parts = dataset_name.replace(".json", "").split("_")

        info = {}

        # Extract service and fault information
        service_part = parts[0]
        if any(char.isdigit() for char in service_part):
            # Find where digits start
            for i, char in enumerate(service_part):
                if char.isdigit():
                    info["service"] = service_part[:i]
                    info["fault_delay_ms"] = service_part[i:]
                    break
        else:
            info["service"] = service_part
            info["fault_delay_ms"] = "0"

        # Extract other parameters
        for part in parts[1:]:
            if part.startswith("users"):
                info["users"] = part.replace("users", "")
            elif part.startswith("spawn"):
                info["spawn_rate"] = part.replace("spawn_rate", "")

        return info

    def load_trace_data(self, dataset_name: str) -> RCAData:
        """
        Load and preprocess trace data.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            Preprocessed RCAData object
        """
        dataset_path = self.trace_dir / dataset_name

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        logger.info(f"Loading dataset: {dataset_name}")

        # Parse dataset information
        dataset_info = self.parse_dataset_info(dataset_name)
        fault_service = dataset_info.get("service", "unknown")

        # Build RCA data from the trace file
        rca_data = self.builder.build_from_files(
            trace_file=str(dataset_path),
            root_causes=[fault_service],  # Use the faulty service as ground truth
            trace_format="jaeger",
        )

        logger.info(
            f"Loaded {len(rca_data.traces or [])} traces with {len(rca_data.nodes)} services"
        )
        logger.info(f"Ground truth root cause: {fault_service}")

        return rca_data

    def run_analysis(
        self, rca_data: RCAData, algorithms: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Run RCA analysis with specified algorithms.

        Args:
            rca_data: Preprocessed RCA data
            algorithms: List of algorithm names to run (default: all)

        Returns:
            Dictionary mapping algorithm names to their results
        """
        if algorithms is None:
            algorithms = list(self.algorithms.keys())

        results = {}

        for algo_name in algorithms:
            if algo_name not in self.algorithms:
                logger.warning(f"Algorithm {algo_name} not found")
                continue

            logger.info(f"Running {algo_name}...")
            start_time = time.time()

            try:
                algorithm = self.algorithms[algo_name]
                result = algorithm.analyze(rca_data)
                execution_time = time.time() - start_time

                results[algo_name] = {
                    "scores": result,
                    "execution_time": execution_time,
                    "top_services": list(result.keys())[:5] if result else [],
                }

                logger.info(f"{algo_name} completed in {execution_time:.2f}s")
                logger.info(f"Top 3 suspected services: {list(result.keys())[:3]}")

            except Exception as e:
                logger.error(f"Error running {algo_name}: {e}")
                results[algo_name] = {"error": str(e)}

        return results

    def evaluate_results(
        self, results: Dict[str, Dict], ground_truth: List[str], top_k: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate algorithm results against ground truth.

        Args:
            results: Algorithm results
            ground_truth: List of true root cause services
            top_k: Number of top predictions to consider

        Returns:
            Evaluation metrics for each algorithm
        """
        evaluation = {}

        for algo_name, result in results.items():
            if "error" in result:
                evaluation[algo_name] = {"error": result["error"]}
                continue

            scores = result.get("scores", {})
            if not scores:
                evaluation[algo_name] = {"precision": 0, "recall": 0, "f1": 0}
                continue

            # Get top-k predictions
            top_predictions = list(scores.keys())[:top_k]

            # Calculate metrics
            true_positives = len(set(top_predictions) & set(ground_truth))
            precision = true_positives / len(top_predictions) if top_predictions else 0
            recall = true_positives / len(ground_truth) if ground_truth else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            evaluation[algo_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "top_predictions": top_predictions,
                "execution_time": result.get("execution_time", 0),
            }

        return evaluation

    def run_demo(
        self, dataset_name: Optional[str] = None, algorithms: Optional[List[str]] = None
    ):
        """
        Run complete demo analysis.

        Args:
            dataset_name: Specific dataset to analyze (default: first available)
            algorithms: List of algorithms to run (default: all)
        """
        logger.info("🚀 Starting ShapleyIQ Real Data Demo")

        # List available datasets
        datasets = self.list_available_datasets()
        if not datasets:
            logger.error("No datasets found! Please check your data directory.")
            return

        if dataset_name is None:
            dataset_name = datasets[0]
            logger.info(f"Using dataset: {dataset_name}")
        elif dataset_name not in datasets:
            logger.error(
                f"Dataset {dataset_name} not found. Available: {datasets[:5]}..."
            )
            return

        # Parse dataset information
        dataset_info = self.parse_dataset_info(dataset_name)
        logger.info(f"📊 Dataset Info: {dataset_info}")

        # Load and preprocess data
        try:
            rca_data = self.load_trace_data(dataset_name)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return

        # Run analysis
        logger.info("🔍 Running RCA Analysis...")
        results = self.run_analysis(rca_data, algorithms)

        # Evaluate results
        ground_truth = [dataset_info.get("service", "unknown")]
        evaluation = self.evaluate_results(results, ground_truth)

        # Display results
        self.display_results(results, evaluation, dataset_info)

    def display_results(self, results: Dict, evaluation: Dict, dataset_info: Dict):
        """Display analysis results in a formatted way."""
        logger.info("\n" + "=" * 80)
        logger.info("📈 SHAPLEYIQ ANALYSIS RESULTS")
        logger.info("=" * 80)

        logger.info(f"Dataset: {dataset_info}")
        logger.info(
            f"Ground Truth Root Cause: {dataset_info.get('service', 'unknown')}"
        )
        logger.info("")

        for algo_name in results.keys():
            logger.info(f"🤖 {algo_name}:")

            if "error" in results[algo_name]:
                logger.info(f"   ❌ Error: {results[algo_name]['error']}")
                continue

            eval_data = evaluation[algo_name]

            logger.info(
                f"   ⏱️  Execution Time: {eval_data.get('execution_time', 0):.2f}s"
            )
            logger.info(f"   🎯 Precision@3: {eval_data.get('precision', 0):.3f}")
            logger.info(f"   📊 Recall@3: {eval_data.get('recall', 0):.3f}")
            logger.info(f"   🔢 F1@3: {eval_data.get('f1', 0):.3f}")
            logger.info(
                f"   🏆 Top Predictions: {eval_data.get('top_predictions', [])[:3]}"
            )
            logger.info("")

        logger.info("=" * 80)


def main():
    """Main demo function."""
    demo = ShapleyIQDemo()

    # List available datasets
    datasets = demo.list_available_datasets()

    if datasets:
        logger.info(f"Available datasets: {len(datasets)}")
        for i, dataset in enumerate(datasets[:10]):  # Show first 10
            info = demo.parse_dataset_info(dataset)
            logger.info(
                f"  {i + 1}. {dataset} -> Service: {info.get('service')}, Delay: {info.get('fault_delay_ms')}ms"
            )

        # Run demo with first dataset
        logger.info("\n🎯 Running demo with first dataset...")
        demo.run_demo(datasets[0])

    else:
        logger.error(
            "No datasets found. Please ensure your data is in the correct directory."
        )
        logger.info("Expected structure:")
        logger.info("  ShapleyIQ/rca4tracing/fault_injection/data/*.json")


if __name__ == "__main__":
    main()
