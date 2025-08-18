"""
Utility functions for RCA algorithms.

This module contains common utility functions used across different algorithms.
"""

from typing import Dict, List, Optional

import numpy as np
from rcabench_platform.v2.logging import logger


def pearson_correlation(
    a: List[float], b: List[float], default_value: float = 0.0
) -> float:
    """
    Calculate Pearson correlation coefficient between two time series.

    Args:
        a: First time series
        b: Second time series
        default_value: Value to return if correlation cannot be calculated

    Returns:
        Pearson correlation coefficient or default_value
    """
    try:
        if not a or not b:
            return default_value

        # Handle different lengths by truncating to minimum
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a = a[-min_len:]
            b = b[-min_len:]

        if len(a) < 2 or not np.var(a) or not np.var(b):
            return default_value

        correlation_matrix = np.corrcoef(a, b)
        correlation = abs(correlation_matrix[0, 1])

        return correlation if not np.isnan(correlation) else default_value

    except Exception as e:
        logger.warning(f"Failed to calculate Pearson correlation: {e}")
        return default_value


def contribution_to_probability(
    contribution_dict: Dict[str, float],
) -> Dict[str, float]:
    """
    Convert contribution scores to probabilities by normalization.

    Args:
        contribution_dict: Dictionary mapping node IDs to contribution scores

    Returns:
        Dictionary mapping node IDs to probability scores
    """
    # Set negative contributions to zero
    adjusted_dict = {k: max(0, v) for k, v in contribution_dict.items()}

    total_contribution = sum(adjusted_dict.values())
    if total_contribution == 0:
        logger.warning("Total contribution is zero, returning uniform distribution")
        n = len(adjusted_dict)
        return {k: 1.0 / n for k in adjusted_dict.keys()}

    probability_dict = {k: v / total_contribution for k, v in adjusted_dict.items()}
    return probability_dict


def calculate_md5_hash(text: str) -> str:
    """
    Calculate MD5 hash of a string.

    Args:
        text: Input string

    Returns:
        MD5 hash as hexadecimal string
    """
    import hashlib

    return hashlib.md5(text.encode("utf-8")).hexdigest()


def normalize_time_series(data: List[float], method: str = "zscore") -> List[float]:
    """
    Normalize time series data.

    Args:
        data: Time series data
        method: Normalization method ('zscore', 'minmax')

    Returns:
        Normalized time series
    """
    if not data:
        return data

    data_array = np.array(data)

    if method == "zscore":
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        if std_val == 0:
            return [0.0] * len(data)
        return ((data_array - mean_val) / std_val).tolist()

    elif method == "minmax":
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        if max_val == min_val:
            return [0.0] * len(data)
        return ((data_array - min_val) / (max_val - min_val)).tolist()

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def detect_anomalies_statistical(
    data: List[float],
    threshold_multiplier: float = 3.0,
    baseline_mean: Optional[float] = None,
    baseline_std: Optional[float] = None,
) -> List[bool]:
    """
    Detect anomalies using statistical thresholds.

    Args:
        data: Time series data
        threshold_multiplier: Number of standard deviations for threshold
        baseline_mean: Pre-computed baseline mean (if None, computed from data)
        baseline_std: Pre-computed baseline std (if None, computed from data)

    Returns:
        List of boolean flags indicating anomalies
    """
    if not data:
        return []

    data_array = np.array(data)

    if baseline_mean is None:
        baseline_mean = np.mean(data_array)
    if baseline_std is None:
        baseline_std = np.std(data_array)

    if baseline_std == 0:
        return [False] * len(data)

    lower_threshold = baseline_mean - threshold_multiplier * baseline_std
    upper_threshold = baseline_mean + threshold_multiplier * baseline_std

    anomalies = [(x < lower_threshold or x > upper_threshold) for x in data]
    return anomalies


def build_adjacency_matrix(edges: List[tuple], node_ids: List[str]) -> np.ndarray:
    """
    Build adjacency matrix from edge list.

    Args:
        edges: List of (source, target) tuples
        node_ids: List of node identifiers

    Returns:
        Adjacency matrix as numpy array
    """
    n_nodes = len(node_ids)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    adjacency_matrix = np.zeros((n_nodes, n_nodes))

    for source, target in edges:
        if source in node_to_idx and target in node_to_idx:
            src_idx = node_to_idx[source]
            tgt_idx = node_to_idx[target]
            adjacency_matrix[src_idx, tgt_idx] = 1

    return adjacency_matrix


def rank_results(scores: Dict[str, float], ascending: bool = False) -> List[tuple]:
    """
    Rank results by score.

    Args:
        scores: Dictionary mapping items to scores
        ascending: If True, rank in ascending order

    Returns:
        List of (item, score) tuples sorted by score
    """
    return sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)


def evaluate_top_k_accuracy(
    predicted_ranking: List[str], true_root_causes: List[str], k_values: List[int]
) -> Dict[int, float]:
    """
    Evaluate top-k accuracy for different k values.

    Args:
        predicted_ranking: List of node IDs in predicted order
        true_root_causes: List of true root cause node IDs
        k_values: List of k values to evaluate

    Returns:
        Dictionary mapping k to accuracy score
    """
    results = {}
    true_set = set(true_root_causes)

    for k in k_values:
        if k <= len(predicted_ranking):
            predicted_top_k = set(predicted_ranking[:k])
            accuracy = len(predicted_top_k.intersection(true_set)) / len(true_set)
            results[k] = accuracy
        else:
            results[k] = 0.0

    return results


def probability_conversion(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Convert scores to probabilities using softmax normalization.

    Args:
        scores: Dictionary mapping node IDs to scores

    Returns:
        Dictionary mapping node IDs to probabilities
    """
    if not scores:
        return {}

    # Get score values
    score_values = list(scores.values())

    # Apply softmax to prevent overflow
    max_score = max(score_values)
    exp_scores = {
        node_id: np.exp(score - max_score) for node_id, score in scores.items()
    }

    # Normalize to get probabilities
    total_exp = sum(exp_scores.values())

    if total_exp == 0:
        # Uniform distribution if all scores are the same
        uniform_prob = 1.0 / len(scores)
        return {node_id: uniform_prob for node_id in scores.keys()}

    probabilities = {
        node_id: exp_score / total_exp for node_id, exp_score in exp_scores.items()
    }

    return probabilities
