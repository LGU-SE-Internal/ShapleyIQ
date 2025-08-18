# ShapleyIQ

Influence Quantification by Shapley Values for Performance Debugging of Microservices

## Overview

ShapleyIQ is a comprehensive root cause analysis (RCA) framework designed for microservice architectures. It uses Shapley values to quantify the influence of different service components on performance anomalies, providing precise and interpretable results for performance debugging.

## Features

- **ShapleyValue RCA**: Main algorithm using cooperative game theory to quantify service influence
- **Baseline Algorithms**: Implementation of MicroHECL, MicroRCA, MicroRank, and TON for comparison
- **Multi-format Support**: Compatible with Jaeger, Zipkin, and DbaAS trace formats
- **Comprehensive Preprocessing**: Built-in data preprocessing pipeline for trace normalization
- **Type Safety**: Full type annotations and data validation throughout the codebase
- **Modular Design**: Clean separation between preprocessing and algorithm execution

## Installation

### From Source

```bash
git clone <repository-url>
cd ShapleyIQ
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from shapleyiq import ShapleyValueRCA, RCAData, Edge, ServiceNode, TraceData
from shapleyiq.preprocessing import RCADataBuilder

# Load and preprocess data
builder = RCADataBuilder()
rca_data = builder.build_from_files(
    trace_files=["traces.json"],
    metric_files=["metrics.json"]
)

# Run ShapleyIQ analysis
algorithm = ShapleyValueRCA()
results = algorithm.analyze(rca_data)

# Print top root causes
for service, score in results.get_top_root_causes(top_k=5):
    print(f"{service}: {score:.4f}")
```

### Using Baseline Algorithms

```python
from shapleyiq.algorithms import MicroHECL, MicroRCA, MicroRank, TON

# Compare with baseline algorithms
algorithms = {
    "ShapleyIQ": ShapleyValueRCA(),
    "MicroHECL": MicroHECL(),
    "MicroRCA": MicroRCA(),
    "MicroRank": MicroRank(),
    "TON": TON()
}

for name, algorithm in algorithms.items():
    results = algorithm.analyze(rca_data)
    print(f"{name} top cause: {results.get_top_root_causes(1)[0]}")
```

### Command Line Demo

```bash
# Run the demonstration script
shapleyiq-demo
```

## Data Format

### Trace Data Structure

```python
@dataclass
class TraceData:
    trace_id: str
    spans: List[Dict[str, Any]]
    start_time: int
    end_time: int
    duration: int
    has_anomaly: bool
```

### Service Graph Structure

```python
@dataclass
class Edge:
    source: str
    target: str
    weight: float = 1.0
    call_count: int = 1
    avg_latency: float = 0.0

@dataclass  
class ServiceNode:
    service_name: str
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
```

## Algorithm Details

### ShapleyValue RCA

The main algorithm uses Shapley values from cooperative game theory to fairly attribute performance anomalies to different services:

1. **Timeline Analysis**: Analyzes service call timelines within traces
2. **Contribution Calculation**: Computes each service's contribution using Shapley values
3. **Influence Aggregation**: Aggregates influence scores across multiple traces

### Baseline Algorithms

- **MicroHECL**: Correlation-based anomaly propagation analysis
- **MicroRCA**: Graph-based root cause analysis with service dependencies
- **MicroRank**: PageRank-inspired algorithm for service importance ranking
- **TON**: Time-window based outlier detection and ranking

## Configuration

### Algorithm Parameters

```python
# ShapleyValue RCA configuration
algorithm = ShapleyValueRCA(
    contribution_threshold=0.01,  # Minimum contribution to consider
    max_iterations=1000,          # Maximum Shapley value iterations
    sample_size=100              # Sample size for approximation
)

# MicroHECL configuration
microhecl = MicroHECL(
    correlation_threshold=0.8,    # Correlation threshold for anomaly propagation
    propagation_steps=5          # Maximum propagation steps
)
```

## Performance

ShapleyIQ has been evaluated on various microservice architectures:

- **Accuracy**: Superior root cause identification compared to baseline methods
- **Scalability**: Handles service graphs with 100+ nodes efficiently  
- **Interpretability**: Provides quantitative influence scores for each service

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 src/
black src/
mypy src/
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ShapleyIQ in your research, please cite:

```bibtex
@article{shapleyiq2024,
  title={Influence Quantification by Shapley Values for Performance Debugging of Microservices},
  author={ShapleyIQ Team},
  journal={Conference/Journal Name},
  year={2024}
}
```

## Acknowledgments

- Original research from the rca4tracing project
- Shapley value implementation inspired by cooperative game theory literature
- Baseline algorithms adapted from various microservice RCA research papers
