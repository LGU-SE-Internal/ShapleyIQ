"""
ShapleyIQ Platform CLI
新平台的命令行接口
"""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from .algorithms import TON, MicroHECL, MicroRank, MicroRCA, ShapleyRCA
from .data_loader import load_logs, load_metrics, load_traces
from .interface import AlgorithmArgs

app = typer.Typer(name="shapleyiq", help="ShapleyIQ RCA Platform")

# Available algorithms
ALGORITHMS = {
    "shapley": ShapleyRCA,
    "microhecl": MicroHECL,
    "microrca": MicroRCA,
    "microrank": MicroRank,
    "ton": TON,
}


@app.command()
def run(
    data_path: Annotated[Path, typer.Argument(help="Path to the data directory")],
    algorithm: Annotated[str, typer.Option(help="Algorithm to use")] = "shapley",
    env_file: Annotated[
        Optional[Path], typer.Option(help="Environment configuration file")
    ] = None,
    output: Annotated[Optional[Path], typer.Option(help="Output file path")] = None,
    verbose: Annotated[bool, typer.Option(help="Enable verbose output")] = False,
):
    """
    Run RCA analysis using the specified algorithm.
    """
    if algorithm not in ALGORITHMS:
        typer.echo(
            f"Error: Unknown algorithm '{algorithm}'. Available: {list(ALGORITHMS.keys())}"
        )
        raise typer.Exit(1)

    # Load environment configuration
    env_config = {}
    if env_file:
        if env_file.exists():
            with open(env_file, "r") as f:
                env_config = json.load(f)
                if verbose:
                    typer.echo(f"Loaded environment config from {env_file}")
        else:
            typer.echo(f"Warning: Environment file {env_file} not found")
    else:
        # Try to find env.json in data directory
        env_path = data_path / "env.json"
        if env_path.exists():
            with open(env_path, "r") as f:
                env_config = json.load(f)
                if verbose:
                    typer.echo(f"Loaded environment config from {env_path}")

    if verbose:
        typer.echo(f"Running {algorithm} algorithm on data at {data_path}")

    try:
        # Load data
        if verbose:
            typer.echo("Loading traces...")

        # Define time ranges from env config
        normal_start = env_config.get("NORMAL_START")
        normal_end = env_config.get("NORMAL_END")
        abnormal_start = env_config.get("ABNORMAL_START")
        abnormal_end = env_config.get("ABNORMAL_END")

        if not all([normal_start, normal_end, abnormal_start, abnormal_end]):
            typer.echo(
                "Warning: Time ranges not found in environment config, using default files"
            )
            # Try to load from normal/abnormal trace files
            normal_traces_file = data_path / "normal_traces.parquet"
            abnormal_traces_file = data_path / "abnormal_traces.parquet"

            if normal_traces_file.exists() and abnormal_traces_file.exists():
                normal_traces = load_traces(normal_traces_file)
                abnormal_traces = load_traces(abnormal_traces_file)
            else:
                typer.echo("Error: Could not find trace files or time ranges")
                raise typer.Exit(1)
        else:
            # Load traces using time ranges
            traces_file = data_path / "traces.parquet"
            if not traces_file.exists():
                typer.echo(f"Error: Traces file {traces_file} not found")
                raise typer.Exit(1)

            normal_traces = load_traces(
                traces_file, start_time=normal_start, end_time=normal_end
            )
            abnormal_traces = load_traces(
                traces_file, start_time=abnormal_start, end_time=abnormal_end
            )

        if verbose:
            typer.echo(
                f"Loaded {len(normal_traces)} normal traces and {len(abnormal_traces)} abnormal traces"
            )

        # Load metrics if available
        metrics_file = data_path / "metrics.parquet"
        normal_metrics = None
        abnormal_metrics = None

        if metrics_file.exists():
            if verbose:
                typer.echo("Loading metrics...")
            try:
                if normal_start and normal_end and abnormal_start and abnormal_end:
                    normal_metrics = load_metrics(
                        metrics_file, start_time=normal_start, end_time=normal_end
                    )
                    abnormal_metrics = load_metrics(
                        metrics_file, start_time=abnormal_start, end_time=abnormal_end
                    )
                else:
                    # Try separate files
                    normal_metrics_file = data_path / "normal_metrics.parquet"
                    abnormal_metrics_file = data_path / "abnormal_metrics.parquet"
                    if normal_metrics_file.exists() and abnormal_metrics_file.exists():
                        normal_metrics = load_metrics(normal_metrics_file)
                        abnormal_metrics = load_metrics(abnormal_metrics_file)

                if verbose and normal_metrics is not None:
                    typer.echo("Loaded metrics data")
            except Exception as e:
                if verbose:
                    typer.echo(f"Warning: Could not load metrics: {e}")

        # Load logs if available
        logs_file = data_path / "logs.parquet"
        normal_logs = None
        abnormal_logs = None

        if logs_file.exists():
            if verbose:
                typer.echo("Loading logs...")
            try:
                if normal_start and normal_end and abnormal_start and abnormal_end:
                    normal_logs = load_logs(
                        logs_file, start_time=normal_start, end_time=normal_end
                    )
                    abnormal_logs = load_logs(
                        logs_file, start_time=abnormal_start, end_time=abnormal_end
                    )
                else:
                    # Try separate files
                    normal_logs_file = data_path / "normal_logs.parquet"
                    abnormal_logs_file = data_path / "abnormal_logs.parquet"
                    if normal_logs_file.exists() and abnormal_logs_file.exists():
                        normal_logs = load_logs(normal_logs_file)
                        abnormal_logs = load_logs(abnormal_logs_file)

                if verbose and normal_logs is not None:
                    typer.echo("Loaded logs data")
            except Exception as e:
                if verbose:
                    typer.echo(f"Warning: Could not load logs: {e}")

        # Create algorithm instance
        algorithm_class = ALGORITHMS[algorithm]
        alg_instance = algorithm_class()

        # Prepare algorithm arguments
        args = AlgorithmArgs(
            normal_traces=normal_traces,
            abnormal_traces=abnormal_traces,
            normal_metrics=normal_metrics,
            abnormal_metrics=abnormal_metrics,
            normal_logs=normal_logs,
            abnormal_logs=abnormal_logs,
        )

        if verbose:
            typer.echo(f"Running {algorithm} algorithm...")

        # Run algorithm
        results = alg_instance(args)

        # Process results
        if verbose:
            typer.echo(f"Algorithm completed with {len(results)} results")

        # Output results
        output_data = []
        for i, result in enumerate(results):
            output_data.append(
                {
                    "rank": i + 1,
                    "node_name": result.node_name,
                    "probability": result.probability,
                    "score": getattr(result, "score", result.probability),
                }
            )

        if output:
            # Save to file
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)
            typer.echo(f"Results saved to {output}")
        else:
            # Print to console
            typer.echo("\nRoot Cause Analysis Results:")
            typer.echo("-" * 50)
            for item in output_data[:10]:  # Show top 10
                typer.echo(
                    f"{item['rank']:2d}. {item['node_name']:<30} {item['probability']:.4f}"
                )

    except Exception as e:
        typer.echo(f"Error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def list_algorithms():
    """List available algorithms."""
    typer.echo("Available algorithms:")
    for name, alg_class in ALGORITHMS.items():
        typer.echo(f"  {name:<12} - {alg_class.__doc__ or 'No description'}")


@app.command()
def validate_data(
    data_path: Annotated[Path, typer.Argument(help="Path to the data directory")],
    verbose: Annotated[bool, typer.Option(help="Enable verbose output")] = False,
):
    """Validate data format and structure."""
    typer.echo(f"Validating data at {data_path}")

    # Check required files
    required_files = ["traces.parquet", "env.json"]
    optional_files = [
        "metrics.parquet",
        "logs.parquet",
        "normal_traces.parquet",
        "abnormal_traces.parquet",
    ]

    missing_required = []
    for file in required_files:
        file_path = data_path / file
        if file_path.exists():
            typer.echo(f"✓ Found {file}")
        else:
            missing_required.append(file)
            typer.echo(f"✗ Missing {file}")

    available_optional = []
    for file in optional_files:
        file_path = data_path / file
        if file_path.exists():
            available_optional.append(file)
            typer.echo(f"✓ Found {file}")
        else:
            if verbose:
                typer.echo(f"  {file} (optional)")

    if missing_required:
        typer.echo(f"\nError: Missing required files: {missing_required}")
        raise typer.Exit(1)

    # Validate env.json
    env_file = data_path / "env.json"
    try:
        with open(env_file, "r") as f:
            env_config = json.load(f)

        required_keys = ["NORMAL_START", "NORMAL_END", "ABNORMAL_START", "ABNORMAL_END"]
        missing_keys = [key for key in required_keys if key not in env_config]

        if missing_keys:
            typer.echo(f"Warning: Missing time range keys in env.json: {missing_keys}")
        else:
            typer.echo("✓ Environment configuration is valid")

    except Exception as e:
        typer.echo(f"Error reading env.json: {e}")
        raise typer.Exit(1)

    typer.echo("\nData validation completed successfully!")


if __name__ == "__main__":
    app()
