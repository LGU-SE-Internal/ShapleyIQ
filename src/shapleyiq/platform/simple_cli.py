"""
ShapleyIQ Platform CLI - Simplified Version
新平台的命令行接口 - 简化版本
"""

import typer
from pathlib import Path
from typing import Optional, Annotated
import json

from .data_loader import load_traces, load_metrics, load_logs
from .algorithms import ShapleyRCA, MicroHECL, MicroRCA, MicroRank, TON
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
    output: Annotated[Optional[Path], typer.Option(help="Output file path")] = None,
    verbose: Annotated[bool, typer.Option(help="Enable verbose output")] = False,
):
    """Run RCA analysis using the specified algorithm."""
    if algorithm not in ALGORITHMS:
        typer.echo(f"Error: Unknown algorithm '{algorithm}'. Available: {list(ALGORITHMS.keys())}")
        raise typer.Exit(1)
    
    if verbose:
        typer.echo(f"Running {algorithm} algorithm on data at {data_path}")
    
    try:
        # Load data using the platform data loaders
        if verbose:
            typer.echo("Loading data...")
        
        # Load traces (required)
        traces_lf = load_traces(data_path)
        if verbose:
            typer.echo("✓ Traces loaded")
        
        # Load metrics (optional)
        metrics_lf = None
        try:
            metrics_lf = load_metrics(data_path)
            if verbose:
                typer.echo("✓ Metrics loaded")
        except Exception as e:
            if verbose:
                typer.echo(f"⚠ Metrics not available: {e}")
        
        # Load logs (optional)
        logs_lf = None
        try:
            logs_lf = load_logs(data_path)
            if verbose:
                typer.echo("✓ Logs loaded")
        except Exception as e:
            if verbose:
                typer.echo(f"⚠ Logs not available: {e}")
        
        # Create algorithm instance
        algorithm_class = ALGORITHMS[algorithm]
        alg_instance = algorithm_class()
        
        # Prepare algorithm arguments
        args = AlgorithmArgs(
            input_folder=data_path,
            traces=traces_lf,
            metrics=metrics_lf,
            logs=logs_lf
        )
        
        if verbose:
            typer.echo(f"Running {algorithm} algorithm...")
        
        # Run algorithm
        results = alg_instance(args)
        
        # Process results
        if verbose:
            typer.echo(f"Algorithm completed with {len(results)} result sets")
        
        # Output results - take the first result set
        if results:
            result = results[0]
            output_data = []
            
            for i, node_name in enumerate(result.ranks):
                score = result.scores.get(node_name, 0.0) if result.scores else 0.0
                output_data.append({
                    "rank": i + 1,
                    "node_name": node_name,
                    "score": score
                })
            
            if output:
                # Save to file
                with open(output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                typer.echo(f"Results saved to {output}")
            else:
                # Print to console
                typer.echo("\nRoot Cause Analysis Results:")
                typer.echo("-" * 50)
                for item in output_data[:10]:  # Show top 10
                    typer.echo(f"{item['rank']:2d}. {item['node_name']:<30} {item['score']:.4f}")
        else:
            typer.echo("No results returned from algorithm")
    
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
        doc = alg_class.__doc__ or 'No description'
        doc_line = doc.split('\n')[0].strip()
        typer.echo(f"  {name:<12} - {doc_line}")


@app.command()
def validate_data(
    data_path: Annotated[Path, typer.Argument(help="Path to the data directory")],
    verbose: Annotated[bool, typer.Option(help="Enable verbose output")] = False,
):
    """Validate data format and structure."""
    typer.echo(f"Validating data at {data_path}")
    
    if not data_path.exists():
        typer.echo(f"Error: Directory {data_path} does not exist")
        raise typer.Exit(1)
    
    # Check required files
    required_files = ["normal_traces.parquet", "abnormal_traces.parquet"]
    optional_files = ["normal_metrics.parquet", "abnormal_metrics.parquet", 
                     "normal_logs.parquet", "abnormal_logs.parquet", "env.json"]
    
    missing_required = []
    for file in required_files:
        file_path = data_path / file
        if file_path.exists():
            typer.echo(f"✓ Found {file}")
        else:
            missing_required.append(file)
            typer.echo(f"✗ Missing {file}")
    
    for file in optional_files:
        file_path = data_path / file
        if file_path.exists():
            typer.echo(f"✓ Found {file}")
        elif verbose:
            typer.echo(f"  {file} (optional)")
    
    if missing_required:
        typer.echo(f"\nError: Missing required files: {missing_required}")
        raise typer.Exit(1)
    
    # Try to load data to validate format
    try:
        if verbose:
            typer.echo("\nValidating data format...")
        
        traces_lf = load_traces(data_path)
        typer.echo("✓ Traces data format is valid")
        
    except Exception as e:
        typer.echo(f"✗ Data format validation failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)
    
    typer.echo("\nData validation completed successfully!")


@app.command()
def test_algorithm(
    data_path: Annotated[Path, typer.Argument(help="Path to the data directory")],
    algorithm: Annotated[str, typer.Option(help="Algorithm to test")] = "shapley",
):
    """Test an algorithm with the provided data."""
    typer.echo(f"Testing {algorithm} algorithm with data at {data_path}")
    run(data_path, algorithm, None, True)


if __name__ == "__main__":
    app()
