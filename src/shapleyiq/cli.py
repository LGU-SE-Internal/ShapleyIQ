"""
Command Line Interface for ShapleyIQ.

This module provides a CLI interface for running ShapleyIQ algorithms.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .algorithms import MicroHECL, MicroRank, MicroRCA, ShapleyValueRCA
from .preprocessing import RCADataBuilder

app = typer.Typer(
    name="shapleyiq",
    help="ShapleyIQ: Influence Quantification by Shapley Values for Performance Debugging of Microservices",
)
console = Console()


@app.command()
def analyze(
    trace_file: Path = typer.Argument(..., help="Path to trace data file"),
    root_causes: str = typer.Option(
        "", help="Comma-separated list of known root cause node IDs"
    ),
    algorithm: str = typer.Option(
        "ShapleyValueRCA",
        help="Algorithm to use: ShapleyValueRCA, MicroHECL, MicroRCA, MicroRank",
    ),
    trace_format: str = typer.Option(
        "jaeger", help="Trace format: jaeger, zipkin, dbaas"
    ),
    output: Optional[Path] = typer.Option(None, help="Output file for results"),
    top_k: int = typer.Option(5, help="Number of top results to show"),
):
    """
    Analyze traces to identify root causes using the specified algorithm.
    """
    console.print(
        f"[bold blue]Running {algorithm} analysis on {trace_file}[/bold blue]"
    )

    # Parse root causes
    root_cause_list = (
        [rc.strip() for rc in root_causes.split(",") if rc.strip()]
        if root_causes
        else []
    )

    try:
        # Build RCA data
        builder = RCADataBuilder()
        data = builder.build_from_files(
            trace_file=str(trace_file),
            root_causes=root_cause_list,
            trace_format=trace_format,
        )

        if data.is_empty:
            console.print(
                "[bold red]Error: No valid data found in trace file[/bold red]"
            )
            raise typer.Exit(1)

        # Select and run algorithm
        if algorithm == "ShapleyValueRCA":
            algo = ShapleyValueRCA()
        elif algorithm == "MicroHECL":
            algo = MicroHECL()
        elif algorithm == "MicroRCA":
            algo = MicroRCA()
        elif algorithm == "MicroRank":
            algo = MicroRank()
        else:
            console.print(
                f"[bold red]Error: Unknown algorithm '{algorithm}'[/bold red]"
            )
            raise typer.Exit(1)

        # Run analysis
        console.print("Running analysis...")
        results = algo.run(data)

        if not results:
            console.print("[yellow]Warning: No results found[/yellow]")
            return

        # Display results
        _display_results(results, top_k, algorithm)

        # Save results if requested
        if output:
            _save_results(results, output)
            console.print(f"[green]Results saved to {output}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error during analysis: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def demo():
    """
    Run a demo analysis with synthetic data.
    """
    console.print("[bold blue]Running ShapleyIQ Demo[/bold blue]")

    # Create synthetic data
    from .data_structures import Edge, RCAData, ServiceNode, TraceData

    # Create simple call graph: A -> B -> C, A -> D
    edges = [
        Edge("ServiceA:op1", "ServiceB:op1"),
        Edge("ServiceB:op1", "ServiceC:op1"),
        Edge("ServiceA:op1", "ServiceD:op1"),
    ]

    nodes = [
        ServiceNode("ServiceA:op1", "ServiceA", "192.168.1.1"),
        ServiceNode("ServiceB:op1", "ServiceB", "192.168.1.2"),
        ServiceNode("ServiceC:op1", "ServiceC", "192.168.1.3"),
        ServiceNode("ServiceD:op1", "ServiceD", "192.168.1.4"),
    ]

    # Create synthetic trace
    spans = [
        {
            "spanId": "1",
            "serviceName": "ServiceA",
            "operationName": "op1",
            "startTime": 1000,
            "duration": 100,
            "parentSpanId": None,
        },
        {
            "spanId": "2",
            "serviceName": "ServiceB",
            "operationName": "op1",
            "startTime": 1010,
            "duration": 50,
            "parentSpanId": "1",
        },
        {
            "spanId": "3",
            "serviceName": "ServiceC",
            "operationName": "op1",
            "startTime": 1020,
            "duration": 30,
            "parentSpanId": "2",
        },
        {
            "spanId": "4",
            "serviceName": "ServiceD",
            "operationName": "op1",
            "startTime": 1070,
            "duration": 20,
            "parentSpanId": "1",
        },
    ]

    trace = TraceData("demo_trace", spans)

    data = RCAData(
        edges=edges,
        nodes=nodes,
        traces=[trace],
        root_causes=["ServiceC:op1"],  # Known root cause for demo
    )

    # Run ShapleyValueRCA
    console.print("Running ShapleyValueRCA on demo data...")
    algo = ShapleyValueRCA()
    results = algo.run(data)

    # Display results
    _display_results(results, 5, "ShapleyValueRCA")

    console.print("\n[green]Demo completed successfully![/green]")


@app.command()
def validate(
    trace_file: Path = typer.Argument(..., help="Path to trace data file"),
    trace_format: str = typer.Option(
        "jaeger", help="Trace format: jaeger, zipkin, dbaas"
    ),
):
    """
    Validate trace data format and structure.
    """
    console.print(f"[bold blue]Validating trace file: {trace_file}[/bold blue]")

    try:
        builder = RCADataBuilder()
        data = builder.build_from_files(
            trace_file=str(trace_file), root_causes=[], trace_format=trace_format
        )

        # Display validation results
        table = Table(title="Trace Data Validation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Format", trace_format)
        table.add_row("Number of Traces", str(len(data.traces) if data.traces else 0))
        table.add_row("Number of Nodes", str(len(data.nodes)))
        table.add_row("Number of Edges", str(len(data.edges)))
        table.add_row("Root ID", data.root_id or "Not found")
        table.add_row(
            "Request Timestamp",
            str(data.request_timestamp) if data.request_timestamp else "Not found",
        )
        table.add_row(
            "Validation Status", "✅ PASSED" if data.validate() else "❌ FAILED"
        )

        console.print(table)

        if data.validate():
            console.print("[green]Trace data is valid and ready for analysis![/green]")
        else:
            console.print(
                "[red]Trace data validation failed. Please check the data format.[/red]"
            )
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[bold red]Error during validation: {e}[/bold red]")
        raise typer.Exit(1)


def _display_results(results: dict, top_k: int, algorithm: str):
    """Display analysis results in a formatted table."""
    table = Table(title=f"{algorithm} Analysis Results")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Node ID", style="magenta")
    table.add_column("Score", justify="right", style="green")

    # Get top k results
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]

    for i, (node_id, score) in enumerate(sorted_results, 1):
        table.add_row(str(i), node_id, f"{score:.4f}")

    console.print(table)


def _save_results(results: dict, output_path: Path):
    """Save results to file."""
    import json

    # Sort results by score
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    with open(output_path, "w") as f:
        json.dump(sorted_results, f, indent=2)


if __name__ == "__main__":
    app()
