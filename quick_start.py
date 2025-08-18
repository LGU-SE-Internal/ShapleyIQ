#!/usr/bin/env python3
"""
Quick start demo for ShapleyIQ with real data.

This script shows how to quickly analyze your TrainTicket data using ShapleyIQ.
"""

import sys

from rcabench_platform.v2.logging import logger


def demo_cli_usage():
    """Demonstrate CLI usage with examples."""

    logger.info("🚀 ShapleyIQ CLI Demo")
    logger.info("=" * 50)

    logger.info("\n📖 Available Commands:")
    logger.info("1. Analyze trace data:")
    logger.info(
        "   python -m shapleyiq analyze trace_file.json --algorithm shapley --root-causes service1"
    )

    logger.info("\n2. Run demonstration:")
    logger.info("   python -m shapleyiq demo --dataset example")

    logger.info("\n3. Validate installation:")
    logger.info("   python -m shapleyiq validate")

    logger.info("\n🔍 Example with your TrainTicket data:")
    logger.info(
        "   python -m shapleyiq analyze rca4tracing/fault_injection/data/traces/ts-basic-service100_users5_spawn_rate5.json \\"
    )
    logger.info("          --algorithm shapley \\")
    logger.info("          --root-causes ts-basic-service \\")
    logger.info("          --format jaeger")

    logger.info("\n📊 Compare all algorithms:")
    logger.info("   python demo_real_data.py")

    logger.info("\n💡 Tips:")
    logger.info("   - Use --help for detailed options")
    logger.info(
        "   - The algorithm names are: shapley, microhecl, microrca, microrank, ton"
    )
    logger.info("   - Root causes should match the service names in your trace data")


def main():
    """Main demo function."""
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        # Run validation script
        import subprocess

        subprocess.run([sys.executable, "validate_setup.py"])
    else:
        demo_cli_usage()


if __name__ == "__main__":
    main()
