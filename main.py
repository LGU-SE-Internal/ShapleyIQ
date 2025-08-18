#!/usr/bin/env -S uv run -s
from src.shapleyiq.platform.algorithms import TON, MicroRank, MicroHECL, MicroRCA, ShapleyRCA
from rcabench_platform.v2.cli.main import main
from rcabench_platform.v2.algorithms.spec import global_algorithm_registry
if __name__ == "__main__":
    registry = global_algorithm_registry()
    registry["shapleyiq"] = ShapleyRCA
    registry["ton"] = TON
    registry["microrank"] = MicroRank
    registry["microhecl"] = MicroHECL
    registry["microrca"] = MicroRCA

    main(enable_builtin_algorithms=False)
