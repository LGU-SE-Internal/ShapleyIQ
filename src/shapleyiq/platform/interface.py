"""
Platform Interface Specifications
定义新平台的算法接口规范
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl

# Import rcabench specs
from rcabench_platform.v2.algorithms.spec import (
    Algorithm as RCABenchAlgorithm,
    AlgorithmArgs as RCABenchAlgorithmArgs,
    AlgorithmAnswer as RCABenchAlgorithmAnswer,
)


@dataclass
class ShapleyIQAlgorithmArgs:
    """Extended algorithm input arguments for ShapleyIQ"""

    input_folder: Path
    traces: Optional[pl.LazyFrame] = None
    inject_time: Optional[Any] = None
    parameters: Optional[Dict[str, Any]] = None


class BaseAdapter:
    """Base adapter for converting between old and new interfaces"""

    def __init__(self, algorithm_func):
        self.algorithm_func = algorithm_func

    def __call__(self, args: ShapleyIQAlgorithmArgs) -> List[RCABenchAlgorithmAnswer]:
        """Convert new interface to old and back"""
        raise NotImplementedError("Subclasses must implement this method")


class ShapleyIQAlgorithmWrapper(RCABenchAlgorithm):
    """Wrapper to adapt ShapleyIQ algorithms to rcabench interface"""

    def __init__(self, adapter: BaseAdapter, cpu_count: Optional[int] = None):
        self.adapter = adapter
        self._cpu_count = cpu_count

    def needs_cpu_count(self) -> int | None:
        return self._cpu_count

    def __call__(self, args: RCABenchAlgorithmArgs) -> List[RCABenchAlgorithmAnswer]:
        # Convert rcabench args to ShapleyIQ args
        from .data_loader import PlatformDataLoader

        loader = PlatformDataLoader(args.input_folder)
        data = loader.load_all_data()

        shapleyiq_args = ShapleyIQAlgorithmArgs(
            input_folder=args.input_folder,
            traces=data.get("traces"),
            inject_time=data.get("inject_time"),
            parameters={},
        )

        return self.adapter(shapleyiq_args)
