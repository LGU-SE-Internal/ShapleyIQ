"""
基于rcabench platform spec的算法实现
"""

from typing import List

from rcabench_platform.v2.algorithms.spec import (
    Algorithm,
    AlgorithmAnswer,
    AlgorithmArgs,
)

from .adapters import (
    MicroHECLAdapter,
    MicroRankAdapter,
    MicroRCAAdapter,
    ShapleyRCAAdapter,
    TONAdapter,
)
from .interface import PlatformDataConverter


class ShapleyRCA(Algorithm):
    """ShapleyValueRCA算法实现"""

    def __init__(self, using_cache: bool = False, sync_overlap_threshold: float = 0.05):
        self.adapter = ShapleyRCAAdapter(
            using_cache=using_cache, sync_overlap_threshold=sync_overlap_threshold
        )

    def needs_cpu_count(self) -> int | None:
        return 4

    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        # 转换数据
        data = PlatformDataConverter.from_rcabench_args(args)

        if data.traces is None:
            return []

        try:
            # 运行算法
            service_scores = self.adapter.run(data.traces)

            # 转换结果
            return PlatformDataConverter.to_rcabench_answers(service_scores)
        except Exception as e:
            print(f"ShapleyRCA error: {e}")
            return []


class MicroHECL(Algorithm):
    """MicroHECL算法实现"""

    def __init__(self, time_window: int = 15):
        self.adapter = MicroHECLAdapter(time_window=time_window)

    def needs_cpu_count(self) -> int | None:
        return 4

    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        # 转换数据
        data = PlatformDataConverter.from_rcabench_args(args)

        if data.traces is None:
            return []

        try:
            # 运行算法
            service_scores = self.adapter.run(data.traces)

            # 转换结果
            return PlatformDataConverter.to_rcabench_answers(service_scores)
        except Exception as e:
            print(f"MicroHECL error: {e}")
            return []


class MicroRCA(Algorithm):
    """MicroRCA算法实现"""

    def __init__(self, time_window: int = 15):
        self.adapter = MicroRCAAdapter(time_window=time_window)

    def needs_cpu_count(self) -> int | None:
        return 4

    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        # 转换数据
        data = PlatformDataConverter.from_rcabench_args(args)

        if data.traces is None:
            return []

        try:
            # 运行算法
            service_scores = self.adapter.run(data.traces)

            # 转换结果
            return PlatformDataConverter.to_rcabench_answers(service_scores)
        except Exception as e:
            print(f"MicroRCA error: {e}")
            return []


class TON(Algorithm):
    """TON算法实现"""

    def __init__(self, time_window: int = 15):
        self.adapter = TONAdapter(time_window=time_window)

    def needs_cpu_count(self) -> int | None:
        return 4

    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        # 转换数据
        data = PlatformDataConverter.from_rcabench_args(args)

        if data.traces is None:
            return []

        try:
            # 运行算法
            service_scores = self.adapter.run(data.traces)

            # 转换结果
            return PlatformDataConverter.to_rcabench_answers(service_scores)
        except Exception as e:
            print(f"TON error: {e}")
            return []


class MicroRank(Algorithm):
    """MicroRank算法实现"""

    def __init__(self, n_sigma: int = 3):
        self.adapter = MicroRankAdapter(n_sigma=n_sigma)

    def needs_cpu_count(self) -> int | None:
        return 4

    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        # 转换数据
        data = PlatformDataConverter.from_rcabench_args(args)

        if data.traces is None:
            return []

        try:
            # 运行算法
            service_scores = self.adapter.run(data.traces)

            # 转换结果
            return PlatformDataConverter.to_rcabench_answers(service_scores)
        except Exception as e:
            print(f"MicroRank error: {e}")
            return []
