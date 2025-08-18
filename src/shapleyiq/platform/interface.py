"""
Platform Interface Specifications
定义新平台的算法接口规范和数据转换
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import polars as pl
from rcabench_platform.v2.algorithms.spec import (
    AlgorithmAnswer as RCABenchAlgorithmAnswer,
)

# Import rcabench specs
from rcabench_platform.v2.algorithms.spec import (
    AlgorithmArgs as RCABenchAlgorithmArgs,
)


@dataclass
class ShapleyIQData:
    """ShapleyIQ平台数据结构"""

    traces: Optional[pl.LazyFrame] = None
    inject_time: Optional[Any] = None


class PlatformDataConverter:
    """平台数据转换器"""

    @staticmethod
    def from_rcabench_args(args: RCABenchAlgorithmArgs) -> ShapleyIQData:
        """从rcabench参数转换为ShapleyIQ数据"""
        from .data_loader import PlatformDataLoader

        loader = PlatformDataLoader(args.input_folder)
        data = loader.load_all_data()

        return ShapleyIQData(
            traces=data.get("traces"), inject_time=data.get("inject_time")
        )

    @staticmethod
    def to_rcabench_answers(
        service_scores: Dict[str, float],
    ) -> List[RCABenchAlgorithmAnswer]:
        """将service级别的分数转换为rcabench答案格式"""
        # 按分数排序
        sorted_services = sorted(
            service_scores.items(), key=lambda x: x[1], reverse=True
        )

        answers = []
        for rank, (service_name, score) in enumerate(sorted_services, start=1):
            answers.append(
                RCABenchAlgorithmAnswer(level="service", name=service_name, rank=rank)
            )
        return answers
