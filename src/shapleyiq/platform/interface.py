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
    metrics: Optional[pl.LazyFrame | None] = None
    inject_time: Optional[Any] = None
    initial_anomalous_node: Optional[str] = None  # 单个初始异常节点（为MicroHECL提供）
    anomalous_services: Optional[List[str]] = (
        None  # 多个异常服务（为TON和MicroRCA提供）
    )


class PlatformDataConverter:
    """平台数据转换器"""

    @staticmethod
    def from_rcabench_args(
        args: RCABenchAlgorithmArgs, need_anomaly_detection=True, need_metrics=False
    ) -> ShapleyIQData:
        """从rcabench参数转换为ShapleyIQ数据"""
        from .alarm_detector import detect_anomalous_services
        from .data_loader import PlatformDataLoader

        loader = PlatformDataLoader(args.input_folder, need_metrics=need_metrics)
        data = loader.load_all_data()

        # 尝试从conclusion数据中自动检测异常节点/服务
        initial_anomalous_node = None
        anomalous_services = []
        if need_anomaly_detection:
            try:
                anomalous_services = detect_anomalous_services(args.input_folder)
                initial_anomalous_node = (
                    anomalous_services[0] if anomalous_services else None
                )
            except Exception as e:
                print(f"Failed to detect anomalous services: {e}")

        return ShapleyIQData(
            traces=data.get("traces"),
            metrics=data.get("metrics") if need_metrics else None,
            inject_time=data.get("inject_time"),
            initial_anomalous_node=initial_anomalous_node,
            anomalous_services=anomalous_services,
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
