#!/usr/bin/env python3
"""
测试完整移植的算法实现
验证原版算法逻辑是否被正确保留
使用真实的TrainTicket数据
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from pathlib import Path

from shapleyiq.platform.algorithms import (
    TONAdapter,
    MicroHECLAdapter,
    MicroRankAdapter,
    MicroRCAAdapter,
    ShapleyRCAAdapter,
)
from shapleyiq.platform.interface import ShapleyIQAlgorithmWrapper, ShapleyIQAlgorithmArgs
from shapleyiq.platform.data_loader import PlatformDataLoader
from rcabench_platform.v2.algorithms.spec import AlgorithmArgs as RCABenchAlgorithmArgs


def load_real_data():
    """
    加载真实的TrainTicket数据
    """
    # 使用实际数据路径
    data_folder = Path("test/ts1-ts-route-plan-service-request-replace-method-qtbhzt")

    if not data_folder.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_folder}")

    print(f"📁 加载数据: {data_folder}")

    # 使用我们的数据加载器
    loader = PlatformDataLoader(data_folder)
    data = loader.load_all_data()

    print("✅ 数据加载完成:")
    if "traces" in data:
        traces_count = data["traces"].select("trace_id").unique().collect().height
        spans_count = data["traces"].collect().height
        print(f"   - Traces: {traces_count} traces, {spans_count} spans")

    return data


def test_algorithm_with_real_data(adapter_class, algorithm_name, **kwargs):
    """
    使用真实数据测试单个算法
    """
    print(f"\n=== 测试 {algorithm_name} (真实数据) ===")

    try:
        # 创建适配器实例
        adapter = adapter_class(**kwargs)
        
        # 包装为rcabench算法
        algorithm = ShapleyIQAlgorithmWrapper(adapter, cpu_count=1)

        # 准备rcabench算法参数
        data_folder = Path("test/ts1-ts-route-plan-service-request-replace-method-qtbhzt")
        args = RCABenchAlgorithmArgs(
            dataset="trainticket",
            datapack="test1", 
            input_folder=data_folder,
            output_folder=data_folder / "output"
        )

        # 运行算法
        results = algorithm(args)

        # 输出结果
        if results and len(results) > 0:
            print(f"✅ {algorithm_name} 成功运行")
            print(f"   找到 {len(results)} 个服务结果:")
            
            for result in results[:10]:  # 显示前10个结果
                print(f"     排名 {result.rank}: {result.name} (级别: {result.level})")

        else:
            print(f"❌ {algorithm_name} 运行失败: 无结果")

    except Exception as e:
        print(f"❌ {algorithm_name} 运行失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    主测试函数
    """
    print("🔍 测试完整移植的ShapleyIQ算法 (使用真实数据)")
    print("=" * 60)

    try:
        # 测试所有算法
        algorithms = [
            (
                ShapleyRCAAdapter,
                "ShapleyRCA",
                {"using_cache": False, "sync_overlap_threshold": 0.05},
            ),
            (MicroHECLAdapter, "MicroHECL", {"time_window": 15}),
            (MicroRCAAdapter, "MicroRCA", {"time_window": 15}),
            (TONAdapter, "TON", {"time_window": 15}),
            (MicroRankAdapter, "MicroRank", {"n_sigma": 3}),
        ]

        success_count = 0
        total_count = len(algorithms)

        for adapter_class, algorithm_name, kwargs in algorithms:
            try:
                test_algorithm_with_real_data(
                    adapter_class, algorithm_name, **kwargs
                )
                success_count += 1
            except Exception as e:
                print(f"❌ {algorithm_name} 测试失败: {e}")

        print("\n" + "=" * 60)
        print(f"📊 测试总结: {success_count}/{total_count} 算法测试成功")

        if success_count == total_count:
            print("🎉 所有算法都成功完成了真实数据测试!")
        else:
            print("⚠️  部分算法需要进一步调试")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()




if __name__ == "__main__":
    main()
