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
    TON,
    MicroHECL,
    MicroRank,
    MicroRCA,
    ShapleyRCA,
)
from shapleyiq.platform.interface import AlgorithmArgs
from shapleyiq.platform.data_loader import NewPlatformDataLoader


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
    loader = NewPlatformDataLoader(data_folder)
    data = loader.load_all_data()
    
    print("✅ 数据加载完成:")
    if "traces" in data:
        traces_count = data["traces"].select("trace_id").unique().collect().height
        spans_count = data["traces"].collect().height
        print(f"   - Traces: {traces_count} traces, {spans_count} spans")
    if "metrics" in data:
        metrics_count = data["metrics"].collect().height
        print(f"   - Metrics: {metrics_count} records")
    if "logs" in data:
        logs_count = data["logs"].collect().height
        print(f"   - Logs: {logs_count} records")
    
    return data


def test_algorithm_with_real_data(algorithm_class, algorithm_name, data, **kwargs):
    """
    使用真实数据测试单个算法
    """
    print(f"\n=== 测试 {algorithm_name} (真实数据) ===")

    try:
        # 创建算法实例
        algorithm = algorithm_class(**kwargs)

        # 准备算法参数
        args = AlgorithmArgs(
            input_folder=Path("test/ts1-ts-route-plan-service-request-replace-method-qtbhzt"),
            traces=data.get("traces"),
            metrics=data.get("metrics"),
            metrics_histogram=data.get("metrics_histogram"),
            logs=data.get("logs"),
            inject_time=data.get("inject_time")
        )

        # 运行算法
        results = algorithm(args)

        # 输出结果
        if results and len(results) > 0:
            result = results[0]
            print(f"✅ {algorithm_name} 成功运行")
            
            # 显示operation级别结果
            if result.ranks:
                print(f"   Operation排序 (前10个): {result.ranks[:10]}")
                if result.scores:
                    print("   前10个分数:")
                    for i, op in enumerate(result.ranks[:10]):
                        score = result.scores.get(op, 0)
                        print(f"     {i+1}. {op}: {score:.2f}")
            
            # 显示service级别结果
            if result.service_ranking:
                print(f"   Service排序: {result.service_ranking}")
                
            if result.metadata:
                print(f"   元数据: {result.metadata}")
                
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
        # 加载真实数据
        data = load_real_data()
        
        # 测试所有算法
        algorithms = [
            (
                ShapleyRCA,
                "ShapleyRCA", 
                {"using_cache": False, "sync_overlap_threshold": 0.05},
            ),
            (MicroHECL, "MicroHECL", {"time_window": 15}),
            (MicroRCA, "MicroRCA", {"time_window": 15}),
            (TON, "TON", {"time_window": 15}),
            (MicroRank, "MicroRank", {"n_sigma": 3}),
        ]

        success_count = 0
        total_count = len(algorithms)

        for algorithm_class, algorithm_name, kwargs in algorithms:
            try:
                test_algorithm_with_real_data(algorithm_class, algorithm_name, data, **kwargs)
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
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
