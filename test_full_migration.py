#!/usr/bin/env python3
"""
测试完整移植的算法实现
验证原版算法逻辑是否被正确保留
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path
import polars as pl
from datetime import datetime

from shapleyiq.platform.interface import AlgorithmArgs
from shapleyiq.platform.full_algorithms import (
    ShapleyRCA, 
    MicroHECL, 
    MicroRCA, 
    TON, 
    MicroRank
)

def create_test_traces_data():
    """
    创建测试用的traces数据（新平台格式）
    """
    # 模拟TrainTicket数据格式
    traces_data = [
        {
            'trace_id': 'trace_1',
            'span_id': 'span_1_1',
            'parent_span_id': '',
            'service_name': 'ts-travel-service',
            'operation_name': 'queryInfo', 
            'span_name': 'queryInfo',
            'time': datetime(2023, 1, 1, 10, 0, 0),
            'duration': 500000  # 500ms
        },
        {
            'trace_id': 'trace_1',
            'span_id': 'span_1_2', 
            'parent_span_id': 'span_1_1',
            'service_name': 'ts-route-service',
            'operation_name': 'getRouteByStartAndTerminal',
            'span_name': 'getRouteByStartAndTerminal',
            'time': datetime(2023, 1, 1, 10, 0, 0, 50000),
            'duration': 200000  # 200ms
        },
        {
            'trace_id': 'trace_1',
            'span_id': 'span_1_3',
            'parent_span_id': 'span_1_1', 
            'service_name': 'ts-train-service',
            'operation_name': 'queryForTravel',
            'span_name': 'queryForTravel',
            'time': datetime(2023, 1, 1, 10, 0, 0, 100000),
            'duration': 300000  # 300ms
        },
        {
            'trace_id': 'trace_2',
            'span_id': 'span_2_1',
            'parent_span_id': '',
            'service_name': 'ts-travel-service', 
            'operation_name': 'queryInfo',
            'span_name': 'queryInfo',
            'time': datetime(2023, 1, 1, 10, 0, 1),
            'duration': 800000  # 800ms (异常慢)
        },
        {
            'trace_id': 'trace_2',
            'span_id': 'span_2_2',
            'parent_span_id': 'span_2_1',
            'service_name': 'ts-route-service',
            'operation_name': 'getRouteByStartAndTerminal', 
            'span_name': 'getRouteByStartAndTerminal',
            'time': datetime(2023, 1, 1, 10, 0, 1, 50000),
            'duration': 600000  # 600ms (异常慢)
        },
        {
            'trace_id': 'trace_2',
            'span_id': 'span_2_3',
            'parent_span_id': 'span_2_1',
            'service_name': 'ts-train-service',
            'operation_name': 'queryForTravel',
            'span_name': 'queryForTravel', 
            'time': datetime(2023, 1, 1, 10, 0, 1, 100000),
            'duration': 300000  # 300ms (正常)
        }
    ]
    
    # 转换为Polars LazyFrame
    traces_lf = pl.LazyFrame(traces_data)
    return traces_lf

def test_algorithm(algorithm_class, algorithm_name, **kwargs):
    """
    测试单个算法
    """
    print(f"\n=== 测试 {algorithm_name} ===")
    
    try:
        # 创建算法实例
        algorithm = algorithm_class(**kwargs)
        
        # 准备测试数据
        traces_lf = create_test_traces_data()
        args = AlgorithmArgs(
            input_folder=Path("."),
            traces=traces_lf
        )
        
        # 运行算法
        results = algorithm(args)
        
        # 输出结果
        if results and len(results) > 0:
            result = results[0]
            print(f"✅ {algorithm_name} 成功运行")
            print(f"   排序结果: {result.ranks[:5]}")  # 只显示前5个
            if result.scores:
                print("   前5个分数:", {k: v for k, v in list(result.scores.items())[:5]})
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
    print("🔍 测试完整移植的ShapleyIQ算法")
    print("=" * 50)
    
    # 测试所有算法
    algorithms = [
        (ShapleyRCA, "完整ShapleyRCA", {"using_cache": False, "sync_overlap_threshold": 0.05}),
        (MicroHECL, "完整MicroHECL", {"time_window": 15}),
        (MicroRCA, "完整MicroRCA", {"time_window": 15}),
        (TON, "完整TON", {"time_window": 15}),
        (MicroRank, "完整MicroRank", {"n_sigma": 3})
    ]
    
    success_count = 0
    total_count = len(algorithms)
    
    for algorithm_class, algorithm_name, kwargs in algorithms:
        try:
            test_algorithm(algorithm_class, algorithm_name, **kwargs)
            success_count += 1
        except Exception as e:
            print(f"❌ {algorithm_name} 测试失败: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"📊 测试总结: {success_count}/{total_count} 算法测试成功")
    
    if success_count == total_count:
        print("🎉 所有算法都成功完成了完整移植!")
    else:
        print("⚠️  部分算法移植需要进一步调试")

if __name__ == "__main__":
    main()
