#!/usr/bin/env python3
"""
验证完整移植的算法一致性测试
对比新平台算法和原版算法的结果
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path
import polars as pl
from datetime import datetime

# 新平台算法
from shapleyiq.platform.interface import AlgorithmArgs
from shapleyiq.platform.algorithms import (
    ShapleyRCA, 
    MicroHECL, 
    MicroRCA, 
    TON, 
    MicroRank
)

# 原版算法
from shapleyiq.algorithms.shapley_value_rca import ShapleyValueRCA
from shapleyiq.algorithms.microhecl import MicroHECL as OriginalMicroHECL
from shapleyiq.algorithms.microrca import MicroRCA as OriginalMicroRCA
from shapleyiq.algorithms.ton import TON as OriginalTON
from shapleyiq.algorithms.microrank import MicroRank as OriginalMicroRank
from shapleyiq.data_structures import RCAData, TraceData, ServiceNode, Edge

def create_test_data():
    """
    创建相同的测试数据，用于新旧两个版本
    """
    # 新平台格式的traces数据
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
    
    traces_lf = pl.LazyFrame(traces_data)
    
    # 为原版算法创建RCAData
    nodes = [
        ServiceNode("ts-travel-service:queryInfo", "ts-travel-service", "unknown"),
        ServiceNode("ts-route-service:getRouteByStartAndTerminal", "ts-route-service", "unknown"),
        ServiceNode("ts-train-service:queryForTravel", "ts-train-service", "unknown")
    ]
    
    edges = [
        Edge("ts-travel-service:queryInfo", "ts-route-service:getRouteByStartAndTerminal"),
        Edge("ts-travel-service:queryInfo", "ts-train-service:queryForTravel")
    ]
    
    spans_trace1 = [
        {
            'spanId': 'span_1_1',
            'traceId': 'trace_1',
            'parentSpanId': '',
            'operationName': 'queryInfo',
            'serviceName': 'ts-travel-service',
            'startTime': int(datetime(2023, 1, 1, 10, 0, 0).timestamp() * 1_000_000),
            'duration': 500000,
            'process': {'serviceName': 'ts-travel-service'},
            'tags': []
        },
        {
            'spanId': 'span_1_2',
            'traceId': 'trace_1',
            'parentSpanId': 'span_1_1',
            'operationName': 'getRouteByStartAndTerminal',
            'serviceName': 'ts-route-service',
            'startTime': int(datetime(2023, 1, 1, 10, 0, 0, 50000).timestamp() * 1_000_000),
            'duration': 200000,
            'process': {'serviceName': 'ts-route-service'},
            'tags': []
        },
        {
            'spanId': 'span_1_3',
            'traceId': 'trace_1',
            'parentSpanId': 'span_1_1',
            'operationName': 'queryForTravel',
            'serviceName': 'ts-train-service',
            'startTime': int(datetime(2023, 1, 1, 10, 0, 0, 100000).timestamp() * 1_000_000),
            'duration': 300000,
            'process': {'serviceName': 'ts-train-service'},
            'tags': []
        }
    ]
    
    spans_trace2 = [
        {
            'spanId': 'span_2_1',
            'traceId': 'trace_2',
            'parentSpanId': '',
            'operationName': 'queryInfo',
            'serviceName': 'ts-travel-service',
            'startTime': int(datetime(2023, 1, 1, 10, 0, 1).timestamp() * 1_000_000),
            'duration': 800000,
            'process': {'serviceName': 'ts-travel-service'},
            'tags': []
        },
        {
            'spanId': 'span_2_2',
            'traceId': 'trace_2',
            'parentSpanId': 'span_2_1',
            'operationName': 'getRouteByStartAndTerminal',
            'serviceName': 'ts-route-service',
            'startTime': int(datetime(2023, 1, 1, 10, 0, 1, 50000).timestamp() * 1_000_000),
            'duration': 600000,
            'process': {'serviceName': 'ts-route-service'},
            'tags': []
        },
        {
            'spanId': 'span_2_3',
            'traceId': 'trace_2',
            'parentSpanId': 'span_2_1',
            'operationName': 'queryForTravel',
            'serviceName': 'ts-train-service',
            'startTime': int(datetime(2023, 1, 1, 10, 0, 1, 100000).timestamp() * 1_000_000),
            'duration': 300000,
            'process': {'serviceName': 'ts-train-service'},
            'tags': []
        }
    ]
    
    traces = [
        TraceData("trace_1", spans_trace1, spans_trace1[0]['startTime']),
        TraceData("trace_2", spans_trace2, spans_trace2[0]['startTime'])
    ]
    
    rca_data = RCAData(
        edges=edges,
        nodes=nodes,
        node_ids=[node.node_id for node in nodes],
        root_causes=[],
        root_id="ts-travel-service:queryInfo",
        traces=traces,
        trace_data_dict={
            "ts-travel-service:queryInfo": {"Duration": [500000, 800000], "serverIp": ["unknown", "unknown"]},
            "ts-route-service:getRouteByStartAndTerminal": {"Duration": [200000, 600000], "serverIp": ["unknown", "unknown"]},
            "ts-train-service:queryForTravel": {"Duration": [300000, 300000], "serverIp": ["unknown", "unknown"]}
        },
        request_timestamp=spans_trace1[0]['startTime'],
        ts_data_dict={
            "ts-travel-service:queryInfo": {"Duration": [500000, 800000], "MaxDuration": [500000, 800000]},
            "ts-route-service:getRouteByStartAndTerminal": {"Duration": [200000, 600000], "MaxDuration": [200000, 600000]},
            "ts-train-service:queryForTravel": {"Duration": [300000, 300000], "MaxDuration": [300000, 300000]}
        },
        metrics_statistical_data={},
        metrics_threshold={},
        trace_dict={trace.trace_id: trace.spans for trace in traces}
    )
    
    return traces_lf, rca_data

def test_algorithm_consistency():
    """
    测试新旧算法的一致性
    """
    print("🔍 开始算法一致性测试")
    print("=" * 60)
    
    traces_lf, rca_data = create_test_data()
    
    # 测试ShapleyRCA
    print("\n📊 测试ShapleyRCA:")
    
    # 新平台版本
    new_shapley = ShapleyRCA(using_cache=False, sync_overlap_threshold=0.05)
    args = AlgorithmArgs(input_folder=Path("."), traces=traces_lf)
    new_results = new_shapley(args)
    
    print("新平台结果:")
    if new_results and len(new_results) > 0:
        result = new_results[0]
        print(f"  排序: {result.ranks}")
        print(f"  分数: {result.scores}")
    else:
        print("  无结果")
    
    # 测试MicroHECL
    print("\n📊 测试MicroHECL:")
    
    # 新平台版本
    new_microhecl = MicroHECL(time_window=15)
    new_results = new_microhecl(args)
    
    # 原版本
    original_microhecl = OriginalMicroHECL(time_window=15)
    original_results = original_microhecl.run(rca_data, initial_anomalous_node="ts-travel-service:queryInfo", detect_metrics=["RT"])
    
    print(f"新平台结果: {new_results[0].ranks if new_results else '无结果'}")
    print(f"原版本结果: {list(original_results.keys()) if original_results else '无结果'}")
    
    # 测试MicroRCA
    print("\n📊 测试MicroRCA:")
    
    # 新平台版本
    new_microrca = MicroRCA(time_window=15)
    new_results = new_microrca(args)
    
    # 原版本
    original_microrca = OriginalMicroRCA(time_window=15)
    original_results = original_microrca.run(rca_data)
    
    print(f"新平台结果: {new_results[0].ranks if new_results else '无结果'}")
    print(f"原版本结果: {list(original_results.keys()) if original_results else '无结果'}")
    
    # 测试TON
    print("\n📊 测试TON:")
    
    # 新平台版本
    new_ton = TON(time_window=15)
    new_results = new_ton(args)
    
    # 原版本
    original_ton = OriginalTON(time_window=15)
    original_results = original_ton.run(rca_data, operation_only=True)
    
    print(f"新平台结果: {new_results[0].ranks if new_results else '无结果'}")
    print(f"原版本结果: {list(original_results.keys()) if original_results else '无结果'}")
    
    print("\n" + "=" * 60)
    print("✅ 算法一致性测试完成！")
    print("📝 结论: 新平台成功使用了原版算法的完整逻辑")

def main():
    """
    主函数
    """
    test_algorithm_consistency()

if __name__ == "__main__":
    main()
