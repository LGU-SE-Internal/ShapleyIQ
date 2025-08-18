#!/usr/bin/env python3
"""
Real TrainTicket Data Demo for ShapleyIQ
使用真实的TrainTicket数据演示ShapleyIQ包的功能
"""

import json
import sys
import time
from pathlib import Path

# 添加src目录到Python路径
sys.path.append("src")

# 导入我们重构的包
from shapleyiq.algorithms.shapley_rca import ShapleyValueRCA
from shapleyiq.preprocessing.trace_preprocessor import TracePreprocessor


def find_trainticket_data():
    """查找TrainTicket数据目录"""
    base_dir = Path(".")

    # 检查可能的数据路径
    possible_paths = [
        "ShapleyIQ/rca4tracing/fault_injection/data",
        "rca4tracing/fault_injection/data",
        "fault_injection/data",
        "data",
    ]

    for path in possible_paths:
        full_path = base_dir / path
        if full_path.exists():
            traces_dir = full_path / "traces"
            if traces_dir.exists() and any(traces_dir.glob("*.json")):
                return full_path

    return None


def load_trace_data(data_dir, max_traces=10):
    """
    加载TrainTicket trace数据

    Args:
        data_dir: 数据目录路径
        max_traces: 最大加载trace数量

    Returns:
        traces: trace数据列表
        trace_ids: trace ID列表
    """
    traces_dir = data_dir / "traces"
    if not traces_dir.exists():
        raise FileNotFoundError(f"Traces directory not found: {traces_dir}")

    # 获取所有JSON文件
    trace_files = list(traces_dir.glob("*.json"))[:max_traces]

    traces = []
    trace_ids = []

    print(f"Loading {len(trace_files)} trace files...")

    for trace_file in trace_files:
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                trace_data = json.load(f)

            # trace文件名就是traceID
            trace_id = trace_file.stem

            traces.append(trace_data)
            trace_ids.append(trace_id)

        except Exception as e:
            print(f"Error loading {trace_file}: {e}")
            continue

    print(f"Successfully loaded {len(traces)} traces")
    return traces, trace_ids


def analyze_trace_structure(traces):
    """分析trace数据结构"""
    if not traces:
        return

    print("\n=== Trace Data Structure Analysis ===")

    # 分析第一个trace的结构
    first_trace = traces[0]
    print(f"First trace type: {type(first_trace)}")

    if isinstance(first_trace, dict):
        print(f"Keys in first trace: {list(first_trace.keys())}")

        # 如果有data字段，分析其结构
        if "data" in first_trace:
            data = first_trace["data"]
            if isinstance(data, list) and data:
                print(f"Number of spans in first trace: {len(data)}")
                print(
                    f"Sample span keys: {list(data[0].keys()) if data else 'No spans'}"
                )

    elif isinstance(first_trace, list):
        print(f"First trace is a list with {len(first_trace)} items")
        if first_trace:
            print(f"Sample span keys: {list(first_trace[0].keys())}")


def extract_services_from_traces(traces):
    """从traces中提取服务信息"""
    services = set()
    operations = set()

    for trace in traces:
        spans = []

        # 处理不同的trace格式
        if isinstance(trace, dict):
            if "data" in trace:
                spans = trace["data"]
            elif "spans" in trace:
                spans = trace["spans"]
            else:
                # 假设整个dict就是spans的容器
                spans = [trace] if "operationName" in trace else []
        elif isinstance(trace, list):
            spans = trace

        # 提取服务和操作信息
        for span in spans:
            if isinstance(span, dict):
                if "process" in span and "serviceName" in span["process"]:
                    services.add(span["process"]["serviceName"])
                if "operationName" in span:
                    operations.add(span["operationName"])

    return list(services), list(operations)


def preprocess_jaeger_traces(traces):
    """
    预处理Jaeger格式的traces，转换为我们算法需要的格式
    """
    preprocessed_traces = []

    for trace in traces:
        spans = []

        # 提取spans
        if isinstance(trace, dict):
            if "data" in trace:
                spans = trace["data"]
            elif "spans" in trace:
                spans = trace["spans"]
        elif isinstance(trace, list):
            spans = trace

        # 转换spans格式
        processed_spans = []
        for span in spans:
            if isinstance(span, dict):
                processed_span = {
                    "spanID": span.get("spanID", ""),
                    "traceID": span.get("traceID", ""),
                    "operationName": span.get("operationName", ""),
                    "startTime": span.get("startTime", 0),
                    "duration": span.get("duration", 0),
                    "references": span.get("references", []),
                    "process": span.get("process", {}),
                    "logs": span.get("logs", []),
                    "tags": span.get("tags", {}),
                }

                # 确保process包含serviceName
                if "serviceName" not in processed_span["process"]:
                    processed_span["process"]["serviceName"] = "unknown-service"

                processed_spans.append(processed_span)

        if processed_spans:
            preprocessed_traces.append(processed_spans)

    return preprocessed_traces


def run_algorithms_on_trainticket_data():
    """在TrainTicket数据上运行RCA算法"""

    print("=== ShapleyIQ Real TrainTicket Data Demo ===")

    # 1. 查找数据
    data_dir = find_trainticket_data()
    if data_dir is None:
        print("❌ TrainTicket data not found!")
        print(
            "Please ensure the TrainTicket data is available in one of these locations:"
        )
        print("- ShapleyIQ/rca4tracing/fault_injection/data/")
        print("- rca4tracing/fault_injection/data/")
        return

    print(f"✅ Found TrainTicket data at: {data_dir}")

    # 2. 加载数据
    try:
        traces, trace_ids = load_trace_data(data_dir, max_traces=5)  # 先用少量数据测试
        if not traces:
            print("❌ No valid traces loaded!")
            return

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 3. 分析数据结构
    analyze_trace_structure(traces)

    # 4. 提取服务信息
    services, operations = extract_services_from_traces(traces)
    print(f"\n=== Discovered Services ({len(services)}) ===")
    for service in sorted(services)[:10]:  # 显示前10个
        print(f"  - {service}")
    if len(services) > 10:
        print(f"  ... and {len(services) - 10} more")

    # 5. 预处理数据
    print("\n=== Preprocessing Traces ===")
    preprocessed_traces = preprocess_jaeger_traces(traces)
    print(f"Preprocessed {len(preprocessed_traces)} traces")

    # 6. 使用我们的TracePreprocessor进一步处理
    try:
        preprocessor = TracePreprocessor()

        # 处理每个trace
        processed_data = []
        for i, trace in enumerate(preprocessed_traces):
            try:
                # 构建调用图
                call_graph = preprocessor.build_call_graph(trace)

                # 计算持续时间统计
                duration_stats = preprocessor.calculate_duration_statistics(trace)

                processed_data.append(
                    {
                        "trace_id": trace_ids[i],
                        "spans": trace,
                        "call_graph": call_graph,
                        "duration_stats": duration_stats,
                    }
                )

            except Exception as e:
                print(f"Warning: Error processing trace {trace_ids[i]}: {e}")
                continue

        print(f"Successfully processed {len(processed_data)} traces")

    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return

    # 7. 运行Shapley Value RCA
    print("\n=== Running Shapley Value RCA ===")
    try:
        shapley_rca = ShapleyValueRCA()

        # 使用原始的预处理traces（Shapley算法期望这种格式）
        start_time = time.time()
        results = shapley_rca.analyze_traces(
            preprocessed_traces[:3]
        )  # 先用3个traces测试
        duration = time.time() - start_time

        print(f"✅ Shapley RCA completed in {duration:.2f}s")
        print("Top 10 results:")

        # 转换为概率并排序
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for i, (service, score) in enumerate(sorted_results[:10]):
            print(f"  {i + 1:2d}. {service:<30} {score:.4f}")

    except Exception as e:
        print(f"❌ Error running Shapley RCA: {e}")
        import traceback

        traceback.print_exc()

    # 8. 运行其他算法（如果数据结构支持）
    print("\n=== Running Other Algorithms ===")

    # 这里需要更多的数据准备工作来适配其他算法
    # 暂时跳过，因为它们需要更复杂的数据结构
    print("⚠️  Other algorithms require additional data preparation")
    print("    (metrics, thresholds, etc.) - skipping for now")

    print("\n=== Demo Completed ===")
    print("✅ Successfully demonstrated ShapleyIQ on real TrainTicket data!")


if __name__ == "__main__":
    run_algorithms_on_trainticket_data()
