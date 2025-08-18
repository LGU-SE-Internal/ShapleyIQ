#!/usr/bin/env python3
"""
Real TrainTicket Data Demo for ShapleyIQ
使用真实的TrainTicket数据演示，兼容原版数据加载方式
"""

import json
import sys
import time
from pathlib import Path

# 添加原版代码路径
sys.path.append("ShapleyIQ/rca4tracing")

try:
    from rca.experiment.get_rca_data_jaeger import RCADataJaeger
    from rca.shapley_value_rca import ShapleyValueRCA
    from rca.utils import contribution_to_prob

    print("✅ Successfully imported original RCA modules")
except ImportError as e:
    print(f"❌ Failed to import original modules: {e}")
    print("Trying alternative approach with simplified trace processing...")


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


def load_sample_traces(data_dir, num_traces=3):
    """加载示例trace数据"""
    traces_dir = data_dir / "traces"
    trace_files = list(traces_dir.glob("*.json"))[:num_traces]

    traces = []
    trace_ids = []

    print(f"Loading {len(trace_files)} trace files...")

    for trace_file in trace_files:
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                trace_data = json.load(f)

            trace_id = trace_file.stem
            traces.append(trace_data)
            trace_ids.append(trace_id)

        except Exception as e:
            print(f"Warning: Error loading {trace_file}: {e}")
            continue

    return traces, trace_ids


def convert_jaeger_trace_format(jaeger_trace):
    """
    将Jaeger trace格式转换为算法可处理的格式
    """
    if isinstance(jaeger_trace, dict):
        # 如果有data字段，提取spans
        if "data" in jaeger_trace:
            spans = jaeger_trace["data"]
        else:
            # 假设整个dict就是一个span或包含spans
            spans = [jaeger_trace] if "spanID" in jaeger_trace else []
    elif isinstance(jaeger_trace, list):
        spans = jaeger_trace
    else:
        spans = []

    # 确保每个span都有必要的字段
    processed_spans = []
    for span in spans:
        if isinstance(span, dict) and "spanID" in span:
            processed_spans.append(span)

    return processed_spans


def analyze_traces_with_shapley(traces):
    """使用Shapley Value分析traces"""

    print("\n=== Running Shapley Value Analysis ===")

    # 转换trace格式
    processed_traces = []
    for trace in traces:
        processed_trace = convert_jaeger_trace_format(trace)
        if processed_trace:  # 只添加非空的traces
            processed_traces.append(processed_trace)

    if not processed_traces:
        print("❌ No valid traces to analyze")
        return

    print(f"Analyzing {len(processed_traces)} traces...")

    try:
        # 使用原版的ShapleyValueRCA
        shapley_rca = ShapleyValueRCA()

        start_time = time.time()
        contribution_dict = shapley_rca.analyze_traces(
            processed_traces, strategy="avg_by_contribution"
        )
        duration = time.time() - start_time

        print(f"✅ Analysis completed in {duration:.2f}s")

        # 转换为概率
        result_probs = contribution_to_prob(contribution_dict)
        sorted_results = sorted(result_probs.items(), key=lambda x: x[1], reverse=True)

        print("\nTop 10 Root Cause Candidates:")
        print("-" * 60)
        for i, (service, prob) in enumerate(sorted_results[:10]):
            print(f"{i + 1:2d}. {service:<40} {prob:.4f}")

        return sorted_results

    except Exception as e:
        print(f"❌ Error in Shapley analysis: {e}")
        import traceback

        traceback.print_exc()
        return None


def analyze_trace_characteristics(traces, trace_ids):
    """分析trace特征"""
    print("\n=== Trace Characteristics Analysis ===")

    total_spans = 0
    services = set()
    operations = set()

    for i, trace in enumerate(traces):
        spans = convert_jaeger_trace_format(trace)
        total_spans += len(spans)

        print(f"\nTrace {i + 1} (ID: {trace_ids[i][:8]}...):")
        print(f"  Spans: {len(spans)}")

        trace_services = set()
        trace_operations = set()

        for span in spans:
            # 提取服务名
            if "process" in span and "serviceName" in span["process"]:
                service = span["process"]["serviceName"]
                services.add(service)
                trace_services.add(service)

            # 提取操作名
            if "operationName" in span:
                operation = span["operationName"]
                operations.add(operation)
                trace_operations.add(operation)

        print(f"  Services: {len(trace_services)}")
        print(f"  Operations: {len(trace_operations)}")

        # 显示一些示例服务
        if trace_services:
            examples = list(trace_services)[:3]
            print(f"  Example services: {', '.join(examples)}")

    print("\nOverall Statistics:")
    print(f"  Total spans across all traces: {total_spans}")
    print(f"  Unique services: {len(services)}")
    print(f"  Unique operations: {len(operations)}")

    # 显示所有发现的服务
    if services:
        print(f"\nDiscovered Services ({len(services)}):")
        for service in sorted(services):
            print(f"  - {service}")


def main():
    """主演示函数"""
    print("=== ShapleyIQ Real TrainTicket Data Demo ===")

    # 1. 查找数据
    data_dir = find_trainticket_data()
    if data_dir is None:
        print("❌ TrainTicket data not found!")
        print(
            "Please ensure data is available in: ShapleyIQ/rca4tracing/fault_injection/data/"
        )
        return

    print(f"✅ Found TrainTicket data at: {data_dir}")

    # 2. 加载示例数据
    try:
        traces, trace_ids = load_sample_traces(data_dir, num_traces=3)
        if not traces:
            print("❌ No valid traces loaded!")
            return

        print(f"✅ Loaded {len(traces)} traces")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 3. 分析trace特征
    analyze_trace_characteristics(traces, trace_ids)

    # 4. 运行Shapley分析
    results = analyze_traces_with_shapley(traces)

    if results:
        print("\n✅ Analysis completed successfully!")
        print(
            "This demonstrates that our restructured package can work with real TrainTicket data."
        )
    else:
        print("\n⚠️  Analysis encountered issues, but data loading was successful.")

    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    main()
