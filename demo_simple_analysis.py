#!/usr/bin/env python3
"""
Simplified TrainTicket Data Analysis Demo
直接分析真实TrainTicket数据的简化版本
"""

import json
from collections import Counter, defaultdict
from pathlib import Path


def find_trainticket_data():
    """查找TrainTicket数据目录"""
    data_dir = Path("ShapleyIQ/rca4tracing/fault_injection/data")
    if data_dir.exists():
        traces_dir = data_dir / "traces"
        if traces_dir.exists() and any(traces_dir.glob("*.json")):
            return data_dir
    return None


def load_traces(data_dir, num_traces=5):
    """加载trace数据"""
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


def extract_trace_info(trace):
    """从trace中提取基本信息"""
    operations = []
    durations = []
    references = []

    for span in trace:
        if isinstance(span, dict):
            # 提取操作名
            if "operationName" in span:
                operations.append(span["operationName"])

            # 提取持续时间
            if "duration" in span:
                durations.append(span["duration"])

            # 提取调用关系
            if "references" in span:
                for ref in span["references"]:
                    if ref.get("refType") == "CHILD_OF":
                        references.append((ref.get("spanID"), span.get("spanID")))

    return {
        "operations": operations,
        "durations": durations,
        "references": references,
        "operation_count": len(operations),
        "total_duration": sum(durations) if durations else 0,
        "avg_duration": sum(durations) / len(durations) if durations else 0,
    }


def analyze_operation_patterns(traces):
    """分析操作模式"""
    operation_frequencies = Counter()
    operation_durations = defaultdict(list)

    for trace in traces:
        trace_info = extract_trace_info(trace)

        # 统计操作出现次数
        for op in trace_info["operations"]:
            operation_frequencies[op] += 1

        # 统计持续时间（简化处理，假设按顺序对应）
        for i, op in enumerate(trace_info["operations"]):
            if i < len(trace_info["durations"]):
                operation_durations[op].append(trace_info["durations"][i])

    return operation_frequencies, operation_durations


def simple_shapley_analysis(traces):
    """简化的Shapley Value分析"""
    print("\n=== Simple Shapley-like Analysis ===")

    # 1. 提取所有操作
    all_operations = set()
    trace_operations = []

    for trace in traces:
        trace_info = extract_trace_info(trace)
        trace_ops = set(trace_info["operations"])
        all_operations.update(trace_ops)
        trace_operations.append(trace_ops)

    print(f"Found {len(all_operations)} unique operations across {len(traces)} traces")

    # 2. 计算操作的"贡献"（简化版本）
    operation_scores = {}

    for operation in all_operations:
        # 计算操作出现在多少个trace中
        appearance_count = sum(
            1 for trace_ops in trace_operations if operation in trace_ops
        )

        # 计算基础分数（出现频率）
        frequency_score = appearance_count / len(traces)

        # 简化的Shapley-like计算：考虑操作的边际贡献
        marginal_contribution = 0

        for trace_ops in trace_operations:
            if operation in trace_ops:
                # 这个操作在这个trace中的"重要性"
                # 简化计算：1 / trace中操作的数量
                marginal_contribution += 1 / len(trace_ops) if trace_ops else 0

        # 综合分数
        operation_scores[operation] = (
            frequency_score + marginal_contribution / len(traces)
        ) / 2

    # 3. 排序结果
    sorted_operations = sorted(
        operation_scores.items(), key=lambda x: x[1], reverse=True
    )

    print("\nOperation Importance Ranking (Shapley-like scores):")
    print("-" * 60)
    for i, (operation, score) in enumerate(sorted_operations):
        print(f"{i + 1:2d}. {operation:<40} {score:.4f}")

    return sorted_operations


def analyze_traces():
    """主分析函数"""
    print("=== TrainTicket Data Analysis Demo ===")

    # 1. 查找数据
    data_dir = find_trainticket_data()
    if data_dir is None:
        print("❌ TrainTicket data not found!")
        return

    print(f"✅ Found data at: {data_dir}")

    # 2. 加载数据
    traces, trace_ids = load_traces(data_dir, num_traces=10)
    if not traces:
        print("❌ No traces loaded!")
        return

    print(f"✅ Loaded {len(traces)} traces")

    # 3. 基本统计分析
    print("\n=== Basic Trace Analysis ===")

    total_spans = sum(len(trace) for trace in traces)
    print(f"Total spans: {total_spans}")
    print(f"Average spans per trace: {total_spans / len(traces):.1f}")

    # 4. 分析操作模式
    operation_frequencies, operation_durations = analyze_operation_patterns(traces)

    print("\nOperation Frequency Analysis:")
    print("-" * 40)
    for op, freq in operation_frequencies.most_common(10):
        avg_duration = (
            sum(operation_durations[op]) / len(operation_durations[op])
            if operation_durations[op]
            else 0
        )
        print(f"{op:<30} {freq:3d} times, avg: {avg_duration:6.0f}μs")

    # 5. 运行简化的Shapley分析
    shapley_results = simple_shapley_analysis(traces)

    # 6. 分析可能的异常模式
    print("\n=== Anomaly Pattern Analysis ===")

    # 查找持续时间异常的操作
    anomaly_candidates = []

    for op, durations in operation_durations.items():
        if len(durations) >= 2:  # 至少出现2次
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)

            # 如果最大持续时间是平均值的3倍以上，可能是异常
            if max_duration > avg_duration * 3 and avg_duration > 1000:  # 1ms以上
                anomaly_ratio = max_duration / avg_duration
                anomaly_candidates.append(
                    (op, anomaly_ratio, max_duration, avg_duration)
                )

    if anomaly_candidates:
        print("Potential anomaly operations (duration spikes):")
        anomaly_candidates.sort(key=lambda x: x[1], reverse=True)

        for op, ratio, max_dur, avg_dur in anomaly_candidates[:5]:
            print(
                f"  {op:<35} {ratio:4.1f}x spike ({max_dur:6.0f}μs vs {avg_dur:6.0f}μs avg)"
            )
    else:
        print("No obvious duration anomalies detected")

    print("\n=== Analysis Complete ===")
    print("✅ Successfully analyzed TrainTicket data!")
    print("This demonstrates the capability to process real trace data")
    print("for root cause analysis using Shapley-like algorithms.")


if __name__ == "__main__":
    analyze_traces()
