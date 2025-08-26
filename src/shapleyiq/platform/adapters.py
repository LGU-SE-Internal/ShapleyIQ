"""
Algorithm Adapters
将原有算法适配到新的平台接口
"""

from datetime import datetime
from typing import Dict, List, Optional

import polars as pl

from ..algorithms.microhecl import MicroHECL
from ..algorithms.microrank import MicroRank
from ..algorithms.microrca import MicroRCA
from ..algorithms.shapley_value_rca import ShapleyValueRCA
from ..algorithms.ton import TON
from ..data_structures import Edge, RCAData, ServiceNode, TraceData


def safe_convert_to_int(value):
    """
    安全地将各种类型的值转换为整数，特别处理datetime对象
    """
    if value is None:
        return 0

    if isinstance(value, datetime):
        # 将datetime转换为微秒级时间戳
        return int(value.timestamp() * 1_000_000)

    if isinstance(value, str):
        try:
            # 尝试解析为浮点数再转换为整数
            return int(float(value))
        except ValueError:
            return 0

    if isinstance(value, (int, float)):
        return int(value)

    return 0


def aggregate_to_service_level(operation_results: Dict[str, float]) -> Dict[str, float]:
    """聚合到service级别"""
    service_scores = {}
    for operation_name, score in operation_results.items():
        if ":" in operation_name:
            service_name = operation_name.split(":", 1)[0]
        else:
            service_name = operation_name

        if service_name not in service_scores:
            service_scores[service_name] = 0
        service_scores[service_name] += score
    return service_scores


def convert_polars_traces_to_rca_data(
    traces_lf: pl.LazyFrame,
    metrics_lf: Optional[pl.LazyFrame] = None,
    metrics_sli_lf: Optional[pl.LazyFrame] = None,
) -> RCAData:
    """
    将Polars LazyFrame的traces数据转换为原版算法需要的RCAData格式
    完整保留原始数据结构和逻辑，并正确分类正常和异常数据
    """
    # Collect the lazy frame to get actual data
    traces_df = traces_lf.collect()

    # Group by trace_id to get individual traces
    trace_groups = traces_df.group_by("trace_id", maintain_order=True)

    traces = []
    all_nodes = set()
    all_edges = []
    service_to_operations = {}
    trace_data_dict = {}
    ts_data_dict = {}

    # 为MicroRank等算法分类正常和异常traces
    normal_traces = {}
    abnormal_traces = {}

    for group_data in trace_groups:
        trace_id_val = group_data[0][0]  # Extract the trace_id value
        trace_spans_df = group_data[1]  # Get the spans DataFrame

        trace_id_str = str(trace_id_val)

        # 检查这个trace是否为异常trace（根据anomal标记）
        is_anomal_trace = any(
            row.get("anomal", 0) == 1 for row in trace_spans_df.iter_rows(named=True)
        )

        # Convert each span to the format expected by TraceData
        spans = []
        span_lookup = {}  # For building parent-child relationships

        for row in trace_spans_df.iter_rows(named=True):
            # 安全转换时间相关字段
            start_time = safe_convert_to_int(row.get("time", 0))
            duration = safe_convert_to_int(row.get("duration", 0))

            # 获取状态码信息（data_loader已经转换为enum值：0=Unset, 1=Ok, 2=Error）
            status_code = safe_convert_to_int(
                row.get("attr.status_code", 1)
            )  # 默认为Ok=1
            is_error = status_code == 2  # Error = 2

            span_id = row.get("span_id", "")
            parent_span_id = row.get("parent_span_id", "")
            service_name = row.get("service_name", "")
            operation_name = row.get("operation_name", row.get("span_name", ""))

            span = {
                "spanId": span_id,
                "traceId": trace_id_str,
                "parentSpanId": parent_span_id,
                "operationName": operation_name,
                "serviceName": service_name,
                "startTime": start_time,
                "duration": duration,
                "TimeStamp": start_time,  # 添加原始算法期望的字段
                "Duration": duration,  # 添加原始算法期望的字段
                "process": {"serviceName": service_name},
                "tags": [],
            }
            spans.append(span)
            span_lookup[span_id] = span

            # 收集服务和操作信息
            if service_name and operation_name:
                node_id = f"{service_name}:{operation_name}"
                all_nodes.add(node_id)

                if service_name not in service_to_operations:
                    service_to_operations[service_name] = set()
                service_to_operations[service_name].add(operation_name)

                # 构建trace_data_dict (用于一些baseline算法)
                if node_id not in trace_data_dict:
                    trace_data_dict[node_id] = {"Duration": [], "serverIp": []}
                trace_data_dict[node_id]["Duration"].append(duration)
                # 不添加"unknown"到serverIp列表中

                # 构建ts_data_dict (时间序列数据) - 添加QPS和EC支持
                if node_id not in ts_data_dict:
                    ts_data_dict[node_id] = {
                        "Duration": [],
                        "MaxDuration": [],
                        "QPS": [],  # 请求数量（每个span计为1个请求）
                        "EC": [],  # 错误数量
                    }
                ts_data_dict[node_id]["Duration"].append(duration)
                ts_data_dict[node_id]["MaxDuration"].append(duration)
                ts_data_dict[node_id]["QPS"].append(1)  # 每个span代表一个请求
                ts_data_dict[node_id]["EC"].append(1 if is_error else 0)  # 错误计数

        # 构建调用关系edges
        for span in spans:
            if span["parentSpanId"] and span["parentSpanId"] in span_lookup:
                parent_span = span_lookup[span["parentSpanId"]]
                parent_node_id = (
                    f"{parent_span['serviceName']}:{parent_span['operationName']}"
                )
                child_node_id = f"{span['serviceName']}:{span['operationName']}"

                if parent_node_id != child_node_id:  # 避免自循环
                    edge = Edge(source_id=parent_node_id, target_id=child_node_id)
                    if edge not in all_edges:
                        all_edges.append(edge)

        # Create TraceData object
        trace_data = TraceData(
            trace_id=trace_id_str,
            spans=spans,
            timestamp=spans[0]["startTime"] if spans else 0,
        )
        traces.append(trace_data)

        # 根据anomal标记分类trace
        if is_anomal_trace:
            abnormal_traces[trace_id_str] = spans
        else:
            normal_traces[trace_id_str] = spans

    # Create ServiceNode objects
    nodes = []
    for node_id in all_nodes:
        if ":" in node_id:
            service_name, operation = node_id.split(":", 1)
            nodes.append(
                ServiceNode(
                    node_id=node_id,
                    service_name=service_name,
                    server_ip="",  # 没有IP信息时使用空字符串
                )
            )

    # 确定root_id和root_id_list (支持多个root节点)
    root_id = ""
    root_id_list = []

    if traces and traces[0].spans:
        # 收集所有没有parent的span作为root节点
        for span in traces[0].spans:
            if not span.get("parentSpanId"):
                node_id = f"{span['serviceName']}:{span['operationName']}"
                if node_id not in root_id_list:
                    root_id_list.append(node_id)
                if not root_id:  # 设置第一个作为主要root_id
                    root_id = node_id

    # Create RCAData object with complete data

    # 使用metrics_sli.parquet文件获取真实的统计数据
    metrics_statistical_data = {}

    if metrics_sli_lf is not None:
        # 收集SLI数据
        sli_df = metrics_sli_lf.collect()

        # 按service_name和span_name分组
        sli_grouped = sli_df.group_by(
            ["service_name", "span_name"], maintain_order=True
        )

        for group_data in sli_grouped:
            service_name = group_data[0][0]
            span_name = group_data[0][1]
            group_df = group_data[1]

            node_id = f"{service_name}:{span_name}"

            # 提取统计数据用于加权计算
            duration_data = []  # (avg_duration, total_count) pairs
            qps_values = []
            error_data = []  # (error_count, total_count) pairs

            for row in group_df.iter_rows(named=True):
                total_count = row.get("total_count", 0)
                if total_count <= 0:  # 跳过没有请求的时间窗口
                    continue

                # Duration统计 - 使用avg_duration按total_count加权
                avg_duration = row.get("avg_duration", 0)
                if avg_duration > 0:
                    duration_data.append((avg_duration, total_count))

                # QPS统计 - 假设每个时间窗口是1分钟
                qps = total_count / 60.0
                qps_values.append(qps)

                # Error Count统计 - 按total_count加权
                error_count = row.get("error_count", 0)
                error_data.append((error_count, total_count))

            if duration_data:
                import statistics

                # 计算加权平均Duration
                total_requests = sum(count for _, count in duration_data)
                weighted_mean_duration = (
                    sum(duration * count for duration, count in duration_data)
                    / total_requests
                )

                # 计算加权标准差
                weighted_variance = (
                    sum(
                        count * (duration - weighted_mean_duration) ** 2
                        for duration, count in duration_data
                    )
                    / total_requests
                )
                weighted_std_duration = (
                    weighted_variance**0.5
                    if weighted_variance > 0
                    else weighted_mean_duration * 0.1
                )

                # QPS统计 - 简单平均即可（因为都是基于相同时间窗口）
                mean_qps = statistics.mean(qps_values) if qps_values else 1.0
                std_qps = (
                    statistics.stdev(qps_values)
                    if len(qps_values) > 1
                    else mean_qps * 0.1
                )

                # Error Count统计 - 计算总体错误率
                total_errors = sum(errors for errors, _ in error_data)
                total_all_requests = sum(count for _, count in error_data)
                error_rate = (
                    total_errors / total_all_requests if total_all_requests > 0 else 0.0
                )

                # 错误率的标准差 (使用二项分布公式)
                error_std = (
                    (error_rate * (1 - error_rate) / len(error_data)) ** 0.5
                    if len(error_data) > 0
                    else 0.1
                )

                duration_count = len(duration_data)

                metrics_statistical_data[node_id] = {
                    "Duration": [
                        weighted_mean_duration,
                        weighted_std_duration,
                        duration_count,
                    ],
                    "MaxDuration": [
                        weighted_mean_duration,
                        weighted_std_duration,
                        duration_count,
                    ],
                    "QPS": [mean_qps, std_qps, duration_count],
                    "EC": [error_rate, error_std, duration_count],
                }
    else:
        # 如果没有SLI数据，回退到从正常traces计算统计数据
        # 从正常traces计算统计数据，需要包含所有metrics (Duration, MaxDuration, QPS, EC)
        normal_ts_data_dict = {}

        # 重新处理正常traces数据以计算完整统计
        for trace_id_str, spans in normal_traces.items():
            for span in spans:
                service_name = span.get("serviceName", "")
                operation_name = span.get("operationName", "")

                if service_name and operation_name:
                    node_id = f"{service_name}:{operation_name}"
                    duration = safe_convert_to_int(
                        span.get("duration", span.get("Duration", 0))
                    )

                    # 对于正常数据，我们假设大部分请求都是成功的
                    is_error = False  # 正常数据中假设错误率很低

                    if node_id not in normal_ts_data_dict:
                        normal_ts_data_dict[node_id] = {
                            "Duration": [],
                            "MaxDuration": [],
                            "QPS": [],
                            "EC": [],
                        }
                    normal_ts_data_dict[node_id]["Duration"].append(duration)
                    normal_ts_data_dict[node_id]["MaxDuration"].append(duration)
                    normal_ts_data_dict[node_id]["QPS"].append(
                        1
                    )  # 每个span代表一个请求
                    normal_ts_data_dict[node_id]["EC"].append(1 if is_error else 0)

        # 从正常数据计算统计信息
        for node_id, ts_data in normal_ts_data_dict.items():
            durations = ts_data.get("Duration", [])
            qps_data = ts_data.get("QPS", [])
            ec_data = ts_data.get("EC", [])

            if durations:
                import statistics

                # Duration统计
                mean_duration = statistics.mean(durations)
                std_duration = statistics.stdev(durations) if len(durations) > 1 else 0
                count = len(durations)

                # QPS统计（请求数量 - 正常情况下每个span都是一个请求）
                mean_qps = statistics.mean(qps_data) if qps_data else 1.0
                std_qps = statistics.stdev(qps_data) if len(qps_data) > 1 else 0.1

                # EC统计（错误数量 - 正常数据中应该很少有错误）
                mean_ec = statistics.mean(ec_data) if ec_data else 0.0
                std_ec = statistics.stdev(ec_data) if len(ec_data) > 1 else 0.1

                metrics_statistical_data[node_id] = {
                    "Duration": [mean_duration, std_duration, count],
                    "MaxDuration": [mean_duration, std_duration, count],
                    "QPS": [mean_qps, std_qps, count],
                    "EC": [mean_ec, std_ec, count],
                }

    # 处理metrics数据（为TON和MicroRCA算法）
    ip_ts_data_dict = {}
    if metrics_lf is not None:
        # 收集metrics数据并按service_name分组
        # 真实数据是行式格式：每行包含metric名称、value和service_name
        metrics_df = metrics_lf.collect()

        # 过滤出我们需要的machine metrics
        machine_metrics_df = metrics_df.filter(
            pl.col("metric").is_in(
                ["k8s.pod.cpu_limit_utilization", "k8s.pod.memory_limit_utilization"]
            )
        )

        # 创建service_name到metrics数据的映射
        service_metrics = {}

        for row in machine_metrics_df.iter_rows(named=True):
            service_name = row.get("service_name", "")
            if not service_name:  # 跳过空的service_name
                continue

            # 获取时间戳和metric信息
            timestamp = row.get("time", 0)
            if hasattr(timestamp, "timestamp"):  # 如果是datetime对象
                timestamp = timestamp.timestamp()
            elif timestamp == 0:
                timestamp = 0

            metric_name = row.get("metric", "")
            metric_value = float(row.get("value", 0.0))

            if service_name not in service_metrics:
                service_metrics[service_name] = {
                    "k8s.pod.cpu_limit_utilization": [],
                    "k8s.pod.memory_limit_utilization": [],
                    "timestamps": [],
                }

            # 将数据添加到对应的metric列表中
            if metric_name in service_metrics[service_name]:
                service_metrics[service_name][metric_name].append(metric_value)
                service_metrics[service_name]["timestamps"].append(timestamp)

        # 转换为ip_ts_data_dict格式（使用service_name作为key）
        # 这样TON算法可以通过service_name获取对应的machine metrics
        ip_ts_data_dict = service_metrics

    rca_data = RCAData(
        edges=all_edges,
        nodes=nodes,
        node_ids=list(all_nodes),
        root_causes=[],  # 由算法确定
        root_id=root_id,
        root_id_list=root_id_list,  # 支持多个root节点
        traces=traces,
        trace_data_dict=trace_data_dict,
        request_timestamp=traces[0].timestamp if traces else 0,
        ts_data_dict=ts_data_dict,
        metrics_statistical_data=metrics_statistical_data,
        metrics_threshold={},
        ip_ts_data_dict=ip_ts_data_dict,  # 添加metrics数据支持
        # 为MicroRank等算法添加正常和异常traces分类
        normal_trace_dict=normal_traces,
        abnormal_trace_dict=abnormal_traces,
        trace_dict={trace.trace_id: trace.spans for trace in traces},
    )

    return rca_data


class ShapleyRCAAdapter:
    """
    ShapleyValueRCA适配器
    """

    def __init__(self, using_cache: bool = False, sync_overlap_threshold: float = 0.05):
        self.using_cache = using_cache
        self.sync_overlap_threshold = sync_overlap_threshold
        self.algorithm = ShapleyValueRCA(
            using_cache=using_cache, sync_overlap_threshold=sync_overlap_threshold
        )

    def run(
        self,
        traces_lf: pl.LazyFrame,
        metrics_sli_lf: Optional[pl.LazyFrame] = None,
        initial_anomalous_node: Optional[str] = None,
        anomalous_services: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """运行算法并返回service级别的结果

        Args:
            traces_lf: trace数据
            metrics_sli_lf: SLI统计数据（可选）
            initial_anomalous_node: 初始异常节点（ShapleyValueRCA不使用此参数）
            anomalous_services: 异常服务列表（ShapleyValueRCA不使用此参数）
        """
        # 转换数据格式
        rca_data = convert_polars_traces_to_rca_data(
            traces_lf, metrics_sli_lf=metrics_sli_lf
        )

        # 使用原版算法运行分析 - ShapleyValueRCA使用全局分析，不依赖初始节点
        results = self.algorithm.analyze(
            rca_data, strategy="avg_by_contribution", sort_result=True
        )

        # 转换结果为service级别
        if isinstance(results, dict):
            return aggregate_to_service_level(results)
        return {}


class MicroHECLAdapter:
    """
    MicroHECL适配器
    """

    def __init__(self, time_window: int = 15):
        self.time_window = time_window
        self.algorithm = MicroHECL(time_window=time_window)

    def run(
        self,
        traces_lf: pl.LazyFrame,
        metrics_sli_lf: Optional[pl.LazyFrame] = None,
        initial_anomalous_node: Optional[str] = None,
        anomalous_services: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """运行算法并返回service级别的结果

        Args:
            traces_lf: trace数据
            metrics_sli_lf: SLI统计数据（可选）
            initial_anomalous_node: 初始异常节点（可选）
            anomalous_services: 异常服务列表（MicroHECL只使用第一个）
        """
        # 转换数据格式
        rca_data = convert_polars_traces_to_rca_data(
            traces_lf, metrics_sli_lf=metrics_sli_lf
        )

        # MicroHECL只使用第一个异常服务，如果有的话
        if not initial_anomalous_node and anomalous_services:
            # 在trace数据中查找匹配的节点
            for node_id in rca_data.node_ids:
                if anomalous_services[0] in node_id:
                    initial_anomalous_node = node_id
                    break

        # 使用传入的初始异常节点，如果没有提供则使用默认逻辑
        if not initial_anomalous_node:
            initial_anomalous_node = rca_data.root_id if rca_data.root_id else None
            if not initial_anomalous_node and rca_data.node_ids:
                initial_anomalous_node = rca_data.node_ids[0]

        # 使用原版算法运行分析
        results = self.algorithm.analyze(
            rca_data,
            initial_anomalous_node=initial_anomalous_node,
            detect_metrics=["RT", "EC", "QPS"],  # 使用完整的metric列表
        )

        # 转换结果为service级别
        if isinstance(results, dict):
            return aggregate_to_service_level(results)
        return {}


class MicroRCAAdapter:
    """
    MicroRCA适配器
    """

    def __init__(self, time_window: int = 15):
        self.time_window = time_window
        self.algorithm = MicroRCA(time_window=time_window)

    def run(
        self,
        traces_lf: pl.LazyFrame,
        metrics_lf: Optional[pl.LazyFrame] = None,
        metrics_sli_lf: Optional[pl.LazyFrame] = None,
        initial_anomalous_node: Optional[str] = None,
        anomalous_services: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """运行算法并返回service级别的结果

        Args:
            traces_lf: trace数据
            metrics_lf: metrics数据
            metrics_sli_lf: SLI统计数据
            initial_anomalous_node: 初始异常节点（MicroRCA可以使用单个节点）
            anomalous_services: 异常服务列表（MicroRCA优先使用此参数）
        """
        # 转换数据格式
        rca_data = convert_polars_traces_to_rca_data(
            traces_lf, metrics_lf, metrics_sli_lf
        )

        # 使用原版算法运行分析
        results = self.algorithm.analyze(
            rca_data,
            initial_anomalous_node=initial_anomalous_node,
            anomalous_services=anomalous_services,
        )

        # 转换结果为service级别
        if isinstance(results, dict):
            return aggregate_to_service_level(results)
        return {}


class TONAdapter:
    """
    TON适配器
    """

    def __init__(self, time_window: int = 15):
        self.time_window = time_window
        self.algorithm = TON(time_window=time_window)

    def run(
        self,
        traces_lf: pl.LazyFrame,
        metrics_lf: Optional[pl.LazyFrame] = None,
        metrics_sli_lf: Optional[pl.LazyFrame] = None,
        initial_anomalous_node: Optional[str] = None,
        anomalous_services: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """运行算法并返回service级别的结果

        Args:
            traces_lf: trace数据
            metrics_lf: metrics数据
            metrics_sli_lf: SLI统计数据
            initial_anomalous_node: 初始异常节点（TON可以使用此参数）
            anomalous_services: 异常服务列表（TON优先使用此参数）
        """
        # 转换数据格式
        rca_data = convert_polars_traces_to_rca_data(
            traces_lf, metrics_lf, metrics_sli_lf
        )

        # 使用原版算法运行分析 - TON支持initial_anomalous_node和anomalous_services
        results = self.algorithm.analyze(
            rca_data,
            operation_only=True,
            initial_anomalous_node=initial_anomalous_node,
            anomalous_services=anomalous_services,
        )

        # 转换结果为service级别
        if isinstance(results, dict):
            return aggregate_to_service_level(results)
        return {}


class MicroRankAdapter:
    """
    MicroRank适配器
    """

    def __init__(self, n_sigma: int = 3):
        self.n_sigma = n_sigma
        self.algorithm = MicroRank(n_sigma=n_sigma)

    def run(
        self,
        traces_lf: pl.LazyFrame,
        metrics_sli_lf: Optional[pl.LazyFrame] = None,
        initial_anomalous_node: Optional[str] = None,
        anomalous_services: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """运行算法并返回service级别的结果

        Args:
            traces_lf: trace数据
            metrics_sli_lf: SLI统计数据（可选）
            initial_anomalous_node: 初始异常节点（MicroRank可以使用此参数）
            anomalous_services: 异常服务列表（MicroRank使用第一个，如果有的话）
        """
        # 转换数据格式
        rca_data = convert_polars_traces_to_rca_data(
            traces_lf, metrics_sli_lf=metrics_sli_lf
        )

        # MicroRank也可以使用第一个异常服务
        if not initial_anomalous_node and anomalous_services:
            # 在trace数据中查找匹配的节点
            for node_id in rca_data.node_ids:
                if anomalous_services[0] in node_id:
                    initial_anomalous_node = node_id
                    break

        # 使用原版算法运行分析 - MicroRank支持initial_anomalous_node
        results = self.algorithm.analyze(
            rca_data,
            phi=0.5,
            omega=0.01,
            d=0.04,
            initial_anomalous_node=initial_anomalous_node,
        )

        # 转换结果为service级别
        if isinstance(results, dict):
            return aggregate_to_service_level(results)
        return {}
