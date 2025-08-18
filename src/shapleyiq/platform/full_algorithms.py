"""
完整的ShapleyIQ算法适配器
使用原版算法逻辑，完整移植到新的平台接口
"""

from functools import partial
from typing import List, Dict, Optional
import numpy as np
import polars as pl
from datetime import datetime

from .interface import Algorithm, AlgorithmArgs, AlgorithmAnswer
from ..data_structures import RCAData, TraceData, ServiceNode, Edge
from ..algorithms.shapley_value_rca import ShapleyValueRCA as OriginalShapleyRCA
from ..algorithms.microhecl import MicroHECL as OriginalMicroHECL
from ..algorithms.microrca import MicroRCA as OriginalMicroRCA
from ..algorithms.microrank import MicroRank as OriginalMicroRank
from ..algorithms.ton import TON as OriginalTON


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


def convert_polars_traces_to_rca_data(traces_lf: pl.LazyFrame) -> RCAData:
    """
    将Polars LazyFrame的traces数据转换为原版算法需要的RCAData格式
    完整保留原始数据结构和逻辑
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
    
    for group_data in trace_groups:
        trace_id_val = group_data[0][0]  # Extract the trace_id value
        trace_spans_df = group_data[1]    # Get the spans DataFrame
        
        trace_id_str = str(trace_id_val)
        
        # Convert each span to the format expected by TraceData
        spans = []
        span_lookup = {}  # For building parent-child relationships
        
        for row in trace_spans_df.iter_rows(named=True):
            # 安全转换时间相关字段
            start_time = safe_convert_to_int(row.get('time', 0))
            duration = safe_convert_to_int(row.get('duration', 0))
            
            span_id = row.get('span_id', '')
            parent_span_id = row.get('parent_span_id', '')
            service_name = row.get('service_name', '')
            operation_name = row.get('operation_name', row.get('span_name', ''))
            
            span = {
                'spanId': span_id,
                'traceId': trace_id_str,
                'parentSpanId': parent_span_id,
                'operationName': operation_name,
                'serviceName': service_name,
                'startTime': start_time,
                'duration': duration,
                'process': {
                    'serviceName': service_name
                },
                'tags': []
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
                    trace_data_dict[node_id] = {
                        'Duration': [],
                        'serverIp': []
                    }
                trace_data_dict[node_id]['Duration'].append(duration)
                trace_data_dict[node_id]['serverIp'].append('unknown')
                
                # 构建ts_data_dict (时间序列数据)
                if node_id not in ts_data_dict:
                    ts_data_dict[node_id] = {
                        'Duration': [],
                        'MaxDuration': []
                    }
                ts_data_dict[node_id]['Duration'].append(duration)
                ts_data_dict[node_id]['MaxDuration'].append(duration)
        
        # 构建调用关系edges
        for span in spans:
            if span['parentSpanId'] and span['parentSpanId'] in span_lookup:
                parent_span = span_lookup[span['parentSpanId']]
                parent_node_id = f"{parent_span['serviceName']}:{parent_span['operationName']}"
                child_node_id = f"{span['serviceName']}:{span['operationName']}"
                
                if parent_node_id != child_node_id:  # 避免自循环
                    edge = Edge(source_id=parent_node_id, target_id=child_node_id)
                    if edge not in all_edges:
                        all_edges.append(edge)
        
        # Create TraceData object
        trace_data = TraceData(
            trace_id=trace_id_str,
            spans=spans,
            timestamp=spans[0]['startTime'] if spans else 0
        )
        traces.append(trace_data)
    
    # Create ServiceNode objects
    nodes = []
    for node_id in all_nodes:
        if ':' in node_id:
            service_name, operation = node_id.split(':', 1)
            nodes.append(ServiceNode(
                node_id=node_id,
                service_name=service_name,
                server_ip="unknown"  # 新格式中不可用
            ))
    
    # 确定root_id (通常是第一个没有parent的span)
    root_id = ""
    if traces and traces[0].spans:
        for span in traces[0].spans:
            if not span.get('parentSpanId'):
                root_id = f"{span['serviceName']}:{span['operationName']}"
                break
    
    # Create RCAData object with complete data
    rca_data = RCAData(
        edges=all_edges,
        nodes=nodes,
        node_ids=list(all_nodes),
        root_causes=[],  # 由算法确定
        root_id=root_id,
        traces=traces,
        trace_data_dict=trace_data_dict,
        request_timestamp=traces[0].timestamp if traces else 0,
        ts_data_dict=ts_data_dict,
        metrics_statistical_data={},
        metrics_threshold={},
        # 为MicroRank等算法添加额外的trace_dict
        trace_dict={trace.trace_id: trace.spans for trace in traces}
    )
    
    return rca_data


class ShapleyRCA(Algorithm):
    """
    完整的ShapleyValueRCA实现，使用原版算法逻辑
    """
    
    def __init__(self, using_cache: bool = False, sync_overlap_threshold: float = 0.05):
        self.using_cache = using_cache
        self.sync_overlap_threshold = sync_overlap_threshold
        self.algorithm = OriginalShapleyRCA(
            using_cache=using_cache,
            sync_overlap_threshold=sync_overlap_threshold
        )
    
    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        if args.traces is None:
            return [AlgorithmAnswer(ranks=[])]
        
        try:
            # 转换数据格式
            rca_data = convert_polars_traces_to_rca_data(args.traces)
            
            # 使用原版算法运行分析
            results = self.algorithm.run(
                rca_data, 
                strategy="avg_by_contribution", 
                sort_result=True
            )
            
            # 转换结果格式
            ranks = []
            scores = {}
            if isinstance(results, dict):
                # 按分数排序
                sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
                ranks = [node_id for node_id, score in sorted_results]
                scores = {node_id: float(score) for node_id, score in sorted_results}
            
            return [AlgorithmAnswer(
                ranks=ranks,
                scores=scores,
                node_names=list(rca_data.node_ids) if rca_data else [],
                metadata={'algorithm': 'ShapleyRCA', 'strategy': 'avg_by_contribution'}
            )]
            
        except Exception as e:
            print(f"ShapleyRCA error: {e}")
            import traceback
            traceback.print_exc()
            return [AlgorithmAnswer(ranks=[])]


class MicroHECL(Algorithm):
    """
    完整的MicroHECL实现，使用原版算法逻辑
    """
    
    def __init__(self, time_window: int = 15):
        self.time_window = time_window
        self.algorithm = OriginalMicroHECL(time_window=time_window)
    
    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        if args.traces is None:
            return [AlgorithmAnswer(ranks=[])]
        
        try:
            # 转换数据格式
            rca_data = convert_polars_traces_to_rca_data(args.traces)
            
            # 确定初始异常节点
            initial_anomalous_node = rca_data.root_id if rca_data.root_id else None
            if not initial_anomalous_node and rca_data.node_ids:
                initial_anomalous_node = rca_data.node_ids[0]
            
            # 使用原版算法运行分析
            results = self.algorithm.run(
                rca_data,
                initial_anomalous_node=initial_anomalous_node,
                detect_metrics=["RT"]
            )
            
            # 转换结果格式
            ranks = []
            scores = {}
            if isinstance(results, dict):
                # 按分数排序
                sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
                ranks = [node_id for node_id, score in sorted_results]
                scores = {node_id: float(score) for node_id, score in sorted_results}
            
            return [AlgorithmAnswer(
                ranks=ranks,
                scores=scores,
                node_names=list(rca_data.node_ids) if rca_data else [],
                metadata={'algorithm': 'MicroHECL', 'time_window': self.time_window}
            )]
            
        except Exception as e:
            print(f"MicroHECL error: {e}")
            import traceback
            traceback.print_exc()
            return [AlgorithmAnswer(ranks=[])]


class MicroRCA(Algorithm):
    """
    完整的MicroRCA实现，使用原版算法逻辑
    """
    
    def __init__(self, time_window: int = 15):
        self.time_window = time_window
        self.algorithm = OriginalMicroRCA(time_window=time_window)
    
    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        if args.traces is None:
            return [AlgorithmAnswer(ranks=[])]
        
        try:
            # 转换数据格式
            rca_data = convert_polars_traces_to_rca_data(args.traces)
            
            # 使用原版算法运行分析
            results = self.algorithm.run(rca_data)
            
            # 转换结果格式
            ranks = []
            scores = {}
            if isinstance(results, dict):
                # 按分数排序
                sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
                ranks = [node_id for node_id, score in sorted_results]
                scores = {node_id: float(score) for node_id, score in sorted_results}
            
            return [AlgorithmAnswer(
                ranks=ranks,
                scores=scores,
                node_names=list(rca_data.node_ids) if rca_data else [],
                metadata={'algorithm': 'MicroRCA', 'time_window': self.time_window}
            )]
            
        except Exception as e:
            print(f"MicroRCA error: {e}")
            import traceback
            traceback.print_exc()
            return [AlgorithmAnswer(ranks=[])]


class TON(Algorithm):
    """
    完整的TON实现，使用原版算法逻辑
    """
    
    def __init__(self, time_window: int = 15):
        self.time_window = time_window
        self.algorithm = OriginalTON(time_window=time_window)
    
    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        if args.traces is None:
            return [AlgorithmAnswer(ranks=[])]
        
        try:
            # 转换数据格式
            rca_data = convert_polars_traces_to_rca_data(args.traces)
            
            # 使用原版算法运行分析
            results = self.algorithm.run(rca_data, operation_only=True)
            
            # 转换结果格式
            ranks = []
            scores = {}
            if isinstance(results, dict):
                # 按分数排序
                sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
                ranks = [node_id for node_id, score in sorted_results]
                scores = {node_id: float(score) for node_id, score in sorted_results}
            
            return [AlgorithmAnswer(
                ranks=ranks,
                scores=scores,
                node_names=list(rca_data.node_ids) if rca_data else [],
                metadata={'algorithm': 'TON', 'time_window': self.time_window, 'operation_only': True}
            )]
            
        except Exception as e:
            print(f"TON error: {e}")
            import traceback
            traceback.print_exc()
            return [AlgorithmAnswer(ranks=[])]


class MicroRank(Algorithm):
    """
    完整的MicroRank实现，使用原版算法逻辑
    """
    
    def __init__(self, n_sigma: int = 3):
        self.n_sigma = n_sigma
        self.algorithm = OriginalMicroRank(n_sigma=n_sigma)
    
    def __call__(self, args: AlgorithmArgs) -> List[AlgorithmAnswer]:
        if args.traces is None:
            return [AlgorithmAnswer(ranks=[])]
        
        try:
            # 转换数据格式
            rca_data = convert_polars_traces_to_rca_data(args.traces)
            
            # 使用原版算法运行分析
            results = self.algorithm.run(rca_data, phi=0.5, omega=0.01, d=0.04)
            
            # 转换结果格式
            ranks = []
            scores = {}
            if isinstance(results, dict):
                # 按分数排序
                sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
                ranks = [node_id for node_id, score in sorted_results]
                scores = {node_id: float(score) for node_id, score in sorted_results}
            
            return [AlgorithmAnswer(
                ranks=ranks,
                scores=scores,
                node_names=list(rca_data.node_ids) if rca_data else [],
                metadata={'algorithm': 'MicroRank', 'n_sigma': self.n_sigma}
            )]
            
        except Exception as e:
            print(f"MicroRank error: {e}")
            import traceback
            traceback.print_exc()
            return [AlgorithmAnswer(ranks=[])]
