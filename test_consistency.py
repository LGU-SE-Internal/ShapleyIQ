"""
算法一致性测试
比较新平台实现与原版算法的结果一致性
"""

import sys
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from shapleyiq.platform.data_loader import load_traces
from shapleyiq.platform.algorithms import ShapleyRCA as SimpleShapleyRCA
from shapleyiq.platform.real_algorithms import RealShapleyRCA
from shapleyiq.platform.interface import AlgorithmArgs


def test_algorithm_consistency():
    """测试算法实现的一致性"""
    print("=" * 60)
    print("算法一致性测试")
    print("=" * 60)
    
    # 加载测试数据
    data_path = Path("test/ts1-ts-route-plan-service-request-replace-method-qtbhzt")
    
    if not data_path.exists():
        print(f"错误: 测试数据目录 {data_path} 不存在")
        return False
    
    print("加载测试数据...")
    traces_lf = load_traces(data_path)
    
    # 准备算法参数
    args = AlgorithmArgs(
        input_folder=data_path,
        traces=traces_lf,
        metrics=None,
        logs=None
    )
    
    print("\n比较ShapleyRCA实现:")
    print("-" * 40)
    
    # 测试简化版算法
    print("运行简化版ShapleyRCA...")
    try:
        simple_algo = SimpleShapleyRCA()
        simple_results = simple_algo(args)
        
        if simple_results:
            simple_result = simple_results[0]
            print(f"简化版前5个结果: {simple_result.ranks[:5]}")
            if simple_result.scores:
                simple_scores = {name: simple_result.scores[name] for name in simple_result.ranks[:5] if name in simple_result.scores}
                print(f"简化版分数: {simple_scores}")
        else:
            print("简化版无结果")
            simple_result = None
    except Exception as e:
        print(f"简化版失败: {e}")
        simple_result = None
    
    # 测试原版算法
    print("\n运行原版ShapleyRCA...")
    try:
        real_algo = RealShapleyRCA()
        real_results = real_algo(args)
        
        if real_results:
            real_result = real_results[0]
            print(f"原版前5个结果: {real_result.ranks[:5]}")
            if real_result.scores:
                real_scores = {name: real_result.scores[name] for name in real_result.ranks[:5] if name in real_result.scores}
                print(f"原版分数: {real_scores}")
        else:
            print("原版无结果")
            real_result = None
    except Exception as e:
        print(f"原版失败: {e}")
        import traceback
        traceback.print_exc()
        real_result = None
    
    # 比较结果
    print("\n结果比较:")
    print("-" * 40)
    
    if simple_result and real_result:
        # 比较前5个排名
        simple_top5 = set(simple_result.ranks[:5])
        real_top5 = set(real_result.ranks[:5])
        
        intersection = simple_top5.intersection(real_top5)
        overlap_ratio = len(intersection) / 5 if 5 > 0 else 0
        
        print(f"前5个结果的重叠度: {overlap_ratio:.2%}")
        print(f"共同的节点: {intersection}")
        print(f"简化版独有: {simple_top5 - real_top5}")
        print(f"原版独有: {real_top5 - simple_top5}")
        
        if overlap_ratio > 0.6:
            print("✓ 算法实现基本一致")
            return True
        else:
            print("✗ 算法实现差异较大")
            return False
    else:
        print("✗ 无法比较 - 某个算法失败")
        return False


def show_detailed_comparison():
    """显示详细的算法逻辑比较"""
    print("\n" + "=" * 60)
    print("算法逻辑差异分析")
    print("=" * 60)
    
    print("\n原版ShapleyValueRCA的核心步骤:")
    print("1. trace_to_timelines() - 将trace转换为时间线")
    print("2. trace_to_calling_tree() - 构建调用树")
    print("3. split_timelines() - 分割调用者时间线")
    print("4. merge_timelines() - 合并同步时间线")
    print("5. shapley_value_for_timelines() - 计算Shapley值")
    print("6. distribute_contribution_to_nodes() - 将贡献分配到节点")
    
    print("\n简化版实现的步骤:")
    print("1. 简单统计操作名和持续时间")
    print("2. 基于频次和平均持续时间计算分数")
    print("3. 归一化为概率")
    
    print("\n主要差异:")
    print("- 原版使用复杂的Shapley值计算，考虑时间线重叠和调用关系")
    print("- 简化版只做基本的统计分析，不考虑调用依赖")
    print("- 原版能更准确地识别真正的根因，简化版可能有偏差")
    
    print("\n建议:")
    print("1. 使用 real_algorithms.py 中的实现替代 algorithms.py")
    print("2. 确保数据转换正确，保持算法语义一致性")
    print("3. 测试和验证新平台实现的正确性")


def main():
    """主函数"""
    # 测试算法一致性
    is_consistent = test_algorithm_consistency()
    
    # 显示详细比较
    show_detailed_comparison()
    
    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    
    if is_consistent:
        print("✓ 当前实现与原版算法基本一致")
    else:
        print("✗ 当前实现与原版算法不一致")
        print("\n问题:")
        print("- 新平台的 algorithms.py 只是简化的统计实现")
        print("- 没有使用原版的复杂Shapley值计算逻辑")
        print("- 可能导致根因分析准确性下降")
        
        print("\n解决方案:")
        print("- 使用 real_algorithms.py 中的实现")
        print("- 这些实现正确调用了原版算法逻辑")
        print("- 但需要修复数据转换的类型问题")
    
    return is_consistent


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
