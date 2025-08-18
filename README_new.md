# ShapleyIQ

一个基于Shapley值的微服务性能调试和根因分析框架。

## 概述

ShapleyIQ是一个现代化的Python包，用于微服务系统的根因分析。它实现了基于Shapley值的影响量化方法，以及多种基线算法，为微服务性能调试提供全面的解决方案。

## 特性

- **🎯 精确分析**: 基于博弈论的Shapley值方法，提供精确的根因定位
- **📊 多算法支持**: 集成ShapleyValueRCA、MicroHECL、MicroRCA、MicroRank、TON等算法
- **🔄 数据预处理**: 支持Jaeger、Zipkin、DbaAS等多种追踪数据格式
- **⚡ 高性能**: 优化的算法实现，支持大规模微服务系统
- **🛠️ 易于使用**: 提供CLI和Python API两种使用方式
- **📈 可视化**: 丰富的结果展示和评估指标

## 安装

### 要求

- Python 3.13+
- 推荐使用虚拟环境

### 从源码安装

```bash
git clone <repository-url>
cd ShapleyIQ
pip install -e .
```

## 快速开始

### 1. 验证安装

```bash
python validate_setup.py
```

### 2. 使用命令行工具

```bash
# 分析单个追踪文件
python -m shapleyiq analyze trace_file.json \
    --algorithm shapley \
    --root-causes service1 \
    --format jaeger

# 运行演示
python -m shapleyiq demo --dataset example
```

### 3. 使用Python API

```python
from shapleyiq import ShapleyValueRCA
from shapleyiq.preprocessing import RCADataBuilder

# 构建数据
builder = RCADataBuilder()
rca_data = builder.build_from_files(
    trace_file="path/to/traces.json",
    root_causes=["faulty-service"],
    trace_format="jaeger"
)

# 运行分析
algorithm = ShapleyValueRCA()
results = algorithm.analyze(rca_data)

print("根因分析结果:")
for service, score in list(results.items())[:5]:
    print(f"{service}: {score:.4f}")
```

### 4. 使用真实数据

如果您有TrainTicket的追踪数据：

```bash
# 分析TrainTicket数据
python demo_real_data.py

# 或使用CLI
python -m shapleyiq analyze \
    rca4tracing/fault_injection/data/traces/ts-basic-service100_users5_spawn_rate5.json \
    --algorithm shapley \
    --root-causes ts-basic-service \
    --format jaeger
```

## 项目结构

```
ShapleyIQ/
├── src/shapleyiq/
│   ├── __init__.py                 # 主包入口
│   ├── cli.py                      # 命令行界面
│   ├── data_structures/            # 数据结构定义
│   │   └── __init__.py
│   ├── preprocessing/              # 数据预处理
│   │   └── __init__.py
│   ├── utils/                      # 工具函数
│   │   └── __init__.py
│   └── algorithms/                 # 算法实现
│       ├── __init__.py
│       ├── base.py                 # 基础算法类
│       ├── shapley_value_rca.py    # 主算法
│       ├── microhecl.py            # MicroHECL基线
│       ├── microrca.py             # MicroRCA基线
│       ├── microrank.py            # MicroRank基线
│       └── ton.py                  # TON基线
├── tests/                          # 测试文件
├── demo_real_data.py               # 真实数据演示
├── quick_start.py                  # 快速开始指南
├── validate_setup.py               # 安装验证
└── pyproject.toml                  # 项目配置
```

## 算法说明

### ShapleyValueRCA (主算法)
基于博弈论中的Shapley值，量化每个微服务对系统性能问题的贡献度。

### 基线算法
- **MicroHECL**: 基于异常传播的分层根因分析
- **MicroRCA**: 基于个性化PageRank的随机游走分析
- **MicroRank**: 结合频谱故障定位和PageRank的混合方法
- **TON**: 基于拓扑的异常检测网络

## 数据格式

支持以下追踪数据格式：

### Jaeger格式
```json
{
  "data": [
    {
      "traceID": "trace-001",
      "spans": [
        {
          "spanID": "span-1",
          "operationName": "GET /api",
          "process": {"serviceName": "frontend"},
          "startTime": 1609459200000000,
          "duration": 50000
        }
      ]
    }
  ]
}
```

### 通用格式
```json
[
  {
    "trace_id": "trace-001",
    "spans": [
      {
        "spanId": "span-1",
        "service_name": "frontend",
        "operation_name": "GET /api",
        "start_time": 1609459200000000,
        "duration": 50000
      }
    ]
  }
]
```

## 开发

### 运行测试

```bash
python -m pytest tests/ -v
```

### 代码格式化

```bash
ruff format src/
ruff check src/
```

## 引用

如果您在研究中使用了ShapleyIQ，请引用相关论文：

```bibtex
@article{shapleyiq2024,
  title={ShapleyIQ: Influence Quantification by Shapley Values for Performance Debugging of Microservices},
  author={...},
  journal={...},
  year={2024}
}
```

## 许可证

[此处添加许可证信息]

## 贡献

欢迎贡献代码！请参考贡献指南。

## 支持

如有问题或建议，请提交issue。
