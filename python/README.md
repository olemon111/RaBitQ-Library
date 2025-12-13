# QG Python Package

Python wheel package for RaBitQ QuantizedGraph.

## 构建和安装

### 从源码构建 wheel 包

```bash
cd python
pip install build
python -m build
```

这会在 `dist/` 目录下生成 `.whl` 文件。

### 安装 wheel 包

```bash
pip install dist/qg-0.1.0-*.whl
```

### 开发模式安装

```bash
cd python
pip install -e .
```

## 使用示例

```python
import numpy as np
from qg import QG

# 创建 QG 实例
qg = QG()

# 加载索引
qg.load_index("output/qg_metadata.bin", "output/qg_data.bin")

# 设置 ef 参数
qg.set_ef(100)

# 单个查询 (对应 load_test_qg.cpp 中的 search_one)
query = np.random.rand(128).astype(np.float32)
results = qg.search_one(query, k=10)
print(f"Found neighbors: {results}")

# 批量查询 (对应 load_test_qg.cpp 中的 search_all)
queries = np.random.rand(100, 128).astype(np.float32)
all_results = qg.search_all(queries, k=10)
print(f"Results shape: {all_results.shape}")  # (100, 10)
```

## API 文档

### QG 类

#### `load_index(meta_filename: str, data_filename: str)`

加载索引文件。

- `meta_filename`: 元数据文件路径（如 `qg_metadata.bin`）
- `data_filename`: 数据文件路径（如 `qg_data.bin`）

#### `set_ef(ef: int)`

设置 ef（探索因子）参数，控制搜索质量和速度的权衡。

#### `search_one(query: np.ndarray, k: int) -> np.ndarray`

对单个查询向量进行搜索，返回 k 个最近邻的 ID。

- `query`: 查询向量（1D numpy 数组，float32）
- `k`: 返回的最近邻数量
- 返回: k 个最近邻 ID 的数组（uint32）

#### `search_all(queries: np.ndarray, k: int) -> np.ndarray`

对多个查询向量进行批量搜索。

- `queries`: 查询向量数组（2D numpy 数组，shape [n_queries, dim]，float32）
- `k`: 每个查询返回的最近邻数量
- 返回: 形状为 [n_queries, k] 的结果数组（uint32）

#### 属性

- `num_vertices`: 图中的顶点数量
- `dimension`: 向量维度
- `degree_bound`: 图的度限制
- `entry_point`: 图的入口点
- `ef`: 当前的 ef 参数值

## 依赖

- Python >= 3.7
- numpy >= 1.19.0
- pybind11 >= 2.6.0
- C++17 编译器（支持 OpenMP 和 AVX512）

## 注意事项

- 需要支持 AVX512 指令集的 CPU
- 需要 OpenMP 支持
- 在 Linux 系统上，数据文件会使用 mmap 进行内存映射以提高性能

