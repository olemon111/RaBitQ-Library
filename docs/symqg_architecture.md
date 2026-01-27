# SymQG (对称量化图) 索引结构详解

## 1. 概述

SymQG (Symmetric Quantized Graph) 是一种基于量化图的近似最近邻搜索索引结构。它结合了图索引的高效搜索能力和量化技术的存储压缩优势，通过对称量化编码邻居向量，实现快速的距离估计和搜索。

## 2. 索引结构布局

### 2.1 整体内存布局

SymQG使用行式存储（Row-Major）布局，每个数据点占用一行连续内存。每行包含以下四个部分：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        单个顶点（数据点）的内存布局                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ 1. Raw Vector (原始向量)                                      │      │
│  │    大小: dim_ * sizeof(T) 字节                                │      │
│  │    存储: 数据点的原始特征向量                                  │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                              ↓                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ 2. Batch Data (量化批次数据)                                  │      │
│  │    大小: (degree_bound_ / 32) * QGBatchDataMap::data_bytes()│      │
│  │    存储: 邻居向量的量化编码和因子                                │      │
│  │    - 每个batch包含32个邻居的量化信息                            │      │
│  │    - 包含: 二进制码 + f_add因子 + f_rescale因子                │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                              ↓                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ 3. Neighbor IDs (邻居ID列表)                                  │      │
│  │    大小: degree_bound_ * sizeof(PID) 字节                      │      │
│  │    存储: 该顶点的所有邻居顶点ID                                 │      │
│  │    注意: degree_bound_必须是32的倍数                           │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  总行大小: row_offset_ = batch_data_offset_ + neighbor_offset_ + ... │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Batch Data 详细结构

每个Batch Data存储32个邻居的量化信息，结构如下：

```
┌─────────────────────────────────────────────────────────────────┐
│                   单个Batch (32个邻居)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ batch_bin_code_ (二进制量化码)                          │   │
│  │ 大小: padded_dim * 32 / 8 字节                          │   │
│  │ 格式: 每个邻居用1-bit编码，共32个邻居                    │   │
│  │ 布局: [neighbor0_code] [neighbor1_code] ... [neighbor31]│   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ f_add_ (加法因子数组)                                    │   │
│  │ 大小: 32 * sizeof(T) 字节                                │   │
│  │ 存储: 32个float值，用于距离估计的加法项                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ f_rescale_ (重缩放因子数组)                              │   │
│  │ 大小: 32 * sizeof(T) 字节                                │   │
│  │ 存储: 32个float值，用于距离估计的缩放项                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  总大小: (padded_dim * 32 / 8) + (32 * sizeof(T) * 2)         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 内存偏移计算

根据代码中的`initialize()`函数：

```cpp
batch_data_offset_ = dim_ * sizeof(T);  // 原始向量后
neighbor_offset_ = batch_data_offset_ + 
                   QGBatchDataMap::data_bytes(padded_dim_) * (degree_bound_ / 32);
row_offset_ = neighbor_offset_ + degree_bound_ * sizeof(PID);
```

### 2.4 完整索引结构示意图

```
索引整体布局 (N个数据点):

┌─────────────────────────────────────────────────────────────┐
│ 顶点0: [RawVec] [Batch0] [Batch1] ... [NeighborIDs]        │
├─────────────────────────────────────────────────────────────┤
│ 顶点1: [RawVec] [Batch0] [Batch1] ... [NeighborIDs]        │
├─────────────────────────────────────────────────────────────┤
│ 顶点2: [RawVec] [Batch0] [Batch1] ... [NeighborIDs]        │
├─────────────────────────────────────────────────────────────┤
│  ...                                                         │
├─────────────────────────────────────────────────────────────┤
│ 顶点N-1: [RawVec] [Batch0] [Batch1] ... [NeighborIDs]      │
└─────────────────────────────────────────────────────────────┘

其中:
- RawVec: dim_ * sizeof(float) 字节
- Batch数量: degree_bound_ / 32
- 每个Batch: QGBatchDataMap::data_bytes(padded_dim_) 字节
- NeighborIDs: degree_bound_ * sizeof(PID) 字节
```

## 3. 构建流程详解

### 3.1 构建流程概览

SymQG的构建过程分为以下几个阶段：

```
初始化阶段
    ↓
随机初始化图结构
    ↓
迭代优化 (默认3次迭代)
    ├─ 迭代1: 搜索新邻居 + 剪枝 + 添加反向边
    ├─ 迭代2: 搜索新邻居 + 剪枝 + 添加反向边
    └─ 迭代3: 搜索新邻居 + 剪枝 + 添加反向边 + 图精炼
    ↓
保存索引
```

### 3.2 详细构建步骤

#### 步骤1: 初始化入口点

```cpp
// 计算数据集的质心
std::vector<float> centroid = compute_centroid(data, num_nodes_, dim_, num_threads_);

// 找到距离质心最近的点作为入口点
PID entry_point = exact_nn(data, centroid.data(), num_nodes_, dim_, num_threads_, euclidean_sqr<float>);

qg_.set_ep(entry_point);
qg_.copy_vectors(data);  // 复制原始向量到索引
```

#### 步骤2: 随机初始化 (`random_init()`)

为每个顶点随机选择`degree_bound_`个邻居：

```cpp
for (每个顶点 i) {
    1. 随机选择 degree_bound_ 个不同的邻居
    2. 计算与这些邻居的精确距离
    3. 调用 update_qg(i, neighbors) 更新量化编码
}
```

#### 步骤3: 迭代优化 (`iter()`)

每次迭代包含以下子步骤：

**3.3.1 搜索新邻居 (`search_new_neighbors`)**

```cpp
for (每个顶点 i) {
    1. 使用当前图结构进行ANN搜索，找到 ef_build_ 个候选邻居
    2. 将当前邻居也加入候选池
    3. 对候选池进行部分排序，保留前 kMaxCandidatePoolSize 个
    4. 调用启发式剪枝算法
}
```

**3.3.2 启发式剪枝 (`heuristic_prune`)**

采用贪心策略选择邻居：

```cpp
算法流程:
1. 按距离排序候选邻居
2. 依次选择邻居，如果候选k被已选邻居j遮挡（djk < dik），则剪枝k
3. 如果refine=true，记录被剪枝的邻居到pruned_neighbors_
4. 最终保留 degree_bound_ 个邻居
```

**3.3.3 添加反向边 (`add_reverse_edges`)**

确保图的对称性：

```cpp
for (每个顶点 i) {
    for (i的每个邻居 j) {
        if (j的邻居列表中没有i) {
            尝试将i添加到j的邻居列表
            - 如果j的邻居数 < degree_bound_: 直接添加
            - 否则: 加入reverse_buffer等待后续处理
        }
    }
}

// 对reverse_buffer中的顶点进行剪枝
for (每个顶点 i) {
    合并reverse_buffer[i]和new_neighbors_[i]
    调用heuristic_prune进行剪枝
}
```

**3.3.4 图精炼 (`graph_refine`) - 仅在最后一次迭代**

确保每个顶点的度数等于`degree_bound_`：

```cpp
for (每个顶点 i) {
    if (i的邻居数 < degree_bound_) {
        1. 使用二分搜索找到合适的cosine阈值
        2. 从pruned_neighbors_中选择邻居补充
        3. 如果仍不足，随机选择顶点补充
    }
}
```

**3.3.5 更新量化编码 (`update_qg`)**

```cpp
for (每个顶点 i) {
    1. 更新邻居ID列表
    2. 对当前顶点和所有邻居向量进行旋转（Rotator）
    3. 以当前顶点为质心，对邻居向量进行量化
    4. 将量化码和因子存储到batch_data中
}
```

### 3.3 量化编码过程

对于顶点`i`，量化其邻居的过程：

```cpp
1. 旋转当前顶点: rotated_centroid = rotator.rotate(vector[i])
2. 旋转所有邻居: rotated_neighbors = [rotator.rotate(vector[neighbor[j]]) for j in neighbors]
3. 对每32个邻居进行批量量化:
   - 使用RaBitQ算法进行1-bit量化
   - 生成: batch_bin_code, f_add, f_rescale
   - 存储到对应的batch_data位置
```

## 4. 搜索流程详解

### 4.1 搜索算法流程

```
输入: 查询向量 query, top-k值 k, ef参数
输出: k个最近邻的ID

1. 旋转查询向量: rotated_query = rotator.rotate(query)
2. 初始化搜索缓冲区: search_pool (大小=ef)
3. 初始化结果缓冲区: res_pool (大小=k)
4. 初始化访问标记: visited_set
5. 将入口点加入search_pool

6. while (search_pool不为空):
   a. 从search_pool中弹出距离最小的节点 cur_node
   b. 如果cur_node已访问，跳过
   c. 标记cur_node为已访问
   d. 计算query与cur_node的精确距离: exact_dist
   e. 扫描cur_node的所有邻居:
      - 使用量化码快速估计距离
      - 将未访问的邻居加入search_pool
   f. 将cur_node加入res_pool

7. 更新结果: 检查res_pool中节点的邻居，补充结果
8. 返回res_pool中的top-k结果
```

### 4.2 关键函数详解

#### 4.2.1 `scan_neighbors()` - 扫描邻居

```cpp
void scan_neighbors(
    const BatchQuery<T>& q_obj,      // 旋转后的查询对象
    PID data_id,                      // 当前节点ID
    T* est_dist,                      // 输出：估计距离数组
    buffer::SearchBuffer<T>& search_pool,  // 搜索缓冲区
    HashBasedBooleanSet& vis,         // 访问标记
    size_t cur_degree                 // 当前节点的度数
) {
    // 1. 获取当前节点的batch_data
    const char* batch_data = get_batch_data(data_id);
    
    // 2. 对每个batch（32个邻居）进行批量距离估计
    for (i = 0; i < cur_degree; i += 32) {
        qg_batch_estdist(batch_data, q_obj, padded_dim_, est_dist + i);
        batch_data += QGBatchDataMap::data_bytes(padded_dim_);
    }
    
    // 3. 获取邻居ID列表
    const PID* neighbors = get_neighbors(data_id);
    
    // 4. 将未访问的邻居加入搜索缓冲区
    for (i = 0; i < cur_degree; ++i) {
        PID neighbor = neighbors[i];
        T dist = est_dist[i];  // 使用估计距离
        
        if (!vis.get(neighbor) && !search_pool.is_full(dist)) {
            search_pool.insert(neighbor, dist);
        }
    }
}
```

#### 4.2.2 距离估计 (`qg_batch_estdist`)

使用量化码快速估计距离：

```
对于batch中的32个邻居:
1. 使用batch_bin_code和查询向量计算内积（快速位运算）
2. 应用f_add和f_rescale因子进行距离估计
3. 返回32个估计距离值
```

#### 4.2.3 `update_results()` - 结果更新

```cpp
void update_results(
    buffer::SearchBuffer<T>& result_pool,
    HashBasedBooleanSet& vis,
    const T* query
) {
    // 如果结果池已满，直接返回
    if (result_pool.is_full()) return;
    
    // 检查结果池中每个节点的邻居
    for (每个结果节点 record) {
        for (record的每个邻居 neighbor) {
            if (neighbor未访问 && 结果池未满) {
                计算query与neighbor的精确距离
                将neighbor加入结果池
            }
        }
        if (结果池已满) break;
    }
}
```

### 4.3 搜索性能优化

1. **量化距离估计**: 使用1-bit量化码快速估计距离，避免加载完整向量
2. **批量处理**: 每次处理32个邻居，利用SIMD指令加速
3. **内存预取**: 在扫描邻居时预取下一个可能访问的向量
4. **访问标记**: 使用哈希集合快速检查节点是否已访问

## 5. 关键数据结构

### 5.1 QuantizedGraph 类

```cpp
class QuantizedGraph {
private:
    size_t num_points_;           // 数据点数量
    size_t degree_bound_;         // 度数上界（32的倍数）
    size_t dim_;                  // 原始维度
    size_t padded_dim_;           // 填充后的维度（64的倍数）
    PID entry_point_;             // 入口点ID
    
    Array<char> data_;            // 主数据数组（行式存储）
    Rotator* rotator_;            // 数据旋转器
    
    // 内存偏移
    size_t batch_data_offset_;    // batch_data的偏移
    size_t neighbor_offset_;      // 邻居ID的偏移
    size_t row_offset_;           // 每行总大小
};
```

### 5.2 QGBuilder 类

```cpp
class QGBuilder {
private:
    QuantizedGraph& qg_;                    // 要构建的图索引
    size_t ef_build_;                       // 构建时的搜索ef参数
    std::vector<CandidateList> new_neighbors_;      // 新邻居列表
    std::vector<CandidateList> pruned_neighbors_;   // 被剪枝的邻居（用于精炼）
    std::vector<HashBasedBooleanSet> visited_list_; // 访问标记列表
    std::vector<uint32_t> degrees_;                 // 每个顶点的度数
};
```

## 6. 使用示例

### 6.1 构建索引

```cpp
// 1. 加载数据
data_type data;
rabitqlib::load_vecs<float, data_type>(data_file, data);

// 2. 创建索引对象
index_type qg(data.rows(), data.cols(), degree, metric_type);

// 3. 创建构建器并构建
rabitqlib::symqg::QGBuilder builder(qg, ef, data.data());
builder.build();  // 默认3次迭代

// 4. 保存索引
qg.save(index_file);
```

### 6.2 查询索引

```cpp
// 1. 加载索引
index_type qg;
qg.load(index_file);

// 2. 设置搜索参数
qg.set_ef(ef_search);

// 3. 执行搜索
std::vector<PID> results(topk);
qg.search(&query[0], topk, results.data());
```

## 7. 性能特点

1. **存储效率**: 使用1-bit量化大幅压缩存储空间
2. **搜索速度**: 批量距离估计和SIMD优化提升搜索速度
3. **内存局部性**: 行式存储保证良好的缓存性能
4. **可扩展性**: 支持大规模数据集和并行构建

## 8. 参数说明

- `degree_bound_`: 每个顶点的最大邻居数，必须是32的倍数
- `ef_build_`: 构建时的搜索宽度，影响构建质量和时间
- `ef`: 查询时的搜索宽度，影响查询质量和速度
- `padded_dim_`: 填充后的维度，必须是64的倍数，用于SIMD优化
