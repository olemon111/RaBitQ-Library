"""
示例：使用 QG Python 包

这个示例展示了如何使用 qg 包进行搜索，对应 load_test_qg.cpp 的功能。
"""
import numpy as np
import time
import os
import sys
from qg import QG
from utils.io import read_fbin, read_ibin


def drop_caches():
    """
    Drop page cache (Linux only, requires root or proper permissions).
    Corresponds to drop_caches() in load_test_qg.cpp.
    """
    if sys.platform == 'linux':
        try:
            os.sync()
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')
        except (IOError, PermissionError) as e:
            print(f"drop_caches failed: {e} (need root or proper permissions)")
    else:
        # No-op on non-Linux
        pass


def calc_recall(nq: int, ngt: int, gts: np.ndarray, k: int, 
                results: np.ndarray, ats: list) -> str:
    """
    Calculate recall metrics.
    Corresponds to calc_recall() in load_test_qg.cpp.
    
    Args:
        nq: Number of queries
        ngt: Number of ground truth neighbors per query
        gts: Ground truth array, shape [nq, ngt], dtype uint32
        k: Number of results per query
        results: Search results, shape [nq, k], dtype uint32
        ats: List of @k values to compute recall at (e.g., [1, 5, 10, 20, 50, 100])
    
    Returns:
        String representation of recall metrics
    """
    recalls = np.zeros(len(ats))
    
    for qi in range(nq):
        gt_row = gts[qi]  # Shape: [ngt]
        res_row = results[qi]  # Shape: [k]
        
        # Preload results into a set for membership checks
        res_set = set(res_row)
        
        for ai, at in enumerate(ats):
            cut = min(at, ngt)
            hit = sum(1 for i in range(cut) if gt_row[i] in res_set)
            recalls[ai] += hit / cut if cut > 0 else 0.0
    
    # Average over all queries
    recalls = recalls / nq
    
    # Format output
    recall_strs = [f"R@{ats[i]}: {recalls[i]:.6f}" for i in range(len(ats))]
    return "[" + ", ".join(recall_strs) + "]"


def load_dataset(query_fbin_path: str, gt_ibin_path: str):
    """
    Load queries and ground truth from files.
    Corresponds to Dataset class in load_test_qg.cpp.
    
    Args:
        query_fbin_path: Path to query .fbin file
        gt_ibin_path: Path to ground truth .ibin file
    
    Returns:
        tuple: (queries, gts, nq, ngt)
            queries: numpy array, shape [nq, dim], dtype float32
            gts: numpy array, shape [nq, ngt], dtype uint32
            nq: number of queries
            ngt: number of ground truth neighbors per query
    """
    # Load queries from fbin file
    queries = read_fbin(query_fbin_path)  # Shape: [nq, dim], dtype: float32
    nq = queries.shape[0]
    dim = queries.shape[1]
    
    # Load ground truth from ibin file
    gts = read_ibin(gt_ibin_path)  # Shape: [n, k], dtype: int32 (will convert to uint32)
    ngt = gts.shape[1]
    
    # Convert gts to uint32
    gts = gts.astype(np.uint32)
    
    # Verify dimensions match
    if gts.shape[0] != nq:
        raise ValueError(f"Query count mismatch: queries has {nq} queries, but gts has {gts.shape[0]}")
    
    return queries, gts, nq, ngt


def main():
    # 创建 QG 实例
    qg = QG()
    
    # 加载索引（对应 load_test_qg.cpp 中的 load_index）
    qg.load_index("output/qg_metadata.bin", "output/qg_data.bin")
    print(f"Loaded index: {qg.num_vertices} vertices, dim={qg.dimension}")
    
    # 加载查询和 ground truth 数据（对应 load_test_qg.cpp 中的 Dataset）
    query_fbin_path = "/home/lbl/vdb-dataset/sift1m/sift1m_query.fbin"
    gt_ibin_path = "/home/lbl/vdb-dataset/sift1m/sift1m_gt100"
    
    print(f"Loading queries from: {query_fbin_path}")
    print(f"Loading ground truth from: {gt_ibin_path}")
    queries, gts, nq, ngt = load_dataset(query_fbin_path, gt_ibin_path)
    print(f"Loaded {nq} queries, {ngt} ground truth neighbors per query")
    
    # Verify query dimension matches index dimension
    if queries.shape[1] != qg.dimension:
        raise ValueError(f"Query dimension mismatch: queries have dim {queries.shape[1]}, "
                         f"but index has dim {qg.dimension}")
    
    # Recall @k values to compute (对应 load_test_qg.cpp 中的 racall_ats)
    recall_ats = [1, 5, 10, 20, 50, 100]
    K = 100  # Number of results to return (对应 load_test_qg.cpp 中的 K)
    
    # 示例 1: search_one - 单个查询（对应 load_test_qg.cpp 中的 search_one）
    print("\n=== search_one 示例 ===")
    query_index = 0  # 默认使用 query_index=0
    for L in [100, 120, 140, 160, 180, 200]:
        drop_caches()
        
        query = queries[query_index]  # Shape: [dim], dtype: float32
        
        # 执行搜索并计时
        ts = time.perf_counter()
        results = qg.search_one(query, k=min(L, K), L=L)
        te = time.perf_counter()
        duration_ms = (te - ts) * 1000.0
        
        # 计算 recall（使用单个 query 的 ground truth）
        # 对应 load_test_qg.cpp: const uint32_t *gt_row = ds.gts() + query_index * ds.ngt();
        gt_row = gts[query_index:query_index+1]  # Shape: [1, ngt]
        res_row = results.reshape(1, -1)  # Shape: [1, k]
        recall_str = calc_recall(1, ngt, gt_row, K, res_row, recall_ats)
        
        print(f"[search_one] L: {L}, query={query_index}, "
              f"duration(ms): {duration_ms:.3f}, recall: {recall_str}")
    
    # 示例 2: search_all - 批量查询（对应 load_test_qg.cpp 中的 search_all）
    print("\n=== search_all 示例 ===")
    for L in [40, 60, 80, 100, 120, 140, 160, 180, 200]:
        # 执行搜索并计时
        ts = time.perf_counter()
        all_results = qg.search_all(queries, k=min(L, K), L=L)
        te = time.perf_counter()
        duration_s = te - ts
        qps = nq / duration_s if duration_s > 0 else 0
        
        # 计算 recall
        recall_str = calc_recall(nq, ngt, gts, K, all_results, recall_ats)
        
        print(f"L: {L}, duration(s): {duration_s:.6f}, qps: {qps:.2f}, recall: {recall_str}")


if __name__ == "__main__":
    main()
