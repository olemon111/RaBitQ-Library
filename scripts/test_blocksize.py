import os
import mmap
import time
import random

FILE = "/home/lbl/dataset/32GB_bigfile.bin"
# FILE = "/home/lbl/dataset/8GB_bigfile.bin"
ROUNDS = 5000

X_LIST = [1024, 2*1024, 4*1024, 8*1024, 16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 1024*1024]

def percentile(data, p):
    data = sorted(data)
    k = int(len(data)*p/100)
    return data[min(k, len(data)-1)]

def run_test(mm, offsets):
    latencies = []
    for off in offsets:
        t0 = time.perf_counter_ns()
        _ = mm[off]  # mmap 访问
        t1 = time.perf_counter_ns()
        latencies.append((t1-t0)/1e6)  # ms
    return {
        "avg_ms": sum(latencies)/len(latencies),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99)
    }

def main():
    fd = os.open(FILE, os.O_RDONLY)
    # 提示内核：随机访问，尽量不要做预读（best-effort）
    try:
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_RANDOM"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    except Exception:
        # 在某些平台/内核上可能不可用或被忽略
        pass
    size = os.stat(FILE).st_size
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    # 对 mmap 提示随机访问，抑制页级预读窗口扩展
    try:
        if hasattr(mm, "madvise") and hasattr(mmap, "MADV_RANDOM"):
            mm.madvise(mmap.MADV_RANDOM)
    except Exception:
        # 旧 Python 或非 Linux 可能不支持
        pass

    print(f"File size: {size / (1024**3):.1f} GB")

    for X in X_LIST:
        max_i = min(ROUNDS, (size)//X)

        # 对齐访问
        offsets_aligned = [i*X for i in range(max_i)]
        random.shuffle(offsets_aligned)
        aligned_stats = run_test(mm, offsets_aligned)

        # 半步偏移访问
        offsets_shifted = [i*X + X//2 for i in range(max_i)]
        random.shuffle(offsets_shifted)
        shifted_stats = run_test(mm, offsets_shifted)

        print(f"\n=== X = {X // 1024} KB ===")
        print(f"Aligned : {aligned_stats}")
        print(f"Shifted : {shifted_stats}")

    mm.close()
    os.close(fd)

if __name__ == "__main__":
    main()
