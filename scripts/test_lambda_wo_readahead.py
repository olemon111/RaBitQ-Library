import os
import mmap
import time
import random

FILE = "/mnt/efs/1GB_bigfile.bin"

DEFAULT_ROUNDS = 2000
DEFAULT_X_LIST = [
    1024, 2*1024, 4*1024, 8*1024,
    16*1024, 32*1024, 64*1024,
    128*1024, 256*1024
]


def percentile(data, p):
    data = sorted(data)
    k = int(len(data) * p / 100)
    return data[min(k, len(data) - 1)]


def run_test(mm, offsets):
    latencies = []
    for off in offsets:
        t0 = time.perf_counter_ns()
        _ = mm[off]
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1e6)  # ms
    return {
        "avg_ms": sum(latencies) / len(latencies),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
    }


def lambda_handler(event, context):
    rounds = int(event.get("rounds", DEFAULT_ROUNDS))
    x_list = event.get("x_list", DEFAULT_X_LIST)

    print(f"[INFO] rounds={rounds}")
    print(f"[INFO] x_list={x_list}")
    print(f"[INFO] file={FILE}")

    fd = None
    mm = None

    try:
        fd = os.open(FILE, os.O_RDONLY)
        # 提示内核随机访问，尽量避免预读（best-effort）
        try:
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_RANDOM"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        except Exception:
            pass

        size = os.stat(FILE).st_size
        mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        # 对映射提示随机访问，抑制页级预读窗口扩展
        try:
            if hasattr(mm, "madvise") and hasattr(mmap, "MADV_RANDOM"):
                mm.madvise(mmap.MADV_RANDOM)
        except Exception:
            pass

        print(f"[INFO] File size: {size / (1024**3):.2f} GB")
        print(f"[INFO] Page size: {os.sysconf('SC_PAGE_SIZE')} bytes")

        results = {}

        for X in x_list:
            max_i = min(rounds, size // X)
            if max_i <= 0:
                continue

            offsets_aligned = [i * X for i in range(max_i)]
            random.shuffle(offsets_aligned)
            aligned_stats = run_test(mm, offsets_aligned)

            offsets_shifted = [i * X + X // 2 for i in range(max_i)]
            random.shuffle(offsets_shifted)
            shifted_stats = run_test(mm, offsets_shifted)

            print(f"\n=== X = {X // 1024 if X >= 1024 else X} "
                  f"{'KB' if X >= 1024 else 'B'} ===")
            print(f"Aligned : {aligned_stats}")
            print(f"Shifted : {shifted_stats}")

            results[X] = {
                "aligned": aligned_stats,
                "shifted": shifted_stats,
            }

        return {
            "status": "ok",
            "results": results,
        }

    finally:
        if mm is not None:
            mm.close()
        if fd is not None:
            os.close(fd)
