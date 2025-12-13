import numpy as np
import struct


def read_ivecs(filename):
    print(f"Reading File - {filename}")
    a = np.fromfile(filename, dtype="int32")
    d = a[0]
    print(f"\t{filename} readed")
    return a.reshape(-1, d + 1)[:, 1:]


def read_fvecs(filename):
    return read_ivecs(filename).view("float32")


def write_ivecs(filename, m):
    print(f"Writing File - {filename}")
    n, d = m.shape
    myimt = "i" * d
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(struct.pack("i", d))
            bin = struct.pack(myimt, *m[i])
            f.write(bin)
    print(f"\t{filename} wrote")


def write_fvecs(filename, m):
    m = m.astype("float32")
    write_ivecs(filename, m.view("int32"))


def read_ibin(filename):
    """
    Read ibin file matching C++ load_ibin() behavior.
    Format: [n (uint32), k (uint32), n*k uint32_t data]
    """
    with open(filename, 'rb') as f:
        # Read header: n and k as uint32 (matching C++ uint32_t)
        n = np.frombuffer(f.read(4), dtype='uint32')[0]
        k = np.frombuffer(f.read(4), dtype='uint32')[0]
        
        # Read exactly n*k uint32_t elements (matching C++ behavior)
        data = np.frombuffer(f.read(n * k * 4), dtype='uint32')
    
    print(f"\t{filename} readed")
    # Reshape to [n, k]
    return data.reshape(n, k)


def read_fbin(filename):
    """
    Read fbin file matching C++ load_fbin() behavior.
    Format: [n (uint32), d (uint32), n*d float data]
    """
    with open(filename, 'rb') as f:
        # Read header: n and d as uint32 (matching C++ uint32_t)
        n = np.frombuffer(f.read(4), dtype='uint32')[0]
        d = np.frombuffer(f.read(4), dtype='uint32')[0]
        
        # Read exactly n*d float elements (matching C++ behavior)
        data = np.frombuffer(f.read(n * d * 4), dtype='float32')
    
    print(f"\t{filename} readed")
    # Reshape to [n, d]
    return data.reshape(n, d)


def write_sampled_data(filename, m, num_sampled, seed=42):
    num_points = m.shape[0]
    d = m.shape[1]
    m = m.view("int32")

    np.random.seed(seed)
    sequence = np.random.permutation(num_points)

    with open(filename, "wb") as f:
        for i in range(num_sampled):
            f.write(struct.pack("i", d))
            bin = struct.pack(f"{d}i", *m[sequence[i]])
            f.write(bin)
