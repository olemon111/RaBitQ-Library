#include <rabitqlib/index/symqg/qg.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

static void drop_caches()
{
#ifdef __linux__
    // Best-effort: drop page cache. Requires root; otherwise will fail gracefully.
    ::sync();
    int fd = ::open("/proc/sys/vm/drop_caches", O_WRONLY);
    if (fd >= 0)
    {
        const char *val = "3\n";
        ssize_t n = ::write(fd, val, std::strlen(val));
        ::close(fd);
        if (n < 0)
        {
            std::cerr << "drop_caches write failed: " << std::strerror(errno) << std::endl;
        }
    }
    else
    {
        std::cerr << "drop_caches open failed: " << std::strerror(errno)
                  << " (need root or proper permissions)" << std::endl;
    }
#else
    // No-op on non-Linux
#endif
}
// Minimal helpers inline in this file
static void expect_eq(size_t a, size_t b)
{
    if (a != b)
    {
        std::cerr << "expect_eq failed: " << a << " != " << b << std::endl;
        std::abort();
    }
}

static std::string calc_recall(size_t nq, size_t ngt, const uint32_t *gts, size_t K,
                               const uint32_t *results, size_t n_ats, const size_t *ats)
{
    std::vector<double> recalls(n_ats, 0.0);

    for (size_t qi = 0; qi < nq; ++qi)
    {
        const uint32_t *gt_row = gts + qi * ngt;
        const uint32_t *res_row = results + qi * K;

        // Preload results into a set for membership checks.
        std::unordered_set<uint32_t> res_set;
        res_set.reserve(K * 2);
        for (size_t j = 0; j < K; ++j)
            res_set.insert(res_row[j]);

        for (size_t ai = 0; ai < n_ats; ++ai)
        {
            const size_t cut = std::min(ats[ai], ngt);
            size_t hit = 0;
            for (size_t k = 0; k < cut; ++k)
            {
                if (res_set.find(gt_row[k]) != res_set.end())
                    ++hit;
            }
            recalls[ai] += static_cast<double>(hit) / static_cast<double>(cut);
        }
    }

    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < n_ats; ++i)
    {
        double avg = recalls[i] / static_cast<double>(nq);
        if (i)
            oss << ", ";
        oss << "R@" << ats[i] << ": " << avg;
    }
    oss << "]";
    return oss.str();
}

// Very small mmap wrapper inline
class MMap
{
public:
    explicit MMap(const std::string &path)
    {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
        {
            throw std::runtime_error("Failed to open file: " + path);
        }
        struct stat st{};
        if (fstat(fd_, &st) != 0)
        {
            ::close(fd_);
            throw std::runtime_error("Failed to stat file: " + path);
        }
        size_ = static_cast<size_t>(st.st_size);
        data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED)
        {
            ::close(fd_);
            throw std::runtime_error("Failed to mmap file: " + path);
        }
    }
    ~MMap()
    {
        if (data_ && data_ != MAP_FAILED)
            ::munmap(data_, size_);
        if (fd_ >= 0)
            ::close(fd_);
    }
    template <typename T>
    T *ptr(size_t offset_bytes = 0) const
    {
        if (offset_bytes > size_)
            throw std::out_of_range("offset out of range");
        return reinterpret_cast<T *>(reinterpret_cast<unsigned char *>(data_) + offset_bytes);
    }
    void *data_ = nullptr;
    size_t size_ = 0;

private:
    int fd_ = -1;
};

// Minimal Dataset loader for .fbin
class Dataset
{
public:
    Dataset(size_t first, size_t count,
            const std::string &base_fbin_path,
            const std::string &query_fbin_path,
            const std::string &gt_ibin_path)
        : first_(first), count_(count)
    {
        load_fbin(base_fbin_path, base_);
        if (first_ >= base_.size())
            throw std::out_of_range("first out of range");
        nvecs_ = std::min(count_, base_.size() - first_);
        dim_ = base_.empty() ? 0 : base_[0].size();

        // load queries from fbin
        std::vector<std::vector<float>> qvecs;
        load_fbin(query_fbin_path, qvecs);
        if (!qvecs.empty() && qvecs[0].size() == dim_)
        {
            nq_ = qvecs.size();
            queries_.resize(nq_ * dim_);
            for (size_t i = 0; i < nq_; ++i)
                std::copy(qvecs[i].begin(), qvecs[i].end(), queries_.begin() + i * dim_);
        }
        else
        {
            throw std::runtime_error("query dim mismatch or empty queries");
        }

        // load ground-truth from ibin (uint32_t) with header (n, k)
        load_ibin(gt_ibin_path, gts_, nq_, ngt_);

        // flatten base vectors for copy_vectors
        vecs_.resize(nvecs_ * dim_);
        for (size_t i = 0; i < nvecs_; ++i)
        {
            const auto &v = base_[first_ + i];
            std::copy(v.begin(), v.end(), vecs_.begin() + i * dim_);
        }
    }

    size_t nvecs() const { return nvecs_; }
    size_t dim() const { return dim_; }
    const float *vecs() const { return vecs_.data(); }
    const float *query(size_t i) const { return queries_.data() + i * dim_; }
    size_t nq() const { return nq_; }
    size_t ngt() const { return ngt_; }
    const uint32_t *gts() const { return gts_.data(); }

    friend std::ostream &operator<<(std::ostream &os, const Dataset &ds)
    {
        os << "nvecs=" << ds.nvecs_ << ", dim=" << ds.dim_ << ", nq=" << ds.nq_;
        return os;
    }

private:
    static void load_fbin(const std::string &path, std::vector<std::vector<float>> &out)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("Failed to open fbin: " + path);

        uint32_t n = 0, d = 0;
        in.read(reinterpret_cast<char *>(&n), sizeof(uint32_t));
        in.read(reinterpret_cast<char *>(&d), sizeof(uint32_t));
        if (!in)
            throw std::runtime_error("Failed to read header from fbin");

        out.resize(n, std::vector<float>(d));
        for (uint32_t i = 0; i < n; ++i)
        {
            in.read(reinterpret_cast<char *>(out[i].data()), sizeof(float) * d);
            if (!in)
                throw std::runtime_error("Failed to read vector from fbin");
        }
    }

    static void load_ibin(const std::string &path, std::vector<uint32_t> &out, size_t &n, size_t &k)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("Failed to open ibin: " + path);
        uint32_t n32 = 0, k32 = 0;
        in.read(reinterpret_cast<char *>(&n32), sizeof(uint32_t));
        in.read(reinterpret_cast<char *>(&k32), sizeof(uint32_t));
        if (!in)
            throw std::runtime_error("Failed to read header from ibin");
        n = n32;
        k = k32;
        out.resize(static_cast<size_t>(n) * static_cast<size_t>(k));
        in.read(reinterpret_cast<char *>(out.data()), sizeof(uint32_t) * out.size());
        if (!in)
            throw std::runtime_error("Failed to read data from ibin");
    }

    size_t first_ = 0;
    size_t count_ = 0;
    size_t nvecs_ = 0;
    size_t dim_ = 0;
    size_t nq_ = 0;
    size_t ngt_ = 0;
    std::vector<std::vector<float>> base_;
    std::vector<float> vecs_;
    std::vector<float> queries_;
    std::vector<uint32_t> gts_;
};

int main()
{
    rabitqlib::symqg::QuantizedGraph<> qg;
    qg.load_index("output/qg_metadata.bin", "output/qg_data.bin");
    Dataset ds(0, 1000000,
               "/home/lbl/vdb-dataset/sift1m/sift1m_base.fbin",
               "/home/lbl/vdb-dataset/sift1m/sift1m_query.fbin",
               "/home/lbl/vdb-dataset/sift1m/sift1m_gt100");
    std::cout << "ds: " << ds << std::endl;

    constexpr size_t racall_ats[] = {1, 5, 10, 20, 50, 100};
    // constexpr size_t racall_ats[] = {1, 10};
    // constexpr size_t K = 10;
    constexpr size_t K = 100;
    expect_eq(ds.ngt(), 100);
    std::vector<uint32_t> results(ds.ngt() * ds.nq());
    auto search_all = [&](size_t L)
    {
        qg.set_ef(L);
        auto ts = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < ds.nq(); ++i)
            qg.search(ds.query(i), std::min(L, K), results.data() + i * K);
        auto te = std::chrono::high_resolution_clock::now();
        auto duration_s = std::chrono::duration<double>(te - ts).count();
        auto qps = ds.nq() / duration_s;
        std::cout << "L: " << L << ", duration(s): " << duration_s << ", qps: " << qps << std::endl;
        auto recall = calc_recall(ds.nq(), ds.ngt(), ds.gts(), K, results.data(),
                                  std::size(racall_ats), racall_ats);
        std::cout << "recall: " << recall << std::endl;
    };

    auto search_one = [&](size_t L, size_t query_index)
    {
        qg.set_ef(L);
        std::vector<uint32_t> one_res(K);
        auto ts = std::chrono::high_resolution_clock::now();
        qg.search(ds.query(query_index), std::min(L, K), one_res.data());
        auto te = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration<double, std::milli>(te - ts).count();

        // 计算单个 query 的 recall，gts 偏移到指定 query
        const uint32_t *gt_row = ds.gts() + query_index * ds.ngt();
        auto recall_one = calc_recall(1, ds.ngt(), gt_row, K, one_res.data(),
                                      std::size(racall_ats), racall_ats);

        std::cout << "[search_one] L: " << L << ", query=" << query_index
                  << ", duration(ms): " << duration_ms
                  << ", recall: " << recall_one << std::endl;
    };

    for (size_t L : {100, 120, 140, 160, 180, 200})
    {
        drop_caches();
        search_one(L, 0);
    }

    for (size_t L : {40, 60, 80, 100, 120, 140, 160, 180, 200})
    {
        search_all(L);
    }

    return 0;
}
