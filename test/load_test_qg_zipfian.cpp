#include <rabitqlib/index/symqg/qg.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <sstream>
#include <random>
#include <cmath>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

static void drop_caches()
{
#ifdef __linux__
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
#endif
}

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

        load_ibin(gt_ibin_path, gts_, nq_, ngt_);

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

class ZipfianSampler
{
public:
    ZipfianSampler(size_t n, double s, uint64_t seed)
        : rng_(seed)
    {
        if (n == 0)
            throw std::runtime_error("ZipfianSampler: n must be > 0");
        if (s <= 0.0)
            throw std::runtime_error("ZipfianSampler: s must be > 0");

        std::vector<double> weights;
        weights.reserve(n);
        for (size_t i = 0; i < n; ++i)
        {
            const double rank = static_cast<double>(i + 1);
            weights.push_back(1.0 / std::pow(rank, s));
        }
        dist_ = std::discrete_distribution<size_t>(weights.begin(), weights.end());
    }

    size_t sample()
    {
        return dist_(rng_);
    }

private:
    std::mt19937_64 rng_;
    std::discrete_distribution<size_t> dist_;
};

static size_t infer_nvecs_from_dataset_name(const std::string &dataset_name)
{
    if (dataset_name.find("10m") != std::string::npos)
        return 10000000;
    if (dataset_name.find("1m") != std::string::npos)
        return 1000000;
    if (dataset_name.find("10k") != std::string::npos)
        return 10000;
    if (dataset_name.find("small") != std::string::npos)
        return 1000;
    throw std::runtime_error("Invalid dataset name: " + dataset_name +
                             " (expected e.g. sift1m, gist1m, sift10m, sift10k, siftsmall)");
}

static bool parse_size_list_arg(const std::string &s, std::vector<size_t> &out)
{
    out.clear();
    size_t i = 0;
    while (i < s.size())
    {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == ','))
            ++i;
        if (i >= s.size())
            break;
        size_t j = i;
        while (j < s.size() && s[j] >= '0' && s[j] <= '9')
            ++j;
        if (j == i)
            return false;
        out.push_back(static_cast<size_t>(std::stoull(s.substr(i, j - i))));
        i = j;
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == ','))
            ++i;
    }
    return !out.empty();
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset_name> e.g. sift1m, gist1m, sift10m, sift10k, siftsmall\n"
                  << "Optional:\n"
                  << "  --dataset_root <dir>   (default: /home/ubuntu/datasets)\n"
                  << "  --output_root <dir>    (default: /mnt/efs/datasets/<dataset_name>/cloudqg)\n"
                  << "  --diskann_suffix <s>   (default: _R32_L125)\n"
                  << "  --K <int>              (default: 10)\n"
                  << "  --L <list>             (default: 40,60,80,100,120,140,160,180,200; e.g. --L 50,75,100)\n"
                  << "  --first <int>          (default: 0)\n"
                  << "  --count <int>          (default: inferred from dataset name)\n"
                  << "  --mode <all|one>       (default: all)\n"
                  << "  --query_index <int>    (default: 0; only used when --mode one)\n"
                  << "  --zipf_s <double>      (default: 0.99; only used when --mode all)\n"
                  << "  --zipf_seed <int>      (default: 42; only used when --mode all)\n"
                  << "  --drop_caches          (best-effort; may require permissions)\n"
                  << "  --print_every          (default: false; print each query latency(ms) in --mode all)\n";
        return 1;
    }

    std::string dataset_name = argv[1];
    std::string dataset_root = "/home/ubuntu/datasets";
    std::string output_root = "/mnt/efs/datasets/" + dataset_name + "/cloudqg";
    std::string diskann_suffix = "_R32_L125";
    size_t first = 0;
    size_t count = 0;
    size_t K = 10;
    std::vector<size_t> Ls{40, 60, 80, 100, 120, 140, 160, 180, 200};
    bool do_drop_caches = false;
    bool print_every = false;
    std::string mode = "all";
    size_t query_index = 0;
    double zipf_s = 0.99;
    uint64_t zipf_seed = 42;

    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto need_val = [&](const char *name) -> std::string
        {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("Missing value for ") + name);
            return std::string(argv[++i]);
        };

        if (arg == "--dataset_root")
            dataset_root = need_val("--dataset_root");
        else if (arg == "--output_root")
            output_root = need_val("--output_root");
        else if (arg == "--diskann_suffix")
            diskann_suffix = need_val("--diskann_suffix");
        else if (arg == "--first")
            first = static_cast<size_t>(std::stoull(need_val("--first")));
        else if (arg == "--count")
            count = static_cast<size_t>(std::stoull(need_val("--count")));
        else if (arg == "--K")
            K = static_cast<size_t>(std::stoull(need_val("--K")));
        else if (arg == "--L")
        {
            std::vector<size_t> parsed;
            if (!parse_size_list_arg(need_val("--L"), parsed))
                throw std::runtime_error("Failed to parse --L list (expected comma/space separated integers)");
            Ls = std::move(parsed);
        }
        else if (arg == "--mode")
            mode = need_val("--mode");
        else if (arg == "--query_index")
            query_index = static_cast<size_t>(std::stoull(need_val("--query_index")));
        else if (arg == "--zipf_s")
            zipf_s = std::stod(need_val("--zipf_s"));
        else if (arg == "--zipf_seed")
            zipf_seed = static_cast<uint64_t>(std::stoull(need_val("--zipf_seed")));
        else if (arg == "--drop_caches")
            do_drop_caches = true;
        else if (arg == "--print_every")
            print_every = true;
        else
            throw std::runtime_error("Unknown argument: " + arg);
    }

    if (count == 0)
        count = infer_nvecs_from_dataset_name(dataset_name);

    std::cout << "dataset_name: " << dataset_name
              << ", first: " << first
              << ", count: " << count
              << ", K: " << K
              << ", mode: " << mode
              << ", query_index: " << query_index
              << ", zipf_s: " << zipf_s
              << ", zipf_seed: " << zipf_seed
              << ", print_every: " << (print_every ? "true" : "false")
              << ", diskann_suffix: " << diskann_suffix
              << ", dataset_root: " << dataset_root
              << ", output_root: " << output_root
              << std::endl;

    const std::string base_fbin_path = dataset_root + "/" + dataset_name + "/" + dataset_name + "_base.fbin";
    const std::string query_fbin_path = dataset_root + "/" + dataset_name + "/" + dataset_name + "_query.fbin";
    const std::string gt_ibin_path = dataset_root + "/" + dataset_name + "/" + dataset_name + "_gt100";

    const std::string output_dir = output_root;
    const std::string qg_data_path = output_dir + "/qg_data" + diskann_suffix + ".bin";
    const std::string qg_metadata_path = output_dir + "/qg_metadata" + diskann_suffix + ".bin";

    rabitqlib::symqg::QuantizedGraph<> qg;
    qg.load_index(qg_metadata_path.c_str(), qg_data_path.c_str());

    Dataset ds(first, count, base_fbin_path, query_fbin_path, gt_ibin_path);
    std::cout << "ds: " << ds << std::endl;

    expect_eq(ds.ngt(), 100);
    std::vector<uint32_t> results(ds.ngt() * ds.nq());

    const std::vector<size_t> default_ats_all{1, 5, 10, 20, 50, 100};
    std::vector<size_t> recall_ats;
    recall_ats.reserve(default_ats_all.size());
    for (size_t a : default_ats_all)
    {
        if (a <= K)
            recall_ats.push_back(a);
    }
    if (recall_ats.empty())
        recall_ats.push_back(K);

    auto search_all_zipfian = [&](size_t L)
    {
        qg.set_ef(L);
        ZipfianSampler sampler(ds.nq(), zipf_s, zipf_seed + static_cast<uint64_t>(L));
        std::vector<size_t> sampled_qids(ds.nq(), 0);

        auto ts = std::chrono::high_resolution_clock::now();
        for (size_t req = 0; req < ds.nq(); ++req)
        {
            const size_t qid = sampler.sample();
            sampled_qids[req] = qid;

            auto q_ts = std::chrono::high_resolution_clock::now();
            qg.search(ds.query(qid), std::min(L, K), results.data() + req * K);
            auto q_te = std::chrono::high_resolution_clock::now();

            if (print_every)
            {
                auto q_ms = std::chrono::duration<double, std::milli>(q_te - q_ts).count();
                std::cout << "[req " << req << ", qid " << qid << "] duration(ms): " << q_ms << std::endl;
            }
        }
        auto te = std::chrono::high_resolution_clock::now();
        auto duration_s = std::chrono::duration<double>(te - ts).count();
        auto qps = ds.nq() / duration_s;

        std::vector<uint32_t> sampled_gts(ds.nq() * ds.ngt());
        for (size_t req = 0; req < ds.nq(); ++req)
        {
            const uint32_t *gt_row = ds.gts() + sampled_qids[req] * ds.ngt();
            std::copy(gt_row, gt_row + ds.ngt(), sampled_gts.data() + req * ds.ngt());
        }

        std::cout << "L: " << L
                  << ", duration(s): " << duration_s
                  << ", qps: " << qps
                  << ", workload: zipfian(s=" << zipf_s << ")"
                  << std::endl;

        auto recall = calc_recall(ds.nq(), ds.ngt(), sampled_gts.data(), K, results.data(),
                                  recall_ats.size(), recall_ats.data());
        std::cout << "recall: " << recall << std::endl;
    };

    auto search_one = [&](size_t L, size_t qi)
    {
        if (qi >= ds.nq())
            throw std::out_of_range("query_index out of range");
        qg.set_ef(L);
        std::vector<uint32_t> one_res(K);
        auto ts = std::chrono::high_resolution_clock::now();
        qg.search(ds.query(qi), std::min(L, K), one_res.data());
        auto te = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration<double, std::milli>(te - ts).count();

        const uint32_t *gt_row = ds.gts() + qi * ds.ngt();
        auto recall_one = calc_recall(1, ds.ngt(), gt_row, K, one_res.data(),
                                      recall_ats.size(), recall_ats.data());

        std::cout << "[search_one] L: " << L << ", query=" << qi
                  << ", duration(ms): " << duration_ms
                  << ", recall: " << recall_one << std::endl;
    };

    if (mode == "all")
    {
        for (size_t L : Ls)
        {
            if (do_drop_caches)
                drop_caches();
            search_all_zipfian(L);
        }
    }
    else if (mode == "one")
    {
        for (size_t L : Ls)
        {
            if (do_drop_caches)
                drop_caches();
            search_one(L, query_index);
        }
    }
    else
    {
        throw std::runtime_error("Invalid --mode: " + mode + " (expected all|one)");
    }

    return 0;
}
