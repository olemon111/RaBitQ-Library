// Print and verify SymQG on-disk layout (meta + data split).
// Reference: docs/symqg_layout_diagram.md

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/utils/rotator.hpp"

namespace fs = std::filesystem;
using rabitqlib::MetricType;
using rabitqlib::PID;
using rabitqlib::RotatorType;

struct IndexHeader {
    size_t num_points{};
    size_t degree_bound{};
    size_t dim{};
    size_t padded_dim{};
    PID entry_point{};
    RotatorType rotator_type{};
    MetricType metric_type{};
};

std::string metric_to_string(MetricType m) {
    switch (m) {
        case rabitqlib::METRIC_L2:
            return "L2";
        case rabitqlib::METRIC_IP:
            return "InnerProduct";
        default:
            return "Unknown";
    }
}

std::string rotator_to_string(RotatorType r) {
    switch (r) {
        case RotatorType::FhtKacRotator:
            return "FhtKacRotator";
        case RotatorType::MatrixRotator:
            return "MatrixRotator";
        default:
            return "Unknown";
    }
}

std::string human_bytes(double bytes) {
    const char* suffixes[] = {"B", "KB", "MB", "GB"};
    int idx = 0;
    while (bytes >= 1024.0 && idx < 3) {
        bytes /= 1024.0;
        ++idx;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << bytes << ' ' << suffixes[idx];
    return oss.str();
}

bool read_header(const std::string& meta_path, IndexHeader& h) {
    std::ifstream meta(meta_path, std::ios::binary);
    if (!meta.is_open()) {
        std::cerr << "Failed to open meta file: " << meta_path << '\n';
        return false;
    }
    meta.read(reinterpret_cast<char*>(&h.num_points), sizeof(size_t));
    meta.read(reinterpret_cast<char*>(&h.degree_bound), sizeof(size_t));
    meta.read(reinterpret_cast<char*>(&h.dim), sizeof(size_t));
    meta.read(reinterpret_cast<char*>(&h.padded_dim), sizeof(size_t));
    meta.read(reinterpret_cast<char*>(&h.entry_point), sizeof(PID));
    meta.read(reinterpret_cast<char*>(&h.rotator_type), sizeof(RotatorType));
    meta.read(reinterpret_cast<char*>(&h.metric_type), sizeof(MetricType));
    if (!meta) {
        std::cerr << "Failed to read full header (file too small?)\n";
        return false;
    }
    // Header layout (on-disk metadata, in order):
    //   num_points   : size_t       (count of data points)
    //   degree_bound : size_t       (max out-degree per node, multiple of 32)
    //   dim          : size_t       (original vector dimension)
    //   padded_dim   : size_t       (dimension padded to multiple of 64)
    //   entry_point  : PID          (graph search entry node id)
    //   rotator_type : RotatorType  (which Rotator implementation is used)
    //   metric_type  : MetricType   (L2 or InnerProduct)
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <meta_file> <data_file>\n";
        return 1;
    }

    const std::string meta_path = argv[1];
    const std::string data_path = argv[2];

    if (!fs::exists(meta_path) || !fs::exists(data_path)) {
        std::cerr << "Meta or data file does not exist.\n";
        return 1;
    }

    IndexHeader h{};
    if (!read_header(meta_path, h)) {
        return 1;
    }

    constexpr size_t kBatch = rabitqlib::fastscan::kBatchSize;
    const size_t batches = (h.degree_bound + kBatch - 1) / kBatch;
    const bool aligned_degree = (h.degree_bound % kBatch) == 0;

    const size_t bin_code_bytes = (h.padded_dim * kBatch) / 8;
    const size_t f_add_bytes = sizeof(float) * kBatch;
    const size_t f_rescale_bytes = sizeof(float) * kBatch;
    const size_t per_batch_bytes = rabitqlib::QGBatchDataMap<float>::data_bytes(h.padded_dim);

    const size_t raw_vec_bytes = h.dim * sizeof(float);
    const size_t batch_region_bytes = per_batch_bytes * batches;
    const size_t neighbor_bytes = h.degree_bound * sizeof(PID);
    const size_t row_offset = raw_vec_bytes + batch_region_bytes + neighbor_bytes;
    const size_t data_region_bytes = row_offset * h.num_points;

    const size_t header_bytes =
        4 * sizeof(size_t) + sizeof(PID) + sizeof(RotatorType) + sizeof(MetricType);

    size_t rotator_bytes = 0;
    if (h.rotator_type == RotatorType::FhtKacRotator) {
        rotator_bytes = h.padded_dim / 2;  // flip_ size = 4 * padded_dim / 8
    } else if (h.rotator_type == RotatorType::MatrixRotator) {
        rotator_bytes = h.dim * h.padded_dim * sizeof(float);
    }

    const size_t expected_meta_min = header_bytes + rotator_bytes;
    const size_t expected_data = data_region_bytes;

    const size_t meta_size = fs::file_size(meta_path);
    const size_t data_size = fs::file_size(data_path);

    const long long meta_gap =
        static_cast<long long>(meta_size) - static_cast<long long>(expected_meta_min);
    const long long data_gap =
        static_cast<long long>(data_size) - static_cast<long long>(expected_data);

    std::cout << "SymQG On-Disk Layout Inspector\n";
    std::cout << "Meta file          : " << meta_path << " (" << meta_size << " bytes)\n";
    std::cout << "Data file          : " << data_path << " (" << data_size << " bytes)\n\n";

    // Detailed metadata layout (header) with offsets and meanings.
    std::cout << "Header (metadata layout)\n";
    std::cout << "  total header bytes : " << header_bytes << " ("
              << human_bytes(static_cast<double>(header_bytes)) << ")\n";
    std::cout << "  fields (in file order):\n";
    std::cout << "    [0.." << sizeof(size_t) - 1
              << "] num_points   (size_t, count of points)            = " << h.num_points
              << '\n';
    std::cout << "    [" << sizeof(size_t) << ".." << 2 * sizeof(size_t) - 1
              << "] degree_bound (size_t, out-degree per node)        = " << h.degree_bound
              << '\n';
    std::cout << "    [" << 2 * sizeof(size_t) << ".." << 3 * sizeof(size_t) - 1
              << "] dim          (size_t, original dimension)         = " << h.dim << '\n';
    std::cout << "    [" << 3 * sizeof(size_t) << ".." << 4 * sizeof(size_t) - 1
              << "] padded_dim   (size_t, padded dimension)           = " << h.padded_dim
              << '\n';
    const size_t entry_off = 4 * sizeof(size_t);
    const size_t rot_off = entry_off + sizeof(PID);
    const size_t metric_off = rot_off + sizeof(RotatorType);
    std::cout << "    [" << entry_off << ".." << entry_off + sizeof(PID) - 1
              << "] entry_point  (PID, start node id)                 = " << h.entry_point
              << '\n';
    std::cout << "    [" << rot_off << ".." << rot_off + sizeof(RotatorType) - 1
              << "] rotator_type (uint8, Rotator implementation)     = "
              << rotator_to_string(h.rotator_type) << '\n';
    std::cout << "    [" << metric_off << ".." << metric_off + sizeof(MetricType) - 1
              << "] metric_type  (uint8, distance metric)            = "
              << metric_to_string(h.metric_type) << "\n\n";

    // Concise header summary (values only), convenient for SIFT1M etc.
    std::cout << "Header (concise values)\n";
    std::cout << "  num_points       : " << h.num_points << '\n';
    std::cout << "  dim              : " << h.dim << '\n';
    std::cout << "  padded_dim       : " << h.padded_dim << '\n';
    std::cout << "  degree_bound     : " << h.degree_bound
              << (aligned_degree ? "" : "  (WARN: not multiple of 32)") << '\n';
    std::cout << "  metric           : " << metric_to_string(h.metric_type) << '\n';
    std::cout << "  rotator          : " << rotator_to_string(h.rotator_type) << '\n';
    std::cout << "  entry_point      : " << h.entry_point << "\n\n";

    std::cout << "Per-batch layout (kBatchSize=" << kBatch << ")\n";
    std::cout << "  bin_code bytes   : " << bin_code_bytes << '\n';
    std::cout << "  f_add bytes      : " << f_add_bytes << '\n';
    std::cout << "  f_rescale bytes  : " << f_rescale_bytes << '\n';
    std::cout << "  per_batch bytes  : " << per_batch_bytes << "\n\n";

    std::cout << "Per-vertex layout\n";
    std::cout << "  raw vector       : " << raw_vec_bytes << " ("
              << human_bytes(static_cast<double>(raw_vec_bytes)) << ")\n";
    std::cout << "  batch data       : " << batch_region_bytes << " (" << batches
              << " batches)\n";
    std::cout << "  neighbors        : " << neighbor_bytes << '\n';
    std::cout << "  row_offset       : " << row_offset << " ("
              << human_bytes(static_cast<double>(row_offset)) << ")\n\n";

    std::cout << "File-level checks\n";
    std::cout << "  expected meta >= : " << expected_meta_min << " (header + rotator)\n";
    std::cout << "  meta gap         : " << meta_gap
              << " (positive means extra bytes/alignment)\n";
    std::cout << "  expected data    : " << expected_data << " (num_points * row_offset)\n";
    std::cout << "  data gap         : " << data_gap
              << " (should be 0; positive means extra padding)\n\n";

    bool ok = true;
    if (data_gap < 0) {
        std::cerr << "[ERROR] Data file smaller than expected layout by " << -data_gap
                  << " bytes.\n";
        ok = false;
    }
    if (meta_gap < 0) {
        std::cerr << "[ERROR] Meta file smaller than expected header+rotator by "
                  << -meta_gap << " bytes.\n";
        ok = false;
    }
    if (!aligned_degree) {
        std::cerr << "[WARN ] degree_bound not multiple of 32; layout assumes padding.\n";
    }

    if (ok) {
        std::cout << "Layout check: PASS (sizes are consistent with derived layout).\n";
    } else {
        std::cout << "Layout check: FAIL (see errors above).\n";
    }

    // Optional: peek first vertex neighbor IDs to ensure offsets map correctly.
    if (data_size >= row_offset) {
        // Map first row from data file into memory (portable read).
        std::ifstream data_in(data_path, std::ios::binary);
        data_in.seekg(0, std::ios::beg);
        std::vector<char> first_row(row_offset);
        data_in.read(first_row.data(), static_cast<std::streamsize>(row_offset));
        if (data_in) {
            const char* base = first_row.data();
            const PID* nbrs =
                reinterpret_cast<const PID*>(base + (raw_vec_bytes + batch_region_bytes));
            std::cout << "First vertex neighbors (first 8 IDs): ";
            size_t show = std::min<size_t>(8, h.degree_bound);
            for (size_t i = 0; i < show; ++i) {
                std::cout << nbrs[i] << (i + 1 == show ? "" : " ");
            }
            std::cout << '\n';
        }
    }

    return ok ? 0 : 2;
}
