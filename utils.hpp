#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename TagT>
struct SearchResult {
    size_t insert_offset;
    size_t query_idx;
    std::vector<TagT> tags;
    std::vector<float> distances;

    SearchResult(size_t offset, size_t idx, const std::vector<TagT> &t)
        : insert_offset(offset), query_idx(idx), tags(t), distances() {}

    SearchResult(size_t offset, size_t idx, const std::vector<TagT> &t,
                 const std::vector<float> &d)
        : insert_offset(offset), query_idx(idx), tags(t), distances(d) {}
};

std::string to_string_with_precision(float value, int precision = 2) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    std::string str = ss.str();

    if (str.empty()) {
        return std::to_string(value);
    }

    size_t pos = str.find_last_not_of('0');
    if (pos != std::string::npos) {
        str.erase(pos + 1, std::string::npos);
    }

    if (!str.empty() && str.back() == '.') {
        str.pop_back();
    }
    return str;
}

struct Stat {
    std::string index_name;
    uint32_t num_points;
    uint32_t R;
    uint32_t Ls;
    uint32_t Lb;
    float alpha = 1.2;
    uint32_t num_threads;
    std::string dataset_name;
    uint32_t batch_size;

    float write_ratio;
    double insert_qps;
    double mean_insert_latency;
    double p95_insert_latency;
    double p99_insert_latency;

    double search_qps;
    double mean_search_latency;
    double p95_search_latency;
    double p99_search_latency;

    float overall_recall_at_10;
    std::string stagewise_result_path;

    Stat(std::string idx_name, std::string ds_name, uint32_t r, uint32_t lb,
         uint32_t ls, float wr, uint32_t threads, uint32_t batch_size,
         std::string res_path)
        : index_name(idx_name),
          dataset_name(ds_name),
          R(r),
          Lb(lb),
          Ls(ls),
          num_threads(threads),
          write_ratio(wr),
          alpha(1.2f),
          batch_size(100),
          stagewise_result_path(
              res_path + "/" + index_name + "_" + dataset_name + "_R" +
              std::to_string(r) + "_Lb" + std::to_string(lb) + "_Ls" +
              std::to_string(ls) + "_w" + to_string_with_precision(wr) + "_t" +
              std::to_string(threads) + ".res") {}
};

void read_results(std::vector<SearchResult<uint32_t>> &res,
                  const std::string &res_path) {
    res.clear();

    std::ifstream in_file(res_path);
    if (!in_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + res_path);
    }

    std::string line;
    size_t current_offset = 0;
    bool first_batch = true;

    while (std::getline(in_file, line)) {
        if (line.empty()) {
            continue;
        }

        if (line.find("batch") == 0) {
            std::istringstream batch_stream(line);
            std::string batch_word;
            batch_stream >> batch_word;
            if (!(batch_stream >> current_offset)) {
                in_file.close();
                throw std::runtime_error("Invalid batch offset format: " +
                                         line);
            }
            continue;
        }

        size_t query_idx;
        std::istringstream iss(line);
        if (!(iss >> query_idx)) {
            in_file.close();
            throw std::runtime_error("Invalid query_idx format at offset " +
                                     std::to_string(current_offset));
        }

        if (!std::getline(in_file, line)) {
            in_file.close();
            throw std::runtime_error("Missing tags line at offset " +
                                     std::to_string(current_offset));
        }

        std::vector<uint32_t> tags;
        std::istringstream tags_stream(line);
        uint32_t tag;
        while (tags_stream >> tag) {
            tags.push_back(tag);
        }

        res.emplace_back(current_offset, query_idx, tags);
    }

    in_file.close();
}

void write_results(std::vector<SearchResult<uint32_t>> &res,
                   const std::string &res_path) {
    std::sort(
        res.begin(), res.end(),
        [](const SearchResult<uint32_t> &a, const SearchResult<uint32_t> &b) {
            return a.insert_offset < b.insert_offset;
        });

    std::ofstream out_file(res_path);
    if (!out_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + res_path);
    }

    size_t current_offset = res.empty() ? 0 : res[0].insert_offset;
    bool first_batch = true;

    for (const auto &result : res) {
        if (result.insert_offset != current_offset) {
            current_offset = result.insert_offset;
            out_file << "\nbatch " << current_offset << "\n";
        } else if (first_batch) {
            out_file << "batch " << current_offset << "\n";
            first_batch = false;
        }

        out_file << result.query_idx << "\n";

        for (size_t i = 0; i < result.tags.size(); ++i) {
            out_file << result.tags[i];
            if (i < result.tags.size() - 1) {
                out_file << " ";
            }
        }
        out_file << "\n";
    }

    out_file.close();
}

template <typename TagT = uint32_t>
void load_gt(std::vector<SearchResult<TagT>> &gt, const std::string &gt_path) {
    std::ifstream in(gt_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file: " + gt_path);
    }

    int n, k, b;
    in.read(reinterpret_cast<char *>(&n), sizeof(int));
    in.read(reinterpret_cast<char *>(&k), sizeof(int));
    in.read(reinterpret_cast<char *>(&b), sizeof(int));

    if (n <= 0 || k <= 0 || b <= 0) {
        throw std::runtime_error(
            "Invalid file header: n, k, or b is non-positive");
    }

    gt.clear();
    gt.reserve(n * b);

    for (int batch_idx = 0; batch_idx < b; ++batch_idx) {
        int current_base_size;
        in.read(reinterpret_cast<char *>(&current_base_size), sizeof(int));
        if (!in.good()) {
            throw std::runtime_error("Failed to read base size for batch " +
                                     std::to_string(batch_idx));
        }

        std::vector<int> indices(n * k);
        in.read(reinterpret_cast<char *>(indices.data()), n * k * sizeof(int));
        if (!in.good()) {
            throw std::runtime_error("Failed to read indices for batch " +
                                     std::to_string(batch_idx));
        }

        std::vector<float> distances(n * k);
        in.read(reinterpret_cast<char *>(distances.data()),
                n * k * sizeof(float));
        if (!in.good()) {
            throw std::runtime_error("Failed to read distances for batch " +
                                     std::to_string(batch_idx));
        }

        for (int query_idx = 0; query_idx < n; ++query_idx) {
            std::vector<TagT> tags(k);
            std::vector<float> query_distances;
            for (int j = 0; j < k; ++j) {
                size_t offset = query_idx * k + j;
                tags[j] = indices[offset];
                query_distances.push_back(distances[offset]);
            }
            gt.emplace_back(current_base_size, query_idx, tags,
                            query_distances);
        }
    }

    in.close();
    std::cout << "Loaded " << gt.size() << " search results from " << gt_path
              << " (queries: " << n << ", k: " << k << ", batches: " << b << ")"
              << std::endl;
}

template <typename T>
inline void load_aligned_bin(const std::string &bin_file, T *&data,
                             size_t &npts, size_t &dim, size_t &rounded_dim) {
    std::ifstream reader;
    reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
        std::cout << "Reading bin file " << bin_file << " ..." << std::flush;
        reader.open(bin_file, std::ios::binary | std::ios::ate);

        uint64_t actual_file_size = reader.tellg();
        reader.seekg(0);

        int npts_i32, dim_i32;
        reader.read((char *)&npts_i32, sizeof(int));
        reader.read((char *)&dim_i32, sizeof(int));
        npts = static_cast<size_t>(npts_i32);
        dim = static_cast<size_t>(dim_i32);

        size_t expected_file_size =
            npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
        if (actual_file_size != expected_file_size) {
            std::stringstream stream;
            stream << "Error: File size mismatch. Actual size is "
                   << actual_file_size << ", expected size is "
                   << expected_file_size << " (npts = " << npts
                   << ", dim = " << dim << ", sizeof(T) = " << sizeof(T) << ")";
            throw std::runtime_error(stream.str());
        }

        rounded_dim = (dim + 7) & ~7;
        std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim
                  << ", aligned_dim = " << rounded_dim << "... " << std::flush;

        size_t alloc_size = npts * rounded_dim * sizeof(T);
        std::cout << "Allocating memory of " << alloc_size << " bytes... "
                  << std::flush;
        data = new T[npts * rounded_dim];
        std::cout << "done. Copying data..." << std::flush;

        for (size_t i = 0; i < npts; i++) {
            reader.read((char *)(data + i * rounded_dim), dim * sizeof(T));
            std::memset(data + i * rounded_dim + dim, 0,
                        (rounded_dim - dim) * sizeof(T));
        }
        std::cout << " done." << std::endl;
    } catch (const std::ios_base::failure &e) {
        std::stringstream stream;
        stream << "Failed to read file " << bin_file << ": " << e.what();
        throw std::runtime_error(stream.str());
    } catch (const std::exception &e) {
        throw;
    }
}

void get_bin_metadata(const std::string &filename, size_t &num_points,
                      size_t &dimensions, size_t offset = 0) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    file.seekg(offset, std::ios::beg);
    if (file.fail()) {
        file.close();
        throw std::runtime_error("Failed to seek to offset: " +
                                 std::to_string(offset));
    }

    uint32_t metadata[2];
    file.read(reinterpret_cast<char *>(metadata), 2 * sizeof(uint32_t));
    if (file.gcount() != 2 * sizeof(uint32_t)) {
        file.close();
        throw std::runtime_error("Failed to read metadata at offset: " +
                                 std::to_string(offset));
    }

    num_points = metadata[0];
    dimensions = metadata[1];
    std::cout << "File " << filename << " contains " << num_points
              << " points and " << dimensions << " dimensions at offset "
              << offset << std::endl;

    file.close();
}

void save_stat(Stat &stat, std::string stat_path) {
    std::ifstream check_file(stat_path);
    bool file_exists = check_file.good();
    check_file.close();
    std::cout << "Stat: " << stat_path << std::endl;

    std::ofstream file(stat_path, std::ios::app);
    if (!file.is_open()) {
        file.open(stat_path, std::ios::out);
        if (!file.is_open()) {
            return;
        }
    }

    if (!file_exists) {
        file << "index_name,num_points,R,Lb,Ls,alpha,num_threads,dataset_name,"
                "batch_size,"
             << "write_ratio,insert_qps,insert_mean_latency,insert_p95_latency,"
             << "insert_p99_latency,search_qps,search_mean_latency,search_p95_"
                "latency,"
             << "search_p99_latency,overall_recall_at_10(%),stagewise_result_"
                "path,"
                "stagewise_recall_path\n";
    }

    std::ostringstream ss;
    ss << stat.index_name << "," << stat.num_points << "," << stat.R << ","
       << stat.Lb << "," << stat.Ls << "," << stat.alpha << ","
       << stat.num_threads << "," << stat.dataset_name << "," << stat.batch_size
       << "," << stat.write_ratio << "," << stat.insert_qps << ","
       << stat.mean_insert_latency << "," << stat.p95_insert_latency << ","
       << stat.p99_insert_latency << "," << stat.search_qps << ","
       << stat.mean_search_latency << "," << stat.p95_search_latency << ","
       << stat.p99_search_latency << "," << stat.overall_recall_at_10 << ","
       << stat.stagewise_result_path;
    file << ss.str() << "\n";
    file.close();
}

double calculate_recall(uint32_t num_queries, uint32_t *gold_std,
                        float *gs_dist, uint32_t dim_gs, uint32_t *our_results,
                        uint32_t dim_or, uint32_t recall_at) {
    double total_recall = 0;
    std::set<uint32_t> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
        gt.clear();
        res.clear();
        uint32_t *gt_vec = gold_std + dim_gs * i;
        uint32_t *res_vec = our_results + dim_or * i;
        size_t tie_breaker = recall_at;
        if (gs_dist != nullptr) {
            tie_breaker = recall_at - 1;
            float *gt_dist_vec = gs_dist + dim_gs * i;
            while (tie_breaker < dim_gs &&
                   gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
                tie_breaker++;
        }

        gt.insert(gt_vec, gt_vec + tie_breaker);
        res.insert(res_vec, res_vec + recall_at);
        uint32_t cur_recall = 0;
        for (auto &v : gt) {
            if (res.find(v) != res.end()) {
                cur_recall++;
            }
        }
        total_recall += cur_recall;
    }
    return total_recall / num_queries * (100.0 / recall_at);
}