#include <immintrin.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <queue>

using PointPair = std::pair<int, float>;

float euclidean_distance_simd(const std::vector<float> &a,
    const std::vector<float> &b) {
    if (a.size() != b.size())
    throw std::runtime_error("Vector dimensions mismatch");

    size_t n = a.size();
    float sum = 0.0f;
    size_t i = 0;

    if (n >= 8) {
        __m256 sum_vec = _mm256_setzero_ps();  
        for (; i <= n - 8; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
        }
        float temp[8];
        _mm256_storeu_ps(temp, sum_vec);
        for (int j = 0; j < 8; ++j) {
            sum += temp[j];
        }
    }

    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;  
}

std::vector<PointPair> exact_knn(
    const std::vector<float> &query,
    const std::vector<std::vector<float>> &base, size_t b_size, int k) {
    using HeapPair = std::pair<float, int>;
    std::priority_queue<HeapPair, std::vector<HeapPair>, std::less<HeapPair>> max_heap;

    for (size_t j = 0; j < b_size && j < base.size(); ++j) {
        float dist = euclidean_distance_simd(query, base[j]);
        if (max_heap.size() < static_cast<size_t>(k)) {
            max_heap.emplace(dist, static_cast<int>(j));
        } else if (dist < max_heap.top().first) {
            max_heap.pop(); 
            max_heap.emplace(dist, static_cast<int>(j));
        }
    }

    std::vector<PointPair> topk;
    topk.reserve(k);
    while (!max_heap.empty()) {
        topk.emplace_back(max_heap.top().second, max_heap.top().first);
        max_heap.pop();
    }
    std::reverse(topk.begin(), topk.end());

    return topk;
}

std::vector<std::vector<PointPair>> compute_batch_groundtruth(
    const std::vector<std::vector<float>> &base,
    const std::vector<std::vector<float>> &queries, size_t b_size, int k) {
    if (!base.empty() && !queries.empty() &&
        base[0].size() != queries[0].size()) {
        throw std::runtime_error("Base and query vector dimensions mismatch");
    }

    std::vector<std::vector<PointPair>> results(queries.size());
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    size_t chunk_size = (queries.size() + num_threads - 1) / num_threads;

    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end && i < queries.size(); ++i) {
            results[i] = exact_knn(queries[i], base, b_size, k);
        }
    };

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, queries.size());
        threads.emplace_back(worker, start, end);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    std::cout << "Computed groundtruth for base size " << b_size << std::endl;
    return results;
}

std::vector<std::vector<float>> read_fvecs(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    std::vector<std::vector<float>> data;
    while (in.good()) {
        int dim;
        in.read(reinterpret_cast<char *>(&dim), sizeof(int));
        if (!in.good()) break;

        std::vector<float> vec_float(dim);
        in.read(reinterpret_cast<char *>(vec_float.data()),
                dim * sizeof(float));

        std::vector<float> vec(dim);
        for (int i = 0; i < dim; ++i) {
            vec[i] = static_cast<float>(
                std::min(std::max(vec_float[i], 0.0f), 255.0f));
        }
        data.push_back(vec);
    }
    in.close();
    std::cout << "Read " << data.size() << " vectors from " << filename
              << std::endl;
    return data;
}

void save_to_bin(
    const std::vector<std::vector<std::vector<PointPair>>> &all_batches,
    const std::string &filename, int k) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Failed to open file: " + filename);

    int n = static_cast<int>(all_batches[0].size());
    int b = static_cast<int>(all_batches.size());
    out.write(reinterpret_cast<const char *>(&n), sizeof(int));
    out.write(reinterpret_cast<const char *>(&k), sizeof(int));
    out.write(reinterpret_cast<const char *>(&b), sizeof(int));

    for (size_t batch_idx = 0; batch_idx < all_batches.size(); ++batch_idx) {
        int current_base_size = static_cast<int>((batch_idx + 1) * 100);
        out.write(reinterpret_cast<const char *>(&current_base_size),
                  sizeof(int));
        if (!out.good()) throw std::runtime_error("write base_size failed at batch " + std::to_string(batch_idx));

        const auto &batch_gt = all_batches[batch_idx];
        for (const auto &result : batch_gt) {
            for (const auto &[id, dist] : result) {
                out.write(reinterpret_cast<const char *>(&id), sizeof(int));
                if (!out.good()) throw std::runtime_error("write query index at batch " + std::to_string(batch_idx));
            }
        }

        size_t dist_count = 0;
        for (const auto &result : batch_gt) {
            for (const auto &[id, dist] : result) {
                out.write(reinterpret_cast<const char *>(&dist), sizeof(float));
                if (!out.good()) throw std::runtime_error("write distance at batch " + std::to_string(batch_idx));
                dist_count++;
            }
        }
    }

    out.close();
    std::cout << "Saved to " << filename << " (queries: " << n << ", k: " << k
              << ", batches: " << b << ")" << std::endl;
}

void compute_and_save_full_groundtruth(
    const std::vector<std::vector<float>> &base,
    const std::vector<std::vector<float>> &queries,
    const std::string &filename, int k) {
    if (!base.empty() && !queries.empty() &&
        base[0].size() != queries[0].size()) {
        throw std::runtime_error("Base and query vector dimensions mismatch");
    }

    std::vector<std::vector<PointPair>> results =
        compute_batch_groundtruth(base, queries, base.size(), k);

    size_t npts = queries.size();  
    size_t ndims = k;             
    std::vector<int32_t> data(npts * ndims);       
    std::vector<float> distances(npts * ndims);    

    for (size_t i = 0; i < npts; ++i) {
        for (size_t j = 0; j < ndims; ++j) {
            size_t idx = i * ndims + j;
            data[idx] = results[i][j].first;      
            distances[idx] = results[i][j].second; 
        }
    }

    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    if (!writer.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    int npts_i32 = static_cast<int>(npts);
    int ndims_i32 = static_cast<int>(ndims);
    writer.write(reinterpret_cast<char *>(&npts_i32), sizeof(int));
    writer.write(reinterpret_cast<char *>(&ndims_i32), sizeof(int));
    std::cout << "Saving full groundtruth in one file (npts, dim, npts*dim id-matrix, "
                 "npts*dim dist-matrix) with npts = "
              << npts << ", dim = " << ndims << ", size = "
              << 2 * npts * ndims * sizeof(int32_t) + 2 * sizeof(int) << "B"
              << std::endl;

    writer.write(reinterpret_cast<char *>(data.data()), npts * ndims * sizeof(int32_t));
    writer.write(reinterpret_cast<char *>(distances.data()), npts * ndims * sizeof(float));
    writer.close();

    std::cout << "Finished writing full groundtruth to " << filename << std::endl;
}

struct Args {
    std::string base_path;
    std::string query_path;
    std::string batch_gt_path;
    std::string gt_path;
    std::string data_type;
    std::string dist_func;
    int k = 100;
};

Args parse_args(int argc, char *argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--base_path" && i + 1 < argc)
            args.base_path = argv[++i];
        else if (arg == "--query_path" && i + 1 < argc)
            args.query_path = argv[++i];
        else if (arg == "--batch_gt_path" && i + 1 < argc)
            args.batch_gt_path = argv[++i];
        else if (arg == "--gt_path" && i + 1 < argc)
            args.gt_path = argv[++i];
        else if (arg == "--data_type" && i + 1 < argc)
            args.data_type = argv[++i];
        else if (arg == "--k" && i + 1 < argc)
            args.k = std::stoi(argv[++i]);
    }
    return args;
}

int main(int argc, char *argv[]) {
    Args args = parse_args(argc, argv);

    if (args.base_path.empty() || args.query_path.empty() || args.k <= 0 || args.data_type.empty() ||
        (args.gt_path.empty() && args.batch_gt_path.empty())) {
        std::cerr << "Error: Missing required arguments (base_path, query_path, data_type, k, and at least one of gt_path or batch_gt_path)" << std::endl;
        return 1;
    }

    std::vector<std::vector<float>> base = read_fvecs(args.base_path);
    std::vector<std::vector<float>> queries = read_fvecs(args.query_path);

    std::cout << "Computing groundtruth for " << args.k << " nearest neighbors" << std::endl;

    if (!args.batch_gt_path.empty()) {
        const int increment = 100;
        size_t total_b = base.size();
        int num_batches = (total_b + increment - 1) / increment;

        std::vector<std::vector<std::vector<PointPair>>> all_batches;
        all_batches.reserve(num_batches);
        for (size_t b_size = increment; b_size <= total_b; b_size += increment) {
            std::cout << "Processing base size " << b_size << " for batch groundtruth" << std::endl;
            auto batch_gt = compute_batch_groundtruth(base, queries, b_size, args.k);
            all_batches.push_back(batch_gt);
        }
        save_to_bin(all_batches, args.batch_gt_path, args.k);
    }

    if (!args.gt_path.empty()) {
        std::cout << "Processing full base size " << base.size() << " for full groundtruth" << std::endl;
        compute_and_save_full_groundtruth(base, queries, args.gt_path, args.k);
    }

    return 0;
}