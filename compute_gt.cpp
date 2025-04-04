#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <immintrin.h>
#include <thread>

using PointPair = std::pair<int, float>;

float euclidean_distance_simd(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size()) throw std::runtime_error("Vector dimensions mismatch");

    size_t n = a.size();
    float sum = 0.0f;
    size_t i = 0;

    if (n >= 32) {
        __m256 sum_vec = _mm256_setzero_ps();
        for (; i <= n - 32; i += 32) {
            __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
            __m256i va_low = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 0));
            __m256i vb_low = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 0));
            __m256i va_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
            __m256i vb_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
            __m256 diff_low = _mm256_cvtepi32_ps(_mm256_sub_epi32(va_low, vb_low));
            __m256 diff_high = _mm256_cvtepi32_ps(_mm256_sub_epi32(va_high, vb_high));
            sum_vec = _mm256_fmadd_ps(diff_low, diff_low, sum_vec);
            sum_vec = _mm256_fmadd_ps(diff_high, diff_high, sum_vec);
        }
        float temp[8];
        _mm256_storeu_ps(temp, sum_vec);
        for (int j = 0; j < 8; ++j) sum += temp[j];
    }

    for (; i < n; ++i) {
        float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        sum += diff * diff;
    }

    return sum;
}

std::vector<PointPair> find_k_nearest(const std::vector<uint8_t>& query, 
                                      const std::vector<std::vector<uint8_t>>& base, 
                                      size_t b_size, int k) {
    std::vector<PointPair> topk;
    for (size_t j = 0; j < b_size && j < base.size(); ++j) {
        float dist = euclidean_distance_simd(query, base[j]);
        if (topk.size() < static_cast<size_t>(k)) {
            topk.emplace_back(static_cast<int>(j), dist);
            std::sort(topk.begin(), topk.end(), [](const PointPair& a, const PointPair& b) {
                return a.second < b.second;
            });
        } else if (dist < topk.back().second) {
            topk.back() = {static_cast<int>(j), dist};
            std::sort(topk.begin(), topk.end(), [](const PointPair& a, const PointPair& b) {
                return a.second < b.second;
            });
        }
    }
    return topk;
}

std::vector<std::vector<PointPair>> compute_batch_groundtruth(
    const std::vector<std::vector<uint8_t>>& base,
    const std::vector<std::vector<uint8_t>>& queries,
    size_t b_size, int k) {
    if (!base.empty() && !queries.empty() && base[0].size() != queries[0].size()) {
        throw std::runtime_error("Base and query vector dimensions mismatch");
    }

    std::vector<std::vector<PointPair>> results(queries.size());
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    size_t chunk_size = (queries.size() + num_threads - 1) / num_threads;

    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end && i < queries.size(); ++i) {
            results[i] = find_k_nearest(queries[i], base, b_size, k);
        }
    };

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, queries.size());
        threads.emplace_back(worker, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "Computed groundtruth for base size " << b_size << std::endl;
    return results;
}

std::vector<std::vector<uint8_t>> read_fvecs(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::vector<std::vector<uint8_t>> data;
    while (in.good()) {
        int dim;
        in.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!in.good()) break;

        std::vector<float> vec_float(dim);
        in.read(reinterpret_cast<char*>(vec_float.data()), dim * sizeof(float));

        std::vector<uint8_t> vec(dim);
        for (int i = 0; i < dim; ++i) {
            vec[i] = static_cast<uint8_t>(std::min(std::max(vec_float[i], 0.0f), 255.0f));
        }
        data.push_back(vec);
    }
    in.close();
    std::cout << "Read " << data.size() << " vectors from " << filename << std::endl;
    return data;
}

void save_to_bin(const std::vector<std::vector<std::vector<PointPair>>>& all_batches, 
                 const std::string& filename, int k) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Failed to open file: " + filename);

    int n = static_cast<int>(all_batches[0].size());
    int b = static_cast<int>(all_batches.size());
    out.write(reinterpret_cast<const char*>(&n), sizeof(int));
    out.write(reinterpret_cast<const char*>(&k), sizeof(int));
    out.write(reinterpret_cast<const char*>(&b), sizeof(int));

    for (size_t batch_idx = 0; batch_idx < all_batches.size(); ++batch_idx) {
        int current_base_size = static_cast<int>((batch_idx + 1) * 100);
        out.write(reinterpret_cast<const char*>(&current_base_size), sizeof(int));

        const auto& batch_gt = all_batches[batch_idx];
        for (const auto& result : batch_gt) {
            for (const auto& [id, dist] : result) {
                out.write(reinterpret_cast<const char*>(&id), sizeof(int));
            }
        }
        for (const auto& result : batch_gt) {
            for (const auto& [id, dist] : result) {
                out.write(reinterpret_cast<const char*>(&dist), sizeof(float));
            }
        }
    }
    out.close();
    std::cout << "Saved to " << filename << " (queries: " << n << ", k: " << k << ", batches: " << b << ")" << std::endl;
}

struct Args {
    std::string base_path;
    std::string query_path;
    std::string gt_path;
    std::string data_type;
    std::string dist_func;
    int k = 100;
};

Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-base_path" && i + 1 < argc) args.base_path = argv[++i];
        else if (arg == "-query_path" && i + 1 < argc) args.query_path = argv[++i];
        else if (arg == "-gt_path" && i + 1 < argc) args.gt_path = argv[++i];
        else if (arg == "-data_type" && i + 1 < argc) args.data_type = argv[++i];
        else if (arg == "-k" && i + 1 < argc) args.k = std::stoi(argv[++i]);
    }
    return args;
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    if (args.base_path.empty() || args.query_path.empty() || args.gt_path.empty()) {
        std::cerr << "Error: Missing required arguments" << std::endl;
        return 1;
    }

    std::vector<std::vector<uint8_t>> base = read_fvecs(args.base_path);
    std::vector<std::vector<uint8_t>> queries = read_fvecs(args.query_path);

    const int increment = 100;
    size_t total_b = base.size();
    int num_batches = (total_b + increment - 1) / increment;

    std::cout << "Computing groundtruth for " << args.k << " nearest neighbors" << std::endl;

    std::vector<std::vector<std::vector<PointPair>>> all_batches;
    all_batches.reserve(num_batches);
    for (size_t b_size = increment; b_size <= total_b; b_size += increment) {
        std::cout << "Processing base size " << b_size << std::endl;
        auto batch_gt = compute_batch_groundtruth(base, queries, b_size, args.k);
        all_batches.push_back(batch_gt);
    }

    save_to_bin(all_batches, args.gt_path, args.k);

    return 0;
}