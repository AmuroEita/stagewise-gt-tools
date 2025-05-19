#include <omp.h>
#include <papi.h>
#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include "algorithms/hnsw.hpp"
#include "utils.hpp"

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
bool concurrent_bench(const std::string &data_path,
                      const std::string &query_file, const size_t begin_num,
                      const float write_ratio, const size_t batch_size,
                      const uint32_t recall_at, const uint32_t Ls,
                      const uint32_t num_threads,
                      std::unique_ptr<IndexBase<T, TagT, LabelT>> &&index,
                      std::vector<SearchResult<TagT>> &search_results,
                      Stat &stat, bool query_new_data = false) {
    std::cout << "Starting concurrent benchmarking with #threads: "
              << num_threads << " #ratio: " << write_ratio << ":"
              << 1 - write_ratio << std::endl;

    omp_set_num_threads(num_threads);

    size_t data_num, data_dim, aligned_dim;
    get_bin_metadata(data_path, data_num, data_dim);
    T *raw_data = nullptr;
    load_aligned_bin(data_path, raw_data, data_num, data_dim, aligned_dim);
    auto data = std::unique_ptr<T[]>(raw_data);

    size_t query_num, query_dim, query_aligned_dim;
    T *raw_query = nullptr;
    load_aligned_bin(query_file, raw_query, query_num, query_dim,
                     query_aligned_dim);
    auto query = std::unique_ptr<T[]>(raw_query);

    std::vector<uint32_t> tags(begin_num);
    std::iota(tags.begin(), tags.end(), static_cast<uint32_t>(0));
    index->build(data.get(), begin_num, tags);

    size_t insert_total = data_num - begin_num;
    size_t search_total = insert_total * ((1 - write_ratio) / write_ratio);
    size_t search_batch_size = batch_size * ((1 - write_ratio) / write_ratio);
    size_t start_insert_offset = 0, end_insert_offset = 0,
           start_search_offset = 0, end_search_offset = 0, query_idx = 0;

    std::exception_ptr last_exception = nullptr;
    std::mutex last_except_mutex, result_mutex, insert_latency_mutex,
        search_latency_mutex;
    std::vector<double> insert_latency_stats, search_latency_stats;

    auto succeed_insert_count = std::make_shared<std::atomic<size_t>>(0);
    auto failed_insert_count = std::make_shared<std::atomic<size_t>>(0);
    auto succeed_search_count = std::make_shared<std::atomic<size_t>>(0);
    auto failed_search_count = std::make_shared<std::atomic<size_t>>(0);

    auto st = std::chrono::high_resolution_clock::now();
    while (end_insert_offset < insert_total ||
           end_search_offset < search_total) {
        end_insert_offset =
            std::min(start_insert_offset + batch_size, insert_total);
        std::cout << "Inserting with insert_offset="
                  << begin_num + end_insert_offset << std::endl;

        std::vector<T *> batch_data;
        std::vector<TagT> batch_tags;
        for (size_t idx = start_insert_offset; idx < end_insert_offset; ++idx) {
            batch_data.push_back(&data.get()[(idx + begin_num) * aligned_dim]);
            batch_tags.push_back(static_cast<TagT>(idx + begin_num));
        }

        auto qs = std::chrono::high_resolution_clock::now();
        int insert_result = index->batch_insert(batch_data, batch_tags);
        if (insert_result != 0)
            failed_insert_count->fetch_add(batch_data.size(),
                                           std::memory_order_seq_cst);
        else
            succeed_insert_count->fetch_add(batch_data.size(),
                                            std::memory_order_seq_cst);

        auto qe = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = qe - qs;
        {
            std::unique_lock<std::mutex> lock(insert_latency_mutex);
            insert_latency_stats.push_back((float)(diff.count() * 1000000));
        }

        if (query_new_data) {
            size_t new_search_batch_size =
                batch_size * ((1 - write_ratio) / write_ratio);
            std::vector<T *> new_batch_queries;
            std::vector<size_t> new_batch_query_indices;

            std::vector<size_t> indices(batch_size);
            std::iota(indices.begin(), indices.end(), start_insert_offset);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);

            for (size_t i = 0; i < new_search_batch_size && i < batch_size;
                 ++i) {
                size_t idx = indices[i];
                new_batch_queries.push_back(
                    &data.get()[(idx + begin_num) * aligned_dim]);
                new_batch_query_indices.push_back(idx + begin_num);
            }

            auto new_search_qs = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<TagT>> new_batch_results;
            new_batch_results.reserve(new_batch_queries.size());
            index->batch_search(new_batch_queries, recall_at, Ls,
                                new_batch_results);

            auto new_search_qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> new_search_diff =
                new_search_qe - new_search_qs;
            {
                std::unique_lock<std::mutex> lock(search_latency_mutex);
                search_latency_stats.push_back(
                    (float)(new_search_diff.count() * 1000000));
            }
            {
                std::unique_lock<std::mutex> lock(result_mutex);
                for (size_t i = 0; i < new_batch_results.size(); ++i) {
                    search_results.emplace_back(begin_num + end_insert_offset,
                                                new_batch_query_indices[i],
                                                new_batch_results[i]);
                }
            }
        }

        start_insert_offset = end_insert_offset;

        if (!query_new_data) {
            end_search_offset =
                std::min(start_search_offset + search_batch_size, search_total);
            size_t cur_offset = begin_num + end_insert_offset;
            std::cout << "Searching with search_offset=" << cur_offset
                      << std::endl;

            std::vector<T *> batch_queries;
            std::vector<size_t> batch_query_indices;
            for (size_t idx = start_search_offset; idx < end_search_offset;
                 ++idx) {
                if (++query_idx >= query_num) query_idx %= query_num;
                batch_queries.push_back(query.get() +
                                        query_idx * query_aligned_dim);
                batch_query_indices.push_back(query_idx);
            }

            auto search_qs = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<TagT>> batch_results;
            batch_results.reserve(batch_queries.size());
            index->batch_search(batch_queries, recall_at, Ls, batch_results);

            auto search_qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> search_diff = search_qe - search_qs;
            {
                std::unique_lock<std::mutex> lock(search_latency_mutex);
                search_latency_stats.push_back(
                    (float)(search_diff.count() * 1000000));
            }
            {
                std::unique_lock<std::mutex> lock(result_mutex);
                for (size_t i = 0; i < batch_results.size(); ++i) {
                    search_results.emplace_back(
                        cur_offset, batch_query_indices[i], batch_results[i]);
                }
            }

            start_search_offset = end_search_offset;
        }
    }

    auto et = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(et - st).count();
    double insert_qps = insert_total / elapsed_sec;
    double search_qps = search_total / elapsed_sec;

    std::sort(insert_latency_stats.begin(), insert_latency_stats.end());
    double mean_insert_latency =
        std::accumulate(insert_latency_stats.begin(),
                        insert_latency_stats.end(), 0.0) /
        static_cast<float>(insert_total);
    double p95_insert_latency =
        insert_latency_stats.empty()
            ? 0.0
            : (float)insert_latency_stats[(uint64_t)(0.95 * insert_total)];
    double p99_insert_latency =
        insert_latency_stats.empty()
            ? 0.0
            : (float)insert_latency_stats[(uint64_t)(0.999 * insert_total)];

    std::sort(search_latency_stats.begin(), search_latency_stats.end());
    double mean_search_latency =
        std::accumulate(search_latency_stats.begin(),
                        search_latency_stats.end(), 0.0) /
        static_cast<float>(search_total);
    double p95_search_latency =
        search_latency_stats.empty()
            ? 0.0
            : (float)search_latency_stats[(uint64_t)(0.95 * search_total)];
    double p99_search_latency =
        search_latency_stats.empty()
            ? 0.0
            : (float)search_latency_stats[(uint64_t)(0.999 * search_total)];

    stat.num_points = data_num;
    stat.insert_qps = insert_qps;
    stat.mean_insert_latency = mean_insert_latency;
    stat.p95_insert_latency = p95_insert_latency;
    stat.p99_insert_latency = p99_insert_latency;
    stat.search_qps = search_qps;
    stat.mean_search_latency = mean_search_latency;
    stat.p95_search_latency = p95_search_latency;
    stat.p99_search_latency = p99_search_latency;

    std::cout << "Total time: " << elapsed_sec << " seconds\n"
              << "Insertion Statistics:\n"
              << "  Overall throughput: " << insert_qps << " points/second\n"
              << "  Mean latency: " << mean_insert_latency << " us\n"
              << "  P95 latency: " << p95_insert_latency << " us\n"
              << "  P99 latency: " << p99_insert_latency << " us\n"
              << "Search Statistics:\n"
              << "  Overall throughput: " << search_qps << " points/second\n"
              << "  Mean latency: " << mean_search_latency << " us\n"
              << "  P95 latency: " << p95_search_latency << " us\n"
              << "  P99 latency: " << p99_search_latency << " us\n";

    return true;
}

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
bool overall_recall(const std::string &query_file, const uint32_t recall_at,
                    const uint32_t Ls,
                    std::unique_ptr<IndexBase<T, TagT, LabelT>> &&index,
                    const std::string &gt_path, Stat &stat) {
    size_t query_num, query_dim, query_aligned_dim;
    T *raw_query = nullptr;
    load_aligned_bin(query_file, raw_query, query_num, query_dim,
                     query_aligned_dim);
    auto query = std::unique_ptr<T[]>(raw_query);

    std::ifstream gt_reader(gt_path, std::ios::binary);
    if (!gt_reader.is_open()) {
        std::cerr << "Failed to open ground truth file: " << gt_path
                  << std::endl;
        return false;
    }

    int gt_npts, gt_k;
    gt_reader.read(reinterpret_cast<char *>(&gt_npts), sizeof(int));
    gt_reader.read(reinterpret_cast<char *>(&gt_k), sizeof(int));

    std::vector<TagT> gt_ids(gt_npts * gt_k);
    std::vector<float> gt_distances(gt_npts * gt_k);
    gt_reader.read(reinterpret_cast<char *>(gt_ids.data()),
                   gt_npts * gt_k * sizeof(TagT));
    gt_reader.read(reinterpret_cast<char *>(gt_distances.data()),
                   gt_npts * gt_k * sizeof(float));
    gt_reader.close();

    uint64_t total_correct = 0;
    double total_recall = 0.0;
    for (uint32_t i = 0; i < query_num; i++) {
        std::vector<TagT> query_result_tags;
        query_result_tags.reserve(recall_at);
        index->search_with_tags(query.get() + i * query_aligned_dim, recall_at,
                                Ls, query_result_tags);

        double query_recall =
            calculate_recall(1, gt_ids.data() + i * gt_k, gt_distances.data(),
                             gt_k, query_result_tags.data(), gt_k, recall_at);
        total_recall += query_recall;
    }

    stat.overall_recall_at_10 = total_recall / query_num;
    std::cout << "Recall@" << recall_at << " = " << stat.overall_recall_at_10
              << "%" << std::endl;

    return true;
}

void handle_PAPI_error(int retval) {
    if (retval != PAPI_OK) {
        std::cerr << "PAPI error: " << PAPI_strerror(retval) << std::endl;
        exit(1);
    }
}

long get_peak_memory() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

void measure_performance(const std::function<void()> &task, bool useL3 = true) {
    int events[1] = {useL3 ? PAPI_L3_TCM : PAPI_L1_DCM};
    long long values[1] = {0};
    int event_set = PAPI_NULL;

    handle_PAPI_error(PAPI_create_eventset(&event_set));
    handle_PAPI_error(PAPI_add_events(event_set, events, 1));
    handle_PAPI_error(PAPI_start(event_set));
    task();
    handle_PAPI_error(PAPI_stop(event_set, values));

    long peak_mem = get_peak_memory();

    if (useL3)
        std::cout << "L3 Total Cache Misses: " << values[0] << std::endl;
    else
        std::cout << "L1 Data Cache Misses: " << values[0] << std::endl;
    std::cout << "Peak Memory Usage: " << peak_mem << " KB" << std::endl;

    PAPI_cleanup_eventset(event_set);
    PAPI_destroy_eventset(&event_set);
    PAPI_shutdown();
}