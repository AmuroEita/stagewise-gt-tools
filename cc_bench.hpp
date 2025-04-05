#include <vector>        
#include <queue>          
#include <thread>         
#include <mutex>          
#include <condition_variable> 
#include <functional>     
#include <atomic>         
#include <memory>        
#include <chrono>         
#include <iostream>       
#include <algorithm>      
#include <numeric>        
#include <cstdint>   

#include "utils.hpp"     

class ThreadPool {
public:
    ThreadPool(size_t num_threads) : stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

    void enqueue_task(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) return;
            tasks.emplace(std::move(task));
        }
        condition.notify_one();
    }

    void wait_for_tasks() {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (tasks.empty()) break;
            }
            std::this_thread::yield();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class IndexBase {
public:
    virtual ~IndexBase() = default;
    virtual void build(T* data, size_t num_points, const std::vector<TagT>& tags) = 0;
    virtual int insert_point(T* point, TagT tag) = 0;
    virtual void search_with_tags(const T* query, size_t k, size_t Ls, TagT* tags, std::vector<T*>& res) = 0;
};

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
bool concurrent_bench(const std::string& data_path, const std::string& query_file, const size_t begin_num,
                      const float write_ratio, const size_t batch_size, const uint32_t recall_at, const uint32_t Ls,
                      const uint32_t num_threads, std::unique_ptr<IndexBase<T, TagT, LabelT>>&& index, 
                      const std::string& res_path)
{
    std::cout << "Starting concurrent benchmarking with #threads: " << num_threads 
              << " #ratio: " << write_ratio << ":" << 1 - write_ratio << std::endl;

    size_t data_num, data_dim, aligned_dim;
    get_bin_metadata(data_path, data_num, data_dim);

    size_t query_num, query_dim, query_aligned_dim;
    T* query = nullptr;
    load_aligned_bin(query_file, query, query_num, query_dim, query_aligned_dim);

    T* data = nullptr;
    load_aligned_bin(data_path, data, data_num, data_dim, aligned_dim);

    std::vector<uint32_t> tags(begin_num);
    std::iota(tags.begin(), tags.end(), 1 + static_cast<uint32_t>(0));
    index->build(data, begin_num, tags);

    size_t insert_total = data_num - begin_num;
    size_t search_total = insert_total * ((1 - write_ratio) / write_ratio);
    size_t search_batch_size = batch_size * ((1 - write_ratio) / write_ratio);
    size_t start_insert_offset = 0, end_insert_offset = 0, start_search_offset = 0, end_search_offset = 0,
           query_idx = 0;

    std::exception_ptr last_exception = nullptr;
    std::mutex last_except_mutex, result_mutex, insert_latency_mutex, search_latency_mutex;
    std::vector<double> insert_latency_stats, search_latency_stats;
    std::vector<SearchResult<T>> search_results;
    ThreadPool pool(num_threads);

    auto succeed_insert_count = std::make_shared<std::atomic<size_t>>(0);
    auto failed_insert_count = std::make_shared<std::atomic<size_t>>(0);
    auto succeed_search_count = std::make_shared<std::atomic<size_t>>(0);
    auto failed_search_count = std::make_shared<std::atomic<size_t>>(0);

    auto st = std::chrono::high_resolution_clock::now();
    while (end_insert_offset < insert_total || end_search_offset < search_total)
    {
        end_insert_offset = std::min(start_insert_offset + batch_size, insert_total);
        for (size_t idx = start_insert_offset; idx < end_insert_offset; ++idx)
        {
            pool.enqueue_task([&, idx] {
                try
                {
                    auto qs = std::chrono::high_resolution_clock::now();
                    int insert_result = index->insert_point(&data[(idx + begin_num) * aligned_dim],
                                                            1 + static_cast<TagT>(idx + begin_num));
                    if (insert_result != 0)
                        failed_insert_count->fetch_add(1, std::memory_order_seq_cst);
                    else
                        succeed_insert_count->fetch_add(1, std::memory_order_seq_cst);

                    auto qe = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> diff = qe - qs;
                    {
                        std::unique_lock<std::mutex> lock(insert_latency_mutex);
                        insert_latency_stats.push_back((float)(diff.count() * 1000000));
                    }
                }
                catch (...)
                {
                    std::unique_lock<std::mutex> lock(last_except_mutex);
                    last_exception = std::current_exception();
                }
            });
        }
        start_insert_offset = end_insert_offset;

        end_search_offset = std::min(start_search_offset + search_batch_size, search_total);
        for (size_t idx = start_search_offset; idx < end_search_offset; ++idx)
        {
            if (++query_idx >= query_num)
                query_idx %= query_num;
            pool.enqueue_task([&, query_idx] {
                try
                {
                    auto qs = std::chrono::high_resolution_clock::now();
                    std::vector<TagT> query_result_tags(recall_at);
                    std::vector<T*> res;
                    index->search_with_tags(query + query_idx * query_aligned_dim, recall_at, Ls, 
                                            query_result_tags.data(), nullptr, res);
                    search_results.emplace_back(end_search_offset, query_idx, query_result_tags);

                    auto qe = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> diff = qe - qs;
                    {
                        std::unique_lock<std::mutex> lock(search_latency_mutex);
                        search_latency_stats.push_back((float)(diff.count() * 1000000));
                    }
                    {
                        std::unique_lock<std::mutex> lock(result_mutex);
                        search_results.emplace_back(end_insert_offset, query_idx, query_result_tags);
                    }
                }
                catch (...)
                {
                    std::unique_lock<std::mutex> lock(last_except_mutex);
                    last_exception = std::current_exception();
                }
            });
        }
        start_search_offset = end_search_offset;
    }

    pool.wait_for_tasks();

    auto et = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(et - st).count();
    double insert_qps = insert_total / elapsed_sec;
    double search_qps = search_total / elapsed_sec;

    std::sort(insert_latency_stats.begin(), insert_latency_stats.end());
    double mean_insert_latency = std::accumulate(insert_latency_stats.begin(), insert_latency_stats.end(), 0.0) /
                                 static_cast<float>(insert_total);
    double p99_insert_latency = insert_latency_stats.empty() ? 0.0 : 
                                (float)insert_latency_stats[(uint64_t)(0.999 * insert_total)];

    std::sort(search_latency_stats.begin(), search_latency_stats.end());
    double mean_search_latency = std::accumulate(search_latency_stats.begin(), search_latency_stats.end(), 0.0) /
                                 static_cast<float>(search_total);
    double p99_search_latency = search_latency_stats.empty() ? 0.0 : 
                                (float)search_latency_stats[(uint64_t)(0.999 * search_total)];

    std::cout << "Total time: " << elapsed_sec << " seconds\n"
              << "Insertion Statistics:\n"
              << "  Overall throughput: " << insert_qps << " points/second\n"
              << "  Mean latency: " << (insert_latency_stats.empty() ? 0.0 : mean_insert_latency) << " microseconds\n"
              << "  P99 latency: " << (insert_latency_stats.empty() ? 0.0 : p99_insert_latency) << " microseconds\n"
              << "Search Statistics:\n"
              << "  Overall throughput: " << search_qps << " points/second\n"
              << "  Mean latency: " << (search_latency_stats.empty() ? 0.0 : mean_search_latency) << " microseconds\n"
              << "  P99 latency: " << (search_latency_stats.empty() ? 0.0 : p99_search_latency) << " microseconds\n";

    delete[] data;
    delete[] query;
    
    write_results(search_results, res_path);

    return true;
}