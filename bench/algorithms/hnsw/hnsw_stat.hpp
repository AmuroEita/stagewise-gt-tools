#pragma once

#include <omp.h>

#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "../index.hpp"
#include "hnswlib/hnswlib/hnswlib.h"

#define ENABLE_CC_STAT

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class HNSW : public IndexBase<T, TagT, LabelT> {
   public:
    HNSW(size_t max_elements, size_t dim, size_t num_threads, size_t M,
         size_t ef_construction)
        : dim_(dim), num_threads_(num_threads), space(dim) {
        index_ = new hnswlib::HierarchicalNSW<T>(&space, max_elements, M,
                                                 ef_construction);
    }

    void build(const T* data, const TagT* tags, size_t num_points) override {
#pragma omp parallel for num_threads(num_threads_)
        for (size_t i = 0; i < num_points; i++) {
            index_->addPoint((void*)(data + i * dim_), tags[i]);
        }
    }

    int insert(const T* data, const TagT tag) override {
        index_->addPoint(data, tag);
        return 0;
    }

    int batch_insert(const T* batch_data, const TagT* batch_tags,
                     size_t num_points) override {
        int success_count = 0;
#ifdef ENABLE_CC_STAT
        std::vector<double> thread_total_time(num_threads_, 0.0);
        std::vector<double> thread_work_time(num_threads_, 0.0);
#endif

#pragma omp parallel num_threads(num_threads_)
        {
            int tid = omp_get_thread_num();
#ifdef ENABLE_CC_STAT
            auto t_total_start = std::chrono::high_resolution_clock::now();
#endif

#pragma omp for reduction(+ : success_count)
            for (size_t i = 0; i < num_points; ++i) {
#ifdef ENABLE_CC_STAT
                auto t_work_start = std::chrono::high_resolution_clock::now();
#endif
                index_->addPoint(batch_data + i * dim_, batch_tags[i]);
#ifdef ENABLE_CC_STAT
                auto t_work_end = std::chrono::high_resolution_clock::now();
                thread_work_time[tid] +=
                    std::chrono::duration<double>(t_work_end - t_work_start)
                        .count();
#endif
                success_count++;
            }

#ifdef ENABLE_CC_STAT
            auto t_total_end = std::chrono::high_resolution_clock::now();
            thread_total_time[tid] +=
                std::chrono::duration<double>(t_total_end - t_total_start)
                    .count();
#endif
        }

#ifdef ENABLE_CC_STAT
        double batch_total_time = 0.0;
        double batch_work_time = 0.0;
        double batch_cc_time = 0.0;
        for (size_t i = 0; i < num_threads_; ++i) {
            batch_total_time += thread_total_time[i];
            batch_work_time += thread_work_time[i];
            batch_cc_time += (thread_total_time[i] - thread_work_time[i]);
        }
        double batch_cc_ratio = batch_cc_time / batch_total_time * 100.0;
        {
            std::lock_guard<std::mutex> lock(stat_mutex);
            batch_stats_.push_back({"write", batch_total_time, batch_work_time,
                                    batch_cc_time, batch_cc_ratio});
        }
#endif

        return success_count == num_points ? 0 : -1;
    }

    void set_query_params(const QParams& params) override {
        index_->setEf(params.ef_search);
    }

    int search(const T* query, size_t k,
               std::vector<TagT>& result_tags) override {
        auto result = index_->searchKnn(query, k);
        while (!result.empty()) {
            result_tags.push_back(result.top().second);
            result.pop();
        }
        return 0;
    }

    int batch_search(const T* batch_queries, uint32_t k, size_t num_queries,
                     TagT** batch_results) override {
#ifdef ENABLE_CC_STAT
        std::vector<double> thread_total_time(num_threads_, 0.0);
        std::vector<double> thread_work_time(num_threads_, 0.0);
#endif

#pragma omp parallel num_threads(num_threads_)
        {
            int tid = omp_get_thread_num();
#ifdef ENABLE_CC_STAT
            auto t_total_start = std::chrono::high_resolution_clock::now();
#endif

#pragma omp for
            for (size_t i = 0; i < num_queries; ++i) {
#ifdef ENABLE_CC_STAT
                auto t_work_start = std::chrono::high_resolution_clock::now();
#endif
                auto result = index_->searchKnn(batch_queries + i * dim_, k);
                size_t j = 0;
                std::vector<TagT> results;
                while (!result.empty()) {
                    results.push_back(result.top().second);
                    result.pop();
                }
                std::reverse(results.begin(), results.end());
                for (j = 0; j < results.size(); ++j) {
                    batch_results[i][j] = results[j];
                }
#ifdef ENABLE_CC_STAT
                auto t_work_end = std::chrono::high_resolution_clock::now();
                thread_work_time[tid] +=
                    std::chrono::duration<double>(t_work_end - t_work_start)
                        .count();
#endif
            }
#ifdef ENABLE_CC_STAT
            auto t_total_end = std::chrono::high_resolution_clock::now();
            thread_total_time[tid] +=
                std::chrono::duration<double>(t_total_end - t_total_start)
                    .count();
#endif
        }

#ifdef ENABLE_CC_STAT
        double batch_total_time = 0.0;
        double batch_work_time = 0.0;
        double batch_cc_time = 0.0;
        for (size_t i = 0; i < num_threads_; ++i) {
            batch_total_time += thread_total_time[i];
            batch_work_time += thread_work_time[i];
            batch_cc_time += (thread_total_time[i] - thread_work_time[i]);
        }
        double batch_cc_ratio = batch_cc_time / batch_total_time * 100.0;
        {
            std::lock_guard<std::mutex> lock(stat_mutex);
            batch_stats_.push_back({"read", batch_total_time, batch_work_time,
                                    batch_cc_time, batch_cc_ratio});
        }
#endif

        return 0;
    }

    size_t num_threads_;
    size_t dim_;
    hnswlib::L2Space space;
    hnswlib::HierarchicalNSW<T>* index_;

#ifdef ENABLE_CC_STAT
    struct BatchStat {
        std::string type;  // "read" or "write"
        double total_time;
        double work_time;
        double cc_time;
        double cc_ratio;
    };
    std::vector<BatchStat> batch_stats_;

    std::mutex stat_mutex;

    void save_stat(const std::string& filename) {
        std::ofstream ofs(filename);
        ofs << "type,batch_total_time,batch_work_time,batch_cc_time,batch_cc_"
               "ratio"
            << std::endl;
        for (const auto& stat : batch_stats_) {
            ofs << stat.type << "," << stat.total_time << "," << stat.work_time
                << "," << stat.cc_time << "," << stat.cc_ratio << std::endl;
        }
        ofs.close();
    }
#endif
};
