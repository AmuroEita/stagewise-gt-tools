#pragma once

#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "../index.hpp"
#include "hnswlib/hnswlib/hnswlib.h"

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class HNSW : public IndexBase<T, TagT, LabelT> {
   public:
    HNSW(size_t max_elements, size_t dim, size_t num_threads, size_t M,
         size_t ef_construction)
        : dim_(dim),
          num_threads_(num_threads),
          space(dim),
          arena_(num_threads) {
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
        arena_.execute([&] {
            tbb::parallel_for(size_t(0), num_points, [&](size_t i) {
                index_->addPoint(batch_data + i * dim_, batch_tags[i]);
                __sync_fetch_and_add(&success_count, 1);
            });
        });
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
        arena_.execute([&] {
            tbb::parallel_for(size_t(0), num_queries, [&](size_t i) {
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
            });
        });
        return 0;
    }

    size_t num_threads_;
    size_t dim_;
    hnswlib::L2Space space;
    hnswlib::HierarchicalNSW<T>* index_;

   private:
    tbb::task_arena arena_;
};
