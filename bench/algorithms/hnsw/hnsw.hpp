#pragma once

#include <chrono>
#include <cstddef>

#include "../index.hpp"
#include "hnswlib/hnswlib/hnswlib.h"

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class HNSW : public IndexBase<T, TagT, LabelT> {
   public:
    HNSW(size_t dim, size_t max_elements, size_t M, size_t ef_construction,
         size_t num_threads)
        : dim_(dim), num_threads_(num_threads), space(dim) {
        index_ = new hnswlib::HierarchicalNSW<T>(&space, max_elements, M,
                                                 ef_construction);
    }

    void build(const T* data, const TagT* tags, size_t num_points) override {
        for (size_t i = 0; i < num_points; i++) {
            index_->addPoint(data + i * dim_, tags[i]);
        }
    }

    int insert(const T* data, const TagT tag) override {
        index_->addPoint(data, tag);
        return 0;
    }

    int batch_insert(const T* batch_data, const TagT* batch_tags,
                     size_t num_points) override {
        int success_count = 0;
#pragma omp parallel for reduction(+ : success_count) num_threads(num_threads_)
        for (size_t i = 0; i < num_points; ++i) {
            index_->addPoint(batch_data + i * dim_, batch_tags[i]);
            success_count++;
        }
        return success_count == num_points ? 0 : -1;
    }

    void set_query_params(const QParams& params) override {
        query_params_ = params;
        index_->setEf(params.Ls);
    }

    int search(const T* query, size_t k, const QParams& params,
               std::vector<TagT>& result_tags) override {
        index_->setEf(params.Ls);
        auto result = index_->searchKnn(query, k);
        while (!result.empty()) {
            result_tags.push_back(result.top().second);
            result.pop();
        }
        return 0;
    }

    int batch_search(const T* batch_queries, uint32_t k, size_t num_queries,
                     TagT** batch_results) override {
        index_->setEf(query_params_.Ls);

#pragma omp parallel for num_threads(num_threads_)
        for (size_t i = 0; i < num_queries; ++i) {
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
        }
        return 0;
    }

    void print_dim() { std::cout << "dim: " << dim_ << std::endl; }

    size_t num_threads_ = 1;
    size_t dim_;
    hnswlib::L2Space space;
    hnswlib::HierarchicalNSW<T>* index_;
    QParams query_params_;
};
