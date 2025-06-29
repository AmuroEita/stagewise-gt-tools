#pragma once

#include <chrono>
#include <cstddef>
#include <iostream>

#include "../index.hpp"
#include "DiskANN/include/index.h"
#include "DiskANN/include/index_factory.h"
#include "DiskANN/include/parameters.h"

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class Vamana : public IndexBase<T, TagT, LabelT> {
   public:
    Vamana(size_t dim, size_t max_elements, size_t M, size_t ef_construction,
           float alpha, size_t num_threads)
        : dim_(dim), num_threads_(num_threads) {
        diskann::Metric metric = diskann::L2;

        diskann::IndexWriteParameters params =
            diskann::IndexWriteParametersBuilder(ef_construction, M)
                .with_filter_list_size(0)
                .with_alpha(alpha)
                .with_saturate_graph(false)
                .with_num_threads(num_threads)
                .build();

        auto params_ptr =
            std::make_shared<diskann::IndexWriteParameters>(params);
        auto search_params_ptr =
            std::make_shared<diskann::IndexSearchParams>(Ls_, num_threads);

        index_ = std::make_unique<diskann::Index<T, TagT, TagT>>(
            metric, dim_, max_elements, params_ptr, search_params_ptr, 0, true,
            true, false, false, 0, false);
    }

    void build(const T* data, const TagT* tags, size_t num_points) override {
#pragma omp parallel for num_threads(num_threads_)
        for (size_t i = 0; i < num_points; i++) {
            index_->insert_point(data + i * dim_, tags[i]);
        }
    }

    int insert(const T* data, const TagT tag) override {
        index_->insert_point(data, tag);
        return 0;
    }

    int batch_insert(const T* batch_data, const TagT* batch_tags,
                     size_t num_points) override {
#pragma omp parallel for num_threads(num_threads_)
        for (size_t i = 0; i < num_points; i++) {
            index_->insert_point(batch_data + i * dim_, batch_tags[i]);
        }
        return 0;
    }

    void set_query_params(const QParams& params) override { Ls_ = params.Ls; }

    int search(const T* query, size_t k, const QParams& params,
               std::vector<TagT>& result_tags) override {
        return 0;
    }

    int batch_search(const T* batch_queries, uint32_t k, size_t num_queries,
                     TagT** batch_results) override {
        for (size_t i = 0; i < num_queries; ++i) {
            batch_results[i] = new TagT[k];
        }

#pragma omp parallel for num_threads(num_threads_)
        for (size_t i = 0; i < num_queries; ++i) {
            std::vector<TagT> tags_res(k);
            std::vector<float> distances(k);
            std::vector<T*> res_vectors;

            index_->search_with_tags(batch_queries + i * dim_, k, Ls_,
                                     tags_res.data(), nullptr, res_vectors);

            for (uint32_t j = 0; j < k; ++j) {
                batch_results[i][j] = tags_res[j];
            }
        }
        return 0;
    }

    uint32_t L_;
    uint32_t R_;
    uint32_t Ls_ = 100;
    float alpha_;
    size_t dim_;
    size_t num_threads_;

    std::unique_ptr<diskann::Index<T, TagT, TagT>> index_;
};
