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
    Vamana(size_t max_elements, size_t dim, size_t num_threads, size_t M,
           size_t ef_construction, float alpha)
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
            std::make_shared<diskann::IndexSearchParams>(50, num_threads);

        auto index_config = diskann::IndexConfigBuilder()
            .with_metric(diskann::L2)
            .with_dimension(dim)
            .with_max_points(max_elements)
            .is_dynamic_index(true)
            .is_enable_tags(true)
            .is_use_opq(false)
            .is_filtered(false)
            .with_num_pq_chunks(0)
            .is_pq_dist_build(false)
            .with_num_frozen_pts(1)
            .with_tag_type("uint32")
            .with_label_type("uint32")
            .with_data_type("float")
            .with_index_write_params(params)
            .with_index_search_params(*search_params_ptr)
            .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
            .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
            .build();

        diskann::IndexFactory index_factory(index_config);
        index_ = std::unique_ptr<diskann::Index<T, TagT, TagT>>(
            dynamic_cast<diskann::Index<T, TagT, TagT>*>(index_factory.create_instance().release())
        );
        index_->set_start_points_at_random(1.0f);

        std::cout << "max_points = " << index_->get_max_points() << std::endl;
        std::cout << "current_points = " << index_->get_num_points() << std::endl;
    }

    void build(const T* data, const TagT* tags, size_t num_points) override {
        std::vector<TagT> tag_vec(tags, tags + num_points);
        index_->build(data, num_points, tag_vec);
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

    void set_query_params(const QParams& params) override {
        Ls_ = params.ef_search;
    }

    int search(const T* query, size_t k,
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
    uint32_t Ls_;
    float alpha_;
    size_t dim_;
    size_t num_threads_;

    std::unique_ptr<diskann::Index<T, TagT, TagT>> index_;
};
