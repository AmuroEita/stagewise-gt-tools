#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>

#include "../index.hpp"
#include "parlayann/algorithms/HNSW/HNSW.hpp"
#include "parlayann/algorithms/utils/types.h"
#include "parlayann/algorithms/utils/euclidian_point.h"
#include "parlayann/algorithms/utils/point_range.h"

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class ParlayHNSW : public IndexBase<T, TagT, LabelT> {
   public:
    using Point = parlayANN::Euclidian_Point<float>;
    using Range = parlayANN::PointRange<Point>;
    using desc = parlayANN::Desc_HNSW<T, Point>;

    ParlayHNSW(size_t dim, size_t max_elements, size_t M,
               size_t ef_construction, float m_l, float alpha,
               size_t num_threads)
        : dim_(dim),
          graph_degree_(M),
          ef_construction_(ef_construction),
          m_l_(m_l),
          alpha_(alpha),
          num_threads_(num_threads) {}

    void build(const T* data, const TagT* tags, size_t num_points) override {
        Range points(data, num_points, dim_);
        auto ps = parlay::delayed_seq<Point>(points.size(),
            [&](size_t i) { return points[i]; });

        std::cout << "[build] start building HNSW index..." << std::endl;
        index_ = std::make_unique<ANN::HNSW<desc>>(ps.begin(), ps.end(), dim_,
                                                   m_l_, graph_degree_,
                                                   ef_construction_, alpha_);
        std::cout << "[build] finished building HNSW index." << std::endl;
    }

    int batch_insert(const T* batch_data, const TagT* batch_tags, size_t num_points) override {
        if (!index_) return -1;
        uint32_t start_id = *batch_tags;

        Range points(batch_data, num_points, dim_);
        auto ps = parlay::delayed_seq<Point>(points.size(),
            [&](size_t i) { return points[i]; });

        index_->batch_insert(ps.begin(), ps.end(), start_id);
        return 0;
    }

    int insert(const T *point, const TagT tag) override {
        std::cerr << "ParlayHNSW does not support dynamic single insertion"
                  << std::endl;
        return -1;
    }

    void set_query_params(size_t Ls) override {
        ef_search_ = Ls;
    }

    int batch_search(const T* batch_queries, uint32_t k,
                      uint32_t Ls, size_t num_queries,
                      TagT** batch_results) override {
        std::vector<std::vector<TagT>> results(num_queries);

        this->search_latencies.resize(num_queries, 0.0);
        parlay::parallel_for(0, num_queries, [&](size_t i) {
            auto start = std::chrono::high_resolution_clock::now();
            search_with_tags(batch_queries + i * dim_, k, Ls, results[i]);
            auto end = std::chrono::high_resolution_clock::now();
            this->search_latencies[i] =
                std::chrono::duration<double, std::micro>(end - start).count();
        });

        for (size_t i = 0; i < num_queries; ++i) {
            for (size_t j = 0; j < k; ++j) {
                batch_results[i][j] = results[i][j];
            }
        }
    }

    int search_with_tags(const T *query, size_t k, size_t Ls,
                          std::vector<TagT> &result_tags) override {

        parlayANN::QueryParams params(k, beam_width, 1.35, -1,
                       std::min<int>(G.max_degree(), 3 * visit_limit));

        return 0;
    }

   private:
    size_t dim_;
    uint32_t graph_degree_;  // M
    uint32_t ef_construction_;
    uint32_t ef_search_;
    float m_l_;
    float alpha_;
    size_t num_threads_;
    std::unique_ptr<ANN::HNSW<desc>> index_;
};